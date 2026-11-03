"""Evidence on the GPU: adaptive tempered SMC with HMC moves on a DALI posterior.

Sequential Monte Carlo sampler (Del Moral, Doucet & Jasra 2006) through the tempered path
π_β ∝ π(x) L_DALI(x)^β, β: 0 → 1, with the evidence as the product of the incremental weights,

    log Z_DALI = Σ_t log (1/N) Σ_i exp(Δβ_t ℓ_i),     ℓ = log L_DALI,

which is unbiased for Z_DALI. Each stage picks Δβ by bisection so that the incremental weights
keep an effective sample size of ``ess_target``·N, resamples (systematic), and moves every
particle with ``n_mcmc`` HMC transitions invariant for π_β. Everything except the scalar
bisection runs on the device: N particles × n_leapfrog gradient evaluations per move in
fixed-shape vmapped chunks, so the cost is set by the DALI expansion's log L + gradient
(~50 µs per point with ``inference.dali_reduce.CompressedDALI`` on an RTX 3070).

Parameterization: HMC runs in z = logit((x − lo)/(hi − lo)) per parameter (sampling boxes
mapped to ℝ, Jacobian in the target), whitened by the particles' covariance before every
stage (z = m + L u, unit mass in u). The prior and its normalization: the initial particles are exact draws from the
separable prior on the (optionally narrowed) boxes restricted to the hard cuts, the cuts
and joint prior terms (``bundle.log_prior_fn``) are part of every π_β, and

    log Z_DALI = log Z_SMC + log(V_narrow / V_box) + log P_cut(narrow) − log P_cut(box).

``n_runs`` independent populations give the reported log Z_DALI (log of the mean of their Z,
which stays unbiased) and its error (scatter of the runs); the final particles of all runs
are the posterior draws (also those of the DALI order check, ``dali_ns.dali_order_check``).

Reference start (``reference="pilot"`` or a posterior file): the path starts from a
normalized multivariate Student-t q in the sampler's logit coordinates, fitted to a pilot
population (or saved posterior draws) and widened, instead of the prior::

    γ_β(z) = q(z)^{1−β} [π̃(x(z)) |dx/dz| L_DALI(x(z))]^β,    π̃ = exp(log_prior_fn) (unnormalized),

which the same machinery runs with (log q, log π̃ + log|J| + ℓ − log q) in place of
(log prior, ℓ). The estimator stays unbiased for any q that covers the posterior; it only
needs far fewer stages than annealing from the prior (KL ~ a few nats instead of ~50–100).
Then log Z_SMC = log Z_path − log ∫ π̃ dx, with ∫ π̃ = (separable masses, analytic) × P_cut.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from gpucosmo.jax.mcmc.dali_ns import narrowed_prior_view, sample_prior

jax.config.update("jax_enable_x64", True)


@dataclass
class DALISMCConfig:
    n_particles: int = 4096
    n_runs: int = 4  # independent populations (log Z error from their scatter)
    ess_target: float = 0.5  # incremental-weight ESS fraction kept by each tempering step
    n_mcmc: int = 6  # HMC transitions per stage
    n_leapfrog: int = 12
    n_final_mcmc: int = 24  # transitions at β = 1 (decorrelate the posterior draws from resampling)
    target_accept: float = 0.75
    step_jitter: float = 0.2  # per-transition step size uniform in ε (1 ± jitter): randomized path length
    chunk: int = 2048  # particles per vmapped log L + gradient call (device memory)
    seed: int = 0
    n_posterior: int = 4096  # posterior draws written (``.ns_posterior.npz``; the DALI order check)
    n_prior_mc: int = 200_000  # prior draws for the mass of the hard cuts
    max_stages: int = 1000
    sampling_box: dict | None = None  # {name: (lo, hi)} narrower uniform boxes (log Z corrected)
    min_edge_sigma: float = 10.0
    loglike_ceiling: float | None = None  # DALI log L above it is rejected (−inf)
    # Waste-free SMC (Dau & Chopin 2022): each stage resamples M = n_particles / chain_length
    # ancestors and runs an HMC chain of chain_length − 1 transitions from each; every state
    # visited is a particle of the next population (n_mcmc / n_final_mcmc are then unused).
    # Same evidence estimator; ~chain_length / n_mcmc × fewer gradients per particle.
    waste_free: bool = False
    chain_length: int = 16
    # Mutation kernel: "hmc" (gradients) or "rwmh" (gradient-free random-walk Metropolis in the
    # whitened coordinates, scale adapted to ``rw_accept``; n_mcmc = random-walk steps per stage).
    move: str = "hmc"
    rw_accept: float = 0.234
    # Start of the tempering path: "prior" (anneal from the prior), "pilot" (a Student-t fitted
    # to a pilot prior-start population of ``pilot_particles``), or a path to an .npz with
    # posterior draws ``pos`` (e.g. a previous run's .smc_particles.npz / .ns_posterior.npz).
    reference: str = "prior"
    pilot_particles: int = 2048
    pilot_ess: float = 0.5  # the pilot only fits the reference's mean / covariance: coarser tempering
    reference_nu: float = 5.0  # Student-t degrees of freedom (heavy tails cover the posterior's)
    reference_scale: float = 1.3  # covariance widened by scale² relative to the draws'
    # Stop adding runs (up to n_runs) once the log Z_DALI error — max(run scatter, ESS prediction)/√runs
    # with the cut-mass error — is ≤ target_err, after at least min_runs. None: always n_runs.
    target_err: float | None = None
    min_runs: int = 4


def _ess(logw: np.ndarray) -> float:
    fin = np.isfinite(logw)
    if not fin.any():
        return 0.0
    w = np.exp(logw[fin] - logw[fin].max())
    return float(w.sum() ** 2 / (w * w).sum())


def next_temperature(ll: np.ndarray, beta: float, ess_target: float) -> float:
    """Largest β' ≤ 1 with ESS(exp((β' − β) ℓ)) ≥ ess_target · N_finite (bisection; particles
    with ℓ = −inf have zero weight for any β' > β)."""
    n_target = ess_target * int(np.isfinite(ll).sum())
    if _ess((1.0 - beta) * ll) >= n_target:
        return 1.0
    lo, hi = 0.0, 1.0 - beta
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _ess(mid * ll) >= n_target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12 * max(hi, 1e-300):
            break
    return beta + max(lo, 1e-12)


def systematic_resample(logw: np.ndarray, rng: np.random.Generator, m: int | None = None) -> np.ndarray:
    """``m`` (default: len(logw)) indices drawn systematically with probabilities ∝ exp(logw)."""
    w = np.exp(logw - np.max(logw))
    w = np.where(np.isfinite(w), w, 0.0)
    c = np.cumsum(w / w.sum())
    c[-1] = 1.0
    m = w.size if m is None else int(m)
    u = (rng.uniform() + np.arange(m)) / m
    return np.searchsorted(c, u)


class _Kernel:
    """Jitted pieces: target parts, and ``n`` HMC transitions for all particles."""

    def __init__(self, bundle, ll_fn: Callable, lo, hi, cfg: DALISMCConfig, ref: dict | None = None):
        self.lo, self.hi = jnp.asarray(lo), jnp.asarray(hi)
        self.width = self.hi - self.lo
        ceiling = cfg.loglike_ceiling
        self.cfg = cfg
        self.set_reference(ref)

        def x_of(z):
            return self.lo + self.width * jax.nn.sigmoid(z)

        def parts_plain(z, data):
            x = x_of(z)
            log_jac = jnp.sum(jnp.log(self.width) - jax.nn.softplus(-z) - jax.nn.softplus(z))
            lp = bundle.log_prior_fn(x) + log_jac
            ll = ll_fn(x, data)
            ok = jnp.isfinite(ll) if ceiling is None else jnp.isfinite(ll) & (ll <= ceiling)
            return lp, jnp.where(ok, ll, -jnp.inf)

        def parts(z, dk):
            """(log prior, ℓ) of the path: plain, or from the reference q when ``ref["on"]``:
            (log q, log π̃ + log|J| + ℓ − log q). The reference is a runtime argument, so one
            compiled program serves the pilot and the production runs (and the disk cache)."""
            data, ref = dk
            lp, ll = parts_plain(z, data)
            y = ref["chol_inv"] @ (z - ref["mean"])
            lq = ref["log_norm"] - 0.5 * (ref["nu"] + z.shape[0]) * jnp.log1p(y @ y / ref["nu"])
            on = ref["on"] > 0
            return jnp.where(on, lq, lp), jnp.where(on, lp + ll - lq, ll)

        self.x_of = jax.jit(jax.vmap(x_of))

        def target(u, beta, m, L, data):
            lp, ll = parts(m + L @ u, data)
            return lp + jnp.where(beta > 0, beta * ll, 0.0), (lp, ll)

        vg = jax.value_and_grad(target, has_aux=True)

        def move(key, u, val, g, lp, ll, beta, eps, m, L, data):
            """One HMC transition of one particle (unit mass in the whitened u)."""
            k1, k2, k3 = jax.random.split(key, 3)
            p0 = jax.random.normal(k1, u.shape, dtype=u.dtype)
            e = eps * jax.random.uniform(k2, minval=1.0 - cfg.step_jitter, maxval=1.0 + cfg.step_jitter,
                                         dtype=u.dtype)

            def leap(c, _):
                u1, p1, _v, g1, _lp, _ll = c
                p1 = p1 + 0.5 * e * g1
                u1 = u1 + e * p1
                (v1, (lp1, ll1)), g1 = vg(u1, beta, m, L, data)
                p1 = p1 + 0.5 * e * g1
                return (u1, p1, v1, g1, lp1, ll1), None

            (u1, p1, v1, g1, lp1, ll1), _ = jax.lax.scan(leap, (u, p0, val, g, lp, ll), None,
                                                          length=cfg.n_leapfrog)
            log_r = (v1 - 0.5 * p1 @ p1) - (val - 0.5 * p0 @ p0)
            acc = jnp.log(jax.random.uniform(k3, dtype=u.dtype)) < log_r  # NaN → reject
            sel = lambda a, b: jnp.where(acc, a, b)  # noqa: E731
            return (sel(u1, u), sel(v1, val), jnp.where(acc, g1, g), sel(lp1, lp), sel(ll1, ll),
                    acc, jnp.sum((u1 - u) ** 2) * acc / (2.0 * u.shape[0]))

        def chunk_fn(args, beta, eps, n_moves, m, L, data):
            keys, u = args
            (val, (lp, ll)), g = jax.vmap(vg, in_axes=(0, None, None, None, None))(u, beta, m, L, data)

            def one(i, c):
                u, val, g, lp, ll, n_acc, jump = c
                ks = jax.random.split(jax.random.fold_in(keys, i), u.shape[0])
                u, val, g, lp, ll, acc, j = jax.vmap(
                    move, in_axes=(0, 0, 0, 0, 0, 0, None, None, None, None, None))(
                        ks, u, val, g, lp, ll, beta, eps, m, L, data)
                return (u, val, g, lp, ll, n_acc + acc, jump + j)

            z0 = jnp.zeros(u.shape[0])
            # dynamic trip count: one program for the stage moves and the final ones
            u, _v, _g, lp, ll, n_acc, jump = jax.lax.fori_loop(0, n_moves, one, (u, val, g, lp, ll, z0, z0))
            return u, lp, ll, n_acc / n_moves, jump / n_moves

        def mutate(key, Z, beta, eps, m, L, data, n_moves):
            n, d = Z.shape
            c = min(cfg.chunk, n)
            n_pad = -(-n // c) * c
            Linv_dz = jax.scipy.linalg.solve_triangular(L, (Z - m).T, lower=True).T
            U = jnp.concatenate([Linv_dz, jnp.repeat(Linv_dz[-1:], n_pad - n, axis=0)])
            keys = jax.random.split(key, n_pad // c)
            out = jax.lax.map(lambda a: chunk_fn(a, beta, eps, n_moves, m, L, data),
                              (keys, U.reshape(n_pad // c, c, d)))
            U, lp, ll, acc, jump = (a.reshape((n_pad,) + a.shape[2:])[:n] for a in out)
            return m + U @ L.T, lp, ll, acc, jump

        mutate_jit = jax.jit(mutate)
        self.mutate = lambda key, Z, beta, eps, m, L, data, n_moves: mutate_jit(
            key, Z, beta, eps, m, L, (data, self.ref_arrays), jnp.asarray(n_moves, dtype=jnp.int32))

        def target_v(u, beta, m, L, data):
            lp, ll = parts(m + L @ u, data)
            return lp + jnp.where(beta > 0, beta * ll, 0.0), lp, ll

        def move_rw(key, u, val, lp, ll, beta, eps, m, L, data):
            """One random-walk Metropolis step (whitened u, proposal N(u, ε² I))."""
            k1, k2 = jax.random.split(key)
            u1 = u + eps * jax.random.normal(k1, u.shape, dtype=u.dtype)
            v1, lp1, ll1 = target_v(u1, beta, m, L, data)
            acc = jnp.log(jax.random.uniform(k2, dtype=u.dtype)) < v1 - val
            sel = lambda a, b: jnp.where(acc, a, b)  # noqa: E731
            return sel(u1, u), sel(v1, val), sel(lp1, lp), sel(ll1, ll), acc, jnp.sum((u1 - u) ** 2) * acc / (2.0 * u.shape[0])

        def chunk_rw(args, beta, eps, n_moves, m, L, data):
            keys, u = args
            val, lp, ll = jax.vmap(target_v, in_axes=(0, None, None, None, None))(u, beta, m, L, data)

            def one(c, k):
                u, val, lp, ll, n_acc, jump = c
                ks = jax.random.split(k, u.shape[0])
                u, val, lp, ll, acc, j = jax.vmap(move_rw, in_axes=(0, 0, 0, 0, 0, None, None, None, None, None))(
                    ks, u, val, lp, ll, beta, eps, m, L, data)
                return (u, val, lp, ll, n_acc + acc, jump + j), None

            z0 = jnp.zeros(u.shape[0])
            (u, _v, lp, ll, n_acc, jump), _ = jax.lax.scan(one, (u, val, lp, ll, z0, z0),
                                                           jax.random.split(keys, n_moves))
            return u, lp, ll, n_acc / n_moves, jump / n_moves

        def mutate_rw(key, Z, beta, eps, m, L, data, n_moves):
            n, d = Z.shape
            c = min(cfg.chunk, n)
            n_pad = -(-n // c) * c
            Linv_dz = jax.scipy.linalg.solve_triangular(L, (Z - m).T, lower=True).T
            U = jnp.concatenate([Linv_dz, jnp.repeat(Linv_dz[-1:], n_pad - n, axis=0)])
            keys = jax.random.split(key, n_pad // c)
            out = jax.lax.map(lambda a: chunk_rw(a, beta, eps, n_moves, m, L, data),
                              (keys, U.reshape(n_pad // c, c, d)))
            U, lp, ll, acc, jump = (a.reshape((n_pad,) + a.shape[2:])[:n] for a in out)
            return m + U @ L.T, lp, ll, acc, jump

        mutate_rw_jit = jax.jit(mutate_rw, static_argnames=("n_moves",))
        self.mutate_rw = lambda key, Z, beta, eps, m, L, data, n_moves: mutate_rw_jit(
            key, Z, beta, eps, m, L, (data, self.ref_arrays), n_moves=n_moves)

        def chunk_traj(args, beta, eps, n_moves, m, L, data):
            """Like ``chunk_fn`` but returns every state: (n_moves + 1, chains, ...)."""
            keys, u = args
            (val, (lp, ll)), g = jax.vmap(vg, in_axes=(0, None, None, None, None))(u, beta, m, L, data)

            def one(c, k):
                u, val, g, lp, ll = c
                ks = jax.random.split(k, u.shape[0])
                u, val, g, lp, ll, acc, j = jax.vmap(
                    move, in_axes=(0, 0, 0, 0, 0, 0, None, None, None, None, None))(
                        ks, u, val, g, lp, ll, beta, eps, m, L, data)
                return (u, val, g, lp, ll), (u, lp, ll, acc, j)

            _, (us, lps, lls, accs, js) = jax.lax.scan(one, (u, val, g, lp, ll), jax.random.split(keys, n_moves))
            return (jnp.concatenate([u[None], us]), jnp.concatenate([lp[None], lps]),
                    jnp.concatenate([ll[None], lls]), jnp.mean(accs, axis=0), jnp.mean(js, axis=0))

        def mutate_traj(key, Z, beta, eps, m, L, data, n_moves):
            """``n_moves`` transitions of a chain from each row of ``Z`` (M, d); returns all
            (n_moves + 1)·M states, state-major (particle k belongs to chain k mod M)."""
            n, d = Z.shape
            c = min(cfg.chunk, n)
            n_pad = -(-n // c) * c
            Linv_dz = jax.scipy.linalg.solve_triangular(L, (Z - m).T, lower=True).T
            U = jnp.concatenate([Linv_dz, jnp.repeat(Linv_dz[-1:], n_pad - n, axis=0)])
            keys = jax.random.split(key, n_pad // c)
            us, lps, lls, acc, jump = jax.lax.map(lambda a: chunk_traj(a, beta, eps, n_moves, m, L, data),
                                                 (keys, U.reshape(n_pad // c, c, d)))
            # (chunks, steps, c, ...) → (steps, chains, ...)
            us, lps, lls = (jnp.swapaxes(a, 0, 1).reshape((n_moves + 1, n_pad) + a.shape[3:])[:, :n]
                            for a in (us, lps, lls))
            acc, jump = (a.reshape(-1)[:n] for a in (acc, jump))
            Zs = m + us.reshape(-1, d) @ L.T
            return Zs, lps.reshape(-1), lls.reshape(-1), acc, jump

        mutate_traj_jit = jax.jit(mutate_traj, static_argnames=("n_moves",))
        self.mutate_traj = lambda key, Z, beta, eps, m, L, data, n_moves: mutate_traj_jit(
            key, Z, beta, eps, m, L, (data, self.ref_arrays), n_moves=n_moves)

        def evaluate(Z, data):
            c = min(cfg.chunk, Z.shape[0])
            n_pad = -(-Z.shape[0] // c) * c
            Zp = jnp.concatenate([Z, jnp.repeat(Z[-1:], n_pad - Z.shape[0], axis=0)])
            lp, ll = jax.lax.map(lambda zz: jax.vmap(parts, in_axes=(0, None))(zz, data),
                                 Zp.reshape(n_pad // c, c, Z.shape[1]))
            return lp.reshape(-1)[:Z.shape[0]], ll.reshape(-1)[:Z.shape[0]]

        evaluate_jit = jax.jit(evaluate)
        self.evaluate = lambda Z, data: evaluate_jit(Z, (data, self.ref_arrays))

        def evaluate_plain(Z, data):  # prior + DALI log L, whatever the path's start
            c = min(cfg.chunk, Z.shape[0])
            n_pad = -(-Z.shape[0] // c) * c
            Zp = jnp.concatenate([Z, jnp.repeat(Z[-1:], n_pad - Z.shape[0], axis=0)])
            lp, ll = jax.lax.map(lambda zz: jax.vmap(parts_plain, in_axes=(0, None))(zz, data),
                                 Zp.reshape(n_pad // c, c, Z.shape[1]))
            return lp.reshape(-1)[:Z.shape[0]], ll.reshape(-1)[:Z.shape[0]]

        self.evaluate_plain = jax.jit(evaluate_plain)


def _reference_arrays(ref: dict | None, d: int) -> dict:
    if ref is None:  # off: identity placeholders of the same shapes (same compiled program)
        return dict(mean=jnp.zeros(d), chol_inv=jnp.eye(d), nu=jnp.asarray(5.0), log_norm=jnp.asarray(0.0),
                    on=jnp.asarray(0.0))
    return dict(mean=jnp.asarray(ref["mean"]), chol_inv=jnp.asarray(ref["chol_inv"]), nu=jnp.asarray(float(ref["nu"])),
                log_norm=jnp.asarray(float(ref["log_norm"])), on=jnp.asarray(1.0))


def _set_reference(self, ref: dict | None) -> None:
    """Start the path at the Student-t ``ref`` (``fit_reference``), or at the prior (None)."""
    self.ref = ref
    self.ref_arrays = _reference_arrays(ref, int(self.lo.shape[0]))


_Kernel.set_reference = _set_reference


def smc_run(kernel: _Kernel, x0: np.ndarray | None, data, cfg: DALISMCConfig, seed: int,
            log: Callable[[str], None] = print, tag: str = "", z0: np.ndarray | None = None) -> dict:
    """One adaptive tempered SMC population to β = 1, from the prior draws ``x0`` or — with a
    reference kernel — from the reference draws ``z0`` (logit coordinates)."""
    if cfg.waste_free:
        if kernel.ref is not None:
            raise ValueError("waste-free SMC supports the prior start only")
        return _smc_run_waste_free(kernel, x0, data, cfg, seed, log=log, tag=tag)
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    lo, hi = np.asarray(kernel.lo), np.asarray(kernel.hi)
    if z0 is not None:
        Z = jnp.asarray(z0)
        n, d = z0.shape
    else:
        n, d = x0.shape
        u01 = np.clip((x0 - lo) / (hi - lo), 1e-15, 1 - 1e-15)
        Z = jnp.asarray(np.log(u01) - np.log1p(-u01))
    lp, ll = kernel.evaluate(Z, data)
    ll = np.asarray(ll)
    beta, logz, var_logz = 0.0, 0.0, 0.0
    rw = cfg.move == "rwmh"
    if cfg.move not in ("hmc", "rwmh"):
        raise ValueError(f"move {cfg.move!r}")
    eps = 2.38 / np.sqrt(d) if rw else 1.0 * d ** -0.25
    target_acc = cfg.rw_accept if rw else cfg.target_accept
    mutate = kernel.mutate_rw if rw else kernel.mutate
    hist = []
    t0 = time.perf_counter()
    for stage in range(cfg.max_stages):
        b1 = next_temperature(ll, beta, cfg.ess_target)
        lw = (b1 - beta) * ll
        fin = np.isfinite(lw)
        if not fin.any():
            raise RuntimeError(f"SMC stage {stage}: every particle has log L = −inf (β = {beta:.3g})")
        mx = lw[fin].max()
        logz += float(mx + np.log(np.sum(np.exp(lw[fin] - mx))) - np.log(n))
        ess = _ess(lw)
        var_logz += (n / max(ess, 1e-300) - 1.0) / n  # relative variance of the weights' mean
        idx = systematic_resample(np.where(fin, lw, -np.inf), rng)
        Z = Z[jnp.asarray(idx)]
        beta = b1
        Zh = np.asarray(Z)
        m = Zh.mean(axis=0)
        cov = np.cov(Zh, rowvar=False) + 1e-10 * np.eye(d)
        L = np.linalg.cholesky(cov)
        final = beta >= 1.0
        key, sub = jax.random.split(key)
        Z, lp, ll_d, acc, jump = mutate(sub, Z, beta, eps, jnp.asarray(m), jnp.asarray(L), data,
                                        n_moves=cfg.n_final_mcmc if final else cfg.n_mcmc)
        ll = np.asarray(ll_d)
        acc_m = float(np.mean(np.asarray(acc)))
        jump_m = float(np.mean(np.asarray(jump)))
        hist.append(dict(stage=stage, beta=beta, ess=ess, accept=acc_m, eps=eps, jump=jump_m, logz=logz,
                         n_unique=int(np.unique(idx).size), t=time.perf_counter() - t0))
        log(f"{tag}stage {stage:3d}: β {beta:.4e}  ESS {ess:7.0f}  unique {hist[-1]['n_unique']:6d}  "
            f"accept {acc_m:.2f}  ε {eps:.3f}  jump {jump_m:.3f}  log Z {logz:.3f}  "
            f"({hist[-1]['t']:.0f} s)")
        eps = float(eps * np.exp(np.clip(2.0 * (acc_m - target_acc), -0.7, 0.7)))
        if final:
            break
    else:
        raise RuntimeError(f"SMC did not reach β = 1 in {cfg.max_stages} stages (β = {beta:.3g})")
    X = np.asarray(kernel.x_of(Z))
    if kernel.ref is not None:  # the path's ℓ is log π̃ + log|J| + ℓ − log q: report the DALI log L
        ll = np.asarray(kernel.evaluate_plain(Z, data)[1])
    return dict(logz=logz, logz_sd_ess=float(np.sqrt(var_logz)), x=X, loglike=ll, n_stages=len(hist), hist=hist, wall=time.perf_counter() - t0,
                n_like_grad=int(n * (1 + sum((cfg.n_final_mcmc if h["beta"] >= 1 else cfg.n_mcmc)
                                             * (1 if rw else cfg.n_leapfrog) + 1 for h in hist))))


def _smc_run_waste_free(kernel: _Kernel, x0: np.ndarray, data, cfg: DALISMCConfig, seed: int,
                        log: Callable[[str], None] = print, tag: str = "") -> dict:
    """Waste-free adaptive tempered SMC (Dau & Chopin 2022). Within-run error: each stage's
    increment is a mean over M chains of P correlated states, so its relative variance is
    estimated from the spread of the chain means (the ESS formula would ignore the within-chain
    correlation); the stage variances are summed."""
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    lo, hi = np.asarray(kernel.lo), np.asarray(kernel.hi)
    n, d = x0.shape
    P = int(cfg.chain_length)
    if n % P:
        raise ValueError(f"waste-free SMC: n_particles {n} is not a multiple of chain_length {P}")
    M = n // P
    u01 = np.clip((x0 - lo) / (hi - lo), 1e-15, 1 - 1e-15)
    Z = jnp.asarray(np.log(u01) - np.log1p(-u01))
    _, ll = kernel.evaluate(Z, data)
    ll = np.asarray(ll)
    beta, logz, var_logz = 0.0, 0.0, 0.0
    eps = 1.0 * d ** -0.25
    hist, n_grad = [], n
    t0 = time.perf_counter()
    iid = True  # the initial population: independent prior draws
    for stage in range(cfg.max_stages):
        b1 = next_temperature(ll, beta, cfg.ess_target)
        lw = (b1 - beta) * ll
        fin = np.isfinite(lw)
        if not fin.any():
            raise RuntimeError(f"SMC stage {stage}: every particle has log L = −inf (β = {beta:.3g})")
        mx = lw[fin].max()
        w = np.where(fin, np.exp(np.where(fin, lw, mx) - mx), 0.0)
        logz += float(mx + np.log(w.mean()))
        ess = _ess(lw)
        if iid:
            var_logz += (n / max(ess, 1e-300) - 1.0) / n
        else:
            cm = w.reshape(P, M).mean(axis=0)  # chain means
            var_logz += float(np.var(cm, ddof=1) / M / w.mean() ** 2)
        wn = w / w.sum()
        Zh = np.asarray(Z)
        m = wn @ Zh
        cov = np.cov(Zh, rowvar=False, aweights=wn) + 1e-10 * np.eye(d)
        L = np.linalg.cholesky(cov)
        idx = systematic_resample(np.where(fin, lw, -np.inf), rng, M)
        beta = b1
        key, sub = jax.random.split(key)
        Z, _lp, ll_d, acc, jump = kernel.mutate_traj(sub, Z[jnp.asarray(idx)], beta, eps, jnp.asarray(m),
                                                     jnp.asarray(L), data, n_moves=P - 1)
        n_grad += M * ((P - 1) * cfg.n_leapfrog + 1)
        iid = False
        ll = np.asarray(ll_d)
        acc_m = float(np.mean(np.asarray(acc)))
        jump_m = float(np.mean(np.asarray(jump)))
        hist.append(dict(stage=stage, beta=beta, ess=ess, accept=acc_m, eps=eps, jump=jump_m, logz=logz,
                         n_unique=int(np.unique(idx).size), t=time.perf_counter() - t0))
        log(f"{tag}stage {stage:3d}: β {beta:.4e}  ESS {ess:7.0f}  ancestors {hist[-1]['n_unique']:6d}/{M}  "
            f"accept {acc_m:.2f}  ε {eps:.3f}  jump {jump_m:.3f}  log Z {logz:.3f}  ({hist[-1]['t']:.0f} s)")
        eps = float(eps * np.exp(np.clip(2.0 * (acc_m - cfg.target_accept), -0.7, 0.7)))
        if beta >= 1.0:
            break
    else:
        raise RuntimeError(f"SMC did not reach β = 1 in {cfg.max_stages} stages (β = {beta:.3g})")
    X = np.asarray(kernel.x_of(Z))
    return dict(logz=logz, logz_sd_ess=float(np.sqrt(var_logz)), x=X, loglike=ll, n_stages=len(hist), hist=hist,
                wall=time.perf_counter() - t0, n_like_grad=int(n_grad))


def log_separable_mass(view) -> float:
    """log ∫ exp(separable_logprior) dx over the boxes of ``view``: Π (box width) over uniform
    parameters × Π √(2π) σ [Φ(b) − Φ(a)] over the (truncated) Gaussian ones — the
    normalization of ``log_prior_fn`` up to the mass of its cuts / joint terms."""
    from scipy.special import log_ndtr

    lo, hi = np.asarray(view.prior_lo, dtype=float), np.asarray(view.prior_hi, dtype=float)
    mu, sd = np.asarray(view.gauss_mu, dtype=float), np.asarray(view.gauss_sd, dtype=float)
    out = 0.0
    for i in range(lo.size):
        if np.isfinite(sd[i]):
            a, b = (lo[i] - mu[i]) / sd[i], (hi[i] - mu[i]) / sd[i]
            la, lb = log_ndtr(a), log_ndtr(b)
            out += float(np.log(np.sqrt(2 * np.pi) * sd[i]) + lb + np.log1p(-np.exp(la - lb)))
        else:
            out += float(np.log(hi[i] - lo[i]))
    return out


def to_logit(x: np.ndarray, lo, hi) -> np.ndarray:
    u = np.clip((np.asarray(x) - lo) / (hi - lo), 1e-15, 1 - 1e-15)
    return np.log(u) - np.log1p(-u)


def fit_reference(z: np.ndarray, nu: float, scale: float) -> dict:
    """Multivariate Student-t (``nu`` degrees of freedom) in the logit coordinates with the draws'
    mean and covariance times scale squared: the start of the reference path."""
    from scipy.special import gammaln

    d = z.shape[1]
    mean = z.mean(axis=0)
    cov = np.cov(z, rowvar=False) * scale**2 + 1e-12 * np.eye(d)
    chol = np.linalg.cholesky(cov)
    log_norm = float(gammaln(0.5 * (nu + d)) - gammaln(0.5 * nu) - 0.5 * d * np.log(nu * np.pi)
                     - np.sum(np.log(np.diag(chol))))
    return dict(mean=mean, chol=chol, chol_inv=np.linalg.inv(chol), nu=float(nu), log_norm=log_norm)


def sample_reference(ref: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    d = ref["mean"].size
    g = rng.standard_normal((n, d))
    w = rng.chisquare(ref["nu"], size=(n, 1))
    return ref["mean"] + (g * np.sqrt(ref["nu"] / w)) @ ref["chol"].T


def run_dali_smc(bundle, ll_fn: Callable, data, cfg: DALISMCConfig, out_prefix: Path,
                      log: Callable[[str], None] = print) -> dict:
    """Tempered SMC on the DALI posterior; ``ll_fn(x, data)`` is the (jittable,
    differentiable) DALI log L. Writes ``<out_prefix>.ns_posterior.npz`` (posterior draws),
    ``.smc_particles.npz`` (all final particles) and ``.ns_summary.json``."""
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    names = list(bundle.param_names)
    view, log_vol_ratio = narrowed_prior_view(bundle, cfg.sampling_box)
    lo, hi = np.asarray(view.prior_lo, dtype=float), np.asarray(view.prior_hi, dtype=float)
    if cfg.sampling_box:
        log(f"sampling boxes {cfg.sampling_box}: log(V_narrow/V_box) = {log_vol_ratio:.4f}")
    rng = np.random.default_rng(cfg.seed)
    _, p_box = sample_prior(bundle, cfg.n_prior_mc, rng)
    _, p_nar = sample_prior(view, cfg.n_prior_mc, rng)
    pc_box, pc_nar = p_box["accept_fraction"], p_nar["accept_fraction"]
    pc_err = float(np.hypot(np.sqrt(pc_box * (1 - pc_box) / p_box["n_tried"]) / pc_box,
                            np.sqrt(pc_nar * (1 - pc_nar) / p_nar["n_tried"]) / pc_nar))
    log(f"prior mass of the hard cuts: box {pc_box:.5f}, sampling box {pc_nar:.5f} (rel. err {pc_err:.1e})")
    log(f"SMC: {cfg.n_runs} runs × {cfg.n_particles} particles, ESS target {cfg.ess_target}, "
        f"{cfg.n_mcmc} HMC moves × {cfg.n_leapfrog} leapfrog per stage ({cfg.n_final_mcmc} at β = 1), "
        f"chunk {cfg.chunk}")
    corr = log_vol_ratio + float(np.log(pc_nar)) - float(np.log(pc_box))
    t0 = time.perf_counter()
    reference = None
    kernel = _Kernel(view, ll_fn, lo, hi, cfg)  # one compiled program: prior start, pilot and reference runs
    if cfg.reference == "prior":
        offset = 0.0
    else:
        if cfg.reference == "pilot":
            from dataclasses import replace

            cfg_p = replace(cfg, ess_target=cfg.pilot_ess)  # host-side tempering only
            xp, _ = sample_prior(view, cfg.pilot_particles, rng)
            rp = smc_run(kernel, xp, data, cfg_p, seed=cfg.seed * 1000, log=log, tag="[pilot] ")
            zr = to_logit(rp["x"], lo, hi)
            log(f"pilot: {cfg.pilot_particles} particles, {rp['n_stages']} stages, {rp['wall']:.0f} s; "
                f"log Z_DALI = {rp['logz'] + corr:.3f} (± {rp['logz_sd_ess']:.3f})")
            reference = dict(source="pilot", n=cfg.pilot_particles, n_stages=rp["n_stages"], wall=rp["wall"],
                             logz_dali=rp["logz"] + corr)
        else:
            zr = to_logit(np.load(cfg.reference)["pos"], lo, hi)
            reference = dict(source=str(cfg.reference), n=int(zr.shape[0]))
        ref = fit_reference(zr, cfg.reference_nu, cfg.reference_scale)
        reference.update(nu=cfg.reference_nu, scale=cfg.reference_scale)
        kernel.set_reference(ref)
        # the path ends at ∫ π̃ L (π̃ = exp(log_prior_fn), unnormalized): divide by ∫ π̃ over the boxes
        offset = -log_separable_mass(view) - float(np.log(pc_nar))
    runs = []
    for r in range(cfg.n_runs):
        if reference is None:
            x0, _ = sample_prior(view, cfg.n_particles, rng)
            res = smc_run(kernel, x0, data, cfg, seed=cfg.seed * 1000 + r + 1, log=log, tag=f"[run {r}] ")
        else:
            res = smc_run(kernel, None, data, cfg, seed=cfg.seed * 1000 + r + 1, log=log, tag=f"[run {r}] ",
                          z0=sample_reference(ref, cfg.n_particles, rng))
            res["logz"] += offset
        log(f"run {r}: log Z_SMC = {res['logz']:.4f} (± {res['logz_sd_ess']:.3f} from the weights' ESS) in "
            f"{res['n_stages']} stages, {res['wall']:.0f} s")
        runs.append(res)
        if cfg.target_err is not None and len(runs) >= max(2, cfg.min_runs):
            lz_ = np.asarray([q["logz"] for q in runs])
            sd_ = max(float(np.std(lz_, ddof=1)), float(np.sqrt(np.mean([q["logz_sd_ess"] ** 2 for q in runs]))))
            err_ = float(np.hypot(sd_ / np.sqrt(len(runs)), pc_err))
            if err_ <= cfg.target_err:
                log(f"log Z_DALI error {err_:.4f} ≤ target {cfg.target_err} after {len(runs)} runs: stopping")
                break
    wall = time.perf_counter() - t0
    lz = np.asarray([r["logz"] for r in runs])
    m = lz.max()
    logz_smc = float(m + np.log(np.mean(np.exp(lz - m))))
    # Per-run error: the larger of the runs' scatter (few runs: a noisy estimate, easily 2–3×
    # low) and the incremental weights' prediction Σ_t (N/ESS_t − 1)/N (exact for perfectly
    # mixing moves, a lower bound otherwise).
    sd_run = float(np.std(lz, ddof=1)) if lz.size > 1 else 0.0
    sd_ess = float(np.sqrt(np.mean([r["logz_sd_ess"] ** 2 for r in runs])))
    sd_use = max(sd_run, sd_ess)
    X = np.concatenate([r["x"] for r in runs])
    LL = np.concatenate([r["loglike"] for r in runs])
    keep = np.isfinite(LL)
    mean = X[keep].mean(axis=0)
    std = X[keep].std(axis=0)
    edge = {}
    lo0, hi0 = np.asarray(bundle.prior_lo, dtype=float), np.asarray(bundle.prior_hi, dtype=float)
    for nm in (cfg.sampling_box or {}):
        i = names.index(nm)
        dd = []
        if lo[i] > lo0[i] + 1e-12 * (hi0[i] - lo0[i]):
            dd.append((mean[i] - lo[i]) / max(std[i], 1e-300))
        if hi[i] < hi0[i] - 1e-12 * (hi0[i] - lo0[i]):
            dd.append((hi[i] - mean[i]) / max(std[i], 1e-300))
        if dd:
            edge[nm] = float(min(dd))
    edge_ok = bool(all(v >= cfg.min_edge_sigma for v in edge.values()))
    if edge:
        log(f"posterior distance to the narrowed edges (σ): {edge}" + ("" if edge_ok else
            f"  WARNING: < {cfg.min_edge_sigma}σ, the volume correction is not valid; widen the box"))
    sel = rng.choice(np.flatnonzero(keep), size=min(cfg.n_posterior, int(keep.sum())), replace=False)
    np.savez(out_prefix.with_suffix(".ns_posterior.npz"), pos=X[sel], loglike=LL[sel], names=np.asarray(names))
    np.savez(out_prefix.with_suffix(".smc_particles.npz"), pos=X, loglike=LL, run=np.repeat(
        np.arange(len(runs)), [r["x"].shape[0] for r in runs]), names=np.asarray(names))
    summary = dict(
        sampler="smc_on_dali", config=asdict(cfg), dim=len(names), names=names,
        logz_runs=lz.tolist(), logz_run_std=sd_run, logz_run_std_ess=sd_ess, logz_smc=logz_smc,
        logz_dali=logz_smc + corr, logz_dali_std=float(np.hypot(sd_use / np.sqrt(lz.size), pc_err)),
        log_volume_ratio=log_vol_ratio, prior_cut_mass=pc_box, prior_cut_mass_sampling_box=pc_nar,
        sampling_box=cfg.sampling_box, edge_sigma=edge, edge_ok=edge_ok,
        n_stages=[r["n_stages"] for r in runs], n_like_grad=int(sum(r["n_like_grad"] for r in runs)),
        run_wall_sec=[r["wall"] for r in runs], n_points=int(X.shape[0]),
        n_eff=int(sum(np.unique(r["x"], axis=0).shape[0] for r in runs)),  # distinct final particles
        n_nonfinite_final=int((~keep).sum()), loglike_max=float(LL[keep].max()),
        posterior_mean=mean.tolist(), posterior_std=std.tolist(), wall_sec=wall,
        hist=[r["hist"] for r in runs], loglike_ceiling=cfg.loglike_ceiling, reference=reference,
    )
    out_prefix.with_suffix(".ns_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log(f"SMC done: log Z_DALI = {summary['logz_dali']:.3f} ± {summary['logz_dali_std']:.3f} (runs {np.round(lz, 3).tolist()}, "
        f"scatter {sd_run:.3f}, ESS prediction {sd_ess:.3f} per run), {summary['n_like_grad']:.3g} log L + gradient "
        f"evaluations, {wall:.0f} s")
    return summary


__all__ = ["DALISMCConfig", "fit_reference", "log_separable_mass", "next_temperature", "run_dali_smc",
           "sample_reference", "smc_run", "systematic_resample"]
