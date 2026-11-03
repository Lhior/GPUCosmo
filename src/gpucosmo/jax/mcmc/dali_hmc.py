#!/usr/bin/env python3
"""DALI-trajectory HMC: leapfrog on a DALI posterior, accept/reject on a chosen target.

Leapfrog integration under *any* fixed potential is reversible and volume preserving, so
proposing with a DALI potential U_s and accepting with the Hamiltonian H = U(q) + K(p) of a
target posterior leaves that target invariant (e.g. Neal 2011, §5.5 "HMC with approximate
gradients"); a trajectory potential that differs from the target only lowers the acceptance
rate. The target (``accept``):

* ``"dali"`` (default): the DALI posterior (``dali_target_logpost``, default the trajectory's own
  ``dali_logpost``), evaluated at the end points — no exact likelihood call, the same
  posterior the evidence pipeline samples;
* ``"exact"``: the exact posterior (``exact_logpost_batch``), one batched exact evaluation per
  iteration, so the chains target the exact posterior whatever the DALI expansion's error.

The DALI expansion and the (post-warm-up) step size must not depend on the chain state; step
size is adapted only during warm-up (dual averaging on the acceptance), then frozen.
"""

from __future__ import annotations

import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def _dual_averaging_step(state, accept_prob, target=0.7, gamma=0.05, t0=10.0, kappa=0.75):
    log_eps, log_eps_bar, h_bar, mu, t = state
    t = t + 1
    h_bar = (1 - 1 / (t + t0)) * h_bar + (target - accept_prob) / (t + t0)
    log_eps = mu - np.sqrt(t) / gamma * h_bar
    w = t ** (-kappa)
    log_eps_bar = w * log_eps + (1 - w) * log_eps_bar
    return (log_eps, log_eps_bar, h_bar, mu, t)


# Jitted proposal per (DALI, chains, dim, path length): converge segments call
# run_dali_hmc repeatedly with the same DALI, and rebuilding the closure
# recompiled the trajectory (≈ 6 min) at every segment. The DALI expansion is kept
# alive in the value so its id cannot be reused.
_PROPOSE_CACHE: dict = {}


def _get_propose(dali_logpost, n_chains, dim, n_leapfrog, with_args: bool = False):
    """Jitted proposal ``propose(key, q, eps, inv_m, m_chol, sargs)``.

    ``with_args``: the DALI expansion is ``dali_logpost(q, sargs)`` and its data
    (expansion coefficients, likelihood arrays such as PR4's 3.3 GB matrix) arrive as
    the jit argument ``sargs``. Captured in a closure instead, JAX embeds them in
    the compiled program as constants: a second device copy of every array.
    """
    key = (id(dali_logpost), n_chains, dim, n_leapfrog, bool(with_args))
    hit = _PROPOSE_CACHE.get(key)
    if hit is not None and hit[0] is dali_logpost:
        return hit[1]
    if with_args:
        vg_s = jax.value_and_grad(dali_logpost)
    else:
        vg_s = jax.value_and_grad(lambda q, _a: dali_logpost(q))

    def kinetic(p, inv_m):
        return 0.5 * p @ (inv_m @ p)

    def trajectory(q, p, eps, n_steps, inv_m, sargs):
        """Leapfrog on the DALI expansion; a non-finite step freezes and invalidates it.

        The (value, gradient) at the end of a step is carried into the next one, so
        a trajectory costs n_steps + 1 DALI evaluations (it was 2·n_leapfrog + 2:
        two gradients per step, masked steps past n_steps, and separate end-point
        values), and its end-point values come for free.
        """
        l0, g0 = vg_s(q, sargs)

        def body(_, carry):
            q, p, g, l, ok = carry
            p_half = p + 0.5 * eps * g
            q_new = q + eps * (inv_m @ p_half)
            l_new, g_new = vg_s(q_new, sargs)
            p_new = p_half + 0.5 * eps * g_new
            step = ok & jnp.all(jnp.isfinite(q_new)) & jnp.all(jnp.isfinite(p_new))
            keep = lambda a, b: jnp.where(step, a, b)  # noqa: E731
            return (keep(q_new, q), keep(p_new, p), keep(g_new, g), keep(l_new, l), step)

        q1, p1, _, l1, ok = jax.lax.fori_loop(0, n_steps, body,
                                              (q, p, g0, l0, jnp.asarray(True)))
        return q1, p1, ok, l0, l1

    @jax.jit
    def propose(key, q, eps, inv_m, m_chol, sargs=None):
        # Mass matrix passed as an argument: adapting it during warm-up does not
        # trigger a recompile.
        k1, k2, k3 = jax.random.split(key, 3)
        z = jax.random.normal(k1, q.shape)
        p0 = z @ m_chol.T
        # One jittered trajectory length per iteration, shared by the chains (it is
        # independent of the state, so the kernel stays exact): the vmapped loop then
        # stops after n_steps instead of always running n_leapfrog masked steps.
        n_steps = jax.random.randint(k2, (), n_leapfrog // 2, n_leapfrog + 1)
        eps_j = eps * jax.random.uniform(k3, (n_chains,), minval=0.8, maxval=1.2)
        q1, p1, ok, ls0, ls1 = jax.vmap(trajectory, in_axes=(0, 0, 0, None, None, None))(
            q, p0, eps_j, n_steps, inv_m, sargs)
        k0 = jax.vmap(kinetic, in_axes=(0, None))(p0, inv_m)
        k1_ = jax.vmap(kinetic, in_axes=(0, None))(p1, inv_m)
        return q1, k0, k1_, ls0, ls1, ok, n_steps

    _PROPOSE_CACHE[key] = (dali_logpost, propose)
    return propose


ACCEPT_MODES = ("dali", "exact")


def run_dali_hmc(
    *,
    dali_logpost: Callable[[jnp.ndarray], jnp.ndarray],
    exact_logpost_batch: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    x_init: np.ndarray,
    inv_mass: np.ndarray,
    n_iter: int,
    n_leapfrog: int = 20,
    n_warmup: int | None = None,
    step_size: float = 0.13,
    target_accept: float = 0.7,
    seed: int = 0,
    out_path: Path | None = None,
    param_names: list[str] | None = None,
    save_every: int = 10,
    adapt_mass_at: tuple[int, ...] = (),
    rng_state: dict | None = None,
    dali_args=None,
    accept: str = "dali",
    dali_target_logpost: Callable | None = None,
) -> dict:
    """Run ``x_init.shape[0]`` chains; returns (and saves) samples and diagnostics.

    Resuming: pass the previous result's final state (``x_init=res["x_last"]``,
    ``inv_mass=res["inv_mass"]``, ``step_size=res["eps_final"]``, ``n_warmup=0``,
    ``rng_state=res["rng_state"]``). The continuation is then the same Markov
    chain, draw for draw, as one uninterrupted run.

    ``dali_args``: pytree passed as ``dali_logpost(q, dali_args)`` (as a
    jit argument, so large arrays are not embedded in the compiled trajectory).

    ``accept``: ``"dali"`` accepts on ``dali_target_logpost`` (same call signature as
    ``dali_logpost``; default ``dali_logpost`` itself), e.g. the DALI posterior with a
    likelihood the trajectory smooths for its gradients; ``"exact"`` on
    ``exact_logpost_batch``.
    """
    if accept not in ACCEPT_MODES:
        raise ValueError(f"accept must be one of {ACCEPT_MODES}, not {accept!r}")
    if accept == "exact" and exact_logpost_batch is None:
        raise ValueError("accept='exact' needs exact_logpost_batch")
    target = dali_logpost if dali_target_logpost is None else dali_target_logpost
    if dali_args is not None:
        _target_b = jax.jit(jax.vmap(target, in_axes=(0, None)))
    else:
        _target_b = jax.jit(jax.vmap(lambda q, _a: target(q), in_axes=(0, None)))
    accept_logpost_batch = (exact_logpost_batch if accept == "exact"
                            else (lambda q: _target_b(q, dali_args)))
    x = jnp.asarray(x_init, dtype=jnp.float64)
    n_chains, dim = x.shape
    n_warmup = n_iter // 4 if n_warmup is None else n_warmup
    adapt_mass_at = tuple(sorted(a for a in adapt_mass_at if 0 < a < n_warmup))
    if adapt_mass_at and n_warmup - adapt_mass_at[-1] < 50:
        print(f"warning: only {n_warmup - adapt_mass_at[-1]} step-size iterations after the "
              f"last mass adaptation ({adapt_mass_at[-1]}); want ≥ 50", flush=True)
    inv_mass = jnp.asarray(inv_mass, dtype=jnp.float64)

    def chol_mass(inv_m):
        return jnp.linalg.cholesky(jnp.linalg.inv(inv_m))  # p = L z ~ N(0, M)

    mass_chol = chol_mass(inv_mass)
    propose = _get_propose(dali_logpost, n_chains, dim, n_leapfrog,
                           with_args=dali_args is not None)
    t_start = time.perf_counter()
    eps_lo, eps_hi = 0.02, 1.5  # whitened units; stops dual averaging collapsing
    # The clamp applies to warm-up tuning only; a frozen (resumed) eps is kept as is.
    eps = float(np.clip(step_size, eps_lo, eps_hi)) if n_warmup > 0 else float(step_size)
    # Compile the trajectory (a throwaway call with a fixed key, so the chain's random
    # stream is untouched) on a second thread while the exact batch compiles: XLA
    # compiles in C++ without the GIL, and the two programs are independent.
    with ThreadPoolExecutor(max_workers=2) as pool:
        warm = pool.submit(lambda: jax.block_until_ready(
            propose(jax.random.PRNGKey(0), x, eps, inv_mass, mass_chol, dali_args)))
        lp = np.asarray(accept_logpost_batch(x))
        warm.result()
    t_init_exact = time.perf_counter() - t_start  # both compilations (overlapped)

    key = jax.random.PRNGKey(seed)
    rng = np.random.default_rng(seed)
    if rng_state is not None:
        key = jnp.asarray(rng_state["key"], dtype=jnp.uint32)
        rng.bit_generator.state = rng_state["np"]
    print(f"DALI-HMC: {n_chains} chains × {n_iter} it (warm-up {n_warmup}), "
          f"dim {dim}; accept/reject on the {accept} posterior; initial log-post {np.round(lp, 2)}",
          flush=True)
    da = (np.log(eps), 0.0, 0.0, np.log(10 * eps), 0.0)
    t_prop = t_exact = 0.0
    samples, logps, accepts, eps_hist, dH_sur, lf_steps = [], [], [], [], [], []
    # proposals whose target log posterior was not finite: outside the prior (physical / prior
    # exclusion) or a failed evaluation (e.g. the CMB solver, ``cls_jax.CLS_SOURCE_FAILURE``); both are
    # rejected, and ``bundle.cmb_solver_status`` tells them apart
    n_exact_nonfinite = 0
    for it in range(n_iter):
        key, sub = jax.random.split(key)
        t0 = time.perf_counter()
        q1, k0, k1, ls0, ls1, ok, n_lf = propose(sub, x, eps, inv_mass, mass_chol, dali_args)
        q1 = jax.block_until_ready(q1)
        t1 = time.perf_counter()
        ok = np.asarray(ok)
        # Invalid (non-finite) trajectories are rejected without evaluating the target.
        q_eval = jnp.where(jnp.asarray(ok)[:, None], q1, x)
        lp1 = np.asarray(accept_logpost_batch(q_eval))
        t2 = time.perf_counter()
        if it > 0:  # first iteration includes compilation
            t_prop += t1 - t0
            t_exact += t2 - t1
        else:
            t_first = (t1 - t0, t2 - t1)
        k0, k1, ls0, ls1 = map(np.asarray, (k0, k1, ls0, ls1))
        log_alpha = (lp1 - k1) - (lp - k0)
        n_exact_nonfinite += int(np.sum(ok & ~np.isfinite(lp1)))
        log_alpha = np.where(np.isfinite(lp1) & ok, log_alpha, -np.inf)
        acc = np.log(rng.uniform(size=n_chains)) < log_alpha
        x = jnp.where(jnp.asarray(acc)[:, None], q1, x)
        lp = np.where(acc, lp1, lp)
        a_prob = float(np.mean(np.exp(np.minimum(log_alpha, 0.0))))
        if it < n_warmup and (it + 1) in adapt_mass_at:
            # Warm-up only: empirical covariance of all chains since the last
            # adaptation, shrunk toward its diagonal; then restart step-size tuning.
            start = max([a for a in adapt_mass_at if a < it + 1] + [0])
            win = np.concatenate(samples[start:], axis=0)
            emp = np.cov(win.T)
            emp = 0.9 * emp + 0.1 * np.diag(np.diag(emp)) + 1e-6 * np.eye(dim)
            ratio = np.sqrt(np.diag(emp) / np.diag(np.asarray(inv_mass)))
            inv_mass = jnp.asarray(emp)
            mass_chol = chol_mass(inv_mass)
            # Restart step-size tuning *from the current eps* (log ε̄ = log eps):
            # resetting ε̄ to 1 left the final averaged eps dominated by the
            # erratic first post-reset iterates (froze at 0.07 instead of ≈1).
            da = (np.log(eps), np.log(eps), 0.0, np.log(10 * eps), 0.0)
            print(f"  [adapt mass at {it + 1}] from {win.shape[0]} samples; "
                  f"σ_new/σ_old median {np.median(ratio):.2f} "
                  f"(min {ratio.min():.2f}, max {ratio.max():.2f})", flush=True)
        if it < n_warmup:
            da = _dual_averaging_step(da, a_prob, target=target_accept)
            eps = float(np.clip(np.exp(da[0]), eps_lo, eps_hi))
            if it == n_warmup - 1:
                eps_bar = float(np.exp(da[1]))
                eps_med = float(np.median(eps_hist[-20:] + [eps]))
                if not (eps_med / 3 <= eps_bar <= 3 * eps_med):
                    print(f"  averaged eps {eps_bar:.3f} disagrees with recent median "
                          f"{eps_med:.3f}; using the median", flush=True)
                    eps_bar = eps_med
                eps = float(np.clip(eps_bar, eps_lo, eps_hi))
                print(f"  warm-up done: eps frozen at {eps:.3f}", flush=True)
        samples.append(np.asarray(x))
        logps.append(lp.copy())
        accepts.append(acc)
        eps_hist.append(eps)
        lf_steps.append(int(n_lf))
        dh = (ls1 - k1) - (ls0 - k0)
        dH_sur.append(np.nanmean(np.where(ok & np.isfinite(dh), dh, np.nan)))
        if (it + 1) % save_every == 0 or it == n_iter - 1:
            el = time.perf_counter() - t_start
            n_t = max(it, 1)
            print(f"[{it + 1:5d}/{n_iter}] {el:7.0f}s  per-it: DALI traj "
                  f"{t_prop / n_t:.2f}s + {accept} accept {t_exact / n_t:.2f}s  "
                  f"acc {np.mean(accepts[-save_every:]):.2f} eps {eps:.3f} "
                  f"DALI dH {np.nanmean(dH_sur[-save_every:]):+.3f} "
                  f"logpost {np.mean(lp):.2f}"
                  + (f"  non-finite {accept} {n_exact_nonfinite}" if n_exact_nonfinite else ""), flush=True)
            if out_path is not None:
                np.savez(out_path, samples=np.stack(samples), logpost=np.stack(logps),
                         accept=np.stack(accepts), eps=np.asarray(eps_hist),
                         n_warmup=n_warmup, names=np.asarray(param_names or []),
                         inv_mass=np.asarray(inv_mass))
    return dict(samples=np.stack(samples), logpost=np.stack(logps),
                accept=np.stack(accepts), eps=np.asarray(eps_hist),
                n_steps=np.asarray(lf_steps, dtype=np.int32), n_exact_nonfinite=n_exact_nonfinite, accept_mode=accept,
                x_last=np.asarray(x), eps_final=eps, inv_mass=np.asarray(inv_mass),
                rng_state=dict(key=np.asarray(key), np=rng.bit_generator.state),
                t_per_it=(t_prop + t_exact) / max(n_iter - 1, 1),
                t_prop_per_it=t_prop / max(n_iter - 1, 1), t_exact_per_it=t_exact / max(n_iter - 1, 1),
                t_first_prop=t_first[0], t_first_exact=t_first[1], t_init_exact=t_init_exact)


__all__ = ["ACCEPT_MODES", "run_dali_hmc"]
