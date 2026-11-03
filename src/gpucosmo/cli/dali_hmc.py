"""``gpucosmo-dali-hmc``: DALI-trajectory HMC on the GPU likelihood bundles (MAP, DALI build, sampling).

Stages (artifacts under ``--outdir``):

``map``     Iterated linearization: expand the theory to first order at x_k (one
            exact Jacobian, which also yields the exact logL at x_k), maximize the
            DALI posterior over all parameters (L-BFGS-B, whitened by the
            DALI Hessian, cosmology kept inside a trust region), repeat.
``build``   DALI expansion of the theory at the MAP to order ``--dali-order`` (default 3, the
            triplet) + DALI-Hessian mass matrix.
``sample``  DALI-trajectory HMC: leapfrog on the DALI posterior, accept/reject on the DALI
            posterior with the official likelihoods (``--accept dali``, default: no exact
            likelihood call), or on the exact posterior (``--accept exact``: one batched
            exact theory call per iteration, so the chains target the exact posterior).
``converge`` ``sample`` in resumable segments until every parameter has split-R̂ <
            ``--rhat`` and bulk/tail ESS ≥ ``--min-ess`` (or ``--max-iters``);
            ``--resume`` continues a killed run from ``converge_state.npz``.

Usage (``scripts/run_dali_hmc.py`` is the same program)::

    gpucosmo-dali-hmc --config my_analysis.yaml --outdir OUT --stage map
    gpucosmo-dali-hmc --config my_analysis.yaml --outdir OUT --stage build
    gpucosmo-dali-hmc --analysis mg_desi_cmb_pr4_actplanck --outdir OUT --stage sample --chains 8 --iters 2000
    gpucosmo-dali-hmc --stage converge --outdir outputs/shmc_sn \
        --init-from outputs/shmc/long1_chain_z.npz --init-center z_map

``--plugin MODULE`` imports a module that registers extra likelihoods (installed plugins
register through the ``gpucosmo.likelihoods`` entry point and need no flag).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from gpucosmo.jax.inference.dali_expansion import DALI_ORDER
from gpucosmo.pipeline import analysis_configs, build_analysis_bundle, enable_jax_cache, lens_override

jax.config.update("jax_enable_x64", True)
# Persistent XLA cache: relaunches skip the multi-minute theory compiles (~4.5 min for
# the exact theory program). Every pipeline / benchmark script imports this module
# before compiling anything. 2 s also keeps the likelihood and DALI programs.
enable_jax_cache()

THEORY_BUNDLE_NAMES = ("logA", "ns", "theta_MC_100", "ombh2", "omch2", "tau", "w", "wa")


# μ–Σ modified-gravity analyses of arXiv:2411.12026 (``inference.mg_bundle``): one
# Cobaya yaml per analysis shipped in gpucosmo/configs/cobaya (the official DESI chain
# configs, scripts/make_mg_cobaya_configs.py); ``--config`` takes any other Cobaya yaml.
MG_ANALYSES = analysis_configs()


def _legacy_order(order: int) -> int:
    if order > 2:
        raise SystemExit("the desi_full bundle's CMB DALITheory is second order: pass --dali-order 2")
    return order


def _cmb_dali_adapter(b, tables):
    """DALI adapter for the desi_full bundle: the 8-parameter CMB DALITheory."""
    from types import SimpleNamespace

    from gpucosmo.jax.cmb.dali_theory import (
        DEFAULT_STEPS,
        DALITheory,
        build_dali_theory,
        dali_spectra,
    )

    steps = np.array([DEFAULT_STEPS[n] for n in
                      ("logA", "ns", "theta_mc_100", "ombh2", "omch2", "tau", "w0", "wa")])
    return SimpleNamespace(
        n=8, steps=steps, names=list(THEORY_BUNDLE_NAMES),
        build=lambda x8, order=2, verbose=True, **_: build_dali_theory(
            tables, x8, lmax=b.lmax, second_order=_legacy_order(order) >= 2, verbose=verbose),
        theory=lambda tt, x: dali_spectra(tt, tables, x[:8]),
        save=lambda tt, path: tt.save(path),
        load=lambda path: DALITheory.load(path))


def other_programs_bytes(args, b, tables) -> int:
    """Device bytes of the programs that stay loaded beside the exact theory during
    sampling: the likelihood/prior program and the DALI expansion trajectory (AOT
    compiled here; the persistent compile cache makes the later compiles hits)."""
    from gpucosmo.jax.inference.autosize import footprint_bytes
    from gpucosmo.jax.mcmc.dali_hmc import _get_propose

    n, dim = int(args.chains), len(b.param_names)
    X = jnp.asarray(np.tile(np.asarray(b.fiducial), (n, 1)))
    th = jax.eval_shape(b.theory_batch, X[:, : b.n_theory])
    like = b.like_program.lower(X, X, th, b.aux).compile()
    lp_s, _, _, _, tt = sampling_targets(args, b, tables)
    prop = _get_propose(lp_s, n, dim, int(args.leapfrog), with_args=True).lower(
        jax.random.PRNGKey(0), jnp.zeros((n, dim)), 0.1, jnp.eye(dim), jnp.eye(dim), tt).compile()
    sizes = [footprint_bytes(c, arguments=False) for c in (like, prop)]
    print(f"other programs: likelihood {sizes[0] / 2**30:.2f} GB, trajectory "
          f"{sizes[1] / 2**30:.2f} GB", flush=True)
    return sum(sizes)


def setup(args):
    """Bundle with a ``dali`` adapter: ``n`` leading theory parameters, ``steps``,
    ``build(x_th, order, verbose)``, ``theory(sur, x)``, ``save``, ``load``."""
    import dataclasses
    from types import SimpleNamespace

    from gpucosmo.jax.background.cosmology import load_background_tables

    tables = load_background_tables()
    if args.config or args.analysis in MG_ANALYSES:
        from gpucosmo.jax.inference.cobaya_yaml import dump_resolved

        over = {k: v for k, v in (("cmb_walker_batch", args.cmb_walker_batch),
                                  ("pk_walker_batch", args.pk_walker_batch)) if v is not None}
        if args.stage == "build" and args.expansion_only:
            # The DALI build evaluates the theory only: HiLLiPoP's 3.3 GB matrix stays off the GPU, and
            # the freed memory takes 16-cosmology CMB / P(k) batches (MG: 129 evaluations 165 → 142 s warm).
            over["defer_hillipop_matrix"] = True
            over.setdefault("cmb_walker_batch", 16)
            over.setdefault("pk_walker_batch", 16)
        b, res, diffs = build_analysis_bundle(
            None if args.config else args.analysis, args.config, walker_batch=args.walker_batch,
            bbn_prior=args.bbn_prior, cmb_nonlinear_lens=lens_override(args),
            cmb_lensing=getattr(args, "cmb_lensing", False) or None,
            fs_tables=args.fs_tables, cmb_legacy_lensing=getattr(args, "cmb_legacy_lensing", False),
            # Map/build stages run other programs (and its own DALI build); they use
            # the configured batches. Sampling stages measure the programs that stay
            # loaded and size the exact theory to what is left.
            autosize=not args.no_autosize and args.stage in ("sample", "converge") and args.accept == "exact",
            n_chains=args.chains, batch_overrides=over or None,
            reserved_fn=(lambda bb: other_programs_bytes(args, bb, bb.tables))
            if args.stage in ("sample", "converge") and args.accept == "exact" else None,
            frag=args.autosize_frag, plugins=args.plugin)
        print(f"theory block {b.dali.names}", flush=True)
        dump_resolved(b, res, args.outdir / f"config.updated.{args.stage}.yaml",
                      extra={"audit_diffs": diffs, "cli": {k: str(v) for k, v in vars(args).items()}})
        return b, b.tables
    from gpucosmo.jax.cmb.bundle import build_desi_full_gpu_bundle

    kw = dict(include_sn=False, sn_dataset="") if args.no_sn else dict(sn_dataset=args.sn_dataset)
    b = build_desi_full_gpu_bundle(tables=tables, walker_batch=args.walker_batch, **kw)
    idx = [b.param_names.index(n) for n in THEORY_BUNDLE_NAMES]
    assert idx == list(range(8)), idx
    b = SimpleNamespace(**{f.name: getattr(b, f.name) for f in dataclasses.fields(b)})
    b.dali = _cmb_dali_adapter(b, tables)
    return b, tables


def dali_logpost_fn(b, tables, fixed=None, smooth: bool = True):
    """``lp(x, tt)``: DALI log-posterior with the DALI-expanded theory as an argument.

    Taking ``tt`` as an (array pytree) argument rather than a closure lets the
    jitted value/grad/Hessian compile once and be reused for every re-expansion.
    ``smooth``: the differentiable SimAll low-ℓ EE (for gradients); ``False`` the official
    stepwise one (the accept/reject target).
    """
    S = b.dali
    fixed = fixed or {}
    # Non-differentiable user likelihoods (Cobaya bridge): central differences on the DALI-expanded theory.
    nd = bool(getattr(getattr(b, "internals", None), "nondiff", ()))

    def lp(x, tt):
        for i, v in fixed.items():
            x = x.at[i].set(v)
        # smooth=True: differentiable SimAll low-ℓ EE (clipy's table lookup has
        # zero gradient, so trajectories never saw the τ constraint). The accept/reject
        # (smooth=False, or exact) keeps the official stepwise SimAll as the target.
        if nd:
            like = (b.log_like_from_theory_fn(x, S.theory(tt, x), smooth=smooth, nondiff="skip")
                    + b.nondiff_loglike(x, lambda xx: S.theory(tt, xx)))
        else:
            like = b.log_like_from_theory_fn(x, S.theory(tt, x), smooth=smooth)
        return like + b.log_prior_fn(x)

    return lp


def _hessian_fn(f, vg, b, steps=None):
    """Hessian of ``f(x, tt)``: ``jax.hessian`` for the CMB bundle; for the MG bundles
    central differences of the compiled gradient ``vg`` (forward-over-reverse
    through REPT for ~25 parameters is a very long compile)."""
    if not hasattr(b, "config"):
        return jax.jit(jax.hessian(f))

    def h(x, tt):
        x = jnp.asarray(x)
        n = x.shape[0]
        e = np.asarray(steps) if steps is not None else 1e-4 * (1.0 + np.abs(np.asarray(x)))
        rows = []
        for i in range(n):
            d = jnp.zeros(n).at[i].set(e[i])
            gp = np.asarray(vg(x + d, tt)[1])
            gm = np.asarray(vg(x - d, tt)[1])
            rows.append((gp - gm) / (2 * e[i]))
        H = np.stack(rows)
        return 0.5 * (H + H.T)

    return h


def stage_map(args, b, tables):
    S = b.dali
    names = list(b.param_names)
    lo, hi = np.asarray(b.prior_lo), np.asarray(b.prior_hi)
    fixed = {names.index("w"): -1.0, names.index("wa"): 0.0} if args.lcdm else {}
    x = np.asarray(b.fiducial, dtype=float)
    if args.start:
        x = np.asarray(json.loads(Path(args.start).read_text())["x"], dtype=float)
        # Coordinates outside the (possibly updated) prior box restart at the ref.
        out = (x <= lo) | (x >= hi)
        if np.any(out):
            print("reset to ref (outside prior box):",
                  [names[i] for i in np.flatnonzero(out)], flush=True)
            x[out] = np.asarray(b.fiducial)[out]
    for i, v in fixed.items():
        x[i] = v
    free = np.array([i for i in range(len(names)) if i not in fixed])
    step8 = np.asarray(S.steps)
    history = []
    t_start = time.perf_counter()
    lp = dali_logpost_fn(b, tables, fixed)
    vg = jax.jit(jax.value_and_grad(lp))
    hess_fn = _hessian_fn(lp, vg, b)
    for it in range(args.map_iters):
        tt = S.build(jnp.asarray(x[:S.n]), order=1, verbose=False, method=args.dali_method,
                     tangent_batch=args.tangent_batch)
        # Jitted (eager op-by-op evaluation of the Planck likelihood OOMs);
        # the DALI expansion equals the exact theory at its expansion point.
        lp_exact_here = float(vg(jnp.asarray(x), tt)[0])
        history.append(dict(it=it, logpost=lp_exact_here, t=time.perf_counter() - t_start))
        print(f"[map {it}] exact log-post at x_k = {lp_exact_here:.3f} "
              f"(t={history[-1]['t']:.0f}s)", flush=True)
        if it > 0 and abs(history[-1]["logpost"] - history[-2]["logpost"]) < args.map_tol:
            break
        hdiag = -np.diag(np.asarray(hess_fn(jnp.asarray(x), tt)))
        scale = np.where(hdiag > 0, 1.0 / np.sqrt(np.maximum(hdiag, 1e-300)), 1e-3)
        # Non-finite curvature (e.g. a NaN finite difference) must not poison the
        # whole scaled problem: fall back to a small fraction of the prior width.
        bad = ~np.isfinite(scale) | (scale <= 0)
        scale[bad] = 1e-3 * (hi - lo)[bad]

        # Trust region on the cosmological parameters (the expansion is local),
        # passed to L-BFGS-B as bounds (clipping inside the objective made the
        # gradient inconsistent and stopped it after 2 iterations).
        tr_lo, tr_hi = lo.copy(), hi.copy()
        for i in range(S.n):
            if i in fixed:
                continue
            tr_lo[i] = max(lo[i], x[i] - args.trust * step8[i])
            tr_hi[i] = min(hi[i], x[i] + args.trust * step8[i])
        x0 = x.copy()

        def to_x(u):
            xx = x0.copy()
            xx[free] = x0[free] + scale[free] * u
            return xx

        def obj(u):
            val, g = vg(jnp.asarray(to_x(u)), tt)
            val = float(val)
            if not np.isfinite(val):
                return 1e30, np.zeros_like(u)
            return -val, -(scale[free] * np.asarray(g)[free])

        bounds = [((tr_lo[i] - x0[i]) / scale[i], (tr_hi[i] - x0[i]) / scale[i]) for i in free]
        t_opt = time.perf_counter()
        res = minimize(obj, np.zeros(free.size), jac=True, method="L-BFGS-B",
                       bounds=bounds, options=dict(maxiter=2000, maxcor=50,
                                                   ftol=1e-14, gtol=1e-8))
        u_best, f_best = res.x, res.fun
        if -res.fun - lp_exact_here < 1e-3:
            # L-BFGS-B made no progress (seen on PR4: 1 iteration, a failed first
            # line search). Projected gradient ascent with backtracking instead.
            lb = np.array([bd[0] for bd in bounds]); ub = np.array([bd[1] for bd in bounds])
            u, (f_u, g_u) = np.zeros(free.size), obj(np.zeros(free.size))
            step = 1.0 / max(np.linalg.norm(g_u), 1e-12)
            for _ in range(300):
                for _bt in range(30):
                    u_new = np.clip(u - step * g_u, lb, ub)
                    f_new, g_new = obj(u_new)
                    if f_new < f_u - 1e-4 * step * float(g_u @ g_u):
                        break
                    step *= 0.5
                else:
                    break
                u, f_u, g_u, step = u_new, f_new, g_new, step * 2.0
                if np.max(np.abs(g_u)) < 1e-6:
                    break
            if f_u < f_best:
                u_best, f_best = u, f_u
            print(f"         L-BFGS-B stalled; gradient fallback reached {-f_u:.3f}", flush=True)
        x = to_x(u_best)
        print(f"         DALI optimum {-f_best:.3f} ({res.nit} it, {res.nfev} ev, "
              f"{time.perf_counter() - t_opt:.1f}s)", flush=True)
    out = dict(names=names, x=x.tolist(), logpost=history[-1]["logpost"], lcdm=args.lcdm,
               history=history)
    path = args.outdir / ("map_lcdm.json" if args.lcdm else "map.json")
    path.write_text(json.dumps(out, indent=2))
    print(f"MAP -2 log-post = {-2 * out['logpost']:.3f}; wrote {path}")
    for n, v in zip(names, x):
        print(f"  {n:18s} {v:+.6g}")


LOGIT, AFFINE, LOWER, UPPER = 0, 1, 2, 3
_KIND_NAMES = {LOGIT: "logit", AFFINE: "affine", LOWER: "softplus-lower", UPPER: "softplus-upper"}


def box_transform(b, spec=None):
    """Unconstrained z ↔ boxed x, with log|dx/dz|, chosen per parameter.

    ``spec=None``: every parameter x = lo + (hi − lo)·σ(z) (logit). Sampling in z
    removes the hard prior walls, but a posterior piled against a bound (ksz_norm
    at 0, xi_sz_cib toward 1) becomes a super-exponential wall in z whose
    curvature forced leapfrog steps of ~0.05. ``spec`` (from
    ``auto_transform_spec``) instead gives each parameter

    - ``affine``          x = c + s·z; the box is enforced by the prior (−∞ → reject);
    - ``softplus-lower``  x = lo + s·softplus(z): exponential tail at the bound,
      linear inside — no wall;
    - ``softplus-upper``  x = hi − s·softplus(z);
    - ``logit``           as above, when both bounds matter.
    """
    lo, hi = jnp.asarray(b.prior_lo), jnp.asarray(b.prior_hi)
    span = hi - lo
    dim = lo.shape[0]
    if spec is None:
        spec = dict(kind=np.zeros(dim, int), c=np.zeros(dim), s=np.ones(dim))
    kind = jnp.asarray(spec["kind"])
    c, sc = jnp.asarray(spec["c"]), jnp.asarray(spec["s"])
    is_l, is_a, is_lo, is_hi = (kind == k for k in (LOGIT, AFFINE, LOWER, UPPER))

    def to_x(z):
        return jnp.where(is_l, lo + span * jax.nn.sigmoid(z),
               jnp.where(is_a, c + sc * z,
               jnp.where(is_lo, lo + sc * jax.nn.softplus(z), hi - sc * jax.nn.softplus(z))))

    def _softplus_inv(y):
        y = jnp.maximum(y, 1e-12)
        return y + jnp.log(-jnp.expm1(-y))

    def to_z(x):
        x = jnp.asarray(x)
        u = jnp.clip((x - lo) / span, 1e-6, 1 - 1e-6)
        return jnp.where(is_l, jnp.log(u) - jnp.log1p(-u),
               jnp.where(is_a, (x - c) / sc,
               jnp.where(is_lo, _softplus_inv((x - lo) / sc), _softplus_inv((hi - x) / sc))))

    def log_jac(z):
        t = jnp.where(is_l, jnp.log(span) + jax.nn.log_sigmoid(z) + jax.nn.log_sigmoid(-z),
            jnp.where(is_a, jnp.log(sc), jnp.log(sc) + jax.nn.log_sigmoid(z)))
        return jnp.sum(t, axis=-1)

    return to_x, to_z, log_jac


def auto_transform_spec(b, x_samples, margin=4.0):
    """Per-parameter transform from pilot x-samples (n, dim): see ``box_transform``."""
    lo, hi = np.asarray(b.prior_lo), np.asarray(b.prior_hi)
    mu, sd = x_samples.mean(0), x_samples.std(0)
    d_lo, d_hi = (mu - lo) / sd, (hi - mu) / sd
    # One-sided softplus toward the nearer bound; the far bound, if it matters at
    # all, is enforced by the prior (rejection). Logit only when the posterior
    # fills the box. (Logit/log maps turn a posterior piled at a bound into a
    # wall whose z-curvature grows like (x/σ_x)²: ksz_norm forced eps ≈ 0.05.)
    kind = np.where(np.minimum(d_lo, d_hi) >= margin, AFFINE,
                    np.where(d_lo <= d_hi, LOWER, UPPER))
    kind = np.where(d_lo + d_hi < 3.0, LOGIT, kind)
    return dict(kind=kind.astype(int), c=mu, s=sd)


def load_transform_spec(outdir):
    path = Path(outdir) / "transform.npz"
    if not path.exists():
        return None
    z = np.load(path)
    return dict(kind=z["kind"], c=z["c"], s=z["s"])


def chain_to_x(b, path):
    """x-space samples (n_iter, n_chains, dim) of a saved z-chain, via its own transform."""
    z = np.load(path)
    spec = dict(kind=z["t_kind"], c=z["t_c"], s=z["t_s"]) if "t_kind" in z else None
    to_x = box_transform(b, spec)[0]
    return np.asarray(jax.jit(jax.vmap(jax.vmap(to_x)))(jnp.asarray(z["samples"])))


def stage_build(args, b, tables):
    S = b.dali
    m = json.loads((args.outdir / "map.json").read_text())
    x = np.asarray(m["x"])
    if args.reuse_expansion and (args.outdir / "dali.npz").exists():
        tt = S.load(args.outdir / "dali.npz")
        print("reusing", args.outdir / "dali.npz", flush=True)
    elif args.expansion_only:  # theory only: the likelihood arrays (HiLLiPoP's matrix) are never loaded
        tay = S.build_expansion(jnp.asarray(x[:S.n]), order=args.dali_order, method=args.dali_method,
                             tangent_batch=args.tangent_batch, pair_batch=args.pair_batch)
        S.save((tay, None), args.outdir / "dali.npz")
    else:
        tt = S.build(jnp.asarray(x[:S.n]), order=args.dali_order, method=args.dali_method,
                     tangent_batch=args.tangent_batch, pair_batch=args.pair_batch)
        S.save(tt, args.outdir / "dali.npz")
    if args.expansion_only:
        print("--expansion-only: skipping the z-mode / cov_z search (it seeds only --stage sample)", flush=True)
        return
    lp2 = dali_logpost_fn(b, tables)
    to_x, to_z, log_jac = box_transform(b)

    def lpz(z, tt_):
        return lp2(to_x(z), tt_) + log_jac(z)

    # The z-space mode is not to_z(x_map): nuisances on a bound at the x-MAP (e.g.
    # ksz_norm = 0) map to z → −∞ where the z log-post is flat. Maximize the z
    # log-post (DALI + log-Jacobian) so the Hessian is taken at a real mode.
    vgz = jax.jit(jax.value_and_grad(lpz))
    hz = _hessian_fn(lpz, vgz, b)

    # Damped Newton on the DALI expansion: L-BFGS stalled after one iteration because
    # the z-curvature spans many decades (ksz_norm on its bound is flat in z).
    z = jnp.asarray(to_z(x))
    t_opt = time.perf_counter()
    val, g = vgz(z, tt)
    n_newton, gnorm = 0, float("inf")
    for n_newton in range(1, 101):
        h = np.asarray(hz(z, tt))
        lam, vec = np.linalg.eigh(-0.5 * (h + h.T))
        lam = np.maximum(np.abs(lam), 1e-3)
        step = vec @ ((vec.T @ np.asarray(g)) / lam)
        step = step * min(1.0, 2.0 / max(np.max(np.abs(step)), 1e-300))
        accepted = False
        for alpha in (1.0, 0.5, 0.25, 0.1, 0.03):
            z_try = z + alpha * jnp.asarray(step)
            v_try, g_try = vgz(z_try, tt)
            if np.isfinite(float(v_try)) and float(v_try) > float(val):
                z, val, g = z_try, v_try, g_try
                accepted = True
                break
        gnorm = float(np.max(np.abs(np.asarray(g))))
        if not accepted or gnorm < 1e-4:
            break
    print(f"z-mode (damped Newton, {n_newton} steps, max|grad| {gnorm:.2e}, "
          f"{time.perf_counter() - t_opt:.1f}s): DALI lpz {float(val):.3f}", flush=True)

    class _R:
        pass

    rz = _R()
    rz.x, rz.fun, rz.nit = np.asarray(z), -float(val), n_newton
    z_map = jnp.asarray(rz.x)
    x_zmode = np.asarray(to_x(z_map))
    exact_zmode = float(b.log_post_batch_fn(jnp.asarray(x_zmode)[None])[0])
    print(f"z-mode: DALI lpz {-rz.fun:.3f} ({rz.nit} it, "
          f"{time.perf_counter() - t_opt:.1f}s); exact log-post at x(z-mode) {exact_zmode:.3f}",
          flush=True)
    moved = np.abs(x_zmode - x) > 1e-3 * (np.asarray(b.prior_hi) - np.asarray(b.prior_lo))
    print("  moved by the Jacobian:", {b.param_names[i]: (round(float(x[i]), 4),
          round(float(x_zmode[i]), 4)) for i in np.flatnonzero(moved)}, flush=True)
    hess = np.asarray(hz(z_map, tt))
    # Covariance from the eigendecomposition of −H: negative-curvature (saddle)
    # directions take |λ|, flat ones a floor. (Adding |min eig|·I to cov gave every
    # z-direction variance ≥ 5, far too wide for tight parameters like θ_MC.)
    neg_h = -0.5 * (hess + hess.T)
    lam, vec = np.linalg.eigh(neg_h)
    n_neg = int(np.sum(lam <= 0))
    if n_neg:
        print(f"warning: -H has {n_neg} non-positive eigenvalue(s); using |λ| there",
              flush=True)
        for j in np.flatnonzero(lam <= 0):
            order = np.argsort(np.abs(vec[:, j]))[::-1][:3]
            print(f"  λ={lam[j]:+.3e} direction: "
                  + ", ".join(f"{b.param_names[k]} {vec[k, j]:+.2f}" for k in order),
                  flush=True)
    lam_eff = np.maximum(np.abs(lam), 1e-2)
    cov_z = (vec / lam_eff) @ vec.T
    ev, evec = np.linalg.eigh(cov_z)
    for j in np.argsort(ev)[::-1][:3]:
        k = int(np.argmax(np.abs(evec[:, j])))
        print(f"  cov_z top eigen-direction σ={np.sqrt(ev[j]):.3f}, "
              f"dominated by {b.param_names[k]}", flush=True)
    np.save(args.outdir / "cov_z.npy", cov_z)
    np.save(args.outdir / "z_map.npy", np.asarray(z_map))
    np.save(args.outdir / "x_zmode.npy", x_zmode)
    # Report the implied x-space σ (delta method) for reference.
    jac = np.asarray(jax.jacfwd(to_x)(z_map))
    sig = np.sqrt(np.clip(np.diag(jac @ cov_z @ jac.T), 0, None))
    print("Laplace σ (x, via z):", dict(zip(b.param_names, np.round(sig, 6))), flush=True)

    # DALI accuracy on Laplace draws in z (always inside the prior box).
    rng = np.random.default_rng(1)
    zd = rng.multivariate_normal(np.asarray(z_map), cov_z, size=args.check_draws)
    xd = np.asarray(jax.vmap(to_x)(jnp.asarray(zd)))
    exact = np.asarray(b.log_post_batch_fn(jnp.asarray(xd)))
    sur = np.asarray(jax.jit(jax.vmap(lp2, in_axes=(0, None)))(jnp.asarray(xd), tt))
    ok = np.isfinite(exact) & np.isfinite(sur)
    d = (sur - exact)[ok]
    print(f"DALI - exact log-post on {ok.sum()}/{len(ok)} Laplace draws: "
          f"mean {d.mean():+.3f} std {d.std():.3f} max|.| {np.abs(d).max():.3f}", flush=True)
    (args.outdir / "build.json").write_text(json.dumps(dict(
        sigma=sig.tolist(), n_valid=int(ok.sum()), dlogpost_mean=float(d.mean()),
        dlogpost_std=float(d.std()), dlogpost_maxabs=float(np.abs(d).max())), indent=2))


def sampling_targets(args, b, tables):
    """z-space DALI ``lp_s(z, tt)`` (the trajectory's, smoothed), the DALI accept/reject target
    ``lp_t(z, tt)`` (official likelihoods), the batched exact log-posterior, the z → x map, and
    the DALI expansion data ``tt``. ``tt`` (expansion coefficients and likelihood arrays,
    e.g. PR4's 3.3 GB matrix) is passed to the trajectory as a jit argument
    (``run_dali_hmc(dali_args=tt)``): captured in a closure, it was
    embedded in the compiled program as a second device copy."""
    tt = b.dali.load(args.outdir / "dali.npz")
    lp2 = dali_logpost_fn(b, tables)
    lp2_t = dali_logpost_fn(b, tables, smooth=False)
    to_x, _, log_jac = box_transform(b, load_transform_spec(args.outdir))

    def lp_s(z, tt_):
        return lp2(to_x(z), tt_) + log_jac(z)

    def lp_t(z, tt_):
        return lp2_t(to_x(z), tt_) + log_jac(z)

    def exact_batch(z):
        return b.log_post_batch_fn(jax.vmap(to_x)(z)) + log_jac(z)

    return lp_s, lp_t, exact_batch, to_x, tt


def initial_state(args, b):
    """Initial z per chain and initial inverse mass matrix, in this run's transform.

    ``--init-from``: mass matrix from the empirical z-covariance of a previous
    chain (iterations ≥ ``--init-skip``, default its second half), mapped through
    x into the current transform. Chains start at that chain's last states
    (``--init-center last``) or at N(z(x_mode), 0.1·cov) around this run's own
    mode (``--init-center z_map``; use when the data changed, e.g. adding SN).
    Otherwise both come from the Laplace ``cov_z`` of ``--stage build`` (logit only).
    """
    spec = load_transform_spec(args.outdir)
    _, to_z, _ = box_transform(b, spec)
    to_z_all = jax.jit(jax.vmap(to_z))
    rng = np.random.default_rng(args.seed)
    if not args.init_from:
        assert spec is None, "the Laplace cov_z is in logit coordinates; use --init-from"
        cov_z = np.load(args.outdir / "cov_z.npy")
        z_map = np.load(args.outdir / "z_map.npy")
        return rng.multivariate_normal(z_map, 0.1 * cov_z, size=args.chains), cov_z
    xs = chain_to_x(b, args.init_from)  # (n_iter, n_chains, dim)
    assert xs.shape[-1] == len(b.param_names), (xs.shape, len(b.param_names))
    skip = xs.shape[0] // 2 if args.init_skip is None else args.init_skip
    zs = np.asarray(to_z_all(jnp.asarray(xs.reshape(-1, xs.shape[-1])))).reshape(xs.shape)
    pool = zs[skip:].reshape(-1, zs.shape[-1])
    inv_mass0 = np.cov(pool.T)
    inv_mass0 = 0.9 * inv_mass0 + 0.1 * np.diag(np.diag(inv_mass0)) + 1e-6 * np.eye(zs.shape[-1])
    if args.init_center == "z_map":
        xm_path = args.outdir / "x_zmode.npy"
        x_map = (np.load(xm_path) if xm_path.exists()
                 else np.asarray(box_transform(b)[0](jnp.asarray(np.load(args.outdir / "z_map.npy")))))
        z_c = np.asarray(to_z(jnp.asarray(x_map)))
        z_init = rng.multivariate_normal(z_c, 0.1 * inv_mass0, size=args.chains)
    else:
        z_init = zs[-1][: args.chains]
        if z_init.shape[0] < args.chains:
            extra = rng.multivariate_normal(pool.mean(0), inv_mass0,
                                            size=args.chains - z_init.shape[0])
            z_init = np.concatenate([z_init, extra])
    print(f"warm start from {args.init_from}: {pool.shape[0]} samples (iterations ≥ {skip}) "
          f"for the mass matrix; chains start at {args.init_center}", flush=True)
    return z_init, inv_mass0


def _spec_arrays(args):
    spec = load_transform_spec(args.outdir)
    return {} if spec is None else dict(t_kind=spec["kind"], t_c=spec["c"], t_s=spec["s"])


def setup_transform(args, b):
    """``--transform auto``: choose per-parameter transforms from a pilot chain."""
    path = args.outdir / "transform.npz"
    if args.resume and path.exists():
        return
    if args.transform != "auto":
        if path.exists():
            print(f"removing stale {path} (--transform logit)", flush=True)
            path.unlink()
        return
    pilot = args.transform_pilot or args.init_from
    assert pilot, "--transform auto needs --transform-pilot or --init-from"
    xs = chain_to_x(b, pilot)
    skip = xs.shape[0] // 2 if args.init_skip is None else args.init_skip
    spec = auto_transform_spec(b, xs[skip:].reshape(-1, xs.shape[-1]))
    np.savez(path, **spec)
    print("transforms from", pilot, ":", {n: _KIND_NAMES[int(k)] for n, k in
          zip(b.param_names, spec["kind"]) if k != AFFINE}, "(others affine)", flush=True)


def _adapt_points(args):
    return tuple(int(a) for a in args.adapt_mass_at.split(",") if a) if args.adapt_mass_at else ()


def stage_sample(args, b, tables):
    from gpucosmo.jax.mcmc.dali_hmc import run_dali_hmc

    lp_s, lp_t, exact_batch, to_x, tt = sampling_targets(args, b, tables)
    z_init, inv_mass0 = initial_state(args, b)
    adapt = _adapt_points(args)
    res = run_dali_hmc(
        dali_logpost=lp_s,
        dali_args=tt,
        accept=args.accept,
        dali_target_logpost=lp_t,
        exact_logpost_batch=exact_batch,
        x_init=z_init,
        inv_mass=inv_mass0,
        adapt_mass_at=adapt,
        n_warmup=args.warmup,
        n_iter=args.iters,
        n_leapfrog=args.leapfrog,
        seed=args.seed,
        out_path=args.outdir / "chain_z.npz",
        param_names=list(b.param_names),
    )
    x_samples = np.asarray(jax.vmap(jax.vmap(to_x))(jnp.asarray(res["samples"])))
    if _spec_arrays(args):
        z = dict(np.load(args.outdir / "chain_z.npz"))
        np.savez(args.outdir / "chain_z.npz", **z, **_spec_arrays(args))
    np.savez(args.outdir / "chain.npz", samples=x_samples, logpost_z=res["logpost"],
             accept=res["accept"], eps=res["eps"], names=np.asarray(b.param_names), accept_mode=args.accept)
    print(f"wrote {args.outdir / 'chain.npz'}  (samples in parameter space)")


def stage_converge(args, b, tables):
    """Resumable segments of DALI HMC until R̂/ESS targets are met."""
    from analyze_dali_chain import diagnostics

    from gpucosmo.jax.mcmc.dali_hmc import run_dali_hmc

    names = list(b.param_names)
    state_path = args.outdir / "converge_state.npz"
    log_path = args.outdir / "converge.log"
    lp_s, lp_t, exact_batch, to_x, tt = sampling_targets(args, b, tables)
    to_x_all = jax.jit(jax.vmap(jax.vmap(to_x)))
    common = dict(dali_logpost=lp_s, dali_args=tt, exact_logpost_batch=exact_batch,
                  accept=args.accept, dali_target_logpost=lp_t,
                  n_leapfrog=args.leapfrog, seed=args.seed, param_names=names,
                  out_path=args.outdir / "segment_z.npz", save_every=25)

    def log(msg):
        print(msg, flush=True)
        with log_path.open("a") as f:
            f.write(msg + "\n")

    if args.resume and state_path.exists():
        st = np.load(state_path, allow_pickle=False)
        mode = str(st["accept_mode"]) if "accept_mode" in st.files else "exact"  # earlier runs: exact
        if mode != args.accept:
            raise SystemExit(f"{state_path} was sampled with --accept {mode}; resume with the same "
                             f"(--accept {args.accept} would change the target posterior mid-chain)")
        chain = {k: list(st[k]) for k in ("samples", "logpost", "accept", "eps")}
        chain["n_steps"] = list(st["n_steps"]) if "n_steps" in st else []
        n_warmup = int(st["n_warmup"])
        last = dict(x_last=st["x_last"], inv_mass=st["inv_mass"],
                    eps_final=float(st["eps_final"]),
                    rng_state=dict(key=st["key"], np=json.loads(str(st["np_state"]))))
        log(f"resume: {len(chain['samples'])} iterations from {state_path}")
    else:
        chain = dict(samples=[], logpost=[], accept=[], eps=[], n_steps=[])
        n_warmup = args.warmup if args.warmup is not None else 150
        last = None
        log_path.write_text("")
    seg_no = 0
    while True:
        n_done = len(chain["samples"])
        if last is None:
            z_init, inv_mass0 = initial_state(args, b)
            n_new = n_warmup + args.segment
            res = run_dali_hmc(x_init=z_init, inv_mass=inv_mass0, n_iter=n_new,
                                    n_warmup=n_warmup, adapt_mass_at=_adapt_points(args),
                                    **common)
        else:
            n_new = min(args.segment, args.max_iters - n_done)
            if n_new <= 0:
                log(f"stopping: reached --max-iters {args.max_iters} without convergence")
                break
            res = run_dali_hmc(x_init=last["x_last"], inv_mass=last["inv_mass"],
                                    step_size=last["eps_final"], n_iter=n_new, n_warmup=0,
                                    rng_state=last["rng_state"], **common)
        last = res
        seg_no += 1
        for k in chain:
            chain[k].extend(list(res.get(k, [])))
        zs = np.stack(chain["samples"])
        xs = np.asarray(to_x_all(jnp.asarray(zs)))
        tsp = _spec_arrays(args)
        np.savez(state_path, **tsp, samples=zs, logpost=np.stack(chain["logpost"]),
                 accept=np.stack(chain["accept"]), eps=np.asarray(chain["eps"]),
                 n_steps=np.asarray(chain["n_steps"], dtype=np.int32),
                 n_warmup=n_warmup, x_last=res["x_last"], inv_mass=res["inv_mass"],
                 eps_final=res["eps_final"], key=res["rng_state"]["key"],
                 np_state=json.dumps(res["rng_state"]["np"]), names=np.asarray(names),
                 accept_mode=args.accept)
        np.savez(args.outdir / "chain_z.npz", **tsp, samples=zs, logpost=np.stack(chain["logpost"]),
                 accept=np.stack(chain["accept"]), eps=np.asarray(chain["eps"]),
                 n_warmup=n_warmup, names=np.asarray(names), inv_mass=res["inv_mass"])
        np.savez(args.outdir / "chain.npz", samples=xs, logpost_z=np.stack(chain["logpost"]),
                 accept=np.stack(chain["accept"]), eps=np.asarray(chain["eps"]),
                 n_steps=np.asarray(chain["n_steps"], dtype=np.int32),  # leapfrog steps/iteration
                 n_warmup=n_warmup, names=np.asarray(names))
        post = xs[n_warmup:]
        acc = float(np.mean(np.stack(chain["accept"])[n_warmup:]))
        diag = diagnostics(post, names)
        worst = sorted(diag, key=lambda d: -d["rhat"])[:3]
        rhat_max = worst[0]["rhat"]
        ess_b = min(d["ess_bulk"] for d in diag)
        ess_t = min(d["ess_tail"] for d in diag)
        wd = {d["name"]: d for d in diag}
        log(f"[segment {seg_no}] {zs.shape[0]} it ({post.shape[0]} post-warm-up × "
            f"{post.shape[1]} chains), {res['t_per_it']:.2f} s/it, eps {res['eps_final']:.3f}, "
            f"acc {acc:.3f} | worst R̂ "
            + ", ".join(f"{d['name']} {d['rhat']:.3f}" for d in worst)
            + f" | min ESS bulk {ess_b:.0f} tail {ess_t:.0f} | "
            + " ".join(f"{n} {wd[n]['mean']:+.3f}±{wd[n]['std']:.3f}"
                       for n in ("w", "wa", "mu0", "Sigma0") if n in wd))
        (args.outdir / "converge_diag.json").write_text(json.dumps(diag, indent=2))
        if rhat_max < args.rhat and ess_b >= args.min_ess and ess_t >= args.min_ess:
            log(f"converged: max R̂ {rhat_max:.4f} < {args.rhat}, min ESS bulk {ess_b:.0f} "
                f"tail {ess_t:.0f} ≥ {args.min_ess}")
            break
        if zs.shape[0] >= args.max_iters:
            log(f"stopping: reached --max-iters {args.max_iters} without convergence")
            break


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="gpucosmo-dali-hmc", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("map", "build", "sample", "converge"), required=True)
    ap.add_argument("--config", type=Path, default=None,
                    help="Cobaya yaml (likelihoods, theory options, params/priors) for the μ–Σ / "
                         "DESI DR1 full-shape bundle; overrides --analysis")
    ap.add_argument("--analysis", default="desi_full",
                    choices=["desi_full"] + sorted(MG_ANALYSES),
                    help="desi_full: DESI DR2 BAO + Planck + ACT lensing (+SN); others: the yamls shipped in "
                         "gpucosmo/configs/cobaya (arXiv:2411.12026 μ–Σ / ΛCDM analyses)")
    ap.add_argument("--plugin", action="append", default=[], metavar="MODULE",
                    help="import MODULE (dotted name or path/to/file.py) first: it registers extra likelihoods "
                         "the yaml can name (repeatable)")
    ap.add_argument("--no-sn", action="store_true")
    ap.add_argument("--sn-dataset", choices=("pantheonplus", "union3", "desy5"),
                    default="pantheonplus")
    ap.add_argument("--lcdm", action="store_true")
    ap.add_argument("--outdir", type=Path, default=Path("outputs/shmc"))
    ap.add_argument("--start", default=None)
    ap.add_argument("--map-iters", type=int, default=8)
    ap.add_argument("--map-tol", type=float, default=0.05)
    ap.add_argument("--trust", type=float, default=2.0)
    ap.add_argument("--check-draws", type=int, default=16)
    ap.add_argument("--reuse-expansion", action="store_true",
                    help="build: load outputs/.../dali.npz instead of re-expanding")
    ap.add_argument("--chains", type=int, default=8)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--leapfrog", type=int, default=20)
    ap.add_argument("--accept", choices=("dali", "exact"), default="dali",
                    help="sample/converge: accept/reject on the DALI posterior with the official likelihoods "
                         "(no exact likelihood call; the posterior the evidence pipeline samples) or on the exact "
                         "posterior (one batched exact theory call per iteration; exact whatever the DALI "
                         "expansion's error)")
    ap.add_argument("--walker-batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-from", default=None,
                    help="sample: warm start from a previous chain_z.npz")
    ap.add_argument("--adapt-mass-at", default="25,50",
                    help="sample: warm-up iterations at which to re-estimate the mass matrix")
    ap.add_argument("--warmup", type=int, default=None,
                    help="sample: warm-up iterations (default iters/4; converge: 150)")
    ap.add_argument("--init-center", choices=("last", "z_map"), default="last",
                    help="with --init-from: start chains at its last states or around z_map")
    ap.add_argument("--init-skip", type=int, default=None,
                    help="with --init-from: first iteration used (default: second half)")
    ap.add_argument("--transform", choices=("logit", "auto"), default="logit",
                    help="sample/converge: per-parameter transforms from a pilot chain")
    ap.add_argument("--transform-pilot", default=None,
                    help="z-chain used to choose transforms (default --init-from)")
    ap.add_argument("--segment", type=int, default=250, help="converge: iterations per segment")
    ap.add_argument("--max-iters", type=int, default=5000)
    ap.add_argument("--rhat", type=float, default=1.01)
    ap.add_argument("--min-ess", type=float, default=400)
    ap.add_argument("--resume", action="store_true", help="converge: continue converge_state.npz")
    ap.add_argument("--bbn-prior", choices=("yaml", "paper"), default="yaml",
                    help="MG analyses without CMB: the yaml's BBN likelihood, or the paper "
                         "text's N(0.02218, 0.00055) (to continue runs started with it)")
    ap.add_argument("--dali-order", type=int, default=DALI_ORDER,
                    help="build: order of the DALI expansion of the theory (1 Fisher, 2 doublet, 3 triplet, ...; "
                         "the MAP stage always refits at order 1)")
    ap.add_argument("--dali-method", choices=("fd", "fwd"), default="fd",
                    help="expansion coefficients (build stage and the MAP stage's linear refits): fd = finite-difference "
                         "design of exact evaluations (steps ~1σ, stencils exact to the expansion's order); fwd = "
                         "JAX forward-mode derivatives at the expansion point (k nested jvps per degree-k monomial)")
    ap.add_argument("--tangent-batch", type=int, default=0, help="fwd: tangents per compiled call (0 = all n)")
    ap.add_argument("--pair-batch", type=int, default=1,
                    help="fwd: higher-order direction tuples per compiled call")
    ap.add_argument("--expansion-only", action="store_true",
                    help="build: stop after dali.npz; the z-mode Newton search and cov_z that follow "
                         "seed only --stage sample (the evidence pipeline never reads them)")
    ap.add_argument("--no-autosize", action="store_true",
                    help="MG analyses: keep the MGConfig batch sizes instead of sizing the CMB / "
                         "P(k) walker batches to the GPU memory (cached per device)")
    ap.add_argument("--autosize-frag", type=float, default=None,
                    help="allocator fragmentation allowance for autosize (default 0.03)")
    ap.add_argument("--cmb-walker-batch", type=int, default=None, help="override (no autosize)")
    ap.add_argument("--pk-walker-batch", type=int, default=None, help="override (no autosize)")
    ap.add_argument("--cmb-lensing", action="store_true",
                    help="add ACT DR6 CMB lensing (candl, ACT-only) to a yaml without it")
    ap.add_argument("--cmb-nonlinear-lens", action="store_true",
                    help="MG analyses with CMB: lens with HMcode φφ (overrides the yaml's NonLinear)")
    ap.add_argument("--cmb-legacy-lensing", action="store_true",
                    help="lens the CMB on the solver's base path, as runs before 2026-10-17 (default: the "
                         "accurate high-ℓ path; use the same choice in every stage and model)")
    ap.add_argument("--fs-tables", action="store_true",
                    help="DESI FS: the DALI approximation expands the REPT multipole tables instead of the linear "
                         "spectra (MGConfig.fs_tables; use the same choice in build and ns)")
    ap.add_argument("--cmb-linear-lens", action="store_true",
                    help="CMB analyses: lens with the linear φφ (overrides the yaml / Cobaya "
                         "NonLinear_lens rule; use the same choice in nested models compared by evidence)")
    args = ap.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    b, tables = setup(args)
    if args.stage in ("sample", "converge"):
        setup_transform(args, b)
    dict(map=stage_map, build=stage_build, sample=stage_sample,
         converge=stage_converge)[args.stage](args, b, tables)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
