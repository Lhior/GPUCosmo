#!/usr/bin/env python3
"""Accuracy/throughput harness for the scalar perturbation evolution.

Evolves a handful of k modes with a candidate integrator and compares the CMB
sources (S_T, S_E, Weyl) on the Cls τ grid against a high-resolution RK4
reference. Sources are what the LOS projection consumes, so this isolates the
integrator from the k/ℓ quadrature.

Usage::

    python scripts/bench_evolve.py                      # default vs reference
    python scripts/bench_evolve.py --mode segmented     # new integrator
    python scripts/bench_evolve.py --mode segmented --batch 1,8,32 --n-k 205
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from gpucosmo.jax.eb.cls_jax import (  # noqa: E402
    CLS_HIERARCHY,
    CLS_N_STEPS,
    CLS_N_SUB,
    CLS_N_TAU_SRC,
    _evolve_q_grid,
    _params_dict_cls,
    _recfast_thermo_device,
)
from gpucosmo.jax.eb.evolve_jax import (  # noqa: E402
    downsample_thermo_for_evolve,
    evolve_scalar_k_jax,
)
from gpucosmo.jax.eb.sources_jax import temperature_polarization_sources  # noqa: E402

# DESI DR2 + CMB + SN-like w0wa point (close to the gate chain point).
FIDUCIAL = dict(
    ombh2=0.02237, omch2=0.1193, h=0.667, tau_reio=0.055, ns=0.965,
    As=2.1e-9, w0=-0.836, wa=-0.701,
)
DEFAULT_KS = (0.0008, 0.004, 0.02, 0.06, 0.12, 0.2, 0.3, 0.4, 0.45)


def device_tau_src(thermo: dict, n_tau_src: int = CLS_N_TAU_SRC) -> jnp.ndarray:
    """Same on-device τ sampling as ``cls_jax._source_grid(device_grids=True)``."""
    tau_full = thermo["tau_mpc"]
    g = thermo["visibility"]
    tau_peak = tau_full[jnp.argmax(g)]
    tau0 = tau_full[-1]
    mid = tau_full.shape[0] // 2
    tau_reio = tau_full[mid + jnp.argmax(g[mid:])]
    half_w = 0.40
    tau_lo = jnp.maximum(tau_peak * (1.0 - half_w), tau_full[0] * 2.0)
    tau_hi = jnp.minimum(tau_peak * (1.0 + half_w), tau0 * 0.55)
    n_rec = (5 * n_tau_src) // 8
    n_reio = n_tau_src // 5
    n_late = n_tau_src - n_rec - n_reio
    reio_lo = jnp.maximum(tau_reio * 0.7, tau_hi)
    reio_hi = jnp.minimum(tau_reio * 1.3, tau0 * 0.95)
    tau_src = jnp.sort(jnp.concatenate([
        jnp.linspace(tau_lo, tau_hi, n_rec),
        jnp.linspace(reio_lo, reio_hi, n_reio),
        jnp.linspace(reio_hi, tau0, n_late + 1)[1:],
    ]))
    eps = 1.0e-6 * (tau0 + 1.0)
    tau_src = jnp.maximum(tau_src, jnp.concatenate([tau_src[:1], tau_src[:-1] + eps]))
    return jnp.minimum(tau_src, tau0)


def setup(cosmo: dict = FIDUCIAL) -> dict:
    params = _params_dict_cls(
        cosmo["ombh2"], cosmo["omch2"], cosmo["h"], cosmo["ns"], cosmo["As"],
        cosmo["tau_reio"], cosmo["w0"], cosmo["wa"],
    )
    thermo = _recfast_thermo_device(
        cosmo["ombh2"], cosmo["omch2"], cosmo["h"], cosmo["ns"], cosmo["As"],
        cosmo["tau_reio"], cosmo["w0"], cosmo["wa"],
    )
    tau_src = device_tau_src(thermo)
    tau_full = thermo["tau_mpc"]
    return dict(
        params=params,
        thermo=thermo,
        thermo_ev=downsample_thermo_for_evolve(thermo),
        tau_src=tau_src,
        kappa=jnp.interp(tau_src, tau_full, thermo["kappa"]),
        vis=jnp.interp(tau_src, tau_full, thermo["visibility"]),
        tau0=tau_full[-1],
    )


def make_sources_fn(ctx: dict, mode: str, n_steps: int, n_sub: int, cfg=CLS_HIERARCHY,
                    seg_cfg=None):
    """Return jitted ``k_arr -> (s_t, s_e, weyl)`` for one cosmology."""
    params, thermo, thermo_ev = ctx["params"], ctx["thermo"], ctx["thermo_ev"]
    tau_src, vis, kappa, tau0 = ctx["tau_src"], ctx["vis"], ctx["kappa"], ctx["tau0"]

    def one_k(k):
        y = evolve_scalar_k_jax(
            k, params, thermo_ev, tau_src, cfg=cfg, n_steps=n_steps, n_sub=n_sub,
            integrate_on_output=True, method=mode, seg_cfg=seg_cfg,
        )
        return temperature_polarization_sources(
            y, k, tau_src, vis, kappa, params, thermo, cfg, tau0=tau0
        )

    return jax.jit(jax.vmap(one_k))


def rel_err(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-k relative L2 error of a against b over τ."""
    num = np.sqrt(np.sum((a - b) ** 2, axis=-1))
    den = np.sqrt(np.sum(b**2, axis=-1)) + 1e-300
    return num / den


def timed(fn, *args, reps: int = 3):
    t0 = time.perf_counter()
    out = jax.block_until_ready(fn(*args))
    t_cold = time.perf_counter() - t0
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = jax.block_until_ready(fn(*args))
        ts.append(time.perf_counter() - t0)
    return out, t_cold, min(ts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="rk4_timesteps",
                    help="evolve method (rk4_timesteps | segmented)")
    ap.add_argument("--n-steps", type=int, default=CLS_N_STEPS)
    ap.add_argument("--n-sub", type=int, default=CLS_N_SUB)
    ap.add_argument("--ref-steps", type=int, default=8192)
    ap.add_argument("--ref-sub", type=int, default=4)
    ap.add_argument("--ks", default=",".join(str(k) for k in DEFAULT_KS))
    ap.add_argument("--n-k", type=int, default=0,
                    help="time the full Evolve_q grid (first n_k modes; 0=skip)")
    ap.add_argument("--no-ref", action="store_true")
    ap.add_argument("--freeze-late", action="store_true",
                    help="segmented: legacy max_etak RSA freeze (matches reference)")
    ap.add_argument("--budget-scale", type=float, default=1.0,
                    help="segmented: scale every segment step count")
    ap.add_argument("--ref-mode", default="rk4_timesteps",
                    help="reference integrator; 'segmented' uses --ref-scale x budgets")
    ap.add_argument("--ref-scale", type=float, default=4.0)
    ap.add_argument("--ref-cache", type=Path, default=None,
                    help="default: outputs/benchmarks/evolve_ref_<ref-mode>.npz")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    print(f"backend={jax.default_backend()} devices={jax.devices()}")
    ctx = setup()
    ks = jnp.asarray([float(x) for x in args.ks.split(",")])
    report: dict = {"mode": args.mode, "n_steps": args.n_steps, "n_sub": args.n_sub,
                    "ks": [float(k) for k in ks], "backend": jax.default_backend()}

    import dataclasses

    from gpucosmo.jax.eb.evolve_segmented_jax import DEFAULT_SEGMENTED

    def scaled_seg(sc: float):
        d = DEFAULT_SEGMENTED
        return dataclasses.replace(
            d, freeze_late=args.freeze_late,
            n_tc=round(d.n_tc * sc), n_full=round(d.n_full * sc),
            n_mid=round(d.n_mid * sc), n_rsa=round(d.n_rsa * sc),
        )

    seg_cfg = None
    if args.mode == "segmented":
        seg_cfg = scaled_seg(args.budget_scale)
        report["seg_cfg"] = dataclasses.asdict(seg_cfg)
    fn = make_sources_fn(ctx, args.mode, args.n_steps, args.n_sub, seg_cfg=seg_cfg)
    (s_t, s_e, weyl), t_cold, t_warm = timed(fn, ks)
    print(f"[{args.mode}] {len(ks)} k: cold {t_cold:.2f}s warm {t_warm:.3f}s")
    report.update(t_cold_s=t_cold, t_warm_s=t_warm)

    if not args.no_ref:
        if args.ref_mode == "segmented":
            ref_seg = scaled_seg(args.ref_scale)
            tag = f"seg{args.ref_scale:g}x:freeze={int(args.freeze_late)}"
        else:
            ref_seg = None
            tag = f"{args.ref_steps}x{args.ref_sub}"
        key = tag + ":" + ",".join(f"{float(k):.6g}" for k in ks)
        cache = args.ref_cache or Path(f"outputs/benchmarks/evolve_ref_{args.ref_mode}.npz")
        if args.ref_mode == "rk4_timesteps" and args.ref_cache is None:
            cache = Path("outputs/benchmarks/evolve_ref_sources.npz")
        ref = None
        if cache.exists():
            z = np.load(cache, allow_pickle=False)
            if str(z["key"]) == key:
                ref = (z["s_t"], z["s_e"], z["weyl"])
        if ref is None:
            print(f"building reference {args.ref_mode} {tag} ...")
            ref_fn = make_sources_fn(ctx, args.ref_mode, args.ref_steps, args.ref_sub,
                                     seg_cfg=ref_seg)
            ref = tuple(np.asarray(r) for r in jax.block_until_ready(ref_fn(ks)))
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, key=key, s_t=ref[0], s_e=ref[1], weyl=ref[2])
        report["ref"] = key
        errs = {name: rel_err(np.asarray(got), r)
                for name, got, r in zip(("s_t", "s_e", "weyl"), (s_t, s_e, weyl), ref)}
        print(f"{'k':>8} {'S_T':>10} {'S_E':>10} {'Weyl':>10}")
        for i, k in enumerate(np.asarray(ks)):
            print(f"{k:8.4f} {errs['s_t'][i]:10.2e} {errs['s_e'][i]:10.2e} "
                  f"{errs['weyl'][i]:10.2e}")
        report["rel_err"] = {n: e.tolist() for n, e in errs.items()}

    if args.n_k > 0:
        k_grid = _evolve_q_grid(k_max=0.45, source_accuracy_boost=1.0,
                                taurst=280.0, tau0=14000.0)[: args.n_k]
        _, tc, tw = timed(fn, k_grid, reps=2)
        print(f"[{args.mode}] Evolve_q {k_grid.shape[0]} k: cold {tc:.2f}s warm {tw:.3f}s")
        report.update(n_k=int(k_grid.shape[0]), t_grid_cold_s=tc, t_grid_warm_s=tw)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
