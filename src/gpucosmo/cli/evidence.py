"""``gpucosmo-evidence``: the whole evidence pipeline in one command.

    gpucosmo-evidence --config my_analysis.yaml --outdir OUT [--plugin my_likelihoods.py]

Stages (each skipped when its output exists, so an interrupted run resumes):

1. ``map``   — posterior maximum (``gpucosmo-dali-hmc --stage map``);
2. ``build`` — DALI expansion of the theory at the MAP to order ``--dali-order`` (default 3, the
   triplet; ``--stage build --expansion-only``);
3. ``ns``    — tempered SMC on the compressed DALI likelihood, started from a pilot-fitted reference,
   adding runs until the log Z_DALI error reaches ``--smc-target-err``; then the DALI order
   check: the expansion truncated one order lower, at the posterior draws, gives the last
   order's effect on log Z and on the posterior means.

The exact stages (map, build) run in this process with one shared likelihood bundle, so the
exact theory program is traced, lowered and loaded once (~85 s per process otherwise), with
the XLA flags they need next to Planck PR4's 3.3 GB matrix. The SMC stage runs as a subprocess
without those flags (its GPU kernels are ~2× faster with Triton GEMMs).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from gpucosmo.jax.inference.dali_expansion import DALI_ORDER

EXACT_XLA_FLAGS = "--xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"


def sampling_boxes_from_moments(priors: dict, names, mean, sd, nsigma: float) -> dict:
    """{name: (lo, hi)}: ``mean ± nsigma·sd`` intersected with the uniform prior of each parameter
    whose prior that narrows (Gaussian priors untouched). The SMC samples these boxes and corrects
    log Z by the volume ratio (``--sampling-box``)."""
    out = {}
    for n, m, s in zip(names, mean, sd):
        p = priors.get(str(n))
        if p is None or p.gauss is not None or not s > 0:
            continue
        lo, hi = max(p.lo, float(m) - nsigma * float(s)), min(p.hi, float(m) + nsigma * float(s))
        if lo > p.lo or hi < p.hi:
            out[str(n)] = (lo, hi)
    return out


def _common(args) -> list[str]:
    out = ["--config", str(args.config)] if args.config else ["--analysis", args.analysis]
    out += ["--outdir", str(args.outdir)]
    for p in args.plugin:
        out += ["--plugin", p]
    if args.fs_tables:
        out.append("--fs-tables")
    if args.cmb_linear_lens:
        out.append("--cmb-linear-lens")
    if args.cmb_nonlinear_lens:
        out.append("--cmb-nonlinear-lens")
    if args.cmb_legacy_lensing:
        out.append("--cmb-legacy-lensing")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="gpucosmo-evidence", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", type=Path, help="Cobaya yaml")
    src.add_argument("--analysis", help="a yaml shipped with the package (gpucosmo.pipeline.analysis_configs)")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--plugin", action="append", default=[], metavar="MODULE",
                    help="module registering user likelihoods (dotted name or path/to/file.py; repeatable)")
    ap.add_argument("--no-fs-tables", dest="fs_tables", action="store_false",
                    help="DESI FS: expand the linear spectra instead of the REPT tables (slower DALI)")
    ap.add_argument("--cmb-linear-lens", action="store_true")
    ap.add_argument("--cmb-nonlinear-lens", action="store_true")
    ap.add_argument("--cmb-legacy-lensing", action="store_true",
                    help="lens the CMB on the solver's base path, as runs before 2026-10-17 (default: accurate high-ℓ path)")
    ap.add_argument("--tag", default="smc")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smc-target-err", type=float, default=0.009, help="log Z_DALI error at which runs stop")
    ap.add_argument("--smc-max-runs", type=int, default=16)
    ap.add_argument("--dali-order", type=int, default=DALI_ORDER,
                    help="order of the DALI expansion (1 Fisher, 2 doublet, 3 triplet, ...)")
    ap.add_argument("--sampling-box", action="append", default=[], metavar="NAME=LO,HI",
                    help="narrower uniform sampling box for the SMC (log Z corrected; repeatable; see "
                         "gpucosmo-dali-ns and scripts/sampling_boxes.py)")
    ap.add_argument("--auto-boxes", type=float, default=0.0, metavar="NSIGMA",
                    help="without --sampling-box: after the build, box every uniform-prior parameter to "
                         "±NSIGMA σ of the DALI expansion's Laplace approximation at the MAP (needs no earlier "
                         "run; the SMC reports the posterior's distance to the moved edges; e.g. 30)")
    ap.add_argument("--ns-args", default="", help="extra arguments for the SMC stage (gpucosmo-dali-ns)")
    ap.add_argument("--stages", default="map,build,ns")
    args = ap.parse_args(argv)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    args.outdir.mkdir(parents=True, exist_ok=True)
    # The exact stages in this process: set the flags before JAX creates its GPU client.
    env_ns = {k: v for k, v in os.environ.items() if k != "XLA_FLAGS"}
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " " + EXACT_XLA_FLAGS).strip()
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    # Freed device memory goes back to the GPU (JAX's default pool keeps all it ever grew to), so the
    # SMC subprocess finds room once this process drops PR4's matrix.
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    from gpucosmo.cli import dali_hmc, dali_ns
    from gpucosmo.pipeline import shared_bundle

    common = _common(args)
    times: dict[str, float] = {}
    out = args.outdir

    def stamp(msg):
        print(f"[{time.strftime('%F %T')}] {msg}", flush=True)

    # One bundle for the exact stages: one walker per likelihood call, 8-cosmology CMB / P(k)
    # groups (fits next to the PR4 matrix), the matrix on the device.
    with shared_bundle(walker_batch=1, batch_overrides=dict(cmb_walker_batch=8, pk_walker_batch=8)):
        if "map" in stages and not (out / "map.json").exists():
            stamp("map")
            t0 = time.perf_counter()
            dali_hmc.main(common + ["--stage", "map", "--no-autosize"])
            times["map"] = time.perf_counter() - t0
        if "build" in stages and not (out / "dali.npz").exists():
            stamp(f"build (order {args.dali_order})")
            t0 = time.perf_counter()
            dali_hmc.main(common + ["--stage", "build", "--expansion-only", "--no-autosize",
                                    "--dali-order", str(args.dali_order)])
            times["build"] = time.perf_counter() - t0
        if (out / "dali.npz").exists():
            import numpy as np

            z = np.load(out / "dali.npz")
            built = int(z["order"]) if "order" in z.files else 2
            if built != args.dali_order:
                raise SystemExit(f"{out / 'dali.npz'} is an order-{built} expansion, --dali-order is "
                                 f"{args.dali_order}: use another --outdir (or --dali-order {built})")
        if args.auto_boxes > 0 and not args.sampling_box and "ns" in stages and not (
                out / f"{args.tag}.ns_summary.json").exists():
            import numpy as np

            from gpucosmo.jax.inference.cobaya_yaml import resolve_cobaya
            from gpucosmo.pipeline import analysis_configs

            box_path = out / "sampling_boxes.json"
            if not box_path.exists():
                stamp(f"auto boxes: Laplace approximation at the MAP, ±{args.auto_boxes:g} σ")
                t0 = time.perf_counter()
                if not (out / "laplace.npz").exists():
                    dali_ns.main(common + ["--stage", "laplace"])
                z = np.load(out / "laplace.npz")
                cfg_path = args.config or analysis_configs()[args.analysis]
                boxes = sampling_boxes_from_moments(resolve_cobaya(cfg_path).priors, [str(n) for n in z["names"]],
                                                    z["mean"], np.sqrt(np.diag(z["cov"])), args.auto_boxes)
                box_path.write_text(json.dumps(dict(nsigma=args.auto_boxes, boxes=boxes), indent=1) + "\n")
                times["laplace"] = time.perf_counter() - t0
            boxes = json.loads(box_path.read_text())["boxes"]
            args.sampling_box = [f"{n}={lo:.8g},{hi:.8g}" for n, (lo, hi) in boxes.items()]
            stamp(f"sampling boxes: {' '.join(args.sampling_box)}")
        if "ns" in stages and not (out / f"{args.tag}.ns_summary.json").exists():
            stamp("ns (SMC, subprocess)")
            t0 = time.perf_counter()
            import gc

            from gpucosmo.pipeline import _SHARED

            for b_, _r, _d in _SHARED["cache"].values():  # the SMC subprocess needs the GPU memory
                b_.release_matrix()
            gc.collect()
            boxes = [a for b in args.sampling_box for a in ("--sampling-box", b)]
            cmd = [sys.executable, "-m", "gpucosmo.cli.dali_ns", *common, "--stage", "ns",
                   "--compressed-dali", "--tag", args.tag, "--seed", str(args.seed),
                   "--smc-target-err", str(args.smc_target_err), "--smc-runs", str(args.smc_max_runs),
                   *boxes, *args.ns_args.split()]
            # this process keeps the exact stages' programs and PR4 matrix on the GPU: the SMC takes
            # 2048-particle chunks (same speed, ~0.55 GB) within a small pool
            env_ns.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")
            if "--smc-chunk" not in args.ns_args:
                cmd += ["--smc-chunk", "2048"]
            rc = subprocess.call(cmd, env=env_ns)
            if rc:
                raise SystemExit(f"SMC stage failed (exit {rc}): {' '.join(cmd)}")
            times["ns"] = time.perf_counter() - t0
    ns = out / f"{args.tag}.ns_summary.json"
    summary = {"stage_wall_sec": times}
    if ns.exists():
        r = json.loads(ns.read_text())
        summary.update(logz_dali=r["logz_dali"], logz_dali_std=r["logz_dali_std"], dali_order=r.get("dali_order", 2))
        msg = (f"log Z_DALI = {r['logz_dali']:.3f} ± {r['logz_dali_std']:.3f} (order-{summary['dali_order']} DALI "
               f"likelihood)")
        keep = ("order", "lower_order", "delta_logz", "delta_logz_err", "ess_fraction", "max_abs_mean_shift_sigma",
                "max_abs_mean_shift_param")
        checks = r.get("order_checks") or ({str(r["order_check"]["lower_order"]): r["order_check"]}
                                           if r.get("order_check") else {})
        if checks:
            summary["order_check"] = {k: r["order_check"][k] for k in keep}
            summary["order_checks"] = {q: {k: oc[k] for k in keep} for q, oc in checks.items()}
            for oc in checks.values():
                msg += (f"; order {oc['lower_order']} → {oc['order']}: Δ log Z {-oc['delta_logz']:+.4f}, largest mean "
                        f"shift {oc['max_abs_mean_shift_sigma']:.3f} σ")
        stamp(f"{msg}; stage times {', '.join(f'{k} {v:.0f} s' for k, v in times.items())}")
    (out / f"{args.tag}.evidence.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
