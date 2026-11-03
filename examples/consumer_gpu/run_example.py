#!/usr/bin/env python3
"""Lensed CMB spectra on a consumer GPU, checked against CAMB, with cold- and warm-start times.

Needs no data products: the inputs are the cosmological parameters in ``settings.json``. The
script

1. records the environment (device, versions, git commit of the checkout);
2. evaluates the lensed TT/TE/EE and κκ spectra of the cosmologies in ``settings.json`` with the
   GPU Einstein–Boltzmann solver, batched (``get_theory_spectra_jax_eb_batch``), and times the
   first call in a fresh process (*cold*: tracing, XLA compilation, host tables and the run) and
   ``warm_repeats`` further calls (*warm*: the compiled program only);
3. compares the spectra with CAMB at the pinned accuracy settings (computed with ``--camb``, or
   read from ``reference/camb_reference.npz``) and reports the relative differences per ℓ band,
   with a check against the documented accuracy;
4. reports the solver's numerical-failure status (``solver_status``; 0 = valid);
5. writes everything to ``<out>/result_<device>.json``.

    pip install ".[gpu]"                                      # JAX for CUDA 12, from the repository root
    python examples/consumer_gpu/run_example.py --out out_consumer_gpu
    python examples/consumer_gpu/run_example.py --camb        # recompute the CAMB reference (needs camb>=2)

The persistent XLA compilation cache is off unless ``--compile-cache DIR`` is given, so *cold*
includes a full compile. The recorded runs are in ``reference/``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BANDS = ((2, 30), (30, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 2508))


def load_settings(path: Path) -> dict:
    s = json.loads(path.read_text())
    s["names"] = list(s["cosmologies"])
    return s


def repo_relative(path: Path) -> str:
    """``path`` relative to the repository root (no local home paths in recorded results)."""
    try:
        return str(path.resolve().relative_to(HERE.parents[1]))
    except ValueError:
        return path.name


def camb_reference(settings: dict) -> dict:
    """CAMB lensed D_ℓ (μK²) and C_L^κκ (candl units) at the pinned settings."""
    import camb

    c = settings["camb"]
    if not camb.__version__.startswith(c["version_required"]):
        raise SystemExit(f"camb {c['version_required']}.x required, found {camb.__version__}")
    lmax = int(settings["lmax"])
    out = {"camb_version": camb.__version__}
    for name, p in settings["cosmologies"].items():
        cp = camb.set_params(
            thetastar=p["theta_mc_100"] / 100.0, ombh2=p["ombh2"], omch2=p["omch2"],
            As=np.exp(p["logA"]) * 1e-10, ns=p["ns"], tau=p["tau"], w=p["w0"], wa=p["wa"],
            mnu=c["mnu"], num_massive_neutrinos=c["num_massive_neutrinos"], nnu=c["nnu"],
            bbn_predictor=c["bbn_predictor"], dark_energy_model=c["dark_energy_model"],
            AccuracyBoost=c["AccuracyBoost"], lSampleBoost=c["lSampleBoost"], lAccuracyBoost=c["lAccuracyBoost"])
        cp.set_for_lmax(lmax, lens_potential_accuracy=c["lens_potential_accuracy"],
                        lens_output_margin=c["lens_output_margin"], nonlinear=c["nonlinear"])
        r = camb.get_results(cp)
        tot = r.get_cmb_power_spectra(cp, CMB_unit="muK", raw_cl=False)["total"]
        pp = r.get_lens_potential_cls(lmax=lmax, raw_cl=True)[:, 0]
        ell = np.arange(pp.size, dtype=float)
        kk = pp * ell**2 * (ell + 1.0) ** 2 / (2.0 * np.pi) * (np.pi / 2.0)
        for key, col in (("tt", 0), ("ee", 1), ("te", 3)):
            out[f"{name}.{key}"] = tot[: lmax + 1, col]
        out[f"{name}.kk"] = kk[: lmax + 1]
        out[f"{name}.H0"] = np.asarray(cp.H0)
        print(f"CAMB {name}: H0 {cp.H0:.4f}", flush=True)
    return out


def environment() -> dict:
    import jax

    dev = jax.devices()[0]
    env = {"python": platform.python_version(), "platform": platform.platform(), "jax": jax.__version__,
           "jaxlib": getattr(__import__("jaxlib"), "__version__", "?"), "numpy": np.__version__,
           "backend": jax.default_backend(), "device": getattr(dev, "device_kind", str(dev)),
           "n_devices": jax.device_count(), "cpu_count": os.cpu_count(),
           # other load on the machine makes the timings meaningless: record it
           "loadavg_1_5_15min": list(os.getloadavg()) if hasattr(os, "getloadavg") else None}
    try:
        import gpucosmo

        env["gpucosmo"] = getattr(gpucosmo, "__version__", "?")
    except Exception:  # noqa: BLE001
        pass
    root = HERE.parents[1]
    try:
        env["git_commit"] = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                                           text=True, check=True).stdout.strip()
        env["git_dirty"] = bool(subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--", "src"],
                                               capture_output=True, text=True).stdout.strip())
    except Exception:  # noqa: BLE001
        env["git_commit"] = None
    if env["backend"] == "gpu":
        try:
            q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout
            env["nvidia_smi"] = q.strip().splitlines()[0]
        except Exception:  # noqa: BLE001
            pass
    from gpucosmo.jax.eb import cls_jax

    import jax.numpy as jnp

    env["sources_dtype"] = jnp.dtype(cls_jax._resolve_sources_dtype(cls_jax.CLS_SOURCES_DTYPE)).name
    env["source_failure_policy"] = cls_jax.CLS_SOURCE_FAILURE
    return env


def gpu_spectra(settings: dict, batch: int, repeats: int) -> tuple[dict, dict]:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    from gpucosmo.jax.background.cosmology import load_background_tables
    from gpucosmo.jax.cmb.jax_eb_theory import get_theory_spectra_jax_eb_batch

    names = settings["names"]
    rows = [settings["cosmologies"][names[i % len(names)]] for i in range(batch)]
    keys = ("logA", "ns", "theta_mc_100", "ombh2", "omch2", "tau", "w0", "wa")
    args = {k: jnp.asarray([r[k] for r in rows]) for k in keys}

    t0 = time.perf_counter()
    tables = load_background_tables()
    f = jax.jit(lambda a: get_theory_spectra_jax_eb_batch(
        tables, **a, lmax=int(settings["lmax"]), nonlinear_lens=bool(settings["nonlinear_lens"]),
        accurate=bool(settings["accurate"]), return_status=True))
    spec, status = jax.block_until_ready(f(args))
    cold = time.perf_counter() - t0
    warm = []
    for _ in range(repeats):
        t1 = time.perf_counter()
        jax.block_until_ready(f(args))
        warm.append(time.perf_counter() - t1)
    timing = {"batch": batch, "cold_first_call_s": cold, "warm_s": warm, "warm_median_s": float(np.median(warm)),
              "warm_median_per_cosmology_s": float(np.median(warm)) / batch}
    out = {"solver_status": np.asarray(status).tolist()}
    for i, name in enumerate(names[:batch]):
        out[name] = {"tt": np.asarray(spec.dl_tt[i]), "ee": np.asarray(spec.dl_ee[i]),
                     "te": np.asarray(spec.dl_te[i]), "kk": np.asarray(spec.dl_kk[i]), "H0": float(spec.H0[i])}
    return out, timing


def compare(gpu: dict, ref: dict, settings: dict) -> dict:
    res = {}
    for name in settings["names"]:
        if name not in gpu:
            continue
        g = gpu[name]
        r = {k: np.asarray(ref[f"{name}.{k}"]) for k in ("tt", "ee", "te", "kk")}
        d = {"H0_jax": g["H0"], "H0_camb": float(ref[f"{name}.H0"]), "bands": {}}
        for lo, hi in BANDS:
            sl = slice(lo, hi + 1)
            n = min(hi + 1, g["tt"].size, r["tt"].size)
            sl = slice(lo, n)
            norm_te = np.sqrt(r["tt"][sl] * r["ee"][sl])
            d["bands"][f"{lo}-{hi}"] = {
                "tt_rel_max": float(np.max(np.abs(g["tt"][sl] / r["tt"][sl] - 1))),
                "ee_rel_max": float(np.max(np.abs(g["ee"][sl] / r["ee"][sl] - 1))),
                "te_rel_max": float(np.max(np.abs(g["te"][sl] - r["te"][sl]) / norm_te)),
                "kk_rel_mean": float(np.mean(g["kk"][sl] / r["kk"][sl] - 1)),
            }
        sl = slice(1000, min(2509, g["tt"].size))
        d["tt_ee_rel_max_l1000_2508"] = float(max(np.max(np.abs(g["tt"][sl] / r["tt"][sl] - 1)),
                                                  np.max(np.abs(g["ee"][sl] / r["ee"][sl] - 1))))
        lim = settings["checks"]["tt_ee_rel_max_l1000_2508"]
        d["check_tt_ee_l1000_2508"] = {"limit": lim, "pass": d["tt_ee_rel_max_l1000_2508"] <= lim}
        res[name] = d
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--settings", type=Path, default=HERE / "settings.json")
    ap.add_argument("--out", type=Path, default=Path("out_consumer_gpu"))
    ap.add_argument("--batch", type=int, default=None, help="cosmologies per call (default 8 on a GPU, 2 on a CPU)")
    ap.add_argument("--camb", action="store_true", help="compute the CAMB reference instead of reading it")
    ap.add_argument("--camb-ref", type=Path, default=HERE / "reference" / "camb_reference.npz")
    ap.add_argument("--compile-cache", type=Path, default=None,
                    help="persistent XLA compilation cache (then 'cold' is a warm-cache start)")
    args = ap.parse_args()

    settings = load_settings(args.settings)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.camb:
        ref = camb_reference(settings)
        args.camb_ref.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.camb_ref, **ref, settings=json.dumps(settings["camb"]))
        print(f"wrote {args.camb_ref}")
    ref = dict(np.load(args.camb_ref, allow_pickle=False))

    import jax

    if args.compile_cache is not None:
        jax.config.update("jax_compilation_cache_dir", str(args.compile_cache))
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 2.0)
    env = environment()
    batch = args.batch or (8 if env["backend"] == "gpu" else 2)
    print(json.dumps(env, indent=1), flush=True)
    gpu, timing = gpu_spectra(settings, batch, int(settings["timing"]["warm_repeats"]))
    timing["persistent_compile_cache"] = str(args.compile_cache) if args.compile_cache else None
    if hasattr(os, "getloadavg"):
        timing["loadavg_after_1_5_15min"] = list(os.getloadavg())
    acc = compare(gpu, ref, settings)
    result = {"environment": env, "settings": {k: v for k, v in settings.items() if k != "names"},
              "camb_reference": {"file": repo_relative(args.camb_ref), "camb_version": str(ref.get("camb_version"))},
              "timing": timing, "solver_status": gpu["solver_status"], "accuracy_vs_camb": acc}
    tag = "".join(ch if ch.isalnum() else "_" for ch in env["device"]).strip("_").lower()
    path = args.out / f"result_{tag}.json"
    path.write_text(json.dumps(result, indent=1))
    print(json.dumps({"timing": timing, "solver_status": gpu["solver_status"],
                      "accuracy": {n: {"tt_ee_rel_max_l1000_2508": a["tt_ee_rel_max_l1000_2508"],
                                       "pass": a["check_tt_ee_l1000_2508"]["pass"],
                                       "H0": [a["H0_jax"], a["H0_camb"]]} for n, a in acc.items()}}, indent=1))
    print(f"wrote {path}")
    ok = all(s == 0 for s in gpu["solver_status"]) and all(a["check_tt_ee_l1000_2508"]["pass"] for a in acc.values())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
