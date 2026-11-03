"""``gpucosmo-dali-ns``: evidence on the DALI likelihood (tempered SMC) and the DALI order check.

Stages (after ``gpucosmo-dali-hmc --stage map`` and ``--stage build`` wrote
``map.json`` / ``dali.npz`` to ``--outdir``; ``scripts/run_dali_ns.py`` is the same
program):

    gpucosmo-dali-ns --config my_analysis.yaml --outdir OUT --stage ns \
        --compressed-dali --smc-particles 8192 --smc-ess 0.8 --smc-mcmc 3

``ns`` samples the DALI posterior with tempered SMC (HMC moves, all on the GPU, log Z_DALI
from the tempering increments) and writes log Z_DALI plus posterior draws; then, for an
expansion of order p ≥ 2, the DALI order check: the same expansion truncated to order p − 1
at the posterior draws, whose importance weights give the last order's effect on log Z and on
the posterior means (``mcmc.dali_ns.dali_order_check``; in ``<tag>.ns_summary.json``).
``--compressed-dali`` evaluates the DALI likelihood through
``dali_reduce.CompressedDALI`` (likelihood linear algebra applied to the expansion coefficients,
no output clipping; ~2× faster without Planck PR4's XLA flags, as it loads no 3.3 GB matrix).
``laplace`` writes the Laplace approximation at the MAP (``gpucosmo-evidence --auto-boxes``).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)


def _parse_boxes(items):
    out = {}
    for it in items:
        name, _, rng = it.partition("=")
        a, b = (float(v) for v in rng.split(","))
        out[name] = (a, b)
    return out or None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="gpucosmo-dali-ns", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plugin", action="append", default=[], metavar="MODULE",
                    help="import MODULE (dotted name or path/to/file.py) first: it registers extra likelihoods "
                         "the yaml can name (repeatable)")
    ap.add_argument("--stage", choices=("laplace", "ns"), required=True,
                    help="laplace: write <outdir>/laplace.npz (Laplace Gaussian of the DALI posterior "
                         "at map.json, for gpucosmo-evidence --auto-boxes; a separate process keeps the gradient "
                         "compile out of the sampler's memory)")
    ap.add_argument("--compressed-dali", action="store_true",
                    help="DALI log L from dali_reduce.CompressedDALI (the unclipped DALI "
                         "expansion; BAO background as expanded outputs; ~6 µs per log L + gradient on an RTX 3070 "
                         "Laptop)")
    ap.add_argument("--compressed-f64", action="store_true",
                    help="with --compressed-dali: HiLLiPoP's quadratic term in float64")
    # Defaults: the settings validated on the DESI DR1 + Planck PR4 + ACT μ–Σ analysis (p25): pilot-fitted
    # reference start, ESS 0.8, 3 HMC moves × 12 leapfrog, 8 final moves, 8 × 8192 particles → log Z_DALI ±0.007
    # in ~5 min on an RTX 3070 Laptop.
    ap.add_argument("--smc-particles", type=int, default=8192)
    ap.add_argument("--smc-runs", type=int, default=8, help="independent SMC populations (the maximum with "
                                                                      "--smc-target-err)")
    ap.add_argument("--smc-target-err", type=float, default=None,
                    help="stop adding runs once the log Z_DALI error is below this (after --smc-min-runs)")
    ap.add_argument("--smc-min-runs", type=int, default=4)
    ap.add_argument("--smc-ess", type=float, default=0.8, help="ESS fraction kept per tempering step")
    ap.add_argument("--smc-mcmc", type=int, default=3, help="HMC transitions per stage")
    ap.add_argument("--smc-leapfrog", type=int, default=12)
    ap.add_argument("--smc-final-mcmc", type=int, default=8, help="HMC transitions at β = 1")
    ap.add_argument("--smc-chunk", type=int, default=4096, help="particles per vmapped log L + gradient call")
    ap.add_argument("--smc-reference", default="pilot", metavar="prior|pilot|FILE.npz",
                    help="start of the tempering path: the prior; a Student-t fitted to a pilot prior-start "
                         "population (--smc-pilot-particles); or one fitted to posterior draws in FILE.npz "
                         "('pos', e.g. an earlier .smc_particles.npz). Same evidence (exact for any start), "
                         "far fewer stages from a fitted reference")
    ap.add_argument("--smc-pilot-particles", type=int, default=2048)
    ap.add_argument("--smc-pilot-ess", type=float, default=0.5, help="the pilot's ESS target (coarser tempering)")
    ap.add_argument("--smc-reference-nu", type=float, default=5.0, help="Student-t degrees of freedom")
    ap.add_argument("--smc-reference-scale", type=float, default=1.3,
                    help="reference covariance = draws' covariance × scale²")
    ap.add_argument("--smc-move", choices=("hmc", "rwmh"), default="hmc",
                    help="SMC mutation: HMC (gradients) or gradient-free random-walk Metropolis (--smc-mcmc steps)")
    ap.add_argument("--analysis", default=None, help="configs/cobaya/<analysis>.yaml")
    ap.add_argument("--config", type=Path, default=None, help="any Cobaya yaml")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--dali-expansion", type=Path, default=None, help="default <outdir>/dali.npz")
    ap.add_argument("--cmb-lensing", action="store_true")
    ap.add_argument("--cmb-nonlinear-lens", action="store_true",
                    help="lens the CMB with the HMcode φφ (overrides the yaml)")
    ap.add_argument("--cmb-legacy-lensing", action="store_true",
                    help="lens the CMB on the solver's base path, as runs before 2026-10-17 (MGConfig.cmb_legacy_lensing)")
    ap.add_argument("--cmb-linear-lens", action="store_true",
                    help="lens the CMB with the linear φφ (overrides the yaml / Cobaya NonLinear_lens "
                         "rule; nested models compared by evidence must use the same choice)")
    ap.add_argument("--bbn-prior", choices=("yaml", "paper"), default="yaml")
    ap.add_argument("--fs-tables", action="store_true",
                    help="DALI built with run_dali_hmc.py --fs-tables (REPT tables expanded)")
    ap.add_argument("--n-posterior", type=int, default=2048)
    ap.add_argument("--n-batch", type=int, default=256, help="DALI points per batched call (order check)")
    ap.add_argument("--no-order-check", dest="order_check", action="store_false",
                    help="skip the DALI order check after the SMC")
    ap.add_argument("--no-ceiling", action="store_true",
                    help="do not reject DALI log L above ll_s(MAP) + max(100, 2·dim)")
    ap.add_argument("--sampling-box", action="append", default=[], metavar="NAME=LO,HI",
                    help="narrower uniform box to sample on (log Z corrected by the volume ratio)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="ns", help="output prefix inside outdir")
    args = ap.parse_args(argv)

    from gpucosmo.cli.dali_hmc import build_analysis_bundle, lens_override  # also enables the JAX cache

    from gpucosmo.jax.inference.cobaya_yaml import dump_resolved
    from gpucosmo.jax.mcmc.dali_ns import make_batched_dali_loglike

    args.outdir.mkdir(parents=True, exist_ok=True)
    log_path = args.outdir / f"{args.tag}.{args.stage}.log"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with log_path.open("a") as fh:
            fh.write(line + "\n")

    b, res, diffs = build_analysis_bundle(
        None if args.config else args.analysis, args.config, walker_batch=1,  # PR4: one walker per likelihood call
        bbn_prior=args.bbn_prior, cmb_lensing=args.cmb_lensing or None,
        cmb_nonlinear_lens=lens_override(args), fs_tables=args.fs_tables, cmb_legacy_lensing=args.cmb_legacy_lensing,
        # the compressed DALI likelihood needs no HiLLiPoP matrix on the GPU
        batch_overrides=dict(defer_hillipop_matrix=args.compressed_dali),
        plugins=args.plugin)
    dump_resolved(b, res, args.outdir / f"config.updated.{args.tag}.{args.stage}.yaml",
                  extra={"audit_diffs": diffs, "cli": {k: str(v) for k, v in vars(args).items()}})
    expansion_path = args.dali_expansion or (args.outdir / "dali.npz")
    expansion = b.dali.load_expansion(expansion_path)
    order = int(getattr(expansion, "order", 2))
    log(f"bundle {b.likelihood_label}: {len(b.param_names)} params; order-{order} DALI expansion {expansion_path}")
    map_path = args.outdir / "map.json"
    if map_path.exists() and "cmb" in expansion.y0:
        # Centre HiLLiPoP's float32 χ² on the expansion point's exact spectra and the MAP
        # nuisances (see HiLLiPoP.loglike).
        b.set_cmb_reference(np.asarray(json.loads(map_path.read_text())["x"]), expansion.y0["cmb"])
    # "compressed2": background as expanded outputs, float32 FS (recorded in the summary)
    dali_kind = "default"
    if args.compressed_dali:
        dali_kind = "compressed2" + ("_f64" if args.compressed_f64 else "")
    if args.compressed_dali:
        from types import SimpleNamespace

        from gpucosmo.jax.inference.dali_reduce import CompressedDALI

        if not map_path.exists():
            raise SystemExit("--compressed-dali centres HiLLiPoP on map.json, which is missing")
        cs = CompressedDALI(b, expansion, np.asarray(json.loads(map_path.read_text())["x"]), log=log,
                            hillipop_f64=args.compressed_f64)
        b0 = b
        b = SimpleNamespace(**vars(b0), dali_loglike=cs.loglike)
        tt = cs.data

        def lower(q):  # the compressed likelihood with the monomials above order q set to zero
            return SimpleNamespace(**vars(b0), dali_loglike=lambda x, d: cs.loglike(x, d, order=q)), tt
    else:
        tt = b.dali.load(expansion_path)

        def lower(q):
            return b, ((tt[0].truncate(q), tt[1]) if hasattr(tt[0], "truncate") else None)

    prefix = args.outdir / args.tag
    ceiling = None
    if args.stage == "ns" and not args.no_ceiling and (args.outdir / "map.json").exists():
        x_map = np.asarray(json.loads((args.outdir / "map.json").read_text())["x"])
        ll_map = float(make_batched_dali_loglike(b, tt, 1)(x_map[None])[0])
        ceiling = ll_map + max(100.0, 2.0 * len(b.param_names))
        log(f"DALI log L at the MAP {ll_map:.3f}; ceiling {ceiling:.3f}")
    if args.stage == "laplace":
        from gpucosmo.jax.mcmc.dali_ns import laplace_gaussian

        lap_path = args.outdir / "laplace.npz"
        x_map = np.asarray(json.loads((args.outdir / "map.json").read_text())["x"])
        t0 = time.perf_counter()
        m, cov, n_bad = laplace_gaussian(b, tt, x_map)
        np.savez(lap_path, mean=m, cov=cov, n_floored=n_bad, names=np.asarray(b.param_names))
        sd = np.sqrt(np.diag(cov))
        log(f"Laplace approximation of the DALI posterior at the MAP ({n_bad} directions "
            f"floored at the prior-box variance; {time.perf_counter() - t0:.0f} s) → {lap_path}: σ " + ", ".join(
                f"{n} {v:.3g}" for n, v in list(zip(b.param_names, sd))[:8]))
        return 0

    def mark_summary(**extra):
        path = prefix.with_suffix(".ns_summary.json")
        summ = json.loads(path.read_text())
        summ.update(dali=dali_kind, dali_order=order, **extra)
        path.write_text(json.dumps(summ, indent=2) + "\n")

    if args.stage == "ns":
        from gpucosmo.jax.mcmc.dali_ns import make_dali_loglike
        from gpucosmo.jax.mcmc.dali_smc import DALISMCConfig, run_dali_smc

        cfg = DALISMCConfig(n_particles=args.smc_particles, n_runs=args.smc_runs, ess_target=args.smc_ess,
                                 n_mcmc=args.smc_mcmc, n_leapfrog=args.smc_leapfrog,
                                 n_final_mcmc=args.smc_final_mcmc, chunk=args.smc_chunk, seed=args.seed,
                                 move=args.smc_move, reference=args.smc_reference,
                                 pilot_particles=args.smc_pilot_particles, pilot_ess=args.smc_pilot_ess,
                                 reference_nu=args.smc_reference_nu,
                                 reference_scale=args.smc_reference_scale, target_err=args.smc_target_err,
                                 min_runs=args.smc_min_runs,
                                 n_posterior=args.n_posterior, sampling_box=_parse_boxes(args.sampling_box),
                                 loglike_ceiling=ceiling)
        run_dali_smc(b, make_dali_loglike(b), tt, cfg, prefix, log=log)
        mark_summary()
        if args.order_check and order >= 2 and lower(order - 1)[1] is not None:
            # p − 1 → p, and from order 4 on also p − 2 → p: consecutive orders add terms of alternating
            # parity, so an even-order error is invisible to an odd step and vice versa
            checks = {str(q): order_check(b, tt, *lower(q), prefix, order, q, args.n_batch, log)
                      for q in [order - 1] + ([order - 2] if order >= 4 else [])}
            mark_summary(order_check=checks[str(order - 1)], order_checks=checks)
        return 0
    raise AssertionError(args.stage)


def order_check(b, tt, b_lower, tt_lower, prefix, order: int, lower_order: int, n_batch: int, log) -> dict:
    """``dali_order_check`` at the SMC's posterior draws: order p against the same expansion
    truncated to ``lower_order``."""
    from gpucosmo.jax.mcmc.dali_ns import dali_order_check, make_batched_dali_loglike

    t0 = time.perf_counter()
    X = np.load(prefix.with_suffix(".ns_posterior.npz"))["pos"]
    ll_p = make_batched_dali_loglike(b, tt, n_batch)(X)
    ll_q = make_batched_dali_loglike(b_lower, tt_lower, n_batch)(X)
    out = dali_order_check(X, ll_p, ll_q, b.param_names, order=order, lower_order=lower_order)
    log(f"DALI order check ({X.shape[0]} draws, {time.perf_counter() - t0:.0f} s): order {lower_order} → {order} "
        f"changes log Z by {-out['delta_logz']:+.4f} ± {out['delta_logz_err']:.4f} (weight ESS/N "
        f"{out['ess_fraction']:.3f}); largest posterior-mean shift {out['max_abs_mean_shift_sigma']:.3f} σ "
        f"({out['max_abs_mean_shift_param']})")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
