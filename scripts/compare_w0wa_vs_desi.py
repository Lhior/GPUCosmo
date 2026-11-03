#!/usr/bin/env python3
"""w0waCDM from DESI DR2 BAO + CMB + SN: GPUCosmo's evidence runs against DESI's official chain.

Reads two ``gpucosmo-evidence`` output directories (the w0waCDM and ΛCDM partners of one data
combination) and DESI's public Cobaya chain for that combination, and writes

* a triangle plot of w0, wa, Ωm and H0 (GPUCosmo's SMC particles against DESI's chain);
* ``<out>.json``: posterior summaries, the χ² of each likelihood at both MAPs, Δχ²_MAP and
  Δ log Z_DALI (with each model's DALI order check), each with its "n σ" equivalent (below).

"n σ" conventions:

* Δχ²_MAP → n σ as in DESI DR2: the probability of a χ² with 2 degrees of freedom (w0waCDM has two
  more parameters than ΛCDM) exceeding −Δχ²_MAP, quoted as the two-sided Gaussian n σ.
* Δ log Z → n σ through the Sellke–Bayarri–Berger calibration (B ≤ −1/(e p ln p) for p < 1/e;
  Trotta 2008, arXiv:0803.4089): the p-value whose largest Bayes factor against the null equals the
  measured B, quoted as a two-sided Gaussian n σ. A Bayes factor depends on the parameters' prior
  widths; Δχ²_MAP does not.

    python scripts/compare_w0wa_vs_desi.py --w0wa OUT_W0WA --lcdm OUT_LCDM \\
        --desi-chain <cobaya packages or data>/desi_dr2/cobaya/base_w_wa/desi-bao-all_desy5sn_<CMB>/chain \\
        --out docs/figures/w0wa_desidr2_cmb_desy5_vs_desi

``--desi-chain`` is the chain's file root (``chain.1.txt`` … ``chain.4.txt``, DESI DR2's public
release). The MAP χ² need the analyses' likelihood data (the bundles are rebuilt; a GPU helps).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import optimize, stats

PLOT = ("w", "wa", "omegam", "H0")
LABELS = {"w": r"w_0", "wa": r"w_a", "omegam": r"\Omega_\mathrm{m}", "H0": r"H_0"}


def sigma_from_dchi2(dchi2: float, dof: int = 2) -> tuple[float, float]:
    """(p, n σ two-sided) of a χ² with ``dof`` degrees of freedom exceeding −Δχ²."""
    p = float(stats.chi2.sf(-dchi2, dof))
    return p, float(stats.norm.isf(p / 2.0))


def sigma_from_dlnz(dlnz: float) -> tuple[float, float]:
    """(p, n σ two-sided) with −1/(e p ln p) = B = exp(Δ ln Z) (Sellke, Bayarri & Berger 2001)."""
    B = float(np.exp(dlnz))
    if B <= 1.0:
        return 1.0, 0.0
    p = optimize.brentq(lambda q: -1.0 / (np.e * q * np.log(q)) - B, 1e-300, np.exp(-1.0) - 1e-12)
    return float(p), float(stats.norm.isf(p / 2.0))


def gpucosmo_samples(outdir: Path):
    """GPUCosmo's posterior, the SMC particles (equal weights) of the DALI posterior, with H0 and Ωm
    from the solver's θ → H0 map. Returns (columns, number of draws)."""
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from gpucosmo.jax.background.cosmology import load_background_tables
    from gpucosmo.jax.cmb.theory import solve_h0_from_theta_mc

    z = np.load(outdir / "smc.smc_particles.npz")
    names, X = [str(n) for n in z["names"]], np.asarray(z["pos"])
    col = {n: X[:, i] for i, n in enumerate(names)}
    tables = load_background_tables()
    w0 = col.get("w", np.full(len(X), -1.0))
    wa = col.get("wa", np.zeros(len(X)))
    f = jax.jit(jax.vmap(lambda ob, oc, a, b, t: solve_h0_from_theta_mc(tables, ob, oc, a, b, t, ext={})))
    H0 = np.asarray(f(*(jnp.asarray(v) for v in (col["ombh2"], col["omch2"], w0, wa, col["theta_MC_100"]))))
    omnuh2 = 0.06 / 93.14  # one massive 0.06 eV state, as in the yamls (CAMB's ω_ν = Σm_ν / 93.14 eV)
    om = (col["ombh2"] + col["omch2"] + omnuh2) / (H0 / 100.0) ** 2
    return dict(w=w0, wa=wa, omegam=om, H0=H0), X.shape[0]


def load_cobaya_chain(root: Path, ignore_rows: float, label: str):
    """Cobaya ``<root>.<i>.txt`` chains (header ``# weight minuslogpost <names>``) as MCSamples,
    each chain's first ``ignore_rows`` fraction dropped as burn-in."""
    from getdist import MCSamples

    files = sorted(root.parent.glob(root.name + ".*.txt"))
    if not files:
        raise SystemExit(f"no chains {root}.*.txt")
    names = files[0].open().readline().lstrip("#").split()
    chains = []
    for f in files:
        a = np.loadtxt(f)
        chains.append(a[int(ignore_rows * len(a)):])
    cols = [n for n in names[2:] if n in PLOT]
    idx = [names.index(n) for n in cols]
    return MCSamples(samples=[c[:, idx] for c in chains], weights=[c[:, 0] for c in chains],
                     loglikes=[c[:, 1] for c in chains], names=cols, labels=[LABELS[n] for n in cols],
                     label=label)


def map_chi2(outdir: Path, analysis: str) -> dict:
    """χ² (−2 log L) of each likelihood and in total, and −2 log posterior, at ``map.json``."""
    import jax.numpy as jnp

    from gpucosmo.pipeline import build_analysis_bundle

    m = json.loads((outdir / "map.json").read_text())
    b, _, _ = build_analysis_bundle(analysis)
    if [str(n) for n in b.param_names] != m["names"]:
        raise SystemExit(f"{outdir}/map.json: parameters differ from the {analysis} bundle's")
    x = jnp.asarray(m["x"])
    n_th = int(b.n_theory)
    comps = b.log_like_from_theory_fn(x, b.theory_exact(x[:n_th]), components=True)
    chi2 = {k: float(-2.0 * v) for k, v in comps.items()}
    return dict(chi2=chi2, chi2_total=float(sum(chi2.values())), minus2_logpost=float(-2.0 * m["logpost"]),
                minus2_logprior=float(-2.0 * b.log_prior_fn(x)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--w0wa", type=Path, required=True, help="gpucosmo-evidence output of the w0waCDM analysis")
    ap.add_argument("--lcdm", type=Path, required=True, help="... of the ΛCDM partner")
    ap.add_argument("--w0wa-analysis", default="w0wa_desidr2_cmb_desy5")
    ap.add_argument("--lcdm-analysis", default="lcdm_desidr2_cmb_desy5")
    ap.add_argument("--desi-chain", type=Path, required=True, help="file root of DESI's chain (…/chain)")
    ap.add_argument("--ignore-rows", type=float, default=0.3, help="burn-in fraction of DESI's chain")
    ap.add_argument("--out", type=Path, required=True, help="output root (.png, .pdf, .json)")
    ap.add_argument("--no-map-chi2", action="store_true", help="skip rebuilding the bundles for the MAP χ²")
    ap.add_argument("--reuse-json", action="store_true",
                    help="take the MAP χ² from an existing <out>.json instead of rebuilding the bundles")
    args = ap.parse_args(argv)

    from getdist import MCSamples, plots

    ours, n_ours = gpucosmo_samples(args.w0wa)
    order = json.loads((args.w0wa / "smc.ns_summary.json").read_text()).get("dali_order", 2)
    label = f"GPUCosmo: SMC on the order-{order} DALI likelihood"
    g_ours = MCSamples(samples=np.column_stack([ours[p] for p in PLOT]), names=list(PLOT),
                       labels=[LABELS[p] for p in PLOT], label=label)
    desi = load_cobaya_chain(args.desi_chain, args.ignore_rows, "DESI DR2 official chain (Cobaya MCMC)")

    summary = {"parameters": {}}
    for p in PLOT:
        a = g_ours.getMargeStats().parWithName(p)
        d = desi.getMargeStats().parWithName(p)
        sd_o, sd_d = float(g_ours.std(p)), float(desi.std(p))
        summary["parameters"][p] = dict(
            gpucosmo_mean=float(a.mean), gpucosmo_sd=sd_o, desi_mean=float(d.mean), desi_sd=sd_d,
            shift_in_desi_sd=float((a.mean - d.mean) / sd_d), sd_ratio=sd_o / sd_d)

    rw = json.loads((args.w0wa / "smc.ns_summary.json").read_text())
    rl = json.loads((args.lcdm / "smc.ns_summary.json").read_text())
    dlnz = rw["logz_dali"] - rl["logz_dali"]
    dlnz_err = float(np.hypot(rw["logz_dali_std"], rl["logz_dali_std"]))
    p_b, n_b = sigma_from_dlnz(dlnz)
    summary["evidence"] = dict(dali_order=order, logz_w0wa=rw["logz_dali"], logz_w0wa_err=rw["logz_dali_std"],
                               logz_lcdm=rl["logz_dali"], logz_lcdm_err=rl["logz_dali_std"], dlnz=dlnz,
                               dlnz_err=dlnz_err, bayes_factor=float(np.exp(dlnz)), p_sellke=p_b, n_sigma_sellke=n_b)
    oc_w, oc_l = rw.get("order_check"), rl.get("order_check")
    if oc_w and oc_l:  # the last DALI order's effect: on each log Z and on the Bayes factor
        summary["evidence"]["order_check"] = dict(
            lower_order=oc_w["lower_order"], dlogz_w0wa=-oc_w["delta_logz"], dlogz_lcdm=-oc_l["delta_logz"],
            dlnz_change=-oc_w["delta_logz"] + oc_l["delta_logz"],
            max_mean_shift_sigma_w0wa=oc_w["max_abs_mean_shift_sigma"],
            mean_shift_sigma_w0wa={k: oc_w["mean_shift_sigma"][k] for k in ("w", "wa") if k in oc_w["mean_shift_sigma"]})
    old = args.out.with_suffix(".json")
    if args.reuse_json and old.exists() and "map" in json.loads(old.read_text()):
        summary["map"] = json.loads(old.read_text())["map"]
    elif not args.no_map_chi2:
        mw = map_chi2(args.w0wa, args.w0wa_analysis)
        ml = map_chi2(args.lcdm, args.lcdm_analysis)
        d_like = mw["chi2_total"] - ml["chi2_total"]
        d_post = mw["minus2_logpost"] - ml["minus2_logpost"]
        summary["map"] = dict(w0wa=mw, lcdm=ml, dchi2_loglike=d_like, dchi2_logpost=d_post)
        for k, v in (("loglike", d_like), ("logpost", d_post)):
            p, n = sigma_from_dchi2(v)
            summary["map"][f"p_{k}"], summary["map"][f"n_sigma_{k}"] = p, n

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=1) + "\n")

    gp = plots.get_subplot_plotter(width_inch=7.0)
    gp.settings.legend_fontsize = 10
    gp.triangle_plot([desi, g_ours], list(PLOT), filled=[True, False], contour_colors=["#888888", "#1f5aa6"],
                     legend_loc="upper right")
    ev = summary["evidence"]
    txt = (r"GPUCosmo, w$_0$w$_a$CDM vs $\Lambda$CDM:" + "\n"
           + rf"$\Delta\ln Z = {ev['dlnz']:.2f} \pm {ev['dlnz_err']:.2f}$ ({ev['n_sigma_sellke']:.1f}$\sigma$, Sellke bound)")
    if "map" in summary:
        mp = summary["map"]
        txt += ("\n" + rf"$\Delta\chi^2_\mathrm{{MAP}} = {mp['dchi2_logpost']:.1f}$ ({mp['n_sigma_logpost']:.1f}$\sigma$, 2 dof)")
    gp.fig.text(0.58, 0.64, txt, fontsize=10, va="top")
    for ext in (".png", ".pdf"):
        gp.export(str(args.out.with_suffix(ext)))
    print(json.dumps(summary, indent=1))
    print(f"wrote {args.out}.png/.pdf/.json ({n_ours} GPUCosmo draws)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
