#!/usr/bin/env python
"""Validate the Cobaya bridge's conventions (p26).

For each likelihood: Cobaya + CAMB at a reference cosmology gives log L_camb; the same CAMB
products copied into ``cobaya_bridge.BridgeTheory`` (through ``CobayaLikelihood.host_loglike``)
must give the same value — any difference is a convention error in the bridge (units,
ℓ factors, z pools, parameter names), not a theory difference. With ``--ours`` the bridge is
also evaluated on our background theory (background-only likelihoods; CPU), the difference
being ours − CAMB's background.

    python scripts/validate_cobaya_bridge.py bao.desi_dr2 sn.desy5 planck_2018_lensing.native ...
"""

from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

REF = dict(H0=67.36, ombh2=0.02237, omch2=0.1200, As=2.1e-9, ns=0.9649, tau=0.0544, mnu=0.06, nnu=3.044)


def camb_products(model, req):
    """The bridge products from a Cobaya model whose theory is CAMB (after a logposterior call)."""
    p = model.provider
    prods = {"params": {}}
    for q, z in req["z"].items():
        prods[q] = np.asarray(getattr(p, f"get_{q}")(z), dtype=float)
    if req["z_pairs"] is not None:
        prods["angular_diameter_distance_2"] = np.asarray(p.get_angular_diameter_distance_2(req["z_pairs"]))
    for which in ("Cl",):
        if req[which]:
            cl = p.get_Cl(ell_factor=True, units="FIRASmuK2")
            T2 = 2.7255e6**2
            prods[which] = {s: (np.asarray(cl[s]) if s == "pp" else np.asarray(cl[s]) / T2)
                            for s in ("tt", "te", "ee", "bb", "pp") if s in cl}
    for n in req["params"]:
        prods["params"][n] = float(p.get_param(n))
    return prods


def run(name, ours: bool):
    from cobaya.model import get_model

    from gpucosmo.cobaya_bridge.bridge import CobayaLikelihood
    from gpucosmo.paths import packages_path

    L = CobayaLikelihood(name)
    pt_nuis = {p: L.params[p].start for p in L.params}
    info = {"likelihood": {name: None},
            "theory": {"camb": {"extra_args": {"lens_potential_accuracy": 1, "num_massive_neutrinos": 1,
                                               "halofit_version": "mead2020",
                                               # the solver's BBN helium table (CAMB's default is PRIMAT)
                                               "bbn_predictor": "PArthENoPE_880.2_standard.dat"}}},
            "params": {**REF, **{p: v for p, v in pt_nuis.items()}}, "packages_path": str(packages_path())}
    t0 = time.perf_counter()
    m = get_model(info)
    ll_camb = float(np.sum(m.loglikes({}, return_derived=False)))
    if m.prior.external:
        ll_camb += float(np.sum(m.prior.logps_external(m.parameterization.to_input({}))))
    t_camb = time.perf_counter() - t0
    prods = camb_products(m, L._req)
    pt = {k: v for k, v in dict(H0=REF["H0"], ombh2=REF["ombh2"], omch2=REF["omch2"], logA=np.log(1e10 * REF["As"]),
                                As=REF["As"], ns=REF["ns"], tau=REF["tau"], w=-1.0, wa=0.0, mu0=0.0, Sigma0=0.0,
                                mnu=0.06, nnu=3.044, omk=0.0, meffsterile=0.0).items() if k in L._theory_in}
    pt.update(pt_nuis)
    prods["params"].update({k: v for k, v in pt.items()})
    t0 = time.perf_counter()
    ll_bridge = L.host_loglike(prods, pt)
    t_bridge = time.perf_counter() - t0
    out = dict(name=name, logL_camb=ll_camb, logL_bridge_camb_products=ll_bridge, diff=ll_bridge - ll_camb,
               t_camb_s=t_camb, t_bridge_eval_s=t_bridge)
    if ours and not L.requires:
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
        from gpucosmo.jax.background.cosmology import load_background_tables
        from gpucosmo.likelihoods.base import Cosmology

        c = Cosmology({k: jnp.asarray(v) for k, v in dict(h=REF["H0"] / 100, ombh2=REF["ombh2"], omch2=REF["omch2"],
                                                         ns=REF["ns"], As=REF["As"], tau=REF["tau"], w0=-1.0,
                                                         wa=0.0).items()} | {"mu0": None, "Sigma0": None},
                      load_background_tables())
        th = jax.tree_util.tree_map(np.asarray, L.theory(c))
        p2 = L.products(th, None)
        p2["params"].update(pt)
        out["logL_bridge_ours"] = L.host_loglike(p2, pt)
        out["diff_ours_vs_camb"] = out["logL_bridge_ours"] - ll_camb
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+")
    ap.add_argument("--ours", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    res = []
    for n in args.names:
        try:
            r = run(n, args.ours)
        except Exception as e:  # noqa: BLE001
            r = dict(name=n, error=f"{type(e).__name__}: {str(e)[:300]}")
        res.append(r)
        print(json.dumps(r), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
