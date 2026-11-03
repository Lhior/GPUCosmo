"""Unlensed CMB damping tail of jax_eb vs converged CAMB (AccuracyBoost = 3).

Guards the Cls hierarchy caps: with the massive-ν hierarchy truncated at ℓ = 10,
TT ran +6…8e-4 and EE +9e-4 high at ℓ > 1000 (n_s −0.12σ against PR4).
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
camb = pytest.importorskip("camb")

# Official DESI DR1 + PR4 ΛCDM posterior mean (outputs/theory_check/camb_gr_mean.npz).
POINT = dict(H0=67.74637, ombh2=0.0222801, omch2=0.1186332, logA=3.054436, ns=0.9694608, tau=0.0614381)
T_CMB_UK = 2.7255e6
BANDS = ((500, 1000), (1000, 1500), (1500, 2000), (2000, 2509))
MAX_MEAN_RESIDUAL = {"tt": 3e-4, "ee": 3e-4}


@pytest.mark.slow
def test_unlensed_damping_tail_matches_converged_camb():
    jax.config.update("jax_enable_x64", True)
    from gpucosmo.jax.eb.cls_jax import compute_cmb_cls

    p = POINT
    As = np.exp(p["logA"]) * 1e-10
    cp = camb.set_params(H0=p["H0"], ombh2=p["ombh2"], omch2=p["omch2"], As=As, ns=p["ns"], tau=p["tau"],
                         mnu=0.06, nnu=3.044, num_massive_neutrinos=1,
                         bbn_predictor="PArthENoPE_880.2_standard.dat", dark_energy_model="ppf",
                         AccuracyBoost=3, lSampleBoost=3, lAccuracyBoost=3)
    cp.set_for_lmax(3008, lens_potential_accuracy=0)
    ref = camb.get_results(cp).get_cmb_power_spectra(cp, CMB_unit="muK", raw_cl=False)["unlensed_scalar"]

    cls = compute_cmb_cls(ombh2=p["ombh2"], omch2=p["omch2"], h=p["H0"] / 100.0, tau_reio=p["tau"], ns=p["ns"],
                          As=As, w0=-1.0, wa=0.0, lmax=2508, nonlinear_lens=False)
    ell = np.asarray(cls["ell"], dtype=float)
    e = ell.astype(int)
    fac = ell * (ell + 1.0) / (2.0 * np.pi) * T_CMB_UK**2
    ours = {"tt": np.asarray(cls["tt"]) * fac,
            "ee": np.asarray(cls["ee"]) * fac * (ell * ell - 1.0) * (ell + 2.0) * ell}
    camb_dl = {"tt": ref[:, 0], "ee": ref[:, 1]}
    for s, cap in MAX_MEAN_RESIDUAL.items():
        for lo, hi in BANDS:
            m = (e >= lo) & (e < hi)
            r = float(np.mean(ours[s][m] / camb_dl[s][e[m]] - 1.0))
            assert abs(r) < cap, f"{s.upper()} ℓ {lo}-{hi}: mean residual {r:+.2e} vs converged CAMB"
