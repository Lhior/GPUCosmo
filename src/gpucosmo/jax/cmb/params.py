#!/usr/bin/env python3
"""Parameter names, priors and fiducials for desi_full (DESI-matched cosmology + plik)."""

from __future__ import annotations

import numpy as np

# Cosmology block (DESI chain.updated.yaml)
COSMO_NAMES = (
    "logA",
    "ns",
    "theta_MC_100",
    "ombh2",
    "omch2",
    "tau",
    "w",
    "wa",
)

# plik_lite / clipy calibration
NUISANCE_NAMES = ("A_planck",)

PARAM_NAMES = COSMO_NAMES + NUISANCE_NAMES

COSMO_PRIOR_LO = np.array([1.61, 0.8, 0.5, 0.005, 0.001, 0.01, -3.0, -3.0])
COSMO_PRIOR_HI = np.array([3.91, 1.2, 10.0, 0.1, 0.99, 0.8, 1.0, 2.0])
COSMO_FIDUCIAL = np.array([3.036, 0.9649, 1.04109, 0.02237, 0.12, 0.0544, -0.83, -0.6])
COSMO_PROPOSAL = np.array([0.001, 0.002, 0.0002, 0.0001, 0.0005, 0.003, 0.02, 0.05])

NUISANCE_PRIOR_LO = np.array([0.9])
NUISANCE_PRIOR_HI = np.array([1.1])
NUISANCE_FIDUCIAL = np.array([1.0])
NUISANCE_PROPOSAL = np.array([0.002])

PRIOR_LO = np.concatenate([COSMO_PRIOR_LO, NUISANCE_PRIOR_LO])
PRIOR_HI = np.concatenate([COSMO_PRIOR_HI, NUISANCE_PRIOR_HI])
FIDUCIAL = np.concatenate([COSMO_FIDUCIAL, NUISANCE_FIDUCIAL])
PROPOSAL_SCALES = np.concatenate([COSMO_PROPOSAL, NUISANCE_PROPOSAL])

T_CMB_MUK = 2.7255e6

# Wide box priors for clipy plik / CamSpec nuisances (fallback when YAML not loaded)
_PLIK_NUISANCE_DEFAULTS: dict[str, tuple[float, float, float, float]] = {
    # name: (lo, hi, fiducial, proposal)
    "A_planck": (0.9, 1.1, 1.0, 0.002),
    "calPlanck": (0.9, 1.1, 1.0, 0.002),
    # CamSpec NPIPE TTTEEE (power-law FG + pol cal)
    "amp_143": (0.0, 50.0, 10.0, 1.0),
    "amp_217": (0.0, 50.0, 20.0, 1.0),
    "amp_143x217": (0.0, 50.0, 10.0, 1.0),
    "n_143": (0.0, 5.0, 1.0, 0.2),
    "n_217": (0.0, 5.0, 1.0, 0.2),
    "n_143x217": (0.0, 5.0, 1.0, 0.2),
    "calTE": (0.9, 1.1, 1.0, 0.01),
    "calEE": (0.9, 1.1, 1.0, 0.01),
}


# Planck 2018 plik TTTEEE nuisance priors, as in Cobaya's
# likelihoods/planck_2018_highl_plik/params_*.yaml + base_classes/planck_calib.yaml:
# name: (lo, hi, ref, proposal, gauss) with gauss = (μ, σ) or None. Gaussian-prior
# parameters get a ±10σ box for bounds only. Plus the SZ prior
# ksz_norm + 1.6·A_sz ~ N(9.5, 3) (prior_SZ.yaml), applied in the bundle.
def _g(mu: float, sd: float, prop: float):
    return (mu - 10 * sd, mu + 10 * sd, mu, prop, (mu, sd))


PLIK_2018_PRIORS: dict[str, tuple[float, float, float, float, tuple | None]] = {
    "A_cib_217": (0.0, 200.0, 67.0, 1.2, None),
    "xi_sz_cib": (0.0, 1.0, 0.1, 0.1, None),
    "A_sz": (0.0, 10.0, 7.0, 0.6, None),
    "ksz_norm": (0.0, 10.0, 3.0, 1.0, None),
    "ps_A_100_100": (0.0, 400.0, 257.0, 17.0, None),
    "ps_A_143_143": (0.0, 400.0, 47.0, 3.0, None),
    "ps_A_143_217": (0.0, 400.0, 40.0, 2.0, None),
    "ps_A_217_217": (0.0, 400.0, 104.0, 2.5, None),
    "gal545_A_100": _g(8.6, 2.0, 1.0),
    "gal545_A_143": _g(10.6, 2.0, 1.0),
    "gal545_A_143_217": _g(23.5, 8.5, 1.5),
    "gal545_A_217": _g(91.9, 20.0, 2.0),
    "galf_TE_A_100": _g(0.130, 0.042, 0.1),
    "galf_TE_A_100_143": _g(0.130, 0.036, 0.1),
    "galf_TE_A_100_217": _g(0.46, 0.09, 0.1),
    "galf_TE_A_143": _g(0.207, 0.072, 0.1),
    "galf_TE_A_143_217": _g(0.69, 0.09, 0.1),
    "galf_TE_A_217": _g(1.938, 0.54, 0.2),
    "calib_100T": _g(1.0002, 0.0007, 0.0005),
    "calib_217T": _g(0.99805, 0.00065, 0.0005),
    "A_planck": _g(1.0, 0.0025, 0.0005),
}
SZ_PRIOR = (1.6, 9.5, 3.0)  # ksz_norm + 1.6·A_sz ~ N(9.5, 3)


def nuisance_gaussian_priors(names) -> tuple[np.ndarray, np.ndarray]:
    """(μ, σ) per parameter; σ = inf where the prior is flat."""
    mu = np.zeros(len(names))
    sd = np.full(len(names), np.inf)
    for i, n in enumerate(names):
        spec = PLIK_2018_PRIORS.get(n)
        if spec is not None and spec[4] is not None:
            mu[i], sd[i] = spec[4]
    return mu, sd


def build_desi_full_params(
    nuisance_names: tuple[str, ...] = NUISANCE_NAMES,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assemble param vectors for cosmology + discovered plik nuisances."""
    names = COSMO_NAMES + nuisance_names
    lo: list[float] = list(COSMO_PRIOR_LO)
    hi: list[float] = list(COSMO_PRIOR_HI)
    fid: list[float] = list(COSMO_FIDUCIAL)
    prop: list[float] = list(COSMO_PROPOSAL)
    for n in nuisance_names:
        if n in PLIK_2018_PRIORS:
            a, b, c, d, _ = PLIK_2018_PRIORS[n]
        elif n in _PLIK_NUISANCE_DEFAULTS:
            a, b, c, d = _PLIK_NUISANCE_DEFAULTS[n]
        else:
            a, b, c, d = (0.0, 200.0, 1.0, 0.1)
        lo.append(a)
        hi.append(b)
        fid.append(c)
        prop.append(d)
    return (
        names,
        np.asarray(lo),
        np.asarray(hi),
        np.asarray(fid),
        np.asarray(prop),
    )
