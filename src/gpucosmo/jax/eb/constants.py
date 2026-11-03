#!/usr/bin/env python3
# This file is part of GPUCosmo. Copyright (C) 2025-2026 Leonel Medina-Varela.
# SPDX-License-Identifier: GPL-3.0-or-later
# Derived in part from the following, translated to JAX and modified for GPUCosmo:
# - CAMB (https://github.com/cmbant/CAMB), Copyright 2000-2025 Antony Lewis and Anthony Challinor.
#   Licence: LICENSES/CAMB.txt.
# See NOTICE.md.
"""CAMB ``constants.f90`` units used by the transfer-only EB path."""

from __future__ import annotations

import numpy as np

# Precision
DL = np.float64

# Mathematical
PI = 3.1415926535897932384626433832795
TWOPI = 2.0 * PI
FOURPI = 4.0 * PI
EIGHTPI = 8.0 * PI
ZETA3 = 1.2020569031595942853997

# SI / astrophysical (CAMB constants.f90, PDG 2022)
C_LIGHT = 2.99792458e8  # m/s
C_KM_S = 2.99792458e5  # km/s
H_P = 6.62607015e-34
HBAR = H_P / TWOPI
G_NEWTON = 6.67430e-11
SIGMA_THOMSON = 6.6524587321e-29  # m^2
K_B = 1.380649e-23
EV = 1.602176634e-19
M_P = 1.67262192369e-27
M_E = 9.1093837015e-31
M_H = 1.673575e-27
M_HE4 = 6.646479073e-27
MASS_RATIO_HE_H = M_HE4 / M_H

MPC = 3.085677581e22  # m
MPC_IN_SEC = MPC / C_LIGHT

BARSSC0 = K_B / M_P / C_LIGHT**2
KAPPA = 8.0 * PI * G_NEWTON
A_RAD = 8.0 * PI**5 * K_B**4 / (15.0 * C_LIGHT**3 * H_P**3)
COMPTON_CT = MPC_IN_SEC * (8.0 / 3.0) * (SIGMA_THOMSON / (M_E * C_LIGHT)) * A_RAD

COBE_CMB_TEMP = 2.7255  # K
DEFAULT_NNU = 3.044
# DESI DR2 base_w_wa: one massive eigenstate, sum m_nu = 0.06 eV.
DEFAULT_MNU = 0.06
DEFAULT_NUM_MASSIVE_NU = 1
# Extended cosmology parameters and their defaults (CAMB's): Σm_ν [eV], N_eff, sterile m_eff [eV], Ω_k.
EXT_DEFAULTS = {"mnu": DEFAULT_MNU, "nnu": DEFAULT_NNU, "meffsterile": 0.0, "omk": 0.0}

# H0 = 100 km/s/Mpc in 1/Mpc (c=1 natural): H0/c with H0=100 → 1/2997.92458 /Mpc
H0_100_INV_MPC = 100.0 / C_KM_S  # ≈ 3.3356e-4 / Mpc

# Radiation density today, derived exactly as CAMB results.f90 does:
#   grhog    = kappa/c^2 * 4 sigma_SB/c^3 * T^4 * Mpc^2
#   grhocrit = 3 (1e5/c)^2
# The previously hardcoded 2.472e-5 was 3.9e-4 low, which biased the neutrino
# mass inversion and the radiation-matter transition by the same fraction --
# several times the 1e-3 accuracy target.
SIGMA_SB = 2.0 * PI**5 * K_B**4 / (15.0 * H_P**3 * C_LIGHT**2)
_GRHOG = KAPPA / C_LIGHT**2 * 4.0 * SIGMA_SB / C_LIGHT**3 * COBE_CMB_TEMP**4 * MPC**2
_GRHOCRIT = 3.0 * (1.0e5 / C_LIGHT) ** 2
OMEGA_G_H2 = _GRHOG / _GRHOCRIT  # 2.472975327846e-05

# ρ_ν / ρ_γ per massless species = (7/8) * (4/11)^{4/3}
RHO_NU_OVER_RHO_G = (7.0 / 8.0) * (4.0 / 11.0) ** (4.0 / 3.0)

# Helium mass fraction default (BBN-ish; CAMB YHe from BBN or set)
YHE_DEFAULT = 0.2453  # close to CAMB BBN; overridden when ombh2 known


def yhe_bbn(ombh2: float, nnu: float = DEFAULT_NNU) -> float:
    """CAMB BBN helium mass fraction, from the same interpolation table CAMB uses.

    The linear fit this replaces was 0.18% off the DESI chains' ``YHe``; the table
    reproduces it to 2e-8. Use ``bbn_jax.yhe_from_bbn`` on traced ``ombh2``.
    """
    from gpucosmo.jax.eb.bbn_jax import delta_neff_for, yhe_from_bbn

    return float(yhe_from_bbn(float(ombh2), delta_neff=delta_neff_for(nnu)))


# Converts Σm_ν [eV] to ω_ν for non-relativistic thermal neutrinos (CAMB constants.f90
# ``neutrino_mass_fac``). ω_ν itself follows CAMB 2.0.4: massive_nu.NeutrinoModel.resolve.
NEUTRINO_MASS_FAC = 94.064105812176123

# Thomson opacity prefactor: akthom = σ_T * n_H0 * Mpc  (CAMB units)
# n_H0 = (1-YHe) * ρ_b0 / m_H ; built in thermo from Ω_b h²

# Transfer-only multipole caps (AccuracyBoost=1)
LMAXG_TRANSFER = 6
LMAXGPOL_TRANSFER = 5
LMAXNR_TRANSFER = 14

# Cls multipole caps. CAMB WantCls at AccuracyBoost 1 uses lmaxg 11–13, lmaxgpol
# 11–14, lmaxnr 14. 12/10/15 left TT/EE −1…−4e-4 short of converged CAMB at
# ℓ > 1500; 16/14/20 is within 5e-5 (TT) / 1.4e-4 (EE) at ℓ > 1000 (24/20/30
# overshoots by −1…−2e-4). The evolve is launch-latency bound: same wall time.
LMAXG_CLS = 16
LMAXGPOL_CLS = 14
LMAXNR_CLS = 20
# Massive-neutrino multipole cap and momentum-bin count (CAMB's 3 optimized
# momenta at AccuracyBoost 1). Every bin is evolved massive from the start, so
# all of ρ_ν sits in this hierarchy; CAMB at boost 1 keeps the two high-q bins
# (99% of ρ_ν) on the massless hierarchy until z ≈ 1050 / 455. Truncated at
# ℓ = 10, the reflections at kτ ≳ 10 raised the acoustic amplitude at k ≥ 0.1
# by 3.5–4e-4 (TT +6–8e-4, EE +9e-4 at ℓ > 1000 vs converged CAMB; n_s −0.12σ).
# 20 is converged (30 gives the same sources).
LMAXNU_CLS = 20
NQMAX_CLS = 3
# Default massive-ν cap of HierarchyConfig (matter power / transfer paths).
LMAXNU_TRANSFER = 10

# ODE tolerances near CAMB base_tol
BASE_TOL = 1.0e-4
