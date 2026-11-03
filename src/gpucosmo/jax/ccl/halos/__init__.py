#!/usr/bin/env python3
# This file is part of GPUCosmo. Copyright (C) 2025-2026 Leonel Medina-Varela.
# SPDX-License-Identifier: GPL-3.0-or-later
# Derived in part from the following, translated to JAX and modified for GPUCosmo:
# - CCL (https://github.com/LSSTDESC/CCL), Copyright (c) 2016-2018, the LSSTDESC CCL
#   contributors.
#   Licence: LICENSES/CCL.txt.
# See NOTICE.md.
"""JAX port of ``pyccl.halos`` for halo-model (intrinsic-alignment) power spectra.

Same names and call syntax as pyccl, so ``import pyccl as ccl`` can be swapped
for ``import gpucosmo.jax.ccl as ccl``::

    hmd = ccl.halos.MassDef200m
    cM = ccl.halos.ConcentrationDuffy08(mass_def=hmd)
    hmc = ccl.halos.HMCalculator(mass_function=ccl.halos.MassFuncTinker08(mass_def=hmd),
                                 halo_bias=ccl.halos.HaloBiasTinker10(mass_def=hmd),
                                 mass_def=hmd, nM=64)
    sat = ccl.halos.SatelliteShearHOD(mass_def=hmd, concentration=cM, a1h=1e-3, b=-2)
    nfw = ccl.halos.HaloProfileNFW(mass_def=hmd, concentration=cM)
    pk_GI_1h = ccl.halos.halomod_Pk2D(cosmo, hmc, nfw, prof2=sat, get_2h=False)

Everything is jittable and differentiable w.r.t. the cosmology (background
splines and linear P(k) table of :class:`gpucosmo.jax.ccl.Cosmology`) and the
HOD / IA parameters. Limitations: massless neutrinos in ``omega_x``/``rho_x``;
``SatelliteShearHOD`` only with ``integration_method='FFTLog'``; FFTLog
profiles need a concrete k array; truncation radii enter the FFTLog profiles
through a step function (no gradient through the truncation edge position).
"""

from gpucosmo.jax.ccl.halos._sigma import (
    compute_sigma,
    compute_sigma_spline,
    cosmology_from_ccl,
    dlnsigM_dlogM,
    get_delta_c,
    sigma8,
    sigmaM,
    sigma_spline_from_ccl,
    sigmaR,
)
from gpucosmo.jax.ccl.halos.halo_model import (
    HMCalculator,
    Profile2pt,
    Profile2ptHOD,
    halomod_bias_1pt,
    halomod_mean_profile_1pt,
    halomod_Pk2D,
    halomod_power_spectrum,
)
from gpucosmo.jax.ccl.halos.ingredients import (
    Concentration,
    ConcentrationBhattacharya13,
    ConcentrationConstant,
    ConcentrationDuffy08,
    ConcentrationKlypin11,
    HaloBias,
    HaloBiasBhattacharya11,
    HaloBiasSheth01,
    HaloBiasSheth99,
    HaloBiasTinker10,
    HMIngredients,
    MassFunc,
    MassFuncAngulo12,
    MassFuncDespali16,
    MassFuncJenkins01,
    MassFuncPress74,
    MassFuncSheth99,
    MassFuncTinker08,
    MassFuncTinker10,
)
from gpucosmo.jax.ccl.halos.massdef import (
    MassDef,
    MassDef200c,
    MassDef200m,
    MassDef500c,
    MassDefFof,
    MassDefVir,
    convert_concentration,
    mass2radius_lagrangian,
    mass_translator,
)
from gpucosmo.jax.ccl.halos.profiles import (
    FFTLogParams,
    HaloProfile,
    HaloProfileHOD,
    HaloProfileMatter,
    HaloProfileNFW,
    SatelliteShearHOD,
)

__all__ = [
    "Concentration",
    "ConcentrationBhattacharya13",
    "ConcentrationConstant",
    "ConcentrationDuffy08",
    "ConcentrationKlypin11",
    "FFTLogParams",
    "HMCalculator",
    "HMIngredients",
    "HaloBias",
    "HaloBiasBhattacharya11",
    "HaloBiasSheth01",
    "HaloBiasSheth99",
    "HaloBiasTinker10",
    "HaloProfile",
    "HaloProfileHOD",
    "HaloProfileMatter",
    "HaloProfileNFW",
    "MassDef",
    "MassDef200c",
    "MassDef200m",
    "MassDef500c",
    "MassDefFof",
    "MassDefVir",
    "MassFunc",
    "MassFuncAngulo12",
    "MassFuncDespali16",
    "MassFuncJenkins01",
    "MassFuncPress74",
    "MassFuncSheth99",
    "MassFuncTinker08",
    "MassFuncTinker10",
    "Profile2pt",
    "Profile2ptHOD",
    "SatelliteShearHOD",
    "compute_sigma",
    "compute_sigma_spline",
    "convert_concentration",
    "cosmology_from_ccl",
    "dlnsigM_dlogM",
    "get_delta_c",
    "halomod_Pk2D",
    "halomod_bias_1pt",
    "halomod_mean_profile_1pt",
    "halomod_power_spectrum",
    "mass2radius_lagrangian",
    "mass_translator",
    "sigma8",
    "sigmaM",
    "sigmaR",
    "sigma_spline_from_ccl",
]
