#!/usr/bin/env python3
"""Map GPUCosmo cosmology parameters to CosmoPower-JAX emulator inputs."""

from __future__ import annotations

import numpy as np

from gpucosmo.jax.power import CosmologyParams

from gpucosmo.jax.sigma8 import LN10AS_REF

# HMCode 2020 baryon feedback defaults (CosmoPower mpk_boost training values).
HMCODE_CMIN = 3.13
HMCODE_ETA0 = 0.603


def gpucosmo_to_cosmopower_lin_params(
    params: CosmologyParams,
    *,
    z: float,
    ln10A_s: float = LN10AS_REF,
) -> dict[str, np.ndarray]:
    """CosmoPower mpk_lin parameter dict for a single redshift."""
    return {
        "omega_b": np.array([params.Omega_b * params.h**2]),
        "omega_cdm": np.array([params.Omega_c * params.h**2]),
        "h": np.array([params.h]),
        "n_s": np.array([params.n_s]),
        "ln10^{10}A_s": np.array([ln10A_s]),
        "z": np.array([float(z)]),
    }


def gpucosmo_to_cosmopower_boost_params(
    params: CosmologyParams,
    *,
    z: float,
    ln10A_s: float = LN10AS_REF,
    cmin: float = HMCODE_CMIN,
    eta_0: float = HMCODE_ETA0,
) -> dict[str, np.ndarray]:
    """CosmoPower mpk_boost / mpk_nonlin parameter dict for a single redshift."""
    out = gpucosmo_to_cosmopower_lin_params(params, z=z, ln10A_s=ln10A_s)
    out["cmin"] = np.array([cmin])
    out["eta_0"] = np.array([eta_0])
    return out


__all__ = [
    "HMCODE_CMIN",
    "HMCODE_ETA0",
    "LN10AS_REF",
    "gpucosmo_to_cosmopower_boost_params",
    "gpucosmo_to_cosmopower_lin_params",
]
