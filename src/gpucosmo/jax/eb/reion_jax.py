#!/usr/bin/env python3
# This file is part of GPUCosmo. Copyright (C) 2025-2026 Leonel Medina-Varela.
# SPDX-License-Identifier: GPL-3.0-or-later
# Derived in part from the following, translated to JAX and modified for GPUCosmo:
# - CAMB (https://github.com/cmbant/CAMB), Copyright 2000-2025 Antony Lewis and Anthony Challinor.
#   Licence: LICENSES/CAMB.txt.
# See NOTICE.md.
"""CAMB-style tanh reionization + HeII, including z_re(τ) solve.

Mirrors ``fortran/reionization.f90`` ``TTanhReionization`` / ``TBaseTauWithHeReionization``:
hydrogen + first helium ionization follow a tanh in ``(1+z)^1.5``, and a second
tanh adds the remaining helium electron below ``helium_redshift``.
``z_re`` is solved so the Romberg-style integral of ``xe_reion * akthom * dτ/da``
equals the target optical depth (CAMB ``GetReionizationOptDepth``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# CAMB defaults
TANH_ZEXP = 1.5
DELTA_REDSHIFT = 0.5
HELIUM_REDSHIFT = 3.5
HELIUM_DELTA_REDSHIFT = 0.4
HELIUM_REDSHIFTSTART = 5.5
MAX_REDSHIFT = 50.0
_N_BISECT = 40
_N_QUAD = 512


def _window_vars(z_re: float, delta_z: float = DELTA_REDSHIFT):
    """``WindowVarMid``, ``WindowVarDelta`` from ``TTanhReionization_SetParamsForZre``."""
    z_re = jnp.asarray(z_re, dtype=jnp.float64)
    delta_z = jnp.asarray(delta_z, dtype=jnp.float64)
    mid = (1.0 + z_re) ** TANH_ZEXP
    delta = TANH_ZEXP * (1.0 + z_re) ** (TANH_ZEXP - 1.0) * delta_z
    return mid, delta


def second_helium_xe(
    z,
    *,
    f_He: float,
    include_helium_fullreion: bool = True,
    helium_redshift: float = HELIUM_REDSHIFT,
    helium_delta_redshift: float = HELIUM_DELTA_REDSHIFT,
    helium_redshiftstart: float = HELIUM_REDSHIFTSTART,
):
    """Extra electrons from HeII (CAMB ``SecondHelium_xe``)."""
    z = jnp.asarray(z, dtype=jnp.float64)
    if not include_helium_fullreion:
        return jnp.zeros_like(z)
    xod = (helium_redshift - z) / helium_delta_redshift
    tgh = jnp.tanh(jnp.clip(xod, -100.0, 100.0))
    xe_he = f_He * (tgh + 1.0) / 2.0
    return jnp.where(z < helium_redshiftstart, xe_he, 0.0)


def tanh_reionization_xe(
    z,
    *,
    z_re: float,
    f_He: float,
    xe_recomb=0.0,
    delta_z: float = DELTA_REDSHIFT,
    fraction: float | None = None,
    include_helium_fullreion: bool = True,
    helium_redshift: float = HELIUM_REDSHIFT,
    helium_delta_redshift: float = HELIUM_DELTA_REDSHIFT,
    helium_redshiftstart: float = HELIUM_REDSHIFTSTART,
):
    """Full ionization history overlay (CAMB ``TTanhReionization_xe``)."""
    z = jnp.asarray(z, dtype=jnp.float64)
    xstart = jnp.asarray(xe_recomb, dtype=jnp.float64)
    frac = (1.0 + f_He) if fraction is None else float(fraction)
    mid, delta = _window_vars(z_re, delta_z)
    xod = (mid - (1.0 + z) ** TANH_ZEXP) / jnp.maximum(delta, 1e-30)
    tgh = jnp.tanh(jnp.clip(xod, -100.0, 100.0))
    xe = (frac - xstart) * (tgh + 1.0) / 2.0 + xstart
    xe = xe + second_helium_xe(
        z,
        f_He=f_He,
        include_helium_fullreion=include_helium_fullreion,
        helium_redshift=helium_redshift,
        helium_delta_redshift=helium_delta_redshift,
        helium_redshiftstart=helium_redshiftstart,
    )
    return xe


def reionization_optical_depth(
    z_re: float,
    *,
    akthom: float,
    a_bg: jnp.ndarray,
    tau_bg: jnp.ndarray,
    f_He: float,
    delta_z: float = DELTA_REDSHIFT,
    include_helium_fullreion: bool = True,
    helium_redshift: float = HELIUM_REDSHIFT,
    helium_delta_redshift: float = HELIUM_DELTA_REDSHIFT,
    helium_redshiftstart: float = HELIUM_REDSHIFTSTART,
    max_redshift: float = MAX_REDSHIFT,
    n_quad: int = _N_QUAD,
) -> jnp.ndarray:
    """∫_0^{z_start} xe_reion(z) akthom (dτ/da) dz with xe_recomb=0.

    ``a_bg``, ``tau_bg`` are the dimensionless ``build_background_jax`` tables
    (τ in units of 1/H0). Physical ``dτ/da`` [Mpc] = (dτ_bg/da) / H0_mpc, but
    ``akthom`` already carries Mpc^-1, so we need physical dτ/da. Pass
    ``tau_mpc_bg = tau_bg / H0_mpc`` via pre-scaled ``tau_bg`` **or** scale
    outside; here ``tau_bg`` is interpreted as conformal time in Mpc.
    """
    z_re = jnp.asarray(z_re, dtype=jnp.float64)
    z_start = z_re + delta_z * 8.0
    z_start = jnp.minimum(z_start, max_redshift)
    # Uniform-z trapezoid (CAMB uses Romberg; this is plenty for τ~0.05).
    z = jnp.linspace(0.0, z_start, n_quad)
    a = 1.0 / (1.0 + z)
    # dτ/da on background table
    dtauda = jnp.gradient(tau_bg, a_bg)
    dtauda_z = jnp.interp(a, a_bg, dtauda)
    xe = tanh_reionization_xe(
        z,
        z_re=z_re,
        f_He=f_He,
        xe_recomb=0.0,
        delta_z=delta_z,
        include_helium_fullreion=include_helium_fullreion,
        helium_redshift=helium_redshift,
        helium_delta_redshift=helium_delta_redshift,
        helium_redshiftstart=helium_redshiftstart,
    )
    integrand = xe * akthom * dtauda_z
    return jnp.trapezoid(integrand, z)


def z_re_from_tau(
    tau_reio: float,
    *,
    akthom: float,
    a_bg: jnp.ndarray,
    tau_mpc_bg: jnp.ndarray,
    f_He: float,
    delta_z: float = DELTA_REDSHIFT,
    include_helium_fullreion: bool = True,
    helium_redshift: float = HELIUM_REDSHIFT,
    helium_delta_redshift: float = HELIUM_DELTA_REDSHIFT,
    helium_redshiftstart: float = HELIUM_REDSHIFTSTART,
    max_redshift: float = MAX_REDSHIFT,
    n_newton: int = 12,
) -> jnp.ndarray:
    """Newton-solve ``z_re`` so reionization optical depth equals ``tau_reio``.

    Newton (not bisection) so the VJP w.r.t. ``tau_reio`` / ``akthom`` is
    nonzero — required for AD through ``tau`` and ``ombh2``.
    """
    tau_reio = jnp.asarray(tau_reio, dtype=jnp.float64)
    # Seed from the sharp-reionization matter-era scaling.
    z = jnp.asarray(7.5 + 140.0 * tau_reio, dtype=jnp.float64)
    z = jnp.clip(z, helium_redshift + 0.5, max_redshift)

    def residual(z_re):
        return reionization_optical_depth(
            z_re,
            akthom=akthom,
            a_bg=a_bg,
            tau_bg=tau_mpc_bg,
            f_He=f_He,
            delta_z=delta_z,
            include_helium_fullreion=include_helium_fullreion,
            helium_redshift=helium_redshift,
            helium_delta_redshift=helium_delta_redshift,
            helium_redshiftstart=helium_redshiftstart,
            max_redshift=max_redshift,
        ) - tau_reio

    def body(_, z_i):
        r, dr = jax.value_and_grad(residual)(z_i)
        z_n = z_i - r / jnp.where(jnp.abs(dr) > 1e-12, dr, 1e-12)
        return jnp.clip(z_n, helium_redshift + 0.5, max_redshift)

    return jax.lax.fori_loop(0, n_newton, body, z)


__all__ = [
    "DELTA_REDSHIFT",
    "HELIUM_REDSHIFT",
    "reionization_optical_depth",
    "second_helium_xe",
    "tanh_reionization_xe",
    "z_re_from_tau",
]
