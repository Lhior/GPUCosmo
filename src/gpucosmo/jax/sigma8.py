#!/usr/bin/env python3
"""sigma8 amplitude rescaling shared by Boltzmann backends."""

from __future__ import annotations

import numpy as np

# Planck 2018 best-fit amplitude used as the reference transfer-function solve.
LN10AS_REF = 3.0445224377

# Standard sigma8 top-hat radius: 8 Mpc/h = 8/h Mpc with h = H0/(100 km/s/Mpc).
_SIGMA8_RADIUS_MPC_H = 8.0


def sigma8_radius_mpc(h: float) -> float:
    """Top-hat radius in Mpc matching the CCL/CAMB sigma8 convention."""
    return _SIGMA8_RADIUS_MPC_H / float(h)


def _tophat_window(kr: np.ndarray) -> np.ndarray:
    """CCL top-hat Fourier transform W(kR) = 3(sin x - x cos x)/x^3."""
    x = np.asarray(kr, dtype=float)
    small = np.abs(x) < 1.0e-8
    w = np.empty_like(x)
    xs = x[~small]
    w[~small] = 3.0 * (np.sin(xs) - xs * np.cos(xs)) / xs**3
    w[small] = 1.0
    return w


def _sigma_r_tophat(
    log_k: np.ndarray,
    pk: np.ndarray,
    *,
    radius_mpc: float,
) -> float:
    """RMS fluctuation in a top-hat sphere of radius R [Mpc] (CCL convention)."""
    lnk = np.asarray(log_k, dtype=float)
    pk_arr = np.asarray(pk, dtype=float)
    k = np.exp(lnk)
    kr = k * float(radius_mpc)
    window = _tophat_window(kr)
    integrand = pk_arr * k**3 / (2.0 * np.pi**2) * window**2
    dlnk = np.diff(lnk)
    trapezoid = 0.5 * (integrand[1:] + integrand[:-1]) * dlnk
    return float(np.sqrt(np.sum(trapezoid)))


def sigma8_from_pk_z0(
    log_k: np.ndarray,
    log_pk_z0: np.ndarray,
    *,
    h: float,
) -> float:
    """Compute sigma8 from a linear P(k, z=0) table (k in 1/Mpc, P in Mpc^3)."""
    pk = np.exp(np.asarray(log_pk_z0, dtype=float))
    return _sigma_r_tophat(
        np.asarray(log_k, dtype=float),
        pk,
        radius_mpc=sigma8_radius_mpc(h),
    )


def ln10A_s_for_sigma8(
    sigma8_ref: float,
    sigma8_target: float,
    *,
    ln10A_s_ref: float = LN10AS_REF,
) -> float:
    """Analytic amplitude map: sigma8 ∝ sqrt(A_s) ∝ exp(ln10A_s / 2)."""
    if sigma8_ref <= 0.0 or sigma8_target <= 0.0:
        raise ValueError("sigma8_ref and sigma8_target must be positive.")
    return float(ln10A_s_ref + 2.0 * np.log(sigma8_target / sigma8_ref))


def amplitude_scale_for_sigma8(sigma8_ref: float, sigma8_target: float) -> float:
    """Scale linear P(k, z) by (sigma8_target / sigma8_ref)^2."""
    if sigma8_ref <= 0.0 or sigma8_target <= 0.0:
        raise ValueError("sigma8_ref and sigma8_target must be positive.")
    return float((sigma8_target / sigma8_ref) ** 2)


def _tophat_window_jax(kr):
    """JAX top-hat Fourier transform W(kR) = 3(sin x - x cos x)/x^3."""
    import jax.numpy as jnp

    x = jnp.asarray(kr, dtype=float)
    safe = jnp.where(jnp.abs(x) < 1.0e-8, 1.0, x)
    w = 3.0 * (jnp.sin(safe) - safe * jnp.cos(safe)) / safe**3
    return jnp.where(jnp.abs(x) < 1.0e-8, 1.0, w)


def sigma8_from_pk_z0_jax(log_k, log_pk_z0, *, h):
    """JAX sigma8 from linear P(k, z=0) (k in 1/Mpc, P in Mpc^3)."""
    import jax.numpy as jnp

    pk = jnp.exp(jnp.asarray(log_pk_z0, dtype=float))
    k = jnp.exp(jnp.asarray(log_k, dtype=float))
    kr = k * float(sigma8_radius_mpc(h))
    window = _tophat_window_jax(kr)
    integrand = pk * k**3 / (2.0 * jnp.pi**2) * window**2
    dlnk = jnp.diff(jnp.asarray(log_k, dtype=float))
    trapezoid = 0.5 * (integrand[1:] + integrand[:-1]) * dlnk
    return jnp.sqrt(jnp.sum(trapezoid))


__all__ = [
    "LN10AS_REF",
    "amplitude_scale_for_sigma8",
    "ln10A_s_for_sigma8",
    "sigma8_from_pk_z0",
    "sigma8_from_pk_z0_jax",
    "sigma8_radius_mpc",
]
