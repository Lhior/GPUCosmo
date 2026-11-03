#!/usr/bin/env python3
# This file is part of GPUCosmo. Copyright (C) 2025-2026 Leonel Medina-Varela.
# SPDX-License-Identifier: GPL-3.0-or-later
# Derived in part from the following, translated to JAX and modified for GPUCosmo:
# - CCL (https://github.com/LSSTDESC/CCL), Copyright (c) 2016-2018, the LSSTDESC CCL
#   contributors.
#   Licence: LICENSES/CCL.txt.
# See NOTICE.md.
"""CCL-matched NG FFTLog Hankel transform (spin 0 x 2)."""

from __future__ import annotations

from functools import lru_cache, partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import loggamma

from gpucosmo.jax.splines import (
    ccl_f1d_eval_jax,
    ccl_f1d_eval_numpy,
    ccl_log_spacing_numpy,
    ccl_spline_params,
    log_ell_corr_grid,
    resample_cell_to_fftlog_jax,
)

jax.config.update("jax_enable_x64", True)


class FFTLogTables(NamedTuple):
    """Precomputed CCL FFTLog static tables."""

    ell_grid: jnp.ndarray
    u_coeff: jnp.ndarray
    L: float
    kcrc: float


def _legacy_ng_response() -> bool:
    import os

    return os.environ.get("GPUCOSMO_LEGACY_NG_RESPONSE", "").lower() in (
        "1",
        "true",
        "yes",
    )


def _legacy_fftlog() -> bool:
    import os

    return os.environ.get("GPUCOSMO_LEGACY_FFTLOG", "").lower() in (
        "1",
        "true",
        "yes",
    )


def _use_native_fftlog_hankel() -> bool:
    """JAX FFTLog Hankel (default unless ``GPUCOSMO_LEGACY_FFTLOG=1``)."""
    if _legacy_fftlog():
        return False
    import os

    # One-release alias: explicit native opt-in still forces native on.
    if os.environ.get("GPUCOSMO_NATIVE_FFTLOG", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        return True
    return True


def _use_ccl_fftlog_callback() -> bool:
    """CCL ``correlation(..., method='FFTLog')`` via ``pure_callback``."""
    return not _use_native_fftlog_hankel()


def _lngamma_4(x: float, y: float) -> tuple[float, float]:
    z = loggamma(x + 1j * y)
    return float(z.real), float(z.imag)


def _polar(r: float, phi: float) -> complex:
    return complex(r * np.cos(phi), r * np.sin(phi))


def _goodkr(N: int, mu: float, q: float, L: float, kr: float) -> float:
    xp = (mu + 1.0 + q) / 2.0
    xm = (mu + 1.0 - q) / 2.0
    y = np.pi * N / (2.0 * L)
    _, argp = _lngamma_4(xp, y)
    _, argm = _lngamma_4(xm, y)
    arg = np.log(2.0 / kr) * N / L + (argp + argm) / np.pi
    iarg = round(arg)
    if arg != iarg:
        kr *= np.exp((arg - iarg) * L / N)
    return float(kr)


def _compute_u_coefficients(
    N: int,
    mu: float,
    q: float,
    L: float,
    kcrc: float,
) -> np.ndarray:
    """CCL ``compute_u_coefficients`` for unbiased (q=0) FFTLog."""
    u = np.zeros(N, dtype=np.complex128)
    y = np.pi / L
    k0r0 = kcrc * np.exp(-L)
    t = -2.0 * y * np.log(k0r0 / 2.0)
    x = (mu + 1.0) / 2.0
    for m in range(N // 2 + 1):
        _, phi = _lngamma_4(x, m * y)
        u[m] = _polar(1.0, m * t + 2.0 * phi)
    for m in range(N // 2 + 1, N):
        u[m] = np.conj(u[N - m])
    if N % 2 == 0:
        u[N // 2] = u[N // 2].real + 0j
    return u


@lru_cache(maxsize=4)
def _fftlog_static_tables(
    n_ell: int,
    ell_min: float,
    ell_max: float,
    mu: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Precompute CCL FFTLog grid, u-coefficients, and geometry."""
    ell_grid = ccl_log_spacing_numpy(ell_min, ell_max, n_ell)
    N = int(ell_grid.size)
    q = 0.0
    L = float(np.log(ell_grid[-1] / ell_grid[0]) * N / (N - 1))
    kcrc = _goodkr(N, mu, q, L, 1.0)
    u = _compute_u_coefficients(N, mu, q, L, kcrc)
    return ell_grid, u, L, kcrc


def fht_xi2d_numpy(
    ell_grid: np.ndarray,
    cell_grid: np.ndarray,
    *,
    mu: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """CCL ``ccl_fftlog_ComputeXi2D`` (dim=2, q=0) in NumPy."""
    ell_grid = np.asarray(ell_grid, dtype=float)
    cell_grid = np.asarray(cell_grid, dtype=float)
    N = ell_grid.size
    dim = 2.0
    q = 0.0
    L = np.log(ell_grid[-1] / ell_grid[0]) * N / (N - 1)
    kcrc = _goodkr(N, mu, q, L, 1.0)
    u = _compute_u_coefficients(N, mu, q, L, kcrc)
    prefac_pk = ell_grid ** (dim / 2.0 - q)
    k0r0 = kcrc * np.exp(-L)
    theta = np.empty(N)
    theta[0] = k0r0 / ell_grid[0]
    for n in range(1, N):
        theta[n] = theta[0] * np.exp(n * L / N)
    a = prefac_pk * cell_grid
    # Match FFTW (unnormalized inverse) used in CCL.
    b = np.fft.ifft(np.fft.fft(a) * u / N) * N
    b = b[::-1]
    prefac_xi = (2.0 * np.pi) ** (-dim / 2.0) * theta ** (-dim / 2.0 - q)
    xi = prefac_xi * b.real
    return theta, xi


def _fht_xi2d_jax(
    ell_grid: jnp.ndarray,
    cell_grid: jnp.ndarray,
    u: jnp.ndarray,
    L: float,
    kcrc: float,
    *,
    mu: float = 2.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """JAX port of CCL ``ccl_fftlog_ComputeXi2D`` (dim=2, q=0)."""
    dim = 2.0
    q = 0.0
    N = ell_grid.shape[0]
    prefac_pk = ell_grid ** (dim / 2.0 - q)
    k0r0 = kcrc * jnp.exp(-L)
    n = jnp.arange(N, dtype=float)
    theta0 = k0r0 / ell_grid[0]
    theta = theta0 * jnp.exp(n * L / N)
    a = prefac_pk * cell_grid
    b = jnp.fft.ifft(jnp.fft.fft(a) * u / N) * N
    b = b[::-1]
    prefac_xi = (2.0 * jnp.pi) ** (-dim / 2.0) * theta ** (-dim / 2.0 - q)
    xi = prefac_xi * jnp.real(b)
    return theta, xi


def tracer_corr_fftlog_ng_numpy(
    ell: np.ndarray,
    cell: np.ndarray,
    theta_deg: np.ndarray,
    ell_grid: np.ndarray,
    *,
    mu: float = 2.0,
) -> np.ndarray:
    """Full CCL ``ccl_tracer_corr_fftlog`` NG pipeline in NumPy."""
    cell = np.asarray(cell, dtype=float)
    if np.all(cell == 0.0):
        return np.zeros_like(theta_deg, dtype=float)
    cell_grid = ccl_f1d_eval_numpy(
        np.asarray(ell, dtype=float),
        cell,
        np.asarray(ell_grid, dtype=float),
        extrap_lo="const",
        extrap_hi="logx_logy",
        y0=float(cell[0]),
        yf=0.0,
    )
    th_arr, xi_arr = fht_xi2d_numpy(ell_grid, cell_grid, mu=mu)
    return ccl_f1d_eval_numpy(
        th_arr,
        xi_arr,
        np.deg2rad(np.asarray(theta_deg, dtype=float)),
        extrap_lo="const",
        extrap_hi="const",
        y0=float(xi_arr[0]),
        yf=0.0,
    )


def tracer_corr_fftlog_ng_jax(
    cell: jnp.ndarray,
    ell: jnp.ndarray,
    theta_deg: jnp.ndarray,
    ell_grid: jnp.ndarray,
    u_coeff: jnp.ndarray,
    L: float,
    kcrc: float,
) -> jnp.ndarray:
    """Full CCL ``ccl_tracer_corr_fftlog`` NG pipeline in JAX."""
    cell_grid = resample_cell_to_fftlog_jax(ell, cell, ell_grid)
    th_arr, xi_arr = _fht_xi2d_jax(ell_grid, cell_grid, u_coeff, L, kcrc)
    theta_rad = jnp.deg2rad(theta_deg)
    return ccl_f1d_eval_jax(
        th_arr,
        xi_arr,
        theta_rad,
        extrap_lo="const",
        extrap_hi="const",
        y0=xi_arr[0],
        yf=0.0,
    )


def _default_fftlog_tables() -> FFTLogTables:
    params = ccl_spline_params()
    grid, u, l_val, k_val = _fftlog_static_tables(
        int(params["N_ELL_CORR"]),
        float(params["ELL_MIN_CORR"]),
        float(params["ELL_MAX_CORR"]),
        mu=2.0,
    )
    return FFTLogTables(
        ell_grid=jnp.asarray(grid, dtype=float),
        u_coeff=jnp.asarray(u, dtype=jnp.complex128),
        L=float(l_val),
        kcrc=float(k_val),
    )


_DEFAULT_TABLES = _default_fftlog_tables()


def gamma_t_ng_fftlog_numpy(
    ell: np.ndarray,
    cell: np.ndarray,
    theta_deg: np.ndarray,
) -> np.ndarray:
    """Reference NG transform using the native CCL-matched FFTLog pipeline."""
    params = ccl_spline_params()
    ell_grid, _, _, _ = _fftlog_static_tables(
        int(params["N_ELL_CORR"]),
        float(params["ELL_MIN_CORR"]),
        float(params["ELL_MAX_CORR"]),
        mu=2.0,
    )
    return tracer_corr_fftlog_ng_numpy(ell, cell, theta_deg, ell_grid)


@lru_cache(maxsize=4)
def _nn_fftlog_tables() -> FFTLogTables:
    """FFTLog static tables for NN (μ=0) galaxy clustering w(θ)."""
    params = ccl_spline_params()
    grid, u, l_val, k_val = _fftlog_static_tables(
        int(params["N_ELL_CORR"]),
        float(params["ELL_MIN_CORR"]),
        float(params["ELL_MAX_CORR"]),
        mu=0.0,
    )
    return FFTLogTables(
        ell_grid=jnp.asarray(grid, dtype=float),
        u_coeff=jnp.asarray(u, dtype=jnp.complex128),
        L=float(l_val),
        kcrc=float(k_val),
    )


def tracer_corr_fftlog_nn_numpy(
    ell: np.ndarray,
    cell: np.ndarray,
    theta_deg: np.ndarray,
    ell_grid: np.ndarray | None = None,
) -> np.ndarray:
    """CCL ``type='NN'`` FFTLog (μ=0) for w(θ)."""
    if ell_grid is None:
        params = ccl_spline_params()
        ell_grid, _, _, _ = _fftlog_static_tables(
            int(params["N_ELL_CORR"]),
            float(params["ELL_MIN_CORR"]),
            float(params["ELL_MAX_CORR"]),
            mu=0.0,
        )
    return tracer_corr_fftlog_ng_numpy(
        ell, cell, theta_deg, ell_grid, mu=0.0
    )


def w_theta_nn_fftlog_numpy(
    ell: np.ndarray,
    cell: np.ndarray,
    theta_deg: np.ndarray,
) -> np.ndarray:
    """w(θ) from C_ℓ^{gg} via NN FFTLog."""
    return tracer_corr_fftlog_nn_numpy(ell, cell, theta_deg)


def _fht_xi2d_nn_jax(
    ell_grid: jnp.ndarray,
    cell_grid: jnp.ndarray,
    u: jnp.ndarray,
    L: float,
    kcrc: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """JAX FFTLog Xi2D with μ=0 (NN); u-coefficients must be precomputed at μ=0."""
    return _fht_xi2d_jax(ell_grid, cell_grid, u, L, kcrc, mu=0.0)


def tracer_corr_fftlog_nn_jax(
    cell: jnp.ndarray,
    ell: jnp.ndarray,
    theta_deg: jnp.ndarray,
    ell_grid: jnp.ndarray,
    u_coeff: jnp.ndarray,
    L: float,
    kcrc: float,
) -> jnp.ndarray:
    """Full CCL NN FFTLog pipeline in JAX (μ=0 u-coefficients)."""
    cell_grid = resample_cell_to_fftlog_jax(ell, cell, ell_grid)
    th_arr, xi_arr = _fht_xi2d_nn_jax(ell_grid, cell_grid, u_coeff, L, kcrc)
    theta_rad = jnp.deg2rad(theta_deg)
    return ccl_f1d_eval_jax(
        th_arr,
        xi_arr,
        theta_rad,
        extrap_lo="const",
        extrap_hi="const",
        y0=xi_arr[0],
        yf=0.0,
    )


def w_theta_nn_fftlog_jax(
    ell: jnp.ndarray,
    cell: jnp.ndarray,
    theta_deg: jnp.ndarray,
) -> jnp.ndarray:
    """JAX NN (μ=0) Hankel transform for w(θ)."""
    tables = _nn_fftlog_tables()
    return tracer_corr_fftlog_nn_jax(
        cell,
        ell,
        theta_deg,
        tables.ell_grid,
        tables.u_coeff,
        tables.L,
        tables.kcrc,
    )


class NNFFTLogTransform:
    """Pre-bound ell/theta grids for NN (wθ) FFTLog transforms."""

    def __init__(self, ell: np.ndarray, theta_deg: np.ndarray):
        self.ell = np.asarray(ell, dtype=float)
        self.theta_deg = np.asarray(theta_deg, dtype=float)
        tables = _nn_fftlog_tables()
        self._ell_grid = tables.ell_grid
        self._u = tables.u_coeff
        self._L = tables.L
        self._kcrc = tables.kcrc

    def w_numpy(self, cell: np.ndarray) -> np.ndarray:
        return tracer_corr_fftlog_nn_numpy(
            self.ell, cell, self.theta_deg, np.asarray(self._ell_grid)
        )

    def w_jax(self, cell) -> jnp.ndarray:
        return tracer_corr_fftlog_nn_jax(
            jnp.asarray(cell, dtype=float),
            jnp.asarray(self.ell, dtype=float),
            jnp.asarray(self.theta_deg, dtype=float),
            self._ell_grid,
            self._u,
            self._L,
            self._kcrc,
        )



def gamma_t_ng_fftlog_ccl_numpy(
    ell: np.ndarray,
    cell: np.ndarray,
    theta_deg: np.ndarray,
) -> np.ndarray:
    """Legacy reference via ``ccl.correlation`` (callback backend)."""
    import pyccl as ccl

    ell = np.asarray(ell, dtype=float)
    cell = np.asarray(cell, dtype=float)
    theta_deg = np.asarray(theta_deg, dtype=float)
    if np.all(cell == 0.0):
        return np.zeros_like(theta_deg, dtype=float)
    cosmo = ccl.Cosmology(
        Omega_c=0.253,
        Omega_b=0.047,
        h=0.7,
        sigma8=0.82,
        n_s=0.96,
        transfer_function="boltzmann_camb",
        matter_power_spectrum="halofit",
    )
    return np.asarray(
        ccl.correlation(
            cosmo,
            ell=ell,
            C_ell=cell,
            theta=theta_deg,
            type="NG",
            method="FFTLog",
        ),
        dtype=float,
    )


class NGFFTLogTransform:
    """Pre-bound ell/theta grids for repeated CCL FFTLog NG transforms."""

    def __init__(self, ell: np.ndarray, theta_deg: np.ndarray):
        self.ell = np.asarray(ell, dtype=float)
        self.theta_deg = np.asarray(theta_deg, dtype=float)
        self._n_theta = int(self.theta_deg.size)
        tables = _default_fftlog_tables()
        self._ell_grid = tables.ell_grid
        self._u_coeff = tables.u_coeff
        self._L = tables.L
        self._kcrc = tables.kcrc

    def gamma_t_numpy(self, cell: np.ndarray) -> np.ndarray:
        if _use_ccl_fftlog_callback():
            return gamma_t_ng_fftlog_ccl_numpy(self.ell, cell, self.theta_deg)
        return gamma_t_ng_fftlog_numpy(self.ell, cell, self.theta_deg)

    def gamma_t_jax(self, cell) -> jnp.ndarray:
        return gamma_t_ng_fftlog_jax(
            jnp.asarray(cell, dtype=float),
            jnp.asarray(self.ell, dtype=float),
            jnp.asarray(self.theta_deg, dtype=float),
            self._ell_grid,
            self._u_coeff,
            self._L,
            self._kcrc,
            n_theta=self._n_theta,
        )


@partial(jax.jit, static_argnames=("n_theta",))
def _gamma_t_ng_fftlog_native_jit(
    cell: jnp.ndarray,
    ell: jnp.ndarray,
    theta_deg: jnp.ndarray,
    ell_grid: jnp.ndarray,
    u_coeff: jnp.ndarray,
    L: float,
    kcrc: float,
    *,
    n_theta: int,
) -> jnp.ndarray:
    del n_theta
    return tracer_corr_fftlog_ng_jax(
        cell,
        ell,
        theta_deg,
        ell_grid,
        u_coeff,
        L,
        kcrc,
    )


def gamma_t_ng_fftlog_vmap(
    cell_batch: jnp.ndarray,
    ell: jnp.ndarray,
    theta_deg: jnp.ndarray,
    ell_grid: jnp.ndarray | None = None,
    u_coeff: jnp.ndarray | None = None,
    L: float | None = None,
    kcrc: float | None = None,
) -> jnp.ndarray:
    """Batched NG FFTLog over leading dimension of ``cell_batch``."""
    if ell_grid is None:
        tables = _DEFAULT_TABLES
        ell_grid = tables.ell_grid
        u_coeff = tables.u_coeff
        L = tables.L
        kcrc = tables.kcrc
    n_theta = int(theta_deg.shape[0])

    def _one(cell_row: jnp.ndarray) -> jnp.ndarray:
        return _gamma_t_ng_fftlog_native_jit(
            cell_row,
            ell,
            theta_deg,
            ell_grid,
            u_coeff,
            L,
            kcrc,
            n_theta=n_theta,
        )

    return jax.vmap(_one)(cell_batch)


def gamma_t_ng_fftlog_jax(
    cell: jnp.ndarray,
    ell: jnp.ndarray,
    theta_deg: jnp.ndarray,
    ell_grid: jnp.ndarray | None = None,
    u_coeff: jnp.ndarray | None = None,
    L: float | None = None,
    kcrc: float | None = None,
    *,
    n_theta: int,
) -> jnp.ndarray:
    """JAX NG Hankel transform (native default; legacy callback opt-in)."""
    if _use_ccl_fftlog_callback():
        out_shape = jax.ShapeDtypeStruct((n_theta,), jnp.float64)

        def _callback(cell_np, ell_np, theta_np):
            return gamma_t_ng_fftlog_ccl_numpy(ell_np, cell_np, theta_np)

        return jax.pure_callback(
            _callback,
            out_shape,
            cell,
            ell,
            theta_deg,
            vmap_method="sequential",
        )
    if ell_grid is None:
        tables = _DEFAULT_TABLES
        ell_grid = tables.ell_grid
        u_coeff = tables.u_coeff
        L = tables.L
        kcrc = tables.kcrc
    return _gamma_t_ng_fftlog_native_jit(
        cell,
        ell,
        theta_deg,
        ell_grid,
        u_coeff,
        L,
        kcrc,
        n_theta=n_theta,
    )


# Backward-compatible alias
gamma_t_ng_fftlog_native_jax = tracer_corr_fftlog_ng_jax


__all__ = [
    "FFTLogTables",
    "NGFFTLogTransform",
    "_legacy_fftlog",
    "_legacy_ng_response",
    "_use_ccl_fftlog_callback",
    "_use_native_fftlog_hankel",
    "fht_xi2d_numpy",
    "gamma_t_ng_fftlog_ccl_numpy",
    "gamma_t_ng_fftlog_jax",
    "gamma_t_ng_fftlog_native_jax",
    "gamma_t_ng_fftlog_numpy",
    "gamma_t_ng_fftlog_vmap",
    "tracer_corr_fftlog_ng_jax",
    "tracer_corr_fftlog_ng_numpy",
]
