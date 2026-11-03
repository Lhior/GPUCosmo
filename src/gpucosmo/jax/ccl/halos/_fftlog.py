#!/usr/bin/env python3
# This file is part of GPUCosmo. Copyright (C) 2025-2026 Leonel Medina-Varela.
# SPDX-License-Identifier: GPL-3.0-or-later
# Derived in part from the following, translated to JAX and modified for GPUCosmo:
# - CCL (https://github.com/LSSTDESC/CCL), Copyright (c) 2016-2018, the LSSTDESC CCL
#   contributors.
#   Licence: LICENSES/CCL.txt.
# See NOTICE.md.
"""JAX port of CCL's ``fftlog_transform`` (``ccl_fftlog.c:fht``) + resampling.

``pyccl.pyutils._fftlog_transform(rs, frs, dim, mu, plaw)`` calls
``ccl_fftlog_ComputeXi{2,3}D`` which run Hamilton's FFTLog with
``q = dim/2 + plaw``, ``kcrc = 1`` and the low-ringing ``goodkr`` shift. The
input grid (and hence every coefficient) is static, so all the complex
gamma-function tables are precomputed in NumPy; only the FFT convolution is
traced. ``pyccl.pyutils.resample_array`` (GSL Akima + CCL linear extrapolation)
is reproduced by :func:`resample_akima_linlin`.
"""

from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import loggamma

from gpucosmo.jax.gsl_splines import akima_spline, spline_eval

jax.config.update("jax_enable_x64", True)


class FHTPlan(NamedTuple):
    prefac_in: np.ndarray  # r^(dim/2 - q)
    prefac_out: np.ndarray  # (2 pi)^(-dim/2) k^(-dim/2 - q)
    u: np.ndarray  # complex FFTLog kernel
    k_out: np.ndarray


def _lngamma(x: float, y: float) -> complex:
    return complex(loggamma(complex(x, y)))


def _goodkr(n: int, mu: float, q: float, L: float, kr: float) -> float:
    xp = (mu + 1.0 + q) / 2.0
    xm = (mu + 1.0 - q) / 2.0
    y = np.pi * n / (2.0 * L)
    argp = _lngamma(xp, y).imag
    argm = _lngamma(xm, y).imag
    arg = np.log(2.0 / kr) * n / L + (argp + argm) / np.pi
    iarg = np.round(arg)
    if arg != iarg:
        kr *= np.exp((arg - iarg) * L / n)
    return float(kr)


def _u_coefficients(n: int, mu: float, q: float, L: float, kcrc: float) -> np.ndarray:
    y = np.pi / L
    k0r0 = kcrc * np.exp(-L)
    t = -2.0 * y * np.log(k0r0 / 2.0)
    u = np.zeros(n, dtype=np.complex128)
    m = np.arange(n // 2 + 1)
    if q == 0:
        x = (mu + 1.0) / 2.0
        phi = np.array([_lngamma(x, mm * y).imag for mm in m])
        u[: n // 2 + 1] = np.exp(1j * (m * t + 2.0 * phi))
    else:
        xp = (mu + 1.0 + q) / 2.0
        xm = (mu + 1.0 - q) / 2.0
        gp = np.array([_lngamma(xp, mm * y) for mm in m])
        gm = np.array([_lngamma(xm, -mm * y) for mm in m])
        amp = np.exp(q * np.log(2.0) + gp.real - gm.real)
        u[: n // 2 + 1] = amp * np.exp(1j * (m * t + gp.imag - gm.imag))
    for mm in range(n // 2 + 1, n):
        u[mm] = np.conj(u[n - mm])
    if n % 2 == 0:
        u[n // 2] = u[n // 2].real + 0j
    return u


@lru_cache(maxsize=64)
def _plan_cached(r_key: tuple, dim: float, mu: float, q: float) -> FHTPlan:
    r0, r1, n = r_key
    r = np.geomspace(r0, r1, n)
    L = np.log(r[-1] / r[0]) * n / (n - 1.0)
    kcrc = _goodkr(n, mu, q, L, 1.0)
    u = _u_coefficients(n, mu, q, L, kcrc)
    k0r0 = kcrc * np.exp(-L)
    k_out = np.empty(n)
    k_out[0] = k0r0 / r[0]
    k_out[1:] = k_out[0] * np.exp(np.arange(1, n) * L / n)
    prefac_in = r ** (dim / 2.0 - q)
    prefac_out = (2.0 * np.pi) ** (-dim / 2.0) * k_out ** (-dim / 2.0 - q)
    return FHTPlan(prefac_in, prefac_out, u, k_out)


def fht_plan(r_min: float, r_max: float, n: int, dim: int, mu: float, q: float) -> FHTPlan:
    """Static plan for a ``np.geomspace(r_min, r_max, n)`` input grid."""
    return _plan_cached((float(r_min), float(r_max), int(n)), float(dim), float(mu), float(q))


def fht_apply(plans, f):
    """Run several FFTLog transforms (sharing the input grid) on rows of ``f``.

    ``plans`` is a sequence of :class:`FHTPlan` with identical ``prefac_in``;
    the forward FFT is shared. Returns a list of ``(..., n)`` outputs.
    """
    a = jnp.asarray(plans[0].prefac_in) * f
    fa = jnp.fft.fft(a, axis=-1)
    outs = []
    for p in plans:
        b = jnp.fft.ifft(fa * jnp.asarray(p.u), axis=-1)
        outs.append(jnp.asarray(p.prefac_out) * jnp.real(b[..., ::-1]))
    return outs


def fftlog_transform(rs, frs, dim: int, mu: float, power_law_index: float):
    """``pyccl.pyutils._fftlog_transform`` (``rs`` must be static geomspace)."""
    rs = np.asarray(rs, dtype=float)
    q = 0.5 * dim + power_law_index
    mu_b = mu + 0.5 if dim == 3 else mu
    plan = fht_plan(rs[0], rs[-1], rs.size, dim, mu_b, q)
    (out,) = fht_apply([plan], jnp.asarray(frs, dtype=float))
    return plan.k_out, out


def resample_akima_linlin(x_in, y_in, x_out):
    """``resample_array(x_in, y_in, x_out, 'linx_liny', 'linx_liny')`` per row.

    ``x_in`` / ``x_out`` are 1D; ``y_in`` is ``(..., n_in)``.
    """
    x_in = jnp.asarray(x_in, dtype=float)
    x_out = jnp.asarray(x_out, dtype=float)

    def one(y):
        s = akima_spline(x_in, y)
        inner = spline_eval(s, jnp.clip(x_out, x_in[0], x_in[-1]))
        der_lo = (y[1] - y[0]) / (x_in[1] - x_in[0])
        der_hi = (y[-1] - y[-2]) / (x_in[-1] - x_in[-2])
        lo = y[0] + der_lo * (x_out - x_in[0])
        hi = y[-1] + der_hi * (x_out - x_in[-1])
        return jnp.where(x_out <= x_in[0], lo, jnp.where(x_out >= x_in[-1], hi, inner))

    y2 = jnp.reshape(y_in, (-1, y_in.shape[-1]))
    out = jax.vmap(one)(y2)
    return jnp.reshape(out, y_in.shape[:-1] + (x_out.shape[0],))


__all__ = ["FHTPlan", "fht_plan", "fht_apply", "fftlog_transform", "resample_akima_linlin"]
