#!/usr/bin/env python3
# This file is part of GPUCosmo. Copyright (C) 2025-2026 Leonel Medina-Varela.
# SPDX-License-Identifier: GPL-3.0-or-later
# Derived in part from the following, translated to JAX and modified for GPUCosmo:
# - CCL (https://github.com/LSSTDESC/CCL), Copyright (c) 2016-2018, the LSSTDESC CCL
#   contributors.
#   Licence: LICENSES/CCL.txt.
# See NOTICE.md.
"""Limber ``angular_cl`` in JAX (``ccl_angular_cls_limber`` port).

The integrand, k-limits and Bessel-derivative terms are copied from
``ccl_cls.c``. CCL integrates in ln k with adaptive GSL QAG (GK41,
epsrel 1e-4); here a fixed composite Gauss-Legendre rule on the same
per-ell interval replaces it, which is jittable, differentiable and
converged well below CCL's own tolerance with the defaults below.

The rule is structure-aware so it stays accurate for narrow or truncated
n(z): breakpoints are placed at k = (l+1/2)/chi_i for the kernel spline nodes
chi_i (subsampled to at most ``LIMBER_MAX_KERNEL_BREAKS`` per kernel; plus
(l+3/2)/chi_i for RSD-type tracers), which also captures the jumps where a
kernel ends, merged with ``LIMBER_N_UNIFORM`` uniform panels for P(k)
features, with ``LIMBER_N_GL`` Gauss-Legendre points per interval.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from gpucosmo.jax.ccl.cosmology import DEFAULT_POWER_SPECTRUM, K_MAX, K_MIN, Cosmology
from gpucosmo.jax.ccl.pk2d import Pk2D
from gpucosmo.jax.ccl.tracers import ClTracer, Tracer

jax.config.update("jax_enable_x64", True)

LIMBER_N_UNIFORM = 32
LIMBER_N_GL = 4
LIMBER_MAX_KERNEL_BREAKS = 96


def _gl_unit(n_gl: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes / weights on [0, 1]."""
    x, w = np.polynomial.legendre.leggauss(n_gl)
    return 0.5 * (x + 1.0), 0.5 * w


def _k_interval(ell, trc1: Tracer, trc2: Tracer):
    """``get_k_interval``: overlap of both collections' chi support."""
    chi_min = jnp.maximum(trc1.chi_min, trc2.chi_min)
    chi_max = jnp.minimum(trc1.chi_max, trc2.chi_max)
    lp1h = ell + 0.5
    chi_min = jnp.where(chi_min <= 0.0, 0.5 * lp1h / K_MAX, chi_min)
    lkmax = jnp.log(jnp.minimum(K_MAX, 2.0 * lp1h / chi_min))
    lkmin = jnp.log(jnp.maximum(K_MIN, lp1h / chi_max))
    return lkmin, lkmax


def _transfer_limber_single(
    tr: ClTracer, ell, lk, k, chi_l, a_l, cosmo: Cosmology, psp: Pk2D
):
    w = tr.get_kernel(chi_l)
    t = tr.get_transfer(lk, a_l)
    lp1h = ell + 0.5
    if tr.der_bessel < 1:
        dd = w * t
        if tr.der_bessel == -1:
            dd = dd / (lp1h * lp1h)
    else:
        lp3h = ell + 1.5
        chi_lp = lp3h / k
        a_lp = cosmo.scale_factor_of_chi(chi_lp)
        pk_ratio = jnp.abs(psp.eval_lk(lk, a_lp, cosmo) / psp.eval_lk(lk, a_l, cosmo))
        w_p = tr.get_kernel(chi_lp)
        t_p = tr.get_transfer(lk, a_lp)
        sqell = jnp.sqrt(lp1h * pk_ratio / lp3h)
        if tr.der_bessel == 1:
            dd = ell * w * t / lp1h - sqell * w_p * t_p
        else:
            dd = sqell * 2.0 * w_p * t_p / lp3h - (0.25 + 2.0 * ell) * w * t / (lp1h * lp1h)
    return dd * tr.get_f_ell(ell)


def _transfer_limber_wrap(trc: Tracer, ell, lk, k, chi, a, cosmo, psp):
    out = 0.0
    for tr in trc._trc:
        out = out + _transfer_limber_single(tr, ell, lk, k, chi, a, cosmo, psp)
    return out


def _kernel_breaks(trc: Tracer, max_breaks: int) -> list[tuple[jnp.ndarray, float]]:
    """(chi nodes, ell offset) whose k = (l+offset)/chi become breakpoints."""
    out = []
    for tr in trc._trc:
        if tr.kernel is None:
            continue
        chi = tr.kernel.x
        n = chi.shape[0]
        if n > max_breaks:
            idx = np.unique(np.round(np.linspace(0, n - 1, max_breaks)).astype(int))
            chi = chi[idx]
        out.append((chi, 0.5))
        if tr.der_bessel >= 1:
            out.append((chi, 1.5))
    return out


def _cl_limber_one(ell, cosmo, trc1, trc2, psp, nodes, weights, breaks, n_uniform):
    lkmin, lkmax = _k_interval(ell, trc1, trc2)
    pts = [jnp.linspace(lkmin, lkmax, n_uniform + 1)]
    for chi, off in breaks:
        pts.append(jnp.log((ell + off) / jnp.maximum(chi, 1.0e-30)))
    bounds = jnp.sort(jnp.clip(jnp.concatenate(pts), lkmin, lkmax))
    lo, hi = bounds[:-1], bounds[1:]
    span = hi - lo
    lk = (lo[:, None] + span[:, None] * nodes[None, :]).ravel()
    w = (span[:, None] * weights[None, :]).ravel()
    k = jnp.exp(lk)
    chi = (ell + 0.5) / k
    a = cosmo.scale_factor_of_chi(chi)
    d1 = _transfer_limber_wrap(trc1, ell, lk, k, chi, a, cosmo, psp)
    d2 = _transfer_limber_wrap(trc2, ell, lk, k, chi, a, cosmo, psp)
    pk = psp.eval_lk(lk, a, cosmo)
    return jnp.sum(w * k * pk * d1 * d2) / (ell + 0.5)


def angular_cl(
    cosmo: Cosmology,
    tracer1: Tracer,
    tracer2: Tracer,
    ell,
    *,
    p_of_k_a=DEFAULT_POWER_SPECTRUM,
    l_limber=-1,
    limber_integration_method: str = "qag_quad",
    limber_n_uniform: int | None = None,
    limber_n_gl: int | None = None,
    limber_max_kernel_breaks: int | None = None,
    return_meta: bool = False,
    **_unused,
):
    """Limber C_ell between two tracers (``pyccl.angular_cl`` signature).

    ``p_of_k_a`` accepts a JAX :class:`Pk2D`, a ``pyccl.Pk2D`` (converted, not
    differentiable), ``'delta_matter:delta_matter'`` / ``None`` (non-linear)
    or ``'linear'``. Only the Limber approximation is implemented.
    """
    if limber_integration_method not in ("qag_quad", "spline"):
        raise ValueError(
            f"Limber integration method {limber_integration_method} not supported"
        )
    if isinstance(l_limber, str) or (
        l_limber > 0 and float(np.min(np.asarray(ell))) < l_limber
    ):
        raise NotImplementedError("Only the Limber approximation is implemented")
    psp = cosmo.parse_pk2d(p_of_k_a, is_linear=False)
    nodes, weights = _gl_unit(int(limber_n_gl or LIMBER_N_GL))
    nodes = jnp.asarray(nodes)
    weights = jnp.asarray(weights)
    n_uniform = int(limber_n_uniform or LIMBER_N_UNIFORM)
    max_breaks = int(limber_max_kernel_breaks or LIMBER_MAX_KERNEL_BREAKS)
    breaks = _kernel_breaks(tracer1, max_breaks) + _kernel_breaks(tracer2, max_breaks)
    ell_arr = jnp.atleast_1d(jnp.asarray(ell, dtype=float))
    cl = jax.vmap(
        lambda l: _cl_limber_one(
            l, cosmo, tracer1, tracer2, psp, nodes, weights, breaks, n_uniform
        )
    )(ell_arr)
    if jnp.ndim(ell) == 0:
        cl = cl[0]
    if return_meta:
        return cl, {"l_limber": l_limber}
    return cl


__all__ = ["LIMBER_MAX_KERNEL_BREAKS", "LIMBER_N_GL", "LIMBER_N_UNIFORM", "angular_cl"]
