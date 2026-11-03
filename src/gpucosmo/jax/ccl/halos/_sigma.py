#!/usr/bin/env python3
# This file is part of GPUCosmo. Copyright (C) 2025-2026 Leonel Medina-Varela.
# SPDX-License-Identifier: GPL-3.0-or-later
# Derived in part from the following, translated to JAX and modified for GPUCosmo:
# - CCL (https://github.com/LSSTDESC/CCL), Copyright (c) 2016-2018, the LSSTDESC CCL
#   contributors.
#   Licence: LICENSES/CCL.txt.
# See NOTICE.md.
"""sigma(M), halo-model background helpers and CCL grids.

CCL (``ccl_massfunc.c:ccl_cosmology_compute_sigma``) tabulates
``ln sigma(R(M), a)`` on ``LOGM_SPLINE_NM = 50`` linear log10 M nodes in
[6, 17] and 18 lin-log scale-factor nodes (``A_SPLINE_{MINLOG,MIN}_SM``,
``A_SPLINE_{NLOG,NA}_SM``), then interpolates with a GSL bicubic 2D spline;
``dlnsigM_dlogM`` is minus its ``x`` derivative. Each node is

    sigma^2(R, a) = ln10 / (2 pi^2) \\int dlog10 k  P_lin(k, a) k^3 W^2(kR)

over ``[log10 K_MIN, log10 K_MAX]`` (CCL: GSL cquad, epsrel 1e-5). Here the
integral uses a fixed composite Gauss-Legendre rule (converged to ~1e-9),
which is differentiable w.r.t. the linear power-spectrum table.

The sigma spline is a pure function of the cosmology. It is cached on
concrete (non-traced) cosmologies, and the halo-model drivers compute it once
per call and publish it through :func:`sigma_context` so that every
ingredient (mass function, bias, concentration) inside a traced loop reuses it.
"""

from __future__ import annotations

import contextlib
import contextvars

import jax
import jax.numpy as jnp
import numpy as np

from gpucosmo.jax.ccl.cosmology import RHO_CRITICAL
from gpucosmo.jax.gsl_splines import Bicubic2D, bicubic_deriv_x, bicubic_eval, bicubic_spline

jax.config.update("jax_enable_x64", True)

# pyccl.spline_params / gsl_params defaults (CCL 3.x)
LOGM_SPLINE_MIN = 6.0
LOGM_SPLINE_MAX = 17.0
LOGM_SPLINE_NM = 50
A_SPLINE_MINLOG_SM = 0.01
A_SPLINE_MIN_SM = 0.1
A_SPLINE_MAX = 1.0
A_SPLINE_NLOG_SM = 6
A_SPLINE_NA_SM = 13
A_SPLINE_MINLOG_PK = 0.01
A_SPLINE_MIN_PK = 0.1
A_SPLINE_NLOG_PK = 11
A_SPLINE_NA_PK = 40
K_MIN = 5.0e-5
K_MAX = 1000.0
N_K = 167

# sigma(R) quadrature: composite Gauss-Legendre in log10 k.
SIGMA_N_PANELS = 256
SIGMA_N_GL = 12


# -- CCL spacing helpers (NumPy, static) ------------------------------------
def linear_spacing(xmin: float, xmax: float, n: int) -> np.ndarray:
    dx = (xmax - xmin) / (n - 1.0)
    x = xmin + dx * np.arange(n, dtype=float)
    x[0], x[-1] = xmin, xmax
    return x


def log_spacing(xmin: float, xmax: float, n: int) -> np.ndarray:
    dlog = (np.log(xmax) - np.log(xmin)) / (n - 1.0)
    ratio = np.exp(dlog)
    x = np.empty(n)
    x[0] = xmin
    for i in range(1, n - 1):
        x[i] = x[i - 1] * ratio
    x[-1] = xmax
    return x


def linlog_spacing(xminlog: float, xmin: float, xmax: float, nlog: int, nlin: int) -> np.ndarray:
    dx = (xmax - xmin) / (nlin - 1.0)
    lmin, lchange = np.log(xminlog), np.log(xmin)
    dl = (lchange - lmin) / (nlog - 1.0)
    x = np.empty(nlin + nlog - 1)
    for i in range(nlin + nlog - 1):
        x[i] = np.exp(lmin + dl * i) if i < nlog else xmin + dx * (i - nlog + 1)
    x[0], x[nlog - 1], x[-1] = xminlog, xmin, xmax
    return x


def get_pk_spline_lk() -> np.ndarray:
    """``pyccl.pyutils.get_pk_spline_lk()`` with default spline parameters."""
    nk = int(np.ceil((np.log10(K_MAX) - np.log10(K_MIN)) * N_K))
    return np.log(log_spacing(K_MIN, K_MAX, nk))


def get_pk_spline_a() -> np.ndarray:
    """``pyccl.pyutils.get_pk_spline_a()`` with default spline parameters."""
    return linlog_spacing(A_SPLINE_MINLOG_PK, A_SPLINE_MIN_PK, A_SPLINE_MAX,
                          A_SPLINE_NLOG_PK, A_SPLINE_NA_PK)


# -- background helpers --------------------------------------------------------
BACKGROUND_KEYS = ("Omega_l", "Omega_g", "Omega_nu_rel", "Omega_k", "w0", "wa")


def h_over_h0_analytic(cosmo, a):
    """CCL's analytic E(a) (``ccl_background.c:h_over_h0``, no massive nu).

    Used by ``ccl_omega_x`` / ``ccl_rho_x`` (not the E(a) spline, which differs
    at the 1e-6 level). Requires ``Omega_l`` in ``cosmo.params``; otherwise the
    E(a) spline is used.
    """
    a = jnp.asarray(a, dtype=float)
    p = cosmo.params
    if "Omega_l" not in p:
        return cosmo.h_over_h0(a)
    w0, wa = p.get("w0", -1.0), p.get("wa", 0.0)
    e2 = (p["Omega_m"] + p["Omega_l"] * a ** (-3 * (w0 + wa)) * jnp.exp(3 * wa * (a - 1))
          + p.get("Omega_k", 0.0) * a + (p.get("Omega_g", 0.0) + p.get("Omega_nu_rel", 0.0)) / a
          ) / a**3
    return jnp.sqrt(e2)


def omega_x(cosmo, a, label: str):
    """``cosmo.omega_x(a, 'matter' | 'critical')`` (massless neutrinos)."""
    a = jnp.asarray(a, dtype=float)
    if label == "critical":
        return jnp.ones_like(a)
    if label != "matter":
        raise ValueError(f"omega_x label {label!r} not supported")
    e = h_over_h0_analytic(cosmo, a)
    return cosmo["Omega_m"] / (a**3 * e * e)


def rho_x(cosmo, a, label: str, is_comoving: bool = False):
    """``cosmo.rho_x`` in M_sun / Mpc^3 (physical unless ``is_comoving``)."""
    a = jnp.asarray(a, dtype=float)
    if label == "matter":
        rho = RHO_CRITICAL * cosmo["h"] ** 2 * cosmo["Omega_m"] / a**3
    else:
        e = h_over_h0_analytic(cosmo, a)
        rho = RHO_CRITICAL * cosmo["h"] ** 2 * e * e * omega_x(cosmo, a, label)
    if is_comoving:
        rho = rho * a**3
    return rho


def rho_m0(cosmo):
    """Comoving matter density ``RHO_CRITICAL * Omega_m * h^2``."""
    return RHO_CRITICAL * cosmo["Omega_m"] * cosmo["h"] ** 2


def get_delta_c(cosmo, a, kind: str = "EdS"):
    """``pyccl.halos.get_delta_c``."""
    dc0 = 1.68647019984
    if kind == "EdS":
        return dc0
    if kind == "EdS_approx":
        return 1.686
    if kind == "NakamuraSuto97":
        om = omega_x(cosmo, a, "matter")
        return dc0 * (1 + 0.012299 * jnp.log10(om))
    if kind == "Mead16":
        om = omega_x(cosmo, a, "matter")
        s8 = sigmaR(cosmo, 8.0 / cosmo["h"], 1.0) * cosmo.growth_factor(a)
        return (1.59 + 0.0314 * jnp.log(s8)) * (1 + 0.0123 * jnp.log10(om))
    raise ValueError(f"Unknown threshold kind {kind}")


# -- sigma(R) ------------------------------------------------------------------
def _gl_nodes():
    x, w = np.polynomial.legendre.leggauss(SIGMA_N_GL)
    lo, hi = np.log10(K_MIN), np.log10(K_MAX)
    edges = np.linspace(lo, hi, SIGMA_N_PANELS + 1)
    half = 0.5 * np.diff(edges)
    mid = 0.5 * (edges[1:] + edges[:-1])
    nodes = (mid[:, None] + half[:, None] * x[None, :]).ravel()
    weights = (half[:, None] * w[None, :]).ravel()
    return nodes, weights


_LOG10K, _W_LOG10K = _gl_nodes()


def w_tophat(kr):
    kr2 = kr * kr
    series = 1.0 + kr2 * (-1.0 / 10.0 + kr2 * (1.0 / 280.0 + kr2 * (
        -1.0 / 15120.0 + kr2 * (1.0 / 1330560.0 + kr2 * (-1.0 / 172972800.0)))))
    safe = jnp.where(kr < 0.1, 1.0, kr)
    full = 3.0 * (jnp.sin(safe) - safe * jnp.cos(safe)) / safe**3
    return jnp.where(kr < 0.1, series, full)


def _pk_lin_at_nodes(cosmo, a):
    """P_lin at the quadrature nodes for each a -> (n_a, n_k)."""
    pk = cosmo.get_linear_power()
    lk = jnp.asarray(_LOG10K * np.log(10.0))
    a = jnp.atleast_1d(jnp.asarray(a, dtype=float))
    return jax.vmap(lambda aa: pk.eval_lk(lk, aa, cosmo))(a)


def _sigma2_from_pk(pk_nodes, R):
    """sigma^2 for radii ``R`` (n_R,) given ``pk_nodes`` (..., n_k)."""
    k = jnp.asarray(10.0**_LOG10K)
    w = w_tophat(k[None, :] * jnp.asarray(R)[:, None])  # (n_R, n_k)
    integ = (k**3 * jnp.asarray(_W_LOG10K))[None, :] * w * w
    s2 = jnp.einsum("...k,rk->...r", pk_nodes, integ)
    return s2 * np.log(10.0) / (2 * np.pi**2)


def sigmaR(cosmo, R, a=1.0):
    """``pyccl.sigmaR``: rms linear overdensity in top-hats of comoving R [Mpc]."""
    R_use = jnp.atleast_1d(jnp.asarray(R, dtype=float))
    pkn = _pk_lin_at_nodes(cosmo, a)[0]
    out = jnp.sqrt(_sigma2_from_pk(pkn, R_use))
    return out[0] if np.ndim(R) == 0 else out


def sigma8(cosmo):
    return sigmaR(cosmo, 8.0 / cosmo["h"], 1.0)


def _lagrangian_radius_ccl(cosmo, M):
    return (3.0 * M / (4 * np.pi * rho_m0(cosmo))) ** (1.0 / 3.0)


_LOGM_NODES = linear_spacing(LOGM_SPLINE_MIN, LOGM_SPLINE_MAX, LOGM_SPLINE_NM)
_A_NODES_SM = linlog_spacing(A_SPLINE_MINLOG_SM, A_SPLINE_MIN_SM, A_SPLINE_MAX,
                             A_SPLINE_NLOG_SM, A_SPLINE_NA_SM)


def compute_sigma_spline(cosmo) -> Bicubic2D:
    """CCL's bicubic ln sigma(log10 M, a) spline (no caching)."""
    return _compute_sigma_spline_jit(cosmo)


@jax.jit
def _compute_sigma_spline_jit(cosmo) -> Bicubic2D:
    R = _lagrangian_radius_ccl(cosmo, jnp.asarray(10.0**_LOGM_NODES))
    pkn = _pk_lin_at_nodes(cosmo, jnp.asarray(_A_NODES_SM))  # (n_a, n_k)
    lns = 0.5 * jnp.log(_sigma2_from_pk(pkn, R))  # (n_a, n_m)
    return bicubic_spline(jnp.asarray(_LOGM_NODES), jnp.asarray(_A_NODES_SM), lns.T)


_ACTIVE: contextvars.ContextVar = contextvars.ContextVar("gpucosmo_sigma_spline", default=None)


@contextlib.contextmanager
def sigma_context(cosmo, spline=None):
    """Publish a sigma spline for ``cosmo`` to all halo-model ingredients."""
    if spline is None:
        spline = get_sigma_spline(cosmo)
    token = _ACTIVE.set((cosmo, spline))
    try:
        yield spline
    finally:
        _ACTIVE.reset(token)


def _is_concrete(tree) -> bool:
    return not any(isinstance(x, jax.core.Tracer) for x in jax.tree_util.tree_leaves(tree))


def get_sigma_spline(cosmo) -> Bicubic2D:
    """Sigma spline for ``cosmo``: active context, cached, or freshly computed."""
    active = _ACTIVE.get()
    if active is not None and active[0] is cosmo:
        return active[1]
    cached = cosmo.__dict__.get("_gpucosmo_sigma_spline")
    if cached is not None:
        return cached
    spline = compute_sigma_spline(cosmo)
    if _is_concrete(spline):
        object.__setattr__(cosmo, "_gpucosmo_sigma_spline", spline)
    return spline


def sigma_spline_from_ccl(ccl_cosmo) -> Bicubic2D:
    """Copy the node values of a ``pyccl.Cosmology``'s own sigma(M) spline.

    CCL integrates each node with GSL cquad at ``epsrel = 1e-5``; its node
    values scatter by ~1e-7 around the converged integral (which the native
    JAX quadrature reproduces to ~1e-10), and the bicubic x-derivative
    amplifies that scatter to ~2e-5 in dln sigma / dlog10 M. Copying the nodes
    isolates that CCL quadrature noise in parity tests.
    """
    import pyccl

    ccl_cosmo.compute_sigma()
    lns = np.empty((_A_NODES_SM.size, _LOGM_NODES.size))
    for j, aa in enumerate(_A_NODES_SM):
        sig, status = pyccl.lib.sigM_vec(ccl_cosmo.cosmo, float(aa), _LOGM_NODES,
                                         _LOGM_NODES.size, 0)
        lns[j] = np.log(sig)
    return bicubic_spline(jnp.asarray(_LOGM_NODES), jnp.asarray(_A_NODES_SM),
                          jnp.asarray(lns.T))


def cosmology_from_ccl(ccl_cosmo, *, sigma_nodes: bool = False, **kwargs):
    """``gpucosmo.jax.ccl.Cosmology.from_ccl`` plus the halo-model extras.

    Adds the density parameters CCL's ``omega_x`` / ``rho_x`` use analytically
    (``Omega_l``, ``Omega_g``, ``Omega_nu_rel``, ``Omega_k``, ``w0``, ``wa``) to
    ``params``; ``sigma_nodes=True`` also copies CCL's sigma(M) spline nodes
    (see :func:`sigma_spline_from_ccl`).
    """
    import dataclasses

    from gpucosmo.jax.ccl.cosmology import Cosmology

    jc = Cosmology.from_ccl(ccl_cosmo, **kwargs)
    if float(ccl_cosmo["N_nu_mass"]) > 0:
        import warnings

        warnings.warn("massive neutrinos: omega_x / rho_x use Omega_m a^-3 and the E(a) "
                      "spline (CCL adds the massive-nu phase-space density)")
        extra = {}
    else:
        extra = {"Omega_l": float(ccl_cosmo["Omega_l"]),
                 "Omega_g": float(ccl_cosmo["Omega_g"]),
                 "Omega_nu_rel": float(ccl_cosmo.cosmo.params.Omega_nu_rel),
                 "Omega_k": float(ccl_cosmo["Omega_k"]),
                 "w0": float(ccl_cosmo["w0"]), "wa": float(ccl_cosmo["wa"])}
    jc = dataclasses.replace(jc, params={**jc.params, **extra})
    if sigma_nodes:
        compute_sigma(jc, from_ccl=ccl_cosmo)
    return jc


def compute_sigma(cosmo, *, from_ccl=None, spline: Bicubic2D | None = None):
    """``cosmo.compute_sigma()``: build and cache the spline on ``cosmo``.

    ``from_ccl`` (a ``pyccl.Cosmology``) copies CCL's node values instead;
    ``spline`` attaches a precomputed spline. The cached spline is used by all
    halo-model ingredients evaluated with this cosmology object.
    """
    if from_ccl is not None:
        spline = sigma_spline_from_ccl(from_ccl)
    if spline is not None:
        object.__setattr__(cosmo, "_gpucosmo_sigma_spline", spline)
    else:
        get_sigma_spline(cosmo)
    return cosmo


def _eval_args(M, a):
    logM = jnp.log10(jnp.atleast_1d(jnp.asarray(M, dtype=float)))
    a = jnp.clip(jnp.asarray(a, dtype=float), _A_NODES_SM[0], _A_NODES_SM[-1])
    a_b = jnp.broadcast_to(a, logM.shape)
    return logM, a_b


def sigmaM(cosmo, M, a):
    """``pyccl.sigmaM`` from CCL's bicubic spline."""
    s = get_sigma_spline(cosmo)
    logM, a_b = _eval_args(M, a)
    out = jnp.exp(bicubic_eval(s, logM, a_b))
    return out[0] if np.ndim(M) == 0 else out


def dlnsigM_dlogM(cosmo, M, a):
    """``-d ln sigma / d log10 M`` (``lib.dlnsigM_dlogM_vec``)."""
    s = get_sigma_spline(cosmo)
    logM, a_b = _eval_args(M, a)
    out = -bicubic_deriv_x(s, logM, a_b)
    return out[0] if np.ndim(M) == 0 else out


__all__ = [
    "compute_sigma",
    "compute_sigma_spline",
    "cosmology_from_ccl",
    "h_over_h0_analytic",
    "dlnsigM_dlogM",
    "get_delta_c",
    "get_pk_spline_a",
    "get_pk_spline_lk",
    "get_sigma_spline",
    "omega_x",
    "rho_x",
    "sigma8",
    "sigmaM",
    "sigmaR",
    "sigma_context",
    "sigma_spline_from_ccl",
    "w_tophat",
]
