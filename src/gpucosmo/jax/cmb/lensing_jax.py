#!/usr/bin/env python3
# This file is part of GPUCosmo. Copyright (C) 2025-2026 Leonel Medina-Varela.
# SPDX-License-Identifier: GPL-3.0-or-later
# Derived in part from the following, translated to JAX and modified for GPUCosmo:
# - CAMB (https://github.com/cmbant/CAMB), Copyright 2000-2025 Antony Lewis and Anthony Challinor.
#   Licence: LICENSES/CAMB.txt.
# See NOTICE.md.
"""CMB remapping lensing: CAMB correlation-function method as batched matmuls.

Ports ``camb.correlations.lensed_cls`` (method-4 / Gauss–Legendre). Geometry
(Legendre + Wigner-d on the GL θ grid) depends only on ``lmax`` / sampling, so
it is built once on the host and cached. Each cosmology then only does dense
matvecs/matmuls over (n_θ × ℓ), which is the GPU-friendly stage.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from gpucosmo.paths import camb_package_file, data_dir

jax.config.update("jax_enable_x64", True)

_TEMPLATE_NAME = "HighLExtrapTemplate_lenspotentialCls.dat"
_TEMPLATE_PATH = data_dir("camb", _TEMPLATE_NAME)
# Also accept the copy inside the installed camb Python package.
_TEMPLATE_FALLBACKS = tuple(p for p in (camb_package_file(_TEMPLATE_NAME),) if p is not None)


@functools.lru_cache(maxsize=1)
def _load_highl_template() -> np.ndarray:
    path = _TEMPLATE_PATH
    if not path.exists():
        for p in _TEMPLATE_FALLBACKS:
            if p.exists():
                path = p
                break
        else:
            raise FileNotFoundError(
                f"High-L lensing template not found at {_TEMPLATE_PATH} (copy CAMB's {_TEMPLATE_NAME} "
                "there, or `pip install camb`, which ships it)"
            )
    data = np.loadtxt(path)
    lmax = int(data[-1, 0])
    out = np.zeros((lmax + 1, 5), dtype=np.float64)
    ell = data[:, 0].astype(int)
    out[ell, :4] = data[:, 1:5]
    out[ell, 4] = data[:, 5]
    out.flags.writeable = False
    return out


def extrapolate_lensing_inputs(
    cls: np.ndarray, clpp: np.ndarray, lmax: int
) -> tuple[np.ndarray, np.ndarray]:
    """Pad TT/EE/BB/TE and CPP with CAMB's high-L template (scaled at last ℓ)."""
    cls = np.asarray(cls, dtype=np.float64)
    clpp = np.asarray(clpp, dtype=np.float64)
    if cls.ndim == 1:
        raise ValueError("cls must be (lmax+1, 4) TT,EE,BB,TE")
    cls_lmax = cls.shape[0] - 1
    clpp_lmax = clpp.shape[0] - 1
    if lmax <= cls_lmax and lmax <= clpp_lmax:
        return cls[: lmax + 1, :4], clpp[: lmax + 1]

    template = _load_highl_template()
    if lmax >= template.shape[0]:
        raise ValueError(f"template only to lmax={template.shape[0] - 1}")

    cls_out = np.zeros((lmax + 1, 4), dtype=np.float64)
    cls_out[: min(cls_lmax, lmax) + 1] = cls[: min(cls_lmax, lmax) + 1, :4]
    clpp_out = np.zeros(lmax + 1, dtype=np.float64)
    clpp_out[: min(clpp_lmax, lmax) + 1] = clpp[: min(clpp_lmax, lmax) + 1]

    if lmax > cls_lmax:
        tt_scale = cls_out[cls_lmax, 0] / template[cls_lmax, 0]
        ee_scale = cls_out[cls_lmax, 1] / template[cls_lmax, 1]
        te_scale = np.sqrt(max(0.0, tt_scale * ee_scale))
        cls_out[cls_lmax + 1 :, 0] = template[cls_lmax + 1 : lmax + 1, 0] * tt_scale
        cls_out[cls_lmax + 1 :, 1] = template[cls_lmax + 1 : lmax + 1, 1] * ee_scale
        cls_out[cls_lmax + 1 :, 2] = template[cls_lmax + 1 : lmax + 1, 2]
        cls_out[cls_lmax + 1 :, 3] = template[cls_lmax + 1 : lmax + 1, 3] * te_scale

    if lmax > clpp_lmax:
        clpp_scale = clpp_out[clpp_lmax] / template[clpp_lmax, 4]
        clpp_out[clpp_lmax + 1 :] = template[clpp_lmax + 1 : lmax + 1, 4] * clpp_scale

    return cls_out, clpp_out


def _legendre_funcs_one(x: float, lmax: int):
    """P, dP, d11, dm11, d20, d22, d2m2 for one cosθ (CAMB correlations.legendre_funcs)."""
    try:
        from scipy.special import legendre_p_all

        allP, alldP = legendre_p_all(lmax, x, diff_n=1)
    except ImportError:
        from scipy.special import lpn

        allP, alldP = lpn(lmax, x)

    allP = np.asarray(allP, dtype=np.float64)
    alldP = np.asarray(alldP, dtype=np.float64)
    ls = np.arange(2, lmax + 1, dtype=np.float64)
    lfacs = ls * (ls + 1)
    lfacs2 = (ls + 2) * (ls - 1)
    lrootfacs = np.sqrt(lfacs * lfacs2)
    lfacs1 = np.arange(1, lmax + 1, dtype=np.float64)
    lfacs1 = lfacs1 * (1 + lfacs1)

    fac1 = 1.0 - x
    fac2 = 1.0 + x
    d11 = fac1 * alldP[1:] / lfacs1 + allP[1:]
    dm11 = fac2 * alldP[1:] / lfacs1 - allP[1:]

    P = allP[2:]
    dP = alldP[2:]
    fac = fac1 / fac2
    d22 = (((4 * x - 8) / fac2 + lfacs) * P + 4 * fac * (fac2 + (x - 2) / lfacs) * dP) / lfacs2
    if x > 0.998:
        d2m2 = np.empty(lmax - 1, dtype=np.float64)
        indser = int(np.sqrt((400.0 + 3 / (1 - x**2)) / 150)) - 1
        d2m2[indser:] = (
            (lfacs[indser:] - (4 * x + 8) / fac1) * P[indser:]
            + 4 / fac * (-fac1 + (x + 2) / lfacs[indser:]) * dP[indser:]
        ) / lfacs2[indser:]
        sin2 = 1 - x**2
        d2m2[:indser] = (
            lfacs[:indser] * lfacs2[:indser] * sin2**2 / 7680 * (20 + sin2 * (16 - lfacs[:indser]))
        )
    else:
        d2m2 = ((lfacs - (4 * x + 8) / fac1) * P + 4 / fac * (-fac1 + (x + 2) / lfacs) * dP) / lfacs2
    d20 = (2 * x * dP - lfacs * P) / lrootfacs
    return allP, alldP, d11, dm11, d20, d22, d2m2


def _apodize_theta_width(theta_max, lmax, apodize_point_width):
    if theta_max is None or apodize_point_width <= 0:
        return 0.0
    return min(float(theta_max), 4.8 * float(apodize_point_width) * np.pi / lmax)


def _low_l_ee_taper(lmax: int) -> np.ndarray:
    taper = np.clip((np.arange(2, lmax + 1, dtype=np.float64) - 2) / 18, 0, 1)
    return taper**2 * (3 - 2 * taper)


@functools.lru_cache(maxsize=8)
def _geometry_tables(
    lmax: int,
    sampling_factor: float,
    theta_max: float | None,
    apodize_point_width: float,
) -> dict[str, np.ndarray | float | int]:
    """Cache GL nodes + Legendre/Wigner-d stacks (cosmology-independent), also on disk
    (``host_table_cache``): building them takes ~30 s, mostly ``leggauss``."""
    from gpucosmo.jax.host_table_cache import cached_npz

    key = (int(lmax), float(sampling_factor), None if theta_max is None else float(theta_max),
           float(apodize_point_width))
    return cached_npz("lensing_geometry", key, lambda: _build_geometry_tables(*key), module=__name__)


# Up to this lmax the Gauss–Legendre rule comes from ``numpy.polynomial.legendre.leggauss``
# (as it always did: the lensed spectra below it are unchanged). Above it (high-ℓ lensing,
# lmax_conv up to ~2·10⁴) leggauss's dense eigvalsh would need O(n²) memory (3 GB at n = 2·10⁴)
# and O(n³) time, and only the ~3 % of nodes with θ < θ_max are used: those are found by
# Newton iteration instead (``_gauss_legendre_upper``).
LEGGAUSS_LMAX = 8000


def _gauss_legendre_upper(n: int, xmin: float) -> tuple[np.ndarray, np.ndarray]:
    """Nodes x ≥ ``xmin`` (ascending) and weights of the ``n``-point Gauss–Legendre rule.

    Newton iteration on P_n (three-term recurrence, O(n) per node) from Tricomi's
    asymptotic guesses x_k ≈ (1 − 1/(8n²) + 1/(8n³)) cos(π(4k−1)/(4n+2)), k = 1, 2, …
    counted from x = 1. Nodes agree with ``leggauss`` to 1e-16; the weights are exact to
    ~1e-15 (Σw − 2 ≈ 4e-16 at n = 7246), ``leggauss``'s are off by up to 3e-6 relative.
    """
    n = int(n)
    th_min = float(np.arccos(np.clip(xmin, -1.0, 1.0)))
    # Node k sits at θ ≈ π(4k−1)/(4n+2); a few spare nodes, trimmed after convergence.
    k_max = min(n, int(np.ceil(th_min * (4 * n + 2) / (4 * np.pi) + 0.25)) + 3)
    k = np.arange(1, k_max + 1, dtype=np.float64)
    x = (1.0 - 1.0 / (8.0 * n**2) + 1.0 / (8.0 * n**3)) * np.cos(np.pi * (4.0 * k - 1.0) / (4.0 * n + 2.0))

    def p_and_dp(x):
        p0, p1 = np.ones_like(x), x.copy()
        for j in range(1, n):
            p0, p1 = p1, ((2.0 * j + 1.0) * x * p1 - j * p0) / (j + 1.0)
        dp = n * (x * p1 - p0) / (x * x - 1.0)
        return p1, dp

    for _ in range(100):
        p, dp = p_and_dp(x)
        dx = p / dp
        x = x - dx
        if np.max(np.abs(dx)) < 1e-16:
            break
    _, dp = p_and_dp(x)
    w = 2.0 / ((1.0 - x * x) * dp * dp)
    keep = x >= xmin
    return x[keep][::-1].copy(), w[keep][::-1].copy()


def _build_geometry_tables(
    lmax: int,
    sampling_factor: float,
    theta_max: float | None,
    apodize_point_width: float,
) -> dict[str, np.ndarray | float | int]:
    npoints = int(sampling_factor * lmax) + 1
    if lmax > LEGGAUSS_LMAX and theta_max is not None:
        xvals, weights = _gauss_legendre_upper(npoints, float(np.cos(theta_max)))
    else:
        xvals, weights = np.polynomial.legendre.leggauss(npoints)
    if theta_max is not None:
        xmin = np.cos(theta_max)
        imin = int(np.searchsorted(xvals, xmin))
        apodize_w = _apodize_theta_width(theta_max, lmax, apodize_point_width)
    else:
        imin = 0
        apodize_w = 0.0

    xs = xvals[imin:]
    ws = weights[imin:].copy()
    if apodize_w > 0.0 and theta_max is not None:
        th = np.arccos(np.clip(xs, -1.0, 1.0))
        taper = np.clip((theta_max - th) / apodize_w, 0.0, 1.0)
        ws *= taper**3 * (10.0 + taper * (-15.0 + 6.0 * taper))

    n_th = xs.size
    P0 = np.zeros((n_th, lmax + 1), dtype=np.float64)
    dP0 = np.zeros((n_th, lmax + 1), dtype=np.float64)
    d11 = np.zeros((n_th, lmax), dtype=np.float64)  # ℓ=1..lmax
    dm11 = np.zeros((n_th, lmax), dtype=np.float64)
    d20 = np.zeros((n_th, lmax - 1), dtype=np.float64)  # ℓ=2..lmax
    d22 = np.zeros((n_th, lmax - 1), dtype=np.float64)
    d2m2 = np.zeros((n_th, lmax - 1), dtype=np.float64)

    # Higher-spin stacks needed for pol (sizes relative to ℓ=2..)
    d12 = np.zeros((n_th, lmax - 1), dtype=np.float64)
    d1m2 = np.zeros((n_th, lmax - 1), dtype=np.float64)
    d13 = np.zeros((n_th, lmax - 2), dtype=np.float64)
    d1m3 = np.zeros((n_th, lmax - 2), dtype=np.float64)
    d2m3 = np.zeros((n_th, lmax - 2), dtype=np.float64)
    d3m3 = np.zeros((n_th, lmax - 2), dtype=np.float64)
    d04 = np.zeros((n_th, lmax - 3), dtype=np.float64)
    d2m4 = np.zeros((n_th, lmax - 3), dtype=np.float64)
    d4m4 = np.zeros((n_th, lmax - 3), dtype=np.float64)

    ls2 = np.arange(2, lmax + 1, dtype=np.float64)
    lfacs = ls2 * (ls2 + 1)
    lfacs2 = (ls2 + 2) * (ls2 - 1)
    lrootfacs = np.sqrt(lfacs * lfacs2)
    rootfac1 = np.sqrt(lfacs2)
    rootfac2 = np.sqrt((ls2[1:] + 3) * (ls2[1:] - 2))
    rootrat = lfacs2[1:] / rootfac1[1:] / rootfac2
    rootfac3 = np.sqrt((ls2[2:] - 3) * (ls2[2:] + 4))

    for i, x in enumerate(xs):
        allP, alldP, d11_i, dm11_i, d20_i, d22_i, d2m2_i = _legendre_funcs_one(float(x), lmax)
        P0[i] = allP
        dP0[i] = alldP
        d11[i] = d11_i
        dm11[i] = dm11_i
        d20[i] = d20_i
        d22[i] = d22_i
        d2m2[i] = d2m2_i

        sinth = np.sqrt(max(1.0 - x * x, 0.0))
        sinfac = 4.0 / max(sinth, 1e-300)
        fac1 = 1.0 - x
        fac2 = 1.0 + x
        d1m2[i] = sinth / rootfac1 * (alldP[2:] - 2 / fac1 * dm11_i[1:])
        d12[i] = sinth / rootfac1 * (alldP[2:] - 2 / fac2 * d11_i[1:])
        d1m3[i] = -(x + 0.5) * sinfac * d1m2[i, 1:] / rootfac2 - rootrat * dm11_i[2:]
        d2m3[i] = (-fac2 * d2m2_i[1:] * sinfac - rootfac1[1:] * d1m2[i, 1:]) / rootfac2
        d3m3[i] = (-(x + 1.5) * d2m3[i] * sinfac - rootfac1[1:] * d1m3[i]) / rootfac2
        d13[i] = (x - 0.5) * sinfac * d12[i, 1:] / rootfac2 - rootrat * d11_i[2:]
        d04[i] = (
            (-lfacs[2:] + (18 * x**2 + 6) / max(sinth**2, 1e-300)) * d20_i[2:]
            - 6 * x * lfacs2[2:] * alldP[4:] / lrootfacs[2:]
        ) / (rootfac2[1:] * rootfac3)
        d2m4[i] = (-(6 * x + 4) / max(sinth, 1e-300) * d2m3[i, 1:] - rootfac2[1:] * d2m2_i[2:]) / rootfac3
        d4m4[i] = (
            -7 / 5.0 * (lfacs[2:] - 6) * d2m2_i[2:]
            + 12 / 5.0 * (-lfacs[2:] + (9 * x + 26) / fac1) * d3m3[i, 1:]
        ) / (lfacs[2:] - 12)

    return {
        "lmax": lmax,
        "P": P0,
        "d11": d11,
        "dm11": dm11,
        "d20": d20,
        "d22": d22,
        "d2m2": d2m2,
        "d13": d13,
        "d1m3": d1m3,
        "d3m3": d3m3,
        "d04": d04,
        "d2m4": d2m4,
        "d4m4": d4m4,
        "weights": ws,
        "n_th": n_th,
    }


def _lens_delta_numpy(
    cls: np.ndarray,
    clpp: np.ndarray,
    *,
    lmax: int,
    sampling_factor: float = 1.4,
    theta_max: float | None = np.pi / 32,
    apodize_point_width: float = 10.0,
    low_l_ee_taper: bool = True,
) -> np.ndarray:
    """ΔD_ℓ = lensed − unlensed via cached geometry + dense matvecs."""
    geo = _geometry_tables(lmax, float(sampling_factor), theta_max, float(apodize_point_width))
    ls = np.arange(0, lmax + 1, dtype=np.float64)
    lfacs = ls * (ls + 1)
    lfacsall = lfacs.copy()
    lfacs_safe = lfacs.copy()
    lfacs_safe[0] = 1.0
    cldd = clpp[1 : lmax + 1] / lfacs_safe[1:]
    cphil3 = (2 * ls[1:] + 1) * cldd / 2.0
    facs = (2 * ls + 1) / (4 * np.pi) * 2 * np.pi / lfacs_safe

    ct = facs * cls[: lmax + 1, 0]
    ee = cls[2 : lmax + 1, 1].copy()
    if low_l_ee_taper:
        ee *= _low_l_ee_taper(lmax)
    cp = facs[2:] * (ee + cls[2 : lmax + 1, 2])
    cm = facs[2:] * (ee - cls[2 : lmax + 1, 2])
    cc = facs[2:] * cls[2 : lmax + 1, 3]

    P = geo["P"]
    d11 = geo["d11"]
    dm11 = geo["dm11"]
    d20 = geo["d20"]
    d22 = geo["d22"]
    d2m2 = geo["d2m2"]
    w = geo["weights"]

    # (n_th,)
    sigma2 = (1.0 - d11) @ cphil3
    Cg2 = dm11 @ cphil3
    c2fac = lfacsall[1:] * Cg2[:, None] / 2.0  # (n_th, lmax)
    c2fac2 = c2fac[:, 1:] ** 2  # (n_th, lmax-1) for ℓ>=2 indices of c2fac[1:]->ℓ>=2

    fac = np.exp(-lfacsall[None, :] * sigma2[:, None] / 2.0)  # (n_th, lmax+1)
    difffac = fac  # delta=True ⇒ fac - 0 was wrong in CAMB for delta; they use fac - 1? 
    # CAMB: delta_diff = 1 if delta else 0; difffac = fac - delta_diff
    difffac = fac - 1.0

    f = ct[None, :] * fac
    corrs0 = (
        (ct[None, :] * difffac * P).sum(axis=1)
        + (f[:, 1:] * c2fac * (dm11 + c2fac * P[:, 1:] / 4.0)).sum(axis=1)
        + (f[:, 2:] * c2fac2 * d2m2).sum(axis=1) / 4.0
    )

    f = cp[None, :] * fac[:, 2:]
    c2_from2 = c2fac[:, 1:]  # ℓ>=2
    c2_from3 = c2fac[:, 2:]  # ℓ>=3
    c22 = c2fac2  # ℓ>=2 squared = c2fac[1:]^2
    c22_from4 = c2fac2[:, 2:]  # ℓ>=4
    corrs1 = (
        (cp[None, :] * difffac[:, 2:] * d22).sum(axis=1)
        + (f[:, 1:] * c2_from3 * geo["d13"]).sum(axis=1)
        + ((f * c22 * d22).sum(axis=1) + (f[:, 2:] * c22_from4 * geo["d04"]).sum(axis=1)) / 4.0
    )

    f = cm[None, :] * fac[:, 2:]
    corrs2 = (
        (cm[None, :] * difffac[:, 2:] * d2m2).sum(axis=1)
        + ((f * c2_from2 * dm11[:, 1:]).sum(axis=1) + (f[:, 1:] * c2_from3 * geo["d3m3"]).sum(axis=1)) / 2.0
        + ((f * c22 * (2 * d2m2 + P[:, 2:])).sum(axis=1) + (f[:, 2:] * c22_from4 * geo["d4m4"]).sum(axis=1)) / 8.0
    )

    f = cc[None, :] * fac[:, 2:]
    corrs3 = (
        (cc[None, :] * difffac[:, 2:] * d20).sum(axis=1)
        + ((f * c2_from2 * d11[:, 1:]).sum(axis=1) + (f[:, 1:] * c2_from3 * geo["d1m3"]).sum(axis=1)) / 2.0
        + ((3 * f * c22 * d20).sum(axis=1) + (f[:, 2:] * c22_from4 * geo["d2m4"]).sum(axis=1)) / 8.0
    )

    lensed = np.zeros((lmax + 1, 4), dtype=np.float64)
    lensed[:, 0] = (w[:, None] * corrs0[:, None] * P).sum(axis=0)
    T2 = (corrs1 * w / 2.0)[:, None] * d22
    T4 = (corrs2 * w / 2.0)[:, None] * d2m2
    lensed[2:, 1] = (T2 + T4).sum(axis=0)
    lensed[2:, 2] = (T2 - T4).sum(axis=0)
    lensed[2:, 3] = (w[:, None] * corrs3[:, None] * d20).sum(axis=0)

    lensed[1, :] *= 2.0
    lensed[2:, :] = (lensed[2:, :].T * lfacs[2:]).T
    return lensed


_GEOMETRY_DEVICE_KEYS = (
    "P", "d11", "dm11", "d20", "d22", "d2m2",
    "d13", "d1m3", "d3m3", "d04", "d2m4", "d4m4", "weights",
)


@functools.lru_cache(maxsize=8)
def _geometry_device(
    lmax: int,
    sampling_factor: float,
    theta_max: float | None,
    apodize_point_width: float,
) -> dict[str, jnp.ndarray]:
    """``_geometry_tables`` as device arrays (built eagerly, safe under tracing)."""
    geo = _geometry_tables(lmax, sampling_factor, theta_max, apodize_point_width)
    with jax.enable_x64(True), jax.ensure_compile_time_eval():  # fp64 even inside fp32 regions
        return {k: jnp.asarray(geo[k], dtype=jnp.float64) for k in _GEOMETRY_DEVICE_KEYS}


def _gl_nodes_weights(lmax: int, sampling_factor: float, theta_max: float,
                      apodize_point_width: float) -> tuple[np.ndarray, np.ndarray]:
    """Host: the (apodized) Gauss–Legendre nodes x = cos θ < θ_max and weights of
    ``_build_geometry_tables`` with Newton-iteration nodes."""
    npoints = int(sampling_factor * lmax) + 1
    xs, ws = _gauss_legendre_upper(npoints, float(np.cos(theta_max)))
    apodize_w = _apodize_theta_width(theta_max, lmax, apodize_point_width)
    if apodize_w > 0.0:
        th = np.arccos(np.clip(xs, -1.0, 1.0))
        taper = np.clip((theta_max - th) / apodize_w, 0.0, 1.0)
        ws = ws * taper**3 * (10.0 + taper * (-15.0 + 6.0 * taper))
    return xs, ws


def geometry_runtime(
    lmax: int,
    sampling_factor: float = 1.4,
    theta_max: float = np.pi / 32,
    apodize_point_width: float = 10.0,
) -> dict[str, jnp.ndarray]:
    """``_geometry_device`` computed on the device at run time (jit / vmap safe).

    The host tables enter compiled programs as embedded constants: 0.3 GB at lmax 8000,
    1.4 GB at 17500, which made XLA's compile exceed 7 GB of host memory. Here only the
    n_θ nodes and weights (Newton iteration, ``_gl_nodes_weights``) are constants; P_ℓ and
    dP_ℓ/dx come from the three-term recurrences (a ``lax.scan`` over ℓ, ~13k steps on n_θ
    ≈ 600 nodes) and the Wigner-d stacks from the closed forms of ``_legendre_funcs_one`` /
    ``_build_geometry_tables``. Lensed spectra agree with those from the host tables (same
    nodes) to ≤ 1.2e-7 relative at lmax 8000; with ``leggauss`` nodes to ~1.6e-5.
    """
    lmax = int(lmax)
    xs, ws = _gl_nodes_weights(lmax, float(sampling_factor), float(theta_max), float(apodize_point_width))
    x = jnp.asarray(xs, dtype=jnp.float64)
    w = jnp.asarray(ws, dtype=jnp.float64)

    def step(carry, ell):
        p_prev, p, dp_prev, dp = carry
        p_next = ((2.0 * ell + 1.0) * x * p - ell * p_prev) / (ell + 1.0)
        dp_next = dp_prev + (2.0 * ell + 1.0) * p
        return (p, p_next, dp, dp_next), (p_next, dp_next)

    one, zero = jnp.ones_like(x), jnp.zeros_like(x)
    _, (ps, dps) = jax.lax.scan(step, (one, x, zero, one), jnp.arange(1, lmax, dtype=jnp.float64), unroll=8)
    allP = jnp.concatenate([one[:, None], x[:, None], ps.T], axis=1)  # (n_th, lmax+1)
    alldP = jnp.concatenate([zero[:, None], one[:, None], dps.T], axis=1)

    xc = x[:, None]
    ls = jnp.arange(2, lmax + 1, dtype=jnp.float64)
    lfacs = ls * (ls + 1)
    lfacs2 = (ls + 2) * (ls - 1)
    lrootfacs = jnp.sqrt(lfacs * lfacs2)
    l1 = jnp.arange(1, lmax + 1, dtype=jnp.float64)
    lfacs1 = l1 * (1 + l1)
    fac1, fac2 = 1.0 - xc, 1.0 + xc
    d11 = fac1 * alldP[:, 1:] / lfacs1 + allP[:, 1:]
    dm11 = fac2 * alldP[:, 1:] / lfacs1 - allP[:, 1:]
    P, dP = allP[:, 2:], alldP[:, 2:]
    fac = fac1 / fac2
    d22 = (((4 * xc - 8) / fac2 + lfacs) * P + 4 * fac * (fac2 + (xc - 2) / lfacs) * dP) / lfacs2
    d2m2 = ((lfacs - (4 * xc + 8) / fac1) * P + 4 / fac * (-fac1 + (xc + 2) / lfacs) * dP) / lfacs2
    # Small-ℓ series near θ = 0 (as _legendre_funcs_one for x > 0.998).
    sin2 = 1 - xc**2
    indser = np.where(xs > 0.998, np.floor(np.sqrt((400.0 + 3 / (1 - xs**2)) / 150)) - 1, 0.0)
    ser_mask = jnp.arange(lmax - 1)[None, :] < jnp.asarray(indser, dtype=jnp.float64)[:, None]
    d2m2_ser = lfacs * lfacs2 * sin2**2 / 7680 * (20 + sin2 * (16 - lfacs))
    d2m2 = jnp.where(ser_mask, d2m2_ser, d2m2)
    d20 = (2 * xc * dP - lfacs * P) / lrootfacs

    sinth = jnp.sqrt(jnp.maximum(1.0 - xc * xc, 0.0))
    sinfac = 4.0 / jnp.maximum(sinth, 1e-300)
    rootfac1 = jnp.sqrt(lfacs2)
    rootfac2 = jnp.sqrt((ls[1:] + 3) * (ls[1:] - 2))
    rootrat = lfacs2[1:] / rootfac1[1:] / rootfac2
    rootfac3 = jnp.sqrt((ls[2:] - 3) * (ls[2:] + 4))
    d1m2 = sinth / rootfac1 * (alldP[:, 2:] - 2 / fac1 * dm11[:, 1:])
    d12 = sinth / rootfac1 * (alldP[:, 2:] - 2 / fac2 * d11[:, 1:])
    d1m3 = -(xc + 0.5) * sinfac * d1m2[:, 1:] / rootfac2 - rootrat * dm11[:, 2:]
    d2m3 = (-fac2 * d2m2[:, 1:] * sinfac - rootfac1[1:] * d1m2[:, 1:]) / rootfac2
    d3m3 = (-(xc + 1.5) * d2m3 * sinfac - rootfac1[1:] * d1m3) / rootfac2
    d13 = (xc - 0.5) * sinfac * d12[:, 1:] / rootfac2 - rootrat * d11[:, 2:]
    d04 = ((-lfacs[2:] + (18 * xc**2 + 6) / jnp.maximum(sinth**2, 1e-300)) * d20[:, 2:]
           - 6 * xc * lfacs2[2:] * alldP[:, 4:] / lrootfacs[2:]) / (rootfac2[1:] * rootfac3)
    d2m4 = (-(6 * xc + 4) / jnp.maximum(sinth, 1e-300) * d2m3[:, 1:] - rootfac2[1:] * d2m2[:, 2:]) / rootfac3
    d4m4 = (-7 / 5.0 * (lfacs[2:] - 6) * d2m2[:, 2:]
            + 12 / 5.0 * (-lfacs[2:] + (9 * xc + 26) / fac1) * d3m3[:, 1:]) / (lfacs[2:] - 12)
    return {"P": allP, "d11": d11, "dm11": dm11, "d20": d20, "d22": d22, "d2m2": d2m2, "d13": d13,
            "d1m3": d1m3, "d3m3": d3m3, "d04": d04, "d2m4": d2m4, "d4m4": d4m4, "weights": w}


def _lens_delta_jax(
    cls: jnp.ndarray,
    clpp: jnp.ndarray,
    geo: dict[str, jnp.ndarray],
    *,
    lmax: int,
    low_l_ee_taper: bool = True,
) -> jnp.ndarray:
    """``_lens_delta_numpy`` in jax.numpy: jit / vmap / grad friendly.

    ``cls`` is (lmax+1, 4) D_ℓ-normalized TT, EE, BB, TE; ``clpp`` is (lmax+1,)
    [L(L+1)]² C_φφ / 2π. Returns ΔD_ℓ (lensed − unlensed), shape (lmax+1, 4).
    """
    ls = jnp.arange(0, lmax + 1, dtype=jnp.float64)
    lfacs = ls * (ls + 1)
    lfacs_safe = lfacs.at[0].set(1.0)
    cldd = clpp[1 : lmax + 1] / lfacs_safe[1:]
    cphil3 = (2 * ls[1:] + 1) * cldd / 2.0
    facs = (2 * ls + 1) / (4 * jnp.pi) * 2 * jnp.pi / lfacs_safe

    ct = facs * cls[: lmax + 1, 0]
    ee = cls[2 : lmax + 1, 1]
    if low_l_ee_taper:
        ee = ee * jnp.asarray(_low_l_ee_taper(lmax))
    cp = facs[2:] * (ee + cls[2 : lmax + 1, 2])
    cm = facs[2:] * (ee - cls[2 : lmax + 1, 2])
    cc = facs[2:] * cls[2 : lmax + 1, 3]

    P, d11, dm11 = geo["P"], geo["d11"], geo["dm11"]
    d20, d22, d2m2 = geo["d20"], geo["d22"], geo["d2m2"]
    w = geo["weights"]

    sigma2 = (1.0 - d11) @ cphil3
    Cg2 = dm11 @ cphil3
    c2fac = lfacs[1:] * Cg2[:, None] / 2.0
    c2fac2 = c2fac[:, 1:] ** 2

    fac = jnp.exp(-lfacs[None, :] * sigma2[:, None] / 2.0)
    difffac = fac - 1.0

    f = ct[None, :] * fac
    corrs0 = (
        (ct[None, :] * difffac * P).sum(axis=1)
        + (f[:, 1:] * c2fac * (dm11 + c2fac * P[:, 1:] / 4.0)).sum(axis=1)
        + (f[:, 2:] * c2fac2 * d2m2).sum(axis=1) / 4.0
    )

    f = cp[None, :] * fac[:, 2:]
    c2_from2 = c2fac[:, 1:]
    c2_from3 = c2fac[:, 2:]
    c22 = c2fac2
    c22_from4 = c2fac2[:, 2:]
    corrs1 = (
        (cp[None, :] * difffac[:, 2:] * d22).sum(axis=1)
        + (f[:, 1:] * c2_from3 * geo["d13"]).sum(axis=1)
        + ((f * c22 * d22).sum(axis=1) + (f[:, 2:] * c22_from4 * geo["d04"]).sum(axis=1)) / 4.0
    )

    f = cm[None, :] * fac[:, 2:]
    corrs2 = (
        (cm[None, :] * difffac[:, 2:] * d2m2).sum(axis=1)
        + ((f * c2_from2 * dm11[:, 1:]).sum(axis=1)
           + (f[:, 1:] * c2_from3 * geo["d3m3"]).sum(axis=1)) / 2.0
        + ((f * c22 * (2 * d2m2 + P[:, 2:])).sum(axis=1)
           + (f[:, 2:] * c22_from4 * geo["d4m4"]).sum(axis=1)) / 8.0
    )

    f = cc[None, :] * fac[:, 2:]
    corrs3 = (
        (cc[None, :] * difffac[:, 2:] * d20).sum(axis=1)
        + ((f * c2_from2 * d11[:, 1:]).sum(axis=1)
           + (f[:, 1:] * c2_from3 * geo["d1m3"]).sum(axis=1)) / 2.0
        + ((3 * f * c22 * d20).sum(axis=1)
           + (f[:, 2:] * c22_from4 * geo["d2m4"]).sum(axis=1)) / 8.0
    )

    tt = (w * corrs0) @ P
    T2 = (corrs1 * w / 2.0) @ d22
    T4 = (corrs2 * w / 2.0) @ d2m2
    te = (w * corrs3) @ d20
    zeros2 = jnp.zeros(2, dtype=jnp.float64)
    lensed = jnp.stack([
        tt,
        jnp.concatenate([zeros2, T2 + T4]),
        jnp.concatenate([zeros2, T2 - T4]),
        jnp.concatenate([zeros2, te]),
    ], axis=1)
    scale = jnp.concatenate([jnp.ones(1), jnp.full(1, 2.0), lfacs[2:]])
    return lensed * scale[:, None]


def extrapolate_lensing_inputs_jax(
    cls: jnp.ndarray, clpp: jnp.ndarray, lmax: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``extrapolate_lensing_inputs`` in jax.numpy (static input/output lengths)."""
    cls_lmax = cls.shape[0] - 1
    clpp_lmax = clpp.shape[0] - 1
    if lmax <= cls_lmax and lmax <= clpp_lmax:
        return cls[: lmax + 1, :4], clpp[: lmax + 1]
    template = _load_highl_template()
    if lmax >= template.shape[0]:
        raise ValueError(f"template only to lmax={template.shape[0] - 1}")
    tmpl = jnp.asarray(template[: lmax + 1])

    n_c = min(cls_lmax, lmax) + 1
    if lmax > cls_lmax:
        tt_s = cls[cls_lmax, 0] / tmpl[cls_lmax, 0]
        ee_s = cls[cls_lmax, 1] / tmpl[cls_lmax, 1]
        te_s = jnp.sqrt(jnp.maximum(0.0, tt_s * ee_s))
        tail = tmpl[cls_lmax + 1 :]
        ext = jnp.stack([tail[:, 0] * tt_s, tail[:, 1] * ee_s, tail[:, 2], tail[:, 3] * te_s],
                        axis=1)
        cls_out = jnp.concatenate([cls[:n_c, :4], ext], axis=0)
    else:
        cls_out = cls[: lmax + 1, :4]
    n_p = min(clpp_lmax, lmax) + 1
    if lmax > clpp_lmax:
        pp_s = clpp[clpp_lmax] / tmpl[clpp_lmax, 4]
        clpp_out = jnp.concatenate([clpp[:n_p], tmpl[clpp_lmax + 1 :, 4] * pp_s])
    else:
        clpp_out = clpp[: lmax + 1]
    return cls_out, clpp_out


def lens_cmb_cls(
    dl_tt,
    dl_ee,
    dl_te,
    cpp,
    *,
    dl_bb=None,
    lmax: int | None = None,
    lmax_lensed: int | None = None,
    lmax_conv: int | None = None,
    sampling_factor: float = 1.4,
    theta_max: float | None = np.pi / 32,
    apodize_point_width: float = 10.0,
    low_l_ee_taper: bool = True,
    use_lensing_template: bool = True,
    device_geometry: bool = False,
    geometry: dict | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Lens CMB D_ℓ = ℓ(ℓ+1)C_ℓ/2π with CPP = [L(L+1)]² C_φφ/2π.

    Returns ``(dl_tt, dl_ee, dl_bb, dl_te)`` lensed, length ``lmax_lensed+1``.
    Pure jax.numpy after the (cached, host-built) geometry tables, so it runs
    under ``jit`` / ``vmap`` / ``grad``; ``_lens_delta_numpy`` is the reference.
    ``device_geometry`` (high-ℓ extension): Legendre / Wigner-d tables computed on the device
    at run time (``geometry_runtime``) instead of host tables embedded in the program;
    ``geometry``: such tables precomputed for the convolution range (shared by a batch).
    """
    tt = jnp.asarray(dl_tt, dtype=jnp.float64).ravel()
    ee = jnp.asarray(dl_ee, dtype=jnp.float64).ravel()
    te = jnp.asarray(dl_te, dtype=jnp.float64).ravel()
    pp = jnp.asarray(cpp, dtype=jnp.float64).ravel()
    bb = jnp.zeros_like(tt) if dl_bb is None else jnp.asarray(dl_bb, dtype=jnp.float64).ravel()
    n = min(tt.size, ee.size, te.size, pp.size, bb.size)
    if lmax is None:
        lmax = n - 1
    cls = jnp.stack([tt[:n], ee[:n], bb[:n], te[:n]], axis=1)
    clpp = pp[:n]

    # The convolution reaches CAMB's template range (8000, CAMB lmax_extrap_highl) or, for
    # inputs computed further (high-ℓ extension), the input range: CAMB does not extrapolate
    # past 8000 either.
    conv_cap = max(8000, int(lmax))
    if lmax_conv is None:
        # Match CAMB LensingExtrapLmax rough default for AccuracyTarget=1.
        out_lmax = lmax
        extrap_margin = max(750, int(np.ceil(0.45 * out_lmax + 400)))
        lmax_conv = min(conv_cap, out_lmax + 50 + extrap_margin)
    lmax_conv = int(min(lmax_conv, conv_cap))

    if use_lensing_template:
        cls, clpp = extrapolate_lensing_inputs_jax(cls, clpp, lmax_conv)
        lmax_use = lmax_conv
    else:
        lmax_use = min(lmax, cls.shape[0] - 1, clpp.shape[0] - 1)
        cls = cls[: lmax_use + 1]
        clpp = clpp[: lmax_use + 1]

    if geometry is not None:
        geo = geometry
        if int(geo["P"].shape[-1]) != int(lmax_use) + 1:
            raise ValueError(f"geometry for lmax {int(geo['P'].shape[-1]) - 1}, convolution needs {lmax_use}")
    elif device_geometry:
        if theta_max is None:
            raise ValueError("device_geometry needs theta_max (only the nodes with θ < θ_max are built)")
        geo = geometry_runtime(int(lmax_use), float(sampling_factor), theta_max, float(apodize_point_width))
    else:
        geo = _geometry_device(
            int(lmax_use), float(sampling_factor), theta_max, float(apodize_point_width)
        )
    delta = _lens_delta_jax(cls, clpp, geo, lmax=int(lmax_use), low_l_ee_taper=low_l_ee_taper)
    lensed = cls[: lmax_use + 1] + delta
    if lmax_lensed is None:
        lmax_lensed = lmax
    lmax_lensed = int(min(lmax_lensed, lmax_use))
    out = lensed[: lmax_lensed + 1]
    return out[:, 0], out[:, 1], out[:, 2], out[:, 3]


def lens_cmb_cls_from_raw(
    cl_tt,
    cl_ee,
    cl_te,
    cl_pp,
    ell,
    *,
    t_cmb_muk: float = 2.7255e6,
    **kw,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Lens raw jax_eb C_ℓ (ΔT/T) including CAMB ctnorm on EE/TE."""
    ell = jnp.asarray(ell, dtype=jnp.float64)
    ct = (ell * ell - 1.0) * (ell + 2.0) * ell
    fac = ell * (ell + 1.0) / (2.0 * jnp.pi) * (t_cmb_muk**2)
    dl_tt = jnp.asarray(cl_tt) * fac
    dl_ee = jnp.asarray(cl_ee) * fac * ct
    dl_te = jnp.asarray(cl_te) * fac * jnp.sqrt(jnp.maximum(ct, 0.0))
    cpp = jnp.asarray(cl_pp) * (ell**2) * ((ell + 1.0) ** 2) / (2.0 * jnp.pi)
    return lens_cmb_cls(dl_tt, dl_ee, dl_te, cpp, **kw)


__all__ = [
    "extrapolate_lensing_inputs",
    "geometry_runtime",
    "extrapolate_lensing_inputs_jax",
    "lens_cmb_cls",
    "lens_cmb_cls_from_raw",
]
