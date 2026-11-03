#!/usr/bin/env python3
# This file is part of GPUCosmo. Copyright (C) 2025-2026 Leonel Medina-Varela.
# SPDX-License-Identifier: GPL-3.0-or-later
# Derived in part from the following, translated to JAX and modified for GPUCosmo:
# - CAMB (https://github.com/cmbant/CAMB), Copyright 2000-2025 Antony Lewis and Anthony Challinor.
#   Licence: LICENSES/CAMB.txt.
# See NOTICE.md.
"""Hyperspherical Bessel functions Phi_l^nu(K, chi) for curved universes, in JAX.

Port of CAMB 2.0.4's fast evaluator (``fortran/hyperspherical_bessels_olver.f90``,
``phi_olver``/``u_olver``) to elementwise, branch-free JAX: every region is
computed with ``jnp.where`` on safe arguments, so values *and* gradients in
``nu`` and ``chi`` stay finite, and ``l`` may be a static int or an int array.

Conventions (identical to ``camb.mathutils.phi_recurs`` / ``phi_olver``)
------------------------------------------------------------------------
``chi`` is the comoving distance in units of the curvature radius, ``nu`` the
dimensionless wavenumber (``k * R_curv``); ``S_K(chi) = sinh, chi, sin`` for
``K = -1, 0, +1``. ``Phi -> j_l(nu chi)`` for ``chi -> 0`` (and exactly for
``K = 0``). For ``K = +1``, ``nu`` must be an integer ``> l`` (``>= 3``); ``chi``
is folded into ``[0, pi/2]`` with the parity signs of ``normalize_chi``.

The reduced function ``u = S_K Phi`` obeys ``u'' = (l(l+1)/S_K^2 - nu^2) u``;
its flat analogue ``z j_l(nu z)`` obeys the same equation with ``S_K -> z``.

Regions (``l >= 3``, ``K = +-1``)
---------------------------------
* Olver/Liouville-Green map (``compute_olver_z_amp``): ``u = amp z j_l(nu z)``
  with ``z(chi)`` matching the curved Liouville-Green action from the turning
  point (closed-form ``qintegral_exact`` + ``invert_flat_action``), a quadratic
  expansion inside ``|chi - chi_t| <= 1e-4 max(chi_t, z_t)``, and
  ``amp = (dz/dchi)^(-1/2)``.
* Near-flat small-chi series (``compute_olver_z_amp_smallchi``) under CAMB's
  ``use_smallchi_map`` gate.
* CAMB's fallback corners (``OLVER_GATE_CLOSED_EPS``, ``OLVER_GATE_OPEN_EPS`` with
  ``open_alpha_cut``), where the leading map is off by up to 3e-2 of the peak.
  CAMB uses a second-order Airy patch (``l >= 100/150``), the open small-nu
  ``K_{i nu}`` form or ``phi_recurs`` there.  Here instead:

  - closed: the exact Gegenbauer form (:func:`closed_gegenbauer_phi`) for
    ``n = nu - l - 1 < GEGENBAUER_N_MAX = 128``.  CAMB's closed gate
    ``chi/(2(nu - l)) > 7e-3`` with ``chi <= pi/2`` only fires for
    ``nu - l < 113``, so this covers the whole closed fallback region exactly.
  - open, ``l <= L_RECURSION_MAX = 200``: l-recursion (:func:`open_recursion_phi`,
    upward / Miller as in ``phi_recurs``) as a static-length ``lax.scan``.
  - open, ``l > 200``: the leading Olver map is kept.  Measured against
    ``phi_recurs`` it stays below 1.5e-4 of the peak for ``chi <= 3`` (CAMB
    calibrated its gate to 1e-4 over the full open range); it degrades beyond
    ``chi ~ 3`` (2.7e-4 at ``l = 1000``, 3e-3 at ``l = 151`` for ``chi <= 6``,
    ``nu/l = 0.03``), outside the cosmological range (``chi <= ~2``).

* ``l <= 2``: exact closed forms (``phi01_exact`` seeds of Abbott & Schaefer
  1986 plus one recurrence step), written smooth through ``K = 0``.

Peak-normalized accuracy vs ``phi_recurs`` (``scripts/validate_hyperspherical_jax.py``):
<= 1.3e-4 everywhere sampled (``l`` 2..13500, ``nu/l`` 0.05..100, open
``chi <= 3``, closed ``chi <= 2 pi``), i.e. CAMB ``phi_olver``'s own level.

``j_l`` and external tables
---------------------------
The default flat ``j_l`` is :func:`spherical_jl` (exact recurrences for
``l <= 8``, Olver's uniform Airy expansion above; elementwise in ``l``, ~40 ns
per point on CPU).  A line-of-sight code holding its own ``j_l`` tables
(``los_jax.eval_bessel_horner``) uses :func:`olver_map` (unit radius) or
:func:`olver_map_phys` (physical ``q``, ``chi``, traced ``K`` of either sign) to get
``Phi = w j_l(x_eff)`` and calls :func:`phi` / :func:`phi_phys` only where the
returned ``ok`` mask is False.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

CACHE_EPS = 1.0e-12  # HypersphericalBesselUtils CACHE_EPS

# --- CAMB phi_olver gates (hyperspherical_bessels_olver.f90) ----------------
OLVER_OPEN_ALPHA_FLOOR = 0.095
OLVER_OPEN_ALPHA_JOIN = 0.12
OLVER_OPEN_L_JOIN = 500.0
OLVER_OPEN_ALPHA_LOW_EXP = 0.70
OLVER_OPEN_ALPHA_HIGH_EXP = 0.14
OLVER_GATE_OPEN_EPS = 3.0e-3
OLVER_GATE_CLOSED_EPS = 7.0e-3
SMALLCHI_GATE_ALPHA = 2.5
SMALLCHI_GATE_LOW_ALPHA_L_MIN = 50
SMALLCHI_GATE_METRIC = 5.0e-2

_LOG2 = math.log(2.0)
_SQRT_PI = math.sqrt(math.pi)

# =============================================================================
# Airy functions Ai, Ai' (real argument)
# =============================================================================
# Maclaurin series (DLMF 9.4.1) for -8 <= x <= 5; asymptotic expansions
# (DLMF 9.7.5-9.7.10) outside, written in xi = (2/3)|x|^(3/2) so callers that
# know xi accurately (phase ~ 1e5 rad for j_l at large x) can pass it directly.
_AI0 = 0.355028053887817239260063186004  # Ai(0)
_AIP0 = 0.258819403792806798405183560189  # -Ai'(0)
_AIRY_SERIES_POS = 5.0
_AIRY_SERIES_NEG = 8.0
_N_MAC = 40


def _airy_mac_coeffs():
    a = [1.0]
    b = [1.0]
    for k in range(1, _N_MAC):
        a.append(a[-1] / ((3 * k - 1) * (3 * k)))
        b.append(b[-1] / ((3 * k) * (3 * k + 1)))
    return np.array(a), np.array(b)


_MAC_A, _MAC_B = _airy_mac_coeffs()


def _airy_asym_coeffs(n):
    u = [1.0]
    for k in range(1, n):
        u.append((6 * k - 5) * (6 * k - 3) * (6 * k - 1) / ((2 * k - 1) * 216 * k) * u[-1])
    v = [1.0] + [-(6 * k + 1) / (6 * k - 1) * u[k] for k in range(1, n)]
    return np.array(u), np.array(v)


_ASYM_U, _ASYM_V = _airy_asym_coeffs(14)


def _horner(coeffs, t):
    acc = jnp.zeros_like(t) + coeffs[-1]
    for c in coeffs[-2::-1]:
        acc = acc * t + c
    return acc


def _airy_series(x):
    """Ai(x), Ai'(x) from the Maclaurin series (use for -8 <= x <= 5)."""
    t = x * x * x
    f = _horner(_MAC_A, t)
    g = x * _horner(_MAC_B, t)
    # f' = sum 3k a_k x^(3k-1), g' = sum (3k+1) b_k x^(3k)
    ka = _MAC_A[1:] * 3.0 * np.arange(1, _N_MAC)
    kb = _MAC_B * (3.0 * np.arange(_N_MAC) + 1.0)
    fp = x * x * _horner(ka, t)
    gp = _horner(kb, t)
    return _AI0 * f - _AIP0 * g, _AI0 * fp - _AIP0 * gp


def _airy_asym_pos(xi, x14):
    """Ai, Ai' * exp(xi) for x > 0, xi = (2/3) x^(3/2), x14 = x^(1/4)."""
    r = -1.0 / xi
    s = _horner(_ASYM_U, r)
    t = _horner(_ASYM_V, r)
    ai = s / (2.0 * _SQRT_PI * x14)
    aip = -x14 * t / (2.0 * _SQRT_PI)
    return ai, aip


def _airy_asym_neg(xi, x14):
    """Ai(-X), Ai'(-X) for X > 0, xi = (2/3) X^(3/2), x14 = X^(1/4)."""
    r2 = -1.0 / (xi * xi)
    ue = _horner(_ASYM_U[0::2], r2)
    uo = _horner(_ASYM_U[1::2], r2) / xi
    ve = _horner(_ASYM_V[0::2], r2)
    vo = _horner(_ASYM_V[1::2], r2) / xi
    ph = xi - 0.25 * math.pi
    c, s = jnp.cos(ph), jnp.sin(ph)
    ai = (c * ue + s * uo) / (_SQRT_PI * x14)
    aip = x14 * (s * ve - c * vo) / _SQRT_PI
    return ai, aip


def airy_ai(x):
    """``(Ai(x), Ai'(x))`` for real ``x`` (float64, ~1e-11 absolute)."""
    x = jnp.asarray(x, dtype=jnp.float64)
    pos = x > _AIRY_SERIES_POS
    neg = x < -_AIRY_SERIES_NEG
    xs = jnp.where(pos | neg, 0.0, x)
    ai_s, aip_s = _airy_series(xs)
    ax = jnp.where(pos | neg, jnp.abs(x), 1.0)
    xi = (2.0 / 3.0) * ax * jnp.sqrt(ax)
    x14 = jnp.sqrt(jnp.sqrt(ax))
    ai_p, aip_p = _airy_asym_pos(xi, x14)
    e = jnp.exp(-jnp.where(pos, xi, 0.0))
    ai_n, aip_n = _airy_asym_neg(xi, x14)
    ai = jnp.where(pos, ai_p * e, jnp.where(neg, ai_n, ai_s))
    aip = jnp.where(pos, aip_p * e, jnp.where(neg, aip_n, aip_s))
    return ai, aip


# =============================================================================
# Flat Liouville-Green action and the spherical Bessel j_l(x)
# =============================================================================
# B0(zeta), A1(zeta) of Olver's uniform expansion (DLMF 10.20.10-11) are
# cancellation-limited near the turning point; on |zeta| <= 1 they are replaced
# by degree-16 fits (Chebyshev nodes, 60-digit mpmath; max residual 8e-16 and
# 2e-14).  B0(0) = 0.0179988..., A1(0) = -1/225.
_B0_POLY = np.array([
    1.79988721413560371e-02, 8.88888888889448683e-03, 1.62568716257030944e-03,
    -3.64284865518475884e-04, -3.02060445985275774e-04, -5.84435678154743160e-05,
    1.67698400331108222e-05, 1.30163681020351698e-05, 2.44696868794771939e-06,
    -7.72498955760391360e-07, -5.79485498267110924e-07, -1.07193537107496807e-07,
    3.60167906515622905e-08, 2.64246667095237883e-08, 3.98688675838176476e-09,
    -2.01532289336705096e-09, -7.77762521694088918e-10])
_A1_POLY = np.array([
    -4.44444444444693637e-03, -1.46370746327314041e-03, 7.06417272826255493e-04,
    6.72887595180247152e-04, 1.54002756255806357e-04, -5.76628639322828885e-05,
    -4.98864082612090088e-05, -1.04305752077203305e-05, 3.87463702061069568e-06,
    3.15228962924451199e-06, 6.30091921220627657e-07, -2.38974178698347527e-07,
    -1.85924796787994215e-07, -2.90264661272433582e-08, 1.63502379471727361e-08,
    6.49447618116177024e-09, 2.26805459403604665e-11])
_U1 = 5.0 / 72.0
_U2 = 385.0 / 10368.0
_V1 = -7.0 / 5.0 * _U1
_V2 = -13.0 / 11.0 * _U2
_SERIES_S = 2.0e-2  # |1 - z^2|^(1/2) below which q/s^3 uses its Taylor series


def flat_action(z):
    """Flat Liouville-Green action about the turning point ``z = 1``.

    Returns ``(q, zeta, phi4)``: ``q = |int_z^1 sqrt|1/t^2 - 1|| dt|`` (CAMB's
    ``qintegral_exact`` for ``K = 0``: ``atanh(s) - s`` below, ``s - atan(s)``
    above the turning point, ``s = sqrt|1 - z^2|``), Olver's
    ``zeta = +-(3q/2)^(2/3)`` (positive for ``z < 1``) and ``phi4 = 4 zeta/(1 - z^2)``.
    Everything is written through ``r = q / s^3`` so that ``z -> 1`` is smooth;
    ``q`` itself is returned as 0 within ``s < 0.02`` (it is only used in the tails).
    """
    z = jnp.asarray(z, dtype=jnp.float64)
    below = z < 1.0
    w = 1.0 - z * z  # signed, so zeta = w (3r/2)^(2/3) is smooth through z = 1
    s2 = jnp.abs(w)
    small = s2 < _SERIES_S**2
    # series: atanh(s)-s = s^3/3 + s^5/5 + ...;  s-atan(s) = s^3/3 - s^5/5 + ...
    r_ser = 1.0 / 3.0 + w * (1.0 / 5.0 + w * (1.0 / 7.0 + w / 9.0))
    s2_safe = jnp.where(small, 1.0, s2)
    s = jnp.sqrt(s2_safe)
    z_safe = jnp.where(small, 0.5, jnp.maximum(z, 1.0e-300))
    # atanh(s) = log((1 + s)/z) for s = sqrt(1 - z^2) (no cancellation as z -> 0)
    q_below = jnp.log1p(s) - jnp.log(z_safe) - s
    q_above = s - jnp.arctan(s)
    r_full = jnp.where(below, q_below, q_above) / (s2_safe * s)
    r = jnp.where(small, r_ser, r_full)
    zeta_over_s2 = (1.5 * r) ** (2.0 / 3.0)
    zeta = w * zeta_over_s2
    q = jnp.where(small, 0.0, r_full * s2_safe * s)  # only needed away from the turn
    phi4 = 4.0 * zeta_over_s2
    return q, zeta, phi4


def _b0_a1(zeta, z):
    """Olver's uniform-expansion coefficients ``B0(zeta), A1(zeta)`` at ``z = x/nu``."""
    near = jnp.abs(zeta) <= 1.0
    zn = jnp.where(near, zeta, 0.0)
    b0_n = _horner(_B0_POLY, zn)
    a1_n = _horner(_A1_POLY, zn)
    za = jnp.where(near, 2.0, jnp.abs(zeta))
    # |zeta| > 1 keeps |1 - z^2| >= 0.6 away from the turning point
    p2 = 1.0 / jnp.where(near, 0.25, jnp.abs(1.0 - z * z))
    b0_f, a1_f = _b0_a1_far(za, p2, z < 1.0)
    return jnp.where(near, b0_n, b0_f), jnp.where(near, a1_n, a1_f)


def _b0_a1_far(zeta_abs, p2, below):
    """Closed forms of ``B0, A1`` for ``|zeta| > 1`` (DLMF 10.20.11, continued to ``z > 1``)."""
    p = jnp.sqrt(p2)
    z12 = jnp.sqrt(zeta_abs)
    z32 = zeta_abs * z12
    # below the turning point (zeta > 0), p = (1 - z^2)^(-1/2)
    u1_b = (3.0 * p - 5.0 * p * p2) / 24.0
    u2_b = (81.0 * p2 - 462.0 * p2 * p2 + 385.0 * p2 * p2 * p2) / 1152.0
    b0_b = -u1_b / z12 - 5.0 / (48.0 * zeta_abs**2)
    a1_b = u2_b + 1.5 * _V1 * u1_b / z32 + 2.25 * _V2 / zeta_abs**3
    # above (zeta < 0), P = (z^2 - 1)^(-1/2), Z = -zeta
    w = (3.0 * p + 5.0 * p * p2) / 24.0
    b0_a = w / z12 - 5.0 / (48.0 * zeta_abs**2)
    a1_a = (-(81.0 * p2 + 462.0 * p2 * p2 + 385.0 * p2 * p2 * p2) / 1152.0
            + 1.5 * _V1 * w / z32 - 2.25 * _V2 / zeta_abs**3)
    return jnp.where(below, b0_b, b0_a), jnp.where(below, a1_b, a1_a)


_JL_EXACT_LMAX = 8  # j_l for l <= 8 from trig + upward recurrence / Taylor series


def _jl_low(lf, x):
    """``j_l(x)`` for ``l <= 8`` (selected by ``lf``): upward recurrence from
    ``j_0, j_1`` for ``x >= 1`` (absolute error ~ eps |y_8(1)| ~ 1e-9 at worst) and
    the Taylor series ``x^l/(2l+1)!! sum_k (-x^2/2)^k / (k! prod_i (2l+2i+1))`` below."""
    small = x < 1.0
    xs = jnp.where(small, 1.0, x)
    sx, cx = jnp.sin(xs), jnp.cos(xs)
    jm, jz = sx / xs, (sx / xs - cx) / xs
    up = [jm, jz]
    for n in range(1, _JL_EXACT_LMAX):
        jm, jz = jz, (2.0 * n + 1.0) / xs * jz - jm
        up.append(jz)
    t = -0.5 * x * x
    out = jnp.zeros_like(x)
    for l in range(_JL_EXACT_LMAX + 1):
        dfac = float(np.prod(np.arange(1, 2 * l + 2, 2)))
        coeffs = [1.0]
        for k in range(1, 10):
            coeffs.append(coeffs[-1] / (k * (2 * l + 2 * k + 1)))
        ser = x**l / dfac * _horner(np.array(coeffs), t)
        out = jnp.where(lf == l, jnp.where(small, ser, up[l]), out)
    return out


def spherical_jl(l, x):
    """Spherical Bessel ``j_l(x)``, elementwise in ``l`` (int, ``l >= 0``) and ``x >= 0``.

    ``l <= 8``: ``j_0, j_1`` and upward recurrence (Taylor series for ``x < 1``).
    ``l >= 9``: Olver's uniform asymptotic expansion of ``J_nu(nu z)``, ``nu = l + 1/2``
    (DLMF 10.20.4 with ``A_0 = 1``, ``B_0``, ``A_1``), with ``Ai`` from
    :func:`airy_ai`; the large-``|zeta|`` Airy asymptotics are evaluated from
    ``xi = nu q`` so the phase stays exact to rounding at large ``x``.  Accuracy
    (vs SciPy, peak-normalized): < 1e-9 for ``l <= 8``, 4e-7 at ``l = 10``,
    < 2e-8 for ``l >= 30``.
    """
    l_arr = jnp.asarray(l)
    x = jnp.asarray(x, dtype=jnp.float64)
    lf = l_arr.astype(jnp.float64)
    lf, x = jnp.broadcast_arrays(lf, x)
    low = _jl_low(lf, x)
    nu = jnp.maximum(lf, float(_JL_EXACT_LMAX + 1)) + 0.5
    # z < 1e-3 means j_l < 1e-27 for l >= 9: return 0 there (keeps logs/powers safe)
    valid = x > 1.0e-3 * nu
    z = jnp.where(valid, x / nu, 0.5)
    q, zeta, phi4 = flat_action(z)
    b0, a1 = _b0_a1(zeta, z)
    nu13 = nu ** (1.0 / 3.0)
    xa = nu13 * nu13 * zeta  # Airy argument nu^(2/3) zeta
    ai, aip = airy_ai(xa)
    # far tails: use xi = nu q directly (phase and decay exact to rounding)
    pos = xa > _AIRY_SERIES_POS
    neg = xa < -_AIRY_SERIES_NEG
    xi = nu * jnp.where(pos | neg, q, 1.0)
    axa = jnp.where(pos | neg, jnp.abs(xa), 1.0)
    x14 = jnp.sqrt(jnp.sqrt(axa))
    ai_p, aip_p = _airy_asym_pos(xi, x14)
    e = jnp.exp(-jnp.minimum(jnp.where(pos, xi, 0.0), 745.0))
    ai_n, aip_n = _airy_asym_neg(xi, x14)
    ai = jnp.where(pos, ai_p * e, jnp.where(neg, ai_n, ai))
    aip = jnp.where(pos, aip_p * e, jnp.where(neg, aip_n, aip))
    jnu = jnp.sqrt(jnp.sqrt(phi4)) * (
        ai / nu13 * (1.0 + a1 / (nu * nu)) + aip * b0 / (nu13 * nu13 * nu13 * nu13 * nu13))
    jl_hi = jnp.sqrt(0.5 * math.pi / (nu * z)) * jnu
    jl_hi = jnp.where(valid, jl_hi, 0.0)
    return jnp.where(lf <= _JL_EXACT_LMAX, low, jl_hi)


# =============================================================================
# Curved geometry helpers (HypersphericalBesselUtils)
# =============================================================================
def curved_radius(K: int, chi):
    """``S_K(chi)``: ``sinh``, ``chi``, ``sin`` for ``K = -1, 0, +1``."""
    if K == -1:
        return jnp.sinh(chi)
    if K == 1:
        return jnp.sin(chi)
    return chi


def normalize_chi(l, K: int, nu, chi):
    """CAMB ``normalize_chi``: fold ``chi >= 0`` into the fundamental domain.

    Closed space uses ``Phi(2 pi - chi) = (-1)^l Phi(chi)`` and
    ``Phi(pi - chi) = (-1)^(nu - l - 1) Phi(chi)`` to reach ``[0, pi/2]``.
    Returns ``(achi, symm)``; open/flat return ``(chi, 1)`` (``chi >= 0`` assumed).
    """
    chi = jnp.asarray(chi, dtype=jnp.float64)
    if K != 1:
        return chi, jnp.ones_like(chi)
    l_i = jnp.asarray(l).astype(jnp.int64)
    inu = jnp.round(jnp.asarray(nu, dtype=jnp.float64)).astype(jnp.int64)
    two_pi = 2.0 * math.pi
    achi = jnp.mod(chi, two_pi)
    flip1 = achi > math.pi
    achi = jnp.where(flip1, two_pi - achi, achi)
    symm = jnp.where(flip1 & (jnp.mod(l_i, 2) != 0), -1.0, 1.0)
    flip2 = achi > 0.5 * math.pi
    achi = jnp.where(flip2, math.pi - achi, achi)
    symm = jnp.where(flip2 & (jnp.mod(inu - l_i - 1, 2) != 0), -symm, symm)
    return achi, symm


def turning_point(ell, nu, K: int):
    """Classical turning point ``S_K(chi_t) = ell/nu`` (``ell < nu`` for ``K = 1``)."""
    t = ell / nu
    if K == -1:
        return jnp.arcsinh(t)
    if K == 1:
        return jnp.arcsin(jnp.minimum(t, 1.0))
    return t


def qintegral_exact(K: int, chi, sin_k, alpha):
    """CAMB ``qintegral_exact``: Liouville-Green action between ``chi`` and ``chi_t``.

    ``|int_chi^chi_t sqrt|1/S_K^2 - alpha^2|| dchi'|`` in closed form, with
    ``x = alpha S_K(chi) > 1`` the oscillatory side.  ``sqrt(x^2 + alpha^2)``
    (open) and ``sqrt(alpha^2 - x^2)`` (closed) are written as
    ``alpha cosh(chi)`` and ``alpha cos(chi)``, which keeps ``chi = pi/2`` smooth.
    The untaken side is fed a benign ``x`` so gradients never see ``sqrt(0)``.
    """
    x = alpha * sin_k
    x2 = x * x
    a2 = alpha * alpha
    osc = x > 1.0
    if K == -1:
        r2 = alpha * jnp.cosh(chi)
        xo2 = jnp.where(osc, x2, 2.0)
        r1 = jnp.sqrt(xo2 - 1.0)
        q_osc = alpha * jnp.arcsinh(r1 / jnp.sqrt(1.0 + a2)) - jnp.arctan2(alpha * r1, r2)
        xe = jnp.where(osc, 0.5, x)
        u = jnp.sqrt(1.0 - xe * xe)
        q_ev = (0.5 * alpha * jnp.arctan2(-2.0 * u * r2, 2.0 * xe * xe + a2 - 1.0)
                + jnp.arcsinh(alpha * u / jnp.maximum(xe * jnp.sqrt(1.0 + a2), CACHE_EPS)))
    elif K == 1:
        r2 = alpha * jnp.cos(chi)
        xo2 = jnp.where(osc, x2, 0.5 * (1.0 + a2))
        r1 = jnp.sqrt(xo2 - 1.0)
        q_osc = (0.5 * alpha * jnp.arctan2(2.0 * r1 * r2, a2 + 1.0 - 2.0 * xo2)
                 - jnp.arctan2(alpha * r1, r2))
        xe = jnp.where(osc, 0.5, x)
        u = jnp.sqrt(1.0 - xe * xe)
        sm = jnp.sqrt(jnp.maximum(a2 - 1.0, CACHE_EPS))
        q_ev = (jnp.arcsinh(alpha * u / jnp.maximum(xe * sm, CACHE_EPS))
                - alpha * jnp.arcsinh(u / sm))
    else:
        raise ValueError("qintegral_exact: K must be +-1 (use flat_action for K = 0)")
    return jnp.where(osc, q_osc, q_ev)


# invert_flat_action: CAMB's polynomial fits in s = (3q)^(2/3) near the turn and
# the large-q asymptotic inverses (worst 1.5e-6 evanescent, 8.5e-9 oscillatory).
_Q_CUT_EVAN = 1.72**3 / 3.0
_Q_CUT_OSC = 1.97**3 / 3.0
_EVAN_POLY = np.array([1.0, -5.0000720190143755e-01, 7.5055874727872618e-02,
                       2.1586618823186233e-04, -3.1231653154203167e-04,
                       -1.7210470362266376e-04, 2.2687533176976695e-05])
_OSC_POLY = np.array([1.0, 5.0000018483290443e-01, 7.4998103457285289e-02,
                      -3.5056578112088423e-04, -4.8620948599887724e-04,
                      8.8500265224195657e-05, -8.3097833638549810e-06,
                      3.4828644185221886e-07])


def invert_flat_action(q, z_turn, below):
    """CAMB ``invert_flat_action``: flat ``z`` whose action from ``z_turn`` is ``q``.

    Below the turn ``q = t - tanh t``, ``z/z_turn = sech t``; above it
    ``q = tan th - th``, ``z/z_turn = sec th``.
    """
    q = jnp.maximum(q, 0.0)
    q_cut = jnp.where(below, _Q_CUT_EVAN, _Q_CUT_OSC)
    near = q < q_cut
    qn = jnp.where(near, jnp.maximum(q, 1.0e-300), 1.0)
    s = (3.0 * qn) ** (2.0 / 3.0)
    u_near = jnp.where(below, _horner(_EVAN_POLY, s), _horner(_OSC_POLY, s))
    qf = jnp.where(near, 10.0, q)
    xe = jnp.exp(-(qf + 1.0))
    e = xe * xe
    u_ev = 2.0 * xe * (1.0 + e * (1.0 + 3.0 * e))
    qp = qf + 0.5 * math.pi
    iq2 = 1.0 / (qp * qp)
    u_osc = qp - (1.0 / qp) * (0.5 + iq2 * (7.0 / 24.0 + iq2 * (83.0 / 240.0 + iq2 * (
        6949.0 / 13440.0 + iq2 * (23399.0 / 26880.0 + iq2 * 266317.0 / 168960.0)))))
    u_far = jnp.where(below, u_ev, u_osc)
    return z_turn * jnp.where(near, u_near, u_far)


def open_alpha_cut(l):
    """CAMB ``open_alpha_cut``: broken power law in ``l`` below which (in ``nu/l``)
    the raw open Olver map is not trusted pointwise."""
    ell = jnp.asarray(l).astype(jnp.float64)
    ell = jnp.maximum(ell, 1.0)
    lo = OLVER_OPEN_ALPHA_JOIN * (OLVER_OPEN_L_JOIN / ell) ** OLVER_OPEN_ALPHA_LOW_EXP
    hi = OLVER_OPEN_ALPHA_JOIN * (OLVER_OPEN_L_JOIN / ell) ** OLVER_OPEN_ALPHA_HIGH_EXP
    return jnp.maximum(OLVER_OPEN_ALPHA_FLOOR, jnp.where(ell < OLVER_OPEN_L_JOIN, lo, hi))


def use_smallchi_map(l, nu, achi):
    """CAMB ``use_smallchi_map``: pointwise near-flat gate (``alpha = nu/l``)."""
    lf = jnp.asarray(l).astype(jnp.float64)
    alpha_gate = nu / lf
    return (((alpha_gate > SMALLCHI_GATE_ALPHA)
             | ((lf >= SMALLCHI_GATE_LOW_ALPHA_L_MIN) & (alpha_gate > 1.0)))
            & (lf * lf * achi**7 / nu < SMALLCHI_GATE_METRIC))


def camb_fallback_region(l, K: int, nu, achi):
    """Points where CAMB's ``olver_reduced`` leaves the leading Olver map
    (``OLVER_GATE_CLOSED_EPS`` / ``OLVER_GATE_OPEN_EPS`` with ``open_alpha_cut``),
    excluding the small-chi map region and ``l <= 2``."""
    lf = jnp.asarray(l).astype(jnp.float64)
    small = use_smallchi_map(l, nu, achi)
    if K == 1:
        fb = achi / (2.0 * jnp.maximum(nu - lf, 1.0)) > OLVER_GATE_CLOSED_EPS
    elif K == -1:
        fb = (achi / (2.0 * jnp.maximum(nu, 1e-300)) > OLVER_GATE_OPEN_EPS) & (nu / lf < open_alpha_cut(l))
    else:
        return jnp.zeros(jnp.shape(achi), bool)
    return fb & ~small & (lf > 2.5)


def olver_z_amp_smallchi(l, K: int, nu, chi):
    """CAMB ``compute_olver_z_amp_smallchi``: ``z = chi F``, ``amp = D^(-1/2)``.

    Perturbative solution of ``(dz/dchi)^2 (alpha^2 - 1/z^2) = alpha^2 - 1/S_K^2``
    through ``O((K/alpha^2)^3)`` with ``h = K l(l+1)/nu^2``, ``a = K chi^2``.
    ``amp = 0`` where ``D <= 0`` (never inside the gate).
    """
    lf = jnp.asarray(l).astype(jnp.float64)
    h = K * lf * (lf + 1.0) / (nu * nu)
    a = K * chi * chi
    F = 1.0 - h * (1.0 / 6.0 + (4.0 * a + 13.0 * h) / 360.0
                   + (48.0 * a * a + 148.0 * h * a + 737.0 * h * h) / 45360.0)
    D = 1.0 - h * (1.0 / 6.0 + (12.0 * a + 13.0 * h) / 360.0
                   + (240.0 * a * a + 444.0 * h * a + 737.0 * h * h) / 45360.0)
    amp = jnp.where(D > 0.0, 1.0 / jnp.sqrt(jnp.where(D > 0.0, D, 1.0)), 0.0)
    return chi * F, amp


def near_flat_shift(l, K: int, nu):
    """CAMB ``DoNearFlatIntegration``: ``(nu_eff, amp)`` with ``nu_eff = nu F0``,
    ``F0 = 1 - h/6 - h^2 (13/360 + 737 h/45360)``, ``h = K l(l+1)/nu^2``, ``amp = sqrt(F0)``.

    Then ``Phi ~ (chi / S_K(chi)) * amp * j_l(nu_eff chi)`` (the chi-independent
    part of the small-chi map; CAMB gates it with ``UseShiftedNuNearFlatIntegration``).
    """
    lf = jnp.asarray(l).astype(jnp.float64)
    h = K * lf * (lf + 1.0) / (nu * nu)
    F0 = 1.0 - h / 6.0 - h * h * (13.0 / 360.0 + 737.0 / 45360.0 * h)
    return nu * F0, jnp.sqrt(F0)


def olver_z_amp(l, K: int, nu, achi, sin_k):
    """CAMB ``compute_olver_z_amp``: leading Olver map ``chi -> z`` and ``amp = (dz/dchi)^(-1/2)``.

    ``z`` matches the curved Liouville-Green action from the turning point
    (``ell = sqrt(l(l+1))``, ``alpha = nu/ell``); inside
    ``|chi - chi_t| <= 1e-4 max(chi_t, z_t)`` the map is expanded to second order,
    ``z = z_t + eps (c1 + c2 eps)``, ``c1^3 = cos_K(chi_t)``.
    """
    lf = jnp.asarray(l).astype(jnp.float64)
    ell = jnp.sqrt(lf * (lf + 1.0))
    alpha = nu / ell
    turn_z = ell / nu
    turn_chi = turning_point(ell, nu, K)
    tiny = achi <= jnp.minimum(1.0e-6, 1.0e-4 * jnp.maximum(turn_chi, 1.0))
    win = 1.0e-4 * jnp.maximum(turn_chi, turn_z)
    near = (jnp.abs(achi - turn_chi) <= win) & ~tiny
    # near-turn expansion
    turn_scale = jnp.sqrt(1.0 - K * turn_z * turn_z)
    c1 = turn_scale ** (1.0 / 3.0)
    c2 = (3.0 * alpha**2 * (c1**4 - turn_scale**2) - K) / (10.0 * alpha * c1**2)
    eps = achi - turn_chi
    z_near = turn_z + eps * (c1 + c2 * eps)
    amp_near = 1.0 / jnp.sqrt(c1 + 2.0 * c2 * eps)
    # general branch on a safe chi (outside the window, away from 0)
    chi_g = jnp.where(near, turn_chi + 2.0 * win, jnp.where(tiny, 0.5 * turn_chi, achi))
    sin_g = jnp.where(near | tiny, curved_radius(K, chi_g), sin_k)
    action = qintegral_exact(K, chi_g, sin_g, alpha)
    z_g = invert_flat_action(action, turn_z, chi_g < turn_chi)
    # analytic_amplitude: |(alpha^2 - 1/z^2) / (alpha^2 - 1/S^2)|^(1/4)
    a2 = alpha * alpha
    z_gs = jnp.maximum(z_g, 1.0e-150)
    flat_term = a2 - 1.0 / (z_gs * z_gs)
    curved_term = a2 - 1.0 / (sin_g * sin_g)
    amp_turn = turn_scale ** (-1.0 / 6.0)
    degenerate = (jnp.abs(flat_term) + jnp.abs(curved_term)
                  <= 100.0 * CACHE_EPS * jnp.maximum(1.0, a2)) | (z_g <= 1.0e-8)
    ratio = jnp.abs(flat_term / jnp.where(degenerate, 1.0, curved_term))
    amp_g = jnp.where(degenerate, jnp.where(z_g <= 1.0e-8, 1.0, amp_turn),
                      jnp.sqrt(jnp.sqrt(jnp.where(degenerate, 1.0, ratio))))
    z = jnp.where(tiny, achi, jnp.where(near, z_near, z_g))
    amp = jnp.where(tiny, 1.0, jnp.where(near, amp_near, amp_g))
    return z, amp


def olver_map(l, K: int, nu, chi):
    """Olver/near-flat map of :func:`phi_olver`'s main path, for external ``j_l`` lookups.

    Returns ``(z, amp, symm, achi, sin_k)`` with
    ``S_K(achi) Phi_l^nu(K, chi) ~= symm * amp * z * j_l(nu z)`` (``sin_k = S_K(achi)``),
    using CAMB's near-flat small-chi map under ``use_smallchi_map`` and the full
    Olver map otherwise.  Good to ~1e-4 of the peak for ``l >= 3`` outside
    :func:`camb_fallback_region`; inside it the error reaches 3e-2 (closed
    ``nu = l + 1``) / 2e-2 (open ``l = 3``, ``nu/l ~ 0.2``), except open ``l > 200``
    (see the module docstring) -- use :func:`phi` there.
    """
    nu = jnp.asarray(nu, dtype=jnp.float64)
    achi, symm = normalize_chi(l, K, nu, chi)
    sin_k = curved_radius(K, achi)
    if K == 0:
        return achi, jnp.ones_like(achi), symm, achi, sin_k
    small = use_smallchi_map(l, nu, achi)
    z_s, amp_s = olver_z_amp_smallchi(l, K, nu, achi)
    z_o, amp_o = olver_z_amp(l, K, nu, achi, sin_k)
    z = jnp.where(small, z_s, z_o)
    amp = jnp.where(small, amp_s, amp_o)
    amp = jnp.where(achi <= CACHE_EPS, 0.0, amp)
    return z, amp, symm, achi, sin_k


# =============================================================================
# Exact low-l forms, closed-space Gegenbauer form, open l-recursion
# =============================================================================
def _chi_over_s_and_chi_cot(a):
    """``r = chi/S_K(chi)`` and ``cm = (chi cot_K(chi) - 1)/a`` as smooth functions of ``a = K chi^2``.

    Works for either sign of ``a`` (closed ``a > 0``, open ``a < 0``) and is
    analytic through ``a = 0`` (Taylor series for ``|a| < 1e-3``).
    """
    small = jnp.abs(a) < 1.0e-3
    r_s = 1.0 + a * (1.0 / 6.0 + a * (7.0 / 360.0 + a * (31.0 / 15120.0 + a * 127.0 / 604800.0)))
    cm_s = -(1.0 / 3.0 + a * (1.0 / 45.0 + a * (2.0 / 945.0 + a / 4725.0)))
    aa = jnp.where(small, 1.0, a)
    closed = aa > 0.0
    tc = jnp.minimum(jnp.sqrt(jnp.where(closed, aa, 1.0)), 3.0)  # callers fold chi <= pi/2
    to = jnp.sqrt(jnp.where(closed, 1.0, -aa))
    r_c = tc / jnp.sin(tc)
    r_o = to / jnp.sinh(to)
    cm_c = (tc * jnp.cos(tc) / jnp.sin(tc) - 1.0) / aa
    cm_o = (to / jnp.tanh(to) - 1.0) / aa
    r = jnp.where(small, r_s, jnp.where(closed, r_c, r_o))
    cm = jnp.where(small, cm_s, jnp.where(closed, cm_c, cm_o))
    return r, cm


def phi_l012(l, x, kc, k2):
    """Exact ``Phi_l`` for ``l = 0, 1, 2`` (CAMB ``phi01_exact`` + one recurrence step).

    Sign-agnostic inputs, smooth through ``K = 0``: ``x = nu chi``,
    ``kc = K chi / nu`` and ``k2 = K / nu^2`` (physically ``x = q chi``,
    ``kc = K chi / q``, ``k2 = K / q^2``); then ``a = K chi^2 = kc x`` and
    ``K j^2/nu^2 = j^2 k2``.  With ``r = chi/S_K``, ``c = chi cot_K = 1 + a cm``
    and ``g_j = sqrt(1 - j^2 k2)`` (Abbott & Schaefer 1986)::

        Phi_0 = sinc(x) r
        Phi_1 = (j_1(x) + sinc(x) cm kc) r / g_1
        Phi_2 = (3 c Phi_1 / x - g_1 Phi_0) / g_2     (x^2 g_1 g_2/15 for x < 1e-4)
    """
    lf = jnp.asarray(l).astype(jnp.float64)
    a = kc * x
    r, cm = _chi_over_s_and_chi_cot(a)
    c = 1.0 + a * cm
    g1 = jnp.sqrt(jnp.maximum(1.0 - k2, 1.0e-300))
    g2 = jnp.sqrt(jnp.maximum(1.0 - 4.0 * k2, 1.0e-300))
    small = x < 1.0e-2
    xs = jnp.where(small, 1.0, x)
    t = x * x
    sinc = jnp.where(small, 1.0 - t / 6.0 * (1.0 - t / 20.0 * (1.0 - t / 42.0)), jnp.sin(xs) / xs)
    j1 = jnp.where(small, x / 3.0 * (1.0 - t / 10.0 * (1.0 - t / 28.0)),
                   (jnp.sin(xs) / xs - jnp.cos(xs)) / xs)
    phi0 = sinc * r
    phi1 = (j1 + sinc * cm * kc) * r / g1
    # tiny x: leading Phi_2 = chi^2 sqrt((nu^2 - K)(nu^2 - 4K))/15 = x^2 g1 g2/15
    tiny = x < 1.0e-4
    phi2 = jnp.where(tiny, t / 15.0 * g1 * g2,
                     (3.0 * c * phi1 / jnp.where(tiny, 1.0, x) - g1 * phi0) / g2)
    return jnp.where(lf < 0.5, phi0, jnp.where(lf < 1.5, phi1, phi2))


GEGENBAUER_N_MAX = 128  # closed: exact Gegenbauer form for n = nu - l - 1 < this
L_RECURSION_MAX = 200  # open: l-recursion in CAMB's fallback corners for l <= this
_MILLER_MARGIN = 80
_BIG = 1.0e100


def closed_gegenbauer_phi(l, nu, achi, n_max: int = GEGENBAUER_N_MAX):
    """Exact closed-space ``Phi_l^nu`` for ``n = nu - l - 1 < n_max`` (``achi`` folded).

    ``Phi = sin^l(chi) C_n^(l+1)(cos chi) / C_n^(l+1)(1) * u0`` with the origin
    coefficient ``u0 = sqrt(prod_{j<=l} (nu^2 - j^2)) / (2l+1)!!`` (CAMB
    ``log_origin_u0_fast``); the normalized Gegenbauer polynomial obeys
    ``P_k = (2(k+l) x P_{k-1} - (k-1) P_{k-2}) / (k + 2l + 1)``, ``|P_k| <= 1``.
    This is the finite closed spectrum of ``phi_closed_gegenbauer_start``
    (Tram 2017); it replaces CAMB's recursion/Airy fallbacks for small ``nu - l``,
    where the leading Olver map is worst (3e-2 of the peak at ``nu = l + 1``).
    Elements with ``n >= n_max`` return garbage-but-finite values.
    """
    lf = jnp.asarray(l).astype(jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    n = jnp.round(nu) - lf - 1.0
    x = jnp.cos(achi)
    alpha = lf + 1.0

    def step(carry, k):
        pm, pz = carry
        pn = (2.0 * (k + alpha - 1.0) * x * pz - (k - 1.0) * pm) / (k + 2.0 * alpha - 1.0)
        act = k <= n
        return (jnp.where(act, pz, pm), jnp.where(act, pn, pz)), None

    p0 = jnp.ones_like(x)
    p1 = x
    if n_max > 2:
        (pm, pz), _ = jax.lax.scan(step, (p0, p1), jnp.arange(2.0, float(n_max)))
    else:
        pm, pz = p0, p1
    pn = jnp.where(n < 0.5, p0, pz)
    nu_s = jnp.maximum(nu, lf + 1.0)
    log_u0 = (0.5 * (jax.scipy.special.gammaln(nu_s + lf + 1.0)
                     - jax.scipy.special.gammaln(nu_s - lf) - jnp.log(nu_s))
              - ((lf + 1.0) * _LOG2 + jax.scipy.special.gammaln(lf + 1.5) - 0.5 * math.log(math.pi)))
    pos = achi > 1.0e-150
    log_s = jnp.log(jnp.sin(jnp.where(pos, achi, 1.0)))
    val = pn * jnp.exp(jnp.minimum(log_u0 + lf * log_s, 700.0))
    return jnp.where(pos, val, jnp.where(lf < 0.5, 1.0, 0.0))


def open_recursion_phi(l, nu, chi, l_max: int):
    """Open-space ``Phi_l^nu(chi)`` by l-recursion (CAMB ``phi_recurs``, ``K = -1``).

    ``b_j Phi_j = (2j-1) coth(chi) Phi_{j-1} - b_{j-1} Phi_{j-2}``, ``b_j = sqrt(nu^2 + j^2)``,
    seeded with the exact ``Phi_0, Phi_1``.  Upward where ``nu sinh(chi) >= l (1 - 5e-3)``
    (oscillatory; plus CAMB's low-beta extension), otherwise Miller backward from
    ``l + 80`` normalized to ``Phi_0``/``Phi_1`` (instead of CAMB's continued
    fraction).  Both passes are ``lax.scan`` over a static ``l_max`` (+80), so
    every element costs ``~2 l_max + 80`` recurrence steps; ``l`` may vary per
    element in ``[3, l_max]``.
    """
    lf = jnp.asarray(l).astype(jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    chi = jnp.maximum(jnp.asarray(chi, dtype=jnp.float64), 1.0e-8)
    lf, nu, chi = jnp.broadcast_arrays(lf, nu, chi)
    cth = 1.0 / jnp.tanh(chi)
    nu2 = nu * nu
    phi0 = phi_l012(0, nu * chi, -chi / nu, -1.0 / nu2)
    phi1 = phi_l012(1, nu * chi, -chi / nu, -1.0 / nu2)
    ratio = nu * jnp.sinh(chi) / lf
    use_up = (ratio >= 1.0 - 5.0e-3) | ((nu <= 2.0e-3 * lf) & (ratio >= 0.8))

    def up(carry, j):
        pm, pz, rec = carry
        pn = ((2.0 * j - 1.0) * cth * pz - jnp.sqrt(nu2 + (j - 1.0) ** 2) * pm) / jnp.sqrt(nu2 + j * j)
        act = j <= lf
        rec = jnp.where(j == lf, pn, rec)
        return (jnp.where(act, pz, pm), jnp.where(act, pn, pz), rec), None

    zero = jnp.zeros_like(chi)
    s0 = jnp.where(use_up, phi0, 0.0)
    s1 = jnp.where(use_up, phi1, 0.0)
    (_, _, ph_up), _ = jax.lax.scan(up, (s0, s1, zero), jnp.arange(2.0, l_max + 1.0))

    start = lf + _MILLER_MARGIN

    def down(carry, j):
        pc, bp, rec, p1d = carry
        live = j <= start
        bj = jnp.sqrt(nu2 + j * j)
        plm1 = ((2.0 * j + 1.0) * cth * pc - bp) / bj
        rec = jnp.where(j == lf, pc, rec)
        p1d = jnp.where(j == 1.0, pc, p1d)
        pc_n = jnp.where(live, plm1, 1.0)
        bp_n = jnp.where(live, bj * pc, 0.0)
        big = jnp.maximum(jnp.abs(pc_n), jnp.abs(bp_n)) > _BIG
        f = jnp.where(big, 1.0 / _BIG, 1.0)
        return (pc_n * f, bp_n * f, rec * f, p1d * f), None

    one = jnp.ones_like(chi)
    (p0d, _, rec, p1d), _ = jax.lax.scan(
        down, (one, zero, zero, zero), jnp.arange(float(l_max + _MILLER_MARGIN), 0.0, -1.0))
    use0 = jnp.abs(phi0) >= jnp.abs(phi1)
    den = jnp.where(use0, p0d, p1d)
    den = jnp.where(jnp.abs(den) > 1.0e-300, den, 1.0)
    ph_down = rec * jnp.where(use0, phi0, phi1) / den
    return jnp.where(use_up, ph_up, ph_down)


# =============================================================================
# Public evaluators: unit curvature radius (K = -1, 0, +1)
# =============================================================================
def _static_int(l):
    return isinstance(l, (int, np.integer))


def _jl_eval(jl, l, x):
    return spherical_jl(l, x) if jl is None else jl(x)


def _recursion_lmax(l, recursion_lmax):
    if _static_int(l):
        return max(int(l), 3)
    return int(min(L_RECURSION_MAX, recursion_lmax or L_RECURSION_MAX))


def phi(l, K: int, nu, chi, *, jl=None, recursion: bool = True, recursion_lmax: int | None = None,
        gegenbauer: bool = True):
    """``Phi_l^nu(K, chi)`` in CAMB's conventions (``camb.mathutils.phi_recurs``).

    Parameters
    ----------
    l : int or int array (broadcasts with ``nu``, ``chi``); ``l >= 0``.
    K : static ``-1``, ``0`` or ``+1``.
    nu, chi : float arrays; ``chi >= 0`` in curvature-radius units.  ``K = +1``
        expects integer ``nu > l``.
    jl : optional callable ``x -> j_l(x)`` (same ``l``/broadcast) replacing the
        default :func:`spherical_jl` on the Olver path (e.g. a Horner table).
    recursion : open ``K = -1``, ``3 <= l <= L_RECURSION_MAX``: evaluate CAMB's
        fallback corners by :func:`open_recursion_phi` (static ``l`` sets the scan
        length to ``l``; an ``l`` array scans to ``recursion_lmax``).
    recursion_lmax : static scan bound for an ``l`` array (default
        ``L_RECURSION_MAX``); pass the largest ``l <= L_RECURSION_MAX`` actually
        present to save work -- elements above it keep the Olver value.
    gegenbauer : closed ``K = +1``: exact :func:`closed_gegenbauer_phi` wherever
        ``nu - l - 1 < GEGENBAUER_N_MAX`` (a 128-step scan for every element).
    """
    nu = jnp.asarray(nu, dtype=jnp.float64)
    chi = jnp.asarray(chi, dtype=jnp.float64)
    if K == 0:
        return _jl_eval(jl, l, nu * jnp.abs(chi))
    lf = jnp.asarray(l).astype(jnp.float64)
    lf, nu, chi = jnp.broadcast_arrays(lf, nu, chi)
    l_b = lf if not _static_int(l) else int(l)
    z, amp, symm, achi, sin_k = olver_map(l_b, K, nu, chi)
    if _static_int(l) and l <= 2:
        return symm * phi_l012(l, nu * achi, K * achi / nu, K / (nu * nu))
    u = amp * z * _jl_eval(jl, l_b, nu * z)
    out = u / jnp.where(sin_k > 0.0, sin_k, 1.0)
    if K == -1 and recursion and (not _static_int(l) or l <= L_RECURSION_MAX):
        l_max = _recursion_lmax(l, recursion_lmax)
        l_rec = jnp.clip(lf, 3.0, float(l_max))
        fb = camb_fallback_region(l_b, K, nu, achi) & (lf <= l_max)
        nu_r = jnp.where(fb, nu, 2.0 * l_rec)
        chi_r = jnp.where(fb, achi, 1.0)
        out = jnp.where(fb, open_recursion_phi(l_rec, nu_r, chi_r, l_max), out)
    if K == 1 and gegenbauer:
        use_g = jnp.round(nu) - lf - 1.0 < GEGENBAUER_N_MAX
        nu_g = jnp.where(use_g, nu, lf + 2.0)
        out = jnp.where(use_g, closed_gegenbauer_phi(lf, nu_g, achi), out)
    if not _static_int(l):
        out = jnp.where(lf <= 2.0, phi_l012(lf, nu * achi, K * achi / nu, K / (nu * nu)), out)
    return symm * out


def u_reduced(l, K: int, nu, chi, **kw):
    """CAMB ``u_olver``: ``S_K(achi) Phi_l^nu(K, chi)`` with ``achi`` the folded ``chi``."""
    achi, _ = normalize_chi(l, K, nu, chi)
    return curved_radius(K, achi) * phi(l, K, nu, chi, **kw)


# =============================================================================
# Physical-variable entry point: traced curvature K [Mpc^-2] of either sign
# =============================================================================
def _phys_branches(l, q, chi, K):
    lf = jnp.asarray(l).astype(jnp.float64)
    q = jnp.asarray(q, dtype=jnp.float64)
    chi = jnp.asarray(chi, dtype=jnp.float64)
    K = jnp.asarray(K, dtype=jnp.float64)
    lf, q, chi, K = jnp.broadcast_arrays(lf, q, chi, K)
    absK = jnp.abs(K)
    # CAMB use_smallchi_map in physical variables (alpha = q R / l, R = |K|^-1/2):
    # alpha > 2.5  <=>  q^2 > 6.25 l^2 |K|;   l^2 chi_u^7 / nu = l^2 chi^7 K^4 / q
    small = (((q * q > SMALLCHI_GATE_ALPHA**2 * lf * lf * absK)
              | ((lf >= SMALLCHI_GATE_LOW_ALPHA_L_MIN) & (q * q > lf * lf * absK)))
             & (lf * lf * chi**7 * absK**4 < SMALLCHI_GATE_METRIC * q))
    # near-flat series: z/chi_u = F(h, a), h = K l(l+1)/q^2, a = K chi^2
    h = K * lf * (lf + 1.0) / (q * q)
    a = K * chi * chi
    F = 1.0 - h * (1.0 / 6.0 + (4.0 * a + 13.0 * h) / 360.0
                   + (48.0 * a * a + 148.0 * h * a + 737.0 * h * h) / 45360.0)
    D = 1.0 - h * (1.0 / 6.0 + (12.0 * a + 13.0 * h) / 360.0
                   + (240.0 * a * a + 444.0 * h * a + 737.0 * h * h) / 45360.0)
    amp_s = jnp.where(D > 0.0, 1.0 / jnp.sqrt(jnp.where(D > 0.0, D, 1.0)), 0.0)
    r_s, _ = _chi_over_s_and_chi_cot(a)
    x_s = q * chi * F
    w_s = amp_s * F * r_s
    # exact unit-radius map on safe arguments (benign values where unused)
    sqk = jnp.sqrt(jnp.where(absK > 0.0, absK, 1.0))
    nu = q / sqk
    chi_u = chi * sqk
    safe_nu = 2.0 * lf + 10.0
    # closed modes need nu > l (CAMB: integer nu >= l + 1); below that Phi := 0
    closed_zero = (K > 0.0) & ~small & ~(nu > lf + 0.5)
    closed = (K > 0.0) & ~small & ~closed_zero
    opened = (K < 0.0) & ~small
    nu_c = jnp.where(closed, nu, safe_nu)
    chi_c = jnp.where(closed, chi_u, 0.5)
    nu_o = jnp.where(opened, nu, safe_nu)
    chi_o = jnp.where(opened, chi_u, 0.5)
    return dict(lf=lf, q=q, chi=chi, K=K, small=small, x_s=x_s, w_s=w_s, closed_zero=closed_zero,
                closed=closed, opened=opened, nu_c=nu_c, chi_c=chi_c, nu_o=nu_o, chi_o=chi_o)


def olver_map_phys(l, q, chi, K):
    """Olver/near-flat map in physical units for a traced curvature ``K`` [Mpc^-2].

    ``q`` [1/Mpc] is CAMB's comoving wavenumber (``nu = q R``, ``R = |K|^-1/2``),
    ``chi`` [Mpc] the comoving radial distance, ``K`` any real (closed ``> 0``,
    open ``< 0``, flat ``0``).  Returns ``(x_eff, w, ok)`` with
    ``Phi_l = w * j_l(x_eff)`` wherever ``ok``:

    * near-flat (CAMB ``use_smallchi_map``, rewritten without ``1/sqrt|K|``):
      ``x_eff = q chi F(h, a)``, ``w = D^(-1/2) F chi_u/S_K(chi_u)`` with
      ``h = K l(l+1)/q^2``, ``a = K chi^2`` -- polynomial in ``K``, so values and
      derivatives are continuous through ``K = 0`` (``x_eff -> q chi``, ``w -> 1``);
    * otherwise the exact Olver map at unit radius, ``x_eff = nu z``,
      ``w = symm amp z / S_K(chi_u)``, closed/open picked by ``jnp.where`` on
      ``sign(K)`` (both computed on safe arguments).

    ``ok`` is False for ``l <= 2`` (exact forms), open fallback corners with
    ``l <= L_RECURSION_MAX`` and closed ``nu - l - 1 < GEGENBAUER_N_MAX``; there
    use :func:`phi_phys`.  For ``K > 0`` the caller should pass ``q R`` integer
    ``> l``; the map itself is continuous in ``nu``, and ``nu = q R <= l + 1/2``
    (no closed mode) returns ``w = 0``.
    """
    b = _phys_branches(l, q, chi, K)
    lf = b["lf"]
    l_b = l if _static_int(l) else lf
    zc, ampc, symc, achc, skc = olver_map(l_b, 1, b["nu_c"], b["chi_c"])
    zo, ampo, symo, acho, sko = olver_map(l_b, -1, b["nu_o"], b["chi_o"])
    x_c = b["nu_c"] * zc
    w_c = symc * ampc * zc / jnp.where(skc > 0.0, skc, 1.0)
    x_o = b["nu_o"] * zo
    w_o = ampo * zo / jnp.where(sko > 0.0, sko, 1.0)
    x_eff = jnp.where(b["small"], b["x_s"], jnp.where(b["closed"], x_c, x_o))
    w = jnp.where(b["small"], b["w_s"], jnp.where(b["closed"], w_c, w_o))
    w = jnp.where(b["closed_zero"], 0.0, w)
    fb_open = b["opened"] & camb_fallback_region(l_b, -1, b["nu_o"], acho) & (lf <= L_RECURSION_MAX)
    geg = (b["K"] > 0.0) & (jnp.round(b["nu_c"]) - lf - 1.0 < GEGENBAUER_N_MAX) & b["closed"]
    ok = (lf > 2.0) & ~fb_open & ~geg
    return x_eff, w, ok


def phi_phys(l, q, chi, K, *, jl=None, recursion: bool = True, recursion_lmax: int | None = None,
             gegenbauer: bool = True):
    """``Phi_l`` in physical units (see :func:`olver_map_phys`) including the exact
    ``l <= 2`` forms, the open l-recursion and the closed Gegenbauer form.

    ``jl``: optional callable ``x -> j_l(x)`` for the main path.  Every branch is
    computed for every element (``jnp.where``), so the recursion / Gegenbauer
    scans cost ``~2 recursion_lmax + 80`` and ``GEGENBAUER_N_MAX`` steps unless
    disabled (static ``l > L_RECURSION_MAX`` skips the recursion at trace time);
    see :func:`phi` for ``recursion_lmax``.  Cheapest use in a line-of-sight
    code: ``olver_map_phys`` everywhere and this function only where ``~ok``.
    """
    b = _phys_branches(l, q, chi, K)
    lf = b["lf"]
    x_eff, w, ok = olver_map_phys(l, q, chi, K)
    l_b = l if _static_int(l) else lf
    out = w * _jl_eval(jl, l_b, x_eff)
    qs = jnp.maximum(b["q"], 1.0e-300)
    x = qs * b["chi"]
    l012 = jnp.minimum(lf, 2.0)
    p012 = phi_l012(l012, x, b["K"] * b["chi"] / qs, b["K"] / (qs * qs))
    # closed chi_u > pi/2: fold first (unit radius, parity signs of normalize_chi)
    sqk = jnp.sqrt(jnp.where(b["K"] > 0.0, b["K"], 1.0))
    fold = (b["K"] > 0.0) & (b["chi"] * sqk > 0.5 * math.pi)
    nu_f = jnp.where(fold, b["q"] / sqk, 10.0)
    achi_f, symm_f = normalize_chi(l012, 1, nu_f, jnp.where(fold, b["chi"] * sqk, 1.0))
    p012_f = symm_f * phi_l012(l012, nu_f * achi_f, achi_f / nu_f, 1.0 / (nu_f * nu_f))
    p012 = jnp.where(fold, p012_f, p012)
    if recursion and (not _static_int(l) or l <= L_RECURSION_MAX):
        l_max = _recursion_lmax(l, recursion_lmax)
        l_rec = jnp.clip(lf, 3.0, float(l_max))
        fb = b["opened"] & ~ok & (lf > 2.0) & (lf <= l_max)
        nu_r = jnp.where(fb, b["nu_o"], 2.0 * l_rec)
        chi_r = jnp.where(fb, b["chi_o"], 1.0)
        out = jnp.where(fb, open_recursion_phi(l_rec, nu_r, chi_r, l_max), out)
    if gegenbauer:
        use_g = b["closed"] & ~ok & (lf > 2.0)
        nu_g = jnp.where(use_g, b["nu_c"], lf + 2.0)
        achi, symm = normalize_chi(lf, 1, nu_g, b["chi_c"])
        out = jnp.where(use_g, symm * closed_gegenbauer_phi(lf, nu_g, achi), out)
    return jnp.where(b["closed_zero"], 0.0, jnp.where(lf <= 2.0, p012, out))
