#!/usr/bin/env python3
# This file is part of GPUCosmo. Copyright (C) 2025-2026 Leonel Medina-Varela.
# SPDX-License-Identifier: GPL-3.0-or-later
# Derived in part from the following, translated to JAX and modified for GPUCosmo:
# - CCL (https://github.com/LSSTDESC/CCL), Copyright (c) 2016-2018, the LSSTDESC CCL
#   contributors.
#   Licence: LICENSES/CCL.txt.
# See NOTICE.md.
"""JAX port of CCL 3.2.1 ``ccl_halofit.c`` (Takahashi et al. 2012 Halofit).

Line-faithful to ``ccl_apply_halofit`` / ``ccl_halofit_struct_new`` /
``ccl_halofit_power``:

- The Halofit scales are solved at every scale-factor node of the linear
  ``Pk2D`` (``fka->yarr``) and the non-linear spectrum is tabulated on the
  linear spline nodes ``(a_arr, lk_arr)``; the result is a log-bicubic
  ``Pk2D`` with ``extrap_order_lok=1``, ``extrap_order_hik=2`` and
  ``D(a)^2`` growth extrapolation below ``a_min`` (as ``ccl_f2d_t_new(...,
  1, 2, ccl_f2d_cclgrowth, 1, 0, 2, ccl_f2d_3)``).
- sigma^2(R) = int_{lkmin}^{max(lkmax, ln(30/R))} dlnk Delta^2_lin(k)
  exp(-k^2 R^2) and its first two R-derivatives are integrated over the
  *same* linear P(k) evaluation CCL uses (``ccl_f2d_t_eval``: bicubic spline
  inside the table, k-extrapolation above ``lkmax``). CCL uses GSL cquad;
  here each spline interval is integrated with fixed Gauss-Legendre nodes
  (the integrand is analytic inside an interval), and the extrapolated tail
  with composite Gauss-Legendre panels placed in units of kR. Both converge
  to ~1e-13 relative.
- sigma^2(R_sigma) = 1 is solved in ln R (CCL: Brent on [1e-64, 1e16] Mpc)
  with a coarse bracket scan followed by Newton iterations to machine
  precision. The gradient uses the implicit-function theorem (one Newton step
  on the stop-gradient root), so the result is differentiable w.r.t. the
  linear P(k) table and all background parameters.
- n_eff and C are CCL's expressions in terms of sigma^2 derivatives
  (Takahashi eqs. A4-A5), Bird et al. (2012) / Takahashi massive-neutrino
  corrections, Smith et al. (2003) f1/f2/f3 interpolation.
- Dark energy: CCL 3.2.1 *computes* w0_eff(a) (distance-to-drag matching,
  1601.07230) and its Omega_m/Omega_de for w0-wa cosmologies but does not
  use it ("not using these to match CLASS better"); ``ccl_halofit_power``
  uses w = w0 and the true Omega_m(a), Omega_de(a) of the (w0, wa)
  cosmology. The same is done here, so the w0_eff root-find is skipped
  (it only matters in CCL as a possible failure mode).
- Omega_m(a) / Omega_de(a) are ``ccl_omega_x`` with CCL's analytic
  ``h_over_h0`` (not the E(a) spline), including radiation and the massive
  neutrino phase-space integral (``ccl_neutrinos.c``, same Akima table).
"""

from __future__ import annotations

from functools import lru_cache, partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from gpucosmo.jax.gsl_splines import akima_spline, spline_eval

jax.config.update("jax_enable_x64", True)

# pyccl.physical_constants (CCL 3.2.1)
KBOLTZ = 1.380649e-23
HPLANCK = 6.62607015e-34
CLIGHT = 299792458.0
GNEWT = 6.6743e-11
MPC_TO_METER = 3.085677581491367e22
EV_IN_J = 1.602176634e-19
STBOLTZ = 5.670374419e-08
SOLAR_MASS = 1.988409871e30
RHO_CRITICAL = 277536627203.6856
DELTAM12_SQ = 7.62e-05
DELTAM13_SQ_POS = 0.00255
DELTAM13_SQ_NEG = -0.00243
T_CMB_DEFAULT = 2.7255
T_NCDM_DEFAULT = 0.71611
NEFF_DEFAULT = 3.044

# ccl_neutrinos.h
_NU_MNUT_MIN = 1e-4
_NU_MNUT_MAX = 500.0
_NU_MNUT_N = 1000
NU_CONST = (
    8.0 * np.pi**5 * (KBOLTZ / HPLANCK) ** 3 * KBOLTZ / (15.0 * CLIGHT**3)
    * (8.0 * np.pi * GNEWT)
    / (3.0 * 100.0**2 * 1000.0**2 / MPC_TO_METER**2 * CLIGHT**2)
)

# ln(kR) break points of the Gauss-Legendre panels for the extrapolated tail
# (CCL's upper limit is ln(30/R)).
_TAIL_BREAKS = np.array([-3.0, -1.0, 0.0, 0.6, 1.0, 1.4, 1.8, 2.3, np.log(30.0)])
_TAIL_NODES = 16
# ln R bracket scan (CCL brackets in [1e-64, 1e16] Mpc).
# Newton on the concave ln sigma^2(ln R) converges to <1e-14 in 4 steps from
# this bracket; 6 are used.
_LNR_SCAN = np.linspace(np.log(1e-24), np.log(1e4), 29)
_NEWTON_ITERS = 6


# ---------------------------------------------------------------------------
# Massive neutrinos and background (ccl_neutrinos.c, ccl_core.c, ccl_background.c)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _nu_phasespace_table() -> tuple[np.ndarray, np.ndarray]:
    """Nodes of CCL's ``nu_spline``: normalised phase-space integral vs ln(m/T)."""
    lmnut = np.linspace(np.log(_NU_MNUT_MIN), np.log(_NU_MNUT_MAX), _NU_MNUT_N)
    # int_0^1000 sqrt(x^2 + r^2) x^2 / (e^x + 1) dx; the integrand is below
    # 1e-300 for x > 700, so composite Gauss-Legendre on [0, 200] is exact
    # to double precision.
    xg, wg = np.polynomial.legendre.leggauss(40)
    edges = np.concatenate([np.linspace(0.0, 40.0, 81), np.linspace(40.0, 200.0, 17)[1:]])
    lo, hi = edges[:-1], edges[1:]
    x = (0.5 * (hi - lo)[:, None] * (xg[None, :] + 1.0) + lo[:, None]).ravel()
    w = (0.5 * (hi - lo)[:, None] * wg[None, :]).ravel()
    r = np.exp(lmnut)[:, None]
    base = x * x / (np.exp(x) + 1.0)
    y = np.sum(np.sqrt(x * x + r * r) * base * w, axis=1)
    return lmnut, y / y[0]


@lru_cache(maxsize=1)
def _nu_spline():
    lmnut, y = _nu_phasespace_table()
    return akima_spline(jnp.asarray(lmnut), jnp.asarray(y))


def nu_phasespace_intg(mnuOT):
    """``nu_phasespace_intg``: Akima table with CCL's asymptotic limits."""
    mnuOT = jnp.asarray(mnuOT, dtype=float)
    safe = jnp.clip(mnuOT, _NU_MNUT_MIN, _NU_MNUT_MAX)
    inner = spline_eval(_nu_spline(), jnp.log(safe)) * 7.0 / 8.0
    return jnp.where(
        mnuOT < _NU_MNUT_MIN,
        7.0 / 8.0,
        jnp.where(mnuOT > _NU_MNUT_MAX, 0.2776566337 * mnuOT, inner),
    )


def omeganuh2(a, m_nu, N_nu_mass, T_CMB, T_ncdm):
    """``ccl_Omeganuh2`` (sum over the massive species ``m_nu``)."""
    a = jnp.asarray(a, dtype=float)
    m_nu = jnp.atleast_1d(jnp.asarray(m_nu, dtype=float))
    if m_nu.shape[0] == 0:
        return jnp.zeros_like(a)
    g = (4.0 / 11.0) ** (1.0 / 3.0)
    Tnu = T_CMB * g
    a4 = a**4
    Tnu_eff = Tnu * T_ncdm / g
    prefix_massive = NU_CONST * Tnu_eff**4
    prefix_massless = NU_CONST * Tnu**4
    out = jnp.zeros_like(a)
    for i in range(m_nu.shape[0]):
        mi = m_nu[i]
        mnuOT = mi / (Tnu_eff / a) * (EV_IN_J / KBOLTZ)
        massive = nu_phasespace_intg(mnuOT) * prefix_massive / a4
        # CCL multiplies the massless limit by N_nu_mass (kept verbatim).
        massless = N_nu_mass * prefix_massless * 7.0 / 8.0 / a4
        out = out + jnp.where(mi < 0.00017, massless, massive)
    return out


def neutrino_masses(m_nu, mass_split: str = "normal"):
    """``pyccl.neutrinos._get_neutrino_masses`` (differentiable in the sum)."""
    if isinstance(m_nu, (list, tuple)) or np.ndim(m_nu) > 0:
        return jnp.asarray(m_nu, dtype=float)
    if isinstance(m_nu, (int, float)) and m_nu == 0:
        return jnp.zeros(0)
    split = str(mass_split).lower()
    m = jnp.asarray(m_nu, dtype=float)
    if split in ("sum", "single"):
        return jnp.atleast_1d(m)
    if split == "equal":
        return jnp.full(3, m / 3.0)
    if split not in ("normal", "inverted"):
        raise ValueError(f"Unknown mass_split {mass_split!r}")

    if split == "normal":
        def masses(m1):
            return jnp.stack([m1, jnp.sqrt(m1 * m1 + DELTAM12_SQ),
                              jnp.sqrt(m1 * m1 + DELTAM13_SQ_POS)])
    else:
        # m3 lightest: m1^2 = m3^2 - D13n, m2^2 = m1^2 + D12
        def masses(m3):
            m1 = jnp.sqrt(m3 * m3 - DELTAM13_SQ_NEG)
            return jnp.stack([m1, jnp.sqrt(m1 * m1 + DELTAM12_SQ), m3])

    def f(ml):
        return jnp.sum(masses(ml)) - m

    def step(_, ml):
        return jnp.maximum(ml - f(ml) / jax.grad(f)(ml), 0.0)

    return masses(jax.lax.fori_loop(0, 50, step, jnp.zeros(())))


def background_params(
    *,
    Omega_c,
    Omega_b,
    h,
    Omega_k=0.0,
    Neff=NEFF_DEFAULT,
    m_nu=0.0,
    mass_split: str = "normal",
    w0=-1.0,
    wa=0.0,
    T_CMB=T_CMB_DEFAULT,
    T_ncdm=T_NCDM_DEFAULT,
    Omega_g=None,
) -> dict:
    """Derived CCL parameters (``Cosmology._build_parameters`` /
    ``ccl_parameters_fill_initial``) needed by Halofit; jittable."""
    rho_g = 4.0 * STBOLTZ / CLIGHT**3 * T_CMB**4
    rho_crit = RHO_CRITICAL * SOLAR_MASS / MPC_TO_METER**3 * h**2
    g = (4.0 / 11.0) ** (1.0 / 3.0)
    T_nu = g * T_CMB
    mnu_list = neutrino_masses(m_nu, mass_split)
    try:  # pyccl drops species below the massless limit (concrete inputs only)
        limit = float(T_nu * KBOLTZ / EV_IN_J)
        keep = np.asarray(mnu_list) > limit
        mnu_list = mnu_list[keep]
    except (jax.errors.ConcretizationTypeError, TypeError):
        pass
    N_nu_mass = float(mnu_list.shape[0])
    N_nu_rel = Neff - N_nu_mass * (T_ncdm / g) ** 4
    rho_nu_rel = N_nu_rel * (7.0 / 8.0) * 4.0 * STBOLTZ / CLIGHT**3 * T_nu**4
    Omega_nu_rel = rho_nu_rel / rho_crit
    Omega_nu_mass = omeganuh2(1.0, mnu_list, N_nu_mass, T_CMB, T_ncdm) / h**2
    Omega_m = Omega_b + Omega_c + Omega_nu_mass
    Omega_l = 1.0 - Omega_m - rho_g / rho_crit - Omega_nu_rel - Omega_k
    if Omega_g is None:
        Omega_g = rho_g / rho_crit
    else:
        Omega_l = Omega_l + rho_g / rho_crit - Omega_g
    return {
        "Omega_c": Omega_c, "Omega_b": Omega_b, "Omega_k": Omega_k,
        "Omega_l": Omega_l, "Omega_g": Omega_g, "Omega_nu_rel": Omega_nu_rel,
        "Omega_nu_mass": Omega_nu_mass, "Omega_m": Omega_m,
        "w0": w0, "wa": wa, "h": h, "m_nu": mnu_list,
        "sum_nu_masses": jnp.sum(mnu_list), "N_nu_mass": N_nu_mass,
        "T_CMB": T_CMB, "T_ncdm": T_ncdm,
    }


_BG_KEYS = (
    "Omega_c", "Omega_b", "Omega_k", "Omega_l", "Omega_g", "Omega_nu_rel",
    "Omega_nu_mass", "Omega_m", "w0", "wa", "h", "sum_nu_masses", "N_nu_mass",
    "T_CMB", "T_ncdm",
)


def background_params_from_ccl(cosmo) -> dict:
    """Copy the derived C-level parameters of a ``pyccl.Cosmology``."""
    p = cosmo.cosmo.params
    out = {k: float(getattr(p, k)) for k in _BG_KEYS}
    out["m_nu"] = np.asarray(cosmo["m_nu"], dtype=float).reshape(-1)[: int(p.N_nu_mass)]
    return out


def background_params_from_dict(params: dict) -> dict:
    """Complete a (possibly partial) parameter dict, e.g. ``Cosmology.params``."""
    if all(k in params for k in _BG_KEYS) and "m_nu" in params:
        return params
    Omega_b = params.get("Omega_b", 0.0)
    if "Omega_c" in params:
        Omega_c = params["Omega_c"]
    else:
        Omega_c = params["Omega_m"] - Omega_b
    kw = {k: params[k] for k in ("Omega_k", "Neff", "m_nu", "mass_split", "w0",
                                  "wa", "T_CMB", "T_ncdm", "Omega_g") if k in params}
    bg = background_params(Omega_c=Omega_c, Omega_b=Omega_b, h=params["h"], **kw)
    if "Omega_c" not in params and "Omega_m" in params:
        # Only Omega_m (total, CCL convention) given: attribute the massive
        # neutrinos to it rather than on top of it.
        bg["Omega_c"] = Omega_c - bg["Omega_nu_mass"]
        bg["Omega_m"] = params["Omega_m"]
        bg["Omega_l"] = bg["Omega_l"] + bg["Omega_nu_mass"]
    return bg


def h_over_h0(bg: dict, a):
    """CCL's analytic ``h_over_h0`` (background.c), not the E(a) spline."""
    a = jnp.asarray(a, dtype=float)
    om_nu = omeganuh2(a, bg["m_nu"], bg["N_nu_mass"], bg["T_CMB"], bg["T_ncdm"]) / bg["h"] ** 2
    return jnp.sqrt(
        (bg["Omega_c"] + bg["Omega_b"]
         + bg["Omega_l"] * a ** (-3.0 * (bg["w0"] + bg["wa"])) * jnp.exp(3.0 * bg["wa"] * (a - 1.0))
         + bg["Omega_k"] * a
         + (bg["Omega_g"] + bg["Omega_nu_rel"]) / a
         + om_nu * a**3) / a**3
    )


def omega_m_of_a(bg: dict, a):
    """``ccl_omega_x(cosmo, a, ccl_species_m_label)`` (includes massive nu)."""
    a = jnp.asarray(a, dtype=float)
    hn = h_over_h0(bg, a)
    om_nu = omeganuh2(a, bg["m_nu"], bg["N_nu_mass"], bg["T_CMB"], bg["T_ncdm"])
    return (bg["Omega_c"] + bg["Omega_b"]) / a**3 / hn**2 + om_nu / bg["h"] ** 2 / hn**2


def omega_de_of_a(bg: dict, a):
    """``ccl_omega_x(cosmo, a, ccl_species_l_label)``."""
    a = jnp.asarray(a, dtype=float)
    hn = h_over_h0(bg, a)
    return (bg["Omega_l"] * a ** (-3.0 * (1.0 + bg["w0"] + bg["wa"]))
            * jnp.exp(3.0 * bg["wa"] * (a - 1.0)) / hn**2)


# ---------------------------------------------------------------------------
# sigma^2(R) integrals over the linear Pk2D
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _gl01(n: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


def _default_nodes_per_interval(n_k: int) -> int:
    # ~0.33 max spacing in ln k on CCL/CAMB grids (193 intervals) -> 10 nodes;
    # CCL-dense (1200+ knots, dlnk ~ 0.014) -> 4 nodes. Converged to <1e-12.
    if n_k >= 800:
        return 4
    if n_k >= 400:
        return 6
    return 10


def _hermite_basis(t):
    t2 = t * t
    t3 = t2 * t
    return 2 * t3 - 3 * t2 + 1, t3 - 2 * t2 + t, -2 * t3 + 3 * t2, t3 - t2


def _linear_rows(x, z, zx, n_per: int, extrap_order_hik: int):
    """Linear spline values at the quadrature nodes of every a row.

    ``z``/``zx`` (nk, na) are the table values and natural-spline x
    derivatives (``Bicubic2D.z``/``.zx``). On a row y = a_j the GSL bicubic
    reduces exactly to the cubic Hermite patch in x built from them (u = 0,
    or u = 1 on the last node), so no 2D evaluation is needed. Returns
    (lk_q, w_q, vals (na, nq), tail) where ``tail`` holds the high-k
    extrapolation coefficients of ``ccl_f2d_t_eval`` per row.
    """
    dx = jnp.diff(x)
    t, w = _gl01(n_per)
    h00, h10, h01, h11 = (jnp.asarray(v) for v in _hermite_basis(t))
    vals = (
        h00[None, :, None] * z[:-1, None, :]
        + h10[None, :, None] * dx[:, None, None] * zx[:-1, None, :]
        + h01[None, :, None] * z[1:, None, :]
        + h11[None, :, None] * dx[:, None, None] * zx[1:, None, :]
    )  # (nk-1, m, na)
    lk_q = (x[:-1, None] + dx[:, None] * jnp.asarray(t)[None, :]).reshape(-1)
    w_q = (dx[:, None] * jnp.asarray(w)[None, :]).reshape(-1)
    vals = jnp.moveaxis(vals.reshape(-1, z.shape[1]), 1, 0)  # (na, nq)
    # Extrapolation above lkmax: value, d/dlk and d2/dlk2 of the last
    # Hermite patch at t = 1 (== bicubic_deriv_x / _xx at lk_ev = lkmax).
    h = dx[-1]
    f_end = z[-1]
    d1_end = zx[-1]
    d2_end = (6.0 * z[-2] + 2.0 * h * zx[-2] - 6.0 * z[-1] + 4.0 * h * zx[-1]) / (h * h)
    if extrap_order_hik < 1:
        d1_end = jnp.zeros_like(d1_end)
    if extrap_order_hik < 2:
        d2_end = jnp.zeros_like(d2_end)
    return lk_q, w_q, vals, (f_end, d1_end, d2_end)


def _gauss_weights(k2, lnR, n_mom: int):
    """exp(-k^2R^2) times CCL's 1st / 2nd R-derivative factors."""
    R = jnp.exp(lnR)
    R2 = R * R
    g = jnp.exp(-k2 * R2)
    out = [g]
    if n_mom > 1:
        out.append(g * (-k2 * 2.0 * R))
    if n_mom > 2:
        out.append(g * (-2.0 * k2 + 4.0 * k2 * k2 * R2))
    return out


def _tail_moments(lnR, lkmax, tail, is_log: bool, n_mom: int):
    """Integrals over the extrapolated range [lkmax, max(lkmax, ln(30/R))].

    Composite Gauss-Legendre with panel edges fixed in ln(kR), clipped to
    start at lkmax (zero-length panels contribute nothing).
    """
    f_end, d1, d2 = tail
    tq, wq = _gl01(_TAIL_NODES)
    brk = jnp.maximum(jnp.asarray(_TAIL_BREAKS) - lnR, lkmax)
    brk = jnp.concatenate([jnp.reshape(lkmax, (1,)), brk])
    lo, hi = brk[:-1], brk[1:]
    lk_t = (lo[:, None] + (hi - lo)[:, None] * jnp.asarray(tq)[None, :]).reshape(-1)
    w_t = ((hi - lo)[:, None] * jnp.asarray(wq)[None, :]).reshape(-1)
    dlk = lk_t - lkmax
    f = f_end + d1 * dlk + 0.5 * d2 * dlk * dlk
    p_t = jnp.exp(f) if is_log else f
    dlt_t = p_t * jnp.exp(3.0 * lk_t) / (2.0 * jnp.pi**2) * w_t
    return [jnp.sum(dlt_t * wt) for wt in _gauss_weights(jnp.exp(2.0 * lk_t), lnR, n_mom)]


def _moments(lnR, k2_q, dlt_q, lkmax, tail, is_log: bool, n_mom: int):
    """CCL's gauss_norm / onederiv / twoderiv integrals at one a node."""
    body = [jnp.sum(dlt_q * wq) for wq in _gauss_weights(k2_q, lnR, n_mom)]
    ext = _tail_moments(lnR, lkmax, tail, is_log, n_mom)
    return [b + e for b, e in zip(body, ext)]


def _solve_lnrsigma(k2_q, dlt_q, k2_s, dlt_s, lkmax, tail, is_log: bool):
    """ln R_sigma with sigma^2(R) = 1 at one a node.

    Bracket from a coarse ln R scan (trapezoid on the spline knots
    ``k2_s``/``dlt_s``, exact tail), then safeguarded Newton on
    ln sigma^2(ln R), which is concave and decreasing, to machine precision.
    """

    def lnsig2_coarse(lnR):
        body = jnp.sum(dlt_s * _gauss_weights(k2_s, lnR, 1)[0])
        return jnp.log(body + _tail_moments(lnR, lkmax, tail, is_log, 1)[0])

    def newton_step(lnR):
        s0, s1 = _moments(lnR, k2_q, dlt_q, lkmax, tail, is_log, 2)
        return lnR - jnp.log(s0) / (jnp.exp(lnR) * s1 / s0)

    scan = jnp.asarray(_LNR_SCAN)
    f_scan = jax.lax.stop_gradient(jax.vmap(lnsig2_coarse)(scan))
    i = jnp.clip(jnp.sum(f_scan > 0.0) - 1, 0, scan.shape[0] - 2)
    f0, f1 = f_scan[i], f_scan[i + 1]
    x0 = scan[i] + (scan[i + 1] - scan[i]) * f0 / (f0 - f1)

    def body(_, x):
        return jnp.clip(newton_step(x), scan[0], scan[-1])

    x = jax.lax.fori_loop(0, _NEWTON_ITERS, body, jax.lax.stop_gradient(x0))
    # Implicit-function gradient: one differentiable Newton step at the root.
    return newton_step(jax.lax.stop_gradient(x))


def _scales_from_rows(x, z, zx, *, is_log: bool, extrap_order_hik: int, n_per: int):
    lk_q, w_q, vals, tail = _linear_rows(x, z, zx, n_per, extrap_order_hik)
    p_q = jnp.exp(vals) if is_log else vals
    dlt_q = p_q * (jnp.exp(3.0 * lk_q) / (2.0 * jnp.pi**2) * w_q)[None, :]
    # Knot trapezoid for the bracket scan.
    dxk = jnp.diff(x)
    w_s = 0.5 * (jnp.concatenate([dxk, jnp.zeros(1)]) + jnp.concatenate([jnp.zeros(1), dxk]))
    p_s = jnp.exp(z.T) if is_log else z.T
    dlt_s = jax.lax.stop_gradient(p_s * (jnp.exp(3.0 * x) / (2.0 * jnp.pi**2) * w_s)[None, :])
    lkmax = x[-1]
    k2_q = jnp.exp(2.0 * lk_q)
    k2_s = jnp.exp(2.0 * x)

    def one(dlt, dls, fe, dd1, dd2):
        tl = (fe, dd1, dd2)
        lnR = _solve_lnrsigma(k2_q, dlt, k2_s, dls, lkmax, tl, is_log)
        s0, s1, s2 = _moments(lnR, k2_q, dlt, lkmax, tl, is_log, 3)
        return lnR, s0, s1, s2

    lnR, sigma2, i1, i2 = jax.vmap(one)(dlt_q, dlt_s, *tail)
    rsigma = jnp.exp(lnR)
    # ccl_halofit_struct_new: n_eff and C from the derivative integrals
    # (Takahashi eq. A5 in terms of sigma^2 derivatives).
    n_eff = -rsigma / sigma2 * i1 - 3.0
    dsig = (n_eff + 3.0) / (-rsigma / sigma2)
    C = -1.0 * (
        i2 * rsigma * rsigma / sigma2
        + dsig * rsigma / sigma2
        - dsig * dsig * rsigma * rsigma / sigma2 / sigma2
    )
    return {"rsigma": rsigma, "sigma2": sigma2, "n_eff": n_eff, "C": C}


def halofit_scales(psp, *, nodes_per_interval: int | None = None):
    """R_sigma, sigma^2, n_eff and C at every a node of the linear table.

    ``psp`` is the ``F2D`` of a (non-factorizable) linear ``Pk2D``.
    """
    if psp.fka is None:
        raise NotImplementedError("Halofit needs a 2D (lk, a) linear Pk2D table")
    s = psp.fka
    n_per = nodes_per_interval or _default_nodes_per_interval(int(s.x.shape[0]))
    return _scales_from_rows(
        s.x, s.z, s.zx, is_log=psp.is_log,
        extrap_order_hik=psp.extrap_order_hik, n_per=n_per,
    )


def halofit_power_table(
    bg: dict,
    a,
    lk,
    pk_lin,
    rsigma,
    n_eff,
    C,
):
    """``ccl_halofit_power`` on (a[:, None], lk[None, :]); pk_lin (na, nk)."""
    a = jnp.asarray(a, dtype=float)[:, None]
    k = jnp.exp(jnp.asarray(lk, dtype=float))[None, :]
    rsigma = rsigma[:, None]
    neff = n_eff[:, None]
    C = C[:, None]
    weffa = bg["w0"]
    omegaMz = omega_m_of_a(bg, a)
    omegaDEwz = omega_de_of_a(bg, a)

    ksigma = 1.0 / rsigma
    neff2 = neff * neff
    neff3 = neff2 * neff
    neff4 = neff3 * neff
    delta2_norm = k * k * k / 2.0 / jnp.pi / jnp.pi

    om_nu = bg["sum_nu_masses"] / 93.14 / bg["h"] / bg["h"]
    fnu = om_nu / bg["Omega_m"]

    an = 10.0 ** (1.5222 + 2.8553 * neff + 2.3706 * neff2 + 0.9903 * neff3
                  + 0.2250 * neff4 - 0.6038 * C + 0.1749 * omegaDEwz * (1.0 + weffa))
    bn = 10.0 ** (-0.5642 + 0.5864 * neff + 0.5716 * neff2 - 1.5474 * C
                  + 0.2279 * omegaDEwz * (1.0 + weffa))
    cn = 10.0 ** (0.3698 + 2.0404 * neff + 0.8161 * neff2 + 0.5869 * C)
    gamman = 0.1971 - 0.0843 * neff + 0.8460 * C
    alphan = jnp.abs(6.0835 + 1.3373 * neff - 0.1959 * neff2 - 5.5274 * C)
    betan = (2.0379 - 0.7354 * neff + 0.3157 * neff2 + 1.2490 * neff3
             + 0.3980 * neff4 - 0.1682 * C)
    mun = 0.0
    nun = 10.0 ** (5.2105 + 3.6902 * neff)

    # Smith et al. C17-C18
    one_m = 1.0 - omegaMz
    use_mix = jnp.abs(one_m) > 0.01
    safe_one_m = jnp.where(use_mix, one_m, 1.0)
    f1a = omegaMz ** (-0.0732)
    f2a = omegaMz ** (-0.1423)
    f3a = omegaMz ** (0.0725)
    f1b = omegaMz ** (-0.0307)
    f2b = omegaMz ** (-0.0585)
    f3b = omegaMz ** (0.0743)
    fb_frac = omegaDEwz / safe_one_m
    f1 = jnp.where(use_mix, fb_frac * f1b + (1.0 - fb_frac) * f1a, 1.0)
    f2 = jnp.where(use_mix, fb_frac * f2b + (1.0 - fb_frac) * f2a, 1.0)
    f3 = jnp.where(use_mix, fb_frac * f3b + (1.0 - fb_frac) * f3a, 1.0)

    betan = betan + fnu * (1.081 + 0.395 * neff2)

    y = k / ksigma
    y2 = y * y
    fy = y / 4.0 + y2 / 8.0
    DeltakL = pk_lin * delta2_norm

    kh = k / bg["h"]
    kh2 = kh * kh
    DeltakL_tilde = DeltakL * (1.0 + fnu * (47.48 * kh2) / (1.0 + 1.5 * kh2))
    DeltakQ = (DeltakL * (1.0 + DeltakL_tilde) ** betan
               / (1.0 + alphan * DeltakL_tilde) * jnp.exp(-fy))

    DeltakHprime = an * y ** (3.0 * f1) / (1.0 + bn * y**f2 + (cn * f3 * y) ** (3.0 - gamman))
    DeltakH = DeltakHprime / (1.0 + mun / y + nun / y2)

    Qnu = fnu * (0.977 - 18.015 * (bg["Omega_m"] - 0.3))
    DeltakH = DeltakH * (1.0 + Qnu)

    DeltakNL = DeltakQ + DeltakH
    return DeltakNL / delta2_norm


@partial(jax.jit, static_argnames=("nodes_per_interval",))
def halofit_log_pk_nodes(psp, bg: dict, *, nodes_per_interval: int | None = None):
    """log P_NL at the (a, lk) nodes of the linear table; shape (na, nk)."""
    sc = halofit_scales(psp, nodes_per_interval=nodes_per_interval)
    s = psp.fka
    pk_lin = jnp.exp(s.z.T) if psp.is_log else s.z.T  # (na, nk), exact at nodes
    pk = halofit_power_table(bg, s.y, s.x, pk_lin, sc["rsigma"], sc["n_eff"], sc["C"])
    return jnp.log(pk)


def halofit_log_pk_rows(
    lk,
    a,
    log_pk_lin,
    bg: dict,
    *,
    nodes_per_interval: int | None = None,
    extrap_order_hik: int = 2,
):
    """Halofit on a raw table: ``log_pk_lin`` (na, nk) on nodes (a, lk).

    Equivalent to ``halofit_log_pk_nodes(Pk2D(a_arr=a, lk_arr=lk,
    pk_arr=exp(log_pk_lin)).psp, bg)`` but skips the a-direction spline set
    up (the a-rows are all Halofit needs). ``a`` need not be sorted.
    """
    from gpucosmo.jax.gsl_splines import _cspline_node_derivs

    lk = jnp.asarray(lk, dtype=float)
    a = jnp.asarray(a, dtype=float)
    z = jnp.asarray(log_pk_lin, dtype=float).T  # (nk, na)
    zx = jax.vmap(lambda col: _cspline_node_derivs(lk, col), in_axes=1, out_axes=1)(z)
    n_per = nodes_per_interval or _default_nodes_per_interval(int(lk.shape[0]))
    sc = _scales_from_rows(lk, z, zx, is_log=True,
                           extrap_order_hik=extrap_order_hik, n_per=n_per)
    pk = halofit_power_table(bg, a, lk, jnp.exp(z.T), sc["rsigma"], sc["n_eff"], sc["C"])
    return jnp.log(pk)


def halofit_pk2d(cosmo: Any, pk_linear=None, *, nodes_per_interval: int | None = None):
    """``pyccl.Pk2D.apply_halofit``: non-linear Pk2D from a linear Pk2D.

    ``cosmo`` is a :class:`gpucosmo.jax.ccl.Cosmology` (or a dict of CCL
    parameters); ``pk_linear`` defaults to ``cosmo.get_linear_power()``.
    """
    from gpucosmo.jax.ccl.pk2d import F2D, Pk2D
    from gpucosmo.jax.gsl_splines import bicubic_spline

    if pk_linear is None:
        pk_linear = cosmo.get_linear_power()
    params = cosmo if isinstance(cosmo, dict) else cosmo.params
    bg = background_params_from_dict(params)
    psp = pk_linear.psp
    log_pk = halofit_log_pk_nodes(psp, bg, nodes_per_interval=nodes_per_interval)
    s = psp.fka
    f2d = F2D(
        fka=bicubic_spline(s.x, s.y, log_pk.T),
        is_log=True,
        extrap_order_lok=1,
        extrap_order_hik=2,
        growth_exponent=2,
    )
    return Pk2D(a_arr=None, lk_arr=None, pk_arr=None, _f2d=f2d)


__all__ = [
    "background_params",
    "background_params_from_ccl",
    "background_params_from_dict",
    "h_over_h0",
    "halofit_log_pk_nodes",
    "halofit_log_pk_rows",
    "halofit_pk2d",
    "halofit_power_table",
    "halofit_scales",
    "neutrino_masses",
    "nu_phasespace_intg",
    "omega_de_of_a",
    "omega_m_of_a",
    "omeganuh2",
]
