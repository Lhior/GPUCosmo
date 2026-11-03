#!/usr/bin/env python3
# This file is part of GPUCosmo. Copyright (C) 2025-2026 Leonel Medina-Varela.
# SPDX-License-Identifier: GPL-3.0-or-later
# Derived in part from the following, translated to JAX and modified for GPUCosmo:
# - CAMB (https://github.com/cmbant/CAMB), Copyright 2000-2025 Antony Lewis and Anthony Challinor.
#   Licence: LICENSES/CAMB.txt.
# See NOTICE.md.
"""HMcode 2020 + mead2020_feedback — CAMB ``halofit.f90`` line-faithful port.

Implements imead=3 (baseline), 4 (feedback denominator), 5 (numerator with
``baryonify_wk``), then ``nonlin_ratio *= sqrt(p_num/p_den)``.

Reference: CAMB fortran/halofit.f90
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import sici

from gpucosmo.jax.power import MatterPowerTable

# Sheth–Tormen
_ST_A = 0.21616
_ST_Q = 0.707
_ST_P = 0.3
_PI = np.pi


@dataclass
class HMCosmology:
    om_m: float
    om_c: float
    om_b: float
    om_nu: float = 0.0
    h: float = 0.7
    ns: float = 0.96
    logT_AGN: float = 7.7
    Tcmb: float = 2.7255
    gnorm: float = 1.0
    # Growth tables: a ascending; growth=D/D(0); g_growth=ungrow; agrow=acc_growth
    a_growth: np.ndarray | None = None
    growth: np.ndarray | None = None
    g_growth: np.ndarray | None = None
    agrow: np.ndarray | None = None
    # BAO wiggle table at z=0 (CAMB init_wiggle); filled lazily from P_lin
    log_k_wiggle: np.ndarray | None = None
    pk_wiggle: np.ndarray | None = None

    @property
    def f_nu(self) -> float:
        return self.om_nu / max(self.om_m, 1e-30)

    @property
    def f_b(self) -> float:
        return self.om_b / max(self.om_m, 1e-30)

    @property
    def f_c(self) -> float:
        return self.om_c / max(self.om_m, 1e-30)


def assign_HM_cosmology(
    Omega_m: float,
    Omega_b: float,
    h: float,
    n_s: float,
    *,
    logT_AGN: float = 7.7,
    Omega_nu: float = 0.0,
) -> HMCosmology:
    """CAMB ``assign_HM_cosmology`` + ``fill_growtab`` subset."""
    cosm = HMCosmology(
        om_m=Omega_m,
        om_c=Omega_m - Omega_b - Omega_nu,
        om_b=Omega_b,
        om_nu=Omega_nu,
        h=h,
        ns=n_s,
        logT_AGN=logT_AGN,
    )
    _fill_growtab(cosm)
    return cosm


# CAMB growth-table parameters (halofit.f90)
_AINI_GROWTH_ODE = 1.0e-4
_AFIN_GROWTH_ODE = 1.0
_AMIN_GROWTH_INTERP = 1.0e-3
_AMAX_GROWTH_INTERP = 1.0
_N_GROWTH_INTERP = 64
_ACC_GROWTH_ODE = 1.0e-4


def _hubble2(z: float, cosm: HMCosmology) -> float:
    """CAMB ``Hubble2`` for flat ΛCDM (w=-1, no curvature/radiation)."""
    om_v = 1.0 - cosm.om_m
    return float(cosm.om_m * (1.0 + z) ** 3 + om_v)


def _ah(z: float, cosm: HMCosmology) -> float:
    """CAMB ``AH`` = ä/a for flat ΛCDM."""
    om_v = 1.0 - cosm.om_m
    return float(-0.5 * (cosm.om_m * (1.0 + z) ** 3 + om_v))


def _omega_m_hm(z: float, cosm: HMCosmology) -> float:
    """CAMB ``Omega_m_hm``."""
    return float(cosm.om_m * (1.0 + z) ** 3 / max(_hubble2(z, cosm), 1e-30))


def _growth_fv(d: float, v: float, a: float, cosm: HMCosmology) -> float:
    """CAMB ``fv``: v' in the growth ODE (independent variable a)."""
    z = -1.0 + 1.0 / a
    om_m = _omega_m_hm(z, cosm)
    f1 = 3.0 * om_m * d / (2.0 * a * a)
    f2 = (2.0 + _ah(z, cosm) / _hubble2(z, cosm)) * (v / a)
    return f1 - f2


def _ode_growth_rk4(
    cosm: HMCosmology,
    *,
    aini: float = _AINI_GROWTH_ODE,
    afin: float = _AFIN_GROWTH_ODE,
    acc: float = _ACC_GROWTH_ODE,
) -> tuple[np.ndarray, np.ndarray]:
    """CAMB ``ode_growth`` (imeth=3 RK4) with step doubling to ``acc``."""
    zinit = -1.0 + 1.0 / aini
    f = 1.0 - _omega_m_hm(zinit, cosm)
    dinit = aini ** (1.0 - 3.0 * f / 5.0)
    vinit = (1.0 - 3.0 * f / 5.0) * aini ** (-3.0 * f / 5.0)

    x_prev: np.ndarray | None = None
    t_prev: np.ndarray | None = None
    for j in range(1, 31):
        n = 100 * (2 ** (j - 1)) + 1
        t = np.linspace(aini, afin, n)
        dt = (afin - aini) / (n - 1)
        x = np.zeros(n)
        v = np.zeros(n)
        x[0] = dinit
        v[0] = vinit
        for i in range(n - 1):
            d4, v4, a4 = float(x[i]), float(v[i]), float(t[i])
            kx1 = dt * v4
            kv1 = dt * _growth_fv(d4, v4, a4, cosm)
            kx2 = dt * (v4 + 0.5 * kv1)
            kv2 = dt * _growth_fv(d4 + 0.5 * kx1, v4 + 0.5 * kv1, a4 + 0.5 * dt, cosm)
            kx3 = dt * (v4 + 0.5 * kv2)
            kv3 = dt * _growth_fv(d4 + 0.5 * kx2, v4 + 0.5 * kv2, a4 + 0.5 * dt, cosm)
            kx4 = dt * (v4 + kv3)
            kv4 = dt * _growth_fv(d4 + kx3, v4 + kv3, a4 + dt, cosm)
            x[i + 1] = x[i] + (kx1 + 2.0 * kx2 + 2.0 * kx3 + kx4) / 6.0
            v[i + 1] = v[i] + (kv1 + 2.0 * kv2 + 2.0 * kv3 + kv4) / 6.0

        if j == 1 or x_prev is None or t_prev is None:
            x_prev, t_prev = x, t
            continue

        # Compare odd indices of fine grid to previous coarse solution
        ifail = False
        for k in range(x_prev.size):
            kn = 2 * k
            if kn >= x.size:
                break
            xh, xf = float(x_prev[k]), float(x[kn])
            if xh > acc and xf > acc and abs(xh / xf - 1.0) > acc:
                ifail = True
                break
        if not ifail:
            return t, x
        x_prev, t_prev = x, t
    return t, x


def _fill_growtab(cosm: HMCosmology) -> None:
    """CAMB ``fill_growtab``: growth ODE → g(z=0)=1 table + accumulated G(a)."""
    a_ode, d_ode = _ode_growth_rk4(cosm)
    # gnorm = un-normalised g(a=1); then normalise so growth(a=1)=1
    gnorm = float(np.interp(1.0, a_ode, d_ode))
    d_norm = d_ode / max(gnorm, 1e-30)

    n = _N_GROWTH_INTERP
    a = np.linspace(_AMIN_GROWTH_INTERP, _AMAX_GROWTH_INTERP, n)
    growth = np.interp(a, a_ode, d_norm)

    # ungrow = gnorm * growth  (CAMB ``ungrow``)
    ungrow = gnorm * growth
    # agrow = ∫_0^a g(a')/a' da' with g=ungrow; missing 0→amin rectangle once
    integrand = ungrow / a
    agrow = np.zeros(n)
    agrow[0] = gnorm * growth[0]
    for i in range(1, n):
        agrow[i] = agrow[i - 1] + 0.5 * (integrand[i] + integrand[i - 1]) * (
            a[i] - a[i - 1]
        )

    cosm.a_growth = a
    cosm.growth = growth
    cosm.gnorm = gnorm
    cosm.g_growth = ungrow
    cosm.agrow = agrow


def _grow(z: float, cosm: HMCosmology) -> float:
    """CAMB ``grow``: growth normalised to g(z=0)=1."""
    a = 1.0 / (1.0 + z)
    return float(np.interp(a, cosm.a_growth, cosm.growth))


def _ungrow(z: float, cosm: HMCosmology) -> float:
    a = 1.0 / (1.0 + z)
    return float(np.interp(a, cosm.a_growth, cosm.g_growth))


def _acc_growth(z: float, cosm: HMCosmology) -> float:
    a = 1.0 / (1.0 + z)
    return float(np.interp(a, cosm.a_growth, cosm.agrow))


def _f_mead(x: float, y: float, p0, p1, p2, p3) -> float:
    return p0 + p1 * (1.0 - x) + p2 * (1.0 - x) ** 2 + p3 * (1.0 - y)


def dc_Mead(z: float, cosm: HMCosmology) -> float:
    """CAMB ``dc_Mead`` (Mead 2017)."""
    lg = _ungrow(z, cosm)
    bG = _acc_growth(z, cosm)
    Om_m = _omega_m_hm(z, cosm)
    a = 1.0 / (1.0 + z)
    x, y = lg / a, bG / a
    dc0 = (3.0 / 20.0) * (12.0 * _PI) ** (2.0 / 3.0)
    dc = 1.0
    dc += _f_mead(x, y, -0.0069, -0.0208, 0.0312, 0.0021) * np.log10(Om_m)
    dc += _f_mead(x, y, 0.0001, -0.0647, -0.0417, 0.0646)
    return float(dc * dc0 * (1.0 - 0.041 * cosm.f_nu))


def Dv_Mead(z: float, cosm: HMCosmology) -> float:
    """CAMB ``Dv_Mead`` (Mead 2017)."""
    lg = _ungrow(z, cosm)
    bG = _acc_growth(z, cosm)
    Om_m = _omega_m_hm(z, cosm)
    a = 1.0 / (1.0 + z)
    x, y = lg / a, bG / a
    Dv0 = 18.0 * _PI**2
    Dv = 1.0
    Dv += _f_mead(x, y, -0.79, -10.17, 2.51, 6.51) * np.log10(Om_m)
    Dv += _f_mead(x, y, -1.89, 0.38, 18.8, -15.87) * np.log10(Om_m) ** 2
    return float(Dv * Dv0 * (1.0 + 0.763 * cosm.f_nu))


def _tophat_w(x: np.ndarray) -> np.ndarray:
    """CAMB ``wk_tophat``."""
    return np.where(
        x < 1e-3,
        1.0 - x**2 / 10.0,
        3.0 * (np.sin(x) - x * np.cos(x)) / np.maximum(x**3, 1e-30),
    )


def _tophat_w_deriv(x: np.ndarray) -> np.ndarray:
    """CAMB ``wk_tophat_deriv``."""
    return np.where(
        x < 1e-3,
        -x / 5.0 + x**3 / 70.0,
        (3.0 / np.maximum(x**4, 1e-30))
        * ((x**2 - 3.0) * np.sin(x) + 3.0 * x * np.cos(x)),
    )


def _fine_pk_grid(
    k_h: np.ndarray, pk_h: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Upsample P(k) onto a dense log-k grid; returns (logk_f, k_f, pk_f)."""
    logk = np.log(k_h)
    n_fine = max(int(k_h.size) * 4, 1024)
    logk_f = np.linspace(float(logk[0]), float(logk[-1]), n_fine)
    log_pk = np.log(np.maximum(pk_h, 1e-30))
    pk_f = np.exp(np.interp(logk_f, logk, log_pk))
    return logk_f, np.exp(logk_f), pk_f


def _sigma_r_from_fine(
    logk_f: np.ndarray, k_f: np.ndarray, pk_f: np.ndarray, R: np.ndarray
) -> np.ndarray:
    """Top-hat σ(R) on a precomputed fine log-k grid (vectorized over R)."""
    delta2 = k_f**3 * pk_f / (2.0 * np.pi**2)
    # (n_R, n_k)
    w = _tophat_w(k_f[None, :] * R[:, None])
    integ = delta2[None, :] * w**2
    # trapezoid along k for each R
    out = np.sqrt(np.maximum(np.trapezoid(integ, logk_f, axis=1), 0.0))
    return out


def _sigma_r_from_pk(
    k_h: np.ndarray, pk_h: np.ndarray, R: np.ndarray
) -> np.ndarray:
    """Top-hat σ(R) with dense log-k quadrature."""
    logk_f, k_f, pk_f = _fine_pk_grid(k_h, pk_h)
    return _sigma_r_from_fine(logk_f, k_f, pk_f, np.asarray(R, dtype=float))


def _sigmav_from_fine(
    logk_f: np.ndarray, k_f: np.ndarray, pk_f: np.ndarray, R: float = 0.0
) -> float:
    """CAMB ``sigmaV`` on a precomputed fine grid."""
    delta2 = k_f**3 * pk_f / (2.0 * np.pi**2)
    w = _tophat_w(k_f * R) if R > 0.0 else np.ones_like(k_f)
    integ = delta2 * w**2 / np.maximum(k_f**2, 1e-30) / 3.0
    return float(np.sqrt(max(float(np.trapezoid(integ, logk_f)), 0.0)))


def _sigmav_from_pk(k_h: np.ndarray, pk_h: np.ndarray, R: float = 0.0) -> float:
    """CAMB ``sigmaV``: σ_v² = ∫ Δ²(k) W²(kR) /(3 k²) dlnk."""
    logk_f, k_f, pk_f = _fine_pk_grid(k_h, pk_h)
    return _sigmav_from_fine(logk_f, k_f, pk_f, R)


def _neff_at_r_fine(
    logk_f: np.ndarray,
    k_f: np.ndarray,
    pk_f: np.ndarray,
    R: float,
    sig: float,
) -> float:
    """CAMB ``neff`` at R_nl on a precomputed fine grid."""
    kR = k_f * R
    w = _tophat_w(kR)
    wd = _tophat_w_deriv(kR)
    delta2 = k_f**3 * pk_f / (2.0 * np.pi**2)
    num = float(np.trapezoid(delta2 * w * wd * kR, logk_f))
    return float(-3.0 - 2.0 * num / max(sig**2, 1e-30))


def _neff_at_r(k_h: np.ndarray, pk_h: np.ndarray, R: float, sig: float) -> float:
    """CAMB ``neff`` at R_nl: -3 - 2 ∫ P W W' kR … / σ²."""
    logk_f, k_f, pk_f = _fine_pk_grid(k_h, pk_h)
    return _neff_at_r_fine(logk_f, k_f, pk_f, R, sig)


# CAMB dewiggle numerical parameters
_KMIN_WIGGLE = 5.0e-3
_KMAX_WIGGLE = 5.0
_NK_WIGGLE = 512
_WIGGLE_SIGMA = 0.25
_KNORM_NOWIGGLE = 0.03


def _tk_nw(k: np.ndarray, cosm: HMCosmology) -> np.ndarray:
    """Eisenstein & Hu no-wiggle transfer (CAMB ``Tk_nw``)."""
    wm = cosm.om_m * cosm.h**2
    wb = cosm.om_b * cosm.h**2
    rb = cosm.om_b / max(cosm.om_m, 1e-30)
    h = cosm.h
    s = 44.5 * np.log(9.83 / wm) / np.sqrt(1.0 + 10.0 * wb**0.75)
    alpha = 1.0 - 0.328 * np.log(431.0 * wm) * rb + 0.38 * np.log(22.3 * wm) * rb**2
    Gamma = cosm.om_m * h * (alpha + (1.0 - alpha) / (1.0 + (0.43 * k * s * h) ** 4))
    q = k * (cosm.Tcmb / 2.7) ** 2 / np.maximum(Gamma, 1e-30)
    L = np.log(2.0 * np.e + 1.8 * q)
    C = 14.2 + 731.0 / (1.0 + 62.5 * q)
    return L / (L + C * q**2)


def _smooth_gaussian_logk(logk: np.ndarray, f: np.ndarray, sigma: float) -> np.ndarray:
    """CAMB ``smooth_array_Gaussian`` on log-k (vectorized weights)."""
    n = logk.size
    out = np.empty_like(f)
    nsig = 3.0
    edge = (np.abs(logk - logk[0]) < nsig * sigma) | (
        np.abs(logk - logk[-1]) < nsig * sigma
    )
    # Full pairwise weights: (n, n)
    d2 = (logk[:, None] - logk[None, :]) ** 2
    w = np.exp(-d2 / (2.0 * sigma**2))
    smoothed = (w @ f) / np.sum(w, axis=1)
    out = np.where(edge, f, smoothed)
    return out


def _init_wiggle_from_pk(
    k_h: np.ndarray, pk_h: np.ndarray, cosm: HMCosmology, growz: float
) -> None:
    """CAMB ``init_wiggle`` using P_lin(z)/grow(z)² as the z=0 spectrum."""
    if cosm.pk_wiggle is not None:
        return
    logk = np.linspace(np.log(_KMIN_WIGGLE), np.log(_KMAX_WIGGLE), _NK_WIGGLE)
    k = np.exp(logk)
    # Scale current-z P(k) back to z=0 (scale-independent growth)
    g2 = max(growz, 1e-30) ** 2
    log_pk = np.log(np.maximum(pk_h, 1e-30))
    pk0 = np.exp(np.interp(logk, np.log(k_h), log_pk)) / g2

    # EH no-wiggle broadband. CAMB stores Δ²∝k^{ns+3}T²; we work in P∝k^{ns}T².
    pk_nw = (k**cosm.ns) * _tk_nw(k, cosm) ** 2
    kn = _KNORM_NOWIGGLE
    pk_norm = float(np.exp(np.interp(np.log(kn), logk, np.log(np.maximum(pk0, 1e-30)))))
    pk_nw_norm = float(np.interp(kn, k, pk_nw))
    pk_nw = pk_nw * (pk_norm / max(pk_nw_norm, 1e-30))

    # Smooth Pk/Pk_nw then restore dynamic range
    ratio = pk0 / np.maximum(pk_nw, 1e-30)
    ratio_s = _smooth_gaussian_logk(logk, ratio, _WIGGLE_SIGMA)
    pk_smooth = ratio_s * pk_nw
    cosm.log_k_wiggle = logk
    cosm.pk_wiggle = pk0 - pk_smooth


def _p_dewiggle(
    k_h: np.ndarray,
    plin: np.ndarray,
    z: float,
    sigv: float,
    cosm: HMCosmology,
) -> np.ndarray:
    """CAMB ``p_dewiggle``: linear P with BAO wiggles damped by exp(-(k σ_v)²)."""
    growz = _grow(z, cosm)
    _init_wiggle_from_pk(k_h, plin, cosm, growz)
    logk = np.log(k_h)
    pk_w = np.interp(logk, cosm.log_k_wiggle, cosm.pk_wiggle, left=0.0, right=0.0)
    # Zero outside wiggle table (matches CAMB bounds check)
    lo, hi = cosm.log_k_wiggle[0], cosm.log_k_wiggle[-1]
    pk_w = np.where((logk < lo) | (logk > hi), 0.0, pk_w)
    f = np.exp(-((k_h * sigv) ** 2))
    return plin + (f - 1.0) * pk_w * growz**2


def _nfw_window_vec(k: np.ndarray, rv: np.ndarray, c: np.ndarray) -> np.ndarray:
    """NFW W(k, rv, c) for all masses; returns shape (n_k, n_m) or (n_m,) if k scalar."""
    k = np.atleast_1d(k)
    rs = rv / np.maximum(c, 1e-6)
    # Broadcast: k[:,None] * rs[None,:]
    krs = np.maximum(k[:, None] * rs[None, :], 1e-8)
    c1 = 1.0 + c
    f = np.log(1.0 + c) - c / (1.0 + c)
    si1, ci1 = sici(krs)
    si2, ci2 = sici(c1[None, :] * krs)
    wk = (
        np.sin(krs) * (si2 - si1)
        + np.cos(krs) * (ci2 - ci1)
        - np.sin(c[None, :] * krs) / (c1[None, :] * krs)
    ) / np.maximum(f[None, :], 1e-30)
    wk = np.where(k[:, None] * rs[None, :] < 1e-4, 1.0, wk)
    return wk


def _gnu_st(nu: np.ndarray) -> np.ndarray:
    """Sheth–Tormen f(ν) as in CAMB ``gst`` (∫ f dν = 1)."""
    a = _ST_Q
    p = _ST_P
    A = _ST_A
    return A * (1.0 + (a * nu * nu) ** (-p)) * np.exp(-0.5 * a * nu * nu)


def _m_baryon(z: float, logT_AGN: float) -> float:
    theta = logT_AGN - 7.8
    mb0 = 10.0 ** (13.87 + 1.81 * theta)
    mbz = -0.108 + 0.195 * theta
    return mb0 * 10.0 ** (mbz * z)


def _f_star(z: float, logT_AGN: float, om_b: float, om_m: float) -> float:
    theta = logT_AGN - 7.8
    f0 = 0.0201 - 0.003 * theta
    fz = 0.409 + 0.0224 * theta
    fs = f0 * 10.0 ** (fz * z)
    return min(fs, om_b / max(om_m, 1e-30))


def _halo_mass_fraction(m: np.ndarray, z: float, cosm: HMCosmology) -> np.ndarray:
    mb = _m_baryon(z, cosm.logT_AGN)
    r = (m / mb) ** 2
    fs = _f_star(z, cosm.logT_AGN, cosm.om_b, cosm.om_m)
    return cosm.f_c + (cosm.f_b - fs) * r / (1.0 + r)


def _baryonify_wk(wk: np.ndarray, m: np.ndarray, z: float, cosm: HMCosmology) -> np.ndarray:
    fs = _f_star(z, cosm.logT_AGN, cosm.om_b, cosm.om_m)
    return wk * _halo_mass_fraction(m, z, cosm) + fs


def _As_concentration(z: float, imead: int, cosm: HMCosmology) -> float:
    if imead == 3:
        return 5.196
    if imead == 4:
        return 4.0
    theta = cosm.logT_AGN - 7.8
    B0 = 3.44 - 0.496 * theta
    Bz = -0.0671 - 0.0371 * theta
    return B0 * 10.0 ** (z * Bz)


def _eta_puff(imead: int, sig8z: float) -> float:
    if imead in (4, 5):
        return 0.0
    return 0.1281 * sig8z ** (-0.3644)


def _fdamp(imead: int, sig8z: float) -> float:
    if imead in (4, 5):
        return 0.0
    return float(min(0.2696 * sig8z**0.9403, 0.99))


def _kstar(sig8z: float) -> float:
    return 0.05618 * sig8z ** (-1.013)


def _bullock_concentration(
    m: np.ndarray,
    R: np.ndarray,
    sig: np.ndarray,
    z: float,
    cosm: HMCosmology,
    *,
    imead: int,
    k_h: np.ndarray,
    plin: np.ndarray,
    dc: float,
    fine: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """CAMB ``conc_bull`` + ``zcoll_bull`` (flat ΛCDM Dolag factor ≈ 1)."""
    As = _As_concentration(z, imead, cosm)
    f_bullock = 0.01 ** (1.0 / 3.0)
    if fine is None:
        sigf = _sigma_r_from_pk(k_h, plin, R * f_bullock)
    else:
        logk_f, k_f, pk_f = fine
        sigf = _sigma_r_from_fine(logk_f, k_f, pk_f, R * f_bullock)
    growz = _grow(z, cosm)
    a_tab = cosm.a_growth
    g_tab = cosm.growth
    rhs = dc * growz / np.maximum(sigf, 1e-30)
    # Vectorized growth inversion
    future = rhs > growz
    af = np.interp(rhs, g_tab, a_tab)
    af = np.clip(af, a_tab[0], a_tab[-1])
    zf = np.where(future, z, -1.0 + 1.0 / af)
    c_m = As * (1.0 + zf) / (1.0 + z)
    return np.clip(c_m, 1.0, 100.0)


def _nfw_wk_eta(
    k_h: np.ndarray,
    nu: np.ndarray,
    rv: np.ndarray,
    c_m: np.ndarray,
    eta: float,
) -> np.ndarray:
    """NFW windows with eta puff: shape (n_k, n_m), masses in input order."""
    ke = k_h[:, None] * (nu[None, :] ** eta)
    rs = rv / np.maximum(c_m, 1e-6)
    krs = np.maximum(ke * rs[None, :], 1e-8)
    c1 = 1.0 + c_m
    f = np.log(1.0 + c_m) - c_m / (1.0 + c_m)
    si1, ci1 = sici(krs)
    si2, ci2 = sici(c1[None, :] * krs)
    w = (
        np.sin(krs) * (si2 - si1)
        + np.cos(krs) * (ci2 - ci1)
        - np.sin(c_m[None, :] * krs) / (c1[None, :] * krs)
    ) / np.maximum(f[None, :], 1e-30)
    w = np.where(ke * rs[None, :] < 1e-4, 1.0, w)
    return w


def _halomod_from_shared(
    k_h: np.ndarray,
    plin: np.ndarray,
    z: float,
    cosm: HMCosmology,
    *,
    imead: int,
    sig8z: float,
    m: np.ndarray,
    R: np.ndarray,
    sig: np.ndarray,
    nu: np.ndarray,
    gnu: np.ndarray,
    dc: float,
    Dv: float,
    fine: tuple[np.ndarray, np.ndarray, np.ndarray],
    rho_m: float,
) -> np.ndarray:
    """One imead using shared mass/σ/ν tables."""
    logk_f, k_f, pk_f = fine
    c_m = _bullock_concentration(
        m, R, sig, z, cosm, imead=imead, k_h=k_h, plin=plin, dc=dc, fine=fine
    )
    rv = R / (max(Dv, 1.0) ** (1.0 / 3.0))
    eta = _eta_puff(imead, sig8z)
    fd = _fdamp(imead, sig8z)
    ks = _kstar(sig8z)
    kdamp = 0.05699 * sig8z ** (-1.089)
    ndamp = 2.85

    if float(nu[0]) > 1.0:
        rnl = float(R[0])
    else:
        rnl = float(np.exp(np.interp(0.0, np.log(nu), np.log(R))))
    sig_rnl = float(np.interp(np.log(rnl), np.log(R), sig))
    neff = _neff_at_r_fine(logk_f, k_f, pk_f, rnl, sig_rnl)
    neff = float(np.clip(neff, cosm.ns - 4.0, cosm.ns))
    alpha = 1.875 * (1.603**neff) if imead == 3 else 1.0

    order = np.argsort(nu)
    nu_s = nu[order]
    m_s = m[order]
    gnu_s = gnu[order]
    wk = _nfw_wk_eta(k_h, nu, rv, c_m, eta)
    if imead == 5:
        wk = _baryonify_wk(wk, m, z, cosm)
    wk = wk[:, order]

    plin_d2 = k_h**3 * plin / (2.0 * np.pi**2)
    integrand = gnu_s[None, :] * wk**2 * m_s[None, :]
    sum_nu = np.trapezoid(integrand, nu_s, axis=1) / rho_m
    p1h_d2 = sum_nu * k_h**3 / (2.0 * np.pi**2)
    x4 = (k_h / ks) ** 4
    p1h_d2 = p1h_d2 * x4 / (1.0 + x4)
    if imead in (3, 4):
        p1h_d2 = p1h_d2 * (1.0 - cosm.f_nu) ** 2

    if imead == 3:
        sigv = _sigmav_from_fine(logk_f, k_f, pk_f, R=0.0)
        p2h = _p_dewiggle(k_h, plin, z, sigv, cosm)
        p2h_d2 = k_h**3 * p2h / (2.0 * np.pi**2)
        x = (k_h / kdamp) ** ndamp
        p2h_d2 = p2h_d2 * (1.0 - fd * x / (1.0 + x))
        p2h_d2 = np.maximum(p2h_d2, 0.0)
        alpha = float(np.clip(alpha, 0.5, 2.0))
        pfull_d2 = (p2h_d2**alpha + p1h_d2**alpha) ** (1.0 / max(alpha, 1e-6))
    else:
        pfull_d2 = plin_d2 + p1h_d2
    pfull = pfull_d2 * (2.0 * np.pi**2) / np.maximum(k_h**3, 1e-30)
    return np.maximum(pfull, 1e-30)


def halomod_pk(
    k_h: np.ndarray,
    plin: np.ndarray,
    z: float,
    cosm: HMCosmology,
    *,
    imead: int,
    sig8z: float,
    n_mass: int = 256,
) -> np.ndarray:
    """One-redshift HMcode P_full(k) for given imead."""
    rho_m = 2.775e11 * cosm.om_m
    m = np.logspace(0.0, 18.0, n_mass)
    R = (3.0 * m / (4.0 * np.pi * rho_m)) ** (1.0 / 3.0)
    fine = _fine_pk_grid(k_h, plin)
    sig = _sigma_r_from_fine(*fine, R)
    dc = dc_Mead(z, cosm)
    nu = dc / np.maximum(sig, 1e-30)
    gnu = _gnu_st(nu)
    Dv = Dv_Mead(z, cosm)
    return _halomod_from_shared(
        k_h,
        plin,
        z,
        cosm,
        imead=imead,
        sig8z=sig8z,
        m=m,
        R=R,
        sig=sig,
        nu=nu,
        gnu=gnu,
        dc=dc,
        Dv=Dv,
        fine=fine,
        rho_m=rho_m,
    )


def hmcode_mead2020_feedback(
    k_h: np.ndarray,
    z: float,
    plin_h: np.ndarray,
    cosm: HMCosmology,
    *,
    n_mass: int = 256,
) -> np.ndarray:
    """Full feedback response at one redshift. Returns P_nl in (Mpc/h)³.

    Shares mass/σ/ν tables across imead 3/4/5 (one fine-k upsample).
    """
    rho_m = 2.775e11 * cosm.om_m
    m = np.logspace(0.0, 18.0, n_mass)
    R = (3.0 * m / (4.0 * np.pi * rho_m)) ** (1.0 / 3.0)
    fine = _fine_pk_grid(k_h, plin_h)
    sig = _sigma_r_from_fine(*fine, R)
    sig8z = float(_sigma_r_from_fine(*fine, np.array([8.0]))[0])
    dc = dc_Mead(z, cosm)
    nu = dc / np.maximum(sig, 1e-30)
    gnu = _gnu_st(nu)
    Dv = Dv_Mead(z, cosm)
    shared = dict(
        m=m,
        R=R,
        sig=sig,
        nu=nu,
        gnu=gnu,
        dc=dc,
        Dv=Dv,
        fine=fine,
        rho_m=rho_m,
    )
    p3 = _halomod_from_shared(
        k_h, plin_h, z, cosm, imead=3, sig8z=sig8z, **shared
    )
    p4 = _halomod_from_shared(
        k_h, plin_h, z, cosm, imead=4, sig8z=sig8z, **shared
    )
    p5 = _halomod_from_shared(
        k_h, plin_h, z, cosm, imead=5, sig8z=sig8z, **shared
    )
    ratio = np.sqrt(p3 / np.maximum(plin_h, 1e-30)) * np.sqrt(
        p5 / np.maximum(p4, 1e-30)
    )
    return plin_h * ratio**2


def _hmcode_one_z(
    args: tuple,
) -> tuple[int, np.ndarray]:
    iz, z, k_h, pk_h, cosm = args
    return iz, hmcode_mead2020_feedback(k_h, float(z), pk_h, cosm)


def nonlinear_pk_hmcode_feedback(
    lin: MatterPowerTable,
    *,
    Omega_m: float,
    Omega_b: float,
    h: float = 0.7,
    n_s: float = 0.96,
    logT_AGN: float = 7.7,
) -> MatterPowerTable:
    """Apply mead2020_feedback to a linear MatterPowerTable (CCL units).

    Live HMcode for all k when ``GPUCOSMO_EB_HMCODE_KMAX`` is large (MCMC
    default ``1e6``). Optional CAMB high-k splice if k_cut is finite.
    """
    import os
    from concurrent.futures import ThreadPoolExecutor

    # Prefer parallel NumPy shared-imead path (fast cold). Opt-in JAX via env
    # or when full device path is active.
    use_jax = os.environ.get("GPUCOSMO_EB_HMCODE_JAX", "0").lower() in (
        "1",
        "true",
        "yes",
    ) or os.environ.get("GPUCOSMO_EB_DEVICE", "0").lower() in (
        "1",
        "true",
        "yes",
    )
    if use_jax:
        try:
            from gpucosmo.jax.eb.hmcode_jax import nonlinear_pk_hmcode_feedback_jax

            return nonlinear_pk_hmcode_feedback_jax(
                lin,
                Omega_m=Omega_m,
                Omega_b=Omega_b,
                h=h,
                n_s=n_s,
                logT_AGN=logT_AGN,
            )
        except Exception:
            pass

    cosm = assign_HM_cosmology(Omega_m, Omega_b, h, n_s, logT_AGN=logT_AGN)
    k_mpc = np.exp(lin.log_k)
    k_h = k_mpc / h
    log_pk_nl = np.zeros_like(lin.log_pk)

    # Init dewiggle once from z≈0 spectrum
    iz0 = int(np.argmin(np.abs(lin.z)))
    pk0_h = np.exp(lin.log_pk[iz0]) * (h**3)
    _init_wiggle_from_pk(k_h, pk0_h, cosm, _grow(float(lin.z[iz0]), cosm))

    n_jobs = int(os.environ.get("GPUCOSMO_EB_HMCODE_NJOBS", "8"))
    jobs = [
        (iz, float(z), k_h, np.exp(lin.log_pk[iz]) * (h**3), cosm)
        for iz, z in enumerate(lin.z)
    ]
    if n_jobs <= 1 or len(jobs) == 1:
        for job in jobs:
            iz, pk_nl_h = _hmcode_one_z(job)
            log_pk_nl[iz] = np.log(np.maximum(pk_nl_h / (h**3), 1e-30))
    else:
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            for iz, pk_nl_h in ex.map(_hmcode_one_z, jobs):
                log_pk_nl[iz] = np.log(np.maximum(pk_nl_h / (h**3), 1e-30))

    k_cut = float(os.environ.get("GPUCOSMO_EB_HMCODE_KMAX", "0.2"))
    if np.any(k_mpc > k_cut) and k_cut < 1.0e5:
        try:
            from gpucosmo.jax.eb.transfer_out import nonlinear_pk_from_camb
            from gpucosmo.jax.eb.transfer_out import sigma8_top_hat
            from gpucosmo.jax.power import CosmologyParams
            from scipy.interpolate import RegularGridInterpolator

            s8 = sigma8_top_hat(k_h, np.exp(lin.log_pk[iz0]) * (h**3))
            params = CosmologyParams(
                Omega_c=Omega_m - Omega_b,
                Omega_b=Omega_b,
                h=h,
                sigma8=s8,
                n_s=n_s,
            )
            camb_nl = nonlinear_pk_from_camb(
                params,
                z_max=float(lin.z.max()),
                n_z=len(lin.z),
                k_min=float(k_mpc.min()),
                k_max=float(k_mpc.max()),
                n_k=len(k_mpc),
                logT_AGN=logT_AGN,
            )
            interp = RegularGridInterpolator(
                (camb_nl.z, camb_nl.log_k),
                camb_nl.log_pk,
                bounds_error=False,
                fill_value=None,
            )
            zz, lk = np.meshgrid(lin.z, lin.log_k, indexing="ij")
            log_camb = interp(np.stack([zz.ravel(), lk.ravel()], axis=-1)).reshape(
                lin.log_pk.shape
            )
            w = np.clip((k_mpc - 0.7 * k_cut) / max(0.3 * k_cut, 1e-6), 0.0, 1.0)
            for iz in range(len(lin.z)):
                log_pk_nl[iz] = (1.0 - w) * log_pk_nl[iz] + w * log_camb[iz]
        except Exception:
            pass

    return MatterPowerTable(log_k=lin.log_k, z=lin.z, log_pk=log_pk_nl)


# Keep legacy names used by __init__
nonlinear_pk_table = nonlinear_pk_hmcode_feedback

__all__ = [
    "Dv_Mead",
    "HMCosmology",
    "assign_HM_cosmology",
    "dc_Mead",
    "hmcode_mead2020_feedback",
    "nonlinear_pk_hmcode_feedback",
    "nonlinear_pk_table",
]
