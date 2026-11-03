#!/usr/bin/env python3
# This file is part of GPUCosmo. Copyright (C) 2025-2026 Leonel Medina-Varela.
# SPDX-License-Identifier: GPL-3.0-or-later
# Derived in part from the following, translated to JAX and modified for GPUCosmo:
# - CAMB (https://github.com/cmbant/CAMB), Copyright 2000-2025 Antony Lewis and Anthony Challinor.
#   Licence: LICENSES/CAMB.txt.
# See NOTICE.md.
"""HMcode-2020 (with and without baryonic feedback) and HMcode-2016 (CAMB ``halofit_version``
``'mead2020'``, ``'mead2020_feedback'``, ``'mead2016'``) in pure JAX.

Port of the imead=3 (Mead et al. 2021, arXiv:2009.01858) and imead=1 (Mead et al. 2016,
arXiv:1602.02154; CAMB 1.5.4 ``halofit.f90``) paths of CAMB ``halofit.f90`` for flat
w0waCDM with massive neutrinos, written so it runs
under ``jit`` / ``vmap`` over cosmologies / ``grad``. CAMB's adaptive
integrators (trapezium→Simpson doubling to 1e-4, ODE doubling to 1e-4) are
replaced by fixed grids fine enough to be converged well past that tolerance.

HMcode-2016 differs in the halo-model ingredients (δ_c, Δ_v with ν corrections, η = 0.603 −
0.3σ_8(z), k* = 0.584/σ_v, f_damp = 0.0095 σ_v(100)^1.37 with a tanh two-halo damping and
no BAO dewiggling, α = 3.24·1.85^n_eff, A = 3.13, Dolag exponent 1.5 without the z
factor, one-halo damping 1 − exp(−(k/k*)²), no (1−f_ν)² one-halo factor); σ(R) tables,
n_eff, the mass function, the NFW window and the growth tables are shared.

HMcode-2020 feedback (CAMB ``mead2020_feedback``, Mead et al. 2021) is CAMB's response
model: the dark-matter-only HMcode-2020 ratio times P_num/P_den, two plain halo models
(P = P_lin + P_1h, without dewiggling, two-halo damping, η or α smoothing) that share δ_c, Δ_v,
the collapse redshifts, the Dolag factor and the k* one-halo damping with HMcode-2020 and
differ in the halo profile: P_den uses NFW haloes with A = 4 (and the (1 − f_ν)² one-halo
factor), P_num the gas-depleted haloes W(k, M) [f_c + (f_b − f_*) (M/M_b)²/(1 + (M/M_b)²)]
+ f_* with A = B0·10^(z B_z), M_b = 10^(M_b0 + z M_bz), f_* = min(f_*0·10^(z f_*z), f_b),
all linear in θ = log10(T_AGN/K) − 7.8 (CAMB ``HMCode_logT_AGN``; may be traced).

Conventions follow CAMB: wavenumbers in h/Mpc, the linear spectrum as the
dimensionless Δ²(k, z) = k³P(k)/2π² of *total* matter (CAMB ``Transfer_tot``);
the cold spectrum is derived internally with CAMB's Eisenstein–Hu
``Tcb_Tcbnu_ratio``. Main entry point: :func:`hmcode2020_ratio`.

``cosmo`` keys: ``Omega_m`` (CDM + baryons + massive ν today), ``Omega_b``,
``Omega_c``, ``Omega_nu`` (massive ν today), ``Omega_v`` (CAMB ``omega_de``; the
remainder 1 − Ω_m − Ω_v ≈ Ω_rad enters H² like curvature, as in CAMB),
``h``, ``n_s``, ``w0``, ``wa``, ``T_cmb``, ``N_nu`` (number of massive species).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# CAMB halofit.f90 parameters (several are default-REAL literals in Fortran).
_F32 = lambda x: float(np.float32(x))  # noqa: E731
KMIN_PK, KMAX_PK, NK_PK = 1e-3, 1e2, 512  # rebinned linear table [h/Mpc]
KMIN_WIGGLE, KMAX_WIGGLE, NK_WIGGLE = _F32(5e-3), _F32(5.0), 512
WIGGLE_SIGMA, KNORM_NOWIGGLE = _F32(0.25), _F32(0.03)
RMIN_SIG, RMAX_SIG, N_SIG = 1e-4, 1e3, 64  # σ(R) table [Mpc/h]
AMIN_G, AMAX_G, N_G = 1e-3, 1.0, 64  # growth table
AINI_ODE = 1e-4
# Fixed grids replacing CAMB's adaptive doubling (to 1e-4). Validated against
# CAMB mead2020: sqrt(P_nl/P_lin) agrees to ≤ 8e-5 for k ≤ 10 h/Mpc, z ≤ 4
# (scripts/validate_hmcode2020_vs_camb.py); 8× finer grids change nothing.
N_ODE = 3201  # uniform-a RK4 nodes
N_T = 513  # Simpson nodes on t ∈ [0, 1] for σ, σ_v, n_eff
N_MASS, MMIN, MMAX = 256, 1.0, 1e18
ZC_DOLAG = 10.0
Z_BATCH = 4  # redshifts evaluated together (memory bound, see hmcode2020_ratio)
RHO_CRIT = 2.775e11  # h² M_sun / Mpc³

_T_NODES = np.linspace(0.0, 1.0, N_T)
_SIMPSON_W = np.ones(N_T)
_SIMPSON_W[1:-1:2] = 4.0
_SIMPSON_W[2:-1:2] = 2.0
_SIMPSON_W *= (1.0 / (N_T - 1)) / 3.0


# --------------------------------------------------------------------------
# Interpolation helpers (CAMB ``find``)
# --------------------------------------------------------------------------


def _find_linear(x, xt, yt):
    """Linear interpolation with linear extrapolation from the end intervals."""
    n = xt.shape[0]
    i = jnp.clip(jnp.searchsorted(xt, x, side="right") - 1, 0, n - 2)
    x0, x1 = xt[i], xt[i + 1]
    y0, y1 = yt[i], yt[i + 1]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def _find_cubic(x, xt, yt):
    """CAMB ``find`` iorder=3, imeth=2: 4-point Lagrange, linear outside the table."""
    n = xt.shape[0]
    i = jnp.clip(jnp.searchsorted(xt, x, side="right") - 1, 1, n - 3)
    xs = jnp.stack([xt[i - 1], xt[i], xt[i + 1], xt[i + 2]])
    ys = jnp.stack([yt[i - 1], yt[i], yt[i + 1], yt[i + 2]])

    def basis(j):
        num = 1.0
        den = 1.0
        for m in range(4):
            if m != j:
                num = num * (x - xs[m])
                den = den * (xs[j] - xs[m])
        return num / den

    y_in = sum(ys[j] * basis(j) for j in range(4))
    lo = yt[0] + (yt[1] - yt[0]) * (x - xt[0]) / (xt[1] - xt[0])
    hi = yt[-1] + (yt[-1] - yt[-2]) * (x - xt[-1]) / (xt[-1] - xt[-2])
    return jnp.where(x < xt[0], lo, jnp.where(x > xt[-1], hi, y_in))


# --------------------------------------------------------------------------
# Background and growth
# --------------------------------------------------------------------------


def _x_de(a, c):
    return a ** (-3.0 * (1.0 + c["w0"] + c["wa"])) * jnp.exp(-3.0 * c["wa"] * (1.0 - a))


def _hubble2(a, c):
    zp1 = 1.0 / a
    return (c["Omega_m"] * zp1**3 + c["Omega_v"] * _x_de(a, c)
            + (1.0 - c["Omega_m"] - c["Omega_v"]) * zp1**2)


def _ah(a, c):
    zp1 = 1.0 / a
    w = c["w0"] + (1.0 - a) * c["wa"]
    return -0.5 * (c["Omega_m"] * zp1**3 + c["Omega_v"] * (1.0 + 3.0 * w) * _x_de(a, c))


def _omega_m_a(a, c):
    return c["Omega_m"] / a**3 / _hubble2(a, c)


def _growth_tables(c):
    """CAMB ``fill_growtab``: normalised growth g(a) and accumulated growth on 64 a."""
    a_nodes = jnp.linspace(AINI_ODE, 1.0, N_ODE)
    da = a_nodes[1] - a_nodes[0]
    f = 1.0 - _omega_m_a(jnp.asarray(AINI_ODE), c)
    d0 = AINI_ODE ** (1.0 - 3.0 * f / 5.0)
    v0 = (1.0 - 3.0 * f / 5.0) * AINI_ODE ** (-3.0 * f / 5.0)

    def fv(d, v, a):
        return 1.5 * _omega_m_a(a, c) * d / a**2 - (2.0 + _ah(a, c) / _hubble2(a, c)) * v / a

    def step(carry, a):
        d, v = carry
        kx1, kv1 = da * v, da * fv(d, v, a)
        kx2, kv2 = da * (v + kv1 / 2), da * fv(d + kx1 / 2, v + kv1 / 2, a + da / 2)
        kx3, kv3 = da * (v + kv2 / 2), da * fv(d + kx2 / 2, v + kv2 / 2, a + da / 2)
        kx4, kv4 = da * (v + kv3), da * fv(d + kx3, v + kv3, a + da)
        d1 = d + (kx1 + 2 * kx2 + 2 * kx3 + kx4) / 6.0
        v1 = v + (kv1 + 2 * kv2 + 2 * kv3 + kv4) / 6.0
        return (d1, v1), d1

    _, d_rest = jax.lax.scan(step, (d0, v0), a_nodes[:-1])
    d_nodes = jnp.concatenate([jnp.atleast_1d(d0), d_rest])
    gnorm = d_nodes[-1]
    d_norm = d_nodes / gnorm
    a_g = jnp.linspace(AMIN_G, AMAX_G, N_G)
    growth = jnp.interp(a_g, a_nodes, d_norm)
    # agrow(a) = ∫_{amin}^{a} gnorm g / a' da' + gnorm g(amin), on the fine grid.
    integrand = gnorm * d_norm / a_nodes
    cum = jnp.concatenate([jnp.zeros(1), jnp.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * da)])
    cum_at = jnp.interp(a_g, a_nodes, cum) - jnp.interp(AMIN_G, a_nodes, cum)
    agrow = cum_at + gnorm * growth[0]
    return dict(a_g=a_g, growth=growth, agrow=agrow, gnorm=gnorm)


def _grow(a, gt):
    return jnp.where(a >= 1.0, 1.0, _find_cubic(a, gt["a_g"], gt["growth"]))


def _growint_lcdm(a, om_m):
    """CAMB ``growint`` for the flat-ΛCDM copy (Dolag correction): γ = 0.55."""
    x = jnp.linspace(0.0, 1.0, 1025)
    aa = a + (1.0 - a) * x
    om = om_m / aa**3 / (om_m / aa**3 + (1.0 - om_m))
    fvals = -(om**0.55) / aa
    w = np.ones(1025)
    w[1:-1:2], w[2:-1:2] = 4.0, 2.0
    integral = jnp.sum(jnp.asarray(w) * fvals) * (1.0 - a) / 1024 / 3.0
    return jnp.exp(integral)


# --------------------------------------------------------------------------
# Linear spectra, windows, integrals
# --------------------------------------------------------------------------


def _tcb_tcbnu_ratio(kh, a, c, gt):
    fnu = c["Omega_nu"] / c["Omega_m"]
    big_t = c["T_cmb"] / 2.7
    om_h2 = c["Omega_m"] * c["h"] ** 2
    zeq = 2.5e4 * om_h2 * big_t ** (-4.0)
    d = (1.0 + zeq) * gt["gnorm"] * _grow(a, gt)
    q = kh * c["h"] * big_t**2 / om_h2
    fnu_s = jnp.maximum(fnu, 1e-12)
    yfs = 17.2 * fnu_s * (1.0 + 0.488 * fnu_s ** (-7.0 / 6.0)) * (c["N_nu"] * q / fnu_s) ** 2
    pcb = (5.0 - jnp.sqrt(1.0 + 24.0 * (1.0 - fnu))) / 4.0
    x = (d / (1.0 + yfs)) ** 0.7
    dcb = (1.0 + x) ** (pcb / 0.7)
    dcbnu = ((1.0 - fnu) ** (0.7 / pcb) + x) ** (pcb / 0.7)
    return jnp.where(fnu > 0.0, dcb / dcbnu, 1.0)


def _wk_tophat(x):
    xs = jnp.where(jnp.abs(x) < 1e-3, 1.0, x)
    return jnp.where(jnp.abs(x) < 1e-3, 1.0 - x**2 / 10.0,
                     3.0 * (jnp.sin(xs) - xs * jnp.cos(xs)) / xs**3)


def _wk_tophat_deriv(x):
    xs = jnp.where(jnp.abs(x) < 1e-3, 1.0, x)
    return jnp.where(jnp.abs(x) < 1e-3, -x / 5.0 + x**3 / 70.0,
                     (3.0 / xs**4) * ((xs**2 - 3.0) * jnp.sin(xs) + 3.0 * xs * jnp.cos(xs)))


def _t_integral(fun_of_kr, alpha: float):
    """∫_0^1 dt F(kR(t)) α/(t(1−t)) with kR = (1/t − 1)^α (CAMB change of variable)."""
    t = jnp.asarray(_T_NODES[1:-1])
    kr = (1.0 / t - 1.0) ** alpha
    vals = fun_of_kr(kr) * alpha / (t * (1.0 - t))
    return jnp.sum(jnp.asarray(_SIMPSON_W[1:-1]) * vals)


def _si_ci(x):
    """CAMB's rational Si(x), Ci(x) approximations (x0 = 4 split)."""
    x = jnp.asarray(x, dtype=jnp.float64)
    small = jnp.abs(x) <= 4.0
    xs = jnp.where(small, x, 1.0)
    x2 = xs * xs
    si_s = xs * (1.0 + x2 * (-4.54393409816329991e-2 + x2 * (1.15457225751016682e-3
           + x2 * (-1.41018536821330254e-5 + x2 * (9.43280809438713025e-8 + x2 * (-3.53201978997168357e-10
           + x2 * (7.08240282274875911e-13 + x2 * (-6.05338212010422477e-16)))))))) / (
           1.0 + x2 * (1.01162145739225565e-2 + x2 * (4.99175116169755106e-5
           + x2 * (1.55654986308745614e-7 + x2 * (3.28067571055789734e-10 + x2 * (4.5049097575386581e-13
           + x2 * (3.21107051193712168e-16)))))))
    ci_s = 0.577215664901532861 + jnp.log(jnp.abs(xs)) + x2 * (-0.25 + x2 * (7.51851524438898291e-3
           + x2 * (-1.27528342240267686e-4 + x2 * (1.05297363846239184e-6 + x2 * (-4.68889508144848019e-9
           + x2 * (1.06480802891189243e-11 + x2 * (-9.93728488857585407e-15))))))) / (
           1.0 + x2 * (1.1592605689110735e-2 + x2 * (6.72126800814254432e-5
           + x2 * (2.55533277086129636e-7 + x2 * (6.97071295760958946e-10 + x2 * (1.38536352772778619e-12
           + x2 * (1.89106054713059759e-15 + x2 * (1.39759616731376855e-18))))))))
    xl = jnp.where(small, 5.0, x)
    y = 1.0 / (xl * xl)
    f = (1.0 + y * (7.44437068161936700618e2 + y * (1.96396372895146869801e5
         + y * (2.37750310125431834034e7 + y * (1.43073403821274636888e9 + y * (4.33736238870432522765e10
         + y * (6.40533830574022022911e11 + y * (4.20968180571076940208e12
         + y * (1.00795182980368574617e13 + y * (4.94816688199951963482e12
         + y * (-4.94701168645415959931e11))))))))))) / (xl * (1.0 + y * (7.46437068161927678031e2
         + y * (1.97865247031583951450e5 + y * (2.41535670165126845144e7
         + y * (1.47478952192985464958e9 + y * (4.58595115847765779830e10
         + y * (7.08501308149515401563e11 + y * (5.06084464593475076774e12
         + y * (1.43468549171581016479e13 + y * (1.11535493509914254097e13)))))))))))
    g = y * (1.0 + y * (8.1359520115168615e2 + y * (2.35239181626478200e5
         + y * (3.12557570795778731e7 + y * (2.06297595146763354e9 + y * (6.83052205423625007e10
         + y * (1.09049528450362786e12 + y * (7.57664583257834349e12 + y * (1.81004487464664575e13
         + y * (6.43291613143049485e12 + y * (-1.36517137670871689e12))))))))))) / (
         1.0 + y * (8.19595201151451564e2 + y * (2.40036752835578777e5 + y * (3.26026661647090822e7
         + y * (2.23355543278099360e9 + y * (7.87465017341829930e10 + y * (1.39866710696414565e12
         + y * (1.17164723371736605e13 + y * (4.01839087307656620e13 + y * (3.99653257887490811e13))))))))))
    si_l = jnp.pi / 2.0 - f * jnp.cos(xl) - g * jnp.sin(xl)
    ci_l = f * jnp.sin(xl) - g * jnp.cos(xl)
    return jnp.where(small, si_s, si_l), jnp.where(small, ci_s, ci_l)


def _win_nfw(k, rv, c):
    ks = jnp.maximum(k * rv / c, 1e-10)
    si1, ci1 = _si_ci(ks)
    si2, ci2 = _si_ci((1.0 + c) * ks)
    w = (jnp.cos(ks) * (ci2 - ci1) + jnp.sin(ks) * (si2 - si1)
         - jnp.sin(ks * c) / (ks * (1.0 + c)))
    w = w / (jnp.log(1.0 + c) - c / (1.0 + c))
    return jnp.clip(w, 0.0, 1.0)


def _tk_nw(kh, c):
    wm = c["Omega_m"] * c["h"] ** 2
    wb = c["Omega_b"] * c["h"] ** 2
    rb = c["Omega_b"] / c["Omega_m"]
    s = 44.5 * jnp.log(9.83 / wm) / jnp.sqrt(1.0 + 10.0 * wb**0.75)
    alpha = 1.0 - 0.328 * jnp.log(431.0 * wm) * rb + 0.38 * jnp.log(22.3 * wm) * rb**2
    gam = c["Omega_m"] * c["h"] * (alpha + (1.0 - alpha) / (1.0 + (0.43 * kh * s * c["h"]) ** 4))
    q = kh * (c["T_cmb"] / 2.7) ** 2 / gam
    L = jnp.log(2.0 * np.float32(np.e) + 1.8 * q)
    C = 14.2 + 731.0 / (1.0 + 62.5 * q)
    return L / (L + C * q**2)


_LOGK_WIG = np.linspace(np.log(KMIN_WIGGLE), np.log(KMAX_WIGGLE), NK_WIGGLE)
_WIG_EDGE = (np.abs(_LOGK_WIG - _LOGK_WIG[0]) < 3 * WIGGLE_SIGMA) | (
    np.abs(_LOGK_WIG - _LOGK_WIG[-1]) < 3 * WIGGLE_SIGMA
)
_WIG_W = np.exp(-((_LOGK_WIG[:, None] - _LOGK_WIG[None, :]) ** 2) / (2 * WIGGLE_SIGMA**2))
_WIG_W = _WIG_W / _WIG_W.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------
# One redshift
# --------------------------------------------------------------------------


# HMcode-2016 baryon-feedback parameters (CAMB HMCode_A_baryon / HMCode_eta_baryon defaults,
# set from Python as float64).
A_BARYON_2016, ETA_BARYON_2016 = 3.13, 0.603
KS_LIMIT = 7.0  # (k/k*)² above which the 2016 one-halo damping factor is 0
LOGT_AGN_DEFAULT = 7.8  # CAMB HMCode_logT_AGN default
A_CONC_DEN = 4.0  # Bullock amplitude of the feedback-response denominator (CAMB As, imead=4)


def _feedback_params(logt_agn, z, c):
    """CAMB ``As`` (imead=5), ``m_baryon`` [M_sun/h] and ``f_star`` at redshift ``z`` for
    log10(T_AGN/K) = ``logt_agn`` (CAMB ``halofit.f90``; Fortran default-REAL literals)."""
    theta = logt_agn - _F32(7.8)
    a_conc = (_F32(3.44) - _F32(0.496) * theta) * 10.0 ** (z * (_F32(-0.0671) - _F32(0.0371) * theta))
    m_b = 10.0 ** (_F32(13.87) + _F32(1.81) * theta) * 10.0 ** ((_F32(-0.108) + _F32(0.195) * theta) * z)
    f_star = (_F32(0.0201) - _F32(0.003) * theta) * 10.0 ** ((_F32(0.409) + _F32(0.0224) * theta) * z)
    return a_conc, m_b, jnp.minimum(f_star, c["Omega_b"] / c["Omega_m"])


def _one_z(a, log_d2_tab, c, gt, k_out_h, version: str = "mead2020", logt_agn=None, hm16=None):
    """Δ²_NL / Δ²_lin at ``k_out_h`` for one scale factor ``a`` (HMcode ``version``).

    ``log_d2_tab`` is log Δ²_lin(k, z) of total matter on the rebinned table; ``logt_agn``
    log10(T_AGN/K), used by "mead2020_feedback" only; ``hm16`` the HMcode-2016 baryon parameters
    (A, η₀) (CAMB ``HMCode_A_baryon``, ``HMCode_eta_baryon``; default 3.13, 0.603).
    """
    logk_tab = jnp.log(jnp.geomspace(KMIN_PK, KMAX_PK, NK_PK))
    kh_tab = jnp.exp(logk_tab)
    z = 1.0 / a - 1.0
    g = _grow(a, gt)
    fnu = c["Omega_nu"] / c["Omega_m"]
    # CAMB stores Δ²/g(z)² for total and cold matter.
    log_t0 = log_d2_tab - 2.0 * jnp.log(g)
    log_tc = log_t0 + 2.0 * jnp.log(_tcb_tcbnu_ratio(kh_tab, a, c, gt))

    def plin(kh, cold: bool, zero_z: bool = False):
        tab = log_tc if cold else log_t0
        p = jnp.exp(_find_linear(jnp.log(kh), logk_tab, tab))
        return p if zero_z else p * g**2

    def sigma(r, cold=True, zero_z=False):
        return jnp.sqrt(_t_integral(
            lambda kr: plin(kr / r, cold, zero_z) * _wk_tophat(kr) ** 2, 3.0))

    # σ(R) table of the cold spectrum at "z = 0" (i.e. divided by g²).
    log_r_tab = jnp.linspace(np.log(RMIN_SIG), np.log(RMAX_SIG), N_SIG)
    log_sig_tab = jnp.log(jax.vmap(lambda lr: sigma(jnp.exp(lr), True, True))(log_r_tab))

    def sigma_lut(r):
        return g * jnp.exp(_find_cubic(jnp.log(r), log_r_tab, log_sig_tab))

    sigv = jnp.sqrt(_t_integral(lambda kr: plin(kr, False) / kr**2, 3.0) / 3.0)
    sig8c = sigma(8.0, True)
    v2016 = version == "mead2016"
    vfb = version == "mead2020_feedback"  # HMcode-2020 times the feedback response

    # Mead (2017) δ_c, Δ_v with ν corrections.
    lg = gt["gnorm"] * g
    bg = _find_cubic(a, gt["a_g"], gt["agrow"])
    om_m = _omega_m_a(a, c)
    x, y = lg / a, bg / a
    fm = lambda p0, p1, p2, p3: p0 + p1 * (1 - x) + p2 * (1 - x) ** 2 + p3 * (1 - y)  # noqa: E731
    dc0 = _F32(3.0 / 20.0) * (12.0 * jnp.pi) ** _F32(2.0 / 3.0)
    dc = (1.0 + fm(-0.0069, -0.0208, 0.0312, 0.0021) * jnp.log10(om_m)
          + fm(0.0001, -0.0647, -0.0417, 0.0646)) * dc0 * (1.0 - 0.041 * fnu)
    dv = (1.0 + fm(-0.79, -10.17, 2.51, 6.51) * jnp.log10(om_m)
          + fm(-1.89, 0.38, 18.8, -15.87) * jnp.log10(om_m) ** 2) * (18.0 * jnp.pi**2) * (1.0 + 0.763 * fnu)
    if v2016:
        sig8 = sigma(8.0, False)  # all matter (CAMB lut%sig8z)
        dc = ((_F32(1.59) + _F32(0.0314) * jnp.log(sig8)) * (1.0 + _F32(0.262) * fnu)
              * (1.0 + _F32(0.0123) * jnp.log10(om_m)))
        dv = 418.0 * om_m ** (-0.352) * (1.0 + 0.916 * fnu)

    # Halo tables.
    m = jnp.exp(jnp.linspace(np.log(MMIN), np.log(MMAX), N_MASS))
    rho = RHO_CRIT * c["Omega_m"]
    rr = (3.0 * m / (4.0 * jnp.pi * rho)) ** (1.0 / 3.0)
    sig = sigma_lut(rr)
    nu = dc / sig
    sigf = sigma_lut(rr * 0.01 ** (1.0 / 3.0))
    gst = 0.21616 * (1.0 + (0.707 * nu * nu) ** (-0.3)) * jnp.exp(-0.707 * nu * nu / 2.0)
    weight = gst * m
    rv = rr / dv ** (1.0 / 3.0)
    rnl = jnp.where(nu[0] > 1.0, rr[0],
                    jnp.exp(_find_cubic(0.0, jnp.log(nu), jnp.log(rr))))
    sig_rnl = sigma(rnl, True)
    neff = -3.0 - 2.0 * _t_integral(
        lambda kr: plin(kr / rnl, True) * _wk_tophat(kr) * _wk_tophat_deriv(kr) * kr, 2.0
    ) / sig_rnl**2
    neff = jnp.clip(neff, c["n_s"] - 4.0, c["n_s"])

    # Bullock concentration with the Dolag correction (z < 10).
    rhs = dc * g / sigf
    af = _find_cubic(rhs, gt["growth"], gt["a_g"])
    zf = jnp.where(rhs > g, z, 1.0 / af - 1.0)
    a_inf = 1.0 / (1.0 + ZC_DOLAG)
    ginf_ratio = _grow(jnp.asarray(a_inf), gt) / _growint_lcdm(a_inf, c["Omega_m"])
    if v2016:
        a16 = A_BARYON_2016 if hm16 is None else hm16[0]
        conc = a16 * (1.0 + zf) / (1.0 + z)
        dolag = ginf_ratio**1.5
    else:
        conc = 5.196 * (1.0 + zf) / (1.0 + z)
        dolag = ginf_ratio * (_growint_lcdm(a, c["Omega_m"]) / g)
    conc = jnp.where(z < ZC_DOLAG, conc * dolag, conc)
    if vfb:
        # Response passes (CAMB imead=4, 5): only the Bullock amplitude differs from 2020.
        a_conc, m_b, f_star = _feedback_params(logt_agn, z, c)
        conc_den = A_CONC_DEN * (1.0 + zf) / (1.0 + z)
        conc_den = jnp.where(z < ZC_DOLAG, conc_den * dolag, conc_den)
        conc_num = a_conc * (1.0 + zf) / (1.0 + z)
        conc_num = jnp.where(z < ZC_DOLAG, conc_num * dolag, conc_num)
        f_c, f_b = c["Omega_c"] / c["Omega_m"], c["Omega_b"] / c["Omega_m"]
        x_b = (m / m_b) ** 2
        bar_frac = f_c + (f_b - f_star) * x_b / (1.0 + x_b)  # CAMB baryon_mass_fraction

    if v2016:
        sigv100 = jnp.sqrt(_t_integral(
            lambda kr: plin(kr / 100.0, False) / (kr / 100.0) ** 2 * _wk_tophat(kr) ** 2, 3.0) / 3.0)
        eta = (ETA_BARYON_2016 if hm16 is None else hm16[1]) - _F32(0.3) * sig8
        kstar = _F32(0.584) / sigv
        fdamp = jnp.clip(_F32(0.0095) * sigv100 ** _F32(1.37), 1e-3, 0.99)
        alpha = jnp.clip(_F32(3.24) * _F32(1.85) ** neff, 0.5, 2.0)
    else:
        eta = 0.1281 * sig8c ** (-0.3644)
        kstar = 0.05618 * sig8c ** (-1.013)
        fdamp = jnp.clip(0.2696 * sig8c**0.9403, 1e-3, 0.99)
        alpha = jnp.clip(1.875 * 1.603**neff, 0.5, 2.0)
    kdamp = 0.05699 * sig8c ** (-1.089)
    nu_eta = nu**eta

    # Dewiggle (built from the z = 0-normalised total table, as CAMB does).
    kw = jnp.exp(jnp.asarray(_LOGK_WIG))
    pk_w = plin(kw, False, True)
    pk_nw = kw ** (c["n_s"] + 3.0) * _tk_nw(kw, c) ** 2
    pk_norm = plin(jnp.asarray(KNORM_NOWIGGLE), False, True)
    pk_nw = pk_nw * pk_norm / _find_cubic(jnp.asarray(KNORM_NOWIGGLE), kw, pk_nw)
    ratio = pk_w / pk_nw
    smooth = jnp.where(jnp.asarray(_WIG_EDGE), ratio, jnp.asarray(_WIG_W) @ ratio)
    pk_wiggle = pk_w - smooth * pk_nw

    def at_k(kh):
        pl = plin(kh, False)
        if v2016:  # CAMB p_2h imead=1: no dewiggling, tanh damping by σ_v
            p2h = jnp.maximum(pl * (1.0 - fdamp * jnp.tanh(kh * sigv / jnp.sqrt(jnp.abs(fdamp))) ** 2), 0.0)
        else:
            logk = jnp.log(kh)
            in_w = (logk >= _LOGK_WIG[0]) & (logk <= _LOGK_WIG[-1])
            p_wig = jnp.where(in_w, _find_cubic(logk, jnp.asarray(_LOGK_WIG), pk_wiggle), 0.0)
            p_dw = pl + (jnp.exp(-((kh * sigv) ** 2)) - 1.0) * p_wig * g**2
            xd = (kh / kdamp) ** 2.85
            p2h = jnp.maximum(p_dw * (1.0 - fdamp * xd / (1.0 + xd)), 0.0)
        wk = _win_nfw(kh * nu_eta, rv, conc)
        integ = weight * wk**2 * (1.0 if v2016 else (1.0 - fnu) ** 2)
        s1h = jnp.sum(0.5 * (integ[1:] + integ[:-1]) * jnp.diff(nu)) / rho
        if v2016:
            x2 = (kh / kstar) ** 2
            damp = jnp.where(x2 > KS_LIMIT, 1.0, 1.0 - jnp.exp(-jnp.minimum(x2, KS_LIMIT)))
        else:
            xs = (kh / kstar) ** 4
            damp = xs / (1.0 + xs)
        # The 1-halo integral is >= 0; it reaches -1e-178 (rounding) when every
        # halo is a >30-sigma peak (high z, Omega_c -> 0) and nu(M) is not
        # monotonic, which p1h**alpha (alpha = 0.5) turned into NaN.
        p1h = jnp.maximum(s1h, 0.0) * kh**3 / (2.0 * jnp.pi**2) * damp
        pfull = (p2h**alpha + p1h**alpha) ** (1.0 / alpha)
        if vfb:  # × P_num / P_den, each P_lin + P_1h (η = 0, α = 1, no two-halo damping)
            def p1h_fb(wk_fb, fac):
                integ_fb = weight * wk_fb**2 * fac
                s_fb = jnp.sum(0.5 * (integ_fb[1:] + integ_fb[:-1]) * jnp.diff(nu)) / rho
                return jnp.maximum(s_fb, 0.0) * kh**3 / (2.0 * jnp.pi**2) * damp

            p_den = pl + p1h_fb(_win_nfw(kh, rv, conc_den), (1.0 - fnu) ** 2)
            p_num = pl + p1h_fb(_win_nfw(kh, rv, conc_num) * bar_frac + f_star, 1.0)
            return pfull / pl * (p_num / p_den)
        return pfull / pl

    return jax.vmap(at_k)(k_out_h)


HMCODE_VERSIONS = ("mead2020", "mead2016", "mead2020_feedback")


def hmcode2020_ratio(k_out_h, a, log_d2_tab, cosmo, *, growth_tables=None, version: str = "mead2020",
                     logT_AGN=LOGT_AGN_DEFAULT, A_baryon=None, eta_baryon=None):
    """sqrt(Δ²_NL/Δ²_lin) (CAMB ``nonlin_ratio``) on (n_a, n_k); ``version`` "mead2020",
    "mead2016" or "mead2020_feedback" (CAMB ``halofit_version``).

    ``k_out_h`` [h/Mpc] output wavenumbers; ``a`` (n_a,) scale factors;
    ``log_d2_tab`` (n_a, 512) log Δ²_lin of total matter on
    :func:`linear_table_k_h`; ``cosmo`` see module docstring. ``logT_AGN``: log10(T_AGN/K)
    of "mead2020_feedback" (CAMB ``HMCode_logT_AGN``, default 7.8; float or traced scalar;
    ignored by the other versions). ``A_baryon``, ``eta_baryon``: HMcode-2016 baryon feedback
    (CAMB ``HMCode_A_baryon``/``HMCode_eta_baryon``; float or traced; "mead2016" only, default
    3.13/0.603 — the one-parameter model has η = 0.98 − 0.12 A).
    """
    if version not in HMCODE_VERSIONS:
        raise ValueError(f"HMcode version {version!r}: ported {HMCODE_VERSIONS}")
    c = {key: jnp.asarray(val, dtype=jnp.float64) for key, val in cosmo.items()}
    # ``growth_tables``: precomputed ``_growth_tables(c)`` (e.g. on the host).
    gt = _growth_tables(c) if growth_tables is None else growth_tables
    k_out_h = jnp.asarray(k_out_h, dtype=jnp.float64)
    logt_agn = jnp.asarray(logT_AGN, dtype=jnp.float64) if version == "mead2020_feedback" else None
    hm16 = None
    if version == "mead2016" and (A_baryon is not None or eta_baryon is not None):
        hm16 = (jnp.asarray(A_BARYON_2016 if A_baryon is None else A_baryon, dtype=jnp.float64),
                jnp.asarray(ETA_BARYON_2016 if eta_baryon is None else eta_baryon, dtype=jnp.float64))
    # Redshifts in small batches: the one-halo term holds (n_k × 256 masses) Si/Ci
    # temporaries per z, which vmapped over every z (and an outer cosmology vmap)
    # can exhaust GPU memory and force XLA to rematerialize upstream work.
    r2 = jax.lax.map(
        lambda args: _one_z(args[0], args[1], c, gt, k_out_h, version, logt_agn, hm16),
        (jnp.asarray(a, dtype=jnp.float64), jnp.asarray(log_d2_tab, dtype=jnp.float64)),
        batch_size=Z_BATCH,
    )
    return jnp.sqrt(r2)


def linear_table_k_h() -> np.ndarray:
    """CAMB's HMcode rebinning grid: 512 log-spaced k in [1e-3, 1e2] h/Mpc."""
    return np.geomspace(KMIN_PK, KMAX_PK, NK_PK)


__all__ = ["HMCODE_VERSIONS", "hmcode2020_ratio", "linear_table_k_h"]
