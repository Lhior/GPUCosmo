#!/usr/bin/env python3
# This file is part of GPUCosmo. Copyright (C) 2025-2026 Leonel Medina-Varela.
# SPDX-License-Identifier: GPL-3.0-or-later
# Derived in part from the following, translated to JAX and modified for GPUCosmo:
# - CAMB (https://github.com/cmbant/CAMB), Copyright 2000-2025 Antony Lewis and Anthony Challinor.
#   Licence: LICENSES/CAMB.txt.
# See NOTICE.md.
"""JAX Recfast 1.5.2 — pure jnp ion RHS + lax.scan RK4 (no host freeze)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from gpucosmo.jax.eb.dark_energy import cpl_f_de
import numpy as np
from jax import lax

from gpucosmo.jax.eb import constants as C
from gpucosmo.jax.eb.params import FlatLCDMParams

jax.config.update("jax_enable_x64", True)

_BIG_H = 100.0e3 / C.MPC
_NOT4 = C.MASS_RATIO_HE_H
_LAMBDA, _LAMBDA_HE = 8.2245809, 51.3
_L_H_ION, _L_H_ALPHA = 1.096787737e7, 8.225916453e6
_L_HE1_ION, _L_HE2_ION = 1.98310772e7, 4.389088863e7
_L_HE_2S, _L_HE_2P = 1.66277434e7, 1.71134891e7
_A2P_S, _A2P_T = 1.798287e9, 177.58
_L_HE_2PT, _L_HE_2ST = 1.690871466e7, 1.5985597526e7
_L_HE2ST_ION = 3.8454693845e6
_SIGMA_HE_2PS, _SIGMA_HE_2PT = 1.436289e-22, 1.484872e-22
_LALPHA, _LALPHA_HE = 1.0 / _L_H_ALPHA, 1.0 / _L_HE_2P
_CDB = C.H_P * C.C_LIGHT * (_L_H_ION - _L_H_ALPHA) / C.K_B
_CDB_HE = C.H_P * C.C_LIGHT * (_L_HE1_ION - _L_HE_2S) / C.K_B
_CB1 = C.H_P * C.C_LIGHT * _L_H_ION / C.K_B
_CB1_HE1 = C.H_P * C.C_LIGHT * _L_HE1_ION / C.K_B
_CB1_HE2 = C.H_P * C.C_LIGHT * _L_HE2_ION / C.K_B
_CR = C.TWOPI * (C.M_E / C.H_P) * (C.K_B / C.H_P)
_CK, _CK_HE = _LALPHA**3 / C.EIGHTPI, _LALPHA_HE**3 / C.EIGHTPI
_CL = C.C_LIGHT * C.H_P / (C.K_B * _LALPHA)
_CL_HE = C.C_LIGHT * C.H_P / (C.K_B / _L_HE_2S)
_CT = C.COMPTON_CT / C.MPC_IN_SEC
_BFACT = C.H_P * C.C_LIGHT * (_L_HE_2P - _L_HE_2S) / C.K_B
_H_FRAC, _X_HE_FREEZE = 1.0e-3, 1.0e-8
# CAMB v2 Recfast defaults: recfast_approx_model = "cosmorec"
# (camb/recombination.py). The older "planck" set (fudge_He=0.86, AGauss=-0.14)
# left x_e 1.1% low around last scattering and fed that straight into visibility.
# No global x_e rescaling: the host Recfast (eb/recfast.py) multiplies by 0.99885 to
# cancel its old RK4 bias; carried over here it left x_e 1.1e-3 low at every z (even
# z = 3000, fully ionized), moved the visibility peak 0.017 Mpc early and shrank
# r_s(τ*) by 5e-5: a +0.23σ θ shift against converged CAMB with Planck PR4.
_FU, _FU_HE = 1.125, 0.8472367977
_AG1, _AG2 = -0.1395272483, 0.0729891952
_ZG1, _ZG2 = 7.2813061282, 6.7667038679
_WG1, _WG2 = 0.1638966410, 0.2785834127
# CAMB RECFAST_He_rate_correction (on by default): multiplies dx_He/dz in 1500<z<3000.
_HE_RATE_ZMIN, _HE_RATE_ZMAX = 1500.0, 3000.0
_HE_RATE_A0, _HE_RATE_A1, _HE_RATE_A2 = (
    0.07805480599148856,
    0.899368710079757,
    -3.063896868095767,
)
_HE_RATE_D2, _HE_RATE_D4 = 3.752520187028219, 18.126758791787946
_HE_RATE_Z0, _HE_RATE_WIDTH = 1968.1134791219417, 774.0312440582285
# Fixed RK4 substeps per output Δz (nz=2000 → Δz = 5). The matter-temperature
# equation is stiff right after it leaves the Tmat = Trad tight-coupling regime
# (Compton rate >> H, more so at low Omega_m where H is lower): with 5 substeps
# (Δz = 1) explicit RK4 went unstable there for Omega_m <~ 0.15, spiking Tmat and
# crashing the residual x_e by up to 200x (BAO / taurend errors downstream).
# 10 matches CAMB's x_e to 1.1e-3 over 300 < z < 1500 for 0.048 <= Omega_m <= 1
# (unchanged at 20, 40, 160).
_NSUB = 10
# RECFAST's hydrogen rate switches from the "Saha rate" form to Peebles at x_H = 0.985.
# The Saha-rate equation relaxes x_H to its equilibrium at ∂f/∂x_H ≈ 240 per unit z
# (z ≈ 1570), so explicit RK4 at h = 0.5 (|hλ| ≈ 120, stability limit 2.78) blew up:
# x_e +1.1% / −0.2% vs CAMB over z ≈ 1550–1572, flipping the sign of dκ/dτ there.
# x_H tracks that equilibrium to (dx_eq/dz)/λ ≈ 2e-6, which is what CAMB's adaptive
# DVERK resolves, so it is set algebraically until it crosses 0.985.
_X_H_PEEBLES = 0.985


def _param_floats(params: FlatLCDMParams | dict) -> dict:
    """Pull scalar cosmology fields; preserve tracers (no ``float()`` cast)."""
    if isinstance(params, dict):
        out = {}
        for k, v in params.items():
            if isinstance(v, str):
                continue
            out[k] = v
        return out
    p = params
    keys = (
        "Omega_r", "Omega_m", "Omega_de", "ombh2", "h", "T_CMB", "YHe",
        "f_He", "H0_mpc", "tau_reio", "Omega_b", "Omega_c", "Omega_g", "Omega_nu",
        "Omega_nu_mass", "nu_mass_am", "w0", "wa", "nnu", "mnu", "num_massive_nu", "Omega_k",
    )
    return {k: getattr(p, k, 0.0) for k in keys}


def _as_flatlcdm(params: FlatLCDMParams | dict) -> FlatLCDMParams:
    if isinstance(params, FlatLCDMParams):
        return params
    pf = _param_floats(params)
    return FlatLCDMParams(
        Omega_c=pf["Omega_c"], Omega_b=pf["Omega_b"], h=pf["h"],
        n_s=0.96, sigma8=0.8, T_CMB=pf["T_CMB"], YHe=pf["YHe"],
        tau_reio=pf.get("tau_reio", 0.0544),
        # Dropping these silently gave the thermal background a cosmological
        # constant and massless neutrinos regardless of the requested model.
        w0=pf.get("w0", -1.0),
        wa=pf.get("wa", 0.0),
        nnu=pf.get("nnu", C.DEFAULT_NNU),
        mnu=pf.get("mnu", C.DEFAULT_MNU),
        num_massive_nu=int(pf.get("num_massive_nu", C.DEFAULT_NUM_MASSIVE_NU)),
    )


def _recfast_scalars(pf: dict) -> dict:
    """Derived Recfast scalars; keep JAX tracers (no ``float()`` casts)."""
    Yp = pf["YHe"]
    mu_H = 1.0 / (1.0 - Yp)
    fHe = Yp / (_NOT4 * (1.0 - Yp))
    HO = pf["h"] * _BIG_H
    Nnow = 3.0 * _BIG_H**2 * pf["ombh2"] / (C.EIGHTPI * C.G_NEWTON * mu_H * C.M_H)
    # CAMB State%z_eq counts massive neutrinos as if massless (results.f90:490).
    om_nu_mass = pf.get("Omega_nu_mass", 0.0)
    om_r_eq = pf["Omega_r"] + jnp.sum(jnp.atleast_1d(jnp.asarray(om_nu_mass, dtype=jnp.float64)), axis=-1)  # per block or scalar
    z_eq = pf["Omega_m"] / jnp.maximum(om_r_eq, 1e-30) - 1.0
    return {
        **pf,
        "Nnow": Nnow,
        "fHe": fHe,
        "HO": HO,
        "Tnow": pf["T_CMB"],
        "fu": _FU,
        "OmegaT": pf["Omega_m"],
        "OmegaK": pf.get("Omega_k", 0.0),
        "z_eq": jnp.maximum(z_eq, 1.0),
        "Omega_nu_mass": om_nu_mass,
        "nu_mass_am": pf.get("nu_mass_am", 0.0),
        "w0": pf.get("w0", -1.0),
        "wa": pf.get("wa", 0.0),
    }


def _hz_si(z, st):
    """Hubble in SI, matching CAMB Recfast's ``ainv2/dtauda`` (full cosmology).

    The previous FlatLCDM form dropped massive neutrinos and w0wa, which left H
    ~1.5% low near recombination and drove the 1.1% x_e / visibility bias that
    capped every Cls gate at ~2%.
    """
    a = 1.0 / (1.0 + z)
    w0, wa = st["w0"], st["wa"]
    f_de = cpl_f_de(a, w0, wa)
    from gpucosmo.jax.eb.massive_nu import nu_rho_p_omega

    nu_term = nu_rho_p_omega(st, a)[0] * a ** (-4)
    ea = jnp.sqrt(
        st["Omega_r"] * a ** (-4)
        + st["Omega_m"] * a ** (-3)
        + nu_term
        + st["OmegaK"] * a ** (-2)
        + st["Omega_de"] * f_de
    )
    return st["HO"] * ea


def ion_rhs(z, y, st):
    """CAMB ION: dy/dz with y=(x_H, x_He, a*Tmat). Pure jnp (Heflag=6 path)."""
    x_H = jnp.clip(y[0], 0.0, 1.0)
    x_He = jnp.clip(y[1], 0.0, 1.0)
    aTmat = jnp.maximum(y[2], 1e-30)
    ainv = 1.0 + z
    Tmat = jnp.maximum(ainv * aTmat, 1.0)
    fHe, Nnow, HO, Tnow, fu = st["fHe"], st["Nnow"], st["HO"], st["Tnow"], st["fu"]
    x = x_H + fHe * x_He
    n = Nnow * ainv**3
    n_He = fHe * Nnow * ainv**3
    Trad = Tnow * ainv
    Hz = jnp.maximum(_hz_si(z, st), 1e-30)

    a_PPB, b_PPB, c_PPB, d_PPB = 4.309, -0.6166, 0.6703, 0.5300
    Rdown = 1.0e-19 * a_PPB * (Tmat / 1.0e4) ** b_PPB / (
        1.0 + c_PPB * (Tmat / 1.0e4) ** d_PPB
    )
    Rup = Rdown * (_CR * Tmat) ** 1.5 * jnp.exp(-_CDB / jnp.maximum(Tmat, 1.0))

    a_VF, b_VF = 10.0 ** (-16.744), 0.711
    T_0, T_1 = 10.0**0.477121, 10.0**5.114
    sq_0, sq_1 = jnp.sqrt(Tmat / T_0), jnp.sqrt(Tmat / T_1)
    Rdown_He = a_VF / (sq_0 * (1.0 + sq_0) ** (1.0 - b_VF))
    Rdown_He = Rdown_He / (1.0 + sq_1) ** (1.0 + b_VF)
    Rup_He = 4.0 * Rdown_He * (_CR * Tmat) ** 1.5 * jnp.exp(
        -_CDB_HE / jnp.maximum(Tmat, 1.0)
    )
    # Enters f2 only as (1 + A·He_Boltz)/(1 + B·He_Boltz), saturated long before the cap
    # (Tmat ≲ 23 K). At e^680 its T-derivative overflowed to inf, and the NaN reached
    # every reverse-mode gradient through the scan (0·NaN at the cold late steps).
    He_Boltz = jnp.exp(jnp.minimum(_BFACT / jnp.maximum(Tmat, 1.0), 300.0))

    z_scale = Tnow / C.COBE_CMB_TEMP * ainv - 1.0
    K = (_CK / Hz) * (
        1.0
        + _AG1 * jnp.exp(-(((jnp.log(1.0 + z_scale) - _ZG1) / _WG1) ** 2))
        + _AG2 * jnp.exp(-(((jnp.log(1.0 + z_scale) - _ZG2) / _WG2) ** 2))
    )

    a_trip, b_trip = 10.0 ** (-16.306), 0.761
    Rdown_trip = a_trip / (sq_0 * (1.0 + sq_0) ** (1.0 - b_trip))
    Rdown_trip = Rdown_trip / (1.0 + sq_1) ** (1.0 + b_trip)
    Rup_trip = (
        Rdown_trip
        * jnp.exp(-C.H_P * C.C_LIGHT * _L_HE2ST_ION / (C.K_B * jnp.maximum(Tmat, 1.0)))
        * (_CR * Tmat) ** 1.5
        * (4.0 / 3.0)
    )

    he_active = (x_He >= _X_HE_FREEZE) & (x_He <= 0.98)
    tauHe_s = _A2P_S * _CK_HE * 3.0 * n_He * (1.0 - x_He) / Hz
    pHe_s = (1.0 - jnp.exp(-tauHe_s)) / jnp.maximum(tauHe_s, 1e-30)
    K_He0 = _CK_HE / Hz
    K_He_base = 1.0 / jnp.maximum(_A2P_S * pHe_s * 3.0 * n_He * (1.0 - x_He), 1e-30)
    Doppler = C.C_LIGHT * _L_HE_2P * jnp.sqrt(
        jnp.maximum(2.0 * C.K_B * Tmat / (C.M_H * _NOT4 * C.C_LIGHT**2), 0.0)
    )
    gamma_2Ps = (
        3.0
        * _A2P_S
        * fHe
        * (1.0 - x_He)
        * C.C_LIGHT**2
        / (
            jnp.sqrt(C.PI)
            * _SIGMA_HE_2PS
            * C.EIGHTPI
            * jnp.maximum(Doppler, 1e-30)
            * jnp.maximum(1.0 - x_H, 1e-30)
        )
        / (C.C_LIGHT * _L_HE_2P) ** 2
    )
    AHcon = _A2P_S / (1.0 + 0.36 * (jnp.maximum(gamma_2Ps, 1e-30) ** _FU_HE))
    K_He_dop = 1.0 / jnp.maximum(
        (_A2P_S * pHe_s + AHcon) * 3.0 * n_He * (1.0 - x_He), 1e-30
    )
    K_He_on = jnp.where(x_H < 0.9999999, K_He_dop, K_He_base)
    K_He = jnp.where(he_active, K_He_on, K_He0)

    tauHe_t = _A2P_T * n_He * (1.0 - x_He) * 3.0 / (C.EIGHTPI * Hz * _L_HE_2PT**3)
    pHe_t = (1.0 - jnp.exp(-tauHe_t)) / jnp.maximum(tauHe_t, 1e-30)
    CL_PSt = C.H_P * C.C_LIGHT * (_L_HE_2PT - _L_HE_2ST) / C.K_B
    Cf_simple = _A2P_T * pHe_t * jnp.exp(-CL_PSt / jnp.maximum(Tmat, 1.0))
    Cf_simple = Cf_simple / jnp.maximum(Rup_trip + Cf_simple, 1e-30)
    Doppler_t = C.C_LIGHT * _L_HE_2PT * jnp.sqrt(
        jnp.maximum(2.0 * C.K_B * Tmat / (C.M_H * _NOT4 * C.C_LIGHT**2), 0.0)
    )
    gamma_2Pt = (
        3.0
        * _A2P_T
        * fHe
        * (1.0 - x_He)
        * C.C_LIGHT**2
        / (
            jnp.sqrt(C.PI)
            * _SIGMA_HE_2PT
            * C.EIGHTPI
            * jnp.maximum(Doppler_t, 1e-30)
            * jnp.maximum(1.0 - x_H, 1e-30)
        )
        / (C.C_LIGHT * _L_HE_2PT) ** 2
    )
    AHcon_t = _A2P_T / (1.0 + 0.66 * jnp.maximum(gamma_2Pt, 1e-30) ** 0.9) / 3.0
    Cf_dop = (_A2P_T * pHe_t + AHcon_t) * jnp.exp(-CL_PSt / jnp.maximum(Tmat, 1.0))
    Cf_dop = Cf_dop / jnp.maximum(Rup_trip + Cf_dop, 1e-30)
    CfHe_t = jnp.where(he_active, jnp.where(x_H > 0.99999, Cf_simple, Cf_dop), 0.0)

    num = (
        x * x_H * n * Rdown - Rup * (1.0 - x_H) * jnp.exp(-_CL / jnp.maximum(Tmat, 1.0))
    ) * (1.0 + K * _LAMBDA * n * (1.0 - x_H))
    den = Hz * ainv * (
        1.0 / fu + K * _LAMBDA * n * (1.0 - x_H) / fu + K * Rup * n * (1.0 - x_H)
    )
    f1_full = num / jnp.maximum(den, 1e-30)
    # Above x_H = 0.985 hydrogen is algebraic (Saha for x_H > 0.99, then the
    # equilibrium of RECFAST's "Saha rate" equation, set in run_recfast_jax's step).
    f1 = jnp.where(x_H > _X_H_PEEBLES, 0.0, f1_full)

    f2_base = (
        (x * x_He * n * Rdown_He - Rup_He * (1.0 - x_He) * jnp.exp(-_CL_HE / jnp.maximum(Tmat, 1.0)))
        * (1.0 + K_He * _LAMBDA_HE * n_He * (1.0 - x_He) * He_Boltz)
    ) / (
        Hz
        * ainv
        * (1.0 + K_He * (_LAMBDA_HE + Rup_He) * n_He * (1.0 - x_He) * He_Boltz)
    )
    f2_trip = (
        x * x_He * n * Rdown_trip
        - (1.0 - x_He)
        * 3.0
        * Rup_trip
        * jnp.exp(-C.H_P * C.C_LIGHT * _L_HE_2ST / (C.K_B * jnp.maximum(Tmat, 1.0)))
    ) * CfHe_t / (Hz * ainv)
    f2 = jnp.where(x_He < _X_HE_FREEZE, 0.0, f2_base + jnp.where(he_active, f2_trip, 0.0))
    # CAMB RECFAST_He_rate_correction: scale dx_He/dz between z=1500 and 3000.
    he_u = (z_scale - _HE_RATE_Z0) / _HE_RATE_WIDTH
    he_u2 = he_u * he_u
    he_rate_factor = 1.0 + (
        _HE_RATE_A0 + _HE_RATE_A1 * he_u + _HE_RATE_A2 * he_u2
    ) / (1.0 + _HE_RATE_D2 * he_u2 + _HE_RATE_D4 * he_u2 * he_u2)
    he_rate_factor = jnp.where(
        (z_scale > _HE_RATE_ZMIN) & (z_scale < _HE_RATE_ZMAX),
        he_rate_factor,
        1.0,
    )
    f2 = f2 * he_rate_factor

    timeTh = (1.0 / (_CT * Trad**4)) * (1.0 + x + fHe) / jnp.maximum(x, 1e-30)
    timeH = 2.0 / (3.0 * HO * ainv**1.5)
    dHdz = (HO**2 / 2.0 / Hz) * (
        4.0 * ainv**3 / (1.0 + st["z_eq"]) * st["OmegaT"]
        + 3.0 * st["OmegaT"] * ainv**2
        + 2.0 * st["OmegaK"] * ainv
    )
    eps = Hz * (1.0 + x + fHe) / (_CT * Trad**3 * jnp.maximum(x, 1e-30))
    daT_tc = (
        (Tnow - aTmat) / ainv
        + eps * ((1.0 + fHe) / (1.0 + fHe + x)) * ((f1 + fHe * f2) / jnp.maximum(x, 1e-30)) / ainv
        - eps * dHdz / (Hz * ainv)
        + 3.0 * eps / ainv**2
    )
    daT_comp = (
        _CT * (Trad**4) * x * (aTmat - Tnow) / (Hz * (1.0 + x + fHe)) + aTmat
    ) / ainv
    daT = jnp.where(timeTh < _H_FRAC * timeH, daT_tc, daT_comp)
    return jnp.array([f1, f2, daT])


def _rk4_step(z0, y0, h, st):
    k1 = ion_rhs(z0, y0, st)
    k2 = ion_rhs(z0 + 0.5 * h, y0 + 0.5 * h * k1, st)
    k3 = ion_rhs(z0 + 0.5 * h, y0 + 0.5 * h * k2, st)
    k4 = ion_rhs(z0 + h, y0 + h * k3, st)
    y = y0 + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return jnp.array(
        [jnp.clip(y[0], 0.0, 1.0), jnp.clip(y[1], 0.0, 1.0), jnp.maximum(y[2], 1e-30)]
    )


def _integrate(z_cur, y, z_end, st, nsub=_NSUB):
    h = (z_end - z_cur) / nsub

    def body(i, yy):
        return _rk4_step(z_cur + i * h, yy, h, st)

    return lax.fori_loop(0, nsub, body, y)


def _h_equilibrium(z, y, st):
    """x_H where the Saha-rate equation x·x_H·n·Rdown = Rup·(1 − x_H)·e^(−CL/Tmat) holds.

    Saha at the matter temperature with x = x_H + fHe·x_He: x_H² + (c + s)·x_H − s = 0.
    """
    ainv = 1.0 + z
    Tmat = jnp.maximum(ainv * jnp.maximum(y[2], 1e-30), 1.0)
    n = st["Nnow"] * ainv**3
    s = jnp.exp(1.5 * jnp.log(_CR * Tmat) - (_CDB + _CL) / Tmat) / n
    b = st["fHe"] * jnp.clip(y[1], 0.0, 1.0) + s
    # s underflows to 0 at low z (unused there, but jnp.where still differentiates it).
    root = b + jnp.sqrt(jnp.maximum(b * b + 4.0 * s, 1e-300))
    return jnp.clip(2.0 * s / root, 0.0, 1.0)


def _saha_y(z, st):
    """GET_INIT-like Saha state at redshift z (jnp.where regimes)."""
    Tnow, Nnow, fHe = st["Tnow"], st["Nnow"], st["fHe"]
    z_scale = Tnow / C.COBE_CMB_TEMP * (z + 1.0) - 1.0
    ainv = 1.0 + z
    rhs_he2 = jnp.exp(1.5 * jnp.log(_CR * Tnow / ainv) - _CB1_HE2 / (Tnow * ainv)) / Nnow
    rhs_he1 = 4.0 * jnp.exp(1.5 * jnp.log(_CR * Tnow / ainv) - _CB1_HE1 / (Tnow * ainv)) / Nnow
    x_He0 = 0.5 * (jnp.sqrt((rhs_he1 - 1.0) ** 2 + 4.0 * (1.0 + fHe) * rhs_he1) - (rhs_he1 - 1.0))
    x_He0 = jnp.clip((x_He0 - 1.0) / fHe, 0.0, 1.0)
    rhs_h = jnp.exp(1.5 * jnp.log(_CR * Tnow / ainv) - _CB1 / (Tnow * ainv)) / Nnow
    x_H0 = jnp.clip(0.5 * (jnp.sqrt(jnp.maximum(rhs_h**2 + 4.0 * rhs_h, 1e-300)) - rhs_h), 0.0, 1.0)
    y = jnp.where(
        z_scale > 3500.0,
        jnp.array([1.0, 1.0, Tnow]),
        jnp.where(
            z_scale > 2000.0,
            jnp.array([1.0, x_He0, Tnow]),
            jnp.array([x_H0, 0.0, Tnow]),
        ),
    )
    return y


def run_recfast_jax(
    params: FlatLCDMParams | dict,
    *,
    z_initial: float = 1.0e4,
    z_final: float = 0.0,
    nz: int = 2000,
    split_high_z: bool = True,
):
    """Integrate Recfast on device; return (z_desc, xe, Tmat).

    ``split_high_z=False`` scans every step (reference for the fast path).
    """
    st = _recfast_scalars(_param_floats(params))
    fHe, Tnow, Nnow = st["fHe"], st["Tnow"], st["Nnow"]
    delta_z = (z_initial - z_final) / nz
    z_out = z_initial - (jnp.arange(nz, dtype=jnp.float64) + 1.0) * delta_z
    y0 = _saha_y(z_initial, st)

    def step(carry, z_end):
        y, z_cur = carry
        z_scale = Tnow / C.COBE_CMB_TEMP * (1.0 + z_end) - 1.0
        ainv = 1.0 + z_end
        # Algebraic hydrogen for 0.985 < x_H <= 0.99 (see _X_H_PEEBLES). In the step
        # where the equilibrium crosses 0.985, Peebles starts at the crossing redshift
        # (linear in the equilibrium; x_He is frozen and a·Tmat ≈ Tnow over the step).
        x_eq_end = _h_equilibrium(z_end, y, st)
        in_mid = (y[0] > _X_H_PEEBLES) & (y[0] <= 0.99)
        cross = in_mid & (x_eq_end <= _X_H_PEEBLES)
        frac = jnp.clip((y[0] - _X_H_PEEBLES) / jnp.maximum(y[0] - x_eq_end, 1e-30), 0.0, 1.0)
        z_start = jnp.where(cross, z_cur + frac * (z_end - z_cur), z_cur)
        y_rk = _integrate(z_start, jnp.where(cross, y.at[0].set(_X_H_PEEBLES), y), z_end, st)
        y_mid = jnp.where(cross, y_rk, y_rk.at[0].set(_h_equilibrium(z_end, y_rk, st)))

        rhs_he2 = jnp.exp(1.5 * jnp.log(_CR * Tnow / ainv) - _CB1_HE2 / (Tnow * ainv)) / Nnow
        x0_he2 = 0.5 * (
            jnp.sqrt((rhs_he2 - 1.0 - fHe) ** 2 + 4.0 * (1.0 + 2.0 * fHe) * rhs_he2)
            - (rhs_he2 - 1.0 - fHe)
        )
        rhs_he1 = 4.0 * jnp.exp(1.5 * jnp.log(_CR * Tnow / ainv) - _CB1_HE1 / (Tnow * ainv)) / Nnow
        x_He0 = 0.5 * (jnp.sqrt((rhs_he1 - 1.0) ** 2 + 4.0 * (1.0 + fHe) * rhs_he1) - (rhs_he1 - 1.0))
        x_He0 = jnp.clip((x_He0 - 1.0) / fHe, 0.0, 1.0)
        rhs_h = jnp.exp(1.5 * jnp.log(_CR * Tnow / ainv) - _CB1 / (Tnow * ainv)) / Nnow
        x_H0 = jnp.clip(0.5 * (jnp.sqrt(jnp.maximum(rhs_h**2 + 4.0 * rhs_h, 1e-300)) - rhs_h), 0.0, 1.0)

        y_full = jnp.array([1.0, 1.0, Tnow])
        y_he1 = jnp.array([1.0, x_He0, Tnow])
        y_h = y_rk.at[0].set(x_H0)

        y_new = jnp.where(
            z_scale > 8000.0,
            y_full,
            jnp.where(
                z_scale > 5000.0,
                y_full,
                jnp.where(
                    z_scale > 3500.0,
                    y_full,
                    jnp.where(
                        y[1] > 0.99,
                        y_he1,
                        jnp.where(y[0] > 0.99, y_h, jnp.where(in_mid, y_mid, y_rk)),
                    ),
                ),
            ),
        )
        x0 = jnp.where(
            z_scale > 8000.0,
            1.0 + 2.0 * fHe,
            jnp.where(
                z_scale > 5000.0,
                x0_he2,
                jnp.where(z_scale > 3500.0, 1.0 + fHe, y_new[0] + fHe * y_new[1]),
            ),
        )
        xe = x0
        tm = ainv * y_new[2]
        return (y_new, z_end), (xe, tm)

    # Above z_scale = 3500 every step returns the algebraic fully-ionized /
    # Saha-helium state and carries y = [1, 1, Tnow], so those steps need no RK4.
    # The z grid is static, so split it on the host: vectorize the high-z part and
    # scan only the rest (~1/3 fewer sequential RK4 steps; identical output).
    z_out_np = z_initial - (np.arange(nz, dtype=np.float64) + 1.0) * delta_z
    scale = float(C.COBE_CMB_TEMP)
    try:
        tnow_f = float(Tnow)
    except (TypeError, jax.errors.ConcretizationTypeError):
        tnow_f = scale
    n_hi = int(np.sum(tnow_f / scale * (1.0 + z_out_np) - 1.0 > 3500.0))
    if split_high_z and 0 < n_hi < nz:
        z_hi = z_out[:n_hi]
        z_scale_hi = Tnow / C.COBE_CMB_TEMP * (1.0 + z_hi) - 1.0
        ainv_hi = 1.0 + z_hi
        rhs_he2 = jnp.exp(1.5 * jnp.log(_CR * Tnow / ainv_hi) - _CB1_HE2 / (Tnow * ainv_hi)) / Nnow
        x0_he2 = 0.5 * (
            jnp.sqrt((rhs_he2 - 1.0 - fHe) ** 2 + 4.0 * (1.0 + 2.0 * fHe) * rhs_he2)
            - (rhs_he2 - 1.0 - fHe)
        )
        x0_hi = jnp.where(
            z_scale_hi > 8000.0, 1.0 + 2.0 * fHe,
            jnp.where(z_scale_hi > 5000.0, x0_he2, 1.0 + fHe),
        )
        xe_hi = x0_hi
        tm_hi = ainv_hi * Tnow
        y_start = jnp.array([1.0, 1.0, Tnow])
        (_, _), (xe_lo, tm_lo) = lax.scan(step, (y_start, z_out[n_hi - 1]), z_out[n_hi:])
        return z_out, jnp.concatenate([xe_hi, xe_lo]), jnp.concatenate([tm_hi, tm_lo])
    (_, _), (xe, tm) = lax.scan(step, (y0, jnp.asarray(z_initial, dtype=jnp.float64)), z_out)
    return z_out, xe, tm


def compare_xe_vs_numpy(params: FlatLCDMParams | dict, *, nz: int = 2000) -> float:
    """Median relative |xe_jax - xe_np| / xe_np on the shared z grid."""
    import numpy as np

    from gpucosmo.jax.eb.background import build_background
    from gpucosmo.jax.eb.recfast import run_recfast

    p = _as_flatlcdm(params)
    bg = build_background(p)
    z_np, xe_np, _ = run_recfast(bg, nz=nz)
    z_j, xe_j, _ = (np.asarray(x) for x in run_recfast_jax(p, nz=nz))
    xe_j_i = np.interp(z_np[::-1], z_j[::-1], xe_j[::-1])[::-1]
    return float(np.median(np.abs(xe_j_i - xe_np) / np.maximum(np.abs(xe_np), 1e-30)))


__all__ = ["ion_rhs", "run_recfast_jax", "compare_xe_vs_numpy", "_param_floats", "_as_flatlcdm"]
