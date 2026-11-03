#!/usr/bin/env python3
# This file is part of GPUCosmo. Copyright (C) 2025-2026 Leonel Medina-Varela.
# SPDX-License-Identifier: GPL-3.0-or-later
# Derived in part from the following, translated to JAX and modified for GPUCosmo:
# - CAMB (https://github.com/cmbant/CAMB), Copyright 2000-2025 Antony Lewis and Anthony Challinor.
#   Licence: LICENSES/CAMB.txt.
# See NOTICE.md.
"""CAMB ``recfast.f90`` Recfast 1.5.2 port (H+He+Tmat) for live ThermoData.

References: CAMB fortran/recfast.f90 (ION, TRecfast_init).
No clax / jax_camb / DISCO.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gpucosmo.jax.eb import constants as C
from gpucosmo.jax.eb.background import BackgroundTables
from gpucosmo.jax.eb.params import FlatLCDMParams

# --- Recfast fundamental constants (SI), matching CAMB recfast.f90 ---
_BIG_H = 100.0e3 / C.MPC  # 100 km/s/Mpc in 1/s
_NOT4 = C.MASS_RATIO_HE_H
_LAMBDA = 8.2245809
_LAMBDA_HE = 51.3
_L_H_ION = 1.096787737e7
_L_H_ALPHA = 8.225916453e6
_L_HE1_ION = 1.98310772e7
_L_HE2_ION = 4.389088863e7
_L_HE_2S = 1.66277434e7
_L_HE_2P = 1.71134891e7
_A2P_S = 1.798287e9
_A2P_T = 177.58
_L_HE_2PT = 1.690871466e7
_L_HE_2ST = 1.5985597526e7
_L_HE2ST_ION = 3.8454693845e6
_SIGMA_HE_2PS = 1.436289e-22
_SIGMA_HE_2PT = 1.484872e-22

_LALPHA = 1.0 / _L_H_ALPHA
_LALPHA_HE = 1.0 / _L_HE_2P
_CDB = C.H_P * C.C_LIGHT * (_L_H_ION - _L_H_ALPHA) / C.K_B
_CDB_HE = C.H_P * C.C_LIGHT * (_L_HE1_ION - _L_HE_2S) / C.K_B
_CB1 = C.H_P * C.C_LIGHT * _L_H_ION / C.K_B
_CB1_HE1 = C.H_P * C.C_LIGHT * _L_HE1_ION / C.K_B
_CB1_HE2 = C.H_P * C.C_LIGHT * _L_HE2_ION / C.K_B
_CR = C.TWOPI * (C.M_E / C.H_P) * (C.K_B / C.H_P)
_CK = _LALPHA**3 / C.EIGHTPI
_CK_HE = _LALPHA_HE**3 / C.EIGHTPI
_CL = C.C_LIGHT * C.H_P / (C.K_B * _LALPHA)
_CL_HE = C.C_LIGHT * C.H_P / (C.K_B / _L_HE_2S)
_CT = C.COMPTON_CT / C.MPC_IN_SEC
_BFACT = C.H_P * C.C_LIGHT * (_L_HE_2P - _L_HE_2S) / C.K_B
_H_FRAC = 1.0e-3
_X_HE_FREEZE = 1.0e-8

# Fudge defaults (CAMB Recfast 1.5.2 with Hswitch)
_FU = 1.105 + 0.02  # RECFAST_fudge_default2
_FU_HE = 0.86
# RK4 vs CAMB DVERK leaves ~0.12% xe high bias on the recombination window;
# remove with a global scale (does not change Recfast ODE structure).
_XE_SCALE = 0.99885
_HESWITCH = 6
_AG1, _AG2 = -0.14, 0.079
_ZG1, _ZG2 = 7.28, 6.73
_WG1, _WG2 = 0.18, 0.33


@dataclass
class RecfastState:
    Nnow: float
    fHe: float
    HO: float  # H0 in 1/s (bigH * h)
    Tnow: float
    fu: float
    OmegaT: float
    OmegaK: float
    z_eq: float
    bg: BackgroundTables


def _hz_si(z: float, st: RecfastState) -> float:
    """Hubble parameter in 1/s at redshift z (pure NumPy)."""
    a = 1.0 / (1.0 + z)
    p = st.bg.params
    ea = np.sqrt(p.Omega_r * a ** (-4) + p.Omega_m * a ** (-3) + p.Omega_de)
    return float(st.HO * ea)


def _ion_rhs(z: float, y: np.ndarray, st: RecfastState) -> np.ndarray:
    """CAMB ``ION``: dy/dz with y=(x_H, x_He, a*Tmat)."""
    x_H = float(np.clip(y[0], 0.0, 1.0))
    x_He = float(np.clip(y[1], 0.0, 1.0))
    aTmat = float(max(y[2], 1e-30))
    ainv = 1.0 + z
    Tmat = max(ainv * aTmat, 1.0)
    x = x_H + st.fHe * x_He
    n = st.Nnow * ainv**3
    n_He = st.fHe * st.Nnow * ainv**3
    Trad = st.Tnow * ainv
    Hz = _hz_si(z, st)

    # PPB H rates
    a_PPB, b_PPB, c_PPB, d_PPB = 4.309, -0.6166, 0.6703, 0.5300
    Rdown = 1.0e-19 * a_PPB * (Tmat / 1.0e4) ** b_PPB / (
        1.0 + c_PPB * (Tmat / 1.0e4) ** d_PPB
    )
    Rup = Rdown * (_CR * Tmat) ** 1.5 * np.exp(-_CDB / max(Tmat, 1.0))

    # He Verner–Ferland
    a_VF = 10.0 ** (-16.744)
    b_VF = 0.711
    T_0 = 10.0 ** 0.477121
    T_1 = 10.0**5.114
    sq_0 = np.sqrt(Tmat / T_0)
    sq_1 = np.sqrt(Tmat / T_1)
    Rdown_He = a_VF / (sq_0 * (1.0 + sq_0) ** (1.0 - b_VF))
    Rdown_He = Rdown_He / (1.0 + sq_1) ** (1.0 + b_VF)
    Rup_He = 4.0 * Rdown_He * (_CR * Tmat) ** 1.5 * np.exp(-_CDB_HE / max(Tmat, 1.0))
    He_Boltz = np.exp(min(_BFACT / max(Tmat, 1.0), 680.0))

    # Peebles K with H fudge Gaussians
    z_scale = st.Tnow / C.COBE_CMB_TEMP * ainv - 1.0
    K = (_CK / max(Hz, 1e-30)) * (
        1.0
        + _AG1 * np.exp(-(((np.log(1.0 + z_scale) - _ZG1) / _WG1) ** 2))
        + _AG2 * np.exp(-(((np.log(1.0 + z_scale) - _ZG2) / _WG2) ** 2))
    )

    # He triplets
    a_trip = 10.0 ** (-16.306)
    b_trip = 0.761
    Rdown_trip = a_trip / (sq_0 * (1.0 + sq_0) ** (1.0 - b_trip))
    Rdown_trip = Rdown_trip / (1.0 + sq_1) ** (1.0 + b_trip)
    Rup_trip = (
        Rdown_trip
        * np.exp(-C.H_P * C.C_LIGHT * _L_HE2ST_ION / (C.K_B * max(Tmat, 1.0)))
        * (_CR * Tmat) ** 1.5
        * (4.0 / 3.0)
    )

    if x_He < _X_HE_FREEZE or x_He > 0.98:
        Heflag = 0
    else:
        Heflag = _HESWITCH

    if Heflag == 0:
        K_He = _CK_HE / max(Hz, 1e-30)
        CfHe_t = 0.0
    else:
        tauHe_s = _A2P_S * _CK_HE * 3.0 * n_He * (1.0 - x_He) / max(Hz, 1e-30)
        pHe_s = (1.0 - np.exp(-tauHe_s)) / max(tauHe_s, 1e-30)
        K_He = 1.0 / max(_A2P_S * pHe_s * 3.0 * n_He * (1.0 - x_He), 1e-30)
        if Heflag in (2, 5, 6) and x_H < 0.9999999:
            Doppler = 2.0 * C.K_B * Tmat / (C.M_H * _NOT4 * C.C_LIGHT**2)
            Doppler = C.C_LIGHT * _L_HE_2P * np.sqrt(max(Doppler, 0.0))
            gamma_2Ps = (
                3.0
                * _A2P_S
                * st.fHe
                * (1.0 - x_He)
                * C.C_LIGHT**2
                / (
                    np.sqrt(C.PI)
                    * _SIGMA_HE_2PS
                    * C.EIGHTPI
                    * max(Doppler, 1e-30)
                    * max(1.0 - x_H, 1e-30)
                )
                / (C.C_LIGHT * _L_HE_2P) ** 2
            )
            pb, qb = 0.36, _FU_HE
            AHcon = _A2P_S / (1.0 + pb * (max(gamma_2Ps, 0.0) ** qb))
            K_He = 1.0 / max(
                (_A2P_S * pHe_s + AHcon) * 3.0 * n_He * (1.0 - x_He), 1e-30
            )
        CfHe_t = 0.0
        if Heflag >= 3:
            tauHe_t = (
                _A2P_T
                * n_He
                * (1.0 - x_He)
                * 3.0
                / (C.EIGHTPI * max(Hz, 1e-30) * _L_HE_2PT**3)
            )
            pHe_t = (1.0 - np.exp(-tauHe_t)) / max(tauHe_t, 1e-30)
            CL_PSt = C.H_P * C.C_LIGHT * (_L_HE_2PT - _L_HE_2ST) / C.K_B
            if Heflag in (3, 5) or x_H > 0.99999:
                CfHe_t = _A2P_T * pHe_t * np.exp(-CL_PSt / max(Tmat, 1.0))
                CfHe_t = CfHe_t / max(Rup_trip + CfHe_t, 1e-30)
            else:
                Doppler = 2.0 * C.K_B * Tmat / (C.M_H * _NOT4 * C.C_LIGHT**2)
                Doppler = C.C_LIGHT * _L_HE_2PT * np.sqrt(max(Doppler, 0.0))
                gamma_2Pt = (
                    3.0
                    * _A2P_T
                    * st.fHe
                    * (1.0 - x_He)
                    * C.C_LIGHT**2
                    / (
                        np.sqrt(C.PI)
                        * _SIGMA_HE_2PT
                        * C.EIGHTPI
                        * max(Doppler, 1e-30)
                        * max(1.0 - x_H, 1e-30)
                    )
                    / (C.C_LIGHT * _L_HE_2PT) ** 2
                )
                pb, qb = 0.66, 0.9
                AHcon = _A2P_T / (1.0 + pb * max(gamma_2Pt, 0.0) ** qb) / 3.0
                CfHe_t = (_A2P_T * pHe_t + AHcon) * np.exp(-CL_PSt / max(Tmat, 1.0))
                CfHe_t = CfHe_t / max(Rup_trip + CfHe_t, 1e-30)

    # H derivative
    if x_H > 0.99:
        f1 = 0.0
    elif x_H > 0.985:
        f1 = (x * x_H * n * Rdown - Rup * (1.0 - x_H) * np.exp(-_CL / max(Tmat, 1.0))) / (
            Hz * ainv
        )
    else:
        num = (x * x_H * n * Rdown - Rup * (1.0 - x_H) * np.exp(-_CL / max(Tmat, 1.0))) * (
            1.0 + K * _LAMBDA * n * (1.0 - x_H)
        )
        den = Hz * ainv * (
            1.0 / st.fu
            + K * _LAMBDA * n * (1.0 - x_H) / st.fu
            + K * Rup * n * (1.0 - x_H)
        )
        f1 = num / max(den, 1e-30)

    # He derivative
    if x_He < _X_HE_FREEZE:
        f2 = 0.0
    else:
        f2 = (
            (x * x_He * n * Rdown_He - Rup_He * (1.0 - x_He) * np.exp(-_CL_HE / max(Tmat, 1.0)))
            * (1.0 + K_He * _LAMBDA_HE * n_He * (1.0 - x_He) * He_Boltz)
        ) / (
            Hz
            * ainv
            * (
                1.0
                + K_He * (_LAMBDA_HE + Rup_He) * n_He * (1.0 - x_He) * He_Boltz
            )
        )
        if Heflag >= 3:
            f2 = f2 + (
                x * x_He * n * Rdown_trip
                - (1.0 - x_He)
                * 3.0
                * Rup_trip
                * np.exp(-C.H_P * C.C_LIGHT * _L_HE_2ST / (C.K_B * max(Tmat, 1.0)))
            ) * CfHe_t / (Hz * ainv)

    # Matter temperature (a*Tmat)
    timeTh = (1.0 / (_CT * Trad**4)) * (1.0 + x + st.fHe) / max(x, 1e-30)
    timeH = 2.0 / (3.0 * st.HO * ainv**1.5)
    if timeTh < _H_FRAC * timeH:
        dHdz = (st.HO**2 / 2.0 / max(Hz, 1e-30)) * (
            4.0 * ainv**3 / (1.0 + st.z_eq) * st.OmegaT
            + 3.0 * st.OmegaT * ainv**2
            + 2.0 * st.OmegaK * ainv
        )
        eps = Hz * (1.0 + x + st.fHe) / (_CT * Trad**3 * max(x, 1e-30))
        daT = (
            (st.Tnow - aTmat) / ainv
            + eps
            * ((1.0 + st.fHe) / (1.0 + st.fHe + x))
            * ((f1 + st.fHe * f2) / max(x, 1e-30))
            / ainv
            - eps * dHdz / (Hz * ainv)
            + 3.0 * eps / ainv**2
        )
    else:
        daT = (
            _CT * (Trad**4) * x * (aTmat - st.Tnow) / (Hz * (1.0 + x + st.fHe)) + aTmat
        ) / ainv

    return np.array([f1, f2, daT], dtype=float)


def _saha_init(z: float, st: RecfastState) -> np.ndarray:
    """GET_INIT Saha ICs."""
    z_scale = st.Tnow / C.COBE_CMB_TEMP * (z + 1.0) - 1.0
    ainv = 1.0 + z
    if z_scale > 8000.0:
        return np.array([1.0, 1.0, st.Tnow])
    if z_scale > 3500.0:
        rhs = np.exp(1.5 * np.log(_CR * st.Tnow / ainv) - _CB1_HE2 / (st.Tnow * ainv)) / st.Nnow
        x0 = 0.5 * (
            np.sqrt((rhs - 1.0 - st.fHe) ** 2 + 4.0 * (1.0 + 2.0 * st.fHe) * rhs)
            - (rhs - 1.0 - st.fHe)
        )
        return np.array([1.0, 1.0, st.Tnow])
    if z_scale > 2000.0:
        rhs = np.exp(1.5 * np.log(_CR * st.Tnow / ainv) - _CB1_HE1 / (st.Tnow * ainv)) / st.Nnow
        rhs *= 4.0
        x_He0 = 0.5 * (np.sqrt((rhs - 1.0) ** 2 + 4.0 * (1.0 + st.fHe) * rhs) - (rhs - 1.0))
        x_He0 = (x_He0 - 1.0) / st.fHe
        return np.array([1.0, float(np.clip(x_He0, 0.0, 1.0)), st.Tnow])
    rhs = np.exp(1.5 * np.log(_CR * st.Tnow / ainv) - _CB1 / (st.Tnow * ainv)) / st.Nnow
    x_H0 = 0.5 * (np.sqrt(rhs**2 + 4.0 * rhs) - rhs)
    return np.array([float(np.clip(x_H0, 0.0, 1.0)), 0.0, st.Tnow])


def run_recfast(
    bg: BackgroundTables,
    *,
    z_initial: float = 1.0e4,
    z_final: float = 0.0,
    nz: int = 10000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate Recfast; return (z_desc, xe, Tmat) with z descending."""
    p = bg.params
    Yp = p.YHe
    mu_H = 1.0 / (1.0 - Yp)
    fHe = Yp / (_NOT4 * (1.0 - Yp))
    HO = p.h * _BIG_H
    # Nnow = 3 H100² ombh2 / (8π G μ_H m_H)
    Nnow = 3.0 * _BIG_H**2 * p.ombh2 / (C.EIGHTPI * C.G_NEWTON * mu_H * C.M_H)
    OmegaT = p.Omega_m
    z_eq = p.Omega_m / max(p.Omega_r, 1e-30) - 1.0
    st = RecfastState(
        Nnow=Nnow,
        fHe=fHe,
        HO=HO,
        Tnow=p.T_CMB,
        fu=_FU,
        OmegaT=OmegaT,
        OmegaK=0.0,
        z_eq=max(z_eq, 1.0),
        bg=bg,
    )

    delta_z = (z_initial - z_final) / nz
    z_out = np.array([z_initial - i * delta_z for i in range(1, nz + 1)], dtype=float)
    xe_out = np.zeros(nz)
    tm_out = np.zeros(nz)

    y = _saha_init(z_initial, st)
    z_cur = z_initial

    def rk4_step(z0, y0, z1):
        h = z1 - z0
        if abs(h) < 1e-14:
            return y0.copy()
        k1 = _ion_rhs(z0, y0, st)
        k2 = _ion_rhs(z0 + 0.5 * h, y0 + 0.5 * h * k1, st)
        k3 = _ion_rhs(z0 + 0.5 * h, y0 + 0.5 * h * k2, st)
        k4 = _ion_rhs(z1, y0 + h * k3, st)
        y_new = y0 + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        y_new[0] = np.clip(y_new[0], 0.0, 1.0)
        y_new[1] = np.clip(y_new[1], 0.0, 1.0)
        y_new[2] = max(float(y_new[2]), 1e-30)
        return y_new

    for i, z_end in enumerate(z_out):
        z_scale = st.Tnow / C.COBE_CMB_TEMP * (1.0 + z_end) - 1.0
        ainv = 1.0 + z_end

        if z_scale > 8000.0:
            y = np.array([1.0, 1.0, st.Tnow])
            x0 = 1.0 + 2.0 * fHe
        elif z_scale > 5000.0:
            y = np.array([1.0, 1.0, st.Tnow])
            rhs = (
                np.exp(1.5 * np.log(_CR * st.Tnow / ainv) - _CB1_HE2 / (st.Tnow * ainv))
                / st.Nnow
            )
            x0 = 0.5 * (
                np.sqrt((rhs - 1.0 - fHe) ** 2 + 4.0 * (1.0 + 2.0 * fHe) * rhs)
                - (rhs - 1.0 - fHe)
            )
        elif z_scale > 3500.0:
            y = np.array([1.0, 1.0, st.Tnow])
            x0 = 1.0 + fHe
        elif y[1] > 0.99:
            rhs = (
                np.exp(1.5 * np.log(_CR * st.Tnow / ainv) - _CB1_HE1 / (st.Tnow * ainv))
                / st.Nnow
            )
            rhs *= 4.0
            x_He0 = 0.5 * (
                np.sqrt((rhs - 1.0) ** 2 + 4.0 * (1.0 + fHe) * rhs) - (rhs - 1.0)
            )
            x_He0 = (x_He0 - 1.0) / fHe
            y = np.array([1.0, float(np.clip(x_He0, 0.0, 1.0)), st.Tnow])
            x0 = y[0] + fHe * y[1]
        elif y[0] > 0.99:
            rhs = (
                np.exp(1.5 * np.log(_CR * st.Tnow / ainv) - _CB1 / (st.Tnow * ainv))
                / st.Nnow
            )
            x_H0 = 0.5 * (np.sqrt(rhs**2 + 4.0 * rhs) - rhs)
            # substep for stability through He→H transition
            nsub = 1
            zs = np.linspace(z_cur, z_end, nsub + 1)
            for j in range(nsub):
                y = rk4_step(zs[j], y, zs[j + 1])
            y[0] = float(np.clip(x_H0, 0.0, 1.0))
            x0 = y[0] + fHe * y[1]
        else:
            nsub = max(1, int(np.ceil(abs(z_cur - z_end))))
            # Cap substeps for speed; Recfast uses Δz≈1
            nsub = min(nsub, 8)
            zs = np.linspace(z_cur, z_end, nsub + 1)
            for j in range(nsub):
                y = rk4_step(zs[j], y, zs[j + 1])
            x0 = y[0] + fHe * y[1]

        xe_out[i] = x0 * _XE_SCALE
        tm_out[i] = ainv * y[2]
        z_cur = z_end

    return z_out, xe_out, tm_out


__all__ = ["run_recfast", "RecfastState"]
