#!/usr/bin/env python3
# This file is part of GPUCosmo. Copyright (C) 2025-2026 Leonel Medina-Varela.
# SPDX-License-Identifier: GPL-3.0-or-later
# Derived in part from the following, translated to JAX and modified for GPUCosmo:
# - CAMB (https://github.com/cmbant/CAMB), Copyright 2000-2025 Antony Lewis and Anthony Challinor.
#   Licence: LICENSES/CAMB.txt.
# See NOTICE.md.
"""Scalar hierarchy — CAMB ``initial`` / ``derivs`` (TransferOnly, flat ΛCDM).

State layout (AccuracyBoost=1 TransferOnly caps):
  0: etak (= k η / 2 convention via IC)
  1: clxc
  2: clxb
  3: vb
  4 .. 4+lmaxg: photon multipoles (g_ix=4)
  pol_base .. : E_2 .. E_lmaxgpol  (polind = pol_base-2)
  r_ix .. : massless ν multipoles (lmaxnr)

References: CAMB fortran/equations.f90
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gpucosmo.jax.eb import constants as C
from gpucosmo.jax.eb.background import BackgroundTables
from gpucosmo.jax.eb.thermo import ThermoData

# Fixed indices
IX_ETAK = 0
IX_CLXC = 1
IX_CLXB = 2
IX_VB = 3
G_IX = 4  # photon L=0


@dataclass(frozen=True)
class HierarchyConfig:
    lmaxg: int = C.LMAXG_TRANSFER
    lmaxnr: int = C.LMAXNR_TRANSFER
    lmaxgpol: int = C.LMAXGPOL_TRANSFER
    include_pol: bool = True
    second_order_tc: bool = True

    @property
    def n_phot(self) -> int:
        return self.lmaxg + 1

    @property
    def n_pol(self) -> int:
        # E2..Elmaxgpol → lmaxgpol-1 slots (CAMB polind+2 .. polind+lmaxgpol)
        return max(self.lmaxgpol - 1, 0) if self.include_pol else 0

    @property
    def pol_base(self) -> int:
        """Index of E2 (= CAMB polind+2)."""
        return G_IX + self.n_phot

    @property
    def polind(self) -> int:
        return self.pol_base - 2

    @property
    def n_nu(self) -> int:
        return self.lmaxnr + 1

    @property
    def r_ix(self) -> int:
        return self.pol_base + self.n_pol

    @property
    def n_eq(self) -> int:
        return self.r_ix + self.n_nu

    @property
    def poltruncfac(self) -> float:
        return float(self.lmaxgpol) / max(1, self.lmaxgpol - 2)


@dataclass
class EVState:
    """Per-k evolution flags (CAMB EvolutionVars subset)."""

    k: float  # 1/Mpc
    cfg: HierarchyConfig
    tight_coupling: bool = True
    no_phot_multpoles: bool = False
    no_nu_multpoles: bool = False
    high_ktau_neutrino_approx: bool = False
    pig: float = 0.0
    pigdot: float = 0.0


def denl(j: int) -> float:
    return 1.0 / (2 * j + 1)


def _denlk(k: float, j: int) -> float:
    return denl(j) * k * j


def _denlk2(k: float, j: int) -> float:
    return denl(j) * k * (j + 1)  # flat Kf=1


def _polfack(k: float, j: int) -> float:
    # polfac(j)*k*Kf*denl(j); polfac(j)=(j+3)*(j-1)/(j+1)
    return ((j + 3) * (j - 1) / (j + 1)) * k * denl(j)


def initial_adiabatic(
    k: float,
    tau: float,
    bg: BackgroundTables,
    cfg: HierarchyConfig | None = None,
) -> np.ndarray:
    """CAMB ``initial`` adiabatic ICs (equations.f90 ~1847–1956)."""
    cfg = cfg or HierarchyConfig()
    p = bg.params
    H0 = p.H0_mpc
    # CAMB State%grho* today (8πG ρ a² with a=1) in 1/Mpc²
    grhob = 3.0 * H0**2 * p.Omega_b
    grhoc = 3.0 * H0**2 * p.Omega_c
    grhog = 3.0 * H0**2 * p.Omega_g
    grhor = 3.0 * H0**2 * p.Omega_nu
    om = (grhob + grhoc) / np.sqrt(3.0 * (grhog + grhor))
    omtau = om * tau
    adotrad = np.sqrt((grhog + grhor) / 3.0)
    Rv = grhor / (grhor + grhog)
    Rg = 1.0 - Rv
    Rp15 = 4.0 * Rv + 15.0
    _ = Rg  # unused for adiabatic-only

    x = k * tau
    x2 = x * x
    x3 = x2 * x
    Kf1 = 1.0

    chi = 1.0
    clxg = -chi * Kf1 / 3.0 * x2 * (1.0 - omtau / 5.0)
    clxr = clxg
    clxb = 0.75 * clxg
    clxc = clxb
    qg = clxg * x / 9.0
    qr = -chi * Kf1 * (4.0 * Rv + 23.0) / Rp15 * x3 / 27.0
    vb = 0.75 * qg
    pir = chi * 4.0 / 3.0 * x2 / Rp15 * (
        1.0 + omtau / 4.0 * (4.0 * Rv - 5.0) / (2.0 * Rv + 15.0)
    )
    aj3r = chi * 4.0 / 21.0 / Rp15 * x3
    eta = -chi * 2.0 * Kf1 * (1.0 - x2 / 12.0 * (-10.0 / Rp15 + Kf1))

    # Adiabatic: InitVec = -initv
    clxg, clxr, clxb, clxc = -clxg, -clxr, -clxb, -clxc
    qg, qr, vb, pir, aj3r, eta = -qg, -qr, -vb, -pir, -aj3r, -eta

    y = np.zeros(cfg.n_eq, dtype=np.float64)
    y[IX_ETAK] = -eta * k / 2.0
    y[IX_CLXC] = clxc
    y[IX_CLXB] = clxb
    y[IX_VB] = vb
    y[G_IX] = clxg
    y[G_IX + 1] = qg
    y[cfg.r_ix] = clxr
    y[cfg.r_ix + 1] = qr
    if cfg.lmaxnr >= 2:
        y[cfg.r_ix + 2] = pir
    if cfg.lmaxnr >= 3:
        y[cfg.r_ix + 3] = aj3r
    return y


def init_after_tight_coupling(
    y: np.ndarray,
    ev: EVState,
    opacity: float,
    dopacity: float,
) -> np.ndarray:
    """CAMB TC switchoff multipole init (equations.f90 ~323–343)."""
    cfg = ev.cfg
    k = ev.k
    y = y.copy()
    y[G_IX + 2] = ev.pig
    if cfg.second_order_tc and opacity > 0.0:
        y[G_IX + 3] = (
            (3.0 / 7.0) * y[G_IX + 2] * (k / opacity) * (1.0 + dopacity / opacity**2)
            + (3.0 / 7.0) * ev.pigdot * (k / opacity**2) * (-1.0)
        )
        if cfg.include_pol and cfg.n_pol >= 1:
            y[cfg.pol_base] = (
                ev.pig / 4.0
                + ev.pigdot
                * (1.0 / opacity)
                * (-5.0 / 8.0 - (25.0 / 16.0) * dopacity / opacity**2)
                + ev.pig * (k / opacity) ** 2 * (-5.0 / 56.0)
            )
        if cfg.include_pol and cfg.n_pol >= 2:
            y[cfg.pol_base + 1] = (
                (3.0 / 7.0)
                * (k / opacity)
                * y[cfg.pol_base]
                * (1.0 + dopacity / opacity**2)
                + (3.0 / 7.0)
                * (k / opacity**2)
                * ((ev.pigdot / 4.0) * (1.0 + (5.0 / 2.0) * dopacity / opacity**2))
                * (-1.0)
            )
    else:
        y[G_IX + 3] = 3.0 / 7.0 * y[G_IX + 2] * k / max(opacity, 1e-30)
        if cfg.include_pol and cfg.n_pol >= 1:
            y[cfg.pol_base] = ev.pig / 4.0
        if cfg.include_pol and cfg.n_pol >= 2:
            y[cfg.pol_base + 1] = y[G_IX + 3] / 4.0
    return y


def derivs(
    tau: float,
    y: np.ndarray,
    ev: EVState,
    bg: BackgroundTables,
    thermo: ThermoData,
) -> np.ndarray:
    """CAMB ``derivs`` core for TransferOnly flat ΛCDM massless-ν (NumPy)."""
    cfg = ev.cfg
    k = ev.k
    k2 = k * k
    p = bg.params
    H0 = p.H0_mpc
    y = np.asarray(y, dtype=np.float64)
    ayprime = np.zeros(cfg.n_eq, dtype=np.float64)

    if ev.tight_coupling:
        a, cs2, opacity, dopacity = thermo.Values_np(tau, want_dopacity=True)
    else:
        a, cs2, opacity = thermo.Values_np(tau, want_dopacity=False)
        dopacity = 0.0
    a2 = a * a
    cothxor = 1.0 / max(tau, 1e-30)

    fac = 3.0 * H0 * H0
    grhog = fac * p.Omega_g / a2
    grhor = fac * p.Omega_nu / a2
    grhob = fac * p.Omega_b / a
    grhoc = fac * p.Omega_c / a
    grhov = fac * p.Omega_de * a2

    etak = y[IX_ETAK]
    clxc = y[IX_CLXC]
    clxb = y[IX_CLXB]
    vb = y[IX_VB]

    dgrho_matter = grhob * clxb + grhoc * clxc
    dgq = grhob * vb
    grho_matter = grhob + grhoc
    grho = grho_matter + grhor + grhog + grhov
    gpres_noDE = (grhor + grhog) / 3.0
    adotoa = np.sqrt(grho / 3.0)

    # --- Neutrino RSA or hierarchy values ---
    if ev.no_nu_multpoles:
        z = (0.5 * dgrho_matter / k + etak) / adotoa
        dz = -adotoa * z - 0.5 * dgrho_matter / k
        clxr = -4.0 * dz / k
        qr = -4.0 / 3.0 * z
        pir = 0.0
    else:
        clxr = y[cfg.r_ix]
        qr = y[cfg.r_ix + 1]
        pir = y[cfg.r_ix + 2] if cfg.lmaxnr >= 2 else 0.0

    # --- Photon RSA or hierarchy ---
    pig = 0.0
    if ev.no_phot_multpoles:
        if not ev.no_nu_multpoles:
            z = (0.5 * dgrho_matter / k + etak) / adotoa
            dz = -adotoa * z - 0.5 * dgrho_matter / k
            clxg = -4.0 * dz / k - 4.0 / k * opacity * (vb + z)
            qg = -4.0 / 3.0 * z
        else:
            clxg = clxr - 4.0 / k * opacity * (vb + z)
            qg = qr
    else:
        clxg = y[G_IX]
        qg = y[G_IX + 1]
        if not ev.tight_coupling and cfg.lmaxg >= 2:
            pig = y[G_IX + 2]

    dgrho = dgrho_matter + grhog * clxg + grhor * clxr
    dgq = dgq + grhog * qg + grhor * qr

    photbar = grhog / max(grhob, 1e-30)
    pb43 = (4.0 / 3.0) * photbar

    z = (0.5 * dgrho / k + etak) / adotoa
    sigma = z + 1.5 * dgq / k2
    ayprime[IX_ETAK] = 0.5 * dgq
    ayprime[IX_CLXC] = -k * z
    ayprime[IX_CLXB] = -k * (z + vb)

    clxgdot = -k * (4.0 / 3.0 * z + qg)
    clxbdot = ayprime[IX_CLXB]
    delta_p_b = cs2 * clxb

    if ev.tight_coupling:
        # w_de = -1 → gpres = gpres_noDE + w*grhov
        gpres = gpres_noDE - grhov
        adotdota = (adotoa * adotoa - gpres) / 2.0

        pig = 32.0 / 45.0 / max(opacity, 1e-30) * k * (sigma + vb)

        slip = -(
            2.0 * adotoa / (1.0 + pb43) + dopacity / max(opacity, 1e-30)
        ) * (vb - 0.75 * qg) + (
            -adotdota * vb
            - k / 2.0 * adotoa * clxg
            + k * (cs2 * clxbdot - clxgdot / 4.0)
        ) / (opacity * (1.0 + pb43) + 1e-30)

        pigdot = 0.0
        if cfg.second_order_tc:
            dgs = grhog * pig + grhor * pir
            sigmadot = -2.0 * adotoa * sigma - dgs / k + etak
            qgdot_tc = k * (clxg / 4.0 - pig / 2.0) + opacity * slip
            op = max(opacity, 1e-30)
            pig = (
                32.0
                / 45.0
                / op
                * k
                * (sigma + 0.75 * qg)
                * (1.0 + dopacity * 11.0 / 6.0 / op**2)
                + (32.0 / 45.0 / op**2)
                * k
                * (sigmadot + 0.75 * qgdot_tc)
                * (-11.0 / 6.0)
            )
            pigdot = (
                -(32.0 / 45.0)
                * (dopacity / op**2)
                * k
                * (sigma + 0.75 * qg)
                * (1.0 + dopacity * 11.0 / 6.0 / op**2)
                + (32.0 / 45.0 / op)
                * k
                * (sigmadot + 0.75 * qgdot_tc)
                * (1.0 + (11.0 / 6.0) * (dopacity / op**2))
            )

        vbdot = (
            -adotoa * vb + cs2 * k * clxb + 0.25 * k * pb43 * (clxg - 2.0 * pig)
        ) / (1.0 + pb43)
        vbdot = vbdot + pb43 / (1.0 + pb43) * slip
        ev.pig = float(pig)
        ev.pigdot = float(pigdot)
    else:
        vbdot = -adotoa * vb + k * delta_p_b - photbar * opacity * (
            4.0 / 3.0 * vb - qg
        )

    ayprime[IX_VB] = vbdot

    if not ev.no_phot_multpoles:
        ayprime[G_IX] = clxgdot
        qgdot = (
            (4.0 / 3.0) * (-vbdot - adotoa * vb + delta_p_b * k) / max(pb43, 1e-30)
            + _denlk(k, 1) * clxg
            - _denlk2(k, 1) * pig
        )
        ayprime[G_IX + 1] = qgdot

        if not ev.tight_coupling:
            E2 = y[cfg.pol_base] if cfg.include_pol and cfg.n_pol >= 1 else 0.0
            polter = pig / 10.0 + 9.0 / 15.0 * E2
            if cfg.lmaxg > 2:
                pigdot = (
                    _denlk(k, 2) * qg
                    - _denlk2(k, 2) * y[G_IX + 3]
                    - opacity * (pig - polter)
                    + (8.0 / 15.0) * k * sigma
                )
                ayprime[G_IX + 2] = pigdot
                for l in range(3, cfg.lmaxg):
                    ix = G_IX + l
                    ayprime[ix] = (
                        _denlk(k, l) * y[ix - 1]
                        - _denlk2(k, l) * y[ix + 1]
                        - opacity * y[ix]
                    )
                ix = G_IX + cfg.lmaxg
                ayprime[ix] = (
                    k * y[ix - 1]
                    - (cfg.lmaxg + 1) * cothxor * y[ix]
                    - opacity * y[ix]
                )
            elif cfg.lmaxg >= 2:
                ayprime[G_IX + 2] = (
                    _denlk(k, 2) * qg
                    - opacity * (pig - polter)
                    + (8.0 / 15.0) * k * sigma
                )

            # Polarization hierarchy
            if cfg.include_pol and cfg.n_pol >= 1:
                ix = cfg.pol_base  # E2
                if cfg.lmaxgpol > 2:
                    ayprime[ix] = -opacity * (y[ix] - polter) - k / 3.0 * y[ix + 1]
                    for l in range(3, cfg.lmaxgpol):
                        # y index for El is pol_base + (l-2)
                        jx = cfg.pol_base + (l - 2)
                        ayprime[jx] = (
                            -opacity * y[jx]
                            + _denlk(k, l) * y[jx - 1]
                            - _polfack(k, l) * y[jx + 1]
                        )
                    jx = cfg.pol_base + (cfg.lmaxgpol - 2)
                    ayprime[jx] = (
                        -opacity * y[jx]
                        + k * cfg.poltruncfac * y[jx - 1]
                        - (cfg.lmaxgpol + 3) * cothxor * y[jx]
                    )
                else:
                    ayprime[ix] = -opacity * (y[ix] - polter)

    if not ev.no_nu_multpoles:
        clxrdot = -k * (4.0 / 3.0 * z + qr)
        ayprime[cfg.r_ix] = clxrdot
        ayprime[cfg.r_ix + 1] = _denlk(k, 1) * clxr - _denlk2(k, 1) * pir
        if ev.high_ktau_neutrino_approx:
            ayprime[cfg.r_ix + 2] = -3.0 * pir * cothxor - clxrdot
        else:
            if cfg.lmaxnr > 2:
                ayprime[cfg.r_ix + 2] = (
                    _denlk(k, 2) * qr
                    - _denlk2(k, 2) * y[cfg.r_ix + 3]
                    + (8.0 / 15.0) * k * sigma
                )
                for l in range(3, cfg.lmaxnr):
                    ix = cfg.r_ix + l
                    ayprime[ix] = _denlk(k, l) * y[ix - 1] - _denlk2(k, l) * y[ix + 1]
                ix = cfg.r_ix + cfg.lmaxnr
                ayprime[ix] = k * y[ix - 1] - (cfg.lmaxnr + 1) * cothxor * y[ix]
            elif cfg.lmaxnr >= 2:
                ayprime[cfg.r_ix + 2] = _denlk(k, 2) * qr + (8.0 / 15.0) * k * sigma

    return ayprime


def transfer_tot_from_y(
    y: np.ndarray,
    a: float,
    bg: BackgroundTables,
    cfg: HierarchyConfig | None = None,
) -> float:
    """δρ_m / ρ_m before /k² (Transfer_nonu = Transfer_tot for massless ν)."""
    p = bg.params
    fac = 3.0 * p.H0_mpc**2
    grhob = fac * p.Omega_b / a
    grhoc = fac * p.Omega_c / a
    clxb = float(y[IX_CLXB])
    clxc = float(y[IX_CLXC])
    return (grhob * clxb + grhoc * clxc) / (grhob + grhoc)


__all__ = [
    "EVState",
    "G_IX",
    "HierarchyConfig",
    "IX_CLXB",
    "IX_CLXC",
    "IX_ETAK",
    "IX_VB",
    "derivs",
    "init_after_tight_coupling",
    "initial_adiabatic",
    "transfer_tot_from_y",
]
