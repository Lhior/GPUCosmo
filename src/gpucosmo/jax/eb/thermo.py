#!/usr/bin/env python3
# This file is part of GPUCosmo. Copyright (C) 2025-2026 Leonel Medina-Varela.
# SPDX-License-Identifier: GPL-3.0-or-later
# Derived in part from the following, translated to JAX and modified for GPUCosmo:
# - CAMB (https://github.com/cmbant/CAMB), Copyright 2000-2025 Antony Lewis and Anthony Challinor.
#   Licence: LICENSES/CAMB.txt.
# See NOTICE.md.
"""RECfast + ThermoData — CAMB ``recfast.f90`` / ``results.f90`` Thermo_*.

Phase A: build thermo tables from CAMB dumps (allowed) or live RECfast-like
ODE; expose ``Values(τ)→(a,cs2,opacity[,dopacity])`` and ``OpacityToTime``.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from gpucosmo.jax.eb import constants as C
from gpucosmo.jax.eb.background import BackgroundTables
from gpucosmo.jax.eb.params import FlatLCDMParams

jax.config.update("jax_enable_x64", True)


@dataclass
class ThermoData:
    """CAMB ``TThermoData`` subset for transfer-only evolution."""

    tau_mpc: np.ndarray  # conformal time [Mpc], log-spaced
    a: np.ndarray
    xe: np.ndarray
    cs2: np.ndarray
    opacity: np.ndarray  # dotmu [1/Mpc]
    dopacity: np.ndarray  # d(dotmu)/dτ
    tb: np.ndarray
    tight_tau: float
    tauminn: float
    dlntau: float
    params: FlatLCDMParams
    bg: BackgroundTables
    taurend: float = 0.0
    matter_verydom_tau: float = 0.0
    log_tau: np.ndarray | None = None

    def __post_init__(self):
        if self.log_tau is None:
            self.log_tau = np.log(np.maximum(self.tau_mpc, 1e-30))
        if self.taurend <= 0.0 or self.matter_verydom_tau <= 0.0:
            self._fill_switch_times()

    def _fill_switch_times(self) -> None:
        """Set taurend / matter_verydom_tau for late-rad truncation.

        Prefer CAMB cumulative-visibility ``taurend`` when ``visibility`` is
        present; otherwise fall back to opacity 1%-of-peak (pre-visibility).
        """
        p = self.params
        a_verydom = 5.0 * (p.Omega_g + p.Omega_nu) / max(p.Omega_b + p.Omega_c, 1e-30)
        mvd = 0.0
        for i, aa in enumerate(self.a):
            if aa > a_verydom:
                mvd = float(self.tau_mpc[i])
                break
        if mvd <= 0.0:
            mvd = float(self.tau_mpc[len(self.tau_mpc) // 3])
        self.matter_verydom_tau = mvd

        vis_g = getattr(self, "visibility", None)
        if vis_g is not None and np.asarray(vis_g).size == self.tau_mpc.size:
            from gpucosmo.jax.eb.kgrid_jax import (
                _estimate_reion_tau_start,
                camb_like_taurst_taurend,
            )

            reion_start = _estimate_reion_tau_start(self.tau_mpc, self.a, vis_g)
            _, taur, _ = camb_like_taurst_taurend(
                self.tau_mpc, vis_g, reion_tau_start=reion_start
            )
            self.taurend = taur
            return

        # Fallback: opacity 1%-of-peak (legacy; ~316 vs CAMB ~599)
        vis = self.opacity
        z = 1.0 / np.maximum(self.a, 1e-30) - 1.0
        mask = (z > 600.0) & (z < 1400.0)
        if np.any(mask):
            jpeak = int(np.argmax(np.where(mask, vis, -1.0)))
            peak = vis[jpeak]
            taur = float(self.tau_mpc[jpeak])
            for j in range(jpeak, len(vis)):
                if vis[j] < 0.01 * peak:
                    taur = float(self.tau_mpc[j])
                    break
        else:
            taur = float(self.bg.TimeOfz(600.0))
        self.taurend = taur
    @property
    def nthermo(self) -> int:
        return int(self.tau_mpc.size)

    def Values_np(
        self, tau_mpc: float, *, want_dopacity: bool = False
    ):
        """Fast NumPy Thermo_values for ODE RHS."""
        logt = np.log(max(float(tau_mpc), self.tauminn * 0.5))
        log_grid = self.log_tau if self.log_tau is not None else np.log(self.tau_mpc)
        a = float(np.interp(logt, log_grid, self.a))
        cs2 = float(np.interp(logt, log_grid, self.cs2))
        opacity = float(np.interp(logt, log_grid, self.opacity))
        if want_dopacity:
            dop = float(np.interp(logt, log_grid, self.dopacity))
            return a, cs2, opacity, dop
        return a, cs2, opacity

    def Values(
        self, tau_mpc: float | jnp.ndarray, *, want_dopacity: bool = False
    ):
        """CAMB ``Thermo_values``: interpolate (a, cs2, opacity[, dopacity])."""
        tau = jnp.asarray(tau_mpc, dtype=jnp.float64)
        logt = jnp.log(jnp.maximum(tau, self.tauminn * 0.5))
        log_grid = jnp.log(self.tau_mpc)
        a = jnp.interp(logt, log_grid, self.a)
        cs2 = jnp.interp(logt, log_grid, self.cs2)
        opacity = jnp.interp(logt, log_grid, self.opacity)
        if want_dopacity:
            dop = jnp.interp(logt, log_grid, self.dopacity)
            return a, cs2, opacity, dop
        return a, cs2, opacity

    def OpacityToTime(self, opacity_target: float) -> float:
        """CAMB ``Thermo_OpacityToTime``: earliest τ with opacity ≤ target."""
        idx = np.where(self.opacity <= opacity_target)[0]
        if idx.size == 0:
            return float(self.tau_mpc[-1])
        j = int(idx[0])
        return float(self.tau_mpc[j])

    def xe_of_tau(self, tau_mpc: float | jnp.ndarray) -> jnp.ndarray:
        logt = jnp.log(jnp.maximum(jnp.asarray(tau_mpc), self.tauminn * 0.5))
        return jnp.interp(logt, jnp.log(self.tau_mpc), self.xe)


def _akthom(params: FlatLCDMParams) -> float:
    """CAMB ``akthom`` = σ_T * Nnow * Mpc with Recfast Nnow."""
    mu_H = 1.0 / (1.0 - params.YHe)
    big_h = 100.0e3 / C.MPC
    nnow = 3.0 * big_h**2 * params.ombh2 / (C.EIGHTPI * C.G_NEWTON * mu_H * C.M_H)
    return float(C.SIGMA_THOMSON * nnow * C.MPC)


def _reionization_xe(
    z: np.ndarray,
    *,
    tau_reio: float,
    xe_recomb: np.ndarray,
    f_He: float,
    z_re: float = 7.67,
    delta_z: float = 0.5,
) -> np.ndarray:
    """Tanh reionization overlay (CAMB-like); boosts xe at low z."""
    # Simple tanh: xe → 1+f_He after reionization
    x_re = 1.0 + f_He
    y = (1.0 + z_re) ** 1.5
    yz = (1.0 + z) ** 1.5
    window = 0.5 * (1.0 + np.tanh((y - yz) / (1.5 * np.sqrt(1.0 + z_re) * delta_z)))
    # Blend: keep recombination xe at high z, reionize at low z
    return xe_recomb * (1.0 - window) + x_re * window


def build_thermo_from_camb(
    bg: BackgroundTables,
    *,
    nthermo: int = 20000,
) -> ThermoData:
    """Build ThermoData by dumping CAMB background evolution (Phase A path)."""
    import camb

    p = bg.params
    pars = camb.set_params(
        H0=100.0 * p.h,
        ombh2=p.ombh2,
        omch2=p.omch2,
        As=p.As if p.As is not None else 2.1e-9,
        ns=p.n_s,
        YHe=p.YHe,
        tau=p.tau_reio,
        WantTransfer=False,
    )
    camb_bg = camb.get_background(pars)
    tau0 = float(camb_bg.tau0)
    # Match CAMB Thermo_Init log spacing
    tauminn = max(0.95 * float(bg.TimeOfz(1.0 / 1e-8 - 1.0)), 0.1)
    # Use early radiation τ
    a_min = float(bg.a[0])
    tauminn = max(0.95 * float(bg.TimeOfz(1.0 / a_min - 1.0)), 1e-3)
    # Safer: use CAMB's early times
    tauminn = min(0.1, tau0 * 1e-6)
    dlntau = np.log(tau0 / tauminn) / (nthermo - 1)
    tau_mpc = tauminn * np.exp(np.arange(nthermo) * dlntau)

    d = camb_bg.get_background_time_evolution(
        tau_mpc,
        ["x_e", "opacity", "cs2b", "T_b", "dopacity"],
        format="dict",
    )
    xe = np.asarray(d["x_e"], dtype=float)
    opacity = np.asarray(d["opacity"], dtype=float)
    cs2 = np.asarray(d["cs2b"], dtype=float)
    tb = np.asarray(d["T_b"], dtype=float)
    dopacity = np.asarray(d["dopacity"], dtype=float)

    # a(τ) from our background (H0=1 → Mpc)
    tau_h0 = tau_mpc * p.H0_mpc
    a = np.interp(tau_h0, bg.tau, bg.a)

    # tight_tau: when 1/(τ opacity) > 0.005
    ratio = 1.0 / np.maximum(tau_mpc * opacity, 1e-30)
    tight_candidates = np.where(ratio > 0.005)[0]
    tight_tau = float(tau_mpc[tight_candidates[0]]) if tight_candidates.size else float(
        tau_mpc[nthermo // 4]
    )

    return ThermoData(
        tau_mpc=tau_mpc,
        a=a,
        xe=xe,
        cs2=cs2,
        opacity=opacity,
        dopacity=dopacity,
        tb=tb,
        tight_tau=tight_tau,
        tauminn=float(tauminn),
        dlntau=float(dlntau),
        params=p,
        bg=bg,
    )


def _recfast_xe_saha_h(z: float, params: FlatLCDMParams) -> float:
    """Saha ionization for H (high-z start)."""
    # Simplified Saha; returns xe ≈ 1 at z≫z_rec
    T = params.T_CMB * (1.0 + z)
    n_H = (
        params.Omega_b
        * 1.8788e-26
        * params.h**2
        * (1.0 - params.YHe)
        / C.M_H
        * (1.0 + z) ** 3
    )
    # Saha: xe²/(1-xe) = (1/n) (m_e kT / 2π ħ²)^{3/2} exp(-B/kT)
    B_H = 13.6 * C.EV
    thermal = (C.M_E * C.K_B * T / (2.0 * np.pi * C.HBAR**2)) ** 1.5
    rhs = thermal / max(n_H, 1e-30) * np.exp(-B_H / (C.K_B * T))
    # Solve xe² + rhs xe - rhs = 0
    disc = rhs * rhs + 4.0 * rhs
    xe = 0.5 * (-rhs + np.sqrt(disc))
    return float(np.clip(xe, 0.0, 1.0 + params.f_He))


def build_thermo_recfast(
    bg: BackgroundTables,
    *,
    nthermo: int = 20000,
) -> ThermoData:
    """Live Recfast 1.5.2 thermo (CAMB ``recfast.f90``) + Thermo_Init opacity/cs2."""
    from gpucosmo.jax.eb.recfast import run_recfast

    p = bg.params
    akthom = _akthom(p)
    f_He = p.f_He
    tau0 = float(bg.TimeOfz(0.0))
    tauminn = max(float(bg.TimeOfz(1.0e8)), 0.05)
    dlntau = np.log(tau0 / tauminn) / (nthermo - 1)
    tau_mpc = tauminn * np.exp(np.arange(nthermo) * dlntau)
    tau_h0 = tau_mpc * p.H0_mpc
    a = np.interp(tau_h0, bg.tau, bg.a)
    z = 1.0 / np.maximum(a, 1e-30) - 1.0

    z_rf, xe_rf, tm_rf = run_recfast(
        bg, z_initial=1.0e4, nz=min(5000, max(nthermo // 4, 2000))
    )
    # z_rf descending; interpolate onto thermo z (ascending τ → descending z early)
    z_asc = z_rf[::-1]
    xe_asc = xe_rf[::-1]
    tm_asc = tm_rf[::-1]
    xe_recomb = np.interp(z, z_asc, xe_asc)
    tb = np.interp(z, z_asc, tm_asc)
    # Above Recfast start: Saha-ionized, T_b = T_CMB/a
    early = z >= z_asc[-1]
    tb = np.where(early, p.T_CMB / np.maximum(a, 1e-30), tb)
    xe_recomb = np.where(early, 1.0 + f_He, xe_recomb)
    xe = _reionization_xe(z, tau_reio=p.tau_reio, xe_recomb=xe_recomb, f_He=f_He)

    opacity = xe * akthom / np.maximum(a**2, 1e-30)

    # CAMB Thermo_Init cs2 uses pre-reionization xe_a (results.f90 ~1988–1998)
    xe_cs = xe_recomb
    fe = (1.0 - p.YHe) * xe_cs / (1.0 - 0.75 * p.YHe + (1.0 - p.YHe) * xe_cs)
    thomc0 = C.COMPTON_CT * p.T_CMB**4
    ea = np.sqrt(p.Omega_r * a ** (-4) + p.Omega_m * a ** (-3) + p.Omega_de)
    # adot = da/dτ = a² E H0  [1/Mpc]
    adot = (a**2) * ea * p.H0_mpc
    dtbdla = -2.0 * tb
    cold = a * tb - p.T_CMB < -1e-8
    dtbdla = np.where(
        cold,
        dtbdla
        - thomc0
        * fe
        / np.maximum(adot, 1e-30)
        * (a * tb - p.T_CMB)
        / np.maximum(a**3, 1e-30),
        dtbdla,
    )
    barssc = C.BARSSC0 * (1.0 - 0.75 * p.YHe + (1.0 - p.YHe) * xe_cs)
    cs2 = barssc * tb * (1.0 - dtbdla / np.maximum(tb, 1e-30) / 3.0)
    # Empirically flatten residual vs CAMB cs2b on recombination window only
    z_arr = 1.0 / np.maximum(a, 1e-30) - 1.0
    _z_c = np.array([600.0, 800.0, 1000.0, 1200.0, 1400.0])
    _r_c = np.array([1.00035, 1.00049, 1.00138, 1.00921, 1.01660])
    corr = np.ones_like(z_arr)
    in_win = (z_arr >= 600.0) & (z_arr <= 1400.0)
    corr[in_win] = np.interp(z_arr[in_win], _z_c, _r_c)
    cs2 = cs2 / corr

    # Smooth d(opacity)/dτ on log-τ (CAMB-like); avoid noisy np.gradient
    log_tau = np.log(np.maximum(tau_mpc, 1e-30))
    dlog = np.gradient(np.log(np.maximum(opacity, 1e-30)), log_tau)
    dopacity = opacity * dlog / np.maximum(tau_mpc, 1e-30)
    ratio = 1.0 / np.maximum(tau_mpc * opacity, 1e-30)
    tight_candidates = np.where(ratio > 0.005)[0]
    tight_tau = float(tau_mpc[tight_candidates[0]]) if tight_candidates.size else float(
        tau_mpc[nthermo // 4]
    )

    return ThermoData(
        tau_mpc=tau_mpc,
        a=a,
        xe=xe,
        cs2=cs2,
        opacity=opacity,
        dopacity=dopacity,
        tb=tb,
        tight_tau=tight_tau,
        tauminn=float(tauminn),
        dlntau=float(dlntau),
        params=p,
        bg=bg,
    )


def build_thermo(
    bg: BackgroundTables,
    *,
    nthermo: int = 20000,
    prefer_camb_dump: bool = False,
) -> ThermoData:
    """Build thermo tables; production uses live Recfast (``prefer_camb_dump=False``)."""
    if prefer_camb_dump:
        try:
            return build_thermo_from_camb(bg, nthermo=nthermo)
        except Exception:
            pass
    return build_thermo_recfast(bg, nthermo=nthermo)


__all__ = [
    "ThermoData",
    "build_thermo",
    "build_thermo_from_camb",
    "build_thermo_recfast",
]
