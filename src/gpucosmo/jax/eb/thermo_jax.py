#!/usr/bin/env python3
# This file is part of GPUCosmo. Copyright (C) 2025-2026 Leonel Medina-Varela.
# SPDX-License-Identifier: GPL-3.0-or-later
# Derived in part from the following, translated to JAX and modified for GPUCosmo:
# - CAMB (https://github.com/cmbant/CAMB), Copyright 2000-2025 Antony Lewis and Anthony Challinor.
#   Licence: LICENSES/CAMB.txt.
# See NOTICE.md.
"""Device thermo tables from JAX Recfast + background (no host freeze)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from gpucosmo.jax.eb.dark_energy import cpl_f_de

from gpucosmo.jax.eb.interp_jax import interp_cubic
from gpucosmo.jax.eb import constants as C
from gpucosmo.jax.eb.params import FlatLCDMParams
from gpucosmo.jax.eb.recfast_jax import _as_flatlcdm, run_recfast_jax

jax.config.update("jax_enable_x64", True)

_BG_KEYS = (
    "H0_mpc", "Omega_r", "Omega_m", "Omega_de", "Omega_b", "Omega_c",
    "Omega_g", "Omega_nu", "YHe", "T_CMB", "tau_reio", "f_He", "ombh2", "h",
    "Omega_nu_mass", "nu_mass_am", "w0", "wa", "Omega_k",
)

# Default thermo table: ``nthermo`` log-τ nodes from TAU_TABLE_START [Mpc] to τ0. Modes start
# at τ = 0.001/k (``evolve_jax._tau_start``), so for k ≳ 0.02/Mpc before that, and the lookups
# clamp below ``tauminn``/2: a(τ) frozen through the super-horizon phase made the linear P(k)
# −1.4e-3 at k = 2, −4.5 % at 8 and −20 % at 16/Mpc. ``k_max`` prepends nodes at the same
# spacing (the τ ≥ TAU_TABLE_START nodes are unchanged) down to CAMB's start for k ≤ k_max.
TAU_TABLE_START = 0.05
# Smallest τ0 [Mpc] the static node counts are sized for (Planck ΛCDM: 14 000).
_TAU0_MIN = 8.0e3
# da/dτ [1/Mpc] in the radiation era = H0 √Ω_rad ∝ √ω_rad, h-independent: 2.15e-6 at
# N_eff = 3.044; the lower bound keeps the background start early enough for N_eff ≳ 2.
_ADOTRAD_MIN = 1.8e-6


def tau_table_start(k_max: float) -> float:
    """CAMB's thermo-table start [Mpc] for modes k ≤ ``k_max`` [1/Mpc]: 0.95 · GetTauStart(1.05
    k_max), GetTauStart(q) = min(0.001/q, 0.1) (cmbmain.f90 InitVars, results.f90 Thermo_Init)."""
    return 0.95 * min(1.0e-3 / (1.05 * float(k_max)), 0.1)


def _prepend_nodes(start: float, target: float | None, dl_min: float) -> int:
    """Static count of log nodes (spacing ≥ ``dl_min``) taking a table from ``start`` to ≤ ``target``."""
    if target is None or target >= start:
        return 0
    return int(np.ceil(np.log(start / target) / dl_min))


def thermo_prepend_nodes(k_max: float | None, nthermo: int = 10000) -> int:
    """Nodes ``build_thermo_jax(k_max=…)`` prepends to its ``nthermo``-node table."""
    if k_max is None:
        return 0
    return _prepend_nodes(TAU_TABLE_START, tau_table_start(k_max),
                          float(np.log(_TAU0_MIN / TAU_TABLE_START)) / (nthermo - 1))


def build_background_jax(
    params: FlatLCDMParams | dict,
    *,
    a_min: float = 1.0e-8,
    n_a: int = 20000,
    a_start: float | None = None,
) -> dict[str, jnp.ndarray | float]:
    """τ(a) tables as pure jnp (no host freeze). ``a_start`` < ``a_min`` prepends nodes at the
    same log-a spacing (the nodes from ``a_min`` on are unchanged)."""
    if isinstance(params, dict):
        bg_in = params
    else:
        bg_in = {k: getattr(params, k, 0.0) for k in _BG_KEYS}
        bg_in.update(
            {
                "Omega_r": params.Omega_r,
                "Omega_m": params.Omega_m,
                "Omega_de": params.Omega_de,
                "Omega_nu_mass": params.Omega_nu_mass,
                "nu_mass_am": params.nu_mass_am,
            }
        )
    n_lo = n_a // 2
    n_hi = n_a - n_lo
    dlna = float(np.log(0.1 / a_min)) / (n_lo - 1)
    n_pre = _prepend_nodes(a_min, a_start, dlna)
    a_lo = jnp.geomspace(a_min * float(np.exp(-n_pre * dlna)), 0.1, n_lo + n_pre)
    a_hi = jnp.linspace(0.1, 1.0, n_hi + 1)[1:]
    a = jnp.concatenate([a_lo, a_hi])
    Om_r = bg_in["Omega_r"]
    Om_m = bg_in["Omega_m"]
    Om_de = bg_in["Omega_de"]
    w0, wa = bg_in.get("w0", -1.0), bg_in.get("wa", 0.0)
    f_de = cpl_f_de(a, w0, wa)
    om_k = bg_in.get("Omega_k", 0.0)
    E = jnp.sqrt(_e2_terms(bg_in, a, Om_r, Om_m, Om_de * f_de, om_k))
    has_nu = not _is_zero(bg_in.get("Omega_nu_mass", 0.0))
    nu_term = _nu_rho(bg_in, a) * a ** (-4) if has_nu else 0.0
    dtauda = 1.0 / jnp.maximum(a**2 * E, 1e-30)
    # τ(a[0]) of radiation + matter, the (still relativistic) massive ν counted as radiation:
    # a/√Ω_r alone ignored them, putting τ(a[0]) ~7 % high and shifting every τ(a) by that.
    rad0 = jnp.maximum(Om_r + (nu_term[0] * a[0] ** 4 if has_nu else 0.0), 1e-30)
    tau0_seed = 2.0 * a[0] / (jnp.sqrt(rad0 + Om_m * a[0]) + jnp.sqrt(rad0))
    da = jnp.diff(a)
    mid = 0.5 * (dtauda[:-1] + dtauda[1:])
    tau = jnp.concatenate([jnp.array([tau0_seed]), tau0_seed + jnp.cumsum(mid * da)])
    out = {k: bg_in[k] for k in _BG_KEYS if k in bg_in}
    for k in _BG_KEYS:
        out.setdefault(k, bg_in.get(k, 0.0))
    out["a"] = a
    out["tau"] = tau
    return out


def _is_zero(x) -> bool:
    """True for a concrete zero (no massive eigenstates); keeps that branch Python-time."""
    try:
        return bool(np.all(np.asarray(x) == 0.0))
    except (TypeError, jax.errors.ConcretizationTypeError, jax.errors.TracerArrayConversionError):
        return False


def _nu_rho(bg: dict, a):
    from gpucosmo.jax.eb.massive_nu import nu_rho_p_omega

    return nu_rho_p_omega(bg, a)[0]


def _e2_terms(bg: dict, a, om_r, om_m, de, om_k):
    """E²(a) = Ω_r a⁻⁴ + Ω_m a⁻³ + Σ_i Ω_νi ρ_i(a) a⁻⁴ + Ω_k a⁻² + Ω_de f_de(a) (CAMB dtauda)."""
    e2 = om_r * a ** (-4) + om_m * a ** (-3) + de
    if not _is_zero(bg.get("Omega_nu_mass", 0.0)):
        e2 = e2 + _nu_rho(bg, a) * a ** (-4)
    if not _is_zero(om_k):
        e2 = e2 + om_k * a ** (-2)
    return e2


def _akthom(ombh2, YHe):
    mu_H = 1.0 / (1.0 - YHe)
    big_h = 100.0e3 / C.MPC
    nnow = 3.0 * big_h**2 * ombh2 / (C.EIGHTPI * C.G_NEWTON * mu_H * C.M_H)
    return C.SIGMA_THOMSON * nnow * C.MPC


def thermo_values(tables: dict, tau_mpc, *, want_dopacity: bool = False):
    """CAMB Thermo_values via jnp.interp on log-τ."""
    tau = jnp.asarray(tau_mpc, dtype=jnp.float64)
    logt = jnp.log(jnp.maximum(tau, tables["tauminn"] * 0.5))
    log_grid = tables["log_tau"]
    a = interp_cubic(logt, log_grid, tables["a"])
    cs2 = interp_cubic(logt, log_grid, tables["cs2"])
    opacity = interp_cubic(logt, log_grid, tables["opacity"])
    if want_dopacity:
        return a, cs2, opacity, interp_cubic(logt, log_grid, tables["dopacity"])
    return a, cs2, opacity


def build_thermo_jax(
    params: FlatLCDMParams | dict,
    *,
    nthermo: int = 10000,
    nz_recfast: int = 2000,
    tau_min: float | None = None,
    bg_a_min: float = 1.0e-8,
    z_reio: float | None = None,
    k_max: float | None = None,
) -> dict[str, jnp.ndarray | float]:
    """Device thermo tables from Recfast + background for hierarchy_jax.

    Defaults are the CMB (Cls) settings. CAMB transfer runs start the table at
    ``tau_min = 0.95 * GetTauStart(1.05 * kmax)`` (results.f90 ThermoData_Init)
    so that every k mode starts inside it; pass that as ``tau_min`` together
    with a ``bg_a_min`` early enough for the background to cover it. ``z_reio``
    fixes the tanh reionization redshift (CAMB's default ``use_optical_depth =
    False``, z_re = 10) instead of solving it from ``tau_reio``. ``k_max`` [1/Mpc] (with the
    default start) prepends ``thermo_prepend_nodes(k_max, nthermo)`` nodes at the table's
    spacing so it starts at or before ``tau_table_start(k_max)``, and the background with it.
    """
    n_pre = thermo_prepend_nodes(k_max, nthermo) if tau_min is None else 0
    a_start = 0.25 * _ADOTRAD_MIN * tau_table_start(k_max) if n_pre else None
    bg = build_background_jax(params, a_min=bg_a_min, a_start=a_start)
    pf = {k: bg[k] for k in (
        "Omega_r", "Omega_m", "Omega_de", "ombh2", "h", "T_CMB", "YHe",
        "f_He", "H0_mpc", "tau_reio", "Omega_b", "Omega_c", "Omega_g", "Omega_nu",
        # Recfast's H(z) must see the same massive-nu / w0wa background as
        # build_background_jax, or x_e around last scattering drifts by ~1%.
        "Omega_nu_mass", "nu_mass_am", "w0", "wa", "Omega_k",
    )}
    H0_mpc = bg["H0_mpc"]
    YHe, f_He, T_CMB = bg["YHe"], bg["f_He"], bg["T_CMB"]
    akthom = _akthom(bg["ombh2"], YHe)

    a_bg, tau_bg = bg["a"], bg["tau"]
    tau0 = jnp.interp(1.0, a_bg, tau_bg) / H0_mpc
    if tau_min is None:
        tauminn = jnp.maximum(jnp.interp(1.0 / (1.0e8 + 1.0), a_bg, tau_bg) / H0_mpc, 0.05)
    else:
        tauminn = jnp.asarray(tau_min, dtype=jnp.float64)
    dlntau = jnp.log(tau0 / tauminn) / (nthermo - 1)
    if n_pre:
        tauminn = tauminn * jnp.exp(-n_pre * dlntau)
    i_base = n_pre  # first node of the default table (index offsets of the fallbacks below)
    nthermo = nthermo + n_pre
    tau_mpc = tauminn * jnp.exp(jnp.arange(nthermo, dtype=jnp.float64) * dlntau)
    tau_h0 = tau_mpc * H0_mpc
    a = interp_cubic(tau_h0, tau_bg, a_bg)
    z = 1.0 / jnp.maximum(a, 1e-30) - 1.0

    z_rf, xe_rf, tm_rf = run_recfast_jax(pf, nz=nz_recfast)
    z_asc, xe_asc, tm_asc = z_rf[::-1], xe_rf[::-1], tm_rf[::-1]
    # C¹ resampling of Recfast's Δz = 5 output (see interp_jax).
    xe_recomb = interp_cubic(z, z_asc, xe_asc)
    tb = interp_cubic(z, z_asc, tm_asc)
    early = z >= z_asc[-1]
    tb = jnp.where(early, T_CMB / jnp.maximum(a, 1e-30), tb)
    # Before Recfast's start (z = 1e4) He is doubly ionized: CAMB recfast
    # x_e = 1 + 2 f_He there (was 1 + f_He: 6.9% low opacity at z > 1e4, which
    # moved the tight-coupling exit of k >~ 3/Mpc modes).
    xe_recomb = jnp.where(early, 1.0 + 2.0 * f_He, xe_recomb)

    # CAMB tanh reionization: solve z_re from τ_reio, then overlay H+HeI and HeII.
    from gpucosmo.jax.eb.reion_jax import tanh_reionization_xe, z_re_from_tau

    tau_mpc_bg = tau_bg / H0_mpc
    if z_reio is None:
        z_re = z_re_from_tau(
            bg["tau_reio"],
            akthom=akthom,
            a_bg=a_bg,
            tau_mpc_bg=tau_mpc_bg,
            f_He=f_He,
        )
    else:
        z_re = jnp.asarray(z_reio, dtype=jnp.float64)
    xe = tanh_reionization_xe(
        z, z_re=z_re, f_He=f_He, xe_recomb=xe_recomb
    )

    opacity = xe * akthom / jnp.maximum(a**2, 1e-30)

    xe_cs = xe_recomb
    fe = (1.0 - YHe) * xe_cs / (1.0 - 0.75 * YHe + (1.0 - YHe) * xe_cs)
    thomc0 = C.COMPTON_CT * T_CMB**4
    ea = jnp.sqrt(_e2_terms(bg, a, bg["Omega_r"], bg["Omega_m"],
                            bg["Omega_de"] * cpl_f_de(a, bg.get("w0", -1.0), bg.get("wa", 0.0)),
                            bg.get("Omega_k", 0.0)))
    adot = (a**2) * ea * H0_mpc
    dtbdla = -2.0 * tb
    cold = a * tb - T_CMB < -1e-8
    dtbdla = jnp.where(
        cold,
        dtbdla
        - thomc0 * fe / jnp.maximum(adot, 1e-30) * (a * tb - T_CMB) / jnp.maximum(a**3, 1e-30),
        dtbdla,
    )
    barssc = C.BARSSC0 * (1.0 - 0.75 * YHe + (1.0 - YHe) * xe_cs)
    cs2 = barssc * tb * (1.0 - dtbdla / jnp.maximum(tb, 1e-30) / 3.0)
    z_c = jnp.array([600.0, 800.0, 1000.0, 1200.0, 1400.0])
    r_c = jnp.array([1.00035, 1.00049, 1.00138, 1.00921, 1.01660])
    in_win = (z >= 600.0) & (z <= 1400.0)
    corr = jnp.where(in_win, jnp.interp(z, z_c, r_c), 1.0)
    cs2 = cs2 / corr

    log_tau = jnp.log(jnp.maximum(tau_mpc, 1e-30))
    log_op = jnp.log(jnp.maximum(opacity, 1e-30))
    dlog_c = (log_op[2:] - log_op[:-2]) / jnp.maximum(log_tau[2:] - log_tau[:-2], 1e-30)
    dlog0 = (log_op[1] - log_op[0]) / jnp.maximum(log_tau[1] - log_tau[0], 1e-30)
    dlogn = (log_op[-1] - log_op[-2]) / jnp.maximum(log_tau[-1] - log_tau[-2], 1e-30)
    dlog = jnp.concatenate([dlog0[None], dlog_c, dlogn[None]])
    dopacity = opacity * dlog / jnp.maximum(tau_mpc, 1e-30)

    from gpucosmo.jax.eb.kgrid_jax import first_crossing

    # Continuous crossings (interpolated, not snapped to table nodes).
    log_ratio = -jnp.log(jnp.maximum(tau_mpc * opacity, 1e-30))
    tight_tau = jnp.exp(first_crossing(
        jnp.log(tau_mpc), log_ratio, jnp.log(0.005),
        fallback=jnp.log(tau_mpc[i_base + (nthermo - i_base) // 4]),
    ))

    a_verydom = 5.0 * (bg["Omega_g"] + bg["Omega_nu"]) / jnp.maximum(
        bg["Omega_b"] + bg["Omega_c"], 1e-30
    )
    matter_verydom_tau = jnp.where(
        a[-1] > a_verydom, jnp.interp(a_verydom, a, tau_mpc), tau_mpc[i_base + (nthermo - i_base) // 3]
    )

    # taurend ≈ end of last-scattering (visibility drop after peak in 600<z<1400)
    z_win = (z > 600.0) & (z < 1400.0)
    vis = opacity
    vis_masked = jnp.where(z_win, vis, -1.0)
    jpeak = jnp.argmax(vis_masked)
    peak = vis[jpeak]
    after = jnp.arange(nthermo) >= jpeak
    below = after & (vis < 0.01 * peak) & z_win
    jend = jnp.argmax(below)
    taurend = jnp.where(jnp.any(below), tau_mpc[jend], jnp.interp(
        jnp.asarray(1.0 / 601.0), a_bg, tau_bg
    ) / H0_mpc)

    return {
        "tau_mpc": tau_mpc,
        "a": a,
        "xe": xe,
        "cs2": cs2,
        "opacity": opacity,
        "dopacity": dopacity,
        "tb": tb,
        "log_tau": log_tau,
        "tight_tau": tight_tau,
        "tauminn": tauminn,
        "dlntau": dlntau,
        "taurend": taurend,
        "matter_verydom_tau": matter_verydom_tau,
        "H0_mpc": H0_mpc,
        "bg_a": a_bg,
        "bg_tau": tau_bg,
        "z_re": z_re,
    }


Values = thermo_values

__all__ = [
    "Values",
    "build_background_jax",
    "build_thermo_jax",
    "tau_table_start",
    "thermo_prepend_nodes",
    "thermo_values",
]
