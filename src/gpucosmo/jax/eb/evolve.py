#!/usr/bin/env python3
# This file is part of GPUCosmo. Copyright (C) 2025-2026 Leonel Medina-Varela.
# SPDX-License-Identifier: GPL-3.0-or-later
# Derived in part from the following, translated to JAX and modified for GPUCosmo:
# - CAMB (https://github.com/cmbant/CAMB), Copyright 2000-2025 Antony Lewis and Anthony Challinor.
#   Licence: LICENSES/CAMB.txt.
# See NOTICE.md.
"""GaugeInterface_EvolveScal — regime switches + ODE per k.

NumPy RHS (``hierarchy.derivs``) + SciPy RK45; process-parallel k-loop.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace

import numpy as np
from scipy.integrate import solve_ivp

from gpucosmo.jax.eb import constants as C
from gpucosmo.jax.eb.background import BackgroundTables
from gpucosmo.jax.eb.hierarchy import (
    EVState,
    HierarchyConfig,
    derivs,
    init_after_tight_coupling,
    initial_adiabatic,
    transfer_tot_from_y,
)
from gpucosmo.jax.eb.thermo import ThermoData


def _tight_switch_time(k: float, thermo: ThermoData, tau0: float) -> float:
    """CAMB tight-coupling switchoff time (second_order_tightcoupling)."""
    ep0 = 0.01
    epsw = 100.0 / max(tau0, 1.0)
    if k > epsw:
        if k > epsw * 5:
            ep = ep0 * 5.0 * 0.65
        else:
            ep = ep0
    else:
        ep = ep0
    ep = ep * 2.0
    t_opacity = thermo.OpacityToTime(k / ep)
    return float(min(thermo.tight_tau, t_opacity, 0.8 * tau0))


def _tau_start(k: float, tau0: float) -> float:
    return float(min(0.001 / max(k, 1e-10), 0.1, 0.05 * tau0))


def evolve_scalar_k(
    k: float,
    tau_out: np.ndarray,
    bg: BackgroundTables,
    thermo: ThermoData,
    cfg: HierarchyConfig | None = None,
    *,
    rtol: float | None = None,
    atol: float | None = None,
) -> np.ndarray:
    """Evolve one k; return y(τ_out) with shape (n_out, n_eq)."""
    cfg = cfg or HierarchyConfig()
    rtol = C.BASE_TOL if rtol is None else rtol
    atol = 1e-6 if atol is None else atol
    tau0 = float(thermo.tau_mpc[-1])
    t0 = _tau_start(k, tau0)
    t_tight = _tight_switch_time(k, thermo, tau0)
    t_nu_rsa = max(20.0, cfg.lmaxnr - 4) / k
    taurend = thermo.taurend if thermo.taurend > 0 else float(bg.TimeOfz(600.0))
    mvd = (
        thermo.matter_verydom_tau
        if thermo.matter_verydom_tau > 0
        else float(bg.TimeOfz(100.0))
    )
    t_drop_nu = max(15.0 / k, min(taurend, mvd))
    t_drop_phot = max(15.0 / k, taurend)

    y_cur = np.asarray(initial_adiabatic(k, t0, bg, cfg), dtype=float)
    ev = EVState(k=k, cfg=cfg, tight_coupling=True)

    tau_out = np.asarray(tau_out, dtype=float)
    tau_out = np.maximum(tau_out, t0 * 1.0001)

    switches = sorted(
        {
            t0,
            t_tight,
            min(t_nu_rsa, tau0),
            min(t_drop_nu, tau0),
            min(t_drop_phot, tau0),
            float(tau_out[-1]),
        }
    )
    switches = [t for t in switches if t >= t0]

    results = np.zeros((len(tau_out), cfg.n_eq), dtype=float)
    out_idx = 0
    t_cur = t0
    did_tc_init = False

    def make_rhs(ev_local: EVState):
        def rhs(tau, y_vec):
            return derivs(float(tau), y_vec, ev_local, bg, thermo)

        return rhs

    for t_next in switches[1:]:
        if t_next <= t_cur + 1e-14:
            continue

        if t_cur >= t_tight - 1e-12 and ev.tight_coupling:
            _ = derivs(t_cur, y_cur, ev, bg, thermo)
            _, _, opacity, dopacity = thermo.Values_np(t_cur, want_dopacity=True)
            y_cur = init_after_tight_coupling(y_cur, ev, opacity, dopacity)
            ev = replace(ev, tight_coupling=False)
            did_tc_init = True
        elif t_cur >= t_tight:
            ev = replace(ev, tight_coupling=False)

        if t_cur >= t_nu_rsa:
            ev = replace(ev, high_ktau_neutrino_approx=True)
        if t_cur >= t_drop_nu:
            ev = replace(ev, no_nu_multpoles=True)
        if t_cur >= t_drop_phot:
            ev = replace(ev, no_phot_multpoles=True)

        mask = (tau_out >= t_cur) & (tau_out <= t_next + 1e-14)
        t_eval = tau_out[mask]
        rhs = make_rhs(ev)

        sol = solve_ivp(
            rhs,
            (t_cur, t_next),
            y_cur,
            method="RK45",
            rtol=rtol,
            atol=atol,
            dense_output=True,
        )
        if not sol.success or sol.sol is None:
            y_cur = _rk4_segment(rhs, t_cur, t_next, y_cur, k)
            for t_req in t_eval:
                results[out_idx] = y_cur
                out_idx += 1
        else:
            for t_req in t_eval:
                results[out_idx] = sol.sol(t_req)
                out_idx += 1
            y_cur = np.asarray(sol.sol(t_next), dtype=float)

        t_cur = t_next
        if (not did_tc_init) and abs(t_cur - t_tight) < 1e-8:
            ev_tc = replace(ev, tight_coupling=True)
            _ = derivs(t_cur, y_cur, ev_tc, bg, thermo)
            _, _, opacity, dopacity = thermo.Values_np(t_cur, want_dopacity=True)
            y_cur = init_after_tight_coupling(y_cur, ev_tc, opacity, dopacity)
            ev = replace(
                ev, tight_coupling=False, pig=ev_tc.pig, pigdot=ev_tc.pigdot
            )
            did_tc_init = True

    while out_idx < len(tau_out):
        results[out_idx] = y_cur
        out_idx += 1
    return results


def _rk4_segment(rhs, t0, t1, y0, k, nsub_max=2000):
    nsub = max(40, int(100 * (t1 - t0) * max(k, 1e-3)))
    nsub = min(nsub, nsub_max)
    ts = np.linspace(t0, t1, nsub)
    yy = np.asarray(y0, dtype=float).copy()
    for i in range(nsub - 1):
        h = ts[i + 1] - ts[i]
        k1 = rhs(ts[i], yy)
        k2 = rhs(ts[i] + 0.5 * h, yy + 0.5 * h * k1)
        k3 = rhs(ts[i] + 0.5 * h, yy + 0.5 * h * k2)
        k4 = rhs(ts[i + 1], yy + h * k3)
        yy = yy + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return yy


def _evolve_one_k_worker(args):
    """Picklable worker for ProcessPoolExecutor."""
    ik, k, tau_sorted, inv, z_grid, bg, thermo, cfg = args
    ys = evolve_scalar_k(k, tau_sorted, bg, thermo, cfg)
    ys = ys[inv]
    col = np.zeros(len(z_grid), dtype=float)
    for iz, z in enumerate(z_grid):
        a = 1.0 / (1.0 + z)
        delta = transfer_tot_from_y(ys[iz], a, bg, cfg)
        col[iz] = delta / (k * k)
    return ik, col


def evolve_transfer_grid(
    k_mpc: np.ndarray,
    z_grid: np.ndarray,
    bg: BackgroundTables,
    thermo: ThermoData,
    cfg: HierarchyConfig | None = None,
    *,
    n_jobs: int = 8,
) -> np.ndarray:
    """Evolve all k; return Transfer_tot/k² with shape (n_z, n_k).

    Evolves on a coarse conformal-time grid then interpolates in z (smooth T).
    """
    cfg = cfg or HierarchyConfig()
    z_grid = np.asarray(z_grid, dtype=float)
    # Coarse z for ODE (~16 pts): enough for smooth T(z) interpolation
    n_coarse = min(max(16, len(z_grid) // 4), len(z_grid))
    if n_coarse < len(z_grid):
        z_c = np.unique(np.concatenate([np.linspace(z_grid.min(), z_grid.max(), n_coarse), z_grid[[0, -1]]]))
    else:
        z_c = z_grid

    tau_out = np.asarray(bg.TimeOfz(z_c), dtype=float)
    order = np.argsort(tau_out)
    tau_sorted = tau_out[order]
    inv = np.argsort(order)

    T_c = np.zeros((len(z_c), len(k_mpc)), dtype=float)
    k_list = [float(k) for k in k_mpc]
    args = [
        (ik, k, tau_sorted, inv, z_c, bg, thermo, cfg) for ik, k in enumerate(k_list)
    ]

    if n_jobs <= 1 or len(k_list) < 4:
        for a in args:
            ik, col = _evolve_one_k_worker(a)
            T_c[:, ik] = col
    else:
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            futs = [ex.submit(_evolve_one_k_worker, a) for a in args]
            for fut in as_completed(futs):
                ik, col = fut.result()
                T_c[:, ik] = col

    if len(z_c) == len(z_grid) and np.allclose(z_c, z_grid):
        return T_c

    # Interpolate T(z) onto requested grid (log|T| for stability)
    T = np.zeros((len(z_grid), len(k_mpc)), dtype=float)
    for ik in range(len(k_mpc)):
        T[:, ik] = np.interp(z_grid, z_c, T_c[:, ik])
    return T


__all__ = ["evolve_scalar_k", "evolve_transfer_grid"]
