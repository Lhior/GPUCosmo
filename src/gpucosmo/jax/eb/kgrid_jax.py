#!/usr/bin/env python3
# This file is part of GPUCosmo. Copyright (C) 2025-2026 Leonel Medina-Varela.
# SPDX-License-Identifier: GPL-3.0-or-later
# Derived in part from the following, translated to JAX and modified for GPUCosmo:
# - CAMB (https://github.com/cmbant/CAMB), Copyright 2000-2025 Antony Lewis and Anthony Challinor.
#   Licence: LICENSES/CAMB.txt.
# See NOTICE.md.
"""CAMB ``SetkValuesForSources``-style source wavenumber grids (flat).

Uses forutils-faithful ``TRanges`` (``Add_delta`` / ``GetArray``) matching
``fortran/cmbmain.f90`` ``SetkValuesForSources`` (CAMB v2), including
``AccuracyTarget>0`` low-k reionization polarization clamp on ``dlnk0``.
"""

from __future__ import annotations

import numpy as np

from gpucosmo.jax.eb.tranges_jax import TRanges

# CAMB default smooth multipole for Silk / high-k source cutoff estimate
_L_SMOOTH_SAMPLE = 2500

# CAMB ``max_eta_k`` for lmax~1000 / AccuracyTarget source TimeSteps dtaurec
_DEFAULT_MAX_ETA_K = 72000.0


def max_etak_scalar(
    max_eta_k: float = _DEFAULT_MAX_ETA_K,
    accuracy_boost: float = 1.0,
) -> float:
    """CAMB ``cmbmain`` scalar late-source cutoff: ``boost * max(1700, Max_eta_k) / 20``."""
    return float(accuracy_boost) * max(1700.0, float(max_eta_k)) / 20.0


def set_k_values_for_sources(
    kmin: float,
    kmax: float,
    taurst: float,
    tau0: float,
    *,
    source_accuracy_boost: float = 1.0,
    accurate_reionization: bool = True,
    accurate_polarization: bool = True,
    accuracy_target: int = 1,
    reionization: bool = True,
    l_smooth_sample: int = _L_SMOOTH_SAMPLE,
) -> np.ndarray:
    """Return flat-source q-grid (Mpc⁻¹) matching CAMB ``SetkValuesForSources``."""
    kmin = float(max(kmin, 1e-6))
    kmax = float(max(kmax, kmin * 1.01))
    taurst = float(max(taurst, 1.0))
    tau0 = float(max(tau0, taurst * 2.0))
    boost = float(max(source_accuracy_boost, 1e-3))

    if reionization and accurate_polarization:
        dlnk0 = 2.0 / 10.0 / boost
    else:
        dlnk0 = 5.0 / 10.0 / boost

    if accurate_reionization:
        dlnk0 *= 0.5

    if (
        accuracy_target > 0
        and reionization
        and accurate_polarization
        and accurate_reionization
    ):
        dlnk0 = min(dlnk0, 0.075)

    dkn1 = 0.6 / taurst / boost
    dkn2 = 0.9 / taurst / boost / 1.2
    q_switch = 2.0 * 6.3 / taurst
    qmax_log = min(dkn1 / max(dlnk0, 1e-12), q_switch)

    q_cmb = 2.0 * float(l_smooth_sample) / tau0 * boost
    q_cmb = max(q_switch * 2.0, q_cmb)
    dksmooth = q_cmb / 2.0 / (boost**2)
    dksmooth = dksmooth / 6.0  # Want_CMB

    evolve_q = TRanges()
    evolve_q.init()
    qmax = kmax
    evolve_q.add_delta(kmin, min(qmax_log, qmax), dlnk0, is_log=True)
    if qmax > qmax_log and q_switch > qmax_log:
        evolve_q.add_delta(qmax_log, min(qmax, q_switch), dkn1, is_log=False)
    if qmax > q_switch:
        evolve_q.add_delta(q_switch, min(q_cmb, qmax), dkn2, is_log=False)
        if qmax > q_cmb:
            dks_log = np.log(1.0 + dksmooth / max(q_cmb, 1e-30))
            evolve_q.add_delta(q_cmb, qmax, dks_log, is_log=True)

    pts = evolve_q.get_array(want_dpoints=False)
    if pts.size == 0:
        return np.geomspace(kmin, kmax, 48)
    return pts


def set_k_values_for_int(
    qmin: float,
    qmax_int: float,
    tau0: float,
    dm_t0: float,
    *,
    int_sample_boost: float = 1.0,
    accuracy_target: int = 1,
    accurate_polarization: bool = True,
    accurate_reionization: bool = True,
    maximum_l: int = 2500,
) -> np.ndarray:
    """CAMB ``SetkValuesForInt`` Cls integration q-grid (flat).

    Sources are evolved on the coarser ``SetkValuesForSources`` grid, then
    cubic-interpolated in ``k`` onto this denser grid before LOS / ``dlnk``
    (see ``cmbmain.f90`` ``SetkValuesForInt`` + ``InterpolateSources``).
    """
    qmin = float(max(qmin, 1e-6))
    qmax_int = float(max(qmax_int, qmin * 1.01))
    tau0 = float(max(tau0, 1.0))
    dm_t0 = float(max(dm_t0, 1.0))
    boost = float(max(int_sample_boost, 1e-3))

    low_q = boost
    if (
        accuracy_target > 0
        and accurate_polarization
        and accurate_reionization
    ):
        low_q = boost * 1.8

    lognum = int(round(10.0 * low_q))
    lognum = max(lognum, 2)
    dlnk1 = 1.0 / float(lognum)
    no = int(round(600.0 * boost))
    no = max(no, 2)
    dk0 = 1.8 / dm_t0 / low_q
    dk = 3.0 / dm_t0 / boost / 1.6
    k_max_log = float(lognum) * dk0
    k_max_0 = float(no) * dk0
    dk2 = 0.04 / boost
    max_k_dk = float(max(3000, 2 * int(maximum_l))) / tau0

    r = TRanges()
    r.init()
    r.add_delta(qmin, min(k_max_log, qmax_int), dlnk1, is_log=True)
    if qmax_int > k_max_log:
        r.add_delta(k_max_log, min(qmax_int, k_max_0), dk0, is_log=False)
    if qmax_int > k_max_0:
        r.add_delta(k_max_0, min(qmax_int, max_k_dk), dk, is_log=False)
        if qmax_int > max_k_dk:
            r.add_delta(max_k_dk, qmax_int, dk2, is_log=False)
    pts = r.get_array(want_dpoints=False)
    if pts.size == 0:
        return np.geomspace(qmin, qmax_int, 256)
    return pts


def _refine_largest_gaps(pts: np.ndarray, n_out: int) -> np.ndarray:
    """Grow a sorted grid to ``n_out`` by splitting the largest gaps only."""
    pts = np.asarray(pts, dtype=np.float64)
    while pts.size < n_out:
        d = np.diff(pts)
        i = int(np.argmax(d))
        pts = np.insert(pts, i + 1, 0.5 * (pts[i] + pts[i + 1]))
    return pts


def set_time_steps_for_sources(
    taurst: float,
    taurend: float,
    tau0: float,
    *,
    dtaurec: float | None = None,
    max_eta_k: float | None = _DEFAULT_MAX_ETA_K,
    time_step_boost: float = 1.0,
    reion_tau_start: float | None = None,
    reion_tau_complete: float | None = None,
    reion_n_steps: int = 50,
    n_out: int | None = None,
) -> np.ndarray:
    """CAMB ``SetTimeSteps`` sampling for CMB sources (no redshift windows).

    Mirrors ``results.f90`` ``SetTimeSteps`` for Want_CMB scalars + reionization:
    ``Add_delta(taurst, taurend, dtaurec)``, ``Add_delta(taurend, tau0, dtau0)``,
    optional reion ``Add``, then ``GetArray``.

    ``dtaurec`` defaults to CAMB ``InitVars`` form
    ``min(4/(max_eta_k/tau0), taurst/40) / boost``.
    """
    boost = float(max(time_step_boost, 0.5))
    taurst = float(max(taurst, 1.0))
    tau0 = float(max(tau0, taurst * 2.0))
    taurend = float(np.clip(taurend, taurst * 1.01, tau0 * 0.99))
    if dtaurec is None:
        if max_eta_k is not None and float(max_eta_k) > 0.0:
            qmax = float(max_eta_k) / tau0
            dtaurec_q = 4.0 / max(qmax, 1e-12)
        else:
            dtaurec_q = (taurend - taurst) / 80.0
        dtaurec = min(dtaurec_q, taurst / 40.0) / boost
    else:
        dtaurec = float(dtaurec) / boost

    dtau0 = tau0 / 500.0 / boost

    ts = TRanges()
    ts.init()
    ts.add_delta(taurst, taurend, dtaurec, is_log=False)
    ts.add_delta(taurend, tau0, dtau0, is_log=False)

    if (
        reion_tau_start is not None
        and reion_tau_complete is not None
        and reion_tau_complete > reion_tau_start
    ):
        nri0 = max(8, int(round(reion_n_steps * boost)))
        ts.add(float(reion_tau_start), float(reion_tau_complete), nri0, is_log=False)

    pts = ts.get_array(want_dpoints=False)
    if pts.size < 8:
        pts = np.linspace(taurst, tau0, 64)

    if n_out is not None:
        n_out = int(n_out)
        if pts.size > n_out:
            idx = np.unique(np.round(np.linspace(0, pts.size - 1, n_out)).astype(int))
            pts = pts[idx]
            while pts.size < n_out:
                pts = _refine_largest_gaps(pts, pts.size + 1)
        elif pts.size < n_out:
            pts = _refine_largest_gaps(pts, n_out)
        pts = pts[:n_out].copy()
        for i in range(1, pts.size):
            if pts[i] <= pts[i - 1]:
                pts[i] = pts[i - 1] + 1e-6 * (tau0 + 1.0)

    return np.asarray(pts, dtype=np.float64)


def camb_like_taurst_taurend(
    tau: np.ndarray,
    visibility: np.ndarray,
    *,
    reion_tau_start: float | None = None,
    cum_start: float = 1.0e-7,
    cum_end: float = 0.995,
) -> tuple[float, float, float]:
    """CAMB ThermoData cumulative-visibility ``taurst`` / ``taurend`` / ``tau_maxvis``.

    Cumulative mass uses pre-reionization visibility only (CAMB clamps
    ``taurend = min(taurend, reion_tau_start)`` and the 0.995 threshold is met
    through last-scattering, not the reionization visibility bump).
    """
    tau = np.asarray(tau, dtype=np.float64)
    g = np.asarray(visibility, dtype=np.float64).copy()
    if reion_tau_start is not None:
        g = np.where(tau < float(reion_tau_start), g, 0.0)
    dtau = np.gradient(tau)
    w = np.maximum(g, 0.0) * np.maximum(dtau, 0.0)
    cum = np.cumsum(w)
    tot = float(cum[-1]) + 1e-30
    cum = cum / tot
    i0 = int(np.searchsorted(cum, cum_start))
    i0 = int(np.clip(i0, 0, tau.size - 1))
    taurst = 0.9 * float(tau[i0])
    i1 = int(np.searchsorted(cum, cum_end))
    i1 = int(np.clip(i1, i0 + 1, tau.size - 1))
    taurend = float(tau[i1])
    # Peak of full visibility (recomb), not reion
    g_full = np.asarray(visibility, dtype=np.float64)
    if reion_tau_start is not None:
        g_peak = np.where(tau < float(reion_tau_start), g_full, 0.0)
    else:
        g_peak = g_full
    ip = int(np.argmax(g_peak))
    tau_maxvis = float(tau[ip])
    if reion_tau_start is not None:
        taurend = min(taurend, float(reion_tau_start))
    taurst = max(taurst, float(tau[0]) * 1.01)
    taurend = min(max(taurend, taurst * 1.05), float(tau[-1]) * 0.99)
    return taurst, taurend, tau_maxvis


def _estimate_reion_tau_start(tau: np.ndarray, a: np.ndarray, g: np.ndarray) -> float:
    """CAMB-like reion window start from visibility peak in z∈[5,20]."""
    tau = np.asarray(tau, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    z = 1.0 / np.maximum(a, 1e-12) - 1.0
    band = (z >= 5.0) & (z <= 20.0)
    if np.any(band):
        idx = np.where(band)[0]
        ip = int(idx[int(np.argmax(g[band]))])
        zre = float(np.clip(z[ip], 6.0, 15.0))
    else:
        zre = 10.0
    z_start = zre + 8.0 * 0.5
    return float(np.interp(z_start, z[::-1], tau[::-1]))


def first_crossing(x, y, level, *, fallback=None):
    """x where ``y`` first reaches ``level`` (y ≥ level), linearly interpolated.

    Continuous in the inputs, unlike ``x[argmax(y >= level)]``, which snaps to a
    table node and puts small jumps into everything downstream (they make finite
    differences non-convergent and would break HMC energy conservation).
    """
    import jax.numpy as jnp

    hit = y >= level
    i = jnp.argmax(hit)
    i0 = jnp.maximum(i - 1, 0)
    dy = y[i] - y[i0]
    frac = jnp.where(jnp.abs(dy) > 0, (level - y[i0]) / jnp.where(dy == 0, 1.0, dy), 1.0)
    xc = x[i0] + jnp.clip(frac, 0.0, 1.0) * (x[i] - x[i0])
    if fallback is None:
        fallback = x[-1]
    return jnp.where(jnp.any(hit), xc, fallback)


def refined_peak(log_x, y):
    """Parabolic-vertex refinement of ``argmax(y)`` on a uniform ``log_x`` grid."""
    import jax.numpy as jnp

    n = y.shape[0]
    i = jnp.clip(jnp.argmax(y), 1, n - 2)
    ym, y0, yp = y[i - 1], y[i], y[i + 1]
    den = ym - 2.0 * y0 + yp
    delta = jnp.where(den < 0, 0.5 * (ym - yp) / jnp.where(den == 0, -1.0, den), 0.0)
    delta = jnp.clip(delta, -0.5, 0.5)
    dx = 0.5 * (log_x[i + 1] - log_x[i - 1])
    return jnp.exp(log_x[i] + delta * dx)


def camb_like_taurst_taurend_jax(tau, visibility, a):
    """Device twin of ``camb_like_taurst_taurend`` + ``_estimate_reion_tau_start``.

    Same algorithm and thresholds, in jax.numpy with no host conversion, so it
    returns identical values eagerly and under jit / vmap. (The previous traced
    fallback kept the reionization bump in the cumulative visibility, putting
    ``taurend`` at τ ≈ 10⁴ instead of ≈ 600 under jit.)
    """
    import jax.numpy as jnp

    tau = jnp.asarray(tau, dtype=jnp.float64)
    g = jnp.asarray(visibility, dtype=jnp.float64)
    a = jnp.asarray(a, dtype=jnp.float64)
    n = tau.shape[0]
    # Reionization start: visibility peak in z ∈ [5, 20] → z_re ± 8Δz, Δz = 0.5.
    z = 1.0 / jnp.maximum(a, 1e-12) - 1.0
    band = (z >= 5.0) & (z <= 20.0)
    ip_r = jnp.argmax(jnp.where(band, g, -jnp.inf))
    zre = jnp.where(jnp.any(band), jnp.clip(z[ip_r], 6.0, 15.0), 10.0)
    reion_start = jnp.interp(zre + 8.0 * 0.5, z[::-1], tau[::-1])
    # Cumulative pre-reionization visibility.
    g_pre = jnp.where(tau < reion_start, g, 0.0)
    w = jnp.maximum(g_pre, 0.0) * jnp.maximum(jnp.gradient(tau), 0.0)
    cum = jnp.cumsum(w)
    cum = cum / (cum[-1] + 1e-30)
    # Continuous crossings / peak (no node snapping; see ``first_crossing``).
    taurst = 0.9 * first_crossing(tau, cum, 1.0e-7)
    taurend = first_crossing(tau, cum, 0.995)
    tau_maxvis = refined_peak(jnp.log(tau), g_pre)
    taurend = jnp.minimum(taurend, reion_start)
    taurst = jnp.maximum(taurst, tau[0] * 1.01)
    taurend = jnp.minimum(jnp.maximum(taurend, taurst * 1.05), tau[-1] * 0.99)
    return taurst, taurend, tau_maxvis


def sync_thermo_taurend_camb_like(thermo: dict) -> dict:
    """Overwrite ``taurend`` with CAMB cumulative-visibility value (for evolve RSA).

    Opacity 1%-of-peak heuristics (~316 Mpc) truncate mid-k photon multipoles
    too early vs CAMB (~599). TimeSteps already used ``camb_like_taurst_taurend``;
    evolve must see the same scalar. Pure jax.numpy, identical eager / traced.
    """
    out = dict(thermo)
    if "visibility" not in out:
        return out
    a = out.get("a", out.get("bg_a"))
    taurst, taurend, tau_maxvis = camb_like_taurst_taurend_jax(
        out["tau_mpc"], out["visibility"], a
    )
    out["taurend"] = taurend
    out["taurst_camb_like"] = taurst
    out["tau_maxvis_camb_like"] = tau_maxvis
    return out


__all__ = [
    "camb_like_taurst_taurend_jax",
    "first_crossing",
    "refined_peak",
    "set_k_values_for_sources",
    "set_k_values_for_int",
    "set_time_steps_for_sources",
    "camb_like_taurst_taurend",
    "sync_thermo_taurend_camb_like",
    "max_etak_scalar",
]
