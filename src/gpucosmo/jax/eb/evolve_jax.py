#!/usr/bin/env python3
# This file is part of GPUCosmo. Copyright (C) 2025-2026 Leonel Medina-Varela.
# SPDX-License-Identifier: GPL-3.0-or-later
# Derived in part from the following, translated to JAX and modified for GPUCosmo:
# - CAMB (https://github.com/cmbant/CAMB), Copyright 2000-2025 Antony Lewis and Anthony Challinor.
#   Licence: LICENSES/CAMB.txt.
# See NOTICE.md.
"""JAX transfer evolution — fixed-step ``lax.scan`` RK4 + ``vmap`` over k.

Regime schedule (matches NumPy ``evolve.py`` / CAMB ``DoLateRadTruncation``):
  1. tight coupling for ``tau < t_tight``
  2. ``init_after_tight_coupling`` when crossing ``t_tight``
  3. full hierarchy until ``t_drop_nu``
  4. ν RSA only until ``t_drop_phot`` (photons kept if ``k ≤ 0.03``)
  5. photon+ν RSA thereafter (only when ``k > 0.03``)

One composite τ grid per k (TC → full → ν-RSA → phot-RSA), denser static
steps for high-k stability. Evolve on coarse z then interpolate onto ``z_out``.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from gpucosmo.jax.eb.dark_energy import cpl_f_de
from jax import lax

from gpucosmo.jax.eb.hierarchy_jax import (
    HierarchyConfig,
    derivs,
    init_after_tight_coupling,
    initial_adiabatic,
    thermo_values,
    transfer_tot_from_y,
)
from gpucosmo.jax.eb.kgrid_jax import max_etak_scalar
from gpucosmo.jax.eb.massive_nu import nu_massive_grho_gp, nu_massive_terms

jax.config.update("jax_enable_x64", True)

# Kept for API / __all__ compat (adaptive odeint removed).
ODE_RTOL = 1.0e-4
ODE_ATOL = 1.0e-6

N_TAU_STEPS = 1536
N_SUBSTEPS = 16
N_Z_COARSE = 16
THERMO_HOT_N = 2048
_THERMO_N_DEFAULT = 10000  # cls_jax._THERMO_N: the table THERMO_HOT_N is sized for
# CAMB WantCls: DoLateRadTruncation does not RSA photons for k ≲ this.
PHOT_RSA_KMIN = 0.03

_THERMO_INTERP_KEYS = ("a", "cs2", "opacity", "dopacity", "xe", "tb")


def _as_params(params_dict) -> dict:
    if isinstance(params_dict, dict):
        return params_dict
    return {
        "Omega_b": float(params_dict.Omega_b),
        "Omega_c": float(params_dict.Omega_c),
        "Omega_g": float(params_dict.Omega_g),
        "Omega_nu": float(params_dict.Omega_nu),
        "Omega_de": float(params_dict.Omega_de),
        "H0_mpc": float(params_dict.H0_mpc),
    }


def _bg_arrays(bg_tau_a):
    if isinstance(bg_tau_a, dict):
        return jnp.asarray(bg_tau_a["a"], dtype=jnp.float64), jnp.asarray(
            bg_tau_a["tau"], dtype=jnp.float64
        )
    a_arr, tau_arr = bg_tau_a
    return jnp.asarray(a_arr, dtype=jnp.float64), jnp.asarray(tau_arr, dtype=jnp.float64)


def time_of_z(z, bg_tau_a, H0_mpc):
    a_tab, tau_h0 = _bg_arrays(bg_tau_a)
    a = 1.0 / (1.0 + jnp.asarray(z, dtype=jnp.float64))
    return jnp.interp(a, a_tab, tau_h0) / H0_mpc


def a_of_tau_mpc(tau_mpc, bg_tau_a, H0_mpc):
    a_tab, tau_h0 = _bg_arrays(bg_tau_a)
    tau_h0_q = jnp.asarray(tau_mpc, dtype=jnp.float64) * H0_mpc
    return jnp.interp(tau_h0_q, tau_h0, a_tab)


def _tau_start(k, tau0):
    return jnp.minimum(0.001 / jnp.maximum(k, 1e-10), jnp.minimum(0.1, 0.05 * tau0))


def _opacity_to_time(thermo, target_op):
    """τ where the opacity first drops to ``target_op`` (log-log interpolated)."""
    from gpucosmo.jax.eb.kgrid_jax import first_crossing

    tau = thermo["tau_mpc"]
    op = thermo["opacity"]
    log_tau = jnp.log(jnp.maximum(tau, 1e-30))
    return jnp.exp(first_crossing(
        log_tau, -jnp.log(jnp.maximum(op, 1e-300)), -jnp.log(target_op),
        fallback=log_tau[-1],
    ))


def _tc_ep(k, tau0, accuracy_boost=1.0):
    """CAMB's tight-coupling switch ratio k/opacity (equations.f90, ×2 for second-order TC)."""
    ep0 = 0.01
    epsw = 100.0 / jnp.maximum(tau0, 1.0)
    ep = jnp.where(k > epsw * 5.0, ep0 * 5.0 / accuracy_boost * 0.65, ep0)
    ep = jnp.where(k > epsw, ep, ep0)
    return ep * 2.0


def _tight_switch_time(k, thermo, tau0, accuracy_boost=1.0):
    t_opacity = _opacity_to_time(thermo, k / _tc_ep(k, tau0, accuracy_boost))
    tight_tau = jnp.asarray(thermo["tight_tau"], dtype=jnp.float64)
    return jnp.minimum(tight_tau, jnp.minimum(t_opacity, 0.8 * tau0))


def downsample_thermo_for_evolve(thermo: dict, n: int | None = None) -> dict:
    """Log-τ downsample thermo tables for cheaper RHS ``jnp.interp``.

    Writes both the primary keys used by ``hierarchy_jax.thermo_values`` and
    ``*_hot`` aliases (e.g. ``tau_mpc_hot``) so callers can detect a hot table.
    Scalar metadata (``tight_tau``, ``taurend``, …) is preserved. ``n`` nodes span the
    table; by default the node density of THERMO_HOT_N over a ``_THERMO_N``-node table,
    counted back from τ0 (a 10000-node table gives THERMO_HOT_N nodes, and tables
    extended to earlier times (``thermo_jax.build_thermo_jax(k_max=…)``) keep those nodes).
    """
    tau = jnp.asarray(thermo["tau_mpc"], dtype=jnp.float64)
    n_tab = int(tau.shape[0])
    ratio = (THERMO_HOT_N - 1) / (_THERMO_N_DEFAULT - 1)
    n_hot = 1 + int(np.floor((n_tab - 1) * ratio + 1e-9)) if n is None else int(n)
    out = dict(thermo)
    if n_tab <= n_hot:
        for key in ("tau_mpc",) + _THERMO_INTERP_KEYS:
            if key in out:
                out[f"{key}_hot"] = jnp.asarray(out[key], dtype=jnp.float64)
        if "log_tau" in out:
            out["log_tau_hot"] = jnp.asarray(out["log_tau"], dtype=jnp.float64)
        return out

    log_tau = jnp.log(jnp.maximum(tau, 1e-30))
    if n is None:
        step = (log_tau[-1] - log_tau[0]) / ((n_tab - 1) * ratio)
        log_new = log_tau[-1] - step * jnp.arange(n_hot - 1, -1, -1, dtype=jnp.float64)
        log_new = log_new.at[0].set(jnp.maximum(log_new[0], log_tau[0]))
    else:
        log_new = jnp.linspace(log_tau[0], log_tau[-1], n_hot)
    tau_new = jnp.exp(log_new)
    out["tau_mpc"] = tau_new
    out["tau_mpc_hot"] = tau_new
    out["log_tau"] = log_new
    out["log_tau_hot"] = log_new
    for key in _THERMO_INTERP_KEYS:
        if key not in thermo:
            continue
        vals = jnp.asarray(thermo[key], dtype=jnp.float64)
        down = jnp.interp(log_new, log_tau, vals)
        out[key] = down
        out[f"{key}_hot"] = down
    return out


def _thermo_for_rhs(thermo: dict) -> dict:
    """Prefer already-downsampled ``*_hot`` tables when present."""
    if "tau_mpc_hot" not in thermo:
        return downsample_thermo_for_evolve(thermo)
    out = dict(thermo)
    out["tau_mpc"] = jnp.asarray(thermo["tau_mpc_hot"], dtype=jnp.float64)
    if "log_tau_hot" in thermo:
        out["log_tau"] = jnp.asarray(thermo["log_tau_hot"], dtype=jnp.float64)
    else:
        out["log_tau"] = jnp.log(jnp.maximum(out["tau_mpc"], 1e-30))
    for key in _THERMO_INTERP_KEYS:
        hot = f"{key}_hot"
        if hot in thermo:
            out[key] = jnp.asarray(thermo[hot], dtype=jnp.float64)
    return out


def _phase_knot_counts(n_steps: int) -> tuple[int, int, int, int]:
    """Composite quotas: denser mid ν-RSA (cheap after ScalEqs pack) + solid g2."""
    n_tot = int(n_steps) + 3
    n1 = max(n_tot // 12, 12)  # TC log (trimmed)
    n4 = max(n_tot // 10, 10)  # phot+ν RSA
    # Boost A1 (mid hierarchy) — fewer eqs than A0 full; still keep g2 dense.
    n3 = max(n_tot // 5, 24)
    n2 = max(n_tot - n1 - n3 - n4, 40)  # tight→ν drop / last-scattering
    return n1, n2, n3, n4


def _composite_tau_grid(t0, t_tight, t_drop_nu, t_drop_phot, t_end, n_steps: int):
    """Static-length composite τ grid: log-TC + full + ν-RSA + phot-RSA.

    More points in the full-hierarchy window (photon/ν multipoles) where
    high-k modes are stiffest. When phot RSA is disabled (``t_drop_phot≈t_end``),
    the last segment collapses to a short tail.
    """
    t0 = jnp.asarray(t0, dtype=jnp.float64)
    t_end = jnp.maximum(jnp.asarray(t_end, dtype=jnp.float64), t0 * 1.01)
    t_tight = jnp.clip(jnp.asarray(t_tight, dtype=jnp.float64), t0 * 1.01, t_end * 0.99)
    t_drop_nu = jnp.clip(
        jnp.asarray(t_drop_nu, dtype=jnp.float64), t_tight * 1.01, t_end
    )
    t_drop_phot = jnp.clip(
        jnp.asarray(t_drop_phot, dtype=jnp.float64), t_drop_nu, t_end
    )

    # n1+n2+n3+n4 - 3 == n_steps  ⇒  sum == n_steps+3
    n1, n2, n3, n4 = _phase_knot_counts(int(n_steps))

    g1 = jnp.exp(jnp.linspace(jnp.log(jnp.maximum(t0, 1e-30)), jnp.log(t_tight), n1))
    g2 = jnp.linspace(t_tight, t_drop_nu, n2)
    g3 = jnp.linspace(t_drop_nu, t_drop_phot, n3)
    g4 = jnp.linspace(t_drop_phot, t_end, n4)
    return jnp.concatenate([g1, g2[1:], g3[1:], g4[1:]])


def _rk4_step(y, t, h, k, params, thermo, cfg, regime, thermo_at_tau=None):
    def yp(tt, yy, th):
        return _yp_regime(tt, yy, k, params, thermo, cfg, regime, thermo_at_tau=th)

    if thermo_at_tau is None:
        k1 = yp(t, y, None)
        k2 = yp(t + 0.5 * h, y + 0.5 * h * k1, None)
        k3 = yp(t + 0.5 * h, y + 0.5 * h * k2, None)
        k4 = yp(t + h, y + h * k3, None)
    else:
        # thermo_at_tau is a callable t -> (a,cs2,op,dop)
        k1 = yp(t, y, thermo_at_tau(t))
        k2 = yp(t + 0.5 * h, y + 0.5 * h * k1, thermo_at_tau(t + 0.5 * h))
        k3 = yp(t + 0.5 * h, y + 0.5 * h * k2, thermo_at_tau(t + 0.5 * h))
        k4 = yp(t + h, y + h * k3, thermo_at_tau(t + h))
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _yp_regime(t, y, k, params, thermo, cfg, regime, thermo_at_tau=None):
    """RHS with concrete Python bools selected by ``lax.switch``.

    Regimes: 0=TC, 1=full, 2=ν-RSA only, 3=phot+ν RSA, 4=full+high_ktau ν.
    """

    def tc(_):
        yp, _, _ = derivs(
            t,
            y,
            k,
            params,
            thermo,
            tight_coupling=True,
            no_phot_multpoles=False,
            no_nu_multpoles=False,
            cfg=cfg,
            thermo_at_tau=thermo_at_tau,
        )
        return yp

    def full(_):
        yp, _, _ = derivs(
            t,
            y,
            k,
            params,
            thermo,
            tight_coupling=False,
            no_phot_multpoles=False,
            no_nu_multpoles=False,
            cfg=cfg,
            thermo_at_tau=thermo_at_tau,
        )
        return yp

    def nu_rsa(_):
        yp, _, _ = derivs(
            t,
            y,
            k,
            params,
            thermo,
            tight_coupling=False,
            no_phot_multpoles=False,
            no_nu_multpoles=True,
            cfg=cfg,
            thermo_at_tau=thermo_at_tau,
        )
        return yp

    def phot_nu_rsa(_):
        yp, _, _ = derivs(
            t,
            y,
            k,
            params,
            thermo,
            tight_coupling=False,
            no_phot_multpoles=True,
            no_nu_multpoles=True,
            cfg=cfg,
            thermo_at_tau=thermo_at_tau,
        )
        return yp

    def full_high_ktau(_):
        yp, _, _ = derivs(
            t,
            y,
            k,
            params,
            thermo,
            tight_coupling=False,
            no_phot_multpoles=False,
            no_nu_multpoles=False,
            high_ktau_neutrino_approx=True,
            cfg=cfg,
            thermo_at_tau=thermo_at_tau,
        )
        return yp

    return lax.switch(
        regime, [tc, full, nu_rsa, phot_nu_rsa, full_high_ktau], None
    )


def _apply_tc_init(y, t_tight, k, params, thermo, cfg):
    _, _, opacity, dopacity = thermo_values(t_tight, thermo, want_dopacity=True)
    _, pig, pigdot = derivs(
        t_tight, y, k, params, thermo, tight_coupling=True, cfg=cfg
    )
    from gpucosmo.jax.eb.hierarchy_jax import _params_floats, curvature_k

    k_eq = curvature_k(k, _params_floats(params), cfg)[0] if cfg is not None else k  # √(q² − K)
    return init_after_tight_coupling(y, k_eq, pig, pigdot, opacity, dopacity, cfg)


# Lean composite backbone length for integrate_on_output (replaces 4096×32 / 1024).
EVOLVE_N_PRE = 640
# Low-k band (k ≤ PHOT_RSA_KMIN): shorter backbone; phot never RSA.
EVOLVE_N_PRE_LOWK = 480
# Default max RK4 microsteps per interval (full / TC).
EVOLVE_MAX_SUB = 8
# Post-RSA phase uses a short static substep budget (no padded max_sub=8).
EVOLVE_RSA_SUB = 2
# Mid ν-RSA (A1) is less stiff than full hierarchy; allow fewer microsteps.
EVOLVE_MID_SUB = 4
# Target k·Δτ_micro for local substep count.
EVOLVE_KDT_TARGET = 0.30
# CAMB cmbmain max_etak_scalar (AccuracyTarget Max_eta_k=72000 → 3600).
MAX_ETAK_SCALAR = max_etak_scalar()
# Small hierarchy for phot+ν RSA phase (CAMB ScalEqsToPropagate shrink).
RSA_HIERARCHY = HierarchyConfig(
    lmaxg=2,
    lmaxgpol=0,
    lmaxnr=2,
    include_pol=False,
    second_order_tc=False,
    use_de_ppf=True,
)


def _rsa_hierarchy(cfg: HierarchyConfig) -> HierarchyConfig:
    """Phot+ν RSA hierarchy; the massive-ν momentum grid is dropped here.

    Phase B runs at ``EVOLVE_RSA_SUB = 2`` substeps, which suits the smooth RSA
    fluid but is far too coarse for the oscillatory (frequency ~ k v) massive-ν
    multipoles -- carrying them here made D_l^TT diverge above l ~ 500.

    Dropping them is not just the stable choice, it is an accurate one: against
    the same solve at 32 RSA substeps (which is stable) the two agree to 1.3e-4
    in D_l^TT at l = 500 and 800. By the time a mode reaches phase B it is deep
    inside the horizon, where neutrino perturbations are free-streaming
    suppressed. CAMB reaches the same conclusion by swapping in a 4-moment fluid
    approximation; the grid is still carried through the earlier mid phase,
    where it does matter.
    """
    return HierarchyConfig(
        lmaxg=2,
        lmaxgpol=0,
        lmaxnr=2,
        include_pol=False,
        second_order_tc=False,
        use_de_ppf=cfg.use_de_ppf,
        mg=cfg.mg,
        curved=cfg.curved,
    )


def _mid_nu_rsa_hierarchy(cfg: HierarchyConfig) -> HierarchyConfig:
    """Photon hierarchy kept; massless ν closed (CAMB ``no_nu_multpoles``)."""
    return HierarchyConfig(
        lmaxg=cfg.lmaxg,
        lmaxgpol=cfg.lmaxgpol,
        lmaxnr=2,
        include_pol=cfg.include_pol,
        second_order_tc=False,
        nqmax=cfg.nqmax,
        lmaxnu=cfg.lmaxnu,
        n_nu_eig=cfg.n_nu_eig,
        nu_light=cfg.nu_light,
        curved=cfg.curved,
        use_de_ppf=cfg.use_de_ppf,
        mg=cfg.mg,
    )


def _copy_nu_block(
    dst: jnp.ndarray,
    src: jnp.ndarray,
    dst_cfg: HierarchyConfig,
    src_cfg: HierarchyConfig,
) -> jnp.ndarray:
    """Move the massive-ν momentum block between two state layouts."""
    if dst_cfg.nqmax <= 0 or src_cfg.nqmax <= 0:
        return dst
    n = min(dst_cfg.n_nu_massive, src_cfg.n_nu_massive)
    return dst.at[dst_cfg.nu_ix : dst_cfg.nu_ix + n].set(
        src[src_cfg.nu_ix : src_cfg.nu_ix + n]
    )


def _copy_ppf_gamma(
    dst: jnp.ndarray,
    src: jnp.ndarray,
    dst_cfg: HierarchyConfig,
    src_cfg: HierarchyConfig,
) -> jnp.ndarray:
    """Copy the PPF Gamma state when both layouts carry it."""
    if not (dst_cfg.use_de_ppf and src_cfg.use_de_ppf):
        return dst
    return dst.at[dst_cfg.w_ix].set(src[src_cfg.w_ix])


def _regime_from_t(t, did_init, t_drop_nu, t_drop_phot):
    return jnp.where(
        jnp.logical_not(did_init),
        0,
        jnp.where(
            t >= t_drop_phot,
            3,
            jnp.where(t >= t_drop_nu, 2, 1),
        ),
    )


def _th_lerp(t, t_lo, t_hi, th_lo, th_hi):
    w = (t - t_lo) / jnp.maximum(t_hi - t_lo, 1e-30)
    w = jnp.clip(w, 0.0, 1.0)
    return tuple(th_lo[i] + w * (th_hi[i] - th_lo[i]) for i in range(4))


def _pack_fluid_to_rsa(
    y_full: jnp.ndarray,
    src_cfg: HierarchyConfig | None = None,
    rsa_cfg: HierarchyConfig | None = None,
) -> jnp.ndarray:
    """Copy matter/metric into RSA hierarchy state (multipoles start at 0)."""
    rsa_cfg = rsa_cfg or RSA_HIERARCHY
    y = jnp.zeros(rsa_cfg.n_eq, dtype=jnp.float64)
    y = y.at[:4].set(y_full[:4])
    if src_cfg is not None:
        y = _copy_ppf_gamma(y, y_full, rsa_cfg, src_cfg)
        y = _copy_nu_block(y, y_full, rsa_cfg, src_cfg)
    return y


def _pack_full_to_mid(
    y_full: jnp.ndarray, cfg: HierarchyConfig, mid_cfg: HierarchyConfig
) -> jnp.ndarray:
    """Keep metric + photon/pol; drop massless ν (filled by RSA in RHS)."""
    from gpucosmo.jax.eb.hierarchy_jax import G_IX

    y = jnp.zeros(mid_cfg.n_eq, dtype=jnp.float64)
    y = y.at[:4].set(y_full[:4])
    n_keep = int(mid_cfg.r_ix)
    y = y.at[G_IX:n_keep].set(y_full[G_IX:n_keep])
    y = _copy_ppf_gamma(y, y_full, mid_cfg, cfg)
    return _copy_nu_block(y, y_full, mid_cfg, cfg)


def _nu_rsa_moments(y4, t, k, params, thermo, thermo_at_tau=None, mg=False, curved=False,
                    y_nu=None, nu_cfg=None):
    """Algebraic massless-ν (clxr, qr) from matter/metric under ν RSA (``curved``: k = √(q² − K)
    and the curvature term in adotoa, as ``derivs``). ``y_nu`` / ``nu_cfg``: a state holding the
    massive-ν block and its layout, whose δρ joins the matter's as in ``derivs``."""
    from gpucosmo.jax.eb.hierarchy_jax import IX_CLXB, IX_CLXC, IX_ETAK

    if thermo_at_tau is not None:
        a, cs2, opacity, _dop = thermo_at_tau
    else:
        a, cs2, opacity = thermo_values(t, thermo, want_dopacity=False)
        _dop = 0.0
    del cs2, opacity, _dop
    p = params
    H0 = p["H0_mpc"]
    a2 = a * a
    fac = 3.0 * H0 * H0
    grhog = fac * p["Omega_g"] / a2
    grhor = fac * p["Omega_nu"] / a2
    grhob = fac * p["Omega_b"] / a
    grhoc = fac * p["Omega_c"] / a
    w0 = p.get("w0", -1.0)
    wa = p.get("wa", 0.0)
    f_de = cpl_f_de(a, w0, wa)
    grhov = fac * p["Omega_de"] * f_de * a2
    etak = y4[IX_ETAK]
    clxc = y4[IX_CLXC]
    clxb = y4[IX_CLXB]
    dgrho_m = grhob * clxb + grhoc * clxc
    if y_nu is not None:
        dgrho_m = dgrho_m + nu_massive_terms(p, a, y_nu, nu_cfg)[2]
    # RSA switches on late, which is exactly where the massive neutrinos have
    # gone non-relativistic, so omitting them here biases adotoa most.
    grhonu_t, _ = nu_massive_grho_gp(p, a)
    grho = grhob + grhoc + grhor + grhog + grhov + fac * grhonu_t
    if curved:
        k = jnp.sqrt(jnp.maximum(k * k - p["curv"], 1e-30))
        grho = grho + fac * p["Omega_k"]
    adotoa = jnp.sqrt(jnp.maximum(grho / 3.0, 1e-30))
    z_rsa = (0.5 * dgrho_m / k + etak) / adotoa
    dz = -adotoa * z_rsa - 0.5 * dgrho_m / k
    clxr = -4.0 * dz / k
    qr = -4.0 / 3.0 * z_rsa
    if mg:
        from gpucosmo.jax.eb import mg_jax

        clx_mg, q_mg = mg_jax.rsa_radiation_mg(k, grhoc, grhob, clxc, clxb)
        clxr = jnp.where(mg_jax.mg_active(a), clx_mg, clxr)
        qr = jnp.where(mg_jax.mg_active(a), q_mg, qr)
    return clxr, qr


def _scatter_mid_to_full(
    y_mid: jnp.ndarray,
    t,
    k,
    params,
    thermo,
    cfg: HierarchyConfig,
    mid_cfg: HierarchyConfig,
    thermo_at_tau=None,
) -> jnp.ndarray:
    """Embed mid (photon+matter) state into full Cls layout; ν via RSA."""
    from gpucosmo.jax.eb.hierarchy_jax import G_IX

    y = jnp.zeros(cfg.n_eq, dtype=jnp.float64)
    y = y.at[:4].set(y_mid[:4])
    n_keep = int(mid_cfg.r_ix)
    y = y.at[G_IX:n_keep].set(y_mid[G_IX:n_keep])
    clxr, qr = _nu_rsa_moments(y_mid[:4], t, k, params, thermo, thermo_at_tau,
                               mg=cfg.mg, curved=cfg.curved, y_nu=y_mid, nu_cfg=mid_cfg)
    y = y.at[cfg.r_ix].set(clxr)
    y = y.at[cfg.r_ix + 1].set(qr)
    y = _copy_ppf_gamma(y, y_mid, cfg, mid_cfg)
    return _copy_nu_block(y, y_mid, cfg, mid_cfg)


def _scatter_rsa_to_full(
    y_rsa: jnp.ndarray,
    t,
    k,
    params,
    thermo,
    cfg: HierarchyConfig,
    thermo_at_tau=None,
    rsa_cfg: HierarchyConfig | None = None,
) -> jnp.ndarray:
    """Embed RSA fluid into full Cls state; fill γ/ν moments via RSA algebra."""
    from gpucosmo.jax.eb.hierarchy_jax import G_IX, IX_CLXB, IX_CLXC, IX_ETAK, IX_VB

    y = jnp.zeros(cfg.n_eq, dtype=jnp.float64)
    y = y.at[:4].set(y_rsa[:4])
    if thermo_at_tau is not None:
        a, cs2, opacity, _dop = thermo_at_tau
    else:
        a, cs2, opacity = thermo_values(t, thermo, want_dopacity=False)
        _dop = 0.0
    del cs2, _dop
    p = params
    H0 = p["H0_mpc"]
    a2 = a * a
    fac = 3.0 * H0 * H0
    grhog = fac * p["Omega_g"] / a2
    grhor = fac * p["Omega_nu"] / a2
    grhob = fac * p["Omega_b"] / a
    grhoc = fac * p["Omega_c"] / a
    w0 = p.get("w0", -1.0)
    wa = p.get("wa", 0.0)
    f_de = cpl_f_de(a, w0, wa)
    grhov = fac * p["Omega_de"] * f_de * a2
    etak = y_rsa[IX_ETAK]
    clxc = y_rsa[IX_CLXC]
    clxb = y_rsa[IX_CLXB]
    vb = y_rsa[IX_VB]
    rsa = rsa_cfg or RSA_HIERARCHY
    dgrho_m = grhob * clxb + grhoc * clxc + nu_massive_terms(p, a, y_rsa, rsa)[2]  # as ``derivs``
    grhonu_t, _ = nu_massive_grho_gp(p, a)
    grho = grhob + grhoc + grhor + grhog + grhov + fac * grhonu_t
    if cfg.curved:  # k = √(q² − K) and the curvature term in adotoa, as ``derivs``
        k = jnp.sqrt(jnp.maximum(k * k - p["curv"], 1e-30))
        grho = grho + fac * p["Omega_k"]
    adotoa = jnp.sqrt(jnp.maximum(grho / 3.0, 1e-30))
    z_rsa = (0.5 * dgrho_m / k + etak) / adotoa
    dz = -adotoa * z_rsa - 0.5 * dgrho_m / k
    clxr = -4.0 * dz / k
    qr = -4.0 / 3.0 * z_rsa
    clxg = clxr - 4.0 / k * opacity * (vb + z_rsa)
    qg = qr
    if cfg.mg:
        from gpucosmo.jax.eb import mg_jax

        clx_mg, q_mg = mg_jax.rsa_radiation_mg(k, grhoc, grhob, clxc, clxb)
        on = mg_jax.mg_active(a)
        clxr, qr = jnp.where(on, clx_mg, clxr), jnp.where(on, q_mg, qr)
        clxg, qg = jnp.where(on, clx_mg, clxg), jnp.where(on, q_mg, qg)
    y = y.at[G_IX].set(clxg)
    y = y.at[G_IX + 1].set(qg)
    y = y.at[cfg.r_ix].set(clxr)
    y = y.at[cfg.r_ix + 1].set(qr)
    y = _copy_ppf_gamma(y, y_rsa, cfg, rsa)
    return _copy_nu_block(y, y_rsa, cfg, rsa)


def _rk4_interval(
    y,
    t_lo,
    t_hi,
    k,
    params,
    thermo,
    cfg,
    regime,
    max_sub: int,
    n_need,
    th_lo,
    th_hi,
):
    """RK4 across one τ interval with endpoint-lerped thermo.

    ``n_need`` may be < ``max_sub`` (k·Δτ clip). ``lax.cond`` skips RHS when
    ``i >= n_need`` (helps non-vmapped paths; under ``vmap`` both sides may run).
    """
    dt = t_hi - t_lo
    n_need = jnp.maximum(n_need, 1.0)
    h = dt / n_need

    def th_at(t):
        return _th_lerp(t, t_lo, t_hi, th_lo, th_hi)

    def sub_body(i, yy):
        def do_step(y_cur):
            tt = t_lo + i * h
            return _rk4_step(
                y_cur, tt, h, k, params, thermo, cfg, regime, thermo_at_tau=th_at
            )

        return lax.cond(i < n_need, do_step, lambda y_cur: y_cur, yy)

    return lax.fori_loop(0, max_sub, sub_body, y)


def _evolve_on_timesteps(
    k,
    params,
    thermo,
    tau_out,
    *,
    cfg: HierarchyConfig,
    max_sub: int,
    kdt_target: float,
    t0,
    t_tight,
    t_drop_nu,
    t_drop_phot,
    n_pre: int = EVOLVE_N_PRE,
):
    """Three-phase evolve: full → mid ν-RSA pack → phot+ν RSA.

    A0 (TC → full → ``t_drop_nu``): full Cls hierarchy (+ high_ktau ν approx).
    A1 (``t_drop_nu`` → ``t_drop_phot``): mid hierarchy (photons kept, ν RSA).
    B (phot+ν RSA → τ_end): ``RSA_HIERARCHY``, max_etak freeze, scatter sources.
    """
    max_sub = int(max(1, max_sub))
    rsa_sub = int(EVOLVE_RSA_SUB)
    mid_sub = int(min(max_sub, EVOLVE_MID_SUB))
    n_steps = int(max(256, n_pre))
    kdt_target = float(max(kdt_target, 1e-3))
    max_etak = float(MAX_ETAK_SCALAR)
    mid_cfg = _mid_nu_rsa_hierarchy(cfg)
    rsa_cfg = _rsa_hierarchy(cfg)
    tau_out = jnp.asarray(tau_out, dtype=jnp.float64)
    tau_out = jnp.maximum(tau_out, t0 * 1.0001)
    taurend = jnp.asarray(thermo.get("taurend", t_drop_nu), dtype=jnp.float64)
    t_end = jnp.maximum(jnp.max(tau_out), jnp.maximum(t_drop_nu, t_drop_phot) * 1.01)
    t_end = jnp.minimum(t_end, jnp.asarray(thermo["tau_mpc"][-1], dtype=jnp.float64))
    k_safe = jnp.maximum(k, 1e-10)
    t_drop_phot = jnp.minimum(t_drop_phot, t_end)
    t_drop_nu = jnp.minimum(t_drop_nu, t_drop_phot)
    # CAMB tau_switch_ktau = max(20, lmaxnr-4) / k
    tau_ktau = jnp.asarray(max(20, int(cfg.lmaxnr) - 4), dtype=jnp.float64) / k_safe

    n1, n2, n3, n4 = _phase_knot_counts(n_steps)
    g1 = jnp.exp(
        jnp.linspace(jnp.log(jnp.maximum(t0, 1e-30)), jnp.log(t_tight), n1)
    )
    g2 = jnp.linspace(t_tight, t_drop_nu, n2)
    g3 = jnp.linspace(t_drop_nu, t_drop_phot, n3)
    g4 = jnp.linspace(t_drop_phot, t_end, n4)
    taus_a0 = jnp.concatenate([g1, g2[1:]])
    taus_a1 = g3
    taus_b = g4

    def _sample_th(t):
        return jnp.stack(thermo_values(t, thermo, want_dopacity=True))

    th_a0 = jax.vmap(_sample_th)(taus_a0)
    th_a1 = jax.vmap(_sample_th)(taus_a1)
    th_b = jax.vmap(_sample_th)(taus_b)

    y0 = initial_adiabatic(k, t0, params, cfg)

    def _frozen(t, t_next):
        return jnp.logical_and(t >= taurend, k_safe * t_next > max_etak)

    # ---- A0: full hierarchy through t_drop_nu ----
    def scan_a0(carry, inputs):
        y, t, did_init = carry
        t_next, th_lo, th_hi = inputs
        # 0=TC until init; then 1=full or 4=full+high_ktau
        regime = jnp.where(
            jnp.logical_not(did_init),
            0,
            jnp.where(t >= tau_ktau, 4, 1),
        )
        n_need = float(max_sub)
        y_new = _rk4_interval(
            y, t, t_next, k, params, thermo, cfg, regime, max_sub, n_need, th_lo, th_hi
        )
        need_init = jnp.logical_and(
            jnp.logical_not(did_init), t_next >= t_tight - 1e-14
        )

        def do_init(yy):
            return _apply_tc_init(yy, t_tight, k, params, thermo, cfg)

        y_new = lax.cond(need_init, do_init, lambda yy: yy, y_new)
        return (y_new, t_next, did_init | need_init), y_new

    init_a0 = (y0, taus_a0[0], jnp.asarray(False))
    xs_a0 = (taus_a0[1:], th_a0[:-1], th_a0[1:])
    (y_nu, _, _), ys_a0 = lax.scan(scan_a0, init_a0, xs_a0)
    ys_a0_full = jnp.concatenate([y0[None, :], ys_a0], axis=0)

    # ---- A1: mid hierarchy (photons + ν RSA) through t_drop_phot ----
    y_mid0 = _pack_full_to_mid(y_nu, cfg, mid_cfg)

    def scan_a1(carry, inputs):
        y_mid, t = carry
        t_next, th_lo, th_hi = inputs
        n_need = float(mid_sub)
        y_new = _rk4_interval(
            y_mid,
            t,
            t_next,
            k,
            params,
            thermo,
            mid_cfg,
            2,  # ν-RSA only
            mid_sub,
            n_need,
            th_lo,
            th_hi,
        )
        return (y_new, t_next), y_new

    init_a1 = (y_mid0, taus_a1[0])
    xs_a1 = (taus_a1[1:], th_a1[:-1], th_a1[1:])
    (y_mid_switch, _), ys_a1_mid = lax.scan(scan_a1, init_a1, xs_a1)
    ys_a1_mid_full = jnp.concatenate([y_mid0[None, :], ys_a1_mid], axis=0)

    def scatter_mid(y_mid, t, th):
        return _scatter_mid_to_full(
            y_mid, t, k, params, thermo, cfg, mid_cfg, thermo_at_tau=tuple(th)
        )

    ys_a1_full = jax.vmap(scatter_mid)(ys_a1_mid_full, taus_a1, th_a1)

    # ---- B: phot+ν RSA ----
    y_rsa0 = _pack_fluid_to_rsa(y_mid_switch, mid_cfg, rsa_cfg)

    def scan_b(carry, inputs):
        y_rsa, t = carry
        t_next, th_lo, th_hi = inputs
        dt = t_next - t
        n_from_kdt = jnp.ceil(k_safe * jnp.maximum(dt, 0.0) / kdt_target)
        n_need = jnp.clip(n_from_kdt, 1.0, float(rsa_sub))

        def do_rk4(_y):
            return _rk4_interval(
                _y,
                t,
                t_next,
                k,
                params,
                thermo,
                rsa_cfg,
                3,  # phot+ν RSA
                rsa_sub,
                n_need,
                th_lo,
                th_hi,
            )

        y_new = lax.cond(_frozen(t, t_next), lambda _y: _y, do_rk4, y_rsa)
        return (y_new, t_next), y_new

    init_b = (y_rsa0, taus_b[0])
    xs_b = (taus_b[1:], th_b[:-1], th_b[1:])
    _, ys_b_rsa = lax.scan(scan_b, init_b, xs_b)
    ys_b_rsa_full = jnp.concatenate([y_rsa0[None, :], ys_b_rsa], axis=0)

    def scatter_rsa(y_rsa, t, th):
        return _scatter_rsa_to_full(
            y_rsa, t, k, params, thermo, cfg,
            thermo_at_tau=tuple(th), rsa_cfg=rsa_cfg,
        )

    ys_b_full = jax.vmap(scatter_rsa)(ys_b_rsa_full, taus_b, th_b)

    # Drop duplicated boundary knots between phases.
    taus = jnp.concatenate([taus_a0, taus_a1[1:], taus_b[1:]])
    ys_full = jnp.concatenate([ys_a0_full, ys_a1_full[1:], ys_b_full[1:]], axis=0)

    def interp_comp(j):
        return jnp.interp(tau_out, taus, ys_full[:, j])

    return jax.vmap(interp_comp)(jnp.arange(cfg.n_eq)).T


def evolve_scalar_k_jax(
    k,
    params_dict,
    thermo_dict,
    tau_out,
    *,
    cfg: HierarchyConfig | None = None,
    n_steps: int = 0,
    n_sub: int = 0,
    integrate_on_output: bool = False,
    kdt_target: float = EVOLVE_KDT_TARGET,
    method: str = "rk4_timesteps",
    seg_cfg=None,
):
    """Evolve one k; return ``y(tau_out)`` with shape ``(n_out, n_eq)``.

    ``integrate_on_output=True`` (Cls default): RK4 between consecutive
    ``tau_out`` (TimeSteps) knots with local substep budget ``n_sub`` =
    max microsteps / interval. ``False``: legacy composite τ grid of length
    ``n_steps`` then interp onto ``tau_out`` (transfer / tests).

    ``method="segmented"`` (requires ``integrate_on_output``) uses the
    regime-static, stiffness-adapted integrator in ``evolve_segmented_jax``
    (``seg_cfg``: optional ``SegmentedConfig``).
    """
    cfg = cfg or HierarchyConfig()
    params = _as_params(params_dict)
    thermo = _thermo_for_rhs(thermo_dict)
    n_steps = int(n_steps) if int(n_steps) > 0 else (
        EVOLVE_N_PRE if integrate_on_output else N_TAU_STEPS
    )
    n_sub = int(n_sub) if int(n_sub) > 0 else (
        EVOLVE_MAX_SUB if integrate_on_output else N_SUBSTEPS
    )

    tau0 = jnp.asarray(thermo["tau_mpc"][-1], dtype=jnp.float64)
    # Prefer full-table endpoint if hot table was remapped shorter
    if "tau_mpc" in thermo_dict:
        tau0 = jnp.maximum(
            tau0, jnp.asarray(thermo_dict["tau_mpc"][-1], dtype=jnp.float64)
        )
    taurend = jnp.asarray(thermo.get("taurend", 300.0), dtype=jnp.float64)
    mvd = jnp.asarray(thermo.get("matter_verydom_tau", taurend), dtype=jnp.float64)
    k_safe = jnp.maximum(k, 1e-10)
    # NumPy evolve.py + CAMB WantCls DoLateRadTruncation:
    #   ν drop: max(15/k, min(taurend, mvd))
    #   phot drop: max(15/k, taurend) only for k > PHOT_RSA_KMIN; else never
    t_drop_nu = jnp.maximum(15.0 / k_safe, jnp.minimum(taurend, mvd))
    t_drop_phot_cand = jnp.maximum(15.0 / k_safe, taurend)
    t0 = _tau_start(k, tau0)
    t_tight = _tight_switch_time(k, thermo, tau0)
    t_tight = jnp.maximum(t_tight, t0 * 1.01)
    t_drop_nu = jnp.maximum(t_drop_nu, t_tight * 1.01)
    # Large-scale CMB: keep photon multipoles through τ₀
    rsa_kmin = getattr(seg_cfg, "phot_rsa_kmin", None)
    rsa_kmin = PHOT_RSA_KMIN if rsa_kmin is None else float(rsa_kmin)
    t_drop_phot = jnp.where(k > rsa_kmin, t_drop_phot_cand, tau0)
    t_drop_phot = jnp.maximum(t_drop_phot, t_drop_nu)
    t_end = jnp.maximum(
        jnp.max(jnp.asarray(tau_out, dtype=jnp.float64)),
        jnp.maximum(t_drop_nu, t_drop_phot) * 1.01,
    )
    t_end = jnp.minimum(t_end, tau0)
    # Ensure phot drop ≤ t_end
    t_drop_phot = jnp.minimum(t_drop_phot, t_end)
    t_drop_nu = jnp.minimum(t_drop_nu, t_drop_phot)

    tau_out = jnp.maximum(jnp.asarray(tau_out, dtype=jnp.float64), t0 * 1.0001)

    if integrate_on_output and method == "segmented":
        from gpucosmo.jax.eb.evolve_segmented_jax import (
            DEFAULT_SEGMENTED,
            evolve_segmented,
        )

        return evolve_segmented(
            k,
            params,
            thermo,
            tau_out,
            cfg=cfg,
            t0=t0,
            t_tight=t_tight,
            t_drop_nu=t_drop_nu,
            t_drop_phot=t_drop_phot,
            t_end=t_end,
            seg=seg_cfg or DEFAULT_SEGMENTED,
        )
    if method not in ("rk4_timesteps", "segmented"):
        raise ValueError(f"unknown evolve method {method!r}")

    if integrate_on_output:
        return _evolve_on_timesteps(
            k,
            params,
            thermo,
            tau_out,
            cfg=cfg,
            max_sub=n_sub,
            kdt_target=kdt_target,
            t0=t0,
            t_tight=t_tight,
            t_drop_nu=t_drop_nu,
            t_drop_phot=t_drop_phot,
            n_pre=n_steps if n_steps > 0 else EVOLVE_N_PRE,
        )

    taus = _composite_tau_grid(
        t0, t_tight, t_drop_nu, t_drop_phot, t_end, n_steps
    )
    y0 = initial_adiabatic(k, t0, params, cfg)

    def scan_body(carry, t_next):
        y, t, did_init = carry
        h_macro = t_next - t
        h = h_macro / float(n_sub)
        # Regime fixed for the macro-step from carry time so we never enter
        # the full hierarchy before ``init_after_tight_coupling``.
        # 0=TC, 1=full, 2=ν-RSA, 3=phot+ν RSA
        regime = _regime_from_t(t, did_init, t_drop_nu, t_drop_phot)

        def sub_body(i, yy):
            tt = t + i * h
            return _rk4_step(yy, tt, h, k, params, thermo, cfg, regime)

        y_new = lax.fori_loop(0, n_sub, sub_body, y)

        need_init = jnp.logical_and(
            jnp.logical_not(did_init), t_next >= t_tight - 1e-14
        )

        def do_init(yy):
            return _apply_tc_init(yy, t_tight, k, params, thermo, cfg)

        y_new = lax.cond(need_init, do_init, lambda yy: yy, y_new)
        return (y_new, t_next, did_init | need_init), y_new

    init_carry = (y0, taus[0], jnp.asarray(False))
    _, ys = lax.scan(scan_body, init_carry, taus[1:])
    ys_full = jnp.concatenate([y0[None, :], ys], axis=0)

    def interp_comp(j):
        return jnp.interp(tau_out, taus, ys_full[:, j])

    return jax.vmap(interp_comp)(jnp.arange(cfg.n_eq)).T


def _coarse_z(z_out, n_z_coarse: int):
    z_out = jnp.asarray(z_out, dtype=jnp.float64)
    n_out = int(z_out.shape[0])
    n_c = min(max(int(n_z_coarse), 2), n_out) if n_out > 0 else int(n_z_coarse)
    # Fixed-size coarse grid spanning the requested z range (host evolve.py style).
    if n_out <= n_c:
        return z_out
    return jnp.linspace(jnp.min(z_out), jnp.max(z_out), n_c)


def evolve_one_k_transfer(
    k,
    z_out,
    params_dict,
    thermo_dict,
    bg_tau_a,
    *,
    cfg: HierarchyConfig | None = None,
    n_steps: int = 0,
    n_sub: int = 0,
    n_z_coarse: int = N_Z_COARSE,
):
    """Evolve one k → Transfer_tot/k² at ``z_out`` (shape n_z)."""
    cfg = cfg or HierarchyConfig()
    params = _as_params(params_dict)
    z_out = jnp.asarray(z_out, dtype=jnp.float64)
    z_c = _coarse_z(z_out, n_z_coarse)
    tau_c = time_of_z(z_c, bg_tau_a, params["H0_mpc"])
    ys = evolve_scalar_k_jax(
        k,
        params,
        thermo_dict,
        tau_c,
        cfg=cfg,
        n_steps=n_steps,
        n_sub=n_sub,
    )
    a_c = a_of_tau_mpc(tau_c, bg_tau_a, params["H0_mpc"])

    def one(y, aa):
        return transfer_tot_from_y(y, aa, params) / jnp.maximum(k * k, 1e-30)

    T_c = jax.vmap(one)(ys, a_c)
    if int(z_c.shape[0]) == int(z_out.shape[0]):
        return T_c
    return jnp.interp(z_out, z_c, T_c)


@partial(jax.jit, static_argnames=("n_steps", "n_sub", "n_z_coarse", "cfg"))
def _evolve_transfer_grid_jit(
    k_arr,
    z_out,
    params,
    thermo,
    bg_tau_a,
    n_steps,
    n_sub,
    n_z_coarse,
    cfg,
):
    def one_k(k):
        return evolve_one_k_transfer(
            k,
            z_out,
            params,
            thermo,
            bg_tau_a,
            cfg=cfg,
            n_steps=n_steps,
            n_sub=n_sub,
            n_z_coarse=n_z_coarse,
        )

    return jax.vmap(one_k)(k_arr).T


def evolve_transfer_grid_jax(
    k_arr,
    z_out,
    params_dict,
    thermo_dict,
    bg_tau_a,
    *,
    n_steps: int = 0,
    n_sub: int = 0,
    n_z_coarse: int = N_Z_COARSE,
    cfg: HierarchyConfig | None = None,
):
    """Evolve all k; return Transfer_tot/k² with shape ``(n_z, n_k)``."""
    cfg = cfg or HierarchyConfig()
    params = _as_params(params_dict)
    thermo = _thermo_for_rhs(thermo_dict)
    k_arr = jnp.asarray(k_arr, dtype=jnp.float64)
    z_out = jnp.asarray(z_out, dtype=jnp.float64)
    n_steps_i = int(n_steps) if int(n_steps) > 0 else N_TAU_STEPS
    n_sub_i = int(n_sub) if int(n_sub) > 0 else N_SUBSTEPS
    n_z_c = int(n_z_coarse) if int(n_z_coarse) > 0 else N_Z_COARSE
    return _evolve_transfer_grid_jit(
        k_arr,
        z_out,
        params,
        thermo,
        bg_tau_a,
        n_steps_i,
        n_sub_i,
        n_z_c,
        cfg,
    )


__all__ = [
    "EVOLVE_KDT_TARGET",
    "EVOLVE_MAX_SUB",
    "EVOLVE_MID_SUB",
    "EVOLVE_N_PRE",
    "EVOLVE_N_PRE_LOWK",
    "EVOLVE_RSA_SUB",
    "MAX_ETAK_SCALAR",
    "N_SUBSTEPS",
    "N_TAU_STEPS",
    "N_Z_COARSE",
    "ODE_ATOL",
    "ODE_RTOL",
    "PHOT_RSA_KMIN",
    "RSA_HIERARCHY",
    "THERMO_HOT_N",
    "a_of_tau_mpc",
    "downsample_thermo_for_evolve",
    "evolve_one_k_transfer",
    "evolve_scalar_k_jax",
    "evolve_transfer_grid_jax",
    "time_of_z",
]
