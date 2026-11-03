#!/usr/bin/env python3
# This file is part of GPUCosmo. Copyright (C) 2025-2026 Leonel Medina-Varela.
# SPDX-License-Identifier: GPL-3.0-or-later
# Derived in part from the following, translated to JAX and modified for GPUCosmo:
# - CAMB (https://github.com/cmbant/CAMB), Copyright 2000-2025 Antony Lewis and Anthony Challinor.
#   Licence: LICENSES/CAMB.txt.
# See NOTICE.md.
"""CAMB ``MakeNonlinearSources`` for the lensing/Weyl source (Takahashi Halofit).

CAMB scales ``ScaledSrc(ik, 3:, i)`` by the amplitude ratio
``sqrt(P_nl/P_lin)`` interpolated over conformal time
(``cmbmain.f90:MakeNonlinearSources``). Default CAMB uses HMcode mead2020;
we use the on-device Takahashi Halofit so the path stays JIT / AD friendly
(user-chosen approximation for this port).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from gpucosmo.jax.eb.interp_jax import interp_cubic
from gpucosmo.jax.eb.hierarchy_jax import HierarchyConfig
from gpucosmo.jax.eb.sources_jax import _background_and_stress
from gpucosmo.jax.halofit import halofit_log_pk_jax

jax.config.update("jax_enable_x64", True)

# CAMB classes.f90 default Min_kh_nonlinear.
MIN_KH_NONLINEAR = 0.005
# Sparse redshift samples for Halofit (CAMB Transfer_Times density for lensing).
_N_Z_NL = 64
_Z_MAX_NL = 10.0


def transfer_tot_from_y_grid(
    y: jnp.ndarray,
    k: jnp.ndarray,
    tau: jnp.ndarray,
    params: dict,
    thermo: dict,
    cfg: HierarchyConfig,
) -> jnp.ndarray:
    """CAMB ``Transfer_tot / k²`` on the evolve grid: shape ``(n_k, n_tau)``.

    ``outtransf`` writes ``dgrho_matter/grho_matter`` then divides by ``k²``
    (``equations.f90:2840`` + ``1115``).
    """

    def one_k(yi, ki):
        q = _background_and_stress(yi, ki, tau, params, thermo, cfg)
        return q["delta_m"] / jnp.maximum(ki * ki, 1e-30)

    # y is (n_k, n_tau, n_eq)
    return jax.vmap(one_k)(y, k)


def _pk_lin_from_transfer(
    T_tot: jnp.ndarray,
    k: jnp.ndarray,
    p_r: jnp.ndarray,
) -> jnp.ndarray:
    """P(k) in Mpc³ from Transfer_tot/k² and the primordial spectrum."""
    two_pi2 = 2.0 * jnp.pi * jnp.pi
    return (T_tot**2) * k * two_pi2 * p_r


def nonlinear_ratio_weyl(
    *,
    y_ev: jnp.ndarray,
    k_ev: jnp.ndarray,
    tau: jnp.ndarray,
    thermo: dict,
    params: dict,
    cfg: HierarchyConfig,
    p_r_ev: jnp.ndarray,
    min_kh: float = MIN_KH_NONLINEAR,
) -> jnp.ndarray:
    """Amplitude ratio ``sqrt(P_nl/P_lin)`` on ``(n_k, n_tau)``, 1 below ``min_kh``.

    Builds Transfer_tot on the evolve grid, evaluates Takahashi Halofit on a
    sparse redshift table, and interpolates the ratio back onto ``tau``.
    """
    T = transfer_tot_from_y_grid(y_ev, k_ev, tau, params, thermo, cfg)
    h = jnp.asarray(params["h"], dtype=jnp.float64)
    Omega_m = float(
        jnp.asarray(
            params["Omega_b"]
            + params["Omega_c"]
            + params.get("Omega_nu_massive_today", 0.0)
        )
    )

    a = jnp.interp(tau, thermo["tau_mpc"], thermo["a"])
    z_tau = 1.0 / jnp.maximum(a, 1e-12) - 1.0

    z_tab = jnp.linspace(0.0, _Z_MAX_NL, _N_Z_NL)
    # Map z → a → tau via the thermo table (a increases with tau).
    tau_of_z = jnp.interp(
        1.0 / (1.0 + z_tab),
        thermo["a"],
        thermo["tau_mpc"],
    )

    def T_at_tau(t):
        return jax.vmap(lambda Tk: jnp.interp(t, tau, Tk))(T)

    T_z = jax.vmap(T_at_tau)(tau_of_z)  # (n_z, n_k)
    pk_lin = jax.vmap(lambda Tz: _pk_lin_from_transfer(Tz, k_ev, p_r_ev))(T_z)

    # Extend k upward with a power-law tail so Halofit σ(R) integrals converge.
    k_ext_max = jnp.maximum(k_ev[-1] * 20.0, 100.0)
    n_ext = 32
    k_tail = jnp.geomspace(k_ev[-1] * 1.01, k_ext_max, n_ext)
    slope = jnp.log(
        jnp.maximum(pk_lin[:, -1], 1e-30) / jnp.maximum(pk_lin[:, -2], 1e-30)
    ) / jnp.log(k_ev[-1] / k_ev[-2])
    pk_tail = pk_lin[:, -1:] * (k_tail[None, :] / k_ev[-1]) ** slope[:, None]
    k_full = jnp.concatenate([k_ev, k_tail])
    pk_full = jnp.concatenate([pk_lin, pk_tail], axis=1)

    log_k = jnp.log(jnp.maximum(k_full, 1e-30))
    log_pk = jnp.log(jnp.maximum(pk_full, 1e-30))
    log_pk_nl = halofit_log_pk_jax(log_k, z_tab, log_pk, Omega_m=Omega_m, h=h)

    ratio_z = jnp.sqrt(
        jnp.exp(log_pk_nl[:, : k_ev.shape[0]] - log_pk[:, : k_ev.shape[0]])
    )
    ratio_z = jnp.where(jnp.isfinite(ratio_z), ratio_z, 1.0)

    def ratio_one_k(rk):
        return jnp.interp(z_tau, z_tab, rk)

    ratio = jax.vmap(ratio_one_k)(ratio_z.T)  # (n_k, n_tau)

    kh = k_ev / jnp.maximum(h, 1e-30)
    mask = (kh > min_kh)[:, None]
    return jnp.where(mask, ratio, 1.0)


def eh98_nowiggle_transfer(k_mpc, params: dict) -> jnp.ndarray:
    """Eisenstein & Hu (1998) no-wiggle transfer function at ``k`` in 1/Mpc.

    Used only for its *shape* above the evolved k range, where the true
    transfer bends as ln(k)/k² rather than following a pure power law.
    """
    h = jnp.asarray(params["h"], dtype=jnp.float64)
    om_m = params["Omega_b"] + params["Omega_c"] + params.get("Omega_nu_massive_today", 0.0)
    om_b = params["Omega_b"]
    wm, wb = om_m * h**2, om_b * h**2
    fb = om_b / om_m
    theta = jnp.asarray(params.get("T_CMB", 2.7255), dtype=jnp.float64) / 2.7
    s = 44.5 * jnp.log(9.83 / wm) / jnp.sqrt(1.0 + 10.0 * wb**0.75)
    alpha = 1.0 - 0.328 * jnp.log(431.0 * wm) * fb + 0.38 * jnp.log(22.3 * wm) * fb**2
    k = jnp.asarray(k_mpc, dtype=jnp.float64)
    gamma = om_m * h * (alpha + (1.0 - alpha) / (1.0 + (0.43 * k * s) ** 4))
    q = k * theta**2 / (gamma * h)
    L = jnp.log(2.0 * jnp.e + 1.8 * q)
    C = 14.2 + 731.0 / (1.0 + 62.5 * q)
    return L / (L + C * q**2)


def eh99_nu_growth(k_mpc, a, params: dict) -> jnp.ndarray:
    """Eisenstein & Hu (1999) growth of the total matter with massive neutrinos, D_cbν(k, a)
    (their eqs. 10–13; ``k`` and ``a`` broadcast), up to a k-independent factor.

    Massive neutrinos stop clustering below their free-streaming length, so above k_fs ≈ 0.8
    (m / 1 eV) h/Mpc today the growth of δ_m is suppressed in a way that still changes with k
    there. The free-streaming scale takes the density-weighted m/(k_B T) of the eigenstates
    (CAMB ``nu_masses``) as the mass at the standard T_ν0, which covers a sterile state's
    lower temperature; several eigenstates are treated as one species with that mass.
    """
    from gpucosmo.jax.eb.massive_nu import nu_temperature_ev, thermal_nu_rho

    h = jnp.asarray(params["h"], dtype=jnp.float64)
    om_nu = jnp.asarray(params.get("Omega_nu_massive_today", 0.0), dtype=jnp.float64)
    om0 = params["Omega_b"] + params["Omega_c"] + om_nu
    f_nu = jnp.maximum(om_nu / om0, 1e-12)
    f_cb = 1.0 - f_nu
    p_cb = (5.0 - jnp.sqrt(1.0 + 24.0 * f_cb)) / 4.0
    am = jnp.atleast_1d(jnp.asarray(params.get("nu_mass_am", 0.0), dtype=jnp.float64))
    w = jnp.atleast_1d(jnp.asarray(params.get("Omega_nu_mass", 0.0), dtype=jnp.float64)) * thermal_nu_rho(am)
    am_eff = jnp.sum(w * am) / jnp.maximum(jnp.sum(w), 1e-300)
    m_eff = jnp.maximum(am_eff * nu_temperature_ev(), 1e-6)  # eV at T_ν0
    theta = jnp.asarray(params.get("T_CMB", 2.7255), dtype=jnp.float64) / 2.7
    om_l = jnp.asarray(params["Omega_de"], dtype=jnp.float64)
    z = 1.0 / jnp.asarray(a, dtype=jnp.float64) - 1.0
    g2 = om0 * (1 + z) ** 3 + (1.0 - om0 - om_l) * (1 + z) ** 2 + om_l
    om_z, ol_z = om0 * (1 + z) ** 3 / g2, om_l / g2
    z_eq = 2.50e4 * om0 * h**2 * theta**-4
    d1 = (1 + z_eq) / (1 + z) * 2.5 * om_z / (om_z ** (4.0 / 7.0) - ol_z + (1 + om_z / 2) * (1 + ol_z / 70))
    # N_ν q / f_ν with Ω_ν h² = N_ν m / 93.14 eV and q = k Θ² / (Ω_0 h²)
    nq_f = 93.14 * jnp.asarray(k_mpc, dtype=jnp.float64) * theta**2 / m_eff
    y_fs = 17.2 * f_nu * (1.0 + 0.488 * f_nu ** (-7.0 / 6.0)) * nq_f**2
    return (f_cb ** (0.7 / p_cb) + (d1 / (1.0 + y_fs)) ** 0.7) ** (p_cb / 0.7) * d1 ** (1.0 - p_cb)


def extend_weyl_high_k(
    weyl: jnp.ndarray,
    k_ev: jnp.ndarray,
    k_ext: jnp.ndarray,
    *,
    slope: float = -2.0,
    params: dict | None = None,
    a_tau: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Continue the Weyl source beyond the evolved k range.

    Above the last evolved k, Weyl(k, τ) ≈ Weyl(k_last, τ)·T(k)/T(k_last) on
    sub-horizon scales at late times. With ``params`` the transfer shape is
    Eisenstein–Hu (1998) no-wiggle, which keeps the ln(k)/k² bend; without, a
    ``k^{slope}`` power law (the old default, which runs low: −3% in linear
    C_L^φφ at L = 2000 vs CAMB). CAMB's Limber branch reads the source out to
    ``Max_eta_k/tau0 ≈ 5``. With ``a_tau`` (the scale factor of the source's τ columns) the
    shape also carries the massive neutrinos' scale-dependent growth (``eh99_nu_growth``):
    without it the continued Weyl ran high by the free-streaming suppression still building
    above k_last, 2.9 % at k = 2/Mpc for a 1.1 eV state (C_L^κκ +2e-3 at L ≳ 2000).
    """
    k_ev = jnp.asarray(k_ev, dtype=jnp.float64)
    k_ext = jnp.asarray(k_ext, dtype=jnp.float64)
    weyl = jnp.asarray(weyl, dtype=jnp.float64)
    k_last = k_ev[-1]
    w_last = weyl[-1, :]
    if params is not None:
        t_last = eh98_nowiggle_transfer(k_last, params)

        def shape(kq):
            return eh98_nowiggle_transfer(kq, params) / t_last
    else:

        def shape(kq):
            return (kq / k_last) ** slope

    shape_tau = shape
    if params is not None and a_tau is not None:
        g_last = eh99_nu_growth(k_last, a_tau, params)

        def shape_tau(kq):  # (n_tau,) per k
            return shape(kq) * eh99_nu_growth(kq, a_tau, params) / g_last

    def at_k(kq):
        interior = jax.vmap(lambda col: jnp.interp(kq, k_ev, col))(weyl.T)
        exterior = w_last * shape_tau(kq)
        return jnp.where(kq <= k_last, interior, exterior)

    w_out = jax.vmap(at_k)(k_ext)
    w_out = jnp.nan_to_num(w_out, nan=0.0, posinf=0.0, neginf=0.0)
    return w_out, k_ext


# HMcode-2020 lensing ratio: redshift sampling of sqrt(P_nl/P_lin).
N_A_HMCODE = 24
Z_MAX_HMCODE = 10.0


def hmcode_weyl_ratio(
    *,
    y_ev: jnp.ndarray | None = None,
    k_ev: jnp.ndarray,
    tau: jnp.ndarray,
    thermo: dict,
    params: dict,
    cfg: HierarchyConfig,
    k_out: jnp.ndarray,
    min_kh: float = MIN_KH_NONLINEAR,
    delta_m: jnp.ndarray | None = None,
    version: str = "mead2020",
    logT_AGN=7.8,
) -> jnp.ndarray:
    """CAMB ``MakeNonlinearSources`` ratio sqrt(P_nl/P_lin) with HMcode ``version``
    ("mead2020", CAMB's default, "mead2016", or "mead2020_feedback" with ``logT_AGN`` =
    CAMB ``HMCode_logT_AGN``).

    Pass either the evolved states ``y_ev`` or, cheaper, ``delta_m`` (n_k, n_tau).

    Returns shape ``(n_out, n_tau)`` on ``k_out`` [1/Mpc] × ``tau``. The linear
    total-matter spectrum Δ² = δ_m² P_R comes from the evolved states; above the
    last evolved k it is continued with δ_m ∝ k² T_EH(k) (the same shape used
    for the Weyl source). Ratio is 1 below ``min_kh`` [h/Mpc] and z > 10.
    """
    from gpucosmo.jax.eb.hmcode2020_jax import hmcode2020_ratio, linear_table_k_h

    def delta_m_one(yi, ki):
        return _background_and_stress(yi, ki, tau, params, thermo, cfg)["delta_m"]

    dm = delta_m if delta_m is not None else jax.vmap(delta_m_one)(y_ev, k_ev)  # (n_k, n_tau)
    a_tab, tau_tab = thermo["a"], thermo["tau_mpc"]
    a_nl = jnp.exp(jnp.linspace(-jnp.log1p(Z_MAX_HMCODE), 0.0, N_A_HMCODE))
    tau_nl = jnp.interp(a_nl, a_tab, tau_tab)
    dm_nl = jax.vmap(lambda row: jnp.interp(tau_nl, tau, row))(dm)  # (n_k, n_a)

    h = jnp.asarray(params["h"], dtype=jnp.float64)
    ns, As, kp = params["n_s"], params["As"], params["k_pivot"]
    kt = jnp.asarray(linear_table_k_h()) * h
    log_k_ev = jnp.log(k_ev)
    log_d2_ev = jnp.log(jnp.maximum(dm_nl**2, 1e-300)) + (
        jnp.log(As) + (ns - 1.0) * jnp.log(k_ev / kp)
    )[:, None]
    k_last = k_ev[-1]
    t_last = eh98_nowiggle_transfer(k_last, params)
    ext_shape = (4.0 * jnp.log(kt / k_last)
                 + 2.0 * jnp.log(eh98_nowiggle_transfer(kt, params) / t_last)
                 + (ns - 1.0) * jnp.log(kt / k_last))

    def table_one_a(col):
        # C¹ cubic: a linear interpolation over the evolved grid (k r_s Δln k ≈ 0.38) damps
        # the BAO wiggles of HMcode's input by ~2 % of their amplitude.
        inner = interp_cubic(jnp.log(kt), log_k_ev, col)
        return jnp.where(kt <= k_last, inner, col[-1] + ext_shape)

    tab = jax.vmap(table_one_a)(log_d2_ev.T)  # (n_a, 512)
    cosmo = dict(
        Omega_m=params["Omega_b"] + params["Omega_c"] + params.get("Omega_nu_massive_today", 0.0),
        Omega_b=params["Omega_b"],
        Omega_c=params["Omega_c"],
        Omega_nu=params.get("Omega_nu_massive_today", 0.0),
        Omega_v=params["Omega_de"],
        h=h,
        n_s=ns,
        w0=params.get("w0", -1.0),
        wa=params.get("wa", 0.0),
        T_cmb=params.get("T_CMB", 2.7255),
        # jnp, not float(): in the walker-batched CMB path the params dict is vmapped and
        # num_massive_nu is a traced array (only used arithmetically by hmcode2020).
        N_nu=jnp.asarray(params.get("num_massive_nu", 1), dtype=jnp.float64),
    )
    kh_out = jnp.asarray(k_out, dtype=jnp.float64) / h
    ratio = hmcode2020_ratio(kh_out, a_nl, tab, cosmo, version=version, logT_AGN=logT_AGN)  # (n_a, n_out)
    ratio = jnp.where(kh_out[None, :] < min_kh, 1.0, ratio)
    ratio = jnp.where(jnp.isfinite(ratio), ratio, 1.0)

    a_tau = jnp.interp(tau, tau_tab, a_tab)

    def to_tau(rk):
        return jnp.where(a_tau >= a_nl[0], jnp.interp(a_tau, a_nl, rk), 1.0)

    return jax.vmap(to_tau)(ratio.T)  # (n_out, n_tau)


def scale_weyl_nonlinear(
    weyl: jnp.ndarray,
    *,
    y_ev: jnp.ndarray,
    k_ev: jnp.ndarray,
    tau: jnp.ndarray,
    thermo: dict,
    params: dict,
    cfg: HierarchyConfig,
    p_r_ev: jnp.ndarray,
    enabled: bool = True,
) -> jnp.ndarray:
    """Apply CAMB-like nonlinear scaling to the Weyl source."""
    if not enabled:
        return weyl
    ratio = nonlinear_ratio_weyl(
        y_ev=y_ev,
        k_ev=k_ev,
        tau=tau,
        thermo=thermo,
        params=params,
        cfg=cfg,
        p_r_ev=p_r_ev,
    )
    return weyl * ratio


__all__ = [
    "MIN_KH_NONLINEAR",
    "hmcode_weyl_ratio",
    "eh98_nowiggle_transfer",
    "eh99_nu_growth",
    "extend_weyl_high_k",
    "nonlinear_ratio_weyl",
    "scale_weyl_nonlinear",
    "transfer_tot_from_y_grid",
]
