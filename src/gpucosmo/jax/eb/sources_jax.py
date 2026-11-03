#!/usr/bin/env python3
# This file is part of GPUCosmo. Copyright (C) 2025-2026 Leonel Medina-Varela.
# SPDX-License-Identifier: GPL-3.0-or-later
# Derived in part from the following, translated to JAX and modified for GPUCosmo:
# - CAMB (https://github.com/cmbant/CAMB), Copyright 2000-2025 Antony Lewis and Anthony Challinor.
#   Licence: LICENSES/CAMB.txt.
# - ISiTGR (https://github.com/mishakb/ISiTGR), Dossett, Ishak, Garcia-Quintero et al.,
#   built on CAMB, Copyright 2000-2018 Antony Lewis and Anthony Challinor.
#   Licence: LICENSES/ISiTGR.txt.
# See NOTICE.md.
"""CMB visibility and LOS source functions for jax_eb Cls.

Builds g(τ)=κ̇ e^{-κ} from opacity and CAMB ``OutputSources``-like
temperature / polarization sources from the scalar hierarchy state ``y(τ)``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from gpucosmo.jax.eb.dark_energy import cpl_f_de

from gpucosmo.jax.eb.interp_jax import interp_cubic
from gpucosmo.jax.eb.hierarchy_jax import (
    G_IX,
    IX_ETAK,
    IX_VB,
    HierarchyConfig,
    curvature_k,
    derivs,
    f_k,
    kf,
    ppf_diff_rhopi_add_term,
    ppf_perturbed_stress_energy,
    thermo_values,
)
from gpucosmo.jax.eb.massive_nu import nu_massive_source_terms, nu_massive_terms

jax.config.update("jax_enable_x64", True)

# Optional low-k floor for quadrupole /k² pieces. CAMB has no floor; with
# f64 and the evolve grid starting at qmin≈7e-6 the full quadrupole is
# stable. Default ``use_quad_floor=False`` matches CAMB exactly.
_QUAD_K_MIN = 1.0e-6


def _natural_cubic_d2(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Natural-cubic second derivatives (CAMB / Numerical Recipes ``spline``)."""
    x = jnp.asarray(x, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)
    n = x.shape[0]
    h = jnp.diff(x)

    # Interior tridiagonal system (points 1..n-2):
    # a[i]*y2[i-1] + b[i]*y2[i] + c[i]*y2[i+1] = rhs[i].
    a = h[:-1]
    b = 2.0 * (x[2:] - x[:-2])
    c = h[1:]
    rhs = 6.0 * ((y[2:] - y[1:-1]) / h[1:] - (y[1:-1] - y[:-2]) / h[:-1])
    # Direct tridiagonal solve (LAPACK gtsv / cuSPARSE gtsv2) instead of a
    # Thomas sweep as two ``lax.scan`` of n steps: same system, solved exactly
    # (strictly diagonally dominant, so no pivoting), without ~2n sequential
    # loop iterations (~40 ms on CPU for the 20821-node transfer thermo table).
    zero = jnp.zeros(1, dtype=jnp.float64)
    dl = jnp.concatenate([zero, a[1:]])
    du = jnp.concatenate([c[:-1], zero])
    interior = jax.lax.linalg.tridiagonal_solve(dl, b, du, rhs[:, None])[:, 0]
    return jnp.concatenate(
        [jnp.zeros(1, dtype=jnp.float64), interior, jnp.zeros(1, dtype=jnp.float64)]
    )


def _cubic_spline_interval_integrals(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """∫ y_spline over each interval [x[i], x[i+1]] (trapz + ``-dx²/24 (y''_i + y''_{i+1})``)."""
    dd = _natural_cubic_d2(x, y)
    dx = jnp.diff(x)
    return dx * ((y[1:] + y[:-1]) / 2.0 - (dx * dx) / 24.0 * (dd[1:] + dd[:-1]))


def optical_depth_and_visibility(
    tau: jnp.ndarray, opacity: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """κ(τ)=∫_τ^{τ0} κ̇ dτ' and g=κ̇ e^{-κ} via CAMB's cubic-spline integral.

    CAMB ``results.f90`` builds a ``TCubicSpline`` of opacity and calls
    ``IntegralArray`` (trapz + ``-dx²/24 (y''_i + y''_{i-1})``). Plain
    ``dpoints`` trapz overestimates κ★ by ~1%, which suppresses the visibility
    peak by the same factor and floors every acoustic Cls gate.
    """
    tau = jnp.asarray(tau, dtype=jnp.float64)
    opacity = jnp.asarray(opacity, dtype=jnp.float64)
    # Accumulated from τ0 backwards, as CAMB's reverse-array IntegralArray: I(τ0) − I(τ) from the
    # table start lost ε·∫κ̇ ≈ 1e-7 absolute (∫κ̇ ~ 3e8 from τ = 4e-4 Mpc) in the late-time κ.
    seg = _cubic_spline_interval_integrals(tau, opacity)
    kappa = jnp.concatenate([jnp.cumsum(seg[::-1])[::-1], jnp.zeros(1, dtype=jnp.float64)])
    # CAMB clamps emmu at exp(-69)
    emmu = jnp.exp(-jnp.minimum(jnp.maximum(kappa, 0.0), 69.0))
    g = opacity * emmu
    return kappa, g


def build_visibility_tables(thermo: dict) -> dict[str, jnp.ndarray]:
    """Add κ, g, ġ, g̈ tables to a thermo dict (device arrays).

    CAMB ``IonizationFunctionsAtTime`` supplies ``dvisibility`` /
    ``ddvisibility`` from the thermo spline. We match that with the
    analytic ``ġ = e^{-κ}(κ̈ + κ̇²)`` on the dense τ grid (FD of ``g`` on
    a sparse source grid is too noisy for the quadrupole).

    Also syncs ``taurend`` to CAMB cumulative-visibility (~599) so evolve RSA
    matches TimeSteps / CAMB ``DoLateRadTruncation``.
    """
    from gpucosmo.jax.eb.kgrid_jax import sync_thermo_taurend_camb_like

    tau = jnp.asarray(thermo["tau_mpc"], dtype=jnp.float64)
    opacity = jnp.asarray(thermo["opacity"], dtype=jnp.float64)
    kappa, g = optical_depth_and_visibility(tau, opacity)
    emmu = jnp.exp(-jnp.maximum(kappa, 0.0))
    dopacity = jnp.asarray(thermo["dopacity"], dtype=jnp.float64)
    # dκ/dτ = -opacity ⇒ ġ = e^{-κ}(dopacity + opacity²)
    gdot = emmu * (dopacity + opacity * opacity)
    dd_g = jnp.gradient(gdot, tau)
    out = dict(thermo)
    out["kappa"] = kappa
    out["visibility"] = g
    out["dvisibility"] = gdot
    out["ddvisibility"] = dd_g
    return sync_thermo_taurend_camb_like(out)

def _background_and_stress(
    y: jnp.ndarray,
    k: jnp.ndarray,
    tau: jnp.ndarray,
    params: dict,
    thermo: dict,
    cfg: HierarchyConfig,
):
    """Return background + constraint quantities along τ (``k``: the grid wavenumber q; with
    curvature the true k = √(q² − K) and Kf(ℓ) of CAMB, returned as ``"k"``/``"kfd"``)."""
    k, k2, kfd = curvature_k(k, params, cfg)
    H0 = params["H0_mpc"]
    fac = 3.0 * H0 * H0
    a, _, _ = thermo_values(tau, thermo, want_dopacity=False)
    a2 = a * a
    w0 = params.get("w0", -1.0)
    wa = params.get("wa", 0.0)
    f_de = cpl_f_de(a, w0, wa)
    grhog = fac * params["Omega_g"] / a2
    grhor = fac * params["Omega_nu"] / a2
    grhob = fac * params["Omega_b"] / a
    grhoc = fac * params["Omega_c"] / a
    grhov = fac * params["Omega_de"] * f_de * a2
    grhonu_t, gpnu_t, dgrho_nu, dgq_nu, dgpi_nu = nu_massive_terms(
        params, a, y, cfg
    )
    grho = grhob + grhoc + grhor + grhog + grhov + grhonu_t
    w_de = w0 + wa * (1.0 - a)
    gpres_noDE = (grhog + grhor) / 3.0 + gpnu_t
    gpres = gpres_noDE + w_de * grhov
    grhok = fac * params["Omega_k"] if cfg.curved else 0.0
    adotoa = jnp.sqrt(jnp.maximum((grho + grhok) / 3.0, 1e-30))

    etak = y[:, IX_ETAK]
    clxc = y[:, 1]
    clxb = y[:, 2]
    vb = y[:, IX_VB]
    clxg = y[:, G_IX]
    qg = y[:, G_IX + 1]
    pig = jnp.where(cfg.lmaxg >= 2, y[:, G_IX + 2], jnp.zeros_like(clxg))
    clxr = y[:, cfg.r_ix]
    qr = y[:, cfg.r_ix + 1]
    pir = jnp.where(cfg.lmaxnr >= 2, y[:, cfg.r_ix + 2], jnp.zeros_like(clxg))

    dgrho_matter = grhob * clxb + grhoc * clxc
    dgrho = dgrho_matter + grhog * clxg + grhor * clxr + dgrho_nu
    dgq = grhob * vb + grhog * qg + grhor * qr + dgq_nu
    dgpi = grhog * pig + grhor * pir + dgpi_nu

    dgrho_de = jnp.zeros_like(dgrho)
    dgq_de = jnp.zeros_like(dgq)
    Gammadot = jnp.zeros_like(dgrho)
    Gamma = jnp.zeros_like(dgrho)
    if cfg.use_de_ppf:
        Gamma = y[:, cfg.w_ix]
        dgrho_de, dgq_de, Gammadot = ppf_perturbed_stress_energy(
            Gamma=Gamma,
            dgrho=dgrho,
            dgq=dgq,
            grho=grho,
            grhov=grhov,
            w=w_de,
            gpres_noDE=gpres_noDE,
            etak=etak,
            adotoa=adotoa,
            k=k,
            kf1=kf(kfd, 1),
        )
        dgrho = dgrho + dgrho_de
        dgq = dgq + dgq_de

    kf1 = kf(kfd, 1)
    z = (0.5 * dgrho / k + etak) / jnp.maximum(adotoa, 1e-30)
    sigma = (z + 1.5 * dgq / jnp.maximum(k2, 1e-30)) / kf1
    phi = -((dgrho + 3.0 * dgq * adotoa / jnp.maximum(k, 1e-30)) / kf1 + dgpi) / (2.0 * k2)
    psi = phi - dgpi / jnp.maximum(k2, 1e-30)

    rho_m = grhob + grhoc + grhonu_t
    delta_m = jnp.where(
        rho_m > 0.0,
        (dgrho_matter + dgrho_nu) / jnp.maximum(rho_m, 1e-30),
        0.0,
    )
    return {
        "a": a,
        "adotoa": adotoa,
        "grho": grho,
        "gpres": gpres,
        "gpres_noDE": gpres_noDE,
        "grhog": grhog,
        "grhor": grhor,
        "grhov": grhov,
        "grhonu_t": grhonu_t,
        "gpnu_t": gpnu_t,
        "w_de": w_de,
        "etak": etak,
        "vb": vb,
        "clxg": clxg,
        "qg": qg,
        "pig": pig,
        "pir": pir,
        "dgrho": dgrho,
        "dgq": dgq,
        "dgpi": dgpi,
        "dgrho_de": dgrho_de,
        "dgq_de": dgq_de,
        "Gamma": Gamma,
        "Gammadot": Gammadot,
        "z": z,
        "sigma": sigma,
        "phi": phi,
        "psi": psi,
        "delta_m": delta_m,
        "dgrho_matter": dgrho_matter,
        "k2": k2,
        "k": k,
        "kfd": kfd,
        "grhok": grhok,
    }


def metric_potentials_from_y(
    y: jnp.ndarray,
    k: jnp.ndarray,
    tau: jnp.ndarray,
    params: dict,
    thermo: dict,
    cfg: HierarchyConfig,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return (phi, psi, sigma, delta_m) — CAMB Weyl φ and ψ=φ−Π/k²."""
    q = _background_and_stress(y, k, tau, params, thermo, cfg)
    return q["phi"], q["psi"], q["sigma"], q["delta_m"]


def _mg_temperature_source(*, q, k, ayprime, pigdot, pidot_sum, dgpi_3wp1_nu, params, g, gdot,
                           dd_g, emmu, opacity, dopacity, E2, E3, Edot2, Edot3, octg,
                           octgdot, qgdot, vbdot):
    """ISiTGR μ–Σ temperature source, Weyl potential Φ+Ψ and ISW (``mg_jax``).

    ISiTGR ``derivs`` OutputSources, ``use_mg`` branch (flat): ISW from
    (Φ̇ + Ψ̇)/2 and the non-integrated-by-parts Doppler / quadrupole terms, with
    σ from Φ and σ̇ = kΨ − ℋσ.
    """
    from gpucosmo.jax.eb import mg_jax

    a = q["a"]
    adotoa = q["adotoa"]
    grho, gpres = q["grho"], q["gpres"]
    grhog, grhor = q["grhog"], q["grhor"]
    dgrho, dgq, dgpi = q["dgrho"], q["dgq"], q["dgpi"]
    k2 = k * k
    Hdot = -(grho + 3.0 * gpres) / 6.0
    rho_delta = dgrho + 3.0 * adotoa * dgq / k
    grhonu_t, gpnu_t = q["grhonu_t"], q["gpnu_t"]
    dgpi_3wp1 = 2.0 * (grhog * q["pig"] + grhor * q["pir"]) + dgpi_3wp1_nu
    grhob = 3.0 * params["H0_mpc"] ** 2 * params["Omega_b"] / a
    grhoc = 3.0 * params["H0_mpc"] ** 2 * params["Omega_c"] / a
    rho_plus_p = (grhoc + grhob) + (grhor + grhog) * 4.0 / 3.0 + (grhonu_t + gpnu_t)
    mu, mudot, sig, _sigdot = mg_jax.mu_sigma(k=k, a=a, adotoa=adotoa, Hdot=Hdot,
                                              grhov=q["grhov"], grho=grho, params=params)
    psi, phi_n, sigma = mg_jax.newtonian_potentials(
        k=k, mu=mu, sig=sig, rho_delta=rho_delta, dgpi=dgpi, etak=q["etak"], adotoa=adotoa)
    etakdot = ayprime[:, IX_ETAK]
    sigmadot = k * psi - adotoa * sigma
    phidot_n = (etakdot - adotoa * sigmadot - Hdot * sigma) / k
    psidot = mg_jax.psi_dot_mg(k=k, mu=mu, mudot=mudot, adotoa=adotoa, rho_delta=rho_delta,
                               dgq=dgq, dgpi=dgpi, dgpi_3wplus1=dgpi_3wp1,
                               pidot_sum=pidot_sum, sigma=sigma, phidot=phidot_n, psi=psi,
                               Hdot=Hdot, rho_plus_p=rho_plus_p)
    isw = (psidot + phidot_n) * emmu
    pig, clxg, qg, vb = q["pig"], q["clxg"], q["qg"], q["vb"]
    s_t = (isw + g * pig / 16.0 + (3.0 / 8.0 * E2 + clxg / 4.0) * g
           + (11.0 / 10.0 * gdot * sigma
              + (-3.0 / 8.0 * E3 - 9.0 / 80.0 * octg + 3.0 / 40.0 * qg + vb) * gdot
              + (3.0 / 40.0 * qgdot + 21.0 / 10.0 * sigmadot + vbdot - 9.0 / 80.0 * octgdot
                 - 3.0 / 8.0 * Edot3) * g) / k
           + ((3.0 / 16.0 * dd_g - 9.0 / 160.0 * g * dopacity - 9.0 / 160.0 * gdot * opacity)
              * pig
              + (9.0 / 8.0 * Edot2 + 3.0 / 16.0 * pigdot - 27.0 / 80.0 * opacity * E2) * gdot
              + ((-9.0 / 160.0 * pigdot - 27.0 / 80.0 * Edot2) * opacity
                 - 27.0 / 80.0 * dopacity * E2) * g
              + 9.0 / 8.0 * dd_g * E2) / k2)
    return s_t, phi_n + psi, isw


def temperature_polarization_sources(
    y: jnp.ndarray,
    k: jnp.ndarray,
    tau: jnp.ndarray,
    visibility: jnp.ndarray,
    kappa: jnp.ndarray,
    params: dict,
    thermo: dict,
    cfg: HierarchyConfig,
    tau0: jnp.ndarray | float | None = None,
    *,
    return_components: bool = False,
    use_quad_floor: bool = False,
    sanitize: bool | None = None,
) -> tuple:
    """CAMB ``OutputSources``-like S_T, S_E, and Weyl φ along τ for one k.

    See ``equations.f90`` ~2870–2905. EE Cls later multiply by CAMB
    ``ctnorm=(ℓ−1)ℓ(ℓ+1)(ℓ+2)`` in the theory adapter.

    If ``return_components``, also returns a dict with ``isw``, ``monopole``,
    ``doppler``, ``quadrupole``, ``phidot``, ``phi``.

    ``sanitize``: replace non-finite values by 0. ``None`` (default) does so unless
    ``GPUCOSMO_STRICT_SOURCES=1``. ``cls_jax._source_one_k`` passes ``False`` and counts the
    non-finite modes itself (``cls_jax.CLS_SOURCE_FAILURE``), so a failure is never hidden.
    """
    q = _background_and_stress(y, k, tau, params, thermo, cfg)
    phi = q["phi"]
    sigma = q["sigma"]
    vb = q["vb"]
    clxg = q["clxg"]
    pig = q["pig"]
    pir = q["pir"]
    etak = q["etak"]
    adotoa = q["adotoa"]
    dgpi = q["dgpi"]
    dgq = q["dgq"]
    dgrho = q["dgrho"]
    grho = q["grho"]
    gpres = q["gpres"]
    grhog = q["grhog"]
    grhor = q["grhor"]
    k2 = q["k2"]
    k_grid = k  # the evolve wavenumber q (derivs takes it)
    k = q["k"]  # √(q² − K) (= q when flat)
    kfd = q["kfd"]
    kf1, kf2 = kf(kfd, 1), kf(kfd, 2)

    g = visibility
    emmu = jnp.exp(-jnp.maximum(kappa, 0.0))
    # Prefer CAMB-like thermo ġ/g̈ (dense-grid analytic); fall back to FD
    tau_th = thermo["tau_mpc"]
    if "dvisibility" in thermo and "ddvisibility" in thermo:
        gdot = interp_cubic(tau, tau_th, thermo["dvisibility"])
        dd_g = interp_cubic(tau, tau_th, thermo["ddvisibility"])
    else:
        gdot = jnp.gradient(g, tau)
        dd_g = jnp.gradient(gdot, tau)

    E2 = jnp.where(
        cfg.include_pol and cfg.n_pol >= 1,
        y[:, cfg.pol_base],
        jnp.zeros_like(clxg),
    )
    E3 = jnp.where(
        cfg.include_pol and cfg.n_pol >= 2,
        y[:, cfg.pol_base + 1],
        jnp.zeros_like(clxg),
    )
    octg = jnp.where(cfg.lmaxg >= 3, y[:, G_IX + 3], jnp.zeros_like(clxg))

    # CAMB uses ayprime from derivs (not FD of hierarchy rows on sparse τ)
    def _row(yi, ti):
        ayprime, _pig, pigdot_i = derivs(
            ti,
            yi,
            k_grid,
            params,
            thermo,
            tight_coupling=False,
            no_phot_multpoles=False,
            no_nu_multpoles=False,
            cfg=cfg,
        )
        return ayprime, pigdot_i

    ayprime, pigdot = jax.vmap(_row)(y, tau)
    vbdot = ayprime[:, IX_VB]
    qgdot = ayprime[:, G_IX + 1]
    octgdot = jnp.where(
        cfg.lmaxg >= 3, ayprime[:, G_IX + 3], jnp.zeros_like(clxg)
    )
    pirdot = jnp.where(
        cfg.lmaxnr >= 2, ayprime[:, cfg.r_ix + 2], jnp.zeros_like(clxg)
    )
    Edot2 = jnp.where(
        cfg.include_pol and cfg.n_pol >= 1,
        ayprime[:, cfg.pol_base],
        jnp.zeros_like(clxg),
    )
    Edot3 = jnp.where(
        cfg.include_pol and cfg.n_pol >= 2,
        ayprime[:, cfg.pol_base + 1],
        jnp.zeros_like(clxg),
    )

    polter = pig / 10.0 + 9.0 / 15.0 * E2
    polterdot = pigdot / 10.0 + 0.6 * Edot2
    polterddot_fd = jnp.gradient(polterdot, tau)

    # CAMB: diff_rhopi = pidot_sum - (4*dgpi + dgpi_diff)*adotoa + DE term
    # (equations.f90:2819-2830). Massive-ν contributions via MassiveNuVarsOut.
    pidot_sum = grhog * pigdot + grhor * pirdot
    dgpi_diff, pidot_nu, dgpi_3wp1_nu = nu_massive_source_terms(
        params, q["a"], adotoa, y, ayprime, cfg
    )
    pidot_sum = pidot_sum + pidot_nu
    de_term = jnp.zeros_like(pidot_sum)
    if cfg.use_de_ppf:
        de_term = ppf_diff_rhopi_add_term(
            dgrhoe=q["dgrho_de"],
            dgqe=q["dgq_de"],
            grho=grho,
            gpres=gpres,
            w=q["w_de"],
            adotoa=adotoa,
            k=k,
            grhov=q["grhov"],
            z=q["z"],
            k2=k2,
            Gammadot=q["Gammadot"],
            Gamma=q["Gamma"],
            kf1=kf1,
            grhok=q["grhok"],
        )
    diff_rhopi = pidot_sum - (4.0 * dgpi + dgpi_diff) * adotoa + de_term

    phidot = 0.5 * (
        adotoa * (-dgpi - 2.0 * k2 * phi)
        + dgq * k
        - diff_rhopi
        + k * sigma * (gpres + grho)
    ) / jnp.maximum(k2, 1e-30)
    sigmadot = -adotoa * sigma - 0.5 * dgpi / jnp.maximum(k, 1e-30) + k * phi

    # CAMB closed-form polterddot (equations.f90); FD if opacity path is wild
    opacity = interp_cubic(tau, tau_th, thermo["opacity"])
    dopacity = interp_cubic(tau, tau_th, thermo["dopacity"])
    polterddot_an = (
        -2.0 / 25.0 * adotoa * dgq / (jnp.maximum(k, 1e-30) * kf1)
        - 4.0 / 75.0 * adotoa * k * sigma
        - 4.0 / 75.0 * dgpi
        - 2.0 / 75.0 * dgrho / kf1
        - 3.0 / 50.0 * k * octgdot * kf2
        + 1.0 / 25.0 * k * qgdot
        - 0.2 * k * kf2 * Edot3
        + (-0.1 * pig + 0.7 * polter - 0.6 * E2) * dopacity
        + (-0.1 * pigdot + 0.7 * polterdot - 0.6 * Edot2) * opacity
    )
    # Early τ has huge dopacity; use FD polterddot there (g≈0 anyway)
    dop_ok = jnp.abs(dopacity) < 1.0e2
    polterddot = jnp.where(dop_ok, polterddot_an, polterddot_fd)

    isw = 2.0 * phidot * emmu
    monopole = (-etak / (jnp.maximum(k, 1e-30) * kf1) + 2.0 * phi + clxg / 4.0) * g
    doppler = ((sigma + vb) * gdot + (sigmadot + vbdot) * g) / jnp.maximum(k, 1e-30)

    # Full CAMB quadrupole (equations.f90 ~2896–2897). Optional low-k floor
    # (_QUAD_K_MIN) avoids /k² blow-up; set use_quad_floor=False for CAMB-exact.
    quadrupole_full = (
        (5.0 / 8.0)
        * (
            3.0 * polter * dd_g
            + 6.0 * polterdot * gdot
            + (k2 * polter + 3.0 * polterddot) * g
        )
        / jnp.maximum(k2, 1e-30)
    )
    quadrupole_lo = (5.0 / 8.0) * polter * g
    if use_quad_floor:
        quadrupole = jnp.where(k >= _QUAD_K_MIN, quadrupole_full, quadrupole_lo)
    else:
        quadrupole = quadrupole_full

    s_t = isw + monopole + doppler + quadrupole
    weyl = 2.0 * phi
    if cfg.mg:
        s_t_mg, weyl_mg, isw_mg = _mg_temperature_source(
            q=q, k=k, ayprime=ayprime, pigdot=pigdot, pidot_sum=pidot_sum,
            dgpi_3wp1_nu=dgpi_3wp1_nu, params=params, g=g, gdot=gdot, dd_g=dd_g, emmu=emmu,
            opacity=opacity, dopacity=dopacity, E2=E2, E3=E3, Edot2=Edot2,
            Edot3=Edot3, octg=octg, octgdot=octgdot, qgdot=qgdot, vbdot=vbdot)
        from gpucosmo.jax.eb.mg_jax import mg_active

        on = mg_active(q["a"])
        s_t = jnp.where(on, s_t_mg, s_t)
        weyl = jnp.where(on, weyl_mg, weyl)
        isw = jnp.where(on, isw_mg, isw)

    t0 = tau[-1] if tau0 is None else jnp.asarray(tau0, dtype=jnp.float64)
    # CAMB uses ang_dist = tau0 - tau with no chi floor and only gates on
    # tau < tau0 (equations.f90:2901-2904). The old (chi*k > 1) & (g > 1e-12)
    # guards zeroed the reionization polarization at low k.
    chi = jnp.maximum(t0 - tau, 0.0)
    ang = f_k(chi, params["curv"]) if cfg.curved else chi  # CAMB ang_dist = f_K(τ0 − τ)
    s_e = g * polter * (15.0 / 8.0) / jnp.maximum(ang * ang * k2, 1e-30)
    s_e = jnp.where(tau < t0, s_e, 0.0)

    import os

    # NaN → 0 for direct callers (diagnostic scripts); GPUCOSMO_STRICT_SOURCES=1 keeps the NaNs.
    # The C_ℓ pipeline calls with sanitize=False and reports failed modes (cls_jax).
    if sanitize is None:
        sanitize = os.environ.get("GPUCOSMO_STRICT_SOURCES", "0") != "1"
    fin = ((lambda v: jnp.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)) if sanitize
           else (lambda v: v))
    s_t, s_e, weyl = fin(s_t), fin(s_e), fin(weyl)
    comps = {"isw": fin(isw), "monopole": fin(monopole), "doppler": fin(doppler),
             "quadrupole": fin(quadrupole), "phidot": fin(phidot), "phi": fin(phi)}
    if return_components:
        return s_t, s_e, weyl, comps
    return s_t, s_e, weyl


__all__ = [
    "build_visibility_tables",
    "metric_potentials_from_y",
    "optical_depth_and_visibility",
    "temperature_polarization_sources",
]
