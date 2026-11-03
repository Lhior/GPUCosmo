#!/usr/bin/env python3
# This file is part of GPUCosmo. Copyright (C) 2025-2026 Leonel Medina-Varela.
# SPDX-License-Identifier: GPL-3.0-or-later
# Derived in part from the following, translated to JAX and modified for GPUCosmo:
# - CAMB (https://github.com/cmbant/CAMB), Copyright 2000-2025 Antony Lewis and Anthony Challinor.
#   Licence: LICENSES/CAMB.txt.
# See NOTICE.md.
"""Primordial tensor modes (gravitational waves, ratio r) for the jax_eb CMB pipeline.

Unlensed tensor C_ℓ^{TT,EE,BB,TE} in CAMB's conventions, flat geometry:

* primordial spectrum: CAMB ``TInitialPowerLaw`` with ``tensor_param_rpivot``
  (InitialPower.f90): P_t(k) = r P_s(k_t) exp[n_t ln(k/k_t) + n_t,run/2 ln²(k/k_t)],
  k_t = ``pivot_tensor``. ``nt=None`` is the inflation-consistency choice CAMB's
  Python ``InitialPowerLaw.set_params`` makes for ``nt=None`` (second order in r):
  n_t = −r/8 (2 − n_s − r/8), n_t,run = r/8 (r/8 + n_s − 1). (The Fortran default
  is n_t = 0; the consistency relation lives in the Python wrapper only.)
* perturbations: CAMB ``initialt`` / ``derivst`` (equations.f90): metric H and shear,
  photon intensity and E/B hierarchies, massless-ν hierarchy. Massive neutrinos are
  in the background and contribute their massless-equivalent density times the
  massless-ν quadrupole to the anisotropic stress — CAMB's own treatment whenever
  the eigenstate turns non-relativistic after 0.8 τ_maxvis (true for m_ν ≲ 1 eV,
  ``EvolveTensorMassiveNu``).
* sources: CAMB ``outputt`` (flat limit), integrated by parts into the j_ℓ/x², ε_ℓ, β_ℓ
  projection kernels (``_tensor_sources``); C_ℓ as CAMB ``CalcTensCls`` (dk/k measure,
  π/4 and ctnorm factors).

Integration scheme (not CAMB's): no tight-coupling switch. Before recombination the
Thomson terms are integrated implicitly with the L-stable, stiffly accurate IMEX
Runge–Kutta ARS(4,4,3) (Ascher, Ruuth & Spiteri 1997) — the photon moments relax
onto the tight-coupling solution π_γ = 32/45 k σ/κ̇, E_2 = π_γ/4 by themselves —
and from κ = ``KAPPA_SWITCH`` on (κ̇ Δτ < 1) everything is classical RK4 on a fixed
τ grid. Grid sizes (k, τ, ℓ) are static; grid positions follow the cosmology, so
the whole pipeline is jit/vmap-safe and differentiable in every input.
"""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np

from gpucosmo.jax.eb.dark_energy import cpl_f_de
from gpucosmo.jax.eb.interp_jax import interp_cubic

jax.config.update("jax_enable_x64", True)

# Reference τ0 [Mpc] for the static k grids (the scalar pipeline uses the same).
TAU0_REF = 14000.0
# Initial time [Mpc]: inside the thermo table (τ ≥ 0.05) and k τ ≤ 0.01 for k ≤ 0.17/Mpc
# (lmax = 600), where CAMB's super-horizon ``initialt`` series is accurate.
TAU_START = 0.06
# End of the logarithmic IMEX segment [Mpc].
TAU_LIN = 30.0
# Optical depth κ(τ) at the IMEX → RK4 switch: κ̇ ≈ 1.75/Mpc there (ΛCDM), so the RK4
# recombination steps (≈ 0.8 Mpc, κ̇Δτ ≈ 1.4 < 2.8) are stable, and e^{-κ} ~ 1e-35
# (no line-of-sight sources before).
KAPPA_SWITCH = 80.0
# Fine recombination segment ends at TAU_REC_END_FAC × visibility peak.
TAU_REC_END_FAC = 1.45


@dataclasses.dataclass(frozen=True)
class TensorAccuracy:
    """Static resolution of the tensor calculation.

    Defaults vs converged CAMB (AccuracyBoost 5, scripts/validate_tensors_vs_camb.py), lmax 600:
    max |ΔC/C| ≤ 8.1e-4 at 10 ≤ ℓ < 400 (TT, EE, BB), ≤ 2.1e-3 (BB) at 400–600; lmax 1000:
    ≤ 9.6e-4 to ℓ 600 and ≤ 1.0e-2 at 600–1000 (where tensor BB is ≲ 0.5 % of lensing BB);
    ≤ 0.8 % at ℓ < 10 (CAMB's own AccuracyBoost 3 vs 4 scatter in EE there is up to 1.3 %). Hierarchy truncations as CAMB's
    ``lmaxt = 8·lAccuracyBoost``, ``lmaxpolt = 4·lAccuracyBoost``, ``lmaxnrt = 6·lAccuracyBoost``
    (lmaxpolt = 8 leaves EE/BB 1e-3 off at ℓ 10–50).
    """

    lmaxt: int = 16
    lmaxpolt: int = 16
    lmaxnrt: int = 12
    # IMEX steps: logarithmic TAU_START → TAU_LIN, linear TAU_LIN → τ(κ = KAPPA_SWITCH).
    n_log: int = 120
    n_pre: int = 150
    # RK4 steps: τ(KAPPA_SWITCH) → TAU_REC_END_FAC·τ_peak, then → taurend.
    n_rec: int = 300
    n_post: int = 80
    # Late RK4 step [Mpc] (uniform from taurend to τ0; capped at 1.6/k_max for RK4 stability).
    # Late-time signal comes only from k τ ≲ 400 (shorter modes have decayed).
    late_dtau: float = 8.0
    # Store the late-time sources every ``late_store`` RK4 steps (LOS grid).
    late_store: int = 1
    # k_max τ0_ref / lmax (CAMB max_eta_k_tensor / max_l_tensor; CAMB's default 2 leaves BB
    # ~2 % low at ℓ 500–600 and ~0.4 % at 300–400). 6 (default, accuracy first): BB ≤ 2.1e-3 at
    # 400–600 vs converged CAMB, against 8.6e-3 with 4, for ~50 % more time.
    k_eta_max: float = 6.0
    # Source k grid (sources are cubic-spline interpolated onto the integration grid, CAMB
    # InterpolateSources): Δln k = src_dlnk up to Δk = src_dk_lo/τ0, linear Δk = src_dk_lo/τ0
    # up to src_k_mid/τ0, then Δk = src_dk_hi/τ0. Fine Δk at low k resolves the late-time
    # (reionization, ISW-like) sources, which oscillate in k with period 2π/τ.
    # ``src_on_int``: evolve every integration k instead (no interpolation).
    src_dlnk: float = 0.05
    src_dk_lo: float = 3.0
    src_k_mid: float = 170.0
    src_dk_hi: float = 12.0
    src_on_int: bool = False
    # Integration k grid (``_int_k_grid``): dk0 = 1.8/τ0/int_boost (CAMB), dk0/int_fine below
    # int_k_fine/τ0, at most k/int_lognum.
    int_boost: float = 0.8
    int_fine: float = 2.0
    int_k_fine: float = 150.0
    int_lognum: float = 40.0
    # ℓ samples: every ℓ up to ell_dense, then steps (cubic interpolation in between).
    ell_dense: int = 20
    ell_step_lo: int = 4
    ell_mid: int = 260
    ell_step_hi: int = 12
    # Line of sight over the late segment in ``los_blocks`` τ blocks, each keeping only
    # k ≤ los_ktau/τ (sources with k τ > 400 change C_ℓ by < 2e-5: the modes have decayed)
    # and the ℓ whose kernels are not identically ~0 there (x < ℓ − 6ℓ^{1/3}).
    los_ktau: float = 400.0
    los_blocks: int = 8
    # Projection-kernel table spacing in x = kχ: bessel_dx_lo up to bessel_x_lo, then bessel_dx.
    bessel_dx_lo: float = 0.1
    bessel_x_lo: float = 20.0
    bessel_dx: float = 0.4


DEFAULT_ACCURACY = TensorAccuracy()

# ARS(4,4,3) IMEX tableau (Ascher, Ruuth & Spiteri 1997, §2.8): explicit A_E, implicit A_I
# (first column zero, γ = 1/2, stiffly accurate: y_{n+1} = Y_5), stage times c.
_ARS_C = np.array([0.0, 0.5, 2.0 / 3.0, 0.5, 1.0])
_ARS_AE = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.0, 0.0, 0.0],
    [11.0 / 18.0, 1.0 / 18.0, 0.0, 0.0, 0.0],
    [5.0 / 6.0, -5.0 / 6.0, 0.5, 0.0, 0.0],
    [0.25, 1.75, 0.75, -1.75, 0.0],
])
_ARS_AI = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.5, 0.0, 0.0, 0.0],
    [0.0, 1.0 / 6.0, 0.5, 0.0, 0.0],
    [0.0, -0.5, 0.5, 0.5, 0.0],
    [0.0, 1.5, -1.5, 0.5, 0.5],
])


# ----------------------------------------------------------------------------------------
# Primordial spectrum
# ----------------------------------------------------------------------------------------
def tensor_spectral_index(r, ns, nt=None, ntrun=None):
    """(n_t, n_t,run): given values, or CAMB Python's inflation consistency for ``nt=None``."""
    r = jnp.asarray(r, dtype=jnp.float64)
    if nt is None:
        if ntrun is not None:
            raise ValueError("ntrun set but using inflation consistency (nt=None)")
        r8 = r / 8.0
        return -r8 * (2.0 - ns - r8), r8 * (r8 + ns - 1.0)
    return jnp.asarray(nt, dtype=jnp.float64), jnp.asarray(0.0 if ntrun is None else ntrun,
                                                           dtype=jnp.float64)


def _tensor_power_per_r(k, r, As, ns, nt, ntrun, pivot_tensor, pivot_scalar):
    """P_t(k)/r (r enters only through the inflation-consistency n_t)."""
    nt_, ntrun_ = tensor_spectral_index(r, ns, nt, ntrun)
    ps_pivot = As * (pivot_tensor / pivot_scalar) ** (ns - 1.0)
    lnrat = jnp.log(jnp.asarray(k, dtype=jnp.float64) / pivot_tensor)
    return ps_pivot * jnp.exp(lnrat * (nt_ + 0.5 * ntrun_ * lnrat))


def tensor_power(k, r, As, ns, *, nt=None, ntrun=None, pivot_tensor=0.05, pivot_scalar=0.05):
    """CAMB ``TInitialPowerLaw_TensorPower`` (``tensor_param_rpivot``, no scalar running)."""
    return r * _tensor_power_per_r(k, r, As, ns, nt, ntrun, pivot_tensor, pivot_scalar)


# ----------------------------------------------------------------------------------------
# Static layout of the tensor hierarchy
# ----------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _layout(lt: int, lp: int, ln: int) -> dict:
    """State-vector layout [H, σ, π_γ,2..lt, E_2..lp, B_2..lp, π_ν,2..ln] and the
    per-ℓ coefficients of CAMB ``derivst`` (flat: Kft = aux = 1, cothxor = 1/τ)."""
    if lt < 3 or lp < 3 or ln < 3:
        raise ValueError("tensor hierarchies need lmax >= 3")
    g0 = 2
    e0 = g0 + lt - 1
    b0 = e0 + lp - 1
    n0 = b0 + lp - 1
    n = n0 + ln - 1

    def tensfac(l):
        return (l + 3.0) * (l - 1.0) / (l + 1.0)

    def lower(L, trunc_form):
        # coefficient of X_{l-1} (× k): l/(2l+1), l=3..L-1; at l=L the truncation
        # L/(L-2) (intensity/ν) or l/(2l+1) (E/B); none at l=2.
        ls = np.arange(2, L + 1, dtype=np.float64)
        c = ls / (2 * ls + 1)
        c[0] = 0.0
        if trunc_form:
            c[-1] = L / (L - 2.0)
        return c

    def upper_int(L):
        # coefficient of X_{l+1} (× k) for intensity/ν: tensfac(l)/(2l+1); 0 at l=L.
        ls = np.arange(2, L + 1, dtype=np.float64)
        c = tensfac(ls) / (2 * ls + 1)
        c[-1] = 0.0
        return c

    def upper_pol(L):
        ls = np.arange(2, L + 1, dtype=np.float64)
        c = tensfac(ls) ** 2 / ((ls + 1) * (2 * ls + 1))
        c[-1] = 0.0
        return c

    lsp = np.arange(2, lp + 1, dtype=np.float64)
    trunc_g = np.zeros(lt - 1)
    trunc_g[-1] = lt + 3.0
    trunc_n = np.zeros(ln - 1)
    trunc_n[-1] = ln + 3.0
    # Opacity operator: diagonal −1 on every photon moment except the (π_γ2, E_2) block.
    damp = np.zeros(n)
    damp[g0:n0] = 1.0
    damp[g0] = 0.0
    damp[e0] = 0.0
    return dict(
        n=n, g0=g0, e0=e0, b0=b0, n0=n0, lt=lt, lp=lp, ln=ln,
        lo_g=lower(lt, True), up_g=upper_int(lt), trunc_g=trunc_g,
        lo_p=lower(lp, False), up_p=upper_pol(lp), d4_p=4.0 / (lsp * (lsp + 1.0)),
        lo_n=lower(ln, True), up_n=upper_int(ln), trunc_n=trunc_n,
        damp=damp,
    )


def _shift_lo(x):
    """X_{l-1} along the last axis (zero below ℓ = 2)."""
    return jnp.concatenate([jnp.zeros_like(x[..., :1]), x[..., :-1]], axis=-1)


def _shift_hi(x):
    """X_{l+1} along the last axis (zero above the truncation)."""
    return jnp.concatenate([x[..., 1:], jnp.zeros_like(x[..., :1])], axis=-1)


def _rhs_free(y, k, adotoa, cg, cnu, itau, L):
    """Non-Thomson part of CAMB ``derivst`` (flat). y: (n_k, n); k: (n_k, 1)."""
    g0, e0, b0, n0 = L["g0"], L["e0"], L["b0"], L["n0"]
    H = y[:, 0:1]
    S = y[:, 1:2]
    G = y[:, g0:e0]
    E = y[:, e0:b0]
    B = y[:, b0:n0]
    N = y[:, n0:]
    src = (8.0 / 15.0) * k * S
    lo_g, up_g, trunc_g = L["lo_g"], L["up_g"], L["trunc_g"]
    dG = k * (lo_g * _shift_lo(G) - up_g * _shift_hi(G)) - (trunc_g * itau) * G
    dG = dG.at[:, 0:1].add(src)
    lo_p, up_p, d4 = L["lo_p"], L["up_p"], L["d4_p"]
    dE = k * (lo_p * _shift_lo(E) - up_p * _shift_hi(E) + d4 * B)
    dB = k * (lo_p * _shift_lo(B) - up_p * _shift_hi(B) - d4 * E)
    lo_n, up_n, trunc_n = L["lo_n"], L["up_n"], L["trunc_n"]
    dN = k * (lo_n * _shift_lo(N) - up_n * _shift_hi(N)) - (trunc_n * itau) * N
    dN = dN.at[:, 0:1].add(src)
    rhopi = cg * G[:, 0:1] + cnu * N[:, 0:1]
    dS = -2.0 * adotoa * S + k * H - rhopi / k
    dH = -k * S
    return jnp.concatenate([dH, dS, dG, dE, dB, dN], axis=1)


def _thomson(y, L):
    """Thomson operator O·y (multiply by κ̇): −(π_γ2 − P), −(E_2 − P), P = π_γ2/10 + 3E_2/5,
    and −X on every other photon moment."""
    g0, e0 = L["g0"], L["e0"]
    out = -L["damp"] * y
    pig, e2 = y[:, g0], y[:, e0]
    polter = 0.1 * pig + 0.6 * e2
    out = out.at[:, g0].set(-(pig - polter))
    out = out.at[:, e0].set(-(e2 - polter))
    return out


def _thomson_solve(rhs, c, L):
    """Solve (I − c O) Y = rhs for Y (c = h a_ii κ̇ ≥ 0)."""
    g0, e0 = L["g0"], L["e0"]
    damp = L["damp"]
    y = rhs / (1.0 + c * damp)
    r_g, r_e = rhs[:, g0], rhs[:, e0]
    # [[1 + 0.9c, −0.6c], [−0.1c, 1 + 0.4c]] (π_γ2, E_2) block.
    det = 1.0 + 1.3 * c + 0.3 * c * c
    y_g = ((1.0 + 0.4 * c) * r_g + 0.6 * c * r_e) / det
    y_e = (0.1 * c * r_g + (1.0 + 0.9 * c) * r_e) / det
    return y.at[:, g0].set(y_g).at[:, e0].set(y_e)


# ----------------------------------------------------------------------------------------
# Background along the τ grid
# ----------------------------------------------------------------------------------------
def _background(tau, params, thermo):
    """(adotoa, κ̇, grhog/a², (grhor + grhormass)/a², 1/τ) at τ (CAMB units, Mpc⁻¹)."""
    from gpucosmo.jax.eb.massive_nu import nu_rho_p_omega

    logt = jnp.log(tau)
    log_tab = jnp.log(thermo["tau_mpc"])
    a = interp_cubic(logt, log_tab, thermo["a"])
    opac = interp_cubic(logt, log_tab, thermo["opacity"])
    fac = 3.0 * params["H0_mpc"] ** 2
    a2 = a * a
    grho_rad = fac * (params["Omega_g"] + params["Omega_nu"]) / a2
    grho_m = fac * (params["Omega_b"] + params["Omega_c"]) / a
    grho_de = fac * params["Omega_de"] * cpl_f_de(a, params["w0"], params["wa"]) * a2
    om_mass = params.get("Omega_nu_mass", 0.0)
    grhormass = fac * jnp.sum(jnp.asarray(om_mass, dtype=jnp.float64), axis=-1) / a2
    grho_nu = fac * nu_rho_p_omega(params, a)[0] / a2
    adotoa = jnp.sqrt((grho_rad + grho_m + grho_de + grho_nu) / 3.0)
    cg = fac * params["Omega_g"] / a2
    cnu = fac * params["Omega_nu"] / a2 + grhormass
    return jnp.stack([adotoa, opac, cg, cnu, 1.0 / tau], axis=-1)


def _source_thermo(tau, thermo):
    """(e^{-κ}, g) at τ."""
    t = thermo["tau_mpc"]
    expk = jnp.exp(-jnp.clip(interp_cubic(tau, t, thermo["kappa"]), 0.0, 690.0))
    vis = interp_cubic(tau, t, thermo["visibility"])
    return jnp.stack([expk, vis], axis=-1)


def _tensor_sources(y, k, sq, L):
    """Line-of-sight sources (S_T, g·P) at one τ; P = π_γ2/10 + 3E_2/5 (CAMB ``polter``).

    CAMB ``outputt`` (flat) writes Δ_X = ∫dτ s_X j_ℓ(kχ) with s_T = (σ e^{-κ} +
    15/8 g P/k)/(kχ²) and s_E, s_B containing g'', g', P'', P'. Integrating those by parts
    (boundary terms vanish for ℓ ≥ 2) gives the same Δ_ℓ as
        Δ_T = ∫dτ [k σ e^{-κ} + 15/8 g P] j_ℓ(x)/x²,
        Δ_E = 15/8 ∫dτ g P [j_ℓ'' + 4j_ℓ'/x + 2j_ℓ/x² − j_ℓ],
        Δ_B = 15/4 ∫dτ g P [j_ℓ' + 2j_ℓ/x],     x = kχ,
    with no derivatives of the visibility and no 1/χ singularity. (CAMB's form with the
    thermo tables' g', g'' — consistent with g to ~1e-3 — left EE 1–2 % low at ℓ 10–100:
    the terms of the E source cancel to ~(k Δτ_vis)² of their size there.)
    """
    g0, e0 = L["g0"], L["e0"]
    expk, vis = sq[0], sq[1]
    gp = vis * (0.1 * y[:, g0] + 0.6 * y[:, e0])
    s_t = k[:, 0] * y[:, 1] * expk + 15.0 / 8.0 * gp
    return jnp.stack([s_t, gp], axis=0)


def _initial_state(k, params, L):
    """CAMB ``initialt`` (flat, H = 1) at τ = TAU_START."""
    om_nu = params["Omega_nu"] + jnp.sum(jnp.atleast_1d(jnp.asarray(params.get("Omega_nu_mass", 0.0))), axis=-1)
    bigR = om_nu / (om_nu + params["Omega_g"])
    x = k[:, 0] * TAU_START
    elec = -(2.0 * bigR + 10.0) / (4.0 * bigR + 15.0)
    y = jnp.zeros((k.shape[0], L["n"]), dtype=jnp.float64)
    y = y.at[:, 0].set(1.0)
    y = y.at[:, 1].set(-2.5 / (bigR + 5.0) * x * elec)
    y = y.at[:, L["n0"]].set(-2.0 / 3.0 / (bigR + 5.0) * x ** 2 * elec)
    y = y.at[:, L["n0"] + 1].set(-2.0 / 21.0 / (bigR + 5.0) * x ** 3 * elec)
    return y


# ----------------------------------------------------------------------------------------
# Time integration
# ----------------------------------------------------------------------------------------
def _imex_evolve(y0, k, tau_nodes, params, thermo, L):
    """ARS(4,4,3) from tau_nodes[0] to tau_nodes[-1] (Thomson terms implicit)."""
    h = jnp.diff(tau_nodes)
    t_st = tau_nodes[:-1, None] + h[:, None] * _ARS_C[None, :]  # (n, 5)
    bg = _background(t_st, params, thermo)  # (n, 5, 5)
    ae, ai = _ARS_AE, _ARS_AI

    def step(y, xs):
        hh, b = xs
        fe = [None] * 5
        fi = [None] * 5
        ys = y
        for i in range(5):
            if i > 0:
                acc = y
                for j in range(i):
                    if ae[i, j] != 0.0:
                        acc = acc + (hh * ae[i, j]) * fe[j]
                    if ai[i, j] != 0.0:
                        acc = acc + (hh * ai[i, j]) * fi[j]
                ys = _thomson_solve(acc, hh * ai[i, i] * b[i, 1], L)
            if i < 4:
                adotoa, opac, cg, cnu, itau = (b[i, m] for m in range(5))
                fe[i] = _rhs_free(ys, k, adotoa, cg, cnu, itau, L)
                fi[i] = opac * _thomson(ys, L)
        return ys, None

    # Rematerialized in reverse mode: only the carry is stored per step.
    y1, _ = jax.lax.scan(jax.checkpoint(step), y0, (h, bg))
    return y1


def _rk4_evolve(y0, k, tau_nodes, params, thermo, L, store_every: int = 1):
    """Classical RK4 over tau_nodes; returns y_end and the sources (S_T, g·P) at every
    ``store_every``-th node after the first: (n_out, 2, n_k)."""
    h = jnp.diff(tau_nodes)
    n = h.shape[0]
    if n % store_every:
        raise ValueError("number of RK4 steps must be a multiple of store_every")
    bg_node = _background(tau_nodes, params, thermo)  # (n+1, 5)
    bg_mid = _background(tau_nodes[:-1] + 0.5 * h, params, thermo)  # (n, 5)
    sq = _source_thermo(tau_nodes[1:], thermo)  # (n, 2)

    def rhs(y, b):
        return _rhs_free(y, k, b[0], b[2], b[3], b[4], L) + b[1] * _thomson(y, L)

    def one(carry, xs):
        y, f = carry
        hh, bm, bn = xs
        k2 = rhs(y + (0.5 * hh) * f, bm)
        k3 = rhs(y + (0.5 * hh) * k2, bm)
        k4 = rhs(y + hh * k3, bn)
        y1 = y + (hh / 6.0) * (f + 2.0 * k2 + 2.0 * k3 + k4)
        f1 = rhs(y1, bn)
        return (y1, f1), None

    def group(carry, xs):
        hs, bms, bns, sqs = xs
        if store_every == 1:
            carry, _ = one(carry, (hs[0], bms[0], bns[0]))
        else:
            carry, _ = jax.lax.scan(one, carry, (hs, bms, bns))
        y1, _ = carry
        return carry, _tensor_sources(y1, k, sqs[-1], L)

    m = store_every
    xs = (h.reshape(n // m, m), bg_mid.reshape(n // m, m, 5),
          bg_node[1:].reshape(n // m, m, 5), sq.reshape(n // m, m, 2))
    f0 = rhs(y0, bg_node[0])
    (y1, _), src = jax.lax.scan(jax.checkpoint(group), (y0, f0), xs)
    return y1, src


def _tau_segments(thermo, plan):
    """Cosmology-dependent node positions of the static-size τ segments."""
    acc = plan["acc"]
    tau_tab = thermo["tau_mpc"]
    tau0 = tau_tab[-1]
    log_k = jnp.log(jnp.maximum(thermo["kappa"], 1e-300))
    # κ decreases monotonically: τ(κ = KAPPA_SWITCH) on the reversed table.
    tau1 = jnp.exp(jnp.interp(np.log(KAPPA_SWITCH), log_k[::-1], jnp.log(tau_tab)[::-1]))
    vis = thermo["visibility"]
    from gpucosmo.jax.eb.kgrid_jax import refined_peak

    early = tau_tab < 0.25 * tau0
    tau_peak = refined_peak(jnp.log(tau_tab), jnp.where(early, vis, 0.0))
    tau_c = jnp.maximum(TAU_REC_END_FAC * tau_peak, tau1 + 50.0)
    tau2 = jnp.maximum(jnp.asarray(thermo.get("taurend", 2.15 * tau_peak)), tau_c + 100.0)
    tau_log = TAU_START * (TAU_LIN / TAU_START) ** (np.arange(acc.n_log + 1) / acc.n_log)
    seg_a = jnp.asarray(tau_log)
    seg_b = jnp.linspace(TAU_LIN, tau1, acc.n_pre + 1)
    seg_c = jnp.concatenate([jnp.linspace(tau1, tau_c, acc.n_rec + 1),
                             jnp.linspace(tau_c, tau2, acc.n_post + 1)[1:]])
    seg_d = jnp.linspace(tau2, tau0, plan["n_late"] + 1)
    return seg_a, seg_b, seg_c, seg_d, tau0


# ----------------------------------------------------------------------------------------
# Static plan: k grids, ℓ samples, Bessel table
# ----------------------------------------------------------------------------------------
def _int_k_grid(k_min: float, k_max: float, acc: TensorAccuracy) -> np.ndarray:
    """C_ℓ integration k grid (CAMB ``SetkValuesForInt``-like, spacing continuous):
    Δk = min(k/int_lognum, Δk_lo) with Δk_lo = dk0/int_fine below int_k_fine/τ0, ramping
    linearly to dk0 = 1.8/τ0/int_boost over the next int_k_fine/τ0.

    Δ_ℓ(k)² oscillates in k with period π/χ ≈ 2.3e-4/Mpc, which CAMB's dk0 = 1.8/τ0
    undersamples; that is harmless where the integrand switches on smoothly (ℓ ≳ 30), but
    at ℓ ≲ 20 (and at abrupt spacing changes) it costs up to ~1–2 % in TT.
    """
    dk0 = 1.8 / TAU0_REF / float(acc.int_boost)
    dk_lo = dk0 / float(acc.int_fine)
    kb = acc.int_k_fine / TAU0_REF
    ks = [k_min]
    k = k_min
    while k < k_max:
        if k < kb:
            dk_cap = dk_lo
        else:
            dk_cap = dk_lo + (dk0 - dk_lo) * min(1.0, (k - kb) / kb)
        k = k + min(k / acc.int_lognum, dk_cap)
        ks.append(k)
    ks = np.array(ks)
    ks[-1] = k_max if ks[-1] - k_max < 0.5 * (ks[-1] - ks[-2]) else ks[-1]
    return ks


def _src_k_grid(k_min: float, k_max: float, acc: TensorAccuracy) -> np.ndarray:
    dk_lo = acc.src_dk_lo / TAU0_REF
    dk_hi = acc.src_dk_hi / TAU0_REF
    k_mid = max(acc.src_k_mid / TAU0_REF, dk_lo / acc.src_dlnk)
    k_sw = dk_lo / acc.src_dlnk
    n_log = max(2, int(np.ceil(np.log(k_sw / k_min) / acc.src_dlnk)))
    k_lo = np.exp(np.linspace(np.log(k_min), np.log(k_sw), n_log + 1))
    n_mid = int(np.ceil((k_mid - k_sw) / dk_lo))
    k_md = np.linspace(k_sw, k_mid, n_mid + 1)[1:] if n_mid > 0 else np.zeros(0)
    n_hi = int(np.ceil((k_max - k_mid) / dk_hi))
    k_hi = k_mid + dk_hi * np.arange(1, n_hi + 1)
    return np.concatenate([k_lo, k_md, k_hi])


def _ell_samples(lmax: int, acc: TensorAccuracy) -> np.ndarray:
    ls = list(range(2, min(lmax, acc.ell_dense) + 1))
    ls += list(range(acc.ell_dense + acc.ell_step_lo, min(lmax, acc.ell_mid) + 1, acc.ell_step_lo))
    ls += list(range(ls[-1] + acc.ell_step_hi, lmax + 1, acc.ell_step_hi))
    if ls[-1] != lmax:
        ls.append(lmax)
    return np.array(sorted(set(ls)), dtype=np.int64)


def _kernel_values(ell: int, x: np.ndarray) -> np.ndarray:
    """(j_ℓ/x², ε_ℓ, β_ℓ) at x ≥ 0: ε = j'' + 4j'/x + 2j/x² − j = 2j'/x + (ℓ(ℓ+1) + 2) j/x²
    − 2j (Bessel equation), β = j' + 2j/x; limits at x = 0."""
    from scipy.special import spherical_jn

    xs = np.where(x > 0, x, 1.0)
    j = spherical_jn(ell, xs)
    jp = spherical_jn(ell, xs, derivative=True)
    u = j / xs**2
    eps = 2.0 * jp / xs + (ell * (ell + 1.0) + 2.0) * j / xs**2 - 2.0 * j
    beta = jp + 2.0 * j / xs
    at0 = x == 0
    u = np.where(at0, 1.0 / 15.0 if ell == 2 else 0.0, u)
    eps = np.where(at0, 0.8 if ell == 2 else 0.0, eps)
    beta = np.where(at0, 0.0, beta)
    return np.stack([u, eps, beta])


def _bessel_table(ells: np.ndarray, x_max: float, acc: TensorAccuracy) -> dict:
    """Cubic splines (not-a-knot) of the projection kernels (j_ℓ/x², ε_ℓ, β_ℓ) on a
    piecewise-uniform x grid: Δx = ``bessel_dx_lo`` below ``bessel_x_lo``, ``bessel_dx`` above
    (≤ 3e-5 of the kernel peak for ℓ ≥ 10 at Δx = 0.4). ``coeffs[m, p, j, i]`` (float32):
    kernel m, power p (Horner order t³, t², t, 1), ℓ index j, interval i (t = x − x_i).
    """
    from scipy.interpolate import CubicSpline

    dx_lo, x_lo, dx = float(acc.bessel_dx_lo), float(acc.bessel_x_lo), float(acc.bessel_dx)
    n_lo = int(round(x_lo / dx_lo))
    x_lo = n_lo * dx_lo
    n_hi = int(np.ceil((x_max - x_lo) / dx))
    knots = np.concatenate([dx_lo * np.arange(n_lo), x_lo + dx * np.arange(n_hi + 1)])
    n_int = knots.size - 1
    coeffs = np.zeros((3, 4, len(ells), n_int), dtype=np.float32)
    for j, ell in enumerate(ells):
        vals = _kernel_values(int(ell), knots)
        for m in range(3):
            coeffs[m, :, j, :] = CubicSpline(knots, vals[m], bc_type="not-a-knot").c
    return dict(coeffs=coeffs, knots=knots, n_lo=n_lo, x_lo=x_lo, dx_lo=dx_lo, dx=dx,
                n_int=n_int)


def _bessel_index(x, bess):
    """Interval index and offset of x on the piecewise-uniform kernel grid."""
    i_lo = jnp.floor(x / bess["dx_lo"])
    i_hi = bess["n_lo"] + jnp.floor((x - bess["x_lo"]) / bess["dx"])
    i = jnp.where(x < bess["x_lo"], i_lo, i_hi)
    i = jnp.clip(i.astype(jnp.int32), 0, bess["n_int"] - 1)
    return i, x - jnp.asarray(bess["knots"])[i]


def _spline_weights(x_src: np.ndarray, x_out: np.ndarray) -> dict:
    """Static pieces of natural-cubic-spline interpolation from x_src to x_out."""
    i = np.clip(np.searchsorted(x_src, x_out, side="right") - 1, 0, len(x_src) - 2)
    ho = x_src[i + 1] - x_src[i]
    a0 = (x_src[i + 1] - x_out) / ho
    b0 = 1.0 - a0
    return dict(i=i, a0=a0, b0=b0, ho=ho)


def _spline_d2(y, x):
    """Natural-spline second derivatives of y (n, m) along axis 0 (x static, increasing)."""
    h = np.diff(x)
    n = x.shape[0]
    rhs = 6.0 * ((y[2:] - y[1:-1]) / h[1:, None] - (y[1:-1] - y[:-2]) / h[:-1, None])
    dl = np.concatenate([[0.0], h[1:-1]])
    du = np.concatenate([h[1:-1], [0.0]])
    d = 2.0 * (x[2:] - x[:-2])
    inner = jax.lax.linalg.tridiagonal_solve(jnp.asarray(dl), jnp.asarray(d), jnp.asarray(du), rhs)
    z = jnp.zeros((1, y.shape[1]), dtype=y.dtype)
    return jnp.concatenate([z, inner, z], axis=0) if n > 2 else jnp.zeros_like(y)


def _spline_eval(y, d2, w):
    i, a0, b0, ho = w["i"], w["a0"][:, None], w["b0"][:, None], w["ho"][:, None]
    return (a0 * y[i] + b0 * y[i + 1]
            - a0 * b0 * ((1.0 + a0) * d2[i] + (1.0 + b0) * d2[i + 1]) * ho * ho / 6.0)


@functools.lru_cache(maxsize=8)
def tensor_plan(lmax: int = 600, acc: TensorAccuracy = DEFAULT_ACCURACY) -> dict:
    """Host-side static grids for ``tensor_cls`` (cached per (lmax, acc))."""
    lmax = int(lmax)
    k_max = acc.k_eta_max * lmax / TAU0_REF
    k_min = 0.1 / TAU0_REF
    k_int = _int_k_grid(k_min, k_max, acc)
    if acc.src_on_int:
        k_src = k_int
    else:
        k_src = _src_k_grid(k_min * 0.999, k_max * 1.0001 + acc.src_dk_hi / TAU0_REF, acc)
    ells = _ell_samples(lmax, acc)
    # τ0 ≲ 16000 Mpc for any sensible cosmology; x = kχ ≤ k_max τ0.
    bess = _bessel_table(ells, 1.15 * k_max * 16000.0, acc)
    # RK4 on the free-streaming hierarchies is stable for k Δτ < 2.8: keep k_max Δτ ≤ 1.6 at
    # τ0 = TAU0_REF (≤ 1.85 at TAU0_MAX) whatever lmax.
    n_late = int(np.ceil((TAU0_REF - 600.0) / min(acc.late_dtau, 1.6 / k_max)))
    n_late += (-n_late) % acc.late_store
    dlnk = np.empty_like(k_int)
    dlnk[1:-1] = 0.5 * (k_int[2:] - k_int[:-2])
    dlnk[0] = k_int[1] - k_int[0]
    dlnk[-1] = k_int[-1] - k_int[-2]
    dlnk /= k_int
    return dict(acc=acc, lmax=lmax, k_int=k_int, k_src=k_src, ells=ells, bessel=bess,
                n_late=n_late, dlnk=dlnk, kspline=_spline_weights(k_src, k_int),
                los_blocks=_los_blocks(acc, k_int, ells, n_late))


# Bounds over cosmologies for the static LOS blocks: τ0 ≤ TAU0_MAX, and late-segment node
# positions ≥ _TAU_SAFETY × their ΛCDM (τ0 = TAU0_REF) values (τ0 ≥ 12000 Mpc).
TAU0_MAX = 16000.0
_TAU_SAFETY = 0.85


def _los_blocks(acc: TensorAccuracy, k_int, ells, n_late: int) -> tuple:
    """Static (i0, i1, n_k, n_ℓ) LOS blocks over the source τ nodes: the recombination
    segment with every k and ℓ, then ``los_blocks`` late blocks with k ≤ los_ktau/τ_lo and
    ℓ − 6ℓ^{1/3} ≤ k_cut·χ_max (k_int and ells ascending, so prefixes)."""
    n_c = acc.n_rec + acc.n_post
    n_d = n_late // acc.late_store
    blocks = [(0, n_c, len(k_int), len(ells))]
    edges = np.linspace(0, n_d, acc.los_blocks + 1).round().astype(int)
    lcut = ells - 6.0 * ells ** (1.0 / 3.0)
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        tau_lo = _TAU_SAFETY * (600.0 + (a + 1) * acc.late_store * (TAU0_REF - 600.0) / n_late)
        k_cut = acc.los_ktau / tau_lo
        nk = int(min(len(k_int), np.searchsorted(k_int, k_cut) + 1))
        x_max = k_int[nk - 1] * (TAU0_MAX - tau_lo)
        nl = int(max(1, np.sum(lcut <= x_max)))
        blocks.append((n_c + int(a), n_c + int(b), nk, nl))
    return tuple(blocks)


# ----------------------------------------------------------------------------------------
# Line of sight and C_ℓ
# ----------------------------------------------------------------------------------------
def _trapz_weights(t):
    d = jnp.zeros_like(t)
    d = d.at[1:-1].set(0.5 * (t[2:] - t[:-2]))
    d = d.at[0].set(0.5 * (t[1] - t[0]))
    return d.at[-1].set(0.5 * (t[-1] - t[-2]))


def tensor_sources(params: dict, thermo: dict, plan: dict):
    """Tensor sources on the source k grid: (tau_src (n_τ,), sources (2, n_k_src, n_τ), τ0).

    Rows: S_T = k σ e^{-κ} + 15/8 g P and g P (``_tensor_sources``).
    """
    acc = plan["acc"]
    L = _layout(acc.lmaxt, acc.lmaxpolt, acc.lmaxnrt)
    k = jnp.asarray(plan["k_src"])[:, None]
    seg_a, seg_b, seg_c, seg_d, tau0 = _tau_segments(thermo, plan)
    y = _initial_state(k, params, L)
    y = _imex_evolve(y, k, jnp.concatenate([seg_a, seg_b[1:]]), params, thermo, L)
    y, src_c = _rk4_evolve(y, k, seg_c, params, thermo, L)
    _, src_d = _rk4_evolve(y, k, seg_d, params, thermo, L, store_every=acc.late_store)
    tau_src = jnp.concatenate([seg_c[1:], seg_d[acc.late_store::acc.late_store]])
    src = jnp.concatenate([src_c, src_d], axis=0)  # (n_τ, 2, n_k)
    return tau_src, jnp.transpose(src, (1, 2, 0)), tau0


def _kernel_eval(coeffs, m, j, xi, t):
    """Kernel m of ℓ index j and its x-derivative at (xi, t)."""
    c = coeffs[m, :, j].astype(jnp.float64)  # (4, n_int)
    c0, c1, c2, c3 = c[0][xi], c[1][xi], c[2][xi], c[3][xi]
    return ((c0 * t + c1) * t + c2) * t + c3, (3.0 * c0 * t + 2.0 * c1) * t + c2


# Δ_T, Δ_E, Δ_B prefactors and source rows (``_tensor_sources``): T kernel on S_T, E and B
# kernels on g·P.
_LOS_FAC = (1.0, 15.0 / 8.0, 15.0 / 4.0)
_LOS_SRC = np.array([0, 1, 1])


@functools.lru_cache(maxsize=32)
def _make_los(plan_key: tuple):
    """Line-of-sight projection over the static blocks of ``plan["los_blocks"]``:
    Δ_m(ℓ, k) = Σ_n w_n s_m(k, τ_n) K_m,ℓ(k χ_n), with the sources s cubic-spline
    interpolated from the source k grid block by block (CAMB InterpolateSources; keeps the
    dense (k, τ) arrays one block at a time). One ``custom_jvp``: the tangent reuses the
    kernel-table gathers, dΔ = Σ_n d(w s) K + Σ_n w s K'(x) k dχ_n (the interpolation is
    linear). Inputs: src (n_k_src, 2, n_τ), its k-spline d2 (same shape), w, χ (n_τ,).
    Returns (n_ℓ, 3, n_k)."""
    plan = tensor_plan(*plan_key)
    bess = plan["bessel"]
    # Host (NumPy) constants only: this closure is cached across traces.
    coeffs_np = np.asarray(bess["coeffs"])
    x_end = float(bess["knots"][-1])
    k_int = np.asarray(plan["k_int"], dtype=np.float64)
    blocks = plan["los_blocks"]
    on_int = plan["acc"].src_on_int
    ksp = plan["kspline"]
    n_l, n_k = len(plan["ells"]), len(k_int)

    def interp(src, d2, nk, i0, i1):
        """(nk, 2, i1 − i0) sources on k_int[:nk] from the source k grid."""
        if on_int:
            return src[:nk, :, i0:i1]
        y = src[:, :, i0:i1].reshape(src.shape[0], -1)
        dd = d2[:, :, i0:i1].reshape(src.shape[0], -1)
        wk = {key: v[:nk] for key, v in ksp.items()}
        return _spline_eval(y, dd, wk).reshape(nk, 2, i1 - i0)

    def run(src, d2, w, chi, tangents=None):
        coeffs = jnp.asarray(coeffs_np)
        out = jnp.zeros((n_l, 3, n_k), dtype=jnp.float64)
        dout = None if tangents is None else jnp.zeros_like(out)
        for i0, i1, nk, nl in blocks:
            x = k_int[:nk, None] * chi[None, i0:i1]
            xi, t = _bessel_index(x, bess)
            inside = (x <= x_end)[:, None, :]
            s_raw = interp(src, d2, nk, i0, i1)
            # (nk, 3, nb): the source of each kernel (T: S_T; E, B: g·P) times its prefactor.
            fac = jnp.asarray(_LOS_FAC)[None, :, None]
            sw = jnp.where(inside, fac * (s_raw * w[i0:i1])[:, _LOS_SRC, :], 0.0)
            if tangents is not None:
                dsrc, dd2, dw, dchi = tangents
                ds = interp(dsrc, dd2, nk, i0, i1) * w[i0:i1] + s_raw * dw[i0:i1]
                dsw = jnp.where(inside, fac * ds[:, _LOS_SRC, :], 0.0)
                ax = sw * (k_int[:nk, None] * dchi[None, i0:i1])[:, None, :]

            def one_ell(_, j):
                kv, kd = zip(*(_kernel_eval(coeffs, m, j, xi, t) for m in range(3)))
                kv = jnp.stack(kv, axis=1)  # (nk, 3, nb)
                # Reductions over τ as contractions (multi-threaded on XLA:CPU, unlike sums).
                row = jnp.einsum("kmn,kmn->mk", sw, kv)
                if tangents is None:
                    return None, row
                drow = (jnp.einsum("kmn,kmn->mk", dsw, kv)
                        + jnp.einsum("kmn,kmn->mk", ax, jnp.stack(kd, axis=1)))
                return None, (row, drow)

            body = one_ell if tangents is not None else jax.checkpoint(one_ell)
            _, res = jax.lax.scan(body, None, jnp.arange(nl))
            if tangents is None:
                out = out.at[:nl, :, :nk].add(res)
            else:
                out = out.at[:nl, :, :nk].add(res[0])
                dout = dout.at[:nl, :, :nk].add(res[1])
        return out if tangents is None else (out, dout)

    @jax.custom_jvp
    def los(src, d2, w, chi):
        return run(src, d2, w, chi)

    @los.defjvp
    def los_jvp(primals, tangents):
        return run(*primals, tangents=tangents)

    return los


def tensor_transfer(params: dict, thermo: dict, plan: dict):
    """Δ_ℓ^{T,E,B}(k) on (plan ells, plan k_int): array (3, n_ℓ, n_k_int).

    Normalized as CAMB's ``Delta_p_l_k`` for tensors (T before the ctnorm factor).
    """
    tau_src, src, tau0 = tensor_sources(params, thermo, plan)
    w = _trapz_weights(tau_src)
    src = jnp.transpose(src, (1, 0, 2))  # (n_k_src, 2, n_τ)
    if plan["acc"].src_on_int:
        d2 = jnp.zeros_like(src)
    else:
        n = src.shape[0]
        d2 = _spline_d2(src.reshape(n, -1), np.asarray(plan["k_src"])).reshape(src.shape)
    delta = _make_los((plan["lmax"], plan["acc"]))(src, d2, w, tau0 - tau_src)
    return jnp.transpose(delta, (1, 0, 2))  # (3, n_ℓ, n_k)


def tensor_cls_from_transfer(delta, plan, r, As, ns, *, nt=None, ntrun=None,
                             pivot_tensor=0.05, pivot_scalar=0.05):
    """CAMB ``CalcTensCls`` on the sampled ℓ, then cubic interpolation to every ℓ ≤ lmax.

    Returns dict(ell, tt, ee, bb, te): ℓ(ℓ+1)C_ℓ/2π (dimensionless, ΔT/T units), zero
    at ℓ < 2.
    """
    k_int = jnp.asarray(plan["k_int"])
    # C_ℓ / r (positive) is interpolated in ℓ and multiplied by r at the end, so the result is
    # exactly ∝ r at fixed n_t and its r-derivative is right down to r = 0.
    pk = _tensor_power_per_r(k_int, r, As, ns, nt, ntrun, pivot_tensor,
                             pivot_scalar) * jnp.asarray(plan["dlnk"])
    dT, dE, dB = delta[0], delta[1], delta[2]
    ells = np.asarray(plan["ells"], dtype=np.float64)
    ctnorm = (ells ** 2 - 1.0) * (ells + 2.0) * ells
    dble = ells * (ells + 1.0) / (2.0 * np.pi) * np.pi / 4.0
    tt = (dT ** 2) @ pk * dble * ctnorm
    ee = (dE ** 2) @ pk * dble
    bb = (dB ** 2) @ pk * dble
    te = (dT * dE) @ pk * dble * np.sqrt(ctnorm)
    lmax = plan["lmax"]
    ell_out = np.arange(lmax + 1, dtype=np.float64)
    w = _spline_weights(ells, ell_out[2:])
    tiny = 1e-300
    # Positive spectra: spline log D_ℓ; TE: spline the correlation coefficient.
    logs = jnp.log(jnp.maximum(jnp.stack([tt, ee, bb], axis=1), tiny))  # (n_ℓ, 3)
    rho = te / jnp.sqrt(jnp.maximum(tt * ee, tiny))
    cols = jnp.concatenate([logs, rho[:, None]], axis=1)
    d2 = _spline_d2(cols, ells)
    vals = _spline_eval(cols, d2, w)
    tt_o, ee_o, bb_o = (jnp.exp(vals[:, i]) for i in range(3))
    te_o = vals[:, 3] * jnp.sqrt(tt_o * ee_o)
    pad = jnp.zeros(2, dtype=jnp.float64)
    out = {"ell": jnp.asarray(ell_out)}
    for name, v in (("tt", tt_o), ("ee", ee_o), ("bb", bb_o), ("te", te_o)):
        out[name] = r * jnp.concatenate([pad, v])
    return out


def tensor_cls_from_thermo(params: dict, thermo: dict, r, *, nt=None, ntrun=None,
                           pivot_tensor=0.05, lmax: int = 600,
                           acc: TensorAccuracy = DEFAULT_ACCURACY) -> dict:
    """``tensor_cls`` from the scalar pipeline's ``cmb_params`` dict and thermo tables
    (``cls_jax._recfast_thermo_device`` / ``recfast_thermo_batched``), sharing Recfast."""
    plan = tensor_plan(int(lmax), acc)
    delta = tensor_transfer(params, thermo, plan)
    return tensor_cls_from_transfer(delta, plan, r, params["As"], params["n_s"], nt=nt,
                                    ntrun=ntrun, pivot_tensor=pivot_tensor,
                                    pivot_scalar=params.get("k_pivot", 0.05))


def tensor_cls(ombh2, omch2, h, ns, As, tau_reio, w0, wa, r, *, nt=None, ntrun=None,
               pivot_tensor=0.05, lmax: int = 600, acc: TensorAccuracy = DEFAULT_ACCURACY,
               thermo: dict | None = None, ext: dict | None = None, nu_model=None) -> dict:
    """Unlensed tensor-mode CMB spectra, CAMB ``get_cmb_power_spectra(CMB_unit=None,
    raw_cl=False)["tensor"]`` convention: dict(ell, tt, ee, bb, te) of ℓ(ℓ+1)C_ℓ/2π.

    Cosmology as ``cls_jax.cmb_params`` (neutrinos from ``ext``/``nu_model``, default one massive
    ν of 0.06 eV, N_eff = 3.044; BBN Y_He, tanh reionization from ``tau_reio``; A_s at
    k = 0.05/Mpc). ``r`` at ``pivot_tensor``;
    ``nt=None``: inflation consistency (CAMB Python default). ``lmax`` and ``acc`` are
    static; everything else may be traced (jit / vmap / grad). ``thermo``: precomputed
    ``cls_jax._recfast_thermo_device`` tables for this cosmology (else computed here).
    """
    from gpucosmo.jax.eb.cls_jax import _recfast_thermo_device, _require_flat, cmb_params, ext_params

    e = ext_params(ext)
    _require_flat(e["omk"], "the tensor-mode solver")
    params, _ = cmb_params(ombh2, omch2, h, ns, As, tau_reio, w0, wa, ext=e, nu_model=nu_model)
    if thermo is None:
        thermo = _recfast_thermo_device(ombh2, omch2, h, ns, As, tau_reio, w0, wa, e["mnu"], e["nnu"],
                                        e["meffsterile"], e["omk"], e.get("omnuh2"), nu_model=nu_model,
                                        use_omnuh2="omnuh2" in e)
    return tensor_cls_from_thermo(params, thermo, r, nt=nt, ntrun=ntrun,
                                  pivot_tensor=pivot_tensor, lmax=lmax, acc=acc)


__all__ = [
    "DEFAULT_ACCURACY",
    "TensorAccuracy",
    "tensor_cls",
    "tensor_cls_from_thermo",
    "tensor_cls_from_transfer",
    "tensor_plan",
    "tensor_power",
    "tensor_sources",
    "tensor_spectral_index",
    "tensor_transfer",
]
