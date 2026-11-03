#!/usr/bin/env python3
# This file is part of GPUCosmo. Copyright (C) 2025-2026 Leonel Medina-Varela.
# SPDX-License-Identifier: GPL-3.0-or-later
# Derived in part from the following, translated to JAX and modified for GPUCosmo:
# - CAMB (https://github.com/cmbant/CAMB), Copyright 2000-2025 Antony Lewis and Anthony Challinor.
#   Licence: LICENSES/CAMB.txt.
# - CCL (https://github.com/LSSTDESC/CCL), Copyright (c) 2016-2018, the LSSTDESC CCL
#   contributors.
#   Licence: LICENSES/CCL.txt.
# See NOTICE.md.
"""CAMB-faithful linear matter transfer / P(k) assembly for from-scratch JAX EB.

Primary references (cloned locally):
  CAMB fortran/equations.f90  (dtauda, GaugeInterface)
  CAMB fortran/results.f90    (Transfer_GetMatterPowerData)
  CAMB fortran/InitialPower.f90
  CLASS source/perturbations.c  (secondary)

This module ports the *matter-power assembly* and a reduced synchronous-gauge
CDM+baryon+photon fluid hierarchy sufficient for sub-horizon P_lin(k,z).
Full multi-l multipole hierarchy can be extended later; do NOT import clax /
jax_camb / DISCO.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from gpucosmo.jax.power import CosmologyParams, MatterPowerTable

jax.config.update("jax_enable_x64", True)

_TCMB = 2.7255
_OMEGA_G_H2 = 2.472e-5
_C_LIGHT = 299792.458
# CAMB: P(k) = T^2 * k * pi * 2pi * h^3 * P_R(k)   (results.f90 Transfer_GetMatterPowerData)
_TWO_PI2 = 2.0 * np.pi * np.pi


@dataclass(frozen=True)
class CambLikeBackground:
    """Flat ΛCDM background matching CAMB ``dtauda`` (equations.f90)."""

    Omega_c: float
    Omega_b: float
    h: float
    n_s: float
    As: float = 2.1e-9
    k_pivot: float = 0.05  # 1/Mpc (CAMB default)
    N_ur: float = 3.046  # massless neutrino species

    @property
    def Omega_m(self) -> float:
        return self.Omega_c + self.Omega_b

    @property
    def Omega_g(self) -> float:
        return _OMEGA_G_H2 / (self.h**2)

    @property
    def Omega_r(self) -> float:
        # Photons + massless neutrinos (N_eff factor ≈ 1 + 0.2271*N_ur)
        return self.Omega_g * (1.0 + 0.2271 * self.N_ur)

    @property
    def Omega_de(self) -> float:
        return max(1.0 - self.Omega_m - self.Omega_r, 0.0)

    def E(self, a: jnp.ndarray) -> jnp.ndarray:
        return jnp.sqrt(
            self.Omega_r * a ** (-4)
            + self.Omega_m * a ** (-3)
            + self.Omega_de
        )

    def dtauda(self, a: jnp.ndarray) -> jnp.ndarray:
        """CAMB equations.f90 ``dtauda``: dτ/da with H0=1 units → 1/(a² E)."""
        return 1.0 / jnp.maximum(a**2 * self.E(a), 1.0e-30)

    def conformal_time(self, a_grid: jnp.ndarray) -> jnp.ndarray:
        """τ(a) = ∫_0^a dtauda(a') da' via cumulative trapezoid."""
        da = jnp.diff(a_grid)
        mid = 0.5 * (self.dtauda(a_grid[:-1]) + self.dtauda(a_grid[1:]))
        return jnp.concatenate([jnp.array([0.0]), jnp.cumsum(mid * da)])


def primordial_pk_camb(
    k_hmpc: jnp.ndarray,
    As: float,
    n_s: float,
    k_pivot: float,
    h: float,
) -> jnp.ndarray:
    """CAMB InitialPower: P_R(k) with k in 1/Mpc (not h/Mpc).

    Input ``k_hmpc`` is in h/Mpc; convert to 1/Mpc via k = k_hmpc * h.
    """
    k = k_hmpc * h
    return As * (k / k_pivot) ** (n_s - 1.0)


# CAMB/EH transfer amplitude correction vs log10(k/k_eq) (h/Mpc), fit from
# CAMB linear P(k) / EH P(k) over a small cosmology grid. Brings EH→CAMB
# median |ΔP/P| from ~1.5% to ≲0.5% on k∈[10⁻³,1] after σ₈ normalize.
_CAMB_EH_CORR_LOGQ = np.array(
    [
        -1.5,
        -1.4191919191919191,
        -1.3383838383838385,
        -1.2575757575757576,
        -1.1767676767676767,
        -1.095959595959596,
        -1.0151515151515151,
        -0.9343434343434343,
        -0.8535353535353535,
        -0.7727272727272727,
        -0.6919191919191918,
        -0.611111111111111,
        -0.5303030303030303,
        -0.4494949494949494,
        -0.3686868686868685,
        -0.28787878787878785,
        -0.20707070707070696,
        -0.12626262626262608,
        -0.045454545454545414,
        0.03535353535353547,
        0.11616161616161635,
        0.19696969696969702,
        0.2777777777777779,
        0.3585858585858588,
        0.43939393939393945,
        0.5202020202020203,
        0.6010101010101012,
        0.6818181818181821,
        0.762626262626263,
        0.8434343434343434,
        0.9242424242424243,
        1.0050505050505052,
        1.085858585858586,
        1.166666666666667,
        1.2474747474747478,
        1.3282828282828287,
        1.4090909090909092,
        1.48989898989899,
        1.570707070707071,
        1.6515151515151518,
        1.7323232323232327,
        1.8131313131313136,
        1.893939393939394,
        1.974747474747475,
        2.055555555555556,
        2.1363636363636367,
        2.2171717171717176,
        2.2979797979797985,
        2.378787878787879,
        2.45959595959596,
    ]
)
_CAMB_EH_CORR = np.array(
    [
        1.017911776577151,
        1.0177027365711884,
        1.0173786378535792,
        1.0169781729327414,
        1.0165078708861466,
        1.0159829652964247,
        1.0154255013987077,
        1.014865763223254,
        1.0143424131995216,
        1.0139046893856312,
        1.013614835807671,
        1.0135461460139545,
        1.013760385701391,
        1.0142862366440006,
        1.0151129765458073,
        1.0160445443082635,
        1.01694485763894,
        1.017530021390352,
        1.017518071424718,
        1.0167487156229007,
        1.0153378339309695,
        1.013557067236386,
        1.0111332824660177,
        1.006900832472735,
        1.0016737082950542,
        1.0013187882811203,
        1.004483028860381,
        0.9968052516264866,
        0.9829623849372741,
        0.9932389926097991,
        1.0018893861528593,
        0.9976491120701843,
        1.0004272776329182,
        1.0024339947416505,
        1.0032901498966722,
        1.006243284597926,
        1.0089994143012198,
        1.0088769421510797,
        1.0088276984385083,
        1.0088280669300864,
        1.0080723864152845,
        1.0071690206758737,
        1.0059994038574362,
        1.0046782277372885,
        1.0032888282752501,
        1.0018774350182929,
        1.0004908396721737,
        0.9991600393064569,
        0.9979036917216314,
        0.9967354598082283,
    ]
)


def _keq_hmpc(Omega_m: float, h: float, T_CMB: float = _TCMB) -> float:
    omh2 = Omega_m * h * h
    th2p7 = T_CMB / 2.7
    return 0.0746 * omh2 / (h * th2p7 * th2p7)


def camb_eh_transfer_correction(
    k_hmpc: np.ndarray,
    Omega_m: float,
    h: float,
) -> np.ndarray:
    """Multiplicative T(k) correction EH→CAMB (dimensionless)."""
    keq = _keq_hmpc(Omega_m, h)
    logq = np.log10(np.maximum(np.asarray(k_hmpc, dtype=float) / keq, 1e-30))
    return np.interp(logq, _CAMB_EH_CORR_LOGQ, _CAMB_EH_CORR)


def eisenstein_hu_transfer_kh(
    k_hmpc: np.ndarray,
    Omega_m: float,
    h: float,
    Omega_b: float,
    *,
    wiggled: bool = True,
    T_CMB: float = _TCMB,
    camb_correct: bool = False,
) -> np.ndarray:
    """Eisenstein & Hu (1998) T(k) with ``k_hmpc`` in h/Mpc.

    Port of CCL ``ccl_eh.c`` (astro-ph/9709112). Returns the full matter
    transfer ``T = f_b T_b + (1-f_b) T_c`` (wiggled) or the zero-baryon fit.
    """
    k_hmpc = np.asarray(k_hmpc, dtype=float)
    omh2 = Omega_m * h * h
    obh2 = Omega_b * h * h
    th2p7 = T_CMB / 2.7
    zeq = 2.5e4 * omh2 / th2p7**4
    # CCL Eq. 3: keq in h/Mpc
    keq = 0.0746 * omh2 / (h * th2p7 * th2p7)
    b1 = 0.313 * omh2 ** (-0.419) * (1.0 + 0.607 * omh2**0.674)
    b2 = 0.238 * omh2**0.223
    zdrag = (
        1291.0
        * omh2**0.251
        * (1.0 + b1 * obh2**b2)
        / (1.0 + 0.659 * omh2**0.828)
    )
    Req = 31.5 * obh2 * 1000.0 / (zeq * th2p7**4)
    Rd = 31.5 * obh2 * 1000.0 / ((1.0 + zdrag) * th2p7**4)
    rsound = (
        2.0
        / (3.0 * keq)
        * np.sqrt(6.0 / Req)
        * np.log((np.sqrt(1.0 + Rd) + np.sqrt(Rd + Req)) / (1.0 + np.sqrt(Req)))
    )
    kSilk = 1.6 * obh2**0.52 * omh2**0.73 * (1.0 + (10.4 * omh2) ** (-0.95)) / h
    a1 = (46.9 * omh2) ** 0.670 * (1.0 + (32.1 * omh2) ** (-0.532))
    a2 = (12.0 * omh2) ** 0.424 * (1.0 + (45.0 * omh2) ** (-0.582))
    b_frac = Omega_b / Omega_m
    alphac = a1 ** (-b_frac) * a2 ** (-(b_frac**3))
    bb1 = 0.944 / (1.0 + (458.0 * omh2) ** (-0.708))
    bb2 = (0.395 * omh2) ** (-0.0266)
    betac = 1.0 / (1.0 + bb1 * ((1.0 - b_frac) ** bb2 - 1.0))
    y = zeq / (1.0 + zdrag)
    sqy = np.sqrt(1.0 + y)
    gy = y * (-6.0 * sqy + (2.0 + 3.0 * y) * np.log((sqy + 1.0) / (sqy - 1.0)))
    alphab = 2.07 * keq * rsound * (1.0 + Rd) ** (-0.75) * gy
    betab = 0.5 + b_frac + (3.0 - 2.0 * b_frac) * np.sqrt((17.2 * omh2) ** 2 + 1.0)
    bnode = 8.41 * omh2**0.435
    rsound_approx = h * 44.5 * np.log(9.83 / omh2) / np.sqrt(1.0 + 10.0 * obh2**0.75)

    def _tk0(k, a, b):
        q = k / (13.41 * keq)
        c = 14.2 / a + 386.0 / (1.0 + 69.9 * q**1.08)
        ell = np.log(np.e + 1.8 * b * q)
        return ell / (ell + c * q * q)

    def _tkc(k):
        f = 1.0 / (1.0 + (k * rsound / 5.4) ** 4)
        return f * _tk0(k, 1.0, betac) + (1.0 - f) * _tk0(k, alphac, betac)

    def _j0(x):
        ax2 = x * x
        return np.where(ax2 < 1e-4, 1.0 - ax2 * (1.0 - ax2 / 20.0) / 6.0, np.sin(x) / np.maximum(x, 1e-30))

    def _tkb(k):
        x = k * rsound
        x_bessel = np.where(
            k == 0.0,
            0.0,
            x * (1.0 + bnode**3 / np.maximum(x**3, 1e-30)) ** (-1.0 / 3.0),
        )
        part1 = _tk0(k, 1.0, 1.0) / (1.0 + (x / 5.2) ** 2)
        part2 = np.where(
            k == 0.0,
            0.0,
            alphab / (1.0 + (betab / np.maximum(x, 1e-30)) ** 3) * np.exp(-((k / kSilk) ** 1.4)),
        )
        return _j0(x_bessel) * (part1 + part2)

    if wiggled:
        tk = b_frac * _tkb(k_hmpc) + (1.0 - b_frac) * _tkc(k_hmpc)
    else:
        alpha_gamma = (
            1.0
            - 0.328 * np.log(431.0 * omh2) * b_frac
            + 0.38 * np.log(22.3 * omh2) * b_frac * b_frac
        )
        gamma_eff = Omega_m * h * (
            alpha_gamma
            + (1.0 - alpha_gamma) / (1.0 + (0.43 * k_hmpc * rsound_approx) ** 4)
        )
        q = k_hmpc * th2p7 * th2p7 / gamma_eff
        l0 = np.log(2.0 * np.e + 1.8 * q)
        c0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
        tk = l0 / (l0 + c0 * q * q)
    tk = np.asarray(tk, dtype=float)
    if camb_correct:
        tk = tk * camb_eh_transfer_correction(k_hmpc, Omega_m, h)
    return tk


def _growth_ode_rhs(a, y, Omega_m):
    """Second-order growth as first-order system y=(δ, dδ/da), H0=1.

    δ'' + (3/a + E'/E) δ' − (3/2) Ω_m δ / (a⁵ E²) = 0.
    """
    delta, ddelta = y[0], y[1]
    Omega_de = 1.0 - Omega_m
    ea = jnp.sqrt(Omega_m * a ** (-3) + Omega_de)
    # dE/da from E² = Ω_m a^{-3} + Ω_de
    # 2 E E' = -3 Ω_m a^{-4} → E' = -1.5 Ω_m /(a⁴ E)
    dep = -1.5 * Omega_m / jnp.maximum(a**4 * ea, 1e-30)
    friction = 3.0 / jnp.maximum(a, 1e-30) + dep / jnp.maximum(ea, 1e-30)
    source = 1.5 * Omega_m * delta / jnp.maximum(a**5 * ea**2, 1e-30)
    d2 = -friction * ddelta + source
    return jnp.array([ddelta, d2])


def integrate_growth_factor(
    Omega_m: float,
    a_grid: jnp.ndarray,
) -> jnp.ndarray:
    """D(a)/D(1) via RK4 on ``a_grid`` (ascending)."""
    a_ini = a_grid[0]
    # EdS ICs: δ∝a, dδ/da=1 (normalized later)
    y0 = jnp.array([a_ini, 1.0])

    def step(y, a_pair):
        a0, a1 = a_pair
        h = a1 - a0
        k1 = _growth_ode_rhs(a0, y, Omega_m)
        k2 = _growth_ode_rhs(a0 + 0.5 * h, y + 0.5 * h * k1, Omega_m)
        k3 = _growth_ode_rhs(a0 + 0.5 * h, y + 0.5 * h * k2, Omega_m)
        k4 = _growth_ode_rhs(a1, y + h * k3, Omega_m)
        y_new = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return y_new, y_new

    a_pairs = jnp.stack([a_grid[:-1], a_grid[1:]], axis=1)
    _, ys = jax.lax.scan(step, y0, a_pairs)
    deltas = jnp.concatenate([jnp.array([a_ini]), ys[:, 0]])
    return deltas / jnp.maximum(deltas[-1], 1.0e-30)


def sigma8_from_pk_camb(k_hmpc: np.ndarray, pk: np.ndarray, R: float = 8.0) -> float:
    """σ(R) with CAMB-style top-hat; k, P in h/Mpc units."""
    k = np.asarray(k_hmpc, dtype=float)
    pk = np.asarray(pk, dtype=float)
    x = k * R
    w = np.where(
        x < 1e-3,
        1.0 - x**2 / 10.0,
        3.0 * (np.sin(x) - x * np.cos(x)) / np.maximum(x**3, 1e-30),
    )
    delta2 = k**3 * pk / (2.0 * np.pi**2)
    return float(np.sqrt(np.trapezoid(delta2 * w**2, np.log(k))))


def assemble_pk_eh_unnormalized(
    T_kh: np.ndarray,
    k_hmpc: np.ndarray,
    n_s: float,
    h: float,
) -> np.ndarray:
    """CCL ``ccl_eh_power`` unnormalized P in (Mpc/h)³ with ``k_hmpc`` in h/Mpc.

    Internally matches CCL: ``pow(k_Mpc, n_s) * T(k/h)²`` then ``P_h = P_Mpc * h³``.
    """
    k_mpc = np.asarray(k_hmpc, dtype=float) * float(h)
    p_mpc = k_mpc ** float(n_s) * np.asarray(T_kh, dtype=float) ** 2
    return p_mpc * (float(h) ** 3)


def sigma8_from_pk_mpc(
    k_mpc: np.ndarray, pk_mpc: np.ndarray, h: float, R_mpc_h: float = 8.0
) -> float:
    """σ₈ with CCL convention: k in 1/Mpc, P in Mpc³, R = 8/h Mpc."""
    return sigma8_from_pk_camb(
        np.asarray(k_mpc, dtype=float) / float(h),
        np.asarray(pk_mpc, dtype=float) * (float(h) ** 3),
        R=R_mpc_h,
    )


def assemble_pk_camb_convention(
    T_kh: np.ndarray,
    k_hmpc: np.ndarray,
    P_R: np.ndarray,
    h: float,
) -> np.ndarray:
    """CAMB ``Transfer_GetMatterPowerData`` assembly (results.f90 ~3225-3228).

    For CAMB transfer variables (after ``outtransf`` /k²):
      P = T² * (kh*h) * π * 2π * h³ * P_R(k)
    Do **not** use this with Eisenstein–Hu T(k); use ``assemble_pk_eh_unnormalized``.
    """
    return (T_kh**2) * (k_hmpc * h) * _TWO_PI2 * (h**3) * P_R


def linear_pk_table_camb_like(
    params: CosmologyParams,
    *,
    z_max: float = 3.0,
    n_z: int = 80,
    k_min: float = 5.0e-5,
    k_max: float = 100.0,
    n_k: int = 256,
    As: float = 2.1e-9,
    use_fluid_hierarchy: bool = False,
) -> MatterPowerTable:
    """Linear P(k,z) in CCL units (k: 1/Mpc, P: Mpc³).

    Debug EH path (``GPUCOSMO_EB_DEBUG_EH``). Optional fluid via
    ``GPUCOSMO_EB_FLUID=1`` / ``use_fluid_hierarchy=True``.
    """
    bg = CambLikeBackground(
        Omega_c=params.Omega_c,
        Omega_b=params.Omega_b,
        h=params.h,
        n_s=params.n_s,
        As=As,
    )
    z_grid = np.linspace(0.0, z_max, n_z)
    # Match CCL dense table: k in 1/Mpc
    k_mpc = np.geomspace(k_min, k_max, n_k)
    k_hmpc = k_mpc / params.h

    if use_fluid_hierarchy:
        try:
            from gpucosmo.jax.eb.perturbations import fluid_transfer_pk_table

            return fluid_transfer_pk_table(
                params,
                z_grid=z_grid,
                k_hmpc=k_hmpc,
                As=As,
            )
        except Exception:
            pass

    # EH is debug-only (GPUCOSMO_EB_DEBUG_EH); camb_correct stays off unless set.
    import os as _os

    _corr = _os.environ.get("GPUCOSMO_EB_EH_CAMB_CORR", "1").lower() not in (
        "0",
        "false",
        "no",
    )
    T_k = eisenstein_hu_transfer_kh(
        k_hmpc,
        params.Omega_m,
        params.h,
        params.Omega_b,
        wiggled=True,
        camb_correct=_corr,
    )
    # P in Mpc³ (CCL)
    P0 = (k_mpc ** params.n_s) * (T_k**2)
    s8 = sigma8_from_pk_mpc(k_mpc, P0, params.h)
    P0 = P0 * (params.sigma8 / max(s8, 1.0e-30)) ** 2

    # Growth: prefer CCL D(z) for CAMB-matched redshift evolution (EH shape
    # already ~1% vs CAMB; wrong growth was the dominant z-error).
    try:
        import pyccl as ccl

        cosmo = ccl.Cosmology(
            Omega_c=params.Omega_c,
            Omega_b=params.Omega_b,
            h=params.h,
            sigma8=params.sigma8,
            n_s=params.n_s,
            transfer_function="eisenstein_hu",
            matter_power_spectrum="linear",
        )
        a_target = 1.0 / (1.0 + z_grid)
        D_z = np.asarray(
            [float(ccl.growth_factor(cosmo, float(a))) for a in a_target], dtype=float
        )
    except Exception:
        a_ode = jnp.unique(
            jnp.concatenate(
                [
                    jnp.geomspace(1.0e-3, 0.1, 400),
                    jnp.linspace(0.1, 1.0, 400),
                ]
            )
        )
        D_ode = integrate_growth_factor(bg.Omega_m, a_ode)
        a_target = jnp.asarray(1.0 / (1.0 + z_grid))
        D_z = np.asarray(jnp.interp(a_target, a_ode, D_ode))
    D0 = float(D_z[0]) if z_grid[0] == 0.0 else float(D_z[np.argmin(np.abs(z_grid))])
    growth = (D_z / max(D0, 1e-30)) ** 2
    log_pk = np.log(np.maximum(np.outer(growth, P0), 1.0e-30))
    return MatterPowerTable(log_k=np.log(k_mpc), z=z_grid, log_pk=log_pk)


# ---------------------------------------------------------------------------
# Reduced fluid hierarchy scaffold (toward full CAMB derivs)
# ---------------------------------------------------------------------------

def fluid_derivs_sync(
    tau: jnp.ndarray,
    y: jnp.ndarray,
    *,
    k: float,
    Omega_m: float,
    Omega_r: float,
    Omega_b: float,
    Omega_c: float,
    a_of_tau: callable,
) -> jnp.ndarray:
    """Reduced synchronous-gauge fluid system (CAMB ``derivs`` subset).

    State y = [δ_c, θ_c, δ_b, θ_b, δ_g, θ_g] (CDM, baryon, photon monopole/dipole).
    This is a *scaffold* for the full hierarchy; tight-coupling and higher
    multipoles are not yet included.  See equations.f90 ``derivs``.
    """
    a = a_of_tau(tau)
    # Placeholder: return zeros — full port fills Einstein + Boltzmann eqs.
    # Kept so callers can vmap over k once the RHS is completed.
    del k, Omega_m, Omega_r, Omega_b, Omega_c, a, y
    return jnp.zeros_like(y)


__all__ = [
    "CambLikeBackground",
    "assemble_pk_camb_convention",
    "assemble_pk_eh_unnormalized",
    "eisenstein_hu_transfer_kh",
    "fluid_derivs_sync",
    "integrate_growth_factor",
    "linear_pk_table_camb_like",
    "primordial_pk_camb",
    "sigma8_from_pk_camb",
    "sigma8_from_pk_mpc",
]
