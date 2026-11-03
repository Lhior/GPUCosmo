#!/usr/bin/env python3
# This file is part of GPUCosmo. Copyright (C) 2025-2026 Leonel Medina-Varela.
# SPDX-License-Identifier: GPL-3.0-or-later
# Derived in part from the following, translated to JAX and modified for GPUCosmo:
# - CAMB (https://github.com/cmbant/CAMB), Copyright 2000-2025 Antony Lewis and Anthony Challinor.
#   Licence: LICENSES/CAMB.txt.
# See NOTICE.md.
"""Massive-neutrino background and perturbations, ported from CAMB v2.

Direct port of ``fortran/massive_neutrinos.f90`` (rho/P fits, momentum grid) and
the massive-nu blocks of ``fortran/equations.f90`` (``Nu_Integrate_L012`` and the
multipole ODEs). Everything is JAX so it traces, jits and vmaps with the rest of
the hierarchy.

Densities are in units of the mean density of **one massless neutrino
eigenstate**, matching CAMB, so ``grhormass * rhonu`` gives the physical value.

The DESI DR2 baseline is one massive eigenstate at 0.06 eV plus 2.0308 massless
states, giving ``N_eff = 3.044``. At that mass the neutrinos are still
ultra-relativistic at recombination (``a*m/T_nu ~ 0.2``), so the CMB effect comes
mostly from the late-time background and the lensing potential -- measured at
0.2-0.5% in TT and 2.0% in C_L^phiphi, both well above the 0.1% target.
"""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# --- constants (CAMB constants.f90) ---------------------------------------
ZETA3 = 1.2020569031595942853997
ZETA5 = 1.0369277551433699263313
ZETA7 = 1.0083492773819228268397

# int q^3 F(q) dq = 7/120 pi^4
FERMI_DIRAC_CONST = 7.0 / 120.0 * np.pi**4
CONST2 = 5.0 / 7.0 / np.pi**2

# Converts omnuh2 into sum m_nu in eV for non-relativistic thermal neutrinos (CAMB
# constants.f90 ``neutrino_mass_fac``; one value for the whole package).
from gpucosmo.jax.eb.constants import NEUTRINO_MASS_FAC  # noqa: E402

# Mass splittings and the normal-hierarchy minimum (CAMB constants.f90, PDG 2015).
DELTA_MNU21 = 7.54e-5  # eV^2
DELTA_MNU31 = 2.46e-3  # eV^2
MNU_MIN_NORMAL = 0.06  # eV (CAMB's threshold for the two-eigenstate normal hierarchy)
MNU_MIN_INVERTED = float(np.sqrt(DELTA_MNU31) + np.sqrt(DELTA_MNU31 + DELTA_MNU21))

# Fit domains (massive_neutrinos.f90).
NU_RHOP_AM_MIN = 0.42
NU_RHOP_AM_MAX = 70.0
AM_MIN = 0.3
AM_MAX = 70.0

NU_RHOP_FIT_C = np.sqrt(NU_RHOP_AM_MIN * NU_RHOP_AM_MAX)
NU_RHOP_FIT_INV_ZMAX = (NU_RHOP_AM_MAX + NU_RHOP_FIT_C) / (NU_RHOP_AM_MAX - NU_RHOP_FIT_C)
NU_FIT_C = np.sqrt(AM_MIN * AM_MAX)
NU_FIT_INV_ZMAX = (AM_MAX + NU_FIT_C) / (AM_MAX - NU_FIT_C)

NU_FIT_RHO_SCALE = 3.0 * ZETA3 / (2.0 * FERMI_DIRAC_CONST)
NU_FIT_P_DENOM_SCALE = FERMI_DIRAC_CONST / ((900.0 / 120.0) * ZETA5)

# Polynomial coefficients in z, increasing order.
NU_FIT_RHO_C = np.array(
    [
        7.499710425926954249e-01,
        1.188066888581748443e-01,
        1.545077526625578401e-01,
        -8.378996019212282820e-02,
        2.123075360029347963e-02,
        -2.544964203886362908e-03,
    ]
)
NU_FIT_P_C = np.array(
    [
        9.283417043601330798e-01,
        3.156134695375645838e-01,
        -2.435843189264263742e-01,
        -2.509793530357270000e-02,
        4.067958090718486880e-02,
        2.378742143669706002e-03,
        -1.965863329520473827e-03,
    ]
)
NU_LOW_RHO_C = np.array([-1.872929199948838302e-02, 1.831728681837861694e-02])
NU_LOW_P_C = np.array([1.667044572156534468e-02, -2.277471498716537868e-02])
NU_FIT_D_C = np.array(
    [
        5.800497880734221123e-01,
        1.801974470841860576e-01,
        -1.602709049602861757e-01,
        3.167230341650149189e-02,
        1.056591590995091534e-02,
        -3.877662402660121861e-03,
    ]
)


def _horner(coef: np.ndarray, z: jnp.ndarray) -> jnp.ndarray:
    """Evaluate a polynomial given ascending-order coefficients."""
    out = jnp.full_like(z, coef[-1])
    for c in coef[-2::-1]:
        out = out * z + c
    return out


# --- momentum grid (TNuPerturbations_init) --------------------------------
# CAMB's optimized rules; the kernel already carries the dlnf/dlnq factor
# because the code evolves 4 F_l / dlfdlq.
_NU_Q_RULES: dict[int, tuple[list[float], list[float]]] = {
    3: ([0.913201, 3.37517, 7.79184], [0.0687359, 3.31435, 2.29911]),
    4: (
        [
            0.5802007037903776,
            2.2150938570691223,
            4.948032138986023,
            9.65253759848097,
        ],
        [
            0.0082119845039711,
            1.1143258498419168,
            3.6819104154615907,
            0.8777790167504481,
        ],
    ),
    5: (
        [
            0.4620995950854295,
            1.7331898360630928,
            3.7956972681313816,
            7.2113928588584990,
            13.2665914595911080,
        ],
        [
            0.0026946402277849193,
            0.46041071394952310,
            2.9207114780286405,
            2.1821643017186352,
            0.11621584305889110,
        ],
    ),
}


@functools.lru_cache(maxsize=8)
def nu_momentum_grid(nqmax: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """``(nu_q, nu_int_kernel)`` for ``nqmax`` momentum bins.

    Mirrors ``TNuPerturbations_init``: hand-optimized rules for nqmax in 3..5,
    otherwise a uniform rule. The kernel is normalized by the Fermi-Dirac
    integral so that summing it reproduces the total density.
    """
    if nqmax in _NU_Q_RULES:
        q, kern = _NU_Q_RULES[nqmax]
        nu_q = np.asarray(q, dtype=np.float64)
        kernel = np.asarray(kern, dtype=np.float64)
    else:
        dq = (12 + nqmax / 5) / float(nqmax)
        i = np.arange(1, nqmax + 1, dtype=np.float64)
        nu_q = (i - 0.5) * dq
        dlfdlq = -nu_q / (1.0 + np.exp(-nu_q))
        kernel = dq * nu_q**3 / (np.exp(nu_q) + 1.0) * (-0.25 * dlfdlq)
    return nu_q, kernel / FERMI_DIRAC_CONST


# CAMB 2.0.4 promotes the momentum rule by its heaviest eigenstate (results.f90,
# massive_nu_transfer_q_boost_mass / _q_5point_mass, active with its default AccuracyTarget = 1, for
# CMB and transfer runs alike): 4 bins above nu_masses = 600 (≈ 0.10 eV for an active state), 5
# above 4300 (≈ 0.72 eV). The 3-bin rule's error grows with the mass: P(k) 1.6e-4 at 0.10 eV and
# 1.9e-3 at 0.63 eV against 100 bins (CAMB docs/changelog/matter_power_transfer_accuracy_review.md).
NU_Q4_AM = 600.0
NU_Q5_AM = 4300.0


def camb_nu_nqmax(max_am: float) -> int:
    """CAMB 2.0.4's momentum-bin count for a heaviest eigenstate of ``max_am`` (CAMB
    ``nu_masses``: m/(k_B T) at the block's temperature, ``NeutrinoModel.resolve``'s ``am``)."""
    am = float(max_am)
    return 5 if am > NU_Q5_AM else 4 if am > NU_Q4_AM else 3


def nu_accuracy_to_nqmax(accuracy: float = 1.0) -> int:
    """CAMB's mapping from neutrino accuracy boost to momentum-bin count."""
    if accuracy > 3:
        return int(round(accuracy * 10))
    if accuracy > 2:
        return 5
    if accuracy > 1:
        return 4
    return 3


# --- background rho / P (ThermalNuBackground) -----------------------------
def thermal_nu_rho_p(am: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(rhonu, pnu)`` in units of one massless eigenstate, for ``am = a*m/T_nu``.

    Three branches (low series / smooth fit / large-am series) selected with
    ``jnp.where`` so the function stays traceable; CAMB quotes ~2.4e-5 relative
    accuracy in rho and 4.7e-5 in P against direct quadrature.
    """
    am = jnp.asarray(am, dtype=jnp.float64)
    am2 = am * am
    am4 = am2 * am2

    rho_low = 1.0 + CONST2 * am2 + am4 * (NU_LOW_RHO_C[0] + am2 * NU_LOW_RHO_C[1])
    p_low = (1.0 - CONST2 * am2) / 3.0 + am4 * (NU_LOW_P_C[0] + am2 * NU_LOW_P_C[1])

    # Large am: expanded in 1/(a*m). Guard am so the unused branch stays finite.
    am_hi = jnp.maximum(am, 1.0)
    rho_hi = (
        3.0
        / (2.0 * FERMI_DIRAC_CONST)
        * (ZETA3 * am_hi + ((15.0 * ZETA5) / 2.0 - 945.0 / 16.0 * ZETA7 / am_hi**2) / am_hi)
    )
    p_hi = (
        900.0
        / 120.0
        / FERMI_DIRAC_CONST
        * (ZETA5 - 63.0 / 4.0 * ZETA7 / am_hi**2)
        / am_hi
    )

    am_mid = jnp.clip(am, NU_RHOP_AM_MIN, NU_RHOP_AM_MAX)
    z = ((am_mid - NU_RHOP_FIT_C) / (am_mid + NU_RHOP_FIT_C)) * NU_RHOP_FIT_INV_ZMAX
    rho_mid = (1.0 + NU_FIT_RHO_SCALE * am_mid) * _horner(NU_FIT_RHO_C, z)
    p_mid = _horner(NU_FIT_P_C, z) / (1.0 + NU_FIT_P_DENOM_SCALE * am_mid)

    rho = jnp.where(am <= NU_RHOP_AM_MIN, rho_low, jnp.where(am >= NU_RHOP_AM_MAX, rho_hi, rho_mid))
    pnu = jnp.where(am <= NU_RHOP_AM_MIN, p_low, jnp.where(am >= NU_RHOP_AM_MAX, p_hi, p_mid))
    return rho, pnu


def thermal_nu_rho(am: jnp.ndarray) -> jnp.ndarray:
    return thermal_nu_rho_p(am)[0]


def thermal_nu_drho(am: jnp.ndarray, adotoa: jnp.ndarray) -> jnp.ndarray:
    """d(rho_nu)/dtau, via CAMB's fit to ``D = rho - 3P``."""
    am = jnp.asarray(am, dtype=jnp.float64)
    am2 = am * am

    am_low = jnp.maximum(am, 1e-12)
    d_low = am2 * (
        2.0 * CONST2
        + am2
        * (
            0.4399706676e-1 * jnp.log(am_low)
            - 0.2970400378e-2
            - 0.29331377855e-1 * am
        )
    )

    am_hi = jnp.maximum(am, 1.0)
    d_hi = (
        3.0
        / (2.0 * FERMI_DIRAC_CONST)
        * (ZETA3 * am_hi + (-(15.0 * ZETA5) / 2.0 + 2835.0 / 16.0 * ZETA7 / am_hi**2) / am_hi)
    )

    am_mid = jnp.clip(am, AM_MIN, AM_MAX)
    z = ((am_mid - NU_FIT_C) / (am_mid + NU_FIT_C)) * NU_FIT_INV_ZMAX
    d_mid = am_mid**2 / (1.0 + 2.0 * am_mid) * _horner(NU_FIT_D_C, z)

    d = jnp.where(am <= AM_MIN, d_low, jnp.where(am >= AM_MAX, d_hi, d_mid))
    return d * adotoa


def find_nu_mass_for_rho_jax(rho) -> jnp.ndarray:
    """``find_nu_mass_for_rho`` for traced ``rho`` (jit/vmap/grad): the exact root of
    ``thermal_nu_rho(am) = rho`` (bisection of [0, max(64, 2·guess)] then Newton steps, so
    derivatives flow); 0 for rho ≤ 1. CAMB 2.0.4 takes the same root (its heavy branch is an
    asymptotic inversion plus one Newton step; its light branch a Brent solve)."""
    rho = jnp.asarray(rho, dtype=jnp.float64)
    guess = FERMI_DIRAC_CONST / (1.5 * ZETA3) * rho

    def bisect(_, lohi):
        lo, hi = lohi
        mid = 0.5 * (lo + hi)
        below = thermal_nu_rho(mid) < rho
        return jnp.where(below, mid, lo), jnp.where(below, hi, mid)

    hi0 = jnp.maximum(64.0, 2.0 * guess)
    lo, hi = jax.lax.fori_loop(0, 90, bisect, (jnp.zeros_like(rho), hi0))
    am = 0.5 * (lo + hi)
    for _ in range(3):  # Newton polish: exact root to roundoff, and d(am)/d(rho) for gradients
        r, dr = jax.jvp(thermal_nu_rho, (am,), (jnp.ones_like(am),))
        am = am - (r - rho) / jnp.where(dr > 0, dr, 1.0)
    return jnp.where(rho <= 1.0, 0.0, jnp.maximum(am, 0.0))


# --- neutrino models (CAMB SetNeutrinoHierarchyBase / SetNeutrinoMasses) ------------------
def nu_temperature_ev(T_cmb: float = 2.7255) -> float:
    """Standard neutrino temperature today, k_B T_ν0 [eV] (CAMB ``omnuh2_from_mass``)."""
    from gpucosmo.jax.eb import constants as C

    return C.K_B * float(T_cmb) / C.EV * (4.0 / 11.0) ** (1.0 / 3.0)


def omega_nu1_h2(T_cmb: float = 2.7255) -> float:
    """ω h² of one massless neutrino eigenstate (CAMB ``grhor/grhocrit·h²``)."""
    from gpucosmo.jax.eb import constants as C

    return C.OMEGA_G_H2 * C.RHO_NU_OVER_RHO_G * (float(T_cmb) / C.COBE_CMB_TEMP) ** 4


@functools.lru_cache(maxsize=None)
def _hierarchy_index(hierarchy: str) -> int:
    names = ("degenerate", "normal", "inverted")
    if hierarchy not in names:
        raise ValueError(f"neutrino_hierarchy {hierarchy!r}: one of {names}")
    return names.index(hierarchy)


@dataclasses.dataclass(frozen=True)
class NeutrinoModel:
    """Static neutrino setup (CAMB ``set_cosmology``'s ``neutrino_hierarchy``,
    ``num_massive_neutrinos``, ``meffsterile`` and ``standard_neutrino_neff``).

    The traced parameters are Σm_ν (``mnu``), N_eff (``nnu``) and the sterile effective mass
    (``meffsterile``); ``resolve`` maps them to CAMB's eigenstates. The number of eigenstate
    blocks is static: 1 for ``degenerate`` (``num_massive`` states of mass Σm_ν/num_massive),
    2 for ``normal``/``inverted`` (CAMB's two-eigenstate approximation: a degenerate pair and a
    single state; below the hierarchy's minimum Σm_ν CAMB falls back to one eigenstate, here the
    other block gets zero density), plus 1 for a thermal massive sterile neutrino.
    """

    hierarchy: str = "degenerate"
    num_massive: int = 1  # degenerate only (normal/inverted always have 3 massive states)
    sterile: bool = False
    standard_neff: float = 3.044
    # Momentum bins of the massive-ν perturbations (3, 4, 5: CAMB's optimized rules). None: the
    # 3-bin rule at the default masses, else 5, which holds for any mass (``hierarchy_jax.nu_nqmax``).
    # Analyses with a known mass range take CAMB's rule for it (``camb_nu_nqmax``, set by
    # ``build_mg_bundle``); as-run theories pin their CAMB's.
    nqmax: int | None = None

    def __post_init__(self):
        _hierarchy_index(self.hierarchy)
        if self.hierarchy == "degenerate" and int(self.num_massive) < 0:
            raise ValueError("num_massive must be ≥ 0")
        object.__setattr__(self, "num_massive",
                           int(self.num_massive) if self.hierarchy == "degenerate" else 3)
        object.__setattr__(self, "sterile", bool(self.sterile))
        object.__setattr__(self, "standard_neff", float(self.standard_neff))
        if self.nqmax is not None:
            if int(self.nqmax) < 3:
                raise ValueError("nqmax must be ≥ 3 (CAMB's smallest momentum rule)")
            object.__setattr__(self, "nqmax", int(self.nqmax))

    @property
    def base(self) -> "NeutrinoModel":
        """The physical model without the solver's quadrature choice (its tables' identity)."""
        return dataclasses.replace(self, nqmax=None)

    @property
    def n_active_eig(self) -> int:
        if self.hierarchy == "degenerate":
            return 1 if self.num_massive > 0 else 0
        return 2

    @property
    def n_eig(self) -> int:
        """Static number of massive eigenstate blocks."""
        return self.n_active_eig + (1 if self.sterile else 0)

    @property
    def numbers(self) -> tuple[int, ...]:
        """States per eigenstate block (CAMB ``nu_mass_numbers``)."""
        act = (self.num_massive,) if self.hierarchy == "degenerate" else (2, 1)
        return act[: self.n_active_eig] + ((1,) if self.sterile else ())

    @property
    def min_mnu(self) -> float:
        """Smallest Σm_ν [eV] of the model's physical branch (0 for degenerate)."""
        return {"degenerate": 0.0, "normal": MNU_MIN_NORMAL + 1e-4,
                "inverted": MNU_MIN_INVERTED + 1e-4}[self.hierarchy]

    def resolve(self, mnu, nnu, meffsterile=0.0, T_cmb: float = 2.7255, omnuh2=None) -> dict:
        """CAMB 2.0.4 eigenstates for traced (Σm_ν, N_eff, m_eff,sterile).

        ``omnuh2`` (instead of ``mnu``, degenerate models): CAMB's ``omnuh2_active`` route — the
        active ω_ν h² today is given and the eigenstate mass is the exact inverse of the thermal
        density (the DES/CosmoSIS chains' parameterization).

        Returns ``N_ur`` (massless N_eff), and per block (shape ``(n_eig,)``): ``deg`` (N_eff
        carried, CAMB ``nu_mass_degeneracies``), ``number``, ``mass`` [eV per state], ``am``
        (m/(k_B T) of the block's temperature, CAMB ``nu_masses``), ``omh2_massless`` (ω h² the
        block would have if massless, CAMB ``grhormass/grhocrit·h²``) and ``omnuh2`` (its ω h²
        today), plus the total ``omnuh2`` and ``omnuh2_sterile``.
        """
        f64 = jnp.float64
        mnu = jnp.asarray(0.0 if mnu is None else mnu, dtype=f64)  # unused with omnuh2
        nnu = jnp.asarray(nnu, dtype=f64)
        std = self.standard_neff
        zero = jnp.zeros_like(mnu + nnu)
        degs, nums, masses = [], [], []
        if omnuh2 is not None and (self.hierarchy != "degenerate" or self.sterile):
            raise NotImplementedError("the omnuh2 route is implemented for degenerate active neutrinos only")
        if self.hierarchy == "degenerate":
            if self.num_massive > 0:
                n_m = float(self.num_massive)
                neff_ms = n_m * std / 3.0
                over = nnu > neff_ms
                N_ur = jnp.where(over, nnu - neff_ms, 0.0)
                g = jnp.where(over, neff_ms, nnu)
                degs.append(g)
                nums.append(n_m)
                if omnuh2 is None:
                    masses.append(mnu / n_m)
                else:  # mass per state from the exact inverse of ω(am) = g ω_1 ρ(am)
                    om = jnp.asarray(omnuh2, dtype=f64)
                    am = find_nu_mass_for_rho_jax(om / (jnp.maximum(g, 1e-30) * omega_nu1_h2(T_cmb)))
                    masses.append(am * nu_temperature_ev(T_cmb) * (jnp.maximum(g, 1e-30) / n_m) ** 0.25)
            else:
                N_ur = nnu + zero
        else:
            normal = self.hierarchy == "normal"
            sgn = 1.0 if normal else 0.0
            two = mnu > (MNU_MIN_NORMAL if normal else MNU_MIN_INVERTED) + 1e-4
            # Two-eigenstate branch: three massive states sharing N_eff.
            neff3 = jnp.minimum(nnu, std)
            N_ur3 = jnp.maximum(nnu - std, 0.0)
            if normal:
                m1 = _solve_sum_mnu(mnu, lambda m: m + jnp.sqrt(m * m + DELTA_MNU21)
                                    + jnp.sqrt(m * m + DELTA_MNU31))
            else:
                m3 = _solve_sum_mnu(mnu, lambda m: m + jnp.sqrt(m * m + DELTA_MNU31)
                                    + jnp.sqrt(m * m + DELTA_MNU31 + DELTA_MNU21))
                m1 = jnp.sqrt(m3 * m3 + DELTA_MNU31)
            m_heavy = mnu - 2.0 * m1  # CAMB: m3 = mnu − 2 m1 (keeps the sum)
            # One-eigenstate fallback: 1 massive state (normal) or a degenerate pair (inverted).
            n_low = 1.0 if normal else 2.0
            neff_low_std = n_low * std / 3.0
            neff_low = jnp.minimum(nnu, neff_low_std)
            N_ur1 = jnp.maximum(nnu - neff_low_std, 0.0)
            N_ur = jnp.where(two, N_ur3, N_ur1)
            # block 1: the pair (n = 2); block 2: the single state (n = 1)
            deg_pair = jnp.where(two, neff3 * 2.0 / 3.0, (1.0 - sgn) * neff_low)
            deg_single = jnp.where(two, neff3 / 3.0, sgn * neff_low)
            m_pair = jnp.where(two, m1, mnu / 2.0)
            m_single = jnp.where(two, m_heavy, mnu)
            degs += [deg_pair, deg_single]
            nums += [2.0, 1.0]
            masses += [m_pair, m_single]
        # CAMB: Σm_ν = 0 makes every active state massless (omnuh2 = 0 → num_nu_massless = nnu).
        if degs:
            massless = (mnu <= 0.0) if omnuh2 is None else (jnp.asarray(omnuh2) <= 0.0)
            N_ur = jnp.where(massless, N_ur + sum(degs), N_ur)
            degs = [jnp.where(massless, 0.0, g) for g in degs]
        T_nu = nu_temperature_ev(T_cmb)
        om1 = omega_nu1_h2(T_cmb)
        out_deg, out_num, out_m, out_am, out_om0 = [], [], [], [], []
        for g, n, m in zip(degs, nums, masses):
            g_safe = jnp.maximum(g, 1e-30)
            # CAMB omnuh2_from_mass: the block's temperature is (g/n)^(1/4) T_ν; nu_masses ≥ 1e-3.
            am = jnp.maximum(m / (T_nu * (g_safe / n) ** 0.25), 1e-3)
            out_deg.append(g)
            out_num.append(jnp.asarray(n, dtype=f64) + zero)
            out_m.append(m + zero)
            out_am.append(am)
            out_om0.append(g * om1)
        omnuh2_sterile = zero
        if self.sterile:
            ms = jnp.asarray(meffsterile, dtype=f64)
            fac = NEUTRINO_MASS_FAC * (2.7255 / float(T_cmb)) ** 3
            omnuh2_sterile = ms / fac + zero  # Planck-paper definition (not thermally corrected)
            # CAMB: active massless = standard − active massive N_eff; the sterile carries ΔN_eff.
            act_deg = sum(out_deg) if out_deg else zero
            has_s = ms > 0.0  # m_eff = 0: CAMB adds no sterile state (ΔN_eff stays massless)
            N_ur = jnp.where(has_s, jnp.maximum(std - act_deg, 0.0), N_ur)
            g_s = jnp.where(has_s, jnp.maximum(nnu - std, 1e-6), 0.0)
            am_s = find_nu_mass_for_rho_jax(omnuh2_sterile / (jnp.maximum(g_s, 1e-30) * om1))
            out_deg.append(g_s)
            out_num.append(jnp.ones_like(zero))
            out_m.append(am_s * T_nu * g_s**0.25)
            out_am.append(jnp.maximum(am_s, 1e-3))
            out_om0.append(g_s * om1)
        if not out_deg:
            e = jnp.zeros(zero.shape + (0,), dtype=f64)
            return dict(N_ur=N_ur, deg=e, number=e, mass=e, am=e, omh2_massless=e, omnuh2_each=e,
                        omnuh2=zero, omnuh2_sterile=zero)
        st = lambda xs: jnp.stack([jnp.asarray(x, dtype=f64) for x in xs], axis=-1)  # noqa: E731
        deg, number, mass, am, om0 = st(out_deg), st(out_num), st(out_m), st(out_am), st(out_om0)
        each = om0 * thermal_nu_rho(am)
        return dict(N_ur=N_ur, deg=deg, number=number, mass=mass, am=am, omh2_massless=om0,
                    omnuh2_each=each, omnuh2=jnp.sum(each, axis=-1), omnuh2_sterile=omnuh2_sterile)


def _solve_sum_mnu(mnu, total):
    """The m ≥ 0 with ``total(m) = Σ`` for the hierarchy sums (increasing and convex in m, with
    total(m) ≥ 3m): Newton from Σ/3 ≥ root converges monotonically from above; unrolled so it
    stays a few fused kernels under jit/vmap. 0 when Σ < total(0) (CAMB's one-eigenstate branch
    is used there)."""
    m = jnp.maximum(mnu, 0.0) / 3.0
    for _ in range(16):
        f, df = jax.jvp(total, (m,), (jnp.ones_like(m),))
        m = jnp.maximum(m - (f - mnu) / df, 0.0)
    return m


DEFAULT_NEUTRINO_MODEL = NeutrinoModel()  # CAMB default: one massive eigenstate


def nu_rho_p_omega(params: dict, a) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(Σ_i Ω_i ρ_i, Σ_i Ω_i p_i)`` over the massive eigenstates at scale factor ``a``:
    multiply by ``3 H0²/a²`` for CAMB's ``grhonu_t``/``gpnu_t`` or by ``a⁻⁴`` for E²(a).
    ``Ω_i`` is ``params["Omega_nu_mass"]`` (density the block would have if massless) and
    ``ρ_i``/``p_i`` its ``thermal_nu_rho_p(a·am_i)``. Accepts the legacy scalar layout too."""
    om = params.get("Omega_nu_mass", 0.0)
    a = jnp.asarray(a, dtype=jnp.float64)
    if isinstance(om, (int, float)) and om == 0.0:
        z = jnp.zeros_like(a)
        return z, z
    om = jnp.asarray(om, dtype=jnp.float64)
    am = jnp.asarray(params["nu_mass_am"], dtype=jnp.float64)
    if om.ndim == 0:
        rho, p = thermal_nu_rho_p(a * am)
        return om * rho, om * p
    rho, p = thermal_nu_rho_p(a[..., None] * am)
    return jnp.sum(om * rho, axis=-1), jnp.sum(om * p, axis=-1)


def nu_massive_grho_gp(params: dict, a: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(grhonu_t, gpnu_t)`` for the massive eigenstates, in ``3 H0^2`` units.

    Matches CAMB's ``grhonu_t = grhormass/a^2 * rhonu``. ``Omega_nu_mass`` is the
    density these states *would* have if massless, so the ``1/a^2`` scaling plus
    the ``rhonu(am)`` factor reproduces the full relativistic-to-matter
    transition. Returns zeros when the cosmology has no massive species, so
    callers need no branch.
    """
    a = jnp.asarray(a, dtype=jnp.float64)
    rho_om, p_om = nu_rho_p_omega(params, a)
    return rho_om / (a * a), p_om / (a * a)


# --- perturbations (Nu_Integrate_L012) -----------------------------------
def nu_integrate_l012(
    y_nu: jnp.ndarray,
    am: jnp.ndarray,
    nu_q: jnp.ndarray,
    kernel: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """``(drhonu, fnu, dpnu, pinu)`` from the massive-nu multipole block.

    ``y_nu`` has shape ``(nqmax, lmaxnu + 1)``: multipole ``l`` of momentum bin
    ``i`` at ``y_nu[i, l]``. Vectorized over bins rather than looped, so this is
    one fused kernel instead of ``nqmax`` sequential ones.
    """
    aq = am / nu_q
    v = 1.0 / jnp.sqrt(1.0 + aq * aq)
    psi0 = y_nu[:, 0]
    psi1 = y_nu[:, 1]
    psi2 = y_nu[:, 2]
    drhonu = jnp.sum(kernel * psi0 / v)
    fnu = jnp.sum(kernel * psi1)
    dpnu = jnp.sum(kernel * psi0 * v) / 3.0
    pinu = jnp.sum(kernel * psi2 * v)
    return drhonu, fnu, dpnu, pinu


def nu_massive_terms(
    params: dict,
    a: jnp.ndarray,
    y: jnp.ndarray,
    cfg,
) -> tuple[jnp.ndarray, ...]:
    """``(grhonu_t, gpnu_t, dgrho_nu, dgq_nu, dgpi_nu)`` in CAMB's ``3 H0^2`` units.

    Works for a scalar state (``y`` shape ``(n_eq,)``) and for a stack over tau
    (``y`` shape ``(n_tau, n_eq)``), so the hierarchy and the source builder
    share one implementation. Returns zeros when ``cfg.nqmax == 0``.

    Note ``grhonu_t * clxnu == grhormass_t * drhonu`` because CAMB defines
    ``clxnu = drhonu / rhonu``, so the raw L012 integrals are used directly.
    """
    a = jnp.asarray(a, dtype=jnp.float64)
    zero = jnp.zeros_like(a)
    if getattr(cfg, "nqmax", 0) <= 0:
        return zero, zero, zero, zero, zero

    nu_q_np, kernel_np = nu_momentum_grid(cfg.nqmax)
    nu_q = jnp.asarray(nu_q_np)
    kernel = jnp.asarray(kernel_np)

    am_e, om_e = nu_eig_arrays(params, cfg.n_nu_eig)
    am = a[..., None] * am_e  # (..., n_eig)
    fac = 3.0 * params["H0_mpc"] ** 2
    grhormass_t = fac * om_e / (a * a)[..., None]
    rhonu, pnu = thermal_nu_rho_p(am)
    y_nu = nu_block(y, cfg)  # (..., n_eig, nq, l)

    v = 1.0 / jnp.sqrt(1.0 + (am[..., None] / nu_q) ** 2)  # (..., n_eig, nq)
    drhonu = jnp.sum(kernel * y_nu[..., 0] / v, axis=-1)
    fnu = jnp.sum(kernel * y_nu[..., 1], axis=-1)
    pinu = jnp.sum(kernel * y_nu[..., 2] * v, axis=-1)

    tot = lambda x: jnp.sum(grhormass_t * x, axis=-1)  # noqa: E731
    return tot(rhonu), tot(pnu), tot(drhonu), tot(fnu), tot(pinu)


def nu_eig_arrays(params: dict, n_eig: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(nu_mass_am, Omega_nu_mass)`` as ``(n_eig,)`` arrays (accepts the legacy scalars) and
    checks they match the hierarchy's static ``n_nu_eig``."""
    am = jnp.atleast_1d(jnp.asarray(params["nu_mass_am"], dtype=jnp.float64))
    om = jnp.atleast_1d(jnp.asarray(params["Omega_nu_mass"], dtype=jnp.float64))
    if am.shape[-1] != n_eig or om.shape[-1] != n_eig:
        raise ValueError(f"params have {am.shape[-1]} massive-ν eigenstate blocks, the hierarchy "
                         f"layout {n_eig} (HierarchyConfig.n_nu_eig; see hierarchy_for_model)")
    return am, om


def nu_block(y: jnp.ndarray, cfg) -> jnp.ndarray:
    """The massive-ν part of a state (or stack of states) as ``(..., n_nu_eig, nqmax, lmaxnu+1)``."""
    block = y[..., cfg.nu_ix: cfg.nu_ix + cfg.n_nu_massive]
    return block.reshape(block.shape[:-1] + (cfg.n_nu_eig, cfg.nqmax, cfg.lmaxnu + 1))


def nu_pinudot_raw(
    y_nu: jnp.ndarray,
    ydot_nu: jnp.ndarray,
    am: jnp.ndarray,
    nu_q: jnp.ndarray,
    kernel: jnp.ndarray,
    adotoa: jnp.ndarray,
) -> jnp.ndarray:
    """CAMB ``Nu_pinudot`` raw integral ``Σ kernel (ψ₂̇ v + ψ₂ v̇)``.

    Works for a single state ``(nqmax, lmax+1)`` or a stack over τ with a
    leading axis. Does not include the high-q massless+pert_scale branch
    (our evolve layout keeps every bin in the massive hierarchy).
    """
    aq = am[..., None] / nu_q if jnp.ndim(am) else am / nu_q
    aqdot = aq * (adotoa[..., None] if jnp.ndim(adotoa) else adotoa)
    v = 1.0 / jnp.sqrt(1.0 + aq * aq)
    vdot = -aq * aqdot / (1.0 + aq * aq) ** 1.5
    psi2 = y_nu[..., 2]
    psi2dot = ydot_nu[..., 2]
    return jnp.sum(kernel * (psi2dot * v + psi2 * vdot), axis=-1)


def nu_massive_source_terms(
    params: dict,
    a: jnp.ndarray,
    adotoa: jnp.ndarray,
    y: jnp.ndarray,
    ayprime: jnp.ndarray,
    cfg,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """``(dgpi_diff, pidot_nu, dgpi_3wp1_nu)`` for CAMB ``MassiveNuVarsOut``, summed over the
    eigenstate blocks.

    ``dgpi_diff = Σ pinu*(3 p - ρ)``, ``pidot_nu = Σ grhonu_t * pinudot`` where
    ``pinudot`` is the CAMB-normalized shear derivative, ``dgpi_3wp1_nu = Σ pinu*(3 p + ρ)``.
    Returns zeros when ``cfg.nqmax == 0``.
    """
    a = jnp.asarray(a, dtype=jnp.float64)
    adotoa = jnp.asarray(adotoa, dtype=jnp.float64)
    zero = jnp.zeros_like(a)
    if getattr(cfg, "nqmax", 0) <= 0:
        return zero, zero, zero

    nu_q_np, kernel_np = nu_momentum_grid(cfg.nqmax)
    nu_q = jnp.asarray(nu_q_np)
    kernel = jnp.asarray(kernel_np)

    am_e, om_e = nu_eig_arrays(params, cfg.n_nu_eig)
    am = a[..., None] * am_e  # (..., n_eig)
    fac = 3.0 * params["H0_mpc"] ** 2
    grhormass_t = fac * om_e / (a * a)[..., None]
    rhonu, pnu = thermal_nu_rho_p(am)
    grhonu_t = grhormass_t * rhonu
    gpnu_t = grhormass_t * pnu

    y_nu = nu_block(y, cfg)
    ydot_nu = nu_block(ayprime, cfg)

    v = 1.0 / jnp.sqrt(1.0 + (am[..., None] / nu_q) ** 2)
    pinu_raw = jnp.sum(kernel * y_nu[..., 2] * v, axis=-1)
    pinu_camb = pinu_raw / jnp.maximum(rhonu, 1e-30)

    adotoa_e = jnp.asarray(adotoa, dtype=jnp.float64)[..., None]
    pinudot_raw = nu_pinudot_raw(y_nu, ydot_nu, am, nu_q, kernel, adotoa_e)
    rhonudot = thermal_nu_drho(am, adotoa_e)
    pinudot = pinudot_raw / jnp.maximum(rhonu, 1e-30) - rhonudot / jnp.maximum(
        rhonu, 1e-30
    ) * pinu_camb

    # Summed over blocks: CAMB's MassiveNuVarsOut loops over the eigenstates.
    dgpi_diff = jnp.sum(pinu_camb * (3.0 * gpnu_t - grhonu_t), axis=-1)
    pidot_nu = jnp.sum(grhonu_t * pinudot, axis=-1)
    # Σ_i π_i (3 p_i + ρ_i): the massive-ν part of Σ (3w+1) ρ π (MG source).
    dgpi_3wp1_nu = jnp.sum(pinu_camb * (3.0 * gpnu_t + grhonu_t), axis=-1)
    return dgpi_diff, pidot_nu, dgpi_3wp1_nu


def nu_multipole_derivs(
    y_nu: jnp.ndarray,
    *,
    k: jnp.ndarray,
    am: jnp.ndarray,
    nu_q: jnp.ndarray,
    z: jnp.ndarray,
    sigma: jnp.ndarray,
    cothxor: jnp.ndarray,
    lmaxnu: int,
    kf: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """d/dtau of the massive-nu multipoles, shape ``(nqmax, lmaxnu + 1)``.

    CAMB ``equations.f90`` massive-neutrino block: identical to the massless
    hierarchy except every streaming term carries the velocity factor
    ``v = 1/sqrt(1 + (am/q)^2)``. ``kf``: CAMB's curvature factors Kf(ℓ), ℓ = 0..lmaxnu
    (None: flat).
    """
    aq = am / nu_q
    v = (1.0 / jnp.sqrt(1.0 + aq * aq))[:, None]  # (nqmax, 1)

    ls = jnp.arange(lmaxnu + 1, dtype=jnp.float64)
    denl = 1.0 / (2.0 * ls + 1.0)
    denlk = denl * k * ls
    denlk2 = denl * k * (ls + 1.0) * (1.0 if kf is None else kf)

    out = jnp.zeros_like(y_nu)
    # l = 0
    out = out.at[:, 0].set(-k * (4.0 / 3.0 * z + v[:, 0] * y_nu[:, 1]))
    # l = 1
    out = out.at[:, 1].set(
        v[:, 0] * (denlk[1] * y_nu[:, 0] - denlk2[1] * y_nu[:, 2])
    )
    if lmaxnu == 2:
        out = out.at[:, 2].set(-out[:, 0] - 3.0 * cothxor * y_nu[:, 2])
        return out

    # l = 2 picks up the shear source
    out = out.at[:, 2].set(
        v[:, 0] * (denlk[2] * y_nu[:, 1] - denlk2[2] * y_nu[:, 3])
        + k * 8.0 / 15.0 * sigma
    )
    # interior l = 3 .. lmaxnu-1, vectorized over (bin, l)
    if lmaxnu > 3:
        lo, hi = 3, lmaxnu
        out = out.at[:, lo:hi].set(
            v
            * (
                denlk[lo:hi] * y_nu[:, lo - 1 : hi - 1]
                - denlk2[lo:hi] * y_nu[:, lo + 1 : hi + 1]
            )
        )
    # truncation
    out = out.at[:, lmaxnu].set(
        k * v[:, 0] * y_nu[:, lmaxnu - 1]
        - (lmaxnu + 1) * cothxor * y_nu[:, lmaxnu]
    )
    return out


__all__ = [
    "FERMI_DIRAC_CONST",
    "NEUTRINO_MASS_FAC",
    "nu_accuracy_to_nqmax",
    "camb_nu_nqmax",
    "NU_Q4_AM",
    "NU_Q5_AM",
    "nu_integrate_l012",
    "nu_block",
    "nu_eig_arrays",
    "nu_rho_p_omega",
    "nu_temperature_ev",
    "omega_nu1_h2",
    "NeutrinoModel",
    "DEFAULT_NEUTRINO_MODEL",
    "find_nu_mass_for_rho_jax",
    "nu_massive_grho_gp",
    "nu_massive_source_terms",
    "nu_massive_terms",
    "nu_momentum_grid",
    "nu_multipole_derivs",
    "nu_pinudot_raw",
    "thermal_nu_drho",
    "thermal_nu_rho",
    "thermal_nu_rho_p",
]
