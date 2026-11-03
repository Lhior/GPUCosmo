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
"""Pure-JAX scalar hierarchy — CAMB ``initial`` / ``derivs`` (TransferOnly).

State layout matches ``hierarchy.py`` (AccuracyBoost=1 TransferOnly caps):
  0: etak, 1: clxc, 2: clxb, 3: vb
  4 .. : photon multipoles (G_IX=4)
  pol_base .. : E2 .. E_lmaxgpol
  r_ix .. : massless ν multipoles

No mutable EVState: flags are args; ``derivs`` returns ``(ayprime, pig, pigdot)``.
Thermo / background are plain dicts of floats / ``jnp`` arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from gpucosmo.jax.eb.dark_energy import cpl_f_de

from gpucosmo.jax.eb.interp_jax import cubic_pack_uniform, interp_cubic, interp_cubic_uniform
from gpucosmo.jax.eb import constants as C
from gpucosmo.jax.eb.vec_builder import VecBuilder

jax.config.update("jax_enable_x64", True)

IX_ETAK = 0
IX_CLXC = 1
IX_CLXB = 2
IX_VB = 3
G_IX = 4


@dataclass(frozen=True)
class HierarchyConfig:
    lmaxg: int = C.LMAXG_TRANSFER
    lmaxnr: int = C.LMAXNR_TRANSFER
    lmaxgpol: int = C.LMAXGPOL_TRANSFER
    include_pol: bool = True
    second_order_tc: bool = True
    # Massive neutrinos: n_nu_eig eigenstate blocks of nqmax momentum bins x (lmaxnu + 1)
    # multipoles appended after the massless block (row-major: block, bin, ℓ). nqmax = 0
    # reproduces the massless-only layout. n_nu_eig must match the params'
    # (``massive_nu.NeutrinoModel.n_eig``; ``hierarchy_for_model`` sets it).
    nqmax: int = 0
    lmaxnu: int = C.LMAXNU_TRANSFER
    n_nu_eig: int = 1
    # An eigenstate may be light (≲ NU_LIGHT_MASS): its momentum bins stream at v ≈ 1 to τ0, so
    # the segmented integrator's late (RSA) segment needs a larger step budget for RK4 stability
    # (``evolve_segmented_jax.NU_LIGHT_RSA_MULT``). Set by ``hierarchy_for_model``.
    nu_light: bool = False
    # Spatial curvature (params "curv" = K [Mpc⁻²]): the solver's wavenumber is CAMB's q (ν = q R),
    # the equations use k = √(q² − K), Kf(ℓ) = 1 − K ℓ(ℓ+2)/k² in the upward couplings, the
    # curved truncation cot_K and the K terms of the Einstein equations (equations.f90). Static so
    # the flat program is unchanged.
    curved: bool = False
    # CAMB DarkEnergyPPF: one Gamma state after massless ν (equations.f90:619-623).
    # When False the layout matches the pre-PPF massless/massive indexing.
    use_de_ppf: bool = False
    # μ–Σ modified gravity (ISiTGR synchronous-gauge branch, ``mg_jax``), active
    # for a ≥ 0.02 and outside tight coupling. Static so the GR program is
    # unchanged; μ0/Σ0 are traced through the params dict.
    mg: bool = False

    @property
    def n_phot(self) -> int:
        return self.lmaxg + 1

    @property
    def n_pol(self) -> int:
        return max(self.lmaxgpol - 1, 0) if self.include_pol else 0

    @property
    def pol_base(self) -> int:
        return G_IX + self.n_phot

    @property
    def polind(self) -> int:
        return self.pol_base - 2

    @property
    def n_nu(self) -> int:
        return self.lmaxnr + 1

    @property
    def r_ix(self) -> int:
        return self.pol_base + self.n_pol

    @property
    def w_ix(self) -> int:
        """PPF Gamma index (after massless ν); unused when ``use_de_ppf`` is False."""
        return self.r_ix + self.n_nu

    @property
    def nu_ix(self) -> int:
        """Start of the massive-neutrino block (row-major over momentum bins)."""
        return self.w_ix + (1 if self.use_de_ppf else 0)

    @property
    def n_nu_massive(self) -> int:
        return self.n_nu_eig * self.nqmax * (self.lmaxnu + 1)

    @property
    def n_eq(self) -> int:
        return self.nu_ix + self.n_nu_massive

    @property
    def poltruncfac(self) -> float:
        return float(self.lmaxgpol) / max(1, self.lmaxgpol - 2)


# Lightest eigenstate mass [eV] the default segmented step budgets are stable for (at 1e-4 eV the
# late segment's RK4 diverges from k ≈ 0.16/Mpc: the bins stream at v ≈ 1 to τ0).
NU_LIGHT_MASS = 0.05


def nu_may_be_light(nu_model=None, ext: dict | None = None) -> bool:
    """Can an eigenstate of ``nu_model`` with these ``ext`` values be lighter than NU_LIGHT_MASS?
    True for the hierarchies (their light states), sterile neutrinos, and any Σm_ν (``mnu``) or ω_ν
    (``omnuh2``) given other than the default Σm_ν with a single heavy state. Deliberately
    value-independent otherwise: a fixed value is a Python float in the staged theory but a traced
    array in the batched exact one, and both must take the same step budgets."""
    from gpucosmo.jax.eb.massive_nu import DEFAULT_NEUTRINO_MODEL

    model = DEFAULT_NEUTRINO_MODEL if nu_model is None else nu_model
    if model.n_eig == 0:
        return False
    if model.hierarchy != "degenerate" or model.sterile:
        return True
    ext = ext or {}
    if "omnuh2" in ext:
        return True
    if "mnu" not in ext:
        return C.DEFAULT_MNU / max(model.num_massive, 1) < NU_LIGHT_MASS
    mnu = ext["mnu"]
    is_default = isinstance(mnu, (int, float, np.floating)) and float(mnu) == C.DEFAULT_MNU
    return not is_default or C.DEFAULT_MNU / max(model.num_massive, 1) < NU_LIGHT_MASS


def nu_nqmax(nu_model=None, ext: dict | None = None) -> int:
    """Momentum bins for the massive-ν perturbations of ``nu_model`` with these ``ext`` values: the
    model's ``nqmax`` when set, else CAMB's 3-bin rule at the default masses (≤ 0.06 eV) and the
    5-bin rule, good for any mass (``massive_nu.camb_nu_nqmax``), once Σm_ν (``mnu``), ω_ν
    (``omnuh2``) or a sterile neutrino is given. Value-independent like ``nu_may_be_light``."""
    from gpucosmo.jax.eb.massive_nu import DEFAULT_NEUTRINO_MODEL

    model = DEFAULT_NEUTRINO_MODEL if nu_model is None else nu_model
    if model.nqmax is not None:
        return model.nqmax
    ext = ext or {}
    if model.sterile or "omnuh2" in ext:
        return 5
    if "mnu" not in ext:
        return 3
    mnu = ext["mnu"]
    is_default = isinstance(mnu, (int, float, np.floating)) and float(mnu) == C.DEFAULT_MNU
    return 3 if is_default else 5


def hierarchy_for_model(cfg: HierarchyConfig, nu_model=None, ext: dict | None = None) -> HierarchyConfig:
    """``cfg`` with the massive-ν layout of ``nu_model`` (``massive_nu.NeutrinoModel``; None:
    CAMB's default single eigenstate): its number of eigenstate blocks, no massive block when the
    model has none, its momentum bins (``nu_nqmax``; a config with more keeps them) and
    ``nu_light`` (``nu_may_be_light`` with the ``ext`` values; a config that already has it keeps
    it). A config without massive neutrinos (nqmax = 0) is returned as is."""
    import dataclasses

    from gpucosmo.jax.eb.massive_nu import DEFAULT_NEUTRINO_MODEL

    model = DEFAULT_NEUTRINO_MODEL if nu_model is None else nu_model
    if cfg.nqmax <= 0:
        return cfg
    if model.n_eig == 0:
        return dataclasses.replace(cfg, nqmax=0, n_nu_eig=1, nu_light=False)
    return dataclasses.replace(cfg, n_nu_eig=model.n_eig, nqmax=max(cfg.nqmax, nu_nqmax(model, ext)),
                               nu_light=cfg.nu_light or nu_may_be_light(model, ext))


# CAMB DarkEnergyPPF: c_Gamma_ppf = 0.4 * sqrt(cs2_lam) with cs2_lam = 1.
C_GAMMA_PPF = 0.4
# Quasi-static branch threshold (DarkEnergyPPF.f90:quasi_static_ckH2).
PPF_QUASI_STATIC_CKH2 = 30.0


def ppf_perturbed_stress_energy(
    *,
    Gamma: jnp.ndarray,
    dgrho: jnp.ndarray,
    dgq: jnp.ndarray,
    grho: jnp.ndarray,
    grhov: jnp.ndarray,
    w: jnp.ndarray,
    gpres_noDE: jnp.ndarray,
    etak: jnp.ndarray,
    adotoa: jnp.ndarray,
    k: jnp.ndarray,
    kf1: float = 1.0,
    c_gamma: float = C_GAMMA_PPF,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """CAMB ``TDarkEnergyPPF_PerturbedStressEnergy``.

    Returns ``(dgrho_de, dgq_de, Gammadot)``. Inputs ``dgrho``/``dgq`` must be
    the matter+radiation(+massive ν) totals *before* DE stress is added.
    """
    k2 = k * k
    adotoa_rec = 1.0 / jnp.maximum(adotoa, 1e-30)
    grhoT = grho - grhov
    vT = dgq / jnp.maximum(grhoT + gpres_noDE, 1e-30)
    sigma = (
        (etak + (dgrho + 3.0 * adotoa / jnp.maximum(k, 1e-30) * dgq) / (2.0 * k)) / kf1
        - k * Gamma
    ) * adotoa_rec
    S_Gamma = grhov * (1.0 + w) * (vT + sigma) * adotoa_rec / (2.0 * jnp.maximum(k, 1e-30))
    ckH = c_gamma * k * adotoa_rec
    ckH2 = ckH * ckH
    Gamma_qs = S_Gamma / (1.0 + ckH2) ** 2
    Gammadot_qs = (1.0 + PPF_QUASI_STATIC_CKH2) * adotoa * (Gamma_qs - Gamma)
    Gammadot_dyn = (S_Gamma / (1.0 + ckH2) - (1.0 + ckH2) * Gamma) * adotoa
    Gammadot = jnp.where(ckH2 > PPF_QUASI_STATIC_CKH2, Gammadot_qs, Gammadot_dyn)
    Fa = 1.0 + 3.0 * (grhoT + gpres_noDE) / (2.0 * jnp.maximum(k2, 1e-30) * kf1)
    dgqe = S_Gamma - Gammadot * adotoa_rec - Gamma
    dgqe = -dgqe / jnp.maximum(Fa, 1e-30) * 2.0 * k * adotoa + vT * grhov * (1.0 + w)
    dgrhoe = -2.0 * k2 * kf1 * Gamma - 3.0 * adotoa / jnp.maximum(k, 1e-30) * dgqe
    return dgrhoe, dgqe, Gammadot


def ppf_diff_rhopi_add_term(
    *,
    dgrhoe: jnp.ndarray,
    dgqe: jnp.ndarray,
    grho: jnp.ndarray,
    gpres: jnp.ndarray,
    w: jnp.ndarray,
    adotoa: jnp.ndarray,
    k: jnp.ndarray,
    grhov: jnp.ndarray,
    z: jnp.ndarray,
    k2: jnp.ndarray,
    Gammadot: jnp.ndarray,
    Gamma: jnp.ndarray,
    kf1: float = 1.0,
    grhok: float = 0.0,
) -> jnp.ndarray:
    """CAMB ``TDarkEnergyPPF_diff_rhopi_Add_Term`` (flat universe: ``grhok=0``)."""
    adotoa_rec = 1.0 / jnp.maximum(adotoa, 1e-30)
    k_rec = 1.0 / jnp.maximum(k, 1e-30)
    hdotoh = (-3.0 * grho - 3.0 * gpres - 2.0 * grhok) / 6.0 * adotoa_rec
    ppiedot = (
        3.0 * dgrhoe
        + dgqe
        * (
            12.0 * k_rec * adotoa
            + k * adotoa_rec
            - 3.0 * k_rec * (adotoa + hdotoh)
        )
        + grhov * (1.0 + w) * k * z * adotoa_rec
        - 2.0 * k2 * kf1 * (Gammadot * adotoa_rec - 2.0 * Gamma)
    )
    return ppiedot * adotoa / kf1


def denl(j: int) -> float:
    return 1.0 / (2 * j + 1)


def curvature_k(q, p: dict, cfg: HierarchyConfig):
    """``(k, k², kfd)``: CAMB's k = √(q² − K) from the grid wavenumber q, and ``kfd`` for ``kf``
    (None when flat: k = q, every Kf = 1)."""
    if not cfg.curved:
        return q, q * q, None
    K = p["curv"]
    k2 = q * q - K
    return jnp.sqrt(jnp.maximum(k2, 1e-30)), k2, (K, k2)


def kf(kfd, ell):
    """CAMB ``Kf(ℓ)`` = 1 − K ℓ(ℓ+2)/k² (1 when flat); ``ell`` an int or an array of ℓ.

    Clipped at 0: in a closed model Kf(ℓ) < 0 exactly for ℓ ≥ ν = q/√K, multipoles that do not
    exist (CAMB truncates every hierarchy at ``FirstZerolForBeta`` − 1 = ν − 1). For integer ν,
    Kf(ν − 1) = 0 already decouples them, so the clip is that truncation; between integers (the
    evolve grid's nodes) it keeps the unphysical ℓ ≥ ν modes from growing and leaking into ℓ < ν
    (the closed C_2 was 1.7 % low)."""
    if kfd is None:
        return 1.0
    K, k2 = kfd
    ell = jnp.asarray(ell, dtype=jnp.float64)
    return jnp.maximum(1.0 - K * (ell * (ell + 2.0)) / k2, 0.0)


def f_k(chi, K):
    """CAMB ``f_K(χ)`` = sin_K(√|K| χ)/√|K| (χ when flat), smooth through K = 0 (series)."""
    chi = jnp.asarray(chi, dtype=jnp.float64)
    u = -K * chi * chi  # > 0 open (sinh), < 0 closed (sin)
    small = jnp.abs(u) < 1e-3
    series = 1.0 + u / 6.0 + u * u / 120.0 + u**3 / 5040.0
    up = jnp.sqrt(jnp.where(u > 0.0, jnp.where(small, 1.0, u), 1.0))
    um = jnp.sqrt(jnp.where(u < 0.0, jnp.where(small, 1.0, -u), 1.0))
    return chi * jnp.where(small, series, jnp.where(u > 0.0, jnp.sinh(up) / up, jnp.sin(um) / um))


def cot_k(tau, K):
    """√|K| cot_K(√|K| τ): CAMB's ``cothxor`` (1/(R tanfunc(τ/R))) — cot for K > 0 (closed), coth for
    K < 0 (open), 1/τ when flat; the series 1/τ − Kτ/3 − K²τ³/45 near Kτ² = 0 keeps it smooth."""
    tau = jnp.maximum(jnp.asarray(tau, dtype=jnp.float64), 1e-30)
    u = K * tau * tau
    small = jnp.abs(u) < 1e-4
    series = (1.0 - u / 3.0 - u * u / 45.0) / tau
    sk = jnp.sqrt(jnp.where(small, 1.0, jnp.abs(K)))
    x = jnp.where(small, 1.0, sk * tau)
    curved = jnp.where(K > 0, sk / jnp.tan(x), sk / jnp.tanh(x))
    return jnp.where(small, series, curved)


def _denlk(k, j: int):
    return denl(j) * k * j


def _denlk2(k, j: int):
    return denl(j) * k * (j + 1)


def _polfack(k, j: int):
    return ((j + 3) * (j - 1) / (j + 1)) * k * denl(j)


def _params_floats(params) -> dict:
    """Accept FlatLCDMParams or dict with Omega_* / H0_mpc / w0 / wa."""
    if isinstance(params, dict):
        out = dict(params)
        out.setdefault("w0", -1.0)
        out.setdefault("wa", 0.0)
        out.setdefault("Omega_nu_mass", 0.0)
        out.setdefault("nu_mass_am", 0.0)
        out.setdefault("mu0", 0.0)
        out.setdefault("Sigma0", 0.0)
        return out
    return {
        "Omega_b": float(params.Omega_b),
        "Omega_c": float(params.Omega_c),
        "Omega_g": float(params.Omega_g),
        "Omega_nu": float(params.Omega_nu),
        "Omega_nu_mass": float(getattr(params, "Omega_nu_mass", 0.0)),
        "nu_mass_am": float(getattr(params, "nu_mass_am", 0.0)),
        "Omega_de": float(params.Omega_de),
        "H0_mpc": float(params.H0_mpc),
        "w0": float(getattr(params, "w0", -1.0)),
        "wa": float(getattr(params, "wa", 0.0)),
    }


def pack_thermo_values(thermo: dict) -> dict:
    """Add a packed (a, cs2, opacity, dopacity) table for ``thermo_values``.

    One direct-index lookup and gather per RHS call instead of four binary
    searches and per-call slope evaluation over the full table. Requires the
    uniform log-tau grid of ``build_thermo_jax``.
    """
    cols = jnp.stack([jnp.asarray(thermo[k], dtype=jnp.float64)
                      for k in ("a", "cs2", "opacity", "dopacity")], axis=1)
    log_grid = jnp.log(jnp.asarray(thermo["tau_mpc"], dtype=jnp.float64))
    return {**thermo, "cubic_pack": cubic_pack_uniform(log_grid, cols)}


def thermo_values(tau, thermo: dict, *, want_dopacity: bool = False):
    """Interpolate (a, cs2, opacity[, dopacity]) from device thermo tables."""
    tauminn = thermo.get("tauminn", 1.0e-6)
    logt = jnp.log(jnp.maximum(jnp.asarray(tau, dtype=jnp.float64), tauminn * 0.5))
    if "cubic_pack" in thermo:  # see pack_thermo_values
        v = interp_cubic_uniform(logt, thermo["cubic_pack"])
        cols = (v[..., 0], v[..., 1], v[..., 2], v[..., 3])
        return cols if want_dopacity else cols[:3]
    log_grid = jnp.log(jnp.asarray(thermo["tau_mpc"], dtype=jnp.float64))
    # C¹ cubic, not jnp.interp: table-node kinks alias against the source/knot
    # grids and make C_ℓ wiggle with the parameters (see interp_jax).
    a = interp_cubic(logt, log_grid, jnp.asarray(thermo["a"], dtype=jnp.float64))
    cs2 = interp_cubic(logt, log_grid, jnp.asarray(thermo["cs2"], dtype=jnp.float64))
    opacity = interp_cubic(
        logt, log_grid, jnp.asarray(thermo["opacity"], dtype=jnp.float64)
    )
    if want_dopacity:
        dop = interp_cubic(
            logt, log_grid, jnp.asarray(thermo["dopacity"], dtype=jnp.float64)
        )
        return a, cs2, opacity, dop
    return a, cs2, opacity


def _masked_block(y_m, y_c, y_p, ls, L, *, interior, trunc):
    del y_m, y_c, y_p
    return jnp.where(ls < L, interior, jnp.where(ls == L, trunc, 0.0))


def _masked_hierarchy_derivs(ayprime, y, k, cfg: HierarchyConfig, lmax_live, *, kfd=None,
                             opacity, cothxor, photons: bool, nu: bool, high_ktau):
    """Per-mode truncation (``derivs(..., lmax_live=...)``) for ℓ >= 3."""
    lg, lp, lr = (jnp.asarray(v) for v in lmax_live)
    zero = jnp.zeros(1, dtype=jnp.float64)
    if photons and cfg.lmaxg >= 3:
        ls = jnp.arange(3, cfg.lmaxg + 1, dtype=jnp.float64)
        den = 2.0 * ls + 1.0
        y_c = y[G_IX + 3 : G_IX + cfg.lmaxg + 1]
        y_m = y[G_IX + 2 : G_IX + cfg.lmaxg]
        y_p = jnp.concatenate([y[G_IX + 4 : G_IX + cfg.lmaxg + 1], zero])
        interior = (k * ls / den) * y_m - (k * (ls + 1.0) / den) * kf(kfd, ls) * y_p - opacity * y_c
        trunc = k * y_m - (ls + 1.0) * cothxor * y_c - opacity * y_c
        ayprime = ayprime.at[G_IX + 3 : G_IX + cfg.lmaxg + 1].set(
            _masked_block(y_m, y_c, y_p, ls, lg, interior=interior, trunc=trunc)
        )
        if cfg.include_pol and cfg.lmaxgpol >= 3:
            pb = cfg.pol_base
            ls = jnp.arange(3, cfg.lmaxgpol + 1, dtype=jnp.float64)
            den = 2.0 * ls + 1.0
            y_c = y[pb + 1 : pb + cfg.lmaxgpol - 1]
            y_m = y[pb : pb + cfg.lmaxgpol - 2]
            y_p = jnp.concatenate([y[pb + 2 : pb + cfg.lmaxgpol - 1], zero])
            polfac = ((ls + 3.0) * (ls - 1.0) / (ls + 1.0)) * k / den * kf(kfd, ls)
            interior = -opacity * y_c + (k * ls / den) * y_m - polfac * y_p
            trunc = (-opacity * y_c + k * (ls / (ls - 2.0)) * y_m
                     - (ls + 3.0) * cothxor * y_c)
            ayprime = ayprime.at[pb + 1 : pb + cfg.lmaxgpol - 1].set(
                _masked_block(y_m, y_c, y_p, ls, lp, interior=interior, trunc=trunc)
            )
    if nu and cfg.lmaxnr >= 3:
        r = cfg.r_ix
        ls = jnp.arange(3, cfg.lmaxnr + 1, dtype=jnp.float64)
        den = 2.0 * ls + 1.0
        y_c = y[r + 3 : r + cfg.lmaxnr + 1]
        y_m = y[r + 2 : r + cfg.lmaxnr]
        y_p = jnp.concatenate([y[r + 4 : r + cfg.lmaxnr + 1], zero])
        interior = (k * ls / den) * y_m - (k * (ls + 1.0) / den) * kf(kfd, ls) * y_p
        trunc = k * y_m - (ls + 1.0) * cothxor * y_c
        dots = _masked_block(y_m, y_c, y_p, ls, lr, interior=interior, trunc=trunc)
        if high_ktau is not False:
            dots = jnp.where(jnp.asarray(high_ktau), 0.0, dots)
        ayprime = ayprime.at[r + 3 : r + cfg.lmaxnr + 1].set(dots)
    return ayprime


def initial_adiabatic(k, tau, params, cfg: HierarchyConfig | None = None):
    """CAMB ``initial`` adiabatic ICs (equations.f90)."""
    cfg = cfg or HierarchyConfig()
    p = _params_floats(params)
    k, _k2, kfd = curvature_k(k, p, cfg)  # CAMB ``initial``: k = √(q² − K), Kf(1)
    H0 = p["H0_mpc"]
    grhob = 3.0 * H0**2 * p["Omega_b"]
    grhoc = 3.0 * H0**2 * p["Omega_c"]
    grhog = 3.0 * H0**2 * p["Omega_g"]
    # CAMB `initial` counts the massive eigenstates as radiation (they are ultra-relativistic
    # at the start): grhonu = grhormass + grhornomass in om and Rv. Massless-only made Rv = 0
    # with three massive states (P(k) −1.8e-3 at k = 2/Mpc vs CAMB) and Rv 2/3 of CAMB's with one.
    grhor = 3.0 * H0**2 * (p["Omega_nu"] + jnp.sum(jnp.atleast_1d(jnp.asarray(p.get("Omega_nu_mass", 0.0))), axis=-1))
    om = (grhob + grhoc) / jnp.sqrt(3.0 * (grhog + grhor))
    omtau = om * tau
    Rv = grhor / (grhor + grhog)
    Rp15 = 4.0 * Rv + 15.0

    x = k * tau
    x2 = x * x
    x3 = x2 * x
    Kf1 = kf(kfd, 1)
    chi = 1.0

    clxg = -chi * Kf1 / 3.0 * x2 * (1.0 - omtau / 5.0)
    clxr = clxg
    clxb = 0.75 * clxg
    clxc = clxb
    qg = clxg * x / 9.0
    qr = -chi * Kf1 * (4.0 * Rv + 23.0) / Rp15 * x3 / 27.0
    vb = 0.75 * qg
    pir = chi * 4.0 / 3.0 * x2 / Rp15 * (
        1.0 + omtau / 4.0 * (4.0 * Rv - 5.0) / (2.0 * Rv + 15.0)
    )
    aj3r = chi * 4.0 / 21.0 / Rp15 * x3
    eta = -chi * 2.0 * Kf1 * (1.0 - x2 / 12.0 * (-10.0 / Rp15 + Kf1))

    clxg, clxr, clxb, clxc = -clxg, -clxr, -clxb, -clxc
    qg, qr, vb, pir, aj3r, eta = -qg, -qr, -vb, -pir, -aj3r, -eta

    y = jnp.zeros(cfg.n_eq, dtype=jnp.float64)
    y = y.at[IX_ETAK].set(-eta * k / 2.0)
    y = y.at[IX_CLXC].set(clxc)
    y = y.at[IX_CLXB].set(clxb)
    y = y.at[IX_VB].set(vb)
    y = y.at[G_IX].set(clxg)
    y = y.at[G_IX + 1].set(qg)
    y = y.at[cfg.r_ix].set(clxr)
    y = y.at[cfg.r_ix + 1].set(qr)
    if cfg.lmaxnr >= 2:
        y = y.at[cfg.r_ix + 2].set(pir)
    if cfg.lmaxnr >= 3:
        y = y.at[cfg.r_ix + 3].set(aj3r)

    if cfg.nqmax > 0:
        # CAMB `initial`: massive neutrinos start ultra-relativistic, so every
        # momentum bin copies the massless l=0..3 multipoles.
        block = jnp.zeros((cfg.n_nu_eig, cfg.nqmax, cfg.lmaxnu + 1), dtype=jnp.float64)
        block = block.at[..., 0].set(clxr)
        block = block.at[..., 1].set(qr)
        if cfg.lmaxnu >= 2:
            block = block.at[..., 2].set(pir)
        if cfg.lmaxnu >= 3:
            block = block.at[..., 3].set(aj3r)
        y = y.at[cfg.nu_ix : cfg.nu_ix + cfg.n_nu_massive].set(block.reshape(-1))
    return y


def init_after_tight_coupling(
    y,
    k,
    pig,
    pigdot,
    opacity,
    dopacity,
    cfg: HierarchyConfig | None = None,
):
    """CAMB TC switchoff multipole init (equations.f90 ~323–343)."""
    cfg = cfg or HierarchyConfig()
    y = y.at[G_IX + 2].set(pig)
    op = jnp.maximum(opacity, 1e-30)

    def second_order(_):
        y3 = (
            (3.0 / 7.0) * pig * (k / op) * (1.0 + dopacity / op**2)
            + (3.0 / 7.0) * pigdot * (k / op**2) * (-1.0)
        )
        yy = y.at[G_IX + 3].set(y3)
        if cfg.include_pol and cfg.n_pol >= 1:
            e2 = (
                pig / 4.0
                + pigdot * (1.0 / op) * (-5.0 / 8.0 - (25.0 / 16.0) * dopacity / op**2)
                + pig * (k / op) ** 2 * (-5.0 / 56.0)
            )
            yy = yy.at[cfg.pol_base].set(e2)
        if cfg.include_pol and cfg.n_pol >= 2:
            e3 = (
                (3.0 / 7.0)
                * (k / op)
                * yy[cfg.pol_base]
                * (1.0 + dopacity / op**2)
                + (3.0 / 7.0)
                * (k / op**2)
                * ((pigdot / 4.0) * (1.0 + (5.0 / 2.0) * dopacity / op**2))
                * (-1.0)
            )
            yy = yy.at[cfg.pol_base + 1].set(e3)
        return yy

    def first_order(_):
        yy = y.at[G_IX + 3].set(3.0 / 7.0 * pig * k / op)
        if cfg.include_pol and cfg.n_pol >= 1:
            yy = yy.at[cfg.pol_base].set(pig / 4.0)
        if cfg.include_pol and cfg.n_pol >= 2:
            yy = yy.at[cfg.pol_base + 1].set(yy[G_IX + 3] / 4.0)
        return yy

    if cfg.second_order_tc:
        return jax.lax.cond(opacity > 0.0, second_order, first_order, None)
    return first_order(None)


BG_KEYS = ("cothxor", "grhog", "grhor", "grhob", "grhoc", "grhov", "w_de", "grhonu_t",
           "gpnu_t", "grho", "gpres_noDE", "adotoa", "photbar")


def background_at(tau, a, params, cfg: HierarchyConfig | None = None):
    """y-independent ``derivs`` inputs at (tau, a), stacked in ``BG_KEYS`` order."""
    cfg = cfg or HierarchyConfig()
    p = _params_floats(params)
    H0 = p["H0_mpc"]
    a2 = a * a
    cothxor = cot_k(tau, p["curv"]) if cfg.curved else 1.0 / jnp.maximum(tau, 1e-30)
    fac = 3.0 * H0 * H0
    grhog = fac * p["Omega_g"] / a2
    grhor = fac * p["Omega_nu"] / a2
    grhob = fac * p["Omega_b"] / a
    grhoc = fac * p["Omega_c"] / a
    w0 = p.get("w0", -1.0)
    wa = p.get("wa", 0.0)
    f_de = cpl_f_de(a, w0, wa)
    grhov = fac * p["Omega_de"] * f_de * a2
    w_de = w0 + wa * (1.0 - a)
    if cfg.nqmax > 0:
        from gpucosmo.jax.eb.massive_nu import nu_rho_p_omega

        rho_om, p_om = nu_rho_p_omega(p, a)
        grhonu_t = fac * rho_om / a2
        gpnu_t = fac * p_om / a2
    else:
        grhonu_t = jnp.zeros_like(a)
        gpnu_t = jnp.zeros_like(a)
    grho = grhob + grhoc + grhor + grhog + grhov + grhonu_t
    gpres_noDE = (grhor + grhog) / 3.0 + gpnu_t
    # CAMB: adotoa = √((grho + grhok)/3), grhok = grhocrit·Ω_k (a² ρ_k is constant)
    adotoa = jnp.sqrt((grho + fac * p["Omega_k"]) / 3.0) if cfg.curved else jnp.sqrt(grho / 3.0)
    photbar = grhog / jnp.maximum(grhob, 1e-30)
    vals = (cothxor, grhog, grhor, grhob, grhoc, grhov, w_de, grhonu_t, gpnu_t,
            grho, gpres_noDE, adotoa, photbar)
    return jnp.stack([jnp.broadcast_to(jnp.asarray(v, jnp.float64), jnp.shape(a)) for v in vals])


def _mg_constraints(*, use_mg, z_gr, sigma_gr, etak_dot_gr, y, k, a, p, cfg, etak,
                    adotoa, grho, grhov, w_de, gpres_noDE, grhog, grhor, grhob, grhoc,
                    grhonu_t, gpnu_t, dgrho, dgq, pig, pir, qg, qr, opacity, cothxor,
                    no_phot_multpoles, no_nu_multpoles):
    """ISiTGR μ–Σ replacement of the (z, σ, η̇_k) constraints (``mg_jax``).

    σ follows from the Newtonian potential Φ, η̇_k from the time derivative of
    the Σ constraint (which needs Σρπ̇ evaluated with that σ), and
    ``z = σ − 3η̇_k/k²``. Selected where ``use_mg`` (a ≥ 0.02); GR elsewhere.
    """
    from gpucosmo.jax.eb import mg_jax

    k2 = k * k
    gpres = gpres_noDE + w_de * grhov
    Hdot = -(grho + 3.0 * gpres) / 6.0
    rho_delta = dgrho + 3.0 * adotoa * dgq / k
    dgpi_rad = grhog * pig + grhor * pir
    dgpi = dgpi_rad
    dgpi_3wp1 = 2.0 * dgpi_rad
    if cfg.nqmax > 0:
        from gpucosmo.jax.eb.massive_nu import (
            nu_momentum_grid,
            nu_multipole_derivs,
            nu_pinudot_raw,
            thermal_nu_drho,
            thermal_nu_rho_p,
        )

        from gpucosmo.jax.eb.massive_nu import nu_block, nu_eig_arrays

        assert cfg.lmaxnu > 2, "MG needs the full massive-nu l=2 equation"
        nu_q_np, kernel_np = nu_momentum_grid(cfg.nqmax)
        nu_q, kernel = jnp.asarray(nu_q_np), jnp.asarray(kernel_np)
        am_e, om_e = nu_eig_arrays(p, cfg.n_nu_eig)
        am = a * am_e  # (n_eig,)
        grhormass_t = 3.0 * p["H0_mpc"] ** 2 * om_e / (a * a)
        rhonu, pnu = thermal_nu_rho_p(am)
        y_nu = nu_block(y, cfg)
        v = 1.0 / jnp.sqrt(1.0 + (am[:, None] / nu_q) ** 2)
        pinu_raw = jnp.sum(kernel * y_nu[..., 2] * v, axis=-1)  # (n_eig,)
        dgpi_nu_e = grhormass_t * pinu_raw
        dgpi = dgpi + jnp.sum(dgpi_nu_e)
        dgpi_3wp1 = dgpi_3wp1 + jnp.sum(dgpi_nu_e * (3.0 * pnu / rhonu + 1.0))
    rho_plus_p = (grhoc + grhob) + (grhor + grhog) * 4.0 / 3.0 + (grhonu_t + gpnu_t)
    mu, mudot, sig, sigdot = mg_jax.mu_sigma(k=k, a=a, adotoa=adotoa, Hdot=Hdot,
                                             grhov=grhov, grho=grho, params=p)
    _psi, _phi, sigma = mg_jax.newtonian_potentials(
        k=k, mu=mu, sig=sig, rho_delta=rho_delta, dgpi=dgpi, etak=etak, adotoa=adotoa)
    # Σρπ̇ with the MG shear (ISiTGR step 3).
    pidot_sum = jnp.zeros_like(sigma)
    if not no_phot_multpoles:
        E2 = y[cfg.pol_base] if cfg.include_pol and cfg.n_pol >= 1 else 0.0
        polter = pig / 10.0 + 9.0 / 15.0 * E2
        pigdot = _denlk(k, 2) * qg - opacity * (pig - polter) + (8.0 / 15.0) * k * sigma
        if cfg.lmaxg > 2:
            pigdot = pigdot - _denlk2(k, 2) * y[G_IX + 3]
        pidot_sum = pidot_sum + grhog * pigdot
    if not no_nu_multpoles:
        pirdot = _denlk(k, 2) * qr + (8.0 / 15.0) * k * sigma
        if cfg.lmaxnr > 2:
            pirdot = pirdot - _denlk2(k, 2) * y[cfg.r_ix + 3]
        pidot_sum = pidot_sum + grhor * pirdot
    if cfg.nqmax > 0:
        # l = 2 rows depend on σ only (z enters l = 0).
        nu_dots = jax.vmap(
            lambda yb, amb: nu_multipole_derivs(yb, k=k, am=amb, nu_q=nu_q, z=jnp.zeros_like(sigma),
                                                sigma=sigma, cothxor=cothxor, lmaxnu=cfg.lmaxnu)
        )(y_nu, am)
        pinudot_raw = nu_pinudot_raw(y_nu, nu_dots, am, nu_q, kernel, adotoa)  # (n_eig,)
        rhonudot = thermal_nu_drho(am, adotoa)
        pidot_sum = pidot_sum + jnp.sum(grhormass_t * (pinudot_raw - rhonudot * pinu_raw / rhonu))
    etak_dot = mg_jax.etak_dot_mg(
        k=k, mu=mu, mudot=mudot, sig=sig, sigdot=sigdot, adotoa=adotoa, Hdot=Hdot,
        rho_delta=rho_delta, dgq=dgq, dgpi=dgpi, dgpi_3wplus1=dgpi_3wp1,
        pidot_sum=pidot_sum, sigma=sigma, rho_plus_p=rho_plus_p)
    z = sigma - 3.0 * etak_dot / k2
    return (jnp.where(use_mg, z, z_gr), jnp.where(use_mg, sigma, sigma_gr),
            jnp.where(use_mg, etak_dot, etak_dot_gr))


def derivs(
    tau,
    y,
    k,
    params,
    thermo: dict,
    *,
    tight_coupling: bool = True,
    no_phot_multpoles: bool = False,
    no_nu_multpoles: bool = False,
    high_ktau_neutrino_approx: bool = False,
    cfg: HierarchyConfig | None = None,
    thermo_at_tau: tuple | None = None,
    lmax_live: tuple | None = None,
    bg_at_tau=None,
    freeze_metric: bool = False,
):
    """CAMB ``derivs`` core; returns ``(ayprime, pig, pigdot)``.

    ``freeze_metric``: stop gradients through z and sigma (values unchanged).

    ``bg_at_tau``: ``background_at(tau, a, params, cfg)`` precomputed for this
    tau (the y-independent densities, ``adotoa``, ...). Fixed-step integrators
    tabulate it per stage outside the time loop, which takes the pow / exp /
    sqrt / divisions off the sequential critical path.

    Optional ``thermo_at_tau=(a, cs2, opacity, dopacity)`` skips table interp
    (used by TimeSteps-interval RK4 with endpoint lerp).

    ``lmax_live=(lmaxg, lmaxgpol, lmaxnr)`` (traced ints, each <= the static
    ``cfg`` size and >= 3) truncates the photon / polarization / massless-ν
    hierarchies per mode with CAMB's closure at those ℓ; multipoles above are
    frozen at zero. This is how CAMB's q-dependent hierarchy sizes are run in
    one vmap over k on a padded layout.
    """
    cfg = cfg or HierarchyConfig()
    p = _params_floats(params)
    if cfg.curved and cfg.mg:
        raise NotImplementedError("μ–Σ modified gravity with curvature (the ISiTGR port is flat)")
    # k: the grid wavenumber q; with curvature the equations use √(q² − K) and Kf(ℓ)
    k, k2, kfd = curvature_k(k, p, cfg)
    H0 = p["H0_mpc"]
    y = jnp.asarray(y, dtype=jnp.float64)
    # One fused concatenate instead of ~25 dynamic-update-slice kernels.
    ayprime = VecBuilder(cfg.n_eq)

    if thermo_at_tau is not None:
        a, cs2, opacity, dopacity = thermo_at_tau
        if not tight_coupling:
            dopacity = jnp.asarray(0.0, dtype=jnp.float64)
    elif tight_coupling:
        a, cs2, opacity, dopacity = thermo_values(tau, thermo, want_dopacity=True)
    else:
        a, cs2, opacity = thermo_values(tau, thermo, want_dopacity=False)
        dopacity = 0.0
    if bg_at_tau is None:
        bg_at_tau = background_at(tau, a, params, cfg)
    (cothxor, grhog, grhor, grhob, grhoc, grhov, w_de, grhonu_t, gpnu_t,
     grho, gpres_noDE, adotoa, photbar) = (bg_at_tau[i] for i in range(len(BG_KEYS)))
    fac = 3.0 * H0 * H0

    etak = y[IX_ETAK]
    clxc = y[IX_CLXC]
    clxb = y[IX_CLXB]
    vb = y[IX_VB]

    # Massive neutrinos: grhormass_t is the density one massless eigenstate
    # would have, so rhonu/pnu (ratios to massless) give the physical values.
    # CAMB's grhonu_t*clxnu reduces to grhormass_t*drhonu, so the raw L012
    # integrals are used directly without dividing by rhonu.
    if cfg.nqmax > 0:
        from gpucosmo.jax.eb.massive_nu import (
            nu_integrate_l012,
            nu_momentum_grid,
        )

        from gpucosmo.jax.eb.massive_nu import nu_block, nu_eig_arrays

        nu_q_np, nu_kernel_np = nu_momentum_grid(cfg.nqmax)
        nu_q = jnp.asarray(nu_q_np)
        nu_kernel = jnp.asarray(nu_kernel_np)
        am_e, om_e = nu_eig_arrays(p, cfg.n_nu_eig)
        am = a * am_e  # (n_eig,)
        grhormass_t = fac * om_e / (a * a)
        y_nu = nu_block(y, cfg)  # (n_eig, nqmax, lmaxnu + 1)
        drhonu, fnu, _dpnu, pinu = jax.vmap(nu_integrate_l012, in_axes=(0, 0, None, None))(
            y_nu, am, nu_q, nu_kernel)
        dgrho_nu = jnp.sum(grhormass_t * drhonu)
        dgq_nu = jnp.sum(grhormass_t * fnu)
        dgpi_nu = jnp.sum(grhormass_t * pinu)
    else:
        dgrho_nu = jnp.asarray(0.0, dtype=jnp.float64)
        dgq_nu = jnp.asarray(0.0, dtype=jnp.float64)
        dgpi_nu = jnp.asarray(0.0, dtype=jnp.float64)

    dgrho_matter = grhob * clxb + grhoc * clxc
    dgq = grhob * vb

    mg_on = cfg.mg and not tight_coupling
    if mg_on:
        from gpucosmo.jax.eb import mg_jax

        use_mg = mg_jax.mg_active(a)
        clx_rsa_mg, q_rsa_mg = mg_jax.rsa_radiation_mg(k, grhoc, grhob, clxc, clxb)
    # The radiation-streaming closures' metric estimate takes the non-radiation density: CAMB
    # adds the massive ν to dgrho_matter (MassiveNuVars) before them. Matter alone left out the
    # non-relativistic states' δρ once the closures switch on (z ≲ 500 at k = 0.03/Mpc): P(k)
    # −1.9e-3 at Σm_ν = 1 eV in one state, ∝ m². ISiTGR's μ–Σ closure is matter-only.
    dgrho_rsa = dgrho_matter + dgrho_nu
    if no_nu_multpoles:
        z_rsa = (0.5 * dgrho_rsa / k + etak) / adotoa
        dz = -adotoa * z_rsa - 0.5 * dgrho_rsa / k
        clxr = -4.0 * dz / k
        qr = -4.0 / 3.0 * z_rsa
        pir = 0.0
        if mg_on:
            clxr = jnp.where(use_mg, clx_rsa_mg, clxr)
            qr = jnp.where(use_mg, q_rsa_mg, qr)
    else:
        clxr = y[cfg.r_ix]
        qr = y[cfg.r_ix + 1]
        pir = y[cfg.r_ix + 2] if cfg.lmaxnr >= 2 else 0.0

    pig = jnp.asarray(0.0, dtype=jnp.float64)
    if no_phot_multpoles:
        if not no_nu_multpoles:
            z_rsa = (0.5 * dgrho_rsa / k + etak) / adotoa
            dz = -adotoa * z_rsa - 0.5 * dgrho_rsa / k
            clxg = -4.0 * dz / k - 4.0 / k * opacity * (vb + z_rsa)
            qg = -4.0 / 3.0 * z_rsa
        else:
            clxg = clxr - 4.0 / k * opacity * (vb + z_rsa)
            qg = qr
        if mg_on:
            clxg = jnp.where(use_mg, clx_rsa_mg, clxg)
            qg = jnp.where(use_mg, q_rsa_mg, qg)
    else:
        clxg = y[G_IX]
        qg = y[G_IX + 1]
        if (not tight_coupling) and cfg.lmaxg >= 2:
            pig = y[G_IX + 2]

    dgrho = dgrho_matter + grhog * clxg + grhor * clxr + dgrho_nu
    dgq = dgq + grhog * qg + grhor * qr + dgq_nu

    # CAMB DarkEnergyPPF: add dgrho_de / dgq_de before z, sigma (equations.f90:2422-2435).
    if cfg.use_de_ppf:
        Gamma = y[cfg.w_ix]
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
        ayprime = ayprime.at[cfg.w_ix].set(Gammadot)

    pb43 = (4.0 / 3.0) * photbar

    z = (0.5 * dgrho / k + etak) / adotoa
    sigma = (z + 1.5 * dgq / k2) / kf(kfd, 1)
    etak_dot = 0.5 * dgq + (p["curv"] * z if cfg.curved else 0.0)
    if mg_on:
        z, sigma, etak_dot = _mg_constraints(
            use_mg=use_mg, z_gr=z, sigma_gr=sigma, etak_dot_gr=etak_dot,
            y=y, k=k, a=a, p=p, cfg=cfg, etak=etak, adotoa=adotoa, grho=grho,
            grhov=grhov, w_de=w_de, gpres_noDE=gpres_noDE, grhog=grhog, grhor=grhor,
            grhob=grhob, grhoc=grhoc, grhonu_t=grhonu_t, gpnu_t=gpnu_t,
            dgrho=dgrho, dgq=dgq, pig=pig, pir=pir, qg=qg, qr=qr, opacity=opacity,
            cothxor=cothxor,
            no_phot_multpoles=no_phot_multpoles, no_nu_multpoles=no_nu_multpoles,
        )
    # Materialize the per-mode scalars every multipole equation reads: fused
    # into the output concatenate, XLA recomputes them per element (a long
    # serial fp64 chain per GPU thread).
    z, sigma, dgq = jax.lax.optimization_barrier((z, sigma, dgq))
    if freeze_metric:
        # Values unchanged; derivatives through the metric feedback (rate ~aH)
        # vanish, so a Jacobian of the hierarchy rows is banded in ell (see
        # transfer_evolve_jax._imex_banded_step_fn).
        z, sigma = jax.lax.stop_gradient(z), jax.lax.stop_gradient(sigma)
    ayprime = ayprime.at[IX_ETAK].set(etak_dot)
    ayprime = ayprime.at[IX_CLXC].set(-k * z)
    ayprime = ayprime.at[IX_CLXB].set(-k * (z + vb))

    clxgdot = -k * (4.0 / 3.0 * z + qg)
    clxbdot = ayprime[IX_CLXB]
    delta_p_b = cs2 * clxb
    pigdot = jnp.asarray(0.0, dtype=jnp.float64)

    if tight_coupling:
        gpres = gpres_noDE + w_de * grhov
        adotdota = (adotoa * adotoa - gpres) / 2.0
        op = jnp.maximum(opacity, 1e-30)
        pig = 32.0 / 45.0 / op * k * (sigma + vb)
        slip = -(
            2.0 * adotoa / (1.0 + pb43) + dopacity / op
        ) * (vb - 0.75 * qg) + (
            -adotdota * vb
            - k / 2.0 * adotoa * clxg
            + k * (cs2 * clxbdot - clxgdot / 4.0)
        ) / (opacity * (1.0 + pb43) + 1e-30)

        if cfg.second_order_tc:
            dgs = grhog * pig + grhor * pir + dgpi_nu
            sigmadot = -2.0 * adotoa * sigma - dgs / k + etak
            qgdot_tc = k * (clxg / 4.0 - pig / 2.0) + opacity * slip
            pig = (
                32.0 / 45.0 / op * k * (sigma + 0.75 * qg) * (1.0 + dopacity * 11.0 / 6.0 / op**2)
                + (32.0 / 45.0 / op**2) * k * (sigmadot + 0.75 * qgdot_tc) * (-11.0 / 6.0)
            )
            pigdot = (
                -(32.0 / 45.0) * (dopacity / op**2) * k * (sigma + 0.75 * qg)
                * (1.0 + dopacity * 11.0 / 6.0 / op**2)
                + (32.0 / 45.0 / op) * k * (sigmadot + 0.75 * qgdot_tc)
                * (1.0 + (11.0 / 6.0) * (dopacity / op**2))
            )

        vbdot = (
            -adotoa * vb + cs2 * k * clxb + 0.25 * k * pb43 * (clxg - 2.0 * kf(kfd, 1) * pig)
        ) / (1.0 + pb43)
        vbdot = vbdot + pb43 / (1.0 + pb43) * slip
    else:
        vbdot = -adotoa * vb + k * delta_p_b - photbar * opacity * (
            4.0 / 3.0 * vb - qg
        )

    vbdot, pig = jax.lax.optimization_barrier((vbdot, pig))
    ayprime = ayprime.at[IX_VB].set(vbdot)

    if not no_phot_multpoles:
        ayprime = ayprime.at[G_IX].set(clxgdot)
        qgdot = (
            (4.0 / 3.0) * (-vbdot - adotoa * vb + delta_p_b * k) / jnp.maximum(pb43, 1e-30)
            + _denlk(k, 1) * clxg
            - _denlk2(k, 1) * kf(kfd, 1) * pig
        )
        ayprime = ayprime.at[G_IX + 1].set(qgdot)

        if not tight_coupling:
            E2 = y[cfg.pol_base] if cfg.include_pol and cfg.n_pol >= 1 else 0.0
            polter = pig / 10.0 + 9.0 / 15.0 * E2
            if cfg.lmaxg > 2:
                pigdot = (
                    _denlk(k, 2) * qg
                    - _denlk2(k, 2) * kf(kfd, 2) * y[G_IX + 3]
                    - opacity * (pig - polter)
                    + (8.0 / 15.0) * k * sigma
                )
                ayprime = ayprime.at[G_IX + 2].set(pigdot)
                if cfg.lmaxg > 3:
                    # Vectorized photon multipoles ℓ=3..lmaxg-1
                    ls = jnp.arange(3, cfg.lmaxg, dtype=jnp.float64)
                    den = 2.0 * ls + 1.0
                    y_m = y[G_IX + 2 : G_IX + cfg.lmaxg - 1]
                    y_c = y[G_IX + 3 : G_IX + cfg.lmaxg]
                    y_p = y[G_IX + 4 : G_IX + cfg.lmaxg + 1]
                    dots = ((k * ls / den) * y_m - (k * (ls + 1.0) / den) * kf(kfd, ls) * y_p
                            - opacity * y_c)
                    ayprime = ayprime.at[G_IX + 3 : G_IX + cfg.lmaxg].set(dots)
                ix = G_IX + cfg.lmaxg
                ayprime = ayprime.at[ix].set(
                    k * y[ix - 1]
                    - (cfg.lmaxg + 1) * cothxor * y[ix]
                    - opacity * y[ix]
                )
            elif cfg.lmaxg >= 2:
                ayprime = ayprime.at[G_IX + 2].set(
                    _denlk(k, 2) * qg
                    - opacity * (pig - polter)
                    + (8.0 / 15.0) * k * sigma
                )

            if cfg.include_pol and cfg.n_pol >= 1:
                ix = cfg.pol_base
                if cfg.lmaxgpol > 2:
                    ayprime = ayprime.at[ix].set(
                        -opacity * (y[ix] - polter) - k / 3.0 * y[ix + 1]
                    )
                    if cfg.lmaxgpol > 3:
                        ls = jnp.arange(3, cfg.lmaxgpol, dtype=jnp.float64)
                        den = 2.0 * ls + 1.0
                        # E_ℓ at pol_base+(ℓ-2); interior ℓ=3..lmaxgpol-1
                        j0 = cfg.pol_base + 1  # E3
                        j_end = cfg.pol_base + (cfg.lmaxgpol - 2)  # exclusive end of E_{lmax-1}
                        y_m = y[j0 - 1 : j_end - 1]
                        y_c = y[j0:j_end]
                        y_p = y[j0 + 1 : j_end + 1]
                        polfac = ((ls + 3.0) * (ls - 1.0) / (ls + 1.0)) * k / den * kf(kfd, ls)
                        dots = -opacity * y_c + (k * ls / den) * y_m - polfac * y_p
                        ayprime = ayprime.at[j0:j_end].set(dots)
                    jx = cfg.pol_base + (cfg.lmaxgpol - 2)
                    ayprime = ayprime.at[jx].set(
                        -opacity * y[jx]
                        + k * cfg.poltruncfac * y[jx - 1]
                        - (cfg.lmaxgpol + 3) * cothxor * y[jx]
                    )
                else:
                    ayprime = ayprime.at[ix].set(-opacity * (y[ix] - polter))

    if not no_nu_multpoles:
        clxrdot = -k * (4.0 / 3.0 * z + qr)
        ayprime = ayprime.at[cfg.r_ix].set(clxrdot)
        ayprime = ayprime.at[cfg.r_ix + 1].set(
            _denlk(k, 1) * clxr - _denlk2(k, 1) * kf(kfd, 1) * pir
        )
        # A traced flag selects the closure per step without a second derivs
        # branch: evaluate the full ν hierarchy, then override ℓ=2 and freeze
        # ℓ≥3 (their derivatives are left at zero in the static branch too).
        hk_traced = not isinstance(high_ktau_neutrino_approx, bool)
        if hk_traced:
            ayprime_hk_l2 = -3.0 * pir * cothxor - clxrdot
        if high_ktau_neutrino_approx is True:
            ayprime = ayprime.at[cfg.r_ix + 2].set(-3.0 * pir * cothxor - clxrdot)
        else:
            if cfg.lmaxnr > 2:
                ayprime = ayprime.at[cfg.r_ix + 2].set(
                    _denlk(k, 2) * qr
                    - _denlk2(k, 2) * kf(kfd, 2) * y[cfg.r_ix + 3]
                    + (8.0 / 15.0) * k * sigma
                )
                if cfg.lmaxnr > 3:
                    ls = jnp.arange(3, cfg.lmaxnr, dtype=jnp.float64)
                    den = 2.0 * ls + 1.0
                    y_m = y[cfg.r_ix + 2 : cfg.r_ix + cfg.lmaxnr - 1]
                    y_c = y[cfg.r_ix + 3 : cfg.r_ix + cfg.lmaxnr]
                    y_p = y[cfg.r_ix + 4 : cfg.r_ix + cfg.lmaxnr + 1]
                    dots = (k * ls / den) * y_m - (k * (ls + 1.0) / den) * kf(kfd, ls) * y_p
                    ayprime = ayprime.at[cfg.r_ix + 3 : cfg.r_ix + cfg.lmaxnr].set(dots)
                ix = cfg.r_ix + cfg.lmaxnr
                ayprime = ayprime.at[ix].set(
                    k * y[ix - 1] - (cfg.lmaxnr + 1) * cothxor * y[ix]
                )
            elif cfg.lmaxnr >= 2:
                ayprime = ayprime.at[cfg.r_ix + 2].set(
                    _denlk(k, 2) * qr + (8.0 / 15.0) * k * sigma
                )
            if hk_traced and cfg.lmaxnr >= 2:
                hk = jnp.asarray(high_ktau_neutrino_approx)
                lo, hi = cfg.r_ix + 2, cfg.r_ix + cfg.lmaxnr + 1
                blk = ayprime[lo:hi]
                blk_hk = jnp.zeros_like(blk).at[0].set(ayprime_hk_l2)
                ayprime = ayprime.at[lo:hi].set(jnp.where(hk, blk_hk, blk))

    if lmax_live is not None:
        ayprime = _masked_hierarchy_derivs(
            ayprime, y, k, cfg, lmax_live, kfd=kfd,
            opacity=opacity, cothxor=cothxor,
            photons=(not tight_coupling) and (not no_phot_multpoles),
            nu=not no_nu_multpoles,
            high_ktau=high_ktau_neutrino_approx,
        )

    if cfg.nqmax > 0:
        from gpucosmo.jax.eb.massive_nu import nu_multipole_derivs

        kf_nu = None if kfd is None else kf(kfd, jnp.arange(cfg.lmaxnu + 1))
        nu_dots = jax.vmap(
            lambda yb, amb: nu_multipole_derivs(yb, k=k, am=amb, nu_q=nu_q, z=z, sigma=sigma,
                                                cothxor=cothxor, lmaxnu=cfg.lmaxnu, kf=kf_nu)
        )(y_nu, am)
        ayprime = ayprime.at[cfg.nu_ix : cfg.nu_ix + cfg.n_nu_massive].set(
            nu_dots.reshape(-1)
        )

    return ayprime.build(), pig, pigdot


def transfer_tot_from_y(y, a, params, cfg: HierarchyConfig | None = None):
    """δρ_m / ρ_m before /k² (Transfer_nonu = Transfer_tot for massless ν)."""
    del cfg  # layout unused; indices fixed
    p = _params_floats(params)
    fac = 3.0 * p["H0_mpc"] ** 2
    grhob = fac * p["Omega_b"] / a
    grhoc = fac * p["Omega_c"] / a
    return (grhob * y[IX_CLXB] + grhoc * y[IX_CLXC]) / (grhob + grhoc)


__all__ = [
    "C_GAMMA_PPF",
    "G_IX",
    "HierarchyConfig",
    "IX_CLXB",
    "IX_CLXC",
    "IX_ETAK",
    "IX_VB",
    "PPF_QUASI_STATIC_CKH2",
    "derivs",
    "background_at",
    "BG_KEYS",
    "init_after_tight_coupling",
    "initial_adiabatic",
    "ppf_diff_rhopi_add_term",
    "ppf_perturbed_stress_energy",
    "thermo_values",
    "pack_thermo_values",
    "transfer_tot_from_y",
]
