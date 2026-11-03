# This file is part of GPUCosmo. Copyright (C) 2025-2026 Leonel Medina-Varela.
# SPDX-License-Identifier: GPL-3.0-or-later
# Derived in part from the following, translated to JAX and modified for GPUCosmo:
# - CAMB (https://github.com/cmbant/CAMB), Copyright 2000-2025 Antony Lewis and Anthony Challinor.
#   Licence: LICENSES/CAMB.txt.
# - ISiTGR (https://github.com/mishakb/ISiTGR), Dossett, Ishak, Garcia-Quintero et al.,
#   built on CAMB, Copyright 2000-2018 Antony Lewis and Anthony Challinor.
#   Licence: LICENSES/ISiTGR.txt.
# See NOTICE.md.
"""Linear matter and velocity power spectra from the CMB-grade EB solver.

What a full-shape model needs at the survey's effective redshifts, with the same
physics as the CMB path (w0wa PPF dark energy, massive ν, μ–Σ modified gravity):

* CAMB ``Transfer_nonu``  δ_cb = (ρ_b δ_b + ρ_c δ_c)/(ρ_b + ρ_c) (synchronous);
* CAMB ``Transfer_Newt_vel_cdm``  v_c = −kσ/ℋ and ``Transfer_Newt_vel_baryon``
  v_b = −k(v_b + σ)/ℋ, with σ the (MG) shear;

so ``P_xy(k) = 2π²/k³ · A_s (k/k_p)^{n_s−1} · T_x T_y`` [Mpc³] are CAMB's
``get_linear_matter_power_spectrum(var1, var2, hubble_units=False,
k_hunit=False)``. These are the ``Pk_grid`` pairs the official DESI DR1
likelihood (``reptvelocileptors``) requests from CAMB/ISiTGR.

Each k is evolved with ``evolve_scalar_k_jax(method="segmented")`` and read out
at the requested redshifts only.
"""

from __future__ import annotations

import dataclasses
import os

import jax
import jax.numpy as jnp
import numpy as np

from gpucosmo.jax.eb.cls_jax import (
    CLS_EVOLVE_KDT_TARGET,
    CLS_EVOLVE_METHOD,
    CLS_HIERARCHY,
    CLS_INTEGRATE_ON_OUTPUT,
    CLS_N_STEPS,
    CLS_N_SUB,
    _cast_floats,
    _params_dict_cls,
    _resolve_sources_dtype,
    recfast_thermo_batched,
)
from gpucosmo.jax.eb.evolve_jax import downsample_thermo_for_evolve, evolve_scalar_k_jax
from gpucosmo.jax.eb.evolve_segmented_jax import DEFAULT_SEGMENTED
from gpucosmo.jax.eb.hierarchy_jax import IX_CLXB, IX_CLXC, IX_VB

jax.config.update("jax_enable_x64", True)

# k grid [1/Mpc]. The DESI likelihood's BAO filter (cosmoprimo peakaverage)
# places spline knots at every k > 1 h/Mpc, so P is needed beyond the PT range:
# truncating CAMB's input at 1/Mpc shifted log L by up to 0.07, at 2/Mpc by
# 5e-4 (scripts/validate_desi_fs_vs_official.py --truncate-kmax).
LINPK_K_MIN = 1.0e-4
LINPK_K_MAX = 2.0
# 300 log points: the BAO wiggles (k r_s ≈ 30 rad per unit ln k at k = 0.2/Mpc) need it for the
# consumers' cubic splines; with 150 the DESI DR1 FS + BAO log L was +0.031…+0.035 off a
# 450-point grid (300: ≤ 0.001), the spline onto other grids 1.7e-3.
LINPK_N_K = 300
# The segmented integrator's step budgets are sized for the CMB grid (k ≤ 0.45);
# its stability limit grows linearly with the budget (k ≈ 0.6/Mpc per unit), so
# higher-k modes run with scaled budgets: (k_upper, multiplier) bands. The limit
# per unit budget moves with the cosmology (≈ 0.6/Mpc at Planck ω_b, 0.47/Mpc at
# ω_b = 0.019, where 4× diverged at k = 1.87 with P ~ 1e13); each band ends ≥ 30%
# below the worst case.
LINPK_BUDGET_BANDS = ((0.35, 1), (0.7, 2), (1.4, 4), (np.inf, 6))
LINPK_SINGLE_BAND = ((np.inf, 6),)
# Massive-ν multipoles: the CMB hierarchy's lmaxnu = 10 leaves P_cb 2.4e-3 high
# at k = 0.26/Mpc (truncation reflects free-streaming power back); 25 brings
# P_cb to ≤ 0.9e-3 of CAMB (AccuracyBoost 2) for 3e-3 ≤ k ≤ 0.5/Mpc.
LINPK_HIERARCHY = dataclasses.replace(CLS_HIERARCHY, lmaxnu=25)
# With several massive eigenstates (DES: three degenerate, no massless species) all of N_eff
# sits in the massive hierarchy, whose ℓ = lmaxnu truncation reflects power at high kτ in the
# radiation era (CAMB evolves relativistic massive ν in the massless hierarchy, with its high-kτ
# closure): 25 left P(k) −4e-4 at k = 0.35 and −1.8e-3 at 2/Mpc vs CAMB; 50 → ≤ 3.7e-4 (k ≤ 2).
LINPK_LMAXNU_MULTI = 50
# Precision of the per-k evolution and transfer read-out ("auto" = float32 on
# accelerators, float64 on CPU; as ``cls_jax.CLS_SOURCES_DTYPE``). Recfast, the
# thermo tables and the P(k) assembly stay float64.
LINPK_EVOLVE_DTYPE = os.environ.get("GPUCOSMO_LINPK_DTYPE", "auto")
# Segmented-integrator settings of the P(k) modes: the output times are knots and only the
# states there are kept (store "outputs": memory n_z × n_eq per mode instead of every knot,
# ~150 MB per mode at k = 15/Mpc), the state update is Kahan-compensated (as the default; the
# float32 rounding of ~10⁵ steps at k ≈ 10/Mpc otherwise accumulates to 5e-3 in P), and photons
# stay out of the streaming approximation to k = 0.1/Mpc: with the CMB's 0.03 the modes just
# above it came out 1e-3 low vs CAMB (AccuracyBoost 3; 1.5e-4 with 0.06, within the ±3e-4
# BAO-phase residuals with 0.1).
LINPK_SEGMENTED = dataclasses.replace(DEFAULT_SEGMENTED, store="outputs", compensated=True, phot_rsa_kmin=0.1)
# Modes past LINPK_K_MAX for the "pk" block's ``pk_kmax`` (``hik_k_nodes``): P(k) has no
# features there, so a sparse log grid (Δ ln k ≤ PK_HIK_DLNK) is evolved and interpolated as
# the ratio to the Eisenstein–Hu shape (``growth_jax.extended_lnpk``), in one band with the step
# budget × 3.2 k [Mpc] (RK4 stability needs ≥ k / 0.47 at ω_b = 0.019, see above). float32 is
# within 7.3e-4 of CAMB (AccuracyBoost 2, high_precision) to k = 15 but its rounding grows past
# k ≈ 20 (7e-3 at 50, float64 smooth −1e-3): modes beyond PK_HIK_F32_KMAX evolve in float64
# (~3.5× slower on the RTX 3070). RTX 3070 Laptop, float32, 11 nodes to k = 15: 3.8 s for one
# cosmology, 5.2 s for 16.
PK_HIK_DLNK = 0.2
PK_HIK_F32_KMAX = 20.0


def _hik_mult(k: float) -> int:
    return int(4 * np.ceil(3.2 * float(k) / 4))


def hik_k_nodes(k_max: float) -> np.ndarray:
    """Static modes (LINPK_K_MAX, k_max] [1/Mpc] evolved for the "pk" block past the solver grid."""
    if float(k_max) <= LINPK_K_MAX:
        return np.zeros(0)
    n = int(np.ceil(np.log(float(k_max) / LINPK_K_MAX) / PK_HIK_DLNK))
    return np.geomspace(LINPK_K_MAX, float(k_max), n + 1)[1:]


def hik_bands(k_max: float) -> tuple:
    """Budget bands ((k_upper, step multiplier, precision), ...) of ``hik_k_nodes(k_max)``."""
    k = hik_k_nodes(k_max)
    if not k.size:
        return ()
    lo = k[k <= PK_HIK_F32_KMAX]
    out = ((PK_HIK_F32_KMAX, _hik_mult(lo[-1]), None),) if lo.size else ()
    if k[-1] > PK_HIK_F32_KMAX:
        out += ((np.inf, _hik_mult(k[-1]), "float64"),)
    return out


def with_hik(k_max: float, budget_bands=None) -> tuple[np.ndarray, tuple]:
    """(k grid, budget bands) of ``linear_cb_velocity_spectra`` for the solver grid plus
    ``hik_k_nodes(k_max)`` (the grid's last band capped at LINPK_K_MAX)."""
    base = LINPK_BUDGET_BANDS if budget_bands is None else budget_bands
    base = tuple(b for b in base if b[0] < LINPK_K_MAX) + (
        (LINPK_K_MAX,) + tuple(next(b for b in base if b[0] >= LINPK_K_MAX)[1:]),)
    return np.concatenate([linpk_k_grid(), hik_k_nodes(k_max)]), base + hik_bands(k_max)


def linpk_k_grid(k_min: float | None = None, k_max: float | None = None,
                 n_k: int | None = None) -> np.ndarray:
    # Uniform in log k. A ~2.5x sparser sampling above 0.5/Mpc (where the
    # expensive banded modes live) moved the DESI log L by ~0.2: the BAO filter's
    # spline knots at k > 1 h/Mpc are sensitive to P(k) interpolation error.
    return np.geomspace(LINPK_K_MIN if k_min is None else k_min, LINPK_K_MAX if k_max is None else k_max,
                        LINPK_N_K if n_k is None else n_k)


def _tau_at_z(thermo, z):
    return jnp.interp(1.0 / (1.0 + jnp.asarray(z)), thermo["a"], thermo["tau_mpc"])


def _transfers_at(y, k, tau, params, thermo, cfg):
    """(δ_cb, v_c, v_b, δ_m) from the evolved state (CAMB output-transfer definitions;
    δ_m is ``Transfer_tot`` without radiation, i.e. cb + massive ν)."""
    from gpucosmo.jax.eb.sources_jax import _background_and_stress

    q = _background_and_stress(y, k, tau, params, thermo, cfg)
    k = q["k"]  # √(q² − K) with curvature (CAMB's output k); the grid k when flat
    a = q["a"]
    fac = 3.0 * params["H0_mpc"] ** 2
    grhob = fac * params["Omega_b"] / a
    grhoc = fac * params["Omega_c"] / a
    d_cb = (grhob * y[:, IX_CLXB] + grhoc * y[:, IX_CLXC]) / (grhob + grhoc)
    sigma = q["sigma"]
    if cfg.mg:
        from gpucosmo.jax.eb import mg_jax

        Hdot = -(q["grho"] + 3.0 * q["gpres"]) / 6.0
        mu, _, sig, _ = mg_jax.mu_sigma(k=k, a=a, adotoa=q["adotoa"], Hdot=Hdot,
                                        grhov=q["grhov"], grho=q["grho"], params=params)
        rho_delta = q["dgrho"] + 3.0 * q["adotoa"] * q["dgq"] / k
        _, _, sigma_mg = mg_jax.newtonian_potentials(
            k=k, mu=mu, sig=sig, rho_delta=rho_delta, dgpi=q["dgpi"], etak=q["etak"],
            adotoa=q["adotoa"])
        sigma = jnp.where(mg_jax.mg_active(a), sigma_mg, sigma)
    v_c = -k * sigma / q["adotoa"]
    v_b = -k * (y[:, IX_VB] + sigma) / q["adotoa"]
    return d_cb, v_c, v_b, q["delta_m"]


def linear_cb_velocity_spectra(
    *,
    ombh2,
    omch2,
    h,
    ns,
    As,
    w0,
    wa,
    z_out,  # static (host) redshifts
    tau_reio=0.0544,
    mu0=None,
    Sigma0=None,
    k_mpc: np.ndarray | None = None,
    k_chunk: int = 32,
    cfg=None,
    seg_cfg=None,
    budget_bands=None,
    compute_dtype=None,
    mnu=None,
    num_massive_nu: int | None = None,
    ext: dict | None = None,
    nu_model=None,
) -> dict:
    """Linear P_cb, P_{v_c v_c}, P_{v_b v_c}, P_{v_b v_b} [Mpc³] at ``z_out``.

    Returns a dict with ``k`` [1/Mpc] (n_k,) and arrays of shape (n_z, n_k):
    ``pk_cb``, ``pk_vc_vc``, ``pk_vb_vc``, ``pk_vb_vb`` and ``pk_m`` (cb + massive ν,
    what CAMB's σ8 uses). ``budget_bands``: ((k_upper, step multiplier[, precision]), ...),
    default ``LINPK_BUDGET_BANDS``; every band compiles its own solver copy, and
    ``LINPK_SINGLE_BAND`` evolves all modes with the largest budget in one batch (a band's
    precision overrides ``compute_dtype``).
    τ_reio only enters
    through the (negligible) reionization opacity at these redshifts.
    ``compute_dtype`` (default ``LINPK_EVOLVE_DTYPE``): precision of the per-k evolution
    (float32 traces it with 64-bit types disabled; inputs cast down, transfers back up).
    ``mnu`` (may be traced) with ``num_massive_nu`` degenerate eigenstates: Σm_ν [eV]; default the
    solver's fixed 0.06 eV in one eigenstate. ``ext`` (``cls_jax.ext_params``) and ``nu_model``
    (``massive_nu.NeutrinoModel``) generalize this (Σm_ν, N_eff, sterile m_eff; mutually exclusive
    with ``mnu``/``num_massive_nu``).
    """
    st, cfg, order, k_np = linpk_setup(ombh2=ombh2, omch2=omch2, h=h, ns=ns, As=As, w0=w0, wa=wa, z_out=z_out,
                                       tau_reio=tau_reio, mu0=mu0, Sigma0=Sigma0, k_mpc=k_mpc, cfg=cfg,
                                       mnu=mnu, num_massive_nu=num_massive_nu, ext=ext, nu_model=nu_model)
    tr = jnp.concatenate([linpk_band_transfers(st, k_np[sel], mult, cfg=cfg, seg_cfg=seg_cfg, k_chunk=k_chunk,
                                               compute_dtype=dt or compute_dtype)
                          for sel, mult, dt in linpk_bands(k_np, budget_bands)], axis=0)
    return linpk_spectra(st, tr, order, k_np, As, ns)


def linpk_setup(*, ombh2, omch2, h, ns, As, w0, wa, z_out, tau_reio=0.0544, mu0=None, Sigma0=None,
                k_mpc=None, cfg=None, mnu=None, num_massive_nu=None, ext=None, nu_model=None):
    """Per-cosmology inputs of the per-k evolution: ``(state, cfg, order, k_np)`` with state =
    {params, thermo, thermo_ev, tau_out} (arrays), ``order`` the static permutation back to
    the caller's redshift order and ``k_np`` the static k grid."""
    from gpucosmo.jax.eb.cls_jax import (
        _require_curvature_enabled,
        ext_params,
        hierarchy_for_model,
        is_flat,
        nu_model_for,
    )

    if (mnu is not None or num_massive_nu is not None) and (ext is not None or nu_model is not None):
        raise ValueError("give mnu/num_massive_nu or ext/nu_model, not both")
    model = nu_model_for(nu_model, num_massive_nu)
    e = ext_params(ext if ext is not None else ({} if mnu is None else {"mnu": mnu}))
    _require_curvature_enabled(e["omk"])
    cfg = hierarchy_for_model(cfg or LINPK_HIERARCHY, model, e)
    if not is_flat(e["omk"]):  # P at the true k = √(q² − K): modes evolved at q = √(k² + K)
        if mu0 is not None or Sigma0 is not None:
            raise NotImplementedError("μ–Σ modified gravity with curvature (the ISiTGR port is flat)")
        cfg = dataclasses.replace(cfg, curved=True)
    # All of N_eff in massive blocks (several degenerate states, a hierarchy, steriles) needs the
    # longer massive-ν hierarchy (LINPK_LMAXNU_MULTI).
    if (model.n_eig > 1 or model.num_massive > 1) and cfg.lmaxnu < LINPK_LMAXNU_MULTI:
        cfg = dataclasses.replace(cfg, lmaxnu=LINPK_LMAXNU_MULTI)
    params = _params_dict_cls(ombh2, omch2, h, ns, As, tau_reio, w0, wa, **e, nu_model=model)
    if mu0 is not None or Sigma0 is not None:
        cfg = dataclasses.replace(cfg, mg=True)
        params["mu0"] = jnp.asarray(0.0 if mu0 is None else mu0, dtype=jnp.float64)
        params["Sigma0"] = jnp.asarray(0.0 if Sigma0 is None else Sigma0, dtype=jnp.float64)
    k_np = np.asarray(linpk_k_grid() if k_mpc is None else k_mpc, dtype=np.float64)
    # Tables start before every mode does (else a(τ) is clamped through their super-horizon phase).
    thermo = recfast_thermo_batched(ombh2, omch2, h, ns, As, tau_reio, w0, wa, ext=e,
                                    k_max=float(np.max(k_np)), nu_model=model)
    thermo_ev = downsample_thermo_for_evolve(thermo)
    # Static redshifts (survey effective z): evolve to τ ascending ⇔ z descending.
    z_np = np.atleast_1d(np.asarray(z_out, dtype=np.float64))
    desc = np.argsort(-z_np)
    order = np.argsort(desc)
    tau_out = _tau_at_z(thermo, jnp.asarray(z_np[desc]))
    return dict(params=params, thermo=thermo, thermo_ev=thermo_ev, tau_out=tau_out), cfg, order, k_np


def linpk_bands(k_np, budget_bands=None):
    """Static [(selection mask over k_np, step multiplier, precision or None), ...] of the budget bands."""
    out, lo = [], 0.0
    for band in (LINPK_BUDGET_BANDS if budget_bands is None else budget_bands):
        k_hi, mult = band[:2]
        sel = (k_np > lo) & (k_np <= k_hi)
        lo = k_hi
        if sel.any():
            out.append((sel, mult, band[2] if len(band) > 2 else None))
    return out


def linpk_band_transfers(st, kb, mult, *, cfg, seg_cfg=None, k_chunk: int = 32, compute_dtype=None):
    """(n_kb, 4, n_z) transfers (δ_cb, v_c, v_b, δ_m) of the modes ``kb`` (static) with step
    budget × ``mult``, in the evolution precision (float32 region, as the full function)."""
    base = seg_cfg or LINPK_SEGMENTED
    sc = dataclasses.replace(base, n_tc=base.n_tc * mult, n_full=base.n_full * mult, n_mid=base.n_mid * mult,
                             n_rsa=base.n_rsa * mult, n_fine=base.n_fine * mult)
    kb = np.asarray(kb, dtype=np.float64)

    def run(params, thermo_ev, thermo, tau_out):
        def one_k(k):
            if cfg.curved:  # the mode whose true k = √(q² − K) is the requested k (no closed rounding:
                k = jnp.sqrt(k * k + params["curv"])  # ν ≫ 1 at every k of P(k))
            y = evolve_scalar_k_jax(
                k, params, thermo_ev, tau_out, cfg=cfg, n_steps=int(CLS_N_STEPS),
                n_sub=int(CLS_N_SUB), integrate_on_output=CLS_INTEGRATE_ON_OUTPUT,
                kdt_target=CLS_EVOLVE_KDT_TARGET, method=CLS_EVOLVE_METHOD, seg_cfg=sc)
            return jnp.stack(_transfers_at(y, k, tau_out, params, thermo, cfg))  # (4, n_z)

        kj = jnp.asarray(kb)
        if k_chunk and 0 < int(k_chunk) < int(kb.shape[0]):
            return jax.lax.map(one_k, kj, batch_size=int(k_chunk))
        return jax.vmap(one_k)(kj)

    args = (st["params"], st["thermo_ev"], st["thermo"], st["tau_out"])
    cd = _resolve_sources_dtype(LINPK_EVOLVE_DTYPE if compute_dtype is None else compute_dtype)
    if jnp.dtype(cd) == jnp.float32:
        args32 = _cast_floats(args, jnp.float32)
        with jax.enable_x64(False):
            tr = run(*args32)
        return tr.astype(jnp.float64)
    return run(*args)


def linpk_spectra(st, tr, order, k_np, As, ns) -> dict:
    """Spectra from the transfers ``tr`` (n_k, 4, n_z) of all k (``linear_cb_velocity_spectra``'s output)."""
    params = st["params"]
    k_grid = jnp.asarray(k_np, dtype=jnp.float64)
    tr = tr[:, :, order]  # (n_k, 4, n_z) in the caller's z order
    p_r = As * (k_grid / params["k_pivot"]) ** (ns - 1.0)
    norm = (2.0 * jnp.pi**2 / k_grid**3 * p_r)[:, None]
    d_cb, v_c, v_b, d_m = tr[:, 0], tr[:, 1], tr[:, 2], tr[:, 3]
    return {
        "k": k_grid,
        "pk_cb": (norm * d_cb * d_cb).T,
        "pk_vc_vc": (norm * v_c * v_c).T,
        "pk_vb_vc": (norm * v_b * v_c).T,
        "pk_vb_vb": (norm * v_b * v_b).T,
        "pk_m": (norm * d_m * d_m).T,
        "pk_m_vc": (norm * d_m * v_c).T,  # CAMB delta_tot × v_newtonian_cdm (f σ8)
        "Omega_b": params["Omega_b"],
        "Omega_c": params["Omega_c"],
        "Omega_nu_mass": params.get("Omega_nu_mass", 0.0),
    }


__all__ = ["LINPK_BUDGET_BANDS", "LINPK_HIERARCHY", "LINPK_LMAXNU_MULTI", "LINPK_K_MAX", "LINPK_K_MIN", "LINPK_N_K", "LINPK_SEGMENTED",
           "LINPK_SINGLE_BAND", "PK_HIK_DLNK", "PK_HIK_F32_KMAX", "hik_bands", "hik_k_nodes",
           "linear_cb_velocity_spectra", "linpk_band_transfers", "linpk_bands", "linpk_k_grid", "linpk_setup",
           "linpk_spectra", "with_hik"]
