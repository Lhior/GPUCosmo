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
"""jax_eb CMB C_ℓ pipeline — device evolve + hybrid LOS + Weyl Limber.

Integrals follow CAMB ``dpoints`` / ``dlnk``. Thermodynamics run on-device
through ``recfast_jax`` / ``build_thermo_jax`` so gradients reach ``ombh2``,
``tau_reio``, ``theta_MC`` (via H0) and ``w0``/``wa``. Sampling follows CAMB
AccuracyTarget≈1: piecewise ``Evolve_q`` (``source_accuracy_boost``),
TimeSteps-like τ (``time_step_boost``), cubic densify before LOS/``dlnk``,
high-L residual-template sparse→dense C_ℓ, and denser ℓ (``l_sample_boost``).

Batch walkers: ``compute_cmb_cls_batch``. Mixed-precision experiments:
``gpucosmo.jax.eb.precision`` (``GPUCOSMO_EB_MIXED_PREC``; f32 evolve stub).
"""

from __future__ import annotations

import functools
import os
from functools import partial

import jax
import jax.numpy as jnp
from gpucosmo.jax.eb.dark_energy import cpl_f_de
import numpy as np

from gpucosmo.jax.eb.interp_jax import interp_cubic
from gpucosmo.jax.eb import constants as C
from gpucosmo.jax.eb.bbn_jax import delta_neff_for, yhe_from_bbn
from gpucosmo.jax.eb.evolve_jax import (
    MAX_ETAK_SCALAR,
    PHOT_RSA_KMIN,
    downsample_thermo_for_evolve,
    evolve_scalar_k_jax,
)
from gpucosmo.jax.eb.hierarchy_jax import HierarchyConfig, hierarchy_for_model
from gpucosmo.jax.eb.kgrid_jax import (
    set_k_values_for_int,
    set_k_values_for_sources,
    sync_thermo_taurend_camb_like,
)
from gpucosmo.jax.eb.los_jax import (
    cl_pp_from_weyl,
    hybrid_cls_tt_ee_te_from_sources,
)
from gpucosmo.jax.eb.sources_jax import (
    build_visibility_tables,
    optical_depth_and_visibility,
    temperature_polarization_sources,
)
from gpucosmo.jax.eb.thermo_jax import build_thermo_jax, thermo_values

jax.config.update("jax_enable_x64", True)

CLS_HIERARCHY = HierarchyConfig(
    lmaxg=C.LMAXG_CLS,
    lmaxgpol=C.LMAXGPOL_CLS,
    lmaxnr=C.LMAXNR_CLS,
    include_pol=True,
    second_order_tc=True,
    nqmax=C.NQMAX_CLS,
    lmaxnu=C.LMAXNU_CLS,
    use_de_ppf=True,
)

# Massless-neutrino config, kept to isolate the massive-nu contribution in gates.
CLS_HIERARCHY_MASSLESS = HierarchyConfig(
    lmaxg=C.LMAXG_CLS,
    lmaxgpol=C.LMAXGPOL_CLS,
    lmaxnr=C.LMAXNR_CLS,
    include_pol=True,
    second_order_tc=True,
    nqmax=0,
    use_de_ppf=True,
)

CLS_AMP_SCALE = 1.0
# T/E Bessel → Limber switch. Limber is invalid for primary CMB sources (sharply
# peaked at recombination): switching at ℓ=1500 gave 250% errors in the damping
# tail vs CAMB. Full Bessel LOS to lmax brings unlensed TT/EE/TE damping to
# ≤ 0.11% max (outputs/benchmarks/gate_cls_segmented_gpu_l2508_nolimber.json).
CLS_LIMBER_ELL = 100_000
# φφ Limber switch: CAMB SourceTerms%limber_phi_lmin (SourceWindows.f90 default).
CLS_LIMBER_ELL_PHI = 100
# CAMB applies its max_etak_scalar late-source cutoff only when .not. WantLateTime,
# and WantLateTime = DoLensing. We want lensing, so it stays off.
CLS_LATE_SOURCE_CUTOFF = False
# CAMB NonLinear_lens: scale the Weyl/lensing source by HMcode-2020
# sqrt(P_nl/P_lin) (``hmcode2020_jax``; ratio matches CAMB mead2020 to ≤ 1e-4).
# The DESI/Planck reference (``reference.camb_v2``) uses NonLinear_lens.
CLS_NONLINEAR_LENS = True
# Pre-evolution step budget is banded in k: n_pre scales like k/CLS_K_STEP_REF
# above the reference, since RK4 stability wants Δτ ∝ 1/k. 640 steps holds to
# k ≈ 0.386 and fails by 0.417, which sets the reference.
CLS_K_STEP_REF = 0.35
# Extended top band for lens_potential_accuracy=4 (qmax ≈ Max_eta_k/τ0 ≈ 5.14).
CLS_K_STEP_BAND_EDGES = (0.35, 0.7, 1.4, 2.8, 5.6)
# Cap the Δτ∝1/k step multiplier. Uncapped mult at k=5.14 is ~23 (n_pre~15k).
# Cap=8 left every mode with k≳2 zeroed by CLS_WEYL_MAX_ABS; 16 keeps the
# top band stable on GPU while still far cheaper than the uncapped budget.
CLS_K_STEP_MULT_CAP = 16.0
# Headroom on the linear scaling: the bare ratio still left 4 unstable points at
# k_max = 1.0, so the stability limit is a little steeper than 1/k.
CLS_K_STEP_SAFETY = 1.6
# Lean composite backbone when integrate_on_output (was 4096×32 / 1024×8).
# 512×8 spikes |S_T|>1; 640×8 holds soft key-ℓ gates with max_etak Phase-B freeze.
CLS_N_STEPS = 640
# Low-k band (k ≤ PHOT_RSA_KMIN): shorter n_pre; phot never RSA.
CLS_N_STEPS_LOWK = 480
# Max RK4 microsteps per interval (local k·Δτ budget in RSA).
CLS_N_SUB = 8
# CAMB-like natural SetTimeSteps ~1030; keep n_out at natural density (no thin).
CLS_N_TAU_SRC = 1030
# Device τ-grid segment sizes at time_step_boost = 1 (CAMB SetTimeSteps-like):
# recombination [taurst, taurend], reionization window z_re ± 4, and the uniform
# late segment [taurend, τ0]. Scaled together by time_step_boost.
CLS_TAU_N_REC = 500
CLS_TAU_N_REIO = 50
CLS_TAU_N_LATE = 480
# CAMB AccuracyTarget≈1; mid-k boost>1 collapses TT220 (same failure mode as denser geomspace).
CLS_SOURCE_ACCURACY_BOOST = 1.0
CLS_TIME_STEP_BOOST = 1.0
# Scales backbone n_steps and max_sub.
CLS_EVOLVE_STEP_BOOST = 1.0
# Default: lean composite backbone + local substeps → TimeSteps (CAMB-like).
CLS_INTEGRATE_ON_OUTPUT = True
CLS_EVOLVE_KDT_TARGET = 0.30
CLS_L_SAMPLE_BOOST = 1.0
# Evolve grid: fixed-step RK4 is stable to ~0.45; beyond that Weyl runs away
# (1e100+) regardless of n_pre. Limber C_L^phiphi still needs sources out to
# Max_eta_k/tau0 ≈ 5.14, which we obtain by log-log extrapolating the stable
# Weyl (see ``extend_weyl_high_k``).
CLS_K_MAX_EVOLVE = 0.45
CLS_K_MAX = CLS_K_MAX_EVOLVE
# Limber φφ k extent (CAMB qmax with lens_potential_accuracy=4).
CLS_K_MAX_LIMBER = 5.143

# ---- High-ℓ extension (``highl_plan``; ``compute_cmb_cls(lmax_highl=...)``) ----
# Used only when lensed spectra are wanted past the base range (``jax_eb_theory``:
# lmax_extra > lmax + LENS_MARGIN); everything below the base range is computed as without it.
# Evaluated on CAMB 2.0.4 inputs (AccuracyBoost 1, lens_potential_accuracy 15, lensed to 9000)
# with this module's lensing operator (outputs/p26/highl/offline_*.log):
# * the lensed C_ℓ to ℓ_out need φφ (and the convolution) to ℓ_out + 4000: + 3000 left
#   3e-3 (EE) at ℓ 8500–9000, + 3500 8.7e-4, + 4000 2.4e-4 (CAMB itself: lens_output_margin
#   2000 → 3e-2 at ℓ 9000–10000);
# * the unlensed T/E C_ℓ only to ℓ_out + 500: continuing them past it with their log slope
#   changes the lensed C_ℓ ≤ ℓ_out by < 1e-6 (they are ≤ 1 % of the lensed power there).
CLS_HIGHL_MARGIN_T = 500
CLS_HIGHL_MARGIN_PHI = 4000
# φφ and the lensing convolution reach at least CAMB's template range (lensing_jax extends
# shorter inputs with the template to 8000 anyway).
CLS_HIGHL_LMIN_PHI = 8000
# ℓ sampling past the base grid: Δℓ = 40 to 5000, then CAMB's AccuracyTarget log sampling
# (lSamples_init above min_l_logl_sampling: Δℓ = 100, growing ×1.1). Lensed C_ℓ from the
# interpolated inputs agree with dense inputs to ≤ 7e-5 to ℓ 9000 (TT/EE/TE/BB).
CLS_HIGHL_ELL_STEP = 40
CLS_HIGHL_ELL_LOG_LMIN = 5000
CLS_HIGHL_ELL_LOG_STEP = 100
CLS_HIGHL_ELL_LOG_GROWTH = 1.1
# High-k source modes (``highl_plan``) to CAMB's qmax_int = 2 ℓ_T / τ0 for the unlensed range
# ℓ_T, evolved with step budgets × ceil(k_max / CLS_HIGHL_K_PER_MULT) (the segmented
# integrator's stability limit per unit budget is 0.47–0.6/Mpc; linear_pk_jax uses 0.35/unit).
CLS_HIGHL_K_PER_MULT = 0.35
# Linear (Δk = dkn2) source sampling to CAMB's q_cmb for maximum_l > 5000 with accurate
# polarization: 2 l_smooth_sample / τ0 × 1.4 with l_smooth_sample = 3000 (cmbmain.f90).
CLS_HIGHL_Q_SMOOTH = 2.0 * 3000.0 / 14000.0 * 1.4
# Limber φφ k extent: CAMB set_for_lmax's default lens_potential_accuracy
# max(4, (ℓ_out − 1500)/500) for the lensed ℓ_out, k_max = 18000 · lpa / τ0. lpa 4 instead of
# 15 changes CAMB's lensed TT by 7e-3 at ℓ 4000–5000 (4 % at 8000–9000) and C_L^φφ by 1.8e-3
# at L 1000–2000.
CLS_HIGHL_LENS_K_ETA = 18000.0
# Typical |Evolve_q| at boost=1.0, k_max=0.45, kmin=0.1/τ0.
CLS_N_K = 240
# Soft abort threshold for evolve instability smoke (|S_T| should be O(0.1))
CLS_ST_MAX_ABS = 5.0
# Weyl potential is O(1) at low k and falls with k, so this only catches runaways.
CLS_WEYL_MAX_ABS = 10.0
# What a numerical failure of the per-mode evolution does (issue #11). A mode fails when its
# T/E or Weyl source has non-finite entries, |S_T| > CLS_ST_MAX_ABS or |Weyl| > CLS_WEYL_MAX_ABS;
# the C_ℓ fail when the projection gives non-finite values. Every result of ``compute_cmb_cls``
# carries the count in ``"solver_status"`` (0: valid), whatever the policy:
#   "reject" (default)  every returned C_ℓ is NaN when the count is > 0, so a likelihood is
#                       non-finite and the inference maps it to log L = −∞ (``mg_bundle``). This
#                       is a numerical failure, not a physical exclusion: diagnostics tell them
#                       apart by ``solver_status`` / ``bundle.cmb_solver_status``.
#   "mask"              the pre-0.x behaviour: failed modes are zeroed and non-finite C_ℓ set to
#                       0, which gives finite but wrong spectra (plots of pathological points).
#   "keep"              no zeroing at all (NaN / runaway values propagate); validation.
# GPUCOSMO_SOURCE_FAILURE selects it; GPUCOSMO_STRICT_SOURCES=1 (older switch) means "keep".
SOURCE_FAILURE_POLICIES = ("reject", "mask", "keep")


def _source_failure_policy() -> str:
    if os.environ.get("GPUCOSMO_STRICT_SOURCES", "0") == "1":
        return "keep"
    pol = os.environ.get("GPUCOSMO_SOURCE_FAILURE", "reject").strip().lower()
    if pol not in SOURCE_FAILURE_POLICIES:
        raise ValueError(f"GPUCOSMO_SOURCE_FAILURE={pol!r} not in {SOURCE_FAILURE_POLICIES}")
    return pol


CLS_SOURCE_FAILURE = _source_failure_policy()
CLS_STRICT_SOURCES = CLS_SOURCE_FAILURE == "keep"  # older name
# Perturbation integrator: "segmented" (regime-static segments on
# stiffness-adapted per-k grids, one fused vmap; ``evolve_segmented_jax``) or
# "rk4_timesteps" (legacy switch-per-step RK4 with banded step budgets).
# Segmented is 4-100x closer to a converged reference in the sources, passes
# more CAMB v2 bands (outputs/benchmarks/gate_cls_*_cpu_l2508.json), and is
# faster (outputs/benchmarks/eb_batch_*_gpu*.json).
CLS_EVOLVE_METHOD = "segmented"
# Segmented-evolve knot storage on the source path (``SegmentedConfig.store``):
# "sources" stores only the state entries the sources and δ_m read (same results,
# ~4× less memory per k mode); "all" stores every entry.
CLS_SEGMENT_STORE = "sources"
# Precision of the per-mode evolution and source functions on the batched source path
# (``source_grids_batched``); the per-cosmology τ grid and visibility stay float64.
# "auto" = float32 on accelerators, float64 on CPU. On the RTX 3070 Laptop (1/64-rate
# f64) float32 is ~5× faster with lensed D_ℓ within ~5e-6 and |Δ log L| ≤ 0.007 for the
# PR4 + ACT/Planck-lensing likelihoods (scripts/bench_fp32_stages.py, ΛCDM and μ–Σ).
CLS_SOURCES_DTYPE = os.environ.get("GPUCOSMO_SOURCES_DTYPE", "auto")


def _resolve_sources_dtype(dtype):
    if isinstance(dtype, str):
        if dtype == "auto":
            return jnp.float64 if jax.default_backend() == "cpu" else jnp.float32
        return {"float32": jnp.float32, "float64": jnp.float64}[dtype]
    return dtype


def _cast_floats(tree, dtype):
    def cast(a):
        a = jnp.asarray(a)
        return a.astype(dtype) if jnp.issubdtype(a.dtype, jnp.floating) else a
    return jax.tree_util.tree_map(cast, tree)
# Evolve + source construction over k in chunks of this many modes (lax.map with
# batch_size) on the fused path; 0 = one vmap over the whole grid. Peak memory
# scales with chunk × cosmology batch, which is what limits batching on GPU.
CLS_K_CHUNK = 0
# Batch (cosmology-vmapped) default: chunking keeps B=16 at ~4 GB at lmax 2508.
CLS_BATCH_K_CHUNK = 48
# Local Δℓ=1 windows around acoustic extrema (not global l_sample_boost).
# Kept empty: densifying ells did not improve TT220; reopen after soft-gate shape work.
_ELL_EXTREMA_WINDOWS: tuple[tuple[int, int], ...] = ()

# Thermo table length from τ = 0.05 Mpc to τ0 (``thermo_jax.TAU_TABLE_START``) ...
_THERMO_N = 10000
# ... plus the nodes before it, down to CAMB's start for modes k ≤ THERMO_K_MAX [1/Mpc]
# (``thermo_jax.tau_table_start``): the CMB evolves k ≤ 2.0 (ℓ 13 500), the linear P(k)
# grid k ≤ 2; ``linear_pk_jax`` passes its larger k_max for modes beyond.
THERMO_K_MAX = 2.0
# Host memo for Recfast (keyed by thermo-affecting params; As/ns ignored).
_RECFAST_CACHE: dict[tuple, np.ndarray] = {}
_RECFAST_CACHE_MAX = 8
_THERMO_ARRAY_KEYS = (
    "tau_mpc",
    "a",
    "xe",
    "cs2",
    "opacity",
    "dopacity",
    "tb",
    "log_tau",
    "visibility",
    "dvisibility",
    "ddvisibility",
    "kappa",
)
_THERMO_SCALAR_KEYS = (
    "tight_tau",
    "tauminn",
    "dlntau",
    "taurend",
    "matter_verydom_tau",
    "H0_mpc",
)


def nu_model_for(nu_model=None, num_massive_nu=None):
    """The static ``NeutrinoModel``: ``nu_model`` if given, else ``num_massive_nu`` degenerate
    eigenstates (CAMB's ``num_massive_neutrinos``; default one, CAMB's default)."""
    from gpucosmo.jax.eb.massive_nu import NeutrinoModel

    if nu_model is not None:
        if num_massive_nu is not None and nu_model.hierarchy == "degenerate" \
                and int(num_massive_nu) != nu_model.num_massive:
            raise ValueError(f"num_massive_nu={num_massive_nu} contradicts {nu_model}")
        return nu_model
    n = C.DEFAULT_NUM_MASSIVE_NU if num_massive_nu is None else int(num_massive_nu)
    return NeutrinoModel("degenerate", n)


# Extended cosmology parameters beyond flat w0waCDM, all traceable: Σm_ν [eV], N_eff, the sterile
# effective mass [eV] and Ω_k. ``ext`` dicts carry any subset; missing entries take CAMB's
# defaults (one 0.06 eV eigenstate, N_eff = 3.044, flat).
EXT_DEFAULTS = C.EXT_DEFAULTS


# Spatial curvature in the perturbations (hyperspherical projections, K terms; validated against
# CAMB for −0.05 ≤ Ω_k ≤ 0.03, docs/extended_models.rst). GPUCOSMO_CURVATURE=0 restricts Ω_k ≠ 0 to
# the background (the CMB and P(k) solvers refuse it).
CURVATURE_PERTURBATIONS = os.environ.get("GPUCOSMO_CURVATURE", "1") != "0"


def is_flat(omk) -> bool:
    """A concrete Ω_k = 0 (static flat program); a traced Ω_k is curved (its sign is unknown)."""
    try:
        return float(omk) == 0.0
    except (TypeError, jax.errors.ConcretizationTypeError, jax.errors.TracerArrayConversionError):
        return False


def _require_flat(omk, what: str = "the perturbation solver") -> None:
    """Refuse Ω_k ≠ 0 (or a traced Ω_k) where curvature is not implemented."""
    if not is_flat(omk):
        raise NotImplementedError(f"spatial curvature (omk) is not supported by {what}")


def _require_curvature_enabled(omk) -> None:
    if not is_flat(omk) and not CURVATURE_PERTURBATIONS:
        raise NotImplementedError("spatial curvature (omk) in the perturbation solver is switched off "
                                  "(GPUCOSMO_CURVATURE=0, cls_jax.CURVATURE_PERTURBATIONS)")


def curved_int_grid(q, K):
    """CAMB's scalar integration q for curvature K: closed (K > 0) modes have integer ν = qR ≥ 3
    (``SetClosedkValuesFromArr``). The static grid is rounded where it is denser than ν (every
    integer present; duplicates get zero weight from the centred dpoints, ``closed_dq_fix``):
    the exact sum over ν. Coarser nodes stay put: there the grid resolves the integrand, so the sum
    over integers equals the integral up to exponentially small (Poisson) terms, while rounding
    them would jitter the nodes by up to half a unit of ν (C_ℓ noise ~1e-3 at Ω_k = −0.02; CAMB
    rounds the spacing instead, which a static grid cannot). Open and flat keep q."""
    rinv = jnp.sqrt(jnp.maximum(jnp.abs(K), 1e-300))
    nu = q / rinv
    dense = jnp.gradient(nu) < 1.0
    nu = jnp.maximum(jnp.where(dense, jnp.round(nu), nu), 3.0)
    return jnp.where(K > 0, nu * rinv, q)


def closed_evolve_q(q, K):
    """Evolve grid for curvature K. Closed models: integer ν = qR ≥ 3 as CAMB (``Evolve_q`` rounded,
    ``SetClosedkValuesFromArr``) — ν_i = max(⌈q_i R⌉, ν_{i−1} + 1), so where the static grid is
    denser than ν the nodes are the consecutive integers 3, 4, 5, ... (the integration grid's nodes
    there: the low ℓ use them exactly) and above it each node moves up by less than one unit of ν.
    ``q`` must be increasing. Open and flat keep q."""
    rinv = jnp.sqrt(jnp.maximum(K, 1e-300))
    i = jnp.arange(q.shape[0], dtype=jnp.float64)
    nu = jax.lax.cummax(jnp.maximum(jnp.ceil(q / rinv), 3.0) - i) + i
    return jnp.where(K > 0, nu * rinv, q)


def closed_dq_fix(q, K):
    """Factor on CAMB's dq (``camb_dpoints``, used by the LOS sums) that turns it into the sum over the
    distinct closed-model ν: each run of equal rounded q carries (q_next − q_prev)/2 on its last node
    (the first run, ν = 3, has q_prev = 2/R, so its weight is 1/R as CAMB's ``dpoints(1)``); 1 for
    distinct nodes and for K ≤ 0."""
    from gpucosmo.jax.eb.los_jax import camb_dpoints

    rinv = jnp.sqrt(jnp.maximum(jnp.abs(K), 1e-300))
    used = camb_dpoints(q, half_ends=False)
    nxt = jnp.concatenate([q[1:], q[-1:]])
    last = jnp.concatenate([q[1:] > q[:-1], jnp.ones((1,), bool)])
    # before the first node: ν = 2 when it is ν = 3 (CAMB's dpoints(1) = 1/R), else its mirror image
    # (camb_dpoints' own end weight q1 − q0, so a nearly flat closed model keeps it)
    q_before = jnp.where(q[0] <= 3.5 * rinv, q[0] - rinv, 2.0 * q[0] - q[1])
    start = jnp.concatenate([q_before[None], jnp.where(q[:-1] < q[1:], q[:-1], -jnp.inf)])
    prev = jax.lax.cummax(start)
    want = jnp.where(last, 0.5 * (nxt - prev), 0.0).at[-1].set(used[-1])
    fix = jnp.where(used > 0, want / jnp.where(used > 0, used, 1.0), 0.0)
    return jnp.where(K > 0, fix, 1.0)


def _curved_p_r(q, params, As, ns):
    wq, ks = curved_power_weight(q, params["curv"])
    return As * (ks / params["k_pivot"]) ** (ns - 1.0) * wq


def curved_power_weight(q, K):
    """q²/k_s², k_s = √(q² − K): CAMB's curved measure dq q/k_s² relative to the flat dq/q (the
    primordial power is evaluated at k_s)."""
    ks2 = jnp.maximum(q * q - K, 1e-300)
    return q * q / ks2, jnp.sqrt(ks2)


def ext_params(ext: dict | None) -> dict:
    """``ext`` with every key of ``EXT_DEFAULTS`` (defaults filled in); unknown keys raise. An
    ``omnuh2`` entry (active ω_ν h², CAMB's ``omnuh2_active``; degenerate models) replaces ``mnu``."""
    ext = {} if ext is None else dict(ext)
    bad = set(ext) - set(EXT_DEFAULTS) - {"omnuh2"}
    if bad:
        raise ValueError(f"unknown extended cosmology parameters {sorted(bad)}; known {sorted(EXT_DEFAULTS)}")
    if "omnuh2" in ext and "mnu" in ext:
        raise ValueError("give mnu or omnuh2, not both")
    out = {k: ext.get(k, v) for k, v in EXT_DEFAULTS.items()}
    if "omnuh2" in ext:
        out["omnuh2"] = ext["omnuh2"]
    return out


def _omega_from_physical(
    ombh2,
    omch2,
    h,
    T_cmb=C.COBE_CMB_TEMP,
    nnu=C.DEFAULT_NNU,
    mnu=C.DEFAULT_MNU,
    num_massive_nu=None,
    *,
    nu_model=None,
    meffsterile=0.0,
    omk=0.0,
    omnuh2=None,
):
    """CAMB-convention density fractions (``CAMBdata`` init in results.f90).

    Neutrinos follow CAMB 2.0.4's ``set_cosmology``: ``nu_model`` (``massive_nu.NeutrinoModel``;
    default ``num_massive_nu`` degenerate eigenstates) maps Σm_ν (``mnu``), N_eff (``nnu``) and a
    sterile ``meffsterile`` to eigenstate blocks. ``Omega_nu`` is the massless species only;
    ``Omega_nu_mass`` and ``nu_mass_am`` have one entry per block (shape ``(n_eig,)``):
    the density the block would have if massless and its m/(k_B T), so that
    ``Omega_nu_mass[i] / a^2 * rhonu(a * nu_mass_am[i])`` is CAMB's ``grhormass(i)/a^2 * rhonu``.
    Every input may be traced. ``omk`` = Ω_k (CAMB sign: Ω_k > 0 open); ``curv`` = K [Mpc⁻²].
    """
    from gpucosmo.jax.eb.massive_nu import omega_nu1_h2

    model = nu_model_for(nu_model, num_massive_nu)
    nu = model.resolve(mnu, nnu, meffsterile, T_cmb, omnuh2=omnuh2)
    h2 = jnp.maximum(h**2, 1e-12)
    Omega_b = ombh2 / h2
    Omega_c = omch2 / h2
    Omega_g = C.OMEGA_G_H2 / h2 * (T_cmb / C.COBE_CMB_TEMP) ** 4
    N_ur = nu["N_ur"]
    Omega_nu = omega_nu1_h2(T_cmb) * N_ur / h2
    if model.n_eig > 0:
        Omega_nu_mass = nu["omh2_massless"] / h2[..., None] if jnp.ndim(h2) else nu["omh2_massless"] / h2
        nu_mass_am = nu["am"]
        Omega_nu_massive_today = nu["omnuh2"] / h2
    else:
        Omega_nu_mass = 0.0
        nu_mass_am = 0.0
        Omega_nu_massive_today = 0.0
    Omega_k = jnp.asarray(omk, dtype=jnp.float64)
    Omega_r = Omega_g + Omega_nu
    Omega_m = Omega_b + Omega_c
    Omega_de = jnp.maximum(1.0 - Omega_m - Omega_r - Omega_nu_massive_today - Omega_k, 0.0)
    H0_mpc = h * C.H0_100_INV_MPC
    return {
        "Omega_b": Omega_b,
        "Omega_c": Omega_c,
        "Omega_g": Omega_g,
        "Omega_nu": Omega_nu,
        "Omega_nu_mass": Omega_nu_mass,
        "Omega_nu_massive_today": Omega_nu_massive_today,
        "nu_mass_am": nu_mass_am,
        # Total number of massive states (traced; HMcode's N_ν).
        "num_massive_nu": jnp.sum(nu["number"] * (nu["deg"] > 0), axis=-1) if model.n_eig else 0,
        "omnuh2": nu["omnuh2"],
        "N_ur": N_ur,
        "Omega_r": Omega_r,
        "Omega_m": Omega_m,
        "Omega_k": Omega_k,
        "curv": -Omega_k * H0_mpc**2,
        "Omega_de": Omega_de,
        "H0_mpc": H0_mpc,
        "h": h,
        "ombh2": ombh2,
        "omch2": omch2,
    }


def _params_dict_cls(
    ombh2,
    omch2,
    h,
    ns,
    As,
    tau_reio,
    w0,
    wa,
    *,
    T_cmb=C.COBE_CMB_TEMP,
    nnu=C.DEFAULT_NNU,
    mnu=C.DEFAULT_MNU,
    num_massive_nu=None,
    nu_model=None,
    meffsterile=0.0,
    omk=0.0,
    omnuh2=None,
    YHe=None,
):
    """The solver's parameter dict. ``mnu``/``nnu``/``meffsterile``/``omk`` may be traced; Y_He
    defaults to BBN at (ω_b, N_eff − 3.044) like CAMB's ``set_cosmology``."""
    d = _omega_from_physical(
        ombh2, omch2, h, T_cmb=T_cmb, nnu=nnu, mnu=mnu, num_massive_nu=num_massive_nu,
        nu_model=nu_model, meffsterile=meffsterile, omk=omk, omnuh2=omnuh2,
    )
    yhe = yhe_from_bbn(ombh2, delta_neff=delta_neff_for(nnu)) if YHe is None else YHe
    d.update(
        {
            "n_s": ns,
            "As": As,
            "k_pivot": 0.05,
            "tau_reio": tau_reio,
            "w0": w0,
            "wa": wa,
            "T_CMB": T_cmb,
            "YHe": yhe,
            "f_He": yhe / (C.MASS_RATIO_HE_H * (1.0 - yhe)),
            "mnu": mnu,
            "nnu": nnu,
            "meffsterile": meffsterile,
            "sigma8": 0.8,
        }
    )
    return d


def _make_ell_sparse(lmax: int, l_sample_boost: float = CLS_L_SAMPLE_BOOST) -> list[int]:
    """Python-time ℓ sample grid (``lmax`` is a JIT static arg).

    Dense through the acoustic band; local Δℓ=1 near extrema; ``l_sample_boost``
    tightens global steps (CAMB ``lSampleBoost`` analogue).
    """
    boost = max(float(l_sample_boost), 0.5)
    step_ac = max(1, int(round(2.0 / boost)))
    step_mid = max(1, int(round(5.0 / boost)))
    step_hi = max(2, int(round(20.0 / boost)))
    xs = (
        list(range(2, min(lmax, 30) + 1))
        + list(range(32, min(lmax, 500) + 1, step_ac))
        + list(range(505, min(lmax, 800) + 1, step_mid))
        + list(range(820, lmax + 1, step_hi))
    )
    for lo, hi in _ELL_EXTREMA_WINDOWS:
        if lo > lmax:
            continue
        xs.extend(range(lo, min(hi, lmax) + 1))
    return sorted({x for x in xs if 2 <= x <= lmax})


def _evolve_q_grid(
    *,
    k_max: float = CLS_K_MAX,
    source_accuracy_boost: float = CLS_SOURCE_ACCURACY_BOOST,
    taurst: float = 280.0,
    tau0: float = 14000.0,
) -> jnp.ndarray:
    """CAMB ``SetkValuesForSources`` piecewise q-grid (host → device).

    ``kmin`` matches CAMB ``qmin = qmin0/tau0/initAccuracyBoost`` with
    ``qmin0 = 0.1`` (``cmbmain.f90:93,797``). A hard floor of ``1e-4``
    previously truncated the super-horizon modes that feed ℓ≲30.
    """
    kmin = 0.1 / float(tau0) / float(source_accuracy_boost)
    k_np = set_k_values_for_sources(
        kmin,
        float(k_max),
        float(taurst),
        float(tau0),
        source_accuracy_boost=float(source_accuracy_boost),
        accurate_reionization=True,
        accurate_polarization=True,
        accuracy_target=1,
        reionization=True,
    )
    return jnp.asarray(k_np, dtype=jnp.float64)


def _int_q_grid(
    *,
    k_evolve: jnp.ndarray,
    tau0: float,
    chi_star: float,
    k_max: float = CLS_K_MAX,
    maximum_l: int = 2508,
    int_sample_boost: float = 1.0,
) -> jnp.ndarray:
    """CAMB ``SetkValuesForInt`` denser Cls q-grid (host → device)."""
    k_ev = np.asarray(k_evolve, dtype=np.float64)
    qmin = float(k_ev[0])
    # Cap at Evolve_q max and CAMB-like Bessel etak / τ0
    qmax_int = min(float(k_max), float(k_ev[-1]), max(3000.0, 2.0 * float(maximum_l)) / float(tau0))
    k_np = set_k_values_for_int(
        qmin,
        qmax_int,
        float(tau0),
        float(max(chi_star, 1.0)),
        int_sample_boost=float(int_sample_boost),
        accuracy_target=1,
        accurate_polarization=True,
        accurate_reionization=True,
        maximum_l=int(maximum_l),
    )
    return jnp.asarray(k_np, dtype=jnp.float64)


def _interpolate_sources_k(
    sources_k: jnp.ndarray, k_ev: jnp.ndarray, k_int: jnp.ndarray
) -> jnp.ndarray:
    """Cubic-spline interpolate sources in ``k`` at each τ (CAMB InterpolateSources).

    On-device natural cubic (same formula as ``InterpolateClArr``) so gradients
    flow through the densification and the SciPy host loop no longer forbids
    ``vmap`` over cosmologies.
    """
    from gpucosmo.jax.eb.sources_jax import _natural_cubic_d2

    s = jnp.asarray(sources_k, dtype=jnp.float64)  # (n_k_ev, n_tau)
    ke = jnp.asarray(k_ev, dtype=jnp.float64)
    ki = jnp.asarray(k_int, dtype=jnp.float64)

    # Second derivatives along k for every τ column: reuse the Thomas solver
    # that already vectorizes over the last axis in los_jax.spline_d2_in_k.
    from gpucosmo.jax.eb.los_jax import spline_d2_in_k

    d2 = spline_d2_in_k(s, ke)  # (n_k_ev, n_tau)

    def eval_one(kq):
        i = jnp.searchsorted(ke, kq, side="right") - 1
        i = jnp.clip(i, 0, ke.shape[0] - 2)
        k0 = ke[i]
        k1 = ke[i + 1]
        ho = k1 - k0
        a0 = (k1 - kq) / jnp.maximum(ho, 1e-30)
        b0 = 1.0 - a0
        return (
            a0 * s[i]
            + b0 * s[i + 1]
            - a0 * b0 * ((1.0 + a0) * d2[i] + (1.0 + b0) * d2[i + 1]) * ho * ho / 6.0
        )

    out = jax.vmap(eval_one)(ki)
    out = jnp.where((ki[:, None] >= ke[0]) & (ki[:, None] <= ke[-1]), out, 0.0)
    return out


def _thermo_k_max(k_max=None):
    """Static table k_max: ``k_max`` or THERMO_K_MAX, at least THERMO_K_MAX (None: no prepended nodes)."""
    if THERMO_K_MAX is None:
        return None if k_max is None else float(k_max)
    return float(THERMO_K_MAX) if k_max is None else max(float(k_max), float(THERMO_K_MAX))


def _recfast_thermo_device(
    ombh2, omch2, h, ns, As, tau_reio, w0, wa, mnu=None, nnu=None, meffsterile=None, omk=None,
    omnuh2=None, *, k_max: float | None = None, num_massive_nu: int | None = None, nu_model=None,
    use_omnuh2: bool = False,
) -> dict:
    """On-device Recfast + visibility tables (differentiable in ombh2/τ/…), starting before
    the modes k ≤ ``k_max`` [1/Mpc] (static; default THERMO_K_MAX) do. ``mnu``, ``nnu``,
    ``meffsterile``, ``omk`` (optional, may be traced; ``ext_params`` defaults) with the static
    ``nu_model`` (default ``num_massive_nu`` degenerate eigenstates); ``use_omnuh2``: ``omnuh2``
    (active ω_ν h²) instead of ``mnu``."""
    del ns, As  # primordial; thermo depends on background only
    kw = dict(mnu=None if use_omnuh2 else mnu, nnu=nnu, meffsterile=meffsterile, omk=omk,
              omnuh2=omnuh2 if use_omnuh2 else None)
    ext = ext_params({k: v for k, v in kw.items() if v is not None})
    params = _params_dict_cls(
        ombh2, omch2, h, 0.96, 2e-9, tau_reio, w0, wa, **ext,
        nu_model=nu_model_for(nu_model, num_massive_nu),
    )
    th = build_visibility_tables(build_thermo_jax(params, nthermo=_THERMO_N, k_max=_thermo_k_max(k_max)))
    return sync_thermo_taurend_camb_like(th)


# Recfast + thermo tables of the batched source path (``recfast_thermo_batched``) on the
# CPU via ``host_offload.on_host``: on the GPU they are a 2000-step chain of scalar fp64
# kernels, ~1.4 s per batch on the RTX 3070 Laptop whatever the batch size, and 9× faster
# in fp32 only by breaking Recfast (x_e off by 89 %). The host path is not differentiable;
# the batched theory is evaluated forward only (DALI expansions use finite differences), and
# the single-cosmology path (``compute_cmb_cls``) keeps the device Recfast.
CLS_HOST_THERMO = os.environ.get("GPUCOSMO_HOST_THERMO", "1") == "1"


@functools.lru_cache(maxsize=None)
def _host_recfast_thermo(k_max: float | None, nu_model, use_omnuh2: bool = False):
    from gpucosmo.jax.eb.host_offload import on_host

    return on_host(functools.partial(_recfast_thermo_device, k_max=k_max, nu_model=nu_model,
                                     use_omnuh2=use_omnuh2))


def recfast_thermo_batched(ombh2, omch2, h, ns, As, tau_reio, w0, wa, mnu=None, *,
                           nnu=None, meffsterile=None, omk=None, ext: dict | None = None,
                           k_max: float | None = None, num_massive_nu: int | None = None,
                           nu_model=None) -> dict:
    """``_recfast_thermo_device`` for one cosmology, on the host if ``CLS_HOST_THERMO``
    (vmap it: the host evaluates the whole batch in one call). Extended parameters as keywords
    or one ``ext`` dict (``ext_params``)."""
    k_max = _thermo_k_max(k_max)
    model = nu_model_for(nu_model, num_massive_nu)
    kw = dict(mnu=mnu, nnu=nnu, meffsterile=meffsterile, omk=omk)
    for k_, v in (ext or {}).items():
        if kw.get(k_) is not None:
            raise ValueError(f"{k_} given twice")
        kw[k_] = v
    e = ext_params({k_: v for k_, v in kw.items() if v is not None})
    use_om = "omnuh2" in e
    f = (_host_recfast_thermo(k_max, model, use_om) if CLS_HOST_THERMO
         else functools.partial(_recfast_thermo_device, k_max=k_max, nu_model=model, use_omnuh2=use_om))
    args = (ombh2, omch2, h, ns, As, tau_reio, w0, wa) + tuple(
        jnp.asarray(e[k_], dtype=jnp.float64) for k_ in ("mnu", "nnu", "meffsterile", "omk")) + (
        jnp.asarray(e.get("omnuh2", 0.0), dtype=jnp.float64),)
    return f(*args)


def _highl_cl_template_column(kind: str) -> jnp.ndarray:
    """CAMB ``highL_CL_template(:, C_Temp/C_E/C_Cross/C_Phi)`` as a dense ℓ array.

    File columns: L, TT, EE, BB, TE, PP, TP, EP. Index 0 is unused (L starts at 2).
    Cached as device arrays so ``compute_cmb_cls`` can stay jitted.
    """
    global _HIGHL_TMPL_CACHE
    if kind not in _HIGHL_TMPL_CACHE:
        from gpucosmo.jax.cmb.lensing_jax import _load_highl_template

        tmpl = _load_highl_template()  # (lmax+1, 5): TT,EE,BB,TE,PP
        col = {"tt": 0, "ee": 1, "te": 3, "pp": 4}[kind]
        # Eager even when first hit while tracing, so no tracer is cached.
        with jax.enable_x64(True), jax.ensure_compile_time_eval():  # fp64 even inside fp32 regions
            _HIGHL_TMPL_CACHE[kind] = jnp.asarray(tmpl[:, col], dtype=jnp.float64)
    return _HIGHL_TMPL_CACHE[kind]


_HIGHL_TMPL_CACHE: dict[str, jnp.ndarray] = {}


def _cubic_spline_interp(
    x: jnp.ndarray, xp: jnp.ndarray, yp: jnp.ndarray, d2: jnp.ndarray
) -> jnp.ndarray:
    """Evaluate a natural cubic spline at ``x`` (CAMB ``InterpolateClArr``)."""
    x = jnp.asarray(x, dtype=jnp.float64)
    xp = jnp.asarray(xp, dtype=jnp.float64)
    yp = jnp.asarray(yp, dtype=jnp.float64)
    d2 = jnp.asarray(d2, dtype=jnp.float64)
    n = xp.shape[0]

    def one(xi):
        i = jnp.searchsorted(xp, xi, side="right") - 1
        i = jnp.clip(i, 0, n - 2)
        x0 = xp[i]
        x1 = xp[i + 1]
        ho = x1 - x0
        a0 = (x1 - xi) / jnp.maximum(ho, 1e-30)
        b0 = 1.0 - a0
        return (
            a0 * yp[i]
            + b0 * yp[i + 1]
            - a0 * b0 * ((1.0 + a0) * d2[i] + (1.0 + b0) * d2[i + 1]) * ho * ho / 6.0
        )

    return jax.vmap(one)(x)


def _natural_cubic_d2_1d(xp: jnp.ndarray, yp: jnp.ndarray) -> jnp.ndarray:
    """Natural-cubic y'' on an irregular 1D grid (device Thomas algorithm)."""
    from gpucosmo.jax.eb.sources_jax import _natural_cubic_d2

    return _natural_cubic_d2(xp, yp)


def _interp_cl_residual(
    ell: jnp.ndarray,
    ell_sparse: jnp.ndarray,
    cl_s: jnp.ndarray,
    lmax: int,
    *,
    template_kind: str | None = None,
    template_l_scale: float = 1.0,
) -> jnp.ndarray:
    """CAMB ``InterpolateClArrTemplated``: cubic-interp residual vs high-L template.

    Operates on CAMB ``Cl_scalar`` units: TT is D_ℓ = ℓ(ℓ+1)C_ℓ/2π; EE/TE
    additionally include ``ctnorm`` / ``√ctnorm`` (``CalcScalCls``). The high-L
    unlensed template is stored in those same units. ``pp`` uses a plain cubic
    spline (CAMB ``template_index > 3`` path).
    """
    fac_s = ell_sparse * (ell_sparse + 1.0) / (2.0 * jnp.pi)
    fac = ell * (ell + 1.0) / (2.0 * jnp.pi)
    dl_s = cl_s * fac_s

    # Match CAMB CalcScalCls normalization before comparing to the template.
    if template_kind == "ee":
        ct_s = (ell_sparse * ell_sparse - 1.0) * (ell_sparse + 2.0) * ell_sparse
        ct = (ell * ell - 1.0) * (ell + 2.0) * ell
        dl_s = dl_s * ct_s
    elif template_kind == "te":
        ct_s = jnp.sqrt(
            jnp.maximum(
                (ell_sparse * ell_sparse - 1.0) * (ell_sparse + 2.0) * ell_sparse, 0.0
            )
        )
        ct = jnp.sqrt(
            jnp.maximum((ell * ell - 1.0) * (ell + 2.0) * ell, 0.0)
        )
        dl_s = dl_s * ct_s
    else:
        ct_s = jnp.ones_like(ell_sparse)
        ct = jnp.ones_like(ell)

    use_tmpl = template_kind in ("tt", "ee", "te")
    if use_tmpl:
        tmpl = _highl_cl_template_column(template_kind)
        scale = float(template_l_scale)
        if abs(scale - 1.0) < 1e-12:
            ell_s_i = jnp.clip(ell_sparse.astype(jnp.int32), 0, tmpl.shape[0] - 1)
            ell_i = jnp.clip(ell.astype(jnp.int32), 0, tmpl.shape[0] - 1)
            dl_tmpl_s = tmpl[ell_s_i]
            dl_tmpl = tmpl[ell_i]
        else:

            def _tmpl_at(l_vals):
                tl = l_vals / scale
                lo = jnp.floor(tl).astype(jnp.int32)
                lo = jnp.clip(lo, 1, tmpl.shape[0] - 2)
                hi = lo + 1
                frac = tl - lo.astype(jnp.float64)
                return (1.0 - frac) * tmpl[lo] + frac * tmpl[hi]

            dl_tmpl_s = _tmpl_at(ell_sparse)
            dl_tmpl = _tmpl_at(ell)
        resid_s = dl_s - dl_tmpl_s
    else:
        resid_s = dl_s
        dl_tmpl = jnp.zeros_like(ell)

    d2 = _natural_cubic_d2_1d(ell_sparse, resid_s)
    resid_i = _cubic_spline_interp(ell, ell_sparse, resid_s, d2)
    resid_i = jnp.where(
        (ell >= ell_sparse[0]) & (ell <= ell_sparse[-1]), resid_i, 0.0
    )
    dl_camb = resid_i + dl_tmpl if use_tmpl else resid_i
    # Undo ctnorm so returned C_ℓ matches compute_cmb_cls's raw EE/TE convention.
    dl = dl_camb / jnp.maximum(ct, 1e-30)
    cl = jnp.where(fac > 0.0, dl / jnp.maximum(fac, 1e-30), 0.0)
    out = jnp.zeros(lmax + 1)
    return out.at[2:].set(cl[2:])


def _synthetic_visibility_thermo(params: dict, *, nthermo: int = 2048) -> dict:
    """Fully traced opacity / visibility (no Recfast) for device AD smoke."""
    H0 = params["H0_mpc"]
    a = jnp.geomspace(1e-8, 1.0, nthermo)
    w0, wa = params["w0"], params["wa"]
    f_de = cpl_f_de(a, w0, wa)
    E = jnp.sqrt(
        params["Omega_r"] * a ** (-4)
        + params["Omega_m"] * a ** (-3)
        + params["Omega_de"] * f_de
    )
    dtauda = 1.0 / jnp.maximum(a**2 * E, 1e-30) / H0
    da = jnp.diff(a)
    mid = 0.5 * (dtauda[:-1] + dtauda[1:])
    tau = jnp.concatenate([jnp.array([a[0] * dtauda[0]]), jnp.cumsum(mid * da)])
    tau = tau - tau[0] + 0.05

    ombh2 = params["ombh2"]
    omch2 = params.get("omch2", params["Omega_c"] * params["h"] ** 2)
    z_star = (
        1048.0
        * (1.0 + 0.00124 * ombh2 ** (-0.738))
        * (
            1.0
            + (0.0783 * ombh2 ** (-0.238))
            / (1.0 + 39.5 * ombh2**0.763)
            * (omch2 + ombh2) ** (0.560 / (1.0 + 21.1 * ombh2**1.81))
        )
    )
    a_star = 1.0 / (1.0 + z_star)
    tau_star = jnp.interp(a_star, a, tau)
    sigma_tau = 8.0
    akthom = 0.11 * params["ombh2"] / 0.022
    opacity_rec = akthom * jnp.exp(-0.5 * ((tau - tau_star) / sigma_tau) ** 2)
    z = 1.0 / jnp.maximum(a, 1e-30) - 1.0
    z_re = 7.5 + 20.0 * params["tau_reio"]
    window = 0.5 * (1.0 + jnp.tanh((z_re - z) / 0.5))
    opacity_re = 0.002 * params["tau_reio"] / 0.054 * window / jnp.maximum(a, 1e-3) ** 2
    opacity = opacity_rec + opacity_re
    kappa, g = optical_depth_and_visibility(tau, opacity)
    dopacity = jnp.gradient(opacity, tau)
    emmu = jnp.exp(-jnp.maximum(kappa, 0.0))
    gdot = emmu * (dopacity + opacity * opacity)
    dd_g = jnp.gradient(gdot, tau)
    cs2 = jnp.where(tau > tau_star, 1e-8, jnp.full_like(tau, 1.0 / 3.0))
    return {
        "tau_mpc": tau,
        "a": a,
        "opacity": opacity,
        "dopacity": dopacity,
        "cs2": cs2,
        "xe": jnp.ones_like(tau),
        "kappa": kappa,
        "visibility": g,
        "dvisibility": gdot,
        "ddvisibility": dd_g,
        "tauminn": tau[0],
        "log_tau": jnp.log(jnp.maximum(tau, 1e-8)),
        "taurend": tau_star + 5.0 * sigma_tau,
        "matter_verydom_tau": tau_star * 3.0,
        "tight_tau": tau_star * 0.3,
        "dlntau": jnp.log(tau[-1] / tau[0]) / float(nthermo),
        "H0_mpc": H0,
        "bg_a": a,
        "bg_tau": tau * H0,
        "tb": jnp.ones_like(tau),
    }


def _recfast_cache_key(vec: np.ndarray) -> tuple:
    """Key on background params only (As/ns do not change Recfast tables)."""
    ombh2, omch2, h, _ns, _As, tau_reio, w0, wa = [float(x) for x in vec]
    return (
        round(ombh2, 12),
        round(omch2, 12),
        round(h, 12),
        round(tau_reio, 12),
        round(w0, 12),
        round(wa, 12),
    )


def _recfast_thermo_host(vec: np.ndarray) -> np.ndarray:
    """Host Recfast + visibility → packed arrays (narrow bridge; default neutrinos, flat)."""
    key = _recfast_cache_key(vec)
    hit = _RECFAST_CACHE.get(key)
    if hit is not None:
        return hit

    ombh2, omch2, h, ns, As, tau_reio, w0, wa = [float(x) for x in vec]
    fp = _params_dict_cls(ombh2, omch2, h, ns, As, tau_reio, w0, wa)
    th = build_visibility_tables(build_thermo_jax(fp, nthermo=_THERMO_N, k_max=_thermo_k_max()))
    # Evolve RSA must use CAMB cumulative-visibility taurend (~599), not the
    # opacity 1%-of-peak heuristic (~316) from build_thermo_jax.
    th = sync_thermo_taurend_camb_like(th)
    arrays = np.stack(
        [np.asarray(th[k], dtype=np.float64) for k in _THERMO_ARRAY_KEYS], axis=0
    )
    scalars = np.array(
        [float(th[k]) for k in _THERMO_SCALAR_KEYS], dtype=np.float64
    )
    packed = np.concatenate([arrays.ravel(), scalars])
    if len(_RECFAST_CACHE) >= _RECFAST_CACHE_MAX:
        _RECFAST_CACHE.pop(next(iter(_RECFAST_CACHE)))
    _RECFAST_CACHE[key] = packed
    return packed


def _thermo_n_total() -> int:
    """Length of the ``THERMO_K_MAX`` tables (``_THERMO_N`` plus the prepended nodes)."""
    from gpucosmo.jax.eb.thermo_jax import thermo_prepend_nodes

    return _THERMO_N + thermo_prepend_nodes(_thermo_k_max(), _THERMO_N)


def _unpack_thermo(packed: jnp.ndarray) -> dict:
    n = _thermo_n_total()
    n_arr = len(_THERMO_ARRAY_KEYS)
    arrays = packed[: n_arr * n].reshape((n_arr, n))
    scalars = packed[n_arr * n :]
    th = {k: arrays[i] for i, k in enumerate(_THERMO_ARRAY_KEYS)}
    for i, k in enumerate(_THERMO_SCALAR_KEYS):
        th[k] = scalars[i]
    th["bg_a"] = th["a"]
    th["bg_tau"] = th["tau_mpc"] * th["H0_mpc"]
    return th


def _recfast_thermo_sg(
    ombh2, omch2, h, ns, As, tau_reio, w0, wa
) -> dict:
    """Recfast tables under stop_gradient (host table build, device arrays)."""
    vec = jax.lax.stop_gradient(
        jnp.stack([ombh2, omch2, h, ns, As, tau_reio, w0, wa])
    )
    n_arr = len(_THERMO_ARRAY_KEYS)
    out_dim = n_arr * _thermo_n_total() + len(_THERMO_SCALAR_KEYS)
    packed = jax.pure_callback(
        lambda v: _recfast_thermo_host(np.asarray(v, dtype=np.float64)),
        jax.ShapeDtypeStruct((out_dim,), jnp.float64),
        vec,
        vmap_method="sequential",
    )
    return _unpack_thermo(packed)


def _source_grid_statics(*, n_tau_src, n_steps, n_sub, time_step_boost, evolve_step_boost,
                         integrate_on_output):
    """Static (host int) sizes of the source grid and evolve budgets."""
    boost = float(max(time_step_boost, 0.5))
    n_tau_src = int(max(64, round(n_tau_src * boost)))
    eboost = float(max(evolve_step_boost, 0.5))
    n_steps = int(max(256, round(n_steps * eboost)))
    if integrate_on_output:
        n_sub = int(max(1, round(n_sub * min(eboost, 1.5))))
    else:
        n_sub = int(max(4, round(n_sub * min(eboost, 1.5))))
    n_pre_hi = int(n_steps)
    n_pre_lo = int(max(256, min(int(CLS_N_STEPS_LOWK), n_pre_hi)))
    if float(evolve_step_boost) != 1.0:
        n_pre_lo = int(max(256, round(n_pre_lo * float(max(evolve_step_boost, 0.5)))))
    return dict(n_tau_src=n_tau_src, n_steps=n_steps, n_sub=n_sub, n_pre_hi=n_pre_hi,
                n_pre_lo=n_pre_lo)


def _source_grid_prep(params: dict, thermo: dict, stat: dict) -> dict:
    """Per-cosmology inputs of the per-k evolve: τ source grid (CAMB SetTimeSteps
    twin), visibility and opacity on it, and the evolve's thermo table."""
    n_tau_src = stat["n_tau_src"]
    tau_full = thermo["tau_mpc"]
    g = thermo["visibility"]
    from gpucosmo.jax.eb.kgrid_jax import refined_peak

    # Continuous visibility peak (χ★ enters the lensing kernel).
    tau_peak = refined_peak(jnp.log(tau_full), g)
    tau0 = tau_full[-1]

    # One τ sampler for every path: the device CAMB-SetTimeSteps twin below.
    # (A separate host TRanges path ran 0.5% low in ΛCDM TT at ℓ ≲ 20 and
    # diverged from what the jitted / batched likelihood evaluates.)
    use_host_timesteps = False
    if not use_host_timesteps:
        # Device-side CAMB SetTimeSteps with static segment sizes (vmap-safe):
        # uniform Δτ over [taurst, taurend] (recombination), uniform τ0/500 over
        # [taurend, τ0] (late ISW), plus a reionization window z_re ± 8Δz. The
        # previous device grid skipped τ ≈ 400–6000 entirely, dropping most of
        # the ISW integral (−0.7% in TT at ℓ ≲ 20 vs the host path).
        taurst_d = jnp.asarray(
            thermo.get("taurst_camb_like", tau_peak * 0.73), dtype=jnp.float64
        )
        taurend_d = jnp.asarray(thermo.get("taurend", tau_peak * 2.1), dtype=jnp.float64)
        taurend_d = jnp.clip(taurend_d, taurst_d * 1.01, tau0 * 0.5)
        scale = n_tau_src / float(CLS_N_TAU_SRC)  # n_tau_src already includes boost
        n_reio = max(8, int(round(CLS_TAU_N_REIO * scale)))
        n_late = max(64, int(round(CLS_TAU_N_LATE * scale)))
        n_rec = max(64, int(round(CLS_TAU_N_REC * scale)))
        tau_rec = jnp.linspace(taurst_d, taurend_d, n_rec, endpoint=False)
        tau_late = jnp.linspace(taurend_d, tau0, n_late)
        a_tab = thermo["a"]
        z_re = jnp.asarray(thermo.get("z_re", 7.5), dtype=jnp.float64)

        def _tau_at_z(z):
            return jnp.interp(1.0 / (1.0 + z), a_tab, tau_full)

        reio_lo = _tau_at_z(z_re + 4.0)
        reio_hi = _tau_at_z(jnp.maximum(z_re - 4.0, 0.0))
        tau_reio_grid = jnp.linspace(reio_lo, reio_hi, n_reio)
        tau_src = jnp.sort(jnp.concatenate([tau_rec, tau_late, tau_reio_grid]))
        dtau_eps = 1.0e-6 * (tau0 + 1.0)
        tau_src = jnp.maximum(
            tau_src, jnp.concatenate([tau_src[:1], tau_src[:-1] + dtau_eps])
        )
        tau_src = jnp.minimum(tau_src, tau0)

    # C¹ resampling of the visibility onto τ_src (see interp_jax).
    kappa = interp_cubic(tau_src, tau_full, thermo["kappa"])
    vis = interp_cubic(tau_src, tau_full, g)
    thermo_ev = downsample_thermo_for_evolve(thermo)
    taurend_src = jnp.asarray(thermo.get("taurend", tau_peak), dtype=jnp.float64)
    return dict(tau_src=tau_src, kappa=kappa, vis=vis, thermo_ev=thermo_ev,
                taurend_src=taurend_src, tau0=tau0, tau_peak=tau_peak)


def _scaled_segmented(seg_cfg, mult: int):
    """``SegmentedConfig`` with every segment's step count (and the placement grid) × ``mult``
    (as ``linear_pk_jax.linpk_band_transfers``): the integrator's stability limit grows
    linearly with the budget, ≈ 0.47–0.6/Mpc per unit."""
    import dataclasses

    m = int(mult)
    return dataclasses.replace(seg_cfg, n_tc=seg_cfg.n_tc * m, n_full=seg_cfg.n_full * m,
                               n_mid=seg_cfg.n_mid * m, n_rsa=seg_cfg.n_rsa * m,
                               n_fine=seg_cfg.n_fine * m)


def _check_mode_sources(s_t, s_e, weyl, policy: str | None = None):
    """Validity of one mode's sources: (S_T, S_E, Weyl, failed), ``failed`` 1.0 if the mode
    failed numerically (non-finite entries, |S_T| > CLS_ST_MAX_ABS, non-finite or
    |Weyl| > CLS_WEYL_MAX_ABS), else 0.0 (float, so it passes through jvp / vmap / lax.map).

    Unless ``policy`` (default ``CLS_SOURCE_FAILURE``) is "keep", a failed mode's sources are
    zeroed as before (non-finite entries → 0; T/E of a runaway mode → 0; the Weyl source → 0),
    so the rest of the pipeline stays finite and differentiable; under "reject" the C_ℓ are then
    replaced by NaN (``_apply_failure_policy``)."""
    policy = CLS_SOURCE_FAILURE if policy is None else policy
    te_finite = jnp.all(jnp.isfinite(s_t)) & jnp.all(jnp.isfinite(s_e))
    weyl_finite = jnp.all(jnp.isfinite(weyl))
    if policy != "keep":  # non-finite entries → 0, as the source builder did before
        s_t, s_e, weyl = (jnp.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for v in (s_t, s_e, weyl))
    bad = jnp.max(jnp.abs(jnp.nan_to_num(s_t, nan=0.0))) > CLS_ST_MAX_ABS
    # A single runaway k poisons every k: the source is densified in k with a
    # cubic spline, which is global, so one 1e32 entry turns the whole
    # interpolated array into garbage. |weyl| is O(1) physically.
    weyl_bad = jnp.max(jnp.abs(jnp.nan_to_num(weyl, nan=0.0))) > CLS_WEYL_MAX_ABS
    failed = (~te_finite | ~weyl_finite | bad | weyl_bad).astype(s_t.dtype)
    if policy != "keep":
        s_t = jnp.where(bad, jnp.zeros_like(s_t), s_t)
        s_e = jnp.where(bad, jnp.zeros_like(s_e), s_e)
        weyl = jnp.where(weyl_bad, jnp.zeros_like(weyl), weyl)
    return s_t, s_e, weyl, failed


def _apply_failure_policy(cls: dict, n_failed, policy: str | None = None) -> dict:
    """Add ``"solver_status"`` (number of failed modes + non-finite C_ℓ samples; 0 = valid)
    to ``cls`` and, under the "reject" policy, make every spectrum NaN when it is > 0."""
    policy = CLS_SOURCE_FAILURE if policy is None else policy
    n_failed = jnp.asarray(n_failed, dtype=jnp.float64)
    out = dict(cls)
    if policy == "reject":
        fail = n_failed > 0
        for key in ("tt", "te", "ee", "pp"):
            out[key] = jnp.where(fail, jnp.nan, out[key])
    out["solver_status"] = n_failed
    return out


def _finite_cl(cl, policy: str | None = None):
    """(C_ℓ with non-finite entries → 0 unless the policy is "keep", their count)."""
    policy = CLS_SOURCE_FAILURE if policy is None else policy
    n_bad = jnp.sum(~jnp.isfinite(cl)).astype(jnp.float64)
    if policy == "keep":
        return cl, n_bad
    return jnp.nan_to_num(cl, nan=0.0, posinf=0.0, neginf=0.0), n_bad


def solver_ok(cls) -> jnp.ndarray:
    """True where a ``compute_cmb_cls`` result (or its ``"solver_status"``) is a valid
    evaluation: no failed mode and finite C_ℓ."""
    st = cls["solver_status"] if isinstance(cls, dict) else cls
    return jnp.asarray(st) == 0


def _source_one_k(k, prep: dict, params: dict, thermo: dict, cfg, *, n_pre: int, n_sub: int,
                  integrate_on_output: bool, kdt_target: float, evolve_method: str,
                  keep_states: bool, seg_mult: int = 1):
    """Evolve one k and return its (S_T, S_E, Weyl, δ_m or full states) on τ_src.

    ``seg_mult`` > 1 (high-k modes of the high-ℓ extension, segmented integrator only)
    multiplies the step budgets (``_scaled_segmented``)."""
    thermo_ev, tau_src, vis, kappa, tau0 = (prep["thermo_ev"], prep["tau_src"], prep["vis"],
                                            prep["kappa"], prep["tau0"])
    seg_cfg = None
    if int(seg_mult) != 1 and evolve_method != "segmented":
        raise NotImplementedError("scaled step budgets (seg_mult) need evolve_method='segmented'")
    if evolve_method == "segmented" and not keep_states:
        # Only the sources and δ_m read the states here (with keep_states the caller
        # gets them all).
        import dataclasses

        from gpucosmo.jax.eb.evolve_segmented_jax import DEFAULT_SEGMENTED

        seg_cfg = dataclasses.replace(DEFAULT_SEGMENTED, store=CLS_SEGMENT_STORE)
    if int(seg_mult) != 1:
        from gpucosmo.jax.eb.evolve_segmented_jax import DEFAULT_SEGMENTED

        seg_cfg = _scaled_segmented(DEFAULT_SEGMENTED if seg_cfg is None else seg_cfg, seg_mult)
    y = evolve_scalar_k_jax(
        k,
        params,
        thermo_ev,
        tau_src,
        cfg=cfg,
        n_steps=n_pre,
        n_sub=n_sub,
        integrate_on_output=integrate_on_output,
        kdt_target=kdt_target,
        method=evolve_method,
        seg_cfg=seg_cfg,
    )
    s_t, s_e, weyl = temperature_polarization_sources(
        y, k, tau_src, vis, kappa, params, thermo, cfg, tau0=tau0, sanitize=False
    )
    s_t, s_e, weyl, failed = _check_mode_sources(s_t, s_e, weyl)
    if keep_states:
        return s_t, s_e, weyl, y, failed
    # δ_m(τ) is all the nonlinear-lensing ratio needs; returning it instead of
    # the full (n_tau, n_eq) state cuts that output 76×, which keeps batched
    # programs under the memory budget where XLA would otherwise rematerialize
    # (recompute) the whole evolution (13 s vs 0.7 s per cosmology at B=8).
    from gpucosmo.jax.eb.sources_jax import _background_and_stress

    dm = _background_and_stress(y, k, tau_src, params, thermo, cfg)["delta_m"]
    return s_t, s_e, weyl, dm, failed


def _source_grid(
    k_grid: jnp.ndarray,
    params: dict,
    thermo: dict,
    cfg: HierarchyConfig,
    *,
    n_tau_src: int = CLS_N_TAU_SRC,
    n_steps: int = CLS_N_STEPS,
    n_sub: int = CLS_N_SUB,
    time_step_boost: float = CLS_TIME_STEP_BOOST,
    evolve_step_boost: float = CLS_EVOLVE_STEP_BOOST,
    integrate_on_output: bool = CLS_INTEGRATE_ON_OUTPUT,
    kdt_target: float = CLS_EVOLVE_KDT_TARGET,
    late_source_cutoff: bool = CLS_LATE_SOURCE_CUTOFF,
    device_grids: bool = False,
    evolve_method: str = CLS_EVOLVE_METHOD,
    k_chunk: int = CLS_K_CHUNK,
    keep_states: bool = False,
    n_base: int | None = None,
    hi_mult: int = 1,
    k_bands=None,
    return_status: bool = False,
):
    """Evolve hierarchy on k-grid; return S_T, S_E, Weyl (n_k, n_tau_src).

    ``return_status``: also return the per-mode failure flags (n_k,), 1.0 where the mode failed
    numerically (``_check_mode_sources``), as a ninth value.

    ``k_bands``: host (static) grid for the k-banded step budgets when ``k_grid`` is traced (the
    closed-model evolve grid, ``closed_evolve_q``); defaults to ``k_grid``.

    The last return value is δ_m (n_k, n_tau_src), or the full states
    (n_k, n_tau_src, n_eq) when ``keep_states``.

    ``n_base`` (high-ℓ extension): the modes ``k_grid[n_base:]`` are evolved with the step
    budgets × ``hi_mult`` (segmented integrator), the first ``n_base`` exactly as without it.

    τ samples follow CAMB TimeSteps: dense through last-scattering and
    reionization, with extra late-ISW coverage. ``time_step_boost`` densifies
    ``tau_src`` only. Default ``integrate_on_output`` integrates RK4 between
    TimeSteps knots (CAMB-like); ``evolve_step_boost`` scales max microsteps
    per interval. LOS uses this same native grid + ``camb_dpoints``.

    ``device_grids=True`` skips host TRanges / NumPy thermo peeks so the
    evolve is ``vmap``-safe over cosmologies (batch walkers).
    """
    stat = _source_grid_statics(n_tau_src=n_tau_src, n_steps=n_steps, n_sub=n_sub,
                                time_step_boost=time_step_boost,
                                evolve_step_boost=evolve_step_boost,
                                integrate_on_output=integrate_on_output)
    prep = _source_grid_prep(params, thermo, stat)
    tau_src, tau0, tau_peak = prep["tau_src"], prep["tau0"], prep["tau_peak"]
    taurend_src = prep["taurend_src"]
    max_etak = float(MAX_ETAK_SCALAR)
    n_pre_hi, n_pre_lo = stat["n_pre_hi"], stat["n_pre_lo"]

    def one_k(k, n_pre_k: int, seg_mult: int = 1):
        return _source_one_k(k, prep, params, thermo, cfg, n_pre=n_pre_k, n_sub=stat["n_sub"],
                             integrate_on_output=integrate_on_output, kdt_target=kdt_target,
                             evolve_method=evolve_method, keep_states=keep_states,
                             seg_mult=seg_mult)

    # K-band: PHOT_RSA_KMIN split with shorter n_pre_lo is available, but
    # sequential dual vmaps on one GPU ~2× warm wall. Fuse one vmap at n_pre_hi
    # for mixed Evolve_q; use n_pre_lo only when the whole grid is low-k.
    #
    # Above ~0.35 Mpc^-1 the fixed-step RK4 pre-evolution goes unstable and the
    # Weyl source runs away (1e5 at k=0.42, 1e32 at k=0.45), so the step budget is
    # banded in k: n_pre scales like k, which is what the RK4 stability limit
    # Δτ ∝ 1/k asks for. C_L^{φφ} needs a much wider k range than T/E, so without
    # this the lensing source is unusable exactly where it matters.
    # Band edges from host view of the (static) Evolve_q grid — k_grid itself is
    # Python-static under JIT (fixed anchors / geomspace). Under ``device_grids``
    # (vmap over cosmologies) skip host ``np.asarray(k)`` and use one fused n_pre.
    # The segmented integrator places steps per k from the stiffness, so it
    # needs no k-banded budgets: always one fused vmap.
    if n_base is not None and 0 < int(n_base) < int(k_grid.shape[0]):
        # High-ℓ extension: the base modes as below, then the high-k modes with scaled budgets.
        def run(kg, mult):
            if k_chunk and 0 < int(k_chunk) < int(kg.shape[0]):
                return jax.lax.map(lambda kk: one_k(kk, n_pre_hi, mult), kg, batch_size=int(k_chunk))
            return jax.vmap(lambda kk: one_k(kk, n_pre_hi, mult))(kg)

        lo = run(k_grid[: int(n_base)], 1)
        hi = run(k_grid[int(n_base):], int(hi_mult))
        s_t, s_e, weyl, y_out, failed = (jnp.concatenate([a, b], axis=0) for a, b in zip(lo, hi))
    elif device_grids or evolve_method == "segmented":
        if k_chunk and 0 < int(k_chunk) < int(k_grid.shape[0]):
            # Bound peak memory by k-chunk instead of the full grid; the chunk is
            # still vmapped, and an outer vmap over cosmologies batches it further.
            s_t, s_e, weyl, y_out, failed = jax.lax.map(
                lambda kk: one_k(kk, n_pre_hi), k_grid, batch_size=int(k_chunk)
            )
        else:
            s_t, s_e, weyl, y_out, failed = jax.vmap(lambda kk: one_k(kk, n_pre_hi))(k_grid)
    else:
        k_np = np.asarray(k_grid if k_bands is None else k_bands, dtype=np.float64)
        n_lo = int(np.searchsorted(k_np, float(PHOT_RSA_KMIN), side="right"))
        n_k_tot = int(k_np.size)
        if 0 < n_lo <= n_k_tot:
            n_pre_base = n_pre_hi
        else:
            n_pre_base = n_pre_lo

        edges = [
            int(np.searchsorted(k_np, e, side="right")) for e in CLS_K_STEP_BAND_EDGES
        ]
        bounds = [0] + [e for e in edges if 0 < e < n_k_tot] + [n_k_tot]
        bounds = sorted(set(bounds))
        parts = []
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            k_hi = float(k_np[hi - 1])
            mult = max(1.0, CLS_K_STEP_SAFETY * k_hi / CLS_K_STEP_REF)
            mult = min(mult, float(CLS_K_STEP_MULT_CAP))
            n_pre_band = int(max(256, round(n_pre_base * mult)))
            parts.append(jax.vmap(lambda kk, n=n_pre_band: one_k(kk, n))(k_grid[lo:hi]))
        s_t, s_e, weyl, y_out, failed = (jnp.concatenate([p[i] for p in parts], axis=0)
                                         for i in range(5))
    # CAMB zeroes late sources where k·τ > max_etak_scalar past taurend
    # (cmbmain.f90:1035) -- but only when .not. WantLateTime, and
    # WantLateTime = CP%DoLensing (cmbmain.f90:140). With lensing on, which is the
    # DESI+Planck configuration, CAMB evolves every k to tau0. Applying the cutoff
    # anyway zeroed the Weyl source above k = max_etak/tau0 ~ 0.022, gutting
    # C_L^phiphi at every L the ACT DR6 band cares about.
    if late_source_cutoff:
        late = (tau_src[None, :] > taurend_src) & (
            k_grid[:, None] * tau_src[None, :] > max_etak
        )
        s_t = jnp.where(late, 0.0, s_t)
        s_e = jnp.where(late, 0.0, s_e)
        weyl = jnp.where(late, 0.0, weyl)
    chi_star = tau0 - tau_peak
    if return_status:
        return tau_src, s_t, s_e, weyl, tau0, chi_star, tau_peak, y_out, failed
    return tau_src, s_t, s_e, weyl, tau0, chi_star, tau_peak, y_out


SOURCE_LAYOUTS = ("w_outer", "k_outer", "flat")


def source_grids_batched(
    k_grid: jnp.ndarray,
    params_b: dict,
    thermo_b: dict,
    cfg: HierarchyConfig,
    *,
    layout: str = "k_outer",
    late_source_cutoff: bool = CLS_LATE_SOURCE_CUTOFF,
    evolve_method: str = CLS_EVOLVE_METHOD,
    keep_states: bool = False,
    compute_dtype=None,
    n_base: int | None = None,
    hi_mult: int = 1,
):
    """``_source_grid`` for W cosmologies at once (leading axis W on every leaf of
    ``params_b`` / ``thermo_b``); returns a dict of arrays with leading axis W.

    ``n_base`` / ``hi_mult``: as ``_source_grid`` (high-ℓ extension; the high-k modes
    ``k_grid[n_base:]`` run as a second batch with the step budgets × ``hi_mult``).

    ``compute_dtype`` (default ``CLS_SOURCES_DTYPE``): precision of the per-mode
    evolution and sources. float32 traces that part with 64-bit types disabled, so
    every float64 in the solver becomes float32; inputs are cast down and outputs back
    to float64.

    ``layout`` is how the W × n_k independent mode evolutions are batched:
    ``w_outer`` vmaps cosmologies around a vmap over k (what vmapping
    ``compute_cmb_cls`` does), ``k_outer`` vmaps k around a vmap over cosmologies,
    ``flat`` is one vmap over (cosmology, k) pairs that gathers each mode's
    cosmology tables. The results are identical; XLA's memory layout and fusion
    differ between them.
    """
    stat = _source_grid_statics(n_tau_src=CLS_N_TAU_SRC, n_steps=CLS_N_STEPS, n_sub=CLS_N_SUB,
                                time_step_boost=CLS_TIME_STEP_BOOST,
                                evolve_step_boost=CLS_EVOLVE_STEP_BOOST,
                                integrate_on_output=CLS_INTEGRATE_ON_OUTPUT)
    prep_b = jax.vmap(lambda p, t: _source_grid_prep(p, t, stat))(params_b, thermo_b)

    def one(k, prep, p, t, seg_mult=1):
        return _source_one_k(k, prep, p, t, cfg, n_pre=stat["n_pre_hi"], n_sub=stat["n_sub"],
                             integrate_on_output=CLS_INTEGRATE_ON_OUTPUT,
                             kdt_target=CLS_EVOLVE_KDT_TARGET, evolve_method=evolve_method,
                             keep_states=keep_states, seg_mult=seg_mult)

    n_w = jax.tree_util.tree_leaves(prep_b)[0].shape[0]
    split = n_base is not None and 0 < int(n_base) < int(k_grid.shape[0])

    def modes(k_grid, prep_b, params_b, thermo_b):
        if split:
            nb = int(n_base)
            lo = modes_one(k_grid[:nb], prep_b, params_b, thermo_b, one)
            hi = modes_one(k_grid[nb:], prep_b, params_b, thermo_b,
                           functools.partial(one, seg_mult=int(hi_mult)))
            return jax.tree_util.tree_map(lambda a, b: jnp.concatenate([a, b], axis=1), lo, hi)
        return modes_one(k_grid, prep_b, params_b, thermo_b, one)

    def modes_one(k_grid, prep_b, params_b, thermo_b, one):
        n_k = int(k_grid.shape[0])
        if layout == "w_outer":
            return jax.vmap(lambda pr, p, t: jax.vmap(lambda k: one(k, pr, p, t))(k_grid))(
                prep_b, params_b, thermo_b)
        if layout == "k_outer":
            out = jax.vmap(lambda k: jax.vmap(lambda pr, p, t: one(k, pr, p, t))(
                prep_b, params_b, thermo_b))(k_grid)
            return jax.tree_util.tree_map(lambda a: jnp.swapaxes(a, 0, 1), out)
        if layout == "flat":
            wi = jnp.repeat(jnp.arange(n_w), n_k)
            kk = jnp.tile(k_grid, n_w)

            def take(tree, w):
                return jax.tree_util.tree_map(lambda a: a[w], tree)

            out = jax.vmap(lambda w, k: one(k, take(prep_b, w), take(params_b, w),
                                            take(thermo_b, w)))(wi, kk)
            return jax.tree_util.tree_map(lambda a: a.reshape((n_w, n_k) + a.shape[1:]), out)
        raise ValueError(f"layout {layout!r} not in {SOURCE_LAYOUTS}")

    cd = _resolve_sources_dtype(CLS_SOURCES_DTYPE if compute_dtype is None else compute_dtype)
    if jnp.dtype(cd) == jnp.float32:
        args32 = _cast_floats((jnp.asarray(k_grid), prep_b, params_b, thermo_b), jnp.float32)
        with jax.enable_x64(False):
            out = modes(*args32)
        out = _cast_floats(out, jnp.float64)
    else:
        out = modes(k_grid, prep_b, params_b, thermo_b)
    s_t, s_e, weyl, y_out, failed = out
    max_etak = float(MAX_ETAK_SCALAR)

    def finish(prep, s_t, s_e, weyl):
        # Same as the end of ``_source_grid``.
        if late_source_cutoff:
            tau_src = prep["tau_src"]
            late = (tau_src[None, :] > prep["taurend_src"]) & (
                k_grid[:, None] * tau_src[None, :] > max_etak)
            s_t = jnp.where(late, 0.0, s_t)
            s_e = jnp.where(late, 0.0, s_e)
            weyl = jnp.where(late, 0.0, weyl)
        return s_t, s_e, weyl, prep["tau0"] - prep["tau_peak"]

    s_t, s_e, weyl, chi_star = jax.vmap(finish)(prep_b, s_t, s_e, weyl)
    # mode_failed (W, n_k): 1.0 where a mode failed numerically (``_check_mode_sources``)
    return dict(tau_src=prep_b["tau_src"], s_t=s_t, s_e=s_e, weyl=weyl, tau0=prep_b["tau0"],
                chi_star=chi_star, tau_peak=prep_b["tau_peak"], y_out=y_out, mode_failed=failed)


def cmb_params(ombh2, omch2, h, ns, As, tau_reio, w0, wa, mu0=None, Sigma0=None, *,
               ext: dict | None = None, nu_model=None):
    """(params, mg) as ``prepare_cmb_sources`` builds them (vmap-safe). ``ext``: extended
    parameters (``ext_params``), ``nu_model``: static ``NeutrinoModel``."""
    params = _params_dict_cls(ombh2, omch2, h, ns, As, tau_reio, w0, wa, **ext_params(ext),
                              nu_model=nu_model_for(nu_model))
    mg = mu0 is not None or Sigma0 is not None
    if mg:
        params["mu0"] = jnp.asarray(0.0 if mu0 is None else mu0, dtype=jnp.float64)
        params["Sigma0"] = jnp.asarray(0.0 if Sigma0 is None else Sigma0, dtype=jnp.float64)
    return params, mg


def prepare_cmb_sources(
    *,
    ombh2: jnp.ndarray,
    omch2: jnp.ndarray,
    h: jnp.ndarray,
    tau_reio: jnp.ndarray,
    ns: jnp.ndarray,
    As: jnp.ndarray,
    w0: jnp.ndarray,
    wa: jnp.ndarray,
    lmax: int = 2508,
    n_k: int | None = None,
    k_max: float = CLS_K_MAX,
    use_recfast_thermo: bool = True,
    cfg: HierarchyConfig | None = None,
    source_accuracy_boost: float = CLS_SOURCE_ACCURACY_BOOST,
    time_step_boost: float = CLS_TIME_STEP_BOOST,
    evolve_step_boost: float = CLS_EVOLVE_STEP_BOOST,
    l_sample_boost: float = CLS_L_SAMPLE_BOOST,
    late_source_cutoff: bool = CLS_LATE_SOURCE_CUTOFF,
    nonlinear_lens: bool | str = CLS_NONLINEAR_LENS,
    device_grids: bool = False,
    k_evolve: jnp.ndarray | None = None,
    k_int: jnp.ndarray | None = None,
    evolve_method: str = CLS_EVOLVE_METHOD,
    k_chunk: int = CLS_K_CHUNK,
    keep_states: bool = False,
    mu0: jnp.ndarray | None = None,
    Sigma0: jnp.ndarray | None = None,
    precomputed: dict | None = None,
    lmax_highl: int = 0,
    ext: dict | None = None,
    nu_model=None,
) -> dict:
    """Everything the ℓ-projections need: sources on the integration k grid.

    ``ext``: extended cosmology parameters (Σm_ν, N_eff, m_eff,sterile, Ω_k; ``ext_params``),
    ``nu_model``: the static ``massive_nu.NeutrinoModel`` (default CAMB's one eigenstate).

    ``lmax_highl`` > ``lmax``: high-ℓ extension (``highl_plan(lmax, lmax_highl)``): the source
    grid is the plan's ``k_ev`` (base modes + high-k modes; ``k_evolve`` / a precomputed grid
    must be on it), and the result also carries ``s_t_hi`` / ``s_e_hi`` on ``k_hi`` (the
    plan's ``k_int_hi``) and the plan itself.

    ``precomputed``: ``dict(params=, thermo=, grid=)`` for one cosmology from
    ``cmb_params`` / ``_recfast_thermo_device`` / ``source_grids_batched`` (sliced),
    which skips the evolve here (batched source path).

    ``mu0`` / ``Sigma0`` (both or neither): μ–Σ modified gravity (``mg_jax``,
    ISiTGR convention); switches the static ``cfg.mg`` on.

    Split out of ``compute_cmb_cls`` so the lensing and φφ projections can be
    compared against each other on one EB solve instead of re-running it.

    ``device_grids=True`` (batch path): on-device τ sampling and fixed Int-q
    anchors so ``jax.vmap`` over cosmologies does not hit host NumPy / TRanges.
    Pass concrete ``k_evolve`` / ``k_int`` from outside a ``vmap`` to avoid
    re-entering host grid builders on tracers.
    """
    model = nu_model_for(nu_model)
    cfg = hierarchy_for_model(cfg or CLS_HIERARCHY, model, ext)
    ext = ext_params(ext)
    _require_curvature_enabled(ext["omk"])
    curved = not is_flat(ext["omk"])
    if curved:
        import dataclasses

        if mu0 is not None or Sigma0 is not None:
            raise NotImplementedError("μ–Σ modified gravity with curvature (the ISiTGR port is flat)")
        cfg = dataclasses.replace(cfg, curved=True)
    params = _params_dict_cls(ombh2, omch2, h, ns, As, tau_reio, w0, wa, **ext, nu_model=model)
    if mu0 is not None or Sigma0 is not None:
        import dataclasses

        cfg = dataclasses.replace(cfg, mg=True)
        params["mu0"] = jnp.asarray(0.0 if mu0 is None else mu0, dtype=jnp.float64)
        params["Sigma0"] = jnp.asarray(0.0 if Sigma0 is None else Sigma0, dtype=jnp.float64)

    if precomputed is not None:
        params, thermo = precomputed["params"], precomputed["thermo"]
        params_ev = params
    elif use_recfast_thermo:
        # On-device Recfast so gradients flow to ombh2 / τ_reio / H0 / w0wa.
        # (Host ``_recfast_thermo_sg`` uses sequential pure_callback — not batchable.)
        thermo = _recfast_thermo_device(
            ombh2, omch2, h, ns, As, tau_reio, w0, wa, ext["mnu"], ext["nnu"], ext["meffsterile"],
            ext["omk"], ext.get("omnuh2"), nu_model=model, use_omnuh2="omnuh2" in ext,
        )
        params_ev = params
    else:
        thermo = _synthetic_visibility_thermo(params)
        params_ev = params

    plan = None
    if lmax_highl and int(lmax_highl) > int(lmax):
        if n_k is not None:
            raise ValueError("the high-ℓ extension (lmax_highl) needs the Evolve_q grid (n_k=None)")
        plan = highl_plan(int(lmax), int(lmax_highl), float(source_accuracy_boost))
        if k_evolve is not None and int(np.shape(k_evolve)[0]) != int(plan["k_ev"].shape[0]):
            raise ValueError("k_evolve does not match highl_plan(lmax, lmax_highl)['k_ev']")
        k_evolve = plan["k_ev"]
        if k_int is None:
            k_int = plan["k_int"]
    if k_evolve is None:
        if n_k is not None:
            # Match CAMB qmin = 0.1/tau0 (same floor as _evolve_q_grid).
            kmin = 0.1 / 14000.0 / float(source_accuracy_boost)
            k_grid = jnp.geomspace(kmin, float(k_max), int(n_k))
        else:
            # Fixed taurst/tau0 keep Evolve_q shape JIT-stable and avoid TT220
            # collapse from thermo-shifted q spacing (boost must stay ≤1).
            k_grid = _evolve_q_grid(
                k_max=float(k_max),
                source_accuracy_boost=float(source_accuracy_boost),
                taurst=280.0,
                tau0=14000.0,
            )
    else:
        k_grid = jnp.asarray(k_evolve, dtype=jnp.float64)
    k_static = k_grid  # host grid for the integration q (``_int_q_grid``) and step bands
    if curved:  # CAMB evolves closed models at integer ν ≥ 3 (``SetClosedkValuesFromArr``)
        nb = k_grid.shape[0] if plan is None else int(plan["n_base"])  # the increasing base modes
        k_grid = jnp.concatenate([closed_evolve_q(k_grid[:nb], params["curv"]), k_grid[nb:]])

    if precomputed is not None:
        g = precomputed["grid"]
        tau_src, s_t, s_e, weyl, tau0, chi_star, tau_peak, y_out = (
            g["tau_src"], g["s_t"], g["s_e"], g["weyl"], g["tau0"], g["chi_star"],
            g["tau_peak"], g["y_out"])
        # a grid made before the status existed counts as valid
        mode_failed = g.get("mode_failed", jnp.zeros(s_t.shape[:1]))
    else:
        tau_src, s_t, s_e, weyl, tau0, chi_star, tau_peak, y_out, mode_failed = _source_grid(
            k_grid,
            params_ev,
            thermo,
            cfg,
            time_step_boost=float(time_step_boost),
            evolve_step_boost=float(evolve_step_boost),
            late_source_cutoff=bool(late_source_cutoff),
            device_grids=bool(device_grids),
            evolve_method=str(evolve_method),
            k_chunk=int(k_chunk),
            keep_states=bool(keep_states),
            k_bands=k_static,
            return_status=True,
            **({} if plan is None else dict(n_base=plan["n_base"], hi_mult=plan["hi_mult"])),
        )
    # CAMB: InterpolateSources Evolve_q → SetkValuesForInt, then LOS + dlnk.
    # Batch / device_grids: fixed τ0/χ★ anchors (same as Evolve_q) keep Int-q
    # host construction outside the traced cosmology axis.
    if k_int is None:
        if device_grids:
            tau0_q, chi_q = 14000.0, 14000.0 - 280.0
        else:
            tau0_q = float(np.asarray(tau0))
            chi_q = float(np.asarray(chi_star))
        k_int_arr = _int_q_grid(
            k_evolve=k_static,
            tau0=tau0_q,
            chi_star=chi_q,
            k_max=float(k_max),
            maximum_l=int(lmax),
            int_sample_boost=1.0,
        )
    else:
        k_int_arr = jnp.asarray(k_int, dtype=jnp.float64)

    try:  # host bound for the curved Bessel tables (the curved q grid is traced)
        k_max_host = float(np.max(np.asarray(k_int_arr)))
    except (TypeError, ValueError, jax.errors.TracerArrayConversionError):
        k_max_host = None
    if curved:  # CAMB: closed integer ν; P(k_s) and dq q/k_s² (cmbmain.f90 CalcScalCls)
        k_int_arr = curved_int_grid(k_int_arr, params["curv"])
        wq, ks = curved_power_weight(k_int_arr, params["curv"])
        p_r = As * (ks / params["k_pivot"]) ** (ns - 1.0) * wq * closed_dq_fix(k_int_arr, params["curv"])
    else:
        p_r = As * (k_int_arr / params["k_pivot"]) ** (ns - 1.0)
    p_r_ev = As * (k_grid / params["k_pivot"]) ** (ns - 1.0)

    # Limber φφ needs Weyl out to Max_eta_k/τ0; evolve only reaches 0.45, so the
    # source is continued with the Eisenstein–Hu transfer shape.
    from gpucosmo.jax.eb.nonlinear_sources_jax import extend_weyl_high_k

    k_limber = jnp.geomspace(
        jnp.asarray(k_grid[0]),
        jnp.asarray(CLS_K_MAX_LIMBER if plan is None else plan["k_limber_max"]),
        160 if plan is None else plan["n_limber"],
    )
    # With the high-ℓ extension the modes are [base grid, high-k modes]; ``hi_idx`` picks a
    # k-sorted set covering the whole range for the k-interpolations.
    srt = slice(None) if plan is None else np.asarray(plan["hi_idx"])
    weyl_limber, k_limber = extend_weyl_high_k(weyl[srt], k_grid[srt], k_limber, params=params,
                                               a_tau=thermo_values(thermo, tau_src)[0])

    # CAMB MakeNonlinearSources (NonLinear_lens): scale the lensing source by
    # sqrt(P_nl/P_lin) on both the evolve grid and the Limber grid. ``nonlinear_lens``:
    # True / "mead2020" (HMcode-2020, CAMB's default) or "mead2016" (HMcode-2016).
    if nonlinear_lens:
        from gpucosmo.jax.eb.nonlinear_sources_jax import hmcode_weyl_ratio

        ratio_l = hmcode_weyl_ratio(
            y_ev=y_out[srt] if keep_states else None,
            delta_m=None if keep_states else y_out[srt],
            k_ev=k_grid[srt], tau=tau_src, thermo=thermo, params=params,
            cfg=cfg, k_out=k_limber,
            version="mead2020" if nonlinear_lens is True else str(nonlinear_lens),
        )
        weyl_limber = weyl_limber * ratio_l
        log_kl = jnp.log(k_limber)
        ratio_ev = jax.vmap(
            lambda col: jnp.interp(jnp.log(k_grid), log_kl, col), in_axes=1, out_axes=1
        )(ratio_l)
        weyl = weyl * ratio_ev
        # Materialize the scaled sources once. Without the barrier XLA fuses the
        # ratio producer into the φφ LOS / Limber loop bodies and recomputes it
        # every iteration (25× slower at B=8 cosmologies).
        weyl, weyl_limber = jax.lax.optimization_barrier((weyl, weyl_limber))
    p_r_limber = As * (k_limber / params["k_pivot"]) ** (ns - 1.0)

    weyl_ev, s_t_ev, s_e_ev = weyl, s_t, s_e
    hi = {}
    if plan is not None:
        # ℓ ≤ lmax projection from the base modes exactly as without the extension; the
        # ℓ > lmax projection from all modes on the plan's k_int_hi.
        nb = plan["n_base"]
        k_hi = plan["k_int_hi"]
        hi = dict(s_t_hi=_interpolate_sources_k(s_t[srt], k_grid[srt], k_hi),
                  s_e_hi=_interpolate_sources_k(s_e[srt], k_grid[srt], k_hi), k_hi=k_hi,
                  p_r_hi=(As * (k_hi / params["k_pivot"]) ** (ns - 1.0) if not curved else
                          _curved_p_r(k_hi, params, As, ns)), plan=plan)
        k_grid, s_t, s_e, weyl = k_grid[:nb], s_t[:nb], s_e[:nb], weyl[:nb]
        s_t_ev, s_e_ev = s_t_ev[:nb], s_e_ev[:nb]
    s_t = _interpolate_sources_k(s_t, k_grid, k_int_arr)
    s_e = _interpolate_sources_k(s_e, k_grid, k_int_arr)
    weyl = _interpolate_sources_k(weyl, k_grid, k_int_arr)
    ell_list = _make_ell_sparse(int(lmax), l_sample_boost=float(l_sample_boost))

    return {
        **hi,
        "s_t": s_t,
        "s_e": s_e,
        "weyl": weyl,
        # Evolve_q grid, pre-densification: reaches higher k than the Bessel
        # integration grid, which is what the phi-phi Limber branch needs.
        "weyl_ev": weyl_limber,
        "s_t_ev": s_t_ev,
        "s_e_ev": s_e_ev,
        # Full evolved states only with keep_states=True, else δ_m (n_k, n_tau).
        "y_ev": y_out if keep_states else None,
        "delta_m_ev": None if keep_states else y_out,
        "k_ev": k_limber,
        "p_r_ev": p_r_limber,
        "tau": tau_src,
        "tau0": tau0,
        "chi_star": chi_star,
        "tau_peak": tau_peak,
        "k": k_int_arr,
        "p_r": p_r,
        "ell_list": ell_list,
        "params": params,
        "thermo": thermo,
        "amp_scale": CLS_AMP_SCALE,
        "curv": params["curv"] if curved else None,
        "k_max_host": k_max_host,
        # number of evolved modes (all of them, high-ℓ ones included) that failed numerically
        "n_failed_modes": jnp.sum(mode_failed).astype(jnp.float64),
    }


def compute_cmb_cls(
    *,
    ombh2: jnp.ndarray,
    omch2: jnp.ndarray,
    h: jnp.ndarray,
    tau_reio: jnp.ndarray,
    ns: jnp.ndarray,
    As: jnp.ndarray,
    w0: jnp.ndarray,
    wa: jnp.ndarray,
    lmax: int = 2508,
    n_k: int | None = None,
    k_max: float = CLS_K_MAX,
    use_recfast_thermo: bool = True,
    cfg: HierarchyConfig | None = None,
    limber_ell: int = CLS_LIMBER_ELL,
    limber_ell_phi: int = CLS_LIMBER_ELL_PHI,
    source_accuracy_boost: float = CLS_SOURCE_ACCURACY_BOOST,
    time_step_boost: float = CLS_TIME_STEP_BOOST,
    evolve_step_boost: float = CLS_EVOLVE_STEP_BOOST,
    l_sample_boost: float = CLS_L_SAMPLE_BOOST,
    late_source_cutoff: bool = CLS_LATE_SOURCE_CUTOFF,
    nonlinear_lens: bool | str = CLS_NONLINEAR_LENS,
    device_grids: bool = False,
    k_evolve: jnp.ndarray | None = None,
    k_int: jnp.ndarray | None = None,
    evolve_method: str = CLS_EVOLVE_METHOD,
    k_chunk: int = CLS_K_CHUNK,
    mu0: jnp.ndarray | None = None,
    Sigma0: jnp.ndarray | None = None,
    precomputed: dict | None = None,
    lmax_highl: int = 0,
    ext: dict | None = None,
    nu_model=None,
) -> dict[str, jnp.ndarray]:
    """Compute C_ℓ^{TT,TE,EE,φφ} on device (dimensionless ΔT/T for TT/TE/EE).

    The result also holds ``"solver_status"``: the number of evolved modes that failed
    numerically plus the non-finite C_ℓ samples; 0 is a valid evaluation (``solver_ok``). Under
    the default ``CLS_SOURCE_FAILURE`` "reject" the spectra are NaN when it is > 0.

    Default k-sampling is CAMB ``SetkValuesForSources`` (piecewise Evolve_q).
    Pass ``n_k`` only for legacy equal-log geomspace overrides.
    ``device_grids=True`` enables the vmap-stable batch path (see
    ``compute_cmb_cls_batch``).

    ``lmax_highl`` > ``lmax`` (high-ℓ extension, for lensed spectra to ``lmax_highl``): the
    arrays run to ``highl_plan(lmax, lmax_highl)['l_p']`` instead of ``lmax``. ℓ ≤ lmax are
    projected as without it; T/E past lmax from extra Evolve_q modes to k = 2 l_t/τ0 (scaled
    step budgets) on a sparse ℓ grid to ``l_t`` and continued past it with their log slope;
    φφ by Limber to ``l_p`` with Weyl modes to the plan's ``k_limber_max``.
    """
    pieces = prepare_cmb_sources(
        ombh2=ombh2,
        omch2=omch2,
        h=h,
        tau_reio=tau_reio,
        ns=ns,
        As=As,
        w0=w0,
        wa=wa,
        lmax=lmax,
        n_k=n_k,
        k_max=k_max,
        use_recfast_thermo=use_recfast_thermo,
        cfg=cfg,
        source_accuracy_boost=source_accuracy_boost,
        time_step_boost=time_step_boost,
        evolve_step_boost=evolve_step_boost,
        l_sample_boost=l_sample_boost,
        late_source_cutoff=late_source_cutoff,
        nonlinear_lens=nonlinear_lens,
        device_grids=device_grids,
        k_evolve=k_evolve,
        k_int=k_int,
        evolve_method=evolve_method,
        k_chunk=k_chunk,
        mu0=mu0,
        Sigma0=Sigma0,
        precomputed=precomputed,
        lmax_highl=lmax_highl,
        ext=ext,
        nu_model=nu_model,
    )
    if "plan" in pieces:
        return _cmb_cls_highl(pieces, limber_ell=limber_ell, limber_ell_phi=limber_ell_phi)
    s_t, s_e, weyl = pieces["s_t"], pieces["s_e"], pieces["weyl"]
    tau_src, tau0, chi_star = pieces["tau"], pieces["tau0"], pieces["chi_star"]
    k_grid, p_r = pieces["k"], pieces["p_r"]
    ell_list = pieces["ell_list"]
    ell_sparse = jnp.asarray(ell_list, dtype=jnp.float64)

    # One shared Bessel LOS pass for T and E (TT, EE, TE together).
    cl_tt_s, cl_ee_s, cl_te_s = hybrid_cls_tt_ee_te_from_sources(
        s_t,
        s_e,
        tau_src,
        tau0,
        k_grid,
        p_r,
        ell_sparse,
        limber_ell=limber_ell,
        ell_list=ell_list,
        curv=pieces["curv"],
        k_max_host=pieces.get("k_max_host"),
    )
    cl_pp_s = cl_pp_from_weyl(
        weyl,
        tau_src,
        tau0,
        chi_star,
        k_grid,
        p_r,
        ell_sparse,
        limber_ell=int(limber_ell_phi),
        ell_list=ell_list,
        weyl_limber=pieces["weyl_ev"],
        k_limber=pieces["k_ev"],
        p_r_limber=pieces["p_r_ev"],
        curv=pieces["curv"],
        k_max_host=pieces.get("k_max_host"),
    )

    fin = [_finite_cl(c) for c in (cl_tt_s, cl_te_s, cl_ee_s, cl_pp_s)]
    (cl_tt_s, _), (cl_te_s, _), (cl_ee_s, _), (cl_pp_s, _) = fin
    n_failed = pieces["n_failed_modes"] + sum(n for _, n in fin)
    cl_tt_s = cl_tt_s * CLS_AMP_SCALE
    cl_te_s = cl_te_s * CLS_AMP_SCALE
    cl_ee_s = cl_ee_s * CLS_AMP_SCALE
    cl_pp_s = cl_pp_s * CLS_AMP_SCALE

    ell = jnp.arange(lmax + 1, dtype=jnp.float64)

    return _apply_failure_policy({
        "ell": ell,
        "tt": _interp_cl_residual(
            ell, ell_sparse, cl_tt_s, int(lmax), template_kind="tt"
        ),
        "te": _interp_cl_residual(
            ell, ell_sparse, cl_te_s, int(lmax), template_kind="te"
        ),
        "ee": _interp_cl_residual(
            ell, ell_sparse, cl_ee_s, int(lmax), template_kind="ee"
        ),
        "pp": _interp_cl_residual(
            ell, ell_sparse, cl_pp_s, int(lmax), template_kind="pp"
        ),
    }, n_failed)


def _cmb_cls_highl(pieces: dict, *, limber_ell: int, limber_ell_phi: int) -> dict[str, jnp.ndarray]:
    """``compute_cmb_cls`` with the high-ℓ extension (``pieces`` from ``prepare_cmb_sources``
    with ``lmax_highl``)."""
    plan = pieces["plan"]
    l_t, l_p = int(plan["l_t"]), int(plan["l_p"])
    tau_src, tau0, chi_star = pieces["tau"], pieces["tau0"], pieces["chi_star"]
    ell_lo = list(pieces["ell_list"])
    ell_t = ell_lo + list(plan["ell_t_hi"])
    ell_p = ell_t + list(plan["ell_p_hi"])
    cl_lo = hybrid_cls_tt_ee_te_from_sources(
        pieces["s_t"], pieces["s_e"], tau_src, tau0, pieces["k"], pieces["p_r"],
        jnp.asarray(ell_lo, dtype=jnp.float64), limber_ell=limber_ell, ell_list=ell_lo, curv=pieces["curv"],
        k_max_host=pieces.get("k_max_host"))
    ell_hi = list(plan["ell_t_hi"])
    cl_hi = hybrid_cls_tt_ee_te_from_sources(
        pieces["s_t_hi"], pieces["s_e_hi"], tau_src, tau0, pieces["k_hi"], pieces["p_r_hi"],
        jnp.asarray(ell_hi, dtype=jnp.float64), limber_ell=limber_ell, ell_list=ell_hi, curv=pieces["curv"])
    cl_pp_s = cl_pp_from_weyl(
        pieces["weyl"], tau_src, tau0, chi_star, pieces["k"], pieces["p_r"],
        jnp.asarray(ell_p, dtype=jnp.float64), limber_ell=int(limber_ell_phi), ell_list=ell_p,
        weyl_limber=pieces["weyl_ev"], k_limber=pieces["k_ev"], p_r_limber=pieces["p_r_ev"],
        curv=pieces["curv"], k_max_host=pieces.get("k_max_host"))

    fin = [_finite_cl(jnp.concatenate([a, b])) for a, b in zip(cl_lo, cl_hi)] + [_finite_cl(cl_pp_s)]
    cl_tt_s, cl_ee_s, cl_te_s, cl_pp_s = (c * CLS_AMP_SCALE for c, _ in fin)
    n_failed = pieces["n_failed_modes"] + sum(n for _, n in fin)
    es_t = jnp.asarray(ell_t, dtype=jnp.float64)
    es_p = jnp.asarray(ell_p, dtype=jnp.float64)
    ell_tt = jnp.arange(l_t + 1, dtype=jnp.float64)
    tt = _interp_cl_residual(ell_tt, es_t, cl_tt_s, l_t, template_kind="tt")
    ee = _interp_cl_residual(ell_tt, es_t, cl_ee_s, l_t, template_kind="ee")
    te = _interp_cl_residual(ell_tt, es_t, cl_te_s, l_t, template_kind="te")
    if l_p > l_t:
        rho = te[l_t] / jnp.sqrt(jnp.maximum(tt[l_t] * ee[l_t], 1e-300))
        tt = _log_slope_continuation(tt, l_t, l_p)
        ee = _log_slope_continuation(ee, l_t, l_p)
        te = jnp.concatenate([te, rho * jnp.sqrt(tt[l_t + 1:] * ee[l_t + 1:])])
    ell = jnp.arange(l_p + 1, dtype=jnp.float64)
    pp = _interp_cl_residual(ell, es_p, cl_pp_s, l_p, template_kind="pp")
    return _apply_failure_policy({"ell": ell, "tt": tt, "te": te, "ee": ee, "pp": pp}, n_failed)


compute_cmb_cls_jit = jax.jit(
    compute_cmb_cls,
    static_argnames=(
        "lmax_highl",
        "lmax",
        "n_k",
        "k_max",
        "use_recfast_thermo",
        "limber_ell",
        "limber_ell_phi",
        "source_accuracy_boost",
        "time_step_boost",
        "evolve_step_boost",
        "l_sample_boost",
        "late_source_cutoff",
        "nonlinear_lens",
        "device_grids",
        "evolve_method",
        "k_chunk",
    ),
)


@functools.lru_cache(maxsize=16)
def _batch_k_grids(n_k, k_max, source_accuracy_boost, lmax):
    """Host-built Evolve_q / Int-q grids for the batch path (fixed τ0/χ★ anchors).

    Cached: they depend only on static settings, and rebuilding them on the host
    every call cost more than the batched solve for small batches.
    """
    with jax.enable_x64(True), jax.ensure_compile_time_eval():  # fp64 even inside fp32 regions
        if n_k is not None:
            kmin = 0.1 / 14000.0 / source_accuracy_boost
            k_ev = jnp.geomspace(kmin, k_max, n_k)
        else:
            k_ev = _evolve_q_grid(
                k_max=k_max,
                source_accuracy_boost=source_accuracy_boost,
                taurst=280.0,
                tau0=14000.0,
            )
        k_int_arr = _int_q_grid(
            k_evolve=k_ev,
            tau0=14000.0,
            chi_star=14000.0 - 280.0,
            k_max=k_max,
            maximum_l=lmax,
            int_sample_boost=1.0,
        )
    return k_ev, k_int_arr


def _highl_ell_samples(l_from: int, l_to: int, step: int | None = None) -> tuple[list[int], int | None]:
    """Sparse ℓ past ``l_from`` to ``l_to`` (included): Δℓ = CLS_HIGHL_ELL_STEP below
    CLS_HIGHL_ELL_LOG_LMIN, then log steps (CLS_HIGHL_ELL_LOG_STEP × CLS_HIGHL_ELL_LOG_GROWTH^n).
    ``step``: the log step to continue from (returned for the next call)."""
    xs, ell = [], int(l_from)
    if step is None:
        while ell + CLS_HIGHL_ELL_STEP <= min(CLS_HIGHL_ELL_LOG_LMIN, l_to):
            ell += CLS_HIGHL_ELL_STEP
            xs.append(ell)
        step = CLS_HIGHL_ELL_LOG_STEP
    while ell + step <= l_to:
        ell += step
        xs.append(ell)
        step = int(round(step * CLS_HIGHL_ELL_LOG_GROWTH))
    if not xs or xs[-1] != l_to:
        if xs and l_to - xs[-1] < 0.3 * step:
            xs[-1] = l_to  # no sliver interval at the end
        else:
            xs.append(l_to)
    return xs, step


def _evolve_q_smooth_start(source_accuracy_boost: float) -> tuple[float, float]:
    """(q_cmb, dkn2) of the base Evolve_q grid (``kgrid_jax.set_k_values_for_sources`` at the
    batch anchors taurst = 280, τ0 = 14000): where its linear Δk = dkn2 part ends."""
    from gpucosmo.jax.eb.kgrid_jax import _L_SMOOTH_SAMPLE

    sab = float(source_accuracy_boost)
    taurst, tau0 = 280.0, 14000.0
    q_switch = 2.0 * 6.3 / taurst
    q_cmb = max(q_switch * 2.0, 2.0 * float(_L_SMOOTH_SAMPLE) / tau0 * sab)
    return q_cmb, 0.9 / taurst / sab / 1.2


def highl_lens_k_max(lmax_out: int) -> float:
    """Limber φφ k extent [1/Mpc] for lensed spectra to ``lmax_out`` (CAMB default
    lens_potential_accuracy; never below the base CLS_K_MAX_LIMBER)."""
    lpa = max(4.0, (float(lmax_out) - 1500.0) / 500.0)
    return max(float(CLS_K_MAX_LIMBER), CLS_HIGHL_LENS_K_ETA * lpa / 14000.0)


@functools.lru_cache(maxsize=8)
def highl_plan(l_base: int, lmax_out: int, source_accuracy_boost: float = 1.0) -> dict:
    """Static (host-built) grids of the high-ℓ extension.

    ``l_base``: the base unlensed range (``lmax`` of ``compute_cmb_cls``, i.e. lmax + LENS_MARGIN);
    ``lmax_out`` (> l_base): lensed spectra are wanted to here. Unlensed T/E C_ℓ are projected
    to ``l_t`` = lmax_out + CLS_HIGHL_MARGIN_T (log-slope continuation past it), φφ and the
    lensing convolution reach ``l_p`` = max(CLS_HIGHL_LMIN_PHI, lmax_out + CLS_HIGHL_MARGIN_PHI).

    Keys: ``l_base``, ``l_t``, ``l_p``, ``k_ev`` (base Evolve_q modes then the high-k modes),
    ``n_base`` (number of base modes), ``hi_mult`` (their step-budget multiplier), ``k_int``
    (the base Int-q grid, unchanged), ``k_int_hi`` (Int-q grid of the ℓ > l_base projection),
    ``ell_base`` / ``ell_t_hi`` / ``ell_p_hi`` (sparse ℓ: base grid, T/E past l_base to l_t, φφ
    past l_t to l_p), ``k_limber_max``, ``n_limber``.
    """
    l_base, lmax_out = int(l_base), int(lmax_out)
    if lmax_out <= l_base:
        raise ValueError(f"highl_plan: lmax_out {lmax_out} must exceed the base range {l_base}")
    sab = float(source_accuracy_boost)
    l_t = lmax_out + CLS_HIGHL_MARGIN_T
    l_p = max(CLS_HIGHL_LMIN_PHI, lmax_out + CLS_HIGHL_MARGIN_PHI, l_t)
    k_ev_base, k_int_base = _batch_k_grids(None, float(CLS_K_MAX), sab, l_base)
    tau0, chi = 14000.0, 14000.0 - 280.0  # the batch path's fixed anchors (_batch_k_grids)
    k_base_np = np.asarray(k_ev_base, dtype=np.float64)
    k_top = float(max(3000.0, 2.0 * l_t) / tau0)
    # The base grid is linear (Δk = dkn2) to its q_cmb and log-spaced above: the damping-tail
    # sources oscillate in k with period 2π/r_s ≈ 0.04/Mpc, which the log part (Δk ≈ 0.03)
    # aliases (unlensed EE +80 % at ℓ 5000–6000). The high-ℓ projection therefore uses the
    # base modes to q_cmb plus new modes: linear Δk = dkn2 to q_cmb_ext, log above it.
    q_cmb, dkn2 = _evolve_q_smooth_start(sab)
    q_ext = min(k_top, max(q_cmb, CLS_HIGHL_Q_SMOOTH * sab))
    k_lin = np.zeros(0)
    if q_ext > q_cmb * (1 + 1e-9):  # else (lmax_out ≲ 1250) the base grid's linear part covers k_top
        n_lin = max(1, int(np.ceil((q_ext - q_cmb) / dkn2 - 1e-9)))
        k_lin = q_cmb + (q_ext - q_cmb) * np.arange(1, n_lin + 1) / n_lin
    k_log = np.zeros(0)
    if k_top > q_ext * (1 + 1e-9):
        dlog = np.log(1.0 + 1.0 / (12.0 * sab**2))  # CAMB: log(1 + dksmooth/q_cmb), Want_CMB
        n_log = max(1, int(np.ceil(np.log(k_top / q_ext) / dlog - 1e-9)))
        k_log = q_ext * (k_top / q_ext) ** (np.arange(1, n_log + 1) / n_log)
    k_hi = np.concatenate([k_lin, k_log])
    k_ev_np = np.concatenate([k_base_np, k_hi])
    n_base = int(k_base_np.size)
    hi_idx = np.concatenate([np.nonzero(k_base_np <= q_cmb * (1 + 1e-9))[0],
                             n_base + np.arange(k_hi.size)]).astype(np.int32)
    if not np.all(np.diff(k_ev_np[hi_idx]) > 0):
        raise RuntimeError("highl_plan: the high-ℓ source modes are not increasing in k")
    hi_mult = int(np.ceil(k_top / CLS_HIGHL_K_PER_MULT - 1e-9))
    # Int-q grid of the high-ℓ projection: CAMB SetkValuesForInt for maximum_l = l_t, from
    # where j_ℓ of the lowest added ℓ can first be non-zero (x ≥ ℓ − 4.2 ℓ^⅓, for χ ≤ 16700 Mpc).
    k_int_full = set_k_values_for_int(float(k_base_np[0]), k_top, tau0, chi, int_sample_boost=1.0,
                                      accuracy_target=1, accurate_polarization=True,
                                      accurate_reionization=True, maximum_l=l_t)
    ell_base = _make_ell_sparse(l_base)
    ell_t_hi, step = _highl_ell_samples(ell_base[-1], l_t)
    ell_p_hi, _ = _highl_ell_samples(l_t, l_p, step) if l_p > l_t else ([], None)
    l_first = ell_t_hi[0]
    k_lo = 0.9 * (l_first - 4.2 * l_first ** (1.0 / 3.0)) / 15000.0
    k_int_hi_np = np.asarray(k_int_full, dtype=np.float64)
    k_int_hi_np = k_int_hi_np[k_int_hi_np >= k_lo]
    k_lim = highl_lens_k_max(lmax_out)
    # Same log density as the base Limber grid (160 points from k_ev[0] to CLS_K_MAX_LIMBER).
    n_lim = int(np.ceil(160 * np.log(k_lim / k_base_np[0]) / np.log(CLS_K_MAX_LIMBER / k_base_np[0])))
    with jax.enable_x64(True), jax.ensure_compile_time_eval():
        k_ev = jnp.asarray(k_ev_np, dtype=jnp.float64)
        k_int_hi = jnp.asarray(k_int_hi_np, dtype=jnp.float64)
    return dict(l_base=l_base, lmax_out=lmax_out, l_t=l_t, l_p=l_p, k_ev=k_ev, n_base=n_base,
                hi_idx=tuple(int(i) for i in hi_idx), hi_mult=hi_mult, k_int=k_int_base,
                k_int_hi=k_int_hi, ell_base=tuple(ell_base), ell_t_hi=tuple(ell_t_hi),
                ell_p_hi=tuple(ell_p_hi), k_limber_max=float(k_lim), n_limber=n_lim)


def _log_slope_continuation(cl: jnp.ndarray, l_from: int, l_to: int, *, span: int = 300) -> jnp.ndarray:
    """``cl`` (length l_from + 1) continued to l_to with its log slope over the last ``span``
    ℓ (never rising): the unlensed damping tail past the projected range."""
    c1 = jnp.maximum(cl[l_from], 1e-300)
    c0 = jnp.maximum(cl[l_from - span], 1e-300)
    slope = jnp.minimum(jnp.log(c1 / c0) / span, 0.0)
    ell = jnp.arange(l_from + 1, l_to + 1, dtype=jnp.float64)
    tail = jnp.where(cl[l_from] > 0.0, cl[l_from] * jnp.exp(slope * (ell - l_from)), 0.0)
    return jnp.concatenate([cl, tail])


def compute_cmb_cls_batch(
    ombh2_b: jnp.ndarray,
    omch2_b: jnp.ndarray,
    h_b: jnp.ndarray,
    tau_reio_b: jnp.ndarray,
    ns_b: jnp.ndarray,
    As_b: jnp.ndarray,
    w0_b: jnp.ndarray,
    wa_b: jnp.ndarray,
    *,
    lmax: int = 2508,
    n_k: int | None = None,
    k_max: float = CLS_K_MAX,
    limber_ell: int = CLS_LIMBER_ELL,
    limber_ell_phi: int = CLS_LIMBER_ELL_PHI,
    source_accuracy_boost: float = CLS_SOURCE_ACCURACY_BOOST,
    time_step_boost: float = CLS_TIME_STEP_BOOST,
    evolve_step_boost: float = CLS_EVOLVE_STEP_BOOST,
    l_sample_boost: float = CLS_L_SAMPLE_BOOST,
    late_source_cutoff: bool = CLS_LATE_SOURCE_CUTOFF,
    evolve_method: str = CLS_EVOLVE_METHOD,
    k_chunk: int = CLS_BATCH_K_CHUNK,
) -> dict[str, jnp.ndarray]:
    """Batched EB Cls over a leading cosmology axis (B walkers).

    ``jax.vmap`` over ``compute_cmb_cls`` with on-device Recfast and
    ``device_grids=True`` (no sequential ``pure_callback``). Static Evolve_q /
    Int-q grids are built once on the host and closed over so the mapped body
    never calls ``np.asarray`` on tracers. Returns the same keys as
    ``compute_cmb_cls`` with a leading batch dimension on each spectrum array
    (``ell`` is shared / broadcast).

    Intended for ensemble / multi-walker proposals; see
    ``gpucosmo.jax.mcmc.ensemble_stub`` for the sampler hook.
    """
    k_ev, k_int_arr = _batch_k_grids(
        None if n_k is None else int(n_k),
        float(k_max),
        float(source_accuracy_boost),
        int(lmax),
    )

    return _compute_cmb_cls_batch_core(
        ombh2_b, omch2_b, h_b, tau_reio_b, ns_b, As_b, w0_b, wa_b,
        k_ev, k_int_arr,
        lmax=lmax,
        n_k=n_k,
        k_max=k_max,
        limber_ell=limber_ell,
        limber_ell_phi=limber_ell_phi,
        source_accuracy_boost=source_accuracy_boost,
        time_step_boost=time_step_boost,
        evolve_step_boost=evolve_step_boost,
        l_sample_boost=l_sample_boost,
        late_source_cutoff=late_source_cutoff,
        evolve_method=evolve_method,
        k_chunk=k_chunk,
    )


@partial(
    jax.jit,
    static_argnames=(
        "lmax",
        "n_k",
        "k_max",
        "limber_ell",
        "limber_ell_phi",
        "source_accuracy_boost",
        "time_step_boost",
        "evolve_step_boost",
        "l_sample_boost",
        "late_source_cutoff",
        "evolve_method",
        "k_chunk",
    ),
)
def _compute_cmb_cls_batch_core(
    ombh2_b, omch2_b, h_b, tau_reio_b, ns_b, As_b, w0_b, wa_b, k_ev, k_int_arr,
    *,
    lmax,
    n_k,
    k_max,
    limber_ell,
    limber_ell_phi,
    source_accuracy_boost,
    time_step_boost,
    evolve_step_boost,
    l_sample_boost,
    late_source_cutoff,
    evolve_method,
    k_chunk,
):
    """Jitted vmap body; the k grids arrive as (host-built) array arguments."""

    def _one(ombh2, omch2, h, tau_reio, ns, As, w0, wa):
        return compute_cmb_cls(
            ombh2=ombh2,
            omch2=omch2,
            h=h,
            tau_reio=tau_reio,
            ns=ns,
            As=As,
            w0=w0,
            wa=wa,
            lmax=lmax,
            n_k=n_k,
            k_max=k_max,
            use_recfast_thermo=True,
            limber_ell=limber_ell,
            limber_ell_phi=limber_ell_phi,
            source_accuracy_boost=source_accuracy_boost,
            time_step_boost=time_step_boost,
            evolve_step_boost=evolve_step_boost,
            l_sample_boost=l_sample_boost,
            late_source_cutoff=late_source_cutoff,
            device_grids=True,
            k_evolve=k_ev,
            k_int=k_int_arr,
            evolve_method=evolve_method,
            k_chunk=k_chunk,
        )

    return jax.vmap(_one)(
        ombh2_b, omch2_b, h_b, tau_reio_b, ns_b, As_b, w0_b, wa_b
    )


# Grid construction is host-side NumPy, so the public batch entry is not
# itself jitted; the heavy vmapped body (``_compute_cmb_cls_batch_core``) is.
# Kept as an alias for existing callers.
compute_cmb_cls_batch_jit = compute_cmb_cls_batch


__all__ = [
    "CLS_AMP_SCALE",
    "prepare_cmb_sources",
    "CLS_EVOLVE_METHOD",
    "CLS_BATCH_K_CHUNK",
    "CLS_K_CHUNK",
    "CLS_EVOLVE_STEP_BOOST",
    "CLS_HIERARCHY",
    "CLS_INTEGRATE_ON_OUTPUT",
    "CLS_EVOLVE_KDT_TARGET",
    "CLS_K_MAX",
    "CLS_LIMBER_ELL",
    "CLS_LIMBER_ELL_PHI",
    "CLS_LATE_SOURCE_CUTOFF",
    "CLS_L_SAMPLE_BOOST",
    "CLS_N_K",
    "CLS_N_STEPS",
    "CLS_N_SUB",
    "CLS_SOURCE_ACCURACY_BOOST",
    "CLS_SOURCE_FAILURE",
    "CLS_TIME_STEP_BOOST",
    "SOURCE_FAILURE_POLICIES",
    "compute_cmb_cls",
    "compute_cmb_cls_batch",
    "compute_cmb_cls_batch_jit",
    "compute_cmb_cls_jit",
    "set_k_values_for_sources",
    "solver_ok",
]
