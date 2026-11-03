"""Likelihood bundles for the μ–Σ modified-gravity analyses of arXiv:2411.12026.

Data combinations (Ishak et al. 2024, Table 2):

* DESI(FS+BAO)+BBN+ns10: DESI DR1 full-shape P0/P2 + post-reconstruction BAO
  (6 bins) + Lyα BAO, with the Schöneberg 2024 BBN prior and n_s ~ N(0.9649, 0.042);
* CMB-nl: Planck 2018 plik TTTEEE + low-ℓ TT (Commander) + low-ℓ EE (SimAll),
  no CMB lensing likelihood ("nl"); optionally DESI and/or DES-SN5YR. As in the
  official chains (ISiTGR ``NonLinear: NonLinear_none``), the primary CMB is
  lensed by the linear φφ.

Parameters: the theory parameters come first (so the DALI approximation expands a
contiguous block), then the Planck plik nuisances, then the full-shape bias
parameters (b1p, b2p, bsp per tracer; α0, α2, SN0, SN2 are marginalized
analytically inside the likelihood). With CMB the basis is CosmoMC's
(ln10¹⁰A_s, n_s, 100θ_MC, ω_b, ω_c, τ) and H0 is derived from θ_MC; without
CMB it is (H0, ω_b, ω_c, ln10¹⁰A_s, n_s). μ0, Σ0 (U[−3, 3], hard prior
μ0 < 2Σ0 + 1) and optionally w0, wa follow.

The bundle exposes the same interface as ``cmb.bundle.build_desi_full_gpu_bundle``
(``param_names``, ``prior_lo/hi``, ``fiducial``, ``log_prior_fn``,
``log_post_batch_fn``, ``log_like_from_theory_fn``) plus ``dali``, the
adapter ``scripts/run_dali_hmc.py`` uses to build/evaluate the generic
DALI expansion (``inference.dali_expansion``) of the expensive theory block.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from gpucosmo.jax.background.cosmology import (
    BackgroundParams,
    background_state,
    comoving_distance,
    comoving_distance_cumulative,
    cumulative_distance_nodes,
    hubble_km_s_mpc,
    load_background_tables,
    rs_drag,
    transverse_distance,
)

jax.config.update("jax_enable_x64", True)

# Paper Table 1 priors (boxes narrowed to the validity range of our background
# tables where the paper's are wider: ω_b ∈ [0.005, 0.04], ω_c ∈ [0.05, 0.3]).
COSMO_BOX = {
    "H0": (20.0, 100.0), "ombh2": (0.005, 0.04), "omch2": (0.05, 0.3), "logA": (1.61, 3.91),
    "ns": (0.8, 1.2), "theta_MC_100": (0.5, 10.0), "tau": (0.01, 0.8), "w": (-3.0, 1.0),
    "wa": (-3.0, 2.0), "mu0": (-3.0, 3.0), "Sigma0": (-3.0, 3.0), "r": (0.0, 3.0),
}
COSMO_FID = {"H0": 68.0, "ombh2": 0.02218, "omch2": 0.119, "logA": 3.04, "ns": 0.9649,
             "theta_MC_100": 1.04109, "tau": 0.0544, "w": -1.0, "wa": 0.0, "mu0": 0.0,
             "Sigma0": 0.0, "r": 0.01}
# Extended parameters (``MGConfig.ext_sampled``): Σm_ν [eV], N_eff, Ω_k, sterile m_eff [eV], in
# this order after the base parameters. Boxes are clipped to the background tables' neutrino
# corrections (``ext_box_limits``).
EXT_NAMES = ("mnu", "nnu", "omk", "meffsterile")
COSMO_BOX.update({"mnu": (0.0, 5.0), "nnu": (0.05, 10.0), "omk": (-0.3, 0.3), "meffsterile": (0.0, 3.0)})
COSMO_FID.update({"mnu": 0.06, "nnu": 3.044, "omk": 0.0, "meffsterile": 0.1})
# DES basis (``MGConfig.basis = "des"``, the DES Y3 chains' parameters and priors): total-matter
# Ωm (with the massive neutrinos), h, Ωb, n_s, 10⁹ A_s and Ω_ν h² in three degenerate eigenstates.
DES_THEORY_NAMES = ["Omega_m", "h", "Omega_b", "ns", "As_1e9", "omnuh2"]
COSMO_BOX.update({"Omega_m": (0.1, 0.9), "h": (0.55, 0.91), "Omega_b": (0.03, 0.07), "As_1e9": (0.5, 5.0),
                  "omnuh2": (0.0006, 0.00644)})
COSMO_FID.update({"Omega_m": 0.3, "h": 0.69, "Omega_b": 0.048, "As_1e9": 2.19, "omnuh2": 0.00083})
DES_NS_BOX = (0.87, 1.07)
# HSC basis (``MGConfig.basis = "hsc"``, the HSC Y3 × SDSS 3x2pt chains of Zhang et al. 2025): total-matter
# Ωm, ω_c, ω_b, n_s, ln(10¹⁰A_s) and the HMcode-2016 baryon amplitude A_b; h derived from
# Ωm h² = ω_c + ω_b + Σm_ν/93.14 (their CosmoSIS consistency relation), Σm_ν = 0.06 eV in one state.
HSC_THEORY_NAMES = ["Omega_m", "omch2", "ombh2", "ns", "log1e10As", "A_bary"]
# "hsc_takahashi": the 2x2pt chain (Takahashi halofit, no baryon parameter)
HSC_TAKAHASHI_THEORY_NAMES = HSC_THEORY_NAMES[:5]
HSC_BASES = ("hsc", "hsc_takahashi")
HSC_MNU = 0.06
COSMO_BOX.update({"log1e10As": (1.0, 5.0), "A_bary": (1.5, 3.13)})
COSMO_FID.update({"log1e10As": 3.04, "A_bary": 2.5})
NEUTRINO_MASS_FAC = 94.06410581217612  # CAMB / CosmoSIS consistency: ω_ν = Σm_ν (N_eff/3)^¾ / fac
# BBN: the official chains' ``bbn_likelihoods.schoneberg2024``, a Gaussian in
# (ω_b, N_eff) evaluated at the fixed N_eff = 3.044, i.e. the conditional
# Gaussian in ω_b (identical up to a constant). The paper's Table 1 quotes
# N(0.02218, 0.00055), 0.37σ higher than what the chains used.
BBN_SCHONEBERG2024 = dict(mean=(0.02196, 3.034),
                          cov=((4.0311226e-07, 7.30390042e-05), (7.30390042e-05, 0.0452831584)))
N_EFF = 3.044


def bbn_conditional(mean, cov, nnu: float = N_EFF) -> tuple[float, float]:
    """ω_b Gaussian of a 2D (ω_b, N_eff) BBN likelihood at fixed N_eff."""
    (m_b, m_n), ((c_bb, c_bn), (_, c_nn)) = mean, cov
    return float(m_b + c_bn / c_nn * (nnu - m_n)), float(np.sqrt(c_bb - c_bn**2 / c_nn))


BBN_PRIOR = bbn_conditional(**BBN_SCHONEBERG2024)  # ≈ (0.021976, 0.000534)
BBN_PRIOR_PAPER = (0.02218, 0.00055)  # Table 1 text; ``MGConfig(bbn_prior="paper")``
NS10_PRIOR = (0.9649, 0.042)  # n_s, 10× Planck σ
# Finite-difference steps of the DALI expansion (~1σ of the target posteriors).
STEPS_NOCMB = {"H0": 0.7, "ombh2": 5e-4, "omch2": 4e-3, "logA": 0.1, "ns": 0.03,
               "w": 0.1, "wa": 0.4, "mu0": 0.3, "Sigma0": 0.3,
               "Omega_m": 0.02, "h": 0.04, "Omega_b": 0.007, "As_1e9": 0.3, "omnuh2": 0.0005,
               "log1e10As": 0.1, "A_bary": 0.3, "mnu": 0.15, "nnu": 0.3, "omk": 0.05, "meffsterile": 0.3}
# HSC Y3 final (order ≥ 2) expansion: ~1σ of its broad posterior (S8-degenerate Ωm–A_s, prior-
# dominated ω_b, n_s, A_b), so the polynomial spans the posterior; the MAP stage keeps STEPS_NOCMB.
STEPS_HSC = {"Omega_m": 0.05, "omch2": 0.008, "ombh2": 0.003, "ns": 0.015, "log1e10As": 0.15, "A_bary": 0.4}
# DES Y3 final (order ≥ 2) expansion: ~1σ of its posterior (prior-wide in h, Ω_b, n_s, Ω_ν h²). Chosen with
# second-order expansions scored at the exact draws of a first run (steps of the MAP stage, at the MAP):
# importance-weight ESS of exact vs DALI likelihood
# 0.18 → 0.29 with these steps, ≥ one step inside the box and ln Ωm; 0.57 with the theory's spline-
# coefficient outputs (the likelihood takes P(k) at the nodes); 1.5σ or 2σ steps, box-centred h, Ω_b,
# n_s, Ω_ν h² or A_s factored out with ln A_s were worse.
STEPS_DES = {"Omega_m": 0.03, "h": 0.08, "Omega_b": 0.009, "ns": 0.05, "As_1e9": 0.3, "omnuh2": 0.0015}
STEPS_CMB = {"logA": 0.015, "ns": 0.004, "theta_MC_100": 3e-4, "ombh2": 1.5e-4,
             "omch2": 1.2e-3, "tau": 0.007, "w": 0.06, "wa": 0.25, "mu0": 0.25, "Sigma0": 0.1,
             "r": 0.02, "mnu": 0.05, "nnu": 0.15, "omk": 0.005, "meffsterile": 0.15}
FS_NUIS_FID = {"BGS_z0": (1.03599, 0.0, 0.0), "LRG_z0": (1.23666, 0.0, 0.0),
               "LRG_z1": (1.12281, 0.0, 0.0), "LRG_z2": (1.01724, 0.0, 0.0),
               "ELG_z1": (0.51609, 0.0, 0.0), "QSO_z0": (0.84509, 0.0, 0.0)}


@dataclass(frozen=True)
class MGConfig:
    desi_fs: bool = True
    cmb: str | None = None  # None | "pr3" (plik) | "pr4" (HiLLiPoP + LoLLiPoP)
    sn: str | None = None  # None | "desy5" | "pantheonplus" | "union3"
    mg: bool = True
    w0wa: bool = False
    bbn_ns_priors: bool | None = None  # default: on without CMB
    bbn_prior: str = "schoneberg2024"  # | "paper" (runs made before the switch)
    cmb_nonlinear_lens: bool = False  # official chains: NonLinear_none
    # HMcode version of the non-linear lensing φφ (CAMB halofit_version): "mead2020" (CAMB
    # default) or "mead2016" (the official DESI ΛCDM chains' yaml).
    cmb_nonlinear_model: str = "mead2020"
    # ACT DR6 CMB lensing on the jax_eb φφ (linear unless ``cmb_nonlinear_lens``):
    # act_dr6_lenslike ``variant`` / ``lens_only``. act_baseline + lens_only is candl's
    # ``ACT_DR6_Lens``; every other combination (ACT+Planck PR4 ``actplanck_*``, the
    # CMB-dependent likelihood corrections of lens_only=False) is ``act_dr6_lens_jax``.
    cmb_lensing: bool = False
    cmb_lensing_variant: str = "act_baseline"
    cmb_lensing_lens_only: bool = True
    walker_batch: int = 0  # likelihood evaluation (0 = vmap all walkers)
    # Set from a Cobaya yaml (``inference.cobaya_yaml``); None = module defaults.
    bbn_gauss: tuple[float, float] | None = None  # overrides bbn_prior
    ns_gauss: tuple[float, float] | None = None  # overrides NS10_PRIOR
    tau_fixed: float = 0.0544  # without CMB
    mg_hard_prior: bool = True  # μ0 < 2Σ0 + 1
    sz_prior: bool = True  # Planck plik ksz_norm + 1.6 A_sz ~ N(9.5, 3)
    # DES basis: hard cut Ω_b h² ≤ ombh2_max (None = none). The DES Y3 chains' CosmoSIS pipeline fails
    # beyond the end of bbn_consistency's BBN table (ω_b = 0.04), so they have no samples there.
    ombh2_max: float | None = None
    # GPU batching (throughput vs memory; RTX 5070 Laptop, outputs/bench_*.py): k modes
    # per vmapped chunk in the CMB C_ℓ and linear P(k) solves (0 = all modes at once).
    # Chunking k inside a walker vmap is pathological for the CMB (≈ 13 s/walker vs
    # 0.6 s/walker unchunked with 4 walkers), so the CMB solve is unchunked.
    cmb_k_chunk: int = 0
    pk_k_chunk: int = 64
    # One P(k) solver with the largest step budget for all modes (≤ 3.5e-6 from
    # the banded solve; 2.4× smaller program) instead of one copy per k band.
    pk_single_band: bool = False
    # Walkers vmapped together per step of a sequential map (0 = all) for the CMB
    # solve, the P(k) solve (``walker_batch`` sets the likelihood evaluation).
    cmb_walker_batch: int = 4  # PR4 (3.3 GB HiLLiPoP C on the GPU) yamls resolve to 8
    pk_walker_batch: int = 4
    # DESI FS in the theory block as the REPT multipole tables (bias-independent, cosmology
    # only: ``DESIDR1FullShape.theory_tables``) instead of the linear spectra: the DALI expansion
    # then expands the tables, and the one-loop integrals run once per exact theory call
    # instead of once per DALI evaluation (4.0 of the 5.1 ms per DALI point on an
    # RTX 3070). Exact evaluations are unchanged; DALI expansions built either way are not
    # interchangeable.
    fs_tables: bool = False
    # CMB walker groups through ``get_theory_spectra_jax_eb_batch`` (all modes of
    # the group's cosmologies batched in one evolve; ``cls_jax.SOURCE_LAYOUTS``);
    # None = vmap the single-cosmology pipeline. Measured (RTX 5070 Laptop, 4
    # walkers): "flat" 3.4 GB / 0.73 s per walker, "w_outer" 5.9 GB / 0.60 s,
    # "k_outer" 3.1 GB / 12.5 s (XLA slow path). ``autosize`` picks per device.
    cmb_layout: str | None = "flat"
    # Leave HiLLiPoP's 3.3 GB inverse covariance off the device until an exact or default-
    # DALI likelihood needs it (samplers on ``dali_reduce.CompressedDALI``
    # never do).
    defer_hillipop_matrix: bool = False
    # CMB spectra for user / Cobaya-port likelihoods (``Likelihood.requirements()["cmb"]``):
    # computed without a built-in CMB likelihood too (``cmb_theory``: CMB parameter basis), with
    # κκ and lensed TT/EE/BB/TE to ``cmb_x_lmax`` (0: the built-ins' outputs only). Set by
    # ``build_mg_bundle`` from the likelihoods it is given.
    cmb_theory: bool = False
    cmb_x_lmax: int = 0
    # Linear P(k, z), σ8(z), f σ8(z) at these redshifts for user / Cobaya-port likelihoods
    # (``requirements()["pk"]``; the ``"pk"`` theory block, ``eb.growth_jax``); shares the DESI
    # full-shape P(k) solve when both are present.
    pk_z: tuple = ()
    # ... beyond the solver's k ≤ 2/Mpc to ``pk_kmax`` (sparse evolved modes,
    # ``linear_pk_jax.hik_k_nodes``, interpolated as the ratio to the Eisenstein–Hu shape, which
    # continues past pk_kmax for HMcode) and the non-linear P_m (HMcode ``cmb_nonlinear_model``,
    # ``growth_jax.nonlinear_lnpk``) on that grid.
    pk_kmax: float = 0.0
    pk_nonlinear: bool = False
    # Primordial tensor modes (``r`` sampled, CMB basis only; ``eb.tensors_jax``): the unlensed tensor
    # TT/TE/EE/BB (ℓ ≤ ``TENSOR_LMAX``) are added to the lensed scalar spectra, as CAMB's "total".
    tensors: bool = False
    # The CMB is lensed on the solver's accurate high-ℓ path by default
    # (``jax_eb_theory.ACCURATE_LENSING``); True reproduces the base path of runs made before
    # 2026-10-17 (lensed TT/EE at ℓ 1000–2509 biased +1.5…+7.8e-4 vs converged CAMB, the ACT DR6
    # lensing corrections' BB −7 % at ℓ 2000–3000). Models compared by evidence: same setting.
    cmb_legacy_lensing: bool = False
    # Theory parameter basis: "default" (above) or "des" (``DES_THEORY_NAMES``: the DES Y3 chains'
    # Ωm, h, Ωb, n_s, 10⁹A_s, Ω_ν h² with three degenerate massive neutrinos). The "des" basis feeds
    # user likelihoods only (the built-in theory blocks fix Σm_ν = 0.06 eV in one eigenstate).
    # "hsc" / "hsc_takahashi": the HSC Y3 × SDSS chains' Ωm, ω_c, ω_b, n_s, ln 10¹⁰A_s (+ A_bary for
    # the 3x2pt's HMcode-2016; the 2x2pt's Takahashi halofit has none), user likelihoods only.
    basis: str = "default"
    # Neutrinos (CAMB ``set_cosmology``): ``nu_hierarchy`` "degenerate" (``num_massive_nu``
    # equal-mass states), "normal" or "inverted" (two-eigenstate approximations), ``nu_sterile``: a
    # thermal massive sterile neutrino (``meffsterile``, ΔN_eff = nnu − 3.044).
    nu_hierarchy: str = "degenerate"
    num_massive_nu: int = 1
    nu_sterile: bool = False
    # Extended parameters sampled (subset of EXT_NAMES, default basis), and fixed values away
    # from the defaults (mnu 0.06, nnu 3.044, omk 0, meffsterile 0) as ((name, value), ...).
    ext_sampled: tuple = ()
    ext_fixed: tuple = ()
    # With N_eff sampled: the BBN likelihood as the joint Gaussian (mean (ω_b, N_eff), 2×2 cov)
    # instead of ``bbn_gauss`` (the ω_b conditional at fixed N_eff).
    bbn_gauss2d: tuple | None = None


def nu_model_of(cfg: MGConfig):
    """The static ``massive_nu.NeutrinoModel`` of ``cfg``."""
    from gpucosmo.jax.eb.massive_nu import NeutrinoModel

    return NeutrinoModel(cfg.nu_hierarchy, int(cfg.num_massive_nu), sterile=bool(cfg.nu_sterile))


def ext_box_limits(tables, name: str) -> tuple[float, float] | None:
    """Range of an extended parameter covered by the background tables' z*/z_drag neutrino
    corrections (None: no limit)."""
    if name == "omk" or tables.nu_corr_axes is None:
        return None
    axes = tables.nu_corr_axes
    if name not in axes:
        return None
    rng = tables.nu_corr_ranges[2 + axes.index(name)]
    return float(rng[0]), float(rng[1])


def cmb_on(cfg: MGConfig) -> bool:
    """The theory block has CMB spectra (a built-in CMB likelihood or ``cmb_theory``)."""
    return bool(cfg.cmb) or bool(cfg.cmb_theory)


# Theory blocks a user likelihood may require (``Likelihood.requirements()``).
REQUIREMENT_BLOCKS = ("cmb", "fs", "pk")
PK_KMAX_EXT = 50.0  # 1/Mpc: the extended P(k) grid's limit
PK_N_EXT = 256  # points of that log grid, from 1e-4/Mpc


def pk_ext_grid(k_max: float) -> np.ndarray:
    """The extended P(k) grid [1/Mpc] (``MGConfig.pk_kmax``)."""
    return np.geomspace(1e-4, float(k_max), PK_N_EXT)


def cmb_lmax_supported() -> int:
    """The highest multipole of the CMB spectra the solver provides to user / Cobaya likelihoods
    (past 2508 + LENS_MARGIN the solver switches to its extended high-ℓ path, ``cls_jax.highl_plan``)."""
    from gpucosmo.jax.cmb.jax_eb_theory import max_lmax_extra

    return int(max_lmax_extra())


def extra_requirements(extra) -> dict:
    """Merged ``requirements()`` of user likelihoods: {"cmb": {"lmax", "spectra"}, "fs": {},
    "pk": {"z", "k_max", "nonlinear"}}."""
    out: dict = {}
    for L in extra:
        req = L.requirements() if hasattr(L, "requirements") else {r: {} for r in L.requires}
        for k, v in (req or {}).items():
            if k not in REQUIREMENT_BLOCKS:
                raise ValueError(f"{L.label}: requires {k!r}; known theory blocks {list(REQUIREMENT_BLOCKS)}")
            cur = out.setdefault(k, {})
            if k == "cmb":
                v = v or {}
                cur["lmax"] = max(int(cur.get("lmax", 0)), int(v.get("lmax", 0)))
                cur["spectra"] = sorted(set(cur.get("spectra", ())) | set(v.get("spectra", ())))
            elif k == "pk":
                v = v or {}
                cur["z"] = sorted({round(float(z), 10) for z in list(cur.get("z", ())) + list(v.get("z", ()))})
                cur["k_max"] = max(float(cur.get("k_max", 0.0)), float(v.get("k_max", 0.0)))
                cur["nonlinear"] = bool(cur.get("nonlinear", False) or v.get("nonlinear", False))
    return out


def effective_config(cfg: MGConfig, extra) -> MGConfig:
    """``cfg`` with the CMB outputs the user likelihoods ``extra`` require."""
    import dataclasses

    allreq = extra_requirements(extra)
    over = {}
    pk = allreq.get("pk")
    if pk is not None:
        from gpucosmo.jax.eb.linear_pk_jax import LINPK_K_MAX

        if pk["k_max"] > PK_KMAX_EXT:
            raise ValueError(f"P(k) to k = {pk['k_max']}/Mpc requested; provided to {PK_KMAX_EXT}/Mpc")
        z = tuple(sorted(set(cfg.pk_z) | set(pk["z"])))
        if z != cfg.pk_z:
            over["pk_z"] = z
        kx = max(float(cfg.pk_kmax), float(pk["k_max"]) if pk["k_max"] > LINPK_K_MAX or pk["nonlinear"] else 0.0)
        if pk["nonlinear"] and kx < LINPK_K_MAX:
            kx = LINPK_K_MAX
        if kx != cfg.pk_kmax:
            over["pk_kmax"] = kx
        if pk["nonlinear"] and not cfg.pk_nonlinear:
            over["pk_nonlinear"] = True
    req = allreq.get("cmb")
    if req is None:
        return dataclasses.replace(cfg, **over) if over else cfg
    if not cfg.cmb and not cfg.cmb_theory:
        over["cmb_theory"] = True
    beyond = req["lmax"] > 2508 or bool(set(req["spectra"]) - {"tt", "te", "ee"})
    lx = max(int(cfg.cmb_x_lmax), int(req["lmax"]) if beyond else 0, 2508 if beyond else 0)
    if lx > cmb_lmax_supported():
        raise ValueError(f"CMB spectra to ℓ {lx} requested; the solver computes them to ℓ {cmb_lmax_supported()}")
    if lx != cfg.cmb_x_lmax:
        over["cmb_x_lmax"] = lx
    return dataclasses.replace(cfg, **over) if over else cfg


def shared_block(r: str, th: dict, static: dict | None = None) -> dict:
    """The built-in theory block ``r`` as user likelihoods receive it (``LikelihoodInputs.shared``):
    for "cmb", plus ``cl`` = {"ell", "tt", "te", "ee", "bb" (lensed D_ℓ, μK²), "pp"
    ([ℓ(ℓ+1)]² C_ℓ^φφ / 2π), "kk" (C_ℓ^κκ)} when the extended spectra were computed; for "pk",
    {"z", "k" (static NumPy: the redshifts / k [1/Mpc] of the rows / columns), "pk_m", "pk_cb"
    (linear P_m, P_cb [Mpc³], (n_z, n_k)), "sigma8", "fsigma8" (n_z,)} and, with ``pk_kmax``,
    "k_x" (static) with "pk_m_x" (linear; evolved modes to pk_kmax past the solver grid) and "pk_m_nl" (non-linear)."""
    blk = th[r]
    if r == "pk":
        st = (static or {}).get("pk", {})
        out = {"z": st.get("z"), "k": st.get("k"), "pk_m": jnp.exp(blk["lnpk_m"]), "pk_cb": jnp.exp(blk["lnpk_cb"]),
               "sigma8": blk["sigma8"], "fsigma8": blk["fsigma8"]}
        if "lnpk_m_x" in blk:  # extended grid (static "k_x"): linear and (if computed) non-linear P_m
            out.update(k_x=st.get("k_x"), pk_m_x=jnp.exp(blk["lnpk_m_x"]))
            if "lnpk_m_nl" in blk:
                out["pk_m_nl"] = jnp.exp(blk["lnpk_m_nl"])
        return out
    if r == "cmb" and "cmb_x" in blk:
        x = blk["cmb_x"]  # κκ, TT, EE, BB, TE
        blk = {**blk, "cl": {"ell": jnp.arange(x.shape[-1], dtype=jnp.float64), "kk": x[0], "pp": x[0] * (2.0 / jnp.pi),
                             "tt": x[1], "ee": x[2], "bb": x[3], "te": x[4]}}
    elif r == "cmb" and "dl_tt" in blk:  # TT/TE/EE to ℓ 2508 only (no extended spectra requested)
        cl = {"ell": jnp.arange(blk["dl_tt"].shape[-1], dtype=jnp.float64), "tt": blk["dl_tt"], "te": blk["dl_te"],
              "ee": blk["dl_ee"]}
        if "dl_kk" in blk:  # C_L^κκ (candl's) from the same solve
            cl.update(kk=blk["dl_kk"], pp=blk["dl_kk"] * (2.0 / jnp.pi))
        blk = {**blk, "cl": cl}
    return blk


def _theory_names(cfg: MGConfig):
    if cfg.ombh2_max is not None and cfg.basis != "des":
        raise ValueError("ombh2_max (the cut Ω_b h² ≤ ombh2_max) is defined in the DES basis")
    if cfg.basis in ("des",) + HSC_BASES:
        bad = [k for k, on in (("a CMB likelihood", cmb_on(cfg)), ("DESI full-shape", cfg.desi_fs),
                                ("supernovae", cfg.sn), ("the 'pk' block", bool(cfg.pk_z)), ("μ–Σ", cfg.mg),
                                ("w0wa", cfg.w0wa), ("tensors", cfg.tensors)) if on]
        if bad:
            raise ValueError(f"the {cfg.basis.upper()} parameter basis supports user likelihoods only; "
                             f"got {', '.join(bad)}")
        return list({"des": DES_THEORY_NAMES, "hsc": HSC_THEORY_NAMES,
                     "hsc_takahashi": HSC_TAKAHASHI_THEORY_NAMES}[cfg.basis])
    if cfg.basis != "default":
        raise ValueError(f"unknown parameter basis {cfg.basis!r}")
    if cmb_on(cfg):
        names = ["logA", "ns", "theta_MC_100", "ombh2", "omch2", "tau"]
    else:
        names = ["H0", "ombh2", "omch2", "logA", "ns"]
    bad = sorted(set(cfg.ext_sampled) - set(EXT_NAMES))
    if bad:
        raise ValueError(f"unknown extended parameters {bad}; known {list(EXT_NAMES)}")
    names += [n for n in EXT_NAMES if n in cfg.ext_sampled]
    if cfg.w0wa:
        names += ["w", "wa"]
    if cfg.mg:
        names += ["mu0", "Sigma0"]
    if cfg.tensors:
        if not cmb_on(cfg):
            raise ValueError("tensor modes (r) need the CMB theory block")
        names += ["r"]
    return names


# Tensor spectra are added to ℓ = TENSOR_LMAX: past it tensor BB is ≤ 1.1e-3 × (r/0.1) of the lensing BB
# (4.7e-3 at ℓ 600), tensor TT/EE ≲ 4e-5 of the scalar ones (``eb.tensors_jax``, validated to ℓ 1000).
TENSOR_LMAX = 1000


def add_tensor_spectra(out: dict, c: dict, nu_model=None) -> dict:
    """``out`` (``_cmb_out`` dict for one cosmology) with the unlensed tensor spectra of ``c``
    (cosmology dict with ``r``) added to TT/TE/EE (and BB in the extended blocks) for ℓ ≤ TENSOR_LMAX
    (or the arrays' length when shorter)."""
    from gpucosmo.jax.cmb.params import T_CMB_MUK
    from gpucosmo.jax.eb.tensors_jax import tensor_cls

    t = tensor_cls(c["ombh2"], c["omch2"], c["h"], c["ns"], c["As"], c["tau"], c["w0"], c["wa"], c["r"],
                   lmax=TENSOR_LMAX, ext=c.get("ext"), nu_model=nu_model)
    d = {k: t[k] * T_CMB_MUK**2 for k in ("tt", "ee", "bb", "te")}
    out = dict(out)
    for k in ("tt", "te", "ee"):
        n = min(TENSOR_LMAX + 1, out[f"dl_{k}"].shape[-1])
        out[f"dl_{k}"] = out[f"dl_{k}"].at[:n].add(d[k][:n])
    for key in ("cmb_ext", "cmb_x"):  # rows: κκ, TT, EE, BB, TE
        if key in out:
            x = out[key]
            n = min(TENSOR_LMAX + 1, x.shape[-1])
            for row, k in ((1, "tt"), (2, "ee"), (3, "bb"), (4, "te")):
                x = x.at[row, :n].add(d[k][:n])
            out[key] = x
    return out


def _check_extra(extra, cfg: MGConfig, taken: list[str]) -> list[str]:
    """Validate user likelihoods; returns their parameter names (in order)."""
    from gpucosmo.likelihoods.base import Likelihood
    from gpucosmo.likelihoods.registry import BUILTIN

    labels, names = set(), []
    have = {"cmb": cmb_on(cfg), "fs": bool(cfg.desi_fs), "pk": bool(cfg.pk_z)}
    for L in extra:
        if not isinstance(L, Likelihood):
            raise TypeError(f"{L!r} is not a gpucosmo.likelihoods.Likelihood instance")
        lab = L.label
        if lab in labels or lab in BUILTIN:
            raise ValueError(f"likelihood name {lab!r} used twice (or by a built-in); set name=... per instance")
        labels.add(lab)
        for r in L.requires:
            if r not in have:
                raise ValueError(f"{lab}: requires {r!r}; known shared theory blocks {sorted(have)}")
            if not have[r]:
                raise ValueError(f"{lab}: requires the {r!r} theory block; the analysis has no "
                                 f"{'primary CMB' if r == 'cmb' else 'DESI full-shape'} likelihood")
        share = bool(getattr(L, "allow_shared_params", False))
        for n in L.params:
            if n in taken or n in names:
                if share:  # Cobaya semantics: one parameter for every likelihood naming it
                    continue
                raise ValueError(f"{lab}: parameter {n!r} already exists in the analysis")
            names.append(n)
    return names


def _analysis_nqmax(nu_model, names, lo, hi, ext_fixed: dict) -> int:
    """CAMB 2.0.4's momentum rule (``massive_nu.camb_nu_nqmax``) for the heaviest eigenstate the
    analysis reaches: the eigenstates at the upper edge of the Σm_ν box and the lower edge of the
    N_eff box (below the standard N_eff CAMB cools the states that carry it), or at their fixed
    values. 5 with a sterile neutrino, whose m/T grows without bound as ΔN_eff → 0."""
    from gpucosmo.jax.eb.constants import DEFAULT_MNU, DEFAULT_NNU
    from gpucosmo.jax.eb.massive_nu import camb_nu_nqmax

    if nu_model.sterile:
        return 5
    mnu = float(hi[names.index("mnu")]) if "mnu" in names else float(ext_fixed.get("mnu", DEFAULT_MNU))
    nnu = float(lo[names.index("nnu")]) if "nnu" in names else float(ext_fixed.get("nnu", DEFAULT_NNU))
    am = np.asarray(nu_model.resolve(jnp.asarray(mnu), jnp.asarray(nnu))["am"])
    return camb_nu_nqmax(float(np.max(am))) if am.size else 3


def build_mg_bundle(cfg: MGConfig = MGConfig(), packages_path: str | None = None,
                    priors: dict | None = None, likelihoods=()):
    """``priors``: {name: ParamPrior(lo, hi, ref, gauss)} overriding the module
    defaults (from ``cobaya_yaml.resolve_cobaya``). ω_b / ω_c boxes are clipped to
    the background tables' validity range (listed in ``bundle.notes``).

    ``likelihoods``: user ``gpucosmo.likelihoods.Likelihood`` instances added to the
    built-in ones. Their parameters follow the built-in ones (defaults from their
    ``params``, overridden by ``priors``); their ``theory`` outputs join the theory block
    under ``"ext"`` (so the DALI approximation expands them); each adds a component
    ``log_like_from_theory_fn(..., components=True)[name]``."""
    from gpucosmo.jax.cmb.params import PLIK_2018_PRIORS, SZ_PRIOR, nuisance_gaussian_priors

    extra = tuple(likelihoods or ())
    cfg = effective_config(cfg, extra)
    ext_used = set(cfg.ext_sampled) | set(dict(cfg.ext_fixed))
    if cfg.basis in ("des",) + HSC_BASES and ext_used:
        raise ValueError(f"the {cfg.basis} basis holds {sorted(ext_used)} at their defaults (its likelihood's "
                         "own theory)")
    for L in extra:
        held = ext_used & set(getattr(L, "fixed_ext", ()))
        if held:
            raise ValueError(f"{L.label}: its own theory holds {sorted(held)} fixed (``fixed_ext``); "
                             "they cannot be sampled or changed in this analysis")
    if cfg.nu_hierarchy == "degenerate" and int(cfg.num_massive_nu) == 0:
        from gpucosmo.jax.eb.constants import EXT_DEFAULTS

        if "mnu" in cfg.ext_sampled or dict(cfg.ext_fixed).get("mnu", EXT_DEFAULTS["mnu"]) != 0.0:
            raise ValueError("non-zero neutrino mass with num_massive_neutrinos = 0 (CAMB raises too): "
                             "fix mnu = 0 or use massive states")
        cfg = dataclasses.replace(cfg, num_massive_nu=1)  # Σm_ν = 0 makes every state massless (CAMB)
    th_names = _theory_names(cfg)
    nu_model = nu_model_of(cfg)
    for L in extra:
        want = getattr(L, "neutrino_model", None)
        if want is not None and "mnu" in ext_used and (nu_model.hierarchy, nu_model.num_massive) != tuple(want):
            raise ValueError(f"{L.label}: its own theory has {want[1]} {want[0]} massive neutrinos; Σm_ν is changed "
                             f"with {nu_model.num_massive} {nu_model.hierarchy} ones (set num_massive_neutrinos / "
                             "neutrino_hierarchy to match)")
    if cfg.basis == "des":  # the DES Y3 chains: three degenerate massive states
        from gpucosmo.jax.eb.massive_nu import NeutrinoModel

        nu_model = NeutrinoModel("degenerate", 3)
    ext_fixed = dict(cfg.ext_fixed)
    bad = sorted((set(ext_fixed) - set(EXT_NAMES)) | (set(ext_fixed) & set(cfg.ext_sampled)))
    if bad:
        raise ValueError(f"ext_fixed: unknown or also sampled {bad}")
    if ({"meffsterile"} & (set(cfg.ext_sampled) | set(ext_fixed))) and not cfg.nu_sterile:
        raise ValueError("meffsterile needs MGConfig(nu_sterile=True)")
    if cfg.nu_sterile and "mnu" in (set(cfg.ext_sampled) | set(ext_fixed)):
        raise ValueError("with a massive sterile neutrino Σm_ν is fixed at 0.06 eV (Planck's convention)")
    tables = load_background_tables(nu_model=nu_model)
    n_th = len(th_names)
    pkw = {} if packages_path is None else {"packages_path": packages_path}
    bbn_ns = (not cmb_on(cfg)) if cfg.bbn_ns_priors is None else cfg.bbn_ns_priors
    if cfg.basis in ("des",) + HSC_BASES:
        bbn_ns = bool(cfg.bbn_ns_priors)
        if bbn_ns:
            raise ValueError("BBN / n_s priors are defined in the default basis")

    # ---- components ----
    plik = lowl = None
    plik_names: tuple[str, ...] = ()
    aux = {}  # large arrays passed to the jitted functions as arguments
    _pending_ref = {}  # deferred HiLLiPoP centring (defer_hillipop_matrix)
    hil = lol = None
    if cfg.cmb == "pr4":
        import dataclasses as _dc

        from gpucosmo.jax.cmb.clipy_likelihood import build_clipy_lowl_loglike, load_clipy_lowl
        from gpucosmo.jax.cmb.pr4_jax import HILLIPOP_PARAMS, load_pr4

        hil, lol = load_pr4()
        if not cfg.defer_hillipop_matrix:
            aux["invkll"] = hil.invkll
        plik_names = tuple(HILLIPOP_PARAMS)
        cl = load_clipy_lowl(**pkw)
        lowl = lowl_smooth = build_clipy_lowl_loglike(_dc.replace(cl, ee_like=None))  # Commander TT
    elif cfg.cmb:
        from gpucosmo.jax.cmb.clipy_likelihood import build_clipy_lowl_loglike, load_clipy_lowl
        from gpucosmo.jax.cmb.clipy_plik import build_clipy_plik_loglike, load_clipy_plik

        cp = load_clipy_plik(**pkw)
        if cp is None:
            raise RuntimeError("Planck plik TTTEEE (clipy) not installed")
        plik = build_clipy_plik_loglike(cp)
        plik_names = tuple(cp.nuisance_names)
        cl = load_clipy_lowl(**pkw)
        lowl = build_clipy_lowl_loglike(cl)
        lowl_smooth = build_clipy_lowl_loglike(cl, smooth_ee=True)
    lens_ll = act = None
    lmax_extra = 0  # lensed CMB / κκ past ℓ 2508 for the ACT likelihood corrections
    candl_lens = cfg.cmb_lensing_variant == "act_baseline" and cfg.cmb_lensing_lens_only
    if cfg.cmb_lensing:
        if not cfg.cmb:
            raise ValueError("cmb_lensing needs a primary CMB likelihood (cmb='pr3' or 'pr4')")
        if candl_lens:
            from gpucosmo.jax.cmb.candl_likelihood import (
                build_candl_lens_loglike,
                load_candl_lens_likelihood,
            )

            lens_ll = build_candl_lens_loglike(load_candl_lens_likelihood())
        else:
            from gpucosmo.jax.cmb.act_dr6_lens_jax import (
                N_ELL,
                build_act_dr6_lens_loglike,
                load_act_dr6_lens,
            )

            act = load_act_dr6_lens(cfg.cmb_lensing_variant, cfg.cmb_lensing_lens_only)
            if act.needs_cmb:
                lmax_extra = N_ELL - 1
            elif np.any(act.a_kk[:, 2509:]):
                raise ValueError("ACT lensing bins reach past L 2508 (TheorySpectra.dl_kk)")
            lens_ll = build_act_dr6_lens_loglike(act)
    fs_like = z_fs = z_bao = None
    fs_names: list[str] = []
    if cfg.desi_fs:
        from gpucosmo.jax.eb.linear_pk_jax import linpk_k_grid
        from gpucosmo.jax.fullshape.desi_dr1 import FS_TRACERS, DESIDR1FullShape

        fs_like = DESIDR1FullShape(linpk_k_grid())
        for t in FS_TRACERS:
            fs_names += [f"{t}.b1p", f"{t}.b2p", f"{t}.bsp"]
        z_fs = fs_like.z_fs
        z_bao = np.asarray(fs_like.fid["bao_z"])
    sn_like = None
    if cfg.sn:
        from gpucosmo.jax.background.likelihoods import build_sn_loglike, load_w0wa_data

        data = load_w0wa_data(sn_datasets=(cfg.sn,), **({} if packages_path is None else
                                                          {"packages_path": packages_path}))
        sn_like = build_sn_loglike(data.sn[cfg.sn], tables)

    # ---- user likelihoods ----
    ext_names = _check_extra(extra, cfg, list(th_names) + list(plik_names) + fs_names)
    ext_param = {}
    for L in extra:
        for n in L.params:
            ext_param.setdefault(n, L.params[n])

    # ---- parameters and priors ----
    names = list(th_names) + list(plik_names) + fs_names + ext_names
    lo, hi, fid = [], [], []
    for n in th_names:
        box = DES_NS_BOX if (n == "ns" and cfg.basis == "des") else COSMO_BOX[n]
        lo.append(box[0]), hi.append(box[1]), fid.append(COSMO_FID[n])
    for n in plik_names:
        if cfg.cmb == "pr4":
            a_, b_, ref, _g = HILLIPOP_PARAMS[n]
        else:
            a_, b_, ref, _prop, _g = PLIK_2018_PRIORS[n]
        lo.append(a_), hi.append(b_), fid.append(ref)
    for n in fs_names:
        tr, kind = n.split(".")
        if kind == "b1p":
            lo.append(0.0), hi.append(3.0), fid.append(FS_NUIS_FID[tr][0])
        else:
            lo.append(-50.0), hi.append(50.0), fid.append(0.0)
    for n in ext_names:
        p = ext_param[n]
        lo.append(float(p.lo)), hi.append(float(p.hi)), fid.append(p.start)
    lo, hi, fid = (np.asarray(v, dtype=np.float64) for v in (lo, hi, fid))
    ndim = len(names)
    g_mu = np.zeros(ndim)
    g_sd = np.full(ndim, np.inf)
    if plik_names and cfg.cmb == "pr4":
        for i, n in enumerate(names):
            if n in HILLIPOP_PARAMS and HILLIPOP_PARAMS[n][3] is not None:
                g_mu[i], g_sd[i] = HILLIPOP_PARAMS[n][3]
    elif plik_names:
        mu_p, sd_p = nuisance_gaussian_priors(tuple(names))
        sel = np.isfinite(sd_p)
        g_mu[sel], g_sd[sel] = mu_p[sel], sd_p[sel]
    for i, n in enumerate(names):
        if n.endswith(".b2p") or n.endswith(".bsp"):
            g_mu[i], g_sd[i] = 0.0, 5.0
        elif n in ext_param and ext_param[n].gauss is not None:
            g_mu[i], g_sd[i] = ext_param[n].gauss
    notes = []
    if priors:
        missing = sorted(set(names) - set(priors) - set(ext_names))  # user likelihoods: their defaults
        if missing:
            hint = ""
            if cfg.cmb_theory and {"theta_MC_100", "tau"} & set(missing):
                hint = (" (a likelihood requires CMB spectra, so the analysis uses the CMB parameter basis: "
                        "sample theta_MC_100 and tau instead of H0)")
            raise ValueError(f"no prior given for {missing}{hint}")
        for i, n in enumerate(names):
            if n not in priors:
                continue
            p = priors[n]
            lo[i], hi[i], fid[i] = p.lo, p.hi, p.ref
            g_mu[i], g_sd[i] = p.gauss if p.gauss is not None else (0.0, np.inf)
    for n, v in ext_fixed.items():
        rng = ext_box_limits(tables, n)
        if rng is not None and not rng[0] <= v <= rng[1]:
            raise ValueError(f"{n} = {v:g} is outside the range the background tables / neutrino model support, "
                             f"[{rng[0]:g}, {rng[1]:g}]")
    ext_ranges = [(n, ext_box_limits(tables, n)) for n in cfg.ext_sampled]
    if nu_model.hierarchy != "degenerate" and "mnu" in cfg.ext_sampled:
        # CAMB's two-eigenstate hierarchy starts at Σm_ν > min + 1e-4 (one eigenstate below)
        ext_ranges.append(("mnu", (nu_model.min_mnu + 1e-6, np.inf)))
    for n, rng in [("ombh2", tables.ombh2_range), ("omch2", tables.omch2_range),
                   *[(n_, r_) for n_, r_ in ext_ranges if r_ is not None]]:
        if n not in names:  # DES basis: ω_b, ω_c derived (the DES theory computes its own background)
            continue
        i = names.index(n)
        if lo[i] < rng[0] or hi[i] > rng[1]:
            if n in EXT_NAMES and priors and n in priors:
                # a narrower uniform prior than the user's changes the evidence (ln of the width ratio)
                raise ValueError(f"the prior on {n}, [{lo[i]:g}, {hi[i]:g}], exceeds the range the background "
                                 f"tables / neutrino model support, [{rng[0]:g}, {rng[1]:g}]: narrow it")
            notes.append(f"{n} box [{lo[i]:g}, {hi[i]:g}] clipped to the background-table / model "
                         f"range [{rng[0]:g}, {rng[1]:g}]")
            lo[i], hi[i] = max(lo[i], rng[0]), min(hi[i], rng[1])
    if cfg.basis == "default":
        # Static, so the staged and the batched exact theories share it (``tables.model``).
        nu_model = dataclasses.replace(nu_model, nqmax=_analysis_nqmax(nu_model, names, lo, hi, ext_fixed))
        tables = dataclasses.replace(tables, nu_model=nu_model)
    bbn2d = None
    if bbn_ns:
        if "nnu" in names and cfg.bbn_gauss2d is None and cfg.bbn_gauss is None and cfg.bbn_prior == "schoneberg2024":
            cfg = dataclasses.replace(cfg, bbn_gauss2d=(tuple(BBN_SCHONEBERG2024["mean"]),
                                                        tuple(map(tuple, BBN_SCHONEBERG2024["cov"]))))
        if cfg.bbn_gauss2d is not None:
            if "nnu" not in names:
                raise ValueError("bbn_gauss2d needs N_eff sampled (ext_sampled with 'nnu')")
            m2, c2 = cfg.bbn_gauss2d
            bbn2d = (jnp.asarray([names.index("ombh2"), names.index("nnu")]),
                     jnp.asarray(m2, dtype=jnp.float64), jnp.asarray(np.linalg.inv(np.asarray(c2)), dtype=jnp.float64))
        else:
            if "nnu" in names:
                raise ValueError("N_eff sampled with a BBN prior at fixed N_eff: give bbn_gauss2d")
            g_mu[names.index("ombh2")], g_sd[names.index("ombh2")] = cfg.bbn_gauss or {
                "schoneberg2024": BBN_PRIOR, "paper": BBN_PRIOR_PAPER}[cfg.bbn_prior]
        g_mu[names.index("ns")], g_sd[names.index("ns")] = cfg.ns_gauss or NS10_PRIOR
    g_mask = jnp.asarray(np.isfinite(g_sd))
    g_mu_j = jnp.asarray(g_mu)
    g_sd_j = jnp.asarray(np.where(np.isfinite(g_sd), g_sd, 1.0))
    idx = {n: i for i, n in enumerate(names)}
    if cfg.cmb == "pr4":
        # Centre HiLLiPoP's float32 χ² on CAMB's lensed ΛCDM template with the fiducial
        # nuisances (``HiLLiPoP.loglike(ref=...)``): exact rewrite, float64-level accuracy.
        from gpucosmo.jax.cmb.lensing_jax import _load_highl_template
        from gpucosmo.jax.cmb.params import T_CMB_MUK

        tmpl = _load_highl_template()[: hil.lmax + 1] * T_CMB_MUK**2  # D_ℓ [μK²]: TT, EE, BB, TE
        fid_nuis = {n: float(fid[idx[n]]) for n in plik_names if n in idx}

        def _template_ref():
            return hil.reference(jnp.asarray(tmpl[:, 0]), jnp.asarray(tmpl[:, 1]), jnp.asarray(tmpl[:, 3]),
                                 {n: jnp.asarray(v) for n, v in fid_nuis.items()})
        # With the matrix deferred (compressed-dali samplers never use the exact HiLLiPoP) the
        # ~8 s centring pass through the 3.3 GB matrix waits until _with_matrix() needs it.
        if cfg.defer_hillipop_matrix:
            _pending_ref["fn"] = _template_ref
        else:
            aux["hil_ref"] = _template_ref()
    sz_idx = ((idx["ksz_norm"], idx["A_sz"]) if cfg.sz_prior and {"ksz_norm", "A_sz"} <= set(names)
              else None)
    prior_lo, prior_hi = jnp.asarray(lo), jnp.asarray(hi)
    extra_prior = [L for L in extra if L.has_log_prior()]

    def _ext_params(L, x):
        return {n: x[..., idx[n]] for n in L.params}

    def log_prior_fn(x):
        inside = jnp.all(x >= prior_lo, axis=-1) & jnp.all(x <= prior_hi, axis=-1)
        if cfg.w0wa:
            inside &= (x[..., idx["w"]] + x[..., idx["wa"]]) < 0.0
        if cfg.mg and cfg.mg_hard_prior:  # ISiTGR stability (paper's hard prior)
            inside &= x[..., idx["mu0"]] < 2.0 * x[..., idx["Sigma0"]] + 1.0
        if cfg.ombh2_max is not None:  # DES basis (_theory_names)
            inside &= x[..., idx["Omega_b"]] * x[..., idx["h"]] ** 2 <= cfg.ombh2_max
        z = (x - g_mu_j) / g_sd_j
        lp = -0.5 * jnp.sum(jnp.where(g_mask, z * z, 0.0), axis=-1)
        if bbn2d is not None:  # joint (ω_b, N_eff) BBN Gaussian
            ii, m2, icov2 = bbn2d
            d2 = x[..., ii] - m2
            lp = lp - 0.5 * jnp.einsum("...i,ij,...j->...", d2, icov2, d2)
        if sz_idx is not None:
            fac, mu_sz, sd_sz = SZ_PRIOR
            comb = x[..., sz_idx[0]] + fac * x[..., sz_idx[1]]
            lp = lp - 0.5 * ((comb - mu_sz) / sd_sz) ** 2
        for L in extra_prior:
            lp = lp + L.log_prior(_ext_params(L, x))
        return jnp.where(inside, lp, -jnp.inf)

    # ---- cosmology from the parameter vector ----
    def cosmo(x):
        p = {n: x[..., idx[n]] for n in th_names}
        if cfg.basis == "des":
            from gpucosmo.jax.eb import constants as _C

            h = p["h"]
            omnuh2 = p["omnuh2"]  # Σm_ν such that the solver's three degenerate states have this ω_ν
            # Σm_ν for reference (CosmoSIS's proxy); the theory and the background use ω_ν itself
            # (CAMB's omnuh2_active route: the three degenerate states have exactly this ω_ν today).
            mnu = omnuh2 * NEUTRINO_MASS_FAC / (_C.DEFAULT_NNU / 3.0) ** 0.75
            return dict(h=h, ombh2=p["Omega_b"] * h**2, omch2=(p["Omega_m"] - p["Omega_b"]) * h**2 - omnuh2,
                        ns=p["ns"], As=p["As_1e9"] * 1e-9, tau=jnp.asarray(cfg.tau_fixed),
                        w0=jnp.asarray(-1.0), wa=jnp.asarray(0.0), mu0=None, Sigma0=None, r=None,
                        mnu=mnu, omnuh2=omnuh2, Omega_m=p["Omega_m"], ext={"omnuh2": omnuh2})
        if cfg.basis in HSC_BASES:
            h = jnp.sqrt((p["omch2"] + p["ombh2"] + HSC_MNU / 93.14) / p["Omega_m"])
            out = dict(h=h, ombh2=p["ombh2"], omch2=p["omch2"], ns=p["ns"], As=jnp.exp(p["log1e10As"]) * 1e-10,
                       tau=jnp.asarray(cfg.tau_fixed), w0=jnp.asarray(-1.0), wa=jnp.asarray(0.0), mu0=None,
                       Sigma0=None, r=None, mnu=jnp.asarray(HSC_MNU), Omega_m=p["Omega_m"], ext={})
            if "A_bary" in p:
                out["A_bary"] = p["A_bary"]
            return out
        w0 = p.get("w", jnp.asarray(-1.0))
        wa = p.get("wa", jnp.asarray(0.0))
        # Extended parameters: sampled ones and fixed non-default values (defaults stay implicit, so
        # the solvers keep their exact default-neutrino paths).
        ext = {**ext_fixed, **{n: p[n] for n in EXT_NAMES if n in p}}
        if cmb_on(cfg):
            from gpucosmo.jax.cmb.theory import solve_h0_from_theta_mc

            h = solve_h0_from_theta_mc(tables, p["ombh2"], p["omch2"], w0, wa,
                                       p["theta_MC_100"], ext=ext) / 100.0
        else:
            h = p["H0"] / 100.0
        return dict(h=h, ombh2=p["ombh2"], omch2=p["omch2"], ns=p["ns"],
                    As=jnp.exp(p["logA"]) * 1e-10, tau=p.get("tau", jnp.asarray(cfg.tau_fixed)),
                    w0=w0, wa=wa, mu0=p.get("mu0"), Sigma0=p.get("Sigma0"), r=p.get("r"), ext=ext)

    # ---- exact theory block (what the DALI approximation expands) ----
    def _cosmo_th(x_th):
        c = cosmo(jnp.concatenate([x_th, jnp.asarray(fid[n_th:])]))
        return c, (dict(mu0=c["mu0"], Sigma0=c["Sigma0"]) if cfg.mg else {})

    nl_lens = cfg.cmb_nonlinear_model if cfg.cmb_nonlinear_lens else False  # HMcode version or linear
    # The solve's extended range: the ACT corrections' and the user likelihoods' (``cmb_x``).
    lmax_solve_extra = max(int(lmax_extra), int(cfg.cmb_x_lmax))
    cmb_accurate = not cfg.cmb_legacy_lensing  # ``jax_eb_theory.solve_lmax_extra``

    def _cmb_out(spec):
        out = {"dl_tt": spec.dl_tt, "dl_te": spec.dl_te, "dl_ee": spec.dl_ee}
        if lmax_extra:
            out["cmb_ext"] = spec.cmb_ext[..., :lmax_extra + 1]  # κκ, lensed TT/EE/BB/TE to L 2999
        elif cfg.cmb_lensing:
            out["dl_kk"] = spec.dl_kk  # candl D_L^κκ from the non-Limber φφ
        if cfg.cmb_x_lmax:
            out["cmb_x"] = spec.cmb_ext[..., :int(cfg.cmb_x_lmax) + 1]  # user likelihoods (``shared_block``)
        return out

    def theory_cmb(x_th):
        from gpucosmo.jax.cmb.jax_eb_theory import get_theory_spectra_jax_eb

        c, mg_kw = _cosmo_th(x_th)
        spec = get_theory_spectra_jax_eb(
            tables, logA=x_th[idx["logA"]], ns=c["ns"], theta_mc_100=x_th[idx["theta_MC_100"]],
            ombh2=c["ombh2"], omch2=c["omch2"], tau=c["tau"], w0=c["w0"], wa=c["wa"],
            lmax=2508, nonlinear_lens=nl_lens, k_chunk=cfg.cmb_k_chunk,
            lmax_extra=lmax_solve_extra, accurate=cmb_accurate, ext=c["ext"], **mg_kw)
        out = _cmb_out(spec)
        return add_tensor_spectra(out, c, nu_model) if cfg.tensors else out

    @jax.jit
    def cmb_solver_status(x_th):
        """Numerical-failure diagnostic of the CMB solve at ``x_th`` (theory parameters): the
        number of failed modes and non-finite C_ℓ (``cls_jax.CLS_SOURCE_FAILURE``); 0 = valid.
        A failed solve gives NaN spectra, so log L = −∞ (``log_like_from_theory``): this tells
        such a numerical rejection apart from a physical exclusion (prior cuts, ``log_prior_fn``)."""
        from gpucosmo.jax.cmb.jax_eb_theory import get_theory_spectra_jax_eb

        c, mg_kw = _cosmo_th(x_th)
        _, status = get_theory_spectra_jax_eb(
            tables, logA=x_th[idx["logA"]], ns=c["ns"], theta_mc_100=x_th[idx["theta_MC_100"]],
            ombh2=c["ombh2"], omch2=c["omch2"], tau=c["tau"], w0=c["w0"], wa=c["wa"],
            lmax=2508, nonlinear_lens=nl_lens, k_chunk=cfg.cmb_k_chunk,
            lmax_extra=lmax_solve_extra, accurate=cmb_accurate, ext=c["ext"], return_status=True, **mg_kw)
        return status

    _PK_KEYS = ("pk_cb", "pk_vc_vc", "pk_vb_vc", "pk_vb_vb")
    _FS_TABLE_KEYS = ("pkpoles", "sigma8", "fsigma8")  # what DESIDR1FullShape.loglike reads

    _dist_nodes = (cumulative_distance_nodes(np.concatenate([np.asarray(z_fs), np.asarray(z_bao)]), n_gl=16)
                   if cfg.desi_fs else None)

    def _fs_inputs(c, pk: dict):
        """FSInputs from the cosmology dict ``c`` and the linear spectra ``pk`` (may be {}:
        with precomputed tables only the background entries are read)."""
        from gpucosmo.jax.fullshape.desi_dr1 import FSInputs

        bp = background_state(tables, BackgroundParams.build(c["ombh2"], c["omch2"], 100.0 * c["h"],
                                                             c["w0"], c["wa"], ext=c["ext"]))
        zf, zb = jnp.asarray(z_fs), jnp.asarray(z_bao)
        # D_M at the 13 FS + BAO redshifts by one cumulative Gauss–Legendre rule (≲ 1e-12
        # relative, as 128 nodes per redshift, with ~16× fewer E(z) evaluations).
        dc = transverse_distance(tables, bp, comoving_distance_cumulative(tables, bp, _dist_nodes))
        nf = len(z_fs)
        return FSInputs(
            **{k: pk.get(k) for k in _PK_KEYS}, h=c["h"], n_s=c["ns"],
            Omega_b=c["ombh2"] / c["h"] ** 2, Omega_cdm=c["omch2"] / c["h"] ** 2,
            rdrag=rs_drag(tables, bp), H_fs=hubble_km_s_mpc(tables, bp, zf),
            DA_fs=dc[:nf] / (1 + zf),
            H_bao=hubble_km_s_mpc(tables, bp, zb),
            DA_bao=dc[nf:] / (1 + zb))

    # One linear-spectra solve for the DESI FS redshifts and the user likelihoods' (``pk_z``).
    z_lin = tuple(float(z) for z in z_fs) if cfg.desi_fs else ()
    n_fs_z = len(z_lin)
    z_lin = z_lin + tuple(float(z) for z in cfg.pk_z)

    # Modes past the solver grid (k > 2/Mpc) evolved for ``pk_kmax`` in the same solve.
    from gpucosmo.jax.eb.linear_pk_jax import LINPK_N_K, hik_k_nodes

    k_hik = hik_k_nodes(cfg.pk_kmax) if cfg.pk_z else np.zeros(0)

    def _lin_spectra(x_th):
        from gpucosmo.jax.eb.linear_pk_jax import LINPK_SINGLE_BAND, linear_cb_velocity_spectra, with_hik

        c, mg_kw = _cosmo_th(x_th)
        bands, k_mpc = LINPK_SINGLE_BAND if cfg.pk_single_band else None, None
        if k_hik.size:
            k_mpc, bands = with_hik(cfg.pk_kmax, bands)
        spec = linear_cb_velocity_spectra(
            ombh2=c["ombh2"], omch2=c["omch2"], h=c["h"], ns=c["ns"], As=c["As"],
            w0=c["w0"], wa=c["wa"], z_out=z_lin, k_mpc=k_mpc,
            k_chunk=cfg.pk_k_chunk, budget_bands=bands, ext=c["ext"], nu_model=nu_model, **mg_kw)
        if k_hik.size:  # split off the high-k columns: (n_z, n_hik) P_m, P_cb
            hik = {k: spec[k][:, LINPK_N_K:] for k in ("pk_m", "pk_cb")}
            spec = {k: (v[..., :LINPK_N_K] if getattr(v, "ndim", 0) >= 1 else v) for k, v in spec.items()}
            spec["hik"] = hik
        return c, spec

    def _pk_block(c, spec):
        from gpucosmo.jax.eb.growth_jax import extended_lnpk, growth_block, nonlinear_lnpk
        from gpucosmo.jax.eb.linear_pk_jax import linpk_k_grid

        sl = {k: (v[n_fs_z:] if k in ("pk_m", "pk_cb", "pk_m_vc") else v) for k, v in spec.items()}
        g = growth_block(sl, c["h"])
        out = {"lnpk_m": jnp.log(g["pk_m"]), "lnpk_cb": jnp.log(g["pk_cb"]), "sigma8": g["sigma8"],
               "fsigma8": g["fsigma8"]}
        if cfg.pk_kmax:
            from gpucosmo.jax.eb.cls_jax import _params_dict_cls

            from gpucosmo.jax.eb.cls_jax import ext_params

            prm = _params_dict_cls(c["ombh2"], c["omch2"], c["h"], c["ns"], c["As"], c["tau"], c["w0"], c["wa"],
                                   **ext_params(c["ext"]), nu_model=nu_model)
            kx = pk_ext_grid(cfg.pk_kmax)
            hk = dict(k_hi=k_hik, pk_hi=spec["hik"]["pk_m"][n_fs_z:]) if k_hik.size else {}
            if cfg.pk_nonlinear:
                out["lnpk_m_x"], out["lnpk_m_nl"] = nonlinear_lnpk(linpk_k_grid(), g["pk_m"], np.asarray(cfg.pk_z), kx,
                                                                   prm, version=cfg.cmb_nonlinear_model, **hk)
            else:
                out["lnpk_m_x"] = extended_lnpk(linpk_k_grid(), g["pk_m"], kx, prm, **hk)
        return out

    def theory_lin(x_th):
        """{"fs": ..., "pk": ...} from one linear-spectra solve."""
        c, spec = _lin_spectra(x_th)
        out = {}
        if cfg.desi_fs:
            out["fs"] = _fs_from_spec(c, {k: (v[:n_fs_z] if getattr(v, "ndim", 0) == 2 else v)
                                          for k, v in spec.items()})
        if cfg.pk_z:
            out["pk"] = _pk_block(c, spec)
        return out

    def theory_fs(x_th):
        return theory_lin(x_th)["fs"]

    def _fs_from_spec(c, spec):
        if cfg.fs_tables:
            tab = fs_like.theory_tables(_fs_inputs(c, {k: spec[k] for k in _PK_KEYS}))
            return {k: tab[k] for k in _FS_TABLE_KEYS}
        return {k: jnp.log(spec[k]) for k in _PK_KEYS}

    from gpucosmo.likelihoods.base import Cosmology, LikelihoodInputs

    extra_theory = [L for L in extra if L.has_theory()]
    block_static = {}  # static axes of the shared blocks (``shared_block``)
    if cfg.pk_z:
        from gpucosmo.jax.eb.linear_pk_jax import linpk_k_grid

        block_static["pk"] = {"z": np.asarray(cfg.pk_z, dtype=float), "k": linpk_k_grid()}
        if cfg.pk_kmax:
            block_static["pk"]["k_x"] = pk_ext_grid(cfg.pk_kmax)

    def theory_ext(x_th):
        """User likelihoods' cosmology-only outputs (``Likelihood.theory``)."""
        cosmo_ = Cosmology(_cosmo_th(x_th)[0], tables)
        return {L.label: L.theory(cosmo_) for L in extra_theory}

    def theory_sn(x_th):
        """Supernova distance moduli 5 log10 D_L at the SN redshifts (``sn_like.theory_vector``): a
        theory output like the others, so the DALI expansion covers the SN likelihood too."""
        c = _cosmo_th(x_th)[0]
        bp = BackgroundParams.build(c["ombh2"], c["omch2"], 100.0 * c["h"], c["w0"], c["wa"], ext=c["ext"])
        return {"mu": sn_like.theory_vector(bp)}

    def theory_exact(x_th):
        out = {}
        if cmb_on(cfg):
            out["cmb"] = theory_cmb(x_th)
        if cfg.desi_fs or cfg.pk_z:
            out.update(theory_lin(x_th))
        if extra_theory:
            out["ext"] = theory_ext(x_th)
        if sn_like is not None:
            out["sn"] = theory_sn(x_th)
        return out

    def theory_cmb_group(Xg):
        """CMB spectra for a group of walkers (leading axis) in one batched solve."""
        from gpucosmo.jax.cmb.jax_eb_theory import get_theory_spectra_jax_eb_batch

        c = jax.vmap(lambda xt: _cosmo_th(xt)[0])(Xg)
        mg_kw = dict(mu0=c["mu0"], Sigma0=c["Sigma0"]) if cfg.mg else {}
        spec = get_theory_spectra_jax_eb_batch(
            tables, logA=Xg[:, idx["logA"]], ns=c["ns"], theta_mc_100=Xg[:, idx["theta_MC_100"]],
            ombh2=c["ombh2"], omch2=c["omch2"], tau=c["tau"], w0=c["w0"], wa=c["wa"],
            lmax=2508, nonlinear_lens=nl_lens, layout=cfg.cmb_layout,
            lmax_extra=lmax_solve_extra, accurate=cmb_accurate, ext=c["ext"], **mg_kw)
        out = _cmb_out(spec)
        if cfg.tensors:
            out = jax.vmap(lambda o, cc: add_tensor_spectra(o, cc, nu_model))(out, c)
        return out

    def _map_groups(fb, X, batch):
        """``fb`` takes a group of walkers; groups of ``batch`` run sequentially (the
        last group padded with copies of the last walker)."""
        n = X.shape[0]
        b = int(batch or 0)
        if not 0 < b < n:
            return fb(X)
        n_pad = -(-n // b) * b
        if n_pad > n:
            X = jnp.concatenate([X, jnp.repeat(X[-1:], n_pad - n, axis=0)])
        out = jax.lax.map(fb, X.reshape(n_pad // b, b, X.shape[1]))
        return jax.tree_util.tree_map(lambda a: a.reshape((n_pad,) + a.shape[2:])[:n], out)

    def _map_walkers(f, X, batch):
        """``batch`` walkers vmapped per step of a sequential map (0 = vmap all)."""
        if batch and 0 < int(batch) < X.shape[0]:
            return jax.lax.map(f, X, batch_size=int(batch))
        return jax.vmap(f)(X)

    def theory_exact_batch(X_th):
        """Walker-batched theory. The CMB solve is mapped walker by walker: vmapping
        cosmologies through it costs ~4× per walker (13 s vs 2.7–3.5 s), while the
        P(k) solve batches well (4 walkers in ~1.1× the time of one)."""
        out = {}
        if cmb_on(cfg) and cfg.cmb_layout:
            out["cmb"] = _map_groups(theory_cmb_group, X_th, cfg.cmb_walker_batch)
        elif cmb_on(cfg):
            out["cmb"] = _map_walkers(theory_cmb, X_th, cfg.cmb_walker_batch)
        if cfg.desi_fs or cfg.pk_z:
            out.update(_map_walkers(theory_lin, X_th, cfg.pk_walker_batch))
        if extra_theory:
            out["ext"] = _map_walkers(theory_ext, X_th, cfg.pk_walker_batch)
        if sn_like is not None:
            out["sn"] = _map_walkers(theory_sn, X_th, cfg.pk_walker_batch)
        return out

    # ---- likelihood given the theory block ----
    def log_like_from_theory(x, th, smooth: bool = False, components: bool = False,
                             include: frozenset | None = None, nondiff: str = "callback"):
        """Total log-likelihood given the theory block; ``components=True`` returns a dict
        {component: log L} instead (per FS tracer: the official per-bin log L, which
        excludes the marginalization-prior term carried in ``fs_mlp``). ``include``
        restricts the evaluation to groups of {"cmb", "lens", "fs", "sn"} (diagnostics;
        default all), e.g. to fit CMB-only without compiling the full-shape model. ``nondiff``:
        "callback" evaluates non-differentiable user likelihoods (``Likelihood.differentiable``
        False, e.g. the Cobaya bridge) by their host callbacks (values only); "skip" leaves them
        out (gradient code adds them through ``nondiff_loglike``)."""
        use = (lambda g: True) if include is None else (lambda g: g in include)
        c = cosmo(x)
        parts = {}
        bp = BackgroundParams.build(c["ombh2"], c["omch2"], 100.0 * c["h"], c["w0"], c["wa"], ext=c["ext"])
        if cfg.cmb == "pr4" and use("cmb"):
            s = th["cmb"]
            pars = {n: x[idx[n]] for n in plik_names}
            parts["hillipop"] = hil.loglike(s["dl_tt"], s["dl_ee"], s["dl_te"], pars,
                                            invkll=th["aux"]["invkll"],
                                            ref=th["aux"].get("hil_ref"))
            ell = jnp.arange(s["dl_ee"].shape[0], dtype=jnp.float64)
            cl_ee = jnp.where(ell >= 2, s["dl_ee"] * 2 * jnp.pi / jnp.maximum(ell * (ell + 1), 1.0),
                              0.0)
            parts["lollipop"] = lol.loglike(cl_ee, pars["A_planck"])
            parts["lowl_tt"] = lowl(s["dl_tt"], s["dl_te"], s["dl_ee"], pars["A_planck"])
        elif cfg.cmb and use("cmb"):
            nuis = jnp.stack([x[idx[n]] for n in plik_names])
            a_pl = x[idx["A_planck"]] if "A_planck" in idx else jnp.asarray(1.0)
            s = th["cmb"]
            parts["plik"] = plik(s["dl_tt"], s["dl_te"], s["dl_ee"], nuis)
            parts["lowl"] = (lowl_smooth if smooth else lowl)(s["dl_tt"], s["dl_te"], s["dl_ee"], a_pl)
        if lens_ll is not None and use("lens"):
            if lmax_extra:
                ext = th["cmb"]["cmb_ext"]
                parts["act_lens"] = lens_ll(ext[0], ext[1:])
            else:
                parts["act_lens"] = lens_ll(th["cmb"]["dl_kk"])
        if cfg.desi_fs and use("fs"):
            from gpucosmo.jax.fullshape.desi_dr1 import FS_TRACERS

            f = th["fs"]
            if cfg.fs_tables:
                inp, fs_tab = _fs_inputs(c, {}), f
            else:
                inp, fs_tab = _fs_inputs(c, {k: jnp.exp(f[k]) for k in _PK_KEYS}), None
            nuis_fs = {t: (x[idx[f"{t}.b1p"]], x[idx[f"{t}.b2p"]], x[idx[f"{t}.bsp"]])
                       for t in FS_TRACERS}
            if components:
                tot, per = fs_like.loglike(inp, nuis_fs, tables=fs_tab, per_bin=True)
                parts.update({f"fs:{k}": v for k, v in per.items()})
                parts["fs_mlp"] = tot - sum(per.values())
            else:
                parts["desi_fs"] = fs_like.loglike(inp, nuis_fs, tables=fs_tab)
        if sn_like is not None and use("sn"):  # the distances from the theory block (exact or DALI-expanded)
            parts["sn"] = sn_like.from_theory(th["sn"]["mu"]) if "sn" in th else sn_like(bp)
        if extra:
            cosmo_ = Cosmology(c, tables)
            for L in extra:
                if not getattr(L, "differentiable", True) and nondiff == "skip":
                    continue
                if use(L.label):
                    inp = LikelihoodInputs(theory=th.get("ext", {}).get(L.label), params=_ext_params(L, x),
                                           cosmo=cosmo_, shared={r: shared_block(r, th, block_static) for r in L.requires})
                    parts[L.label] = jnp.asarray(L.loglike(inp), dtype=jnp.float64)
        if components:
            return parts
        ll = sum(parts.values(), jnp.asarray(0.0))
        # Non-finite: a failed theory evaluation (NaN spectra, ``cls_jax.CLS_SOURCE_FAILURE``) or a
        # likelihood outside its domain. Either way the point is rejected rather than given a finite
        # value; ``cmb_solver_status`` says whether the CMB solver failed.
        return jnp.where(jnp.isfinite(ll), ll, -jnp.inf)

    def _ext_inputs(L, x, th, cosmo_=None):
        """``LikelihoodInputs`` of user likelihood ``L`` at ``x`` with the theory block ``th``."""
        cosmo_ = Cosmology(cosmo(x), tables) if cosmo_ is None else cosmo_
        return LikelihoodInputs(theory=th.get("ext", {}).get(L.label), params=_ext_params(L, x), cosmo=cosmo_,
                                shared={r: shared_block(r, th, block_static) for r in L.requires})

    nondiff_list = tuple(L for L in extra if not getattr(L, "differentiable", True))

    def nondiff_loglike(x, theory_fn, rel_step: float = 0.05):
        """Σ log L of the non-differentiable user likelihoods (``Likelihood.differentiable`` False,
        e.g. the Cobaya bridge) at ``x`` with the theory block ``theory_fn(x)`` (e.g. the
        DALI expansion's), differentiable by central differences (``cobaya_bridge.nondiff.fd_term``)
        in the theory parameters and the likelihood's own: ``rel_step`` × ~1σ steps."""
        from gpucosmo.cobaya_bridge.nondiff import fd_term

        tot = jnp.asarray(0.0)
        for L in nondiff_list:
            sel = list(range(n_th)) + [idx[n] for n in L.params]
            h = [rel_step * float(st) for st in steps] + [
                rel_step * (L.params[n].gauss[1] if L.params[n].gauss else 0.02 * (L.params[n].hi - L.params[n].lo))
                for n in L.params]

            def f(xx, L=L):
                return L.loglike(_ext_inputs(L, xx, theory_fn(xx)))

            tot = tot + fd_term(f, x, sel, h)
        return tot

    def _with_matrix():
        """``aux`` with HiLLiPoP's inverse covariance on the device (``defer_hillipop_matrix``)."""
        if hil is not None and "invkll" not in aux:
            aux["invkll"] = hil.invkll
        if "fn" in _pending_ref:
            aux["hil_ref"] = _pending_ref.pop("fn")()
        return aux

    def _release_matrix():
        """Free HiLLiPoP's matrix on the device (``_with_matrix`` puts it back when needed)."""
        aux.pop("invkll", None)
        if hil is not None:
            hil._invkll = None

    def _log_like_scalar(x, aux_):
        return log_like_from_theory(x, {**theory_exact(x[:n_th]), "aux": aux_})

    def log_like_scalar(x):
        return _log_like_scalar(x, _with_matrix())

    # Theory and likelihood are separate compiled programs: fused into one jit the
    # exact batch took 56 s, against 15.1 s (theory) + 0.14 s (likelihoods) apart
    # (8 walkers, DESI FS + PR3 CMB, RTX 5070 Laptop).
    _theory_jit = jax.jit(lambda xs: theory_exact_batch(xs[:, :n_th]))

    @jax.jit
    def _like_post_jit(x, xs, th, aux_):
        return _like_batch(xs, th, aux_) + log_prior_fn(x)

    def _log_post_batch(x, aux_):
        xs = jnp.clip(x, prior_lo, prior_hi)
        return _like_post_jit(x, xs, _theory_jit(xs), aux_)

    def _like_batch(xs, th, aux_):
        f = lambda xx, tt: log_like_from_theory(xx, {**tt, "aux": aux_})  # noqa: E731
        b = int(cfg.walker_batch or 0)
        if 0 < b < xs.shape[0]:
            return jax.lax.map(lambda a: f(*a), (xs, th), batch_size=b)
        return jax.vmap(f)(xs, th)

    def log_post_batch_fn(x):
        return _log_post_batch(x, _with_matrix())

    # ---- DALI adapter ----
    from gpucosmo.jax.inference.dali_expansion import DALI_ORDER, DALIExpansion, build_dali_fd

    steps_tab = STEPS_CMB if cmb_on(cfg) else STEPS_NOCMB
    steps = np.array([steps_tab[n] for n in th_names])

    @jax.jit
    def theory_batch(X):
        return theory_exact_batch(X)

    def _bounded(y, y0):
        """Clip the DALI expansion's deviation from its expansion point per output bin.

        In the posterior bulk |Δ log P| ≲ 0.1 and |ΔD_ℓ/D_ℓ| ≲ 0.05, so little changes
        there (yet not nothing: at the official ΛCDM posterior draws the clipping moves
        log L by 0.02 on average, up to 0.3, mostly through HiLLiPoP;
        ``scripts/validate_compressed_dali.py``); far out the quadratic extrapolation
        is unbounded and gave log-posteriors of order −1e15, whose rejections collapsed
        the dual-averaging step size. The clipped expansion is the DALI posterior that DALI-trajectory
        HMC samples by default; with ``--accept exact`` the chain is exact regardless.
        """
        def clip_lin(v, v0, axis=-1, floor=1e-3):  # per spectrum (row of ``cmb_ext``)
            a = 0.5 * jnp.abs(v0) + floor * jnp.max(jnp.abs(v0), axis=axis, keepdims=True)
            return v0 + jnp.clip(v - v0, -a, a)

        out = {}
        if "fs" in y and cfg.fs_tables:
            # tables (ℓ, k, component, z) cross zero in k: floor relative to each k-row's max
            out["fs"] = {k: clip_lin(y["fs"][k], y0["fs"][k], axis=1 if y0["fs"][k].ndim == 4 else -1,
                                     floor=1e-2) for k in y["fs"]}
        elif "fs" in y:
            out["fs"] = {k: y0["fs"][k] + jnp.clip(y["fs"][k] - y0["fs"][k], -1.0, 1.0)
                         for k in y["fs"]}
        if "cmb" in y:
            out["cmb"] = {k: clip_lin(y["cmb"][k], y0["cmb"][k]) for k in y["cmb"]}
        if "pk" in y:  # log P within ±1 of the expansion point, σ8 / f σ8 within ±50 %
            out["pk"] = {k: (y0["pk"][k] + jnp.clip(y["pk"][k] - y0["pk"][k], -1.0, 1.0)) if k.startswith("ln")
                         else clip_lin(y["pk"][k], y0["pk"][k]) for k in y["pk"]}
        if "ext" in y:  # user likelihoods' outputs: unclipped (no generic safe bound)
            out["ext"] = y["ext"]
        if "sn" in y:  # SN distance moduli: smooth in the parameters, unclipped
            out["sn"] = y["sn"]
        return out

    def _like_tree():
        return jax.eval_shape(theory_exact, jnp.asarray(fid[:n_th]))

    def load_expansion(path):
        return DALIExpansion.load(Path(path), jax.tree_util.tree_map(
            lambda s: jnp.zeros(s.shape, s.dtype), _like_tree()), names=tuple(th_names))

    def expansion_point(x_th, order=1):
        """The expansion point: ``x_th``; in the DES basis with Ω_ν h² raised to one step above its prior
        minimum, so the finite-difference design stays within the massive-neutrino theory's domain
        (the prior-dominated Ω_ν h² can peak at the box edge; Σm_ν = 0 with three massive
        eigenstates is NaN). Steps past the other box edges are physical and left alone (clipping
        every parameter to the box made the MAP stage's expansion point differ from its iterate).
        HSC and DES bases, final (order ≥ 2) expansion: at least one step inside the box — the
        prior-dominated parameters put the MAP in a corner (HSC: ω_b, n_s, A_b; DES: Ω_ν h²), from which
        the polynomial extrapolated across the whole box."""
        x = np.asarray(x_th, dtype=np.float64).copy()
        if cfg.basis == "des":
            i = th_names.index("omnuh2")
            x[i] = max(x[i], lo[i] + steps[i])
        if cfg.basis in HSC_BASES + ("des",) and order >= 2:
            h = expansion_steps(order)
            x = np.clip(x, np.asarray(lo[:n_th]) + h, np.asarray(hi[:n_th]) - h)
        return x

    def expansion_steps(order):
        if cfg.basis in HSC_BASES and order >= 2:
            return np.array([STEPS_HSC[n] for n in th_names])
        if cfg.basis == "des" and order >= 2:
            return np.array([STEPS_DES[n] for n in th_names])
        return steps

    def expansion_log_idx(order):
        """Parameters the final expansion takes in ln: Ωm in the HSC bases (the broad posterior's
        Ωm range spans h ∝ Ωm^−½ at fixed ω_m; a quadratic in Ωm left 10–20 % of the SMC
        posterior where the exact likelihood is lower by 5–50); likewise in the DES basis."""
        if cfg.basis in HSC_BASES + ("des",) and order >= 2:
            return (th_names.index("Omega_m"),)
        return ()

    def build_expansion(x_th, order=DALI_ORDER, verbose=True, method="fd", tangent_batch=0, pair_batch=1):
        """The order-``order`` DALI expansion of the theory block alone (no likelihood arrays)."""
        x_th = expansion_point(x_th, order)
        if method == "fd":
            return build_dali_fd(theory_batch, np.asarray(x_th), expansion_steps(order), order=order,
                                   verbose=verbose, max_batch=16, log_idx=expansion_log_idx(order))
        return _build_dali_fwd_staged(np.asarray(x_th), order, verbose, tangent_batch, pair_batch)

    dali = SimpleNamespace(
        n=n_th, steps=steps, names=th_names,
        # Finite-difference design over the exact batched theory: the forward-mode
        # Jacobian of the banded P(k) solver took > 15 min to compile.
        # The DALI expansion is (DALIExpansion, aux): large likelihood arrays ride along as
        # pytree data so jitted trajectory code receives them as arguments.
        build=lambda *a, **k: (build_expansion(*a, **k), _with_matrix()),
        build_expansion=build_expansion,
        expansion_point=expansion_point,  # where build/build_expansion expand (x_th, moved in the DES basis)
        theory=lambda sur, x: {**_bounded(sur[0](x[:n_th]), sur[0].y0), "aux": sur[1]},
        save=lambda sur, path: sur[0].save(Path(path), names=tuple(th_names)),
        load=lambda path: (load_expansion(path), _with_matrix()),
        load_expansion=load_expansion,  # the DALI expansion alone (no likelihood arrays)
    )
    _self = {}  # the bundle namespace, for the staged forward-mode build (needs the finished bundle)

    def _build_dali_fwd_staged(x_th, order, verbose, tangent_batch, pair_batch):
        """expansion coefficients from JAX forward-mode derivatives (no finite differences), the
        theory differentiated stage by stage (``inference.staged_fwd``; one forward-mode program
        of the whole theory does not compile within the host memory)."""
        from gpucosmo.jax.inference.staged_fwd import staged_expansion
        from gpucosmo.jax.inference.theory_stages import theory_stages

        log = print if verbose else (lambda m: None)
        groups = [theory_stages(_self["b"], compute_dtype=jnp.float64)]  # higher orders need float64 evolutions
        if extra_theory:  # user likelihoods' outputs: one more (separately differentiated) stage
            groups.append([("ext", lambda xt: {"ext": theory_ext(xt)})])
        if sn_like is not None:  # SN distance moduli: likewise
            groups.append([("sn", lambda xt: {"sn": theory_sn(xt)})])
        coef = {}
        for st in (g for g in groups if g):
            _, c = staged_expansion(st, x_th, order=order, tangent_batch=tangent_batch, batch=max(1, int(pair_batch)),
                                 log=log)
            coef.update(c)
        return DALIExpansion(jnp.asarray(x_th), coef, order=order)

    def set_cmb_reference(x, cmb: dict) -> None:
        """Re-centre HiLLiPoP's float32 χ² (``aux["hil_ref"]``) on the spectra ``cmb``
        (``dl_tt``, ``dl_ee``, ``dl_te``) with the nuisances of ``x``. Exact; with a
        posterior-typical reference (e.g. the DALI expansion's expansion point) the batched
        float32 χ² is within 2e-4 of float64, against 2e-2 with the template default."""
        if cfg.cmb == "pr4":
            nuis = {n: jnp.asarray(np.asarray(x)[idx[n]]) for n in plik_names if n in idx}
            tt_, ee_, te_ = (jnp.asarray(cmb[k]) for k in ("dl_tt", "dl_ee", "dl_te"))

            def _ref():
                return hil.reference(tt_, ee_, te_, nuis)
            if cfg.defer_hillipop_matrix and "invkll" not in aux:  # computed when the matrix is needed
                aux.pop("hil_ref", None)
                _pending_ref["fn"] = _ref
            else:
                aux["hil_ref"] = _ref()
        # user / Cobaya-port likelihoods on the CMB block with a centred float32 χ² (e.g. CamSpec)
        refd = [L for L in extra if callable(getattr(L, "set_reference", None)) and "cmb" in L.requires]
        if refd:
            shared = {"cmb": shared_block("cmb", {"cmb": cmb}, block_static)}
            cosmo_ = Cosmology(cosmo(jnp.asarray(x)), tables)
            for L in refd:
                L.set_reference(LikelihoodInputs(theory=None, params={n: jnp.asarray(np.asarray(x)[idx[n]])
                                                                      for n in L.params}, cosmo=cosmo_, shared=shared))

    lens_tag = ("nl" if not cfg.cmb_lensing else "act_lens" if candl_lens else
                f"{cfg.cmb_lensing_variant}_lens" + ("" if cfg.cmb_lensing_lens_only else "_cmbcorr"))
    label = "+".join([s for s, on in (("desi_dr1_fs_bao", cfg.desi_fs),
                                      (f"cmb_{cfg.cmb}_{lens_tag}", cfg.cmb),
                                      (f"sn_{cfg.sn}", cfg.sn),
                                      ("bbn_ns10", bbn_ns)) if on] + [L.label for L in extra])
    # The likelihood objects, for code that re-implements the likelihood on reduced inputs
    # (``dali_reduce.CompressedDALI``).
    internals = SimpleNamespace(
        hil=hil, lol=lol, lowl=lowl, lowl_lmax=(cl.lmax if cfg.cmb else None), lens_ll=lens_ll, act=act,
        candl_lens=candl_lens and cfg.cmb_lensing, lmax_extra=lmax_extra, fs_like=fs_like,
        fs_inputs=_fs_inputs, z_fs=z_fs, z_bao=z_bao, sn_like=sn_like, idx=idx, plik_names=plik_names,
        n_theory=n_th, extra=extra, ext_names=tuple(ext_names), block_static=block_static,
        nondiff=tuple(L for L in extra if not getattr(L, "differentiable", True)), ext_inputs=_ext_inputs)
    bundle = SimpleNamespace(
        param_names=tuple(names), prior_lo=prior_lo, prior_hi=prior_hi,
        gauss_mu=g_mu, gauss_sd=g_sd, notes=notes, aux=aux,
        theory_program=_theory_jit, like_program=_like_post_jit,
        fiducial=jnp.asarray(fid), proposal_scales=jnp.asarray((hi - lo) / 100.0),
        log_prior_fn=log_prior_fn, log_post_batch_fn=log_post_batch_fn,
        log_like_from_theory_fn=log_like_from_theory, log_like_fn=log_like_scalar,
        theory_exact=theory_exact, theory_batch=theory_batch, dali=dali, n_theory=n_th, lmax=2508,
        theory_cmb=theory_cmb if cmb_on(cfg) else None, theory_fs=theory_fs if cfg.desi_fs else None,
        cmb_solver_status=cmb_solver_status if cmb_on(cfg) else None,
        likelihood_label=label, config=cfg, tables=tables, cosmo=cosmo, cmb_lmax_extra=lmax_solve_extra,
        cmb_out=_cmb_out, nondiff_loglike=nondiff_loglike,
        set_cmb_reference=set_cmb_reference, internals=internals, extra_likelihoods=extra,
        with_matrix=_with_matrix, release_matrix=_release_matrix)
    _self["b"] = bundle
    return bundle


__all__ = ["BBN_PRIOR", "BBN_SCHONEBERG2024", "MGConfig", "bbn_conditional", "build_mg_bundle"]
