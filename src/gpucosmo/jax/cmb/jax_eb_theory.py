#!/usr/bin/env python3
"""jax_eb → TheorySpectra adapter for desi_full (on-device Cls).

Recfast, hierarchy evolve, Bessel LOS, Cls ``dlnk`` sums and CMB lensing all
run on device in jax.numpy, with static host-built k grids
(``device_grids=True``), so the adapter is jit / vmap (over walkers) safe.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from gpucosmo.jax.background.cosmology import BackgroundTables
from gpucosmo.jax.cmb.lensing_jax import lens_cmb_cls
from gpucosmo.jax.cmb.params import T_CMB_MUK
from gpucosmo.jax.cmb.theory import TheorySpectra, solve_h0_from_theta_mc
from gpucosmo.jax.eb.cls_jax import (
    CLS_BATCH_K_CHUNK,
    CLS_K_MAX,
    CLS_NONLINEAR_LENS,
    _batch_k_grids,
    compute_cmb_cls,
    highl_plan,
)

_CANDL_KK_LEN = 3000
# Unlensed C_ℓ are computed this far past the lensed lmax before CAMB's high-ℓ template
# takes over (CAMB: lens_margin / lens_output_margin). With 0 the lensed EE ran
# +1.3e-3 (TT −3.2e-4) above converged CAMB at ℓ 2000–2509 and +6e-4 at 1500–2000;
# 750 brings this to +4e-4 / +1e-4 and halves the ℓ 1000–1500 residual (HiLLiPoP
# Δχ² vs converged −0.5 → −0.37 at the ΛCDM mean, n_s shift −0.03σ → +0.01σ).
# Costs ~30 % of a CMB evaluation (any margin > 0 costs about the same).
LENS_MARGIN = 750

# High-ℓ extension (``cls_jax.highl_plan``): lensed spectra past lmax + lens margin. Validated
# against CAMB to this lensed ℓ (scripts/validate_cmb_high_ell.py; tests/test_cmb_high_ell.py):
# ≤ 7.8e-4 at ℓ 3000–9000 (AccuracyBoost 2 reference), ≤ 1.4e-3 at ℓ 3000–13500 (AccuracyBoost
# 1.5: boost 2 to ℓ 17500 needs > 7 GB of host memory).
LMAX_EXTRA_MAX = 13500
# Batched high-ℓ path: cosmologies per vmapped group of the post-source stage (post_batch).
HIGHL_POST_BATCH = 4
# Accurate lensing by default: every lensed solve runs on the high-ℓ path (unlensed T/E to
# ℓ_solve + 500 with a log-slope tail, φφ and the lensing convolution to ℓ_solve + 4000; ℓ_solve ≥
# lmax + lens margin + 1) and returns the requested range. Against converged CAMB (AB 2, lens
# accuracy 15, margin 4000) the mean bias of the lensed TT / EE at ℓ 1000–2509 drops from
# +1.5…+7.8e-4 (the base path's lensing stopped at ℓ 5175) to ≤ 1.3e-4, and the extended block
# read by the ACT DR6 lensing corrections (BB −7 % at ℓ 2000–3000 on the base path) is accurate;
# +33 % time per evaluation (batch of 8, RTX 3070 Laptop: 2.1 → 2.8 s), slightly less memory.
# ``accurate=False`` reproduces the base path (earlier results).
ACCURATE_LENSING = True

jax.config.update("jax_enable_x64", True)


def max_lmax_extra() -> int:
    """Largest ``lmax_extra`` (lensed TT/EE/BB/TE and κκ in ``cmb_ext``) the solver supports."""
    return int(LMAX_EXTRA_MAX)


def _l_cls(lmax: int, lens_cmb: bool, lens_margin: int | None) -> int:
    """Unlensed C_ℓ range of the base path: lmax + lens margin (when lensing)."""
    return int(lmax) + (LENS_MARGIN if lens_margin is None else int(lens_margin)) * bool(lens_cmb)


def uses_highl(lmax: int, lmax_extra: int, *, lens_cmb: bool = True, lens_margin: int | None = None) -> bool:
    """Whether ``lmax_extra`` needs the high-ℓ extension (lensed spectra past the base range
    lmax + lens margin). Below it the results are those of the base path, unchanged."""
    return bool(lens_cmb) and int(lmax_extra) > _l_cls(lmax, lens_cmb, lens_margin)


def solve_lmax_extra(lmax: int, lmax_extra: int, *, lens_cmb: bool = True, lens_margin: int | None = None,
                     n_k: int | None = None, accurate: bool | None = None) -> int:
    """The lensed range actually solved for: the requested ``lmax_extra``, raised past the base range
    (onto the high-ℓ path) when ``accurate`` (default ``ACCURATE_LENSING``)."""
    acc = ACCURATE_LENSING if accurate is None else bool(accurate)
    if acc and lens_cmb and n_k is None:
        return max(int(lmax_extra), _l_cls(lmax, lens_cmb, lens_margin) + 1)
    return int(lmax_extra)


def _check_lmax_extra(lmax_extra: int) -> None:
    if int(lmax_extra) > LMAX_EXTRA_MAX:
        raise ValueError(f"lmax_extra {lmax_extra} > {LMAX_EXTRA_MAX} (max_lmax_extra()): lensed spectra "
                         "past it are not validated")


def _nl_lens(nonlinear_lens) -> bool | str:
    """False (linear φφ), True (HMcode-2020) or an HMcode version name ("mead2020",
    "mead2016"); a static argument of the compiled C_ℓ program."""
    if isinstance(nonlinear_lens, str):
        from gpucosmo.jax.eb.hmcode2020_jax import HMCODE_VERSIONS

        if nonlinear_lens not in HMCODE_VERSIONS:
            raise ValueError(f"nonlinear_lens {nonlinear_lens!r}: False, True or one of {HMCODE_VERSIONS}")
        return nonlinear_lens
    return bool(nonlinear_lens)


def _cl_to_dl_temper(cl: jnp.ndarray, ell: jnp.ndarray) -> jnp.ndarray:
    return cl * (ell * (ell + 1.0) / (2.0 * jnp.pi)) * (T_CMB_MUK**2)


def _camb_ctnorm(ell: jnp.ndarray) -> jnp.ndarray:
    """CAMB ``ctnorm = (ℓ²−1)(ℓ+2)ℓ = (ℓ−1)ℓ(ℓ+1)(ℓ+2)`` for EE/TE."""
    return (ell * ell - 1.0) * (ell + 2.0) * ell


def _cl_pp_to_dl_kk(cl_pp: jnp.ndarray, ell: jnp.ndarray) -> jnp.ndarray:
    dl_pp = cl_pp * (ell**2) * ((ell + 1.0) ** 2) / (2.0 * jnp.pi)
    return dl_pp * (jnp.pi / 2.0)


def _spectra_from_cls(cls: dict, h0, *, lmax: int, lens_cmb: bool,
                      lmax_extra: int = 0, highl: bool = False,
                      geometry: dict | None = None) -> TheorySpectra:
    """Raw C_ℓ (ΔT/T) → D_ℓ in μK² (CAMB ctnorm) up to ``lmax``, lensed if ``lens_cmb``
    (with every computed ℓ, which may run past ``lmax``); φφ → κκ. ``lmax_extra`` > 0
    also returns ``cmb_ext`` (κκ, lensed TT, EE, BB, TE to ``lmax_extra``).
    ``highl``: ``cls`` from the high-ℓ extension (``compute_cmb_cls(lmax_highl=...)``); the
    lensing tables are then computed on the device (``lensing_jax.geometry_runtime``), or
    taken from ``geometry`` (the same, precomputed once for a batch)."""
    ell = cls["ell"]
    if lmax_extra and (not lens_cmb or int(ell.shape[0]) - 1 < int(lmax_extra)):
        raise ValueError(f"lmax_extra {lmax_extra} needs lensing and C_ℓ computed that far "
                         f"(computed to {int(ell.shape[0]) - 1})")
    ct = _camb_ctnorm(ell)
    # CAMB CalcScalCls: EE *= ctnorm, TE *= sqrt(ctnorm)
    dl_tt = _cl_to_dl_temper(cls["tt"], ell)
    dl_ee = _cl_to_dl_temper(cls["ee"], ell) * ct
    dl_te = _cl_to_dl_temper(cls["te"], ell) * jnp.sqrt(jnp.maximum(ct, 0.0))
    cpp = cls["pp"] * (ell**2) * ((ell + 1.0) ** 2) / (2.0 * jnp.pi)
    ext = None
    if lens_cmb:
        # The lensed spectra are computed to CAMB's lmax_conv either way; ℓ ≤ lmax are
        # identical with or without ``lmax_extra``.
        dl_tt, dl_ee, dl_bb, dl_te = lens_cmb_cls(
            dl_tt,
            dl_ee,
            dl_te,
            cpp,
            lmax=int(ell.shape[0]) - 1,
            lmax_lensed=max(int(lmax), int(lmax_extra)),
            use_lensing_template=True,
            low_l_ee_taper=True,
            **(dict(device_geometry=True, geometry=geometry) if highl else {}),
        )
        if lmax_extra:
            n_x = int(lmax_extra) + 1
            kk_x = _cl_pp_to_dl_kk(cls["pp"], ell)[:n_x]
            ext = jnp.stack([kk_x, dl_tt[:n_x], dl_ee[:n_x], dl_bb[:n_x], dl_te[:n_x]])
    dl_tt, dl_ee, dl_te = (d[: int(lmax) + 1] for d in (dl_tt, dl_ee, dl_te))
    dl_kk_full = _cl_pp_to_dl_kk(cls["pp"], ell)
    dl_kk = jnp.zeros(_CANDL_KK_LEN, dtype=jnp.float64)
    n_kk = min(_CANDL_KK_LEN, int(lmax) + 1)
    dl_kk = dl_kk.at[:n_kk].set(dl_kk_full[:n_kk])
    return TheorySpectra(dl_tt=dl_tt, dl_te=dl_te, dl_ee=dl_ee, dl_kk=dl_kk, H0=h0, cmb_ext=ext)


def get_theory_spectra_jax_eb(
    tables: BackgroundTables,
    *,
    logA: jnp.ndarray,
    ns: jnp.ndarray,
    theta_mc_100: jnp.ndarray,
    ombh2: jnp.ndarray,
    omch2: jnp.ndarray,
    tau: jnp.ndarray,
    w0: jnp.ndarray,
    wa: jnp.ndarray,
    lmax: int = 2508,
    n_k: int | None = None,
    source_accuracy_boost: float = 1.0,
    lens_cmb: bool = True,
    k_chunk: int = CLS_BATCH_K_CHUNK,
    mu0: jnp.ndarray | None = None,
    Sigma0: jnp.ndarray | None = None,
    nonlinear_lens: bool | str = CLS_NONLINEAR_LENS,
    lens_margin: int | None = None,
    lmax_extra: int = 0,
    accurate: bool | None = None,
    ext: dict | None = None,
    return_status: bool = False,
) -> TheorySpectra:
    """CMB D_ℓ from on-device jax_eb Cls with H0 from θ_MC.

    ``return_status``: return ``(spectra, solver_status)``, ``solver_status`` the number of failed
    modes and non-finite C_ℓ (0 = valid; ``cls_jax.CLS_SOURCE_FAILURE``).

    ``ext``: Σm_ν, N_eff, m_eff,sterile, Ω_k (``cls_jax.ext_params``; missing: CAMB defaults),
    with the neutrino model of ``tables`` (``load_background_tables(nu_model=…)``).

    Default k-grid is CAMB Evolve_q (``n_k=None``). Pass ``n_k`` only for
    legacy equal-log overrides. When ``lens_cmb`` is True (default), TT/TE/EE
    are remapped with CAMB's correlation-function lensing; ``dl_kk`` stays the
    unlensed φφ spectrum that candl expects. ``k_chunk`` bounds peak memory when
    this is vmapped over many walkers (0 = one vmap over all k). ``mu0`` /
    ``Sigma0``: μ–Σ modified gravity (ISiTGR convention, ``eb.mg_jax``).
    ``nonlinear_lens=False`` lenses with the linear φφ (CAMB ``NonLinear_none``); True or
    "mead2020" with HMcode-2020, "mead2016" with HMcode-2016 (CAMB ``halofit_version``).
    ``lens_margin`` (default ``LENS_MARGIN``): extra unlensed ℓ computed for lensing.
    ``lmax_extra`` > 0: also return ``cmb_ext`` (κκ and lensed TT/EE/BB/TE to
    ``lmax_extra``, from the same solve; ``_spectra_from_cls``). Up to lmax + lens margin
    this is the base solve; past it (≤ ``max_lmax_extra()``) the high-ℓ extension
    (``cls_jax.highl_plan``) adds T/E projections to lmax_extra + 500, φφ and the lensing
    convolution to lmax_extra + 4000, without changing the base ℓ range's projections.
    ``accurate`` (default ``ACCURATE_LENSING``, True): lens on the high-ℓ path whatever
    ``lmax_extra`` (``solve_lmax_extra``); False reproduces the base path.
    """
    l_cls = _l_cls(lmax, lens_cmb, lens_margin)
    l_solve = solve_lmax_extra(lmax, lmax_extra, lens_cmb=lens_cmb, lens_margin=lens_margin, n_k=n_k,
                               accurate=accurate)
    highl = uses_highl(lmax, l_solve, lens_cmb=lens_cmb, lens_margin=lens_margin)
    if highl:
        _check_lmax_extra(l_solve)
        if n_k is not None:
            raise ValueError("lmax_extra past lmax + lens margin needs the Evolve_q grid (n_k=None)")
    h0 = solve_h0_from_theta_mc(
        tables, ombh2, omch2, w0, wa, theta_mc_100, ext=ext
    )
    h = h0 / 100.0
    As = jnp.exp(logA) * 1.0e-10
    cls_kw: dict = dict(
        ext=ext,
        nu_model=tables.model,
        ombh2=ombh2,
        omch2=omch2,
        h=h,
        tau_reio=tau,
        ns=ns,
        As=As,
        w0=w0,
        wa=wa,
        lmax=l_cls,
        use_recfast_thermo=True,
        source_accuracy_boost=float(source_accuracy_boost),
    )
    if n_k is not None:
        cls_kw["n_k"] = int(n_k)
    # Static grids built on the host (cached) so nothing inside is host-bound.
    if highl:
        plan = highl_plan(l_cls, int(l_solve), float(source_accuracy_boost))
        k_ev, k_int = plan["k_ev"], plan["k_int"]
        cls_kw["lmax_highl"] = int(l_solve)
    else:
        k_ev, k_int = _batch_k_grids(
            None if n_k is None else int(n_k), float(CLS_K_MAX),
            float(source_accuracy_boost), l_cls,
        )
    if mu0 is not None or Sigma0 is not None:
        cls_kw.update(mu0=mu0, Sigma0=Sigma0)
    cls = compute_cmb_cls(**cls_kw, device_grids=True, k_evolve=k_ev, k_int=k_int,
                          k_chunk=int(k_chunk), nonlinear_lens=_nl_lens(nonlinear_lens))
    spec = _spectra_from_cls(cls, h0, lmax=int(lmax), lens_cmb=lens_cmb, lmax_extra=int(lmax_extra),
                             highl=highl)
    return (spec, cls["solver_status"]) if return_status else spec


THEORY_PARAM_NAMES = ("logA", "ns", "theta_mc_100", "ombh2", "omch2", "tau", "w0", "wa")


def theory_value_and_jacobian(
    tables: BackgroundTables,
    x: jnp.ndarray,
    *,
    lmax: int = 2508,
    k_chunk: int = CLS_BATCH_K_CHUNK,
    tangent_batch: int = 0,
):
    """(TheorySpectra, Jacobian) at ``x`` (THEORY_PARAM_NAMES order).

    The Jacobian is a TheorySpectra whose leaves carry a leading axis of length
    8 (∂/∂x_i), from forward mode with one primal and 8 tangents.
    """
    x = jnp.asarray(x, dtype=jnp.float64)

    def f(xx):
        return get_theory_spectra_jax_eb(
            tables, **dict(zip(THEORY_PARAM_NAMES, (xx[i] for i in range(8)))),
            lmax=lmax, k_chunk=k_chunk,
        )

    eye = jnp.eye(8, dtype=x.dtype)
    if tangent_batch and 0 < tangent_batch < 8:
        y = f(x)
        jac = jax.lax.map(lambda v: jax.jvp(f, (x,), (v,))[1], eye,
                          batch_size=int(tangent_batch))
        return y, jac
    return jax.vmap(lambda v: jax.jvp(f, (x,), (v,)), out_axes=(None, 0))(eye)


def theory_spectra_fwd_jacobian(
    tables: BackgroundTables,
    x: jnp.ndarray,
    *,
    lmax: int = 2508,
    k_chunk: int = CLS_BATCH_K_CHUNK,
    tangent_batch: int = 0,
) -> TheorySpectra:
    """``get_theory_spectra_jax_eb`` of ``x`` (THEORY_PARAM_NAMES order), cheap to grad.

    Reverse mode through the EB solver would store every RK4 stage of every
    k-mode. The theory has only 8 inputs, so its Jacobian is computed in the
    forward pass with forward mode (8 tangents, one primal) and the backward pass
    is Jᵀ·cotangent (``custom_vjp``). ``jax.grad`` of a full likelihood therefore
    costs one Jacobian of the theory plus reverse mode through the cheap
    likelihood code. ``tangent_batch`` > 0 evaluates the tangents that many at a
    time (bounds memory at the cost of recomputing the primal per group).
    """

    def f(xx):
        return get_theory_spectra_jax_eb(
            tables, **dict(zip(THEORY_PARAM_NAMES, (xx[i] for i in range(8)))),
            lmax=lmax, k_chunk=k_chunk,
        )

    @jax.custom_vjp
    def g(xx):
        return f(xx)

    def g_fwd(xx):
        return theory_value_and_jacobian(tables, xx, lmax=lmax, k_chunk=k_chunk,
                                         tangent_batch=tangent_batch)

    def g_bwd(jac, ct):
        terms = [
            jnp.tensordot(jl, cl, axes=cl.ndim)
            for jl, cl in zip(jax.tree_util.tree_leaves(jac), jax.tree_util.tree_leaves(ct))
        ]
        return (sum(terms),)

    g.defvjp(g_fwd, g_bwd)
    return g(jnp.asarray(x, dtype=jnp.float64))


__all__ = [
    "ACCURATE_LENSING",
    "LMAX_EXTRA_MAX",
    "THEORY_PARAM_NAMES",
    "get_theory_spectra_jax_eb",
    "get_theory_spectra_jax_eb_batch",
    "max_lmax_extra",
    "solve_lmax_extra",
    "theory_spectra_fwd_jacobian",
    "theory_value_and_jacobian",
    "uses_highl",
]


def get_theory_spectra_jax_eb_batch(
    tables: BackgroundTables,
    *,
    logA: jnp.ndarray,
    ns: jnp.ndarray,
    theta_mc_100: jnp.ndarray,
    ombh2: jnp.ndarray,
    omch2: jnp.ndarray,
    tau: jnp.ndarray,
    w0: jnp.ndarray,
    wa: jnp.ndarray,
    lmax: int = 2508,
    source_accuracy_boost: float = 1.0,
    lens_cmb: bool = True,
    mu0: jnp.ndarray | None = None,
    Sigma0: jnp.ndarray | None = None,
    nonlinear_lens: bool | str = CLS_NONLINEAR_LENS,
    layout: str = "k_outer",
    lens_margin: int | None = None,
    lmax_extra: int = 0,
    post_batch: int | None = None,
    accurate: bool | None = None,
    ext: dict | None = None,
    return_status: bool = False,
) -> TheorySpectra:
    """``get_theory_spectra_jax_eb`` for W cosmologies (all inputs shape (W,); ``ext`` values
    too). ``return_status``: return ``(spectra, solver_status)`` with ``solver_status`` (W,).

    The per-k evolutions of all W cosmologies are batched together
    (``cls_jax.source_grids_batched``, ``layout``) instead of vmapping the whole
    single-cosmology pipeline, whose nested vmap made XLA materialize transposed
    copies of the dense τ-grid states (0.8 GB → 3.9 GB temporaries from 1 to 2
    cosmologies). Everything else is vmapped per cosmology. Same results.

    ``post_batch``: the per-cosmology stage after the sources (ℓ projections, φφ, lensing)
    runs over groups of this many cosmologies (``lax.map``) instead of one vmap over all W;
    default: all W on the base path, HIGHL_POST_BATCH with the high-ℓ extension (lmax_extra
    past lmax + lens margin). At lmax_extra 9000 on an RTX 3070 Laptop: W = 8 in one vmap
    3.50 s / 5.3 GB peak, in groups of 1 4.18 s / 4.3 GB; W = 4 in one vmap 2.04 s / 3.4 GB.
    ``accurate``: as ``get_theory_spectra_jax_eb`` (default: the high-ℓ path for every lensed solve).
    """
    import dataclasses

    from gpucosmo.jax.eb.cls_jax import (
        CLS_HIERARCHY,
        cmb_params,
        _require_curvature_enabled,
        hierarchy_for_model,
        is_flat,
        recfast_thermo_batched,
        source_grids_batched,
    )

    ext = dict(ext or {})
    model = tables.model

    l_cls = _l_cls(lmax, lens_cmb, lens_margin)
    l_solve = solve_lmax_extra(lmax, lmax_extra, lens_cmb=lens_cmb, lens_margin=lens_margin, accurate=accurate)
    highl = uses_highl(lmax, l_solve, lens_cmb=lens_cmb, lens_margin=lens_margin)
    h0 = jax.vmap(lambda ob, oc, w, a, th, e: solve_h0_from_theta_mc(tables, ob, oc, w, a, th, ext=e))(
        ombh2, omch2, w0, wa, theta_mc_100, ext)
    h = h0 / 100.0
    As = jnp.exp(logA) * 1.0e-10
    if highl:
        _check_lmax_extra(l_solve)
        plan = highl_plan(l_cls, int(l_solve), float(source_accuracy_boost))
        k_ev, k_int = plan["k_ev"], plan["k_int"]
        grid_kw = dict(n_base=plan["n_base"], hi_mult=plan["hi_mult"])
        cls_extra = dict(lmax_highl=int(l_solve))
    else:
        k_ev, k_int = _batch_k_grids(None, float(CLS_K_MAX), float(source_accuracy_boost), l_cls)
        grid_kw, cls_extra = {}, {}
    mg = mu0 is not None or Sigma0 is not None
    zeros = jnp.zeros_like(ombh2)
    mu_b = (zeros if mu0 is None else mu0) if mg else zeros
    sig_b = (zeros if Sigma0 is None else Sigma0) if mg else zeros

    def params_one(ob, oc, hh, n, a, t, w, wa_, m, sg, e):
        return cmb_params(ob, oc, hh, n, a, t, w, wa_, *((m, sg) if mg else (None, None)), ext=e,
                          nu_model=model)[0]

    params_b = jax.vmap(params_one)(ombh2, omch2, h, ns, As, tau, w0, wa, mu_b, sig_b, ext)
    thermo_b = jax.vmap(lambda *a, e: recfast_thermo_batched(*a, ext=e, nu_model=model))(
        ombh2, omch2, h, ns, As, tau, w0, wa, e=ext)
    cfg = hierarchy_for_model(dataclasses.replace(CLS_HIERARCHY, mg=True) if mg else CLS_HIERARCHY, model, ext)
    if "omk" in ext and not is_flat(ext["omk"]):  # batched Ω_k: traced → the curved program
        _require_curvature_enabled(ext["omk"])
        cfg = dataclasses.replace(cfg, curved=True)
    grid_b = source_grids_batched(k_ev, params_b, thermo_b, cfg, layout=layout, **grid_kw)

    geometry = None
    if highl:
        from gpucosmo.jax.cmb.lensing_jax import geometry_runtime

        # Cosmology-independent: built once for the batch (lax.map below would rebuild it
        # per group).
        geometry = geometry_runtime(int(plan["l_p"]))

    def post(ob, oc, hh, n, a, t, w, wa_, m, sg, h0_, p, th, g, e):
        cls = compute_cmb_cls(
            ombh2=ob, omch2=oc, h=hh, tau_reio=t, ns=n, As=a, w0=w, wa=wa_, lmax=l_cls,
            ext=e, nu_model=model,
            use_recfast_thermo=True, source_accuracy_boost=float(source_accuracy_boost),
            device_grids=True, k_evolve=k_ev, k_int=k_int, nonlinear_lens=_nl_lens(nonlinear_lens),
            **(dict(mu0=m, Sigma0=sg) if mg else {}), **cls_extra,
            precomputed=dict(params=p, thermo=th, grid=g))
        spec = _spectra_from_cls(cls, h0_, lmax=int(lmax), lens_cmb=lens_cmb,
                                 lmax_extra=int(lmax_extra), highl=highl, geometry=geometry)
        return (spec, cls["solver_status"]) if return_status else spec

    args = (ombh2, omch2, h, ns, As, tau, w0, wa, mu_b, sig_b, h0, params_b, thermo_b, grid_b, ext)
    n_w = int(ombh2.shape[0])
    pb = (HIGHL_POST_BATCH if highl else n_w) if post_batch is None else int(post_batch)
    if 0 < pb < n_w:
        return jax.lax.map(lambda a: post(*a), args, batch_size=pb)
    return jax.vmap(post)(*args)
