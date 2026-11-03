#!/usr/bin/env python3
"""Assemble desi_full LikelihoodBundle: BAO + SN + clipy plik + clipy lowl + candl lensing."""

from __future__ import annotations

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from gpucosmo.jax.background.cosmology import (
    BackgroundParams,
    BackgroundTables,
    load_background_tables,
)
from gpucosmo.jax.background.likelihoods import (
    DEFAULT_PACKAGES_PATH,
    W0WaLikelihoodData,
    build_bao_loglike,
    build_sn_loglike,
    load_w0wa_data,
)
from gpucosmo.jax.cmb.candl_likelihood import (
    CandlLensLike,
    build_candl_lens_loglike,
    load_candl_lens_likelihood,
)
from gpucosmo.jax.cmb.clipy_likelihood import (
    ClipyLowl,
    build_clipy_lowl_loglike,
    load_clipy_lowl,
)
from gpucosmo.jax.cmb.clipy_plik import (
    ClipyPlik,
    build_clipy_plik_loglike,
    load_clipy_plik,
)
from gpucosmo.jax.cmb.params import (
    COSMO_NAMES,
    NUISANCE_NAMES,
    SZ_PRIOR,
    build_desi_full_params,
    nuisance_gaussian_priors,
)
from gpucosmo.jax.cmb.plik_lite import build_plik_lite_loglike, load_plik_lite_data
from gpucosmo.jax.cmb.theory import TheoryBackend, get_theory_spectra
from gpucosmo.jax.inference.bundle import LikelihoodBundle

_PLIK_LITE_SUBDIR = "data/planck_2018_pliklite_native"


def _default_plik_lite_dir(packages_path: str) -> Path:
    env = os.environ.get("GPUCOSMO_PLIK_LITE_DIR")
    if env:
        return Path(env)
    return Path(packages_path) / _PLIK_LITE_SUBDIR


def _background_from_cosmo(
    tables: BackgroundTables, p: dict[str, jnp.ndarray], h0: jnp.ndarray
) -> BackgroundParams:
    return BackgroundParams.build(p["ombh2"], p["omch2"], h0, p["w"], p["wa"])  # default neutrinos, flat


def build_desi_full_gpu_bundle(
    data: W0WaLikelihoodData | None = None,
    tables: BackgroundTables | None = None,
    *,
    sn_dataset: str = "pantheonplus",
    packages_path: str = DEFAULT_PACKAGES_PATH,
    include_bao: bool = True,
    include_sn: bool = True,
    include_plik: bool = True,
    include_clipy_lowl: bool = True,
    include_candl_lens: bool = True,
    plik_lite_dir: str | Path | None = None,
    prefer_full_plik: bool = True,
    theory_backend: TheoryBackend = "jax_eb",
    walker_batch: int = 8,
) -> LikelihoodBundle:
    """
    Joint DESI-like GPU bundle with cosmology + plik nuisances + ACT lensing.

    Batched functions evaluate walkers ``walker_batch`` at a time
    (``lax.map(batch_size=...)``): the native EB theory saturates a GPU at a few
    cosmologies, and vmapping many more only raises peak memory until XLA
    rematerializes (16 walkers on an 8 GB card ran 20× slower than 2 × 8).

    High-ell Planck uses **clipy full plik** (``plik_rd12_HM_v22b_TTTEEE.clik``) when
    installed under ``packages/data/planck_2018/`` (see ``install_desi_full_data.sh``),
    otherwise falls back to JAX **plik_lite**.

    ``theory_backend`` defaults to ``jax_eb`` (real w0/wa Cls). Pass
    ``cosmopower`` only for emulator regression.
    """
    if data is None:
        data = load_w0wa_data(
            packages_path,
            sn_datasets=(sn_dataset,) if sn_dataset else (),
        )
    if tables is None:
        tables = load_background_tables()

    plik_mode = "none"
    clipy_plik: ClipyPlik | None = None
    clipy_plik_loglike = None
    plik_lite_data = None
    plik_lite_loglike = None
    nuisance_names: tuple[str, ...] = NUISANCE_NAMES

    if include_plik and prefer_full_plik:
        clipy_plik = load_clipy_plik(packages_path)
        if clipy_plik is not None:
            clipy_plik_loglike = build_clipy_plik_loglike(clipy_plik)
            nuisance_names = clipy_plik.nuisance_names
            plik_mode = "clipy_plik"

    if include_plik and plik_mode == "none":
        pdir = Path(plik_lite_dir) if plik_lite_dir else _default_plik_lite_dir(packages_path)
        if pdir.is_dir():
            plik_lite_data = load_plik_lite_data(pdir)
            plik_lite_loglike = build_plik_lite_loglike(plik_lite_data)
            plik_mode = "plik_lite"
        else:
            include_plik = False

    param_names, prior_lo_np, prior_hi_np, fiducial_np, proposal_np = (
        build_desi_full_params(nuisance_names)
    )
    param_names_tuple = tuple(param_names)
    cosmo_idx = {n: i for i, n in enumerate(param_names_tuple) if n in COSMO_NAMES}
    nuis_idx = [i for i, n in enumerate(param_names_tuple) if n not in COSMO_NAMES]

    clipy_lowl: ClipyLowl | None = None
    clipy_lowl_loglike = clipy_lowl_loglike_smooth = None
    if include_clipy_lowl:
        clipy_lowl = load_clipy_lowl(packages_path)
        clipy_lowl_loglike = build_clipy_lowl_loglike(clipy_lowl)
        clipy_lowl_loglike_smooth = build_clipy_lowl_loglike(clipy_lowl, smooth_ee=True)
        if not clipy_lowl.active:
            include_clipy_lowl = False

    candl_lens: CandlLensLike | None = None
    candl_loglike = None
    if include_candl_lens:
        candl_lens = load_candl_lens_likelihood()
        candl_loglike = build_candl_lens_loglike(candl_lens)

    bao_loglike = build_bao_loglike(data.bao, tables) if include_bao else None
    sn_loglike = (
        build_sn_loglike(data.sn[sn_dataset], tables)
        if include_sn and sn_dataset
        else None
    )

    prior_lo = jnp.asarray(prior_lo_np)
    prior_hi = jnp.asarray(prior_hi_np)
    lmax = (
        int(clipy_plik.lmax)
        if clipy_plik is not None
        else int(plik_lite_data["lmax"])
        if plik_lite_data is not None
        else 2508
    )

    def _params_from_vector(x: jnp.ndarray) -> dict[str, jnp.ndarray]:
        return {name: x[..., i] for i, name in enumerate(param_names_tuple)}

    def _nuisance_vector(p: dict[str, jnp.ndarray]) -> jnp.ndarray:
        if not nuis_idx:
            return jnp.zeros(0)
        return jnp.stack([p[param_names_tuple[i]] for i in nuis_idx], axis=-1)

    def _a_planck(p: dict[str, jnp.ndarray]) -> jnp.ndarray:
        if "A_planck" in p:
            return p["A_planck"]
        if "calPlanck" in p:
            return p["calPlanck"]
        return jnp.asarray(1.0)

    def theory_fn(x: jnp.ndarray):
        """Theory spectra (TheorySpectra) for a full parameter vector."""
        p = _params_from_vector(x)
        with jax.named_scope("like/theory"):
            return get_theory_spectra(
                tables,
                logA=p["logA"],
                ns=p["ns"],
                theta_mc_100=p["theta_MC_100"],
                ombh2=p["ombh2"],
                omch2=p["omch2"],
                tau=p["tau"],
                w0=p["w"],
                wa=p["wa"],
                lmax=lmax,
                theory_backend=theory_backend,
            )

    def log_like_from_theory(x: jnp.ndarray, spec, smooth: bool = False) -> jnp.ndarray:
        """Likelihood given precomputed theory spectra (e.g. a DALI-expanded theory).

        ``smooth=True`` swaps SimAll low-ℓ EE for its differentiable twin
        (``clipy_likelihood._SmoothSimall``); use it for gradient-driven paths
        (DALI trajectories, MAP), not as the exact target.
        """
        p = _params_from_vector(x)
        bg = _background_from_cosmo(tables, p, spec.H0)
        ll = jnp.asarray(0.0)
        if bao_loglike is not None:
            with jax.named_scope("like/bao"):
                ll = ll + bao_loglike(bg)
        if sn_loglike is not None:
            with jax.named_scope("like/sn"):
                ll = ll + sn_loglike(bg)
        if clipy_plik_loglike is not None:
            with jax.named_scope("like/cmb_plik"):
                ll = ll + clipy_plik_loglike(
                    spec.dl_tt, spec.dl_te, spec.dl_ee, _nuisance_vector(p)
                )
        elif plik_lite_loglike is not None:
            with jax.named_scope("like/cmb_plik_lite"):
                ll = ll + plik_lite_loglike(
                    spec.dl_tt, spec.dl_te, spec.dl_ee, _a_planck(p)
                )
        if include_clipy_lowl and clipy_lowl_loglike is not None:
            with jax.named_scope("like/cmb_lowl"):
                ll = ll + (clipy_lowl_loglike_smooth if smooth else clipy_lowl_loglike)(
                    spec.dl_tt, spec.dl_te, spec.dl_ee, _a_planck(p)
                )
        if candl_loglike is not None:
            with jax.named_scope("like/cmb_lensing"):
                ll = ll + candl_loglike(spec.dl_kk)
        return ll

    def log_like_scalar(x: jnp.ndarray) -> jnp.ndarray:
        return log_like_from_theory(x, theory_fn(x))

    g_mu_np, g_sd_np = nuisance_gaussian_priors(param_names_tuple)
    g_mask = jnp.asarray(np.isfinite(g_sd_np))
    g_mu = jnp.asarray(g_mu_np)
    g_sd = jnp.asarray(np.where(np.isfinite(g_sd_np), g_sd_np, 1.0))
    sz_idx = (
        (param_names_tuple.index("ksz_norm"), param_names_tuple.index("A_sz"))
        if {"ksz_norm", "A_sz"} <= set(param_names_tuple) else None
    )

    def log_prior_fn(x: jnp.ndarray) -> jnp.ndarray:
        """Box + w+wa<0 support, Planck plik Gaussian nuisance priors, SZ prior."""
        inside = jnp.all(x >= prior_lo, axis=-1) & jnp.all(x <= prior_hi, axis=-1)
        w = x[..., cosmo_idx["w"]]
        wa = x[..., cosmo_idx["wa"]]
        inside &= (w + wa) < 0.0
        z = (x - g_mu) / g_sd
        lp = -0.5 * jnp.sum(jnp.where(g_mask, z * z, 0.0), axis=-1)
        if sz_idx is not None:
            fac, mu_sz, sd_sz = SZ_PRIOR
            comb = x[..., sz_idx[0]] + fac * x[..., sz_idx[1]]
            lp = lp - 0.5 * ((comb - mu_sz) / sd_sz) ** 2
        return jnp.where(inside, lp, -jnp.inf)

    def _map_walkers(x: jnp.ndarray) -> jnp.ndarray:
        if walker_batch and walker_batch > 0:
            return jax.lax.map(log_like_scalar, x, batch_size=int(walker_batch))
        return jax.vmap(log_like_scalar)(x)

    log_like_batch = jax.jit(_map_walkers)

    @jax.jit
    def log_post_batch_fn(x: jnp.ndarray) -> jnp.ndarray:
        x_safe = jnp.clip(x, prior_lo, prior_hi)
        like = _map_walkers(x_safe)
        return like + log_prior_fn(x)

    @jax.jit
    def log_post_scalar_fn(x: jnp.ndarray) -> jnp.ndarray:
        x_safe = jnp.clip(x, prior_lo, prior_hi)
        like = log_like_scalar(x_safe)
        return like + log_prior_fn(x)

    labels: list[str] = []
    if include_bao:
        labels.append("desi_dr2_bao")
    if include_sn and sn_dataset:
        labels.append(f"sn_{sn_dataset}")
    if include_plik:
        labels.append(plik_mode)
    if include_clipy_lowl:
        labels.append("clipy_lowl")
    if include_candl_lens:
        labels.append("candl_act_dr6_lens")

    minuslogprior = float(np.sum(np.log(prior_hi_np - prior_lo_np)))

    return LikelihoodBundle(
        param_names=param_names_tuple,
        prior_lo=prior_lo,
        prior_hi=prior_hi,
        proposal_scales=jnp.asarray(proposal_np),
        fiducial=jnp.asarray(fiducial_np),
        minuslogprior=minuslogprior,
        log_like_fn=log_like_scalar,
        log_like_batch_fn=log_like_batch,
        log_prior_fn=log_prior_fn,
        log_post_batch_fn=log_post_batch_fn,
        likelihood_label="+".join(labels) if labels else "desi_full",
        fixed_omega_b=0.0,
        log_post_scalar_fn=log_post_scalar_fn,
        theory_fn=theory_fn,
        log_like_from_theory_fn=log_like_from_theory,
        lmax=lmax,
    )


__all__ = ["build_desi_full_gpu_bundle"]
