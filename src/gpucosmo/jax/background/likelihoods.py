#!/usr/bin/env python3
# This file is part of GPUCosmo. Copyright (C) 2025-2026 Leonel Medina-Varela.
# SPDX-License-Identifier: GPL-3.0-or-later
# Derived in part from the following, translated to JAX and modified for GPUCosmo:
# - Cobaya (https://github.com/CobayaSampler/cobaya), Copyright 2017-2026 Jesus Torrado and
#   Antony Lewis.
#   Licence: LICENSES/Cobaya.txt.
# See NOTICE.md.
"""
JAX implementations of the public DESI DR2 BAO, SN and compressed-CMB
likelihoods for w0waCDM fits, exactly mirroring the cobaya CPU versions.

Data handling: the cobaya likelihood classes are instantiated once on CPU and
their fully-processed arrays (redshift/observable tables, masked data vectors,
absolute-magnitude-marginalized inverse covariances) are copied verbatim into
JAX constants. This guarantees the data side is bit-identical to cobaya; only
the theory predictions are recomputed (in float64 JAX, see cosmology.py).

Likelihoods:

- DESI DR2 BAO (bao.desi_dr2): 13 D_V/r_d, D_M/r_d, D_H/r_d measurements.
- SN: sn.pantheonplus, sn.pantheonplusshoes, sn.union3, sn.desy5
  (Mb analytically marginalized, matching cobaya's `_marginalize_abs_mag`).
  Pantheon+SH0ES replaces calibrator luminosity distances with Cepheid
  host distances (cobaya `PantheonPlusShoes.alpha_beta_logp`).
- Compressed CMB: Gaussian on (theta_star, ombh2, ombh2+omch2) with the
  mean/covariance from DESI DR2 (arXiv:2503.14738, eqs. 35-36), the prior
  DESI validated against the full Planck+ACT likelihood for w0waCDM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from gpucosmo.jax.background.cosmology import (
    C_KM_S,
    BackgroundParams,
    BackgroundTables,
    background_state,
    comoving_distance,
    comoving_distance_cumulative,
    cumulative_distance_nodes,
    hubble_e2_of_ln_a,
    load_background_tables,
    rs_drag,
    theta_star,
    transverse_distance,
)
from gpucosmo.jax.inference.bundle import LikelihoodBundle

DEFAULT_PACKAGES_PATH = None  # → gpucosmo.paths.packages_path() (env / Cobaya config)

# DESI DR2 (arXiv:2503.14738) compressed CMB prior, eqs. (35)-(36):
# Gaussian on (theta_star, ombh2, ombh2 + omch2).
CMB_COMPRESSED_MEAN = np.array([0.01041, 0.02223, 0.14208])
CMB_COMPRESSED_COV = 1.0e-9 * np.array(
    [
        [0.006621, 0.12444, -1.1929],
        [0.12444, 21.344, -94.001],
        [-1.1929, -94.001, 1488.4],
    ]
)

# Observable codes for the BAO data vector.
_BAO_OBS_CODES = {"DV_over_rs": 0, "DM_over_rs": 1, "DH_over_rs": 2}

SN_DATASETS = ("pantheonplus", "pantheonplusshoes", "union3", "desy5")

# DESI DR2 baseline priors (w0 U[-3,1], wa U[-3,2], w0+wa<0), with
# (ombh2, omch2) ranges matching the CAMB recombination tables and a wide H0.
PARAM_NAMES = ("ombh2", "omch2", "H0", "w0", "wa")
PRIOR_LO = np.array([0.005, 0.05, 20.0, -3.0, -3.0])
PRIOR_HI = np.array([0.04, 0.30, 100.0, 1.0, 2.0])
FIDUCIAL = np.array([0.02237, 0.1180, 67.5, -0.83, -0.6])
PROPOSAL_SCALES = np.array([1.5e-4, 1.5e-3, 0.6, 0.06, 0.25])


@dataclass(frozen=True)
class BAOData:
    z: np.ndarray
    value: np.ndarray
    obs_code: np.ndarray
    invcov: np.ndarray


@dataclass(frozen=True)
class SNData:
    name: str
    zcmb: np.ndarray
    zhel: np.ndarray
    mag: np.ndarray
    invcov: np.ndarray
    # Pantheon+SH0ES: Cepheid calibrators use host distances, not cosmology.
    is_calibrator: np.ndarray | None = None
    ceph_dist: np.ndarray | None = None


@dataclass(frozen=True)
class W0WaLikelihoodData:
    bao: BAOData
    sn: dict[str, SNData]
    cmb_mean: np.ndarray = field(
        default_factory=lambda: CMB_COMPRESSED_MEAN.copy()
    )
    cmb_cov: np.ndarray = field(default_factory=lambda: CMB_COMPRESSED_COV.copy())


def load_w0wa_data(
    packages_path: str | None = DEFAULT_PACKAGES_PATH,
    sn_datasets: Sequence[str] = SN_DATASETS,
) -> W0WaLikelihoodData:
    """Instantiate the cobaya likelihoods and copy their processed arrays (``packages_path``:
    Cobaya packages; default from ``gpucosmo.paths.packages_path``)."""
    from cobaya.likelihoods.bao.desi_dr2 import desi_dr2

    from gpucosmo.paths import resolve_packages_path

    info = {"packages_path": resolve_packages_path(packages_path)}
    bao_like = desi_dr2(info)
    if float(bao_like.rs_rescale) != 1.0:
        raise ValueError("Unexpected rs_rescale in bao.desi_dr2")
    bao = BAOData(
        z=bao_like.data["z"].values.astype(float),
        value=bao_like.data["value"].values.astype(float),
        obs_code=np.array(
            [_BAO_OBS_CODES[obs] for obs in bao_like.data["observable"]], dtype=int
        ),
        invcov=np.asarray(bao_like.invcov, dtype=float),
    )

    sn: dict[str, SNData] = {}
    for name in sn_datasets:
        if name == "pantheonplus":
            from cobaya.likelihoods.sn.pantheonplus import PantheonPlus as cls
        elif name == "pantheonplusshoes":
            from cobaya.likelihoods.sn.pantheonplusshoes import PantheonPlusShoes as cls
        elif name == "union3":
            from cobaya.likelihoods.sn.union3 import Union3 as cls
        elif name == "desy5":
            from cobaya.likelihoods.sn.desy5 import DESy5 as cls
        else:
            raise ValueError(f"Unknown SN dataset {name!r}")
        like = cls(info)
        if like.use_abs_mag:
            raise ValueError("JAX port assumes Mb-marginalized SN likelihoods")
        is_cal = getattr(like, "is_calibrator", None)
        ceph = getattr(like, "ceph_dist", None)
        sn[name] = SNData(
            name=name,
            zcmb=np.asarray(like.zcmb, dtype=float),
            zhel=np.asarray(like.zhel, dtype=float),
            mag=np.asarray(like.mag, dtype=float),
            invcov=np.asarray(like.invcov, dtype=float),
            is_calibrator=(
                np.asarray(is_cal, dtype=bool) if is_cal is not None else None
            ),
            ceph_dist=np.asarray(ceph, dtype=float) if ceph is not None else None,
        )
    return W0WaLikelihoodData(bao=bao, sn=sn)


def build_bao_loglike(
    data: BAOData, tables: BackgroundTables
) -> Callable[[BackgroundParams], jnp.ndarray]:
    z = jnp.asarray(data.z)
    value = jnp.asarray(data.value)
    obs_code = jnp.asarray(data.obs_code)
    invcov = jnp.asarray(data.invcov)

    def theory_vector(params: BackgroundParams) -> jnp.ndarray:
        params = background_state(tables, params)
        rd = rs_drag(tables, params)
        dm = transverse_distance(tables, params, comoving_distance(tables, params, z))
        ln_a = -jnp.log1p(z)
        hz = params.H0 * jnp.sqrt(hubble_e2_of_ln_a(tables, params, ln_a))
        dh = C_KM_S / hz
        dv = jnp.cbrt(z * dm**2 * dh)
        theory = jnp.where(obs_code == 0, dv, jnp.where(obs_code == 1, dm, dh))
        return theory / rd

    def loglike(params: BackgroundParams) -> jnp.ndarray:
        resid = theory_vector(params) - value
        return -0.5 * resid @ invcov @ resid

    loglike.theory_vector = theory_vector  # type: ignore[attr-defined]
    return loglike


def build_sn_loglike(
    data: SNData, tables: BackgroundTables
) -> Callable[[BackgroundParams], jnp.ndarray]:
    zhel = jnp.asarray(data.zhel)
    mag = jnp.asarray(data.mag)
    invcov = jnp.asarray(data.invcov)
    has_calibrators = (
        data.is_calibrator is not None and data.ceph_dist is not None
    )
    if has_calibrators:
        is_calibrator = jnp.asarray(data.is_calibrator)
        ceph_lumdist = jnp.asarray(data.ceph_dist) - 25.0

    # Cumulative quadrature over the sorted SN redshifts (see
    # comoving_distance_cumulative): the 128-node-per-SN rule dominated the
    # cost of every DALI-HMC leapfrog step in f64 on consumer GPUs.
    dc_nodes = cumulative_distance_nodes(np.asarray(data.zcmb))

    def theory_vector(params: BackgroundParams) -> jnp.ndarray:
        # cobaya: lumdists = 5 log10((1+zhel)(1+zcmb) D_A(zcmb))
        #                  = 5 log10((1+zhel) D_M(zcmb)); Mb marginalized.
        params = background_state(tables, params)
        dm = transverse_distance(tables, params, comoving_distance_cumulative(tables, params, dc_nodes))
        lumdists = 5.0 * jnp.log10((1.0 + zhel) * dm)
        if has_calibrators:
            # Pantheon+SH0ES: Cepheid hosts anchor absolute magnitude.
            lumdists = jnp.where(is_calibrator, ceph_lumdist, lumdists)
        return lumdists

    def loglike_from_theory(lumdists: jnp.ndarray) -> jnp.ndarray:
        """log L from the distance moduli 5 log10 D_L of ``theory_vector`` (e.g. DALI-expanded)."""
        resid = mag - lumdists
        return -0.5 * resid @ invcov @ resid

    def loglike(params: BackgroundParams) -> jnp.ndarray:
        return loglike_from_theory(theory_vector(params))

    loglike.theory_vector = theory_vector  # type: ignore[attr-defined]
    loglike.from_theory = loglike_from_theory  # type: ignore[attr-defined]
    loglike.mag = np.asarray(data.mag, dtype=float)  # type: ignore[attr-defined]
    loglike.invcov = np.asarray(data.invcov, dtype=float)  # type: ignore[attr-defined]
    return loglike


def build_cmb_loglike(
    tables: BackgroundTables,
    mean: np.ndarray = CMB_COMPRESSED_MEAN,
    cov: np.ndarray = CMB_COMPRESSED_COV,
) -> Callable[[BackgroundParams], jnp.ndarray]:
    mean_j = jnp.asarray(mean)
    invcov = jnp.asarray(np.linalg.inv(cov))

    def theory_vector(params: BackgroundParams) -> jnp.ndarray:
        return jnp.stack([theta_star(tables, params), params.ombh2, params.ombh2 + params.omch2])

    def loglike(params: BackgroundParams) -> jnp.ndarray:
        resid = theory_vector(params) - mean_j
        return -0.5 * resid @ invcov @ resid

    loglike.theory_vector = theory_vector  # type: ignore[attr-defined]
    return loglike


def _params_from_vector(x: jnp.ndarray) -> BackgroundParams:
    """The 5-parameter flat w0waCDM vector of this bundle (default neutrinos)."""
    return BackgroundParams.build(x[..., 0], x[..., 1], x[..., 2], x[..., 3], x[..., 4])


def build_w0wa_bundle(
    data: W0WaLikelihoodData | None = None,
    tables: BackgroundTables | None = None,
    *,
    sn_dataset: str | None = "pantheonplus",
    include_bao: bool = True,
    include_cmb: bool = True,
    packages_path: str | None = DEFAULT_PACKAGES_PATH,
) -> LikelihoodBundle:
    """Joint DESI-BAO + compressed-CMB + SN likelihood bundle for GPU samplers."""
    if data is None:
        data = load_w0wa_data(
            packages_path,
            sn_datasets=(sn_dataset,) if sn_dataset else (),
        )
    if tables is None:
        tables = load_background_tables()

    parts: list[tuple[str, Callable[[BackgroundParams], jnp.ndarray]]] = []
    labels: list[str] = []
    if include_bao:
        parts.append(("like/bao", build_bao_loglike(data.bao, tables)))
        labels.append("desi_dr2_bao")
    if include_cmb:
        parts.append(("like/cmb_compressed", build_cmb_loglike(tables, data.cmb_mean, data.cmb_cov)))
        labels.append("cmb_compressed")
    if sn_dataset is not None:
        parts.append(("like/sn", build_sn_loglike(data.sn[sn_dataset], tables)))
        labels.append(f"sn_{sn_dataset}")

    prior_lo = jnp.asarray(PRIOR_LO)
    prior_hi = jnp.asarray(PRIOR_HI)

    def log_like_scalar(x: jnp.ndarray) -> jnp.ndarray:
        params = _params_from_vector(x)
        total = jnp.asarray(0.0)
        for scope, part in parts:
            with jax.named_scope(scope):
                total = total + part(params)
        return total

    def log_prior_fn(x: jnp.ndarray) -> jnp.ndarray:
        inside = jnp.all(x >= prior_lo, axis=-1) & jnp.all(x <= prior_hi, axis=-1)
        # DESI DR2: enforce early matter domination, w0 + wa < 0.
        inside &= (x[..., 3] + x[..., 4]) < 0.0
        # Flatness closure must leave positive dark energy density (background_state clamps
        # ω_de h² at 1e-10).
        inside &= background_state(tables, _params_from_vector(x)).omega_de_h2 > 1.0e-10
        return jnp.where(inside, 0.0, -jnp.inf)

    log_like_batch = jax.jit(jax.vmap(log_like_scalar))

    @jax.jit
    def log_post_batch_fn(x: jnp.ndarray) -> jnp.ndarray:
        # Clip into the valid box so rejected points still evaluate finitely
        # (keeps NUTS gradients NaN-free); the prior assigns them -inf.
        x_safe = jnp.clip(x, prior_lo, prior_hi)
        like = jax.vmap(log_like_scalar)(x_safe)
        inside = jnp.isfinite(log_prior_fn(x))
        return jnp.where(inside, like, -jnp.inf)

    @jax.jit
    def log_post_scalar_fn(x: jnp.ndarray) -> jnp.ndarray:
        x_safe = jnp.clip(x, prior_lo, prior_hi)
        like = log_like_scalar(x_safe)
        inside = jnp.isfinite(log_prior_fn(x))
        return jnp.where(inside, like, -jnp.inf)

    minuslogprior = float(np.sum(np.log(PRIOR_HI - PRIOR_LO)))

    return LikelihoodBundle(
        param_names=PARAM_NAMES,
        prior_lo=prior_lo,
        prior_hi=prior_hi,
        proposal_scales=jnp.asarray(PROPOSAL_SCALES),
        fiducial=jnp.asarray(FIDUCIAL),
        minuslogprior=minuslogprior,
        log_like_fn=log_like_scalar,
        log_like_batch_fn=log_like_batch,
        log_prior_fn=log_prior_fn,
        log_post_batch_fn=log_post_batch_fn,
        likelihood_label="+".join(labels),
        fixed_omega_b=0.0,
        log_post_scalar_fn=log_post_scalar_fn,
    )


__all__ = [
    "BAOData",
    "CMB_COMPRESSED_COV",
    "CMB_COMPRESSED_MEAN",
    "FIDUCIAL",
    "PARAM_NAMES",
    "PRIOR_HI",
    "PRIOR_LO",
    "PROPOSAL_SCALES",
    "SNData",
    "SN_DATASETS",
    "W0WaLikelihoodData",
    "build_bao_loglike",
    "build_cmb_loglike",
    "build_sn_loglike",
    "build_w0wa_bundle",
    "load_w0wa_data",
]
