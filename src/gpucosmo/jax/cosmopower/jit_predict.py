#!/usr/bin/env python3
"""JIT + vmap CosmoPower-JAX forward passes and P(k, z) table assembly."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.nn import sigmoid

from gpucosmo.jax.cosmopower.emulators import _mpk_lin_emulator, _mpk_nonlin_emulator
from gpucosmo.jax.cosmopower.params import HMCODE_CMIN, HMCODE_ETA0, LN10AS_REF
from gpucosmo.jax.power import CosmologyParams

_LOG_FLOOR = 1.0e-30


def _as_jnp_tree(obj: Any) -> Any:
    if isinstance(obj, (list, tuple)):
        return type(obj)(_as_jnp_tree(x) for x in obj)
    return jnp.asarray(obj)


def _cp_activation(x, alpha, beta):
    return jnp.multiply(
        jnp.add(beta, jnp.multiply(sigmoid(jnp.multiply(alpha, x)), jnp.subtract(1.0, beta))),
        x,
    )


def _cp_forward(
    weights: tuple,
    hyper_params: tuple,
    param_train_mean: jnp.ndarray,
    param_train_std: jnp.ndarray,
    feature_train_mean: jnp.ndarray,
    feature_train_std: jnp.ndarray,
    input_vec: jnp.ndarray,
    *,
    log_spectrum: bool,
) -> jnp.ndarray:
    """Batched CosmoPower MLP forward pass (input_vec shape (batch, n_params))."""
    layer_out = [(input_vec - param_train_mean) / param_train_std]
    for i in range(len(weights) - 1):
        w, b = weights[i]
        alpha, beta = hyper_params[i]
        act = jnp.dot(layer_out[-1], w.T) + b
        layer_out.append(_cp_activation(act, alpha, beta))
    w, b = weights[-1]
    preds = jnp.dot(layer_out[-1], w.T) + b[-1]
    preds = preds * feature_train_std + feature_train_mean
    if log_spectrum:
        preds = 10.0**preds
    return preds


@lru_cache(maxsize=1)
def _lin_predict_jit():
    emu = _mpk_lin_emulator()
    weights = _as_jnp_tree(emu.weights)
    hyper = _as_jnp_tree(emu.hyper_params)
    pmean = jnp.asarray(emu.param_train_mean)
    pstd = jnp.asarray(emu.param_train_std)
    fmean = jnp.asarray(emu.feature_train_mean)
    fstd = jnp.asarray(emu.feature_train_std)

    @jax.jit
    def predict(input_vec: jnp.ndarray) -> jnp.ndarray:
        return _cp_forward(
            weights, hyper, pmean, pstd, fmean, fstd, input_vec, log_spectrum=True
        )

    return predict


@lru_cache(maxsize=1)
def _nonlin_predict_jit():
    emu = _mpk_nonlin_emulator()
    weights_b = _as_jnp_tree(emu.weights)
    hyper_b = _as_jnp_tree(emu.hyper_params)
    weights_l = _as_jnp_tree(emu.weights_l)
    hyper_l = _as_jnp_tree(emu.hyper_params_l)
    pmean = jnp.asarray(emu.param_train_mean)
    pstd = jnp.asarray(emu.param_train_std)
    fmean = jnp.asarray(emu.feature_train_mean)
    fstd = jnp.asarray(emu.feature_train_std)
    pmean_l = jnp.asarray(emu.param_train_mean_l)
    pstd_l = jnp.asarray(emu.param_train_std_l)
    fmean_l = jnp.asarray(emu.feature_train_mean_l)
    fstd_l = jnp.asarray(emu.feature_train_std_l)

    @jax.jit
    def predict(input_vec: jnp.ndarray) -> jnp.ndarray:
        boost = _cp_forward(
            weights_b, hyper_b, pmean, pstd, fmean, fstd, input_vec, log_spectrum=True
        )
        lin_in = jnp.concatenate((input_vec[:, :5], input_vec[:, 7:8]), axis=1)
        lin = _cp_forward(
            weights_l,
            hyper_l,
            pmean_l,
            pstd_l,
            fmean_l,
            fstd_l,
            lin_in,
            log_spectrum=True,
        )
        return boost * lin

    return predict


@jax.jit
def _interp_log_pk_rows(
    log_k_src: jnp.ndarray,
    log_pk_rows: jnp.ndarray,
    log_k_dst: jnp.ndarray,
) -> jnp.ndarray:
    """Interpolate log P(k) rows onto log_k_dst (vmap over z)."""

    def _row(log_pk_row: jnp.ndarray) -> jnp.ndarray:
        lk_dst = jnp.clip(log_k_dst, log_k_src[0], log_k_src[-1])
        return jnp.interp(lk_dst, log_k_src, log_pk_row)

    return jax.vmap(_row)(log_pk_rows)


@lru_cache(maxsize=1)
def _log_k_cp_jax() -> jnp.ndarray:
    return jnp.log(jnp.asarray(_mpk_lin_emulator().modes, dtype=float))


@lru_cache(maxsize=1)
def _sample_lin_logpk_jit():
    log_k_cp = _log_k_cp_jax()
    predict = _lin_predict_jit()

    @jax.jit
    def sample(
        omega_b: jnp.ndarray,
        omega_c: jnp.ndarray,
        h: jnp.ndarray,
        n_s: jnp.ndarray,
        z_grid: jnp.ndarray,
        log_k_dst: jnp.ndarray,
        ln10A_s: jnp.ndarray,
    ) -> jnp.ndarray:
        n = z_grid.shape[0]
        param_batch = jnp.stack(
            [
                jnp.full((n,), omega_b * h**2),
                jnp.full((n,), omega_c * h**2),
                jnp.full((n,), h),
                jnp.full((n,), n_s),
                jnp.full((n,), ln10A_s),
                z_grid,
            ],
            axis=1,
        )
        pk = predict(param_batch)
        log_pk_cp = jnp.log(jnp.maximum(pk, _LOG_FLOOR))
        return _interp_log_pk_rows(log_k_cp, log_pk_cp, log_k_dst)

    return sample


@lru_cache(maxsize=1)
def _sample_nonlin_logpk_jit():
    log_k_cp = _log_k_cp_jax()
    predict = _nonlin_predict_jit()

    @jax.jit
    def sample(
        omega_b: jnp.ndarray,
        omega_c: jnp.ndarray,
        h: jnp.ndarray,
        n_s: jnp.ndarray,
        z_grid: jnp.ndarray,
        log_k_dst: jnp.ndarray,
        ln10A_s: jnp.ndarray,
    ) -> jnp.ndarray:
        n = z_grid.shape[0]
        param_batch = jnp.stack(
            [
                jnp.full((n,), omega_b * h**2),
                jnp.full((n,), omega_c * h**2),
                jnp.full((n,), h),
                jnp.full((n,), n_s),
                jnp.full((n,), ln10A_s),
                jnp.full((n,), HMCODE_CMIN),
                jnp.full((n,), HMCODE_ETA0),
                z_grid,
            ],
            axis=1,
        )
        pk = predict(param_batch)
        log_pk_cp = jnp.log(jnp.maximum(pk, _LOG_FLOOR))
        return _interp_log_pk_rows(log_k_cp, log_pk_cp, log_k_dst)

    return sample


def warmup_cosmopower_jit(
    *,
    z_max: float = 1.7,
    n_z: int = 80,
    n_k: int = 256,
    k_min: float = 1.0e-4,
    k_max: float = 50.0,
) -> None:
    """Compile CosmoPower JIT kernels (call once before timing)."""
    z_grid = jnp.linspace(0.0, float(z_max), int(n_z))
    k_grid = jnp.logspace(jnp.log10(float(k_min)), jnp.log10(float(k_max)), int(n_k))
    log_k_dst = jnp.log(k_grid)
    ob = jnp.asarray(0.047)
    oc = jnp.asarray(0.239)
    h = jnp.asarray(0.7)
    ns = jnp.asarray(0.96)
    ln10 = jnp.asarray(LN10AS_REF)
    lin = _sample_lin_logpk_jit()(ob, oc, h, ns, z_grid, log_k_dst, ln10)
    nl = _sample_nonlin_logpk_jit()(ob, oc, h, ns, z_grid, log_k_dst, ln10)
    jax.block_until_ready(lin)
    jax.block_until_ready(nl)


def sample_cp_tables_on_device(
    params: CosmologyParams,
    *,
    z_grid: jnp.ndarray,
    log_k_grid: jnp.ndarray,
    ln10A_s: float = LN10AS_REF,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """CosmoPower lin + HMCode nonlin log P(k, z) on the active JAX device."""
    ob = jnp.asarray(params.Omega_b, dtype=float)
    oc = jnp.asarray(params.Omega_c, dtype=float)
    h = jnp.asarray(params.h, dtype=float)
    ns = jnp.asarray(params.n_s, dtype=float)
    ln10 = jnp.asarray(ln10A_s, dtype=float)
    log_pk_lin = _sample_lin_logpk_jit()(ob, oc, h, ns, z_grid, log_k_grid, ln10)
    log_pk_nl = _sample_nonlin_logpk_jit()(ob, oc, h, ns, z_grid, log_k_grid, ln10)
    return log_pk_lin, log_pk_nl


def sample_cp_tables_batched(
    params: CosmologyParams,
    *,
    z_grid: np.ndarray,
    log_k_grid: np.ndarray,
    ln10A_s: float = LN10AS_REF,
) -> tuple[np.ndarray, np.ndarray]:
    """Batched JIT CosmoPower lin + HMCode nonlin log P(k, z) tables."""
    log_pk_lin, log_pk_nl = sample_cp_tables_on_device(
        params,
        z_grid=jnp.asarray(z_grid, dtype=float),
        log_k_grid=jnp.asarray(log_k_grid, dtype=float),
        ln10A_s=ln10A_s,
    )
    return (
        np.asarray(log_pk_lin, dtype=float),
        np.asarray(log_pk_nl, dtype=float),
    )


def clear_jit_predict_caches() -> None:
    _lin_predict_jit.cache_clear()
    _nonlin_predict_jit.cache_clear()
    _log_k_cp_jax.cache_clear()
    _sample_lin_logpk_jit.cache_clear()
    _sample_nonlin_logpk_jit.cache_clear()


__all__ = [
    "clear_jit_predict_caches",
    "sample_cp_tables_batched",
    "sample_cp_tables_on_device",
    "warmup_cosmopower_jit",
]
