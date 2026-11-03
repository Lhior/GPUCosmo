#!/usr/bin/env python3
"""CosmoPower-JAX linear and HMCode nonlinear P(k, z) tables."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from gpucosmo.jax.sigma8 import amplitude_scale_for_sigma8, sigma8_from_pk_z0
from gpucosmo.jax.cosmopower.emulators import (
    clear_emulator_cache,
    cosmopower_k_modes,
)
from gpucosmo.jax.cosmopower.jit_predict import (
    sample_cp_tables_batched,
    warmup_cosmopower_jit,
)
from gpucosmo.jax.cosmopower.params import LN10AS_REF
from gpucosmo.jax.power import CosmologyParams, MatterPowerTable


@dataclass(frozen=True)
class _CosmopowerTransferCacheEntry:
    """Cached emulator output at reference amplitude (sigma8-independent)."""

    log_k: np.ndarray
    z_grid: np.ndarray
    log_pk_ref: np.ndarray
    log_pk_nl_ref: np.ndarray
    sigma8_ref: float


def _rounded_transfer_key(
    params: CosmologyParams,
    *,
    z_max: float,
    n_z: int,
    n_k: int,
    k_min: float,
    k_max: float,
) -> tuple[float, float, float, float, float, int, int, float, float]:
    """Cache key excluding sigma8 (amplitude applied analytically)."""
    return (
        round(params.Omega_c, 8),
        round(params.Omega_b, 8),
        round(params.h, 8),
        round(params.n_s, 8),
        float(z_max),
        int(n_z),
        int(n_k),
        float(k_min),
        float(k_max),
    )


@lru_cache(maxsize=64)
def _get_cosmopower_transfer_entry(
    omega_c: float,
    omega_b: float,
    h: float,
    n_s: float,
    z_max: float,
    n_z: int,
    n_k: int,
    k_min: float,
    k_max: float,
) -> _CosmopowerTransferCacheEntry:
    """Run CosmoPower-JAX once at reference amplitude for lin + HMCode nonlin."""
    params = CosmologyParams(
        Omega_c=omega_c,
        Omega_b=omega_b,
        h=h,
        sigma8=1.0,
        n_s=n_s,
    )
    z_grid = np.linspace(0.0, float(z_max), int(n_z))
    k_grid = np.logspace(np.log10(float(k_min)), np.log10(float(k_max)), int(n_k))
    log_k_grid = np.log(k_grid)

    log_pk_ref, log_pk_nl_ref = sample_cp_tables_batched(
        params,
        z_grid=z_grid,
        log_k_grid=log_k_grid,
        ln10A_s=LN10AS_REF,
    )
    sigma8_ref = sigma8_from_pk_z0(log_k_grid, log_pk_ref[0], h=h)
    return _CosmopowerTransferCacheEntry(
        log_k=log_k_grid,
        z_grid=z_grid,
        log_pk_ref=log_pk_ref,
        log_pk_nl_ref=log_pk_nl_ref,
        sigma8_ref=sigma8_ref,
    )


def clear_cosmopower_pk_caches() -> None:
    """Clear CosmoPower transfer LRU caches (retain JIT-compiled emulators)."""
    _get_cosmopower_transfer_entry.cache_clear()


def _build_from_entry(
    entry: _CosmopowerTransferCacheEntry,
    params: CosmologyParams,
    *,
    nonlinear: bool,
) -> MatterPowerTable:
    scale = amplitude_scale_for_sigma8(entry.sigma8_ref, params.sigma8)
    log_scale = np.log(scale)
    log_pk = entry.log_pk_ref + log_scale
    if nonlinear:
        log_pk = entry.log_pk_nl_ref + log_scale
    return MatterPowerTable(
        log_k=entry.log_k,
        z=entry.z_grid,
        log_pk=log_pk,
    )


def build_linear_pk_cosmopower(
    params: CosmologyParams,
    *,
    z_max: float = 3.0,
    n_z: int = 80,
    k_min: float = 1.0e-4,
    k_max: float = 50.0,
    n_k: int = 256,
) -> MatterPowerTable:
    """Linear P(k, z) from CosmoPower-JAX with analytic sigma8 rescaling."""
    key = _rounded_transfer_key(
        params, z_max=z_max, n_z=n_z, n_k=n_k, k_min=k_min, k_max=k_max
    )
    omega_c, omega_b, h, n_s, z_max_r, n_z_r, n_k_r, k_min_r, k_max_r = key
    entry = _get_cosmopower_transfer_entry(
        omega_c, omega_b, h, n_s, z_max_r, n_z_r, n_k_r, k_min_r, k_max_r
    )
    return _build_from_entry(entry, params, nonlinear=False)


def build_nonlinear_pk_cosmopower(
    params: CosmologyParams,
    *,
    z_max: float = 3.0,
    n_z: int = 80,
    k_min: float = 1.0e-4,
    k_max: float = 50.0,
    n_k: int = 256,
) -> MatterPowerTable:
    """HMCode-emulated nonlinear P(k, z) from CosmoPower-JAX mpk_nonlin."""
    key = _rounded_transfer_key(
        params, z_max=z_max, n_z=n_z, n_k=n_k, k_min=k_min, k_max=k_max
    )
    omega_c, omega_b, h, n_s, z_max_r, n_z_r, n_k_r, k_min_r, k_max_r = key
    entry = _get_cosmopower_transfer_entry(
        omega_c, omega_b, h, n_s, z_max_r, n_z_r, n_k_r, k_min_r, k_max_r
    )
    return _build_from_entry(entry, params, nonlinear=True)


__all__ = [
    "build_linear_pk_cosmopower",
    "build_nonlinear_pk_cosmopower",
    "clear_cosmopower_pk_caches",
    "cosmopower_k_modes",
    "warmup_cosmopower_jit",
]
