#!/usr/bin/env python3
"""CAMB-parity Einstein–Boltzmann + mead2020_feedback P(k,z).

Cold-path design: one cosmology → one linear solve → one HMcode pass →
``(pk_nl, pk_lin)``. ``_PK_CACHE`` is optional; MCMC must not rely on hits.

Env knobs:
  - ``GPUCOSMO_EB_LINEAR_SOURCE``: ``camb`` (MCMC default via shared path) |
    ``boltzmann`` (accuracy) | ``camb_transfer``
  - ``GPUCOSMO_EB_LIVE_HMCODE``: default on
  - ``GPUCOSMO_EB_HMCODE_KMAX``: high-k CAMB splice threshold (shared jax_eb
    path sets ``1e6`` to avoid per-step CAMB NL)
  - ``GPUCOSMO_EB_DEVICE=1``: full on-device Recfast + hierarchy + HMcode
    (no host Recfast / CAMB / solve_ivp in the P(k) path)
  - ``GPUCOSMO_EB_STAGE_TIMERS=1``: print stage timings
"""

from __future__ import annotations

import os
import time
from typing import Any

from gpucosmo.jax.power import CosmologyParams, MatterPowerTable

_PK_CACHE: dict = {}
_LAST_STAGE_TIMES: dict[str, float] = {}


def clear_pk_cache() -> None:
    """Drop module-level P(k) cache (honest MCMC / bakeoff)."""
    _PK_CACHE.clear()


def last_stage_times() -> dict[str, float]:
    """Copy of timings from the most recent ``build_jax_eb_pk_pair`` call."""
    return dict(_LAST_STAGE_TIMES)


def _cache_key(
    params: CosmologyParams,
    *,
    z_max: float,
    n_z: int,
    n_k: int,
    nonlinear: bool,
    k_min: float,
    k_max: float,
    linear_source: str,
) -> tuple:
    return (
        round(float(params.Omega_c), 8),
        round(float(params.Omega_b), 8),
        round(float(params.h), 8),
        round(float(params.sigma8), 8),
        round(float(params.n_s), 8),
        float(z_max),
        int(n_z),
        int(n_k),
        bool(nonlinear),
        float(k_min),
        float(k_max),
        os.environ.get("GPUCOSMO_EB_DEBUG_EH", ""),
        os.environ.get("GPUCOSMO_EB_SKIP_HALOFIT", ""),
        os.environ.get("GPUCOSMO_EB_LIVE_HMCODE", "1"),
        linear_source,
        os.environ.get("GPUCOSMO_EB_LOGT_AGN", "7.7"),
        os.environ.get("GPUCOSMO_EB_CAMB_DIRECT", ""),
        os.environ.get("GPUCOSMO_EB_CCL_PK", ""),
        os.environ.get("GPUCOSMO_EB_BOLTZMANN_KMAX", "3.0"),
        os.environ.get("GPUCOSMO_EB_HMCODE_KMAX", "0.2"),
        os.environ.get("GPUCOSMO_EB_DEVICE", "0"),
    )


def _build_linear(
    params: CosmologyParams,
    *,
    z_max: float,
    n_z: int,
    n_k: int,
    k_min: float,
    k_max: float,
    linear_source: str,
) -> MatterPowerTable:
    debug_eh = os.environ.get("GPUCOSMO_EB_DEBUG_EH", "").lower() in (
        "1",
        "true",
        "yes",
    )
    if debug_eh:
        from gpucosmo.jax.eb.transfer import linear_pk_table_camb_like

        return linear_pk_table_camb_like(
            params,
            z_max=z_max,
            n_z=n_z,
            k_min=k_min,
            k_max=k_max,
            n_k=n_k,
            use_fluid_hierarchy=False,
        )
    from gpucosmo.jax.eb.transfer_out import linear_pk_table

    return linear_pk_table(
        params,
        z_max=z_max,
        n_z=n_z,
        k_min=k_min,
        k_max=k_max,
        n_k=n_k,
        source=linear_source,
    )


def _apply_nl(
    params: CosmologyParams,
    lin: MatterPowerTable,
    *,
    logT: float,
) -> MatterPowerTable:
    live_hm_env = os.environ.get("GPUCOSMO_EB_LIVE_HMCODE", "1").lower()
    live_hm = live_hm_env not in ("0", "false", "no")
    if live_hm:
        from gpucosmo.jax.eb.hmcode import nonlinear_pk_hmcode_feedback

        return nonlinear_pk_hmcode_feedback(
            lin,
            Omega_m=float(params.Omega_m),
            Omega_b=float(params.Omega_b),
            h=float(params.h),
            n_s=float(params.n_s),
            logT_AGN=logT,
        )
    from gpucosmo.jax.eb.transfer_out import nonlinear_pk_from_camb

    return nonlinear_pk_from_camb(
        params,
        z_max=float(lin.z.max()) if lin.z.size else 3.0,
        n_z=int(lin.z.size),
        k_min=float(np_exp_min(lin)),
        k_max=float(np_exp_max(lin)),
        n_k=int(lin.log_k.size),
        logT_AGN=logT,
    )


def np_exp_min(lin: MatterPowerTable) -> float:
    import numpy as np

    return float(np.exp(lin.log_k[0]))


def np_exp_max(lin: MatterPowerTable) -> float:
    import numpy as np

    return float(np.exp(lin.log_k[-1]))


def build_jax_eb_pk_pair(
    params: CosmologyParams,
    *,
    z_max: float = 3.0,
    n_z: int = 80,
    n_k: int = 256,
    k_min: float = 5.0e-5,
    k_max: float = 100.0,
    linear_source: str | None = None,
) -> tuple[MatterPowerTable, MatterPowerTable]:
    """One linear solve + one HMcode pass → ``(pk_nl, pk_lin)`` aligned grids.

    This is the MCMC cold path: no second Boltzmann/CAMB linear solve.
    """
    global _LAST_STAGE_TIMES
    src = (linear_source or os.environ.get("GPUCOSMO_EB_LINEAR_SOURCE", "boltzmann")).strip()
    logT = float(os.environ.get("GPUCOSMO_EB_LOGT_AGN", "7.7"))
    skip_nl = os.environ.get("GPUCOSMO_EB_SKIP_HALOFIT", "").lower() in (
        "1",
        "true",
        "yes",
    )
    camb_direct = os.environ.get("GPUCOSMO_EB_CAMB_DIRECT", "").lower() in (
        "1",
        "true",
        "yes",
    )
    use_ccl = os.environ.get("GPUCOSMO_EB_CCL_PK", "").lower() in (
        "1",
        "true",
        "yes",
    )
    timers = os.environ.get("GPUCOSMO_EB_STAGE_TIMERS", "").lower() in (
        "1",
        "true",
        "yes",
    )
    stages: dict[str, float] = {}

    key_nl = _cache_key(
        params,
        z_max=z_max,
        n_z=n_z,
        n_k=n_k,
        nonlinear=True,
        k_min=k_min,
        k_max=k_max,
        linear_source=src,
    )
    key_lin = _cache_key(
        params,
        z_max=z_max,
        n_z=n_z,
        n_k=n_k,
        nonlinear=False,
        k_min=k_min,
        k_max=k_max,
        linear_source=src,
    )
    cached_nl = _PK_CACHE.get(key_nl)
    cached_lin = _PK_CACHE.get(key_lin)
    if cached_nl is not None and cached_lin is not None:
        _LAST_STAGE_TIMES = {"t_cache": 0.0}
        return cached_nl, cached_lin

    # CCL / CAMB-direct shortcuts (still return a linear companion)
    if use_ccl and not skip_nl:
        from gpucosmo.jax.power import build_dense_pk_table_from_ccl

        t0 = time.perf_counter()
        pk_nl = build_dense_pk_table_from_ccl(
            params,
            z_max=z_max,
            n_z=n_z,
            n_k=n_k,
            nonlinear=True,
            nonlinear_prescription="hmcode",
            hmcode_logT_AGN=logT,
        )
        pk_lin = build_dense_pk_table_from_ccl(
            params,
            z_max=z_max,
            n_z=n_z,
            n_k=n_k,
            nonlinear=False,
            nonlinear_prescription="hmcode",
            hmcode_logT_AGN=logT,
        )
        stages["t_ccl"] = time.perf_counter() - t0
        _PK_CACHE[key_nl] = pk_nl
        _PK_CACHE[key_lin] = pk_lin
        _LAST_STAGE_TIMES = stages
        return pk_nl, pk_lin

    if camb_direct and not skip_nl:
        from gpucosmo.jax.eb.transfer_out import (
            linear_pk_from_camb_spectrum,
            nonlinear_pk_from_camb,
        )

        t0 = time.perf_counter()
        pk_nl = nonlinear_pk_from_camb(
            params,
            z_max=z_max,
            n_z=n_z,
            k_min=k_min,
            k_max=k_max,
            n_k=n_k,
            logT_AGN=logT,
        )
        stages["t_camb_nl"] = time.perf_counter() - t0
        t1 = time.perf_counter()
        pk_lin = linear_pk_from_camb_spectrum(
            params, z_max=z_max, n_z=n_z, k_min=k_min, k_max=k_max, n_k=n_k
        )
        stages["t_linear"] = time.perf_counter() - t1
        _PK_CACHE[key_nl] = pk_nl
        _PK_CACHE[key_lin] = pk_lin
        _LAST_STAGE_TIMES = stages
        return pk_nl, pk_lin

    device = os.environ.get("GPUCOSMO_EB_DEVICE", "0").lower() in (
        "1",
        "true",
        "yes",
    )
    if device:
        from gpucosmo.jax.eb.boltzmann_jax import build_device_pk_pair

        t0 = time.perf_counter()
        pk_nl, lin = build_device_pk_pair(
            params,
            z_max=z_max,
            n_z=n_z,
            n_k=n_k,
            k_min=k_min,
            k_max=k_max,
            logT_AGN=logT,
        )
        stages["t_device_pair"] = time.perf_counter() - t0
        if skip_nl:
            pk_nl = lin
        _PK_CACHE[key_lin] = lin
        _PK_CACHE[key_nl] = pk_nl
        _LAST_STAGE_TIMES = stages
        if timers:
            print(f"jax_eb stages: {stages}", flush=True)
        return pk_nl, lin

    t0 = time.perf_counter()
    lin = _build_linear(
        params,
        z_max=z_max,
        n_z=n_z,
        n_k=n_k,
        k_min=k_min,
        k_max=k_max,
        linear_source=src,
    )
    stages["t_linear"] = time.perf_counter() - t0

    if skip_nl:
        _PK_CACHE[key_lin] = lin
        _PK_CACHE[key_nl] = lin
        _LAST_STAGE_TIMES = stages
        if timers:
            print(f"jax_eb stages: {stages}", flush=True)
        return lin, lin

    t1 = time.perf_counter()
    pk_nl = _apply_nl(params, lin, logT=logT)
    stages["t_hmcode"] = time.perf_counter() - t1

    _PK_CACHE[key_lin] = lin
    _PK_CACHE[key_nl] = pk_nl
    _LAST_STAGE_TIMES = stages
    if timers:
        print(f"jax_eb stages: {stages}", flush=True)
    return pk_nl, lin


def build_jax_eb_pk_table(
    params: CosmologyParams,
    *,
    z_max: float = 3.0,
    n_z: int = 80,
    n_k: int = 256,
    nonlinear: bool = True,
    k_min: float = 5.0e-5,
    k_max: float = 100.0,
) -> MatterPowerTable:
    """Dense P(k,z). Prefer ``build_jax_eb_pk_pair`` for MCMC (NL+lin together)."""
    pk_nl, pk_lin = build_jax_eb_pk_pair(
        params,
        z_max=z_max,
        n_z=n_z,
        n_k=n_k,
        k_min=k_min,
        k_max=k_max,
    )
    return pk_nl if nonlinear else pk_lin


def linear_matter_power_table(params: CosmologyParams, **kw) -> MatterPowerTable:
    nonlinear = kw.pop("nonlinear", False)
    return build_jax_eb_pk_table(params, nonlinear=nonlinear, **kw)


__all__ = [
    "build_jax_eb_pk_pair",
    "build_jax_eb_pk_table",
    "clear_pk_cache",
    "last_stage_times",
    "linear_matter_power_table",
    "_PK_CACHE",
]
