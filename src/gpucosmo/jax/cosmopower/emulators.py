#!/usr/bin/env python3
"""Lazy CosmoPower-JAX emulator singletons."""

from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def _mpk_lin_emulator():
    from cosmopower_jax.cosmopower_jax import CosmoPowerJAX as CPJ

    return CPJ(probe="mpk_lin", verbose=False)


@lru_cache(maxsize=1)
def _mpk_nonlin_emulator():
    from cosmopower_jax.cosmopower_jax import CosmoPowerJAX as CPJ

    return CPJ(probe="mpk_nonlin", verbose=False)


def cosmopower_k_modes() -> np.ndarray:
    """Fixed log-spaced k grid [1/Mpc] from the bundled mpk_lin emulator."""
    return np.asarray(_mpk_lin_emulator().modes, dtype=float)


def predict_mpk_lin(params: dict) -> np.ndarray:
    """Single-sample linear P(k) prediction on the emulator k grid."""
    return np.asarray(_mpk_lin_emulator().predict(params), dtype=float)


def predict_mpk_nonlin(params: dict) -> np.ndarray:
    """Single-sample HMCode-emulated nonlinear P(k) on the emulator k grid."""
    return np.asarray(_mpk_nonlin_emulator().predict(params), dtype=float)


def clear_emulator_cache() -> None:
    """Drop lazily loaded CosmoPower-JAX emulator instances."""
    _mpk_lin_emulator.cache_clear()
    _mpk_nonlin_emulator.cache_clear()


__all__ = [
    "clear_emulator_cache",
    "cosmopower_k_modes",
    "predict_mpk_lin",
    "predict_mpk_nonlin",
]
