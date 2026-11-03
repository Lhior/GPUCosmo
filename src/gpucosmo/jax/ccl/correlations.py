#!/usr/bin/env python3
# This file is part of GPUCosmo. Copyright (C) 2025-2026 Leonel Medina-Varela.
# SPDX-License-Identifier: GPL-3.0-or-later
# Derived in part from the following, translated to JAX and modified for GPUCosmo:
# - CCL (https://github.com/LSSTDESC/CCL), Copyright (c) 2016-2018, the LSSTDESC CCL
#   contributors.
#   Licence: LICENSES/CCL.txt.
# See NOTICE.md.
"""Angular correlation functions (``pyccl.correlation``) via JAX FFTLog."""

from __future__ import annotations

import jax.numpy as jnp

from gpucosmo.jax.fftlog import gamma_t_ng_fftlog_jax, w_theta_nn_fftlog_jax


def correlation(cosmo, *, ell, C_ell, theta, type: str = "NN", method: str = "fftlog"):
    """Real-space correlation at ``theta`` [deg] from C_ell.

    Supports CCL's FFTLog method for ``type='NN'`` (w(theta)) and
    ``type='NG'`` (gamma_t).
    """
    del cosmo
    if method.lower() != "fftlog":
        raise NotImplementedError("Only method='fftlog' is implemented")
    ell = jnp.asarray(ell, dtype=float)
    C_ell = jnp.asarray(C_ell, dtype=float)
    theta = jnp.atleast_1d(jnp.asarray(theta, dtype=float))
    if type == "NG":
        return gamma_t_ng_fftlog_jax(C_ell, ell, theta, n_theta=int(theta.shape[0]))
    if type == "NN":
        return w_theta_nn_fftlog_jax(ell, C_ell, theta)
    raise NotImplementedError(f"correlation type {type!r} not implemented")


__all__ = ["correlation"]
