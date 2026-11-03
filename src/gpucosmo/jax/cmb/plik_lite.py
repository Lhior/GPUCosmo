#!/usr/bin/env python3
# This file is part of GPUCosmo. Copyright (C) 2025-2026 Leonel Medina-Varela.
# SPDX-License-Identifier: GPL-3.0-or-later
# Derived in part from the following, translated to JAX and modified for GPUCosmo:
# - Cobaya (https://github.com/CobayaSampler/cobaya), Copyright 2017-2026 Jesus Torrado and
#   Antony Lewis.
#   Licence: LICENSES/Cobaya.txt.
# See NOTICE.md.
"""JAX Planck plik_lite likelihood (mirrors Cobaya's ``planck_2018_highl_plik.TTTEEE_lite_native``)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import numpy as np

import jax
import jax.numpy as jnp
from jax import jit

jax.config.update("jax_enable_x64", True)


def _parse_dataset(dataset_path: str) -> dict:
    config: dict = {}
    with open(dataset_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip()
                if v.lower() in ("true", "false"):
                    config[k] = v.lower() == "true"
                elif v.replace(".", "").replace("-", "").isdigit():
                    config[k] = int(v) if "." not in v and "e" not in v.lower() else float(v)
                else:
                    config[k] = v
    return config


def load_plik_lite_data(
    data_dir: str | Path,
    dataset_file: str = "plik_lite_v22.dataset",
) -> dict:
    base = Path(data_dir)
    dataset_path = base / dataset_file
    config = _parse_dataset(str(dataset_path))
    base_dir = dataset_path.parent

    use_cl = [c.lower() for c in config.get("use_cl", "TT TE EE").split()]
    nbintt = config["nbintt"]
    nbinte = config["nbinte"]
    nbinee = config["nbinee"]
    lmax = config["lmax"]
    bin_lmin_offset = config["bin_lmin_offset"]

    data = np.loadtxt(base_dir / config["data"])
    blmin = np.loadtxt(base_dir / config["blmin"]).astype(int) + bin_lmin_offset
    blmax = np.loadtxt(base_dir / config["blmax"]).astype(int) + bin_lmin_offset
    bweight_raw = np.loadtxt(base_dir / config["weights"])
    ls = np.arange(len(bweight_raw)) + bin_lmin_offset
    weights_cl = bweight_raw * 2 * np.pi / ls / (ls + 1)
    weights_cl = np.hstack((np.zeros(bin_lmin_offset), weights_cl))

    nbins = nbintt + nbinee + nbinte
    cov_file_binary = config.get("cov_file_binary")
    cov_file_ascii = config.get("cov_file")
    cov_path_bin = base_dir / cov_file_binary if cov_file_binary else None
    cov_path_asc = base_dir / cov_file_ascii if cov_file_ascii else None

    if cov_file_binary and cov_path_bin and cov_path_bin.is_file():
        from scipy.io import FortranFile

        with FortranFile(str(cov_path_bin), "r") as f:
            cov = f.read_reals(dtype=float).reshape((nbins, nbins))
        cov = np.tril(cov) + np.tril(cov, -1).T
    elif cov_file_ascii and cov_path_asc and cov_path_asc.is_file():
        cov = np.loadtxt(cov_path_asc)
        if cov.size != nbins * nbins:
            cov = cov.reshape((nbins, nbins))
    else:
        raise FileNotFoundError(f"Covariance not found under {base_dir}")

    cl_names = ["tt", "te", "ee"]
    nbin_list = [nbintt, nbinte, nbinee]
    used_bins: list[np.ndarray] = []
    used_indices: list[int] = []
    offset = 0
    for cl, nbin in zip(cl_names, nbin_list):
        if cl in use_cl:
            used_bins.append(np.arange(nbin, dtype=int))
            used_indices.extend((used_bins[-1] + offset).tolist())
        else:
            used_bins.append(np.arange(0, dtype=int))
        offset += nbin

    used_indices_arr = np.array(used_indices, dtype=int)
    x_data_cl = data[used_indices_arr, 1]
    cov_used = cov[np.ix_(used_indices_arr, used_indices_arr)]
    invcov = np.linalg.inv(cov_used)

    return {
        "blmin": blmin,
        "blmax": blmax,
        "weights_Cl": weights_cl,
        "used_bins": used_bins,
        "X_data": x_data_cl,
        "invcov": invcov,
        "lmax": lmax,
        "bin_lmin_offset": bin_lmin_offset,
    }


def build_plik_lite_loglike(data: dict) -> Callable[..., jnp.ndarray]:
    """Return ``loglike(ctt, cte, cee, A_planck)`` with D_l in muK^2."""
    blmin = data["blmin"]
    blmax = data["blmax"]
    weights = jnp.asarray(data["weights_Cl"])
    used_bins = data["used_bins"]
    x_data = jnp.asarray(data["X_data"])
    invcov = jnp.asarray(data["invcov"])

    def bin_cl(cell: jnp.ndarray, tp: int) -> jnp.ndarray:
        bins = used_bins[tp]
        if len(bins) == 0:
            return jnp.array([])
        out = []
        for i in bins:
            sl = slice(int(blmin[i]), int(blmax[i]) + 1)
            w = weights[blmin[i] : blmax[i] + 1]
            out.append(jnp.dot(cell[sl], w))
        return jnp.stack(out) if out else jnp.array([])

    def loglike(ctt: jnp.ndarray, cte: jnp.ndarray, cee: jnp.ndarray, a_planck: jnp.ndarray):
        cl_list = []
        for tp, cell in enumerate([ctt, cte, cee]):
            b = bin_cl(cell, tp)
            if b.size > 0:
                cl_list.append(b)
        cl = jnp.concatenate(cl_list) if cl_list else jnp.array([])
        cl = cl / (a_planck**2)
        diff = x_data - cl
        chi2 = diff @ invcov @ diff
        return -0.5 * chi2

    return jit(loglike)
