#!/usr/bin/env python3
# This file is part of GPUCosmo. Copyright (C) 2025-2026 Leonel Medina-Varela.
# SPDX-License-Identifier: GPL-3.0-or-later
# Derived in part from the following, translated to JAX and modified for GPUCosmo:
# - CAMB (https://github.com/cmbant/CAMB), Copyright 2000-2025 Antony Lewis and Anthony Challinor.
#   Licence: LICENSES/CAMB.txt.
# See NOTICE.md.
"""outtransf + Transfer_GetMatterPowerData assembly (results.f90).

Linear T(k,z) sources (``GPUCOSMO_EB_LINEAR_SOURCE``):
  - ``camb`` (default): CAMB ``get_matter_transfer_data`` — exact Fortran
    Transfer_tot/k², assembled with our ``Transfer_GetMatterPowerData`` formula
  - ``boltzmann``: live hierarchy (``evolve.py`` / ``hierarchy.py``)
"""

from __future__ import annotations

import os
import warnings

import numpy as np

from gpucosmo.jax.eb.background import build_background
from gpucosmo.jax.eb.hierarchy import HierarchyConfig
from gpucosmo.jax.eb.params import FlatLCDMParams, transfer_k_grid, transfer_z_grid
from gpucosmo.jax.eb.thermo import build_thermo
from gpucosmo.jax.power import CosmologyParams, MatterPowerTable

_TWO_PI2 = 2.0 * np.pi * np.pi


def _camb_params(
    fp: FlatLCDMParams,
    *,
    As: float = 2.0e-9,
    WantTransfer: bool = True,
):
    import camb

    # mnu=0 matches CCL default massless-ν setup used in GGL bakeoff.
    return camb.set_params(
        H0=100.0 * fp.h,
        ombh2=fp.ombh2,
        omch2=fp.omch2,
        As=As,
        ns=fp.n_s,
        YHe=fp.YHe,
        mnu=0.0,
        WantTransfer=WantTransfer,
    )


def assemble_matter_power(
    T_tot: np.ndarray,
    k_mpc: np.ndarray,
    P_R: np.ndarray,
    h: float,
) -> np.ndarray:
    """CAMB ``Transfer_GetMatterPowerData`` (results.f90 ~3225).

    Returns P in (Mpc/h)³.
    """
    return (T_tot**2) * k_mpc * _TWO_PI2 * (h**3) * P_R


def primordial_PR(
    k_mpc: np.ndarray, As: float, n_s: float, k_pivot: float
) -> np.ndarray:
    return As * (k_mpc / k_pivot) ** (n_s - 1.0)


def sigma8_top_hat(k_hmpc: np.ndarray, pk_h: np.ndarray, R: float = 8.0) -> float:
    k = np.asarray(k_hmpc, dtype=float)
    pk = np.asarray(pk_h, dtype=float)
    x = k * R
    w = np.where(
        x < 1e-3,
        1.0 - x**2 / 10.0,
        3.0 * (np.sin(x) - x * np.cos(x)) / np.maximum(x**3, 1e-30),
    )
    delta2 = k**3 * pk / (2.0 * np.pi**2)
    return float(np.sqrt(np.trapezoid(delta2 * w**2, np.log(k))))


def transfer_tot_from_camb(
    params: FlatLCDMParams,
    k_mpc: np.ndarray,
    z_grid: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Dump CAMB Transfer_tot/k² onto (z, k) grid. Returns (T, As)."""
    import camb
    from camb.model import Transfer_tot

    As = params.As if params.As is not None else 2.1e-9
    kmax_h = float(np.max(k_mpc) / params.h * 1.1)
    z_list = list(np.sort(np.asarray(z_grid, dtype=float))[::-1])  # earliest first

    pars = _camb_params(params, As=As, WantTransfer=True)
    pars.set_matter_power(redshifts=z_list, kmax=max(kmax_h, 5.0), silent=True)
    results = camb.get_results(pars)
    tr = results.get_matter_transfer_data()
    kh = np.asarray(tr.transfer_z("k/h"), dtype=float)
    # Transfer_tot is 1-based index 7 → 0-based 6
    T_raw = np.asarray(tr.transfer_data[Transfer_tot - 1, :, :], dtype=float)  # (nk, nz)
    zs_camb = np.asarray(z_list, dtype=float)

    k_h_target = k_mpc / params.h
    log_kh = np.log(np.maximum(kh, 1e-30))
    z_asc = zs_camb[::-1]
    T_asc = T_raw[:, ::-1]

    T_out = np.zeros((len(z_grid), len(k_mpc)), dtype=float)
    for iz, z in enumerate(z_grid):
        if z >= z_asc[-1]:
            T_at_z = T_asc[:, -1]
        elif z <= z_asc[0]:
            T_at_z = T_asc[:, 0]
        else:
            j = int(np.searchsorted(z_asc, z))
            j = min(max(j, 1), len(z_asc) - 1)
            w = (z - z_asc[j - 1]) / max(z_asc[j] - z_asc[j - 1], 1e-30)
            T_at_z = (1.0 - w) * T_asc[:, j - 1] + w * T_asc[:, j]
        T_out[iz] = np.exp(
            np.interp(
                np.log(np.maximum(k_h_target, 1e-30)),
                log_kh,
                np.log(np.maximum(np.abs(T_at_z), 1e-30)),
            )
        ) * np.sign(np.interp(np.log(k_h_target), log_kh, T_at_z))
        # Prefer linear interp on T itself (can be negative? usually +)
        T_out[iz] = np.interp(np.log(k_h_target), log_kh, T_at_z)
    return T_out, float(As)


def transfer_tot_from_boltzmann(
    params: FlatLCDMParams,
    k_mpc: np.ndarray,
    z_grid: np.ndarray,
) -> np.ndarray:
    """Live hierarchy Transfer_tot/k².

    Modes with k > ``GPUCOSMO_EB_BOLTZMANN_KMAX`` (default 3/Mpc) are filled
    from CAMB Transfer_tot so full dense tables stay practical; the gate window
    k∈[10⁻³,1] is always live.
    """
    from gpucosmo.jax.eb.evolve import evolve_transfer_grid

    k_mpc = np.asarray(k_mpc, dtype=float)
    k_cut = float(os.environ.get("GPUCOSMO_EB_BOLTZMANN_KMAX", "3.0"))
    bg = build_background(params)
    thermo = build_thermo(bg)
    live = k_mpc <= k_cut
    T = np.zeros((len(z_grid), len(k_mpc)), dtype=float)
    if np.any(live):
        T[:, live] = evolve_transfer_grid(
            k_mpc[live], z_grid, bg, thermo, HierarchyConfig()
        )
    if np.any(~live):
        T_camb, _ = transfer_tot_from_camb(params, k_mpc, z_grid)
        T[:, ~live] = T_camb[:, ~live]
    return T


def _pk_table_from_T(
    T: np.ndarray,
    k_mpc: np.ndarray,
    z_grid: np.ndarray,
    params: FlatLCDMParams,
    As: float,
) -> MatterPowerTable:
    P_R = primordial_PR(k_mpc, As, params.n_s, params.k_pivot)
    pk_h = np.zeros_like(T)
    for iz in range(len(z_grid)):
        pk_h[iz] = assemble_matter_power(T[iz], k_mpc, P_R, params.h)
    k_h = k_mpc / params.h
    iz0 = int(np.argmin(np.abs(z_grid)))
    s8 = sigma8_top_hat(k_h, pk_h[iz0], R=8.0)
    pk_h *= (params.sigma8 / max(s8, 1e-30)) ** 2
    pk_mpc = pk_h / (params.h**3)
    return MatterPowerTable(
        log_k=np.log(k_mpc),
        z=z_grid,
        log_pk=np.log(np.maximum(pk_mpc, 1e-30)),
    )


def linear_pk_from_camb_spectrum(
    params: CosmologyParams,
    *,
    z_max: float = 3.0,
    n_z: int = 80,
    k_min: float = 5.0e-5,
    k_max: float = 100.0,
    n_k: int = 256,
) -> MatterPowerTable:
    """Linear P via CAMB ``get_matter_power_spectrum`` (CCL units).

    Equivalent to ``Transfer_GetMatterPowerData`` on CAMB's native k samples,
    interpolated onto the dense CCL-like grid and σ₈-rescaled.
    """
    import camb

    fp = FlatLCDMParams.from_cosmo(params)
    k_mpc = transfer_k_grid(k_min, k_max, n_k)
    z_grid = transfer_z_grid(z_max, n_z)
    z_list = list(np.sort(z_grid)[::-1])
    kmax_h = float(np.max(k_mpc) / fp.h * 1.05)

    pars = _camb_params(fp, As=2.0e-9, WantTransfer=True)
    pars.NonLinear = camb.model.NonLinear_none
    pars.set_matter_power(redshifts=z_list, kmax=max(kmax_h, 5.0), silent=True)
    results = camb.get_results(pars)
    s8 = float(results.get_sigma8_0())
    kh, z_pk, pk = results.get_matter_power_spectrum(
        minkh=float(k_min / fp.h),
        maxkh=min(kmax_h, 200.0),
        npoints=max(n_k * 2, 512),
    )
    kh = np.asarray(kh, dtype=float)
    z_pk = np.asarray(z_pk, dtype=float)
    pk = np.asarray(pk, dtype=float) * (fp.sigma8 / max(s8, 1e-30)) ** 2

    k_h_t = k_mpc / fp.h
    log_kh = np.log(np.maximum(kh, 1e-30))
    log_pk_out = np.zeros((len(z_grid), len(k_mpc)), dtype=float)
    for iz, z in enumerate(z_grid):
        j = int(np.argmin(np.abs(z_pk - z)))
        # linear interp in z if neighbors exist
        if len(z_pk) > 1:
            order = np.argsort(z_pk)
            z_asc = z_pk[order]
            pk_asc = pk[order]
            if z <= z_asc[0]:
                pk_z = pk_asc[0]
            elif z >= z_asc[-1]:
                pk_z = pk_asc[-1]
            else:
                jj = int(np.searchsorted(z_asc, z))
                w = (z - z_asc[jj - 1]) / max(z_asc[jj] - z_asc[jj - 1], 1e-30)
                pk_z = (1.0 - w) * pk_asc[jj - 1] + w * pk_asc[jj]
        else:
            pk_z = pk[j]
        pk_i = np.exp(
            np.interp(np.log(k_h_t), log_kh, np.log(np.maximum(pk_z, 1e-30)))
        )
        log_pk_out[iz] = np.log(np.maximum(pk_i / (fp.h**3), 1e-30))
    return MatterPowerTable(log_k=np.log(k_mpc), z=z_grid, log_pk=log_pk_out)


def linear_pk_table(
    params: CosmologyParams,
    *,
    z_max: float = 3.0,
    n_z: int = 80,
    k_min: float = 5.0e-5,
    k_max: float = 100.0,
    n_k: int = 256,
    source: str | None = None,
) -> MatterPowerTable:
    """Linear P(k,z) in CCL units (k: 1/Mpc, P: Mpc³)."""
    source = source or os.environ.get("GPUCOSMO_EB_LINEAR_SOURCE", "camb")
    fp = FlatLCDMParams.from_cosmo(params)
    k_mpc = transfer_k_grid(k_min, k_max, n_k)
    z_grid = transfer_z_grid(z_max, n_z)

    if source == "boltzmann":
        As = fp.As if fp.As is not None else 2.1e-9
        T = transfer_tot_from_boltzmann(fp, k_mpc, z_grid)
        return _pk_table_from_T(T, k_mpc, z_grid, fp, As)

    if source == "camb_transfer":
        # Explicit Transfer_tot/k² assembly (slightly noisier after regrid)
        T, As = transfer_tot_from_camb(fp, k_mpc, z_grid)
        return _pk_table_from_T(T, k_mpc, z_grid, fp, As)

    return linear_pk_from_camb_spectrum(
        params, z_max=z_max, n_z=n_z, k_min=k_min, k_max=k_max, n_k=n_k
    )


def nonlinear_pk_from_camb(
    params: CosmologyParams,
    *,
    z_max: float = 3.0,
    n_z: int = 80,
    k_min: float = 5.0e-5,
    k_max: float = 100.0,
    n_k: int = 256,
    logT_AGN: float = 7.7,
) -> MatterPowerTable:
    """CAMB ``mead2020_feedback`` NL P on dense grid (CCL units)."""
    import camb

    fp = FlatLCDMParams.from_cosmo(params)
    k_mpc = transfer_k_grid(k_min, k_max, n_k)
    z_grid = transfer_z_grid(z_max, n_z)
    z_list = list(np.sort(z_grid)[::-1])
    kmax_h = float(np.max(k_mpc) / fp.h * 1.1)

    pars = _camb_params(fp, As=2.0e-9, WantTransfer=True)
    pars.NonLinear = camb.model.NonLinear_pk
    pars.NonLinearModel.set_params(
        halofit_version="mead2020_feedback",
        HMCode_logT_AGN=logT_AGN,
    )
    pars.set_matter_power(redshifts=z_list, kmax=max(kmax_h, 5.0), silent=True)
    results = camb.get_results(pars)
    s8 = float(results.get_sigma8_0())
    scale = (fp.sigma8 / max(s8, 1e-30)) ** 2

    kh, z_pk, pk = results.get_matter_power_spectrum(
        minkh=float(k_min / fp.h),
        maxkh=min(kmax_h, 200.0),
        npoints=max(n_k * 2, 512),
    )
    kh = np.asarray(kh, dtype=float)
    z_pk = np.asarray(z_pk, dtype=float)
    pk = np.asarray(pk, dtype=float) * scale  # (nz, nk) in (Mpc/h)^3

    k_h_t = k_mpc / fp.h
    log_pk_out = np.zeros((len(z_grid), len(k_mpc)), dtype=float)
    log_kh = np.log(np.maximum(kh, 1e-30))
    order = np.argsort(z_pk)
    z_asc = z_pk[order]
    pk_asc = pk[order]
    for iz, z in enumerate(z_grid):
        if z <= z_asc[0]:
            pk_z = pk_asc[0]
        elif z >= z_asc[-1]:
            pk_z = pk_asc[-1]
        else:
            jj = int(np.searchsorted(z_asc, z))
            w = (z - z_asc[jj - 1]) / max(z_asc[jj] - z_asc[jj - 1], 1e-30)
            pk_z = (1.0 - w) * pk_asc[jj - 1] + w * pk_asc[jj]
        pk_i = np.exp(
            np.interp(np.log(k_h_t), log_kh, np.log(np.maximum(pk_z, 1e-30)))
        )
        log_pk_out[iz] = np.log(np.maximum(pk_i / (fp.h**3), 1e-30))
    return MatterPowerTable(log_k=np.log(k_mpc), z=z_grid, log_pk=log_pk_out)


__all__ = [
    "assemble_matter_power",
    "linear_pk_from_camb_spectrum",
    "linear_pk_table",
    "nonlinear_pk_from_camb",
    "primordial_PR",
    "sigma8_top_hat",
    "transfer_tot_from_boltzmann",
    "transfer_tot_from_camb",
]
