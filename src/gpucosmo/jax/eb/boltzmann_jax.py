#!/usr/bin/env python3
"""On-device linear P(k,z): Recfast + hierarchy scan + σ₈ normalize.

High-k (k > BOLTZMANN_KMAX): Eisenstein–Hu splice matched at the cut
(no CAMB in the hot path). Keep arrays on device until MatterPowerTable.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np

from gpucosmo.jax.eb.evolve_jax import (
    downsample_thermo_for_evolve,
    evolve_transfer_grid_jax,
)
from gpucosmo.jax.eb.params import FlatLCDMParams, transfer_k_grid, transfer_z_grid
from gpucosmo.jax.eb.thermo_jax import build_thermo_jax
from gpucosmo.jax.eb.transfer import eisenstein_hu_transfer_kh
from gpucosmo.jax.power import CosmologyParams, MatterPowerTable

jax.config.update("jax_enable_x64", True)

_TWO_PI2 = 2.0 * jnp.pi * jnp.pi


def _params_dict(fp: FlatLCDMParams) -> dict:
    return {
        "Omega_b": float(fp.Omega_b),
        "Omega_c": float(fp.Omega_c),
        "Omega_g": float(fp.Omega_g),
        "Omega_nu": float(fp.Omega_nu),
        "Omega_de": float(fp.Omega_de),
        "Omega_m": float(fp.Omega_m),
        "Omega_r": float(fp.Omega_r),
        "H0_mpc": float(fp.H0_mpc),
        "h": float(fp.h),
        "n_s": float(fp.n_s),
        "sigma8": float(fp.sigma8),
        "YHe": float(fp.YHe),
        "T_CMB": float(fp.T_CMB),
        "ombh2": float(fp.ombh2),
        "f_He": float(fp.f_He),
        "tau_reio": float(fp.tau_reio),
        "w0": float(getattr(fp, "w0", -1.0)),
        "wa": float(getattr(fp, "wa", 0.0)),
    }


def _eh_transfer_tot_jnp(
    k_mpc: jnp.ndarray,
    z_grid: jnp.ndarray,
    *,
    Omega_m: float,
    h: float,
    Omega_b: float,
    n_s: float,
    As: float,
    k_pivot: float,
) -> jnp.ndarray:
    """EH-shaped Transfer_tot/k² on device (host EH once, then jnp growth)."""
    k_np = np.asarray(k_mpc)
    T_eh = eisenstein_hu_transfer_kh(
        k_np / h, Omega_m, h, Omega_b, wiggled=True, camb_correct=True
    )
    T_eh = jnp.asarray(T_eh, dtype=jnp.float64)
    P_R = As * (k_mpc / k_pivot) ** (n_s - 1.0)
    T0 = jnp.sqrt(
        jnp.maximum(
            k_mpc ** (n_s - 1.0) * T_eh**2 / (_TWO_PI2 * jnp.maximum(P_R, 1e-30)),
            0.0,
        )
    )
    a = 1.0 / (1.0 + z_grid)
    D = a / jnp.maximum(a[-1], 1e-30)
    return T0[None, :] * D[:, None]


def _assemble_pk_h(T: jnp.ndarray, k_mpc: jnp.ndarray, P_R: jnp.ndarray, h: float) -> jnp.ndarray:
    """Vectorized Transfer_GetMatterPowerData → P in (Mpc/h)³, shape (n_z, n_k)."""
    return (T**2) * k_mpc[None, :] * _TWO_PI2 * (h**3) * P_R[None, :]


def _sigma8_top_hat_jnp(k_hmpc: jnp.ndarray, pk_h: jnp.ndarray, R: float = 8.0) -> jnp.ndarray:
    x = k_hmpc * R
    w = jnp.where(
        x < 1e-3,
        1.0 - x**2 / 10.0,
        3.0 * (jnp.sin(x) - x * jnp.cos(x)) / jnp.maximum(x**3, 1e-30),
    )
    delta2 = k_hmpc**3 * pk_h / (2.0 * jnp.pi**2)
    return jnp.sqrt(jnp.trapezoid(delta2 * w**2, jnp.log(k_hmpc)))


def transfer_tot_from_boltzmann_jax(
    fp: FlatLCDMParams,
    k_mpc: np.ndarray | jnp.ndarray,
    z_grid: np.ndarray | jnp.ndarray,
    *,
    thermo: dict | None = None,
) -> jnp.ndarray:
    """Device hierarchy for k ≤ cut; EH splice for high-k. Returns jnp (n_z, n_k)."""
    k_mpc = jnp.asarray(k_mpc, dtype=jnp.float64)
    z_grid = jnp.asarray(z_grid, dtype=jnp.float64)
    k_cut = float(os.environ.get("GPUCOSMO_EB_BOLTZMANN_KMAX", "1.0"))
    if thermo is None:  # starting before the evolved modes do (``thermo_jax.tau_table_start``)
        thermo = build_thermo_jax(fp, k_max=min(k_cut, float(np.max(np.asarray(k_mpc)))))
    thermo = downsample_thermo_for_evolve(thermo)
    params = _params_dict(fp)
    bg_tau_a = {"a": thermo["bg_a"], "tau": thermo["bg_tau"]}
    live_np = np.asarray(k_mpc) <= k_cut
    As = fp.As if fp.As is not None else 2.1e-9
    n_z = int(z_grid.shape[0])
    n_k = int(k_mpc.shape[0])
    T = jnp.zeros((n_z, n_k), dtype=jnp.float64)

    if np.any(live_np):
        T_live = evolve_transfer_grid_jax(
            k_mpc[live_np], z_grid, params, thermo, bg_tau_a
        )
        # Sanitize unstable high-k modes from fixed-step scan
        T_live = jnp.where(jnp.isfinite(T_live), T_live, 0.0)
        T_live = jnp.clip(T_live, -1.0e20, 1.0e20)
        T = T.at[:, live_np].set(T_live)

    # Always build EH; use for high-k and to repair non-finite live cells
    T_eh = _eh_transfer_tot_jnp(
        k_mpc,
        z_grid,
        Omega_m=float(fp.Omega_m),
        h=float(fp.h),
        Omega_b=float(fp.Omega_b),
        n_s=float(fp.n_s),
        As=float(As),
        k_pivot=float(fp.k_pivot),
    )
    if np.any(live_np):
        i_cut = int(np.where(live_np)[0][-1])
        scale = T[:, i_cut] / jnp.maximum(jnp.abs(T_eh[:, i_cut]), 1e-30)
        scale = jnp.where(jnp.isfinite(scale), scale, 1.0)
        T_eh_scaled = T_eh * scale[:, None]
    else:
        T_eh_scaled = T_eh

    if np.any(~live_np):
        T = T.at[:, ~live_np].set(T_eh_scaled[:, ~live_np])

    # Repair any remaining non-finite live entries with scaled EH
    bad = ~jnp.isfinite(T) | (jnp.abs(T) > 1.0e18)
    T = jnp.where(bad, T_eh_scaled, T)
    return T


def linear_pk_table_jax(
    params: CosmologyParams,
    *,
    z_max: float = 3.0,
    n_z: int = 80,
    k_min: float = 5.0e-5,
    k_max: float = 100.0,
    n_k: int = 256,
) -> MatterPowerTable:
    """Linear P(k,z) in CCL units from on-device Boltzmann + EH high-k."""
    fp = FlatLCDMParams.from_cosmo(params)
    k_mpc = jnp.asarray(transfer_k_grid(k_min, k_max, n_k), dtype=jnp.float64)
    z_grid = jnp.asarray(transfer_z_grid(z_max, n_z), dtype=jnp.float64)
    As = fp.As if fp.As is not None else 2.1e-9
    T = transfer_tot_from_boltzmann_jax(fp, k_mpc, z_grid)
    P_R = As * (k_mpc / fp.k_pivot) ** (fp.n_s - 1.0)
    pk_h = _assemble_pk_h(T, k_mpc, P_R, float(fp.h))
    k_h = k_mpc / fp.h
    iz0 = int(jnp.argmin(jnp.abs(z_grid)))
    s8 = _sigma8_top_hat_jnp(k_h, pk_h[iz0], R=8.0)
    pk_h = pk_h * (fp.sigma8 / jnp.maximum(s8, 1e-30)) ** 2
    pk_mpc = pk_h / (fp.h**3)
    return MatterPowerTable(
        log_k=np.asarray(jnp.log(k_mpc)),
        z=np.asarray(z_grid),
        log_pk=np.asarray(jnp.log(jnp.maximum(pk_mpc, 1e-30))),
    )


def build_device_pk_pair(
    params: CosmologyParams,
    *,
    z_max: float = 3.0,
    n_z: int = 80,
    n_k: int = 256,
    k_min: float = 5.0e-5,
    k_max: float = 100.0,
    logT_AGN: float = 7.7,
) -> tuple[MatterPowerTable, MatterPowerTable]:
    """Device linear + HMcode → (pk_nl, pk_lin)."""
    lin = linear_pk_table_jax(
        params, z_max=z_max, n_z=n_z, k_min=k_min, k_max=k_max, n_k=n_k
    )
    use_jax_hm = os.environ.get("GPUCOSMO_EB_HMCODE_JAX", "1").lower() not in (
        "0",
        "false",
        "no",
    )
    if use_jax_hm:
        try:
            from gpucosmo.jax.eb.hmcode_jax import nonlinear_pk_hmcode_feedback_jax

            pk_nl = nonlinear_pk_hmcode_feedback_jax(
                lin,
                Omega_m=float(params.Omega_m),
                Omega_b=float(params.Omega_b),
                h=float(params.h),
                n_s=float(params.n_s),
                logT_AGN=logT_AGN,
            )
            return pk_nl, lin
        except Exception:
            pass
    from gpucosmo.jax.eb.hmcode import nonlinear_pk_hmcode_feedback

    pk_nl = nonlinear_pk_hmcode_feedback(
        lin,
        Omega_m=float(params.Omega_m),
        Omega_b=float(params.Omega_b),
        h=float(params.h),
        n_s=float(params.n_s),
        logT_AGN=logT_AGN,
    )
    return pk_nl, lin


__all__ = [
    "build_device_pk_pair",
    "linear_pk_table_jax",
    "transfer_tot_from_boltzmann_jax",
]
