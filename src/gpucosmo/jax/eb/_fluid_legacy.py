#!/usr/bin/env python3
"""Legacy experimental reduced-fluid P(k) path (EH default; optional fluid)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from gpucosmo.jax.eb.transfer import (
    CambLikeBackground,
    assemble_pk_camb_convention,
    eisenstein_hu_transfer_kh,
    fluid_derivs_sync,
    integrate_growth_factor,
    primordial_pk_camb,
    sigma8_from_pk_mpc,
)
from gpucosmo.jax.power import CosmologyParams, MatterPowerTable

jax.config.update("jax_enable_x64", True)

_IX_ETA = 0
_IX_DC = 1
_IX_DB = 2
_IX_VB = 3
_IX_DG = 4
_IX_TG = 5
_IX_DUR = 6
_IX_TUR = 7
_IX_SUR = 8
_NVAR = 9


def fluid_derivs_reduced(
    a: jnp.ndarray,
    y: jnp.ndarray,
    *,
    k: float,
    bg: CambLikeBackground,
    opacity: float,
) -> jnp.ndarray:
    """Reduced dy/da scaffold — prefer ``hierarchy.derivs`` for production."""
    del opacity
    eta, dc, db, vb, dg, tg, dur, tur, sur = (
        y[_IX_ETA],
        y[_IX_DC],
        y[_IX_DB],
        y[_IX_VB],
        y[_IX_DG],
        y[_IX_TG],
        y[_IX_DUR],
        y[_IX_TUR],
        y[_IX_SUR],
    )
    # Minimal matter growth stub (not CAMB-accurate)
    ea = bg.E(a)
    ddc = -k * vb / jnp.maximum(a**2 * ea, 1e-30)  # placeholder
    ddb = ddc
    dvb = -1.5 * bg.Omega_m * db / jnp.maximum(a**2 * ea, 1e-30) - vb / jnp.maximum(a, 1e-30)
    return jnp.array([0.0, ddc, ddb, dvb, 0.0, 0.0, 0.0, 0.0, 0.0])


def _opacity_approx(a: jnp.ndarray, bg: CambLikeBackground) -> jnp.ndarray:
    return 1.0e-8 / jnp.maximum(a**2, 1e-30)


def _initial_conditions(k: float, a_ini: float, bg: CambLikeBackground) -> jnp.ndarray:
    y = np.zeros(_NVAR)
    y[_IX_DC] = a_ini
    y[_IX_DB] = a_ini
    return jnp.asarray(y)


def _evolve_one_k(k: float, a_out: jnp.ndarray, bg: CambLikeBackground) -> jnp.ndarray:
    a_ini = 1e-6
    y0 = _initial_conditions(k, a_ini, bg)
    a_grid = jnp.unique(
        jnp.concatenate(
            [
                jnp.array([a_ini]),
                jnp.geomspace(a_ini, 1e-3, 100),
                jnp.linspace(0.1, 1.0, 100),
                a_out,
            ]
        )
    )

    def step(y, a_pair):
        a0, a1 = a_pair
        h = a1 - a0

        def rhs(aa, yy):
            return fluid_derivs_reduced(aa, yy, k=k, bg=bg, opacity=_opacity_approx(aa, bg))

        k1 = rhs(a0, y)
        k2 = rhs(a0 + 0.5 * h, y + 0.5 * h * k1)
        k3 = rhs(a0 + 0.5 * h, y + 0.5 * h * k2)
        k4 = rhs(a1, y + h * k3)
        return y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), y + (h / 6.0) * (
            k1 + 2 * k2 + 2 * k3 + k4
        )

    a_pairs = jnp.stack([a_grid[:-1], a_grid[1:]], axis=1)
    _, ys = jax.lax.scan(step, y0, a_pairs)
    y_all = jnp.concatenate([y0[None, :], ys], axis=0)
    grhoc = 3.0 * bg.Omega_c / a_grid
    grhob = 3.0 * bg.Omega_b / a_grid
    d_m = (grhob * y_all[:, _IX_DB] + grhoc * y_all[:, _IX_DC]) / jnp.maximum(
        grhob + grhoc, 1e-30
    )
    T = d_m / jnp.maximum(k**2, 1e-30)
    return jnp.interp(a_out, a_grid, T)


def fluid_transfer_pk_table(
    params: CosmologyParams,
    *,
    z_grid: np.ndarray,
    k_hmpc: np.ndarray,
    As: float = 2.1e-9,
) -> MatterPowerTable:
    """P_lin via EH (default). ``GPUCOSMO_EB_FLUID=1`` enables experimental fluid."""
    import os

    bg = CambLikeBackground(
        Omega_c=params.Omega_c,
        Omega_b=params.Omega_b,
        h=params.h,
        n_s=params.n_s,
        As=As,
    )
    z_grid = np.asarray(z_grid, dtype=float)
    k_hmpc = np.asarray(k_hmpc, dtype=float)
    k_mpc = k_hmpc * params.h
    a_out = jnp.asarray(1.0 / (1.0 + z_grid))

    T_eh = eisenstein_hu_transfer_kh(
        k_hmpc, params.Omega_m, params.h, params.Omega_b, wiggled=True, camb_correct=False
    )
    P_eh0 = (k_mpc ** params.n_s) * (T_eh**2)
    try:
        import pyccl as ccl

        cosmo = ccl.Cosmology(
            Omega_c=params.Omega_c,
            Omega_b=params.Omega_b,
            h=params.h,
            sigma8=params.sigma8,
            n_s=params.n_s,
            transfer_function="eisenstein_hu",
            matter_power_spectrum="linear",
        )
        D_z = np.asarray(
            [float(ccl.growth_factor(cosmo, float(a))) for a in np.asarray(a_out)],
            dtype=float,
        )
    except Exception:
        a_ode = jnp.unique(
            jnp.concatenate([jnp.geomspace(1e-3, 0.1, 400), jnp.linspace(0.1, 1.0, 400)])
        )
        D_ode = integrate_growth_factor(bg.Omega_m, a_ode)
        D_z = np.asarray(jnp.interp(a_out, a_ode, D_ode))
    D0 = float(D_z[0]) if z_grid[0] == 0.0 else float(D_z[np.argmin(np.abs(z_grid))])
    growth = (D_z / max(D0, 1e-30)) ** 2
    P_eh = np.outer(growth, P_eh0)
    s8 = sigma8_from_pk_mpc(
        k_mpc, P_eh[0] if z_grid[0] == 0 else P_eh[np.argmin(z_grid)], params.h
    )
    P_eh = P_eh * (params.sigma8 / max(s8, 1e-30)) ** 2

    use_fluid = os.environ.get("GPUCOSMO_EB_FLUID", "0").lower() in ("1", "true", "yes")
    if not use_fluid:
        return MatterPowerTable(
            log_k=np.log(k_mpc), z=z_grid, log_pk=np.log(np.maximum(P_eh, 1e-30))
        )

    T_kz = np.zeros((z_grid.size, k_hmpc.size))
    for i, kh in enumerate(k_hmpc):
        try:
            T_kz[:, i] = np.asarray(_evolve_one_k(float(kh), a_out, bg))
        except Exception:
            T_kz[:, i] = np.nan
    P_R = np.asarray(
        primordial_pk_camb(jnp.asarray(k_hmpc), bg.As, bg.n_s, bg.k_pivot, bg.h)
    )
    pk = np.zeros_like(T_kz)
    for iz in range(z_grid.size):
        T = T_kz[iz]
        if not np.all(np.isfinite(T)):
            continue
        pk[iz] = assemble_pk_camb_convention(T, k_hmpc, P_R, params.h) / (params.h**3)
    bad = ~np.isfinite(pk) | (pk <= 0)
    pk = np.where(bad, P_eh, pk)
    s8 = sigma8_from_pk_mpc(
        k_mpc, pk[0] if z_grid[0] == 0 else pk[np.argmin(z_grid)], params.h
    )
    pk = pk * (params.sigma8 / max(s8, 1e-30)) ** 2
    return MatterPowerTable(
        log_k=np.log(k_mpc), z=z_grid, log_pk=np.log(np.maximum(pk, 1e-30))
    )


__all__ = ["fluid_derivs_reduced", "fluid_derivs_sync", "fluid_transfer_pk_table"]
