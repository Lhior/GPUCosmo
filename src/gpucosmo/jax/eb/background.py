#!/usr/bin/env python3
# This file is part of GPUCosmo. Copyright (C) 2025-2026 Leonel Medina-Varela.
# SPDX-License-Identifier: GPL-3.0-or-later
# Derived in part from the following, translated to JAX and modified for GPUCosmo:
# - CAMB (https://github.com/cmbant/CAMB), Copyright 2000-2025 Antony Lewis and Anthony Challinor.
#   Licence: LICENSES/CAMB.txt.
# See NOTICE.md.
"""Background — CAMB ``dtauda`` / densities / TimeOfz (equations.f90 / results.f90)."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from gpucosmo.jax.eb.dark_energy import cpl_f_de
import numpy as np

from gpucosmo.jax.eb import constants as C
from gpucosmo.jax.eb.params import FlatLCDMParams
from gpucosmo.jax.eb.transfer import CambLikeBackground  # re-export compat

jax.config.update("jax_enable_x64", True)


@dataclass
class BackgroundTables:
    """Precomputed a↔τ tables matching CAMB conformal-time integration."""

    a: np.ndarray
    tau: np.ndarray
    params: FlatLCDMParams

    # Densities today in CAMB 8πG a^4 ρ / 3 units with H0=1 → Ω_i
    # Internal evolution uses grho_i(a) = Ω_i / a^{n} with H0=1.

    @property
    def tau0(self) -> float:
        return float(self.tau[-1])

    def E(self, a: jnp.ndarray) -> jnp.ndarray:
        p = self.params
        a = jnp.asarray(a, dtype=jnp.float64)
        # CPL: ρ_de(a)/ρ_de0 = a^{-3(1+w0+wa)} exp(-3 wa (1-a))
        w0 = getattr(p, "w0", -1.0)
        wa = getattr(p, "wa", 0.0)
        f_de = cpl_f_de(a, w0, wa)
        return jnp.sqrt(
            p.Omega_r * a ** (-4) + p.Omega_m * a ** (-3) + p.Omega_de * f_de
        )

    def dtauda(self, a: jnp.ndarray) -> jnp.ndarray:
        """CAMB ``dtauda`` in H0=1 units: dτ/da = 1/(a² E).

        Physical conformal time [Mpc] = τ_H0=1 / H0_mpc.
        """
        return 1.0 / jnp.maximum(a**2 * self.E(a), 1.0e-30)

    def tau_physical(self, a: jnp.ndarray) -> jnp.ndarray:
        """Conformal time in Mpc (CAMB units)."""
        return self.tau_of_a(a) / self.params.H0_mpc

    def tau_of_a(self, a: jnp.ndarray | float) -> jnp.ndarray:
        """τ(a) in H0=1 units via table interp."""
        a = jnp.asarray(a, dtype=jnp.float64)
        return jnp.interp(a, self.a, self.tau)

    def a_of_tau(self, tau: jnp.ndarray | float) -> jnp.ndarray:
        """a(τ) with τ in H0=1 units."""
        tau = jnp.asarray(tau, dtype=jnp.float64)
        return jnp.interp(tau, self.tau, self.a)

    def TimeOfz(self, z: float | np.ndarray) -> np.ndarray:
        """CAMB ``TimeOfz``: conformal time in Mpc at redshift z."""
        z = np.asarray(z, dtype=float)
        a = 1.0 / (1.0 + z)
        tau_h0 = np.interp(a, self.a, self.tau)
        return tau_h0 / self.params.H0_mpc

    def grho_components(self, a: float | jnp.ndarray) -> dict[str, jnp.ndarray]:
        """8πG ρ a² / 3 in H0=1 units (= Ω_i * a^{2-n}).

        Returns dict with keys g, r (massless ν), b, c, de, tot.
        """
        a = jnp.asarray(a, dtype=jnp.float64)
        p = self.params
        grhog = p.Omega_g / a**2
        grhor = p.Omega_nu / a**2
        grhob = p.Omega_b / a
        grhoc = p.Omega_c / a
        w0 = getattr(p, "w0", -1.0)
        wa = getattr(p, "wa", 0.0)
        f_de = cpl_f_de(a, w0, wa)
        grhov = p.Omega_de * f_de * a**2
        grho = grhog + grhor + grhob + grhoc + grhov
        return {
            "g": grhog,
            "r": grhor,
            "b": grhob,
            "c": grhoc,
            "de": grhov,
            "tot": grho,
        }

    def adot(self, a: float | jnp.ndarray) -> jnp.ndarray:
        """ȧ = a² H = a E (H0=1)."""
        a = jnp.asarray(a, dtype=jnp.float64)
        return a * self.E(a)


def build_background(
    params: FlatLCDMParams,
    *,
    a_min: float = 1.0e-8,
    n_a: int = 20000,
) -> BackgroundTables:
    """Integrate τ(a) = ∫ dtauda from a_min → 1 (H0=1 units)."""
    # Dense early + late grid
    a_lo = np.geomspace(a_min, 0.1, n_a // 2)
    a_hi = np.linspace(0.1, 1.0, n_a // 2 + 1)[1:]
    a = np.unique(np.concatenate([a_lo, a_hi]))

    def E(aa):
        w0 = getattr(params, "w0", -1.0)
        wa = getattr(params, "wa", 0.0)
        f_de = aa ** (-3.0 * (1.0 + w0 + wa)) * np.exp(-3.0 * wa * (1.0 - aa))
        return np.sqrt(
            params.Omega_r * aa ** (-4)
            + params.Omega_m * aa ** (-3)
            + params.Omega_de * f_de
        )

    dtauda = 1.0 / np.maximum(a**2 * E(a), 1e-30)
    # Cumulative trapezoid; τ(a_min)≈a_min/(a_min² E) * a_min/2 ≈ radiation τ∝a
    # Seed: in RD, τ ≈ a / (a² * sqrt(Ω_r)/a²) = a/sqrt(Ω_r) → τ(a_min)=a_min/sqrt(Ω_r)
    tau0_seed = a[0] / np.sqrt(max(params.Omega_r, 1e-30))
    da = np.diff(a)
    mid = 0.5 * (dtauda[:-1] + dtauda[1:])
    tau = np.concatenate([[tau0_seed], tau0_seed + np.cumsum(mid * da)])
    return BackgroundTables(a=a, tau=tau, params=params)


# Keep CambLikeBackground importable from background
__all__ = ["BackgroundTables", "CambLikeBackground", "build_background"]
