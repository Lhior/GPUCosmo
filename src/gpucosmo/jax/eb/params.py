#!/usr/bin/env python3
"""Flat ΛCDM params and transfer k/z grids for CAMB-parity JAX EB."""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np

from gpucosmo.jax.eb import constants as C
from gpucosmo.jax.power import CosmologyParams


@dataclass(frozen=True)
class FlatLCDMParams:
    """CAMB-aligned flat cosmology for transfer / Cls (ΛCDM or CPL dark energy)."""

    Omega_c: float
    Omega_b: float
    h: float
    n_s: float
    sigma8: float
    As: float | None = None  # if None, set later via σ₈ rescale
    k_pivot: float = 0.05  # 1/Mpc
    T_CMB: float = C.COBE_CMB_TEMP
    YHe: float = C.YHE_DEFAULT
    tau_reio: float = 0.0544  # optical depth (reionization overlay)
    w0: float = -1.0  # CPL dark energy
    wa: float = 0.0
    # Neutrino sector. N_eff is shared over three states, so one massive
    # eigenstate at 0.06 eV leaves N_ur = 3.044 - 3.044/3 = 2.0293 massless.
    nnu: float = C.DEFAULT_NNU
    mnu: float = C.DEFAULT_MNU
    num_massive_nu: int = C.DEFAULT_NUM_MASSIVE_NU

    @classmethod
    def from_cosmo(cls, params: CosmologyParams, **kw) -> FlatLCDMParams:
        from gpucosmo.jax.eb.constants import yhe_bbn

        yhe = kw.pop("YHe", None)
        if yhe is None:
            yhe = yhe_bbn(float(params.Omega_b) * float(params.h) ** 2)
        return cls(
            Omega_c=float(params.Omega_c),
            Omega_b=float(params.Omega_b),
            h=float(params.h),
            n_s=float(params.n_s),
            sigma8=float(params.sigma8),
            YHe=float(yhe),
            **kw,
        )

    @property
    def Omega_m(self) -> float:
        return self.Omega_c + self.Omega_b

    @property
    def ombh2(self) -> float:
        return self.Omega_b * self.h**2

    @property
    def omch2(self) -> float:
        return self.Omega_c * self.h**2

    @property
    def Omega_g(self) -> float:
        # Scale with (T/T0)^4
        return C.OMEGA_G_H2 / (self.h**2) * (self.T_CMB / C.COBE_CMB_TEMP) ** 4

    @functools.cached_property
    def _nu(self) -> dict:
        """Neutrino eigenstates (``cls_jax._omega_from_physical``: CAMB 2.0.4 conventions)."""
        from gpucosmo.jax.eb.cls_jax import _omega_from_physical

        d = _omega_from_physical(self.ombh2, self.omch2, self.h, T_cmb=self.T_CMB, nnu=self.nnu,
                                 mnu=self.mnu, num_massive_nu=self.num_massive_nu)
        return {k: (float(v) if np.ndim(v) == 0 else np.asarray(v, dtype=np.float64))
                for k, v in d.items() if k in ("N_ur", "Omega_nu", "omnuh2", "nu_mass_am",
                                               "Omega_nu_mass", "Omega_nu_massive_today")}

    @property
    def N_ur(self) -> float:
        """N_eff remaining in massless states after the massive eigenstates."""
        return self._nu["N_ur"]

    @property
    def Omega_nu(self) -> float:
        """Massless neutrino species only."""
        return self._nu["Omega_nu"]

    @property
    def omnuh2(self) -> float:
        return float(np.sum(self._nu["omnuh2"]))

    @property
    def nu_mass_am(self):
        """``m c^2 / (k_B T)`` per massive eigenstate block (shape ``(n_eig,)``); 0 if massless."""
        return self._nu["nu_mass_am"]

    @property
    def Omega_nu_mass(self):
        """Density each massive eigenstate block would have if massless (shape ``(n_eig,)``)."""
        return self._nu["Omega_nu_mass"]

    @property
    def Omega_nu_massive_today(self) -> float:
        return self._nu["Omega_nu_massive_today"]

    @property
    def Omega_r(self) -> float:
        """Photons + massless neutrinos (the strictly a^-4 part)."""
        return self.Omega_g + self.Omega_nu

    @property
    def Omega_de(self) -> float:
        return max(
            1.0 - self.Omega_m - self.Omega_r - self.Omega_nu_massive_today, 0.0
        )

    @property
    def f_He(self) -> float:
        """n_He / n_H = YHe / (4 (1-YHe)) with not4≈MASS_RATIO."""
        return self.YHe / (C.MASS_RATIO_HE_H * (1.0 - self.YHe))

    @property
    def H0_mpc(self) -> float:
        """H0 in 1/Mpc (c=1)."""
        return self.h * C.H0_100_INV_MPC


def transfer_k_grid(
    k_min: float = 5.0e-5,
    k_max: float = 100.0,
    n_k: int = 256,
) -> np.ndarray:
    """Log-spaced k in 1/Mpc (CCL dense-table convention)."""
    return np.geomspace(k_min, k_max, n_k)


def transfer_z_grid(z_max: float = 3.0, n_z: int = 80) -> np.ndarray:
    return np.linspace(0.0, z_max, n_z)


__all__ = ["FlatLCDMParams", "transfer_k_grid", "transfer_z_grid"]
