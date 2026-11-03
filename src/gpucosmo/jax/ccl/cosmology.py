#!/usr/bin/env python3
# This file is part of GPUCosmo. Copyright (C) 2025-2026 Leonel Medina-Varela.
# SPDX-License-Identifier: GPL-3.0-or-later
# Derived in part from the following, translated to JAX and modified for GPUCosmo:
# - CCL (https://github.com/LSSTDESC/CCL), Copyright (c) 2016-2018, the LSSTDESC CCL
#   contributors.
#   Licence: LICENSES/CCL.txt.
# See NOTICE.md.
"""Tabulated JAX cosmology exposing the pyccl background / P(k) interface.

Tracers and ``angular_cl`` only need chi(a), a(chi), E(a), D(a), f(a) and the
linear / non-linear P(k, a). CCL stores all background quantities as GSL
Akima splines; storing the same nodes here reproduces CCL to spline
precision. Build with :meth:`Cosmology.from_ccl` for parity work, or with
:meth:`Cosmology.from_tables` inside jitted likelihoods (all arrays may be
traced).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from gpucosmo.jax.ccl.pk2d import Pk2D
from gpucosmo.jax.gsl_splines import Spline1D, akima_spline, spline_deriv, spline_eval

jax.config.update("jax_enable_x64", True)

# pyccl.physical_constants (CCL 3.x)
CLIGHT_HMPC = 2997.92458  # c / (100 km/s/Mpc) in Mpc/h
RHO_CRITICAL = 277536627203.6856  # h^2 M_sun / Mpc^3
IA_C1 = 5.0e-14  # h^-2 M_sun^-1 Mpc^3 (Brown et al. 2002)
# pyccl.spline_params
A_SPLINE_MIN = 0.1
K_MIN = 5.0e-5
K_MAX = 1000.0

DEFAULT_POWER_SPECTRUM = "delta_matter:delta_matter"


@dataclass(frozen=True)
class Cosmology:
    """Background splines, P(k) tables and scalar parameters."""

    chi_of_a: Spline1D
    a_of_chi: Spline1D
    e_of_a: Spline1D
    d_of_a: Spline1D
    f_of_a: Spline1D
    params: dict
    pk_lin: Pk2D | None = None
    pk_nl: Pk2D | None = None

    # -- pyccl-style accessors -------------------------------------------
    def __getitem__(self, key: str) -> Any:
        if key in self.params:
            return self.params[key]
        if key in ("Omega_k", "sigma_0", "Omega_g", "m_nu"):
            return 0.0
        raise KeyError(key)

    def _clip_a(self, a, s: Spline1D):
        return jnp.clip(jnp.asarray(a, dtype=float), s.x[0], s.x[-1])

    def comoving_radial_distance(self, a):
        return spline_eval(self.chi_of_a, self._clip_a(a, self.chi_of_a))

    def scale_factor_of_chi(self, chi):
        chi = jnp.clip(jnp.asarray(chi, dtype=float), self.a_of_chi.x[0], self.a_of_chi.x[-1])
        return spline_eval(self.a_of_chi, chi)

    def h_over_h0(self, a):
        return spline_eval(self.e_of_a, self._clip_a(a, self.e_of_a))

    def growth_factor(self, a):
        return spline_eval(self.d_of_a, self._clip_a(a, self.d_of_a))

    def growth_rate(self, a):
        return spline_eval(self.f_of_a, self._clip_a(a, self.f_of_a))

    def get_linear_power(self, name: str = DEFAULT_POWER_SPECTRUM) -> Pk2D:
        if self.pk_lin is None:
            raise ValueError("Cosmology has no linear power spectrum")
        return self.pk_lin

    def get_nonlin_power(self, name: str = DEFAULT_POWER_SPECTRUM) -> Pk2D:
        if self.pk_nl is None:
            raise ValueError("Cosmology has no non-linear power spectrum")
        return self.pk_nl

    def compute_nonlin_power(
        self, matter_power_spectrum: str = "halofit", *, pk_linear: Pk2D | None = None
    ) -> "Cosmology":
        """Return a copy whose non-linear P(k) is built from the linear one.

        Mirrors ``pyccl.Cosmology._compute_nonlin_power``: ``'halofit'``
        applies the CCL-exact Takahashi Halofit
        (:func:`gpucosmo.jax.ccl.halofit.halofit_pk2d`) to the linear Pk2D,
        ``'linear'`` reuses it. Jittable and differentiable.
        """
        pkl = self.get_linear_power() if pk_linear is None else pk_linear
        mps = str(matter_power_spectrum).lower()
        if mps == "halofit":
            from gpucosmo.jax.ccl.halofit import halofit_pk2d

            pk_nl = halofit_pk2d(self, pkl)
        elif mps == "linear":
            pk_nl = pkl
        else:
            raise ValueError(
                f"matter_power_spectrum={matter_power_spectrum!r} cannot be computed "
                "in JAX (supported: 'halofit', 'linear'); pass pk_nonlin instead"
            )
        return dataclasses.replace(self, pk_lin=pkl, pk_nl=pk_nl)

    def with_halofit(self, pk_linear: Pk2D | None = None) -> "Cosmology":
        """Shorthand for ``compute_nonlin_power('halofit')``."""
        return self.compute_nonlin_power("halofit", pk_linear=pk_linear)

    def linear_matter_power(self, k, a):
        return self.get_linear_power()(k, a, self)

    def nonlin_matter_power(self, k, a):
        return self.get_nonlin_power()(k, a, self)

    def parse_pk2d(self, p_of_k_a, *, is_linear: bool = False) -> Pk2D:
        """Resolve ``p_of_k_a`` like ``pyccl.Cosmology.parse_pk2d``."""
        if p_of_k_a is None or (
            isinstance(p_of_k_a, str)
            and p_of_k_a in (DEFAULT_POWER_SPECTRUM, "nonlinear")
        ):
            return self.get_linear_power() if is_linear else self.get_nonlin_power()
        if isinstance(p_of_k_a, str):
            if p_of_k_a == "linear":
                return self.get_linear_power()
            raise ValueError(f"Unknown power spectrum {p_of_k_a!r}")
        if isinstance(p_of_k_a, Pk2D):
            return p_of_k_a
        try:  # pyccl.Pk2D (e.g. halo-model IA power spectrum)
            import pyccl

            if isinstance(p_of_k_a, pyccl.Pk2D):
                return Pk2D.from_ccl(p_of_k_a)
        except ImportError:
            pass
        raise TypeError(f"Unsupported p_of_k_a of type {type(p_of_k_a)}")

    # -- constructors -----------------------------------------------------
    @classmethod
    def from_tables(
        cls,
        *,
        z,
        chi,
        h_over_h0,
        growth,
        growth_rate=None,
        Omega_m,
        h,
        pk_linear: Pk2D | None = None,
        pk_nonlin: Pk2D | None = None,
        matter_power_spectrum: str | None = None,
        **params,
    ) -> "Cosmology":
        """Background from tables on an ascending redshift grid (z[0] = 0).

        With ``matter_power_spectrum='halofit'`` (and no ``pk_nonlin``) the
        non-linear power is computed from ``pk_linear`` with the CCL-exact
        JAX Halofit; its background (Omega_m(a), Omega_de(a), f_nu) uses the
        CCL parameters in ``params`` (``Omega_c``/``Omega_b`` or ``Omega_m``,
        ``h``, optional ``w0``, ``wa``, ``m_nu``, ``Neff``, ``T_CMB``, ...).
        """
        z = jnp.asarray(z, dtype=float)
        a = (1.0 / (1.0 + z))[::-1]
        chi_a = jnp.asarray(chi, dtype=float)[::-1]
        e_a = jnp.asarray(h_over_h0, dtype=float)[::-1]
        d_a = jnp.asarray(growth, dtype=float)[::-1]
        d_spl = akima_spline(a, d_a)
        if growth_rate is None:
            f_a = a * spline_deriv(d_spl, a) / d_a
        else:
            f_a = jnp.asarray(growth_rate, dtype=float)[::-1]
        out = cls(
            chi_of_a=akima_spline(a, chi_a),
            a_of_chi=akima_spline(chi_a[::-1], a[::-1]),
            e_of_a=akima_spline(a, e_a),
            d_of_a=d_spl,
            f_of_a=akima_spline(a, f_a),
            params={"Omega_m": Omega_m, "h": h, **params},
            pk_lin=pk_linear,
            pk_nl=pk_nonlin,
        )
        if pk_nonlin is None and matter_power_spectrum is not None:
            out = out.compute_nonlin_power(matter_power_spectrum)
        return out

    @classmethod
    def from_ccl(
        cls,
        cosmo,
        *,
        pk_linear=None,
        pk_nonlin=None,
        with_power: bool = True,
    ) -> "Cosmology":
        """Mirror a ``pyccl.Cosmology`` on CCL's own background spline nodes."""
        import pyccl as ccl
        from pyccl.pyutils import loglin_spacing

        sp = ccl.spline_params
        a = loglin_spacing(
            sp.A_SPLINE_MINLOG, sp.A_SPLINE_MIN, sp.A_SPLINE_MAX,
            sp.A_SPLINE_NLOG, sp.A_SPLINE_NA,
        )
        chi_a = np.asarray(ccl.comoving_radial_distance(cosmo, a), dtype=float)
        e_a = np.asarray(ccl.h_over_h0(cosmo, a), dtype=float)
        d_a = np.asarray(ccl.growth_factor(cosmo, a), dtype=float)
        f_a = np.asarray(ccl.growth_rate(cosmo, a), dtype=float)
        # a(chi): CCL tabulates on a linear chi grid with dchi <= 5 Mpc.
        chi0, chif = chi_a[-1], chi_a[0]
        n_chi = int((chif - chi0) / 5.0)
        chi_lin = np.linspace(chi0, chif, n_chi)
        a_chi = np.asarray(ccl.scale_factor_of_chi(cosmo, chi_lin), dtype=float)
        a_chi[0], a_chi[-1] = a[-1], a[0]
        if with_power:
            if pk_linear is None:
                pk_linear = cosmo.get_linear_power()
            if pk_nonlin is None:
                pk_nonlin = cosmo.get_nonlin_power()
        to_jax = lambda p: None if p is None else (
            p if isinstance(p, Pk2D) else Pk2D.from_ccl(p)
        )
        params = {
            key: float(cosmo[key])
            for key in ("Omega_m", "Omega_c", "Omega_b", "h", "n_s")
        }
        for key in ("sigma8", "A_s"):
            val = cosmo[key]
            if val is not None and np.isfinite(val):
                params[key] = float(val)
        # Derived C-level background parameters (radiation, massive nu, w0/wa)
        # used by the JAX Halofit (compute_nonlin_power).
        from gpucosmo.jax.ccl.halofit import background_params_from_ccl

        params.update(background_params_from_ccl(cosmo))
        return cls(
            chi_of_a=akima_spline(jnp.asarray(a), jnp.asarray(chi_a)),
            a_of_chi=akima_spline(jnp.asarray(chi_lin), jnp.asarray(a_chi)),
            e_of_a=akima_spline(jnp.asarray(a), jnp.asarray(e_a)),
            d_of_a=akima_spline(jnp.asarray(a), jnp.asarray(d_a)),
            f_of_a=akima_spline(jnp.asarray(a), jnp.asarray(f_a)),
            params=params,
            pk_lin=to_jax(pk_linear),
            pk_nl=to_jax(pk_nonlin),
        )


jax.tree_util.register_dataclass(
    Cosmology,
    data_fields=[
        "chi_of_a", "a_of_chi", "e_of_a", "d_of_a", "f_of_a",
        "params", "pk_lin", "pk_nl",
    ],
    meta_fields=[],
)


__all__ = [
    "A_SPLINE_MIN",
    "CLIGHT_HMPC",
    "Cosmology",
    "DEFAULT_POWER_SPECTRUM",
    "IA_C1",
    "K_MAX",
    "K_MIN",
    "RHO_CRITICAL",
]
