#!/usr/bin/env python3
# This file is part of GPUCosmo. Copyright (C) 2025-2026 Leonel Medina-Varela.
# SPDX-License-Identifier: GPL-3.0-or-later
# Derived in part from the following, translated to JAX and modified for GPUCosmo:
# - CCL (https://github.com/LSSTDESC/CCL), Copyright (c) 2016-2018, the LSSTDESC CCL
#   contributors.
#   Licence: LICENSES/CCL.txt.
# See NOTICE.md.
"""Halo mass definitions (``pyccl.halos.massdef``)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from gpucosmo.jax.ccl.halos._sigma import omega_x, rho_x


def _squeeze_like(out, ref):
    return out[0] if np.ndim(ref) == 0 else out


def mass2radius_lagrangian(cosmo, M):
    """Lagrangian radius (comoving Mpc) of mass ``M`` [M_sun]."""
    M_use = jnp.atleast_1d(jnp.asarray(M, dtype=float))
    R = (M_use / (4.18879020479 * rho_x(cosmo, 1.0, "matter"))) ** (1.0 / 3.0)
    return _squeeze_like(R, M)


def convert_concentration(cosmo, *, c_old, Delta_old, Delta_new, n_iter: int = 60):
    """NFW concentration for a new overdensity: f(c_old) D_old = f(c_new) D_new.

    CCL uses a root finder (``ROOT_EPSREL``); here a fixed number of
    safeguarded Newton steps in ln c (differentiable, converged to ~1e-14).
    """
    c_old = jnp.asarray(c_old, dtype=float)

    def fx(x):
        return x**3 / (jnp.log1p(x) - x / (1 + x))

    target = fx(c_old) * Delta_old / Delta_new
    lc = jnp.log(c_old)
    for _ in range(n_iter):
        c = jnp.exp(lc)
        f = fx(c)
        dlnf = 3.0 - c * c / ((1 + c) ** 2 * (jnp.log1p(c) - c / (1 + c)))
        lc = lc - jnp.clip((jnp.log(f) - jnp.log(target)) / dlnf, -1.0, 1.0)
    return jnp.exp(lc)


class MassDef:
    """Halo mass definition: overdensity ``Delta`` w.r.t. matter / critical."""

    def __init__(self, Delta, rho_type):
        if isinstance(Delta, str):
            if Delta.isdigit():
                Delta = int(Delta)
            elif Delta not in ("fof", "vir"):
                raise ValueError(f"Unknown Delta type {Delta}.")
        if isinstance(Delta, (int, float)) and Delta < 0:
            raise ValueError("Delta must be a positive number.")
        if rho_type not in ("matter", "critical"):
            raise ValueError("rho_type must be {'matter', 'critical'}.")
        self.Delta = Delta
        self.rho_type = rho_type

    @property
    def name(self) -> str:
        if isinstance(self.Delta, (int, float)):
            return f"{self.Delta}{self.rho_type[0]}"
        return f"{self.Delta}"

    def __repr__(self):
        return f"MassDef(Delta={self.Delta}, rho_type={self.rho_type})"

    def __eq__(self, other):
        return isinstance(other, MassDef) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __call__(self):  # pyccl < 3 style ``MassDef200m()``
        return self

    def get_Delta_vir(self, cosmo, a):
        """Bryan & Norman (1998) virial overdensity for this ``rho_type``."""
        om = omega_x(cosmo, a, "matter")
        x = om - 1
        dv = 18 * np.pi**2 + 82 * x - 39 * x**2
        if self.rho_type == "matter":
            dv = dv / om
        return dv

    def get_Delta(self, cosmo, a):
        if self.Delta == "fof":
            raise ValueError("FoF masses don't have an associated overdensity."
                             "Nor can they be translated into other masses")
        if self.Delta == "vir":
            return self.get_Delta_vir(cosmo, a)
        return self.Delta

    def _get_Delta_m(self, cosmo, a):
        delta = self.get_Delta(cosmo, a)
        if self.rho_type == "matter":
            return delta
        return delta * omega_x(cosmo, a, self.rho_type) / omega_x(cosmo, a, "matter")

    def get_mass(self, cosmo, R, a):
        """Mass [M_sun] for a physical radius R [Mpc]."""
        R_use = jnp.atleast_1d(jnp.asarray(R, dtype=float))
        M = 4.18879020479 * rho_x(cosmo, a, self.rho_type) * self.get_Delta(cosmo, a) * R_use**3
        return _squeeze_like(M, R)

    def get_radius(self, cosmo, M, a):
        """Physical radius [Mpc] for mass M [M_sun]."""
        M_use = jnp.atleast_1d(jnp.asarray(M, dtype=float))
        R = (M_use / (4.18879020479 * self.get_Delta(cosmo, a)
                      * rho_x(cosmo, a, self.rho_type))) ** (1.0 / 3.0)
        return _squeeze_like(R, M)

    @classmethod
    def from_name(cls, name):
        table = {"200m": MassDef200m, "200c": MassDef200c, "500c": MassDef500c,
                 "vir": MassDefVir, "fof": MassDefFof}
        if name in table:
            return table[name]
        parser = {"c": "critical", "m": "matter"}
        if len(name) < 2 or name[-1] not in parser:
            raise ValueError("Could not parse mass definition string.")
        return cls(name[:-1], parser[name[-1]])

    @classmethod
    def create_instance(cls, input_):
        if isinstance(input_, cls):
            return input_
        return cls.from_name(input_)

    @classmethod
    def from_specs(cls, mass_def=None, *, mass_function=None, halo_bias=None, concentration=None):
        """Instantiate ``mass_def`` and any string-specified ingredients."""
        from gpucosmo.jax.ccl.halos.ingredients import Concentration, HaloBias, MassFunc

        specs = [(v, T) for v, T in ((mass_function, MassFunc), (halo_bias, HaloBias),
                                     (concentration, Concentration)) if v is not None]
        if mass_def is None:
            if any(isinstance(v, str) for v, _ in specs):
                raise ValueError("Need mass_def if an ingredient is a str.")
            mass_def = specs[0][0].mass_def
        else:
            mass_def = cls.create_instance(mass_def)
        out = [T.create_instance(v, mass_def=mass_def) for v, T in specs]
        if out and {x.mass_def for x in out} != {mass_def}:
            raise ValueError("Inconsistent mass definitions.")
        return (mass_def, *out)


MassDef200m = MassDef(200, "matter")
MassDef200c = MassDef(200, "critical")
MassDef500c = MassDef(500, "critical")
MassDefVir = MassDef("vir", "critical")
MassDefFof = MassDef("fof", "matter")


def mass_translator(*, mass_in, mass_out, concentration):
    """Translate masses between definitions assuming NFW (pyccl syntax)."""
    from gpucosmo.jax.ccl.halos.ingredients import Concentration

    mass_in = MassDef.create_instance(mass_in)
    mass_out = MassDef.create_instance(mass_out)
    concentration = Concentration.create_instance(concentration, mass_def=mass_in)

    def translate(cosmo, M, a):
        if mass_in == mass_out:
            return M
        c_in = concentration(cosmo, M, a)
        d_in = mass_in.get_Delta(cosmo, a) * omega_x(cosmo, a, mass_in.rho_type)
        r_in = mass_in.get_radius(cosmo, M, a)
        d_out = mass_out.get_Delta(cosmo, a) * omega_x(cosmo, a, mass_out.rho_type)
        c_out = convert_concentration(cosmo, c_old=c_in, Delta_old=d_in, Delta_new=d_out)
        return mass_out.get_mass(cosmo, r_in * c_out / c_in, a)

    return translate


__all__ = [
    "MassDef",
    "MassDef200c",
    "MassDef200m",
    "MassDef500c",
    "MassDefFof",
    "MassDefVir",
    "convert_concentration",
    "mass2radius_lagrangian",
    "mass_translator",
]
