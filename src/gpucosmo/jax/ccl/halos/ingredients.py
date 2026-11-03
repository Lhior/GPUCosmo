#!/usr/bin/env python3
# This file is part of GPUCosmo. Copyright (C) 2025-2026 Leonel Medina-Varela.
# SPDX-License-Identifier: GPL-3.0-or-later
# Derived in part from the following, translated to JAX and modified for GPUCosmo:
# - CCL (https://github.com/LSSTDESC/CCL), Copyright (c) 2016-2018, the LSSTDESC CCL
#   contributors.
#   Licence: LICENSES/CCL.txt.
# See NOTICE.md.
"""Mass functions, halo bias and concentration-mass relations (pyccl.halos).

Formulae are transcribed from ``pyccl.halos.{hmfunc,hbias,concentration}``;
sigma(M) and dln sigma / dlog10 M come from CCL's bicubic spline
(:mod:`gpucosmo.jax.ccl.halos._sigma`). Every ``__call__`` is jittable and
differentiable w.r.t. the cosmology tables and parameters.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from gpucosmo.jax.ccl.halos._sigma import (
    dlnsigM_dlogM,
    get_delta_c,
    omega_x,
    rho_m0,
    sigmaM,
)
from gpucosmo.jax.ccl.halos.massdef import MassDef


def _all_subclasses(cls):
    out = []
    for sub in cls.__subclasses__():
        out.append(sub)
        out.extend(_all_subclasses(sub))
    return out


class HMIngredients:
    """Base class: mass-definition bookkeeping and name registry."""

    name = None
    _mass_def_strict_always = False

    def __init__(self, *, mass_def, mass_def_strict=True):
        mass_def = MassDef.create_instance(mass_def)
        self.mass_def_strict = mass_def_strict
        self._check_mass_def(mass_def)
        self.mass_def = mass_def
        self._setup()

    def _check_mass_def_strict(self, mass_def) -> bool:
        return False

    def _setup(self) -> None:
        pass

    def _check_mass_def(self, mass_def) -> None:
        msg = f"{type(self).__name__} is not defined for {mass_def.name} masses"
        if self._check_mass_def_strict(mass_def):
            if self._mass_def_strict_always:
                raise ValueError(f"{msg} and this requirement cannot be relaxed.")
            if self.mass_def_strict:
                raise ValueError(f"{msg}. To relax this check set `mass_def_strict=False`.")

    @classmethod
    def from_name(cls, name):
        mod = {p.name: p for p in _all_subclasses(cls) if getattr(p, "name", None)}
        if name not in mod:
            raise KeyError(f"Invalid model {name}.")
        return mod[name]

    @classmethod
    def create_instance(cls, input_, **kwargs):
        if isinstance(input_, cls):
            return input_
        if isinstance(input_, str):
            return cls.from_name(input_)(**kwargs)
        raise TypeError(f"Expected {cls.__name__} or str but received {type(input_).__name__}.")

    def _get_logM_sigM(self, cosmo, M, a, *, return_dlns=False):
        logM = jnp.log10(M)
        sigM = sigmaM(cosmo, M, a)
        if not return_dlns:
            return logM, sigM
        return logM, sigM, dlnsigM_dlogM(cosmo, M, a)


def _squeeze(out, M):
    return out[0] if np.ndim(M) == 0 else out


# == mass functions ============================================================
class MassFunc(HMIngredients):
    """dn/dlog10M = f(sigma) rho_m / M dln sigma^-1 / dlog10 M  [Mpc^-3]."""

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        raise NotImplementedError

    def __call__(self, cosmo, M, a):
        M_use = jnp.atleast_1d(jnp.asarray(M, dtype=float))
        logM, sigM, dlns = self._get_logM_sigM(cosmo, M_use, a, return_dlns=True)
        f = self._get_fsigma(cosmo, sigM, a, 2.302585092994046 * logM)
        return _squeeze(f * rho_m0(cosmo) * dlns / M_use, M)


def _interp_table(ld, xp, fp):
    """``scipy.interpolate.interp1d`` (linear) on the Tinker Delta tables."""
    return jnp.interp(ld, jnp.asarray(xp), jnp.asarray(fp))


_TINKER_LDELTA = np.log10(np.array([200., 300., 400., 600., 800., 1200., 1600., 2400., 3200.]))


class MassFuncTinker08(MassFunc):
    name = "Tinker08"

    def __init__(self, *, mass_def="200m", mass_def_strict=True):
        super().__init__(mass_def=mass_def, mass_def_strict=mass_def_strict)

    def _check_mass_def_strict(self, mass_def):
        return mass_def.Delta == "fof"

    def _setup(self):
        self.alpha = np.array([0.186, 0.200, 0.212, 0.218, 0.248, 0.255, 0.260, 0.260, 0.260])
        self.beta = np.array([1.47, 1.52, 1.56, 1.61, 1.87, 2.13, 2.30, 2.53, 2.66])
        self.gamma = np.array([2.57, 2.25, 2.05, 1.87, 1.59, 1.51, 1.46, 1.44, 1.41])
        self.phi = np.array([1.19, 1.27, 1.34, 1.45, 1.58, 1.80, 1.97, 2.24, 2.44])

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        ld = jnp.log10(self.mass_def._get_Delta_m(cosmo, a))
        pA = _interp_table(ld, _TINKER_LDELTA, self.alpha) * a**0.14
        pa = _interp_table(ld, _TINKER_LDELTA, self.beta) * a**0.06
        pd = 10.0 ** (-((0.75 / (ld - 1.8750612633)) ** 1.2))
        pb = _interp_table(ld, _TINKER_LDELTA, self.gamma) * a**pd
        pc = _interp_table(ld, _TINKER_LDELTA, self.phi)
        return pA * ((pb / sigM) ** pa + 1) * jnp.exp(-pc / sigM**2)


class MassFuncTinker10(MassFunc):
    name = "Tinker10"

    def __init__(self, *, mass_def="200m", mass_def_strict=True, norm_all_z=False):
        self.norm_all_z = norm_all_z
        super().__init__(mass_def=mass_def, mass_def_strict=mass_def_strict)

    def _check_mass_def_strict(self, mass_def):
        return mass_def.Delta == "fof"

    def _setup(self):
        self.t_alpha = np.array([0.368, 0.363, 0.385, 0.389, 0.393, 0.365, 0.379, 0.355, 0.327])
        self.t_beta = np.array([0.589, 0.585, 0.544, 0.543, 0.564, 0.623, 0.637, 0.673, 0.702])
        self.t_gamma = np.array([0.864, 0.922, 0.987, 1.09, 1.20, 1.34, 1.50, 1.68, 1.81])
        self.t_phi = np.array([-0.729, -0.789, -0.910, -1.05, -1.20, -1.26, -1.45, -1.50, -1.49])
        self.t_eta = np.array([-0.243, -0.261, -0.261, -0.273, -0.278, -0.301, -0.301,
                               -0.319, -0.336])
        self.t_p = np.array([-0.158, -0.195, -0.213, -0.254, -0.281, -0.349, -0.367,
                             -0.435, -0.504])
        self.t_q = np.array([0.0128, 0.0128, 0.0143, 0.0154, 0.0172, 0.0174, 0.0199,
                             0.0203, 0.0205])

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        ld = jnp.log10(self.mass_def._get_Delta_m(cosmo, a))
        nu = get_delta_c(cosmo, a, "EdS_approx") / sigM
        a = jnp.clip(a, 0.25, 1)
        pa = _interp_table(ld, _TINKER_LDELTA, self.t_eta) * a ** (-0.27)
        pb = _interp_table(ld, _TINKER_LDELTA, self.t_beta) * a ** (-0.20)
        pc = _interp_table(ld, _TINKER_LDELTA, self.t_gamma) * a**0.01
        pd = _interp_table(ld, _TINKER_LDELTA, self.t_phi) * a**0.08
        pA0 = _interp_table(ld, _TINKER_LDELTA, self.t_alpha)
        if self.norm_all_z:
            z = 1 / a - 1
            pp = _interp_table(ld, _TINKER_LDELTA, self.t_p)
            pq = _interp_table(ld, _TINKER_LDELTA, self.t_q)
            pA0 = pA0 * jnp.exp(z * (pp + pq * z))
        return nu * pA0 * (1 + (pb * nu) ** (-2 * pd)) * (
            nu ** (2 * pa) * jnp.exp(-0.5 * pc * nu**2))


class MassFuncSheth99(MassFunc):
    name = "Sheth99"

    def __init__(self, *, mass_def="fof", mass_def_strict=True, use_delta_c_fit=False):
        self.use_delta_c_fit = use_delta_c_fit
        super().__init__(mass_def=mass_def, mass_def_strict=mass_def_strict)

    def _check_mass_def_strict(self, mass_def):
        return mass_def.Delta != "fof"

    def _setup(self):
        self.A, self.p, self.a = 0.21615998645, 0.3, 0.707

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        kind = "NakamuraSuto97" if self.use_delta_c_fit else "EdS"
        nu = get_delta_c(cosmo, a, kind) / sigM
        return nu * self.A * (1.0 + (self.a * nu**2) ** (-self.p)) * jnp.exp(-self.a * nu**2 / 2.0)


class MassFuncPress74(MassFunc):
    name = "Press74"

    def __init__(self, *, mass_def="fof", mass_def_strict=True):
        super().__init__(mass_def=mass_def, mass_def_strict=mass_def_strict)
        self._norm = np.sqrt(2 / np.pi)

    def _check_mass_def_strict(self, mass_def):
        return mass_def.Delta != "fof"

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        nu = get_delta_c(cosmo, a, kind="EdS") / sigM
        return self._norm * nu * jnp.exp(-0.5 * nu**2)


class MassFuncDespali16(MassFunc):
    name = "Despali16"

    def __init__(self, *, mass_def="200m", mass_def_strict=True, ellipsoidal=False):
        self.ellipsoidal = ellipsoidal
        super().__init__(mass_def=mass_def, mass_def_strict=mass_def_strict)

    def _check_mass_def_strict(self, mass_def):
        return mass_def.Delta == "fof"

    def _setup(self):
        vals = {True: (0.3953, -0.1768, 0.7057, 0.2125, 0.3268, 0.2206, 0.1937, -0.04570),
                False: (0.3292, -0.1362, 0.7665, 0.2263, 0.4332, 0.2488, 0.2554, -0.1151)}
        self._coef = vals[self.ellipsoidal]

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        A0, A1, a0, a1, a2, p0, p1, p2 = self._coef
        delta_c = get_delta_c(cosmo, a, "NakamuraSuto97")
        dv = self.mass_def.get_Delta_vir(cosmo, a)
        x = jnp.log10(self.mass_def.get_Delta(cosmo, a) / dv)
        A = A1 * x + A0
        aa = (a2 * x + a1) * x + a0
        p = (p2 * x + p1) * x + p0
        nu_p = aa * (delta_c / sigM) ** 2
        return 2.0 * A * jnp.sqrt(nu_p / 2.0 / np.pi) * (jnp.exp(-0.5 * nu_p) * (1.0 + nu_p**-p))


class MassFuncJenkins01(MassFunc):
    name = "Jenkins01"

    def __init__(self, *, mass_def="fof", mass_def_strict=True):
        super().__init__(mass_def=mass_def, mass_def_strict=mass_def_strict)

    def _check_mass_def_strict(self, mass_def):
        return mass_def.Delta != "fof"

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        return 0.315 * jnp.exp(-jnp.abs(-jnp.log(sigM) + 0.61) ** 3.8)


class MassFuncAngulo12(MassFunc):
    name = "Angulo12"

    def __init__(self, *, mass_def="fof", mass_def_strict=True):
        super().__init__(mass_def=mass_def, mass_def_strict=mass_def_strict)

    def _check_mass_def_strict(self, mass_def):
        return mass_def.Delta != "fof"

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        return 0.201 * ((2.08 / sigM) ** 1.7 + 1.0) * jnp.exp(-1.172 / sigM**2)


# == halo bias =================================================================
class HaloBias(HMIngredients):
    """Halo bias b(sigma_M)."""

    def _get_bsigma(self, cosmo, sigM, a):
        raise NotImplementedError

    def __call__(self, cosmo, M, a):
        M_use = jnp.atleast_1d(jnp.asarray(M, dtype=float))
        _, sigM = self._get_logM_sigM(cosmo, M_use, a)
        return _squeeze(self._get_bsigma(cosmo, sigM, a), M)


class HaloBiasTinker10(HaloBias):
    name = "Tinker10"

    def __init__(self, *, mass_def="200m", mass_def_strict=True):
        super().__init__(mass_def=mass_def, mass_def_strict=mass_def_strict)

    def _check_mass_def_strict(self, mass_def):
        return mass_def.Delta == "fof"

    def _setup(self):
        self.B, self.b, self.c = 0.183, 1.5, 2.4
        self.dc = get_delta_c(None, None, kind="EdS")

    def _get_bsigma(self, cosmo, sigM, a):
        nu = self.dc / sigM
        ld = jnp.log10(self.mass_def._get_Delta_m(cosmo, a))
        xp = jnp.exp(-((4.0 / ld) ** 4.0))
        A = 1.0 + 0.24 * ld * xp
        C = 0.019 + 0.107 * ld + 0.19 * xp
        aa = 0.44 * ld - 0.88
        nupa = nu**aa
        return 1 - A * nupa / (nupa + self.dc**aa) + (self.B * nu**self.b + C * nu**self.c)


class HaloBiasSheth99(HaloBias):
    name = "Sheth99"

    def __init__(self, *, mass_def="fof", mass_def_strict=True, use_delta_c_fit=False):
        self.use_delta_c_fit = use_delta_c_fit
        super().__init__(mass_def=mass_def, mass_def_strict=mass_def_strict)

    def _check_mass_def_strict(self, mass_def):
        return mass_def.Delta != "fof"

    def _setup(self):
        self.p, self.a = 0.3, 0.707

    def _get_bsigma(self, cosmo, sigM, a):
        kind = "NakamuraSuto97" if self.use_delta_c_fit else "EdS"
        delta_c = get_delta_c(cosmo, a, kind=kind)
        anu2 = self.a * (delta_c / sigM) ** 2
        return 1 + (anu2 - 1.0 + 2.0 * self.p / (1.0 + anu2**self.p)) / delta_c


class HaloBiasSheth01(HaloBias):
    name = "Sheth01"

    def __init__(self, *, mass_def="fof", mass_def_strict=True):
        super().__init__(mass_def=mass_def, mass_def_strict=mass_def_strict)

    def _check_mass_def_strict(self, mass_def):
        return mass_def.Delta != "fof"

    def _setup(self):
        self.a, self.sqrta, self.b, self.c = 0.707, 0.84083292038, 0.5, 0.6
        self.dc = get_delta_c(None, None, kind="EdS")

    def _get_bsigma(self, cosmo, sigM, a):
        anu2 = self.a * (self.dc / sigM) ** 2
        anu2c = anu2**self.c
        t1 = self.b * (1.0 - self.c) * (1.0 - 0.5 * self.c)
        return 1 + (self.sqrta * anu2 * (1 + self.b / anu2c) - anu2c / (anu2c + t1)) / (
            self.sqrta * self.dc)


class HaloBiasBhattacharya11(HaloBias):
    name = "Bhattacharya11"

    def __init__(self, *, mass_def="fof", mass_def_strict=True):
        super().__init__(mass_def=mass_def, mass_def_strict=mass_def_strict)

    def _check_mass_def_strict(self, mass_def):
        return mass_def.Delta != "fof"

    def _setup(self):
        self.a, self.az, self.p, self.q = 0.788, 0.01, 0.807, 1.795
        self.dc = get_delta_c(None, None, kind="EdS")

    def _get_bsigma(self, cosmo, sigM, a):
        nu = self.dc / sigM
        anu2 = self.a * a**self.az * nu**2
        return 1 + (anu2 - self.q + 2 * self.p / (1 + anu2**self.p)) / self.dc


# == concentrations ============================================================
class Concentration(HMIngredients):
    """Concentration-mass relation c(M, a)."""

    _mass_def_strict_always = True

    def __init__(self, *, mass_def):
        super().__init__(mass_def=mass_def, mass_def_strict=True)

    def _concentration(self, cosmo, M, a):
        raise NotImplementedError

    def __call__(self, cosmo, M, a):
        M_use = jnp.atleast_1d(jnp.asarray(M, dtype=float))
        return _squeeze(self._concentration(cosmo, M_use, a), M)


class ConcentrationDuffy08(Concentration):
    name = "Duffy08"

    def __init__(self, *, mass_def="200c"):
        super().__init__(mass_def=mass_def)

    def _check_mass_def_strict(self, mass_def):
        return mass_def.name not in ("vir", "200m", "200c")

    def _setup(self):
        vals = {"vir": (7.85, -0.081, -0.71), "200m": (10.14, -0.081, -1.01),
                "200c": (5.71, -0.084, -0.47)}
        self.A, self.B, self.C = vals[self.mass_def.name]

    def _concentration(self, cosmo, M, a):
        return self.A * (M * (cosmo["h"] * 5e-13)) ** self.B * a ** (-self.C)


class ConcentrationBhattacharya13(Concentration):
    name = "Bhattacharya13"

    def __init__(self, *, mass_def="200c"):
        super().__init__(mass_def=mass_def)

    def _check_mass_def_strict(self, mass_def):
        return mass_def.name not in ("vir", "200m", "200c")

    def _setup(self):
        vals = {"vir": (7.7, 0.9, -0.29), "200m": (9.0, 1.15, -0.29),
                "200c": (5.9, 0.54, -0.35)}
        self.A, self.B, self.C = vals[self.mass_def.name]

    def _concentration(self, cosmo, M, a):
        gz = cosmo.growth_factor(a)
        nu = get_delta_c(cosmo, a, kind="NakamuraSuto97") / sigmaM(cosmo, M, a)
        return self.A * gz**self.B * nu**self.C


class ConcentrationKlypin11(Concentration):
    name = "Klypin11"

    def __init__(self, *, mass_def="vir"):
        super().__init__(mass_def=mass_def)

    def _check_mass_def_strict(self, mass_def):
        return mass_def.name != "vir"

    def _concentration(self, cosmo, M, a):
        return 9.6 * (M * (cosmo["h"] * 1e-12)) ** (-0.075)


class ConcentrationConstant(Concentration):
    name = "Constant"

    def __init__(self, c=1, *, mass_def="200c"):
        self.c = c
        super().__init__(mass_def=mass_def)

    def _check_mass_def_strict(self, mass_def):
        return False

    def _concentration(self, cosmo, M, a):
        return jnp.full_like(M, 1.0) * self.c


__all__ = [
    "Concentration",
    "ConcentrationBhattacharya13",
    "ConcentrationConstant",
    "ConcentrationDuffy08",
    "ConcentrationKlypin11",
    "HMIngredients",
    "HaloBias",
    "HaloBiasBhattacharya11",
    "HaloBiasSheth01",
    "HaloBiasSheth99",
    "HaloBiasTinker10",
    "MassFunc",
    "MassFuncAngulo12",
    "MassFuncDespali16",
    "MassFuncJenkins01",
    "MassFuncPress74",
    "MassFuncSheth99",
    "MassFuncTinker08",
    "MassFuncTinker10",
    "omega_x",
]
