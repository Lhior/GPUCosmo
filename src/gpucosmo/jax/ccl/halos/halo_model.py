#!/usr/bin/env python3
# This file is part of GPUCosmo. Copyright (C) 2025-2026 Leonel Medina-Varela.
# SPDX-License-Identifier: GPL-3.0-or-later
# Derived in part from the following, translated to JAX and modified for GPUCosmo:
# - CCL (https://github.com/LSSTDESC/CCL), Copyright (c) 2016-2018, the LSSTDESC CCL
#   contributors.
#   Licence: LICENSES/CCL.txt.
# See NOTICE.md.
"""``HMCalculator``, ``Profile2pt`` and halo-model power spectra (pyccl syntax).

Mass integrals use pyccl's defaults: ``nM = 128`` log10 M nodes in [8, 16]
and ``integration_method_M='simpson'`` -- ``scipy.integrate.simpson`` is
linear in the integrand, so its exact weights (including SciPy's
even-sample correction) are precomputed once. ``'spline'`` uses the GSL Akima
integral like ``pyccl.pyutils._spline_integrate``. The low-mass remainder
(``mf0``, ``mbf0``) is handled exactly as in pyccl so that the matter
consistency relations hold.

``halomod_power_spectrum`` evaluates the scale factors with ``jax.lax.map``
(``a_batch_size`` controls how many are vectorised together); the linear power
spectrum defaults to ``cosmo``'s linear ``Pk2D`` without growth extrapolation
(``extrap_pk=False``), as in pyccl.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from gpucosmo.jax.ccl.halos._sigma import get_pk_spline_a, get_pk_spline_lk, rho_m0, sigma_context
from gpucosmo.jax.ccl.halos.massdef import MassDef
from gpucosmo.jax.ccl.halos.profiles import HaloProfile, HaloProfileHOD
from gpucosmo.jax.ccl.pk2d import Pk2D
from gpucosmo.jax.gsl_splines import akima_spline, spline_integral

jax.config.update("jax_enable_x64", True)


def _simpson_weights(x: np.ndarray) -> np.ndarray:
    from scipy.integrate import simpson

    return np.asarray(simpson(np.eye(x.size), x=x, axis=-1), dtype=float)


class HMCalculator:
    """Halo-model mass integrals (``pyccl.halos.HMCalculator``)."""

    def __init__(self, *, mass_function, halo_bias, mass_def=None, log10M_min=8.,
                 log10M_max=16., nM=128, integration_method_M="simpson"):
        out = MassDef.from_specs(mass_def, mass_function=mass_function, halo_bias=halo_bias)
        if len(out) != 3:
            raise ValueError("A valid mass function and halo bias is needed")
        self.mass_def, self.mass_function, self.halo_bias = out
        self.precision = {"log10M_min": log10M_min, "log10M_max": log10M_max, "nM": nM,
                          "integration_method_M": integration_method_M}
        self._lmass = np.linspace(log10M_min, log10M_max, nM)
        self._mass = 10.0**self._lmass
        self._m0 = self._mass[0]
        if integration_method_M == "simpson":
            self._w = jnp.asarray(_simpson_weights(self._lmass))
        elif integration_method_M == "spline":
            self._w = None
        else:
            raise ValueError("Invalid integration method.")

    # -- integration ------------------------------------------------------
    def _integrator(self, fM, log10M=None):
        if self._w is not None:
            return jnp.tensordot(fM, self._w, axes=([-1], [0]))
        x = jnp.asarray(self._lmass)
        flat = jnp.reshape(fM, (-1, fM.shape[-1]))
        res = jax.vmap(lambda y: spline_integral(akima_spline(x, y), x[0], x[-1]))(flat)
        return jnp.reshape(res, fM.shape[:-1])

    def _check_mass_def(self, *others):
        if {x.mass_def for x in others} != {self.mass_def}:
            raise ValueError("Inconsistent mass definitions.")

    def _mf(self, cosmo, a):
        mf = self.mass_function(cosmo, self._mass, a)
        mf0 = (rho_m0(cosmo) - self._integrator(mf * self._mass)) / self._m0
        return mf, mf0

    def _get_ingredients(self, cosmo, a, *, get_bf):
        mf, mf0 = self._mf(cosmo, a)
        if not get_bf:
            return mf, mf0, None, None
        bf = self.halo_bias(cosmo, self._mass, a)
        mbf0 = (rho_m0(cosmo) - self._integrator(mf * bf * self._mass)) / self._m0
        return mf, mf0, bf, mbf0

    def _integrate_over_mf(self, ingr, array_2):
        mf, mf0, _, _ = ingr
        return self._integrator(mf * array_2) + mf0 * array_2[..., 0]

    def _integrate_over_mbf(self, ingr, array_2):
        mf, _, bf, mbf0 = ingr
        return self._integrator(mf * bf * array_2) + mbf0 * array_2[..., 0]

    def integrate_over_massfunc(self, func, cosmo, a):
        fM = func(jnp.asarray(self._mass))
        return self._integrate_over_mf(self._get_ingredients(cosmo, a, get_bf=False), fM)

    def get_mass_function(self, cosmo, a):
        return self._mf(cosmo, a)[0]

    # -- halo-model integrals (pyccl signatures) --------------------------
    def I_0_1(self, cosmo, k, a, prof):
        self._check_mass_def(prof)
        with sigma_context(cosmo):
            uk = _fourier_2d(prof, cosmo, k, self._mass, a).T
            return _squeeze_k(self._integrate_over_mf(
                self._get_ingredients(cosmo, a, get_bf=False), uk), k)

    def I_1_1(self, cosmo, k, a, prof):
        self._check_mass_def(prof)
        with sigma_context(cosmo):
            uk = _fourier_2d(prof, cosmo, k, self._mass, a).T
            return _squeeze_k(self._integrate_over_mbf(
                self._get_ingredients(cosmo, a, get_bf=True), uk), k)

    def I_0_2(self, cosmo, k, a, prof, *, prof2=None, prof_2pt):
        prof2 = prof if prof2 is None else prof2
        self._check_mass_def(prof, prof2)
        with sigma_context(cosmo):
            uk = _as_2d(prof_2pt.fourier_2pt(cosmo, k, self._mass, a, prof, prof2=prof2), k).T
            return _squeeze_k(self._integrate_over_mf(
                self._get_ingredients(cosmo, a, get_bf=False), uk), k)

    def I_1_2(self, cosmo, k, a, prof, *, prof2=None, prof_2pt, diag=True):
        if not diag:
            raise NotImplementedError("only diag=True is implemented")
        prof2 = prof if prof2 is None else prof2
        self._check_mass_def(prof, prof2)
        with sigma_context(cosmo):
            uk = _as_2d(prof_2pt.fourier_2pt(cosmo, k, self._mass, a, prof, prof2=prof2), k).T
            return _squeeze_k(self._integrate_over_mbf(
                self._get_ingredients(cosmo, a, get_bf=True), uk), k)


def _as_2d(uk, k):
    return uk[:, None] if np.ndim(k) == 0 else uk


def _fourier_2d(prof, cosmo, k, M, a):
    return _as_2d(prof.fourier(cosmo, k, M, a), k)


def _squeeze_k(x, k):
    return x[0] if np.ndim(k) == 0 else x


class Profile2pt:
    """Fourier two-point halo profile ``(1 + r_corr) u_1 u_2``."""

    def __init__(self, *, r_corr=0.0):
        self.r_corr = r_corr

    def update_parameters(self, *, r_corr=None):
        if r_corr is not None:
            self.r_corr = r_corr

    def fourier_2pt(self, cosmo, k, M, a, prof, *, prof2=None, diag=True, _uk=None):
        prof2 = prof if prof2 is None else prof2
        if not (isinstance(prof, HaloProfile) and isinstance(prof2, HaloProfile)):
            raise TypeError("prof and prof2 must be HaloProfile")
        uk1, uk2 = _uk if _uk is not None else (None, None)
        if uk1 is None:
            uk1 = prof.fourier(cosmo, k, M, a)
        if uk2 is None:
            uk2 = uk1 if prof2 is prof else prof2.fourier(cosmo, k, M, a)
        if not diag:
            raise NotImplementedError("only diag=True is implemented")
        return uk1 * uk2 * (1 + self.r_corr)


class Profile2ptHOD(Profile2pt):
    """HOD auto-correlation with the central/satellite variance."""

    def fourier_2pt(self, cosmo, k, M, a, prof, *, prof2=None, diag=True, _uk=None):
        prof2 = prof if prof2 is None else prof2
        if not (isinstance(prof, HaloProfileHOD) and isinstance(prof2, HaloProfileHOD)):
            raise TypeError("prof and prof2 must be HaloProfileHOD")
        if not diag:
            raise NotImplementedError("only diag=True is implemented")
        if prof2 is prof:
            return prof._fourier_variance(cosmo, k, M, a)
        return super().fourier_2pt(cosmo, k, M, a, prof, prof2=prof2, _uk=_uk)


def _map_a(fn, a_use, batch_size):
    if a_use.shape[0] == 1:
        return fn(a_use[0])[None]
    return jax.lax.map(fn, a_use, batch_size=batch_size)


def halomod_power_spectrum(cosmo, hmc, k, a, prof, *, prof2=None, prof_2pt=None,
                           p_of_k_a=None, get_1h=True, get_2h=True, smooth_transition=None,
                           suppress_1h=None, extrap_pk=False, a_batch_size=None):
    """Halo-model P(k, a) = (P_1h + P_2h) / (norm1 norm2), shape (na, nk)."""
    if smooth_transition is not None and not (get_1h and get_2h):
        raise ValueError("Transition region can only be modified "
                         "when both 1-halo and 2-halo terms are queried.")
    if suppress_1h is not None and not get_1h:
        raise ValueError("Can't suppress the 1-halo term when get_1h is False.")
    prof2 = prof if prof2 is None else prof2
    prof_2pt = Profile2pt() if prof_2pt is None else prof_2pt
    hmc._check_mass_def(prof, prof2)
    same = prof2 is prof
    pk2d = cosmo.parse_pk2d(p_of_k_a, is_linear=True)
    extrap = cosmo if extrap_pk else None
    k_use = np.atleast_1d(np.asarray(k, dtype=float)) if not isinstance(k, jax.core.Tracer) \
        else jnp.atleast_1d(k)
    a_use = jnp.atleast_1d(jnp.asarray(a, dtype=float))
    M = hmc._mass

    with sigma_context(cosmo):

        def one_a(aa):
            ingr = hmc._get_ingredients(cosmo, aa, get_bf=get_2h)
            norm1 = prof.get_normalization(cosmo, aa, hmc=hmc)
            norm2 = norm1 if same else prof2.get_normalization(cosmo, aa, hmc=hmc)
            uk1 = _fourier_2d(prof, cosmo, k_use, M, aa)
            uk2 = uk1 if same else _fourier_2d(prof2, cosmo, k_use, M, aa)
            out = 0.0
            pk_2h = 0.0
            if get_2h:
                i11_1 = hmc._integrate_over_mbf(ingr, uk1.T)
                i11_2 = i11_1 if same else hmc._integrate_over_mbf(ingr, uk2.T)
                pk_2h = pk2d(k_use, aa, cosmo=extrap) * i11_1 * i11_2
            pk_1h = 0.0
            if get_1h:
                uk12 = prof_2pt.fourier_2pt(cosmo, k_use, M, aa, prof, prof2=prof2,
                                            _uk=(uk1, uk2))
                pk_1h = hmc._integrate_over_mf(ingr, uk12.T)
                if suppress_1h is not None:
                    ks = suppress_1h(aa)
                    pk_1h = pk_1h * (k_use / ks) ** 4 / (1 + (k_use / ks) ** 4)
            if smooth_transition is None:
                out = (pk_1h + pk_2h) / (norm1 * norm2)
            else:
                alpha = smooth_transition(aa)
                out = (pk_1h**alpha + pk_2h**alpha) ** (1 / alpha) / (norm1 * norm2)
            return jnp.broadcast_to(out, (k_use.shape[0],))

        out = _map_a(one_a, a_use, a_batch_size)
    if np.ndim(a) == 0:
        out = jnp.squeeze(out, axis=0)
    if np.ndim(k) == 0:
        out = jnp.squeeze(out, axis=-1)
    return out


def halomod_Pk2D(cosmo, hmc, prof, *, prof2=None, prof_2pt=None, p_of_k_a=None,
                 get_1h=True, get_2h=True, lk_arr=None, a_arr=None, extrap_order_lok=1,
                 extrap_order_hik=2, smooth_transition=None, suppress_1h=None,
                 extrap_pk=False, a_batch_size=None):
    """Halo-model :class:`~gpucosmo.jax.ccl.Pk2D` (linear-P spline, like pyccl)."""
    lk_arr = get_pk_spline_lk() if lk_arr is None else np.asarray(lk_arr, dtype=float)
    a_arr = get_pk_spline_a() if a_arr is None else a_arr
    pk_arr = halomod_power_spectrum(
        cosmo, hmc, np.exp(lk_arr), a_arr, prof, prof2=prof2, prof_2pt=prof_2pt,
        p_of_k_a=p_of_k_a, get_1h=get_1h, get_2h=get_2h, smooth_transition=smooth_transition,
        suppress_1h=suppress_1h, extrap_pk=extrap_pk, a_batch_size=a_batch_size)
    return Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=pk_arr, is_logp=False,
                extrap_order_lok=extrap_order_lok, extrap_order_hik=extrap_order_hik)


def _Ix1(name, cosmo, hmc, k, a, prof, a_batch_size=None):
    a_use = jnp.atleast_1d(jnp.asarray(a, dtype=float))
    k_use = np.atleast_1d(np.asarray(k, dtype=float))
    integ = getattr(hmc, name)
    with sigma_context(cosmo):
        out = _map_a(lambda aa: integ(cosmo, k_use, aa, prof)
                     / prof.get_normalization(cosmo, aa, hmc=hmc), a_use, a_batch_size)
    if np.ndim(a) == 0:
        out = jnp.squeeze(out, axis=0)
    if np.ndim(k) == 0:
        out = jnp.squeeze(out, axis=-1)
    return out


def halomod_mean_profile_1pt(cosmo, hmc, k, a, prof):
    return _Ix1("I_0_1", cosmo, hmc, k, a, prof)


def halomod_bias_1pt(cosmo, hmc, k, a, prof):
    return _Ix1("I_1_1", cosmo, hmc, k, a, prof)


__all__ = [
    "HMCalculator",
    "Profile2pt",
    "Profile2ptHOD",
    "halomod_Pk2D",
    "halomod_bias_1pt",
    "halomod_mean_profile_1pt",
    "halomod_power_spectrum",
]
