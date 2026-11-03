#!/usr/bin/env python3
# This file is part of GPUCosmo. Copyright (C) 2025-2026 Leonel Medina-Varela.
# SPDX-License-Identifier: GPL-3.0-or-later
# Derived in part from the following, translated to JAX and modified for GPUCosmo:
# - CCL (https://github.com/LSSTDESC/CCL), Copyright (c) 2016-2018, the LSSTDESC CCL
#   contributors.
#   Licence: LICENSES/CCL.txt.
# See NOTICE.md.
"""Halo profiles (``pyccl.halos.profiles``): NFW, HOD and SatelliteShearHOD.

Array conventions follow pyccl: ``fourier(cosmo, k, M, a)`` returns
``(nM, nk)`` with the scalar axes squeezed. Profile parameters (HOD, IA
amplitude ``a1h``) may be JAX tracers; the SatelliteShearHOD slope ``b`` and
``lmax`` are static (they set the angular coefficients).

FFTLog-based transforms reproduce ``HaloProfile._fftlog_wrap``: the real
profile is sampled on ``np.geomspace(k_min, k_max, n)`` with pyccl's padding
and ``n_per_decade`` settings, transformed with the CCL FFTLog kernel and
resampled onto ``k`` with GSL Akima + linear extrapolation. The ``k`` array
must therefore be concrete (NumPy) for FFTLog profiles; the analytic profiles
accept traced ``k``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import erf, sici

from gpucosmo.jax.ccl.halos._fftlog import fht_apply, fht_plan, resample_akima_linlin
from gpucosmo.jax.ccl.halos._sigma import rho_m0
from gpucosmo.jax.ccl.halos.massdef import MassDef

jax.config.update("jax_enable_x64", True)


class FFTLogParams(dict):
    """``pyccl.FFTLogParams`` defaults (dict-like)."""

    DEFAULTS = dict(
        padding_lo_fftlog=0.1, padding_hi_fftlog=10.0, n_per_decade=100,
        extrapol="linx_liny", padding_lo_extra=0.1, padding_hi_extra=10.0,
        large_padding_2D=False, plaw_fourier=-1.5, plaw_projected=-1.0,
    )

    def __init__(self, **kw):
        super().__init__(self.DEFAULTS)
        self.update_parameters(**kw)

    def update_parameters(self, **kw):
        for key, val in kw.items():
            if key not in self.DEFAULTS:
                raise KeyError(f"Unknown FFTLog parameter {key}")
            self[key] = val


def _squeeze_kM(prof, k, M):
    if np.ndim(k) == 0:
        prof = jnp.squeeze(prof, axis=-1)
    if np.ndim(M) == 0:
        prof = jnp.squeeze(prof, axis=0)
    return prof


def _as1d(x):
    """1D view that keeps NumPy inputs concrete (static FFTLog grids)."""
    if isinstance(x, (jax.Array, jax.core.Tracer)):
        return jnp.atleast_1d(x.astype(float))
    return np.atleast_1d(np.asarray(x, dtype=float))


class HaloProfile:
    """Base halo profile with FFTLog Fourier transforms."""

    def __init__(self, *, mass_def, concentration=None, is_number_counts=False):
        if not (hasattr(self, "_real") or hasattr(self, "_fourier")):
            raise TypeError(f"Can't instantiate {type(self).__name__} with no "
                            "_real or _fourier implementation.")
        self.precision_fftlog = FFTLogParams()
        self.is_number_counts = is_number_counts
        self.mass_def, *out = MassDef.from_specs(mass_def, concentration=concentration)
        if out:
            self.concentration = out[0]

    def get_normalization(self, cosmo=None, a=None, *, hmc=None):
        return 1.0

    def update_precision_fftlog(self, **kwargs):
        self.precision_fftlog.update_parameters(**kwargs)

    def _get_plaw_fourier(self, cosmo, a):
        return self.precision_fftlog["plaw_fourier"]

    def real(self, cosmo, r, M, a):
        if getattr(self, "_real", None):
            return self._real(cosmo, r, M, a)
        return self._fftlog_wrap(cosmo, r, M, a, fourier_out=False)

    def fourier(self, cosmo, k, M, a):
        if getattr(self, "_fourier", None):
            return self._fourier(cosmo, k, M, a)
        return self._fftlog_wrap(cosmo, k, M, a, fourier_out=True)

    # -- FFTLog ----------------------------------------------------------
    def _fftlog_grid(self, k, large_padding=True):
        k_use = np.atleast_1d(np.asarray(k, dtype=float))
        p = self.precision_fftlog
        if large_padding:
            k_min = p["padding_lo_fftlog"] * np.amin(k_use)
            k_max = p["padding_hi_fftlog"] * np.amax(k_use)
        else:
            k_min = p["padding_lo_extra"] * np.amin(k_use)
            k_max = p["padding_hi_extra"] * np.amax(k_use)
        n_k = int(np.log10(k_max / k_min)) * p["n_per_decade"]
        return k_use, k_min, k_max, n_k

    def _fftlog_multi(self, cosmo, k, M, a, *, ells, weights, fourier_out=True,
                      large_padding=True):
        """sum_l weights[l] * FFTLog_l[profile] resampled onto ``k``: (nM, nk)."""
        try:
            k_use, k_min, k_max, n_k = self._fftlog_grid(k, large_padding)
        except (TypeError, jax.errors.TracerArrayConversionError) as err:
            raise TypeError("FFTLog profiles need a concrete (NumPy) k array") from err
        if self.precision_fftlog["extrapol"] != "linx_liny":
            raise NotImplementedError("only extrapol='linx_liny' is implemented")
        r_arr = np.geomspace(k_min, k_max, n_k)
        M_use = _as1d(M)
        p_func = self._real if fourier_out else self._fourier
        p_in = p_func(cosmo, r_arr, M_use, a)
        p_in = jnp.reshape(p_in, (M_use.shape[0], n_k))
        q = 1.5 + self._get_plaw_fourier(cosmo, a)
        plans = [fht_plan(k_min, k_max, n_k, 3, ell + 0.5, q) for ell in ells]
        outs = fht_apply(plans, p_in)
        lk_use = np.log(k_use)
        total = 0.0
        for plan, out, w in zip(plans, outs, weights):
            total = total + w * resample_akima_linlin(np.log(plan.k_out), out, lk_use)
        if fourier_out:
            total = total * (2 * np.pi) ** 3
        return total

    def _fftlog_wrap(self, cosmo, k, M, a, fourier_out=False, large_padding=True, ell=0):
        out = self._fftlog_multi(cosmo, k, M, a, ells=(ell,), weights=(1.0,),
                                 fourier_out=fourier_out, large_padding=large_padding)
        return _squeeze_kM(out, k, M)


class HaloProfileMatter(HaloProfile):
    """Profiles normalised by the comoving matter density."""

    def get_normalization(self, cosmo, a=None, *, hmc=None):
        return rho_m0(cosmo)


class HaloProfileNFW(HaloProfileMatter):
    """Navarro-Frenk-White profile (truncated at r_Delta by default)."""

    def __init__(self, *, mass_def, concentration, fourier_analytic=True,
                 projected_analytic=False, cumul2d_analytic=False, truncated=True):
        self.truncated = truncated
        self.fourier_analytic = fourier_analytic
        if projected_analytic or cumul2d_analytic:
            raise NotImplementedError("projected / cumul2d NFW profiles not ported")
        if fourier_analytic:
            self._fourier = self._fourier_analytic
        super().__init__(mass_def=mass_def, concentration=concentration)
        self.update_precision_fftlog(padding_hi_fftlog=1e2, padding_lo_fftlog=1e-2,
                                     n_per_decade=1000, plaw_fourier=-2.0)

    def _norm(self, M, Rs, c):
        return M / (4 * np.pi * Rs**3 * (jnp.log(1 + c) - c / (1 + c)))

    def _radii(self, cosmo, M_use, a):
        R_M = self.mass_def.get_radius(cosmo, M_use, a) / a
        c_M = self.concentration(cosmo, M_use, a)
        return R_M, c_M, R_M / c_M

    def _real(self, cosmo, r, M, a):
        r_use, M_use = _as1d(r), _as1d(M)
        R_M, c_M, R_s = self._radii(cosmo, M_use, a)
        x = r_use[None, :] / R_s[:, None]
        prof = 1.0 / (x * (1 + x) ** 2)
        if self.truncated:
            prof = jnp.where(r_use[None, :] > R_M[:, None], 0.0, prof)
        prof = prof * self._norm(M_use, R_s, c_M)[:, None]
        return _squeeze_kM(prof, r, M)

    def _fourier_analytic(self, cosmo, k, M, a):
        M_use, k_use = _as1d(M), _as1d(k)
        R_M, c_M, R_s = self._radii(cosmo, M_use, a)
        x = k_use[None, :] * R_s[:, None]
        Si2, Ci2 = sici(x)
        P1 = M_use / (jnp.log(1 + c_M) - c_M / (1 + c_M))
        if self.truncated:
            Si1, Ci1 = sici((1 + c_M[:, None]) * x)
            P2 = jnp.sin(x) * (Si1 - Si2) + jnp.cos(x) * (Ci1 - Ci2)
            P3 = jnp.sin(c_M[:, None] * x) / ((1 + c_M[:, None]) * x)
            prof = P1[:, None] * (P2 - P3)
        else:
            P2 = jnp.sin(x) * (0.5 * np.pi - Si2) - jnp.cos(x) * Ci2
            prof = P1[:, None] * P2
        return _squeeze_kM(prof, k, M)


_HOD_PARAMS = ("log10Mmin_0", "log10Mmin_p", "siglnM_0", "siglnM_p", "log10M0_0",
               "log10M0_p", "log10M1_0", "log10M1_p", "alpha_0", "alpha_p", "fc_0",
               "fc_p", "bg_0", "bg_p", "bmax_0", "bmax_p", "a_pivot", "ns_independent")


class HaloProfileHOD(HaloProfile):
    """Zheng et al. (2005) HOD with NFW satellites, as in ``pyccl``."""

    def __init__(self, *, mass_def, concentration, log10Mmin_0=12., log10Mmin_p=0.,
                 siglnM_0=0.4, siglnM_p=0., log10M0_0=7., log10M0_p=0., log10M1_0=13.3,
                 log10M1_p=0., alpha_0=1., alpha_p=0., fc_0=1., fc_p=0., bg_0=1., bg_p=0.,
                 bmax_0=1., bmax_p=0., a_pivot=1., ns_independent=False,
                 is_number_counts=True):
        self.log10Mmin_0, self.log10Mmin_p = log10Mmin_0, log10Mmin_p
        self.log10M0_0, self.log10M0_p = log10M0_0, log10M0_p
        self.log10M1_0, self.log10M1_p = log10M1_0, log10M1_p
        self.siglnM_0, self.siglnM_p = siglnM_0, siglnM_p
        self.alpha_0, self.alpha_p = alpha_0, alpha_p
        self.fc_0, self.fc_p = fc_0, fc_p
        self.bg_0, self.bg_p = bg_0, bg_p
        self.bmax_0, self.bmax_p = bmax_0, bmax_p
        self.a_pivot = a_pivot
        self.ns_independent = ns_independent
        super().__init__(mass_def=mass_def, concentration=concentration,
                         is_number_counts=is_number_counts)

    def update_parameters(self, **kwargs):
        for key, val in kwargs.items():
            if key not in _HOD_PARAMS:
                raise TypeError(f"Unknown HOD parameter {key}")
            if val is not None:
                setattr(self, key, val)

    def _sat_geometry(self, cosmo, M_use, a):
        bg = self.bg_0 + self.bg_p * (a - self.a_pivot)
        bmax = self.bmax_0 + self.bmax_p * (a - self.a_pivot)
        R_M = self.mass_def.get_radius(cosmo, M_use, a) / a
        c_M = self.concentration(cosmo, M_use, a)
        R_s = R_M / c_M
        return bg, bmax, R_M, R_s, c_M * bmax / bg

    def _usat_real(self, cosmo, r, M, a):
        r_use, M_use = _as1d(r), _as1d(M)
        bg, bmax, R_M, R_s, c_M = self._sat_geometry(cosmo, M_use, a)
        x = r_use[None, :] / (R_s[:, None] * bg)
        prof = 1.0 / (x * (1 + x) ** 2)
        prof = jnp.where(r_use[None, :] > R_M[:, None] * bmax, 0.0, prof)
        norm = 1.0 / (4 * np.pi * (bg * R_s) ** 3 * (jnp.log(1 + c_M) - c_M / (1 + c_M)))
        return _squeeze_kM(prof * norm[:, None], r, M)

    def _usat_fourier(self, cosmo, k, M, a):
        M_use, k_use = _as1d(M), _as1d(k)
        bg, bmax, R_M, R_s, c_M = self._sat_geometry(cosmo, M_use, a)
        x = k_use[None, :] * R_s[:, None] * bg
        Si1, Ci1 = sici((1 + c_M[:, None]) * x)
        Si2, Ci2 = sici(x)
        P1 = 1.0 / (jnp.log(1 + c_M) - c_M / (1 + c_M))
        P2 = jnp.sin(x) * (Si1 - Si2) + jnp.cos(x) * (Ci1 - Ci2)
        P3 = jnp.sin(c_M[:, None] * x) / ((1 + c_M[:, None]) * x)
        return _squeeze_kM(P1[:, None] * (P2 - P3), k, M)

    def _combine(self, Nc, Ns, fc, u):
        if self.ns_independent:
            return Nc[:, None] * fc + Ns[:, None] * u
        return Nc[:, None] * (fc + Ns[:, None] * u)

    def _real(self, cosmo, r, M, a):
        r_use, M_use = _as1d(r), _as1d(M)
        ur = jnp.reshape(self._usat_real(cosmo, r_use, M_use, a), (M_use.shape[0], -1))
        prof = self._combine(self._Nc(M_use, a), self._Ns(M_use, a), self._fc(a), ur)
        return _squeeze_kM(prof, r, M)

    def _fourier(self, cosmo, k, M, a):
        k_use, M_use = _as1d(k), _as1d(M)
        uk = jnp.reshape(self._usat_fourier(cosmo, k_use, M_use, a), (M_use.shape[0], -1))
        prof = self._combine(self._Nc(M_use, a), self._Ns(M_use, a), self._fc(a), uk)
        return _squeeze_kM(prof, k, M)

    def _fourier_variance(self, cosmo, k, M, a):
        k_use, M_use = _as1d(k), _as1d(M)
        Nc, Ns, fc = self._Nc(M_use, a), self._Ns(M_use, a), self._fc(a)
        uk = jnp.reshape(self._usat_fourier(cosmo, k_use, M_use, a), (M_use.shape[0], -1))
        prof = Ns[:, None] * uk
        if self.ns_independent:
            prof = 2 * Nc[:, None] * fc * prof + prof**2
        else:
            prof = Nc[:, None] * (2 * fc * prof + prof**2)
        return _squeeze_kM(prof, k, M)

    def get_normalization(self, cosmo, a, *, hmc):
        def integ(M):
            Nc, Ns, fc = self._Nc(M, a), self._Ns(M, a), self._fc(a)
            return Nc * fc + Ns if self.ns_independent else Nc * (fc + Ns)

        return hmc.integrate_over_massfunc(integ, cosmo, a)

    def _fc(self, a):
        return self.fc_0 + self.fc_p * (a - self.a_pivot)

    def _Nc(self, M, a):
        Mmin = 10.0 ** (self.log10Mmin_0 + self.log10Mmin_p * (a - self.a_pivot))
        siglnM = self.siglnM_0 + self.siglnM_p * (a - self.a_pivot)
        return 0.5 * (1 + erf(jnp.log(M / Mmin) / siglnM))

    def _Ns(self, M, a):
        M0 = 10.0 ** (self.log10M0_0 + self.log10M0_p * (a - self.a_pivot))
        M1 = 10.0 ** (self.log10M1_0 + self.log10M1_p * (a - self.a_pivot))
        alpha = self.alpha_0 + self.alpha_p * (a - self.a_pivot)
        # heaviside(M - M0, 1) * (|M - M0| / M1)**alpha, gradient-safe below M0.
        above = M > M0
        base = jnp.where(above, (M - M0) / M1, 1.0)
        return jnp.where(above, base**alpha, 0.0)


# Hard-coded angular coefficients of pyccl (b = 0 and b = -2).
_FL_B0 = np.array([2.77582637, -0.19276603, 0.04743899, -0.01779024, 0.00832446, -0.00447308])
_FL_BM2 = np.array([4.71238898, -2.61799389, 2.06167032, -1.76714666, 1.57488973, -1.43581368])


def _I_integral(a, b):
    from scipy.special import gamma

    return (1 + (-1) ** b) * gamma(a / 2 + 1) * gamma((b + 1) / 2) / (2 * gamma(a / 2 + b / 2 + 3 / 2))


def _fl(ell, thk=np.pi / 2, b=-2):
    """Eq. (C8) of Fortuna et al. (2021) (``SatelliteShearHOD._fl``)."""
    from scipy.special import binom

    gj = np.array([0, 0, np.pi / 2, 0, np.pi / 2, 0, 15 * np.pi / 32, 0, 7 * np.pi / 16, 0,
                   105 * np.pi / 256, 0, 99 * np.pi / 256])
    var1_add = 1 if b == 0 else b
    l_sum = 0.0
    for m in range(0, ell + 1):
        m_sum = 0.0
        for j in range(0, m + 1):
            m_sum += (binom(m, j) * gj[j] * _I_integral(j + var1_add, m - j)
                      * np.sin(thk) ** j * np.cos(thk) ** (m - j))
        l_sum += binom(ell, m) * binom((ell + m - 1) / 2, ell) * m_sum
    return 2**ell * l_sum


class SatelliteShearHOD(HaloProfileHOD):
    """Satellite intrinsic-shear HOD profile (Fortuna et al. 2021).

    Fourier profile ``sum_{l even >= 2} i^l (2l+1) f_l / (4 pi) * FFTLog_l``
    of ``gamma_I(r) u_sat(r|M)`` with pyccl's FFTLog precision
    (``padding_lo=1e-2, padding_hi=1e3, n_per_decade=350, plaw=-3.7``).
    Only ``integration_method='FFTLog'`` (the pyccl default) is ported.
    """

    def __init__(self, *, mass_def, concentration, a1h=0.001, b=-2, lmax=6,
                 log10Mmin_0=12., log10Mmin_p=0., siglnM_0=0.4, siglnM_p=0.,
                 log10M0_0=7., log10M0_p=0., log10M1_0=13.3, log10M1_p=0., alpha_0=1.,
                 alpha_p=0., bg_0=1., bg_p=0., bmax_0=1., bmax_p=0., a_pivot=1.,
                 ns_independent=False, integration_method="FFTLog", rmin=0.001, N_r=512,
                 N_jn=10000):
        import warnings

        if lmax >= 13:
            lmax = 12
            warnings.warn("Maximum l provided too high. Using lmax=12.")
        elif lmax < 2:
            lmax = 2
            warnings.warn("Maximum l provided too low. Using lmax=2.")
        if integration_method not in ("FFTLog", "simpson", "spline"):
            raise ValueError("Integration method provided not supported. Use `FFTLog`, "
                             "`simpson`, or `spline`.")
        if integration_method != "FFTLog":
            raise NotImplementedError("Only integration_method='FFTLog' is ported to JAX")
        self.integration_method = integration_method
        self.a1h = a1h
        self.b = b
        if lmax % 2 != 0:
            lmax = lmax - 1
        self.lmax = lmax
        self.rmin, self.N_r, self.N_jn = rmin, N_r, N_jn
        super().__init__(mass_def=mass_def, concentration=concentration,
                         log10Mmin_0=log10Mmin_0, log10Mmin_p=log10Mmin_p, siglnM_0=siglnM_0,
                         siglnM_p=siglnM_p, log10M0_0=log10M0_0, log10M0_p=log10M0_p,
                         log10M1_0=log10M1_0, log10M1_p=log10M1_p, fc_0=0.0, fc_p=0.0,
                         alpha_0=alpha_0, alpha_p=alpha_p, bg_0=bg_0, bg_p=bg_p,
                         bmax_0=bmax_0, bmax_p=bmax_p, a_pivot=a_pivot,
                         ns_independent=ns_independent)
        self.update_precision_fftlog(padding_lo_fftlog=1e-2, padding_hi_fftlog=1e3,
                                     n_per_decade=350, plaw_fourier=-3.7)

    @property
    def _angular_fl(self):
        if self.b == 0:
            fl = _FL_B0
        elif self.b == -2:
            fl = _FL_BM2
        else:
            fl = np.array([_fl(ell, b=self.b) for ell in range(2, self.lmax + 1, 2)])
        return np.asarray(fl).reshape(-1, 1)

    def update_parameters(self, **kwargs):
        for key in ("a1h", "b", "lmax", "rmin", "N_r", "N_jn"):
            if kwargs.get(key) is not None:
                setattr(self, key, kwargs.pop(key))
        super().update_parameters(**kwargs)

    def get_normalization(self, cosmo, a, *, hmc):
        def integ(M):
            Nc, Ns = self._Nc(M, a), self._Ns(M, a)
            return Nc + Ns if self.ns_independent else Nc * (1 + Ns)

        return hmc.integrate_over_massfunc(integ, cosmo, a)

    def gamma_I(self, r, r_vir):
        if self.b == 0:
            return self.a1h
        r_use = jnp.maximum(_as1d(r), 0.06)
        r_vir = jnp.reshape(jnp.asarray(r_vir, dtype=float), (-1, 1))
        return jnp.minimum(self.a1h * (r_use / r_vir) ** self.b, 0.3)

    def _real(self, cosmo, r, M, a):
        M_use, r_use = _as1d(M), _as1d(r)
        rvir = self.mass_def.get_radius(cosmo, M_use, a) / a
        u = jnp.reshape(self._usat_real(cosmo, r_use, M_use, a), (M_use.shape[0], -1))
        return _squeeze_kM(self.gamma_I(r_use, rvir) * u, r, M)

    def _usat_fourier(self, cosmo, k, M, a):
        ells = tuple(range(2, self.lmax + 1, 2))
        fl = self._angular_fl[:, 0]
        weights = tuple(float((1j**ell).real * (2 * ell + 1) * fl[j] / (4 * np.pi))
                        for j, ell in enumerate(ells))
        prof = self._fftlog_multi(cosmo, k, M, a, ells=ells, weights=weights)
        return _squeeze_kM(prof, k, M)


__all__ = [
    "FFTLogParams",
    "HaloProfile",
    "HaloProfileHOD",
    "HaloProfileMatter",
    "HaloProfileNFW",
    "SatelliteShearHOD",
]
