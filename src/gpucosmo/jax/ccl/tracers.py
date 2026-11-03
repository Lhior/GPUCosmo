#!/usr/bin/env python3
# This file is part of GPUCosmo. Copyright (C) 2025-2026 Leonel Medina-Varela.
# SPDX-License-Identifier: GPL-3.0-or-later
# Derived in part from the following, translated to JAX and modified for GPUCosmo:
# - CCL (https://github.com/LSSTDESC/CCL), Copyright (c) 2016-2018, the LSSTDESC CCL
#   contributors.
#   Licence: LICENSES/CCL.txt.
# See NOTICE.md.
"""CCL-syntax tracers in JAX (``pyccl.tracers`` port).

Every numerical step follows ``ccl_tracers.c`` (CCL 3.x):

- n(z) is an Akima spline normalised by exact spline integration
  (``NZ_NORM_SPLINE_INTEGRATION``); no clipping of negative values.
- Density kernels live on the n(z) nodes, ``W = p(z) H(z)``.
- Lensing kernels are sampled at ``n_samples`` redshifts ``z_max i / n``;
  the radial integral is an Akima spline integral on the n(z) nodes plus
  CCL's trapezoid correction (``LENSING_KERNEL_SPLINE_INTEGRATION``).
- Kernels are Akima splines in chi, zero outside their nodes; kernel support
  uses ``CCL_FRAC_RELEVANT``; transfers use natural cubic splines.
- Intrinsic alignments: ``ia_bias=(z, A_IA(z))`` with ``use_A_ia=True``
  applies ``-A C1 rho_crit Omega_m / D(z)`` (NLA); ``use_A_ia=False`` passes
  the raw amplitude (use with PT / halo-model IA power spectra).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from gpucosmo.jax.ccl.cosmology import (
    A_SPLINE_MIN,
    CLIGHT_HMPC,
    IA_C1,
    RHO_CRITICAL,
    Cosmology,
)
from gpucosmo.jax.ccl.pk2d import F2D, transfer_f2d
from gpucosmo.jax.gsl_splines import (
    Spline1D,
    akima_spline,
    f1d_eval,
    spline_integral,
)

jax.config.update("jax_enable_x64", True)

CCL_FRAC_RELEVANT = 5.0e-4


def _check_array_params(arr, name: str, *, like=None):
    if arr is None:
        return None, None
    if not isinstance(arr, (tuple, list)):
        if like is None or jnp.ndim(arr) != 0:
            raise ValueError(f"{name} must be a tuple of two arrays.")
        z = jnp.asarray(like, dtype=float)  # convenience: constant value
        return z, jnp.full(z.shape, arr, dtype=float)
    if len(arr) != 2:
        raise ValueError(f"{name} must be a tuple of two arrays.")
    return jnp.asarray(arr[0], dtype=float), jnp.asarray(arr[1], dtype=float)


@dataclass(frozen=True)
class ClTracer:
    """One ``ccl_cl_tracer_t``: kernel(chi) x transfer(k, a) x f(ell)."""

    kernel: Spline1D | None
    transfer: F2D | None
    chi_min: jnp.ndarray
    chi_max: jnp.ndarray
    der_bessel: int = field(default=0, metadata={"static": True})
    der_angles: int = field(default=0, metadata={"static": True})

    def get_kernel(self, chi):
        if self.kernel is None:
            return jnp.ones_like(jnp.asarray(chi, dtype=float))
        return f1d_eval(self.kernel, jnp.asarray(chi, dtype=float))

    def get_transfer(self, lk, a):
        if self.transfer is None:
            return jnp.ones(jnp.broadcast_shapes(jnp.shape(lk), jnp.shape(a)))
        return self.transfer.eval(lk, a)

    def get_f_ell(self, ell):
        return f_ell(ell, self.der_angles)


jax.tree_util.register_dataclass(
    ClTracer,
    data_fields=["kernel", "transfer", "chi_min", "chi_max"],
    meta_fields=["der_bessel", "der_angles"],
)


def f_ell(ell, der_angles: int):
    """``ccl_cl_tracer_t_get_f_ell`` including CCL's piecewise approximations."""
    ell = jnp.asarray(ell, dtype=float)
    if der_angles == 1:
        return ell * (ell + 1.0)
    if der_angles == 2:
        lp1h2 = (ell + 0.5) ** 2
        exact = jnp.sqrt(jnp.maximum((ell + 2.0) * (ell + 1.0) * ell * (ell - 1.0), 0.0))
        return jnp.where(
            ell <= 1.0,
            0.0,
            jnp.where(
                ell <= 10.0,
                exact,
                jnp.where(ell <= 1000.0, lp1h2 * (1.0 - 1.25 / lp1h2), lp1h2),
            ),
        )
    return jnp.ones_like(ell)


def _kernel_support(chi: jnp.ndarray, w: jnp.ndarray):
    """chi_min / chi_max where |W| exceeds CCL_FRAC_RELEVANT of its peak."""
    w_abs = jnp.abs(w)
    w_thr = jnp.max(w_abs) * CCL_FRAC_RELEVANT
    lo = jnp.argmax(w_abs[1:] >= w_thr)
    hi = chi.shape[0] - 2 - jnp.argmax((w_abs[:-1] >= w_thr)[::-1])
    return chi[lo], chi[hi + 1]


class Tracer:
    """Collection of ``ClTracer`` contributions (``pyccl.Tracer``)."""

    def __init__(self):
        self._trc: list[ClTracer] = []
        self._dndz = None

    def __bool__(self) -> bool:
        return bool(self._trc)

    @property
    def chi_min(self):
        return None if not self._trc else jnp.min(jnp.stack([t.chi_min for t in self._trc]))

    @property
    def chi_max(self):
        return None if not self._trc else jnp.max(jnp.stack([t.chi_max for t in self._trc]))

    def get_kernel(self, chi):
        return jnp.stack([t.get_kernel(chi) for t in self._trc if t.kernel is not None])

    def get_transfer(self, lk, a):
        return jnp.stack([t.get_transfer(lk, a) for t in self._trc])

    def get_f_ell(self, ell):
        return jnp.stack([t.get_f_ell(ell) for t in self._trc])

    def get_bessel_derivative(self):
        return np.array([t.der_bessel for t in self._trc])

    def get_angles_derivative(self):
        return np.array([t.der_angles for t in self._trc])

    def get_dndz(self, z):
        if self._dndz is None:
            raise AttributeError("Tracer has no redshift distribution")
        z_n, n = self._dndz
        return jnp.interp(jnp.asarray(z, dtype=float), z_n, n, left=0.0, right=0.0)

    def add_tracer(
        self,
        cosmo: Cosmology,
        *,
        kernel=None,
        transfer_ka=None,
        transfer_k=None,
        transfer_a=None,
        der_bessel: int = 0,
        der_angles: int = 0,
        is_logt: bool = False,
        extrap_order_lok: int = 0,
        extrap_order_hik: int = 2,
    ) -> None:
        if der_angles not in (0, 1, 2):
            raise ValueError("der_angles must be between 0 and 2")
        if der_bessel not in (-1, 0, 1, 2):
            raise ValueError("der_bessel must be between -1 and 2")
        if kernel is None:
            spl = None
            chi_min = jnp.asarray(0.0)
            chi_max = cosmo.comoving_radial_distance(A_SPLINE_MIN)
        else:
            chi_s = jnp.asarray(kernel[0], dtype=float)
            w_s = jnp.asarray(kernel[1], dtype=float)
            spl = akima_spline(chi_s, w_s)
            chi_min, chi_max = _kernel_support(chi_s, w_s)
        transfer = transfer_f2d(
            transfer_ka=transfer_ka,
            transfer_k=transfer_k,
            transfer_a=transfer_a,
            is_logt=is_logt,
            extrap_order_lok=extrap_order_lok,
            extrap_order_hik=extrap_order_hik,
        )
        self._trc.append(
            ClTracer(
                kernel=spl,
                transfer=transfer,
                chi_min=chi_min,
                chi_max=chi_max,
                der_bessel=int(der_bessel),
                der_angles=int(der_angles),
            )
        )


def _tracer_flatten(t: Tracer):
    return (tuple(t._trc), t._dndz), None


def _tracer_unflatten(_aux, children):
    t = Tracer()
    t._trc = list(children[0])
    t._dndz = children[1]
    return t


jax.tree_util.register_pytree_node(Tracer, _tracer_flatten, _tracer_unflatten)

NzTracer = Tracer


# -- kernels -----------------------------------------------------------------
def _nz_norm(z_n: jnp.ndarray, n: jnp.ndarray) -> jnp.ndarray:
    return spline_integral(akima_spline(z_n, n), z_n[0], z_n[-1])


def get_density_kernel(cosmo: Cosmology, *, dndz):
    """(chi, p(z) H(z)) on the n(z) nodes; H in 1/Mpc."""
    z_n, n = _check_array_params(dndz, "dndz")
    a = 1.0 / (1.0 + z_n)
    chi = cosmo.comoving_radial_distance(a)
    hz = cosmo["h"] * cosmo.h_over_h0(a) / CLIGHT_HMPC
    return chi, hz * n / _nz_norm(z_n, n)


def get_nchi_lensing_kernel(z_n) -> int:
    z_n = np.asarray(z_n, dtype=float)
    dz = (z_n[-1] - z_n[0]) / (z_n.size - 1)
    return int(z_n[-1] / dz + 0.5)


def get_lensing_kernel(cosmo: Cosmology, *, dndz, mag_bias=None, n_chi=None):
    """(chi, W_L(chi)) exactly as ``ccl_get_lensing_mag_kernel`` (spline mode).

    ``W = 3 H0^2 Om / (2a) chi int dz p(z) (1 - 5 s(z)/2) (chi(z)-chi)/chi(z)``.
    """
    z_n, n = _check_array_params(dndz, "dndz")
    if n_chi is None:
        n_chi = get_nchi_lensing_kernel(np.asarray(z_n))
    n_chi = int(n_chi)
    z_max = z_n[-1]
    z_chi = z_max / n_chi * jnp.arange(n_chi) + 1.0e-15
    chi_arr = cosmo.comoving_radial_distance(1.0 / (1.0 + z_chi))
    chi_nodes = cosmo.comoving_radial_distance(1.0 / (1.0 + z_n))
    if mag_bias is not None:
        z_s, s = _check_array_params(mag_bias, "mag_bias")
        qz = 1.0 - 2.5 * f1d_eval(akima_spline(z_s, s), z_n, y_lo=s[0], y_hi=s[-1])
    else:
        qz = jnp.ones_like(z_n)
    i_norm = 1.0 / _nz_norm(z_n, n)
    hub = cosmo["h"] / CLIGHT_HMPC
    lens_prefac = 1.5 * hub * hub * cosmo["Omega_m"]
    nz_count = z_n.shape[0]

    def one(chi_end):
        a_end = cosmo.scale_factor_of_chi(chi_end)
        z_end = 1.0 / a_end - 1.0
        below = chi_nodes < chi_end
        safe_chi = jnp.where(chi_nodes == 0.0, 1.0, chi_nodes)
        integrand = jnp.where(
            below,
            0.0,
            jnp.where(
                chi_nodes == 0.0,
                n * qz,
                n * qz * (chi_nodes - chi_end) / safe_chi,
            ),
        )
        i0 = jnp.minimum(jnp.sum(below), nz_count - 1)
        result = spline_integral(akima_spline(z_n, integrand), z_n[i0], z_n[-1])
        trapz = 0.5 * (z_n[i0] - z_end) * integrand[i0]
        result = result + jnp.where(z_end > z_n[0], trapz, 0.0)
        return result * lens_prefac * i_norm * chi_end / a_end

    return chi_arr, jax.vmap(one)(chi_arr)


def get_kappa_kernel(cosmo: Cosmology, *, z_source, n_samples: int = 100):
    """CMB-lensing-style kernel for a single source plane."""
    chi_source = cosmo.comoving_radial_distance(1.0 / (1.0 + z_source))
    chi = jnp.linspace(0.0, chi_source, int(n_samples))
    a = cosmo.scale_factor_of_chi(chi)
    hub = cosmo["h"] / CLIGHT_HMPC
    prefac = 1.5 * hub * hub * cosmo["Omega_m"] / chi_source
    return chi, prefac * (chi_source - chi) * chi / a


# -- concrete tracers ----------------------------------------------------------
def NumberCountsTracer(
    cosmo: Cosmology,
    *,
    dndz,
    bias=None,
    mag_bias=None,
    has_rsd: bool,
    n_samples: int = 256,
) -> Tracer:
    """Galaxy clustering: linear bias, optional RSD and magnification."""
    tracer = Tracer()
    z_n, n = _check_array_params(dndz, "dndz")
    tracer._dndz = (z_n, n)
    if bias is None and not has_rsd and mag_bias is None:
        raise ValueError(
            "Number counts tracers must have a non-zero bias, "
            "RSDs, or a magnification bias contribution."
        )
    kernel_d = None
    if bias is not None:
        kernel_d = get_density_kernel(cosmo, dndz=(z_n, n))
        z_b, b = _check_array_params(bias, "bias", like=z_n)
        tracer.add_tracer(
            cosmo, kernel=kernel_d, transfer_a=(1.0 / (1.0 + z_b[::-1]), b[::-1])
        )
    if has_rsd:
        if kernel_d is None:
            kernel_d = get_density_kernel(cosmo, dndz=(z_n, n))
        a_s = 1.0 / (1.0 + z_n[::-1])
        tracer.add_tracer(
            cosmo,
            kernel=kernel_d,
            transfer_a=(a_s, -cosmo.growth_rate(a_s)),
            der_bessel=2,
        )
    if mag_bias is not None:
        mag = _check_array_params(mag_bias, "mag_bias", like=z_n)
        chi, w = get_lensing_kernel(cosmo, dndz=(z_n, n), mag_bias=mag, n_chi=n_samples)
        tracer.add_tracer(cosmo, kernel=(chi, -2.0 * w), der_bessel=-1, der_angles=1)
    return tracer


def WeakLensingTracer(
    cosmo: Cosmology,
    *,
    dndz,
    has_shear: bool = True,
    ia_bias=None,
    use_A_ia: bool = True,
    n_samples: int = 256,
) -> Tracer:
    """Galaxy shapes: lensing shear plus optional (N)LA intrinsic alignments."""
    tracer = Tracer()
    z_n, n = _check_array_params(dndz, "dndz")
    tracer._dndz = (z_n, n)
    if has_shear:
        kernel_l = get_lensing_kernel(cosmo, dndz=(z_n, n), n_chi=n_samples)
        tracer.add_tracer(cosmo, kernel=kernel_l, der_bessel=-1, der_angles=2)
    elif ia_bias is None:
        raise ValueError(
            "Weak lensing tracers with no shear must have a non-zero "
            "intrinsic alignment amplitude."
        )
    if ia_bias is not None:
        z_a, tmp_a = _check_array_params(ia_bias, "ia_bias", like=z_n)
        kernel_i = get_density_kernel(cosmo, dndz=(z_n, n))
        if use_A_ia:
            # Joachimi et al. (2011) Eq. 6 with C_1 = 5e-14 (Brown et al. 2002).
            growth = cosmo.growth_factor(1.0 / (1.0 + z_a))
            rho_m = RHO_CRITICAL * cosmo["Omega_m"]
            amp = -tmp_a * IA_C1 * rho_m / growth
        else:
            amp = tmp_a
        tracer.add_tracer(
            cosmo,
            kernel=kernel_i,
            transfer_a=(1.0 / (1.0 + z_a[::-1]), amp[::-1]),
            der_bessel=-1,
            der_angles=2,
        )
    return tracer


def CMBLensingTracer(cosmo: Cosmology, *, z_source, n_samples: int = 100) -> Tracer:
    tracer = Tracer()
    kernel = get_kappa_kernel(cosmo, z_source=z_source, n_samples=n_samples)
    tracer.add_tracer(cosmo, kernel=kernel, der_bessel=-1, der_angles=1)
    return tracer


__all__ = [
    "CCL_FRAC_RELEVANT",
    "CMBLensingTracer",
    "ClTracer",
    "NumberCountsTracer",
    "NzTracer",
    "Tracer",
    "WeakLensingTracer",
    "f_ell",
    "get_density_kernel",
    "get_kappa_kernel",
    "get_lensing_kernel",
    "get_nchi_lensing_kernel",
]
