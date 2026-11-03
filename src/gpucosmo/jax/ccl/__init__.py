#!/usr/bin/env python3
"""pyccl-compatible JAX API for tracers, IA models and Limber C_ell.

Written so analysis code can swap ``import pyccl as ccl`` for
``import gpucosmo.jax.ccl as ccl`` with the same tracer definitions::

    cosmo = ccl.Cosmology.from_ccl(ccl_cosmo)          # or Cosmology.from_tables
    src = ccl.WeakLensingTracer(cosmo, dndz=(z, nz), ia_bias=(z, A_IA))
    lens = ccl.NumberCountsTracer(cosmo, dndz=(z, nz), bias=(z, b), has_rsd=False)
    cl = ccl.angular_cl(cosmo, lens, src, ell)

Intrinsic-alignment models available as in CCL:

- NLA / LA: ``WeakLensingTracer(ia_bias=(z, A(z)), use_A_ia=True)``; pass
  ``p_of_k_a='linear'`` for the linear-alignment model.
- TATT (tidal alignment + tidal torquing): :mod:`gpucosmo.jax.ccl.nl_pt`
  ``PTIntrinsicAlignmentTracer`` + ``EulerianPTCalculator`` with an
  ``ia_bias=(z, 1), use_A_ia=False`` weak-lensing tracer.
- Halo-model IA (``pyccl.halos.SatelliteShearHOD``): build the IA ``Pk2D``
  natively with :mod:`gpucosmo.jax.ccl.halos` (same syntax as ``pyccl.halos``,
  jittable and differentiable), or pass a ``pyccl.Pk2D`` as ``p_of_k_a``
  (converted, exact but not differentiable in cosmology).
"""

from __future__ import annotations

from gpucosmo.jax.ccl import halos, nl_pt
from gpucosmo.jax.ccl.cells import angular_cl
from gpucosmo.jax.ccl.correlations import correlation
from gpucosmo.jax.ccl.cosmology import (
    CLIGHT_HMPC,
    DEFAULT_POWER_SPECTRUM,
    IA_C1,
    RHO_CRITICAL,
    Cosmology,
)
from gpucosmo.jax.ccl.halos import sigma8, sigmaM, sigmaR
from gpucosmo.jax.ccl.pk2d import Pk2D
from gpucosmo.jax.ccl.halofit import halofit_pk2d
from gpucosmo.jax.ccl.tracers import (
    CMBLensingTracer,
    NumberCountsTracer,
    NzTracer,
    Tracer,
    WeakLensingTracer,
    get_density_kernel,
    get_kappa_kernel,
    get_lensing_kernel,
)

__all__ = [
    "CLIGHT_HMPC",
    "CMBLensingTracer",
    "Cosmology",
    "DEFAULT_POWER_SPECTRUM",
    "IA_C1",
    "NumberCountsTracer",
    "NzTracer",
    "Pk2D",
    "RHO_CRITICAL",
    "Tracer",
    "WeakLensingTracer",
    "angular_cl",
    "correlation",
    "get_density_kernel",
    "get_kappa_kernel",
    "get_lensing_kernel",
    "halofit_pk2d",
    "halos",
    "nl_pt",
    "sigma8",
    "sigmaM",
    "sigmaR",
]
