"""Example likelihoods, usable as they are and as templates (``docs/adding_likelihoods.rst``).

* ``h0_gaussian`` — a Gaussian measurement of H0 (default: SH0ES 2022, 73.04 ± 1.04);
* ``bao_distances`` — BAO distance ratios D_M/r_d, D_H/r_d, D_V/r_d, D_A/r_d at any
  redshifts with a covariance, from yaml options or a text file.

In a Cobaya yaml::

    likelihood:
      h0_gaussian: {mean: 73.04, sigma: 1.04}
      bao_distances:
        z: [0.51, 0.51]
        quantities: [DM_over_rd, DH_over_rd]
        data: [13.62, 20.98]
        cov: [[0.0625, -0.0244], [-0.0244, 0.3364]]
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from gpucosmo.likelihoods.base import Cosmology, GaussianLikelihood
from gpucosmo.likelihoods.registry import register_likelihood


@register_likelihood
class GaussianH0(GaussianLikelihood):
    """Gaussian measurement of H0 [km/s/Mpc] (options ``mean``, ``sigma``)."""

    name = "h0_gaussian"

    def __init__(self, mean: float = 73.04, sigma: float = 1.04, **options):
        super().__init__(data=[mean], cov=[[sigma**2]], **options)

    def model(self, cosmo: Cosmology):
        return jnp.atleast_1d(cosmo.H0)


_QUANTITIES = {
    "DM_over_rd": lambda c, z: c.transverse_comoving_distance(z) / c.rdrag,
    "DH_over_rd": lambda c, z: c.hubble_distance(z) / c.rdrag,
    "DV_over_rd": lambda c, z: c.volume_distance(z) / c.rdrag,
    "DA_over_rd": lambda c, z: c.angular_diameter_distance(z) / c.rdrag,
    "rd_over_DV": lambda c, z: c.rdrag / c.volume_distance(z),
}


@register_likelihood
class BAODistances(GaussianLikelihood):
    """BAO distance ratios at given redshifts with a full covariance.

    Options: ``z``, ``quantities`` (each one of ``DM_over_rd``, ``DH_over_rd``,
    ``DV_over_rd``, ``DA_over_rd``, ``rd_over_DV``), ``data`` and ``cov`` — or ``file``: a
    text file with columns ``z value quantity`` and ``cov_file`` with the covariance (Cobaya's
    ``bao`` likelihood format).
    """

    name = "bao_distances"

    def __init__(self, z=None, quantities=None, data=None, cov=None, file=None, cov_file=None, **options):
        if file is not None:
            rows = [ln.split() for ln in open(file) if ln.strip() and not ln.lstrip().startswith("#")]
            z = [float(r[0]) for r in rows]
            data = [float(r[1]) for r in rows]
            quantities = [r[2] for r in rows]
            cov = np.loadtxt(cov_file) if cov_file is not None else cov
        if z is None or quantities is None or data is None or cov is None:
            raise ValueError("bao_distances: give z, quantities, data and cov (or file + cov_file)")
        unknown = sorted(set(quantities) - set(_QUANTITIES))
        if unknown:
            raise ValueError(f"bao_distances: unknown quantities {unknown}; known {sorted(_QUANTITIES)}")
        if not len(z) == len(quantities) == len(data):
            raise ValueError("bao_distances: z, quantities and data must have the same length")
        self.z = np.asarray(z, dtype=np.float64)
        self.quantities = list(quantities)
        super().__init__(data=data, cov=cov, **options)

    def model(self, cosmo: Cosmology):
        return jnp.stack([_QUANTITIES[q](cosmo, z) for q, z in zip(self.quantities, self.z)])


__all__ = ["BAODistances", "GaussianH0"]
