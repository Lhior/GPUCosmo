"""Example GPUCosmo plug-in: a Gaussian likelihood on r_d·h [Mpc] (the BAO scale in h⁻¹ Mpc).

Installed, it is selected in a Cobaya yaml by its entry-point name::

    likelihood:
      rdh_gaussian: {mean: 99.1, sigma: 1.0}
"""

import jax.numpy as jnp

from gpucosmo.likelihoods import GaussianLikelihood


class RdhGaussian(GaussianLikelihood):
    """Gaussian measurement of r_d h [Mpc] (options ``mean``, ``sigma``)."""

    name = "rdh_gaussian"

    def __init__(self, mean: float = 99.1, sigma: float = 1.0, **options):
        super().__init__(data=[mean], cov=[[sigma**2]], **options)

    def model(self, cosmo):
        return jnp.atleast_1d(cosmo.rdrag * cosmo.h)
