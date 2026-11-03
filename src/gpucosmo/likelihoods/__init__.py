"""User likelihoods for the GPU bundles: base classes, registry and examples.

Minimal example (a Gaussian measurement of the BAO scale at z = 0.5)::

    import jax.numpy as jnp
    from gpucosmo.likelihoods import GaussianLikelihood, register_likelihood

    @register_likelihood
    class MyDV(GaussianLikelihood):
        name = "my_dv"

        def __init__(self, value=13.0, sigma=0.2, **options):
            super().__init__(data=[value], cov=[[sigma**2]], **options)

        def model(self, cosmo):
            return jnp.atleast_1d(cosmo.volume_distance(0.5) / cosmo.rdrag)

Then name it in a Cobaya yaml (``likelihood: {my_dv: {value: 13.1}}``, with ``--plugin
my_module`` on the command line), or pass an instance:
``build_analysis_bundle(config=..., likelihoods=[MyDV(value=13.1)])``.
See ``docs/adding_likelihoods.rst``.
"""

from gpucosmo.likelihoods.base import C_KM_S, Cosmology, GaussianLikelihood, Likelihood, LikelihoodInputs, Param
from gpucosmo.likelihoods.registry import (
    available_likelihoods,
    get_likelihood_class,
    load_entry_points,
    load_plugin_module,
    register_likelihood,
)

__all__ = ["C_KM_S", "Cosmology", "GaussianLikelihood", "Likelihood", "LikelihoodInputs", "Param",
           "available_likelihoods", "get_likelihood_class", "load_entry_points", "load_plugin_module",
           "register_likelihood"]
