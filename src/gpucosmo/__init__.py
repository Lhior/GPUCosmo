"""GPUCosmo: GPU (JAX) cosmological likelihoods and Bayesian evidence.

Entry points:

* ``gpucosmo.pipeline.build_analysis_bundle`` — Cobaya yaml → GPU likelihood bundle (exact
  theory, DALI expansion, priors);
* ``gpucosmo.likelihoods`` — add your own likelihoods (``Likelihood``, ``GaussianLikelihood``,
  ``register_likelihood``; docs/adding_likelihoods.rst);
* ``gpucosmo.paths`` — where data, Cobaya packages and caches are looked up;
* command line: ``gpucosmo-evidence`` (MAP, DALI expansion, evidence), ``gpucosmo-dali-hmc`` (MAP,
  DALI build, HMC), ``gpucosmo-dali-ns`` (evidence: tempered SMC on the DALI likelihood + the
  DALI order check), ``gpucosmo-likelihoods``, ``gpucosmo-paths``.
"""

from ._version import __version__

__all__ = ["__version__"]
