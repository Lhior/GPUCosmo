GPUCosmo
========

GPU-native cosmological likelihoods and Bayesian evidence in JAX.

.. note::

   **Research preview.** Usable for research within the scope in :doc:`research_preview`; the
   independent full-likelihood validation campaign is pending. Read the scope, the known
   limitations and what the evidence uncertainties do and do not include before relying on
   results.

* **Exact theory on the GPU**: an Einstein–Boltzmann solver (CMB C_ℓ, lensing, linear P(k))
  for w0waCDM, with CAMB's neutrino models (Σm_ν, N_eff, normal/inverted hierarchies, massive
  sterile neutrinos) and spatial curvature (see :doc:`extended_models`).
* **Likelihoods**: DESI DR2 BAO, Planck 2018 low-ℓ TT/EE and CamSpec NPIPE high-ℓ, ACT DR6
  lensing and supernovae (DES-SN5YR, Pantheon+, Union3), configured by Cobaya yamls — and
  **your own**, through a small plug-in API.
* **Evidence**: a higher-order DALI expansion of the theory (default the triplet, third order;
  the order is configurable), tempered SMC with HMC moves on its compressed likelihood, and the
  DALI order check, which measures the last order's effect on the evidence and the posterior
  (error budget: :ref:`evidence-error-budget`).

This preview ships three demos: a user likelihood on a CPU, CMB spectra on a consumer GPU
checked against CAMB, and the w0waCDM vs ΛCDM evidence from DESI DR2 BAO + CMB + SN
(:doc:`usage`).

Quick start:

.. code-block:: bash

    pip install ".[desi-full]"                    # from a checkout; see Installation
    gpucosmo-paths                                # where data and caches are looked up
    gpucosmo-likelihoods                          # likelihoods a yaml can name

    gpucosmo-evidence --config my_analysis.yaml --outdir OUT   # MAP → DALI expansion → SMC → order check

.. toctree::
   :maxdepth: 2

   Home page <self>
   research_preview
   installation
   data
   usage
   adding_likelihoods
   extended_models
   cobaya
   cli
   development
   API Reference <autoapi/index>
