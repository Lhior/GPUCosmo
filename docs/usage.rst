Usage
=====

From a Cobaya yaml to a likelihood bundle
-----------------------------------------

Analyses are described by Cobaya input files: likelihoods, theory options, parameters and
priors. ``gpucosmo.pipeline.build_analysis_bundle`` resolves the yaml and checks that the
GPU pipeline reproduces every setting, refusing unsupported ones with a clear message. It
then builds the bundle:

.. code-block:: python

    from gpucosmo.pipeline import analysis_configs, build_analysis_bundle

    print(sorted(analysis_configs()))          # the DESI DR2 + CMB + SN configs shipped with the package
    bundle, resolved, diffs = build_analysis_bundle(analysis="w0wa_desidr2_cmb_desy5")

    bundle.param_names            # theory parameters first, then nuisances
    bundle.prior_lo, bundle.prior_hi, bundle.fiducial
    bundle.log_prior_fn(x)        # log prior
    bundle.log_post_batch_fn(X)   # exact log posterior, batched over the rows of X
    bundle.theory_exact(x_th)     # the expensive theory block (what the DALI approximation expands)
    bundle.log_like_from_theory_fn(x, theory, components=True)   # per-likelihood log L

``resolved.notes`` and ``diffs`` record every interpretation of the yaml and any difference
between the bundle and the yaml (both are written to ``config.updated.*.yaml`` by the
command-line tools).

The DALI likelihood
-------------------

The evidence pipeline samples the likelihood in the DALI approximation (Derivative Approximation
for LIkelihoods; Sellentin, Quartin & Amendola 2014, arXiv:1401.6892; Sellentin 2015,
arXiv:1506.04866): the model is expanded in the parameters around a fiducial point (the MAP)
and the likelihood is evaluated on the expansion. The expansion's order sets the DALI order:
1 is the Fisher matrix, 2 the doublet, 3 the triplet (the DALI tensors of Šarčević, van der Wild
& Sellentin 2026, arXiv:2610.02104), and so on. ``bundle.dali.build(x_th, order=3)`` expands the
theory block (CMB spectra, distances, P(k), ...) to that order around ``x_th`` in the monomial
basis of ``gpucosmo.jax.inference.dali_expansion``:

* by default from a finite-difference design of exact theory evaluations with steps of about
  one posterior σ, each stencil exact for polynomials of the expansion's degree
  (1 + 2n² evaluations at second order, 1 + 4n + 2n(n − 1) + 8 C(n, 3) at third: 129 and 593
  for the eight theory parameters of w0waCDM + CMB);
* or ``method="fwd"``: JAX forward-mode derivatives at ``x_th`` (k nested jvps per degree-k
  monomial, the theory differentiated stage by stage).

``bundle.dali.theory(expansion, x)`` evaluates the expansion, and the analysis's likelihood codes
run on it, so every DALI term of the chosen order is included, including the terms of a
non-zero residual at the fiducial point. Two differences from the Gaussian DALI of the papers:

* the likelihoods keep their own form (Planck's low-ℓ Commander and SimAll, CamSpec, ACT DR6
  lensing, supernovae, user likelihoods), evaluated on the expanded theory;
* nuisance parameters are not expanded: they enter the likelihood codes exactly.

The evidence and posterior are those of the DALI likelihood. Its accuracy is checked within the
DALI hierarchy: after the SMC, the same expansion truncated one order lower is evaluated at the
posterior draws, and importance weights between the two give the last order's effect on log Z
and on every posterior mean (the *order check*, in ``<tag>.ns_summary.json``). It needs no exact
likelihood evaluation. A converged hierarchy shows a small change; increase ``--dali-order``
otherwise. Consecutive orders add terms of alternating parity in each parameter, so one step
can miss an error dominated by the next term of the other parity: a theory whose leading
neglected dependence is quartic (CamSpec's high-ℓ spectra in ω_c, for example) changes little
from order 2 to 3. From order 4 on, the check therefore also reports p − 2 → p.

``gpucosmo.jax.inference.dali_reduce.CompressedDALI`` applies every likelihood's linear algebra
to the expansion's coefficients once, so a DALI log L + gradient costs microseconds per point
on a GPU. This is what the SMC engine samples. Its cost grows with the number of monomials,
C(n + p, p) for n theory parameters at order p (45 at second and 165 at third order for n = 8).

Evidence
--------

The command-line pipeline computes the Bayesian evidence of the DALI likelihood. One command
runs every stage (resumable: stages whose outputs exist are skipped):

.. code-block:: bash

    gpucosmo-evidence --config my_analysis.yaml --outdir OUT [--plugin my_likelihoods.py] [--dali-order 3]

It keeps the exact stages in one process (the exact theory program is traced, lowered and loaded
once) and runs the SMC as a subprocess. The stages separately:

1. ``gpucosmo-dali-hmc --stage map`` — the posterior maximum;
2. ``gpucosmo-dali-hmc --stage build --expansion-only --dali-order 3`` — the DALI expansion at the MAP;
3. ``gpucosmo-dali-ns --stage ns --compressed-dali`` — tempered SMC on the DALI likelihood:
   log Z_DALI and posterior draws, then the order check.

Defaults: ``--smc-reference pilot``, which starts the tempering from a Student-t fitted to a
2048-particle pilot instead of the prior; ESS target 0.8; 3 HMC moves × 12 leapfrog per stage;
8 final moves; 8192 particles per run. ``--smc-target-err`` adds runs until the log Z_DALI error is
reached. Compare models with the same DALI order and lensing treatment (``--cmb-linear-lens`` /
``--cmb-nonlinear-lens``) in every stage.

**Wide priors.** The pilot and the SMC start from the prior. Where the prior is much wider
than the posterior (the DESI DR2 + CMB yamls: θ_MC ∈ [0.5, 10], w ∈ [−3, 1], wa ∈ [−3, 2]),
the DALI expansion is evaluated far outside the region it describes, and likelihoods such as
Planck's Commander low-ℓ TT return non-finite values for almost every prior draw: the pilot
then collapses onto a single particle. ``--auto-boxes NSIGMA`` avoids this: after the build it
fits a Laplace approximation at the MAP and samples every uniform-prior parameter in
mean ± NSIGMA σ (within its prior), correcting log Z by the volume ratio. With ``NSIGMA = 10``
the likelihood outside the boxes is negligible.

Adding data
-----------

See :doc:`adding_likelihoods`.
