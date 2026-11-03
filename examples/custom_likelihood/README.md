# Custom likelihood example

* `my_likelihoods.py` — two user likelihoods: `cosmic_chronometers` (a `GaussianLikelihood`)
  and `tt_band_power` (a general `Likelihood` with a nuisance parameter that reads the
  built-in CMB spectra; needs an analysis with a primary CMB likelihood).
* `analysis.yaml` — a background-only analysis (BBN + n_s priors, the built-in example
  `bao_distances`, and `cosmic_chronometers`) that runs in minutes on a CPU.
* `run_example.py` — the same analysis from Python (bundle, exact log posterior, DALI
  expansion, compressed DALI likelihood).

Command line (from the repository root), in one command:

```bash
gpucosmo-evidence --config examples/custom_likelihood/analysis.yaml \
    --plugin examples/custom_likelihood/my_likelihoods.py --outdir out_example
```

or stage by stage:

```bash
P="--config examples/custom_likelihood/analysis.yaml --plugin examples/custom_likelihood/my_likelihoods.py --outdir out_example"
gpucosmo-dali-hmc $P --stage map
gpucosmo-dali-hmc $P --stage build --expansion-only
gpucosmo-dali-ns  $P --stage ns --compressed-dali --smc-particles 2048 --smc-runs 4 --smc-ess 0.8 --smc-mcmc 3 --tag smc
```

The last line prints log Z_DALI, the evidence of the third-order (triplet) DALI likelihood, and
the DALI order check: how much going from order 2 to 3 changed log Z and the posterior means.
± is the Monte Carlo precision (docs/research_preview.rst). On 2026-11-02, on a laptop CPU, the
one-command run gave log Z_DALI = −14.720 ± 0.004 in 27 s; order 2 → 3 changed log Z by −0.004
and no posterior mean by more than 0.005σ. (An earlier version reweighted the second-order
DALI posterior by the exact likelihood and got −14.717 ± 0.005.)
