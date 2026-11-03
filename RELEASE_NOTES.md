# Release notes

## 0.1.0 — research preview (2026-11-03)

The first public release of GPUCosmo. It is a **research preview**: usable for research within
the scope in [docs/research_preview.rst](docs/research_preview.rst), not a validated production
code. The independent full-likelihood validation campaign is pending. In it, the evidences and
posteriors will be compared with exact-likelihood samplers run independently at full cost on
data-centre GPUs. Until then, evidences are research results. Their quoted ± is Monte Carlo
precision only. It does not include the DALI truncation error (estimated by the order check),
the theory's systematic error or posterior mass far from the expansion point.

It is distributed on GitHub only, not on PyPI: install from a clone or the tag,
`pip install "GPUCosmo @ git+https://github.com/Lhior/GPUCosmo@v0.1.0"`
([docs/installation.rst](docs/installation.rst)).

### What is in it

- `jax_eb`, an emulator-free Einstein–Boltzmann solver in JAX: lensed CMB spectra and φφ
  (ℓ ≤ 2508, extended lensed spectra to 13500), linear P(k), HMcode-2020/2016. It covers flat
  ΛCDM and w0waCDM, neutrino masses, hierarchies, N_eff, a sterile neutrino and spatial
  curvature (validated at |Ω_k| ≤ 0.05).
- Likelihoods configured by Cobaya yamls: DESI DR2 BAO, Planck 2018 low-ℓ TT / EE (clipy),
  Planck CamSpec NPIPE, ACT DR6 lensing, supernovae (DES-SN5YR, Pantheon+, Union3), BBN, as
  native JAX ports; any other Cobaya likelihood through a bridge.
- A plug-in API for user likelihoods (`gpucosmo.likelihoods`, entry point
  `gpucosmo.likelihoods`).
- Evidence pipeline (`gpucosmo-evidence`): MAP → DALI expansion of the theory to order
  `--dali-order` (default 3, the triplet; finite-difference stencils exact to that order, or JAX
  forward mode) → tempered SMC on the compressed DALI likelihood → DALI order check (the same
  expansion one order lower, at the posterior draws: the last order's effect on log Z and on the
  posterior means).
- Three demos: [examples/custom_likelihood](examples/custom_likelihood) (CPU, no data files),
  [examples/consumer_gpu](examples/consumer_gpu) (GPU spectra against CAMB, no data files) and
  the w0waCDM vs ΛCDM evidence from DESI DR2 BAO + CMB + SN (README).

### Behaviour worth knowing

- **Numerical failures are reported and rejected.** A failed mode of the CMB perturbation solve
  (non-finite or runaway sources) makes the spectra of that evaluation NaN, so the likelihood is
  non-finite and the inference rejects the point (log L = −∞). Every `compute_cmb_cls` result
  carries `solver_status`; `GPUCOSMO_SOURCE_FAILURE=mask` zeroes failed modes instead, and
  `keep` (alias `GPUCOSMO_STRICT_SOURCES=1`) is for validation.
- **The DALI build stops on failed exact evaluations**: a non-finite theory at any design point
  is an error naming that point.
- **Evidences are those of the DALI likelihood.** No exact-likelihood reweighting: check the
  order check's changes in `<tag>.evidence.json` and raise `--dali-order` if they matter.
- **Wide priors need sampling boxes.** With priors far wider than the posterior, use
  `gpucosmo-evidence --auto-boxes 10` (docs/usage.rst).
- **CI** runs pull-request code only on GitHub-hosted runners (lint, docs, CPU tests without
  data on Python 3.11 and 3.13), with least-privilege tokens and third-party actions pinned to
  commit SHAs ([.github/workflows/README.md](.github/workflows/README.md)).
- **License: GPL-3.0-or-later.** GPUCosmo translates parts of CAMB, ISiTGR and Cobaya (LGPL-3.0
  with additional conditions) and of the Planck PR4 likelihoods HiLLiPoP and LoLLiPoP (GPL-3.0)
  to JAX. Derived files name their sources; the upstream notices are in [NOTICE.md](NOTICE.md)
  and [LICENSES/](LICENSES). The README's *Citing* section lists the papers to cite for each part.

### Known limitations

- Closed models: the low-ℓ accuracy of the Olver hyperspherical map at |Ω_k| ≳ 0.1 is not
  validated, and the P(k) at k ≲ 4.5√K differs from CAMB by up to 20 % (few discrete modes).
- Tensor modes and μ–Σ modified gravity with curvature are not supported (refused).
- Every non-default Σm_ν uses the conservative light-neutrino step budget (+28 % CMB time).

### Not included

- Analyses and likelihoods still in development: the DESI DR1 full-shape likelihood, μ–Σ
  analyses, DES Y3 and HSC Y3, and further Cobaya likelihood ports. The resolver refuses them
  with a message.
- CUDA-native evolution and line-of-sight kernels (experimental).
- Likelihood data products. They are public, but they are not redistributed here
  ([docs/data.rst](docs/data.rst)).
