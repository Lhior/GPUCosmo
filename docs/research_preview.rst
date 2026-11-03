Research preview: scope, status and limitations
===============================================

GPUCosmo is released as a **research preview**. It is usable for research, and the scope below
is tested. It is not a validated production code: the independent full-likelihood validation
campaign is **pending**. In that campaign, the evidences and posteriors of the GPU pipeline are
compared with exact-likelihood samplers run independently, at full likelihood cost, on
data-centre GPUs. No result in this documentation has been checked by that campaign yet. Until
it is complete, treat the evidences in particular as research results with the error budget
described in :ref:`evidence-error-budget`, not as validated numbers.

Bounded preview scope
---------------------

The preview supports the following. Each entry points to the test or script that checks it.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Area
     - In scope
   * - Theory
     - The ``jax_eb`` Einstein–Boltzmann solver (default): lensed CMB TT/TE/EE/BB and φφ to
       ℓ = 2508 (extended path for lensed spectra to ℓ = 13500), linear P(k), HMcode-2020 / 2016.
       Accuracy against CAMB: ``examples/consumer_gpu``, :doc:`cobaya`, :doc:`extended_models`.
   * - Models
     - Flat ΛCDM, w0waCDM; Σm_ν, N_eff, neutrino
       hierarchies and a massive sterile neutrino (:doc:`extended_models`); spatial curvature
       validated at Ω_k = +0.03, ±0.02, −0.05.
   * - Likelihoods
     - DESI DR2 BAO, Planck 2018 low-ℓ TT (Commander) / EE (SimAll) through clipy, Planck CamSpec
       NPIPE high-ℓ, ACT DR6 lensing (ACT-only and ACT+Planck), supernovae (DES-SN5YR, Pantheon+,
       Union3), BBN, configured by Cobaya yamls (``configs/cobaya``). Plug-in likelihoods
       (:doc:`adding_likelihoods`). Other Cobaya likelihoods through the bridge (:doc:`cobaya`).
   * - Inference
     - ``gpucosmo-evidence``: MAP → order-p DALI expansion (default the triplet, p = 3) → tempered
       SMC on the compressed DALI likelihood → DALI order check (order p − 1 against p).
       DALI-trajectory HMC, accept/reject on the DALI posterior (default) or the exact one
       (``--accept exact``).
   * - Devices
     - NVIDIA GPUs through JAX's CUDA 12 build (extra ``gpu``); CPUs, which work but are slow for
       the CMB. Consumer GPUs used in development: RTX 3070 Laptop (8 GB) and RTX 5070
       Laptop (8 GB).
   * - Precision
     - float64 everywhere except on accelerators: the per-mode evolution and source functions run
       in float32 by default (``GPUCOSMO_SOURCES_DTYPE`` / ``GPUCOSMO_LINPK_DTYPE=float64`` restore
       float64; lensed D_ℓ within 6e-6), and the CamSpec χ² uses a centred float32 product
       (exact algebra; ~1e-6 in χ² near the posterior).
   * - Input ranges
     - Prior boxes are clipped to the background tables: ω_b ∈ [0.005, 0.04], ω_c ∈ [0.05, 0.3],
       Σm_ν ≤ 1.5 eV, 1 ≤ N_eff ≤ 6, m_eff ≤ 3 eV. The clipped boxes are listed in
       ``bundle.notes`` and in each run's ``config.updated.*.yaml``. Unsupported yaml options raise
       an error instead of being ignored (``inference.cobaya_yaml``).
   * - Python / JAX
     - Python ≥ 3.11, JAX ≥ 0.10. CI tests Python 3.11 and 3.13 on CPU without data; the
       developer runs the data and GPU tests on Python 3.13.

Outside the preview
~~~~~~~~~~~~~~~~~~~

These are in the repository but not part of the supported preview:

* **Not supported, refused with an error:** tensor modes with Ω_k ≠ 0; the DESI DR1 full-shape
  likelihood and the DES Y3 / HSC Y3 likelihoods (not part of this release).
* **Limited validation:** closed models with \|Ω_k\| ≳ 0.1 at low ℓ (Olver map) and the
  closed-model P(k) at k ≲ 4.5√K, which differs from CAMB by up to 20 %. The solver's μ–Σ
  modified gravity (ISiTGR conventions) ships, but no μ–Σ analysis is part of the preview.
* **Performance:** every non-default Σm_ν uses the conservative light-neutrino step budget
  (+28 % CMB time).

.. _numerical-failures:

Numerical failures are rejected, not hidden
-------------------------------------------

The perturbation integrator can fail at a pathological point: non-finite values, or a source that
runs away (\|S_T\| > 5, \|Weyl\| > 10, both O(0.1–1) physically). Before this release such modes
were zeroed silently, and the C_ℓ came out finite and plausible but wrong. Now:

* Every ``compute_cmb_cls`` result carries ``solver_status``: the number of failed modes plus the
  non-finite C_ℓ samples (0 = valid; ``cls_jax.solver_ok``).
  ``get_theory_spectra_jax_eb[_batch](..., return_status=True)`` and an analysis bundle's
  ``cmb_solver_status(x_theory)`` return it.
* The default policy, ``GPUCOSMO_SOURCE_FAILURE=reject``, makes the spectra NaN when it is > 0.
  The likelihood is then non-finite and the inference maps it to log L = −∞, so the MAP, the
  exact accept/reject of DALI-trajectory HMC (``--accept exact``) and the DALI build cannot
  accept a failed evaluation as a finite
  value. Valid points are unchanged bit for bit, and JIT, vmap and gradients are unaffected.
* ``mask`` restores the old finite masking (for plots of pathological points). ``keep`` lets
  NaN and runaway values propagate, for validation; ``GPUCOSMO_STRICT_SOURCES=1`` is an older
  name for it.
* A rejected point is a **numerical failure, not a physical exclusion**. Prior boxes and hard cuts
  (``log_prior_fn``) are physical or prior exclusions. A non-finite likelihood inside the prior is
  a failed evaluation, and ``cmb_solver_status`` tells the CMB solver's failures apart.
* The DALI build stops if the exact theory is non-finite at any point of its finite-difference
  design: a failed solve would otherwise enter every expansion coefficient. The error names the
  design point. DALI-trajectory HMC with ``--accept exact`` logs the number of non-finite exact
  evaluations.

The production integrator gave ``solver_status = 0`` at the fiducial cosmology and at the corners
of the w0–wa prior that were checked. The test ``tests/test_solver_validity.py`` uses a genuine
failure: the legacy RK4 integrator without its k-banded step budgets runs away at k = 0.45/Mpc,
and that mode is flagged.

.. _evidence-error-budget:

What the evidence uncertainties mean
------------------------------------

The pipeline reports log Z_DALI, the evidence of the order-p DALI likelihood: the analysis's
likelihood codes evaluated on the order-p expansion of the theory about the MAP, with the nuisance
parameters exact. Its quoted ± (for example ±0.009 for the README's DESI DR2 BAO + CMB + SN
analyses) covers only part of the error:

* **Monte Carlo precision (quoted).** The scatter of the SMC runs, combined in quadrature with the
  Monte Carlo error of the prior mass of the hard cuts. Running longer reduces it.
* **DALI truncation (estimated, not quoted).** The order check evaluates the same expansion
  truncated to order p − 1 at the posterior draws and reports the change of log Z and of every
  posterior mean from order p − 1 to p (``order_check`` in ``<tag>.ns_summary.json``, also in
  ``<tag>.evidence.json``). This is the size of the order-(p − 1) truncation error; that of
  order p is expected smaller when the hierarchy converges, but nothing bounds it. One step can
  also miss an error of the other parity (a missing quartic term barely changes order 2 → 3);
  from order 4 on, the check also reports p − 2 → p. For a Bayes factor compare both models'
  changes. If they are not small against the precision you need,
  raise ``--dali-order``.
* **Region of the expansion (not quoted).** The finite-difference design spans about ±2σ
  around the MAP (steps of about 1σ), so the expansion fits the theory over the posterior bulk and
  extrapolates beyond it. A posterior with mass far from the MAP (a second mode, a long
  curved degeneracy) is outside what an expansion about one point describes, and the order check
  need not reveal it.
* **Theoretical error (not quoted).** The theory is GPUCosmo's own code. Its differences from
  CAMB/CLASS and from the official likelihood implementations shift log Z by amounts the ±
  does not contain. Examples: lensed TT/EE at ℓ 1000–2508 within 4e-4 of converged CAMB,
  float32 evolution with \|Δ log L\| ≤ 0.007 for HiLLiPoP, clipped prior boxes. The float32 effect
  alone is comparable to the quoted precision.
* **Independent validation (pending).** The checks so far:

  * independent estimators of the DALI evidence (importance sampling from a fitted
    Student-t, the Laplace approximation) against the SMC;
  * reference-start against prior-start SMC;
  * the DALI orders against the exact likelihood, for the README's DESI DR2 + CMB + DES-SN5YR
    analyses. An earlier version reweighted the order-2 posterior by the exact likelihood (704
    and 256 exact evaluations, weight ESS/N ≥ 0.99). Those exact values, re-weighted to each
    order's likelihood at the same draws, give:

    ==================  =================  =================  ====  ==================  ==============
    DALI order          log Z, w0waCDM     log Z, ΛCDM        ln B  posterior means     widths
    ==================  =================  =================  ====  ==================  ==============
    2                   −6375.663 ± 0.009  −6379.794 ± 0.009  4.13  ≤ 0.026σ            ≤ 3.5 %
    3                   −6375.683 ± 0.006  −6379.787 ± 0.006  4.10  ≤ 0.010σ            ≤ 4.9 %
    4                   −6375.632 ± 0.008  −6379.767 ± 0.009  4.13  ≤ 0.017σ            ≤ 1.2 %
    exact likelihood    −6375.629 ± 0.009  −6379.779 ± 0.009  4.15
    ==================  =================  =================  ====  ==================  ==============

    (means and widths: largest difference from the exact posterior over the parameters, ω_c's
    width the slowest to converge; Monte Carlo noise ≈ 0.005σ). Against the exact log L at those
    draws, every likelihood but CamSpec converges at order 3 (rms of log L_exact − log L_DALI:
    low-ℓ EE 0.011, DESI BAO 0.007, ACT lensing 0.002); CamSpec's quartic ω_c dependence
    leaves the triplet 0.04–0.05 low in log Z, and the total rms falls from 0.084 (order 2)
    to 0.079 (3) and 0.043 (4);
  * posteriors against official DESI chains (README).

  These share GPUCosmo's likelihood code, or test posteriors rather than evidences. The planned
  independent full-likelihood campaign is what would test the evidence end to end.

Reproducing the published numbers
---------------------------------

The README's timings and log Z values (e.g. "log Z_DALI −6375.632 ± 0.008" for w0waCDM from DESI
DR2 BAO + CMB + DES-SN5YR at DALI order 4) were measured by the developer on an RTX 5070 Laptop from a clean install.
The run directories are not part of this release. They are developer measurements, and an
independent rerun has not reproduced them. ``examples/consumer_gpu`` is a small reproducible GPU example that needs no
data products. Its pinned reference results record the hardware, versions and cold- and
warm-start times.
