Cobaya likelihoods
==================

Any likelihood Cobaya ships, and those of the external packages Cobaya lists, can be named in a
GPUCosmo yaml. Each runs in one of three ways, chosen automatically in this order:

1. **Built in** — the Planck PR3 (plik + low-ℓ TT + EE) / PR4 (HiLLiPoP + LoLLiPoP + low-ℓ TT)
   combinations and ACT DR6 lensing with them (``inference.cobaya_yaml``).
2. **Native port** (``gpucosmo.likelihoods.cobaya``: BAO, H0 / M_b, SN, CMBlikes, Planck
   CamSpec / plik-lite / low-ℓ (clipy), ACT DR6 lensing) — a JAX re-implementation of the
   likelihood's ``logp`` registered under its Cobaya name. Cobaya still loads the data, the
   yaml options and the parameter priors (the port reads them off the loaded Cobaya
   instance), so values agree with Cobaya to rounding; the port is jitted, vmapped and
   differentiated with the rest of the analysis on the GPU.
3. **Bridge** (``gpucosmo.cobaya_bridge``) — anything else Cobaya can load: Cobaya evaluates
   the likelihood on the host, on products computed by the GPU theory. It is exact but not
   differentiable (see :ref:`bridge-in-pipelines`).

``gpucosmo-likelihoods`` lists built-ins and ports.

Example
-------

.. code-block:: yaml

    theory:
      camb:
        extra_args: {lens_potential_accuracy: 1}
    likelihood:
      bao.desi_dr2:                      # native port (BAO)
      sn.desy5:                          # native port (SN, M_b marginalized)
      planck_2018_lensing.native:        # native port (CMBlikes)
      planck_2018_highl_CamSpec2021.TTTEEE:
      planck_2018_lowl.TT_clik:          # native port (clipy)
      tdcosmo.tdcosmo.TDCOSMO:           # bridge (hierarc), with options
        tdcosmo_lenses: True
    params:
      logA: {prior: {min: 1.61, max: 3.91}}
      ns: {prior: {min: 0.8, max: 1.2}}
      theta_MC_100: {prior: {min: 0.5, max: 10}}
      ombh2: {prior: {min: 0.005, max: 0.04}}
      omch2: {prior: {min: 0.05, max: 0.3}}
      tau: {prior: {min: 0.01, max: 0.8}}
      mnu: 0.06                          # or sampled: see Neutrinos and curvature

The likelihoods' nuisance parameters come with Cobaya's default priors; entries in ``params``
override them, and parameters with the same name are shared (Cobaya semantics, e.g.
``A_planck``). A likelihood that needs CMB spectra puts the analysis in the CMB parameter
basis (θ_MC, τ sampled); BAO / SN-only analyses sample H0.

Theory blocks
-------------

What a Cobaya likelihood requests is computed on the GPU and expanded by the DALI expansion
like every other theory output, and handed to ports (``LikelihoodInputs.shared``) and to the
bridge (as Cobaya provider products, CAMB conventions):

* **Background** — H(z), angular-diameter, comoving and luminosity distances, D_A(z₁, z₂),
  Ω_b/Ω_cdm/Ω_ν(z), r_drag, z_drag, z*, r*, θ*, cosmomc_theta, Y_He, Ω_m, … (w0waCDM with the
  analysis' neutrinos and curvature, as the solver: :doc:`extended_models`).
* **CMB spectra** (``"cmb"``) — lensed TT, TE, EE, BB, φφ (and κκ) to the requested ℓ (up to
  13 500), also without any built-in CMB likelihood. Every lensed solve runs on the solver's
  accurate path (high-k source modes with scaled step budgets; with ℓ_s = max(ℓ_max, 2508 + 751),
  unlensed spectra to ℓ_s + 500 with a log-slope tail, φφ and the lensing step to ℓ_s + 4000);
  ``--cmb-legacy-lensing`` (``MGConfig.cmb_legacy_lensing``) reproduces the base path of runs
  made before 2026-10-17 (see below).
* **Tensor modes** — with ``r`` sampled, the unlensed tensor TT/TE/EE/BB (ℓ ≤ 1000, n_t from the
  inflation-consistency relation as CAMB's Python default) are added to the lensed scalar spectra,
  as CAMB's "total".
* **Matter power** (``"pk"``) — linear P_m (δ_tot) and P_cb (δ_nonu) at the requested
  redshifts on the solver's grid (300 log points to k = 2/Mpc), σ8(z) and f σ8(z) (CAMB's σ²_vd/σ_dd), and on an extended k grid to the
  requested k_max (≤ 50/Mpc) the linear P_m and the non-linear P_m from HMcode-2020/2016.
  Past k = 2/Mpc the extended grid interpolates sparse evolved modes (Δ ln k ≤ 0.2, in the
  same solve) as their ratio to the Eisenstein–Hu shape, which continues only past k_max (for
  HMcode's integrals).

Accuracy against CAMB:

=====================================  ==============================================================
Product                                vs CAMB
=====================================  ==============================================================
σ8(z), f σ8(z), z ≤ 2.3                +9e-5 relative (AccuracyBoost 2)
linear P(k), k ≤ 2/Mpc                 ≤ 6.4e-4 (AB 2, ``Transfer.high_precision``; ±3e-4 BAO phase)
linear P(k), 2 < k ≤ 15/Mpc            ≤ 7.3e-4 (−1e-3 at 50/Mpc in float64)
non-linear P(k) (HMcode-2020)          ≤ 6.9e-4 to k = 15/Mpc (DES Y1 grid, 100 z ≤ 3.5)
lensed TT/EE, ℓ 1000–2508              mean −1.3e-4 / −1e-5, max 4e-4 (converged CAMB: AB 2,
                                       lens accuracy 15, margin 4000)
C_L^κκ, L 1000–2000                    −1.7e-3 (−5.9e-4 with ``source_accuracy_boost=2``)
lensed TT/EE/BB, ℓ 3000–9000           ≤ 6e-4 (converged CAMB)
lensed spectra, ℓ 3000–13 500          ≤ 1.4e-3
tensor TT/EE/BB/TE, 10 ≤ ℓ < 400       ≤ 0.21 % (converged CAMB, AB 4)
=====================================  ==============================================================

and on likelihoods, through the GPU bundle: BAO log L within 2e-4 and SN within 1e-9 of Cobaya +
CAMB; DES Y1 cosmic shear on our "pk" block vs on converged CAMB's −0.005; BICEP/Keck 2018 with r = 0.05 −280.121 vs
−280.095; ACT DR6 CMB-only (ℓ ≤ 9001) −82.304 vs −82.434 with *converged* CAMB — Cobaya's
default CAMB settings give −82.538 there, its lensed spectra being 33 % low at ℓ = 9000 (too
little lensing margin past ℓ_max), so for high-ℓ data compare against converged CAMB.

Before 2026-10-17 the lensed CMB came from a base path whose extended block (read by the built-in
ACT DR6 lensing corrections) had BB up to 13 % off at ℓ 2000–3000 (ACT lensing log L −0.03) and
whose lensed TT/EE at ℓ 1000–2509 were biased +1.5…+7.8e-4; the accurate path is now the default
(+33 % time per CMB evaluation: 2.8 s instead of 2.1 s for a batch of 8 on an RTX 3070 Laptop,
slightly less GPU memory). ``--cmb-legacy-lensing`` (``MGConfig.cmb_legacy_lensing``)
reproduces the old path, e.g. to continue or compare with earlier runs; compare models by evidence
with one setting.

.. _bridge-in-pipelines:

The bridge in the pipelines
---------------------------

A bridged likelihood is a ``jax.pure_callback``: values only. The pipelines use it

* **exactly** wherever values suffice — the exact log posterior (MAP, DALI build), and the
  accept/reject of DALI-trajectory HMC (on the DALI-expanded theory by default, on the exact
  theory with ``--accept exact``);
* with a **central-difference gradient** in the parameters it depends on, on the DALI expansion
  theory, in the MAP stage (``cobaya_bridge.nondiff.fd_term``);
* as a **quadratic proxy** in the samplers on the compressed DALI likelihood: a second-order
  expansion of its log L in (theory parameters, own nuisances), fitted once at the
  expansion point by finite differences. This proxy is an approximation of log L itself,
  outside the DALI hierarchy: the DALI order check does not measure its error, which enters the
  evidence uncorrected. Prefer a native (JAX) port for evidences.

Stochastic likelihoods (hierarc's Monte Carlo in TDCOSMO) are evaluated with common random
numbers (a fixed seed per evaluation), so they are smooth functions of the parameters.
Log-uniform priors are sampled on their box with a normalized log-prior correction, which
keeps the evidence normalization exact; Cobaya ``prior:`` terms with a known JAX form (Planck's
SZ prior) enter ``log_prior`` (differentiable), others stay in the callback.

Writing a native port
---------------------

Subclass ``gpucosmo.likelihoods.cobaya.base.CobayaPort``, list the Cobaya names it serves in
``cobaya_names``, copy the arrays you need from the loaded Cobaya instance in ``setup`` and
write ``loglike(inp)`` in ``jax.numpy`` reading the same products the bridge would pass. Add
the module to ``PORTS`` in ``gpucosmo/likelihoods/cobaya/__init__.py`` and validate it with
``gpucosmo.likelihoods.cobaya.testing.compare`` (port vs bridge on identical inputs, plus
timings).
