Neutrinos and curvature
=======================

Beyond flat w0waCDM with CAMB's default neutrinos (one 0.06 eV eigenstate, N_eff = 3.044), the
theory takes the parameters CAMB's ``set_cosmology`` takes, with CAMB 2.0.4's conventions:

=====================  ====================================================================
``mnu``                Σm_ν [eV] of the active neutrinos
``nnu``                N_eff
``meffsterile``        effective mass of a thermal massive sterile neutrino [eV] (Planck's
                       definition, ΔN_eff = ``nnu`` − 3.044)
``omk``                Ω_k (CAMB sign: Ω_k > 0 is open)
=====================  ====================================================================

and CAMB's neutrino options in the theory block's ``extra_args``: ``num_massive_neutrinos`` (equal-mass
states sharing Σm_ν) and ``neutrino_hierarchy`` (``degenerate``, ``normal``, ``inverted``).

.. code-block:: yaml

    theory:
      camb:
        extra_args: {num_massive_neutrinos: 3, neutrino_hierarchy: normal}
    params:
      mnu: {prior: {min: 0.0602, max: 1.0}}   # the normal hierarchy needs Σm_ν > 0.0601 eV
      nnu: {prior: {min: 2.0, max: 4.5}}
      ...

Sampled (``prior``) or fixed to another value (``value``), they enter every theory block: the
background, θ* → H0, recombination (Y_He from BBN at ΔN_eff), the CMB and matter-power Boltzmann
solves, HMcode, tensors, the Cobaya bridge's products and derived
parameters, and the ``Cosmology`` that plug-in likelihoods receive. With N_eff sampled, Cobaya's
``bbn.BBN_Schoneberg2024``-style likelihood becomes the joint (ω_b, N_eff) Gaussian. In Python the
same is ``MGConfig(ext_sampled=("mnu", "nnu"), nu_hierarchy="normal", ...)``.

Neutrino models
---------------

``gpucosmo.jax.eb.massive_nu.NeutrinoModel`` reproduces CAMB 2.0.4's ``SetNeutrinoMasses``: the
eigenstate masses are exact (Σm_ν / n per state, or CAMB's two-eigenstate approximation of the
hierarchies: a degenerate pair m₁ and m₃ = Σ − 2m₁), each block's temperature is (g/n)^¼ T_ν with g
the N_eff it carries, and ω_ν is the thermal density (so Σm_ν → ω_ν includes the finite-temperature
correction). Massless states take the remaining N_eff. Below a hierarchy's minimum Σm_ν CAMB falls
back to one eigenstate; that branch is reproduced, but CAMB's own switch at Σ = 0.0601 eV (normal)
is a discontinuity, so start such priors above it. Checked against CAMB's ``nu_masses``,
``omnuh2`` and degeneracies to 1e-10 (``tests/test_neutrinos_curvature.py``). The DES basis samples
ω_ν directly (CAMB's ``omnuh2_active``): the mass is then the exact inverse of the thermal density.

The Boltzmann hierarchy has one massive block per eigenstate (``HierarchyConfig.n_nu_eig``): one
for degenerate states, two for the hierarchies (a block whose state is massless in CAMB's
one-eigenstate fallback carries no density), plus one for a sterile neutrino. When an eigenstate
can be light (≲ 0.05 eV: the hierarchies, a sampled Σm_ν), its momentum bins stream at v ≈ 1 to
today and the late (radiation-streaming) segment of the integrator gets 4× its step budget
(``evolve_segmented_jax.NU_LIGHT_RSA_MULT``; +28 % CMB time); with the default budget the explicit
RK4 diverged from k ≈ 0.16/Mpc at Σm_ν = 1e-4 eV.

Each block samples the momentum distribution with CAMB's optimized rules, chosen as CAMB 2.0.4
chooses them from the heaviest eigenstate's m/(k_B T) (its ``nu_masses``;
``massive_nu.camb_nu_nqmax``): 3 bins up to 600 (≈ 0.10 eV for an active state), 4 up to 4300
(≈ 0.72 eV) and 5 above. The 3-bin rule's error grows with the mass. CAMB measures 1.6e-4 in P(k)
at 0.10 eV and 1.9e-3 at 0.63 eV against 100 bins. An analysis (``build_mg_bundle``) takes the
rule for the heaviest state its Σm_ν box (or fixed value) reaches, at the lower edge of its N_eff
box, and 5 bins with a sterile neutrino. The choice is static (``NeutrinoModel.nqmax``), so the
staged and exact theories agree. Called directly, the theory uses 5 bins whenever Σm_ν is given,
unless the model sets ``nqmax``. The DES Y3 plug-in keeps the 3 bins of its chain's CAMB. For
one eigenstate, the CMB (ℓ ≤ 2508) takes 2 % longer with 4 bins and 10 % longer with 5 (RTX 3070
Laptop).

Background and recombination
----------------------------

The background (``gpucosmo.jax.background.cosmology``) uses CAMB's thermal ρ_ν(a) fit per eigenstate
and f_K(χ) for every transverse distance (D_M, D_A, D_L, D_V, D_A(z₁, z₂), θ*), smooth through
Ω_k = 0; it agrees with CAMB to 1e-10 in H(z) and distances, 1e-8 in the sound horizons, for every
neutrino model and \|Ω_k\| ≤ 0.1. z* and z_drag come from Chebyshev fits to CAMB's RECFAST with the
solver's BBN helium (PArthENoPE): the base surface in (ω_b, ω_c) at the default neutrinos
(``w0wa_background_tables_v2.npz``) plus, per neutrino model, a 4-D least-squares correction in
(ω_b, ω_c, Σm_ν, N_eff) — (N_eff, m_eff) with a sterile neutrino — split at the kink of CAMB's N_eff
sharing (``scripts/build_background_tables.py``, ``scripts/build_nu_background_tables.py``; the tables
ship in ``gpucosmo/_data``). Held-out accuracy for ω_b ∈ [0.019, 0.026], ω_c ∈ [0.08, 0.16] and the
full neutrino ranges: z* ≤ 1.8e-6, z_drag ≤ 3e-5 (≤ 2e-6 near N_eff = 3); both are limited by the
wiggles of the BBN-table spline (Y_p is tabulated to five digits) and CAMB's ~3e-6 z_drag jitter.
r_d is then good to ~1e-6, i.e. ~2e-3 in the DESI DR2 BAO log L (0.002σ). The tables' ranges
(Σm_ν ≤ 1.5 eV, 1 ≤ N_eff ≤ 6, m_eff ≤ 3 eV) clip the prior boxes (listed in ``bundle.notes``).
Curvature moves z*, z_drag by < 3e-6 for \|Ω_k\| ≤ 0.1 and is not tabulated.

The version-1 base table (before 2026-10-22) was fitted with CAMB's default BBN table (PRIMAT
2024) while recombination uses PArthENoPE: z* and z_drag were 9e-6 / 8e-6 off, H0 from θ* +0.0017
km/s/Mpc and the DESI DR2 BAO log L up to ~0.01 (results within 0.01σ).

Curvature
---------

Ω_k enters every theory block. The background and θ* use f_K(χ) throughout. The perturbations
follow CAMB's curved equations (``HierarchyConfig.curved``):

* k² = q² − K and the Kf(ℓ) = 1 − Kℓ(ℓ+2)/k² factors of the photon, polarisation and neutrino
  hierarchies; σ/Kf(1); η̇ += Kz; the curved sources.
* Projection with hyperspherical Bessel functions: ``hyperspherical_jax``, a uniform
  (Olver/Liouville–Green) map onto the j_ℓ tables, with the exact forms for ℓ ≤ 2.
* CAMB's measure dq q/k_s² P(k_s).
* Closed models (Ω_k < 0) keep only integer ν = q/√K ≥ 3, and drop ν ≤ ℓ:

  * The static integration grid is rounded where it is denser than ν (the exact sum over ν) and
    kept where coarser, where the sum equals the integral.
  * The evolve grid takes the consecutive integers ν = 3, 4, … where it is denser.
  * Kf(ℓ) is clipped at 0, which is CAMB's truncation of every hierarchy at ℓ = ν − 1. At
    non-integer ν, the ℓ ≥ ν modes would otherwise grow and leak into the quadrupole (C_2 was
    1.7 % low before this).

* C_ℓ^φφ uses the f_K lensing kernel and, above L = 100, CAMB's curved transfer Limber:
  f_K(χ) = (ℓ+½)/q, with weight √(1 − Kℓ²/q²). CAMB uses this, not the flat k = (ℓ+½)/χ,
  whenever it has no source windows.

The linear P(k) evolves k² = q² − K. HMcode takes Ω_k through its H(z) like CAMB's: P_nl/P_lin
agrees with CAMB to ≤ 4e-4 at Ω_k = ±0.05. Likelihoods that read P(k) (``"pk"``) project it with
their own code, as under Cobaya + CAMB: Cobaya's DES/KiDS-style likelihoods use flat Limber
kernels in χ, the same as with CAMB.

``GPUCOSMO_CURVATURE=0`` restores the earlier behaviour: Ω_k at the background level only, with
CMB and P(k) likelihoods refused.

Not supported with Ω_k ≠ 0, and refused loudly:

* tensor modes (``r``), which would need the curved tensor hierarchy and kernels;
* μ–Σ modified gravity (the ISiTGR port is flat).

Validated at Ω_k = +0.03, ±0.02 and −0.05, and at −0.02 with three degenerate 0.2 eV states (θ*
fixed: H0 from 52 to 93):

* C_ℓ responses ≤ 1.6e-3 for TT and ≤ 1.8e-3 for EE at every ℓ, TE ≤ 2.6e-3 (relative to
  √(TT·EE));
* C_L^κκ ≤ 5e-4;
* P(k) ≤ 7e-4 for k > 1e-3/Mpc;
* H0 within 0.002 km/s/Mpc.

Not covered by these points:

* The leading-order Olver map limits closed models at ℓ ≥ 3 to ≲ 2e-3 at ℓ = 3 for
  \|Ω_k\| = 0.1, ~1e-4 by ℓ = 8, far inside cosmic variance.
* A closed P(k) at ν ≲ 4.5 (k ≲ 4.5√K ~ 1.5e-4/Mpc) is not meaningful: there are only a few
  discrete modes there and none below ν = 3. It differs from CAMB's interpolation by up to 20 %.

Accuracy against CAMB
---------------------

The development validation (its scripts are not part of this preview) compared lensed C_ℓ, H0
from θ*, the linear P(k) and σ8 with CAMB 2.0.4 (AccuracyBoost 2) for each model; the *response* column is the change of the
jax/CAMB residual from the default-neutrino point, which isolates the extension from the solver's
fiducial residuals. All runs use ``GPUCOSMO_STRICT_SOURCES=1`` (the
``keep`` failure policy), so a non-finite source fails the run instead of being zeroed. Since the
research preview the default policy rejects such points as well (:ref:`numerical-failures`).

Residuals:

* Hierarchies and light states: ≤ 4e-4, and H0 from θ* within 0.001 km/s/Mpc.
* N_eff: ≤ 1.8e-3 at ℓ 30–1000.
* Heavy states: P(k) ≤ 7.5e-4 and σ8 ≤ 3e-5 of CAMB for a 0.3 eV state and the 1.1 eV sterile
  state (m_eff = 0.5 eV, ΔN_eff = 0.36). For the sterile state, the κκ response is 6.8e-4, and
  TT/EE are at the level of the same ΔN_eff without mass. Three causes made this error grow with
  the mass before 2026-10-23 (P(k) 2.3e-3, σ8 5.1e-4, κκ 2.5e-3 for the sterile state):

  * The radiation-streaming closures' metric estimate left out the massive neutrinos' density
    perturbation, which CAMB includes. This was −1.9e-3 in P(k) at 1 eV, scaling as m².
  * The 3-bin momentum rule was used at every mass (see *Neutrino models*).
  * The Limber φφ source past the evolved k = 0.45/Mpc was continued without free-streaming. It
    now carries Eisenstein & Hu (1999)'s scale-dependent growth. For the sterile state, the
    remaining shape error is −4e-3 at k = 1/Mpc and −1e-2 at k = 2/Mpc, which gives κκ 1.1e-3
    at L 2000–2500. This continuation sets the high-L κκ accuracy for every model.

.. include:: extended_models_table.rst
