Adding a likelihood
===================

GPUCosmo's analyses are a sum of independent log-likelihood terms. Besides the built-in
ones (DESI DR2 BAO, Planck, ACT DR6 lensing, supernovae, BBN), you
can add your own as small Python classes. They then work everywhere the built-ins do:

* in exact evaluations (``bundle.log_post_batch_fn``, the MAP and the DALI build);
* in the **DALI expansion** (order 3 by default), which expands your likelihood's
  cosmology-dependent outputs together with the built-in theory;
* in the **compressed DALI likelihood** sampled by the GPU SMC engine
  (``gpucosmo-dali-ns --compressed-dali``);
* in Cobaya yamls, by name, with their options and parameter priors.

The code lives in ``gpucosmo.likelihoods``. Complete, runnable examples are in
``examples/custom_likelihood`` (a plug-in file, a yaml and a Python script) and
``examples/gpucosmo_example_plugin`` (an installable plug-in package).

.. contents:: On this page
   :local:
   :depth: 2


The two halves of a likelihood
------------------------------

A GPUCosmo likelihood is split into two methods, because the pipeline treats them
differently:

``theory(cosmo)`` — **cosmology only**
    Everything that depends on the cosmological parameters alone: distances, a model data
    vector, spectra you compute yourself. Exact evaluations call it once per cosmology. The
    DALI approximation expands its outputs (a dict of arrays) to the DALI order (default 3)
    around the expansion point. An expensive ``theory`` therefore costs nothing per evaluation of the
    DALI likelihood.

``loglike(inp)`` — **per point**
    The log-likelihood from the ``theory`` outputs and your nuisance parameters. It runs at
    every sampler step (millions of times), so keep it cheap.

Both run inside ``jax.jit``, ``jax.vmap`` and ``jax.grad`` (the samplers use gradients),
so write them with ``jax.numpy``. :ref:`jax-rules` lists what that implies.


A Gaussian likelihood in five lines
-----------------------------------

Most data are a Gaussian data vector with a covariance. Subclass
``GaussianLikelihood``, pass ``data`` and ``cov`` (or ``precision``) to its
constructor, and implement ``model(cosmo)``:

.. code-block:: python

    import jax.numpy as jnp
    import numpy as np
    from gpucosmo.likelihoods import GaussianLikelihood, register_likelihood

    @register_likelihood
    class CosmicChronometers(GaussianLikelihood):
        """H(z) from cosmic chronometers."""

        name = "cosmic_chronometers"          # the yaml key and component label

        def __init__(self, z, H, sigma, **options):   # options come from the yaml block
            self.z = np.asarray(z, dtype=float)
            super().__init__(data=H, cov=np.diag(np.asarray(sigma) ** 2), **options)

        def model(self, cosmo):
            return cosmo.H(self.z)            # km/s/Mpc, a JAX array of len(z)

χ² = (d − m)ᵀ C⁻¹ (d − m). Because the model depends on the cosmology only and the
χ² is quadratic in it, the compressed DALI likelihood folds the covariance's whitening into the
expansion coefficients. That is exact, and needs no n × n product per point.

The ``cosmo`` object (``gpucosmo.likelihoods.Cosmology``) gives the background at the
current parameters:

========================================  ===================================================================================================================================
``cosmo.H(z)``                            Hubble rate [km/s/Mpc]
``cosmo.comoving_distance(z)``            D_C = D_M (flat) [Mpc]
``cosmo.angular_diameter_distance(z)``    D_A [Mpc]
``cosmo.luminosity_distance(z)``          D_L [Mpc]
``cosmo.hubble_distance(z)``              D_H = c/H [Mpc]
``cosmo.volume_distance(z)``              D_V [Mpc]
``cosmo.rdrag``                           sound horizon at the drag epoch r_d [Mpc]
``cosmo.theta_star``                      θ* = r_s(z*)/D_M(z*)
``cosmo.Omega_m``                         total matter density today
parameters                                ``h``, ``H0``, ``ombh2``, ``omch2``, ``ns``, ``As``, ``logA``, ``tau``, ``w0``, ``wa``, ``mu0``, ``Sigma0`` (None if not sampled)
``cosmo.mnu``, ``nnu``, ``omk``, …        Σm_ν, N_eff, Ω_k, ``meffsterile`` (CAMB's defaults when not sampled); ``cosmo.omnuh2``, ``cosmo.Omega_k``
========================================  ===================================================================================================================================

The background is w0waCDM with the analysis' neutrinos and curvature (D_M, D_A, D_L, D_V use
f_K), the same as the built-in theory (:doc:`extended_models`).

A likelihood whose ``theory`` runs its own physics, rather than going through ``cosmo``, declares
what that physics holds fixed. Then an analysis that samples or changes those parameters is
refused, instead of being silently inconsistent:

* ``fixed_ext = ("nnu", "omk")``: the extended parameters held fixed;
* ``neutrino_model = ("degenerate", 3)``: the neutrino states its theory assumes when it follows
  ``cosmo.mnu``.

Two Gaussian examples ship with the package and are registered already:
``h0_gaussian`` (an H0 measurement) and ``bao_distances`` (BAO distance ratios
D_M/r_d, D_H/r_d, D_V/r_d, D_A/r_d at any redshifts, from yaml options or a text file).
See ``gpucosmo/likelihoods/examples.py``.


Nuisance parameters and general likelihoods
-------------------------------------------

Declare sampled nuisance parameters in ``params`` with ``Param(lo, hi, ref=None,
gauss=None)`` (a uniform box, optionally a Gaussian prior). ``loglike`` receives their
values in ``inp.params``:

.. code-block:: python

    from gpucosmo.likelihoods import GaussianLikelihood, Param

    class CalibratedH0(GaussianLikelihood):
        name = "calibrated_h0"
        params = {"h0_cal": Param(0.9, 1.1, gauss=(1.0, 0.02))}

        def __init__(self, **options):
            super().__init__(data=[73.0], cov=[[1.0]], **options)

        def model(self, cosmo):                 # cosmology only: DALI-expanded
            return jnp.atleast_1d(cosmo.H0)

        def vector(self, model, params):        # per point: apply the nuisance
            return model * params["h0_cal"]

For anything that is not a Gaussian in a model vector, subclass ``Likelihood`` and write
``loglike`` yourself:

.. code-block:: python

    from gpucosmo.likelihoods import Likelihood, Param, register_likelihood

    @register_likelihood
    class StudentTH0(Likelihood):
        """H0 with a Student-t likelihood (heavy tails)."""

        name = "h0_student_t"
        params = {"h0_shift": Param(-2.0, 2.0, gauss=(0.0, 0.5))}

        def __init__(self, mean=73.0, sigma=1.0, nu=4.0, **options):
            super().__init__(**options)
            self.mean, self.sigma, self.nu = float(mean), float(sigma), float(nu)

        def theory(self, cosmo):
            return {"H0": cosmo.H0}

        def loglike(self, inp):
            t = (inp.theory["H0"] + inp.params["h0_shift"] - self.mean) / self.sigma
            return -0.5 * (self.nu + 1.0) * jnp.log1p(t**2 / self.nu)

``loglike`` receives a ``LikelihoodInputs`` named tuple:

* ``inp.theory`` — the output of ``theory`` at this point (exact, or the
  DALI expansion's), or None if you define no ``theory``;
* ``inp.params`` — your parameters' values;
* ``inp.cosmo`` — the ``Cosmology`` at this point, from the *exact* background;
* ``inp.shared`` — the built-in theory blocks named in ``requires`` (next section).

You can skip ``theory`` and compute everything from ``inp.cosmo`` in ``loglike``. That
is exact and fine for cheap quantities. In the compressed DALI likelihood it costs an exact
background calculation per point (with a CMB likelihood, a θ* → H0 solve), whereas
outputs of ``theory`` cost almost nothing there.

``log_prior(params)`` adds prior terms beyond the per-parameter boxes and Gaussians, e.g.
a correlated prior on several nuisances.

Parameter names must be unique across the whole analysis (built-in and user
likelihoods). The cosmological parameters are the built-in ones, and a plug-in cannot add
new ones (see :ref:`limitations`).


Using the built-in CMB theory
-----------------------------

A likelihood can read the built-in spectra instead of computing its own. Name the blocks
in ``requires``:

.. code-block:: python

    class TTBandPower(Likelihood):
        name = "tt_band_power"
        requires = ("cmb",)
        params = {"tt_cal": Param(0.9, 1.1, gauss=(1.0, 0.01))}

        def loglike(self, inp):
            dl = inp.shared["cmb"]["dl_tt"][200:251]       # D_ℓ^TT [μK²], ℓ = 200…250
            band = inp.params["tt_cal"] ** 2 * jnp.mean(dl)
            return -0.5 * ((band - 5700.0) / 60.0) ** 2

``shared["cmb"]`` holds the lensed ``dl_tt``, ``dl_te``, ``dl_ee`` (D_ℓ in μK², ℓ = 0 …
2508), plus ``dl_kk`` (or ``cmb_ext`` up to ℓ = 2999) when the analysis includes ACT
lensing. The analysis must contain a primary CMB likelihood; otherwise building the bundle
fails with a clear error.


Making the likelihood known
---------------------------

A Cobaya yaml's ``likelihood:`` key selects your class in one of these ways:

1. **By registered name.** ``@register_likelihood`` registers the class under its
   ``name`` (and its ``aliases``). The module must be imported before the yaml is read:
   pass ``--plugin my_module`` (a dotted module name or ``path/to/file.py``) to the
   command-line tools, or ``plugins=[...]`` to ``build_analysis_bundle``. Calling
   ``gpucosmo.likelihoods.load_plugin_module`` in your own script also works.

2. **By entry point.** An installed package declares its classes in its
   ``pyproject.toml``; they are found automatically, with no flag:

   .. code-block:: toml

       [project.entry-points."gpucosmo.likelihoods"]
       my_bao = "my_package.bao:MyBAO"

   ``examples/gpucosmo_example_plugin`` is a complete plug-in package
   (``pip install -e examples/gpucosmo_example_plugin``).

3. **By import path.** Use the key ``my_package.module.MyLikelihood`` (or
   ``my_package.module:MyLikelihood``), or any key with a ``class:`` option in its block.

``gpucosmo-likelihoods [--plugin MODULE]`` lists every likelihood a yaml can name.


Using it in a yaml
------------------

.. code-block:: yaml

    likelihood:
      cosmic_chronometers:            # registered name: options go to __init__
        z: [0.18, 0.35, 0.59]
        H: [75.0, 83.0, 104.0]
        sigma: [4.0, 14.0, 13.0]
      h0_riess:                       # a second instance under its own name
        class: h0_gaussian
        mean: 73.04
        sigma: 1.04
      calibrated_h0: null             # no options
    params:
      h0_cal:                         # overrides the Param default of calibrated_h0
        prior: {dist: norm, loc: 1.0, scale: 0.05}

The rules:

* **Options.** Every key of the block is passed to ``__init__``, except Cobaya's own
  (``class``, ``python_path``, ``speed``, ``stop_at_error``, ``input_params``,
  ``output_params``, ``version``, ``aliases``). Load data files in ``__init__``, as NumPy
  arrays.
* **Instance name.** The name is the yaml key when the key is the class's ``name`` or the
  block has a ``class:`` option. With an import-path or alias key, it is the class's
  ``name``. A ``name:`` option sets it explicitly. Names label the log-likelihood
  components and must be unique.
* **Priors.** A ``params`` entry in the yaml overrides your ``Param`` default (Cobaya
  ``prior: {min, max}`` or ``{dist: norm, loc, scale}``). Parameters the yaml omits keep
  their defaults, and the run notes say so.
* **Records.** ``config.updated.*.yaml`` lists the user likelihoods of each run.

Then run the usual pipeline, in one command:

.. code-block:: bash

    gpucosmo-evidence --config my_analysis.yaml --plugin my_likelihoods.py --outdir OUT

or stage by stage:

.. code-block:: bash

    P="--config my_analysis.yaml --plugin my_likelihoods.py --outdir OUT"
    gpucosmo-dali-hmc $P --stage map
    gpucosmo-dali-hmc $P --stage build --expansion-only
    gpucosmo-dali-ns  $P --stage ns --compressed-dali --tag smc


Using it from Python
--------------------

.. code-block:: python

    from gpucosmo.pipeline import build_analysis_bundle

    bundle, resolved, diffs = build_analysis_bundle(
        config="my_analysis.yaml",                    # path or dict
        likelihoods=[CosmicChronometers(z=..., H=..., sigma=...)],   # added to the yaml's
    )
    bundle.param_names                 # built-in parameters, then yours
    bundle.log_post_batch_fn(X)        # exact log posterior of the rows of X
    th = bundle.theory_exact(x[:bundle.n_theory])
    bundle.log_like_from_theory_fn(x, th, components=True)["cosmic_chronometers"]

``gpucosmo.jax.inference.mg_bundle.build_mg_bundle(cfg, priors=..., likelihoods=[...])``
is the lower-level entry point.


What happens in each stage
--------------------------

========================  ==================================================================
exact evaluation          ``theory`` at each cosmology (batched over walkers), then
                          ``loglike``
DALI expansion            ``theory``'s outputs are expanded to ``--dali-order`` with the rest
(``--stage build``)       of the theory block, by a finite-difference design (``--dali-method
                          fd``) or JAX forward-mode derivatives (``fwd``). User outputs are
                          not clipped (the built-in CMB/FS outputs are), so keep them smooth
                          in the cosmology.
compressed DALI           a ``GaussianLikelihood`` whose ``model``, ``vector`` and
likelihood (SMC engine)   ``loglike`` are not overridden: χ² = \|Lᵀd − Σ_t φ_t Lᵀm_t\|², with
                          the whitening folded into the coefficients. Otherwise the expanded
                          outputs are rebuilt per point (float64 expansion point, float32
                          deviation) and your ``loglike`` is called on them.
order check               your ``loglike`` on the expansion truncated one order lower, at the
                          posterior draws: the last order's effect on log Z and the means
========================  ==================================================================

A DALI expansion saved before you added a likelihood has no outputs for it. Rebuild it
(``--stage build``) after changing the analysis's likelihoods; the compressed DALI likelihood
refuses a stale expansion.


.. _jax-rules:

Rules for JAX code
------------------

* Use ``jax.numpy`` for everything that depends on parameters. NumPy is fine for
  constants in ``__init__``: data, covariances, redshift grids.
* Shapes must not depend on parameter values: no boolean masking with traced values. Use
  ``jnp.where`` instead.
* No Python ``if`` / ``for`` on parameter values. Python loops over static things (your
  data points, redshifts) are fine.
* Return a scalar from ``loglike`` and a dict of arrays (or None) from ``theory``.
* Computations are in float64 (``jax_enable_x64`` is on). Return ``-jnp.inf`` (or let a
  NaN happen) for invalid regions; the pipeline treats non-finite values as zero
  likelihood.
* No side effects (printing, file I/O, global state) inside ``theory`` / ``loglike``.


Testing your likelihood
-----------------------

Check the four things the pipeline relies on: the exact value, the DALI expansion accuracy,
the compressed form, and finite gradients. ``tests/test_likelihood_plugins.py`` does this
for the examples. A minimal version on a cheap background-only analysis
(``examples/custom_likelihood/analysis.yaml``):

.. code-block:: python

    import jax, numpy as np
    from gpucosmo.pipeline import build_analysis_bundle
    from gpucosmo.jax.inference.dali_reduce import CompressedDALI

    b, _, _ = build_analysis_bundle(config="analysis.yaml", likelihoods=[MyLike()], verbose=False)
    x0 = np.asarray(b.fiducial)
    X = jax.numpy.asarray(x0 + 1e-3 * np.random.default_rng(0).normal(size=(8, x0.size)) * x0)
    tt = b.dali.build(x0[:b.n_theory], verbose=False)
    exact = jax.vmap(lambda x: b.log_like_from_theory_fn(x, b.theory_exact(x[:b.n_theory]), components=True))(X)
    surr = jax.vmap(lambda x: b.log_like_from_theory_fn(x, b.dali.theory(tt, x), components=True))(X)
    cs = CompressedDALI(b, tt[0], x0, log=lambda m: None)
    comp = jax.vmap(lambda x: cs.components(x, cs.data))(X)
    print(np.abs(surr["my_like"] - exact["my_like"]).max())      # DALI truncation error
    print(np.abs(comp["my_like"] - surr["my_like"]).max())       # ≲ 1e-6
    print(np.isfinite(jax.vmap(jax.grad(lambda x: cs.loglike(x, cs.data)))(X)).all())


Performance
-----------

* Put the expensive cosmology-only work in ``theory``. It then runs only in the few
  hundred exact evaluations (MAP, DALI build), never in the sampler.
* Keep ``loglike`` small: the SMC engine evaluates the compressed DALI likelihood millions of
  times (~6 µs per point for the full DESI + Planck + ACT analysis on a laptop GPU).
* Prefer ``GaussianLikelihood`` without ``vector`` when the data allow it: it gets the
  whitened fast path.
* If ``loglike`` reads the built-in ``"cmb"`` block (``requires``), override ``cmb_lmax()`` to
  return the largest multipole it uses: the compressed DALI likelihood then evaluates the shared
  spectra only that far (otherwise to ℓ ≈ 4000 at every point).
* Avoid ``inp.cosmo`` in ``loglike`` for analyses with a CMB likelihood: each use costs a
  θ* → H0 solve per point in the compressed DALI likelihood.


.. _limitations:

Limitations
-----------

* **Cosmological model.** The cosmological parameters and the built-in theory are fixed:
  w0waCDM (+ the solver's μ–Σ modified gravity) with CAMB's neutrino models and spatial curvature
  (:doc:`extended_models`). A plug-in adds
  likelihoods and nuisance parameters, not new cosmological parameters or theory codes. A
  new theory quantity must be computed inside ``theory`` from the available parameters; for
  example, call ``gpucosmo.jax.eb.linear_pk_jax`` yourself for P(k) at other redshifts.
* **Smoothness.** The DALI expansion assumes outputs that are well described by a polynomial
  of the DALI order over the posterior. The order check reports how much the last order
  changed log Z and the posterior means; a large change means a higher ``--dali-order`` (or a
  smoother parameterization of your outputs) is needed.
* **Built-in likelihoods** keep their dedicated code paths (and compressed forms). They are
  listed by ``gpucosmo-likelihoods`` as ``[built-in]``, and their names are reserved.
* **Independence.** Each likelihood is an independent term of the total log L. Data sets
  with correlated errors must be implemented as one likelihood with the joint covariance.
