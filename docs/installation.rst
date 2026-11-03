Installation
============

GPUCosmo needs Python ≥ 3.11 and JAX ≥ 0.10. The core dependencies are NumPy, SciPy, JAX and PyYAML.
The research preview is distributed on GitHub, not on PyPI: install from a clone, or from the
tag with ``pip install "GPUCosmo @ git+https://github.com/Lhior/GPUCosmo@v0.1.0"``.

.. code-block:: bash

    pip install .                        # from a source checkout (-e . to develop)
    pip install ".[desi-full]"           # with the extras of the DESI DR2 + CMB + SN demo

(or ``pip install dist/GPUCosmo-*.whl`` for a built wheel; ``python -m build`` makes one).

For speed, install a GPU build of JAX (``pip install ".[gpu]"`` for CUDA 12, or see
the JAX installation guide). The package was developed with JAX 0.10–0.11 on NVIDIA RTX 3070
and RTX 5070 Laptop GPUs (8 GB), and the preview's demos were run from a clean install with JAX
0.12; everything also runs on a CPU, slower.

Optional extras:

==============  ==========================================================================
``planck``      Planck 2018 plik / low-ℓ likelihoods (clipy) and candl
``cobaya``      Cobaya: packages-path lookup, the likelihoods' data loaders
``desi-full``   everything the DESI DR2 + CMB + SN demo needs (``planck``, ``cobaya`` and
                ``camb``); add ``act_dr6_lenslike`` (see :doc:`data`)
``docs``        building this documentation
``dev``         tests and docs
==============  ==========================================================================

Modules that need a missing extra raise ``ModuleNotFoundError`` naming it when imported;
the core pipeline (bundles, DALI expansions, SMC, user likelihoods) needs none of them.

Check the installation:

.. code-block:: bash

    gpucosmo-paths          # data, Cobaya packages and cache locations; which data sets exist
    gpucosmo-likelihoods    # built-in and registered likelihoods

The background-only example in ``examples/custom_likelihood`` runs with the core
dependencies alone and needs no data files (see :doc:`adding_likelihoods`).
