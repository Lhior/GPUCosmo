Development
===========

Create an environment and install the package in editable mode with the development
extras:

.. code-block:: console

   >> conda create -n gpucosmo python=3.13
   >> conda activate gpucosmo
   >> pip install -e '.[dev,desi-full]'
   >> pre-commit install

Tests (``pytest``; ``-m "not slow"`` skips the multi-minute ones):

.. code-block:: console

   >> JAX_PLATFORMS=cpu pytest -m "not slow" tests/

The slow tests (``-m slow``) need the GPU (``JAX_PLATFORMS=cuda,cpu``; the CPU runs host
callbacks). Some tests also read data sets (``data/``), Cobaya packages, or generated files in
``outputs/``, and skip without them.

The tests keep every compiled XLA program in the persistent compilation cache
(``GPUCOSMO_JAX_CACHE_DIR``, default ``~/.cache/gpucosmo_jax``; ``tests/conftest.py``), so a
rerun mostly loads instead of compiling; ``GPUCOSMO_TEST_JAX_CACHE=0`` turns it off. The
tests compute the theory with ``jax_eb`` only; the other Boltzmann back ends (CCL, CosmoPower)
have no tests.

CI (workflow "CI", ``.github/workflows/testing-and-coverage.yml``) is split by trust. Every pull
request runs on GitHub-hosted machines: lint, docs and the tests without data on Python 3.11 and
3.13. Pull-request code never runs on a self-hosted machine: the jobs with data and the GPU run
only in the development repository, on trusted code. A pull request that only changes
documentation (``docs/``, ``*.md``, ``*.rst``) runs lint and docs only.
``.github/workflows/README.md`` describes the jobs and the security rules.

Documentation (``docs/``, Sphinx with sphinx-autoapi):

.. code-block:: console

   >> sphinx-build -b html docs docs/_build/html

Building a release: ``python -m build`` (sdist + wheel). Build from a clean commit:
setuptools-scm takes the version from git, and the sdist contains the files git tracks.
