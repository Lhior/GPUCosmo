"""Pytest configuration for GPUCosmo."""

from __future__ import annotations

import os

import pytest

# JAX + pyccl together can crash on some GPU backends in WSL; force CPU.
os.environ.setdefault("JAX_PLATFORMS", "cpu")


def pytest_configure(config):
    """Persistent XLA compilation cache for every test (``gpucosmo.paths.jax_cache_dir``:
    ``GPUCOSMO_JAX_CACHE_DIR`` or ``~/.cache/gpucosmo_jax``): reruns, the per-file slow-test
    processes and later CI runs load the compiled programs instead of compiling them again
    (compilation is most of the suite's time). ``GPUCOSMO_TEST_JAX_CACHE=0`` turns it off.

    Every program is kept, however quick to compile: the tests run thousands of single
    operations outside ``jit``, and the same small programs are compiled again many times
    within one run (one ``multiply`` up to 84 times; 16 000 compiles of ~0.025 s, 445 s, in
    the DES Y3 + autosize slow tests). With them on disk those tests take 405 s warm instead
    of 900 s (1 s threshold) or ~1030 s (no cache)."""
    if os.environ.get("GPUCOSMO_TEST_JAX_CACHE", "1") == "0":
        return
    try:
        from gpucosmo.jax_cache import enable_jax_cache

        enable_jax_cache(min_compile_time_secs=0.0)
    except ImportError as e:  # no JAX: the JAX tests skip themselves
        config.issue_config_time_warning(pytest.PytestConfigWarning(f"no JAX compilation cache: {e}"), 2)


# Run tests in the firecrown_developer conda env:
#   conda run -n firecrown_developer env JAX_PLATFORMS=cpu pytest tests/ -v
