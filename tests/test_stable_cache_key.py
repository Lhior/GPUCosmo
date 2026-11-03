"""``pipeline.stable_cache_key``: deterministic persistent-cache keys across processes."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

jax = pytest.importorskip("jax")


def test_accelerator_component_is_deterministic():
    import hashlib

    import numpy as np
    from jax._src import cache_key

    from gpucosmo.pipeline import stable_cache_key

    assert stable_cache_key() and stable_cache_key()  # idempotent
    devs = np.array(jax.devices())
    h1, h2 = hashlib.sha256(), hashlib.sha256()
    cache_key._hash_accelerator_config(h1, devs)
    cache_key._hash_accelerator_config(h2, devs)
    assert h1.hexdigest() == h2.hexdigest()


@pytest.mark.slow
def test_one_cache_entry_across_processes(tmp_path):
    """Four processes compiling one program write one cache entry (jaxlib 0.11's GPU topology
    fingerprint alternates between two values per process, which gave two)."""
    script = textwrap.dedent(f"""
        import jax, jax.numpy as jnp
        jax.config.update("jax_compilation_cache_dir", {str(tmp_path)!r})
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
        from gpucosmo.pipeline import stable_cache_key
        assert stable_cache_key()
        jax.jit(lambda x: jnp.sin(x) @ x.T + 1.0)(jnp.ones((32, 32))).block_until_ready()
    """)
    env = {k: v for k, v in os.environ.items() if k != "GPUCOSMO_STABLE_CACHE_KEY"}
    for _ in range(4):
        subprocess.run([sys.executable, "-c", script], check=True, env=env, capture_output=True)
    assert len([p for p in tmp_path.iterdir() if p.name.startswith("jit__lambda")]) == 1
