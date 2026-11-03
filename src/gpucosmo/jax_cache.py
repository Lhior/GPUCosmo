"""Persistent XLA compilation cache, shared by the pipelines, the command-line tools and the
tests (``tests/conftest.py``). Importing this module has no side effects (``gpucosmo.pipeline``
switches on float64)."""

from __future__ import annotations


def enable_jax_cache(min_compile_time_secs: float = 2.0) -> None:
    """Persistent XLA compilation cache (``gpucosmo.paths.jax_cache_dir``): relaunches skip the
    multi-minute theory compiles; programs taking ≥ ``min_compile_time_secs`` to compile are
    kept. The cache keys get a deterministic accelerator component (``stable_cache_key``)."""
    import jax

    from gpucosmo.paths import jax_cache_dir

    jax.config.update("jax_compilation_cache_dir", str(jax_cache_dir()))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", min_compile_time_secs)
    stable_cache_key()


def stable_cache_key() -> bool:
    """Make the persistent-cache key's accelerator component deterministic.

    JAX hashes ``xla_client.get_topology_for_devices(devices).fingerprint()`` into every key; with
    jaxlib 0.11 on CUDA that fingerprint takes one of two values at random per process (the same
    RTX 3070, the same environment; 6 + 6 of 12 processes), so a relaunch found its programs with
    probability ~1/2 (a μ–Σ evidence run recompiled ~5 min of programs, 2026-10-18). The topology
    is replaced by the devices' platform, kind, compute capability, core count and number, with
    the platform version as before; jaxlib/backend versions and XLA flags stay in the key.
    ``GPUCOSMO_STABLE_CACHE_KEY=0`` keeps JAX's key. Returns whether the patch is active.
    """
    import os

    if os.environ.get("GPUCOSMO_STABLE_CACHE_KEY", "1") == "0":
        return False
    try:
        from jax._src import cache_key as _ck
    except ImportError:  # pragma: no cover - JAX internals moved
        return False
    if getattr(_ck, "_gpucosmo_stable", False) or not hasattr(_ck, "_hash_accelerator_config"):
        return getattr(_ck, "_gpucosmo_stable", False)

    def _hash_accelerator_config(hash_obj, accelerators):
        parts = [f"{d.platform}|{getattr(d, 'device_kind', '')}|{getattr(d, 'compute_capability', '')}|"
                 f"{getattr(d, 'core_count', '')}" for d in accelerators.flat]
        try:
            parts.append(str(next(iter(accelerators.flat)).client.platform_version))
        except Exception:  # noqa: BLE001
            pass
        hash_obj.update(("gpucosmo-accelerators:" + ";".join(parts)).encode())

    _ck._hash_accelerator_config = _hash_accelerator_config
    _ck._gpucosmo_stable = True
    return True
