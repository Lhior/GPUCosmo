"""JAX CMB theory + clipy/candl likelihoods for desi_full inference."""

import importlib

# Public names, imported on first access (PEP 562): importing a submodule does not
# pull in the others (and their optional dependencies).
_LAZY = {
    "build_desi_full_gpu_bundle": "gpucosmo.jax.cmb.bundle",
}


def __getattr__(name):
    if name in _LAZY:
        value = getattr(importlib.import_module(_LAZY[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY))


__all__ = sorted(_LAZY)
