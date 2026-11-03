"""JAX w0waCDM background cosmology and background-level likelihoods."""

import importlib

# Public names, imported on first access (PEP 562): importing a submodule does not
# pull in the others (and their optional dependencies).
_LAZY = {
    "BackgroundParams": "gpucosmo.jax.background.cosmology",
    "BackgroundTables": "gpucosmo.jax.background.cosmology",
    "load_background_tables": "gpucosmo.jax.background.cosmology",
    "W0WaLikelihoodData": "gpucosmo.jax.background.likelihoods",
    "build_w0wa_bundle": "gpucosmo.jax.background.likelihoods",
    "load_w0wa_data": "gpucosmo.jax.background.likelihoods",
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
