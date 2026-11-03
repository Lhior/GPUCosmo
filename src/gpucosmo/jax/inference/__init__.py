"""Inference bundles: the analyses of a Cobaya yaml (``mg_bundle``) and their DALI expansions."""

import importlib

# Public names, imported on first access (PEP 562): importing a submodule does not
# pull in the others (and their optional dependencies).
_LAZY = {
    "LikelihoodBundle": "gpucosmo.jax.inference.bundle",
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
