"""Registry of likelihood classes, so Cobaya yamls (and ``--plugin`` modules) can name them.

A yaml ``likelihood:`` key selects a class by, in order:

1. a registered name or alias (``@register_likelihood``, or a ``gpucosmo.likelihoods`` entry
   point of an installed package);
2. the block's ``class:`` option, or the key itself, as an import path
   (``my_package.module.MyLikelihood`` or ``my_package.module:MyLikelihood``).

Entry points (in the plugin package's ``pyproject.toml``)::

    [project.entry-points."gpucosmo.likelihoods"]
    my_bao = "my_package.bao:MyBAO"
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

from gpucosmo.likelihoods.base import Likelihood

ENTRY_POINT_GROUP = "gpucosmo.likelihoods"
_REGISTRY: dict[str, type] = {}
_ENTRY_POINTS_LOADED = False

# Yaml keys handled by the built-in likelihoods (``inference.cobaya_yaml``); listed by
# ``available_likelihoods`` and reserved as names.
BUILTIN = {
    "desi_fs": "DESI DR1 full-shape + BAO (desi_fs_bao_all)",
    "plik": "Planck 2018 high-ℓ plik TTTEEE (clipy)",
    "hillipop": "Planck PR4 HiLLiPoP TTTEEE",
    "lollipop": "Planck PR4 LoLLiPoP low-ℓ EE",
    "lowl_tt": "Planck 2018 Commander low-ℓ TT",
    "lowl_ee": "Planck 2018 SimAll low-ℓ EE",
    "act_lens": "ACT DR6 (+ Planck PR4) CMB lensing",
    "sn": "DES-Y5 / Pantheon+ / Union3 supernovae",
    "bbn": "BBN prior on ω_b (Schöneberg 2024)",
    "ns10": "n_s prior (10× Planck σ)",
}


def register_likelihood(cls: type | None = None, *, name: str | None = None, aliases: tuple[str, ...] = ()):
    """Class decorator: make a ``Likelihood`` subclass selectable by name (its ``name``
    attribute, else the class name), its ``aliases`` and its import path::

        @register_likelihood
        class MyBAO(GaussianLikelihood):
            name = "my_bao"
            ...
    """
    def deco(c):
        if not (isinstance(c, type) and issubclass(c, Likelihood)):
            raise TypeError(f"register_likelihood: {c!r} is not a gpucosmo.likelihoods.Likelihood subclass")
        key = name or c.name or c.__name__
        if key in BUILTIN:
            raise ValueError(f"likelihood name {key!r} is reserved for a built-in likelihood")
        keys = {key, f"{c.__module__}.{c.__qualname__}", *c.aliases, *aliases}
        for k in keys:
            old = _REGISTRY.get(k)
            if old is not None and old is not c and f"{old.__module__}.{old.__qualname__}" != f"{c.__module__}.{c.__qualname__}":
                raise ValueError(f"likelihood name {k!r} already registered by {old.__module__}.{old.__qualname__}")
            _REGISTRY[k] = c
        if name and not c.name:
            c.name = name
        return c

    return deco(cls) if cls is not None else deco


def load_entry_points(force: bool = False) -> None:
    """Register the likelihoods installed packages declare under the ``gpucosmo.likelihoods``
    entry-point group (a class, or a module that registers its classes on import)."""
    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED and not force:
        return
    _ENTRY_POINTS_LOADED = True
    from importlib.metadata import entry_points

    for ep in entry_points(group=ENTRY_POINT_GROUP):
        try:
            obj = ep.load()
        except Exception as e:  # noqa: BLE001 - one broken plug-in must not hide the others
            import warnings

            warnings.warn(f"gpucosmo.likelihoods entry point {ep.name!r} ({ep.value}) failed to load: {e!r}",
                          stacklevel=2)
            continue
        if isinstance(obj, type) and issubclass(obj, Likelihood):
            register_likelihood(obj, name=None if obj.name else ep.name,
                                aliases=(ep.name,) if obj.name and obj.name != ep.name else ())


def load_plugin_module(spec: str):
    """Import a module by dotted name or file path (``path/to/my_likelihoods.py``); its
    ``@register_likelihood`` classes become available."""
    if spec.endswith(".py") or "/" in spec:
        path = Path(spec).expanduser().resolve()
        mod_name = f"gpucosmo_plugin_{path.stem}"
        if mod_name in sys.modules:
            return sys.modules[mod_name]
        s = importlib.util.spec_from_file_location(mod_name, path)
        if s is None or s.loader is None:
            raise ImportError(f"cannot import plugin file {path}")
        mod = importlib.util.module_from_spec(s)
        sys.modules[mod_name] = mod
        s.loader.exec_module(mod)
        return mod
    return importlib.import_module(spec)


def _import_path(path: str):
    mod, sep, attr = path.partition(":")
    if not sep:
        mod, _, attr = path.rpartition(".")
    if not mod:
        return None
    try:
        obj = getattr(importlib.import_module(mod), attr)
    except (ImportError, AttributeError):
        return None
    return obj if isinstance(obj, type) and issubclass(obj, Likelihood) else None


def get_likelihood_class(key: str, block: dict | None = None) -> type | None:
    """The ``Likelihood`` class a yaml key (and its block's ``class:`` option) selects, or None."""
    import gpucosmo.likelihoods.examples  # noqa: F401  (registers the example likelihoods)
    from gpucosmo.likelihoods.cobaya import register_ports

    register_ports()  # native ports of Cobaya likelihoods, under their Cobaya names

    block = block or {}
    for k in ([block["class"]] if block.get("class") else []) + [key]:
        if k in _REGISTRY:
            return _REGISTRY[k]
    load_entry_points()
    for k in ([block["class"]] if block.get("class") else []) + [key]:
        if k in _REGISTRY:
            return _REGISTRY[k]
        cls = _import_path(k)
        if cls is not None:
            return cls
    return None


def available_likelihoods(builtin: bool = True) -> dict[str, str]:
    """{name: description} of the built-in likelihoods and every registered class."""
    import gpucosmo.likelihoods.examples  # noqa: F401  (registers the example likelihoods)

    load_entry_points()
    out = {}
    if builtin:
        from gpucosmo.jax.inference.cobaya_yaml import _LIKES

        for k, v in BUILTIN.items():
            keys = [key for comp, pats in _LIKES.items() if comp == k or comp.startswith(f"{k}:") for key in pats[:1]]
            out[k] = f"[built-in] {v}" + (f" (yaml: {', '.join(keys)})" if keys else "")
    from gpucosmo.likelihoods.cobaya import register_ports

    for n, c in sorted(register_ports().items()):
        out[n] = f"[Cobaya port] {c.__module__}.{c.__qualname__}"
    seen = set(register_ports().values())
    for k, c in sorted(_REGISTRY.items()):
        if c in seen or "." in k:
            continue
        seen.add(c)
        doc = (c.__doc__ or "").strip().splitlines()
        out[c.name or k] = f"{c.__module__}.{c.__qualname__}" + (f" — {doc[0]}" if doc else "")
    return out


def main(argv: list[str] | None = None) -> int:
    """``gpucosmo-likelihoods``: list the likelihoods a Cobaya yaml can name."""
    import argparse

    ap = argparse.ArgumentParser(prog="gpucosmo-likelihoods", description=main.__doc__)
    ap.add_argument("--plugin", action="append", default=[], metavar="MODULE",
                    help="import MODULE first (dotted name or path/to/file.py)")
    args = ap.parse_args(argv)
    for m in args.plugin:
        load_plugin_module(m)
    for k, v in available_likelihoods().items():
        print(f"{k:22s} {v}")
    return 0


__all__ = ["BUILTIN", "ENTRY_POINT_GROUP", "available_likelihoods", "get_likelihood_class", "load_entry_points",
           "load_plugin_module", "register_likelihood"]


if __name__ == "__main__":
    raise SystemExit(main())
