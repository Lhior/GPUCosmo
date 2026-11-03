"""Disk cache for static tables built on the host (NumPy / SciPy) while a program traces.

The exact theory rebuilds some cosmology-independent tables in every process: the
lensing Gauss–Legendre nodes (``numpy.polynomial.legendre.leggauss``, an ``eigvalsh`` of
a ~4600² matrix) with their Wigner-d stacks, and the spherical-Bessel spline coefficients
(SciPy ``spherical_jn`` + ``CubicSpline`` for ~450 ℓ). That is ~40 s of every pipeline
stage's start-up, which the JAX persistent compilation cache cannot remove (it keys on
the traced program, so tracing must run first). The tables are deterministic in their
arguments, so they are stored as ``.npz`` files under ``data/cache/`` (gitignored).

The key covers the build arguments, the source of the module that builds the table (any
edit to it invalidates its entries) and the NumPy / SciPy versions.
``GPUCOSMO_TABLE_CACHE=0`` disables the cache; ``GPUCOSMO_TABLE_CACHE=<dir>`` moves it.
"""

from __future__ import annotations

import functools
import hashlib
import os
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

def _cache_dir() -> Path | None:
    """``GPUCOSMO_TABLE_CACHE`` (``0`` disables), default ``<data dir>/cache`` (``gpucosmo.paths``)."""
    from gpucosmo.paths import table_cache_dir

    return table_cache_dir()


@functools.lru_cache(maxsize=None)
def _module_digest(module: str) -> str:
    path = getattr(sys.modules.get(module), "__file__", None)
    data = Path(path).read_bytes() if path else module.encode()
    return hashlib.sha256(data).hexdigest()[:16]


def cached_npz(name: str, key: tuple, build: Callable[[], dict], *, module: str) -> dict:
    """``build()`` (a dict of arrays and Python scalars), loaded from / saved to disk.

    0-d entries come back as Python scalars, so ints used as static shapes stay ints.
    """
    root = _cache_dir()
    if root is None:
        return build()
    import scipy

    ident = repr((name, key, _module_digest(module), np.__version__, scipy.__version__))
    path = root / f"{name}_{hashlib.sha256(ident.encode()).hexdigest()[:20]}.npz"
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as z:
                return {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}
        except (OSError, ValueError, EOFError):
            pass  # unreadable entry: rebuild and overwrite
    out = build()
    try:
        root.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp.npz")
        np.savez(tmp, **{k: np.asarray(v) for k, v in out.items()})
        os.replace(tmp, path)
    except OSError:
        pass  # read-only checkout: the table is still returned
    return out


__all__ = ["cached_npz"]
