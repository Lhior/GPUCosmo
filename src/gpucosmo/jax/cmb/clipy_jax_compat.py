#!/usr/bin/env python3
"""Compatibility shim: enable clipy JAX backend on stacks that reject int64 arrays."""

from __future__ import annotations

import os
from functools import wraps

import numpy as np

_PATCHED = False


def _patch_smica_init() -> None:
    from clipy import smica

    if getattr(smica.smica_lkl.__init__, "_gpucosmo_patched", False):
        return

    _orig = smica.smica_lkl.__init__
    _jnp_array = smica.jnp.array

    def _jnp_array_int32(x, *args, **kwargs):
        if isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.integer):
            x = np.asarray(x, dtype=np.int32)
        return _jnp_array(x, *args, **kwargs)

    @wraps(_orig)
    def _patched(self, lkl, **options):
        smica.jnp.array = _jnp_array_int32
        try:
            return _orig(self, lkl, **options)
        finally:
            smica.jnp.array = _jnp_array

    _patched._gpucosmo_patched = True  # type: ignore[attr-defined]
    smica.smica_lkl.__init__ = _patched


def _patch_gibbs_searchsorted() -> None:
    """Replace gibbs int64 searchsorted counters with int32 for JAX GPU."""
    from clipy import gibbs

    if getattr(gibbs.gibbs_lkl._searchsorted, "_gpucosmo_patched", False):
        return

    import jax.numpy as jnp

    def _searchsorted(self, A, v):
        return jnp.sum(
            jnp.where(
                A < v,
                jnp.ones(len(A), dtype=jnp.int32),
                jnp.zeros(len(A), dtype=jnp.int32),
            )
        )

    _searchsorted._gpucosmo_patched = True  # type: ignore[attr-defined]
    gibbs.gibbs_lkl._searchsorted = _searchsorted


def _patch_gibbs_splint() -> None:
    """Vectorize ``gibbs_lkl.splint_gauss_and_deriv`` (Commander TT's per-ℓ cubic spline).

    clipy finds each multipole's bracket by counting the knots below it (a masked sum over
    the 1000 knots, one reduction per ℓ = 2–29) and reads the knots through (nl, nl) gathers
    and their diagonals; under vmap on the GPU that was 1.5 ms per 4096 points (~5 % of the
    compressed DALI likelihood). Here: a binary search per ℓ (same index: side="left" counts the
    knots < Cℓ), per-row gathers, the same spline arithmetic — 0.16 ms."""
    from clipy import gibbs

    if getattr(gibbs.gibbs_lkl.splint_gauss_and_deriv, "_gpucosmo_patched", False):
        return

    import jax
    import jax.numpy as jnp

    search = jax.vmap(lambda a, v: jnp.searchsorted(a, v, side="left", method="scan_unrolled"))

    def splint_gauss_and_deriv(self, cls):
        n = self.nbin
        x = self.cl2x[0]
        cnt = search(x, cls)  # number of knots < Cℓ, per ℓ
        klo = jnp.where(cnt > n - 2, n - 2, cnt)
        klo = jnp.where(klo < 1, 1, klo) - 1
        khi = klo + 1

        def at(a, k):
            return jnp.take_along_axis(a, k[:, None], axis=1)[:, 0]

        x_lo, x_hi = at(x, klo), at(x, khi)
        c1_lo, c1_hi = at(self.cl2x[1], klo), at(self.cl2x[1], khi)
        c2_lo, c2_hi = at(self.cl2x[2], klo), at(self.cl2x[2], khi)
        h = x_hi - x_lo
        a = (x_hi - cls) / h
        b = (cls - x_lo) / h
        six = jnp.asarray(6.0, dtype=h.dtype)
        gauss = a * c1_lo + b * c1_hi + ((a**3 - a) * c2_lo + (b**3 - b) * c2_hi) * (h**2) / six
        deriv = ((c1_hi - c1_lo) / h - (3.0 * a**2 - 1.0) / six * h * c2_lo
                 + (3.0 * b**2 - 1.0) / six * h * c2_hi)
        return gauss, deriv

    splint_gauss_and_deriv._gpucosmo_patched = True  # type: ignore[attr-defined]
    splint_gauss_and_deriv._clipy_original = gibbs.gibbs_lkl.splint_gauss_and_deriv  # type: ignore[attr-defined]
    gibbs.gibbs_lkl.splint_gauss_and_deriv = splint_gauss_and_deriv


def sanitize_clipy_templates(like) -> int:
    """Zero the non-finite entries of a loaded clipy likelihood's component templates; returns how
    many. plik's correlated-noise ("cnoise") templates are normalized by their value at a pivot ℓ,
    which is 0 for frequency pairs a file does not use (e.g. the TT-only plik), leaving 0/0
    entries: they drop out of log L but turn every gradient through a multiplicative component
    (calibrations, A_planck) into NaN. Same value, finite gradients. Clears JAX's caches (the
    components' jitted methods take ``self`` as a static argument)."""
    import jax
    import jax.numpy as jnp

    n, seen = 0, set()

    def walk(o, depth=0):
        nonlocal n
        if id(o) in seen or depth > 8:
            return
        seen.add(id(o))
        for name in ("cmp", "_internal", "lkl", "_lkl", "lkls", "_lkls", "smica"):
            x = getattr(o, name, None)
            if x is None:
                continue
            for y in (x if isinstance(x, (list, tuple)) else [x]):
                t = getattr(y, "template", None)
                if t is not None:
                    bad = ~np.isfinite(np.asarray(t))
                    if bad.any():
                        n += int(bad.sum())
                        y.template = jnp.where(jnp.isfinite(t), t, 0.0)
                walk(y, depth + 1)

    walk(like)
    if n:
        jax.clear_caches()
    return n


def ensure_clipy_jax() -> None:
    """
    Enable clipy JAX mode and apply the int64 compatibility patches and the vectorized
    Commander TT spline (``_patch_gibbs_splint``).

    Must be called before ``clipy.clik_candl(...)``. Idempotent.
    """
    global _PATCHED
    os.environ.pop("CLIPY_NOJAX", None)

    if not _PATCHED:
        import clipy.gibbs  # noqa: F401
        import clipy.smica  # noqa: F401

        _patch_smica_init()
        _patch_gibbs_searchsorted()
        _patch_gibbs_splint()
        _PATCHED = True

    import clipy

    if not getattr(clipy, "hasjax", False):
        raise RuntimeError(
            "clipy JAX backend unavailable (hasjax=False). "
            "Ensure jax is installed and CLIPY_NOJAX is unset."
        )


__all__ = ["ensure_clipy_jax"]
