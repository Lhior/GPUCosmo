#!/usr/bin/env python3
"""clipy JAX patches: the vectorized Commander TT spline equals clipy's own."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
gibbs = pytest.importorskip("clipy.gibbs")


def test_vectorized_gibbs_spline_matches_clipy():
    from gpucosmo.jax.cmb.clipy_jax_compat import ensure_clipy_jax

    ensure_clipy_jax()
    patched = gibbs.gibbs_lkl.splint_gauss_and_deriv
    original = patched._clipy_original
    rng = np.random.default_rng(1)
    nl, nbin = 28, 1000
    obj = gibbs.gibbs_lkl.__new__(gibbs.gibbs_lkl)  # only nbin, cl2x and _searchsorted are read
    obj.nbin = nbin
    knots = np.sort(rng.uniform(0.0, 5000.0, (nl, nbin)), axis=1)
    obj.cl2x = jnp.asarray(np.stack([knots, rng.normal(size=(nl, nbin)), rng.normal(size=(nl, nbin))]))
    for cls in (rng.uniform(100.0, 4900.0, nl),  # interior
                np.concatenate([[-1.0, 1e5], rng.uniform(0.0, 5000.0, nl - 2)])):  # below / above every knot
        cls = jnp.asarray(cls)
        g0, d0 = original(obj, cls)
        g1, d1 = jax.jit(lambda c: patched(obj, c))(cls)
        np.testing.assert_allclose(g1, g0, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(d1, d0, rtol=1e-14, atol=1e-14)
        j0 = jax.jacfwd(lambda c: original(obj, c)[0])(cls)
        j1 = jax.jacfwd(lambda c: patched(obj, c)[0])(cls)
        np.testing.assert_allclose(j1, j0, rtol=1e-12, atol=1e-14)
