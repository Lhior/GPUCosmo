#!/usr/bin/env python3
"""Helpers behind the jax_eb speed-ups: they must not change any values."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)


def test_vec_builder_matches_at_set_chain():
    from gpucosmo.jax.eb.vec_builder import VecBuilder

    rng = np.random.default_rng(0)
    base = jnp.asarray(rng.normal(size=12))
    ops = [(3, 1.5), (slice(5, 9), rng.normal(size=4)), (slice(0, 2), 7.0),
           (6, -2.0), (slice(8, 12), rng.normal(size=4)), (11, 0.25)]
    for start in (None, base):
        ref = jnp.zeros(12) if start is None else base
        b = VecBuilder(12, base=start)
        for idx, val in ops:
            ref = ref.at[idx].set(val)
            b = b.at[idx].set(val)
        assert np.array_equal(np.asarray(b.build()), np.asarray(ref))
        assert float(b[6]) == float(ref[6])
        assert np.array_equal(np.asarray(b[4:10]), np.asarray(ref[4:10]))


def test_interp_cubic_uniform_matches_interp_cubic():
    from gpucosmo.jax.eb.interp_jax import cubic_pack_uniform, interp_cubic, interp_cubic_uniform

    x = jnp.log(1e-3 * jnp.exp(jnp.arange(2001) * (np.log(1.4e4 / 1e-3) / 2000)))
    ys = jnp.stack([jnp.sin(x), jnp.exp(-0.1 * x), x**3], axis=1)
    pack = cubic_pack_uniform(x, ys)
    xq = jnp.asarray(np.random.default_rng(1).uniform(float(x[0]) - 1, float(x[-1]) + 1, 500))
    got = interp_cubic_uniform(xq, pack)
    for j in range(3):
        ref = interp_cubic(xq, x, ys[:, j])
        assert float(jnp.max(jnp.abs(got[:, j] - ref))) <= 1e-12 * float(jnp.max(jnp.abs(ref)))


def test_on_host_inlines_on_cpu_and_vmaps():
    from gpucosmo.jax.eb.host_offload import on_host

    def fn(a, b):
        return {"s": a + b, "v": jnp.arange(3.0) * a * b}

    g = on_host(fn)
    out = jax.jit(jax.vmap(g))(jnp.asarray([1.0, 2.0]), jnp.asarray([3.0, 4.0]))
    assert np.allclose(out["s"], [4.0, 6.0])
    assert np.allclose(out["v"], [[0, 3, 6], [0, 8, 16]])


def test_natural_cubic_d2_matches_scipy():
    from scipy.interpolate import CubicSpline

    from gpucosmo.jax.eb.sources_jax import _natural_cubic_d2

    x = np.geomspace(1e-3, 1e4, 3000)
    y = np.exp(-x / 300.0) / (1.0 + x)
    got = np.asarray(_natural_cubic_d2(jnp.log(x), y))
    ref = CubicSpline(np.log(x), y, bc_type="natural")(np.log(x), 2)
    assert np.max(np.abs(got - ref)) < 1e-10 * np.max(np.abs(ref))
