#!/usr/bin/env python3
"""Host-only check: on-device Miller spherical Bessel vs SciPy."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy
jax.config.update("jax_enable_x64", True)

from gpucosmo.jax.eb.los_jax import (  # noqa: E402
    spherical_bessel_jl_scan,
    spherical_bessel_jn,
)


@pytest.mark.parametrize(
    "ell_max,x_vals",
    [
        (5, [0.5, 1.0, 3.0, 10.0]),
        (20, [1.0, 5.0, 15.0, 50.0]),
        (50, [10.0, 30.0, 100.0]),
        (100, [20.0, 80.0, 200.0]),
        (40, [0.1, 0.5, 2.0, 10.0]),
    ],
)
def test_spherical_bessel_vs_scipy(ell_max, x_vals):
    from scipy.special import spherical_jn

    x_np = np.asarray(x_vals, dtype=np.float64)
    jl = np.asarray(spherical_bessel_jl_scan(ell_max, jnp.asarray(x_np)))
    for ell in range(ell_max + 1):
        ref = spherical_jn(ell, x_np)
        denom = np.maximum(np.abs(ref), 1e-12)
        rel = np.abs(jl[ell] - ref) / denom
        tol = np.where(x_np < 0.5, 0.05, 3e-2)
        near_turn = np.abs(x_np - float(ell)) < max(15.0, 0.5 * float(ell))
        tol = np.where(near_turn, np.maximum(tol, 0.05), tol)
        assert np.all(rel < tol), f"ell={ell} rel={rel} x={x_vals}"


def test_spherical_bessel_jn_suppressed():
    """n ≫ x must be tiny (not clipped to ~1)."""
    from scipy.special import spherical_jn

    x = jnp.asarray([1.0, 5.0, 20.0], dtype=jnp.float64)
    for n in (50, 100, 200):
        dev = np.asarray(spherical_bessel_jn(n, x))
        ref = spherical_jn(n, np.asarray(x))
        assert np.max(np.abs(dev)) < 1e-6 or np.allclose(dev, ref, rtol=1e-4, atol=1e-30)
        # Must not sit at the clip boundary
        assert np.max(np.abs(dev)) < 0.99 or np.max(np.abs(ref)) > 0.5


def test_spherical_bessel_jit():
    fn = jax.jit(spherical_bessel_jn, static_argnums=(0,))
    x = jnp.linspace(0.1, 80.0, 32)
    out = fn(40, x)
    assert out.shape == (32,)
    assert np.isfinite(np.asarray(out)).all()



@pytest.mark.parametrize("n", [500, 2100, 2200, 2300, 2500, 3000, 3500])
def test_spherical_bessel_jn_high_n_peak(n):
    """Issue #10: the unscaled Miller recurrence overflowed for n ≳ 2200 and j_n came back 0
    around the first peak (error 0.6 of the peak). Now ≲ 1e-12 of the peak over the turning
    point, the first peak and its oscillations."""
    from scipy.special import spherical_jn

    x = np.linspace(n - 3 * n ** (1 / 3), n + 40 * n ** (1 / 3), 300)
    ref = spherical_jn(n, x)
    peak = np.abs(ref).max()
    dev = np.asarray(jax.jit(spherical_bessel_jn, static_argnums=(0,))(n, jnp.asarray(x)))
    assert np.isfinite(dev).all()
    assert np.max(np.abs(dev - ref)) < 1e-12 * peak


@pytest.mark.parametrize("n", [2, 3, 10, 200])
def test_spherical_bessel_jn_small_x_series(n):
    """Issue #10: below x = 0.5 only the leading series term was kept (1.3e-4 of the peak off at
    n = 3); the full series is now summed."""
    from scipy.special import spherical_jn

    x = np.concatenate([np.geomspace(1e-6, 0.499, 60), [0.5, 0.51]])
    ref = spherical_jn(n, x)
    dev = np.asarray(spherical_bessel_jn(n, jnp.asarray(x)))
    tiny = np.abs(ref) < 1e-290  # underflow
    np.testing.assert_allclose(dev[~tiny], ref[~tiny], rtol=1e-12, atol=0)
