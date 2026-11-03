#!/usr/bin/env python3
"""Forward-mode expansion coefficients: staged nested jvps and the implicit θ → H0 rule."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpucosmo.jax.inference.staged_fwd import staged_derivatives, staged_expansion
from gpucosmo.jax.inference.dali_expansion import multi_indices

jax.config.update("jax_enable_x64", True)

ELL = jnp.arange(6.0)


def _stages():
    """Three stages; the state carries a constant ℓ grid and an integer table, and the last
    stage takes √(ℓ(ℓ−1)) — singular derivative at ℓ = 0, 1 if the constant were differentiated."""
    s1 = lambda x: {"y": x[0] ** 2 * jnp.exp(x[1]) * jnp.cos(x[2]) * jnp.ones(6), "ell": ELL,  # noqa: E731
                    "idx": jnp.arange(6), "x": x}
    s2 = lambda st: {**st, "z": jnp.sin(st["y"] * st["x"][1]) + st["x"][2] ** 3}  # noqa: E731
    s3 = lambda st: {"out": st["z"] * jnp.sqrt(jnp.maximum(st["ell"] * (st["ell"] - 1.0), 0.0))  # noqa: E731
                     + st["idx"].astype(jnp.float64) * st["x"][0]}
    return [("s1", s1), ("s2", s2), ("s3", s3)]


def _composed(x):
    st = x
    for _, f in _stages():
        st = f(st)
    return st["out"]


@pytest.mark.parametrize("batch", [1, 2])
def test_staged_derivatives_match_jax_on_the_composition(batch):
    x0 = jnp.asarray([1.3, 0.2, -0.4])
    quiet = dict(log=lambda m: None, release=False)
    y, J = staged_derivatives(_stages(), x0, np.arange(3)[:, None], batch=batch, **quiet)
    _, H = staged_derivatives(_stages(), x0, multi_indices(3, 2)[2], batch=batch, **quiet)
    _, T = staged_derivatives(_stages(), x0, multi_indices(3, 3)[3], batch=batch, **quiet)
    np.testing.assert_allclose(y["out"], _composed(x0), rtol=1e-14)
    np.testing.assert_allclose(J["out"], jax.jacfwd(_composed)(x0).T, rtol=1e-12, atol=1e-14)
    H_ref = jnp.moveaxis(jax.jacfwd(jax.jacfwd(_composed))(x0), 0, -1)
    T_ref = jnp.moveaxis(jax.jacfwd(jax.jacfwd(jax.jacfwd(_composed)))(x0), 0, -1)
    for D, ref, idx in ((H, H_ref, multi_indices(3, 2)[2]), (T, T_ref, multi_indices(3, 3)[3])):
        assert np.all(np.isfinite(D["out"]))
        np.testing.assert_allclose(D["out"], ref[tuple(idx.T)], rtol=1e-11, atol=1e-11)


def test_staged_expansion_is_the_composition_s_polynomial():
    from gpucosmo.jax.inference.dali_expansion import DALIExpansion

    x0 = jnp.asarray([1.3, 0.2, -0.4])
    y0, coef = staged_expansion(_stages(), x0, order=3, batch=4, log=lambda m: None, release=False)
    t = DALIExpansion(x0, coef, order=3)
    np.testing.assert_allclose(t.y0["out"], _composed(x0), rtol=1e-14)
    # the error of the order-p polynomial falls as |d|^(p+1)
    errs = [float(jnp.max(jnp.abs(t(x0 + e)["out"] - _composed(x0 + e)))) for e in (1e-2, 5e-3)]
    assert 12 < errs[0] / errs[1] < 20


def test_theta_to_h0_higher_derivatives_are_exact():
    """The implicit-function custom_jvp rule differentiates again: H0's Hessian in
    (ω_b, ω_c, 100θ*) matches central differences of the converged solve (the earlier
    stop_gradient Newton step was exact only to first order)."""
    from gpucosmo.jax.background.cosmology import load_background_tables
    from gpucosmo.jax.cmb.theory import solve_h0_from_theta_mc

    t = load_background_tables()
    f = lambda p: solve_h0_from_theta_mc(t, p[0], p[1], -1.0, 0.0, p[2])  # noqa: E731
    p0 = np.asarray([0.02236, 0.1177, 1.04110])
    H = np.asarray(jax.jacfwd(jax.jacfwd(f))(jnp.asarray(p0)))
    hs = np.array([1e-5, 1e-4, 1e-5])
    fn = lambda p: float(f(jnp.asarray(p)))  # noqa: E731
    Hfd = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            ei, ej = np.eye(3)[i] * hs[i], np.eye(3)[j] * hs[j]
            Hfd[i, j] = (fn(p0 + ei + ej) - fn(p0 + ei - ej) - fn(p0 - ei + ej) + fn(p0 - ei - ej)) / (4 * hs[i] * hs[j])
    np.testing.assert_allclose(H, Hfd, rtol=2e-5, atol=1e-3)
    # and once more (the triplet DALI expansion's forward mode): third derivatives against
    # central differences of the Hessian
    T = np.asarray(jax.jacfwd(jax.jacfwd(jax.jacfwd(f)))(jnp.asarray(p0)))
    hf = jax.jit(jax.jacfwd(jax.jacfwd(f)))
    h3 = np.array([1e-4, 1e-3, 1e-4])
    Tfd = np.stack([(np.asarray(hf(p0 + np.eye(3)[i] * h3[i])) - np.asarray(hf(p0 - np.eye(3)[i] * h3[i])))
                    / (2 * h3[i]) for i in range(3)])
    assert np.max(np.abs(T - Tfd)) < 2e-3 * np.max(np.abs(T))
    g = jax.grad(lambda p: f(p) ** 2)(jnp.asarray(p0))  # reverse mode through the rule
    assert np.all(np.isfinite(np.asarray(g)))
