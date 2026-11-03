"""Centred float32 χ² of the large Cobaya-port inverse covariances (CamSpec): algebraically exact,
float32 only in the small quadratic term about the reference residual."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def test_centred_chi2_matches_cobaya_semantics():
    from gpucosmo.likelihoods.cobaya.planck_python import _Chi2

    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(0)
    n = 1600  # > 1500: Cobaya's lower-triangle (dsymv) path
    A = rng.normal(size=(n, n)) / np.sqrt(n)
    C = A @ A.T + 0.05 * np.eye(n)
    Ci = np.linalg.inv(C) + 1e-9 * rng.normal(size=(n, n))  # cached inverses are not exactly symmetric
    L = np.linalg.cholesky(C)
    d0 = L @ rng.normal(size=n)
    ref = lambda d: 2 * d @ np.tril(Ci) @ d - np.sum(np.diag(Ci) * d * d)  # noqa: E731
    ch = _Chi2(Ci)
    ch.set_reference(d0)
    for s in (0.0, 0.05, 0.3):
        d = d0 + L @ (s * np.random.default_rng(int(100 * s) + 1).normal(size=n))
        assert abs(float(ch(jnp.asarray(d))) - ref(d)) < 1e-4
    D = jnp.asarray(d0[None] + 0.1 * (L @ rng.normal(size=(n, 4))).T)
    assert float(jnp.max(jnp.abs(jax.jit(jax.vmap(ch))(D) - jnp.asarray([ref(np.asarray(x)) for x in D])))) < 1e-4
