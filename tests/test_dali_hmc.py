#!/usr/bin/env python3
"""DALI-trajectory HMC: with ``accept="exact"`` it targets the exact posterior whatever the DALI
expansion; by default it accepts on (and targets) the DALI posterior without an exact call."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


def test_exact_accept_samples_exact_target_with_a_wrong_dali():
    from gpucosmo.jax.mcmc.dali_hmc import run_dali_hmc

    mu = jnp.asarray([1.0, -2.0])
    cov = jnp.asarray([[1.0, 0.8], [0.8, 2.0]])
    prec = jnp.linalg.inv(cov)

    def exact(q):
        d = q - mu
        return -0.5 * d @ prec @ d

    # Deliberately wrong DALI: shifted mean, wrong correlation and scale.
    mu_s = mu + jnp.asarray([0.4, -0.3])
    prec_s = jnp.linalg.inv(jnp.asarray([[1.5, 0.2], [0.2, 1.2]]))

    def dali(q):
        d = q - mu_s
        return -0.5 * d @ prec_s @ d

    res = run_dali_hmc(
        dali_logpost=dali,
        exact_logpost_batch=jax.jit(jax.vmap(exact)),
        accept="exact",
        x_init=np.zeros((16, 2)),
        inv_mass=np.eye(2),
        n_iter=1500,
        n_leapfrog=10,
        n_warmup=300,
        step_size=0.3,
        seed=3,
        save_every=10_000,
    )
    post = res["samples"][300:].reshape(-1, 2)
    acc = res["accept"][300:].mean()
    assert 0.2 < acc < 0.99
    np.testing.assert_allclose(post.mean(0), np.asarray(mu), atol=0.08)
    np.testing.assert_allclose(np.cov(post.T), np.asarray(cov), atol=0.15)
    # And clearly not the DALI expansion's distribution.
    assert abs(post.mean(0)[0] - float(mu_s[0])) > 0.25


def test_mass_adaptation_recovers_scale_and_keeps_eps_sane():
    from gpucosmo.jax.mcmc.dali_hmc import run_dali_hmc

    scales = jnp.asarray([0.01, 1.0, 30.0])

    def target(q):
        return -0.5 * jnp.sum((q / scales) ** 2)

    res = run_dali_hmc(
        dali_logpost=target,
        exact_logpost_batch=jax.jit(jax.vmap(target)),
        accept="exact",
        x_init=np.zeros((8, 3)) + 1e-3,
        inv_mass=np.eye(3),
        n_iter=700,
        n_leapfrog=10,
        n_warmup=300,
        adapt_mass_at=(100, 200),
        seed=1,
        save_every=10_000,
    )
    post = res["samples"][300:].reshape(-1, 3)
    np.testing.assert_allclose(post.std(0), np.asarray(scales), rtol=0.25)
    assert res["eps"][-1] > 0.2  # did not collapse onto the floor


def test_resumed_segments_continue_the_same_chain():
    from gpucosmo.jax.mcmc.dali_hmc import run_dali_hmc

    prec = jnp.asarray([[2.0, 0.5], [0.5, 1.0]])

    def target(q):
        return -0.5 * q @ prec @ q

    def dali(q):
        return -0.5 * q @ (1.1 * prec) @ q

    kw = dict(dali_logpost=dali, exact_logpost_batch=jax.jit(jax.vmap(target)), accept="exact",
              n_leapfrog=8, seed=5, save_every=10_000)
    x0 = np.full((4, 2), 0.3)
    full = run_dali_hmc(x_init=x0, inv_mass=np.eye(2), n_iter=120, n_warmup=60,
                             adapt_mass_at=(30,), **kw)
    seg1 = run_dali_hmc(x_init=x0, inv_mass=np.eye(2), n_iter=80, n_warmup=60,
                             adapt_mass_at=(30,), **kw)
    seg2 = run_dali_hmc(x_init=seg1["x_last"], inv_mass=seg1["inv_mass"],
                             step_size=seg1["eps_final"], n_iter=40, n_warmup=0,
                             rng_state=seg1["rng_state"], **kw)
    joined = np.concatenate([seg1["samples"], seg2["samples"]])
    np.testing.assert_allclose(joined, full["samples"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(seg2["eps"], full["eps"][80:])


def test_box_transform_kinds_roundtrip_and_jacobian():
    import sys
    import types
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_dali_hmc import AFFINE, LOGIT, LOWER, UPPER, box_transform

    b = types.SimpleNamespace(prior_lo=np.array([0.0, 0.0, -3.0, 0.0]),
                              prior_hi=np.array([10.0, 1.0, 1.0, 200.0]))
    spec = dict(kind=np.array([LOWER, UPPER, AFFINE, LOGIT]),
                c=np.array([0.0, 0.0, -0.8, 0.0]), s=np.array([0.6, 0.2, 0.06, 1.0]))
    to_x, to_z, log_jac = box_transform(b, spec)
    x = jnp.asarray([0.3, 0.85, -0.83, 40.0])
    z = to_z(x)
    np.testing.assert_allclose(to_x(z), x, rtol=1e-10)
    jac = jax.jacfwd(to_x)(z)
    np.testing.assert_allclose(log_jac(z), np.log(np.abs(np.diag(jac))).sum(), rtol=1e-10)
    assert np.allclose(jac - np.diag(np.diag(jac)), 0)


def test_dali_args_path_samples_exact_target():
    """DALI data passed as a jit argument (``dali_args``) gives the same
    exact sampler as the closure form (2-D Gaussian with a wrong DALI)."""
    import jax.numpy as jnp
    import numpy as np

    from gpucosmo.jax.mcmc.dali_hmc import run_dali_hmc

    cov = np.array([[1.0, 0.6], [0.6, 2.0]])
    icov = jnp.asarray(np.linalg.inv(cov))

    def exact(x):
        return -0.5 * jnp.einsum("ni,ij,nj->n", x, icov, x)

    def dali(q, a):  # deliberately wrong scale, data passed as an argument
        return -0.5 * a["s"] * q @ (a["icov"] @ q)

    res = run_dali_hmc(dali_logpost=dali, dali_args={"s": 0.7, "icov": icov},
                            exact_logpost_batch=exact, accept="exact", x_init=np.zeros((8, 2)),
                            inv_mass=cov, n_iter=1200, n_warmup=200, seed=1)
    s = res["samples"][200:].reshape(-1, 2)
    np.testing.assert_allclose(np.cov(s.T), cov, rtol=0.25, atol=0.15)


def test_default_dali_accept_samples_the_dali_target_without_exact_calls():
    """Default ``accept="dali"``: the chain targets the DALI posterior, no exact call is made
    (``exact_logpost_batch`` absent), and a separate accept target (``dali_target_logpost``: e.g.
    the official likelihood where the trajectory uses a smoothed one) is the one sampled."""
    from gpucosmo.jax.mcmc.dali_hmc import run_dali_hmc

    mu_t = np.array([0.5, -1.0])
    cov_t = np.array([[1.0, 0.5], [0.5, 1.5]])
    icov_t = jnp.asarray(np.linalg.inv(cov_t))

    def target(q, a):  # the DALI posterior the chain should sample
        d = q - a["mu"]
        return -0.5 * d @ (a["icov"] @ d)

    def smoothed(q, a):  # the trajectory's potential: a slightly different one
        d = q - a["mu"] - 0.2
        return -0.4 * d @ (a["icov"] @ d)

    args = {"mu": jnp.asarray(mu_t), "icov": icov_t}
    res = run_dali_hmc(dali_logpost=smoothed, dali_target_logpost=target, dali_args=args,
                       x_init=np.zeros((16, 2)), inv_mass=cov_t, n_iter=1500, n_warmup=300, n_leapfrog=10,
                       seed=2, save_every=10_000)
    assert res["accept_mode"] == "dali"
    post = res["samples"][300:].reshape(-1, 2)
    np.testing.assert_allclose(post.mean(0), mu_t, atol=0.08)
    np.testing.assert_allclose(np.cov(post.T), cov_t, atol=0.15)
    # without a separate target the trajectory's own potential is sampled
    res2 = run_dali_hmc(dali_logpost=smoothed, dali_args=args, x_init=np.zeros((16, 2)), inv_mass=cov_t,
                        n_iter=1500, n_warmup=300, n_leapfrog=10, seed=2, save_every=10_000)
    np.testing.assert_allclose(res2["samples"][300:].reshape(-1, 2).mean(0), mu_t + 0.2, atol=0.08)
    with pytest.raises(ValueError, match="needs exact_logpost_batch"):
        run_dali_hmc(dali_logpost=smoothed, dali_args=args, accept="exact", x_init=np.zeros((2, 2)),
                     inv_mass=cov_t, n_iter=2)
