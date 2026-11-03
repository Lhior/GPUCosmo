#!/usr/bin/env python3
"""DALI evidence (tempered SMC) and the DALI order check on a toy with a known evidence."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from scipy.special import logsumexp  # noqa: E402
from scipy.stats import truncnorm  # noqa: E402

from gpucosmo.jax.mcmc.dali_ns import (  # noqa: E402
    dali_order_check,
    make_batched_dali_loglike,
    make_dali_loglike,
    sample_prior,
)
from gpucosmo.jax.mcmc.dali_smc import DALISMCConfig, run_dali_smc  # noqa: E402

LO, HI = -5.0, 5.0
MEAN = np.array([0.5, 0.5, 0.2, -0.2])
SIG = 0.3
BIAS = 0.1  # DALI-expanded theory offset


def _toy_bundle():
    lo = jnp.full(4, LO)
    hi = jnp.full(4, HI)
    g_mu = np.array([0.0, 0.0, 0.0, 0.0])
    g_sd = np.array([np.inf, np.inf, 1.0, 1.0])  # two uniform, two N(0,1) truncated to the box
    g_mask = jnp.asarray(np.isfinite(g_sd))
    g_sd_j = jnp.asarray(np.where(np.isfinite(g_sd), g_sd, 1.0))

    def log_prior_fn(x):
        inside = jnp.all(x >= lo, axis=-1) & jnp.all(x <= hi, axis=-1)
        inside &= (x[..., 0] + x[..., 1]) < 2.0  # hard cut
        z = (x - jnp.asarray(g_mu)) / g_sd_j
        lp = -0.5 * jnp.sum(jnp.where(g_mask, z * z, 0.0), axis=-1)
        return jnp.where(inside, lp, -jnp.inf)

    def log_like_from_theory(x, th):
        return -0.5 * jnp.sum(((th["y"] - jnp.asarray(MEAN)) / SIG) ** 2)

    def log_post_batch_fn(x):
        return jax.vmap(lambda xx: log_like_from_theory(xx, {"y": xx}))(x) + log_prior_fn(x)

    dali = SimpleNamespace(theory=lambda tt, x: {"y": x + tt})
    return SimpleNamespace(param_names=("a", "b", "c", "d"), prior_lo=lo, prior_hi=hi, gauss_mu=g_mu,
                           gauss_sd=g_sd, log_prior_fn=log_prior_fn,
                           log_like_from_theory_fn=log_like_from_theory,
                           log_post_batch_fn=jax.jit(log_post_batch_fn), dali=dali)


def _reference_logz(bias: float, n: int = 2_000_000, seed: int = 1) -> float:
    """log ∫ L(x + bias) π(x) dx under the normalized prior (Monte Carlo over prior draws)."""
    rng = np.random.default_rng(seed)
    x = np.empty((n, 4))
    x[:, :2] = rng.uniform(LO, HI, (n, 2))
    a, b = (LO - 0.0) / 1.0, (HI - 0.0) / 1.0
    x[:, 2:] = truncnorm.rvs(a, b, size=(n, 2), random_state=rng)
    ok = (x[:, 0] + x[:, 1]) < 2.0
    ll = -0.5 * np.sum(((x + bias - MEAN) / SIG) ** 2, axis=1)
    return float(logsumexp(ll[ok]) - np.log(n) - np.log(ok.mean()))


def test_sample_prior_matches_normalized_prior():
    b = _toy_bundle()
    x, info = sample_prior(b, 20000, np.random.default_rng(0))
    assert x.shape == (20000, 4)
    assert np.all(x[:, 0] + x[:, 1] < 2.0)
    assert abs(info["accept_fraction"] - 0.68) < 0.02  # P(u0 + u1 < 2) = 1 - 8²/(2·100) for u ~ U(-5, 5)
    assert abs(np.std(x[:, 2]) - 1.0) < 0.03 and abs(np.mean(x[:, 2])) < 0.03
    assert abs(np.mean(x[:, 0]) + 1.10) < 0.1  # the cut skews the uniform coordinates: -0.32·(7/3)/0.68


def test_dali_smc_and_order_check_recover_evidences(tmp_path):
    """SMC on a DALI likelihood whose theory is offset by BIAS recovers that likelihood's evidence;
    the order check against a "lower order" without the offset recovers the evidence difference
    and the posterior-mean shift from the same draws."""
    jax.config.update("jax_enable_x64", True)
    b = _toy_bundle()
    tt = jnp.asarray(BIAS)
    cfg = DALISMCConfig(n_particles=4000, n_runs=4, n_mcmc=4, n_leapfrog=8, n_final_mcmc=8, chunk=2000,
                        n_prior_mc=200_000, n_posterior=2000, seed=0)
    summ = run_dali_smc(b, make_dali_loglike(b), tt, cfg, tmp_path / "toy", log=lambda s: None)
    ref_s = _reference_logz(BIAS)
    ref = _reference_logz(0.0)
    assert abs(summ["logz_dali"] - ref_s) < 4 * summ["logz_dali_std"] + 0.05
    X = np.load(tmp_path / "toy.ns_posterior.npz")["pos"]
    ll_p = make_batched_dali_loglike(b, tt, 500)(X)
    ll_q = make_batched_dali_loglike(b, jnp.asarray(0.0), 500)(X)
    out = dali_order_check(X, ll_p, ll_q, b.param_names, order=3)
    assert out["lower_order"] == 2 and out["ess_fraction"] > 0.3
    assert abs(out["delta_logz"] - (ref - ref_s)) < 4 * out["delta_logz_err"] + 0.05
    # the offset theory's posterior sits BIAS low: the other one is BIAS/σ higher in every parameter
    sd = X.std(axis=0)
    for i, n in enumerate(b.param_names):
        assert abs(out["mean_shift_sigma"][n] - BIAS / sd[i]) < 0.15, n
