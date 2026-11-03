#!/usr/bin/env python3
"""dali_smc: tempered SMC evidences against analytic / brute-force values."""

from __future__ import annotations

import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpucosmo.jax.mcmc.dali_ns import sample_prior
from gpucosmo.jax.mcmc.dali_smc import DALISMCConfig, next_temperature, run_dali_smc

jax.config.update("jax_enable_x64", True)


def _toy(d=6, gauss_param=None, cut=False, seed=0):
    rng = np.random.default_rng(seed)
    lo, hi = -5.0 * np.ones(d), 5.0 * np.ones(d)
    mu_p, sd_p = np.zeros(d), np.full(d, np.inf)
    if gauss_param is not None:
        sd_p[gauss_param] = 2.0
    A = rng.standard_normal((d, d))
    cov = 0.25 * (A @ A.T / d + 0.5 * np.eye(d))  # posterior σ ~ 0.5, correlated
    mean = rng.uniform(-1.0, 1.0, d)
    P = np.linalg.inv(cov)
    lo_j, hi_j, g = jnp.asarray(lo), jnp.asarray(hi), np.isfinite(sd_p)

    def log_prior_fn(x):
        inside = jnp.all(x >= lo_j, axis=-1) & jnp.all(x <= hi_j, axis=-1)
        if cut:
            inside &= x[..., 0] < x[..., 1] + 0.5
        z = jnp.where(g, (x - mu_p) / np.where(g, sd_p, 1.0), 0.0)
        return jnp.where(inside, -0.5 * jnp.sum(z * z, axis=-1), -jnp.inf)

    bundle = SimpleNamespace(param_names=tuple(f"p{i}" for i in range(d)), prior_lo=lo_j, prior_hi=hi_j,
                             gauss_mu=mu_p, gauss_sd=sd_p, log_prior_fn=log_prior_fn)
    data = {"mean": jnp.asarray(mean), "P": jnp.asarray(P)}

    def ll(x, data):
        r = x - data["mean"]
        return -0.5 * r @ data["P"] @ r

    return bundle, ll, data, mean, cov


def _cfg(**kw):
    base = dict(n_particles=4000, n_runs=8, n_mcmc=4, n_leapfrog=8, n_final_mcmc=8, chunk=2000,
                n_prior_mc=100_000, n_posterior=4000)
    base.update(kw)
    return DALISMCConfig(**base)


def test_next_temperature_hits_the_ess_target():
    ll = np.random.default_rng(1).normal(-50.0, 10.0, 4000)
    b = next_temperature(ll, 0.0, 0.5)
    w = np.exp((b * ll) - (b * ll).max())
    assert abs(w.sum() ** 2 / (w * w).sum() / ll.size - 0.5) < 1e-3
    assert next_temperature(np.zeros(10), 0.3, 0.5) == 1.0


WF = {"standard": {}, "waste_free": dict(waste_free=True, chain_length=8)}


@pytest.mark.parametrize("variant", list(WF))
def test_gaussian_in_a_box_matches_analytic_evidence(tmp_path, variant):
    d = 6
    bundle, ll, data, mean, cov = _toy(d)
    exact = 0.5 * d * np.log(2 * np.pi) + 0.5 * np.linalg.slogdet(cov)[1] - d * np.log(10.0)
    s = run_dali_smc(bundle, ll, data, _cfg(**WF[variant]), tmp_path / "toy", log=lambda m: None)
    err = s["logz_dali_std"]
    assert abs(s["logz_dali"] - exact) < 4 * err, (s["logz_dali"], exact, err)
    post = np.load(tmp_path / "toy.ns_posterior.npz")
    assert np.all(np.abs(post["pos"].mean(axis=0) - mean) < 0.1)
    assert np.allclose(np.cov(post["pos"], rowvar=False), cov, atol=0.06)
    assert json.loads((tmp_path / "toy.ns_summary.json").read_text())["sampler"] == "smc_on_dali"


@pytest.mark.parametrize("variant", list(WF))
def test_cut_gaussian_prior_and_narrowed_box(tmp_path, variant):
    """Hard cut + truncated-Gaussian prior parameter + a narrowed sampling box: same evidence
    as brute-force Monte Carlo over the normalized prior."""
    d = 4
    bundle, ll, data, mean, cov = _toy(d, gauss_param=2, cut=True, seed=3)
    x, info = sample_prior(bundle, 1_000_000, np.random.default_rng(0))
    lv = np.asarray(jax.vmap(lambda xx: ll(xx, data))(jnp.asarray(x)))
    m = lv.max()
    ref = m + np.log(np.mean(np.exp(lv - m)))
    ref_err = np.std(np.exp(lv - m)) / np.mean(np.exp(lv - m)) / np.sqrt(lv.size)
    box = {"p3": (mean[3] - 3.5, mean[3] + 3.5)}  # posterior σ ≲ 0.6: ≥ 5σ inside
    s = run_dali_smc(bundle, ll, data, _cfg(sampling_box=box, min_edge_sigma=5.0, **WF[variant]), tmp_path / "cut",
                          log=lambda m: None)
    err = np.hypot(s["logz_dali_std"], ref_err)
    assert 0.0 < info["accept_fraction"] < 1.0 and s["prior_cut_mass"] < 1.0
    assert abs(s["logz_dali"] - ref) < 4 * err, (s["logz_dali"], ref, err)
    assert s["edge_ok"]


@pytest.mark.parametrize("variant", list(WF))
def test_run_scatter_matches_the_ess_error_model(tmp_path, variant):
    """Unbiased over many runs, and the per-run error predicted within a run (incremental
    weights' ESS; chain-mean spread for waste-free) matches the runs' scatter."""
    d = 6
    bundle, ll, data, mean, cov = _toy(d)
    exact = 0.5 * d * np.log(2 * np.pi) + 0.5 * np.linalg.slogdet(cov)[1] - d * np.log(10.0)
    s = run_dali_smc(bundle, ll, data, _cfg(n_particles=1000, n_runs=24, chunk=1000, **WF[variant]),
                          tmp_path / "many", log=lambda m: None)
    lz = np.asarray(s["logz_runs"])
    assert abs(lz.mean() - exact) < 3 * lz.std(ddof=1) / np.sqrt(lz.size) + 0.01
    if variant == "standard":
        assert 0.6 < s["logz_run_std"] / s["logz_run_std_ess"] < 1.6, (s["logz_run_std"], s["logz_run_std_ess"])
    else:  # the chain-mean estimate ignores the genealogy: it can only underestimate
        assert s["logz_run_std_ess"] < 1.2 * s["logz_run_std"]


REF = {"pilot": dict(reference="pilot", pilot_particles=1000)}


@pytest.mark.parametrize("variant", list(REF))
def test_reference_start_gaussian_in_a_box(tmp_path, variant):
    """Path from a fitted Student-t: the analytic evidence and the posterior, in few stages."""
    d = 6
    bundle, ll, data, mean, cov = _toy(d)
    exact = 0.5 * d * np.log(2 * np.pi) + 0.5 * np.linalg.slogdet(cov)[1] - d * np.log(10.0)
    s = run_dali_smc(bundle, ll, data, _cfg(**REF[variant]), tmp_path / "ref", log=lambda m: None)
    assert abs(s["logz_dali"] - exact) < 4 * s["logz_dali_std"], (s["logz_dali"], exact, s["logz_dali_std"])
    assert max(s["n_stages"]) < 8  # the prior start needs ~20 on this toy
    post = np.load(tmp_path / "ref.ns_posterior.npz")
    assert np.all(np.abs(post["pos"].mean(axis=0) - mean) < 0.1)
    assert np.allclose(np.cov(post["pos"], rowvar=False), cov, atol=0.06)
    lv = np.asarray(jax.vmap(lambda xx: ll(xx, data))(jnp.asarray(post["pos"])))
    np.testing.assert_allclose(post["loglike"], lv, rtol=1e-10, atol=1e-10)  # the DALI log L, not the path's


@pytest.mark.parametrize("variant", list(REF))
def test_reference_start_cut_gaussian_prior_and_narrowed_box(tmp_path, variant):
    """Hard cut + truncated-Gaussian prior + narrowed box: the prior normalization (separable
    masses × cut mass) is applied correctly — same evidence as brute-force Monte Carlo."""
    d = 4
    bundle, ll, data, mean, cov = _toy(d, gauss_param=2, cut=True, seed=3)
    x, info = sample_prior(bundle, 1_000_000, np.random.default_rng(0))
    lv = np.asarray(jax.vmap(lambda xx: ll(xx, data))(jnp.asarray(x)))
    m = lv.max()
    ref = m + np.log(np.mean(np.exp(lv - m)))
    ref_err = np.std(np.exp(lv - m)) / np.mean(np.exp(lv - m)) / np.sqrt(lv.size)
    box = {"p3": (mean[3] - 3.5, mean[3] + 3.5)}
    s = run_dali_smc(bundle, ll, data, _cfg(sampling_box=box, min_edge_sigma=5.0, **REF[variant]),
                          tmp_path / "refcut", log=lambda m: None)
    err = np.hypot(s["logz_dali_std"], ref_err)
    assert abs(s["logz_dali"] - ref) < 4 * err, (s["logz_dali"], ref, err)


def test_reference_start_is_unbiased_over_many_runs(tmp_path):
    d = 6
    bundle, ll, data, mean, cov = _toy(d)
    exact = 0.5 * d * np.log(2 * np.pi) + 0.5 * np.linalg.slogdet(cov)[1] - d * np.log(10.0)
    s = run_dali_smc(bundle, ll, data, _cfg(n_particles=1000, n_runs=24, chunk=1000, reference="pilot",
                                                 pilot_particles=1000), tmp_path / "refmany", log=lambda m: None)
    lz = np.asarray(s["logz_runs"])
    assert abs(lz.mean() - exact) < 3 * lz.std(ddof=1) / np.sqrt(lz.size) + 0.005
    assert 0.5 < s["logz_run_std"] / s["logz_run_std_ess"] < 1.8, (s["logz_run_std"], s["logz_run_std_ess"])


def test_target_error_stops_adding_runs(tmp_path):
    d = 6
    bundle, ll, data, mean, cov = _toy(d)
    s = run_dali_smc(bundle, ll, data, _cfg(n_particles=1000, n_runs=12, chunk=1000, reference="pilot",
                                                 pilot_particles=1000, target_err=0.03, min_runs=3),
                          tmp_path / "tgt", log=lambda m: None)
    assert 3 <= len(s["logz_runs"]) < 12 and s["logz_dali_std"] <= 0.03
