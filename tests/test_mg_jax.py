#!/usr/bin/env python3
"""μ–Σ modified gravity in the EB solver (ISiTGR port, ``eb.mg_jax``)."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


def _setup():
    from gpucosmo.jax.eb.cls_jax import CLS_HIERARCHY, _params_dict_cls, _recfast_thermo_device

    params = _params_dict_cls(0.02237, 0.12, 0.6736, 0.9649, 2.1e-9, 0.0544, -1.0, 0.0)
    thermo = _recfast_thermo_device(0.02237, 0.12, 0.6736, 0.9649, 2.1e-9, 0.0544, -1.0, 0.0)
    return params, thermo, CLS_HIERARCHY


def test_mu_sigma_today_and_time_derivative():
    from gpucosmo.jax.eb.hierarchy_jax import background_at
    from gpucosmo.jax.eb.mg_jax import mu_sigma

    params, thermo, cfg = _setup()
    params = {**params, "mu0": 0.4, "Sigma0": -0.2}
    tau_tab, a_tab = np.asarray(thermo["tau_mpc"]), np.asarray(thermo["a"])

    def at(tau):
        a = jnp.interp(tau, tau_tab, a_tab)
        bg = background_at(tau, a, params, cfg)
        grhov, grho, adotoa = bg[5], bg[9], bg[11]
        gpres = bg[10] + bg[6] * grhov
        Hdot = -(grho + 3.0 * gpres) / 6.0
        return mu_sigma(k=0.1, a=a, adotoa=adotoa, Hdot=Hdot, grhov=grhov, grho=grho,
                        params=params)

    mu, mudot, sig, sigdot = at(tau_tab[-1])
    np.testing.assert_allclose([float(mu), float(sig)], [1.4, 0.8], rtol=2e-4)
    tau = 0.7 * tau_tab[-1]
    eps = 1.0
    num_mu = (at(tau + eps)[0] - at(tau - eps)[0]) / (2 * eps)
    num_sig = (at(tau + eps)[2] - at(tau - eps)[2]) / (2 * eps)
    m = at(tau)
    np.testing.assert_allclose(float(m[1]), float(num_mu), rtol=2e-3)
    np.testing.assert_allclose(float(m[3]), float(num_sig), rtol=2e-3)


def test_mg_constraints_reduce_to_gr_at_mu_sigma_zero():
    """ISiTGR's (σ from Φ, η̇ from the Σ constraint, z = σ − 3η̇/k²) equals the
    GR constraints for μ = Σ = 1 in ΛCDM, for an arbitrary full-hierarchy state."""
    from gpucosmo.jax.eb.hierarchy_jax import IX_CLXC, IX_ETAK, derivs

    params, thermo, cfg = _setup()
    cfg_mg = dataclasses.replace(cfg, mg=True)
    rng = np.random.default_rng(0)
    tau = 0.5 * float(thermo["tau_mpc"][-1])  # a ≈ 0.3: MG branch active
    for k in (0.003, 0.05, 0.2):
        y = jnp.asarray(rng.normal(size=cfg.n_eq) * 1e-3)
        kw = dict(tight_coupling=False, no_phot_multpoles=False, no_nu_multpoles=False)
        gr, _, _ = derivs(tau, y, k, params, thermo, cfg=cfg, **kw)
        mg, _, _ = derivs(tau, y, k, {**params, "mu0": 0.0, "Sigma0": 0.0}, thermo,
                          cfg=cfg_mg, **kw)
        # η̇_k and δ̇_c = −k z carry the constraint solutions.
        np.testing.assert_allclose(float(mg[IX_ETAK]), float(gr[IX_ETAK]), rtol=1e-8, atol=1e-14)
        np.testing.assert_allclose(float(mg[IX_CLXC]), float(gr[IX_CLXC]), rtol=1e-8, atol=1e-14)
        np.testing.assert_allclose(np.asarray(mg), np.asarray(gr), rtol=1e-7, atol=1e-12)


def test_mg_branch_inactive_before_a_002():
    from gpucosmo.jax.eb.hierarchy_jax import derivs

    params, thermo, cfg = _setup()
    cfg_mg = dataclasses.replace(cfg, mg=True)
    a_tab, tau_tab = np.asarray(thermo["a"]), np.asarray(thermo["tau_mpc"])
    tau = float(np.interp(0.01, a_tab, tau_tab))
    y = jnp.asarray(np.random.default_rng(1).normal(size=cfg.n_eq) * 1e-3)
    kw = dict(tight_coupling=False, no_phot_multpoles=False, no_nu_multpoles=False)
    gr, _, _ = derivs(tau, y, 0.05, params, thermo, cfg=cfg, **kw)
    mg, _, _ = derivs(tau, y, 0.05, {**params, "mu0": 1.0, "Sigma0": 0.5}, thermo, cfg=cfg_mg,
                      **kw)
    np.testing.assert_array_equal(np.asarray(mg), np.asarray(gr))


def test_bbn_prior_is_schoneberg2024_at_fixed_neff():
    """The ω_b prior equals the official 2D (ω_b, N_eff) Gaussian at N_eff = 3.044 up to a constant."""
    from gpucosmo.jax.inference.mg_bundle import BBN_PRIOR, BBN_SCHONEBERG2024, N_EFF

    mean, cov = np.asarray(BBN_SCHONEBERG2024["mean"]), np.asarray(BBN_SCHONEBERG2024["cov"])
    icov = np.linalg.inv(cov)
    ob = np.linspace(0.020, 0.024, 9)
    d = np.stack([ob - mean[0], np.full_like(ob, N_EFF - mean[1])], 1)
    lp2d = -0.5 * np.einsum("ni,ij,nj->n", d, icov, d)
    lp1d = -0.5 * ((ob - BBN_PRIOR[0]) / BBN_PRIOR[1]) ** 2
    np.testing.assert_allclose(lp2d - lp2d[0], lp1d - lp1d[0], atol=1e-9)


def test_cosmomc_theta_matches_camb():
    """CAMB's approximate cosmomc_theta (what Cobaya samples as theta_MC_100) to ≤ 3e-5."""
    camb = pytest.importorskip("camb")
    import jax.numpy as jnp

    from gpucosmo.jax.background.cosmology import BackgroundParams, cosmomc_theta, load_background_tables

    t = load_background_tables()
    for H0, ob, oc, w, wa in [(68.2, 0.02263, 0.1173, -1.0, 0.0), (64.0, 0.0225, 0.12, -0.6, -1.2)]:
        p = camb.set_params(H0=H0, ombh2=ob, omch2=oc, mnu=0.06, num_massive_neutrinos=1,
                            nnu=3.044, dark_energy_model="ppf", w=w, wa=wa)
        ref = camb.get_background(p).cosmomc_theta()
        ours = float(cosmomc_theta(t, BackgroundParams.build(*map(jnp.asarray, (ob, oc, H0, w, wa)))))
        assert abs(ours / ref - 1) < 3e-5
