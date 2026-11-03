#!/usr/bin/env python3
"""The supernova term on the DALI expansion (whitened coefficients) and CamSpec's compressed χ²
(``dali_reduce.CamSpecCompressed``) against their direct forms, without data products."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)


def test_psd_whitening_reproduces_a_singular_quadratic_form():
    """An inverse covariance with an analytically marginalized offset is singular: xᵀAx = |Wᵀx|²."""
    from gpucosmo.jax.inference.dali_reduce import psd_whitening

    rng = np.random.default_rng(0)
    B = rng.normal(size=(40, 40))
    C = B @ B.T + 40 * np.eye(40)
    Ci = np.linalg.inv(C)
    one = np.ones(40)
    A = Ci - np.outer(Ci @ one, Ci @ one) / (one @ Ci @ one)  # marginalized over a common offset
    W = psd_whitening(A)
    for _ in range(5):
        x = rng.normal(size=40)
        assert abs(x @ A @ x - np.sum((W.T @ x) ** 2)) < 1e-10 * abs(x @ A @ x)
    with pytest.raises(ValueError, match="positive semi-definite"):
        psd_whitening(-np.eye(3))


def _fake_camspec(rng, n_terms):
    """A CamSpec port with three segments (TT 143, TE, EE), a power-law foreground with sampled
    amplitude and index, and calibrations, without Cobaya: the attributes the port's own
    ``residual`` / ``cals`` / ``_param`` use."""
    from gpucosmo.likelihoods.cobaya.planck_python import CamSpec, _Chi2

    L = object.__new__(CamSpec)
    ls = {1: np.arange(30, 330, 3), 4: np.arange(30, 230, 4), 5: np.arange(30, 230, 5)}
    L.segments = [(i, v) for i, v in ls.items()]
    n = sum(v.size for v in ls.values())
    B = rng.normal(size=(n, n))
    L.chi2 = _Chi2(np.linalg.inv(B @ B.T / n + np.eye(n)))
    L.fg_2018, L.fg_powerlaw, L.sz_prior = False, True, False
    L.spectra = ["ee", "te", "tt"]
    L.fixed = {"cal0": 1.0, "cal2": 1.0}
    L.params = {k: None for k in ("A_planck", "calTE", "calEE", "amp_143", "n_143")}
    L.requires = ("cmb",)
    L.data = rng.normal(size=n) * 10 + 100
    spec = {s: rng.normal(size=(n_terms, 400)) * (100.0 / (1 + np.arange(n_terms)))[:, None] for s in ("tt", "te", "ee")}
    spec["tt"][0] += 100.0
    return L, spec


def test_compressed_camspec_equals_the_port():
    """Compressed χ² (float32 quadratic term) against the port's own on the same expanded spectra, at
    random parameters, with all theory and nuisance dependences (calibrations, power-law amplitude and
    index); also truncated to a lower order."""
    from gpucosmo.jax.inference.dali_expansion import monomials, n_terms
    from gpucosmo.jax.inference.dali_reduce import CamSpecCompressed
    from gpucosmo.likelihoods.base import LikelihoodInputs

    rng = np.random.default_rng(1)
    n, order = 2, 2
    nt = n_terms(n, order)
    L, spec = _fake_camspec(rng, nt)
    names = ["A_planck", "calTE", "calEE", "amp_143", "n_143"]
    idx = {k: i for i, k in enumerate(names)}
    bundle = SimpleNamespace(internals=SimpleNamespace(idx=idx), prior_lo=np.array([0.9, 0.9, 0.9, 0.0, 0.0]),
                             prior_hi=np.array([1.1, 1.1, 1.1, 50.0, 5.0]))
    cmb_terms = {"dl_tt": spec["tt"], "dl_te": spec["te"], "dl_ee": spec["ee"]}
    x_ref = np.array([1.0, 1.0, 1.0, 10.0, 1.0])
    assert CamSpecCompressed.supports(L)
    cs = CamSpecCompressed(L, cmb_terms, bundle, x_ref, np.zeros(n), order, np.full(n, 0.3), log=lambda m: None)
    assert cs.ref_check < 1e-10 and cs.basis_change_err < 1e-10
    f = jax.jit(lambda d, x: cs.loglike(d, x, cs.arrays))
    f1 = jax.jit(lambda d, x: cs.loglike(d, x, cs.arrays, order=1))
    for _ in range(6):
        d = rng.normal(size=n) * 0.3
        x = x_ref + rng.normal(size=5) * np.array([0.003, 0.01, 0.01, 3.0, 0.5])
        for q, fn in ((order, f), (1, f1)):
            phi = np.array(monomials(jnp.asarray(d), order))
            phi[n_terms(n, q):] = 0.0
            cl = {s: jnp.asarray(phi @ spec[s]) for s in ("tt", "te", "ee")}
            inp = LikelihoodInputs(theory=None, params={k: jnp.asarray(x[idx[k]]) for k in names}, cosmo=None,
                                   shared={"cmb": {"cl": cl}})
            want = float(L.loglike(inp))
            got = float(fn(jnp.asarray(d), jnp.asarray(x)))
            assert abs(got - want) < 1e-6 * max(1.0, abs(want)), (q, got, want)
    g = jax.grad(lambda x: cs.loglike(jnp.asarray(d), x, cs.arrays))(jnp.asarray(x))
    assert np.all(np.isfinite(np.asarray(g)))


def test_act_port_projection_equals_the_port():
    """The ACT DR6 lensing port's bandpowers are linear in κκ (from the φφ the port converts, 0 at
    ℓ = 0) and the lensed D_ℓ: their expansion coefficients reproduce the port's log L."""
    from gpucosmo.jax.cmb.act_dr6_lens_jax import ACTDR6Lens, build_act_dr6_lens_loglike
    from gpucosmo.jax.inference.dali_reduce import act_port_projectable, act_port_terms
    from gpucosmo.jax.inference.mg_bundle import shared_block
    from gpucosmo.likelihoods.base import LikelihoodInputs
    from gpucosmo.likelihoods.cobaya.act_lens import ACTDR6LensLike

    rng = np.random.default_rng(2)
    n_ell, nb, nt = 60, 7, 5
    B = rng.normal(size=(nb, nb))
    lens = ACTDR6Lens(variant="act_baseline", lens_only=False, data=rng.normal(size=nb), cinv=B @ B.T + nb * np.eye(nb),
                      a_kk=rng.normal(size=(nb, n_ell)) * 1e-2, a_cmb=rng.normal(size=(4, nb, n_ell)) * 1e-5,
                      offset=rng.normal(size=nb), n_act=nb)
    L = object.__new__(ACTDR6LensLike)
    L.lens, L.n_ell, L._ll = lens, n_ell, build_act_dr6_lens_loglike(lens)
    x = rng.normal(size=(nt, 5, n_ell + 10)) * np.array([1.0, 100.0, 10.0, 1.0, 10.0])[None, :, None]
    cmb_terms = {"cmb_x": x}
    assert act_port_projectable(L, cmb_terms)
    T = act_port_terms(L, cmb_terms)
    for _ in range(4):
        phi = np.r_[1.0, rng.normal(size=nt - 1)]
        r = lens.data - (phi @ T + lens.offset)
        got = -0.5 * r @ lens.cinv @ r
        shared = {"cmb": shared_block("cmb", {"cmb": {"cmb_x": jnp.asarray(np.einsum("t,txl->xl", phi, x))}})}
        want = float(L.loglike(LikelihoodInputs(theory=None, params={}, cosmo=None, shared=shared)))
        assert abs(got - want) < 1e-9 * max(1.0, abs(want)), (got, want)
    assert not act_port_projectable(L, {"cmb_x": x[..., :n_ell - 1]})  # spectra too short


def test_clik_ports_declare_the_multipoles_they_read():
    from gpucosmo.likelihoods.base import Likelihood
    from gpucosmo.likelihoods.cobaya.planck_clik import LowlClik

    L = object.__new__(LowlClik)
    L.spectra = [("tt", 29), ("te", 29), ("ee", 29)]
    assert L.cmb_lmax() == 29
    assert Likelihood.cmb_lmax(object.__new__(Likelihood)) is None  # undeclared: the full block
