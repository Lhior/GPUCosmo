#!/usr/bin/env python3
"""Unit tests for CCL-matched JAX integrators."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from gpucosmo.jax.fftlog import (
    gamma_t_ng_fftlog_ccl_numpy,
    gamma_t_ng_fftlog_jax,
    gamma_t_ng_fftlog_numpy,
)
from gpucosmo.jax.limber_quad import integrate_gk41_logk, limber_k_bounds
from gpucosmo.jax.splines import (
    akima_interp_jax,
    ccl_f1d_eval_jax,
    ccl_f1d_eval_numpy,
    ccl_log_spacing_numpy,
    log_k_grid,
    resample_logx_logy_numpy,
)


def test_log_k_grid_spacing():
    grid = log_k_grid()
    steps = np.diff(grid)
    assert np.allclose(steps, steps[0], rtol=1.0e-6)


def test_ccl_log_spacing_endpoints():
    x = ccl_log_spacing_numpy(0.01, 60000.0, 5000)
    assert x[0] == pytest.approx(0.01)
    assert x[-1] == pytest.approx(60000.0)
    assert np.all(np.diff(x) > 0.0)


def test_ccl_f1d_const_logx_logy_matches_pyccl_swap():
    pytest.importorskip("pyccl")
    from pyccl.pyutils import resample_array

    x = np.geomspace(2.0, 5000.0, 80)
    y = 1.0e-8 * (x / 200.0) ** (-1.5)
    x_out = ccl_log_spacing_numpy(0.01, 60000.0, 200)
    ours = ccl_f1d_eval_numpy(
        x,
        y,
        x_out,
        extrap_lo="const",
        extrap_hi="logx_logy",
        y0=float(y[0]),
        yf=0.0,
    )
    # pyccl SWIG swaps extrap_lo/hi; this is the CCL-true const-lo/logx_logy-hi combo.
    ref = resample_array(
        x,
        y,
        x_out,
        extrap_lo="logx_logy",
        extrap_hi="constant",
        fill_value_lo=float(y[0]),
        fill_value_hi=0.0,
    )
    assert np.max(np.abs(ours / ref - 1.0)) < 1.0e-10


def test_ccl_f1d_jax_matches_numpy():
    x = np.geomspace(2.0, 5000.0, 60)
    y = 1.0e-8 * (x / 200.0) ** (-2.0)
    xq = np.geomspace(1.0, 8000.0, 40)
    ref = ccl_f1d_eval_numpy(
        x,
        y,
        xq,
        extrap_lo="const",
        extrap_hi="const",
        y0=float(y[0]),
        yf=0.0,
    )
    out = np.asarray(
        ccl_f1d_eval_jax(
            jnp.asarray(x),
            jnp.asarray(y),
            jnp.asarray(xq),
            extrap_lo="const",
            extrap_hi="const",
            y0=float(y[0]),
            yf=0.0,
        )
    )
    assert np.nanmedian(np.abs(out / np.maximum(np.abs(ref), 1.0e-30) - 1.0)) < 1.0e-3
    xq_hi = np.geomspace(4000.0, 80000.0, 20)
    ref_hi = ccl_f1d_eval_numpy(
        x,
        y,
        xq_hi,
        extrap_lo="const",
        extrap_hi="logx_logy",
        y0=float(y[0]),
        yf=0.0,
    )
    out_hi = np.asarray(
        ccl_f1d_eval_jax(
            jnp.asarray(x),
            jnp.asarray(y),
            jnp.asarray(xq_hi),
            extrap_lo="const",
            extrap_hi="logx_logy",
            y0=float(y[0]),
            yf=0.0,
        )
    )
    assert np.nanmedian(np.abs(out_hi / np.maximum(np.abs(ref_hi), 1.0e-30) - 1.0)) < 1.0e-4


def test_akima_reproduces_linear():
    x = jnp.linspace(0.0, 1.0, 20)
    y = 2.0 * x + 1.0
    xq = jnp.linspace(0.1, 0.9, 7)
    out = akima_interp_jax(x, y, xq)
    assert np.allclose(np.asarray(out), np.asarray(2.0 * xq + 1.0), rtol=1.0e-5)


def test_resample_logx_logy_flat():
    x = np.geomspace(1.0, 100.0, 50)
    y = np.ones_like(x) * 3.0
    x_out = np.geomspace(2.0, 80.0, 30)
    out = resample_logx_logy_numpy(x, y, x_out)
    assert np.allclose(out, 3.0, rtol=1.0e-6)


def test_limber_k_bounds_respects_ccl_limits():
    k_lo, k_hi = limber_k_bounds(500.0, chi_min=10.0, chi_max=5000.0)
    assert k_lo > 0.0
    assert k_hi > k_lo
    assert k_lo >= 500.5 / 5000.0


def test_fftlog_jax_matches_numpy():
    ell = np.logspace(np.log10(2), np.log10(5000), 40)
    cell = 1.0e-8 * (ell / 500.0) ** (-1.5)
    theta = np.geomspace(1.5, 30.0, 12)
    ref = gamma_t_ng_fftlog_numpy(ell, cell, theta)
    out = np.asarray(
        gamma_t_ng_fftlog_jax(
            jnp.asarray(cell),
            jnp.asarray(ell),
            jnp.asarray(theta),
            n_theta=int(theta.size),
        )
    )
    assert np.median(np.abs(out / ref - 1.0)) < 5.0e-4


def test_fftlog_native_matches_ccl():
    pytest.importorskip("pyccl")
    ell = np.logspace(np.log10(2), np.log10(5000), 80)
    cell = 1.0e-8 * (ell / 200.0) ** (-2.0)
    theta = np.geomspace(0.0026, 25.0, 12)
    native = gamma_t_ng_fftlog_numpy(ell, cell, theta)
    ccl = gamma_t_ng_fftlog_ccl_numpy(ell, cell, theta)
    rel = np.abs(native - ccl) / np.maximum(np.abs(ccl), 1.0e-30)
    assert np.median(rel) < 0.02
    assert np.max(rel) < 0.05


def test_fftlog_ccl_callback_matches_ccl(monkeypatch):
    pytest.importorskip("pyccl")
    monkeypatch.setenv("GPUCOSMO_LEGACY_FFTLOG", "1")
    ell = np.logspace(np.log10(2), np.log10(5000), 80)
    cell = 1.0e-8 * (ell / 200.0) ** (-2.0)
    theta = np.geomspace(0.5, 25.0, 10)
    ref = gamma_t_ng_fftlog_ccl_numpy(ell, cell, theta)
    out = np.asarray(
        gamma_t_ng_fftlog_jax(
            jnp.asarray(cell),
            jnp.asarray(ell),
            jnp.asarray(theta),
            n_theta=int(theta.size),
        )
    )
    rel = np.abs(out - ref) / np.maximum(np.abs(ref), 1.0e-30)
    assert np.median(rel) < 1.0e-6


def test_fftlog_jax_grad_finite():
    ell = np.logspace(np.log10(2), np.log10(5000), 40)
    cell = 1.0e-8 * (ell / 500.0) ** (-1.5)
    theta = np.geomspace(0.01, 5.0, 8)

    def gamma_sum(c):
        return jnp.sum(
            gamma_t_ng_fftlog_jax(
                c,
                jnp.asarray(ell),
                jnp.asarray(theta),
                n_theta=int(theta.size),
            )
        )

    grad = jax.grad(gamma_sum)(jnp.asarray(cell))
    assert np.all(np.isfinite(np.asarray(grad)))


def test_gk_logk_integrates_power_law():
    alpha = -2.0

    def integrand(k):
        return k**alpha

    log_lo = jnp.log(1.0e-3)
    log_hi = jnp.log(1.0e-1)
    numeric = float(integrate_gk41_logk(integrand, log_lo, log_hi))
    assert np.isfinite(numeric)
    assert numeric > 0.0


def test_pk2d_spline_matches_ccl():
    """The JAX P(k, z) spline against pyccl's ``Pk2D`` on the same (smooth, analytic) table."""
    pytest.importorskip("pyccl")
    import jax.numpy as jnp
    import numpy as np
    import pyccl as ccl

    from gpucosmo.jax.splines import pk2d_eval_from_z_jax

    log_k = np.log(np.geomspace(1.0e-4, 50.0, 256))
    z_tab = np.linspace(0.0, 1.7, 80)
    k = np.exp(log_k)
    log_pk = (np.log(2.0e4 * k / (1.0 + (k / 0.02) ** 2) ** 1.5)[None, :]
              - 2.0 * np.log1p(z_tab)[:, None])
    a_arr = 1.0 / (1.0 + z_tab[::-1])
    pk2d = ccl.Pk2D(a_arr=a_arr, lk_arr=log_k, pk_arr=log_pk[::-1], is_logp=True,
                    extrap_order_lok=1, extrap_order_hik=2)
    rng = np.random.default_rng(0)
    kq = jnp.asarray(rng.uniform(1.0e-3, 10.0, 32))
    zq = jnp.asarray(rng.uniform(0.05, 1.5, 32))
    pk_jax = np.asarray(pk2d_eval_from_z_jax(kq, zq, jnp.asarray(log_k), jnp.asarray(z_tab),
                                             jnp.asarray(log_pk)))
    pk_ccl = np.array(
        [float(pk2d(float(kv), 1.0 / (1.0 + float(zv)), cosmo=None)) for kv, zv in zip(kq, zq)]
    )
    rel = np.abs(pk_jax / np.maximum(pk_ccl, 1.0e-30) - 1.0)
    assert np.median(rel) < 1.0e-4  # 4e-6 / 9e-5 measured
    assert np.max(rel) < 1.0e-3
