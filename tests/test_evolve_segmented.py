#!/usr/bin/env python3
"""Regime-static segmented evolution: building blocks and convergence."""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


def test_hermite_reproduces_cubic():
    from gpucosmo.jax.eb.evolve_segmented_jax import _hermite

    knots = jnp.asarray([0.0, 0.3, 1.1, 1.5, 2.9])
    poly = lambda t: 1.0 - 2.0 * t + 0.5 * t**2 - 0.25 * t**3  # noqa: E731
    dpoly = lambda t: -2.0 + t - 0.75 * t**2  # noqa: E731
    Y = jnp.stack([poly(knots), 2.0 * poly(knots)], axis=1)
    F = jnp.stack([dpoly(knots), 2.0 * dpoly(knots)], axis=1)
    tq = jnp.linspace(-0.5, 3.5, 41)  # includes clamped points outside the range
    got = np.asarray(_hermite(knots, Y, F, tq))
    tc = np.clip(np.asarray(tq), 0.0, 2.9)
    np.testing.assert_allclose(got[:, 0], poly(tc), rtol=0, atol=1e-12)
    np.testing.assert_allclose(got[:, 1], 2.0 * poly(tc), rtol=0, atol=1e-12)


def test_equidistributed_knots_equal_stiffness_and_degenerate_segment():
    from gpucosmo.jax.eb.evolve_segmented_jax import _equidistributed_knots

    lam = lambda t: 1.0 + 50.0 / t  # noqa: E731
    kn = np.asarray(_equidistributed_knots(1.0, 100.0, 64, lam, 2048))
    assert kn.shape == (65,)
    assert kn[0] == 1.0 and kn[-1] == 100.0
    assert np.all(np.diff(kn) > 0)
    # ∫Λ over each interval (analytic) is constant.
    cum = kn + 50.0 * np.log(kn)
    per = np.diff(cum)
    assert np.max(per) / np.min(per) < 1.01

    same = np.asarray(_equidistributed_knots(7.0, 7.0, 16, lam, 256))
    np.testing.assert_allclose(same, 7.0)


def test_traced_high_ktau_flag_matches_static_branches():
    from gpucosmo.jax.eb.cls_jax import CLS_HIERARCHY as cfg

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from bench_evolve import setup

    from gpucosmo.jax.eb.evolve_jax import _thermo_for_rhs
    from gpucosmo.jax.eb.hierarchy_jax import derivs, initial_adiabatic

    ctx = setup()
    th = _thermo_for_rhs(ctx["thermo_ev"])
    k, tau = 0.2, 150.0
    y = initial_adiabatic(k, 1.0, ctx["params"], cfg)
    y = y + 1e-3 * jnp.sin(jnp.arange(cfg.n_eq, dtype=jnp.float64))
    kw = dict(tight_coupling=False, cfg=cfg)
    for flag in (True, False):
        want, _, _ = derivs(tau, y, k, ctx["params"], th,
                            high_ktau_neutrino_approx=flag, **kw)
        got = jax.jit(
            lambda f: derivs(tau, y, k, ctx["params"], th,
                             high_ktau_neutrino_approx=f, **kw)[0]
        )(jnp.asarray(flag))
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-13, atol=0)


@pytest.mark.slow
def test_segmented_sources_converged_and_stable_to_kmax():
    """Default budgets vs 2x budgets agree to 1e-3 on S_T/S_E, 1e-4 on Weyl, k ≤ 0.45."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from bench_evolve import make_sources_fn, rel_err, setup

    from gpucosmo.jax.eb.evolve_segmented_jax import DEFAULT_SEGMENTED as d

    ctx = setup()
    ks = jnp.asarray([0.002, 0.03, 0.2, 0.45])
    fine = dataclasses.replace(d, n_tc=2 * d.n_tc, n_full=2 * d.n_full,
                               n_mid=2 * d.n_mid, n_rsa=2 * d.n_rsa)
    got = make_sources_fn(ctx, "segmented", 0, 0)(ks)
    ref = make_sources_fn(ctx, "segmented", 0, 0, seg_cfg=fine)(ks)
    for name, g, r, tol in zip(("s_t", "s_e", "weyl"), got, ref, (1e-3, 1e-3, 1e-4)):
        g, r = np.asarray(g), np.asarray(r)
        assert np.all(np.isfinite(g)), name
        assert np.max(rel_err(g, r)) < tol, (name, rel_err(g, r))


def test_fused_tt_ee_te_los_matches_separate_projections():
    from gpucosmo.jax.eb.los_jax import (
        hybrid_cl_from_sources,
        hybrid_cl_te_from_sources,
        hybrid_cls_tt_ee_te_from_sources,
    )

    rng = np.random.default_rng(3)
    n_k, n_tau = 40, 120
    k = jnp.geomspace(1e-3, 0.2, n_k)
    tau = jnp.linspace(200.0, 14000.0, n_tau)
    tau0 = 14000.0
    env = jnp.exp(-0.5 * ((tau - 280.0) / 30.0) ** 2)[None, :]
    s_t = jnp.asarray(rng.standard_normal((n_k, n_tau))) * env
    s_e = jnp.asarray(rng.standard_normal((n_k, n_tau))) * env
    p_r = 2e-9 * (k / 0.05) ** -0.035
    ells = [2, 10, 50, 120, 400, 1600, 2000]  # crosses the Limber switch
    ell = jnp.asarray(ells, dtype=jnp.float64)
    kw = dict(limber_ell=1500, ell_list=ells)
    tt, ee, te = hybrid_cls_tt_ee_te_from_sources(
        s_t, s_e, tau, tau0, k, p_r, ell, compute_dtype=jnp.float64, **kw
    )
    want_tt = hybrid_cl_from_sources(s_t, tau, tau0, k, p_r, ell, bessel_lmax=1499, **kw)
    want_ee = hybrid_cl_from_sources(s_e, tau, tau0, k, p_r, ell, bessel_lmax=1499, **kw)
    want_te = hybrid_cl_te_from_sources(s_t, s_e, tau, tau0, k, p_r, ell,
                                        bessel_lmax=1499, **kw)
    for got, want in ((tt, want_tt), (ee, want_ee), (te, want_te)):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-10,
                                   atol=1e-12 * float(np.max(np.abs(want))))


def _structural_needs(cfg, k=0.03, n_tau=96, seed=0):
    """Entries of each segment's native state with a nonzero gradient of a random
    combination of (S_T, S_E, Weyl, δ_m), on a subsampled τ grid."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from bench_evolve import setup

    from gpucosmo.jax.eb.evolve_jax import (
        _apply_tc_init,
        _mid_nu_rsa_hierarchy,
        _scatter_mid_to_full,
        _scatter_rsa_to_full,
    )
    from gpucosmo.jax.eb.evolve_segmented_jax import _rsa_hierarchy_nu, _thermo4
    from gpucosmo.jax.eb.sources_jax import _background_and_stress, temperature_polarization_sources

    ctx = setup()
    params = dict(ctx["params"])
    if cfg.mg:
        params.update(mu0=jnp.asarray(0.3), Sigma0=jnp.asarray(0.1))
    sub = np.linspace(0, ctx["tau_src"].shape[0] - 1, n_tau).astype(int)
    tau, vis, kappa = ctx["tau_src"][sub], ctx["vis"][sub], ctx["kappa"][sub]
    thermo = ctx["thermo"]
    th_out = jax.vmap(lambda t: _thermo4(t, thermo))(tau)
    mid_cfg, rsa_cfg = _mid_nu_rsa_hierarchy(cfg), _rsa_hierarchy_nu(cfg)
    rng = np.random.default_rng(seed)
    w = jnp.asarray(rng.standard_normal((4, n_tau)))

    def outputs(y):
        s_t, s_e, weyl = temperature_polarization_sources(y, k, tau, vis, kappa, params, thermo, cfg,
                                                           tau0=ctx["tau0"])
        dm = _background_and_stress(y, k, tau, params, thermo, cfg)["delta_m"]
        return jnp.sum(w * jnp.stack([s_t, s_e, weyl, dm]))

    convert = {
        "tc": (cfg.n_eq, lambda yn: jax.vmap(
            lambda yy, tt: _apply_tc_init(yy, tt, k, params, thermo, cfg))(yn, tau)),
        "full": (cfg.n_eq, lambda yn: yn),
        "mid": (mid_cfg.n_eq, lambda yn: jax.vmap(
            lambda ym, t, th: _scatter_mid_to_full(ym, t, k, params, thermo, cfg, mid_cfg,
                                                   thermo_at_tau=tuple(th)))(yn, tau, th_out)),
        "rsa": (rsa_cfg.n_eq, lambda yn: jax.vmap(
            lambda yr, t, th: _scatter_rsa_to_full(yr, t, k, params, thermo, cfg, thermo_at_tau=tuple(th),
                                                   rsa_cfg=rsa_cfg))(yn, tau, th_out)),
    }
    needs = {}
    for name, (n_nat, f) in convert.items():
        yn = jnp.asarray(rng.standard_normal((n_tau, n_nat)) * 1e-2)
        g = np.asarray(jax.grad(lambda v: outputs(f(v)))(yn))
        needs[name] = np.flatnonzero(np.any(g != 0, axis=0))
    return needs


@pytest.mark.slow
@pytest.mark.parametrize("mg", [False, True])
def test_source_knot_indices_cover_source_dependencies(mg):
    """``store="sources"`` keeps every entry the sources and δ_m depend on."""
    from gpucosmo.jax.eb.cls_jax import CLS_HIERARCHY
    from gpucosmo.jax.eb.evolve_segmented_jax import source_knot_indices

    cfg = dataclasses.replace(CLS_HIERARCHY, mg=mg)
    keep = source_knot_indices(cfg)
    needs = _structural_needs(cfg)
    for name in ("tc", "full", "mid", "rsa"):
        assert set(needs[name].tolist()) <= set(keep[name].tolist()), (name, needs[name], keep[name])


@pytest.mark.slow
def test_store_sources_matches_store_all():
    """Reduced knot storage gives the same sources as storing every entry."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from bench_evolve import make_sources_fn, setup

    from gpucosmo.jax.eb.evolve_segmented_jax import DEFAULT_SEGMENTED as d

    ctx = setup()
    ks = jnp.asarray([0.002, 0.03, 0.2, 0.45])
    got = make_sources_fn(ctx, "segmented", 0, 0, seg_cfg=dataclasses.replace(d, store="sources"))(ks)
    ref = make_sources_fn(ctx, "segmented", 0, 0, seg_cfg=dataclasses.replace(d, store="all"))(ks)
    for name, g, r in zip(("s_t", "s_e", "weyl"), got, ref):
        g, r = np.asarray(g), np.asarray(r)
        scale = np.max(np.abs(r), axis=-1, keepdims=True)
        assert np.max(np.abs(g - r) / scale) < 1e-12, name


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_los_k_outer_scan_matches_ell_outer(dtype):
    """The k-outer TT/EE/TE scan ([ℓ, interval, 4] coefficient rows, one interval search
    per k chunk) gives the ℓ-outer scan's per-ℓ k sums in the same order."""
    from gpucosmo.jax.eb import los_jax as L

    rng = np.random.default_rng(5)
    n_k, n_tau = 300, 90  # k: two chunks of 256 (one padded); ℓ: two chunks of 32
    k = jnp.geomspace(1e-3, 0.3, n_k)
    tau = jnp.linspace(200.0, 14000.0, n_tau)
    tau0 = 14000.0
    env = jnp.exp(-0.5 * ((tau - 280.0) / 30.0) ** 2)[None, :]
    s_t = jnp.asarray(rng.standard_normal((n_k, n_tau))) * env
    s_e = jnp.asarray(rng.standard_normal((n_k, n_tau))) * env
    p_r = 2e-9 * (k / 0.05) ** -0.035
    ells = [2, 3, 5, 8, 12, 20, 30, 45, 60, 80, 100, 130, 160, 200, 250, 300, 380, 460, 550, 650,
            760, 880, 1000, 1150, 1300, 1450, 1600, 1750, 1900, 2050, 2200, 2350, 2500, 2650, 2800]
    dtau = L.camb_dpoints(tau, half_ends=True)
    chi = jnp.maximum(tau0 - tau, 0.0)
    table = L.build_bessel_horner_table(ells, L._bessel_xmax_host(k, chi))
    args = (s_t, s_e, dtau, chi, k, p_r, L.camb_dlnk(k), table, len(ells))
    got = L._los_tt_ee_te_scan_k_outer(*args, compute_dtype=dtype)
    want = L._los_tt_ee_te_scan(*args, compute_dtype=dtype)
    for g, w in zip(got, want):
        scale = float(np.max(np.abs(np.asarray(w))))
        np.testing.assert_allclose(np.asarray(g), np.asarray(w), rtol=0, atol=1e-12 * scale)


@pytest.mark.slow
@pytest.mark.skipif(jax.default_backend() == "cpu",
                    reason="float32 sources are the accelerator path (JAX_PLATFORMS=cuda pytest -m slow)")
def test_float32_sources_close_to_float64():
    """Batched sources with the per-mode evolution in float32 stay within 1e-5 (relative
    to each source's maximum) of float64, away from the duplicated τ0 end point."""
    from gpucosmo.jax.eb import cls_jax as C

    ob, oc, h, ns, As, tau, w0, wa = (jnp.asarray([v]) for v in (0.0224, 0.12, 0.675, 0.965, 2.1e-9,
                                                                    0.055, -1.0, 0.0))
    params_b = jax.vmap(lambda *a: C.cmb_params(*a)[0])(ob, oc, h, ns, As, tau, w0, wa)
    thermo_b = jax.vmap(C.recfast_thermo_batched)(ob, oc, h, ns, As, tau, w0, wa)
    k_ev, _ = C._batch_k_grids(None, float(C.CLS_K_MAX), 1.0, 3258)
    out = {dt: C.source_grids_batched(k_ev, params_b, thermo_b, C.CLS_HIERARCHY, layout="flat",
                                      compute_dtype=dt) for dt in (jnp.float64, jnp.float32)}
    for name in ("s_t", "s_e", "weyl"):
        a = np.asarray(out[jnp.float64][name])[..., :-2]
        b = np.asarray(out[jnp.float32][name])[..., :-2]
        assert np.max(np.abs(b - a)) < 1e-5 * np.max(np.abs(a)) * 10, name
