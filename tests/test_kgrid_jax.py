#!/usr/bin/env python3
"""CAMB SetkValuesForSources-style Evolve_q grid smoke tests."""

from __future__ import annotations

import numpy as np


def test_set_k_values_piecewise_not_geomspace():
    from gpucosmo.jax.eb.kgrid_jax import set_k_values_for_sources

    k = set_k_values_for_sources(
        1e-4,
        0.45,
        280.0,
        14000.0,
        source_accuracy_boost=1.0,
        accuracy_target=1,
    )
    assert k.ndim == 1 and k.size > 48
    assert np.all(np.diff(k) > 0.0)
    assert k[0] >= 1e-4 - 1e-15
    assert k[-1] <= 0.45 + 1e-12
    # Low-k roughly log-spaced; mid-k more linear (dk ~ const)
    dlnk_lo = np.diff(np.log(k[:12])).mean()
    dk_mid = np.diff(k[len(k) // 3 : 2 * len(k) // 3]).mean()
    dlnk_mid = np.diff(np.log(k[len(k) // 3 : 2 * len(k) // 3])).mean()
    assert dlnk_lo < 0.12
    assert dk_mid > 0.0
    # Mid band should not be pure log with same dlnk as low-k
    assert abs(dlnk_mid - dlnk_lo) > 0.01 or dk_mid < k[len(k) // 2] * dlnk_lo * 0.5


def test_boost_increases_density():
    from gpucosmo.jax.eb.kgrid_jax import set_k_values_for_sources

    k1 = set_k_values_for_sources(1e-4, 0.45, 280.0, 14000.0, source_accuracy_boost=1.0)
    k15 = set_k_values_for_sources(1e-4, 0.45, 280.0, 14000.0, source_accuracy_boost=1.5)
    assert k15.size > k1.size
    assert k1[-1] >= 0.4  # Silk reach past q_cmb


def test_tranges_add_delta_counting_and_dlnk():
    from gpucosmo.jax.eb.tranges_jax import TRanges, dlnk_from_points

    r = TRanges()
    r.init()
    r.add_delta(1e-4, 0.01, 0.075, is_log=True)
    r.add_delta(0.01, 0.1, 0.002, is_log=False)
    pts = r.get_array(want_dpoints=True)
    assert pts.size >= 8
    assert np.all(np.diff(pts) > 0.0)
    dlnk = dlnk_from_points(pts)
    assert dlnk.shape == pts.shape
    assert np.all(np.isfinite(dlnk))
    assert float(dlnk.sum()) > 1.0


def test_time_steps_fixed_n_out_and_rec_dense():
    from gpucosmo.jax.eb.kgrid_jax import set_time_steps_for_sources

    # CAMB-like wide recombination window + max_eta_k→dtaurec
    tau = set_time_steps_for_sources(
        206.0,
        600.0,
        14176.0,
        time_step_boost=1.0,
        reion_tau_start=3840.0,
        reion_tau_complete=5740.0,
        n_out=640,
    )
    assert tau.shape == (640,)
    assert tau[0] >= 206.0 - 1e-9
    assert tau[-1] <= 14176.0 + 1e-6
    assert np.all(np.diff(tau) > 0.0)
    rec = (tau > 206.0) & (tau < 600.0)
    late = (tau > 10000.0) & (tau < 14176.0)
    assert rec.sum() > 50 and late.sum() > 5
    assert float(np.median(np.diff(tau[rec]))) < float(np.median(np.diff(tau[late])))


def test_camb_like_taurst_taurend_from_visibility():
    from gpucosmo.jax.eb.kgrid_jax import camb_like_taurst_taurend

    tau = np.linspace(50.0, 14000.0, 2000)
    peak = 280.0
    g = np.exp(-0.5 * ((tau - peak) / 25.0) ** 2)
    g = g / np.trapezoid(g, tau)
    taurst, taurend, tmax = camb_like_taurst_taurend(tau, g)
    assert taurst < peak < taurend
    assert abs(tmax - peak) < 5.0
    assert taurend - taurst > 40.0


def test_sync_thermo_taurend_near_camb():
    """Recfast + visibility → taurend ~599 (CAMB), not opacity heuristic ~316."""
    import jax.numpy as jnp

    from gpucosmo.jax.eb.cls_jax import _recfast_thermo_sg
    from gpucosmo.jax.eb.evolve_jax import PHOT_RSA_KMIN

    th = _recfast_thermo_sg(
        jnp.asarray(0.02237),
        jnp.asarray(0.12),
        jnp.asarray(0.673305),
        jnp.asarray(0.9649),
        jnp.asarray(np.exp(3.036) * 1e-10),
        jnp.asarray(0.0544),
        jnp.asarray(-1.0),
        jnp.asarray(0.0),
    )
    taurend = float(th["taurend"])
    assert 550.0 <= taurend <= 650.0, f"taurend={taurend}"
    # Mid-k phot RSA drop uses max(15/k, taurend) ≈ taurend when k=0.05
    k = 0.05
    assert k > PHOT_RSA_KMIN
    t_phot = max(15.0 / k, taurend)
    assert t_phot >= 550.0


def test_max_etak_scalar_camb_formula():
    from gpucosmo.jax.eb.kgrid_jax import max_etak_scalar

    # AccuracyTarget Max_eta_k=72000 → boost * max(1700, 72000) / 20 = 3600
    assert abs(max_etak_scalar() - 3600.0) < 1e-9
    assert abs(max_etak_scalar(5000.0) - 250.0) < 1e-9
