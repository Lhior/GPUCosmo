#!/usr/bin/env python3
"""Smoke tests for jax_eb CMB sources (device-safe magnitudes).

Uses the same solver configuration as the DESI chains
(``scripts/run_desi_full_mcmc.py`` -> ``get_theory_spectra_jax_eb`` ->
``compute_cmb_cls(use_recfast_thermo=True, device_grids=True)``): on-device
Recfast thermo, the CAMB-like TimeSteps tau grid of ``_source_grid`` and the
``CLS_EVOLVE_METHOD`` ("segmented") perturbation integrator.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy
jax.config.update("jax_enable_x64", True)

# Planck 2018 best fit with a w0wa point, as sampled by the DESI chains.
FIDUCIAL = dict(
    ombh2=0.02237, omch2=0.1200, h=0.6736, ns=0.9649, As=2.1e-9,
    tau_reio=0.0544, w0=-1.0, wa=0.0,
)


@pytest.fixture(scope="module")
def production_sources():
    """Sources and states on k in {1e-4, 0.05, 0.3} Mpc^-1 (production path)."""
    from gpucosmo.jax.eb.cls_jax import (
        CLS_EVOLVE_METHOD,
        CLS_HIERARCHY,
        _params_dict_cls,
        _recfast_thermo_device,
        _source_grid,
    )

    assert CLS_EVOLVE_METHOD == "segmented"
    p = {key: jnp.asarray(val) for key, val in FIDUCIAL.items()}
    order = ("ombh2", "omch2", "h", "ns", "As", "tau_reio", "w0", "wa")
    params = _params_dict_cls(*(p[key] for key in order))
    thermo = _recfast_thermo_device(*(p[key] for key in order))
    k = jnp.asarray([1.0e-4, 0.05, 0.3])
    tau_src, s_t, s_e, weyl, tau0, _chi_star, tau_peak, y = _source_grid(
        k, params, thermo, CLS_HIERARCHY, device_grids=True,
        evolve_method=CLS_EVOLVE_METHOD, keep_states=True,
    )
    return dict(
        thermo=thermo, k=np.asarray(k), tau=np.asarray(tau_src),
        s_t=np.asarray(s_t), s_e=np.asarray(s_e), weyl=np.asarray(weyl),
        y=np.asarray(y), tau0=float(tau0), tau_peak=float(tau_peak),
    )


def test_thermo_visibility_tables(production_sources):
    thermo = production_sources["thermo"]
    for key in ("visibility", "dvisibility", "ddvisibility", "kappa"):
        assert key in thermo and np.isfinite(np.asarray(thermo[key])).all()
    # Recombination: visibility peaks at tau ~ 280 Mpc for Planck LCDM.
    assert 250.0 < production_sources["tau_peak"] < 310.0


def test_sources_finite_and_bounded(production_sources):
    from gpucosmo.jax.eb.cls_jax import CLS_ST_MAX_ABS, CLS_WEYL_MAX_ABS

    s = production_sources
    assert np.isfinite(s["y"]).all()
    for name in ("s_t", "s_e", "weyl"):
        assert np.isfinite(s[name]).all()
    # _source_grid zeroes a k-mode whose sources run away; none may be zeroed.
    assert np.all(np.max(np.abs(s["s_t"]), axis=1) > 0.0)
    assert np.all(np.max(np.abs(s["weyl"]), axis=1) > 0.0)
    assert np.max(np.abs(s["s_t"])) < CLS_ST_MAX_ABS
    assert np.max(np.abs(s["weyl"])) < CLS_WEYL_MAX_ABS


def test_polarization_source_at_recombination(production_sources):
    s = production_sources
    near = np.abs(s["tau"] - s["tau_peak"]) < 0.2 * s["tau_peak"]
    for ik in (1, 2):  # sub-horizon modes at recombination
        assert np.max(np.abs(s["s_e"][ik, near])) > 0.0


def test_sources_quad_guard_low_k(production_sources):
    """Low-k quadrupole /k^2 pieces must not explode (k = 1e-4 Mpc^-1)."""
    s = production_sources
    assert s["k"][0] == pytest.approx(1.0e-4)
    assert np.max(np.abs(s["s_t"][0])) < 5.0
    assert np.max(np.abs(s["s_e"][0])) < 5.0
