#!/usr/bin/env python3
"""RSA / TimeSteps drop-time smoke for CAMB-faithful photon truncation."""

from __future__ import annotations

import numpy as np


def test_phot_rsa_kmin_gate_logic():
    """Large-scale k never RSA photons; small-scale k does after max(15/k, taurend)."""
    from gpucosmo.jax.eb.evolve_jax import PHOT_RSA_KMIN

    assert PHOT_RSA_KMIN == 0.03
    taurend, mvd, tau0 = 600.0, 400.0, 14000.0

    def drops(k: float):
        t_nu = max(15.0 / k, min(taurend, mvd))
        t_phot_cand = max(15.0 / k, taurend)
        t_phot = t_phot_cand if k > PHOT_RSA_KMIN else tau0
        t_phot = max(t_phot, t_nu)
        return t_nu, t_phot

    t_nu, t_phot = drops(0.01)  # ℓ~150
    assert t_phot >= tau0 - 1e-9  # never truncates photons
    assert t_nu < t_phot

    t_nu, t_phot = drops(0.1)
    assert t_phot < tau0
    assert t_phot == max(15.0 / 0.1, taurend)
    assert t_nu <= t_phot


def test_composite_tau_grid_length_and_order():
    import jax.numpy as jnp
    from gpucosmo.jax.eb.evolve_jax import _composite_tau_grid

    n_steps = 128
    taus = np.asarray(
        _composite_tau_grid(0.1, 50.0, 400.0, 800.0, 14000.0, n_steps)
    )
    assert taus.shape == (n_steps,)
    assert np.all(np.diff(taus) > 0.0)
    assert taus[0] >= 0.1 - 1e-12
    assert taus[-1] <= 14000.0 + 1e-6
