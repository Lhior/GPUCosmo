"""Native ports of Cobaya's BAO and H0 / M_b likelihoods vs the Cobaya bridge
(``likelihoods.cobaya.testing.compare``). CPU; skipped without Cobaya or the data."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)
pytest.importorskip("cobaya")

SHIFTS = [{"h": 0.01}, {"omch2": 0.005}]


def _port(name, block=None):
    from gpucosmo.cobaya_bridge.bridge import BridgeUnsupported
    from gpucosmo.likelihoods.cobaya import ported

    try:
        return ported()[name](name, block)
    except BridgeUnsupported as e:
        pytest.skip(f"{name}: {e}")


@pytest.mark.parametrize("name", [
    "bao.desi_dr2", "bao.desi_2024_bao_all", "bao.sdss_dr12_consensus_bao",
    "bao.sdss_dr16_baoplus_lyauto",  # 2D grid
    "bao.sdss_dr16_bao_elg",  # 1D grid
    "bao.sdss_dr7_mgs",  # probability table
    "bao.sdss_dr16_baoplus_elg",  # 3D grid with f σ8
    "bao.sdss_dr12_consensus_final",  # Gaussian with f σ8
    "H0.riess2020", "H0.riess2020Mb",
])
def test_port_matches_bridge(name):
    from gpucosmo.likelihoods.cobaya.testing import compare

    r = compare(_port(name), n=3, cosmo_shifts=SHIFTS, batch=32, n_bridge_timing=2)
    assert r["max_abs_diff"] < 1e-9, r


def test_bao_gradient_is_finite_and_matches_finite_differences():
    from gpucosmo.likelihoods.cobaya.testing import inputs, reference

    L = _port("bao.desi_dr2")
    ref = reference()

    def f(dh):
        inp = inputs(L, ref, {}, {"h": dh})
        return L.loglike(inp)

    g = float(jax.grad(f)(jnp.asarray(0.0)))
    eps = 1e-5
    fd = (float(f(jnp.asarray(eps))) - float(f(jnp.asarray(-eps)))) / (2 * eps)
    assert np.isfinite(g) and abs(g - fd) < 1e-4 * max(1.0, abs(fd))


def test_registry_lists_ports_under_cobaya_names():
    from gpucosmo.likelihoods.registry import available_likelihoods, get_likelihood_class

    assert get_likelihood_class("bao.desi_dr2").__name__ == "BAO"
    assert "bao.sixdf_2011_bao" in available_likelihoods()
