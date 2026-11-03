"""Native ports of Cobaya's clik-based Planck likelihoods vs the Cobaya bridge. CPU; skipped
without Cobaya, clipy or the data."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

jax.config.update("jax_enable_x64", True)
pytest.importorskip("cobaya")
pytest.importorskip("clipy")


def _port(name):
    from gpucosmo.cobaya_bridge.bridge import BridgeUnsupported
    from gpucosmo.likelihoods.cobaya import ported

    try:
        return ported()[name](name)
    except BridgeUnsupported as e:
        pytest.skip(f"{name}: {e}")


@pytest.mark.parametrize("name", ["planck_2018_lowl.TT_clik", "planck_2018_lowl.EE_clik",
                                  "planck_2018_highl_plik.TT_lite", "planck_2018_lensing.clik",
                                  "planck_2018_highl_plik.EE"])
def test_clik_port_matches_bridge(name):
    from gpucosmo.likelihoods.cobaya.testing import compare

    r = compare(_port(name), n=3, batch=8, n_bridge_timing=2)
    assert r["max_abs_diff"] < 1e-9, r


def test_plik_tt_sz_prior_and_gradient():
    """plik TT carries Cobaya's SZ prior (now in log_prior, differentiable) and the port is
    differentiable in the spectra and the nuisances."""
    from gpucosmo.likelihoods.cobaya.testing import inputs, reference

    L = _port("planck_2018_highl_plik.TT")
    assert L.has_log_prior() and L.bridge._jax_priors and not L.bridge._host_priors
    p = {k: jnp.asarray(v.start) for k, v in L.params.items()}
    lp = float(L.log_prior(p))
    want = -0.5 * ((float(p["ksz_norm"]) + 1.6 * float(p["A_sz"]) - 9.5) / 3.0) ** 2 - np.log(3.0 * np.sqrt(2 * np.pi))
    assert abs(lp - want) < 1e-12
    ref = reference()

    def f(scale, a_planck):
        inp = inputs(L, ref, {**{k: v.start for k, v in L.params.items()}, "A_planck": a_planck})
        cl = dict(inp.shared["cmb"]["cl"])
        cl["tt"] = cl["tt"] * scale
        return L.loglike(inp._replace(shared={"cmb": {**inp.shared["cmb"], "cl": cl}}))

    g = jax.grad(f, argnums=(0, 1))(jnp.asarray(1.0), jnp.asarray(1.0))
    assert all(np.isfinite(float(v)) for v in g)
