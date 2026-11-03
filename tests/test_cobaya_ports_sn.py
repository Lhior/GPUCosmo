"""Native ports of Cobaya's supernova (``sn.*``) likelihoods against the Cobaya bridge
(``likelihoods.cobaya.testing.compare``), and their gradients. CPU only; skipped where Cobaya or
the SN data are not installed."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

cobaya = pytest.importorskip("cobaya")

TOL = 1e-8
MB = {"Mb": {"prior": {"min": -20.0, "max": -18.5}, "ref": -19.25}}
# JLA_lite on a 13-point (α, β) grid (the default 149-point grid is 0.65 GB of inverse covariances)
SMALL_GRID = {"marginalize_params": {"marge_steps": 2, "alpha_centre": 0.14, "beta_centre": 3.123,
                                     "step_width_alpha": 0.003, "step_width_beta": 0.04}}


def _port(name, block=None, **kw):
    from gpucosmo.cobaya_bridge.bridge import BridgeUnsupported
    from gpucosmo.likelihoods.cobaya import ported
    from gpucosmo.likelihoods.cobaya.supernovae import SN

    try:
        return ported().get(name, SN)(name, block, **kw)  # sn.jla_lite: ported, not registered
    except BridgeUnsupported as e:
        if name.startswith("sn.") and "could not build" in str(e):  # data not installed
            pytest.skip(f"{name} not installed: {e}")
        raise


def _compare(L, batch, shifts):
    from gpucosmo.likelihoods.cobaya.testing import compare

    r = compare(L, n=2, batch=batch, cosmo_shifts=shifts, n_bridge_timing=1)
    assert np.all(np.isfinite(r["bridge"])), r["bridge"]
    return r


def _grads(L):
    """d log L / d h (through ``theory``) and d log L / d (own parameters) at the start point."""
    from gpucosmo.jax.background.cosmology import load_background_tables
    from gpucosmo.likelihoods.base import Cosmology, LikelihoodInputs
    from gpucosmo.likelihoods.cobaya.testing import reference

    ref = reference()["cosmo"]
    tables = load_background_tables()
    par = {p: jnp.asarray(v.start) for p, v in L.params.items()}

    def f(h, par):
        cosmo = Cosmology({**ref, "h": h}, tables)
        return L.loglike(LikelihoodInputs(L.theory(cosmo), par, cosmo, {}))

    return jax.grad(f, argnums=(0, 1))(jnp.asarray(ref["h"]), par)


# ---- supernovae --------------------------------------------------------------------------------
SN_CASES = [
    ("sn.pantheon", None, 16),
    ("sn.pantheonplus", None, 16),
    ("sn.pantheonplusshoes", None, 16),
    ("sn.union3", None, 16),
    ("sn.desy5", None, 16),
    ("sn.desdovekie", None, 16),
    ("sn.jla", None, 4),
    ("sn.jla_lite", SMALL_GRID, 4),
    ("sn.pantheonplusshoes", {"use_abs_mag": True, "params": MB}, 16),
]


@pytest.mark.parametrize("name,block,batch", SN_CASES,
                         ids=[f"{n}{'-absmag' if b and 'use_abs_mag' in b else ''}" for n, b, _ in SN_CASES])
def test_sn_port_matches_bridge_and_is_differentiable(name, block, batch):
    L = _port(name, block)
    r = _compare(L, batch, [{"h": 0.01}])
    assert r["max_abs_diff"] <= TOL, r
    gh, gp = _grads(L)
    assert np.isfinite(gh) and gh != 0.0
    for p, g in gp.items():
        assert np.isfinite(g) and g != 0.0, p


def test_sn_names_registered():
    from gpucosmo.likelihoods.cobaya import ported
    from gpucosmo.likelihoods.cobaya.supernovae import SN, SN_SETS

    P = ported()
    for m, c in SN_SETS.items():
        assert P[f"sn.{m}"] is SN and P[f"sn.{m}.{c}"] is SN
    assert "sn.jla_lite" not in P  # left to the bridge (memory)


def test_sn_missing_nuisance_parameter_is_rejected():
    # a block ``params:`` replaces JLA's default α, β: Cobaya's logp would fail, the port refuses
    with pytest.raises(NotImplementedError):
        _port("sn.jla", {"use_abs_mag": True, "params": MB})
