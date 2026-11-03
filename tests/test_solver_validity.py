#!/usr/bin/env python3
"""Solver validity reporting (issue #11): a numerical failure of the CMB perturbation solve is
reported (``solver_status``) and, by default, rejected (NaN spectra → log L = −∞) instead of being
turned into finite, plausible but wrong spectra (a DALI expansion built on such spectra stops:
``tests/test_dali_expansion_and_pr4.py``)."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy
jax.config.update("jax_enable_x64", True)

from gpucosmo.jax.eb import cls_jax as C  # noqa: E402

N_TAU = 32


def _sources(st_scale=0.1, weyl_scale=0.3):
    tau = jnp.linspace(0.0, 1.0, N_TAU)
    return st_scale * jnp.sin(7 * tau), 0.5 * st_scale * jnp.cos(5 * tau), weyl_scale * jnp.exp(-tau)


@pytest.mark.parametrize("policy", C.SOURCE_FAILURE_POLICIES)
def test_valid_mode_untouched(policy):
    s_t, s_e, w = _sources()
    o_t, o_e, o_w, failed = C._check_mode_sources(s_t, s_e, w, policy=policy)
    assert float(failed) == 0.0
    for a, b in ((o_t, s_t), (o_e, s_e), (o_w, w)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize("case", ["nan_t", "inf_e", "nan_weyl", "runaway_t", "runaway_weyl"])
def test_pathological_mode_flagged(case):
    s_t, s_e, w = _sources()
    if case == "nan_t":
        s_t = s_t.at[3].set(jnp.nan)
    elif case == "inf_e":
        s_e = s_e.at[5].set(jnp.inf)
    elif case == "nan_weyl":
        w = w.at[7].set(jnp.nan)
    elif case == "runaway_t":
        s_t = s_t.at[4].set(10 * C.CLS_ST_MAX_ABS)
    else:
        w = w.at[2].set(1e32)
    for policy in C.SOURCE_FAILURE_POLICIES:
        o_t, o_e, o_w, failed = C._check_mode_sources(s_t, s_e, w, policy=policy)
        assert float(failed) == 1.0, (case, policy)
        outs = np.concatenate([np.asarray(v) for v in (o_t, o_e, o_w)])
        if policy == "keep":  # nothing hidden
            assert not np.isfinite(outs).all() or np.abs(outs).max() > C.CLS_ST_MAX_ABS
        else:  # the rest of the pipeline stays finite (the C_ℓ are rejected later)
            assert np.isfinite(outs).all()
    if case.startswith("runaway"):  # "mask"/"reject" zero the whole failed source, as before
        o_t, o_e, o_w, _ = C._check_mode_sources(s_t, s_e, w, policy="mask")
        zeroed = (o_t, o_e) if case == "runaway_t" else (o_w,)
        assert all(np.all(np.asarray(v) == 0.0) for v in zeroed)


def test_flags_batch_under_jit_vmap_and_grad():
    s_t, s_e, w = _sources()
    st = jnp.stack([s_t, s_t.at[1].set(jnp.nan), s_t.at[2].set(50.0)])
    flags = jax.jit(jax.vmap(lambda a: C._check_mode_sources(a, s_e, w, policy="reject")[3]))(st)
    np.testing.assert_array_equal(np.asarray(flags), [0.0, 1.0, 1.0])

    def loss(scale):  # the check keeps valid points differentiable
        o_t, o_e, o_w, _ = C._check_mode_sources(scale * s_t, s_e, w, policy="reject")
        return jnp.sum(o_t**2) + jnp.sum(o_w)

    g = jax.grad(loss)(1.0)
    assert np.isfinite(float(g)) and abs(float(g) - 2 * float(jnp.sum(s_t**2))) < 1e-12


def test_failure_policy_on_cls():
    cls = {k: jnp.linspace(1.0, 2.0, 10) for k in ("tt", "te", "ee", "pp")}
    cls["ell"] = jnp.arange(10.0)
    ok = C._apply_failure_policy(cls, 0.0, policy="reject")
    assert float(ok["solver_status"]) == 0.0 and bool(C.solver_ok(ok))
    np.testing.assert_array_equal(np.asarray(ok["tt"]), np.asarray(cls["tt"]))
    bad = C._apply_failure_policy(cls, 2.0, policy="reject")
    assert not bool(C.solver_ok(bad))
    assert all(np.isnan(np.asarray(bad[k])).all() for k in ("tt", "te", "ee", "pp"))
    np.testing.assert_array_equal(np.asarray(bad["ell"]), np.arange(10.0))
    masked = C._apply_failure_policy(cls, 2.0, policy="mask")  # finite, but the status says why not to trust it
    assert np.isfinite(np.asarray(masked["tt"])).all() and float(masked["solver_status"]) == 2.0
    cl, n = C._finite_cl(jnp.asarray([1.0, jnp.nan, jnp.inf]), policy="reject")
    assert float(n) == 2.0 and np.isfinite(np.asarray(cl)).all()


def test_policy_from_environment(monkeypatch):
    monkeypatch.delenv("GPUCOSMO_STRICT_SOURCES", raising=False)
    monkeypatch.delenv("GPUCOSMO_SOURCE_FAILURE", raising=False)
    assert C._source_failure_policy() == "reject"
    monkeypatch.setenv("GPUCOSMO_SOURCE_FAILURE", "mask")
    assert C._source_failure_policy() == "mask"
    monkeypatch.setenv("GPUCOSMO_STRICT_SOURCES", "1")  # the older validation switch
    assert C._source_failure_policy() == "keep"
    monkeypatch.delenv("GPUCOSMO_STRICT_SOURCES")
    monkeypatch.setenv("GPUCOSMO_SOURCE_FAILURE", "zero")
    with pytest.raises(ValueError):
        C._source_failure_policy()


@pytest.fixture(scope="module")
def fiducial_inputs():
    fid = dict(ombh2=0.02237, omch2=0.1200, h=0.6736, ns=0.9649, As=2.1e-9, tau_reio=0.0544, w0=-1.0, wa=0.0)
    order = ("ombh2", "omch2", "h", "ns", "As", "tau_reio", "w0", "wa")
    p = [jnp.asarray(fid[k]) for k in order]
    return C._params_dict_cls(*p), C._recfast_thermo_device(*p)


def test_real_integrator_failure_is_reported(fiducial_inputs):
    """A genuine numerical failure, not an injected one: the legacy fixed-step RK4 integrator
    without its k-banded step budgets (device_grids) runs away above k ≈ 0.42/Mpc (Weyl source
    1e32 at 0.45). The mode used to be zeroed silently; now it is flagged."""
    params, thermo = fiducial_inputs
    k = jnp.asarray([0.05, 0.45])
    out = C._source_grid(k, params, thermo, C.CLS_HIERARCHY, device_grids=True,
                         evolve_method="rk4_timesteps", return_status=True)
    assert len(out) == 9
    np.testing.assert_array_equal(np.asarray(out[-1]), [0.0, 1.0])
    # without return_status the public 8-tuple is unchanged
    assert len(C._source_grid(k[:1], params, thermo, C.CLS_HIERARCHY, device_grids=True,
                              evolve_method="rk4_timesteps")) == 8


def test_production_integrator_valid_at_fiducial(fiducial_inputs):
    params, thermo = fiducial_inputs
    k = jnp.asarray([1.0e-4, 0.05, 0.45])
    out = C._source_grid(k, params, thermo, C.CLS_HIERARCHY, device_grids=True,
                         evolve_method=C.CLS_EVOLVE_METHOD, return_status=True)
    np.testing.assert_array_equal(np.asarray(out[-1]), [0.0, 0.0, 0.0])
