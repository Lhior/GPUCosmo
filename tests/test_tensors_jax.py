"""Primordial tensor modes (``gpucosmo.jax.eb.tensors_jax``).

Fast: r-linearity, inflation-consistency n_t, finiteness and forward-mode derivatives on a
coarse grid. Slow: default accuracy against converged CAMB 2.0.4 (AccuracyBoost 4,
max_eta_k_tensor 6750; ``scripts/build_tensor_camb_reference.py --export-tests``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy
jax.config.update("jax_enable_x64", True)

REF = Path(__file__).parent / "data" / "tensor_cls_camb_reference.npz"
NAMES = ("tt", "ee", "bb", "te")


def _cosmo(case: dict) -> dict:
    return dict(ombh2=case["ombh2"], omch2=case["omch2"], h=case["H0"] / 100.0, ns=case["ns"],
                As=np.exp(case["logA"]) * 1e-10, tau_reio=case["tau"], w0=case["w0"],
                wa=case["wa"])


# DESI DR1 + PR4 ΛCDM mean (the reference cosmology).
LCDM = _cosmo(dict(H0=67.74637, ombh2=0.0222801, omch2=0.1186332, logA=3.054436,
                   ns=0.9694608, tau=0.0614381, w0=-1.0, wa=0.0))


@pytest.fixture(scope="module")
def coarse():
    """A cheap configuration (lmax 100, coarse grids) for the structural tests."""
    from gpucosmo.jax.eb.tensors_jax import TensorAccuracy

    return dict(lmax=100, acc=TensorAccuracy(lmaxt=8, lmaxpolt=8, lmaxnrt=8, n_log=60,
                                             n_pre=60, n_rec=120, n_post=40, late_dtau=20.0,
                                             ell_dense=10, ell_step_lo=6, ell_mid=100))


def test_spectral_index_matches_camb_consistency():
    from gpucosmo.jax.eb.tensors_jax import tensor_spectral_index

    # CAMB InitialPowerLaw.set_params(ns=0.9694608, r=0.1, nt=None): stored in the reference.
    nt, ntrun = tensor_spectral_index(0.1, 0.9694608)
    assert float(nt) == pytest.approx(-0.01272549, abs=1e-8)
    assert float(ntrun) == pytest.approx(-0.00022549, abs=1e-9)
    with pytest.raises(ValueError):
        tensor_spectral_index(0.1, 0.96, None, 0.01)


def test_linear_in_r_finite_and_differentiable(coarse):
    from gpucosmo.jax.eb.cls_jax import _recfast_thermo_device, cmb_params
    from gpucosmo.jax.eb.tensors_jax import (
        tensor_cls,
        tensor_cls_from_transfer,
        tensor_plan,
        tensor_transfer,
    )

    def spectra(x):
        out = tensor_cls(**{**LCDM, "tau_reio": x[1]}, r=x[0], nt=0.0, **coarse)
        return jnp.stack([out[n] for n in NAMES])

    x0 = jnp.array([0.1, LCDM["tau_reio"]])
    f = jax.jit(spectra)
    a = np.asarray(f(x0))
    b = np.asarray(f(x0.at[0].set(0.03)))
    assert np.all(np.isfinite(a))
    assert np.all(a[:3, 2:] > 0.0)
    assert np.all(a[:, :2] == 0.0)
    # Fixed n_t: exactly linear in r.
    np.testing.assert_allclose(b[:, 2:], 0.3 * a[:, 2:], rtol=1e-11, atol=0.0)
    # nt=None is the inflation-consistency n_t(r): same as passing it explicitly (the
    # r-dependence is all in tensor_cls_from_transfer, so reuse one transfer).
    args = [LCDM[k] for k in ("ombh2", "omch2", "h", "ns", "As", "tau_reio", "w0", "wa")]
    params, _ = cmb_params(*args)
    plan = tensor_plan(coarse["lmax"], coarse["acc"])
    delta = jax.jit(lambda p, t: tensor_transfer(p, t, plan))(params, _recfast_thermo_device(*args))
    r, ns = 0.1, LCDM["ns"]
    cons = tensor_cls_from_transfer(delta, plan, r, LCDM["As"], ns)
    expl = tensor_cls_from_transfer(delta, plan, r, LCDM["As"], ns, nt=-r / 8 * (2 - ns - r / 8),
                                    ntrun=r / 8 * (r / 8 + ns - 1))
    zero = tensor_cls_from_transfer(delta, plan, 0.0, LCDM["As"], ns, nt=0.0)
    for n in NAMES:
        np.testing.assert_allclose(np.asarray(cons[n]), np.asarray(expl[n]), rtol=1e-12, atol=0)
        assert np.all(np.asarray(zero[n]) == 0.0)
    # Forward-mode derivatives in (r, τ_reio): finite, d/dr = C/r, d/dτ matches differences.
    jac = np.asarray(jax.jit(jax.jacfwd(spectra))(x0))
    assert np.all(np.isfinite(jac))
    np.testing.assert_allclose(jac[:, 2:, 0], a[:, 2:] / 0.1, rtol=1e-10, atol=0.0)
    h = 1e-4
    fd = (np.asarray(f(x0.at[1].add(h))) - np.asarray(f(x0.at[1].add(-h)))) / (2 * h)
    bb = slice(2, 30)  # reionization bump: strongly τ-dependent
    np.testing.assert_allclose(jac[2, bb, 1], fd[2, bb], rtol=2e-3)


def _residuals(ours: np.ndarray, ref: np.ndarray, lo: int, hi: int) -> dict:
    s = slice(lo, hi)
    out = {n: float(np.max(np.abs(ours[s, i] / ref[s, i] - 1.0))) for i, n in enumerate(NAMES[:3])}
    out["te"] = float(np.max(np.abs(ours[s, 3] - ref[s, 3]) / np.sqrt(ref[s, 0] * ref[s, 1])))
    return out


# Max |residual| per ℓ band (TE in units of sqrt(TT·EE)). Default accuracy reaches ≲ 2e-3 at
# ℓ < 400 and ≲ 9e-3 (BB) at 500–600 (k_max = 4 lmax/τ0); CAMB's own AccuracyBoost 3 vs 4
# scatter in EE / TE at ℓ < 10 is up to 1.3 % / 1.8 %.
BANDS = {(2, 10): 0.02, (10, 20): 5e-3, (20, 400): 5e-3, (400, 601): 1.5e-2}


@pytest.mark.slow
@pytest.mark.parametrize("case", ["lcdm_r0.1", "w0wa_r0.1"])
def test_matches_converged_camb(case):
    from gpucosmo.jax.eb.tensors_jax import tensor_cls

    data = np.load(REF)
    meta = json.loads(str(data["meta"]))[case]["case"]
    ref = data[case]
    out = jax.jit(lambda r: tensor_cls(**_cosmo(meta), r=r, lmax=600))(meta["r"])
    ours = np.stack([np.asarray(out[n]) for n in NAMES], axis=1)
    assert ours.shape == ref.shape
    for (lo, hi), tol in BANDS.items():
        res = _residuals(ours, ref, lo, hi)
        bad = {n: v for n, v in res.items() if v > tol}
        assert not bad, f"{case} ℓ {lo}–{hi - 1}: max residual {bad} > {tol}"


def test_add_tensor_spectra_adds_tensor_dl_to_the_cmb_block():
    """``mg_bundle.add_tensor_spectra``: T_CMB²·(tensor D_ℓ) added to TT/TE/EE and the extended blocks'
    TT/EE/BB/TE rows for ℓ ≤ 600 only; vmapped over cosmologies = one at a time."""
    import jax
    import jax.numpy as jnp

    from gpucosmo.jax.cmb.params import T_CMB_MUK
    from gpucosmo.jax.eb.tensors_jax import tensor_cls
    from gpucosmo.jax.inference.mg_bundle import TENSOR_LMAX, add_tensor_spectra

    c = {k: jnp.asarray(v) for k, v in dict(ombh2=0.02237, omch2=0.12, h=0.6736, ns=0.9649, As=2.1e-9, tau=0.0544,
                                             w0=-1.0, wa=0.0, r=0.1).items()}
    out = {k: jnp.zeros(2509) for k in ("dl_tt", "dl_te", "dl_ee")} | {"cmb_x": jnp.zeros((5, 3000))}
    got = add_tensor_spectra(out, c)
    t = tensor_cls(c["ombh2"], c["omch2"], c["h"], c["ns"], c["As"], c["tau"], c["w0"], c["wa"], c["r"],
                   lmax=TENSOR_LMAX)
    n = TENSOR_LMAX + 1
    np.testing.assert_allclose(np.asarray(got["dl_tt"][:n]), np.asarray(t["tt"]) * T_CMB_MUK**2, rtol=1e-12)
    assert float(jnp.abs(got["dl_tt"][n:]).max()) == 0.0
    np.testing.assert_allclose(np.asarray(got["cmb_x"][3, :n]), np.asarray(t["bb"]) * T_CMB_MUK**2, rtol=1e-12)
    assert float(jnp.abs(got["cmb_x"][0]).max()) == 0.0  # κκ untouched
    c2 = jax.tree_util.tree_map(lambda a: jnp.stack([a, a * 1.0]), c)
    c2["r"] = jnp.asarray([0.1, 0.05])
    outb = jax.tree_util.tree_map(lambda a: jnp.stack([a, a]), out)
    gb = jax.vmap(add_tensor_spectra)(outb, c2)
    one = add_tensor_spectra(out, {**c, "r": jnp.asarray(0.05)})  # n_t depends on r (consistency): not ×½
    np.testing.assert_allclose(np.asarray(gb["dl_ee"][1, :n]), np.asarray(one["dl_ee"][:n]), rtol=1e-9)
    np.testing.assert_allclose(np.asarray(gb["dl_ee"][0, :n]), np.asarray(got["dl_ee"][:n]), rtol=1e-9)
