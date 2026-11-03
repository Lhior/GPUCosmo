"""GPU batch sizing (inference.autosize) and the batched CMB spectra it sizes."""

from __future__ import annotations

import numpy as np
import pytest

from gpucosmo.jax.inference.autosize import candidate_batches


def test_candidates_divide_chains_largest_first():
    c = candidate_batches(8, cmb=True, fs=True)
    assert c[0] == (8, 8, "flat") and c[1] == (8, 8, "w_outer")
    assert all(8 % cb == 0 and 8 % pb == 0 for cb, pb, _ in c)
    # Memory-starved CMB batches keep ≥ 4 P(k) walkers.
    assert (1, 4, "flat") in c and (2, 4, "flat") in c and (1, 4, None) in c
    assert [cb for cb, _, _ in c] == sorted([cb for cb, _, _ in c], reverse=True)


def test_candidates_without_cmb_only_size_pk():
    assert candidate_batches(4, cmb=False, fs=True) == [(1, 4, "flat"), (1, 2, "flat"),
                                                       (1, 1, "flat")]


@pytest.mark.slow
@pytest.mark.parametrize("layout", ["flat", "w_outer"])
def test_batched_cmb_spectra_match_single(layout):
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    from gpucosmo.jax.background.cosmology import load_background_tables
    from gpucosmo.jax.cmb.jax_eb_theory import (
        get_theory_spectra_jax_eb,
        get_theory_spectra_jax_eb_batch,
    )

    t = load_background_tables()
    ob, mu, sg = jnp.array([0.0222, 0.0226]), jnp.array([0.0, 0.3]), jnp.array([0.1, 0.4])
    fixed = dict(logA=3.04, ns=0.9665, theta_mc_100=1.0411, omch2=0.1193, tau=0.054, w0=-1.0, wa=0.0)
    batch = get_theory_spectra_jax_eb_batch(
        t, ombh2=ob, mu0=mu, Sigma0=sg, nonlinear_lens=False, layout=layout,
        **{k: jnp.full(2, v) for k, v in fixed.items()})
    for i in range(2):
        one = get_theory_spectra_jax_eb(t, ombh2=ob[i], mu0=mu[i], Sigma0=sg[i],
                                        nonlinear_lens=False, k_chunk=0, **fixed)
        for name in ("dl_tt", "dl_ee", "dl_te"):
            a, b = np.asarray(getattr(batch, name))[i, 2:], np.asarray(getattr(one, name))[2:]
            # Batching changes the reduction order: 2e-9 of the amplitude at a TE zero in float64 (#15);
            # on accelerators the sources are float32 (cls_jax.CLS_SOURCES_DTYPE): 8e-6 relative,
            # 6e-7 of the amplitude at the TE zero near ℓ = 50
            cpu = jax.default_backend() == "cpu"
            rtol, atol = (1e-6, 1e-8) if cpu else (3e-5, 3e-6)
            np.testing.assert_allclose(a, b, rtol=rtol, atol=atol * np.abs(b).max())


def test_autosize_falls_back_to_smallest_when_nothing_fits(monkeypatch):
    from gpucosmo.jax.inference import autosize as A
    from gpucosmo.jax.inference.mg_bundle import MGConfig

    monkeypatch.setattr(A, "CACHE", A.CACHE.parent / "autosize_test_fallback.json")
    cfg = MGConfig(desi_fs=True, cmb="pr3")
    out = A.autosize_batches(lambda c: None, cfg, analysis="t", n_chains=8,
                             resident_bytes=1 << 62, verbose=False)  # nothing can fit
    smallest = A.candidate_batches(8, cmb=True, fs=True)[-1]
    assert (out.cmb_walker_batch, out.pk_walker_batch, out.cmb_layout) == smallest
    assert smallest == (1, 4, None)
