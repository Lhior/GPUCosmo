#!/usr/bin/env python3
"""CMB theory_backend seam: jax_eb default, cosmopower available."""

from __future__ import annotations

import inspect

import jax.numpy as jnp
import pytest

from gpucosmo.jax.cmb.bundle import build_desi_full_gpu_bundle
from gpucosmo.jax.cmb.theory import get_theory_spectra


def test_bundle_default_theory_backend_is_jax_eb():
    sig = inspect.signature(build_desi_full_gpu_bundle)
    assert sig.parameters["theory_backend"].default == "jax_eb"


def test_unknown_theory_backend_raises():
    from gpucosmo.jax.background.cosmology import load_background_tables

    tables = load_background_tables()
    kwargs = dict(
        logA=jnp.asarray(3.044),
        ns=jnp.asarray(0.9649),
        theta_mc_100=jnp.asarray(1.0409),
        ombh2=jnp.asarray(0.02237),
        omch2=jnp.asarray(0.1200),
        tau=jnp.asarray(0.0544),
        w0=jnp.asarray(-1.0),
        wa=jnp.asarray(0.0),
        lmax=64,
    )
    with pytest.raises(ValueError, match="Unknown theory_backend"):
        get_theory_spectra(tables, theory_backend="spline", **kwargs)  # type: ignore[arg-type]


@pytest.mark.slow
def test_jax_eb_theory_spectra_finite():
    from gpucosmo.jax.background.cosmology import load_background_tables

    tables = load_background_tables()
    spec = get_theory_spectra(
        tables,
        logA=jnp.asarray(3.044),
        ns=jnp.asarray(0.9649),
        theta_mc_100=jnp.asarray(1.0409),
        ombh2=jnp.asarray(0.02237),
        omch2=jnp.asarray(0.1200),
        tau=jnp.asarray(0.0544),
        w0=jnp.asarray(-0.83),
        wa=jnp.asarray(-0.6),
        lmax=128,
        theory_backend="jax_eb",
    )
    assert jnp.isfinite(spec.H0)
    assert jnp.isfinite(spec.dl_tt[100])
    assert jnp.isfinite(spec.dl_kk[50])
