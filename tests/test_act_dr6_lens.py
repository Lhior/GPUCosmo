"""ACT DR6 (+ Planck PR4) lensing likelihood port (``act_dr6_lens_jax``) and its theory input."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DDIR = ROOT / "data/act_dr6_lens/v1.2"
REF_SRC = ROOT / "data/external/act_dr6_lenslike_src/act_dr6_lenslike-1.2.1"
needs_data = pytest.mark.skipif(not (DDIR / "covmat_actplanck.txt").exists(),
                                reason="scripts/fetch_act_dr6_lens_data.sh not run")


def _fiducial_spectra():
    from gpucosmo.jax.cmb.act_dr6_lens_jax import N_ELL, _fiducials

    fid = _fiducials(DDIR)
    ell = np.arange(N_ELL, dtype=float)
    dl = np.stack([fid[s] * ell * (ell + 1) / (2 * np.pi) for s in ("tt", "ee", "bb", "te")])
    return fid["kk"], dl


@needs_data
@pytest.mark.parametrize("variant", ["act_baseline", "actplanck_baseline"])
def test_corrections_vanish_at_the_fiducial_spectra(variant):
    from gpucosmo.jax.cmb.act_dr6_lens_jax import build_act_dr6_lens_loglike, load_act_dr6_lens

    corr = load_act_dr6_lens(variant, lens_only=False, verbose=False)
    plain = load_act_dr6_lens(variant, lens_only=True, verbose=False)  # same binning, no corrections
    kk, dl = _fiducial_spectra()
    assert corr.needs_cmb and not plain.needs_cmb
    assert corr.data.size == plain.data.size == (19 if "planck" in variant else 10)
    r = corr.data - plain.a_kk @ kk
    expect = -0.5 * r @ corr.cinv @ r
    got = float(build_act_dr6_lens_loglike(corr)(kk, dl))
    assert got == pytest.approx(expect, rel=1e-10, abs=1e-10)
    # ... and respond to the CMB: 1 % more lensed TT power shifts ln L.
    got2 = float(build_act_dr6_lens_loglike(corr)(kk, dl * np.array([1.01, 1, 1, 1])[:, None]))
    assert abs(got2 - got) > 1e-3


@pytest.mark.slow
@needs_data
@pytest.mark.skipif(not REF_SRC.exists(), reason="act_dr6_lenslike sdist not unpacked")
def test_matches_reference_act_dr6_lenslike():
    import jax.numpy as jnp

    sys.path.insert(0, str(REF_SRC))
    from act_dr6_lenslike import act_dr6_lenslike as ref

    from gpucosmo.jax.cmb.act_dr6_lens_jax import build_act_dr6_lens_loglike, load_act_dr6_lens

    kk, dl = _fiducial_spectra()
    rng = np.random.default_rng(3)
    kk = kk * (1 + 0.05 * rng.standard_normal(kk.size))
    dl = dl * (1 + 0.02 * rng.standard_normal(dl.shape))
    ell = np.arange(kk.size, dtype=float)
    cl = dl * 2 * np.pi / np.maximum(ell * (ell + 1), 1)
    d = ref.load_data("actplanck_baseline", ddir=str(DDIR) + "/", lens_only=False,
                      like_corrections=True, nsims_act=792.0, nsims_planck=400.0)
    want = ref.generic_lnlike(d, ell, kk, ell, cl[0], cl[1], cl[3], cl[2], 2998)
    got = float(build_act_dr6_lens_loglike(load_act_dr6_lens("actplanck_baseline", False,
                                                             verbose=False))(
        jnp.asarray(kk), jnp.asarray(dl)))
    assert got == pytest.approx(want, rel=1e-10)


def test_lmax_extra_leaves_the_lensed_spectra_below_lmax_unchanged():
    import jax.numpy as jnp

    from gpucosmo.jax.cmb.jax_eb_theory import _spectra_from_cls

    lmax, l_cls = 400, 700
    ell = jnp.arange(l_cls + 1, dtype=jnp.float64)
    x = jnp.maximum(ell, 2.0)
    cls = {"ell": ell, "tt": 1e-9 / x**2, "ee": 1e-20 / x**2, "te": 1e-15 / x**2,
           "pp": 1e-8 / x**4}
    a = _spectra_from_cls(cls, 67.0, lmax=lmax, lens_cmb=True)
    b = _spectra_from_cls(cls, 67.0, lmax=lmax, lens_cmb=True, lmax_extra=600)
    for f in ("dl_tt", "dl_te", "dl_ee", "dl_kk"):
        np.testing.assert_array_equal(np.asarray(getattr(a, f)), np.asarray(getattr(b, f)))
    assert a.cmb_ext is None and b.cmb_ext.shape == (5, 601)
    np.testing.assert_allclose(np.asarray(b.cmb_ext[1, :lmax + 1]), np.asarray(a.dl_tt), rtol=0)
    np.testing.assert_allclose(np.asarray(b.cmb_ext[0, :lmax + 1]), np.asarray(a.dl_kk[:lmax + 1]))
    assert np.all(np.asarray(b.cmb_ext[3, 2:]) > 0)  # lensing B modes
    with pytest.raises(ValueError, match="lmax_extra"):
        _spectra_from_cls(cls, 67.0, lmax=lmax, lens_cmb=True, lmax_extra=800)


def test_candl_act_baseline_lens_only_matches_the_port():
    """The candl path (``cmb.candl_likelihood``) and ``act_dr6_lens_jax`` agree for act_baseline,
    lens_only (candl's theory vector starts at L = 2: it was shifted by two multipoles before)."""
    pytest.importorskip("candl")
    pytest.importorskip("candl_data")
    import jax.numpy as jnp

    from gpucosmo.jax.cmb.act_dr6_lens_jax import build_act_dr6_lens_loglike, load_act_dr6_lens
    from gpucosmo.jax.cmb.candl_likelihood import build_candl_lens_loglike, load_candl_lens_likelihood

    try:
        act = load_act_dr6_lens("act_baseline", True)
    except FileNotFoundError:
        pytest.skip("ACT DR6 lensing data not available")
    L = np.arange(3000.0)
    kk = jnp.asarray(1e-7 * (L * (L + 1)) ** 2 / (2 * np.pi) * 2.0e-9 * np.exp(-L / 900.0) * (L >= 2) + 1e-12)
    ll_candl = float(build_candl_lens_loglike(load_candl_lens_likelihood())(kk))
    ll_port = float(build_act_dr6_lens_loglike(act)(kk))
    assert abs(ll_candl - ll_port) < 1e-6 * max(1.0, abs(ll_port)), (ll_candl, ll_port)
