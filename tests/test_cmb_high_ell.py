"""High-ℓ lensed CMB: the ``lmax_extra`` > lmax + LENS_MARGIN extension of the jax_eb theory.

* The legacy base path (``accurate=False``): ``lmax_extra`` ≤ 2508 + 750 = 3258 reproduces the
  spectra of the solver before the extension (``tests/data/cmb_base_path_reference.npz``, written on
  the CPU by that code: ``outputs/p26/highl/baseline_dump.py``, with the version-1 background tables
  kept as ``tests/data/w0wa_background_tables_v1.npz``) to 2e-6 (κκ and the extended block:
  2e-5) — bit for bit until the 2026-10-17 solver fixes (thermo tables from CAMB's start time,
  background τ seed with the relativistic massive ν, Kahan-compensated update: ≤ 7e-7; HMcode's
  input table interpolated cubically, not linearly: non-linear κκ 1.2e-5).
* By default every lensed solve runs on the high-ℓ path (``ACCURATE_LENSING``), which is closer to
  converged CAMB at ℓ ≤ 2508 too.
* Past it, lensed TT/EE/BB/TE and κκ to ℓ 9000 agree with CAMB 2.0.4 (AccuracyBoost 2, its
  default lens_potential_accuracy for lmax 9000, lens_output_margin 4000, mead2020 NonLinear_lens;
  ``tests/data/cmb_high_ell_camb_reference.npz`` from ``scripts/validate_cmb_high_ell.py``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")

DATA = Path(__file__).parent / "data"
BASE_REF = DATA / "cmb_base_path_reference.npz"
CAMB_REF = DATA / "cmb_high_ell_camb_reference.npz"
P1 = dict(logA=3.044, ns=0.9649, theta_mc_100=1.04092, ombh2=0.02237, omch2=0.1200, tau=0.0544, w0=-1.0, wa=0.0)
P2 = dict(logA=3.03, ns=0.972, theta_mc_100=1.0410, ombh2=0.0225, omch2=0.118, tau=0.060, w0=-0.9, wa=-0.3)
FIELDS = ("dl_tt", "dl_te", "dl_ee", "dl_kk", "H0", "cmb_ext")


def _setup():
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from gpucosmo.jax.background.cosmology import load_background_tables

    return jnp, load_background_tables()


def _setup_v1():
    """The background tables the base-path reference was written with: version 1 (z*, z_drag fitted
    with CAMB's PRIMAT helium; 2026-10-22 replaced by v2, which moves these spectra by ≤ 8e-5).
    With them the current solver reproduces the reference to ≤ 1e-6."""
    jnp, _ = _setup()
    from gpucosmo.jax.background.cosmology import load_background_tables

    return jnp, load_background_tables(DATA / "w0wa_background_tables_v1.npz")


def _assert_matches_base(sp, ref, tag):
    for f in FIELDS:
        got, want = np.asarray(getattr(sp, f)), ref[f"{tag}_{f}"]
        assert got.shape == want.shape, f
        rtol = 2e-5 if f in ("dl_kk", "cmb_ext") else 2e-6
        # atol: TE and the extended block's rows cross zero
        np.testing.assert_allclose(got, want, rtol=rtol, atol=rtol * np.max(np.abs(want)), err_msg=f)


def test_highl_threshold_and_plan_keep_the_base_grids():
    jax.config.update("jax_enable_x64", True)
    from gpucosmo.jax.cmb.jax_eb_theory import LENS_MARGIN, max_lmax_extra, uses_highl
    from gpucosmo.jax.eb.cls_jax import CLS_K_MAX, _batch_k_grids, _make_ell_sparse, highl_plan

    base = 2508 + LENS_MARGIN
    assert not uses_highl(2508, 0) and not uses_highl(2508, base)
    assert uses_highl(2508, base + 1) and not uses_highl(2508, base + 1, lens_cmb=False)
    assert not uses_highl(2508, 2608, lens_margin=100) and uses_highl(2508, 2609, lens_margin=100)
    assert max_lmax_extra() >= 9000
    plan = highl_plan(base, 9000)
    k_ev, k_int = _batch_k_grids(None, float(CLS_K_MAX), 1.0, base)
    nb = plan["n_base"]
    np.testing.assert_array_equal(np.asarray(plan["k_ev"][:nb]), np.asarray(k_ev))
    np.testing.assert_array_equal(np.asarray(plan["k_int"]), np.asarray(k_int))
    assert list(plan["ell_base"]) == _make_ell_sparse(base)
    k_sorted = np.asarray(plan["k_ev"])[np.asarray(plan["hi_idx"])]
    assert np.all(np.diff(k_sorted) > 0) and k_sorted[-1] >= 2 * plan["l_t"] / 14000 * (1 - 1e-12)
    ell_t = list(plan["ell_base"]) + list(plan["ell_t_hi"])
    assert ell_t == sorted(set(ell_t)) and ell_t[-1] == plan["l_t"] == 9500
    assert plan["l_p"] == 13000 and plan["ell_p_hi"][-1] == plan["l_p"]


def test_lmax_extra_up_to_the_base_range_is_unchanged():
    """With ``accurate=False`` (the legacy base path), lmax_extra = 3258 (the largest the base range
    covers) gives the pre-extension spectra (to 2e-6 / 2e-5 since the 2026-10-17 solver fixes)."""
    if jax.default_backend() != "cpu":
        pytest.skip("reference written on the CPU backend")
    jnp, tables = _setup_v1()
    from gpucosmo.jax.cmb.jax_eb_theory import get_theory_spectra_jax_eb

    ref = np.load(BASE_REF)
    sp = get_theory_spectra_jax_eb(tables, **{k: jnp.asarray(v) for k, v in P1.items()}, lmax=2508,
                                   nonlinear_lens=False, lmax_extra=3258, accurate=False)
    _assert_matches_base(sp, ref, "single_lin_x3258")


@pytest.mark.slow
def test_batch_lmax_extra_up_to_the_base_range_is_unchanged():
    if jax.default_backend() != "cpu":
        pytest.skip("reference written on the CPU backend")
    jnp, tables = _setup_v1()
    from gpucosmo.jax.cmb.jax_eb_theory import get_theory_spectra_jax_eb_batch

    ref = np.load(BASE_REF)
    # One cosmology: on the CPU backend two already exceed 7 GB of host memory.
    b = {k: jnp.asarray([P2[k]]) for k in P2}
    sp = get_theory_spectra_jax_eb_batch(tables, **b, lmax=2508, nonlinear_lens="mead2020", layout="flat",
                                         lmax_extra=3258, mu0=jnp.asarray([0.1]), Sigma0=jnp.asarray([-0.05]),
                                         accurate=False)
    _assert_matches_base(sp, ref, "batch1_flat_nl_mg_x3258")


# Max |relative residual| per band vs the CAMB reference (TE relative to sqrt(TT·EE)); measured
# (GPU, float32 sources): ℓ < 3000 ≤ 2.0e-3 (the unchanged base range: EE/TE at ℓ < 30, BB/κκ at
# ℓ < 500), ℓ ≥ 3000 ≤ 6.2e-4 (TT), 3.7e-4 (BB), 3.3e-4 (TE), 3.1e-4 (EE), κκ ≤ 6.2e-4 at L ≥ 500.
TOL = (((2, 3000), dict(tt=3e-3, ee=3e-3, bb=3e-3, te=3e-3, kk=3e-3)),
       ((3000, 9001), dict(tt=1.5e-3, ee=1.5e-3, bb=1.5e-3, te=1.5e-3, kk=1.5e-3)))


@pytest.mark.slow
def test_high_ell_lensed_spectra_match_camb():
    jnp, tables = _setup_v1()  # the reference's CAMB run took the H0 these tables give
    from gpucosmo.jax.cmb.jax_eb_theory import get_theory_spectra_jax_eb

    ref = np.load(CAMB_REF)
    assert json.loads(str(ref["cosmo"])) == P1
    sp = get_theory_spectra_jax_eb(tables, **{k: jnp.asarray(v) for k, v in P1.items()}, lmax=2508,
                                   nonlinear_lens="mead2020", lmax_extra=9000)
    np.testing.assert_allclose(float(sp.H0), float(ref["H0"]), rtol=1e-8)  # 4e-10 after the 2026-10 θ* solve
    ext = np.asarray(sp.cmb_ext)
    assert ext.shape == (5, 9001)
    ell, lensed, kk = ref["ell"], ref["lensed"], ref["kk"]
    for (lo, hi), tol in TOL:
        m = (ell >= lo) & (ell < hi)
        e = ell[m]
        L = lensed[m]
        res = dict(tt=ext[1, e] / L[:, 0] - 1, ee=ext[2, e] / L[:, 1] - 1, bb=ext[3, e] / L[:, 2] - 1,
                   te=(ext[4, e] - L[:, 3]) / np.sqrt(L[:, 0] * L[:, 1]), kk=ext[0, e] / kk[m] - 1)
        for k, r in res.items():
            worst = float(np.max(np.abs(r)))
            assert worst < tol[k], f"{k} ℓ {lo}–{hi - 1}: max |rel| {worst:.2e} ≥ {tol[k]:.1e}"
    assert sp.dl_tt.shape == (2509,) and sp.dl_kk.shape == (3000,)


def test_accurate_lensing_is_the_default():
    """The default solve runs on the high-ℓ path whatever lmax_extra (``solve_lmax_extra``); the
    legacy base path only with ``accurate=False``."""
    from gpucosmo.jax.cmb.jax_eb_theory import LENS_MARGIN, solve_lmax_extra, uses_highl

    base = 2508 + LENS_MARGIN
    assert solve_lmax_extra(2508, 0) == base + 1 and uses_highl(2508, solve_lmax_extra(2508, 0))
    assert solve_lmax_extra(2508, 2999) == base + 1
    assert solve_lmax_extra(2508, 9000) == 9000
    assert solve_lmax_extra(2508, 2999, accurate=False) == 2999 and not uses_highl(2508, 2999)
    assert solve_lmax_extra(2508, 0, lens_cmb=False) == 0  # unlensed: nothing to extend


@pytest.mark.slow
def test_default_lensing_matches_converged_camb_below_2508():
    """The default (accurate) solve with no extended range requested: lensed TT/EE at ℓ 1000–2508
    within 5e-4 of converged CAMB, mean bias ≤ 2.5e-4 (the legacy base path: +3.5e-4 / +7.8e-4)."""
    from gpucosmo.jax.background.cosmology import load_background_tables
    from gpucosmo.jax.cmb.jax_eb_theory import get_theory_spectra_jax_eb

    z = np.load(Path(__file__).parent / "data" / "cmb_high_ell_camb_reference.npz", allow_pickle=True)
    c = json.loads(str(z["cosmo"]))
    ell, ref = z["ell"], z["lensed"]
    sp = jax.jit(lambda: get_theory_spectra_jax_eb(load_background_tables(), **c, lmax=2508,
                                                   nonlinear_lens="mead2020"))()
    sel = (ell >= 1000) & (ell <= 2508)
    for i, k in ((0, "dl_tt"), (1, "dl_ee")):
        r = np.asarray(getattr(sp, k))[ell[sel]] / ref[sel, i] - 1
        assert np.max(np.abs(r)) < 5e-4 and abs(np.mean(r)) < 2.5e-4, (k, np.max(np.abs(r)), np.mean(r))


def test_highl_plan_builds_for_low_lmax():
    """The default solve's plan for lmax ≲ 1250 (k_top below the base grid's q_cmb): no added
    modes, the base grid covers the projection (#16)."""
    jax.config.update("jax_enable_x64", True)
    from gpucosmo.jax.cmb.jax_eb_theory import solve_lmax_extra
    from gpucosmo.jax.eb.cls_jax import highl_plan

    for lmax in (2, 128, 600, 1000, 1250, 1500, 2508):
        plan = highl_plan(lmax, solve_lmax_extra(lmax, 0))
        k_sorted = np.asarray(plan["k_ev"])[np.asarray(plan["hi_idx"])]
        assert np.all(np.diff(k_sorted) > 0), lmax
        assert k_sorted[-1] >= max(3000.0, 2 * plan["l_t"]) / 14000 * (1 - 1e-12), lmax
    assert highl_plan(128, solve_lmax_extra(128, 0))["k_ev"].shape[0] == plan["n_base"]


@pytest.mark.slow
def test_low_lmax_solve_matches_the_full_range():
    """A default solve at lmax 1000 (no added high-k modes, #16) agrees with the lmax 2508 solve
    below ℓ = 1000 to the ℓ sampling's interpolation: TT 1.4e-4 at ℓ = 1000, ≤ 5e-5 below 900."""
    jnp, tables = _setup()
    from gpucosmo.jax.cmb.jax_eb_theory import get_theory_spectra_jax_eb

    kw = {k: jnp.asarray(v) for k, v in P1.items()}
    lo = jax.jit(lambda: get_theory_spectra_jax_eb(tables, **kw, lmax=1000))()
    hi = jax.jit(lambda: get_theory_spectra_jax_eb(tables, **kw, lmax=2508))()
    for f in ("dl_tt", "dl_ee", "dl_te"):
        a, b = np.asarray(getattr(lo, f))[2:1001], np.asarray(getattr(hi, f))[2:1001]
        scale = np.sqrt(np.asarray(lo.dl_tt)[2:1001] * np.asarray(lo.dl_ee)[2:1001]) if f == "dl_te" else np.abs(b)
        r = np.abs(a - b) / scale
        assert np.max(r) < 2e-4 and np.max(r[:898]) < 1e-4, (f, float(np.max(r)), float(np.max(r[:898])))
