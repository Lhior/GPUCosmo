"""Cobaya bridge (``gpucosmo.cobaya_bridge``) and the pipeline's handling of non-differentiable
likelihoods. CPU only; skipped where Cobaya or its data are not installed."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

cobaya = pytest.importorskip("cobaya")


def _bridge(name, **kw):
    from gpucosmo.cobaya_bridge.bridge import BridgeUnsupported, CobayaLikelihood

    try:
        return CobayaLikelihood(name, **kw)
    except BridgeUnsupported as e:
        if "could not build" in str(e):
            pytest.skip(f"{name} not installed: {e}")
        raise


def _cosmo(h=0.6736, omch2=0.12):
    from gpucosmo.jax.background.cosmology import load_background_tables
    from gpucosmo.likelihoods.base import Cosmology

    p = dict(h=h, ombh2=0.02237, omch2=omch2, ns=0.9649, As=2.1e-9, tau=0.0544, w0=-1.0, wa=0.0)
    return Cosmology({k: jnp.asarray(v) for k, v in p.items()} | {"mu0": None, "Sigma0": None},
                     load_background_tables())


# ---- nondiff helpers -------------------------------------------------------------------------
def test_fd_term_gradient_is_the_central_difference_and_reverse_mode_works():
    from gpucosmo.cobaya_bridge.nondiff import fd_term

    def f_host(x):
        x = np.asarray(x)
        return np.asarray(-0.5 * np.sum((x - 1.0) ** 2 * np.arange(1, x.size + 1)) + np.sin(x[0]))

    def f(x):
        return jax.pure_callback(f_host, jax.ShapeDtypeStruct((), jnp.float64), x, vmap_method="sequential")

    x = jnp.asarray([0.3, -0.2, 2.0])
    g = jax.grad(lambda z: fd_term(f, z, [0, 1, 2], [1e-4] * 3))(x)
    exact = -(np.asarray(x) - 1.0) * np.arange(1, 4) + np.array([np.cos(0.3), 0, 0])
    np.testing.assert_allclose(np.asarray(g), exact, rtol=1e-6, atol=1e-7)
    # only the selected coordinates get a derivative
    g2 = jax.grad(lambda z: fd_term(f, z, [1], [1e-4]))(x)
    assert g2[0] == 0.0 and g2[2] == 0.0 and abs(g2[1] - exact[1]) < 1e-6


def test_fit_quadratic_is_exact_for_a_quadratic():
    from gpucosmo.cobaya_bridge.nondiff import fit_quadratic

    rng = np.random.default_rng(1)
    A = rng.normal(size=(3, 3))
    H = -(A @ A.T + np.eye(3))
    x0 = np.array([0.1, 0.5, -1.0, 7.0])
    b = np.array([0.3, -0.2, 0.05])

    def f_batch(X):
        d = X[:, :3] - x0[:3]
        return 2.0 + d @ b + 0.5 * np.einsum("ni,ij,nj->n", d, H, d)

    q = fit_quadratic(f_batch, x0, [0, 1, 2], [0.1, 0.1, 0.1])
    pts = x0 + rng.normal(size=(5, 4))
    np.testing.assert_allclose(np.asarray(jax.vmap(q)(jnp.asarray(pts))), f_batch(pts), rtol=1e-8, atol=1e-8)


# ---- bridge conventions ----------------------------------------------------------------------
@pytest.mark.parametrize("name", ["bao.desi_dr2", "sn.desy5"])
def test_bridge_with_camb_products_reproduces_cobaya_camb(name):
    """The bridge's provider conventions: CAMB's own products through BridgeTheory give Cobaya +
    CAMB's log L exactly."""
    pytest.importorskip("camb")
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "validate_cobaya_bridge.py"
    spec = importlib.util.spec_from_file_location("validate_cobaya_bridge", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["validate_cobaya_bridge"] = mod
    spec.loader.exec_module(mod)
    _bridge(name)  # skips when not installed
    r = mod.run(name, ours=True)
    assert abs(r["diff"]) < 1e-9, r
    assert abs(r["diff_ours_vs_camb"]) < 5e-3, r  # our background vs CAMB's


def test_bridge_background_products_and_requirements():
    L = _bridge("bao.desi_dr2")
    assert L.params == {} and L.requires == ()
    th = L.theory(_cosmo())
    assert {"angular_diameter_distance", "Hubble", "params"} <= set(th)
    assert "rdrag" in th["params"]
    assert np.all(np.diff(L._req["z"]["Hubble"]) > 0)  # ascending pool


def test_bridge_loglike_vmaps_and_rejects_failures():
    from gpucosmo.likelihoods.base import LikelihoodInputs

    L = _bridge("bao.sdss_dr12_consensus_bao")
    c1, c2 = _cosmo(), _cosmo(h=0.70)
    v1 = float(L.loglike(LikelihoodInputs(L.theory(c1), {}, c1, {})))
    v2 = float(L.loglike(LikelihoodInputs(L.theory(c2), {}, c2, {})))
    th = jax.tree_util.tree_map(lambda a, b: jnp.stack([a, b]), L.theory(c1), L.theory(c2))
    vb = jax.vmap(lambda t: L.loglike(LikelihoodInputs(t, {}, c1, {})))(th)
    np.testing.assert_allclose(np.asarray(vb), [v1, v2], rtol=1e-12)
    bad = jax.tree_util.tree_map(lambda a: a * jnp.nan, L.theory(c1))
    assert float(L.loglike(LikelihoodInputs(bad, {}, c1, {}))) == -np.inf


def test_unsupported_requirements_are_named():
    from gpucosmo.cobaya_bridge.bridge import BridgeUnsupported, CobayaLikelihood

    try:
        CobayaLikelihood("des_y1.shear")
    except BridgeUnsupported as e:
        if "could not build" in str(e):
            pytest.skip("DES Y1 not installed")
        assert "Pk_grid" in str(e) or "non-linear" in str(e)
    else:  # the theory gained non-linear P(k): nothing to check
        pass


# ---- pipeline ----------------------------------------------------------------------------------
YAML_BG = """
theory: {camb: {extra_args: {}}}
likelihood:
  bao.desi_dr2:
  sn.pantheonplus:
  H0.riess2020:
params:
  H0: {prior: {min: 40, max: 100}, ref: 68.}
  ombh2: {prior: {min: 0.005, max: 0.04}, ref: 0.0224}
  omch2: {prior: {min: 0.05, max: 0.3}, ref: 0.12}
  logA: {prior: {min: 1.61, max: 3.91}, ref: 3.05}
  ns: {prior: {min: 0.8, max: 1.2}, ref: 0.965}
  mnu: 0.06
"""


@pytest.fixture(scope="module")
def bg_bundle(tmp_path_factory):
    import yaml

    from gpucosmo.pipeline import build_analysis_bundle

    _bridge("bao.desi_dr2")
    info = yaml.safe_load(YAML_BG)
    b, res, diffs = build_analysis_bundle(config=info, verbose=False)
    return b, res


def test_yaml_routes_cobaya_names_to_ports_or_bridge(bg_bundle):
    b, res = bg_bundle
    kinds = {L.label: type(L).__name__ for L in b.extra_likelihoods}
    # BAO / H0 have native ports (registered under their Cobaya names); SN is built in.
    assert kinds == {"bao.desi_dr2": "BAO", "H0.riess2020": "H0"}
    assert "sn" in b.likelihood_label


def test_bridged_and_native_agree_inside_the_bundle(bg_bundle):
    """The same yaml with the bridge forced: same exact log posterior."""
    from gpucosmo.cobaya_bridge.bridge import CobayaLikelihood
    from gpucosmo.jax.inference.mg_bundle import build_mg_bundle

    b, res = bg_bundle
    x = jnp.asarray(b.fiducial)[None].repeat(2, 0).at[1, 0].add(1.5)
    lp_native = np.asarray(b.log_post_batch_fn(x))
    bridged = [CobayaLikelihood(L.label) for L in b.extra_likelihoods]
    b2 = build_mg_bundle(b.config, priors=res.priors, likelihoods=bridged)
    lp_bridge = np.asarray(b2.log_post_batch_fn(x))
    np.testing.assert_allclose(lp_bridge, lp_native, rtol=0, atol=1e-9)
    # gradient: finite differences through the callbacks = autodiff through the ports
    th = lambda z, bb=b2: {**bb.theory_exact(z[:bb.n_theory]), "aux": bb.aux}  # noqa: E731
    g_b = jax.grad(lambda z: b2.log_like_from_theory_fn(z, th(z), nondiff="skip") + b2.nondiff_loglike(z, th))(x[0])
    g_n = jax.grad(lambda z: b.log_like_from_theory_fn(z, {**b.theory_exact(z[:b.n_theory]), "aux": b.aux}))(x[0])
    np.testing.assert_allclose(np.asarray(g_b), np.asarray(g_n), rtol=1e-4, atol=1e-3)


def test_resolver_reroutes_incomplete_builtin_sets():
    from gpucosmo.jax.inference.cobaya_yaml import resolve_cobaya

    info = {"theory": {"camb": {"extra_args": {"lens_potential_accuracy": 1}}},
            "likelihood": {"planck_2018_lowl.TT_clik": None, "planck_2018_lensing.native": None},
            "params": {"logA": {"prior": {"min": 1.61, "max": 3.91}}, "ns": {"prior": {"min": 0.8, "max": 1.2}},
                       "theta_MC_100": {"prior": {"min": 0.5, "max": 10}},
                       "ombh2": {"prior": {"min": 0.005, "max": 0.04}},
                       "omch2": {"prior": {"min": 0.05, "max": 0.3}}, "tau": {"prior": {"min": 0.01, "max": 0.8}},
                       "mnu": 0.06}}
    try:
        res = resolve_cobaya(info)
    except Exception as e:  # noqa: BLE001
        if "could not build" in str(e) or "not installed" in str(e):
            pytest.skip(str(e)[:200])
        raise
    assert res.config.cmb is None and res.config.cmb_theory
    assert {L.label for L in res.extra_likelihoods} == {"planck_2018_lowl.TT_clik", "planck_2018_lensing.native"}
    assert any("not part of a built-in" in n for n in res.notes)
    assert "A_planck" in res.priors  # shared by both, Cobaya defaults


@pytest.mark.slow
def test_pk_block_in_the_jitted_bundle_matches_the_eager_theory():
    """A likelihood reading f σ8 (SDSS DR12 consensus final, native port) through the bundle's
    jitted, walker-batched path and the eager theory: same log L; the port and the bridge agree."""
    from gpucosmo.cobaya_bridge.bridge import CobayaLikelihood
    from gpucosmo.jax.inference.mg_bundle import MGConfig, build_mg_bundle
    from gpucosmo.likelihoods.cobaya import ported

    name = "bao.sdss_dr12_consensus_final"
    _bridge(name)
    cfg = MGConfig(desi_fs=False, cmb=None, mg=False, w0wa=False)
    b = build_mg_bundle(cfg, likelihoods=[ported()[name](name)])
    assert b.config.pk_z and "pk" in b.extra_likelihoods[0].requires
    x = jnp.asarray(b.fiducial)[None].repeat(2, 0).at[1, 2].add(0.004)
    lp = np.asarray(b.log_post_batch_fn(x))
    eager = [float(b.log_like_from_theory_fn(xi, {**b.theory_exact(xi[:b.n_theory]), "aux": b.aux})
                   + b.log_prior_fn(xi)) for xi in x]
    # float64 on CPU; on accelerators the P(k) modes evolve in float32 (linear_pk_jax.LINPK_EVOLVE_DTYPE):
    # the batched and eager programs then differ by ~1e-6 in log L
    tol = 1e-8 if jax.default_backend() == "cpu" else 1e-5
    np.testing.assert_allclose(lp, eager, rtol=0, atol=tol)
    b2 = build_mg_bundle(cfg, likelihoods=[CobayaLikelihood(name)])
    np.testing.assert_allclose(np.asarray(b2.log_post_batch_fn(x)), lp, rtol=0, atol=tol)


def test_compressed_dali_fits_quadratic_proxies_for_bridged_likelihoods(bg_bundle):
    """Background analysis with the BAO / H0 likelihoods forced through the bridge: the compressed
    DALI replaces them by quadratic proxies (differentiable), close to the exact bridged
    value near the expansion point, and its total matches the native-port DALI there."""
    from gpucosmo.cobaya_bridge.bridge import CobayaLikelihood
    from gpucosmo.jax.inference.mg_bundle import build_mg_bundle
    from gpucosmo.jax.inference.dali_reduce import CompressedDALI

    b, res = bg_bundle
    bridged = [CobayaLikelihood(L.label) for L in b.extra_likelihoods]
    b2 = build_mg_bundle(b.config, priors=res.priors, likelihoods=bridged)
    x0 = np.asarray(b2.fiducial, dtype=float)
    tay = b2.dali.build_expansion(x0[:b2.n_theory], verbose=False)
    cs = CompressedDALI(b2, tay, x0, log=lambda m: None)
    assert set(cs.proxies) == {L.label for L in bridged}
    lab = "bao.desi_dr2"
    x1 = x0.copy()
    x1[0] += 0.5  # H0 + 0.5
    parts = cs.components(jnp.asarray(x1), cs.data)
    exact = float(b2.log_like_from_theory_fn(jnp.asarray(x1), {**b2.theory_exact(jnp.asarray(x1[:b2.n_theory])),
                                                               "aux": b2.aux}, components=True)[lab])
    assert abs(float(parts[lab]) - exact) < 0.05 * max(1.0, abs(exact))
    g = jax.grad(lambda z: cs.loglike(z, cs.data))(jnp.asarray(x1))
    assert np.all(np.isfinite(np.asarray(g)))
    # the native ports' compressed DALI likelihood (exact in the BAO / H0 theory outputs) at x0
    tay_n = b.dali.build_expansion(x0[:b.n_theory], verbose=False)
    cs_n = CompressedDALI(b, tay_n, x0, log=lambda m: None)
    assert abs(float(cs.loglike(jnp.asarray(x0), cs.data)) - float(cs_n.loglike(jnp.asarray(x0), cs_n.data))) < 1e-6


def test_shared_cmb_block_always_has_cl():
    """Spectra to ℓ 2508 without the extended block (e.g. a low-ℓ likelihood alone) still reach
    user / Cobaya likelihoods as ``cl``; the extended block, when computed, takes precedence."""
    from gpucosmo.jax.inference.mg_bundle import shared_block

    dl = {k: jnp.arange(2509.0) * (i + 1) for i, k in enumerate(("dl_tt", "dl_te", "dl_ee"))}
    cl = shared_block("cmb", {"cmb": dl})["cl"]
    assert set(cl) == {"ell", "tt", "te", "ee"} and float(cl["ee"][10]) == 30.0
    x = jnp.ones((5, 3000))
    cl2 = shared_block("cmb", {"cmb": {**dl, "cmb_x": x}})["cl"]
    assert cl2["bb"].shape == (3000,) and abs(float(cl2["pp"][5]) - 2 / np.pi) < 1e-15
