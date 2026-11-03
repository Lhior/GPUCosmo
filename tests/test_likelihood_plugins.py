#!/usr/bin/env python3
"""User likelihoods (gpucosmo.likelihoods): registry, yaml resolution, bundle, DALI expansions."""

from __future__ import annotations

import dataclasses
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpucosmo.likelihoods import (
    GaussianLikelihood,
    Likelihood,
    Param,
    available_likelihoods,
    get_likelihood_class,
    load_plugin_module,
    register_likelihood,
)
from gpucosmo.likelihoods.examples import BAODistances, GaussianH0

jax.config.update("jax_enable_x64", True)

BBN = {"mean": [0.02196, 3.034], "cov": [[4.0311226e-07, 7.30390042e-05], [7.30390042e-05, 0.0452831584]]}


def _yaml(likelihoods: dict, params: dict | None = None) -> dict:
    """A cheap background-only analysis (BBN + n_s priors, no Boltzmann theory)."""
    return {
        "theory": {"camb": {"extra_args": {"num_massive_neutrinos": 1, "dark_energy_model": "ppf",
                                           "bbn_predictor": "PArthENoPE_880.2_standard.dat"}}},
        "likelihood": {"bbn_likelihoods.schoneberg2024": BBN, "cmb_likelihoods.ns10_planck2018": {}, **likelihoods},
        "params": {"H0": {"prior": {"min": 50, "max": 90}, "ref": 68.0},
                   "ombh2": {"prior": {"min": 0.005, "max": 0.04}, "ref": 0.0224},
                   "omch2": {"prior": {"min": 0.05, "max": 0.3}, "ref": 0.12},
                   "logA": {"prior": {"min": 1.61, "max": 3.91}, "ref": 3.04},
                   "ns": {"prior": {"min": 0.8, "max": 1.2}, "ref": 0.965},
                   "tau": 0.0544, "mnu": 0.06, **(params or {})},
    }


BAO = {"z": [0.51, 0.51, 1.5], "quantities": ["DM_over_rd", "DH_over_rd", "DV_over_rd"],
       "data": [13.6, 21.0, 26.0], "cov": [[0.06, -0.02, 0.0], [-0.02, 0.33, 0.0], [0.0, 0.0, 0.3]]}


@register_likelihood
class ScaledH0(GaussianLikelihood):
    """H0 with a calibration nuisance: not compressible (generic compressed path)."""

    name = "test_scaled_h0"
    params = {"test_cal": Param(0.8, 1.2, gauss=(1.0, 0.02))}

    def __init__(self, **options):
        super().__init__(data=[70.0], cov=[[4.0]], **options)

    def model(self, cosmo):
        return jnp.atleast_1d(cosmo.H0)

    def vector(self, model, params):
        return model * params["test_cal"]

    def log_prior(self, params):  # an extra (non-separable) prior term
        return -0.5 * ((params["test_cal"] - 1.0) / 0.1) ** 2


class ExactDV(Likelihood):
    """No theory: evaluated from the exact background at every point."""

    name = "test_exact_dv"

    def loglike(self, inp):
        return -0.5 * ((inp.cosmo.volume_distance(0.3) / inp.cosmo.rdrag - 8.0) / 0.2) ** 2


def test_registry_lookup():
    avail = available_likelihoods()
    assert {"h0_gaussian", "bao_distances", "hillipop", "desi_fs"} <= set(avail)
    assert get_likelihood_class("h0_gaussian") is GaussianH0
    assert get_likelihood_class("gpucosmo.likelihoods.examples.BAODistances") is BAODistances
    assert get_likelihood_class("gpucosmo.likelihoods.examples:BAODistances") is BAODistances
    assert get_likelihood_class("anything", {"class": "bao_distances"}) is BAODistances
    assert get_likelihood_class("no_such_likelihood") is None
    with pytest.raises(ValueError, match="reserved"):
        register_likelihood(type("Bad", (Likelihood,), {"name": "hillipop"}))
    with pytest.raises(ValueError, match="already registered"):
        register_likelihood(type("Dup", (Likelihood,), {"name": "h0_gaussian"}))


def test_plugin_module_from_file(tmp_path):
    f = tmp_path / "my_plugin.py"
    f.write_text(textwrap.dedent('''
        import jax.numpy as jnp
        from gpucosmo.likelihoods import GaussianLikelihood, register_likelihood

        @register_likelihood
        class FileH0(GaussianLikelihood):
            name = "test_file_h0"
            def __init__(self, mean=70.0, **options):
                super().__init__(data=[mean], cov=[[1.0]], **options)
            def model(self, cosmo):
                return jnp.atleast_1d(cosmo.H0)
    '''))
    load_plugin_module(str(f))
    from gpucosmo.jax.inference.cobaya_yaml import resolve_cobaya

    r = resolve_cobaya(_yaml({"test_file_h0": {"mean": 71.0}}))
    (L,) = r.extra_likelihoods
    assert L.label == "test_file_h0" and float(L.data[0]) == 71.0


def test_entry_points_are_registered(monkeypatch):
    import gpucosmo.likelihoods.registry as reg

    class EP:
        name = "test_ep_like"

        def load(self):
            return type("EPLike", (GaussianH0,), {"name": None, "__module__": "test_ep_module"})

    monkeypatch.setattr("importlib.metadata.entry_points", lambda group: [EP()] if group == reg.ENTRY_POINT_GROUP else [])
    reg.load_entry_points(force=True)
    cls = get_likelihood_class("test_ep_like")
    assert cls is not None and cls.__name__ == "EPLike" and cls.name == "test_ep_like"


def test_yaml_resolution():
    from gpucosmo.jax.inference.cobaya_yaml import UnsupportedCobayaInput, resolve_cobaya

    r = resolve_cobaya(_yaml({"h0_gaussian": {"mean": 73.0, "sigma": 1.0}, "bao_distances": BAO,
                              # a second instance of a class under its own key (Cobaya's ``class:``)
                              "h0_other": {"class": "gpucosmo.likelihoods.examples.GaussianH0", "mean": 67.0},
                              "test_scaled_h0": None},
                             params={"test_cal": {"prior": {"dist": "norm", "loc": 1.0, "scale": 0.05}}}))
    labels = [L.label for L in r.extra_likelihoods]
    assert labels == ["h0_gaussian", "bao_distances", "h0_other", "test_scaled_h0"]
    assert r.likelihoods["ext:h0_other"] == "h0_other"
    assert float(r.extra_likelihoods[2].data[0]) == 67.0
    assert r.priors["test_cal"].gauss == (1.0, 0.05)  # the yaml overrides the class default
    with pytest.raises(UnsupportedCobayaInput, match="not available on the GPU pipeline"):
        resolve_cobaya(_yaml({"my_unknown_likelihood": {}}))


def test_validation_errors():
    from gpucosmo.jax.inference.cobaya_yaml import resolve_cobaya
    from gpucosmo.jax.inference.mg_bundle import build_mg_bundle

    r = resolve_cobaya(_yaml({}))

    class NeedsCMB(Likelihood):
        name = "test_needs_cmb"
        requires = ("cmb",)

    # CMB spectra are computed on demand, in the CMB parameter basis (θ_MC, τ sampled)
    with pytest.raises(ValueError, match="CMB parameter basis"):
        build_mg_bundle(r.config, priors=r.priors, likelihoods=[NeedsCMB()])

    class NeedsFS(Likelihood):
        name = "test_needs_fs"
        requires = ("fs",)

    import dataclasses

    with pytest.raises(ValueError, match="full-shape"):
        build_mg_bundle(dataclasses.replace(r.config, desi_fs=False), priors=r.priors, likelihoods=[NeedsFS()])
    with pytest.raises(ValueError, match="already exists"):
        build_mg_bundle(r.config, priors=r.priors, likelihoods=[
            type("P", (ExactDV,), {"name": "test_p", "params": {"H0": Param(0, 1)}})()])
    with pytest.raises(ValueError, match="used twice"):
        build_mg_bundle(r.config, priors=r.priors, likelihoods=[ExactDV(), ExactDV()])


@pytest.fixture(scope="module")
def bundle():
    from gpucosmo.jax.inference.cobaya_yaml import resolve_cobaya
    from gpucosmo.jax.inference.mg_bundle import build_mg_bundle

    r = resolve_cobaya(_yaml({"h0_gaussian": {"mean": 73.0, "sigma": 1.0}, "bao_distances": BAO}))
    r.add_likelihoods([ScaledH0(), ExactDV()])
    b = build_mg_bundle(r.config, priors=r.priors, likelihoods=r.extra_likelihoods)
    x0 = np.asarray(b.fiducial)
    rng = np.random.default_rng(0)
    X = jnp.asarray(x0 + rng.normal(size=(6, x0.size)) * 0.005 * (np.asarray(b.prior_hi) - np.asarray(b.prior_lo)))
    return b, X


def test_bundle_components_and_priors(bundle):
    b, X = bundle
    assert b.param_names[-1] == "test_cal"
    assert b.likelihood_label.endswith("h0_gaussian+bao_distances+test_scaled_h0+test_exact_dv")
    x = jnp.asarray(b.fiducial)
    th = b.theory_exact(x[:b.n_theory])
    assert set(th["ext"]) == {"h0_gaussian", "bao_distances", "test_scaled_h0"}  # ExactDV has no theory
    comp = b.log_like_from_theory_fn(x, th, components=True)
    assert np.isclose(comp["h0_gaussian"], -0.5 * (73.0 - 68.0) ** 2)
    # the BAO model is the documented distance ratios
    from gpucosmo.likelihoods import Cosmology

    c = Cosmology(b.cosmo(x), b.tables)
    dm = float(c.comoving_distance(0.51) / c.rdrag)
    np.testing.assert_allclose(th["ext"]["bao_distances"]["model"][0], dm, rtol=1e-12)
    # Param Gaussian + the extra log_prior both enter log_prior_fn
    i = b.param_names.index("test_cal")
    x1 = x.at[i].set(1.05)
    d = float(b.log_prior_fn(x1) - b.log_prior_fn(x))
    np.testing.assert_allclose(d, -0.5 * (0.05 / 0.02) ** 2 - 0.5 * (0.05 / 0.1) ** 2, rtol=1e-12)
    lp = np.asarray(b.log_post_batch_fn(X))
    assert np.all(np.isfinite(lp))


def test_dalis_reproduce_user_likelihoods(bundle):
    from gpucosmo.jax.inference.dali_reduce import CompressedDALI

    b, X = bundle
    x0 = np.asarray(b.fiducial)
    tt = b.dali.build(x0[:b.n_theory], verbose=False)
    ex = jax.vmap(lambda x: b.log_like_from_theory_fn(x, b.theory_exact(x[:b.n_theory]), components=True))(X)
    sur = jax.vmap(lambda x: b.log_like_from_theory_fn(x, b.dali.theory(tt, x), components=True))(X)
    cs = CompressedDALI(b, tt[0], x0, log=lambda m: None)
    assert "extw:h0_gaussian" in cs.slices and "extw:bao_distances" in cs.slices  # whitened fast path
    assert "test_scaled_h0" in cs.ext_tree  # generic path (vector() uses a nuisance)
    cc = jax.vmap(lambda x: cs.components(x, cs.data))(X)
    for k in ex:
        assert np.max(np.abs(np.asarray(sur[k]) - np.asarray(ex[k]))) < 1e-3, k  # DALI truncation error
        assert np.max(np.abs(np.asarray(cc[k]) - np.asarray(sur[k]))) < 1e-5, k  # float32 deviation
    g = jax.vmap(jax.grad(lambda x: cs.loglike(x, cs.data)))(X)
    assert np.all(np.isfinite(np.asarray(g)))


def test_forward_mode_expansion_includes_user_theory(bundle):
    b, _ = bundle
    x0 = np.asarray(b.fiducial)[:b.n_theory]
    fd = b.dali.build(x0, verbose=False)[0]
    fwd = b.dali.build(x0, verbose=False, method="fwd")[0]
    for k in ("h0_gaussian", "bao_distances", "test_scaled_h0"):
        np.testing.assert_allclose(fwd.y0["ext"][k]["model"], fd.y0["ext"][k]["model"], rtol=1e-12)
        j_fwd, j_fd = np.asarray(fwd.jac["ext"][k]["model"]), np.asarray(fd.jac["ext"][k]["model"])
        np.testing.assert_allclose(j_fwd, j_fd, rtol=2e-3, atol=1e-6 * np.abs(j_fd).max())


def test_pipeline_build_analysis_bundle():
    from gpucosmo.pipeline import build_analysis_bundle

    b, res, diffs = build_analysis_bundle(config=_yaml({"h0_gaussian": None}), likelihoods=[ExactDV()],
                                          verbose=False)
    assert diffs == []
    assert [L.label for L in b.extra_likelihoods] == ["h0_gaussian", "test_exact_dv"]
    assert dataclasses.asdict(b.config)["desi_fs"] is False


def _doc_classes():
    """Execute the class-defining Python blocks of docs/adding_likelihoods.rst, in order."""
    import re
    from pathlib import Path

    rst = (Path(__file__).resolve().parents[1] / "docs" / "adding_likelihoods.rst").read_text()
    blocks = re.findall(r"\.\. code-block:: python\n\n((?:    .*\n|\n)+)", rst)
    ns = {"__name__": "gpucosmo_docs_example"}
    n = 0
    for b in blocks:
        code = textwrap.dedent(b)
        if "class " in code and "Likelihood" in code:
            exec(compile(code, "adding_likelihoods.rst", "exec"), ns)
            n += 1
    assert n >= 4
    return ns


def test_documented_examples_run():
    """The guide's classes work through exact, DALI and compressed evaluation."""
    from gpucosmo.jax.inference.cobaya_yaml import resolve_cobaya
    from gpucosmo.jax.inference.mg_bundle import build_mg_bundle
    from gpucosmo.jax.inference.dali_reduce import CompressedDALI

    ns = _doc_classes()
    likes = [ns["CosmicChronometers"](z=[0.2, 0.5, 1.0], H=[75.0, 90.0, 120.0], sigma=[5.0, 8.0, 12.0]),
             ns["CalibratedH0"](), ns["StudentTH0"]()]
    assert ns["TTBandPower"].requires == ("cmb",)
    r = resolve_cobaya(_yaml({}))
    r.add_likelihoods(likes)
    b = build_mg_bundle(r.config, priors=r.priors, likelihoods=r.extra_likelihoods)
    x0 = np.asarray(b.fiducial)
    X = jnp.asarray(x0 + np.random.default_rng(1).normal(size=(4, x0.size)) * 0.004 * (np.asarray(b.prior_hi) - np.asarray(b.prior_lo)))
    tt = b.dali.build(x0[:b.n_theory], verbose=False)
    ex = jax.vmap(lambda x: b.log_like_from_theory_fn(x, b.theory_exact(x[:b.n_theory]), components=True))(X)
    sur = jax.vmap(lambda x: b.log_like_from_theory_fn(x, b.dali.theory(tt, x), components=True))(X)
    cs = CompressedDALI(b, tt[0], x0, log=lambda m: None)
    cc = jax.vmap(lambda x: cs.components(x, cs.data))(X)
    for k in ("cosmic_chronometers", "calibrated_h0", "h0_student_t"):
        assert np.all(np.isfinite(np.asarray(ex[k]))), k
        assert np.max(np.abs(np.asarray(sur[k]) - np.asarray(ex[k]))) < 1e-3, k
        assert np.max(np.abs(np.asarray(cc[k]) - np.asarray(sur[k]))) < 1e-5, k
    assert "extw:cosmic_chronometers" in cs.slices  # compressible Gaussian
    assert np.all(np.isfinite(np.asarray(jax.vmap(jax.grad(lambda x: cs.loglike(x, cs.data)))(X))))


MG_CFG = ("data/desi_dr1/official_mg_chains/base_mu_sigma/desi-reptvelocileptors-fs-bao-all_planck2018-lowl-TT-clik_"
          "planck2020-lollipop-lowlE_planck2020-hillipop-TTTEEE_planck-act-dr6-lensing/chain.updated.yaml")


@pytest.mark.slow
def test_shared_cmb_block_on_the_mg_dali():
    """A user likelihood reading the built-in CMB spectra (``requires=("cmb",)``) on the real
    μ–Σ DALI: compressed (full shared blocks) = default DALI = exact at the MAP."""
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    out = root / "outputs" / "sns_p18_mg"
    if not ((root / MG_CFG).exists() and (out / "dali.npz").exists()):
        pytest.skip("needs the official MG config and outputs/sns_p18_mg")
    from gpucosmo.jax.inference.dali_reduce import CompressedDALI
    from gpucosmo.pipeline import build_analysis_bundle

    ns = _doc_classes()
    band = ns["TTBandPower"]()
    b, _, _ = build_analysis_bundle(config=root / MG_CFG, likelihoods=[band], fs_tables=True, verbose=False,
                                    batch_overrides=dict(defer_hillipop_matrix=True))
    tay = b.dali.load_expansion(out / "dali.npz")  # TTBandPower has no theory(): the saved expansion serves
    xm = np.asarray(json.loads((out / "map.json").read_text())["x"])
    x = np.concatenate([xm, [1.0]])  # + tt_cal
    rng = np.random.default_rng(0)
    X = jnp.asarray(x + rng.normal(size=(4, x.size)) * 1e-3 * (np.asarray(b.prior_hi) - np.asarray(b.prior_lo)))
    only = frozenset({"tt_band_power"})
    sur = jax.vmap(lambda xx: b.log_like_from_theory_fn(xx, {**tay(xx[:b.n_theory]), "aux": {}}, components=True,
                                                        include=only))(X)
    cs = CompressedDALI(b, tay, xm, log=lambda m: None)
    assert "shared:cmb:dl_tt" in cs.slices
    cc = jax.vmap(lambda xx: cs.components(xx, cs.data))(X)
    np.testing.assert_allclose(np.asarray(cc["tt_band_power"]), np.asarray(sur["tt_band_power"]), rtol=1e-6, atol=1e-6)
    ex = b.log_like_from_theory_fn(X[0], {**b.theory_exact(X[0, :b.n_theory]), "aux": {}}, components=True,
                                   include=only)
    assert abs(float(ex["tt_band_power"]) - float(sur["tt_band_power"][0])) < 0.05  # DALI truncation error near the MAP


def test_stale_expansion_file_is_refused(bundle, tmp_path):
    """A DALI expansion saved without a user likelihood's outputs cannot be loaded."""
    from gpucosmo.jax.inference.cobaya_yaml import resolve_cobaya
    from gpucosmo.jax.inference.mg_bundle import build_mg_bundle

    b, _ = bundle
    r = resolve_cobaya(_yaml({"h0_gaussian": None}))
    small = build_mg_bundle(r.config, priors=r.priors, likelihoods=r.extra_likelihoods)
    tt = small.dali.build(np.asarray(small.fiducial)[:small.n_theory], verbose=False)
    small.dali.save(tt, tmp_path / "dali.npz")
    with pytest.raises(ValueError, match="rebuild it"):
        b.dali.load_expansion(tmp_path / "dali.npz")
