"""Cobaya yaml input (inference.cobaya_yaml): the analyses shipped with the preview, and the
inputs the preview refuses."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from gpucosmo.jax.inference.cobaya_yaml import (
    PLIK_FIXED,
    UnsupportedCobayaInput,
    load_yaml,
    resolve_cobaya,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "cobaya"
EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "custom_likelihood" / "analysis.yaml"
# analysis: (SN dataset, number of sampled parameters)
EXPECTED = {
    "lcdm_desidr2_cmb_desy5": ("desy5", 15), "lcdm_desidr2_cmb_pantheonplus": ("pantheonplus", 15),
    "lcdm_desidr2_cmb_union3": ("union3", 15), "w0wa_desidr2_cmb_desy5": ("desy5", 17),
    "w0wa_desidr2_cmb_pantheonplus": ("pantheonplus", 17), "w0wa_desidr2_cmb_union3": ("union3", 17),
}


def _needs_cmb_data():
    """The DESI DR2 + CMB analyses load Planck (clik, CamSpec NPIPE) and ACT DR6 lensing through
    Cobaya: skip without Cobaya, its packages path or the ACT likelihood package."""
    pytest.importorskip("cobaya")
    pytest.importorskip("clipy")
    pytest.importorskip("act_dr6_lenslike")
    from gpucosmo.paths import packages_path

    p = packages_path()
    if p is None or not (Path(p) / "data" / "planck_NPIPE_CamSpec").is_dir():
        pytest.skip("Cobaya packages (Planck CamSpec NPIPE) not installed")


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_preview_configs_resolve(name):
    _needs_cmb_data()
    r = resolve_cobaya(CONFIGS / f"{name}.yaml")
    sn, n = EXPECTED[name]
    c = r.config
    assert (c.sn, c.mg, c.desi_fs, len(r.priors)) == (sn, False, False, n)
    assert ("w" in r.priors and "wa" in r.priors) == name.startswith("w0wa")


def _example(edit=None):
    info = copy.deepcopy(load_yaml(EXAMPLE))
    info["likelihood"].pop("cosmic_chronometers")  # registered by the example's --plugin file
    if edit:
        edit(info)
    return info


def test_example_analysis_resolves_without_data():
    r = resolve_cobaya(_example())
    assert not r.config.desi_fs and not r.config.mg and r.config.cmb is None
    assert {"H0", "ombh2", "omch2"} <= set(r.priors)


@pytest.mark.parametrize("edit, match", [
    (lambda i: i["likelihood"].__setitem__(
        "desi_y1_cosmo_bindings.cobaya_likelihoods.fs_bao_likelihoods.desi_fs_bao_all", {}),
     "DESI DR1 full-shape"),
    (lambda i: i["params"].__setitem__("Omega_m", {"prior": {"min": 0.1, "max": 0.9}}), "DES Y3 / HSC Y3"),
    (lambda i: i["params"].__setitem__("A_bary", {"prior": {"min": 2.0, "max": 3.13}}), "DES Y3 / HSC Y3"),
    (lambda i: i["likelihood"].__setitem__("no_such_package.NoSuchLikelihood", {}), "not available"),
], ids=["desi_full_shape", "des_basis", "hsc_basis", "unknown"])
def test_inputs_outside_the_preview_are_refused(edit, match):
    with pytest.raises(UnsupportedCobayaInput, match=match):
        resolve_cobaya(_example(edit))


def test_plik_fixed_values_match_clipy_defaults():
    clipy_plik = pytest.importorskip("gpucosmo.jax.cmb.clipy_plik")
    try:
        cp = clipy_plik.load_clipy_plik()
    except FileNotFoundError:  # no Cobaya packages path configured
        cp = None
    if cp is None:
        pytest.skip("plik clik data not installed")
    defaults = dict(cp.like._default)
    for k, v in PLIK_FIXED.items():
        assert np.isclose(defaults[k], v), k


def test_clipy_priors_are_not_counted_twice():
    """clipy's ``all_priors=True`` adds Gaussian nuisance priors inside ``log_like``; the bundle
    applies the same priors itself, so the loaded likelihoods must carry none (keeping only
    clipy's fixed baseline values)."""
    import dataclasses

    import jax.numpy as jnp

    from gpucosmo.jax.cmb.clipy_likelihood import build_clipy_lowl_loglike, load_clipy_lowl

    cl = load_clipy_lowl()
    if not cl.active:
        pytest.skip("Planck low-ell clipy data not installed")
    for like in (cl.tt_like, cl.ee_like):
        assert like is None or not like._prior
    from gpucosmo.jax.cmb.lensing_jax import _load_highl_template
    from gpucosmo.jax.cmb.params import T_CMB_MUK

    f = build_clipy_lowl_loglike(dataclasses.replace(cl, ee_like=None))
    t = _load_highl_template()[:40] * T_CMB_MUK**2
    tt, ee, te = (jnp.asarray(t[:, i]) for i in (0, 1, 3))
    # Δ log L from A_planck 1 → 1.005 on the ΛCDM template: +0.35 from the calibration alone;
    # with clipy's prior also inside it was −1.65 (−½(0.005/0.0025)² = −2 more)
    d = float(f(tt, te, ee, jnp.asarray(1.005)) - f(tt, te, ee, jnp.asarray(1.0)))
    assert 0.0 < d < 1.0, d
    clipy_plik = pytest.importorskip("gpucosmo.jax.cmb.clipy_plik")
    cp = clipy_plik.load_clipy_plik()
    if cp is not None:
        assert not cp.like._prior and "cib_index" not in cp.nuisance_names
