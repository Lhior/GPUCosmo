"""Native CMBlikes ports (``likelihoods.cobaya.cmblikes``) against Cobaya through the bridge.

Planck 2018 / PR4 lensing (full and CMB-marginalized) and BICEP/Keck 2018 (HL likelihood with
foregrounds; our theory's BB is the lensed scalar one, r = 0): value for value at the reference
spectra and at rescaled spectra, BK18's decorrelation branch, the exact full-sky form on a
Cobaya forecast data set, and finite gradients w.r.t. the spectra and nuisance parameters.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("cobaya")
jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

TOL = 1e-6
NAMES = ("planck_2018_lensing.native", "planck_2018_lensing.CMBMarged", "plancklensing.PlanckLensing",
         "plancklensing.PlanckLensingMarged", "planckpr4lensing.PlanckPR4Lensing",
         "planckpr4lensing.PlanckPR4LensingMarged", "bicep_keck_2018")
SCALES = {"tt": 1.03, "te": 0.97, "ee": 1.05, "bb": 1.2, "pp": 0.93}
_CACHE: dict = {}


def _port(name, cls=None, *args, **kw):
    """The port (built once per configuration); skips when Cobaya cannot build it (data missing)."""
    from gpucosmo.cobaya_bridge.bridge import BridgeUnsupported

    key = (name, repr(args), repr(sorted(kw.items())))
    if key not in _CACHE:
        if cls is None:
            from gpucosmo.likelihoods.cobaya import ported

            cls = ported()[name]
        try:
            _CACHE[key] = cls(name, *args, **kw)
        except BridgeUnsupported as e:
            pytest.skip(f"{name}: {e}")
    return _CACHE[key]


def _inputs(L, scales=None, params=None):
    from gpucosmo.likelihoods.cobaya.testing import inputs, points, reference

    ref = reference()
    pts = points(L, 4, seed=1)
    par = params or {k: v[1] for k, v in pts.items()}
    inp = inputs(L, ref, par)
    cl = dict(inp.shared["cmb"]["cl"])
    for s, f in (scales or {}).items():
        cl[s] = cl[s] * f
    return inp._replace(shared={**inp.shared, "cmb": {**inp.shared["cmb"], "cl": cl}})


def _diff(L, inp):
    f = jax.jit(lambda p, cmb: L.loglike(inp._replace(params=p, shared={**inp.shared, "cmb": cmb})))
    nat = float(f(inp.params, inp.shared["cmb"]))
    brg = float(L.bridge.loglike(inp))
    assert np.isfinite(nat) and np.isfinite(brg)
    return abs(nat - brg)


@pytest.mark.parametrize("name", NAMES)
def test_compare(name):
    from gpucosmo.likelihoods.cobaya.testing import compare

    L = _port(name)
    r = compare(L, n=4, batch=16, n_bridge_timing=2)
    assert r["max_abs_diff"] < TOL, r


@pytest.mark.parametrize("name", NAMES)
def test_rescaled_spectra(name):
    L = _port(name)
    assert _diff(L, _inputs(L, SCALES)) < TOL


def test_bk18_decorrelation():
    """Sampled Δ_dust, Δ_sync: the ℓ-dependent (per map pair) foreground path."""
    from gpucosmo.likelihoods.cobaya.cmblikes import CMBLikes

    L = _port("bicep_keck_2018", CMBLikes, params_info={
        "delta_dust": {"prior": {"min": 0.8, "max": 1.0}, "ref": 0.95},
        "delta_sync": {"prior": {"min": 0.8, "max": 1.0}, "ref": 0.9}})
    assert L.fold is None and not L.fg.separable(_inputs(L))
    assert _diff(L, _inputs(L, SCALES)) < TOL


def test_exact_forecast(tmp_path):
    """The exact full-sky (unbinned) form on a ``make_forecast_cmb_dataset`` data set."""
    from cobaya.likelihoods.base_classes.cmblikes import make_forecast_cmb_dataset

    from gpucosmo.likelihoods.cobaya.cmblikes import CMBLikes
    from gpucosmo.likelihoods.cobaya.testing import reference

    x = np.asarray(reference()["cmb"]["cmb_x"])
    cls = {"tt": x[1], "ee": x[2], "te": x[4], "pp": x[0] * 2 / np.pi}
    make_forecast_cmb_dataset(cls, "fc", output_dir=str(tmp_path), noise_muK_arcmin_T=10.0, fwhm_arcmin=5.0, lmin=2,
                              lmax=1000, fsky=0.4, lens_recon_noise=np.full(x.shape[1], 1e-8), cl_dict_lmin=0)
    L = _port(f"forecast_exact_{os.getpid()}", CMBLikes, {"class": "cobaya.likelihoods.base_classes.CMBlikes",
                                                          "dataset_file": str(tmp_path / "fc.dataset")})
    assert L.approx == "exact" and not L.binned and L.nmaps == 3
    inp = _inputs(L, SCALES)
    assert abs(float(L.loglike(inp))) > 1.0  # non-trivial away from the fiducial
    assert _diff(L, inp) < TOL


def _grad_check(f, x0, h):
    g = float(jax.grad(f)(x0))
    fd = (float(f(x0 + h)) - float(f(x0 - h))) / (2 * h)
    assert np.isfinite(g) and g != 0.0
    assert abs(g - fd) <= 1e-4 * abs(fd) + 1e-6, (g, fd)


@pytest.mark.parametrize("name,spec,nuis", [("planck_2018_lensing.native", "pp", "A_planck"),
                                            ("planckpr4lensing.PlanckPR4LensingMarged", "pp", None),
                                            ("bicep_keck_2018", "bb", "BBdust")])
def test_gradients(name, spec, nuis):
    L = _port(name)
    inp = _inputs(L)

    def by_scale(a):
        cl = dict(inp.shared["cmb"]["cl"])
        cl[spec] = cl[spec] * a
        return L.loglike(inp._replace(shared={**inp.shared, "cmb": {**inp.shared["cmb"], "cl": cl}}))

    _grad_check(jax.jit(by_scale), jnp.asarray(1.0), 1e-4)
    if nuis:
        def by_param(v):
            return L.loglike(inp._replace(params={**inp.params, nuis: v}))

        _grad_check(jax.jit(by_param), jnp.asarray(inp.params[nuis], dtype=jnp.float64), 1e-5)
    if name == "bicep_keck_2018":  # every foreground parameter, through the HL eigendecompositions
        g = jax.grad(lambda p: L.loglike(inp._replace(params=p)))({k: jnp.asarray(v, dtype=jnp.float64)
                                                                     for k, v in inp.params.items()})
        assert all(np.isfinite(float(v)) for v in g.values()), g
