"""Native ports of Cobaya's pure-Python Planck likelihoods (``likelihoods.cobaya.planck_python``)
against Cobaya through the bridge.

CamSpec (2018 / 2021 / NPIPE), plik-lite, low-ℓ TT and EE: value for value at the reference
spectra (``testing.compare``) and at rescaled spectra, the −inf branches of the low-ℓ
likelihoods, and finite gradients w.r.t. a spectrum scale and a nuisance parameter. The
large CamSpec sets (inverse covariances up to 11786²) run on an ℓ-range subset with every
spectrum, still above Cobaya's 1500-point ``dsymv`` threshold; the full sets are validated by
``testing.compare`` outside the test suite (they need several GB of RAM).
"""

from __future__ import annotations

import contextlib
import os

import numpy as np
import pytest

pytest.importorskip("cobaya")
jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

TOL = 1e-6
SUBSET = "30-300 500-800 1990-2000"  # 2945 points with all six CamSpec spectra
ALL_CL = "100x100 143x143 217x217 143x217 TE EE"
_FG2018 = {"aps100": 150.0, "aps143": 50.0, "aps217": 55.0, "psr": 0.9, "cibr": 0.5, "ncib": 0.3, "cibrun": 0.1,
           "xi": 0.1, "dust100": 1.0, "dust143": 1.0, "dust217": 1.0, "dust143x217": 1.0,
           "acib143": {"prior": {"min": -1, "max": 10}, "ref": -0.2},
           "acib217": {"prior": {"min": 0, "max": 80}, "ref": 50},
           "asz143": {"prior": {"min": 0, "max": 10}, "ref": 4}, "aksz": {"prior": {"min": 0, "max": 10}, "ref": 1}}
_POWERLAW = {"amp_100": 0.0, "n_100": 1.0, "amp_143": 10.0, "amp_217": 20.0, "n_143": 1.0, "n_217": 1.2,
             "n_143x217": 0.9, "amp_143x217": {"prior": {"min": 0, "max": 50}, "ref": 10}}
CONFIGS = {  # id: (Cobaya name, block)
    "lowl.TT": ("planck_2018_lowl.TT", None),
    "lowl.EE": ("planck_2018_lowl.EE", None),
    "lowl.EE_sroll2": ("planck_2018_lowl.EE_sroll2", None),
    "plik.TTTEEE_lite": ("planck_2018_highl_plik.TTTEEE_lite_native", None),
    "plik.TT_lite": ("planck_2018_highl_plik.TT_lite_native", None),
    "CamSpec2021.EE": ("planck_2018_highl_CamSpec2021.EE", None),
    "NPIPE.TE": ("planck_NPIPE_highl_CamSpec.TE", None),  # strongly asymmetric cached inverse
    "CamSpec.TTTEEE.sub": ("planck_2018_highl_CamSpec.TTTEEE", {"dataset_params": {"use_cl": ALL_CL,
                                                                                  "use_range": SUBSET}}),
    "CamSpec2021.TTTEEE.sub": ("planck_2018_highl_CamSpec2021.TTTEEE",
                               {"dataset_params": {"use_cl": "143x143 217x217 143x217 TE EE", "use_range": SUBSET}}),
    "NPIPE.TTTE.sub": ("planck_NPIPE_highl_CamSpec.TTTE",
                       {"dataset_params": {"use_cl": "143x143 217x217 143x217 TE", "use_range": SUBSET}}),
    # use_fg_residual_model 2: 2018 foreground model + power-law residuals (acib143 on both sides of 0)
    "CamSpec2021.TT.fg2.sub": ("planck_2018_highl_CamSpec2021.TT",
                               {"dataset_params": {"use_cl": "100x100 143x143 217x217 143x217", "use_range": SUBSET},
                                "params": {"use_fg_residual_model": 2, **_FG2018, **_POWERLAW,
                                           "A_planck": {"prior": {"dist": "norm", "loc": 1, "scale": 0.0025}}}}),
}
SCALES = {"tt": 1.03, "te": 0.97, "ee": 1.05}
_CACHE: dict = {}


def _port(key, **kw):
    """The port (built once per configuration); skips when Cobaya cannot build it (data missing).
    Subset configurations do not write a covinv cache into the packages directory."""
    import cobaya.likelihoods.base_classes.planck_2018_CamSpec_python as cs

    from gpucosmo.cobaya_bridge.bridge import BridgeUnsupported
    from gpucosmo.likelihoods.cobaya import ported
    from gpucosmo.paths import packages_path

    ck = (key, repr(sorted(kw.items())))
    if ck not in _CACHE:
        name, block = CONFIGS[key]
        old, old_lock = cs.use_cache, cs.FileLock
        cs.use_cache = block is None
        path = packages_path()
        if path is not None and not os.access(path, os.W_OK):
            # Read-only packages (CI sandbox): Cobaya cannot create the lock file next to the cached
            # inverse covariance, which it reports as "locked"; read the existing cache without it
            cs.FileLock = lambda *a, **k: contextlib.nullcontext()
        try:
            _CACHE[ck] = ported()[name](name, block, **kw)
        except BridgeUnsupported as e:
            pytest.skip(f"{name}: {e}")
        finally:
            cs.use_cache, cs.FileLock = old, old_lock
    return _CACHE[ck]


def _inputs(L, scales=None):
    from gpucosmo.likelihoods.cobaya.testing import inputs, points, reference

    pts = points(L, 4, seed=1)
    inp = inputs(L, reference(), {k: v[1] for k, v in pts.items()})
    return _scaled(inp, scales or {})


def _scaled(inp, scales):
    cl = {s: v * scales[s] if s in scales else v for s, v in inp.shared["cmb"]["cl"].items()}
    return inp._replace(shared={**inp.shared, "cmb": {**inp.shared["cmb"], "cl": cl}})


def _pair(L, inp):
    f = jax.jit(lambda p, cmb: L.loglike(inp._replace(params=p, shared={**inp.shared, "cmb": cmb})))
    return float(f(inp.params, inp.shared["cmb"])), float(L.bridge.loglike(inp))


def test_registered():
    from gpucosmo.likelihoods.cobaya import ported
    from gpucosmo.likelihoods.cobaya.planck_python import CamSpec, LowlEE, LowlTT, PlikLite

    P = ported()
    for cls in (CamSpec, PlikLite, LowlTT, LowlEE):
        assert all(P[n] is cls for n in cls.cobaya_names)
    assert len(CamSpec.cobaya_names) == 14


@pytest.mark.parametrize("key", list(CONFIGS))
def test_compare(key):
    from gpucosmo.likelihoods.cobaya.testing import compare

    L = _port(key)
    r = compare(L, n=4, batch=16, n_bridge_timing=2)
    assert np.all(np.isfinite(r["native"])) and r["max_abs_diff"] < TOL, r


def test_cobaya_inverse_untouched():
    """The port's device copy of the inverse covariance (lower triangle, zeroed above the
    diagonal in place) leaves Cobaya's array as it was."""
    L = _port("NPIPE.TE")
    c = L.cobaya.covinv
    assert np.any(np.triu(c, 1) != 0) and not np.array_equal(c, c.T)
    assert np.array_equal(np.asarray(L.chi2.M), np.tril(c))


@pytest.mark.parametrize("key", ["lowl.TT", "lowl.EE", "plik.TTTEEE_lite", "NPIPE.TE", "CamSpec.TTTEEE.sub",
                                 "CamSpec2021.TT.fg2.sub"])
def test_rescaled_spectra(key):
    L = _port(key)
    nat, brg = _pair(L, _inputs(L, SCALES))
    assert np.isfinite(nat) and abs(nat - brg) < TOL


@pytest.mark.parametrize("acib143", [-0.5, 2.0])
def test_camspec_acib143_branches(acib143):
    """A_CIB,143 < 0: fixed from A_CIB,217 (Cobaya's default); ≥ 0: used as given."""
    L = _port("CamSpec2021.TT.fg2.sub")
    inp = _inputs(L)
    nat, brg = _pair(L, inp._replace(params={**inp.params, "acib143": acib143}))
    assert abs(nat - brg) < TOL


@pytest.mark.parametrize("key,spec", [("lowl.TT", "tt"), ("lowl.EE", "ee")])
def test_lowl_rejected(key, spec):
    """Outside the TT spline prior bounds / beyond the EE table: −inf, as Cobaya."""
    L = _port(key)
    nat, brg = _pair(L, _inputs(L, {spec: 1e4}))
    assert nat == brg == -np.inf


def _grad_check(f, x0, h):
    g = float(jax.grad(f)(x0))
    fd = (float(f(x0 + h)) - float(f(x0 - h))) / (2 * h)
    assert np.isfinite(g) and g != 0.0
    assert abs(g - fd) <= 1e-4 * abs(fd) + 1e-6, (g, fd)


@pytest.mark.parametrize("key,spec,nuis", [("CamSpec.TTTEEE.sub", "tt", "asz143"),
                                           ("CamSpec2021.TTTEEE.sub", "ee", "n_217"),
                                           ("CamSpec2021.TT.fg2.sub", "tt", "acib217"),
                                           ("plik.TTTEEE_lite", "te", "A_planck"),
                                           ("lowl.TT", "tt", None)])
def test_gradients(key, spec, nuis):
    L = _port(key)
    inp = _inputs(L)
    _grad_check(jax.jit(lambda a: L.loglike(_scaled(inp, {spec: a}))), jnp.asarray(1.0), 1e-4)
    if nuis:
        def by_param(v):
            return L.loglike(inp._replace(params={**inp.params, nuis: v}))

        _grad_check(jax.jit(by_param), jnp.asarray(inp.params[nuis], dtype=jnp.float64), 1e-5)
    g = jax.grad(lambda p: L.loglike(inp._replace(params=p)))({k: jnp.asarray(v, dtype=jnp.float64)
                                                               for k, v in inp.params.items()})
    assert all(np.isfinite(float(v)) for v in g.values()), g


def test_lowl_ee_smooth():
    """The tabulated EE likelihood: zero gradient (exact port); ``smooth=True`` interpolates
    the table (finite, non-zero gradient, value within the table's bin-to-bin variation)."""
    exact, smooth = _port("lowl.EE"), _port("lowl.EE", smooth=True)
    assert smooth.smooth and not exact.smooth
    inp = _inputs(exact)

    def f(L):
        return jax.jit(lambda a: L.loglike(_scaled(inp, {"ee": a})))

    assert float(jax.grad(f(exact))(1.0)) == 0.0
    g = float(jax.grad(f(smooth))(1.0))
    assert np.isfinite(g) and g != 0.0
    jump = np.max(np.abs(np.diff(exact.table, axis=0)), axis=0).sum()
    assert abs(float(f(smooth)(1.0)) - float(f(exact)(1.0))) < jump
