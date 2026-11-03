"""Tests for the pure-JAX FAST-PT port (gpucosmo.jax.fastpt_jax).

FAST-PT evaluates the direct log-k convolutions of P_IA_deltaE2, P_IA_B,
P_13_reg and Y1_reg_NL with ``scipy.signal.fftconvolve``, whose round-off
(kernel dynamic range ~1e30) reaches ~1e-6..1e-4 of max|term| at high k. The
JAX port evaluates them exactly (direct Toeplitz sum), so the 1e-9 reference
comparisons patch FAST-PT to use ``np.convolve`` for those four helpers; the
stock-FAST-PT comparison is checked separately with a looser tolerance.
"""

from __future__ import annotations

import contextlib
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

fastpt = pytest.importorskip("fastpt")
pyccl = pytest.importorskip("pyccl")

import jax
import jax.numpy as jnp

from gpucosmo.jax.fastpt_jax import FASTPTJax, ept_ia_templates, jaxpt_available, make_fastpt

K_S = np.logspace(-4, 2, int(6 * 20))
N_PAD = int(1.0 * len(K_S))
FPT_KW = dict(low_extrap=-5, high_extrap=3, n_pad=N_PAD)
TOL = 1e-9

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning:fastpt")

METHODS = ("IA_tt", "IA_ta", "IA_mix", "one_loop_dd", "one_loop_dd_bias_b3nl")
WINDOWS = [
    dict(P_window=None, C_window=0.75),
    dict(P_window=None, C_window=None),
    dict(P_window=np.array([0.2, 0.2]), C_window=0.75),
]
WINDOW_IDS = ["C0.75", "Cnone", "P0.2-C0.75"]
# (method, index) of outputs that go through FAST-PT's fftconvolve kernels.
# FAST-PT 4.0–4.1.0 dropped P_window from the scalar (one-loop) path; 2.x/3.x and ≥ 4.1.1 apply it,
# as FASTPTJax does. jax-pt (the "jaxpt" backend) follows 4.0–4.1.0.
_FPT_V = tuple(int(x) for x in fastpt.__version__.split(".")[:3] if x.isdigit())
SCALAR_WINDOW_IN_REF = not ((4, 0, 0) <= _FPT_V < (4, 1, 1))
SCALAR_METHODS = ("one_loop_dd", "one_loop_dd_bias_b3nl")


def _skip_scalar_window(method, kw, ref_windows: bool):
    if method in SCALAR_METHODS and kw.get("P_window") is not None and not ref_windows:
        pytest.skip("the reference ignores P_window on the scalar path (FAST-PT 4.0–4.1.0 / jax-pt)")


KERNEL_TERMS = {("IA_ta", 1), ("IA_mix", 1), ("one_loop_dd", 0),
                ("one_loop_dd_bias_b3nl", 0), ("one_loop_dd_bias_b3nl", 8)}


@pytest.fixture(autouse=True, scope="module")
def _numpy_trapz_shim():
    # FAST-PT 4.0.0 calls np.trapz (removed in NumPy 2.4) for sig4.
    with pytest.MonkeyPatch.context() as mp:
        if not hasattr(np, "trapz"):
            mp.setattr(np, "trapz", np.trapezoid, raising=False)
        yield


@contextlib.contextmanager
def _exact_fastpt_convolutions():
    import fastpt.IA.IA_ABD as ia_abd
    import fastpt.IA.IA_ta as ia_ta
    import fastpt.utils.matter_power_spt as mps

    direct = lambda a, b: np.convolve(a, b)  # noqa: E731
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(mps, "fftconvolve", direct)
        mp.setattr(ia_ta, "convolve", direct)
        mp.setattr(ia_abd, "convolve", direct)
        yield


def _rel_err(ref, got) -> float:
    ref = np.asarray(ref, dtype=float)
    got = np.asarray(got, dtype=float)
    return float(np.max(np.abs(got - ref)) / np.max(np.abs(ref)))


@pytest.fixture(scope="module")
def cosmo():
    return pyccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, sigma8=0.8,
                           n_s=0.96, transfer_function="eisenstein_hu")


@pytest.fixture(scope="module")
def pk_lin(cosmo):
    return np.asarray(cosmo.linear_matter_power(K_S, 1.0))


@pytest.fixture(scope="module")
def pt():
    return FASTPTJax(K_S, **FPT_KW)


def _fastpt_ref(method, P, exact: bool, **kw):
    ctx = _exact_fastpt_convolutions() if exact else contextlib.nullcontext()
    with ctx:
        ref = fastpt.FASTPT(K_S, **FPT_KW)
        return getattr(ref, method)(P, **kw)


@pytest.mark.parametrize("kw", WINDOWS, ids=WINDOW_IDS)
@pytest.mark.parametrize("method", METHODS)
def test_matches_fastpt(pt, pk_lin, method, kw):
    _skip_scalar_window(method, kw, SCALAR_WINDOW_IN_REF)
    ref = _fastpt_ref(method, pk_lin, exact=True, **kw)
    got = getattr(pt, method)(jnp.asarray(pk_lin), **kw)
    assert len(got) == len(ref)
    for i, (r, g) in enumerate(zip(ref, got)):
        assert np.shape(g) == np.shape(r), (method, i)
        assert _rel_err(r, g) < TOL, (method, i, _rel_err(r, g))


@pytest.mark.parametrize("kw", WINDOWS[:2], ids=WINDOW_IDS[:2])
@pytest.mark.parametrize("method", METHODS)
def test_matches_stock_fastpt(pt, pk_lin, method, kw):
    ref = _fastpt_ref(method, pk_lin, exact=False, **kw)
    got = getattr(pt, method)(jnp.asarray(pk_lin), **kw)
    for i, (r, g) in enumerate(zip(ref, got)):
        tol = 1e-3 if (method, i) in KERNEL_TERMS else TOL
        assert _rel_err(r, g) < tol, (method, i, _rel_err(r, g))


needs_jaxpt = pytest.mark.skipif(not jaxpt_available(), reason="jax-pt not installed")


@needs_jaxpt
@pytest.mark.parametrize("kw", WINDOWS[::2], ids=WINDOW_IDS[::2])
@pytest.mark.parametrize("method", METHODS)
def test_jaxpt_backend_matches_stock_fastpt(pk_lin, method, kw):
    """JAX-PT (through the FAST-PT-signature adapter) vs stock FAST-PT."""
    _skip_scalar_window(method, kw, not SCALAR_WINDOW_IN_REF)  # jax-pt ignores it: compare where FAST-PT does too
    jpt = make_fastpt(K_S, backend="jaxpt", **FPT_KW)
    ref = _fastpt_ref(method, pk_lin, exact=False, **kw)
    got = getattr(jpt, method)(jnp.asarray(pk_lin), **kw)
    assert len(got) == len(ref)
    for i, (r, g) in enumerate(zip(ref, got)):
        assert np.shape(g) == np.shape(r), (method, i)
        # With a C_window JAX-PT matches FAST-PT to <1e-12 (<1.1e-6 adding a
        # P_window); without one it is off by up to 0.4% (make_fastpt avoids it).
        tol = 1e-3 if (method, i) in KERNEL_TERMS else 2e-6
        assert _rel_err(r, g) < tol, (method, i, _rel_err(r, g))


def test_make_fastpt_backend_selection():
    assert isinstance(make_fastpt(K_S, backend="gpucosmo", **FPT_KW), FASTPTJax)
    assert isinstance(make_fastpt(K_S, backend="auto", C_window=None, **FPT_KW), FASTPTJax)
    if jaxpt_available():
        assert not isinstance(make_fastpt(K_S, backend="auto", **FPT_KW), FASTPTJax)
    with pytest.raises(ValueError):
        make_fastpt(K_S, backend="numpy")


@needs_jaxpt
def test_jaxpt_backend_grad(pk_lin):
    jpt = make_fastpt(K_S, backend="jaxpt", **FPT_KW)
    P0 = jnp.asarray(pk_lin)
    w = jnp.asarray(np.exp(-0.5 * np.log(K_S / 0.1) ** 2))

    def f(amp):
        P = amp ** 2 * P0
        return jnp.sum(w * (jpt.IA_mix(P, C_window=0.75)[0] + jpt.IA_ta(P, C_window=0.75)[2]))

    # Every term is quadratic in P: f(amp) = amp^4 f(1).
    assert abs(float(jax.jit(jax.grad(f))(1.0)) / (4 * float(f(1.0))) - 1.0) < 1e-10


def test_jit_matches_eager(pt, pk_lin):
    def run(P):
        kw = dict(C_window=0.75)
        return (pt.IA_tt(P, **kw), pt.IA_ta(P, **kw), pt.IA_mix(P, **kw),
                pt.one_loop_dd_bias_b3nl(P, **kw))

    P = jnp.asarray(pk_lin)
    for e, j in zip(jax.tree_util.tree_leaves(run(P)),
                    jax.tree_util.tree_leaves(jax.jit(run)(P))):
        assert _rel_err(e, j) < 1e-10  # XLA fusion reorders sums


def test_grad_matches_finite_difference(pt, pk_lin):
    P0 = jnp.asarray(pk_lin)
    w = jnp.asarray(np.exp(-0.5 * np.log(K_S / 0.1) ** 2))
    tilt = jnp.asarray(np.log(K_S / 0.05))

    def f(theta):
        amp, dn = theta
        P = amp ** 2 * P0 * jnp.exp(dn * tilt)
        kw = dict(C_window=0.75)
        P_A = pt.IA_mix(P, **kw)[0]
        P_0E0E = pt.IA_ta(P, **kw)[2]
        P_1loop = pt.one_loop_dd_bias_b3nl(P, **kw)[0]
        return (jnp.sum(w * P_A) + jnp.sum(w * P_0E0E)) / 1e3 + jnp.sum(w * P_1loop) / 1e4

    theta = jnp.array([1.0, 0.0])
    grad = np.asarray(jax.grad(f)(theta))
    h = 1e-4
    for i in range(2):
        e = np.zeros(2)
        e[i] = h
        fd = (f(theta + e) - f(theta - e)) / (2 * h)
        assert abs(grad[i] - fd) < 1e-6 * abs(fd), (i, grad[i], fd)
    # Every term is quadratic in P: f(amp) = amp^4 f(1).
    assert abs(grad[0] - 4 * f(theta)) < 1e-10 * abs(grad[0])


def test_ept_templates_match_ccl(pt, cosmo):
    from pyccl.nl_pt import EulerianPTCalculator

    with _exact_fastpt_convolutions():
        ept = EulerianPTCalculator(with_NC=True, with_IA=True, cosmo=cosmo)
    np.testing.assert_allclose(ept.k_s, K_S, rtol=0, atol=0)
    tpl = ept_ia_templates(pt, jnp.asarray(cosmo.linear_matter_power(K_S, 1.0)),
                           P_window=None, C_window=0.75)
    for name in ("ia_ta", "ia_tt", "ia_mix", "dd_bias", "one_loop_dd"):
        ref = getattr(ept, name)
        assert len(tpl[name]) == len(ref), name
        for i, (r, g) in enumerate(zip(ref, tpl[name])):
            assert _rel_err(r, g) < TOL, (name, i, _rel_err(r, g))
