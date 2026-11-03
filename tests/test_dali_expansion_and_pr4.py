#!/usr/bin/env python3
"""Generic DALI expansions (``inference.dali_expansion``) and the JAX Planck PR4 likelihoods."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
REPO = Path(__file__).resolve().parents[1]


def _quadratic_tree(x):
    A = jnp.arange(12.0).reshape(3, 4) / 10.0
    return {"a": 1.0 + x @ A + 0.5 * (x[0] * x[1]) * jnp.ones(4), "b": {"c": jnp.sum(x**2) * jnp.ones(2)}}


def test_fd_expansion_reproduces_a_quadratic_exactly():
    from gpucosmo.jax.inference.dali_expansion import build_dali_fd

    fb = jax.jit(jax.vmap(_quadratic_tree))
    sur = build_dali_fd(fb, np.array([0.1, -0.2, 0.3]), [0.5, 0.7, 0.3], verbose=False,
                          max_batch=5)
    for x in (np.array([1.0, 2.0, -1.0]), np.array([-0.4, 0.1, 0.9])):
        got, want = sur(jnp.asarray(x)), _quadratic_tree(jnp.asarray(x))
        np.testing.assert_allclose(np.asarray(got["a"]), np.asarray(want["a"]), atol=1e-12)
        np.testing.assert_allclose(np.asarray(got["b"]["c"]), np.asarray(want["b"]["c"]), atol=1e-12)


def test_fd_expansion_in_log_coordinates(tmp_path):
    """``log_idx``: the expansion is quadratic in ln x_i, so a function quadratic in (ln x₀, x₁) is
    reproduced exactly far from the expansion point; the coordinates survive save/load."""
    from gpucosmo.jax.inference.dali_expansion import DALIExpansion, build_dali_fd

    def f(x):
        u = jnp.log(x[0])
        return {"y": jnp.stack([1.0 + 2.0 * u + 0.7 * u**2 + 0.3 * u * x[1], x[1] ** 2 - u])}

    fb = jax.jit(jax.vmap(f))
    sur = build_dali_fd(fb, np.array([0.3, 1.0]), [0.05, 0.2], verbose=False, log_idx=(0,))
    assert sur.log_idx == (0,)
    for x in (np.array([0.1, 2.0]), np.array([0.55, -0.5])):
        np.testing.assert_allclose(np.asarray(sur(jnp.asarray(x))["y"]), np.asarray(f(jnp.asarray(x))["y"]),
                                   atol=1e-10)
    lin = build_dali_fd(fb, np.array([0.3, 1.0]), [0.05, 0.2], verbose=False)  # in x₀: not exact
    assert np.max(np.abs(np.asarray(lin(jnp.asarray([0.1, 2.0]))["y"] - f(jnp.asarray([0.1, 2.0]))["y"]))) > 1e-2
    sur.save(tmp_path / "t.npz")
    back = DALIExpansion.load(tmp_path / "t.npz", {"y": jnp.zeros(2)})
    assert back.log_idx == (0,)
    x = jnp.asarray([0.2, 0.4])
    np.testing.assert_allclose(np.asarray(back(x)["y"]), np.asarray(sur(x)["y"]), rtol=1e-14)
    # a pytree: jit keeps the static coordinates
    np.testing.assert_allclose(np.asarray(jax.jit(lambda t, x: t(x))(sur, x)["y"]), np.asarray(sur(x)["y"]))


def test_forward_mode_expansion_matches_fd_on_a_quadratic():
    from gpucosmo.jax.inference.dali_expansion import build_dali_fd, build_dali_fwd

    x0 = np.array([0.1, -0.2, 0.3])
    s1 = build_dali_fwd(_quadratic_tree, x0, order=2, verbose=False)
    s2 = build_dali_fd(jax.vmap(_quadratic_tree), x0, [0.5, 0.7, 0.3], order=2, verbose=False)
    x = jnp.asarray([0.7, 0.2, -0.5])
    np.testing.assert_allclose(np.asarray(s1(x)["a"]), np.asarray(s2(x)["a"]), atol=1e-12)


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_order_p_expansions_reproduce_degree_p_polynomials(order):
    """Both builders are exact for a theory of total degree ≤ p (the FD stencils by construction);
    the order-p FD design reduces to the classic 1 + 2n (p = 1) and 1 + 2n² (p = 2) points."""
    from gpucosmo.jax.inference.dali_expansion import build_dali_fd, build_dali_fwd, fd_design, monomials, n_terms

    n = 4
    C = np.random.default_rng(order).normal(size=(n_terms(n, order), 3))
    f = lambda x: {"y": monomials(x, order) @ C}  # noqa: E731
    x0 = np.array([0.2, -0.1, 0.4, 0.3])
    fd = build_dali_fd(jax.vmap(f), x0, [0.3, 0.5, 0.2, 0.4], order=order, verbose=False, max_batch=16)
    fwd = build_dali_fwd(f, x0, order=order, pair_batch=8, verbose=False)
    X = x0 + np.random.default_rng(9).normal(size=(6, n))
    want = np.asarray(jax.vmap(f)(X)["y"])
    for t in (fd, fwd):
        assert t.order == order
        np.testing.assert_allclose(np.asarray(jax.vmap(t)(X)["y"]), want, atol=1e-11)
    n_pts = {1: 1 + 2 * n, 2: 1 + 2 * n * n}
    if order in n_pts:
        assert fd_design(n, order)[0].shape[0] == n_pts[order]


def test_dali_hierarchy_converges_and_truncates():
    """Away from polynomials the error falls with the order; ``truncate`` is the lower-order
    expansion with the same derivatives."""
    from gpucosmo.jax.inference.dali_expansion import build_dali_fwd

    g = lambda x: {"y": jnp.exp(0.5 * x[0]) * jnp.cos(x[1]) + x[0] * x[1] ** 2}  # noqa: E731
    x0 = np.array([0.1, 0.2])
    X = x0 + 0.5 * np.random.default_rng(3).normal(size=(500, 2))
    want = np.asarray(jax.vmap(g)(X)["y"])
    t4 = build_dali_fwd(g, x0, order=4, verbose=False)
    errs = []
    for p in (1, 2, 3, 4):
        tp = build_dali_fwd(g, x0, order=p, verbose=False)
        np.testing.assert_allclose(np.asarray(t4.truncate(p).coef["y"]), np.asarray(tp.coef["y"]), rtol=1e-12,
                                   atol=1e-14)
        errs.append(float(np.sqrt(np.mean((np.asarray(jax.vmap(tp)(X)["y"]) - want) ** 2))))
    assert errs[3] < errs[2] < errs[1] < errs[0]


def test_fd_expansion_refuses_non_finite_theory():
    """A failed exact evaluation (NaN spectra from a rejected solve) at a design point stops the build."""
    from gpucosmo.jax.inference.dali_expansion import build_dali_fd

    f = lambda X: {"y": jnp.where(X[:, :1] > 0.55, jnp.nan, X ** 2)}  # noqa: E731
    with pytest.raises(ValueError, match="non-finite at 1 of 13 design points"):
        build_dali_fd(f, np.array([0.3, 0.1]), [0.15, 0.1], order=3, verbose=False)


def test_second_order_files_of_the_earlier_format_load(tmp_path):
    """dali.npz written before the order-p format (y0, jac, hess) loads as an order-2 expansion."""
    from gpucosmo.jax.inference.dali_expansion import DALIExpansion

    rng = np.random.default_rng(0)
    y0, J, H = rng.normal(size=4), rng.normal(size=(3, 4)), rng.normal(size=(3, 3, 4))
    H = H + H.transpose(1, 0, 2)
    np.savez(tmp_path / "old.npz", x0=np.zeros(3), log_idx=np.zeros(0, dtype=np.int64),
             **{"y0['y']": y0, "jac['y']": J, "hess['y']": H})
    t = DALIExpansion.load(tmp_path / "old.npz", {"y": jnp.zeros(4)})
    assert t.order == 2
    d = np.array([0.3, -0.2, 0.5])
    np.testing.assert_allclose(np.asarray(t(jnp.asarray(d))["y"]), y0 + d @ J + 0.5 * np.einsum("i,j,ijm->m", d, d, H),
                               rtol=1e-13)


@pytest.mark.slow
@pytest.mark.skipif(not (REPO / "data/planck_pr4/reference.npz").exists(),
                    reason="run scripts/ref_planck_pr4_dump.py first")
def test_pr4_likelihoods_match_official():
    from gpucosmo.jax.cmb.pr4_jax import load_pr4

    r = np.load(REPO / "data/planck_pr4/reference.npz")
    hil, lol = load_pr4()
    for ip in (0, 1):
        pre = f"p{ip}_"
        p = {str(n): jnp.asarray(float(v)) for n, v in zip(r[pre + "nuis_names"],
                                                           r[pre + "nuis_values"]) if np.isfinite(v)}
        ll_h = float(hil.loglike(jnp.asarray(r[pre + "dl_tt"]), jnp.asarray(r[pre + "dl_ee"]),
                                 jnp.asarray(r[pre + "dl_te"]), p))
        # official χ² uses a float32 covariance and data vector (relative precision ~3e-7)
        assert abs(ll_h - float(r[pre + "loglike_hillipop"])) < 0.02
        ll_l = float(lol.loglike(jnp.asarray(r[pre + "cl_ee"]), p["A_planck"]))
        np.testing.assert_allclose(ll_l, float(r[pre + "loglike_lollipop"]), atol=1e-6)


def test_hillipop_centred_quadratic_form_is_exact():
    """χ² = x0ᵀCx0 + 2(Cx0)·δx + δxᵀCδx (``HiLLiPoP.loglike(ref=...)``) equals xᵀCx, and
    with a nearby reference the float32 product is far more accurate than uncentred."""
    from gpucosmo.jax.cmb.pr4_jax import _quad_form_sym

    rng = np.random.default_rng(0)
    n = 600
    a = rng.standard_normal((n, n))
    C = (a @ a.T / n + np.eye(n)).astype(np.float32)  # SPD, exactly float32
    x0 = 1e3 * rng.standard_normal(n)
    x = x0 + rng.standard_normal(n)  # δx ≪ x: what a posterior-typical reference gives
    C64 = C.astype(np.float64)
    exact = float(x @ C64 @ x)
    cx0 = C64 @ x0
    centred = float(x0 @ cx0 + 2.0 * cx0 @ (x - x0) + _quad_form_sym(jnp.asarray(C), jnp.asarray(x - x0)))
    plain = float(_quad_form_sym(jnp.asarray(C), jnp.asarray(x)))
    assert abs(centred - exact) < 1e-9 * abs(exact)
    assert abs(centred - exact) < 1e-2 * abs(plain - exact) or abs(plain - exact) < 1e-12 * abs(exact)
