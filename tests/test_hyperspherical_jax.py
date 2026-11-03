"""Hyperspherical Bessel functions (``gpucosmo.jax.eb.hyperspherical_jax``) vs CAMB.

Fast CPU checks; the broad grid lives in ``scripts/validate_hyperspherical_jax.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import spherical_jn

from gpucosmo.jax.eb import hyperspherical_jax as hs

jax.config.update("jax_enable_x64", True)

TOL = 2e-4  # peak-normalized (CAMB phi_olver target: 1e-4, max < 2e-4)
R_H = 4450.0  # Mpc

_phi = jax.jit(hs.phi, static_argnums=(0, 1))


def _peak_err(val, ref, cm, l, K, nu):
    return np.max(np.abs(np.asarray(val) - ref)) / cm.phi_first_peak_amplitude(l, K, nu)


CASES = {
    -1: [(2, 0.7), (3, 1.0), (10, 3.3), (100, 12.9), (180, 22.0), (1000, 3848.0), (2500, 2700.0)],
    1: [(2, 3.0), (3, 8.0), (100, 101.0), (1000, 1040.0), (1000, 1500.0), (3000, 30000.0)],
}


@pytest.mark.parametrize("K", [-1, 1])
def test_phi_matches_phi_recurs(K):
    """All cases of one K in a single traced-l call (one compilation)."""
    cm = pytest.importorskip("camb.mathutils")
    chi_max = 3.0 if K < 0 else np.pi / 2
    rows = []
    for l, nu in CASES[K]:
        ell = np.sqrt(l * (l + 1.0))
        ct = np.arcsinh(ell / nu) if K < 0 else np.arcsin(ell / nu)
        chi = np.concatenate([np.linspace(1e-5, chi_max, 600), ct * (1 + np.linspace(-0.02, 0.02, 81))])
        rows.append(np.clip(chi, 1e-5, chi_max))
    chi = np.stack(rows)
    ls = np.array([c[0] for c in CASES[K]])[:, None]
    nus = np.array([c[1] for c in CASES[K]])[:, None]
    val = np.asarray(jax.jit(hs.phi, static_argnums=1)(ls, K, nus, chi))
    assert np.all(np.isfinite(val))
    for i, (l, nu) in enumerate(CASES[K]):
        ref = cm.phi_recurs(l, K, nu, chi[i])
        assert _peak_err(val[i], ref, cm, l, K, nu) < TOL, (l, nu)


def test_l_array_matches_static():
    cm = pytest.importorskip("camb.mathutils")
    ls = np.array([0, 1, 2, 3, 30, 150, 1200])
    nus = np.array([2.0, 5.0, 9.0, 1.5, 6.0, 25.0, 1500.0])
    chi = np.linspace(1e-3, 2.5, 60)
    val = np.asarray(jax.jit(hs.phi, static_argnums=1)(ls[:, None], -1, nus[:, None], chi[None, :]))
    for i, (l, nu) in enumerate(zip(ls, nus)):
        ref = cm.phi_recurs(int(l), -1, float(nu), chi)
        assert np.max(np.abs(val[i] - ref)) / cm.phi_first_peak_amplitude(int(l), -1, float(nu)) < TOL


def test_flat_limit_is_spherical_bessel():
    x = np.linspace(0.0, 400.0, 4001)
    ls = np.array([0, 1, 2, 3, 7, 40, 300])[:, None]
    ref = spherical_jn(ls, x[None, :])
    val = np.asarray(jax.jit(hs.phi, static_argnums=1)(ls, 0, 1.0, x[None, :]))
    assert np.all(np.max(np.abs(val - ref), axis=1) < 2e-5 * np.max(np.abs(ref), axis=1) + 1e-15)
    # elementwise l, large l and large x
    l = np.array([3, 30, 1000, 3500, 13500])
    xs = l * 1.07 + 3.0
    assert np.allclose(np.asarray(hs.spherical_jl(l, xs)), spherical_jn(l, xs), rtol=0, atol=1e-9)
    assert abs(float(hs.spherical_jl(10, 1.0e5)) - spherical_jn(10, 1.0e5)) < 1e-12


def test_small_chi_limit():
    """Phi_l^nu(K, chi) -> j_l(nu chi) when chi and l/nu are both small."""
    nu = 2.0e5
    chi = np.linspace(1e-6, 3e-4, 300)[None, :]
    ls = np.array([3, 10, 40])[:, None]
    f = jax.jit(lambda l, c, K: hs.phi(l, K, nu, c, recursion=False, gegenbauer=False), static_argnums=2)
    ref = spherical_jn(ls, nu * chi)
    for K in (-1, 1):
        assert np.max(np.abs(np.asarray(f(ls, chi, K)) - ref)) < 1e-6


def test_closed_parity_folding():
    cm = pytest.importorskip("camb.mathutils")
    l, nu = 7, 40.0
    chi = np.linspace(0.05, np.pi / 2 - 0.05, 50)
    base = np.asarray(_phi(l, 1, nu, chi))
    s_pi = (-1) ** (int(nu) - l - 1)
    assert np.allclose(np.asarray(_phi(l, 1, nu, np.pi - chi)), s_pi * base, atol=1e-12)
    assert np.allclose(np.asarray(_phi(l, 1, nu, 2 * np.pi - chi)), (-1) ** l * base, atol=1e-12)
    for l, nu in ((7, 40.0), (300, 310.0), (300, 700.0)):
        chi = np.linspace(0.0, 2 * np.pi, 301)
        assert _peak_err(_phi(l, 1, nu, chi), cm.phi_recurs(l, 1, nu, chi), cm, l, 1, nu) < TOL


def test_exact_low_l():
    cm = pytest.importorskip("camb.mathutils")
    chi = np.concatenate([np.geomspace(1e-7, 1e-2, 30), np.linspace(1e-2, 1.5, 200)])
    for K, nu in ((-1, 0.3), (-1, 40.0), (1, 3.0), (1, 500.0)):
        for l in (0, 1, 2):
            ref = cm.phi_recurs(l, K, nu, chi)
            assert np.max(np.abs(np.asarray(_phi(l, K, nu, chi)) - ref)) < 1e-10


def test_gradients_finite_and_match_fd():
    pts = {-1: [(2, 3.0, 0.7), (500, 3000.0, 0.15), (100, 150.0, 1.2), (30, 5.0, 2.0)],
           1: [(1000, 1500.0, 0.9), (500, 560.0, 1.3), (2, 7.0, 1.1)]}
    # plus the whole chi range incl. chi = 0, the turning point and closed chi = pi/2
    chi_all = np.concatenate([[0.0], np.linspace(1e-4, 1.6, 200), [np.pi / 2]])
    for K, rows in pts.items():
        npt = len(rows)
        l, nu, chi = (np.array(c, dtype=float) for c in zip(*rows))
        for l0, nu0 in ((50, 80.0), (20, 25.0 if K == 1 else 3.0)):
            l = np.concatenate([l, np.full(chi_all.shape, l0)])
            nu = np.concatenate([nu, np.full(chi_all.shape, nu0)])
            chi = np.concatenate([chi, chi_all])
        l = l.astype(int)

        def f(l, n, c, K=K):
            return hs.phi(l, K, n, c, recursion_lmax=50)

        fj = jax.jit(f)
        gn, gc = (np.asarray(a) for a in jax.jit(jax.vmap(jax.grad(f, argnums=(1, 2))))(l, nu, chi))
        assert np.all(np.isfinite(gn)) and np.all(np.isfinite(gc))
        sl = slice(0, npt)
        h = 1e-6 * np.maximum(chi, 1e-3)
        fd = (np.asarray(fj(l, nu, chi + h)) - np.asarray(fj(l, nu, chi - h))) / (2 * h)
        assert np.all(np.abs(gc[sl] - fd[sl]) <= 1e-6 * (np.abs(fd[sl]) + 1e-3))
        if K == -1:
            hn = 1e-6 * nu
            fdn = (np.asarray(fj(l, nu + hn, chi)) - np.asarray(fj(l, nu - hn, chi))) / (2 * hn)
            assert np.all(np.abs(gn[sl] - fdn[sl]) <= 1e-6 * (np.abs(fdn[sl]) + 1e-3))


def test_olver_map_with_external_jl():
    l, nu = 400, 900.0
    chi = np.linspace(1e-3, 1.5, 200)
    z, amp, symm, achi, sin_k = jax.jit(lambda n, c: hs.olver_map(l, -1, n, c))(nu, chi)
    from scipy.special import spherical_jn as jn

    phi_ext = np.asarray(symm * amp * z) * jn(l, nu * np.asarray(z)) / np.asarray(sin_k)
    phi_jax = np.asarray(jax.jit(lambda n, c: hs.phi(l, -1, n, c, jl=lambda x: hs.spherical_jl(l, x)))(nu, chi))
    assert np.max(np.abs(phi_ext - phi_jax)) < 1e-8


def test_jit_vmap():
    f = jax.jit(jax.vmap(lambda nu, chi: hs.phi(300, -1, nu, chi), in_axes=(0, None)))
    out = f(jnp.array([100.0, 400.0, 2000.0]), jnp.linspace(0.0, 2.0, 50))
    assert out.shape == (3, 50) and bool(jnp.all(jnp.isfinite(out)))


@pytest.mark.parametrize("sgn", [-1, 1])
def test_physical_units_vs_phi_recurs(sgn):
    cm = pytest.importorskip("camb.mathutils")
    f = jax.jit(hs.phi_phys)
    fmap = jax.jit(hs.olver_map_phys)
    K = sgn * 0.05 / R_H**2
    R = 1.0 / np.sqrt(abs(K))
    chi = np.linspace(1.0, 14000.0, 300)
    for l, qx in ((2, 2e-4), (3, 1e-4), (20, 3e-3), (150, 0.012), (1000, 0.08), (3000, 0.5)):
        nu = qx * R
        if sgn > 0:
            nu = float(max(np.round(nu), l + 5))
        q = nu / R
        ref = cm.phi_recurs(l, sgn, nu, chi / R)
        val = np.asarray(f(jnp.full(chi.shape, l), q, chi, K))
        assert np.max(np.abs(val - ref)) / cm.phi_first_peak_amplitude(l, sgn, nu) < TOL
        x_eff, w, ok = fmap(jnp.full(chi.shape, l), q, chi, K)
        if l > 2:
            jl = spherical_jn(l, np.asarray(x_eff))
            m = np.asarray(ok)
            assert np.max(np.abs(np.asarray(w)[m] * jl[m] - ref[m]), initial=0.0) / \
                cm.phi_first_peak_amplitude(l, sgn, nu) < TOL


def test_physical_units_continuous_through_flat():
    rng = np.random.default_rng(0)
    n = 300
    l = rng.integers(2, 3001, n)
    q = np.exp(rng.uniform(np.log(1e-5), np.log(0.5), n))
    chi = rng.uniform(1.0, 14000.0, n)
    flat = spherical_jn(l, q * chi)
    def main(l, q, c, K):  # fallbacks never fire this close to K = 0
        return hs.phi_phys(l, q, c, K, recursion=False, gegenbauer=False)

    fphi = jax.jit(main)
    gphi = jax.jit(jax.vmap(jax.grad(main, argnums=(1, 2, 3)), in_axes=(0, 0, 0, None)))
    g0 = [np.asarray(a) for a in gphi(l, q, chi, 0.0)]
    assert np.max(np.abs(np.asarray(fphi(l, q, chi, 0.0)) - flat)) < 1e-6
    for K in (-1e-20, 1e-20, -1e-12, 1e-12):
        val = np.asarray(fphi(l, q, chi, K))
        assert np.max(np.abs(val - flat)) < 2e-5
        x, w, ok = (np.asarray(a) for a in hs.olver_map_phys(l, q, chi, K))
        nf = np.abs(K) * l * (l + 1.0) / q**2 < 1e-2
        assert np.max(np.abs(x[nf] / (q[nf] * chi[nf]) - 1)) < 1e-2 * 0.2
        assert np.max(np.abs(w[nf] - 1)) < 1e-2
        g = [np.asarray(a) for a in gphi(l, q, chi, K)]
        assert all(np.all(np.isfinite(a)) for a in g)
        if abs(K) < 1e-15:  # every point near-flat: derivatives continuous through 0
            for gi, g0i in zip(g, g0):
                assert np.max(np.abs(gi - g0i)) <= 1e-6 * np.max(np.abs(g0i))


def test_physical_units_closed_beyond_quarter_sphere():
    """Omega_k = -0.3-like closed case with chi_u up to 3 (folding, incl. l <= 2)."""
    cm = pytest.importorskip("camb.mathutils")
    K = 0.3 / R_H**2
    R = 1.0 / np.sqrt(K)
    chi = np.linspace(1.0, 3.0 * R, 200)
    ls = np.array([0, 1, 2, 3, 50])
    nus = ls + 20.0
    val = np.asarray(jax.jit(hs.phi_phys)(ls[:, None], (nus / R)[:, None], chi[None, :], K))
    for i, (l, nu) in enumerate(zip(ls, nus)):
        ref = cm.phi_recurs(int(l), 1, float(nu), chi / R)
        assert np.max(np.abs(val[i] - ref)) / cm.phi_first_peak_amplitude(int(l), 1, float(nu)) < TOL
