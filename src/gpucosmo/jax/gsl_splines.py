#!/usr/bin/env python3
"""GSL-exact 1D/2D splines in JAX (Akima, natural cubic, bicubic).

CCL builds every radial kernel, n(z), background quantity and Pk2D on GSL
interpolators. Reproducing CCL to better than 1e-4 therefore requires the
same interpolants, not just "a" cubic spline:

- ``gsl_interp_akima``: n(z), tracer kernels W(chi), chi(a), a(chi), E(a), D(a)
- ``gsl_interp_cspline`` (natural): factorizable transfers T(a) / T(k)
- ``gsl_interp2d_bicubic``: Pk2D / 2D transfers, derivatives from natural
  cubic splines along each axis

Splines are stored as per-interval polynomial coefficients
``y_i + b_i dx + c_i dx^2 + d_i dx^3`` so evaluation, derivatives and exact
integration share one representation. Everything is jittable and
differentiable with respect to the node values.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


class Spline1D(NamedTuple):
    """Piecewise-cubic spline on nodes ``x`` (ascending)."""

    x: jnp.ndarray
    y: jnp.ndarray
    b: jnp.ndarray
    c: jnp.ndarray
    d: jnp.ndarray


def _search_index(x, x_eval):
    n = x.shape[0]
    return jnp.clip(jnp.searchsorted(x, x_eval, side="right") - 1, 0, n - 2)


def _corrected(x, x_eval, guess):
    """Exact ``searchsorted`` interval from a guess within one node of it."""
    n = x.shape[0]
    i = jnp.clip(guess, 0, n - 2)
    i = i - (x_eval < x[i]).astype(i.dtype) + (x_eval >= x[i + 1]).astype(i.dtype)
    return jnp.clip(i, 0, n - 2)


def _uniform_index(x, x_eval):
    n = x.shape[0]
    g = jnp.floor((x_eval - x[0]) * ((n - 1) / (x[-1] - x[0])))
    return _corrected(x, x_eval, jnp.nan_to_num(g).astype(jnp.int32))


def _inv1p_uniform_index(x, x_eval):
    # x = 1/(1+z) with z uniform (ascending x <-> descending z): index the z grid.
    n = x.shape[0]
    z_lo, z_hi = 1.0 / x[-1] - 1.0, 1.0 / x[0] - 1.0
    zq = 1.0 / jnp.maximum(x_eval, 1e-300) - 1.0
    jz = jnp.floor((zq - z_lo) * ((n - 1) / (z_hi - z_lo)))
    return _corrected(x, x_eval, (n - 2) - jnp.nan_to_num(jz).astype(jnp.int32))


def _grid_kind(x):
    """0: uniform in x, 1: uniform in 1/x - 1 (a = 1/(1+z)), 2: general.

    "Uniform" means every node within a quarter spacing of the linear grid,
    which keeps the direct-index guess within one node; ``_corrected`` then
    returns exactly the binary-search interval.
    """
    n = x.shape[0]
    t = jnp.arange(n, dtype=x.dtype) / (n - 1)

    def ok(u):
        lin = u[0] + (u[-1] - u[0]) * t
        return jnp.all(jnp.abs(u - lin) <= 0.25 * jnp.abs(u[-1] - u[0]) / (n - 1))

    zz = 1.0 / jnp.maximum(x[::-1], 1e-300) - 1.0
    return jnp.where(ok(x), 0, jnp.where((x[0] > 0) & ok(zz), 1, 2))


def _interval_index(x: jnp.ndarray, x_eval: jnp.ndarray) -> jnp.ndarray:
    """GSL bsearch: i with x_i <= x < x_{i+1}; x == x_{n-1} maps to n-2.

    On uniform (or a = 1/(1+z), z-uniform) grids the interval is computed
    directly instead of by a binary search (a sequential loop over the whole
    query array on GPU); the result is identical. The grid test depends only
    on ``x``, so under ``vmap`` over cosmologies with a shared grid the branch
    is a real conditional.
    """
    x = jnp.asarray(x)
    x_eval = jnp.asarray(x_eval)
    n = x.shape[0]
    if n < 4:
        return _search_index(x, x_eval)
    return jax.lax.switch(_grid_kind(x), (_uniform_index, _inv1p_uniform_index, _search_index),
                          x, x_eval)


def akima_spline(x: jnp.ndarray, y: jnp.ndarray) -> Spline1D:
    """Non-periodic ``gsl_interp_akima`` coefficients (GSL akima.c)."""
    x = jnp.asarray(x, dtype=float)
    y = jnp.asarray(y, dtype=float)
    h = jnp.diff(x)
    m = jnp.diff(y) / h
    # GSL boundary slopes: m[-2], m[-1] and m[n-1], m[n].
    m_lo2 = 3.0 * m[0] - 2.0 * m[1]
    m_lo1 = 2.0 * m[0] - m[1]
    m_hi1 = 2.0 * m[-1] - m[-2]
    m_hi2 = 3.0 * m[-1] - 2.0 * m[-2]
    mm = jnp.concatenate([jnp.stack([m_lo2, m_lo1]), m, jnp.stack([m_hi1, m_hi2])])
    # For interval i (0..n-2): m[i-2]..m[i+2] are mm[i]..mm[i+4].
    m_im2 = mm[:-4]
    m_im1 = mm[1:-3]
    m_i = mm[2:-2]
    m_ip1 = mm[3:-1]
    m_ip2 = mm[4:]
    ne = jnp.abs(m_ip1 - m_i) + jnp.abs(m_im1 - m_im2)
    ne_next = jnp.abs(m_ip2 - m_ip1) + jnp.abs(m_i - m_im1)
    safe_ne = jnp.where(ne == 0.0, 1.0, ne)
    safe_ne_next = jnp.where(ne_next == 0.0, 1.0, ne_next)
    alpha_i = jnp.abs(m_im1 - m_im2) / safe_ne
    alpha_ip1 = jnp.abs(m_i - m_im1) / safe_ne_next
    tl_ip1 = jnp.where(
        ne_next == 0.0, m_i, (1.0 - alpha_ip1) * m_i + alpha_ip1 * m_ip1
    )
    b_gen = (1.0 - alpha_i) * m_im1 + alpha_i * m_i
    c_gen = (3.0 * m_i - 2.0 * b_gen - tl_ip1) / h
    d_gen = (b_gen + tl_ip1 - 2.0 * m_i) / (h * h)
    flat = ne == 0.0
    b = jnp.where(flat, m_i, b_gen)
    c = jnp.where(flat, 0.0, c_gen)
    d = jnp.where(flat, 0.0, d_gen)
    return Spline1D(x, y, b, c, d)


def cspline_spline(x: jnp.ndarray, y: jnp.ndarray) -> Spline1D:
    """Natural ``gsl_interp_cspline`` coefficients (GSL cspline.c)."""
    x = jnp.asarray(x, dtype=float)
    y = jnp.asarray(y, dtype=float)
    n = x.shape[0]
    h = jnp.diff(x)
    slope = jnp.diff(y) / h
    if n == 2:
        cc = jnp.zeros(2)
    else:
        rhs = 3.0 * (slope[1:] - slope[:-1])
        diag = 2.0 * (h[:-1] + h[1:])
        off = h[1:-1]
        lower = jnp.concatenate([jnp.zeros(1), off])
        upper = jnp.concatenate([off, jnp.zeros(1)])
        # The matrix depends only on the nodes: invert it once (one multi-RHS
        # solve) and apply it as a matmul. Splines built on a shared grid (the
        # rows / columns of a bicubic table, a batch of cosmologies) then share
        # the factorization instead of launching one solve per system.
        inv = jax.lax.linalg.tridiagonal_solve(
            lower, diag, upper, jnp.eye(n - 2, dtype=rhs.dtype)
        )
        interior = inv @ rhs
        cc = jnp.concatenate([jnp.zeros(1), interior, jnp.zeros(1)])
    b = slope - h * (cc[1:] + 2.0 * cc[:-1]) / 3.0
    d = (cc[1:] - cc[:-1]) / (3.0 * h)
    return Spline1D(x, y, b, cc[:-1], d)


def spline_eval(s: Spline1D, x_eval: jnp.ndarray) -> jnp.ndarray:
    """Evaluate inside [x0, x_{n-1}] (callers handle extrapolation)."""
    i = _interval_index(s.x, x_eval)
    dx = x_eval - s.x[i]
    return s.y[i] + dx * (s.b[i] + dx * (s.c[i] + dx * s.d[i]))


def spline_deriv(s: Spline1D, x_eval: jnp.ndarray) -> jnp.ndarray:
    i = _interval_index(s.x, x_eval)
    dx = x_eval - s.x[i]
    return s.b[i] + dx * (2.0 * s.c[i] + 3.0 * s.d[i] * dx)


def spline_deriv2(s: Spline1D, x_eval: jnp.ndarray) -> jnp.ndarray:
    i = _interval_index(s.x, x_eval)
    dx = x_eval - s.x[i]
    return 2.0 * s.c[i] + 6.0 * s.d[i] * dx


def _poly_integral(s: Spline1D, i: jnp.ndarray, dx: jnp.ndarray) -> jnp.ndarray:
    return dx * (
        s.y[i] + dx * (0.5 * s.b[i] + dx * (s.c[i] / 3.0 + 0.25 * s.d[i] * dx))
    )


def spline_cumulative(s: Spline1D) -> jnp.ndarray:
    """Integral from x0 to each node; shape (n,)."""
    h = jnp.diff(s.x)
    seg = _poly_integral(s, jnp.arange(h.shape[0]), h)
    return jnp.concatenate([jnp.zeros(1), jnp.cumsum(seg)])


def spline_integral(s: Spline1D, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """``gsl_spline_eval_integ`` over [a, b] with a, b inside the node range."""
    cum = spline_cumulative(s)

    def antideriv(xv):
        i = _interval_index(s.x, xv)
        return cum[i] + _poly_integral(s, i, xv - s.x[i])

    return antideriv(b) - antideriv(a)


def f1d_eval(
    s: Spline1D,
    x_eval: jnp.ndarray,
    *,
    y_lo: jnp.ndarray | float = 0.0,
    y_hi: jnp.ndarray | float = 0.0,
) -> jnp.ndarray:
    """``ccl_f1d_t_eval`` with ``ccl_f1d_extrap_const`` on both sides."""
    inner = spline_eval(s, jnp.clip(x_eval, s.x[0], s.x[-1]))
    return jnp.where(x_eval < s.x[0], y_lo, jnp.where(x_eval > s.x[-1], y_hi, inner))


class Bicubic2D(NamedTuple):
    """``gsl_interp2d_bicubic`` on (x, y) with z of shape (nx, ny)."""

    x: jnp.ndarray
    y: jnp.ndarray
    z: jnp.ndarray
    zx: jnp.ndarray
    zy: jnp.ndarray
    zxy: jnp.ndarray


def _cspline_node_derivs(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    s = cspline_spline(x, y)
    # Derivative at nodes: b_i for i < n-1, end node from last interval.
    h_last = x[-1] - x[-2]
    last = s.b[-1] + h_last * (2.0 * s.c[-1] + 3.0 * s.d[-1] * h_last)
    return jnp.concatenate([s.b, last[None]])


def bicubic_spline(x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray) -> Bicubic2D:
    """Precompute GSL bicubic derivative tables (bicubic.c init)."""
    x = jnp.asarray(x, dtype=float)
    y = jnp.asarray(y, dtype=float)
    z = jnp.asarray(z, dtype=float)
    zx = jax.vmap(lambda col: _cspline_node_derivs(x, col), in_axes=1, out_axes=1)(z)
    zy = jax.vmap(lambda row: _cspline_node_derivs(y, row))(z)
    zxy = jax.vmap(lambda col: _cspline_node_derivs(x, col), in_axes=1, out_axes=1)(zy)
    return Bicubic2D(x, y, z, zx, zy, zxy)


def _hermite(t: jnp.ndarray):
    t2 = t * t
    t3 = t2 * t
    return (
        2.0 * t3 - 3.0 * t2 + 1.0,
        t3 - 2.0 * t2 + t,
        -2.0 * t3 + 3.0 * t2,
        t3 - t2,
    )


def bicubic_eval(s: Bicubic2D, xv: jnp.ndarray, yv: jnp.ndarray) -> jnp.ndarray:
    """Evaluate inside the grid (bicubic Hermite patch == GSL expansion)."""
    i = _interval_index(s.x, xv)
    j = _interval_index(s.y, yv)
    dx = s.x[i + 1] - s.x[i]
    dy = s.y[j + 1] - s.y[j]
    t = (xv - s.x[i]) / dx
    u = (yv - s.y[j]) / dy
    h0t, h1t, h2t, h3t = _hermite(t)
    h0u, h1u, h2u, h3u = _hermite(u)

    def corner(arr, ii, jj):
        return arr[ii, jj]

    out = 0.0
    for ci, (hvt, hdt) in ((i, (h0t, h1t)), (i + 1, (h2t, h3t))):
        for cj, (hvu, hdu) in ((j, (h0u, h1u)), (j + 1, (h2u, h3u))):
            out = out + (
                hvt * hvu * corner(s.z, ci, cj)
                + hdt * dx * hvu * corner(s.zx, ci, cj)
                + hvt * hdu * dy * corner(s.zy, ci, cj)
                + hdt * hdu * dx * dy * corner(s.zxy, ci, cj)
            )
    return out


def bicubic_deriv_x(s: Bicubic2D, xv: jnp.ndarray, yv: jnp.ndarray) -> jnp.ndarray:
    return jax.jvp(lambda xx: bicubic_eval(s, xx, yv), (xv,), (jnp.ones_like(xv),))[1]


def bicubic_deriv_xx(s: Bicubic2D, xv: jnp.ndarray, yv: jnp.ndarray) -> jnp.ndarray:
    return jax.jvp(
        lambda xx: bicubic_deriv_x(s, xx, yv), (xv,), (jnp.ones_like(xv),)
    )[1]


__all__ = [
    "Bicubic2D",
    "Spline1D",
    "akima_spline",
    "bicubic_deriv_x",
    "bicubic_deriv_xx",
    "bicubic_eval",
    "bicubic_spline",
    "cspline_spline",
    "f1d_eval",
    "spline_cumulative",
    "spline_deriv",
    "spline_deriv2",
    "spline_eval",
    "spline_integral",
]
