#!/usr/bin/env python3
"""Smooth (C¹) interpolation for thermo tables.

``jnp.interp`` is piecewise linear: every table node is a kink. Tabulated
recombination quantities (x_e from Recfast, the visibility and its derivatives)
are resampled onto parameter-dependent τ grids, so those kinks alias against the
LOS quadrature and make the C_ℓ wiggle as parameters change (≈1e-3 in low-ℓ TT
over δω_c ~ 1e-4), which breaks finite-difference / AD gradient agreement and
HMC energy conservation. Cubic Hermite interpolation with centred-difference
slopes removes the kinks while keeping the table values exactly.
"""

from __future__ import annotations

import jax.numpy as jnp


def interp_cubic(xq, x, y):
    """Cubic Hermite interpolation of ``y(x)`` at ``xq`` (``x`` increasing).

    Node slopes are 3-point (quadratic) derivatives, one-sided at the ends, so
    the result is C¹, reproduces the table at the nodes, and is exact for
    quadratics on any grid. Outside ``[x[0], x[-1]]`` the value is held at the
    end point, like ``jnp.interp``.
    """
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    n = x.shape[0]
    m = _node_slopes(x, y)
    dx = jnp.diff(x)
    xq = jnp.asarray(xq)
    xc = jnp.clip(xq, x[0], x[-1])
    i = jnp.clip(jnp.searchsorted(x, xc, side="right") - 1, 0, n - 2)
    h = dx[i]
    t = (xc - x[i]) / h
    t2, t3 = t * t, t * t * t
    return ((2 * t3 - 3 * t2 + 1) * y[i] + (t3 - 2 * t2 + t) * h * m[i]
            + (-2 * t3 + 3 * t2) * y[i + 1] + (t3 - t2) * h * m[i + 1])


def _node_slopes(x, y):
    """3-point node slopes of ``interp_cubic`` (``y`` may carry trailing axes)."""
    dx = jnp.diff(x)
    if y.ndim > 1:
        dx = dx.reshape(dx.shape + (1,) * (y.ndim - 1))
    h0, h1 = dx[:-1], dx[1:]
    y0, y1, y2 = y[:-2], y[1:-1], y[2:]
    m_mid = (-h1 / (h0 * (h0 + h1)) * y0 + (h1 - h0) / (h0 * h1) * y1
             + h0 / (h1 * (h0 + h1)) * y2)
    a0, a1 = dx[0], dx[1]
    m_first = (-(2 * a0 + a1) / (a0 * (a0 + a1)) * y[0] + (a0 + a1) / (a0 * a1) * y[1]
               - a0 / (a1 * (a0 + a1)) * y[2])
    b0, b1 = dx[-2], dx[-1]
    m_last = (b1 / (b0 * (b0 + b1)) * y[-3] - (b0 + b1) / (b0 * b1) * y[-2]
              + (2 * b1 + b0) / (b1 * (b0 + b1)) * y[-1])
    return jnp.concatenate([m_first[None], m_mid, m_last[None]])


def cubic_pack_uniform(x, ys):
    """Pre-pack tables on a uniform grid ``x`` for ``interp_cubic_uniform``.

    ``ys``: (n, n_tab). Node slopes are computed once here instead of inside
    every ODE right-hand side.
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    ys = jnp.asarray(ys, dtype=jnp.float64)
    return {"x": x, "y": ys, "m": _node_slopes(x, ys),
            "x0": x[0], "inv_dx": (x.shape[0] - 1) / (x[-1] - x[0])}


def interp_cubic_uniform(xq, pack):
    """``interp_cubic`` of every packed table at ``xq`` (tables on the last axis).

    The grid is uniform (CAMB's log-tau thermo table), so the interval index is
    computed directly, as CAMB does, rather than by a binary search (a
    sequential loop on GPU). The interpolant is the same C¹ Hermite cubic; off
    a node boundary the interval is identical, and on one both neighbours agree.
    """
    x, y, m = pack["x"], pack["y"], pack["m"]
    n = x.shape[0]
    xc = jnp.clip(xq, x[0], x[-1])
    i = jnp.clip(jnp.floor((xc - pack["x0"]) * pack["inv_dx"]).astype(jnp.int32), 0, n - 2)
    x_i, x_j = x[i], x[i + 1]
    h = (x_j - x_i)[..., None]
    t = ((xc - x_i)[..., None]) / h
    t2, t3 = t * t, t * t * t
    return ((2 * t3 - 3 * t2 + 1) * y[i] + (t3 - 2 * t2 + t) * h * m[i]
            + (-2 * t3 + 3 * t2) * y[i + 1] + (t3 - t2) * h * m[i + 1])


__all__ = ["interp_cubic", "cubic_pack_uniform", "interp_cubic_uniform"]
