"""JAX evaluation of SciPy (FITPACK) splines and regular-grid interpolators, value for value.

Ports read the knots and coefficients of the SciPy objects Cobaya builds (``UnivariateSpline``,
``RectBivariateSpline``, ``RegularGridInterpolator``) and evaluate them here, differentiably.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import lax


def _basis(t: jnp.ndarray, k: int, n_coef: int, x):
    """(first index, the k+1 non-zero B-spline basis values) at scalar ``x`` (Cox–de Boor)."""
    i = jnp.clip(jnp.searchsorted(t, x, side="right") - 1, k, n_coef - 1)
    N = [jnp.ones_like(x)]
    left, right = [None], [None]
    for j in range(1, k + 1):
        left.append(x - t[i + 1 - j])
        right.append(t[i + j] - x)
        saved = jnp.zeros_like(x)
        for r in range(j):
            temp = N[r] / (right[r + 1] + left[j - r])
            N[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        N.append(saved)
    return i - k, jnp.stack(N)


class Spline1D:
    """``scipy.interpolate.UnivariateSpline`` (or any (t, c, k) B-spline). ``ext``: 0
    extrapolate, 2 → ``out_value`` outside [t_k, t_{n−k−1}] (SciPy raises), 3 clamp."""

    def __init__(self, spl=None, *, tck=None, ext: int = 0, out_value=-jnp.inf):
        t, c, k = tck if tck is not None else spl._eval_args
        self.t = np.asarray(t, dtype=np.float64)
        self.k = int(k)
        self.n = len(self.t) - self.k - 1
        self.c = np.asarray(c, dtype=np.float64)[: self.n]
        self.lo, self.hi = float(self.t[self.k]), float(self.t[self.n])
        self.ext, self.out = int(ext), out_value

    def __call__(self, x):
        x = jnp.asarray(x, dtype=jnp.float64)
        xe = jnp.clip(x, self.lo, self.hi) if self.ext == 3 else x
        i0, N = _basis(jnp.asarray(self.t), self.k, self.n, xe)
        v = jnp.dot(lax.dynamic_slice(jnp.asarray(self.c), (i0,), (self.k + 1,)), N)
        if self.ext == 2:
            v = jnp.where((x >= self.lo) & (x <= self.hi), v, self.out)
        return v


class Spline2D:
    """``scipy.interpolate.RectBivariateSpline``: arguments clamped to the knot range, as
    FITPACK's ``fpbisp``."""

    def __init__(self, spl):
        tx, ty, c = spl.tck
        kx, ky = spl.degrees
        self.tx, self.ty = np.asarray(tx, dtype=np.float64), np.asarray(ty, dtype=np.float64)
        self.kx, self.ky = int(kx), int(ky)
        self.nx, self.ny = len(self.tx) - self.kx - 1, len(self.ty) - self.ky - 1
        self.c = np.asarray(c, dtype=np.float64).reshape(self.nx, self.ny)

    def __call__(self, x, y):
        x = jnp.clip(jnp.asarray(x, dtype=jnp.float64), self.tx[self.kx], self.tx[self.nx])
        y = jnp.clip(jnp.asarray(y, dtype=jnp.float64), self.ty[self.ky], self.ty[self.ny])
        ix, Nx = _basis(jnp.asarray(self.tx), self.kx, self.nx, x)
        iy, Ny = _basis(jnp.asarray(self.ty), self.ky, self.ny, y)
        blk = lax.dynamic_slice(jnp.asarray(self.c), (ix, iy), (self.kx + 1, self.ky + 1))
        return Nx @ blk @ Ny


class RegularGrid:
    """``scipy.interpolate.RegularGridInterpolator`` (method "linear") with ``fill_value``
    outside the grid (``bounds_error=False``)."""

    def __init__(self, rgi):
        if rgi.method != "linear":
            raise NotImplementedError(f"RegularGridInterpolator method {rgi.method!r}")
        self.grid = [np.asarray(g, dtype=np.float64) for g in rgi.grid]
        self.values = np.asarray(rgi.values, dtype=np.float64)
        self.fill = rgi.fill_value

    def __call__(self, pt):
        pt = [jnp.asarray(p, dtype=jnp.float64) for p in pt]
        idx, w = [], []
        inside = jnp.asarray(True)
        for g, p in zip(self.grid, pt):
            gj = jnp.asarray(g)
            i = jnp.clip(jnp.searchsorted(gj, p, side="right") - 1, 0, len(g) - 2)
            idx.append(i)
            w.append((p - gj[i]) / (gj[i + 1] - gj[i]))
            inside &= (p >= g[0]) & (p <= g[-1])
        vals = jnp.asarray(self.values)
        out = jnp.asarray(0.0)
        d = len(self.grid)
        for corner in range(2**d):
            bits = [(corner >> j) & 1 for j in range(d)]
            wt = jnp.asarray(1.0)
            ii = []
            for j, b in enumerate(bits):
                wt = wt * (w[j] if b else 1.0 - w[j])
                ii.append(idx[j] + b)
            out = out + wt * vals[tuple(ii)]
        if self.fill is None:
            return out
        return jnp.where(inside, out, self.fill)


__all__ = ["RegularGrid", "Spline1D", "Spline2D"]
