"""Interpolation operators on static knots (host-built, applied on device).

The full-shape pipeline interpolates between fixed grids (velocileptors'
k / q grids, the likelihood k grid) and, through Alcock–Paczyński and h, at
cosmology-dependent points. Cubic splines are linear in the data, so for fixed
knots the spline is a fixed linear operator: built here once with SciPy (the
same routines the reference code calls, applied to identity data) and applied
with ``jnp``. Points that depend on the cosmology use the piecewise-polynomial
form (``PPoly`` coefficients, also linear in the data) with a traced
``searchsorted``; outside the knots the end pieces are continued, as SciPy's
``extrapolate=True`` does.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from scipy import interpolate


def _scipy_spline(x, y, kind):
    if kind == "natural":
        return interpolate.CubicSpline(x, y, bc_type="natural", axis=0)
    if kind == "not-a-knot":
        return interpolate.make_interp_spline(x, y, k=3, axis=0)
    raise ValueError(kind)


def _ppoly_tensor(x, kind):
    """Coefficients (4, n-1, n) of the spline of identity data on every [x_i, x_{i+1}].

    Re-expanded at each knot from derivatives at the interval midpoint, so the
    pieces are indexed by the knots for either boundary condition (a
    not-a-knot B-spline has no breakpoints at x_1 and x_{n-2}).
    """
    n = len(x)
    sp = _scipy_spline(x, np.eye(n), kind)
    xm = 0.5 * (x[1:] + x[:-1])
    d = [sp(xm, nu) for nu in range(4)]  # each (n-1, n)
    delta = (xm - x[:-1])[:, None]
    c3 = d[3] / 6.0
    c2 = d[2] / 2.0 - d[3] / 2.0 * delta
    c1 = d[1] - d[2] * delta + d[3] / 2.0 * delta**2
    c0 = d[0] - d[1] * delta + d[2] / 2.0 * delta**2 - d[3] / 6.0 * delta**3
    return np.stack([c3, c2, c1, c0])


class StaticSpline:
    """Cubic spline through data ``y`` (leading axis = knots) at fixed knots ``x``.

    ``kind="natural"`` (SciPy ``CubicSpline(bc_type='natural')``) or
    ``"not-a-knot"`` (``interp1d(kind='cubic')`` / ``make_interp_spline(k=3)``).
    """

    def __init__(self, x, kind="natural"):
        import jax

        self.x = np.asarray(x, dtype=np.float64)
        self.kind = kind
        # A concrete constant even when built inside a trace: instances are cached
        # (``growth_jax._static_spline``) and reused by other jitted programs.
        with jax.ensure_compile_time_eval():
            self._c = jnp.asarray(_ppoly_tensor(self.x, kind))  # (4, n-1, n)

    def matrix(self, x_eval, nu=0):
        """Dense (n_eval, n) operator for static evaluation points (derivative ``nu``)."""
        sp = _scipy_spline(self.x, np.eye(len(self.x)), self.kind)
        return jnp.asarray(sp(np.asarray(x_eval, dtype=np.float64), nu))

    def coeffs(self, y):
        return jnp.tensordot(self._c, y, axes=([2], [0]))  # (4, n-1, ...)

    def __call__(self, y, xq):
        """Spline of ``y`` at traced points ``xq`` (end pieces extrapolate)."""
        c = self.coeffs(y)
        x = jnp.asarray(self.x)
        i = jnp.clip(jnp.searchsorted(x, xq, side="right") - 1, 0, len(self.x) - 2)
        dx = xq - x[i]
        ci = c[:, i]
        dx = dx.reshape(dx.shape + (1,) * (ci.ndim - 1 - dx.ndim))
        return ((ci[0] * dx + ci[1]) * dx + ci[2]) * dx + ci[3]


def linear_interp_operator(x_eval, xp):
    """(idx, w) so that ``f[..., idx]*(1-w) + f[..., idx+1]*w == np.interp(x_eval, xp, f)``."""
    x_eval = np.asarray(x_eval, dtype=np.float64)
    xp = np.asarray(xp, dtype=np.float64)
    xe = np.clip(x_eval, xp[0], xp[-1])
    idx = np.clip(np.searchsorted(xp, xe, side="right") - 1, 0, len(xp) - 2)
    w = (xe - xp[idx]) / (xp[idx + 1] - xp[idx])
    return jnp.asarray(idx), jnp.asarray(w)


def apply_linear(op, f):
    idx, w = op
    return f[..., idx] * (1.0 - w) + f[..., idx + 1] * w


class LogInterp:
    """velocileptors ``Utils.loginterp`` for static (x, xx), positive data.

    Natural cubic spline in log x inside [x0, xN]; power laws with the spline's
    end log-slopes outside. The reference first searches for a same-sign run of
    points; power spectra are positive so that run is the whole grid.
    """

    def __init__(self, x, xx):
        x = np.asarray(x, dtype=np.float64)
        xx = np.asarray(xx, dtype=np.float64)
        sp = StaticSpline(np.log(x), "natural")
        self.x0, self.xn = x[0], x[-1]
        self.xx = jnp.asarray(xx)
        self.inside = jnp.asarray((xx > x[0]) & (xx < x[-1]))
        self.lo = jnp.asarray(xx <= x[0])
        self.hi = jnp.asarray(xx >= x[-1])
        self.M = sp.matrix(np.log(np.clip(xx, x[0], x[-1])))
        d = sp.matrix(np.log([x[0], x[-1]]), nu=1)
        self.d0, self.dn = d[0], d[1]

    def __call__(self, y):
        y = jnp.asarray(y)
        lneff = (self.d0 @ y) / y[0]
        rneff = (self.dn @ y) / y[-1]
        inner = self.M @ y
        low = y[0] * jnp.where(self.lo, (self.xx / self.x0) ** lneff, 0.0)
        high = y[-1] * jnp.where(self.hi, (self.xx / self.xn) ** rneff, 0.0)
        return jnp.where(self.inside, inner, 0.0) + low + high


__all__ = ["LogInterp", "StaticSpline", "apply_linear", "linear_interp_operator"]
