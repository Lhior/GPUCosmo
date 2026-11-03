"""Non-differentiable likelihood terms (host callbacks, e.g. the Cobaya bridge) in gradient code.

* ``fd_term(f, x, sel, h)`` — ``f(x)`` with a central-difference derivative in the
  coordinates ``sel``: the 2·len(sel) shifted values are computed in the forward pass and
  the tangent is linear in them, so ``jax.grad`` works through it (reverse mode included).
  For optimizers and short gradient runs on a cheap theory (the MAP stage's DALI likelihood).
* ``QuadraticProxy`` / ``fit_quadratic`` — a second-order expansion of the term in
  ``x[sel]``, fitted once by finite differences: exact gradients at no per-point host cost,
  for the samplers on the compressed DALI likelihood. A quadratic in log L itself, not part of
  the DALI hierarchy: its error enters the evidence uncorrected and the DALI order check
  (``mcmc.dali_ns.dali_order_check``) does not measure it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def fd_term(f, x, sel, h):
    """``f(x)`` (scalar, any JAX-traceable function, e.g. a ``pure_callback``) whose derivative
    is the central difference with steps ``h`` along the coordinates ``sel`` (zero along the
    others)."""
    sel = np.asarray(sel, dtype=int)
    h = np.asarray(h, dtype=np.float64)
    m = sel.size

    @jax.custom_jvp
    def g(x):
        return f(x)

    @g.defjvp
    def g_jvp(primals, tangents):
        (x,), (dx,) = primals, tangents
        E = jnp.zeros((m, x.shape[0]), dtype=x.dtype).at[jnp.arange(m), sel].set(jnp.asarray(h))
        X = jnp.concatenate([x[None] + E, x[None] - E, x[None]])
        F = jax.vmap(f)(X)
        grad = (F[:m] - F[m:2 * m]) / (2.0 * jnp.asarray(h))
        grad = jnp.where(jnp.isfinite(grad), grad, 0.0)
        return F[-1], jnp.dot(grad, dx[sel])

    return g(x)


class QuadraticProxy:
    """c + gᵀd + ½ dᵀHd with d = x[sel] − x0 (arrays usable as jit arguments: ``arrays``)."""

    def __init__(self, sel, x0, c, g, H, info=None):
        self.sel = np.asarray(sel, dtype=int)
        self.arrays = {"x0": jnp.asarray(x0, dtype=jnp.float64), "c": jnp.asarray(c, dtype=jnp.float64),
                       "g": jnp.asarray(g, dtype=jnp.float64), "H": jnp.asarray(H, dtype=jnp.float64)}
        self.info = info or {}

    def __call__(self, x, arrays=None):
        a = self.arrays if arrays is None else arrays
        d = x[..., self.sel] - a["x0"]
        return a["c"] + d @ a["g"] + 0.5 * jnp.einsum("...i,ij,...j->...", d, a["H"], d)


def fit_quadratic(f_batch, x0, sel, h, *, refine: bool = True, h_max=None, log=None) -> QuadraticProxy:
    """Fit ``QuadraticProxy`` to ``f_batch`` (rows of a (n_pts, ndim) array → (n_pts,) values,
    NumPy in and out) around ``x0`` by central differences on the coordinates ``sel`` with
    steps ``h``. ``refine``: first estimate the diagonal curvature, then re-step each
    coordinate to ~1σ of the term (clipped to [h/10, h_max]) so the quadratic is the secant over
    the region the posterior occupies rather than the local tangent."""
    x0 = np.asarray(x0, dtype=np.float64)
    sel = np.asarray(sel, dtype=int)
    h = np.asarray(h, dtype=np.float64).copy()
    m = sel.size
    h_max = np.full(m, np.inf) if h_max is None else np.asarray(h_max, dtype=np.float64)

    def pts_diag(hh):
        E = np.zeros((m, x0.size))
        E[np.arange(m), sel] = hh
        return np.concatenate([x0[None], x0[None] + E, x0[None] - E])

    if refine:
        F = np.asarray(f_batch(pts_diag(h)), dtype=np.float64)
        f0, fp, fm = F[0], F[1:m + 1], F[m + 1:]
        curv = (fp - 2 * f0 + fm) / h**2
        sig = np.where((curv < 0) & np.isfinite(curv), 1.0 / np.sqrt(np.maximum(-curv, 1e-300)), h)
        h = np.clip(sig, h / 10.0, h_max)
    F = np.asarray(f_batch(pts_diag(h)), dtype=np.float64)
    f0, fp, fm = F[0], F[1:m + 1], F[m + 1:]
    g = (fp - fm) / (2 * h)
    H = np.diag((fp - 2 * f0 + fm) / h**2)
    if m > 1:
        ii, jj = np.triu_indices(m, 1)
        n_pairs = ii.size
        P = np.repeat(x0[None], 4 * n_pairs, axis=0)
        for k, (si, sj) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
            rows = np.arange(k * n_pairs, (k + 1) * n_pairs)  # one (i, j) pair per row
            P[rows, sel[ii]] += si * h[ii]
            P[rows, sel[jj]] += sj * h[jj]
        G = np.asarray(f_batch(P), dtype=np.float64).reshape(4, n_pairs)
        hij = (G[0] - G[1] - G[2] + G[3]) / (4 * h[ii] * h[jj])
        H[ii, jj] = H[jj, ii] = hij
    ok = np.all(np.isfinite(g)) and np.all(np.isfinite(H)) and np.isfinite(f0)
    if not ok:
        raise ValueError("fit_quadratic: non-finite values on the stencil (likelihood failed near x0)")
    n_eval = (2 * m + 1) * (2 if refine else 1) + (2 * m * (m - 1) if m > 1 else 0)
    if log:
        log(f"  quadratic proxy: {m} coordinates, {n_eval} evaluations, log L(x0) = {f0:.3f}")
    return QuadraticProxy(sel, x0[sel], f0, g, H, info=dict(h=h.tolist(), n_eval=int(n_eval)))


__all__ = ["QuadraticProxy", "fd_term", "fit_quadratic"]
