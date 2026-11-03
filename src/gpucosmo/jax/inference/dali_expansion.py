"""Order-p DALI expansion of an arbitrary theory ``f: x (n,) → pytree of arrays``.

DALI (Sellentin, Quartin & Amendola 2014; Sellentin 2015) expands the *model* about the
fiducial point and keeps the likelihood exact; order p = 1 is the Fisher (Gauss–Newton)
approximation, p = 2 the doublet, p = 3 the triplet (Šarčević, van der Wild & Sellentin 2026),
and so on. Here the theory block is expanded to order p,

    f(x) ≈ Σ_{α: deg ≤ p} c_α d^α,   c_α = ∂^α f(x₀) / α!,   d = u(x) − u(x₀),

with u the identity, or ln on the parameters ``log_idx`` (a polynomial in ln Ωm follows the
h ∝ Ωm^−½ power laws of a broad posterior at fixed ω_m far better than one in Ωm), and
evaluated inside the likelihood codes, so every DALI tensor of the chosen order (F, D, T, …,
and the residual terms b_a, b_ab, b_abc of a non-zero fiducial residual) is implied, for
Gaussian and non-Gaussian likelihoods alike.

The coefficients are stored in the monomial basis φ(d) = [1, d_i, d_i d_j (i ≤ j), d_i d_j d_k
(i ≤ j ≤ k), …] (``multi_indices``): f ≈ φ(d) · C. They come from

* ``build_dali_fd``: central finite differences over a design of exact evaluations with steps
  h ≈ 1σ, each stencil exact for polynomials of degree ≤ p (a polynomial theory of degree ≤ p
  is reproduced exactly; otherwise the expansion fits the derivatives over the posterior's
  extent);
* ``build_dali_fwd``: JAX forward-mode derivatives at x₀ (nested jvp, ``staged_fwd``).

The expansion is a pytree, so it can be passed into jitted code; it must stay fixed while
sampling for DALI-trajectory HMC to be exact (``mcmc.dali_hmc``). ``DALIExpansion.truncate`` gives
the lower orders of the same expansion (the DALI hierarchy), whose differences measure the
truncation error.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations_with_replacement, product
from math import comb, factorial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

DALI_ORDER = 3  # default expansion order: the triplet


def expansion_coord(x, log_idx: tuple = ()):
    """The expansion coordinates u(x): ln on the parameters ``log_idx`` (last axis), else x."""
    x = jnp.asarray(x, dtype=jnp.float64)
    if not log_idx:
        return x
    idx = np.asarray(log_idx)
    return x.at[..., idx].set(jnp.log(x[..., idx]))


# ---------------------------------------------------------------- monomial basis
@lru_cache(maxsize=None)
def multi_indices(n: int, order: int) -> tuple:
    """Per degree k = 0 … order, the (n_k, k) sorted index tuples i₁ ≤ … ≤ i_k of the monomials
    d_{i₁}⋯d_{i_k}, in the order of ``monomials``."""
    out = []
    for k in range(order + 1):
        tuples = list(combinations_with_replacement(range(n), k))
        out.append(np.asarray(tuples, dtype=np.int64).reshape(len(tuples), k))
    return tuple(out)


def n_terms(n: int, order: int) -> int:
    """Number of monomials of degree ≤ order in n variables, C(n + order, order)."""
    return comb(n + order, order)


def monomials(d: jnp.ndarray, order: int) -> jnp.ndarray:
    """φ(d) (..., n_terms): all monomials of degree ≤ ``order`` in the last axis of ``d``, by degree,
    lexicographic within a degree (``multi_indices``)."""
    n = d.shape[-1]
    out = [jnp.ones(d.shape[:-1] + (1,), d.dtype)]
    prev, prev_idx = None, None
    for k, idx in enumerate(multi_indices(n, order)[1:], start=1):
        if k == 1:
            cur = d
        else:  # degree k = (degree k − 1 monomial without its last index) × d_last
            pos = {tuple(t): q for q, t in enumerate(prev_idx)}
            head = np.asarray([pos[tuple(t[:-1])] for t in idx])
            cur = prev[..., head] * d[..., idx[:, -1]]
        out.append(cur)
        prev, prev_idx = cur, idx
    return jnp.concatenate(out, axis=-1)


def _factorial_of_counts(t) -> int:
    return int(np.prod([factorial(c) for c in np.unique(np.asarray(t), return_counts=True)[1]])) if len(t) else 1


def coef_from_derivatives(derivs: list) -> np.ndarray:
    """Monomial coefficients (n_terms, ...) from the full symmetric derivative tensors
    ``derivs[k]`` (shape (n,)*k + out): c_α = ∂^α f / α! (α! = Π multiplicity!)."""
    derivs = [np.asarray(a, dtype=np.float64) for a in derivs]
    order = len(derivs) - 1
    n = derivs[1].shape[0] if order >= 1 else 0
    out = [derivs[0][None]]
    for k, idx in enumerate(multi_indices(n, order)[1:], start=1):
        fac = np.asarray([_factorial_of_counts(t) for t in idx], dtype=np.float64)
        vals = derivs[k][tuple(idx.T)]
        out.append(vals / fac.reshape((-1,) + (1,) * (vals.ndim - 1)))
    return np.concatenate(out, axis=0)


@jax.tree_util.register_pytree_node_class
@dataclass
class DALIExpansion:
    x0: jnp.ndarray  # (n,), parameter space
    coef: dict  # leaves (n_terms(n, order), ...): monomial coefficients in u = expansion_coord(x, log_idx)
    order: int = DALI_ORDER  # (static)
    log_idx: tuple = ()  # parameters expanded in ln (static)

    def tree_flatten(self):
        return (self.x0, self.coef), (int(self.order), tuple(self.log_idx))

    @classmethod
    def tree_unflatten(cls, aux, children):
        order, log_idx = aux
        return cls(*children, order=order, log_idx=tuple(log_idx or ()))

    @property
    def n(self) -> int:
        return int(self.x0.shape[0])

    @property
    def y0(self):
        """The theory at the expansion point."""
        return jax.tree_util.tree_map(lambda c: c[0], self.coef)

    @property
    def jac(self):
        """First derivatives (n, ...) in the expansion coordinates."""
        return jax.tree_util.tree_map(lambda c: c[1:1 + self.n], self.coef)

    def truncate(self, order: int) -> DALIExpansion:
        """The same expansion to a lower ``order`` (its leading monomials)."""
        if not 0 <= order <= self.order:
            raise ValueError(f"cannot truncate an order-{self.order} expansion to order {order}")
        nt = n_terms(self.n, order)
        return DALIExpansion(self.x0, jax.tree_util.tree_map(lambda c: c[:nt], self.coef), order=order,
                      log_idx=self.log_idx)

    def __call__(self, x):
        d = expansion_coord(x, self.log_idx) - expansion_coord(self.x0, self.log_idx)
        phi = monomials(d, self.order)
        nt = phi.shape[-1]

        def ev(c):
            # one flat (n_terms)·(n_terms, m) product: vmapped over points this is a single matmul
            return (phi @ c.reshape(nt, -1)).reshape(c.shape[1:])

        return jax.tree_util.tree_map(ev, self.coef)

    def save(self, path: Path, names: tuple | None = None) -> None:
        """``names``: the expansion's parameter names (checked by ``load``)."""
        flat, _ = jax.tree_util.tree_flatten_with_path(self.coef)
        arrays = {"x0": np.asarray(self.x0), "log_idx": np.asarray(self.log_idx, dtype=np.int64),
                  "order": np.asarray(int(self.order))}
        if names is not None:
            arrays["param_names"] = np.asarray([str(n) for n in names])
        for kp, leaf in flat:
            arrays[f"coef{jax.tree_util.keystr(kp)}"] = np.asarray(leaf)
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: Path, like: dict, names: tuple | None = None) -> DALIExpansion:
        """Load into the structure of ``like`` (a pytree with the theory's output keys). ``names``:
        the analysis' theory parameters, which must match those the file was built for (files
        written without names are accepted when their dimension matches). Second-order files
        written before the order-p format (y0 / jac / hess arrays) are converted."""
        z = np.load(path)
        if names is not None:
            if "param_names" in z.files and [str(n) for n in z["param_names"]] != [str(n) for n in names]:
                raise ValueError(f"{path} was built for the theory parameters {list(z['param_names'])}, the "
                                 f"analysis has {list(names)}: rebuild it")
            if len(z["x0"]) != len(names):
                raise ValueError(f"{path} expands {len(z['x0'])} parameters, the analysis has {len(names)}: rebuild it")
        flat, treedef = jax.tree_util.tree_flatten_with_path(like)
        keys = [jax.tree_util.keystr(kp) for kp, _ in flat]
        legacy = "order" not in z.files
        first = "y0" if legacy else "coef"
        missing = [k for k in keys if f"{first}{k}" not in z.files]
        if missing:
            raise ValueError(f"{path} has no expansion coefficients for {missing[:4]}{' …' if len(missing) > 4 else ''}: "
                             "it was built for an analysis with other theory outputs (e.g. before a likelihood "
                             "was added); rebuild it (gpucosmo-dali-hmc --stage build)")
        if legacy:  # f₀ + J·d + ½ dᵀHd
            leaves = [jnp.asarray(coef_from_derivatives([z[f"y0{k}"], z[f"jac{k}"], z[f"hess{k}"]])) for k in keys]
            order = 2
        else:
            leaves = [jnp.asarray(z[f"coef{k}"]) for k in keys]
            order = int(z["order"])
        log_idx = tuple(int(i) for i in z["log_idx"]) if "log_idx" in z.files else ()
        return cls(jnp.asarray(z["x0"]), jax.tree_util.tree_unflatten(treedef, leaves), order=order, log_idx=log_idx)


# ---------------------------------------------------------------- finite-difference design
@lru_cache(maxsize=None)
def _stencil_1d(k: int, q: int) -> tuple:
    """((s, w), …): central weights for the k-th derivative on the points −m … m (unit step),
    m = ⌈q/2⌉, exact for polynomials of degree ≤ max(q, 2m) (zero weights dropped)."""
    if k == 0:
        return ((0, 1.0),)
    m = max(1, -(-q // 2))
    s = np.arange(-m, m + 1)
    V = s[None, :].astype(np.float64) ** np.arange(2 * m + 1)[:, None]
    rhs = np.zeros(2 * m + 1)
    rhs[k] = factorial(k)
    w = np.linalg.solve(V, rhs)
    return tuple((int(si), float(wi)) for si, wi in zip(s, w) if abs(wi) > 1e-12 * np.abs(w).max())


def fd_design(n: int, order: int) -> tuple[np.ndarray, np.ndarray]:
    """(S, W): integer design offsets S (P, n) (row 0 the expansion point) and weights W
    (n_terms, P) with c = W · f(u₀ + h ⊙ S) / h^α. The coefficient of d^α uses the product of
    1-D central stencils, the one along i for ∂^{α_i} exact for polynomials of degree
    p − deg α + α_i, so the estimate of every c_α is exact for theories of total degree ≤ p
    (and the second-order design is the classic 1 + 2n² point one)."""
    pts = {(0,) * n: 0}
    rows = []
    for k, idx in enumerate(multi_indices(n, order)):
        for t in idx:
            dims, counts = np.unique(t, return_counts=True)
            stencils = [_stencil_1d(int(c), order - k + int(c)) for c in counts]
            row = {}
            for combo in product(*stencils):
                s = [0] * n
                w = 1.0
                for dim, (si, wi) in zip(dims, combo):
                    s[int(dim)] = si
                    w *= wi
                p = pts.setdefault(tuple(s), len(pts))
                row[p] = row.get(p, 0.0) + w / _factorial_of_counts(t)
            rows.append(row)
    W = np.zeros((len(rows), len(pts)))
    for r, row in enumerate(rows):
        for p, w in row.items():
            W[r, p] = w
    S = np.zeros((len(pts), n), dtype=np.int64)
    for s, p in pts.items():
        S[p] = s
    return S, W


def build_dali_fd(fn_batch, x0, steps, *, order: int = DALI_ORDER, verbose: bool = True,
                    max_batch: int = 0, log_idx: tuple = ()) -> DALIExpansion:
    """Order-``order`` DALI expansion from a finite-difference design of exact evaluations.

    ``fn_batch(X)`` maps (m, n) points to a pytree with leading axis m (e.g. a jitted ``lax.map``
    of the exact theory), so no derivative program has to be compiled. Design: ``fd_design``
    with steps h_i (≈ 1σ, so the expansion fits the theory over the posterior's extent): order 1
    has 1 + 2n points, order 2 1 + 2n², order 3 1 + 4n + 2n(n − 1) + 8 C(n, 3).
    ``log_idx``: parameters expanded in ln (design ln x0 ± h/x0, ``steps`` in x units).
    """
    x0 = np.asarray(x0, dtype=np.float64)
    n = x0.size
    log_idx = tuple(int(i) for i in log_idx)
    u0 = np.asarray(expansion_coord(x0, log_idx))
    h = np.asarray(steps, dtype=np.float64).copy()
    for i in log_idx:
        h[i] = h[i] / x0[i]
    S, W = fd_design(n, int(order))
    X = u0 + S * h
    for i in log_idx:  # design points back to parameter space
        X[:, i] = np.exp(X[:, i])
    t0 = time.perf_counter()
    if max_batch and X.shape[0] > max_batch:
        chunks = []
        for a_ in range(0, X.shape[0], max_batch):
            xb = X[a_:a_ + max_batch]
            pad = max_batch - xb.shape[0]  # keep one compiled shape
            yb = fn_batch(jnp.asarray(np.concatenate([xb, np.repeat(xb[-1:], pad, 0)])))
            chunks.append(jax.tree_util.tree_map(lambda t, m=xb.shape[0]: np.asarray(t)[:m], yb))
        Y = jax.tree_util.tree_map(lambda *c: np.concatenate(c), *chunks)
    else:
        Y = jax.tree_util.tree_map(np.asarray, fn_batch(jnp.asarray(X)))
    if verbose:
        print(f"dali-fd: order {order}, {X.shape[0]} exact evaluations in {time.perf_counter() - t0:.1f}s",
              flush=True)
    bad = np.zeros(X.shape[0], dtype=bool)
    for leaf_ in jax.tree_util.tree_leaves(Y):
        bad |= ~np.all(np.isfinite(np.asarray(leaf_).reshape(X.shape[0], -1)), axis=1)
    if bad.any():
        i = int(np.flatnonzero(bad)[0])
        raise ValueError(f"exact theory non-finite at {int(bad.sum())} of {X.shape[0]} design points (e.g. design "
                         f"offset {S[i].tolist()} steps, x = {X[i].tolist()}; a solver failure or a step outside the "
                         "theory's domain): change the steps or the expansion point")
    # c_α = W_α · f / h^α
    hpow = np.concatenate([np.prod(h[idx], axis=1) if idx.shape[1] else np.ones(1)
                           for idx in multi_indices(n, int(order))])

    def leaf(Yl):
        Yl = np.asarray(Yl, dtype=np.float64)
        c = (W @ Yl.reshape(Yl.shape[0], -1)) / hpow[:, None]
        return jnp.asarray(c.reshape((W.shape[0],) + Yl.shape[1:]))

    coef = jax.tree_util.tree_map(leaf, Y)
    return DALIExpansion(jnp.asarray(x0), coef, order=int(order), log_idx=log_idx)


def build_dali_fwd(fn, x0, *, order: int = DALI_ORDER, tangent_batch: int = 0, pair_batch: int = 1,
                     verbose: bool = True) -> DALIExpansion:
    """Order-``order`` DALI expansion from JAX forward-mode derivatives of ``fn`` at ``x0`` — no
    finite differences: ∂^α f from k nested jvps along the basis directions of each sorted index
    tuple (C(n + k − 1, k) of them at degree k; ``staged_fwd.staged_expansion``). ``tangent_batch``
    first-order tangents and ``pair_batch`` higher-order direction tuples per compiled call bound
    the device memory. The expansion is the local one at ``x0`` (``build_dali_fd`` with steps
    h ≈ 1σ fits the derivatives over the posterior's extent instead)."""
    from gpucosmo.jax.inference.staged_fwd import staged_expansion

    log = print if verbose else (lambda m: None)
    x0 = jnp.asarray(x0, dtype=jnp.float64)
    _, coef = staged_expansion([("theory", fn)], x0, order=int(order), tangent_batch=tangent_batch,
                            batch=pair_batch, log=log, release=False)
    return DALIExpansion(x0, coef, order=int(order))


__all__ = ["DALI_ORDER", "DALIExpansion", "build_dali_fd", "build_dali_fwd", "coef_from_derivatives", "fd_design",
           "monomials", "multi_indices", "n_terms", "expansion_coord"]
