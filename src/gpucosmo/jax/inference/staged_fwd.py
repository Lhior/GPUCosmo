"""Forward-mode expansion coefficients of the exact theory, compiled stage by stage.

``jax.jvp`` of the whole theory (CMB + DESI FS) is one XLA program with the primal and its
tangents; compiling it needed > 20 GB of host memory (10 GB RAM + 10 GB swap, then killed).
Forward mode composes stage by stage, so each stage s' = f(s) is differentiated and compiled
on its own and the tangents are passed between stages.

For a k-th derivative along directions (a₁, …, a_k) the state carries 2^k components s_m,
m a bit mask of the directions it has been differentiated along (s_0 the primal, s_{2^j} = a_j
at the parameters, the rest 0); a stage maps them by k nested jvps of f (``_lift``):

* first order: (s, ṡ_a) → (f(s), Df[ṡ_a]);
* second order: (s, ṡ_a, ṡ_b, s̈_ab) → (f(s), Df[ṡ_a], Df[ṡ_b], D²f[ṡ_a, ṡ_b] + Df[s̈_ab]);
* order k: the chain rule (Faà di Bruno) for all 2^k mixed partials.

At the output, s_{2^k − 1} = ∂^k y/∂x_{a₁}⋯∂x_{a_k} exactly (no finite differences).
``stages`` is a list of ``(name, f)`` with ``f(state) -> state`` on pytrees; directions are
vmapped (the primal is computed once per stage call) in batches of ``batch``.
"""

from __future__ import annotations

import gc
import time
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np


def _batched(fn, batch: int, n: int):
    """Run ``fn(idx)`` on index batches [0, n) of size ``batch`` (last batch padded) and
    concatenate the outputs along axis 0."""
    outs = []
    for a in range(0, n, batch):
        idx = np.arange(a, min(a + batch, n))
        pad = batch - idx.size
        idx_p = np.concatenate([idx, np.repeat(idx[-1:], pad)]) if pad else idx
        o = fn(idx_p)
        outs.append(jax.tree_util.tree_map(lambda t, m=idx.size: np.asarray(t)[:m], o))
    return jax.tree_util.tree_map(lambda *c: np.concatenate(c), *outs)


def _const_mask(*tangent_trees) -> tuple:
    """Per state leaf: True if every tangent (all directions, all given trees) is exactly
    zero — a constant carried through the state (e.g. the ℓ grid, k grids, integer tables)."""
    leaves = [jax.tree_util.tree_leaves(t) for t in tangent_trees]
    out = []
    for parts in zip(*leaves):
        const = True
        for t in parts:
            a = np.asarray(t)
            if a.dtype != jax.dtypes.float0 and np.any(a != 0):
                const = False
                break
        out.append(const)
    return tuple(out)


def _freeze(f, mask):
    """``f`` with the state leaves flagged in ``mask`` passed through ``stop_gradient``: their
    (exactly zero) tangents become symbolic zeros, so derivative rules that are singular at
    the constant's value (√x at ℓ = 0, 1) are never evaluated (0·∞ = NaN otherwise)."""
    def g(st):
        leaves, tdef = jax.tree_util.tree_flatten(st)
        return f(jax.tree_util.tree_unflatten(
            tdef, [jax.lax.stop_gradient(l) if c else l for l, c in zip(leaves, mask)]))
    return g


def _release():
    """Free the compiled stage programs (their host memory) before compiling the next stage."""
    jax.clear_caches()
    gc.collect()


def _lift(f, k: int):
    """S (2^k state components, bit masks of the directions differentiated along) → f's 2^k
    components: k nested jvps of f."""
    if k == 0:
        return lambda S: (f(S[0]),)
    g = _lift(f, k - 1)

    def h(S):
        half = len(S) // 2
        out, tan = jax.jvp(g, (tuple(S[:half]),), (tuple(S[half:]),))
        return tuple(out) + tuple(tan)
    return h


def staged_derivatives(stages, x0, dirs, *, batch: int = 1, log: Callable = print, release: bool = True):
    """(y, D): the output y and D leaves (P, ...) = ∂^k y/∂x_{dirs[p, 0]}⋯∂x_{dirs[p, k−1]} for the
    P direction tuples ``dirs`` (P, k) of parameter indices. ``release``: free each stage's
    compiled programs before the next (``_release``; it clears every JAX cache)."""
    x0 = jnp.asarray(x0, dtype=jnp.float64)
    n = int(x0.shape[0])
    dirs = np.asarray(dirs, dtype=np.int64)
    P, k = dirs.shape
    eye = np.eye(n)
    s = x0
    T = tuple(jnp.asarray(eye[dirs[:, int(np.log2(m))]]) if m & (m - 1) == 0 else jnp.zeros((P, n), x0.dtype)
              for m in range(1, 2 ** k))
    batch = max(1, min(int(batch), P))
    for name, f in stages:
        t0 = time.perf_counter()
        F = _lift(_freeze(f, _const_mask(*T)), k)

        def one(st, *ts, F=F):
            out = F((st,) + ts)
            return out[0], out[1:]

        if batch == 1:
            # One tuple per call without vmap: XLA 0.11.2 failed buffer assignment ("DUS result
            # must alias fusion parameter") on the batch-1 vmapped nested jvp of the CMB sources.
            step1 = jax.jit(one)

            def step(st, *ts, step1=step1):
                # slice, not index: integer leaves carry float0 tangents, and t[0] of a float0
                # array is a numpy scalar JAX rejects
                y, out = step1(st, *(jax.tree_util.tree_map(lambda t: t[0:1].reshape(t.shape[1:]), v) for v in ts))
                return y, jax.tree_util.tree_map(lambda t: t[None], out)
        else:
            step = jax.jit(jax.vmap(one, in_axes=(None,) + (0,) * len(T), out_axes=(None, 0)))
        s_new = None

        def run(idx, s=s, T=T, step=step):
            nonlocal s_new
            y, out = step(s, *(jax.tree_util.tree_map(lambda a: a[idx], t) for t in T))
            s_new = y
            return out
        T = tuple(jax.tree_util.tree_map(jnp.asarray, t) for t in _batched(run, batch, P))
        s = s_new
        del step, run
        if release:
            _release()
        log(f"staged-fwd order {k}: stage {name} {time.perf_counter() - t0:.1f}s ({P} directions, batch {batch})")
    return s, T[-1]


def staged_expansion(stages, x0, *, order: int, tangent_batch: int = 0, batch: int = 1, log: Callable = print,
                  release: bool = True):
    """(y0, coef): the order-``order`` monomial coefficients (``dali_expansion.multi_indices``; leaves
    (n_terms, ...)) of the output through ``stages``: ∂^α y / α! per sorted index tuple.
    ``tangent_batch`` first-order directions (0: all n) and ``batch`` higher-order tuples per call."""
    from gpucosmo.jax.inference.dali_expansion import _factorial_of_counts, multi_indices

    n = int(np.asarray(x0).shape[0])
    blocks, y0 = [], None
    for k, idx in enumerate(multi_indices(n, order)[1:], start=1):
        b = (tangent_batch or n) if k == 1 else batch
        y, D = staged_derivatives(stages, x0, idx, batch=b, log=log, release=release)
        y0 = y if y0 is None else y0
        fac = np.asarray([_factorial_of_counts(t) for t in idx], dtype=np.float64)
        blocks.append(jax.tree_util.tree_map(
            lambda t: np.asarray(t, dtype=np.float64) / fac.reshape((-1,) + (1,) * (np.ndim(t) - 1)), D))
    coef = jax.tree_util.tree_map(lambda y, *bl: jnp.asarray(np.concatenate([np.asarray(y)[None], *bl])), y0, *blocks)
    return y0, coef


__all__ = ["staged_derivatives", "staged_expansion"]
