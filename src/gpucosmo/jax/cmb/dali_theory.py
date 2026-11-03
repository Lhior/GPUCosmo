#!/usr/bin/env python3
"""Second-order DALI expansion of the JAX EB CMB theory.

Around a reference cosmology x₀ (THEORY_PARAM_NAMES order) the spectra are

    C(x) ≈ C₀ + J·d + ½ dᵀ H d,   d = x − x₀,

with J from forward mode (``theory_value_and_jacobian``) and H from central
differences of J (16 Jacobians). H0 is not expanded: it is recomputed exactly
from θ_MC (cheap). Evaluating the DALI expansion costs microseconds, so it can drive
HMC trajectories (``mcmc.dali_hmc``) and optimizers, while the exact theory
is used only where exactness matters (accept/reject, final checks).

The DALI expansion is a plain pytree (``DALITheory``), so it can be closed over in
jitted functions, saved with ``save`` / ``load`` and rebuilt between warm-up
stages (it must stay *fixed* while sampling for the chain to be exact).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gpucosmo.jax.background.cosmology import BackgroundTables
from gpucosmo.jax.cmb.jax_eb_theory import THEORY_PARAM_NAMES, theory_value_and_jacobian
from gpucosmo.jax.cmb.theory import TheorySpectra, solve_h0_from_theta_mc

jax.config.update("jax_enable_x64", True)

_LEAVES = ("dl_tt", "dl_te", "dl_ee", "dl_kk")
# ~1σ (DESI+CMB w0wa) finite-difference half-steps for the Hessian: a fit of the
# curvature over the posterior's extent rather than an infinitesimal derivative.
DEFAULT_STEPS = dict(logA=0.015, ns=0.004, theta_mc_100=3e-4, ombh2=1.5e-4, omch2=1.2e-3,
                     tau=0.007, w0=0.06, wa=0.25)


@jax.tree_util.register_pytree_node_class
@dataclass
class DALITheory:
    x0: jnp.ndarray  # (8,)
    y0: dict  # leaf -> (n,)
    jac: dict  # leaf -> (8, n)
    hess: dict  # leaf -> (8, 8, n)

    def tree_flatten(self):
        return (self.x0, self.y0, self.jac, self.hess), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)

    def save(self, path: Path) -> None:
        arrays = {"x0": np.asarray(self.x0)}
        for k in _LEAVES:
            arrays[f"y0_{k}"] = np.asarray(self.y0[k])
            arrays[f"jac_{k}"] = np.asarray(self.jac[k])
            arrays[f"hess_{k}"] = np.asarray(self.hess[k])
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> DALITheory:
        z = np.load(path)
        return cls(
            jnp.asarray(z["x0"]),
            {k: jnp.asarray(z[f"y0_{k}"]) for k in _LEAVES},
            {k: jnp.asarray(z[f"jac_{k}"]) for k in _LEAVES},
            {k: jnp.asarray(z[f"hess_{k}"]) for k in _LEAVES},
        )


_VJ_CACHE: dict = {}


def _jitted_value_and_jacobian(tables: BackgroundTables, lmax: int, k_chunk: int):
    """One jitted (value, Jacobian) function per (tables, lmax, k_chunk).

    Building a fresh ``jax.jit(lambda ...)`` per expansion retraced and
    recompiled the whole theory Jacobian every time (≈5–8 min of CPU with the
    GPU idle); cached, re-expansions cost only the ~10 s device evaluation.
    """
    key = (id(tables), lmax, k_chunk)
    fn = _VJ_CACHE.get(key)
    if fn is None:
        fn = jax.jit(lambda x: theory_value_and_jacobian(tables, x, lmax=lmax,
                                                         k_chunk=k_chunk))
        _VJ_CACHE[key] = (fn, tables)  # keep tables alive so id() stays unique
        return fn
    return fn[0]


def build_dali_theory(
    tables: BackgroundTables,
    x0,
    *,
    lmax: int = 2508,
    steps: dict | None = None,
    second_order: bool = True,
    k_chunk: int = 48,
    verbose: bool = True,
) -> DALITheory:
    """Expand the theory around ``x0``: 1 Jacobian (+16 for the Hessian)."""
    steps = {**DEFAULT_STEPS, **(steps or {})}
    x0 = jnp.asarray(x0, dtype=jnp.float64)
    vj = _jitted_value_and_jacobian(tables, int(lmax), int(k_chunk))
    t0 = time.perf_counter()
    y, jac = vj(x0)
    y0 = {k: getattr(y, k) for k in _LEAVES}
    j0 = {k: getattr(jac, k) for k in _LEAVES}
    if verbose:
        print(f"dali: value+jacobian {time.perf_counter() - t0:.1f}s", flush=True)
    hess = {k: jnp.zeros((8, 8) + y0[k].shape) for k in _LEAVES}
    if second_order:
        rows = {k: [] for k in _LEAVES}
        for i, name in enumerate(THEORY_PARAM_NAMES):
            h = steps[name]
            e = jnp.zeros(8).at[i].set(h)
            _, jp = vj(x0 + e)
            _, jm = vj(x0 - e)
            for k in _LEAVES:
                rows[k].append((getattr(jp, k) - getattr(jm, k)) / (2.0 * h))
            if verbose:
                print(f"dali: hessian row {name} {time.perf_counter() - t0:.1f}s", flush=True)
        for k in _LEAVES:
            hk = jnp.stack(rows[k])  # (8 rows i, 8 cols j, n) = ∂_i J_j
            hess[k] = 0.5 * (hk + jnp.swapaxes(hk, 0, 1))
    return DALITheory(x0, y0, j0, hess)


def dali_spectra(tt: DALITheory, tables: BackgroundTables, x8) -> TheorySpectra:
    """DALI TheorySpectra at ``x8``; H0 is solved exactly from θ_MC."""
    x8 = jnp.asarray(x8, dtype=jnp.float64)
    d = x8 - tt.x0
    out = {}
    for k in _LEAVES:
        lin = jnp.tensordot(d, tt.jac[k], axes=1)
        quad = 0.5 * jnp.einsum("i,j,ijn->n", d, d, tt.hess[k])
        val = tt.y0[k] + lin + quad
        if k != "dl_te":
            # Far-tail extrapolation can go negative (NaN in the likelihood); those
            # proposals are rejected by the exact accept anyway, so just floor them.
            val = jnp.maximum(val, 1e-3 * jnp.abs(tt.y0[k]))
        out[k] = val
    p = dict(zip(THEORY_PARAM_NAMES, (x8[i] for i in range(8))))
    h0 = solve_h0_from_theta_mc(tables, p["ombh2"], p["omch2"], p["w0"], p["wa"],
                                p["theta_mc_100"])
    return TheorySpectra(dl_tt=out["dl_tt"], dl_te=out["dl_te"], dl_ee=out["dl_ee"],
                         dl_kk=out["dl_kk"], H0=h0)


__all__ = ["DEFAULT_STEPS", "DALITheory", "build_dali_theory", "dali_spectra"]
