#!/usr/bin/env python3
"""candl ACT DR6 lensing likelihood wrapper (JAX-differentiable via candl)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp


@dataclass(frozen=True)
class CandlLensLike:
    """Loaded candl LensLike instance."""

    like: object
    n_ell: int


def load_candl_lens_likelihood() -> CandlLensLike:
    import candl
    import candl_data

    like = candl.LensLike(candl_data.ACT_DR6_Lens)
    return CandlLensLike(like=like, n_ell=int(like.N_ell_bins_theory))


def build_candl_lens_loglike(
    candl_like: CandlLensLike,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Return ``loglike(dl_kk)`` using candl's JAX-jitted ``log_like``.

    ``dl_kk`` must be the candl convention
    ``D_L^{κκ} = (π/2) L²(L+1)² C_L^{φφ}/(2π)``, indexed by L from 0 (``TheorySpectra.dl_kk``).
    candl's theory vectors start at L = 2, so L = 2 … 2 + n_ell − 1 is passed (until 2026-10-17
    this passed L = 0 … n_ell − 1: theory shifted by two multipoles, −0.11 in log L at the
    μ–Σ MAP; tests/test_act_dr6_lens.py compares it with ``act_dr6_lens_jax`` now). Feed the
    non-Limber ``cl_pp_from_weyl`` spectrum from ``jax_eb``; Limber-only φφ biases ACT DR6
    L=40–763 at the few-percent level.
    """

    like = candl_like.like
    n_ell = candl_like.n_ell

    def loglike(dl_kk: jnp.ndarray) -> jnp.ndarray:
        kk = dl_kk[2:2 + n_ell]
        if kk.shape[0] < n_ell:  # past the computed range: zero (the ACT window functions end at L 763)
            kk = jnp.concatenate([kk, jnp.zeros(n_ell - kk.shape[0], dtype=kk.dtype)])
        return jnp.asarray(like.log_like({"Dl": {"kk": kk}}))

    return loglike
