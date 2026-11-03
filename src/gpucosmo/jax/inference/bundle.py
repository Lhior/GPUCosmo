"""``LikelihoodBundle``: a JAX likelihood with its priors, as the samplers and DALI expansions use it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax.numpy as jnp


@dataclass(frozen=True)
class LikelihoodBundle:
    """JAX-native likelihood + priors for GPU MCMC."""

    param_names: tuple[str, ...]
    prior_lo: jnp.ndarray
    prior_hi: jnp.ndarray
    proposal_scales: jnp.ndarray
    fiducial: jnp.ndarray
    minuslogprior: float
    log_like_fn: Callable[..., jnp.ndarray]
    log_like_batch_fn: Callable[..., jnp.ndarray]
    log_prior_fn: Callable[[jnp.ndarray], jnp.ndarray]
    log_post_batch_fn: Callable[[jnp.ndarray], jnp.ndarray]
    likelihood_label: str
    fixed_omega_b: float
    # Optional jitted ``(n_params,) -> scalar`` log posterior (no batch-1 vmap).
    log_post_scalar_fn: Callable[[jnp.ndarray], jnp.ndarray] | None = None
    # Split theory / likelihood (bundles that expose it): lets a DALI-expanded theory
    # (e.g. cmb.dali_theory) drive the exact likelihood code.
    theory_fn: Callable[[jnp.ndarray], Any] | None = None
    log_like_from_theory_fn: Callable[[jnp.ndarray, Any], jnp.ndarray] | None = None
    lmax: int | None = None

    @property
    def n_params(self) -> int:
        return len(self.param_names)


__all__ = ["LikelihoodBundle"]
