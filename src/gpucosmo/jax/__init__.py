"""JAX theory, likelihoods and inference.

Keep this package init lazy: importing ``gpucosmo.jax.gpu_env`` must not pull in JAX before
callers can set ``CUDA_VISIBLE_DEVICES`` / ``JAX_PLATFORMS``.
"""

from __future__ import annotations

from typing import Any

__all__ = ["configure_jax_gpu_env"]


def __getattr__(name: str) -> Any:
    if name == "configure_jax_gpu_env":
        from gpucosmo.jax.gpu_env import configure_jax_gpu_env

        return configure_jax_gpu_env
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
