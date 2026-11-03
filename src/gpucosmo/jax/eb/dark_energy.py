"""CPL dark-energy background helper shared by the EB solver modules."""

from __future__ import annotations

import jax.numpy as jnp


def cpl_f_de(a, w0, wa):
    """ρ_DE(a)/ρ_DE(1) for w(a) = w0 + wa (1 − a): a^{−3(1+w0+wa)} e^{−3 wa (1−a)}."""
    return a ** (-3.0 * (1.0 + w0 + wa)) * jnp.exp(-3.0 * wa * (1.0 - a))


__all__ = ["cpl_f_de"]
