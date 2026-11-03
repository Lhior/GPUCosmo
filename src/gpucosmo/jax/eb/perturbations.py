#!/usr/bin/env python3
"""Perturbations — thin re-export; hierarchy lives in ``hierarchy.py`` / ``evolve.py``.

Kept for back-compat with ``transfer.linear_pk_table_camb_like`` fluid path.
"""

from __future__ import annotations

from gpucosmo.jax.eb.hierarchy import derivs, initial_adiabatic, transfer_tot_from_y
from gpucosmo.jax.eb.transfer import fluid_derivs_sync, integrate_growth_factor

# Legacy experimental fluid table (EH fallback + optional reduced fluid)
from gpucosmo.jax.eb._fluid_legacy import (  # noqa: F401
    fluid_derivs_reduced,
    fluid_transfer_pk_table,
)

__all__ = [
    "derivs",
    "fluid_derivs_reduced",
    "fluid_derivs_sync",
    "fluid_transfer_pk_table",
    "initial_adiabatic",
    "integrate_growth_factor",
    "transfer_tot_from_y",
]
