#!/usr/bin/env python3
# This file is part of GPUCosmo. Copyright (C) 2025-2026 Leonel Medina-Varela.
# SPDX-License-Identifier: GPL-3.0-or-later
# Derived in part from the following, translated to JAX and modified for GPUCosmo:
# - CAMB (https://github.com/cmbant/CAMB), Copyright 2000-2025 Antony Lewis and Anthony Challinor.
#   Licence: LICENSES/CAMB.txt.
# See NOTICE.md.
"""BBN helium abundance from CAMB's interpolation tables, in JAX.

Replaces the linear fit ``YHe = 0.23596569 + 0.44121095*ombh2``, which was 0.18%
off the DESI chains' ``YHe`` column and fed that error straight into
recombination.

CAMB interpolates a table with a ``RectBivariateSpline`` over
``(ombh2, DeltaN)``. With a concrete ``DeltaN`` (fixed N_eff) the problem collapses
to one dimension: the host builds the identical scipy spline once, samples it densely
along ``ombh2``, and JAX evaluates a natural cubic spline through those samples. A
traced ``DeltaN`` (sampled N_eff) evaluates scipy's tensor-product cubic B-spline
itself (same knots and coefficients, de Boor in JAX), so it equals CAMB's value to
roundoff. Both paths have no host callback and are differentiable.

Which table matters: CAMB v2 defaults to PRIMAT 2024, but the DESI DR2 chains
were run with ``PArthENoPE_880.2_standard.dat`` (``DH = 2.5589`` against PRIMAT's
``2.4203``), so that is the default here.
"""

from __future__ import annotations

import functools
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gpucosmo.paths import camb_package_file, data_dir, data_file

jax.config.update("jax_enable_x64", True)

BBN_DATA_DIR = data_dir("bbn")  # CAMB's tables; the installed camb package's copies otherwise

# The DESI DR2 base_w_wa chains: reproduces their YHe/Y_p/DHBBN columns to 8 s.f.
DESI_BBN_TABLE = "PArthENoPE_880.2_standard.dat"

# CAMB v2 constants.default_nnu, the zero point of the tables' DeltaN axis.
DEFAULT_NNU = 3.044

# CAMB constants used by the Yp^BBN <-> YHe mass-fraction conversion.
_M_H = 1.673575e-27
_M_HE4 = 6.646479073e-27

_N_DENSE = 1024


def yhe_to_ypbbn(yhe: jnp.ndarray) -> jnp.ndarray:
    return -4 * _M_H * yhe / (yhe * _M_HE4 - 4 * yhe * _M_H - _M_HE4)


def ypbbn_to_yhe(ypbbn: jnp.ndarray) -> jnp.ndarray:
    """Nucleon fraction -> helium mass fraction (CAMB ``bbn.ypBBN_to_yhe``)."""
    return -ypbbn * _M_HE4 / (-ypbbn * _M_HE4 + 4 * ypbbn * _M_H - 4 * _M_H)


def _parse_table(path: Path) -> tuple[list[str], np.ndarray]:
    """Column names from the last comment line, plus the numeric block."""
    comment = None
    with path.open() as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                comment = s[1:]
            else:
                break
    if comment is None:
        raise ValueError(f"no column header comment in {path}")
    return comment.split(), np.loadtxt(path)


@functools.lru_cache(maxsize=8)
def _dense_samples(
    table: str, column: str, delta_neff: float
) -> tuple[np.ndarray, np.ndarray]:
    """Dense ``(ombh2, value)`` samples from CAMB's own spline at fixed DeltaN."""
    from scipy.interpolate import RectBivariateSpline

    path = data_file("bbn", table, fallbacks=(camb_package_file(table),))
    columns, data = _parse_table(path)
    i_omb = columns.index("ombh2")
    i_dn = columns.index("DeltaN")
    i_col = columns.index(column)

    ombh2s = list(np.unique(data[:, i_omb]))
    deltans = list(np.unique(data[:, i_dn]))
    grid = np.zeros((len(ombh2s), len(deltans)))
    for row in data:
        grid[ombh2s.index(row[i_omb]), deltans.index(row[i_dn])] = row[i_col]
    spline = RectBivariateSpline(ombh2s, deltans, grid)

    omb_dense = np.linspace(ombh2s[0], ombh2s[-1], _N_DENSE)
    values = spline(omb_dense, np.atleast_1d(delta_neff)).ravel()
    return omb_dense, values


@functools.lru_cache(maxsize=8)
def _spline_coeffs(
    table: str, column: str, delta_neff: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(x, y, y2)`` for a natural cubic spline on the dense samples."""
    x, y = _dense_samples(table, column, delta_neff)
    n = x.size
    h = np.diff(x)
    # Standard natural-spline tridiagonal solve for the second derivatives.
    alpha = np.zeros(n)
    alpha[1:-1] = 3.0 * ((y[2:] - y[1:-1]) / h[1:] - (y[1:-1] - y[:-2]) / h[:-1])
    lo = np.zeros(n)
    mu = np.zeros(n)
    z = np.zeros(n)
    lo[0] = 1.0
    for i in range(1, n - 1):
        lo[i] = 2.0 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / lo[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / lo[i]
    y2 = np.zeros(n)
    for i in range(n - 2, -1, -1):
        y2[i] = z[i] - mu[i] * y2[i + 1]
    return x, y, 2.0 * y2  # store d2y/dx2


def _eval_cubic(x_nodes, y_nodes, y2_nodes, xq):
    """Cubic-spline value at ``xq``; clamps outside the table range."""
    xq = jnp.asarray(xq, dtype=jnp.float64)
    x = jnp.asarray(x_nodes)
    y = jnp.asarray(y_nodes)
    y2 = jnp.asarray(y2_nodes)
    n = x.size
    xc = jnp.clip(xq, x[0], x[-1])
    i = jnp.clip(jnp.searchsorted(x, xc, side="right") - 1, 0, n - 2)
    h = x[i + 1] - x[i]
    t = (xc - x[i]) / h
    a = 1.0 - t
    return (
        a * y[i]
        + t * y[i + 1]
        + ((a**3 - a) * y2[i] + (t**3 - t) * y2[i + 1]) * (h * h) / 6.0
    )


@functools.lru_cache(maxsize=8)
def _bspline2d(table: str, column: str):
    """Knots and coefficients of CAMB's ``RectBivariateSpline`` (kx = ky = 3, s = 0)."""
    from scipy.interpolate import RectBivariateSpline

    path = data_file("bbn", table, fallbacks=(camb_package_file(table),))
    columns, data = _parse_table(path)
    i_omb, i_dn, i_col = columns.index("ombh2"), columns.index("DeltaN"), columns.index(column)
    ombh2s = list(np.unique(data[:, i_omb]))
    deltans = list(np.unique(data[:, i_dn]))
    grid = np.zeros((len(ombh2s), len(deltans)))
    for row in data:
        grid[ombh2s.index(row[i_omb]), deltans.index(row[i_dn])] = row[i_col]
    spl = RectBivariateSpline(ombh2s, deltans, grid)
    tx, ty, c = spl.tck
    return np.asarray(tx), np.asarray(ty), np.asarray(c).reshape(len(tx) - 4, len(ty) - 4)


def _cubic_bspline_basis(t: np.ndarray, x):
    """(span index i, the 4 nonzero cubic B-splines at x) for knots ``t``; x is clamped to
    [t[3], t[-4]] as FITPACK's ``fpbisp`` does. Coefficient ``c[i-3+r]`` pairs with ``N[r]``."""
    t = jnp.asarray(t)
    n = t.shape[0]
    x = jnp.clip(jnp.asarray(x, dtype=jnp.float64), t[3], t[n - 4])
    i = jnp.clip(jnp.searchsorted(t, x, side="right") - 1, 3, n - 5)
    left = [None] + [x - t[i + 1 - j] for j in range(1, 4)]
    right = [None] + [t[i + j] - x for j in range(1, 4)]
    N = [jnp.ones_like(x)]
    for j in range(1, 4):
        saved = jnp.zeros_like(x)
        new = []
        for r in range(j):
            temp = N[r] / (right[r + 1] + left[j - r])
            new.append(saved + right[r + 1] * temp)
            saved = left[j - r] * temp
        N = new + [saved]
    return i, jnp.stack(N, axis=-1)


def _bbn_2d(table: str, column: str, ombh2, delta_neff):
    tx, ty, c = _bspline2d(table, column)
    ix, nx = _cubic_bspline_basis(tx, ombh2)
    iy, ny = _cubic_bspline_basis(ty, delta_neff)
    ix, iy = jnp.broadcast_arrays(ix, iy)
    nx, ny = jnp.broadcast_arrays(nx, ny)
    cj = jnp.asarray(c)
    r = jnp.arange(4)
    patch = cj[(ix[..., None, None] - 3 + r[:, None]), (iy[..., None, None] - 3 + r[None, :])]
    return jnp.einsum("...r,...rs,...s->...", nx, patch, ny)


def _is_concrete_number(x) -> bool:
    try:
        float(x)
    except (TypeError, jax.errors.ConcretizationTypeError, jax.errors.TracerArrayConversionError):
        return False
    return np.ndim(x) == 0


def yp_bbn(
    ombh2: jnp.ndarray,
    *,
    delta_neff=0.0,
    table: str = DESI_BBN_TABLE,
) -> jnp.ndarray:
    """BBN helium *nucleon* fraction ``Yp^BBN`` (CAMB ``Y_p``). ``delta_neff`` may be traced
    (sampled N_eff): then CAMB's 2-D spline is evaluated directly."""
    if _is_concrete_number(delta_neff):
        x, y, y2 = _spline_coeffs(table, "Yp^BBN", float(delta_neff))
        return _eval_cubic(x, y, y2, ombh2)
    return _bbn_2d(table, "Yp^BBN", ombh2, delta_neff)


def yhe_from_bbn(
    ombh2: jnp.ndarray,
    *,
    delta_neff=0.0,
    table: str = DESI_BBN_TABLE,
) -> jnp.ndarray:
    """Helium *mass* fraction ``YHe`` (CAMB ``Y_He``), what recombination needs."""
    return ypbbn_to_yhe(yp_bbn(ombh2, delta_neff=delta_neff, table=table))


def dh_bbn(
    ombh2: jnp.ndarray,
    *,
    delta_neff=0.0,
    table: str = DESI_BBN_TABLE,
) -> jnp.ndarray:
    """Deuterium ratio D/H (the chains' ``DHBBN`` column is this times 1e5)."""
    if _is_concrete_number(delta_neff):
        x, y, y2 = _spline_coeffs(table, "D/H", float(delta_neff))
        return _eval_cubic(x, y, y2, ombh2)
    return _bbn_2d(table, "D/H", ombh2, delta_neff)


def delta_neff_for(nnu, *, relative_to: float = DEFAULT_NNU):
    """``DeltaN`` as CAMB v2 computes it: ``N_eff - default_nnu`` with 3.044 (a float for a
    concrete ``nnu``, else a traced array).

    Exactly 0 for the DESI configuration, which is what the chains' YHe column
    reproduces (ΔN = -0.002, the older 3.046 convention, is 5000x worse).
    """
    if _is_concrete_number(nnu):
        return float(nnu) - relative_to
    return jnp.asarray(nnu, dtype=jnp.float64) - relative_to


__all__ = [
    "BBN_DATA_DIR",
    "DESI_BBN_TABLE",
    "delta_neff_for",
    "dh_bbn",
    "yhe_from_bbn",
    "yhe_to_ypbbn",
    "yp_bbn",
    "ypbbn_to_yhe",
]
