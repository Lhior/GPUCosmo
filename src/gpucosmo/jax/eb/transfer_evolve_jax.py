#!/usr/bin/env python3
# This file is part of GPUCosmo. Copyright (C) 2025-2026 Leonel Medina-Varela.
# SPDX-License-Identifier: GPL-3.0-or-later
# Derived in part from the following, translated to JAX and modified for GPUCosmo:
# - CAMB (https://github.com/cmbant/CAMB), Copyright 2000-2025 Antony Lewis and Anthony Challinor.
#   Licence: LICENSES/CAMB.txt.
# See NOTICE.md.
"""CAMB transfer-run scalar evolution (``WantCls = False``) on GPU.

Solves the same equations, approximations and switch times as CAMB 1.6/2.0
``GetTransfer`` for the matter transfer functions (``EV%TransferOnly``,
``Transfer%high_precision`` as set by ``set_matter_power``):

* hierarchy sizes per CAMB (equations.f90 ``GetNumEqns``): photons lmax = 6,
  or 10 for 0.04 < k < 0.5 Mpc^-1; polarization lmax = 5; massless neutrinos
  lmax = 45. Static per k band (``BAND_EDGES``);
* tight coupling until CAMB's switch time; massless-ν UFA closure from
  ``tau_switch_ktau = max(20, lmaxnr - 4)/k`` in any photon regime;
  DoLateRadTruncation drops ν at ``max(15/k, min(taurend, matter_verydom_tau))``
  and photons at ``max(15/k, taurend)`` for every k (the ``k > 0.03`` gate only
  applies when CAMB computes Cls);
* output at the requested times exactly: output times and switch times are
  integration knots, states are read at those knots (no interpolation).

Integration differs from CAMB's adaptive ``dverk`` in method only. After tight
coupling the Thomson terms (``opacity * M``, with ``M`` the constant-structure
drag / damping operator) are stiff: explicit schemes need ~1.5e4 steps at
k = 10 Mpc^-1 just to stay stable. They are integrated implicitly with the
L-stable IMEX Runge-Kutta ARS(3,4,3) (Ascher, Ruuth & Spiteri 1997) whose
implicit stages have a closed-form solve (rank-1 baryon-photon slip block,
2x2 quadrupole/E2 block, diagonal damping), everything else explicitly. Step
counts are static (vmap over k); placement is per k on the non-stiff rate
max(k, w_tau/tau, w_tr (lmax+1)/tau). Convergence is checked by doubling the
budgets (tests/test_jax_eb_matter_pk.py).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import numpy as np
import jax.numpy as jnp
from jax import lax

from gpucosmo.jax.eb.evolve_jax import (
    _apply_tc_init,
    _mid_nu_rsa_hierarchy,
    _pack_fluid_to_rsa,
    _pack_full_to_mid,
    _tau_start,
    _thermo_for_rhs,
    _tight_switch_time,
)
from gpucosmo.jax.eb.evolve_segmented_jax import (
    _equidistributed_knots,
    _rsa_hierarchy_nu,
    _stiffness,
    _thermo4,
)
from gpucosmo.jax.eb.vec_builder import VecBuilder
from gpucosmo.jax.eb.hierarchy_jax import (
    background_at,
    G_IX,
    IX_CLXB,
    IX_CLXC,
    IX_VB,
    HierarchyConfig,
    derivs,
    initial_adiabatic,
)

jax.config.update("jax_enable_x64", True)

# CAMB lmax rules for transfer-only modes with high_precision (lAccuracyBoost=1).
LMAXG_TRANSFER_CAMB = 6
LMAXG_TRANSFER_CAMB_BAO = 10  # 0.04 < q < 0.5
LMAXGPOL_TRANSFER_CAMB = 5
LMAXNR_TRANSFER_CAMB = 45
BAO_BAND = (0.04, 0.5)


def camb_transfer_hierarchy(k_band: str = "default") -> HierarchyConfig:
    """Static HierarchyConfig for massless-ν ΛCDM transfer modes.

    ``k_band="padded"`` is the layout big enough for every q under CAMB v2's
    rules (:func:`camb_transfer_lmax`), used with per-mode ``lmax_live``.
    """
    if k_band == "padded":
        lmaxg, lmaxgpol, lmaxnr = PADDED_LMAX
    else:
        lmaxg = LMAXG_TRANSFER_CAMB_BAO if k_band == "bao" else LMAXG_TRANSFER_CAMB
        lmaxgpol, lmaxnr = LMAXGPOL_TRANSFER_CAMB, LMAXNR_TRANSFER_CAMB
    return HierarchyConfig(
        lmaxg=lmaxg,
        lmaxgpol=lmaxgpol,
        lmaxnr=lmaxnr,
        include_pol=True,
        second_order_tc=True,
        nqmax=0,
        use_de_ppf=False,
    )


# Static sizes covering camb_transfer_lmax for every q (lmaxg <= 12, lmaxgpol
# <= 5, lmaxnr <= 45).
PADDED_LMAX = (12, 5, 45)


def _nint(x):
    """Fortran NINT for x >= 0."""
    return jnp.floor(x + 0.5)


def camb_transfer_lmax(q, *, accuracy_target: bool = True):
    """(lmaxg, lmaxgpol, lmaxnr) of a CAMB v2 transfer mode, as int32 arrays.

    CAMB_v2 equations.f90 (``SetupScalarArrays`` hierarchy sizes) with
    WantCls = Want_CMB = False, ``Transfer%high_precision`` (set by
    ``set_matter_power``), AccuratePolarization, AccurateReionization,
    ``EV%TransferOnly``, lAccuracyBoost = 1, massless neutrinos and
    ``AccuracyTarget = 1`` (config.f90 default).
    """
    q = jnp.asarray(q, dtype=jnp.float64)
    min_lmaxnr = 3.0
    lg = jnp.asarray(11.0)  # AccuratePolarization
    lnr = jnp.asarray(14.0)
    lp = lg
    low = q < 0.05
    scal = 4.0  # AccuratePolarization
    lp_low = jnp.maximum(3.0, jnp.minimum(8.0, _nint(scal * 150.0 * q)))
    lnr_low = jnp.maximum(min_lmaxnr, jnp.minimum(8.0, _nint(jnp.sqrt(scal) * 150.0 * q)))
    lg_low = jnp.maximum(3.0, jnp.minimum(8.0, _nint(jnp.sqrt(scal) * 300.0 * q)))
    lg_low = lg_low * (8.0 if accuracy_target else 4.0)  # AccurateReionization
    lp_low = lp_low * 2.0
    lg = jnp.where(low, lg_low, lg)
    lp = jnp.where(low, lp_low, lp)
    lnr = jnp.where(low, lnr_low, lnr)
    # EV%TransferOnly
    lp = jnp.minimum(lp, 5.0)
    lg = jnp.minimum(lg, 6.0)
    if accuracy_target:
        mid = (q > 0.03) & (q < 0.2)
        lg = jnp.where(mid, jnp.maximum(lg, 12.0), lg)
        lp = jnp.where(mid, jnp.maximum(lp, 5.0), lp)
    # high_precision
    lnr = jnp.where(q >= 0.005, jnp.maximum(lnr, 4.0), lnr)
    r1 = (q >= 0.010) & (q < 0.05)
    lnr = jnp.where(r1, jnp.maximum(jnp.maximum(lnr, _nint(8.0 + 6.0 * (q - 0.010) / 0.040)),
                                    min_lmaxnr), lnr)
    r2 = (q >= 0.05) & (q < 0.12)
    lnr = jnp.where(r2, jnp.maximum(jnp.maximum(lnr, _nint(14.0 + 31.0 * (q - 0.05) / 0.07)),
                                    min_lmaxnr), lnr)
    lnr = jnp.where(q >= 0.12, jnp.maximum(45.0, min_lmaxnr), lnr)
    bao = (q > 0.04) & (q < 0.5)
    lg = jnp.where(bao, jnp.maximum(lg, 10.0), lg)
    return lg.astype(jnp.int32), lp.astype(jnp.int32), lnr.astype(jnp.int32)


@dataclass(frozen=True)
class TransferStepConfig:
    """Static step counts per segment (vmap-uniform) and knot-placement weights."""

    n_tc: int = 512
    n_full: int = 2048
    n_mid: int = 1024
    n_rsa: int = 256
    n_fine: int = 1024
    w_tau: float = 4.0
    w_tr: float = 0.35
    unroll: int = 2
    # Full-hierarchy and mid segments (photons kept; mid: ν RSA):
    # "banded" = implicit metric-frozen Jacobian of the photon / polarization /
    #   massless-ν hierarchies (banded in ell, once per step), explicit rest.
    #   Stable for any k*h: required over the prior, since the segments
    #   [t_tight, min(taurend, tau_mvd)] and [.., 2 taurend] grow ~2x towards
    #   low Omega_m (taurend ~1000 Mpc at Omega_m = 0.05 vs 590 at 0.3) and the
    #   explicit free-streaming limit (k*h ~ 1.7 for ARS(3,4,3)) was exceeded at
    #   k >~ 6 (full) and k ~ 0.4 (low-k mid) for Omega_m <~ 0.15: NaN.
    # "dense" = same split with a jacfwd Jacobian (incl. metric) + batched LU:
    #   same results, ~2x slower.
    # "opacity" = closed-form implicit Thomson terms only (explicit free
    #   streaming): cheapest, valid only while k*h stays below the limit.
    full_scheme: str = "banded"
    mid_scheme: str = "banded"

    def scaled(self, factor: float) -> "TransferStepConfig":
        f = float(factor)
        return replace(
            self,
            n_tc=max(16, int(round(self.n_tc * f))),
            n_full=max(16, int(round(self.n_full * f))),
            n_mid=max(16, int(round(self.n_mid * f))),
            n_rsa=max(16, int(round(self.n_rsa * f))),
        )


# --------------------------------------------------------------------------
# Stiff Thomson operator L = opacity * M (post tight coupling)
# --------------------------------------------------------------------------


def _has_e2(cfg: HierarchyConfig) -> bool:
    return bool(cfg.include_pol and cfg.n_pol >= 1)


def stiff_apply(y, cfg: HierarchyConfig, op, photbar):
    """``L y``: the opacity terms of ``derivs`` (tight_coupling=False)."""
    out = VecBuilder(y.shape[0])
    w = (4.0 / 3.0) * y[IX_VB] - y[G_IX + 1]
    out = out.at[IX_VB].set(-photbar * op * w).at[G_IX + 1].set(op * w)
    if cfg.lmaxg >= 2:
        pig = y[G_IX + 2]
        if _has_e2(cfg):
            e2 = y[cfg.pol_base]
            out = out.at[G_IX + 2].set(-op * (0.9 * pig - 0.6 * e2))
            out = out.at[cfg.pol_base].set(-op * (0.4 * e2 - 0.1 * pig))
        else:
            out = out.at[G_IX + 2].set(-op * 0.9 * pig)
    if cfg.lmaxg >= 3:
        sl = slice(G_IX + 3, G_IX + cfg.lmaxg + 1)
        out = out.at[sl].set(-op * y[sl])
    if _has_e2(cfg) and cfg.n_pol >= 2:
        sl = slice(cfg.pol_base + 1, cfg.pol_base + cfg.n_pol)
        out = out.at[sl].set(-op * y[sl])
    return out.build()


def stiff_solve(b, cfg: HierarchyConfig, c, photbar):
    """Solve ``(I - c M) x = b`` with ``c = h * gamma * opacity`` (closed form)."""
    x = VecBuilder(b.shape[0], base=b)
    pb43 = (4.0 / 3.0) * photbar
    w_x = ((4.0 / 3.0) * b[IX_VB] - b[G_IX + 1]) / (1.0 + c * (1.0 + pb43))
    x = x.at[IX_VB].set(b[IX_VB] - c * photbar * w_x)
    x = x.at[G_IX + 1].set(b[G_IX + 1] + c * w_x)
    if cfg.lmaxg >= 2:
        b1 = b[G_IX + 2]
        if _has_e2(cfg):
            b2 = b[cfg.pol_base]
            det = 1.0 + 1.3 * c + 0.3 * c * c
            x = x.at[G_IX + 2].set(((1.0 + 0.4 * c) * b1 + 0.6 * c * b2) / det)
            x = x.at[cfg.pol_base].set((0.1 * c * b1 + (1.0 + 0.9 * c) * b2) / det)
        else:
            x = x.at[G_IX + 2].set(b1 / (1.0 + 0.9 * c))
    if cfg.lmaxg >= 3:
        sl = slice(G_IX + 3, G_IX + cfg.lmaxg + 1)
        x = x.at[sl].set(b[sl] / (1.0 + c))
    if _has_e2(cfg) and cfg.n_pol >= 2:
        sl = slice(cfg.pol_base + 1, cfg.pol_base + cfg.n_pol)
        x = x.at[sl].set(b[sl] / (1.0 + c))
    return x.build()


# ARS(3,4,3): explicit first stage, L-stable SDIRK implicit part, b_E = b_I.
_G = 0.435866521508459
_B1 = -1.5 * _G * _G + 4.0 * _G - 0.25
_B2 = 1.5 * _G * _G - 5.0 * _G + 1.25
_ARS_C = (0.0, _G, 0.5 * (1.0 + _G), 1.0)
_ARS_AE = (
    (),
    (_G,),
    (0.3212788860286278, 0.3966543747256017),
    (-0.1058582960718797, 0.5529291480359398, 0.5529291480359398),
)
_ARS_AI = ((), (0.0,), (0.0, 0.5 * (1.0 - _G)), (0.0, _B1, _B2))
_ARS_B = (0.0, _B1, _B2, _G)


# --------------------------------------------------------------------------
# Fixed-knot scans that land on output / switch times exactly
# --------------------------------------------------------------------------


def _merged_knots(ta, tb, base_n, lam_fn, n_fine, tau_out, extra):
    """Per-k knots on [ta, tb]: equidistributed base + outputs + switch times.

    Returns ``(knots, tags)``; ``tags[i] = j`` when knot i is output j lying in
    (ta, tb], else -1. Out-of-segment outputs are clipped onto the endpoints and
    produce zero-length (identity) steps, keeping the count static.
    """
    base = _equidistributed_knots(ta, tb, base_n, lam_fn, n_fine)
    n_out = tau_out.shape[0]
    inside = (tau_out > ta) & (tau_out <= tb)
    t_out = jnp.clip(tau_out, ta, tb)
    t_ex = jnp.clip(jnp.asarray(extra, dtype=jnp.float64).reshape(-1), ta, tb)
    t_all = jnp.concatenate([base, t_out, t_ex])
    tags = jnp.concatenate([
        -jnp.ones(base.shape[0], dtype=jnp.int32),
        jnp.where(inside, jnp.arange(n_out, dtype=jnp.int32), -1),
        -jnp.ones(t_ex.shape[0], dtype=jnp.int32),
    ])
    # Outputs after equal-time base knots so the step *ending* there is tagged.
    order = jnp.lexsort((tags, t_all))
    return t_all[order], tags[order]


def _scan_collect(step, y0, knots, tags, n_out, unroll):
    """Run ``step(y, t0, t1, i)`` over the knots; collect (clxc, clxb) at tags."""
    t_lo, t_hi, tag_end = knots[:-1], knots[1:], tags[1:]
    idx = jnp.arange(t_lo.shape[0])
    buf0 = jnp.zeros((n_out, 2), dtype=jnp.float64)

    def body(carry, xs):
        y, buf = carry
        t0, t1, tag, i = xs
        y1 = step(y, t0, t1, i)
        slot = jnp.maximum(tag, 0)
        new = jnp.stack([y1[IX_CLXC], y1[IX_CLXB]])
        buf = buf.at[slot].set(jnp.where(tag >= 0, new, buf[slot]))
        return (y1, buf), None

    (y_end, buf), _ = lax.scan(body, (y0, buf0), (t_lo, t_hi, tag_end, idx),
                               unroll=int(unroll))
    return y_end, buf


def _stage_table(thermo, params, cfg: HierarchyConfig):
    """Per-time RHS inputs: (a, cs2, opacity, dopacity) + ``background_at``.

    Tabulated for every stage time before the scan, so the step loop only does
    the y-dependent work.
    """
    def tab(t):
        th4 = _thermo4(t, thermo)
        return jnp.concatenate([th4, background_at(t, th4[0], params, cfg)])
    return jax.vmap(tab)


def _rk4_step_fn(rhs, tab, knots):
    t_mid = 0.5 * (knots[:-1] + knots[1:])
    th_k = tab(knots)
    th_m = tab(t_mid)

    def step(y, t0, t1, i):
        h = t1 - t0
        tm = 0.5 * (t0 + t1)
        k1 = rhs(t0, y, th_k[i], t0)
        k2 = rhs(tm, y + 0.5 * h * k1, th_m[i], t0)
        k3 = rhs(tm, y + 0.5 * h * k2, th_m[i], t0)
        k4 = rhs(t1, y + h * k3, th_k[i + 1], t0)
        return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return step


def _imex_step_fn(rhs, tab, knots, cfg: HierarchyConfig, params):
    """ARS(3,4,3) step: explicit ``rhs - L y``, implicit ``L y``."""
    t_lo, t_hi = knots[:-1], knots[1:]
    th_st = [tab(t_lo + c * (t_hi - t_lo)) for c in _ARS_C]
    om_ratio = params["Omega_g"] / params["Omega_b"]

    def step(y, t0, t1, i):
        h = t1 - t0
        fe, fi = [], []
        for s in range(4):
            th = th_st[s][i]
            t_s = t0 + _ARS_C[s] * h
            a_s, op_s = th[0], th[2]
            photbar = om_ratio / a_s
            acc = y
            for j in range(s):
                acc = acc + h * _ARS_AE[s][j] * fe[j] + h * _ARS_AI[s][j] * fi[j]
            ys = acc if s == 0 else stiff_solve(acc, cfg, h * _G * op_s, photbar)
            li = stiff_apply(ys, cfg, op_s, photbar)
            fi.append(li)
            fe.append(rhs(t_s, ys, th, t0) - li)
        out = y
        for s in range(4):
            out = out + h * _ARS_B[s] * (fe[s] + fi[s])
        return out

    return step


def _photon_block_index(cfg: HierarchyConfig, *, nu_l: int = -1) -> jnp.ndarray:
    """State indices of v_b, the photon / polarization hierarchies and, if
    ``nu_l >= 0``, massless-ν multipoles ℓ <= nu_l."""
    idx = [IX_VB] + list(range(G_IX, G_IX + cfg.lmaxg + 1))
    if cfg.include_pol:
        idx += list(range(cfg.pol_base, cfg.pol_base + cfg.n_pol))
    if nu_l >= 0:
        idx += list(range(cfg.r_ix, cfg.r_ix + min(nu_l, cfg.lmaxnr) + 1))
    return jnp.asarray(idx, dtype=jnp.int32)


def _imex_dense_step_fn(rhs, tab, knots, idx):
    """ARS(3,4,3) with the ``idx`` block of the (linear) RHS implicit.

    ``derivs`` is linear in y, so the block Jacobian ``J`` (``jax.jacfwd``, one
    batched evaluation) splits it exactly: implicit ``J y_blk``, explicit the
    rest. Used after tight coupling ends, where photons free-stream at rate k
    (Silk-damped at high k) and the explicit stability limit ~2.5/k would
    otherwise set the step count.
    """
    t_lo, t_hi = knots[:-1], knots[1:]
    th_st = [tab(t_lo + c * (t_hi - t_lo)) for c in _ARS_C]
    n_blk = int(idx.shape[0])
    eye = jnp.eye(n_blk, dtype=jnp.float64)

    def step(y, t0, t1, i):
        h = t1 - t0
        fe, fi = [], []
        for s in range(4):
            th = th_st[s][i]
            t_s = t0 + _ARS_C[s] * h

            def f(yy, th=th, t_s=t_s):
                return rhs(t_s, yy, th, t0)

            def f_blk(yb, th=th, t_s=t_s):
                return f(jnp.zeros_like(y).at[idx].set(yb))[idx]

            jac = jax.jacfwd(f_blk)(y[idx])  # (n_blk, n_blk)
            acc = y
            for j in range(s):
                acc = acc + h * _ARS_AE[s][j] * fe[j] + h * _ARS_AI[s][j] * fi[j]
            if s == 0:
                ys = acc
            else:
                xb = jnp.linalg.solve(eye - (h * _G) * jac, acc[idx])
                ys = acc.at[idx].set(xb)
            li = jnp.zeros_like(y).at[idx].set(jac @ ys[idx])
            fi.append(li)
            fe.append(f(ys) - li)
        out = y
        for s in range(4):
            out = out + h * _ARS_B[s] * (fe[s] + fi[s])
        return out

    return step


# Banded implicit block: photon / polarization / massless-nu hierarchies.
_BAND = 2


def _banded_order(cfg: HierarchyConfig, *, nu: bool) -> list[int]:
    """State indices in an order that makes the metric-frozen Jacobian banded.

    [v_b, g0, g1, g2, E2, g3, E3, g4, E4, ..., g_lmax, nu_0 .. nu_lmax]: every
    hierarchy coupling (free streaming, closures, Thomson v_b-g1 and g2-E2) is
    then within ``_BAND`` of the diagonal; photons and nu couple only through
    the metric, which ``derivs(freeze_metric=True)`` removes from the Jacobian.
    """
    order = [IX_VB, G_IX, G_IX + 1]
    if cfg.lmaxg >= 2:
        order.append(G_IX + 2)
    pol = [cfg.pol_base + j for j in range(cfg.n_pol)] if cfg.include_pol else []
    phot_hi = [G_IX + ell for ell in range(3, cfg.lmaxg + 1)]
    if pol:
        order.append(pol[0])
    for j in range(max(len(phot_hi), len(pol) - 1)):
        if j < len(phot_hi):
            order.append(phot_hi[j])
        if j + 1 < len(pol):
            order.append(pol[j + 1])
    if nu:
        order += [cfg.r_ix + ell for ell in range(cfg.lmaxnr + 1)]
    return order


def _band_diagonals(f_frozen, y, order, b=_BAND):
    """Exact band of the (linear) ``f_frozen`` Jacobian on ``order``.

    Columns ``2b+1`` apart never touch the same rows, so one JVP per colour
    class gives every band entry (``2b+1`` JVPs instead of one per column).
    Returns ``{d: (n,) array}`` with ``diag[d][q] = J[order[q], order[q+d]]``.
    """
    n, nc = len(order), 2 * b + 1
    idx = np.asarray(order)
    seeds = np.zeros((nc, y.shape[0]))
    for c in range(nc):
        seeds[c, idx[c::nc]] = 1.0
    jv = jax.vmap(lambda v: jax.jvp(f_frozen, (y,), (v,))[1])(jnp.asarray(seeds, y.dtype))
    g = jv[:, idx]  # (nc, n): rows in ``order``
    q = np.arange(n)
    diags = {}
    for d in range(-b, b + 1):
        p = q + d
        ok = (p >= 0) & (p < n)
        diags[d] = jnp.where(jnp.asarray(ok), g[np.mod(p, nc), q], 0.0)
    return diags


def _band_matvec(diags, x, b=_BAND):
    n = x.shape[0]
    out = diags[0] * x
    for d in range(1, b + 1):
        out = out.at[: n - d].add(diags[d][: n - d] * x[d:])
        out = out.at[d:].add(diags[-d][d:] * x[: n - d])
    return out


def _band_lu(diags, c, b=_BAND):
    """Band LU (no pivoting) of ``I - c J``; returns (multipliers, U rows).

    Unrolled over the static size, so XLA fuses it into a few kernels per mode
    batch. No pivoting: I - cJ has a positive-definite symmetric part here
    (free-streaming couplings are sign-alternating, damping is negative).
    """
    n = diags[0].shape[0]
    a = [[None] * (2 * b + 1) for _ in range(n)]
    for qq in range(n):
        for d in range(-b, b + 1):
            if 0 <= qq + d < n:
                a[qq][d + b] = (1.0 if d == 0 else 0.0) - c * diags[d][qq]
    mult = {}
    for i in range(n):
        piv = a[i][b]
        for rr in range(i + 1, min(n, i + b + 1)):
            f = a[rr][i - rr + b] / piv
            mult[(rr, i)] = f
            for cc in range(i + 1, min(n, i + b + 1)):
                a[rr][cc - rr + b] = a[rr][cc - rr + b] - f * a[i][cc - i + b]
    return mult, a


def _band_lu_solve(lu, rhs, b=_BAND):
    mult, a = lu
    n = rhs.shape[0]
    r = [rhs[qq] for qq in range(n)]
    for i in range(n):
        for rr in range(i + 1, min(n, i + b + 1)):
            r[rr] = r[rr] - mult[(rr, i)] * r[i]
    x = [None] * n
    for i in reversed(range(n)):
        acc = r[i]
        for cc in range(i + 1, min(n, i + b + 1)):
            acc = acc - a[i][cc - i + b] * x[cc]
        x[i] = acc / a[i][b]
    return jnp.stack(x)


def _band_solve(diags, c, rhs, b=_BAND):
    """Solve ``(I - c J) x = rhs`` for banded J."""
    return _band_lu_solve(_band_lu(diags, c, b), rhs, b)


def _imex_banded_step_fn(rhs, rhs_frozen, tab, knots, order):
    """ARS(3,4,3) with the banded hierarchy block implicit (exact split).

    Implicit: ``J y`` on the hierarchy block, J the Jacobian with the metric
    feedback frozen, taken once per step (at the second stage time) and
    LU-factorised once for the three implicit stages; explicit: ``rhs - J y``.
    Any fixed linear split is exact; the k-rate free-streaming couplings are
    time-independent, so what J leaves to the explicit part (the change of
    opacity / 1/tau within a step, metric feedback) is far below the explicit
    stability limit. Stable for any k*h, like the dense scheme, for 2*_BAND+1
    batched JVPs and one O(n) band LU per step.
    """
    t_lo, t_hi = knots[:-1], knots[1:]
    th_st = [tab(t_lo + c * (t_hi - t_lo)) for c in _ARS_C]
    idx = jnp.asarray(order, dtype=jnp.int32)

    def step(y, t0, t1, i):
        h = t1 - t0
        th_j = th_st[1][i]
        t_j = t0 + _ARS_C[1] * h
        diags = _band_diagonals(lambda yy: rhs_frozen(t_j, yy, th_j, t0), y, order)
        lu = _band_lu(diags, h * _G)
        fe, fi = [], []
        for s in range(4):
            th = th_st[s][i]
            t_s = t0 + _ARS_C[s] * h
            acc = y
            for j in range(s):
                acc = acc + h * _ARS_AE[s][j] * fe[j] + h * _ARS_AI[s][j] * fi[j]
            if s == 0:
                ys = acc
            else:
                ys = acc.at[idx].set(_band_lu_solve(lu, acc[idx]))
            li = jnp.zeros_like(y).at[idx].set(_band_matvec(diags, ys[idx]))
            fi.append(li)
            fe.append(rhs(t_s, ys, th, t0) - li)
        out = y
        for s in range(4):
            out = out + h * _ARS_B[s] * (fe[s] + fi[s])
        return out

    return step


# --------------------------------------------------------------------------
# One mode
# --------------------------------------------------------------------------


def evolve_transfer_mode(
    k,
    params: dict,
    thermo: dict,
    tau_out,
    *,
    cfg: HierarchyConfig,
    steps: TransferStepConfig = TransferStepConfig(),
    lmax_live: tuple | None = None,
    late_phot_switch_boost: float = 1.0,
):
    """(clxc, clxb) of one k mode at ``tau_out`` (ascending), CAMB transfer run.

    ``thermo`` is the full thermo table (not downsampled); outputs must lie in
    (t0, max(tau_out)]. ``lmax_live`` (per-mode hierarchy sizes on the padded
    ``cfg``, see :func:`camb_transfer_lmax`) and ``late_phot_switch_boost``
    (CAMB v2 ``latePhotSwitchBoost = 2`` for high-precision transfer runs)
    select CAMB v2 behaviour.
    """
    tau_out = jnp.asarray(tau_out, dtype=jnp.float64)
    n_out = int(tau_out.shape[0])
    th = _thermo_for_rhs(thermo) if "tau_mpc_hot" in thermo else dict(thermo)
    tau0 = jnp.asarray(th["tau_mpc"][-1], dtype=jnp.float64)
    k_safe = jnp.maximum(k, 1e-10)
    taurend = jnp.asarray(th["taurend"], dtype=jnp.float64)
    mvd = jnp.asarray(th.get("matter_verydom_tau", taurend), dtype=jnp.float64)
    t_end = jnp.max(tau_out)

    t0 = _tau_start(k, tau0)
    t_tight = jnp.clip(_tight_switch_time(k, th, tau0), t0 * 1.01, t_end)
    t_drop_nu = jnp.clip(jnp.maximum(15.0 / k_safe, jnp.minimum(taurend, mvd)), t_tight, t_end)
    t_drop_phot = jnp.clip(
        jnp.maximum(15.0 / k_safe, taurend) * late_phot_switch_boost, t_drop_nu, t_end
    )
    lnr_live = cfg.lmaxnr if lmax_live is None else lmax_live[2]
    t_ktau = jnp.clip(jnp.maximum(20.0, jnp.asarray(lnr_live, jnp.float64) - 4.0) / k_safe,
                      t0, t_drop_nu)

    mid_cfg = _mid_nu_rsa_hierarchy(cfg)
    rsa_cfg = _rsa_hierarchy_nu(cfg)

    def rhs_for(c: HierarchyConfig, *, hk: bool = False, freeze: bool = False, **flags):
        def rhs(t, y, th4, t_step):
            extra = {"high_ktau_neutrino_approx": t_step >= t_ktau} if hk else {}
            yp, _, _ = derivs(t, y, k, params, th, cfg=c,
                              thermo_at_tau=tuple(th4[i] for i in range(4)),
                              bg_at_tau=th4[4:], freeze_metric=freeze,
                              lmax_live=lmax_live, **flags, **extra)
            return yp
        return rhs

    def tab_for(c: HierarchyConfig):
        return _stage_table(th, params, c)

    def lam(use_lmax):
        return lambda t: _stiffness(t, k, params, th, use_op=False, lmax=use_lmax,
                                    w=_StiffW(steps))

    # 0: tight coupling (explicit; opacity stiffness removed by the TCA).
    kn0, tg0 = _merged_knots(t0, t_tight, steps.n_tc, lam(2), steps.n_fine, tau_out, t_ktau)
    y0 = initial_adiabatic(k, t0, params, cfg)
    step0 = _rk4_step_fn(rhs_for(cfg, hk=True, tight_coupling=True), tab_for(cfg), kn0)
    y_tc, out0 = _scan_collect(step0, y0, kn0, tg0, n_out, steps.unroll)
    y1_init = _apply_tc_init(y_tc, t_tight, k, params, th, cfg)

    # 1: full hierarchy, IMEX (stiff Thomson drag implicit).
    lmax_full = max(cfg.lmaxg, cfg.lmaxnr, cfg.lmaxgpol)
    kn1, tg1 = _merged_knots(t_tight, t_drop_nu, steps.n_full, lam(lmax_full), steps.n_fine,
                             tau_out, t_ktau)
    if steps.full_scheme == "banded":
        step1 = _imex_banded_step_fn(
            rhs_for(cfg, hk=True, tight_coupling=False),
            rhs_for(cfg, hk=True, freeze=True, tight_coupling=False),
            tab_for(cfg), kn1, _banded_order(cfg, nu=True))
    elif steps.full_scheme == "dense":
        step1 = _imex_dense_step_fn(rhs_for(cfg, hk=True, tight_coupling=False), tab_for(cfg), kn1,
                                    _photon_block_index(cfg, nu_l=2))
    else:
        step1 = _imex_step_fn(rhs_for(cfg, hk=True, tight_coupling=False), tab_for(cfg), kn1,
                              cfg, params)
    y_full, out1 = _scan_collect(step1, y1_init, kn1, tg1, n_out, steps.unroll)

    # 2: photons kept, massless ν RSA (explicit; opacity small after taurend..).
    kn2, tg2 = _merged_knots(t_drop_nu, t_drop_phot, steps.n_mid,
                             lam(max(mid_cfg.lmaxg, mid_cfg.lmaxgpol)), steps.n_fine,
                             tau_out, t_drop_phot)
    rhs2 = rhs_for(mid_cfg, tight_coupling=False, no_nu_multpoles=True)
    if steps.mid_scheme == "banded":
        step2 = _imex_banded_step_fn(
            rhs2, rhs_for(mid_cfg, freeze=True, tight_coupling=False, no_nu_multpoles=True),
            tab_for(mid_cfg), kn2, _banded_order(mid_cfg, nu=False))
    elif steps.mid_scheme == "dense":
        step2 = _imex_dense_step_fn(rhs2, tab_for(mid_cfg), kn2, _photon_block_index(mid_cfg))
    else:
        step2 = _imex_step_fn(rhs2, tab_for(mid_cfg), kn2, mid_cfg, params)
    y_mid, out2 = _scan_collect(step2, _pack_full_to_mid(y_full, cfg, mid_cfg), kn2, tg2,
                                n_out, steps.unroll)

    # 3: photon + ν RSA to the last output.
    lam3 = lambda t: _stiffness(t, k, params, th, use_op=False, lmax=2,  # noqa: E731
                                w=_StiffW(steps), oscillatory=False)
    kn3, tg3 = _merged_knots(t_drop_phot, t_end, steps.n_rsa, lam3, steps.n_fine,
                             tau_out, t_end)
    step3 = _rk4_step_fn(
        rhs_for(rsa_cfg, tight_coupling=False, no_phot_multpoles=True, no_nu_multpoles=True),
        tab_for(rsa_cfg), kn3,
    )
    _, out3 = _scan_collect(step3, _pack_fluid_to_rsa(y_mid, mid_cfg, rsa_cfg), kn3, tg3,
                            n_out, steps.unroll)

    seg = (
        (tau_out > t_tight).astype(jnp.int32)
        + (tau_out > t_drop_nu)
        + (tau_out > t_drop_phot)
    )[:, None]
    out = jnp.where(seg == 0, out0, out1)
    out = jnp.where(seg == 2, out2, out)
    return jnp.where(seg == 3, out3, out)


@dataclass(frozen=True)
class _StiffW:
    """Knot-placement weights in the ``evolve_segmented_jax._stiffness`` API."""

    steps: TransferStepConfig

    @property
    def w_tau(self):
        return self.steps.w_tau

    @property
    def w_tr(self):
        return self.steps.w_tr

    w_op: float = 0.0
    w_nu: float = 0.0


__all__ = [
    "BAO_BAND",
    "PADDED_LMAX",
    "camb_transfer_lmax",
    "TransferStepConfig",
    "camb_transfer_hierarchy",
    "evolve_transfer_mode",
    "stiff_apply",
    "stiff_solve",
]
