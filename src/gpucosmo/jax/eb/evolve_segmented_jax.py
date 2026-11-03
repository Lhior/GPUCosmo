#!/usr/bin/env python3
# This file is part of GPUCosmo. Copyright (C) 2025-2026 Leonel Medina-Varela.
# SPDX-License-Identifier: GPL-3.0-or-later
# Derived in part from the following, translated to JAX and modified for GPUCosmo:
# - CAMB (https://github.com/cmbant/CAMB), Copyright 2000-2025 Antony Lewis and Anthony Challinor.
#   Licence: LICENSES/CAMB.txt.
# See NOTICE.md.
"""GPU-oriented scalar evolution: regime-static segments + stiffness-adapted grids.

Why a separate integrator: ``evolve_jax._evolve_on_timesteps`` picks the CAMB
regime (TC / full / high-kτ / RSA) per step with ``lax.switch`` on a per-k
index. Under ``vmap`` over k that lowers to a select, so every RK4 stage runs
*all* ``derivs`` variants; the ``lax.cond`` substep skips likewise run both
sides. Steps are also spaced uniformly between regime switches, although the
stiffness (photon–baryon drag, rate ``opacity·(1+R⁻¹)``) is concentrated just
after tight coupling ends.

Here each k is integrated through four segments whose regime is a Python
constant, so each ``lax.scan`` body contains exactly one ``derivs`` variant:

  0. tight coupling          [t0,        t_tight]
  1. full hierarchy          [t_tight,   t_drop_nu]
     (segments 0 and 1: CAMB's high-kτ ν closure from ``t_ktau`` on, as a
     traced per-step flag touching only the ν ℓ≥2 rows)
  2. mid (photons, ν RSA)    [t_drop_nu, t_drop_phot]
  3. photon + ν RSA          [t_drop_phot, t_end]  (massive-ν momentum
     hierarchy kept, as CAMB does to τ0 for k ≳ 0.02 with lensing on)

Segment step *counts* are static (vmap-uniform); step *placement* is per k,
equidistributing a local stiffness rate Λ(τ) so that each step has the same
Λ·Δτ. Zero-length segments degenerate to Δτ = 0 identity steps.

States are stored at every knot together with the RHS (the RK4 ``k1`` stage,
free), and cubic-Hermite interpolated onto ``tau_out`` in each segment's
native layout before scattering to the full Cls layout.
"""

from __future__ import annotations

import os

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from gpucosmo.jax.eb.evolve_jax import (
    MAX_ETAK_SCALAR,
    _apply_tc_init,
    _mid_nu_rsa_hierarchy,
    _pack_fluid_to_rsa,
    _pack_full_to_mid,
    _scatter_mid_to_full,
    _scatter_rsa_to_full,
)
from gpucosmo.jax.eb.hierarchy_jax import (
    G_IX,
    HierarchyConfig,
    derivs,
    initial_adiabatic,
    thermo_values,
)
from gpucosmo.jax.eb.massive_nu import nu_momentum_grid

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class SegmentedConfig:
    """Static step counts per segment and stiffness weights for placement.

    Λ(τ) = max(k, w_op·opacity·(1+R⁻¹), w_tr·(ℓmax+1)/τ, w_tau/τ); only the
    ratios matter because each segment rescales Λ to its own step count.
    In the RSA segment the state is non-oscillatory, so k is replaced by the
    baryon sound-crossing rate k·c_s.

    Budgets are sized from max_k ∫Λ dτ over the Evolve_q grid (k ≤ 0.45) at a
    target Λ·Δτ ≈ 0.5: tc ≈ 73, full ≈ 486 (drag-dominated at high k),
    mid ≈ 412 (k ≈ 0.03 keeps photons to τ0), rsa ≈ 350 (massive-ν streaming
    k·v_max at w_nu = 0.25). With w_op = 0.25, Λ·Δτ = 0.5 keeps
    opacity·(1+R⁻¹)·Δτ ≤ 2, inside RK4's real-axis limit 2.78; w_nu = 0.25 does
    the same for the free-streaming massive-ν hierarchy (stability, not phase
    accuracy, is what matters once those modes have free-streamed away).
    """

    n_tc: int = 160
    n_full: int = 1024
    n_mid: int = 832
    n_rsa: int = 768
    n_fine: int = 512
    # Hold the RSA state fixed once k·τ > max_etak_scalar past taurend, as the
    # legacy path does. Off by default: with lensing CAMB evolves every k to τ0
    # (WantLateTime), and a frozen δ under an evolving background biases the
    # late Weyl potential that high-L φφ integrates over.
    freeze_late: bool = False
    # CAMB high-kτ massless-ν closure from t_ktau = 20/k on (per-step flag).
    high_ktau: bool = True
    w_op: float = 0.25
    w_tr: float = 0.35
    w_tau: float = 4.0
    w_nu: float = 0.25
    # ``lax.scan`` unroll factor. The per-step cost on GPU is launch latency,
    # not arithmetic (1 or 384 k modes take the same time), so unrolling cuts
    # wall time without changing the result.
    unroll: int = 1
    # RK4 stages as a 4-trip loop (one inlined ``derivs`` per step body instead of
    # four): same arithmetic, ~4× smaller programs for the segment bodies.
    stage_loop: bool = False
    # Knot storage: "all" keeps every state entry (and its RHS) at every knot;
    # "sources" keeps only the entries the CMB sources and δ_m read
    # (``source_knot_indices``), ~4× less memory per k mode. The integration and the
    # kept entries are unchanged; the other entries of the returned states are zero.
    # "outputs" stores no knots: the output times become knots and only the states
    # there are kept (memory n_out × n_eq per mode; for few outputs, e.g. P(k) at
    # survey redshifts).
    store: str = "all"
    # Kahan-compensated RK4 state update (``_rk4_segment``): the float32 rounding of y + Δy
    # no longer accumulates over the steps. Batched CMB (float32 sources) vs float64: max
    # TT/EE error 5.6e-6/1.5e-5 → 2.0e-6/2.1e-6; P(k) at k = 15/Mpc 4.7e-3 → 7e-4 (CAMB);
    # +0.3 % time, no memory.
    compensated: bool = True
    # Photons go to the radiation-streaming approximation after max(15/k, τ_rend) only for
    # k above this [1/Mpc] (None: ``evolve_jax.PHOT_RSA_KMIN``, CAMB's WantCls 0.03).
    phot_rsa_kmin: float | None = None


# Late (RSA) segment budget × this when an eigenstate may be light (``HierarchyConfig.nu_light``):
# the streaming rate k·v of near-massless bins stays ≈ k to τ0, so ∫ w_nu k v dτ / n_rsa ≈ 0.68 at
# k = 0.16/Mpc with the default budget — the RK4 limit (k v Δτ ≈ 2.8): S_T diverged there for
# Σm_ν = 1e-4 eV. ×4 holds Λ Δτ ≤ 0.5 to k = 0.45 (+28 % CMB evolve time).
NU_LIGHT_RSA_MULT = 4

# GPUCOSMO_RK4_STAGE_LOOP=1: RK4 stage loop (see SegmentedConfig.stage_loop).
DEFAULT_SEGMENTED = SegmentedConfig(stage_loop=os.environ.get("GPUCOSMO_RK4_STAGE_LOOP", "0") == "1")


def _thermo4(t, thermo):
    return jnp.stack(thermo_values(t, thermo, want_dopacity=True))


def _stiffness(
    tau, k, params, thermo, *, use_op: bool, lmax: int, w: SegmentedConfig,
    oscillatory: bool = True, nu_qmax: float = 0.0,
):
    a, cs2, op, _dop = thermo_values(tau, thermo, want_dopacity=True)
    tau = jnp.maximum(tau, 1e-30)
    k_rate = k if oscillatory else k * jnp.sqrt(jnp.maximum(cs2, 0.0))
    lam = jnp.maximum(k_rate, w.w_tau / tau)
    lam = jnp.maximum(lam, w.w_tr * (lmax + 1.0) / tau)
    if use_op:
        # u = 4/3 v_b − q_γ decays at opacity·(1 + 4/3 ρ_γ/ρ_b) = opacity·(1+R⁻¹).
        pb43 = (4.0 / 3.0) * params["Omega_g"] / (params["Omega_b"] * a)
        lam = jnp.maximum(lam, w.w_op * op * (1.0 + pb43))
    if nu_qmax > 0.0:
        # Fastest massive-ν momentum bin streams at k·v, v = (1 + (a m/q)²)^-½.
        # The lightest eigenstate block streams fastest.
        aq = a * jnp.min(jnp.atleast_1d(jnp.asarray(params["nu_mass_am"]))) / nu_qmax
        lam = jnp.maximum(lam, w.w_nu * k / jnp.sqrt(1.0 + aq * aq))
    return lam


def _rsa_hierarchy_nu(cfg: HierarchyConfig) -> HierarchyConfig:
    """Photon+ν RSA hierarchy that keeps the massive-ν momentum grid.

    CAMB never switches high-k massive neutrinos to its fluid approximation
    (``a_massive = 20000 k / m_ν ≥ 0.99``) and evolves them to τ0 when lensing
    is on. Dropping the grid here also drops ρ_ν from the background, which
    over-grows late-time structure: +3.5% in C_L^φφ at L ≳ 200 for Σm_ν=0.06 eV.
    """
    return HierarchyConfig(
        lmaxg=2,
        lmaxgpol=0,
        lmaxnr=2,
        include_pol=False,
        second_order_tc=False,
        nqmax=cfg.nqmax,
        lmaxnu=cfg.lmaxnu,
        n_nu_eig=cfg.n_nu_eig,
        nu_light=cfg.nu_light,
        curved=cfg.curved,
        use_de_ppf=cfg.use_de_ppf,
        mg=cfg.mg,
    )


def source_knot_indices(cfg: HierarchyConfig) -> dict[str, np.ndarray]:
    """Native-layout entries of each segment that the CMB sources and δ_m depend on.

    The sources (``temperature_polarization_sources``) read the metric, matter and
    the ℓ ≤ 2 moments directly, and the RHS of v_b, q_γ, Θ_3, Π_ν, E_2, E_3 and the
    massive-ν ℓ = 2 moments, which couple to one multipole above: photons ℓ ≤ 4,
    E ℓ ≤ 4, massless ν ℓ ≤ 3, massive ν ℓ ≤ 3 in every momentum bin. In tight
    coupling the photon ℓ = 2, 3 and E ℓ = 2, 3 moments are refilled algebraically
    (``_apply_tc_init``); in the mid segment the massless ν and in the RSA segment the
    photons and massless ν are algebraic (``_scatter_*_to_full``). Checked against the
    structural gradient of the sources in ``tests/test_evolve_segmented.py``.
    """
    mid_cfg = _mid_nu_rsa_hierarchy(cfg)
    rsa_cfg = _rsa_hierarchy_nu(cfg)

    def common(c: HierarchyConfig) -> list[int]:
        ix = [0, 1, 2, 3]
        if c.use_de_ppf:
            ix.append(c.w_ix)
        for b in range(c.n_nu_eig * c.nqmax if c.nqmax > 0 else 0):  # every eigenstate block's bins
            base = c.nu_ix + b * (c.lmaxnu + 1)
            ix += [base + ell for ell in range(min(4, c.lmaxnu + 1))]
        return ix

    def photons(c: HierarchyConfig, ells) -> list[int]:
        ix = [G_IX + ell for ell in ells if ell <= c.lmaxg]
        if c.include_pol:
            ix += [c.pol_base + (ell - 2) for ell in ells if 2 <= ell <= c.lmaxgpol]
        return ix

    def nu_massless(c: HierarchyConfig) -> list[int]:
        return [c.r_ix + ell for ell in range(min(4, c.lmaxnr + 1))]

    full = common(cfg) + photons(cfg, range(5)) + nu_massless(cfg)
    tc = [i for i in full if i not in set(photons(cfg, (2, 3)))]
    mid = common(mid_cfg) + photons(mid_cfg, range(5))
    rsa = common(rsa_cfg)
    return {name: np.asarray(sorted(set(ix)), dtype=np.int32)
            for name, ix in (("tc", tc), ("full", full), ("mid", mid), ("rsa", rsa))}


def _expand(yk: jnp.ndarray, keep, n: int) -> jnp.ndarray:
    """(…, len(keep)) → (…, n) with zeros outside ``keep`` (identity if keep is None).

    A gather from a zero-padded copy (static indices), not a scatter: XLA fuses it
    into the consumers, where a scatter materializes every segment's dense
    (n_k, n_τ, n_eq) output at once (~0.8 GB per 240 k modes).
    """
    if keep is None:
        return yk
    inv = np.full(n, len(keep), dtype=np.int32)
    inv[np.asarray(keep)] = np.arange(len(keep), dtype=np.int32)
    pad = jnp.concatenate([yk, jnp.zeros(yk.shape[:-1] + (1,), dtype=yk.dtype)], axis=-1)
    return jnp.take(pad, jnp.asarray(inv), axis=-1)


def _equidistributed_knots(ta, tb, n: int, lam_fn, n_fine: int):
    """``n+1`` knots on [ta, tb] with equal ∫Λ dτ between neighbours."""
    ta = jnp.asarray(ta, dtype=jnp.float64)
    tb = jnp.maximum(jnp.asarray(tb, dtype=jnp.float64), ta)
    # Log-spaced fine grid resolves Λ ∝ 1/τ and the opacity drop alike.
    fine = jnp.exp(jnp.linspace(jnp.log(jnp.maximum(ta, 1e-30)),
                                jnp.log(jnp.maximum(tb, 1e-30)), n_fine))
    fine = fine.at[0].set(ta).at[-1].set(tb)
    lam = lam_fn(fine)
    cum = jnp.concatenate([
        jnp.zeros(1, dtype=jnp.float64),
        jnp.cumsum(0.5 * (lam[1:] + lam[:-1]) * jnp.diff(fine)),
    ])
    # Strictly increasing even for tb == ta (then all knots collapse onto ta).
    cum = cum + jnp.linspace(0.0, 1e-12, n_fine) * (cum[-1] + 1.0)
    u = jnp.linspace(cum[0], cum[-1], n + 1)
    knots = jnp.interp(u, cum, fine)
    return knots.at[0].set(ta).at[-1].set(tb)


_RK4_A = (0.0, 0.5, 0.5, 1.0)  # stage offsets (× h) and abscissae
_RK4_W = (1.0, 2.0, 2.0, 1.0)


def _rk4_segment(y0, knots, rhs, thermo, *, freeze=None, unroll: int = 1,
                 stage_loop: bool = False, keep=None, compensated: bool = False, record=None):
    """Fixed-knot RK4; returns (y_end, Y (n+1, n_eq), F (n+1, n_eq)).

    ``rhs(t, y, th4, t_step)`` where ``t_step`` is the step's start time (for
    per-step regime flags). ``freeze(t, t_next)`` (optional) holds y constant.
    ``keep`` (static index array): store only these entries of Y and F (the scan
    still integrates the whole state). ``compensated``: Kahan-summed state update
    (the rounding of y + Δy no longer accumulates over the steps). ``record =
    (slot, n_out)``: store no knots; the state after step i goes to row ``slot[i]``
    of the returned Y (n_out, n_eq) (row n_out: discarded), F is None.
    """
    sel = (lambda a: a) if keep is None else (lambda a: a[..., keep])
    t_lo, t_hi = knots[:-1], knots[1:]
    t_mid = 0.5 * (t_lo + t_hi)
    th_k = jax.vmap(lambda t: _thermo4(t, thermo))(knots)
    th_m = jax.vmap(lambda t: _thermo4(t, thermo))(t_mid)

    def body(carry, xs):
        y, comp, rec = carry
        t0, t1, tm, th0, thm, th1 = xs[:6]
        h = t1 - t0
        if stage_loop:
            ts = jnp.stack([t0, tm, tm, t1])
            ths = jnp.stack([th0, thm, thm, th1])
            a, w = jnp.asarray(_RK4_A), jnp.asarray(_RK4_W)

            def stage(i, carry):
                k_prev, ksum, k1 = carry
                ki = rhs(ts[i], y + (a[i] * h) * k_prev, ths[i], t0)
                ksum = jnp.where(i == 0, ki, ksum + w[i] * ki)
                return ki, ksum, jnp.where(i == 0, ki, k1)

            z = jnp.zeros_like(y)
            _, ksum, k1 = lax.fori_loop(0, 4, stage, (z, z, z))
            dy = (h / 6.0) * ksum
        else:
            k1 = rhs(t0, y, th0, t0)
            k2 = rhs(tm, y + 0.5 * h * k1, thm, t0)
            k3 = rhs(tm, y + 0.5 * h * k2, thm, t0)
            k4 = rhs(t1, y + h * k3, th1, t0)
            dy = (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if compensated:
            inc = dy - comp
            y1 = y + inc
            comp1 = (y1 - y) - inc
        else:
            y1, comp1 = y + dy, comp
        if freeze is not None:
            frz = freeze(t0, t1)
            y1 = jnp.where(frz, y, y1)
            comp1 = jnp.where(frz, comp, comp1)
            k1 = jnp.where(frz, jnp.zeros_like(k1), k1)
        if record is not None:
            rec = lax.dynamic_update_slice(rec, sel(y1)[None].astype(rec.dtype), (xs[6], jnp.zeros_like(xs[6])))
            return (y1, comp1, rec), None
        return (y1, comp1, rec), (sel(y), sel(k1))

    xs = (t_lo, t_hi, t_mid, th_k[:-1], th_m, th_k[1:])
    if record is not None:
        slot, n_out = record
        rec0 = jnp.broadcast_to(sel(y0), (int(n_out) + 1,) + sel(y0).shape)
        (y_end, _, rec), _ = lax.scan(body, (y0, jnp.zeros_like(y0), rec0), xs + (slot,),
                                      unroll=int(unroll))
        return y_end, rec[:int(n_out)], None
    (y_end, _, _), (ys, fs) = lax.scan(body, (y0, jnp.zeros_like(y0), None), xs, unroll=int(unroll))
    f_end = rhs(knots[-1], y_end, th_k[-1], knots[-1])
    if freeze is not None:
        f_end = jnp.where(freeze(knots[-1], knots[-1]), jnp.zeros_like(f_end), f_end)
    Y = jnp.concatenate([ys, sel(y_end)[None]], axis=0)
    F = jnp.concatenate([fs, sel(f_end)[None]], axis=0)
    return y_end, Y, F


def _hermite(knots, Y, F, tau):
    """Cubic Hermite interpolation of knot states onto ``tau`` (clamped)."""
    n = knots.shape[0]
    tq = jnp.clip(tau, knots[0], knots[-1])
    i = jnp.clip(jnp.searchsorted(knots, tq, side="right") - 1, 0, n - 2)
    t0, t1 = knots[i], knots[i + 1]
    h = t1 - t0
    s = jnp.where(h > 0, (tq - t0) / jnp.where(h > 0, h, 1.0), 0.0)[:, None]
    s2, s3 = s * s, s * s * s
    h00 = 2 * s3 - 3 * s2 + 1
    h10 = s3 - 2 * s2 + s
    h01 = -2 * s3 + 3 * s2
    h11 = s3 - s2
    hh = h[:, None]
    return h00 * Y[i] + h10 * hh * F[i] + h01 * Y[i + 1] + h11 * hh * F[i + 1]


def evolve_segmented(
    k,
    params,
    thermo,
    tau_out,
    *,
    cfg: HierarchyConfig,
    t0,
    t_tight,
    t_drop_nu,
    t_drop_phot,
    t_end,
    seg: SegmentedConfig = DEFAULT_SEGMENTED,
):
    """Evolve one k; return ``y(tau_out)`` in the full ``cfg`` layout (with
    ``seg.store == "sources"`` only the entries the sources read are filled).

    Switch times follow ``evolve_jax.evolve_scalar_k_jax`` (CAMB TC switch,
    DoLateRadTruncation ν/photon drops, ``tau_switch_ktau``).
    """
    tau_out = jnp.asarray(tau_out, dtype=jnp.float64)
    k_safe = jnp.maximum(k, 1e-10)
    mid_cfg = _mid_nu_rsa_hierarchy(cfg)
    rsa_cfg = _rsa_hierarchy_nu(cfg)
    nu_qmax = float(nu_momentum_grid(cfg.nqmax)[0][-1]) if cfg.nqmax > 0 else 0.0
    taurend = jnp.asarray(thermo.get("taurend", t_drop_nu), dtype=jnp.float64)
    # CAMB tau_switch_ktau (equations.f90 GaugeInterface_EvolveScal): the
    # massless-ν UFA closure starts at max(20, lmaxnr-4)/k whatever the photon
    # regime, i.e. also inside tight coupling for k ≳ 20/t_tight. Restricting it
    # to the post-TC segment left reflections from the ℓ = lmaxnr truncation in
    # the radiation-era potentials: -0.4% in the CDM transfer at k = 2/Mpc.
    t_ktau = jnp.asarray(max(20, int(cfg.lmaxnr) - 4), dtype=jnp.float64) / k_safe
    t_ktau = jnp.clip(t_ktau, t0, t_drop_nu)
    lmax_full = max(cfg.lmaxg, cfg.lmaxnr, cfg.lmaxgpol)
    outputs = seg.store == "outputs"
    if seg.store == "sources":
        keep = source_knot_indices(cfg)
    elif seg.store in ("all", "outputs"):
        keep = dict(tc=None, full=None, mid=None, rsa=None)
    else:
        raise ValueError(f"SegmentedConfig.store {seg.store!r} not in ('all', 'sources', 'outputs')")
    n_out = int(tau_out.shape[0])

    def segment(y_start, kn, rhs, keep_ix, **kw):
        """``_rk4_segment`` → (y_end, states at ``tau_out`` (n_out, len)): Hermite dense output
        over the stored knots, or (store "outputs") the output times made knots and recorded."""
        if not outputs:
            y_end, Y, F = _rk4_segment(y_start, kn, rhs, thermo, keep=keep_ix, **kw)
            return y_end, _hermite(kn, Y, F, tau_out)
        tq = jnp.clip(tau_out, kn[0], kn[-1])
        allk = jnp.concatenate([kn, tq])
        order = jnp.argsort(allk, stable=True)  # an output equal to a knot sorts after it
        pos = jnp.argsort(order)[kn.shape[0]:]  # ≥ 1: kn[0] ≤ every tq sorts first
        n_steps = int(allk.shape[0]) - 1
        slot = jnp.full(n_steps, n_out, dtype=jnp.int32).at[pos - 1].set(
            jnp.arange(n_out, dtype=jnp.int32))
        y_end, Yo, _ = _rk4_segment(y_start, allk[order], rhs, thermo, keep=keep_ix,
                                    record=(slot, n_out), **kw)
        return y_end, Yo

    def rhs_for(c: HierarchyConfig, *, t_hk=None, **flags):
        def rhs(t, y, th, t_step):
            extra = {} if t_hk is None else {"high_ktau_neutrino_approx": t_step >= t_hk}
            yp, _, _ = derivs(t, y, k, params, thermo, cfg=c,
                              thermo_at_tau=tuple(th[i] for i in range(4)),
                              **flags, **extra)
            return yp
        return rhs

    def knots_for(ta, tb, n, *, use_op, lmax, oscillatory=True, nu=False):
        return _equidistributed_knots(
            ta, tb, n,
            lambda t: _stiffness(t, k, params, thermo, use_op=use_op, lmax=lmax,
                                 w=seg, oscillatory=oscillatory,
                                 nu_qmax=nu_qmax if nu else 0.0),
            seg.n_fine,
        )

    def with_switch(kn):
        # CAMB restarts the ODE at tau_switch_ktau: make it a knot so no step
        # straddles the closure change (else first-order in the step size).
        return jnp.sort(jnp.concatenate([kn, jnp.clip(t_ktau, kn[0], kn[-1])[None]]))

    # 0: tight coupling (opacity stiffness removed by the TCA).
    kn0 = knots_for(t0, t_tight, seg.n_tc, use_op=False, lmax=2)
    if seg.high_ktau:
        kn0 = with_switch(kn0)
    y0 = initial_adiabatic(k, t0, params, cfg)
    rk = dict(unroll=seg.unroll, stage_loop=seg.stage_loop, compensated=seg.compensated)
    y_tc, Yo0 = segment(
        y0, kn0, rhs_for(cfg, t_hk=t_ktau if seg.high_ktau else None, tight_coupling=True),
        keep["tc"], **rk)
    y_full0 = _apply_tc_init(y_tc, t_tight, k, params, thermo, cfg)

    # 1: full hierarchy; high-kτ massless-ν closure once t ≥ t_ktau (if not already).
    kn1 = knots_for(t_tight, t_drop_nu, seg.n_full, use_op=True, lmax=lmax_full)
    if seg.high_ktau:
        kn1 = with_switch(kn1)
    y_nu, Yo1 = segment(
        y_full0, kn1, rhs_for(cfg, t_hk=t_ktau if seg.high_ktau else None, tight_coupling=False),
        keep["full"], **rk)

    # 2: mid hierarchy — photons kept, massless ν algebraic.
    kn2 = knots_for(t_drop_nu, t_drop_phot, seg.n_mid, use_op=True,
                    lmax=max(mid_cfg.lmaxg, mid_cfg.lmaxgpol))
    y_mid0 = _pack_full_to_mid(y_nu, cfg, mid_cfg)
    y_mid_end, Yo2 = segment(
        y_mid0, kn2, rhs_for(mid_cfg, tight_coupling=False, no_nu_multpoles=True),
        keep["mid"], **rk)

    # 3: photon + ν RSA (small, non-oscillatory state).
    n_rsa = seg.n_rsa * (NU_LIGHT_RSA_MULT if (cfg.nu_light and cfg.nqmax > 0) else 1)
    kn3 = knots_for(t_drop_phot, t_end, n_rsa, use_op=True,
                    lmax=max(2, rsa_cfg.lmaxnu if rsa_cfg.nqmax else 2),
                    oscillatory=False, nu=True)
    y_rsa0 = _pack_fluid_to_rsa(y_mid_end, mid_cfg, rsa_cfg)
    max_etak = float(MAX_ETAK_SCALAR)

    def freeze(t, t_next):
        return jnp.logical_and(t >= taurend, k_safe * t_next > max_etak)

    _, Yo3 = segment(
        y_rsa0, kn3,
        rhs_for(rsa_cfg, tight_coupling=False, no_phot_multpoles=True, no_nu_multpoles=True),
        keep["rsa"], freeze=freeze if seg.freeze_late else None, **rk)

    # Dense output per segment in native layout, then scatter + select.
    th_out = jax.vmap(lambda t: _thermo4(t, thermo))(tau_out)
    # During tight coupling the photon ℓ ≥ 2 / polarization multipoles are
    # algebraic (not evolved), so fill them from the TC relations at every output
    # time, as CAMB does when it evaluates sources in the TC regime. Without this
    # the sources jump at t_tight, and as t_tight moves with the parameters that
    # jump slides between τ samples (1e-3 wiggles in low-ℓ TT vs ω_c).
    y_s0 = jax.vmap(lambda yy, tt: _apply_tc_init(yy, tt, k, params, thermo, cfg))(
        _expand(Yo0, keep["tc"], cfg.n_eq), tau_out
    )
    y_s1 = _expand(Yo1, keep["full"], cfg.n_eq)
    y_s2 = jax.vmap(
        lambda ym, t, th: _scatter_mid_to_full(
            ym, t, k, params, thermo, cfg, mid_cfg, thermo_at_tau=tuple(th)
        )
    )(_expand(Yo2, keep["mid"], mid_cfg.n_eq), tau_out, th_out)
    y_s3 = jax.vmap(
        lambda yr, t, th: _scatter_rsa_to_full(
            yr, t, k, params, thermo, cfg, thermo_at_tau=tuple(th), rsa_cfg=rsa_cfg
        )
    )(_expand(Yo3, keep["rsa"], rsa_cfg.n_eq), tau_out, th_out)

    seg_id = (
        (tau_out >= t_tight).astype(jnp.int32)
        + (tau_out >= t_drop_nu)
        + (tau_out >= t_drop_phot)
    )[:, None]
    out = jnp.where(seg_id == 0, y_s0, y_s1)
    out = jnp.where(seg_id == 2, y_s2, out)
    return jnp.where(seg_id == 3, y_s3, out)


__all__ = ["DEFAULT_SEGMENTED", "SegmentedConfig", "evolve_segmented"]
