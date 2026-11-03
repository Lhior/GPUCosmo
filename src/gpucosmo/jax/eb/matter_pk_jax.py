#!/usr/bin/env python3
# This file is part of GPUCosmo. Copyright (C) 2025-2026 Leonel Medina-Varela.
# SPDX-License-Identifier: GPL-3.0-or-later
# Derived in part from the following, translated to JAX and modified for GPUCosmo:
# - CAMB (https://github.com/cmbant/CAMB), Copyright 2000-2025 Antony Lewis and Anthony Challinor.
#   Licence: LICENSES/CAMB.txt.
# - CCL (https://github.com/LSSTDESC/CCL), Copyright (c) 2016-2018, the LSSTDESC CCL
#   contributors.
#   Licence: LICENSES/CCL.txt.
# See NOTICE.md.
"""Production ``jax_eb`` matter power spectrum: CAMB transfer run + CCL assembly.

Reproduces what ``pyccl`` does with ``transfer_function='boltzmann_camb'`` and
``matter_power_spectrum='camb'`` (``extra_parameters={'camb': {'halofit_version':
'mead2020', 'kmax': 10}}``), with the Boltzmann solve on GPU:

1. CAMB ``GetTransfer`` (``WantCls=False``) for the matter transfer at CCL's
   ``get_pk_spline_a()`` output times on CAMB's ``InitTransfer`` k grid
   (:mod:`gpucosmo.jax.eb.transfer_evolve_jax`);
2. ``P = 2 pi^2 / k^3 * A_s (k/k_p)^(n_s-1) * T_tot^2`` on those nodes, a CCL
   ``Pk2D`` of ln P (``extrap_order_lok=1``, ``extrap_order_hik=2``) and the
   sigma8 rescale of ``A_s`` exactly as ``pyccl.boltzmann.get_camb_pk_lin``;
3. HMcode-2020 (CAMB ``halofit_version='mead2020'``,
   :mod:`gpucosmo.jax.eb.hmcode2020_jax`) fed CAMB's ``MatterPowerData`` table:
   natural cubic spline of ln P in ln k on the native nodes, linear ln P-ln k
   extrapolation past the last two nodes, rebinned to 512 points in
   [1e-3, 1e2] h/Mpc, ratio evaluated at the native nodes.

The k grid is CAMB's for a reference ``taurst`` (the node count depends on
``taurst``, which must be static under ``jit``; the CMB path does the same).
Scope: flat ΛCDM with massless neutrinos (N_eff = 3.044), as the GGL
likelihoods use; w0wa needs CAMB's fluid dark-energy perturbations (not yet).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import numpy as np

from gpucosmo.jax.eb import constants as C
from gpucosmo.jax.eb.hierarchy_jax import pack_thermo_values
from gpucosmo.jax.eb.transfer_evolve_jax import (
    TransferStepConfig,
    camb_transfer_hierarchy,
    camb_transfer_lmax,
    evolve_transfer_mode,
)

jax.config.update("jax_enable_x64", True)

AS_FIDUCIAL = 2.1e-9
K_PIVOT = 0.05


def ccl_pk_spline_a() -> np.ndarray:
    """``pyccl.get_pk_spline_a()`` (CCL defaults), ascending."""
    try:
        import pyccl

        return np.asarray(pyccl.get_pk_spline_a(), dtype=float)
    except ImportError:  # CCL spline_params defaults
        a_log = np.geomspace(0.01, 0.1, 11)
        a_lin = np.linspace(0.1, 1.0, 40)
        return np.concatenate([a_log[:-1], a_lin])


def camb_transfer_k_grid(
    kmax: float = 10.0,
    taurst: float = 280.0,
    *,
    high_precision: bool = True,
    accuracy_boost: float = 1.0,
) -> np.ndarray:
    """CAMB ``InitTransfer`` optimized q grid [1/Mpc] (cmbmain.f90), k_per_logint=0."""
    boost = accuracy_boost * (1.5 if high_precision else 1.0)
    q_switch_lowk1 = 0.7 / taurst
    dlog_lowk1 = 2 * boost
    q_switch_lowk = 8 / taurst
    dlog_lowk = 8 * boost * 2.5
    q_switch_osc = min(kmax, 30 / taurst)
    d_osc = 200 * boost * 1.8
    dlog_osc = 17 * boost
    q_switch_highk = min(kmax, 90 / taurst)
    dlog_highk = 3 * boost
    amin = 5e-5
    q = []
    nq = int(np.log(q_switch_lowk1 / amin) * dlog_lowk1) + 1
    q += [amin * np.exp(i / dlog_lowk1) for i in range(nq)]
    nq = int(np.log(q_switch_lowk / q[-1]) * dlog_lowk) + 1
    last = q[-1]
    q += [last * np.exp(i / dlog_lowk) for i in range(1, nq + 1)]
    if q_switch_osc > q_switch_lowk:
        last = q[-1]
        nq = int((q_switch_osc - last) * d_osc) + 1
        q += [last + i / d_osc for i in range(1, nq + 1)]
    if kmax > q[-1]:
        last = q[-1]
        nq = int(np.log(q_switch_highk / last) * dlog_osc) + 1
        q += [last * np.exp(i / dlog_osc) for i in range(1, nq + 1)]
    if kmax > q[-1]:
        last = q[-1]
        nq = int(np.log(kmax / last) * dlog_highk) + 1
        dlog_highk = nq / np.log(kmax * 1.05 / last)
        q += [last * np.exp(i / dlog_highk) for i in range(1, nq + 1)]
    return np.asarray(q, dtype=float)


@dataclass(frozen=True)
class MatterPkConfig:
    """Static settings (hashable: one compiled program per config)."""

    kmax: float = 10.0
    # CAMB taurst (0.9 tau at cumulative visibility 1e-7) for Omega_m=0.3, h=0.7:
    # reproduces its 194-node grid to 1e-7.
    taurst_ref: float = 204.55
    # BBN table of CAMB's default ``bbn_predictor`` (camb/bbn.py), which pyccl uses.
    bbn_table: str = "PRIMAT_Yp_DH_ErrorMC_2024.dat"
    z_reio: float = 10.0
    # Validated against CAMB 2.0 (tests/test_jax_eb_matter_pk.py) for
    # 0.05 <= Omega_m <= 0.6: the hierarchies are implicit ("banded"), so the
    # step counts are set by accuracy, not by the k*h stability limit.
    steps_main: TransferStepConfig = field(
        default_factory=lambda: TransferStepConfig(n_mid=256)
    )
    # Modes with k < k_split run a separate vmap with ``steps_low`` (step counts
    # are uniform within a vmap, and ~70% of CAMB's nodes are at k < 0.5 where a
    # fraction of the k = 10 budget suffices). ``k_split = None``: one vmap.
    k_split: float | None = 0.5
    steps_low: TransferStepConfig = field(
        default_factory=lambda: TransferStepConfig(n_tc=128, n_full=384, n_mid=128, n_rsa=128)
    )
    # CAMB evaluates HMcode at every output a; the (a, ln k) bicubic spline is
    # not local in a, so skipping early nodes leaks ~3e-3 into z <= 3 at high k.
    hmcode_a_min: float = 0.0
    # Node density relative to CAMB's (AccuracyBoost-style). CAMB's grid moves
    # with the cosmology's taurst; a static grid at boost 1 therefore puts its
    # nodes between CAMB's away from the reference cosmology, and the ln P spline
    # error across the first BAO wiggles (k ~ 0.005-0.08) reaches ~1e-3. Extra
    # modes cost almost nothing on GPU (latency-bound).
    k_density_boost: float = 1.0
    # Run the scalar sequential stages (Recfast + thermo table, HMcode growth
    # ODE) on the host CPU: same XLA programs, 11 ms there vs 1.3 s as
    # single-thread GPU kernels. Not differentiable; False keeps everything on
    # the device.
    host_scalar_stages: bool = True

    @property
    def k_nodes(self) -> np.ndarray:
        return camb_transfer_k_grid(self.kmax, self.taurst_ref,
                                    accuracy_boost=self.k_density_boost)

    @property
    def a_nodes(self) -> np.ndarray:
        return ccl_pk_spline_a()

    @property
    def tau_min(self) -> float:
        """CAMB thermo table start: 0.95 * GetTauStart(1.05 * kmax)."""
        return 0.95 * min(0.001 / (1.05 * self.kmax), 0.1)

    @property
    def nthermo(self) -> int:
        """CAMB results.f90: nint(20000 ln(1.4e4/taumin) / ln(1.4e4/2e-4)), taumin
        = GetTauStart(1.05 kmax) (the table itself starts at 0.95 taumin)."""
        taumin = min(0.001 / (1.05 * self.kmax), 0.1)
        return int(np.floor(20000.0 * np.log(1.4e4 / taumin) / np.log(1.4e4 / 2e-4) + 0.5))


def _camb_params(Omega_c, Omega_b, h, n_s, *, bbn_table: str = "PRIMAT_Yp_DH_ErrorMC_2024.dat"):
    from gpucosmo.jax.eb.bbn_jax import delta_neff_for, yhe_from_bbn
    from gpucosmo.jax.eb.cls_jax import _params_dict_cls

    ombh2 = Omega_b * h * h
    omch2 = Omega_c * h * h
    yhe = yhe_from_bbn(ombh2, delta_neff=delta_neff_for(C.DEFAULT_NNU), table=bbn_table)
    return _params_dict_cls(
        ombh2, omch2, h, n_s, AS_FIDUCIAL, 0.0544, -1.0, 0.0,
        mnu=0.0, num_massive_nu=0, YHe=yhe,
    )


def _thermo(params, cfg: MatterPkConfig):
    from gpucosmo.jax.eb.kgrid_jax import sync_thermo_taurend_camb_like
    from gpucosmo.jax.eb.sources_jax import build_visibility_tables
    from gpucosmo.jax.eb.thermo_jax import build_thermo_jax

    th = build_thermo_jax(
        params, nthermo=cfg.nthermo, tau_min=cfg.tau_min, bg_a_min=1e-11, z_reio=cfg.z_reio,
    )
    return sync_thermo_taurend_camb_like(build_visibility_tables(th))


@lru_cache(maxsize=None)
def _host_thermo_fn(cfg: MatterPkConfig):
    from gpucosmo.jax.eb.host_offload import on_host

    def build(Omega_c, Omega_b, h, n_s):
        th = _thermo(_camb_params(Omega_c, Omega_b, h, n_s, bbn_table=cfg.bbn_table), cfg)
        return {key: jnp.asarray(val, dtype=jnp.float64) for key, val in th.items()}

    return on_host(build)


@lru_cache(maxsize=None)
def _host_growth_fn():
    from gpucosmo.jax.eb.hmcode2020_jax import _growth_tables
    from gpucosmo.jax.eb.host_offload import on_host

    def build(Omega_m, Omega_v):
        return _growth_tables(dict(Omega_m=Omega_m, Omega_v=Omega_v,
                                   w0=jnp.float64(-1.0), wa=jnp.float64(0.0)))

    return on_host(build)


def transfer_tot_nodes(Omega_c, Omega_b, h, n_s, cfg: MatterPkConfig):
    """T_tot = delta_m (synchronous gauge) on (a_nodes, k_nodes), shape (n_a, n_k)."""
    params = _camb_params(Omega_c, Omega_b, h, n_s, bbn_table=cfg.bbn_table)
    if cfg.host_scalar_stages:
        th = _host_thermo_fn(cfg)(Omega_c, Omega_b, h, n_s)
    else:
        th = _thermo(params, cfg)
    # Keep the thermo build a separate stage: fused into its many consumers
    # (knot placement, per-stage lookups) XLA made it ~3x slower (1.4 s vs 0.5 s).
    arrays = {key: val for key, val in th.items() if hasattr(val, "shape")}
    arrays = jax.lax.optimization_barrier(arrays)
    th = pack_thermo_values({**th, **arrays})
    a_nodes = jnp.asarray(cfg.a_nodes)
    tau_out = jnp.interp(a_nodes, th["a"], th["tau_mpc"])  # ascending
    ks = jnp.asarray(cfg.k_nodes)
    grhob = params["Omega_b"]
    grhoc = params["Omega_c"]
    hier = camb_transfer_hierarchy("padded")

    def evolve(ks_group, steps):
        def one(k):
            return evolve_transfer_mode(
                k, params, th, tau_out, cfg=hier, steps=steps,
                lmax_live=camb_transfer_lmax(k), late_phot_switch_boost=2.0,
            )
        return jax.vmap(one)(ks_group)  # (n_k_group, n_a, 2)

    k_np = cfg.k_nodes
    if cfg.k_split is None or not (k_np < cfg.k_split).any():
        cb = evolve(ks, cfg.steps_main)
    else:
        n_lo = int((k_np < cfg.k_split).sum())
        cb = jnp.concatenate(
            [evolve(ks[:n_lo], cfg.steps_low), evolve(ks[n_lo:], cfg.steps_main)], axis=0
        )
    dm = (grhoc * cb[..., 0] + grhob * cb[..., 1]) / (grhoc + grhob)
    return dm.T, params


def _sigma8_ccl(pk, h):
    from gpucosmo.jax.ccl.halos._sigma import _LOG10K, _sigma2_from_pk

    lk = jnp.asarray(_LOG10K * np.log(10.0))
    pkn = pk.eval_lk(lk, 1.0)
    return jnp.sqrt(_sigma2_from_pk(pkn, jnp.asarray([8.0 / h])))[0]


def _camb_matterpower_rebin(lk_nodes, lnp, kh_out_ln, h):
    """CAMB ``MatterPowerData_k``: natural spline in ln k inside, linear beyond."""
    from gpucosmo.jax.gsl_splines import cspline_spline, spline_eval

    s = cspline_spline(lk_nodes, lnp)
    lk_q = kh_out_ln + jnp.log(h)  # h/Mpc -> 1/Mpc
    inside = spline_eval(s, jnp.clip(lk_q, lk_nodes[0], lk_nodes[-1]))
    slope_hi = (lnp[-1] - lnp[-2]) / (lk_nodes[-1] - lk_nodes[-2])
    slope_lo = (lnp[1] - lnp[0]) / (lk_nodes[1] - lk_nodes[0])
    hi = lnp[-1] + slope_hi * (lk_q - lk_nodes[-1])
    lo = lnp[0] + slope_lo * (lk_q - lk_nodes[0])
    return jnp.where(lk_q > lk_nodes[-1], hi, jnp.where(lk_q < lk_nodes[0], lo, inside))


def _hmcode_ratio(lk_nodes, lnp_lin, a_nodes, params, cfg: MatterPkConfig):
    from gpucosmo.jax.eb.hmcode2020_jax import hmcode2020_ratio, linear_table_k_h

    h = params["h"]
    kt = jnp.asarray(linear_table_k_h())  # h/Mpc
    lkt = jnp.log(kt)
    use = np.asarray(cfg.a_nodes) >= cfg.hmcode_a_min
    a_nl = a_nodes[np.where(use)[0]]
    lnp_nl = lnp_lin[np.where(use)[0]]

    def tab_one(lnp_row):
        lnp_q = _camb_matterpower_rebin(lk_nodes, lnp_row, lkt, h)
        # Delta^2 = k^3 P / 2 pi^2 (k [1/Mpc], P [Mpc^3]) = kh^3 P_h / 2 pi^2
        return 3.0 * (lkt + jnp.log(h)) + lnp_q - jnp.log(2.0 * np.pi**2)

    tab = jax.vmap(tab_one)(lnp_nl)
    cosmo = dict(
        Omega_m=params["Omega_m"], Omega_b=params["Omega_b"], Omega_c=params["Omega_c"],
        Omega_nu=0.0, Omega_v=params["Omega_de"], h=h, n_s=params["n_s"],
        w0=-1.0, wa=0.0, T_cmb=params["T_CMB"], N_nu=0.0,
    )
    kh_nodes = jnp.exp(lk_nodes) / h
    gt = (_host_growth_fn()(params["Omega_m"], params["Omega_de"])
          if cfg.host_scalar_stages else None)
    ratio = hmcode2020_ratio(kh_nodes, a_nl, tab, cosmo, growth_tables=gt)  # (n_a_nl, n_k)
    full = jnp.ones_like(lnp_lin)
    return full.at[np.where(use)[0]].set(ratio)


@partial(jax.jit, static_argnames=("cfg", "nonlinear"))
def matter_pk_nodes(Omega_c, Omega_b, h, n_s, sigma8, *, cfg: MatterPkConfig, nonlinear=True):
    """(lk_nodes, a_nodes, lnP_lin, lnP_nl) on CAMB/CCL nodes, sigma8-normalised."""
    from gpucosmo.jax.ccl.pk2d import Pk2D

    dm, params = transfer_tot_nodes(Omega_c, Omega_b, h, n_s, cfg)
    k = jnp.asarray(cfg.k_nodes)
    lk = jnp.log(k)
    a_nodes = jnp.asarray(cfg.a_nodes)
    p_r = AS_FIDUCIAL * (k / K_PIVOT) ** (n_s - 1.0)
    lnp = jnp.log(2.0 * np.pi**2 / k**3 * p_r)[None, :] + 2.0 * jnp.log(jnp.abs(dm))
    pk_raw = Pk2D(a_arr=a_nodes, lk_arr=lk, pk_arr=jnp.exp(lnp), is_logp=True,
                  extrap_order_lok=1, extrap_order_hik=2)
    s8_raw = _sigma8_ccl(pk_raw, h)
    lnp = lnp + 2.0 * jnp.log(sigma8 / s8_raw)
    if not nonlinear:
        return lk, a_nodes, lnp, lnp
    ratio = _hmcode_ratio(lk, lnp, a_nodes, params, cfg)
    return lk, a_nodes, lnp, lnp + 2.0 * jnp.log(ratio)


def matter_pk2d_pair(Omega_c, Omega_b, h, n_s, sigma8, *, cfg: MatterPkConfig | None = None):
    """``(pk_lin, pk_nl)`` JAX ``Pk2D`` as CCL builds them from CAMB (mead2020)."""
    from gpucosmo.jax.ccl.pk2d import Pk2D

    cfg = cfg or MatterPkConfig()
    lk, a, lnp_lin, lnp_nl = matter_pk_nodes(Omega_c, Omega_b, h, n_s, sigma8, cfg=cfg)
    mk = lambda lnp: Pk2D(a_arr=a, lk_arr=lk, pk_arr=jnp.exp(lnp), is_logp=True,  # noqa: E731
                          extrap_order_lok=1, extrap_order_hik=2)
    return mk(lnp_lin), mk(lnp_nl)


__all__ = [
    "MatterPkConfig",
    "camb_transfer_k_grid",
    "ccl_pk_spline_a",
    "matter_pk2d_pair",
    "matter_pk_nodes",
    "transfer_tot_nodes",
]
