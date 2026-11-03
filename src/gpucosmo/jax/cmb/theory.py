#!/usr/bin/env python3
"""JAX CMB theory for desi_full: pluggable backends → TheorySpectra.

Default ``theory_backend='jax_eb'`` (real w0/wa Einstein–Boltzmann Cls).
``cosmopower`` remains available for regression (ΛCDM-trained emulators).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from gpucosmo.jax.background.cosmology import (
    BackgroundParams,
    BackgroundTables,
    theta_star,
)
from gpucosmo.jax.cmb.params import T_CMB_MUK

_CANDL_KK_LEN = 3000

TheoryBackend = Literal["cosmopower", "jax_eb"]


class TheorySpectra(NamedTuple):
    """D_l spectra for clipy/candl/plik_lite (muK^2 for TT/TE/EE; pp/kk in candl units)."""

    dl_tt: jnp.ndarray
    dl_te: jnp.ndarray
    dl_ee: jnp.ndarray
    dl_kk: jnp.ndarray
    H0: jnp.ndarray
    # Optional (jax_eb ``lmax_extra``): rows C_L^κκ, lensed D_ℓ^TT, EE, BB, TE (same units
    # as above) for ℓ = 0 … lmax_extra, past ``lmax`` (ACT DR6 lensing corrections).
    cmb_ext: jnp.ndarray | None = None


@lru_cache(maxsize=1)
def _emulators():
    from cosmopower_jax.cosmopower_jax import CosmoPowerJAX as CPJ

    return {
        "tt": CPJ(probe="cmb_tt"),
        "te": CPJ(probe="cmb_te"),
        "ee": CPJ(probe="cmb_ee"),
        "pp": CPJ(probe="cmb_pp"),
    }


def _cl_to_dl_temper(cl: jnp.ndarray, ell: jnp.ndarray) -> jnp.ndarray:
    return cl * (ell * (ell + 1.0) / (2.0 * jnp.pi)) * (T_CMB_MUK**2)


def _cl_pp_to_dl_kk(cl_pp: jnp.ndarray, ell: jnp.ndarray) -> jnp.ndarray:
    """candl convention: D_l^kk = D_l^pp * pi/2 with D_l^pp = C_l * ell^2(ell+1)^2/(2pi)."""
    dl_pp = cl_pp * (ell**2) * ((ell + 1.0) ** 2) / (2.0 * jnp.pi)
    dl_kk = dl_pp * (jnp.pi / 2.0)
    return dl_kk


def solve_h0_from_theta_mc(
    tables: BackgroundTables,
    ombh2: jnp.ndarray,
    omch2: jnp.ndarray,
    w0: jnp.ndarray,
    wa: jnp.ndarray,
    theta_mc_100: jnp.ndarray,
    *,
    ext: dict | None = None,
    n_iter: int = 20,
    tol: float = 1e-10,
) -> jnp.ndarray:
    """Invert ``theta_MC_100`` to ``H0`` using ``theta_* = r_s/D_M`` (CAMB convention).

    ``ext``: Σm_ν, N_eff, Ω_k, m_eff,sterile (``cls_jax.ext_params``; missing: the defaults)
    with the neutrino model of ``tables``. Python numbers are fixed into the solver, arrays
    are differentiable inputs.

    Newton iterations with the exact ∂θ/∂H0 (forward-mode AD) until the step is below ``tol``
    km/s/Mpc (at most ``n_iter``; from 67.5 that is 3–5 steps in the posterior; vmapped
    batches stop when every member has converged). Derivatives come from the implicit
    function theorem through a ``custom_jvp`` rule, Ḣ0 = −(∂F/∂p · ṗ)/(∂F/∂H0) with
    F = θ*(H0, p) − θ_target, written in differentiable JAX so that differentiating the rule
    again gives exact higher derivatives (forward-over-forward Taylor coefficients). The
    earlier "final Newton step under stop_gradient" was exact only to first order.
    """
    from gpucosmo.jax.eb.cls_jax import ext_params

    e = ext_params(ext)
    static = tuple(sorted((k, float(v)) for k, v in e.items() if isinstance(v, (int, float))))
    dyn_keys = tuple(sorted(k for k, v in e.items() if not isinstance(v, (int, float))))
    dyn = tuple(jnp.asarray(e[k], dtype=jnp.float64) for k in dyn_keys)
    return _h0_solver(tables, int(n_iter), float(tol), static, dyn_keys)(
        ombh2, omch2, w0, wa, theta_mc_100, *dyn)


_H0_SOLVERS: dict = {}


def _h0_solver(tables: BackgroundTables, n_iter: int, tol: float, static: tuple = (),
               dyn_keys: tuple = ()):
    key = (id(tables), n_iter, tol, static, dyn_keys)
    hit = _H0_SOLVERS.get(key)
    if hit is not None:
        return hit[0]

    def theta_of(h, ob, oc, w, a, *dyn):
        ext = dict(static, **dict(zip(dyn_keys, dyn)))
        return theta_star(tables, BackgroundParams.build(ob, oc, h, w, a, ext=ext))

    def newton_solve(ombh2, omch2, w0, wa, theta_mc_100, *dyn):
        target = theta_mc_100 / 100.0

        def step(h):
            th, dth = jax.jvp(lambda hh: theta_of(hh, ombh2, omch2, w0, wa, *dyn), (h,),
                              (jnp.ones_like(h),))
            return h - (th - target) / jnp.where(jnp.abs(dth) > 1e-12, dth, 1e-12)

        def body(state):
            h, _, it = state
            h1 = step(h)
            return h1, jnp.abs(h1 - h), it + 1

        def cond(state):
            _, st, it = state
            return (st > tol) & (it < n_iter)

        h_init = jnp.full(jnp.shape(target), 67.5, dtype=jnp.float64)
        h0, _, _ = jax.lax.while_loop(cond, body, (h_init, jnp.full_like(h_init, jnp.inf), jnp.asarray(0)))
        return h0

    @jax.custom_jvp
    def solve(ombh2, omch2, w0, wa, theta_mc_100, *dyn):
        return newton_solve(ombh2, omch2, w0, wa, theta_mc_100, *dyn)

    @solve.defjvp
    def solve_jvp(primals, tangents):
        h = solve(*primals)
        ob, oc, w, a, th100, *dyn = primals

        def F(hh, ob_, oc_, w_, a_, th_, *dyn_):
            return theta_of(hh, ob_, oc_, w_, a_, *dyn_) - th_ / 100.0

        _, f_p = jax.jvp(lambda *q: F(h, *q), primals, tangents)
        _, f_h = jax.jvp(lambda hh: F(hh, ob, oc, w, a, th100, *dyn), (h,), (jnp.ones_like(h),))
        return h, -f_p / f_h

    _H0_SOLVERS[key] = (solve, tables)  # keep ``tables`` alive: the key is its id
    return solve


def _get_theory_spectra_cosmopower(
    tables: BackgroundTables,
    *,
    logA: jnp.ndarray,
    ns: jnp.ndarray,
    theta_mc_100: jnp.ndarray,
    ombh2: jnp.ndarray,
    omch2: jnp.ndarray,
    tau: jnp.ndarray,
    w0: jnp.ndarray,
    wa: jnp.ndarray,
    lmax: int,
) -> TheorySpectra:
    """CosmoPower-JAX CMB emulators with H0 solved from ``theta_MC_100``."""
    h0 = solve_h0_from_theta_mc(tables, ombh2, omch2, w0, wa, theta_mc_100)
    h = h0 / 100.0
    cp_in = jnp.stack([ombh2, omch2, h, tau, ns, logA])

    em = _emulators()
    cl_tt = em["tt"].predict(cp_in).squeeze()
    cl_te = em["te"].predict(cp_in).squeeze()
    cl_ee = em["ee"].predict(cp_in).squeeze()
    cl_pp = em["pp"].predict(cp_in).squeeze()
    ell_cp = jnp.asarray(em["tt"].modes, dtype=float)

    dl_tt_cp = _cl_to_dl_temper(cl_tt, ell_cp)
    dl_te_cp = _cl_to_dl_temper(cl_te, ell_cp)
    dl_ee_cp = _cl_to_dl_temper(cl_ee, ell_cp)
    dl_kk_cp = _cl_pp_to_dl_kk(cl_pp, ell_cp)

    n_cp = int(cl_tt.shape[0])

    def _embed(dl_cp: jnp.ndarray) -> jnp.ndarray:
        out = jnp.zeros(lmax + 1)
        out = out.at[2 : 2 + n_cp].set(dl_cp)
        return out

    dl_kk = jnp.zeros(_CANDL_KK_LEN)
    n_kk = min(_CANDL_KK_LEN, n_cp)
    dl_kk = dl_kk.at[:n_kk].set(dl_kk_cp[:n_kk])

    return TheorySpectra(
        dl_tt=_embed(dl_tt_cp),
        dl_te=_embed(dl_te_cp),
        dl_ee=_embed(dl_ee_cp),
        dl_kk=dl_kk,
        H0=h0,
    )


def _get_theory_spectra_jax_eb(
    tables: BackgroundTables,
    *,
    logA: jnp.ndarray,
    ns: jnp.ndarray,
    theta_mc_100: jnp.ndarray,
    ombh2: jnp.ndarray,
    omch2: jnp.ndarray,
    tau: jnp.ndarray,
    w0: jnp.ndarray,
    wa: jnp.ndarray,
    lmax: int,
    fwd_jacobian: bool = True,
) -> TheorySpectra:
    """jax_eb theory; ``fwd_jacobian`` makes ``jax.grad`` use the 8-parameter
    forward-mode Jacobian (``theory_spectra_fwd_jacobian``) instead of reverse
    mode through the solver. Values are identical; forward-mode ``jvp`` of the
    caller is then unavailable (custom_vjp)."""
    from gpucosmo.jax.cmb.jax_eb_theory import (
        get_theory_spectra_jax_eb,
        theory_spectra_fwd_jacobian,
    )

    if fwd_jacobian:
        x = jnp.stack([jnp.asarray(v, dtype=jnp.float64)
                       for v in (logA, ns, theta_mc_100, ombh2, omch2, tau, w0, wa)])
        return theory_spectra_fwd_jacobian(tables, x, lmax=lmax)
    return get_theory_spectra_jax_eb(
        tables,
        logA=logA,
        ns=ns,
        theta_mc_100=theta_mc_100,
        ombh2=ombh2,
        omch2=omch2,
        tau=tau,
        w0=w0,
        wa=wa,
        lmax=lmax,
    )


def get_theory_spectra(
    tables: BackgroundTables,
    *,
    logA: jnp.ndarray,
    ns: jnp.ndarray,
    theta_mc_100: jnp.ndarray,
    ombh2: jnp.ndarray,
    omch2: jnp.ndarray,
    tau: jnp.ndarray,
    w0: jnp.ndarray,
    wa: jnp.ndarray,
    lmax: int = 2508,
    theory_backend: TheoryBackend = "jax_eb",
) -> TheorySpectra:
    """
    CMB D_ℓ theory for desi_full with H0 solved from ``theta_MC_100``.

    ``theory_backend``:
      - ``jax_eb`` (default): in-repo EB Cls with real CPL w0/wa
      - ``cosmopower``: ΛCDM-trained emulators (w0/wa via H0 only; regression)
    """
    if theory_backend == "jax_eb":
        return _get_theory_spectra_jax_eb(
            tables,
            logA=logA,
            ns=ns,
            theta_mc_100=theta_mc_100,
            ombh2=ombh2,
            omch2=omch2,
            tau=tau,
            w0=w0,
            wa=wa,
            lmax=lmax,
        )
    if theory_backend == "cosmopower":
        return _get_theory_spectra_cosmopower(
            tables,
            logA=logA,
            ns=ns,
            theta_mc_100=theta_mc_100,
            ombh2=ombh2,
            omch2=omch2,
            tau=tau,
            w0=w0,
            wa=wa,
            lmax=lmax,
        )
    raise ValueError(
        f"Unknown theory_backend={theory_backend!r}; "
        "expected 'jax_eb' or 'cosmopower'."
    )


__all__ = [
    "TheoryBackend",
    "TheorySpectra",
    "get_theory_spectra",
    "solve_h0_from_theta_mc",
]
