#!/usr/bin/env python3
"""CCL-matched Limber integration in log(k) with Gauss-Kronrod quadrature."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from gpucosmo.jax.splines import (
    INTEGRATION_LIMBER_EPSREL,
    INTEGRATION_LIMBER_MAX_SUBDIV,
    ccl_spline_params,
    log_k_grid,
)

_C_LIGHT = 299792.458

from numpy.polynomial.legendre import leggauss

# 25-point Gauss-Legendre nodes on [-1, 1] (used for composite Limber integration).
_GL25_X, _GL25_W = leggauss(25)
_GL25_X_J = jnp.asarray(_GL25_X)
_GL25_W_J = jnp.asarray(_GL25_W)

# 41-point Gauss-Kronrod nodes on [-1, 1] (orders 4 and 20).
_GK41_X = np.array(
    [
        -0.9956571630258080,
        -0.9764421408059000,
        -0.9429745712285932,
        -0.8949914453360103,
        -0.8337093438151741,
        -0.7602092560552659,
        -0.6747411767844591,
        -0.5787626957072917,
        -0.4730027022547945,
        -0.3592104156718870,
        -0.2386145241982145,
        -0.1121550686959141,
        0.0,
        0.1121550686959141,
        0.2386145241982145,
        0.3592104156718870,
        0.4730027022547945,
        0.5787626957072917,
        0.6747411767844591,
        0.7602092560552659,
        0.8337093438151741,
        0.8949914453360103,
        0.9429745712285932,
        0.9764421408059000,
        0.9956571630258080,
    ],
    dtype=float,
)
_GK41_W = np.array(
    [
        0.0059591418384372,
        0.0176140071391521,
        0.0293033978629760,
        0.0406014298233940,
        0.0511403351834587,
        0.0609279728438139,
        0.0698282019798600,
        0.0777331034559707,
        0.0844442355049328,
        0.0898216941014135,
        0.0936913988097003,
        0.0969048595900451,
        0.0984715853860409,
        0.0969048595900451,
        0.0936913988097003,
        0.0898216941014135,
        0.0844442355049328,
        0.0777331034559707,
        0.0698282019798600,
        0.0609279728438139,
        0.0511403351834587,
        0.0406014298233940,
        0.0293033978629760,
        0.0176140071391521,
        0.0059591418384372,
    ],
    dtype=float,
)

_GK41_X_J = jnp.asarray(_GK41_X)
_GK41_W_J = jnp.asarray(_GK41_W)


def limber_k_bounds(
    ell: float,
    chi_min: float,
    chi_max: float,
    *,
    k_min: float | None = None,
    k_max: float | None = None,
) -> tuple[float, float]:
    """Per-mode k limits from Limber mapping chi = (ell+1/2)/k."""
    params = ccl_spline_params()
    k_lo = float(k_min if k_min is not None else params["K_MIN"])
    k_hi = float(k_max if k_max is not None else params["K_MAX"])
    ell_p = float(ell) + 0.5
    if chi_max > 0.0:
        k_lo = max(k_lo, ell_p / chi_max)
    if chi_min > 0.0:
        k_hi = min(k_hi, ell_p / chi_min)
    if k_hi <= k_lo:
        k_hi = k_lo * (1.0 + 1.0e-6)
    return k_lo, k_hi


def _gk41_integral(
    log_k_lo: jnp.ndarray,
    log_k_hi: jnp.ndarray,
    integrand_fn,
) -> jnp.ndarray:
    """Integrate integrand(k) k d(ln k) on [log_k_lo, log_k_hi] with GL25."""
    mid = 0.5 * (log_k_hi + log_k_lo)
    half = 0.5 * (log_k_hi - log_k_lo)
    nodes = mid + half * _GL25_X_J
    k = jnp.exp(nodes)
    vals = integrand_fn(k) * k
    return half * jnp.sum(_GL25_W_J * vals)


def _adaptive_gk_integral(
    log_k_lo: float,
    log_k_hi: float,
    integrand_fn,
    *,
    epsrel: float = INTEGRATION_LIMBER_EPSREL,
    max_subdiv: int = INTEGRATION_LIMBER_MAX_SUBDIV,
) -> float:
    """Adaptive GK on log(k), matching CCL qag_quad tolerance (NumPy driver)."""
    log_lo = float(log_k_lo)
    log_hi = float(log_k_hi)
    stack = [(log_lo, log_hi)]
    total = 0.0
    for _ in range(max_subdiv):
        if not stack:
            break
        a, b = stack.pop()
        mid = 0.5 * (a + b)
        half = 0.5 * (b - a)
        nodes = mid + half * _GK41_X
        k = np.exp(nodes)
        vals = np.asarray(integrand_fn(k), dtype=float) * k
        est = half * np.sum(_GK41_W * vals)
        # one-level refinement
        left_nodes = mid - 0.25 * (b - a) + 0.25 * (b - a) * _GK41_X
        right_nodes = mid + 0.25 * (b - a) + 0.25 * (b - a) * _GK41_X
        k_l = np.exp(left_nodes)
        k_r = np.exp(right_nodes)
        est_l = 0.25 * (b - a) * np.sum(_GK41_W * np.asarray(integrand_fn(k_l)) * k_l)
        est_r = 0.25 * (b - a) * np.sum(_GK41_W * np.asarray(integrand_fn(k_r)) * k_r)
        err = abs(est - (est_l + est_r))
        if err <= epsrel * max(abs(est), 1.0e-30):
            total += est
        else:
            stack.append((a, mid))
            stack.append((mid, b))
    return total


def limber_cl_qag_numpy(
    ell: float,
    integrand_fn,
    *,
    chi_min: float,
    chi_max: float,
    k_min: float | None = None,
    k_max: float | None = None,
) -> float:
    """CCL-style Limber C_ell with log-k qag_quad (matches z-GK 2/c prefactor)."""
    k_lo, k_hi = limber_k_bounds(ell, chi_min, chi_max, k_min=k_min, k_max=k_max)
    integral = _adaptive_gk_integral(np.log(k_lo), np.log(k_hi), integrand_fn)
    return 2.0 * integral / (_C_LIGHT * (float(ell) + 0.5))


def limber_cl_qag_jax(
    ell: jnp.ndarray,
    log_k_lo: jnp.ndarray,
    log_k_hi: jnp.ndarray,
    pk_w1_w2_fn,
) -> jnp.ndarray:
    """JAX GK Limber integral for batched ell (fixed subinterval, JIT-friendly)."""

    def one_mode(log_lo, log_hi, ell_val):
        def integrand(k):
            return pk_w1_w2_fn(k, ell_val)

        return _gk41_integral(log_lo, log_hi, integrand) * 2.0 / (_C_LIGHT * (ell_val + 0.5))

    if ell.ndim == 0:
        return one_mode(log_k_lo, log_k_hi, ell)
    return jax.vmap(one_mode)(log_k_lo, log_k_hi, ell)


def build_log_k_bounds_batch(
    ell: np.ndarray,
    chi_min: float,
    chi_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build per-ell log(k) integration limits."""
    log_lo = np.empty_like(ell, dtype=float)
    log_hi = np.empty_like(ell, dtype=float)
    for i, ell_val in enumerate(ell):
        k_lo, k_hi = limber_k_bounds(float(ell_val), chi_min, chi_max)
        log_lo[i] = np.log(k_lo)
        log_hi[i] = np.log(k_hi)
    return log_lo, log_hi


def integrate_gk41_logx(
    integrand_fn,
    log_x_lo: jnp.ndarray,
    log_x_hi: jnp.ndarray,
) -> jnp.ndarray:
    """∫ integrand(x) dx on [log_x_lo, log_x_hi] with 41-point GK in log-x."""
    mid = 0.5 * (log_x_lo + log_x_hi)
    half = 0.5 * (log_x_hi - log_x_lo)
    log_nodes = mid + half * _GL25_X_J
    x = jnp.exp(log_nodes)
    vals = integrand_fn(x) * x
    return half * jnp.sum(_GL25_W_J * vals)


def integrate_gl25_logk(
    integrand_fn,
    log_k_lo: jnp.ndarray,
    log_k_hi: jnp.ndarray,
) -> jnp.ndarray:
    """∫ integrand(k) k d(ln k) on [log_k_lo, log_k_hi] with 25-point GL."""
    mid = 0.5 * (log_k_lo + log_k_hi)
    half = 0.5 * (log_k_hi - log_k_lo)
    log_nodes = mid + half * _GL25_X_J
    k = jnp.exp(log_nodes)
    vals = integrand_fn(k) * k
    return half * jnp.sum(_GL25_W_J * vals)


def integrate_gl25_logk_multisub(
    integrand_fn,
    log_k_lo: jnp.ndarray,
    log_k_hi: jnp.ndarray,
    *,
    max_subdiv: int = INTEGRATION_LIMBER_MAX_SUBDIV,
) -> jnp.ndarray:
    """Multi-interval GL25 on log(k), sized by CCL ``DLOGK_INTEGRATION`` (JIT-safe).

    Uses a static ``fori_loop`` upper bound so reverse-mode AD (NUTS) works.
    Inactive subdivisions contribute zero.
    """
    params = ccl_spline_params()
    dlogk = float(params["DLOGK_INTEGRATION"])
    total = log_k_hi - log_k_lo
    n_sub = jnp.minimum(
        jnp.maximum(jnp.ceil(total / dlogk).astype(jnp.int32), 1),
        int(max_subdiv),
    )
    n_loop = int(max_subdiv)

    def body(i, acc):
        # Clamp inactive iters so segment math stays well-defined under AD.
        i_eff = jnp.minimum(i, n_sub - 1)
        frac_lo = i_eff / n_sub
        frac_hi = (i_eff + 1) / n_sub
        seg_lo = log_k_lo + total * frac_lo
        seg_hi = log_k_lo + total * frac_hi
        contrib = integrate_gl25_logk(integrand_fn, seg_lo, seg_hi)
        return acc + jnp.where(i < n_sub, contrib, 0.0)

    return jax.lax.fori_loop(0, n_loop, body, jnp.asarray(0.0, dtype=float))

# Backward-compatible alias (historical misnomer: uses GL25 nodes, not GK41).
integrate_gk41_logk = integrate_gl25_logk_multisub


def limber_cl_logk_batched(
    ell: jnp.ndarray,
    log_k_lo: jnp.ndarray,
    log_k_hi: jnp.ndarray,
    log_k_tab: jnp.ndarray,
    z_pk: jnp.ndarray,
    log_pk: jnp.ndarray,
    z_grid: jnp.ndarray,
    chi_grid: jnp.ndarray,
    tracer: jnp.ndarray,
    nc: jnp.ndarray,
) -> jnp.ndarray:
    """
    Batched Limber C_ell with log-k GK quadrature on chi-mapped weights.

    ``tracer`` and ``nc`` are sampled along ``chi_grid`` (not re-mapped via z).
    """
    from gpucosmo.jax.power import pk_ccl_spline_jax

    def one_ell(ell_val, lk_lo, lk_hi, tr, n_c):
        def integrand(k):
            chi = (ell_val + 0.5) / jnp.clip(k, 1.0e-30)
            z = jnp.interp(chi, chi_grid, z_grid, left=0.0, right=z_grid[-1])
            pk = pk_ccl_spline_jax(log_k_tab, z_pk, log_pk, k, z)
            tr_k = jnp.interp(chi, chi_grid, tr, left=0.0, right=0.0)
            return pk * tr_k

        return (
            2.0
            * integrate_gl25_logk_multisub(integrand, lk_lo, lk_hi)
            / (_C_LIGHT * (ell_val + 0.5))
        )

    if ell.ndim == 0:
        return one_ell(ell, log_k_lo, log_k_hi, tracer, nc)
    if tracer.ndim == 1:
        tracer = jnp.broadcast_to(tracer, (ell.size, tracer.size))
    if nc.ndim == 1:
        nc = jnp.broadcast_to(nc, (ell.size, nc.size))
    return jax.vmap(one_ell)(ell, log_k_lo, log_k_hi, tracer, nc)


def default_log_k_nodes(
    k_min: float | None = None,
    k_max: float | None = None,
) -> np.ndarray:
    return log_k_grid(k_min=k_min, k_max=k_max)


__all__ = [
    "build_log_k_bounds_batch",
    "default_log_k_nodes",
    "integrate_gk41_logk",
    "integrate_gk41_logx",
    "integrate_gl25_logk",
    "integrate_gl25_logk_multisub",
    "limber_cl_logk_batched",
    "limber_cl_qag_jax",
    "limber_cl_qag_numpy",
    "limber_k_bounds",
]
