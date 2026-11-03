#!/usr/bin/env python3
"""JAX-friendly spline utilities matching CCL GSL conventions."""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

# CCL default correlation / integration grids (ccl_core.c).
ELL_MIN_CORR = 0.01
ELL_MAX_CORR = 60000.0
N_ELL_CORR = 5000
K_MIN_DEFAULT = 5.0e-5
K_MAX_DEFAULT = 1000.0
DLOGK_INTEGRATION = 0.025
N_K_PER_DECADE = 167
INTEGRATION_LIMBER_EPSREL = 1.0e-4
INTEGRATION_LIMBER_MAX_SUBDIV = 12


def ccl_spline_params() -> dict[str, float | int]:
    """Return CCL spline defaults, optionally overridden by installed pyccl."""
    try:
        import pyccl as ccl

        sp = ccl.spline_params
        return {
            "K_MIN": float(sp.K_MIN),
            "K_MAX": float(sp.K_MAX),
            "DLOGK_INTEGRATION": float(sp.DLOGK_INTEGRATION),
            "N_K": int(sp.N_K),
            "ELL_MIN_CORR": float(sp.ELL_MIN_CORR),
            "ELL_MAX_CORR": float(sp.ELL_MAX_CORR),
            "N_ELL_CORR": int(sp.N_ELL_CORR),
        }
    except ImportError:
        return {
            "K_MIN": K_MIN_DEFAULT,
            "K_MAX": K_MAX_DEFAULT,
            "DLOGK_INTEGRATION": DLOGK_INTEGRATION,
            "N_K": N_K_PER_DECADE,
            "ELL_MIN_CORR": ELL_MIN_CORR,
            "ELL_MAX_CORR": ELL_MAX_CORR,
            "N_ELL_CORR": N_ELL_CORR,
        }


def log_k_grid(
    k_min: float | None = None,
    k_max: float | None = None,
    dlogk: float | None = None,
) -> np.ndarray:
    """Uniform grid in ln(k) with CCL ``DLOGK_INTEGRATION`` spacing."""
    params = ccl_spline_params()
    k_lo = float(k_min if k_min is not None else params["K_MIN"])
    k_hi = float(k_max if k_max is not None else params["K_MAX"])
    step = float(dlogk if dlogk is not None else params["DLOGK_INTEGRATION"])
    log_lo = np.log(k_lo)
    log_hi = np.log(k_hi)
    n = max(2, int(np.ceil((log_hi - log_lo) / step)) + 1)
    return np.linspace(log_lo, log_hi, n)


def ccl_log_spacing_numpy(
    xmin: float,
    xmax: float,
    n: int,
) -> np.ndarray:
    """Match CCL ``ccl_log_spacing`` (endpoint pinning on log grid)."""
    if n < 2:
        raise ValueError("ccl_log_spacing requires at least 2 points")
    if xmin <= 0.0 or xmax <= 0.0:
        raise ValueError("ccl_log_spacing endpoints must be positive")
    log_xmin = np.log(float(xmin))
    log_xmax = np.log(float(xmax))
    dlog = (log_xmax - log_xmin) / float(n - 1)
    x = np.exp(log_xmin + dlog * np.arange(n, dtype=float))
    x[0] = float(xmin)
    x[-1] = float(xmax)
    return x


def ccl_log_spacing_jax(
    xmin: float,
    xmax: float,
    n: int,
) -> jnp.ndarray:
    """JAX version of :func:`ccl_log_spacing_numpy`."""
    log_xmin = jnp.log(jnp.asarray(xmin, dtype=float))
    log_xmax = jnp.log(jnp.asarray(xmax, dtype=float))
    dlog = (log_xmax - log_xmin) / jnp.asarray(n - 1, dtype=float)
    idx = jnp.arange(n, dtype=float)
    x = jnp.exp(log_xmin + dlog * idx)
    x = x.at[0].set(xmin)
    x = x.at[n - 1].set(xmax)
    return x


def log_ell_corr_grid() -> np.ndarray:
    """Internal CCL FFTLog multipole grid (log-spaced)."""
    params = ccl_spline_params()
    return np.log(
        ccl_log_spacing_numpy(
            float(params["ELL_MIN_CORR"]),
            float(params["ELL_MAX_CORR"]),
            int(params["N_ELL_CORR"]),
        )
    )


def _logx_logy_extrap_hi_jax(
    x: jnp.ndarray,
    y: jnp.ndarray,
    x_eval: jnp.ndarray,
) -> jnp.ndarray:
    """CCL ``ccl_f1d_extrap_logx_logy`` above the spline range."""
    x_hi = x[-1]
    der = jnp.log(y[-1] / y[-2]) / jnp.where(
        jnp.abs(x[-1] - x[-2]) > 0.0,
        jnp.log(x[-1] / x[-2]),
        1.0,
    )
    return y[-1] * (x_eval / x_hi) ** der


def _logx_logy_extrap_hi_numpy(
    x: np.ndarray,
    y: np.ndarray,
    x_eval: np.ndarray,
) -> np.ndarray:
    x_hi = x[-1]
    der = np.log(y[-1] / y[-2]) / np.log(x[-1] / x[-2])
    return y[-1] * (x_eval / x_hi) ** der


def ccl_f1d_eval_jax(
    x: jnp.ndarray,
    y: jnp.ndarray,
    x_eval: jnp.ndarray,
    *,
    extrap_lo: str,
    extrap_hi: str,
    y0: float | jnp.ndarray,
    yf: float | jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate a CCL ``ccl_f1d`` Akima spline with CCL extrapolation rules."""
    x = jnp.asarray(x, dtype=float)
    y = jnp.asarray(y, dtype=float)
    x_eval = jnp.asarray(x_eval, dtype=float)
    x_lo = x[0]
    x_hi = x[-1]
    y_inner = akima_interp_jax(x, y, jnp.clip(x_eval, x_lo, x_hi))

    def _lo(val: jnp.ndarray) -> jnp.ndarray:
        if extrap_lo == "const":
            return jnp.broadcast_to(jnp.asarray(y0, dtype=float), val.shape)
        raise ValueError(f"unsupported extrap_lo={extrap_lo!r}")

    def _hi(val: jnp.ndarray) -> jnp.ndarray:
        if extrap_hi == "const":
            return jnp.broadcast_to(jnp.asarray(yf, dtype=float), val.shape)
        if extrap_hi == "logx_logy":
            return _logx_logy_extrap_hi_jax(x, y, val)
        raise ValueError(f"unsupported extrap_hi={extrap_hi!r}")

    return jnp.where(
        x_eval <= x_lo,
        _lo(x_eval),
        jnp.where(x_eval >= x_hi, _hi(x_eval), y_inner),
    )


def ccl_f1d_eval_numpy(
    x: np.ndarray,
    y: np.ndarray,
    x_eval: np.ndarray,
    *,
    extrap_lo: str,
    extrap_hi: str,
    y0: float,
    yf: float,
) -> np.ndarray:
    """NumPy reference for :func:`ccl_f1d_eval_jax` (GSL Akima via scipy)."""
    from scipy.interpolate import Akima1DInterpolator

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_eval = np.asarray(x_eval, dtype=float)
    x_lo = x[0]
    x_hi = x[-1]
    out = np.empty_like(x_eval, dtype=float)
    interp = Akima1DInterpolator(x, y)
    for i, xv in enumerate(x_eval):
        if xv <= x_lo:
            if extrap_lo == "const":
                out[i] = y0
            else:
                raise ValueError(f"unsupported extrap_lo={extrap_lo!r}")
        elif xv >= x_hi:
            if extrap_hi == "const":
                out[i] = yf
            elif extrap_hi == "logx_logy":
                out[i] = _logx_logy_extrap_hi_numpy(x, y, xv)
            else:
                raise ValueError(f"unsupported extrap_hi={extrap_hi!r}")
        else:
            out[i] = float(interp(xv))
    return out


def resample_cell_to_fftlog_grid_numpy(
    ell: np.ndarray,
    cell: np.ndarray,
    ell_out: np.ndarray,
) -> np.ndarray:
    """Resample C_ell onto the FFTLog grid (CCL const-lo / logx_logy-hi)."""
    cell = np.asarray(cell, dtype=float)
    return ccl_f1d_eval_numpy(
        np.asarray(ell, dtype=float),
        cell,
        np.asarray(ell_out, dtype=float),
        extrap_lo="const",
        extrap_hi="logx_logy",
        y0=float(cell[0]),
        yf=0.0,
    )


def resample_logx_logy_numpy(
    x_in: np.ndarray,
    y_in: np.ndarray,
    x_out: np.ndarray,
) -> np.ndarray:
    """CCL ``logx_logy`` 1D resampling via pyccl when available."""
    x_in = np.asarray(x_in, dtype=float)
    y_in = np.asarray(y_in, dtype=float)
    x_out = np.asarray(x_out, dtype=float)
    try:
        from pyccl.pyutils import resample_array

        return np.asarray(
            resample_array(
                x_in,
                y_in,
                x_out,
                extrap_lo="logx_logy",
                extrap_hi="logx_logy",
            ),
            dtype=float,
        )
    except ImportError:
        from scipy.interpolate import Akima1DInterpolator

        log_x = np.log(np.maximum(x_in, 1.0e-30))
        log_y = np.log(np.maximum(y_in, 1.0e-30))
        interp = Akima1DInterpolator(log_x, log_y, extrapolate=True)
        return np.exp(interp(np.log(np.maximum(x_out, 1.0e-30))))


def akima_interp_jax(
    x: jnp.ndarray,
    y: jnp.ndarray,
    x_eval: jnp.ndarray,
) -> jnp.ndarray:
    """Piecewise cubic Hermite (Akima-style) interpolation in JAX."""
    x = jnp.asarray(x, dtype=float)
    y = jnp.asarray(y, dtype=float)
    x_eval = jnp.asarray(x_eval, dtype=float)
    n = x.shape[0]
    dx = jnp.diff(x)
    dy = jnp.diff(y)
    m = dy / jnp.where(jnp.abs(dx) > 0.0, dx, 1.0)
    mk = jnp.concatenate([m[:2], m, m[-2:]])

    def akima_tangent(i: jnp.ndarray) -> jnp.ndarray:
        m0, m1, m2, m3 = mk[i], mk[i + 1], mk[i + 2], mk[i + 3]
        d0 = jnp.abs(m2 - m1)
        d1 = jnp.abs(m3 - m0)
        w = d0 + d1
        return jnp.where(w > 0.0, (d0 * m1 + d1 * m2) / w, 0.5 * (m1 + m2))

    t = jax.vmap(akima_tangent)(jnp.arange(n))

    idx = jnp.clip(jnp.searchsorted(x, x_eval, side="right") - 1, 0, n - 2)
    x0 = x[idx]
    x1 = x[idx + 1]
    y0 = y[idx]
    y1 = y[idx + 1]
    t0 = t[idx]
    t1 = t[idx + 1]
    h = x1 - x0
    s = (x_eval - x0) / jnp.where(jnp.abs(h) > 0.0, h, 1.0)
    h00 = (1.0 + 2.0 * s) * (1.0 - s) ** 2
    h10 = s * (1.0 - s) ** 2
    h01 = s**2 * (3.0 - 2.0 * s)
    h11 = s**2 * (s - 1.0)
    return h00 * y0 + h10 * h * t0 + h01 * y1 + h11 * h * t1


def resample_cell_to_fftlog_jax(
    x_in: jnp.ndarray,
    y_in: jnp.ndarray,
    x_out: jnp.ndarray,
) -> jnp.ndarray:
    """Resample C_ell onto the FFTLog grid (CCL const-lo / logx_logy-hi)."""
    y_in = jnp.asarray(y_in, dtype=float)
    return ccl_f1d_eval_jax(
        jnp.asarray(x_in, dtype=float),
        y_in,
        jnp.asarray(x_out, dtype=float),
        extrap_lo="const",
        extrap_hi="logx_logy",
        y0=y_in[0],
        yf=0.0,
    )


def resample_logx_logy_jax(
    x_in: jnp.ndarray,
    y_in: jnp.ndarray,
    x_out: jnp.ndarray,
) -> jnp.ndarray:
    """CCL-like resample: const extrap below, logx_logy above, Akima inside."""
    x_in = jnp.asarray(x_in, dtype=float)
    y_in = jnp.asarray(y_in, dtype=float)
    x_out = jnp.asarray(x_out, dtype=float)
    x_pos = jnp.maximum(x_in, 1.0e-30)
    y_pos = jnp.maximum(y_in, 1.0e-30)
    log_x = jnp.log(x_pos)
    log_y = jnp.log(y_pos)
    log_x_out = jnp.log(jnp.maximum(x_out, 1.0e-30))

    # logx_logy extrapolation above the input range (CCL f1d_extrap_logx_logy).
    log_x_hi = log_x[-1]
    log_y_hi = log_y[-1]
    slope_hi = (log_y[-1] - log_y[-2]) / jnp.where(
        jnp.abs(log_x[-1] - log_x[-2]) > 0.0,
        log_x[-1] - log_x[-2],
        1.0,
    )
    log_y_hi_extrap = log_y_hi + slope_hi * (log_x_out - log_x_hi)

    y_inner = akima_interp_jax(log_x, log_y, log_x_out)
    y_inner = jnp.exp(y_inner)
    y_hi = jnp.exp(log_y_hi_extrap)
    y_lo = y_pos[0]
    return jnp.where(
        x_out < x_pos[0],
        y_lo,
        jnp.where(x_out > x_pos[-1], y_hi, y_inner),
    )


def akima_interp_logx_logy_jax(
    x: jnp.ndarray,
    y: jnp.ndarray,
    x_eval: jnp.ndarray,
) -> jnp.ndarray:
    """Akima interpolation with ``logx_logy`` coordinates."""
    x_pos = jnp.maximum(x, 1.0e-30)
    y_pos = jnp.maximum(y, 1.0e-30)
    xq = jnp.maximum(x_eval, 1.0e-30)
    return jnp.exp(
        akima_interp_jax(
            jnp.log(x_pos),
            jnp.log(y_pos),
            jnp.log(xq),
        )
    )


def _extrap_logpk_lo_jax(
    log_k: jnp.ndarray,
    log_pk_row: jnp.ndarray,
    lk_eval: jnp.ndarray,
) -> jnp.ndarray:
    """CCL Pk2D extrap_order_lok=1: linear in log P vs log k below range."""
    slope = (log_pk_row[1] - log_pk_row[0]) / jnp.where(
        jnp.abs(log_k[1] - log_k[0]) > 0.0,
        log_k[1] - log_k[0],
        1.0,
    )
    return log_pk_row[0] + slope * (lk_eval - log_k[0])


def _extrap_logpk_hi_jax(
    log_k: jnp.ndarray,
    log_pk_row: jnp.ndarray,
    lk_eval: jnp.ndarray,
) -> jnp.ndarray:
    """CCL Pk2D extrap_order_hik=2: quadratic in log P vs log k above range."""
    lk_hi = log_k[-1]
    d1 = log_k[-1] - log_k[-2]
    d2 = log_k[-2] - log_k[-3]
    p1 = log_pk_row[-1]
    p2 = log_pk_row[-2]
    p3 = log_pk_row[-3]
    der1 = (p1 - p2) / jnp.where(jnp.abs(d1) > 0.0, d1, 1.0)
    der2 = (p2 - p3) / jnp.where(jnp.abs(d2) > 0.0, d2, 1.0)
    der = 0.5 * (der1 + der2)
    dlk = lk_eval - lk_hi
    return p1 + der * dlk + 0.5 * (der - der1) / jnp.where(jnp.abs(d1) > 0.0, d1, 1.0) * dlk**2


def _interp_logpk_along_k_jax(
    log_k: jnp.ndarray,
    log_pk_row: jnp.ndarray,
    lk_eval: jnp.ndarray,
) -> jnp.ndarray:
    """Interpolate log P along log k with CCL low/high extrapolation orders."""
    lk_lo = log_k[0]
    lk_hi = log_k[-1]
    inner = akima_interp_jax(log_k, log_pk_row, jnp.clip(lk_eval, lk_lo, lk_hi))
    lo = _extrap_logpk_lo_jax(log_k, log_pk_row, lk_eval)
    hi = _extrap_logpk_hi_jax(log_k, log_pk_row, lk_eval)
    return jnp.where(lk_eval < lk_lo, lo, jnp.where(lk_eval > lk_hi, hi, inner))


def pk2d_eval_jax(
    k: jnp.ndarray,
    a: jnp.ndarray,
    log_k: jnp.ndarray,
    a_arr: jnp.ndarray,
    log_pk: jnp.ndarray,
) -> jnp.ndarray:
    """
    Evaluate P(k, a) with tensor-product Akima splines (CCL Pk2D analogue).

    ``log_pk`` has shape ``(n_a, n_k)`` with ``a_arr`` ascending.
    Supports scalar or broadcast-compatible ``k`` and ``a``.
    """
    k_b = jnp.asarray(k, dtype=float)
    a_b = jnp.broadcast_to(jnp.asarray(a, dtype=float), k_b.shape)
    log_k = jnp.asarray(log_k, dtype=float)
    a_arr = jnp.asarray(a_arr, dtype=float)
    log_pk = jnp.asarray(log_pk, dtype=float)

    def one(k_one: jnp.ndarray, a_one: jnp.ndarray) -> jnp.ndarray:
        lk = jnp.log(jnp.maximum(k_one, 1.0e-30))
        vals_k = jax.vmap(lambda row: _interp_logpk_along_k_jax(log_k, row, lk))(
            log_pk
        )
        a_lo = a_arr[0]
        a_hi = a_arr[-1]
        inner = akima_interp_jax(a_arr, vals_k, jnp.clip(a_one, a_lo, a_hi))
        slope_lo = (vals_k[1] - vals_k[0]) / jnp.where(
            jnp.abs(a_arr[1] - a_arr[0]) > 0.0,
            a_arr[1] - a_arr[0],
            1.0,
        )
        slope_hi = (vals_k[-1] - vals_k[-2]) / jnp.where(
            jnp.abs(a_arr[-1] - a_arr[-2]) > 0.0,
            a_arr[-1] - a_arr[-2],
            1.0,
        )
        lo = vals_k[0] + slope_lo * (a_one - a_lo)
        hi = vals_k[-1] + slope_hi * (a_one - a_hi)
        log_p = jnp.where(a_one < a_lo, lo, jnp.where(a_one > a_hi, hi, inner))
        return jnp.exp(log_p)

    flat = jax.vmap(one)(k_b.reshape(-1), a_b.reshape(-1))
    return flat.reshape(k_b.shape)


def pk2d_eval_from_z_jax(
    k: jnp.ndarray,
    z: jnp.ndarray,
    log_k: jnp.ndarray,
    z_tab: jnp.ndarray,
    log_pk: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate P(k, z) using CCL-style Pk2D interpolation (z_tab ascending)."""
    a_tab = 1.0 / (1.0 + jnp.asarray(z_tab, dtype=float))
    log_pk_a = jnp.asarray(log_pk, dtype=float)[::-1, :]
    a_arr = a_tab[::-1]
    a_eval = 1.0 / (1.0 + jnp.asarray(z, dtype=float))
    return pk2d_eval_jax(k, a_eval, log_k, a_arr, log_pk_a)


def resample_pk_to_ccl_dense_jax(
    log_k_src: jnp.ndarray,
    log_pk_src: jnp.ndarray,
    log_k_dst: jnp.ndarray,
) -> jnp.ndarray:
    """Upsample log P(k) rows onto a denser log-k grid (Akima logx_logy per z-slice)."""
    k_src = jnp.exp(log_k_src)
    k_dst = jnp.exp(log_k_dst)

    def one_row(row: jnp.ndarray) -> jnp.ndarray:
        return jnp.log(
            akima_interp_logx_logy_jax(k_src, jnp.exp(row), k_dst)
        )

    return jax.vmap(one_row)(log_pk_src)


__all__ = [
    "DLOGK_INTEGRATION",
    "ELL_MAX_CORR",
    "ELL_MIN_CORR",
    "INTEGRATION_LIMBER_EPSREL",
    "INTEGRATION_LIMBER_MAX_SUBDIV",
    "K_MAX_DEFAULT",
    "K_MIN_DEFAULT",
    "N_ELL_CORR",
    "akima_interp_jax",
    "pk2d_eval_from_z_jax",
    "pk2d_eval_jax",
    "resample_pk_to_ccl_dense_jax",
    "ccl_f1d_eval_jax",
    "ccl_f1d_eval_numpy",
    "ccl_log_spacing_jax",
    "ccl_log_spacing_numpy",
    "ccl_spline_params",
    "log_ell_corr_grid",
    "log_k_grid",
    "resample_cell_to_fftlog_grid_numpy",
    "resample_cell_to_fftlog_jax",
    "resample_logx_logy_jax",
]
