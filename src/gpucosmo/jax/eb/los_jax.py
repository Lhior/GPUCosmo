#!/usr/bin/env python3
# This file is part of GPUCosmo. Copyright (C) 2025-2026 Leonel Medina-Varela.
# SPDX-License-Identifier: GPL-3.0-or-later
# Derived in part from the following, translated to JAX and modified for GPUCosmo:
# - CAMB (https://github.com/cmbant/CAMB), Copyright 2000-2025 Antony Lewis and Anthony Challinor.
#   Licence: LICENSES/CAMB.txt.
# See NOTICE.md.
"""Line-of-sight / Limber projection of CMB sources → Δ_ℓ(k).

Integrals follow CAMB ``cmbmain.f90`` / forutils ``TRanges_Getdpoints``
(not trapezoidal quadrature):

- LOS (flat): ``DoFlatIntegration`` — ``Δ_ℓ(q) = Σ_n S(τ_n) j_ℓ(q χ_n) dτ_n``
  with ``dτ_n = TimeSteps%dpoints(n)`` (``half_ends`` default true).
- Cls: ``Σ_q P(k) Δ_ℓ(q)² dlnk`` with ``dlnk = q%dpoints/q`` and
  ``Getdpoints(half_ends=.false.)`` on the transfer-q grid.
- High-ℓ Limber: CAMB ``UseLimber`` path — evaluate the source at χ_*=ν/k.

Spherical Bessel on the Cls hot path uses CAMB-style cubic Horner tables
(host knot build once; device O(1) eval). Miller recurrence remains for
single-ℓ helpers / tests.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def camb_dpoints(x: jnp.ndarray, *, half_ends: bool = True) -> jnp.ndarray:
    """CAMB ``TRanges_Getdpoints`` (forutils ``RangeUtils.f90``)."""
    x = jnp.asarray(x, dtype=jnp.float64)
    d = jnp.zeros_like(x)
    d = d.at[1:-1].set(0.5 * (x[2:] - x[:-2]))
    edge0 = x[1] - x[0]
    edge1 = x[-1] - x[-2]
    if half_ends:
        d = d.at[0].set(0.5 * edge0)
        d = d.at[-1].set(0.5 * edge1)
    else:
        d = d.at[0].set(edge0)
        d = d.at[-1].set(edge1)
    return d


def camb_dlnk(k: jnp.ndarray) -> jnp.ndarray:
    """CAMB Cls measure ``dlnk = q%dpoints / q`` (flat; ``cmbmain.f90``)."""
    k = jnp.maximum(jnp.asarray(k, dtype=jnp.float64), 1e-30)
    return camb_dpoints(k, half_ends=False) / k


# Peak VRAM bound for Bessel under vmap(k): process k in chunks.
# Horner O(1) eval allows larger chunks than Miller (was 64).
LOS_K_CHUNK = 256
# ℓ batch for fused Horner LOS (keeps jl[ℓ,k,τ] peak small on 8GB).
LOS_ELL_CHUNK = 32
# Precision of j_ℓ and the τ quadrature in the fused TT/EE/TE LOS (k sums stay
# float64). float32 changes C_ℓ^{TT,EE,TE} by ≤ 8e-6 (median 1e-8) at lmax 2508
# vs float64, and is ~14% faster end-to-end on a consumer GPU (1/64-rate f64).
# "auto" = float32 on accelerators, float64 on CPU: XLA:CPU (jax 0.10) segfaults
# on the float32 scan at the default 256×32 chunking, and gains nothing there.
LOS_COMPUTE_DTYPE = "auto"
# Loop order of the fused TT/EE/TE LOS scan. "k_outer": each k chunk finds the
# Bessel-table interval of every x = kχ once and gathers the four Horner
# coefficients of an (ℓ, interval) as one contiguous row of an [ℓ, interval, 4]
# table; the per-ℓ k sums accumulate in the same order as "ell_outer" (the
# original), which redoes the interval search for every ℓ chunk.
LOS_LOOP_ORDER = os.environ.get("GPUCOSMO_LOS_LOOP_ORDER", "k_outer")


def _resolve_los_dtype(dtype):
    if isinstance(dtype, str) and dtype == "auto":
        return jnp.float64 if jax.default_backend() == "cpu" else jnp.float32
    return dtype
# Deprecated: host SciPy path retired; Horner tables used for densified Int-k.
LOS_HOST_BESSEL_K_MIN = 10**9
# CAMB bjl_pre_peak_start_factor — zero j_ℓ for x ≪ turning point
_BJL_PRE_PEAK = 4.2

# Host-built Horner tables keyed by (ells_tuple, xmax_bin, boost)
_BESSEL_TABLE_CACHE: dict[tuple, dict] = {}


def _bessel_x_knots(xmax: float, *, bessel_boost: float = 1.0) -> np.ndarray:
    """CAMB ``BessRanges`` knot grid in ``x = k χ`` (AccuracyBoost=1)."""
    from gpucosmo.jax.eb.tranges_jax import TRanges

    boost = float(max(bessel_boost, 1e-3))
    acc = boost  # AccuracyBoost * BesselBoost with AccuracyBoost=1
    xmax = float(max(xmax, 25.0))
    r = TRanges()
    r.init()
    r.add_delta(0.0, 1.0, 0.01 / boost, is_log=False)
    r.add_delta(1.0, 5.0, 0.1 / boost, is_log=False)
    r.add_delta(5.0, 25.0, 0.2 / boost, is_log=False)
    r.add_delta(25.0, min(150.0, xmax), 0.5 / acc, is_log=False)
    if xmax > 150.0:
        r.add_delta(150.0, xmax, 0.7 / acc, is_log=False)
    pts = r.get_array(want_dpoints=False)
    if pts.size < 4:
        pts = np.linspace(0.0, xmax, 64)
    return np.asarray(pts, dtype=np.float64)


def build_bessel_horner_table(
    ells: list[int] | tuple[int, ...] | np.ndarray,
    xmax: float,
    *,
    bessel_boost: float = 1.0,
) -> dict:
    """Build CAMB-style cubic spline coeffs for ``j_ℓ(x)`` on a shared x-grid.

    Knot values from SciPy (host, once). Device eval uses Horner in
    ``(x - x_i)`` with SciPy ``CubicSpline`` packed coeffs.
    """
    from scipy.interpolate import CubicSpline
    from scipy.special import spherical_jn

    ells_t = tuple(int(e) for e in ells)
    xmax = float(max(xmax, 25.0))
    xmax_bin = float(np.ceil(xmax / 50.0) * 50.0)
    key = (ells_t, xmax_bin, float(bessel_boost))
    hit = _BESSEL_TABLE_CACHE.get(key)
    if hit is not None:
        return hit

    def build():
        knots = _bessel_x_knots(xmax_bin, bessel_boost=bessel_boost)
        n_int = int(knots.size - 1)
        n_ell = len(ells_t)
        # coeffs[m, interval, ell_ix]: S = ((c0*dx+c1)*dx+c2)*dx+c3
        coeffs = np.zeros((4, n_int, n_ell), dtype=np.float64)
        xlims = np.zeros(n_ell, dtype=np.float64)

        for j, ell in enumerate(ells_t):
            ell_f = float(ell)
            xlim = max(0.0, ell_f - _BJL_PRE_PEAK * ell_f ** (1.0 / 3.0))
            xlims[j] = xlim
            vals = spherical_jn(ell, knots)
            vals = np.where(knots < xlim, 0.0, vals)
            vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
            cs = CubicSpline(knots, vals, bc_type="natural", extrapolate=False)
            # cs.c shape (4, n_int)
            coeffs[:, :, j] = cs.c
        return {"knots": knots, "coeffs": coeffs, "xlims": xlims}

    from gpucosmo.jax.host_table_cache import cached_npz

    # ~12 s of SciPy for the ~450-ℓ CMB table: kept on disk across processes.
    built = cached_npz("bessel_horner", key, build, module=__name__)
    knots, coeffs, xlims = built["knots"], built["coeffs"], built["xlims"]

    # Host (NumPy) arrays: they enter the compiled programs as constants either way,
    # and cached device copies would stay resident on the GPU for the whole run.
    # ``coeffs_t`` / ``coeffs32_t``: [ℓ, interval, 4] rows for the k-outer LOS scan.
    coeffs_t = np.ascontiguousarray(np.transpose(coeffs, (2, 1, 0)))
    table = {
        "ells": ells_t,
        "knots": knots,
        "coeffs": coeffs,
        "coeffs32": coeffs.astype(np.float32),
        "coeffs_t": coeffs_t,
        "coeffs32_t": coeffs_t.astype(np.float32),
        "xlims": xlims,
        "xmax": xmax_bin,
    }
    _BESSEL_TABLE_CACHE[key] = table
    return table


def eval_bessel_horner(
    table: dict, ell_index: int | jnp.ndarray, x: jnp.ndarray, *, dtype=jnp.float64
) -> jnp.ndarray:
    """Evaluate packed cubic spline ``j_ℓ(x)`` (device).

    The interval search and ``dx`` are always float64 (``x = kχ`` reaches ~10⁴);
    ``dtype=float32`` runs the coefficient gathers and Horner cubic in single
    precision, which is where consumer GPUs have their throughput.
    """
    knots = jnp.asarray(table["knots"], dtype=jnp.float64)
    use32 = jnp.dtype(dtype) == jnp.float32 and "coeffs32" in table
    coeffs = jnp.asarray(table["coeffs32"] if use32 else table["coeffs"])
    xlims = jnp.asarray(table["xlims"], dtype=jnp.float64)
    x = jnp.asarray(x, dtype=jnp.float64)
    x_clip = jnp.clip(x, knots[0], knots[-1])
    i = jnp.searchsorted(knots, x_clip, side="right") - 1
    i = jnp.clip(i, 0, knots.shape[0] - 2)
    x0 = knots[i]
    dx = (x_clip - x0).astype(coeffs.dtype)
    ell_index = jnp.asarray(ell_index, dtype=jnp.int32)
    c0 = coeffs[0, i, ell_index]
    c1 = coeffs[1, i, ell_index]
    c2 = coeffs[2, i, ell_index]
    c3 = coeffs[3, i, ell_index]
    jl = ((c0 * dx + c1) * dx + c2) * dx + c3
    jl = jnp.where(x < xlims[ell_index], 0.0, jl)
    jl = jnp.where(jnp.isfinite(jl), jl, 0.0)
    return jl


def eval_bessel_horner_many(
    table: dict, ell_indices: jnp.ndarray, x: jnp.ndarray, *, dtype=jnp.float64
) -> jnp.ndarray:
    """``j_ℓ(x)`` for many ℓ: returns shape ``(n_ell, *x.shape)``."""
    ell_indices = jnp.asarray(ell_indices, dtype=jnp.int32)
    return jax.vmap(lambda j: eval_bessel_horner(table, j, x, dtype=dtype))(ell_indices)


def bessel_table_vs_scipy_rel_err(
    table: dict, *, n_samp: int = 512, floor: float = 1e-8
) -> dict:
    """Peak-normalized max |Δ| gate vs SciPy (host). Target ≲1e-4."""
    from scipy.special import spherical_jn

    knots = np.asarray(table["knots"])
    xs = np.linspace(float(knots[1]), float(knots[-1]) * 0.99, n_samp)
    out = {}
    for j, ell in enumerate(table["ells"]):
        jl_t = np.asarray(eval_bessel_horner(table, j, jnp.asarray(xs)))
        jl_s = spherical_jn(int(ell), xs)
        peak = max(float(np.max(np.abs(jl_s))), floor)
        near = np.abs(jl_s) > 0.01 * peak
        rel = np.abs(jl_t - jl_s) / peak
        out[int(ell)] = {
            "max_peak_norm": float(np.max(rel)),
            "p95_near_peak": float(np.percentile(rel[near], 95)) if np.any(near) else None,
        }
    return out


def spherical_bessel_jn(n: int, x: jnp.ndarray) -> jnp.ndarray:
    """On-device j_n(x) for a Python-static multipole ``n``.

    - ``n < x``: upward recurrence from j0, j1
    - ``n > x``: Miller downward (normalized to j0/j1), rescaled as it grows so it cannot
      overflow, with a hard zero when ``n > x + margin`` (exponentially small)
    - small ``x`` (< 0.5): the power series, summed to double precision

    Scans are carry-only (no stacked ``ys`` of length O(ℓ)), so peak memory
    stays O(shape(x)) rather than O(ℓ · shape(x)).

    Diagnostics only: the production line of sight uses the Horner tables
    (``build_bessel_horner_table`` / ``eval_bessel_horner``). Before issue #10 was fixed this
    returned 0 around the peak for n ≳ 2200 (the unscaled Miller recurrence overflowed) and
    kept one series term for x < 0.5 (1e-4 of the peak off at n = 3);
    ``tests/test_spherical_bessel_device.py`` checks n to 3500 against SciPy.
    """
    n = int(n)
    x = jnp.maximum(jnp.asarray(x, dtype=jnp.float64), 1e-12)
    j0 = jnp.sin(x) / x
    if n <= 0:
        return j0
    j1 = jnp.sin(x) / (x * x) - jnp.cos(x) / x
    if n == 1:
        return j1

    n_f = float(n)
    # Deep into the forbidden region j_n is exponentially tiny
    margin = max(25.0, 0.25 * n_f)

    # Small x: j_n = x^n/(2n+1)!! Σ_k (−x²/2)^k / (k! (2n+3)(2n+5)…(2n+2k+1)). The leading factor
    # is built in log space (no under/overflow for any n); at x < 0.5 the sum has converged to
    # double precision after 12 terms (ratio ≤ x²/(2k(2n+2k+1)) ≤ 0.04/k).
    from scipy.special import gammaln

    log_dfact = float(gammaln(2 * n + 2) - n * np.log(2.0) - gammaln(n + 1))  # ln (2n+1)!!
    xs = jnp.minimum(x, 0.5)
    lead = jnp.exp(n * jnp.log(xs) - log_dfact)
    term, tot = jnp.ones_like(x), jnp.ones_like(x)
    for kk in range(1, 13):
        term = term * (-0.5 * xs * xs) / (kk * (2.0 * n + 2.0 * kk + 1.0))
        tot = tot + term
    j_series = lead * tot

    def up_step(carry, m):
        jm1, jm = carry
        jp = ((2.0 * m + 1.0) / x) * jm - jm1
        return (jm, jp), None

    (_, j_up), _ = jax.lax.scan(
        up_step, (j0, j1), jnp.arange(1, n, dtype=jnp.float64)
    )

    n_extra = max(40, n // 2 + 20)
    n_start = n + n_extra
    eps = 1.0e-30
    # Downward from n_start → 1: stash j_n, j_1, j_0 without materializing all j_m.
    # m runs n_start, n_start-1, ..., 1; after step at m we have j_lo = j_{m-1}.
    n_f64 = jnp.asarray(float(n), dtype=jnp.float64)

    def down_step(carry, m):
        j_hi, j_cur, jn_s, j1_s, j0_s = carry
        j_lo = ((2.0 * m + 1.0) / x) * j_cur - j_hi
        # At step m: j_cur = j_m, j_lo = j_{m-1}
        jn_s = jnp.where(m == n_f64, j_cur, jn_s)
        j1_s = jnp.where(m == 1.0, j_cur, j1_s)
        j0_s = jnp.where(m == 1.0, j_lo, j0_s)
        # Through the forbidden region (m > x) the sequence grows like exp(∫ arccosh(m/x) dm),
        # past 1e308 for n ≳ 2200: rescale everything kept so far (only ratios are used).
        big = jnp.abs(j_lo) > 1e150
        sc = jnp.where(big, 1e-150, 1.0)
        return (j_cur * sc, j_lo * sc, jn_s * sc, j1_s * sc, j0_s * sc), None

    z = jnp.zeros_like(x)
    init = (z, jnp.full_like(x, eps), z, z, z)
    ms = jnp.arange(n_start, 0, -1, dtype=jnp.float64)
    (_, _, jn_d, j1_d, j0_d), _ = jax.lax.scan(down_step, init, ms)
    use_j0 = jnp.abs(j0) > 1e-14
    scale = jnp.where(
        use_j0,
        j0 / jnp.where(jnp.abs(j0_d) > 1e-300, j0_d, 1.0),
        j1 / jnp.where(jnp.abs(j1_d) > 1e-300, j1_d, 1.0),
    )
    j_mill = jn_d * scale

    up_ok = jnp.isfinite(j_up) & (jnp.abs(j_up) <= 1.0)
    mill_ok = jnp.isfinite(j_mill) & (jnp.abs(j_mill) <= 1.0)
    # Never use upward below the turning point. A failed Miller value is NaN (not 0): this is
    # a diagnostic, a failure must show.
    j_below = jnp.where(mill_ok, j_mill, jnp.nan)
    j_above = jnp.where(up_ok, j_up, j_below)
    j_rec = jnp.where(n_f < x, j_above, j_below)
    j = jnp.where(x < 0.5, j_series, j_rec)
    j = jnp.where(n_f > x + margin, 0.0, j)
    return jnp.clip(j, -1.0, 1.0)


def spherical_bessel_jl_scan(ell_max: int, x: jnp.ndarray) -> jnp.ndarray:
    """On-device j_ℓ(x) for ℓ=0..ell_max via per-ℓ ``spherical_bessel_jn``.

    Prefer ``spherical_bessel_jn`` on the LOS hot path; this table helper is
    for tests / callers that need the full stack.
    """
    ell_max = int(ell_max)
    x = jnp.asarray(x, dtype=jnp.float64)
    rows = [spherical_bessel_jn(ell, x) for ell in range(ell_max + 1)]
    return jnp.stack(rows, axis=0)



def los_delta_ell(
    source: jnp.ndarray,
    tau: jnp.ndarray,
    tau0: jnp.ndarray,
    k: jnp.ndarray,
    ell: int,
    *,
    bessel_lmax: int | None = None,
) -> jnp.ndarray:
    """CAMB flat LOS: ``Δ_ℓ(k) = Σ_n S_n j_ℓ(k(τ0−τ_n)) dτ_n``."""
    del bessel_lmax
    return _los_delta_n(source, tau, tau0, k, int(ell))


def limber_project_source(
    source: jnp.ndarray,
    tau: jnp.ndarray,
    tau0: jnp.ndarray,
    k: jnp.ndarray,
    ell: jnp.ndarray,
) -> jnp.ndarray:
    """CAMB Limber: ``Δ_ℓ ≈ S(k,τ_*)/k * √(π/(2ν))`` with ν=ℓ+½."""
    nu = ell.astype(jnp.float64) + 0.5
    chi_star = nu / jnp.maximum(k, 1e-30)
    tau_star = tau0 - chi_star
    s_star = jnp.interp(tau_star, tau, source, left=0.0, right=0.0)
    return s_star / jnp.maximum(k, 1e-30) * jnp.sqrt(jnp.pi / (2.0 * nu))


def cl_from_delta_l_k(
    delta_l_k: jnp.ndarray,
    p_r: jnp.ndarray,
    dlnk: jnp.ndarray,
) -> jnp.ndarray:
    """CAMB Cls assemble: ``C_ℓ = 4π Σ_q P(k) Δ_ℓ(q)² dlnk``."""
    return 4.0 * jnp.pi * jnp.sum(p_r * delta_l_k**2 * dlnk, axis=-1)


def limber_cl_from_sources(
    sources_k: jnp.ndarray,
    tau: jnp.ndarray,
    tau0: jnp.ndarray,
    k: jnp.ndarray,
    p_r: jnp.ndarray,
    ell: jnp.ndarray,
) -> jnp.ndarray:
    """C_ℓ via Limber Δ_ℓ(k) and CAMB ``Σ P Δ² dlnk``."""
    dlnk = camb_dlnk(k)

    def one_ell(l):
        def one_k(i):
            return limber_project_source(sources_k[i], tau, tau0, k[i], l[None])[0]

        delta = jax.vmap(one_k)(jnp.arange(k.shape[0]))
        return cl_from_delta_l_k(delta, p_r, dlnk)

    return jax.vmap(one_ell)(ell)


def limber_cl_te_from_sources(
    sources_t: jnp.ndarray,
    sources_e: jnp.ndarray,
    tau: jnp.ndarray,
    tau0: jnp.ndarray,
    k: jnp.ndarray,
    p_r: jnp.ndarray,
    ell: jnp.ndarray,
) -> jnp.ndarray:
    """C_ℓ^{TE} = 4π Σ P(k) Δ_T Δ_E dlnk."""
    dlnk = camb_dlnk(k)

    def one_ell(l):
        def one_k(i):
            dt = limber_project_source(sources_t[i], tau, tau0, k[i], l[None])[0]
            de = limber_project_source(sources_e[i], tau, tau0, k[i], l[None])[0]
            return p_r[i] * dt * de * dlnk[i]

        return 4.0 * jnp.pi * jnp.sum(jax.vmap(one_k)(jnp.arange(k.shape[0])))

    return jax.vmap(one_ell)(ell)


def _los_delta_n(
    source,
    tau,
    tau0,
    k,
    ell: int,
    *,
    n_los: int = 1536,
    use_native_grid: bool = True,
):
    """LOS Δ_ℓ for a Python-static multipole ``ell``.

    Default (CAMB-like): integrate on the native source ``TimeSteps`` τ grid
    with ``camb_dpoints``. Optional dual-window χ resample is kept as fallback
    when ``use_native_grid=False``.
    """
    tau = jnp.asarray(tau, dtype=jnp.float64)
    source = jnp.asarray(source, dtype=jnp.float64)
    t0 = jnp.asarray(tau0, dtype=jnp.float64)
    kk = jnp.asarray(k, dtype=jnp.float64)

    if use_native_grid:
        dtau = camb_dpoints(tau, half_ends=True)
        chi = jnp.maximum(t0 - tau, 0.0)
        jl = spherical_bessel_jn(int(ell), kk * chi)
        return jnp.sum(source * jl * dtau)

    # Fallback: dual-window χ sampling (vis + Limber) + interp of S
    del n_los  # kept for API compat when callers pass it
    t_lo = tau[0]
    t_hi = jnp.minimum(tau[-1], t0)
    chi_max = jnp.maximum(t0 - t_lo, 1.0)
    chi_min = jnp.maximum(t0 - t_hi, 0.0)
    n_tau = source.shape[0]
    n_early = jnp.maximum((3 * n_tau) // 5, 8)
    ip = jnp.argmax(jnp.abs(source[:n_early]))
    chi_vis = jnp.clip(t0 - tau[ip], chi_min, chi_max)
    chi_lim = jnp.clip((float(ell) + 0.5) / jnp.maximum(kk, 1e-30), chi_min, chi_max)
    n_samp = 1536
    n_a = n_samp // 3
    n_b = n_samp // 3
    n_c = n_samp - n_a - n_b
    half = jnp.maximum(0.04 * chi_max, 80.0)
    chi_a = jnp.linspace(
        jnp.maximum(chi_min, chi_vis - half),
        jnp.minimum(chi_max, chi_vis + half),
        n_a,
    )
    chi_b = jnp.linspace(
        jnp.maximum(chi_min, chi_lim - half),
        jnp.minimum(chi_max, chi_lim + half),
        n_b,
    )
    chi_c = jnp.linspace(chi_min, chi_max, n_c)
    chi = jnp.sort(jnp.concatenate([chi_a, chi_b, chi_c]))
    chi = jnp.maximum(chi, jnp.concatenate([chi[:1], chi[:-1] + 1e-10 * chi_max]))
    chi = jnp.clip(chi, chi_min, chi_max)
    tau_asc = (t0 - chi)[::-1]
    chi_asc = t0 - tau_asc
    s_asc = jnp.interp(tau_asc, tau, source, left=0.0, right=0.0)
    dtau = camb_dpoints(tau_asc, half_ends=True)
    jl = spherical_bessel_jn(int(ell), kk * jnp.maximum(chi_asc, 0.0))
    return jnp.sum(s_asc * jl * dtau)


def _los_delta_traced(source, tau, tau0, k, ell, bessel_lmax: int):
    """Backward-compatible LOS; ``ell`` may be traced — uses table up to lmax."""
    dtau = camb_dpoints(tau, half_ends=True)
    chi = jnp.maximum(tau0 - tau, 0.0)
    x = k * chi
    jl_all = spherical_bessel_jl_scan(bessel_lmax, x)
    ell_i = jnp.clip(jnp.asarray(ell, dtype=jnp.int32), 0, bessel_lmax)
    jl = jl_all[ell_i]
    return jnp.sum(source * jl * dtau)


def _los_delta_horner_chunk(
    sources_k: jnp.ndarray,
    dtau: jnp.ndarray,
    chi: jnp.ndarray,
    k_chunk: jnp.ndarray,
    table: dict,
    ell_index: int,
) -> jnp.ndarray:
    """Δ_ℓ(k) for one ell_index over a k-chunk via Horner j_ℓ."""
    x = k_chunk[:, None] * chi[None, :]
    jl = eval_bessel_horner(table, int(ell_index), x)
    return jnp.sum(sources_k * jl * dtau[None, :], axis=-1)


def _vmap_los_chunked(sources_k, tau, tau0, k, ell_n: int, *, chunk: int = LOS_K_CHUNK):
    """LOS Δ_ℓ(k); prefers Horner table, else Miller (small n_k / single ell)."""
    n_k = int(k.shape[0])
    chunk = max(int(chunk), 1)
    t0 = float(np.asarray(tau0))
    chi_max = float(np.max(np.maximum(t0 - np.asarray(tau), 0.0)))
    k_max = float(np.max(np.asarray(k)))
    xmax = max(k_max * chi_max * 1.05, 50.0)
    table = build_bessel_horner_table([int(ell_n)], xmax)
    dtau = camb_dpoints(jnp.asarray(tau, dtype=jnp.float64), half_ends=True)
    chi = jnp.maximum(jnp.asarray(tau0, dtype=jnp.float64) - jnp.asarray(tau), 0.0)

    parts = []
    for i0 in range(0, n_k, chunk):
        i1 = min(i0 + chunk, n_k)
        parts.append(
            _los_delta_horner_chunk(
                sources_k[i0:i1], dtau, chi, k[i0:i1], table, 0
            )
        )
    return jnp.concatenate(parts, axis=0)


def _los_delta_pair_shared_jl(source_t, source_e, tau, tau0, k, ell: int):
    """Δ_T and Δ_E with a single shared j_ℓ(kχ) (Horner table)."""
    tau = jnp.asarray(tau, dtype=jnp.float64)
    t0 = jnp.asarray(tau0, dtype=jnp.float64)
    kk = jnp.asarray(k, dtype=jnp.float64)
    dtau = camb_dpoints(tau, half_ends=True)
    chi = jnp.maximum(t0 - tau, 0.0)
    xmax = float(np.asarray(kk)) * float(np.max(np.asarray(chi))) * 1.05 + 10.0
    table = build_bessel_horner_table([int(ell)], max(xmax, 50.0))
    jl = eval_bessel_horner(table, 0, kk * chi)
    dt = jnp.sum(jnp.asarray(source_t, dtype=jnp.float64) * jl * dtau)
    de = jnp.sum(jnp.asarray(source_e, dtype=jnp.float64) * jl * dtau)
    return dt, de


def _pad_to_chunk(n: int, chunk: int) -> int:
    """Smallest multiple of ``chunk`` that is ≥ ``n`` (Python-static sizes)."""
    chunk = max(int(chunk), 1)
    n = int(n)
    return ((n + chunk - 1) // chunk) * chunk


def _los_delta_ell_k_scan(
    sources_k: jnp.ndarray,
    dtau: jnp.ndarray,
    chi: jnp.ndarray,
    k: jnp.ndarray,
    table: dict,
    n_ell: int,
    *,
    k_chunk: int = LOS_K_CHUNK,
    ell_chunk: int = LOS_ELL_CHUNK,
) -> jnp.ndarray:
    """Δ_ℓ(k) via ``lax.scan`` over (ℓ, k) chunks — no Python chunk unroll.

    Pads ℓ/k to chunk multiples so scan lengths are static; masks padded
    rows/cols to zero. Peak VRAM stays O(ell_chunk · k_chunk · n_τ).
    """
    n_k = int(k.shape[0])
    k_chunk = max(int(k_chunk), 1)
    ell_chunk = max(int(ell_chunk), 1)
    n_ell_pad = _pad_to_chunk(n_ell, ell_chunk)
    n_k_pad = _pad_to_chunk(n_k, k_chunk)
    n_e_chunks = n_ell_pad // ell_chunk
    n_k_chunks = n_k_pad // k_chunk

    sources_pad = jnp.pad(
        sources_k, ((0, n_k_pad - n_k), (0, 0)), constant_values=0.0
    )
    k_pad = jnp.pad(k, (0, n_k_pad - n_k), constant_values=k[-1])
    # Clamp padded ell indices into the Horner table; mask after.
    last_ix = jnp.int32(max(n_ell - 1, 0))

    def ell_chunk_body(_carry, e_ix):
        e0 = e_ix * jnp.int32(ell_chunk)
        idx_raw = jnp.arange(ell_chunk, dtype=jnp.int32) + e0
        idx = jnp.minimum(idx_raw, last_ix)
        ell_ok = idx_raw < n_ell

        def k_chunk_body(_c, i_ix):
            i0 = i_ix * jnp.int32(k_chunk)
            kk = jax.lax.dynamic_slice(k_pad, (i0,), (k_chunk,))
            src = jax.lax.dynamic_slice(
                sources_pad,
                (i0, jnp.int32(0)),
                (k_chunk, sources_pad.shape[1]),
            )
            x = kk[:, None] * chi[None, :]
            jl = eval_bessel_horner_many(table, idx, x)  # (ell_chunk, ck, nτ)
            delta = jnp.sum(src[None, :, :] * jl * dtau[None, None, :], axis=-1)
            k_ok = (jnp.arange(k_chunk, dtype=jnp.int32) + i0) < n_k
            delta = jnp.where(ell_ok[:, None] & k_ok[None, :], delta, 0.0)
            return None, delta

        _, deltas = jax.lax.scan(
            k_chunk_body, None, jnp.arange(n_k_chunks, dtype=jnp.int32)
        )
        # (n_k_chunks, ell_chunk, k_chunk) → (ell_chunk, n_k_pad)
        delta_e = jnp.transpose(deltas, (1, 0, 2)).reshape(ell_chunk, n_k_pad)
        return None, delta_e

    _, delta_chunks = jax.lax.scan(
        ell_chunk_body, None, jnp.arange(n_e_chunks, dtype=jnp.int32)
    )
    delta_all = delta_chunks.reshape(n_ell_pad, n_k_pad)
    return delta_all[:n_ell, :n_k]


def _los_te_cross_scan(
    sources_t: jnp.ndarray,
    sources_e: jnp.ndarray,
    dtau: jnp.ndarray,
    chi: jnp.ndarray,
    k: jnp.ndarray,
    p_r: jnp.ndarray,
    dlnk: jnp.ndarray,
    table: dict,
    n_ell: int,
    *,
    k_chunk: int = LOS_K_CHUNK,
    ell_chunk: int = LOS_ELL_CHUNK,
) -> jnp.ndarray:
    """Σ_q P Δ_T Δ_E dlnk via ``lax.scan`` over (ℓ, k) chunks."""
    n_k = int(k.shape[0])
    k_chunk = max(int(k_chunk), 1)
    ell_chunk = max(int(ell_chunk), 1)
    n_ell_pad = _pad_to_chunk(n_ell, ell_chunk)
    n_k_pad = _pad_to_chunk(n_k, k_chunk)
    n_e_chunks = n_ell_pad // ell_chunk
    n_k_chunks = n_k_pad // k_chunk

    st_pad = jnp.pad(sources_t, ((0, n_k_pad - n_k), (0, 0)), constant_values=0.0)
    se_pad = jnp.pad(sources_e, ((0, n_k_pad - n_k), (0, 0)), constant_values=0.0)
    k_pad = jnp.pad(k, (0, n_k_pad - n_k), constant_values=k[-1])
    pr_pad = jnp.pad(p_r, (0, n_k_pad - n_k), constant_values=0.0)
    dlnk_pad = jnp.pad(dlnk, (0, n_k_pad - n_k), constant_values=0.0)
    last_ix = jnp.int32(max(n_ell - 1, 0))

    def ell_chunk_body(_carry, e_ix):
        e0 = e_ix * jnp.int32(ell_chunk)
        idx_raw = jnp.arange(ell_chunk, dtype=jnp.int32) + e0
        idx = jnp.minimum(idx_raw, last_ix)
        ell_ok = idx_raw < n_ell

        def k_chunk_body(cross, i_ix):
            i0 = i_ix * jnp.int32(k_chunk)
            kk = jax.lax.dynamic_slice(k_pad, (i0,), (k_chunk,))
            st = jax.lax.dynamic_slice(
                st_pad, (i0, jnp.int32(0)), (k_chunk, st_pad.shape[1])
            )
            se = jax.lax.dynamic_slice(
                se_pad, (i0, jnp.int32(0)), (k_chunk, se_pad.shape[1])
            )
            pr = jax.lax.dynamic_slice(pr_pad, (i0,), (k_chunk,))
            dl = jax.lax.dynamic_slice(dlnk_pad, (i0,), (k_chunk,))
            x = kk[:, None] * chi[None, :]
            jl = eval_bessel_horner_many(table, idx, x)
            w = jl * dtau[None, None, :]
            dt = jnp.sum(st[None, :, :] * w, axis=-1)
            de = jnp.sum(se[None, :, :] * w, axis=-1)
            # Padded-k rows contribute 0 via pr_pad / dlnk_pad.
            term = jnp.sum(pr[None, :] * dt * de * dl[None, :], axis=-1)
            return cross + term, None

        cross0 = jnp.zeros(ell_chunk, dtype=jnp.float64)
        cross, _ = jax.lax.scan(
            k_chunk_body, cross0, jnp.arange(n_k_chunks, dtype=jnp.int32)
        )
        cross = jnp.where(ell_ok, cross, 0.0)
        return None, cross

    _, cross_chunks = jax.lax.scan(
        ell_chunk_body, None, jnp.arange(n_e_chunks, dtype=jnp.int32)
    )
    return cross_chunks.reshape(n_ell_pad)[:n_ell]


def _los_tt_ee_te_scan(
    sources_t: jnp.ndarray,
    sources_e: jnp.ndarray,
    dtau: jnp.ndarray,
    chi: jnp.ndarray,
    k: jnp.ndarray,
    p_r: jnp.ndarray,
    dlnk: jnp.ndarray,
    table: dict,
    n_ell: int,
    *,
    k_chunk: int = LOS_K_CHUNK,
    ell_chunk: int = LOS_ELL_CHUNK,
    compute_dtype=jnp.float64,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Σ_q P dlnk (Δ_T², Δ_E², Δ_T Δ_E) in one (ℓ, k) scan.

    One ``j_ℓ`` evaluation per chunk feeds both projections, where separate
    TT / EE / TE calls evaluate it three times and project T and E twice.
    ``compute_dtype`` sets the precision of ``j_ℓ`` and the τ quadrature for
    Δ_ℓ(k); the k sums always accumulate in float64.
    """
    n_k = int(k.shape[0])
    k_chunk = max(int(k_chunk), 1)
    ell_chunk = max(int(ell_chunk), 1)
    n_ell_pad = _pad_to_chunk(n_ell, ell_chunk)
    n_k_pad = _pad_to_chunk(n_k, k_chunk)
    n_e_chunks = n_ell_pad // ell_chunk
    n_k_chunks = n_k_pad // k_chunk

    st_pad = jnp.pad(sources_t, ((0, n_k_pad - n_k), (0, 0)), constant_values=0.0)
    se_pad = jnp.pad(sources_e, ((0, n_k_pad - n_k), (0, 0)), constant_values=0.0)
    k_pad = jnp.pad(k, (0, n_k_pad - n_k), constant_values=k[-1])
    # Padded-k columns carry zero weight.
    w_pad = jnp.pad(p_r * dlnk, (0, n_k_pad - n_k), constant_values=0.0)
    last_ix = jnp.int32(max(n_ell - 1, 0))

    def ell_chunk_body(_carry, e_ix):
        e0 = e_ix * jnp.int32(ell_chunk)
        idx_raw = jnp.arange(ell_chunk, dtype=jnp.int32) + e0
        idx = jnp.minimum(idx_raw, last_ix)
        ell_ok = idx_raw < n_ell

        def k_chunk_body(acc, i_ix):
            i0 = i_ix * jnp.int32(k_chunk)
            kk = jax.lax.dynamic_slice(k_pad, (i0,), (k_chunk,))
            st = jax.lax.dynamic_slice(
                st_pad, (i0, jnp.int32(0)), (k_chunk, st_pad.shape[1])
            )
            se = jax.lax.dynamic_slice(
                se_pad, (i0, jnp.int32(0)), (k_chunk, se_pad.shape[1])
            )
            w = jax.lax.dynamic_slice(w_pad, (i0,), (k_chunk,))
            x = kk[:, None] * chi[None, :]
            cd = compute_dtype
            jl = eval_bessel_horner_many(table, idx, x, dtype=cd)
            jl = jl * dtau.astype(cd)[None, None, :]
            dt = jnp.sum(st.astype(cd)[None, :, :] * jl, axis=-1).astype(jnp.float64)
            de = jnp.sum(se.astype(cd)[None, :, :] * jl, axis=-1).astype(jnp.float64)
            tt, ee, te = acc
            return (
                tt + jnp.sum(w[None, :] * dt * dt, axis=-1),
                ee + jnp.sum(w[None, :] * de * de, axis=-1),
                te + jnp.sum(w[None, :] * dt * de, axis=-1),
            ), None

        z = jnp.zeros(ell_chunk, dtype=jnp.float64)
        (tt, ee, te), _ = jax.lax.scan(
            k_chunk_body, (z, z, z), jnp.arange(n_k_chunks, dtype=jnp.int32)
        )
        mask = lambda a: jnp.where(ell_ok, a, 0.0)  # noqa: E731
        return None, (mask(tt), mask(ee), mask(te))

    _, (tt, ee, te) = jax.lax.scan(
        ell_chunk_body, None, jnp.arange(n_e_chunks, dtype=jnp.int32)
    )
    fac = 4.0 * jnp.pi
    return tuple(fac * a.reshape(n_ell_pad)[:n_ell] for a in (tt, ee, te))


def _los_tt_ee_te_scan_k_outer(
    sources_t: jnp.ndarray,
    sources_e: jnp.ndarray,
    dtau: jnp.ndarray,
    chi: jnp.ndarray,
    k: jnp.ndarray,
    p_r: jnp.ndarray,
    dlnk: jnp.ndarray,
    table: dict,
    n_ell: int,
    *,
    k_chunk: int = LOS_K_CHUNK,
    ell_chunk: int = LOS_ELL_CHUNK,
    compute_dtype=jnp.float64,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """``_los_tt_ee_te_scan`` with the k chunks outside: the interval search and
    ``dx`` of x = kχ are done once per k chunk for all ℓ chunks, and the Horner
    coefficients are gathered as [ℓ, interval, 4] rows. Each ℓ receives the same
    per-chunk k sums in the same order; only the order of the τ quadrature sums
    (in ``compute_dtype``) can differ."""
    n_k = int(k.shape[0])
    k_chunk = max(int(k_chunk), 1)
    ell_chunk = max(int(ell_chunk), 1)
    n_ell_pad = _pad_to_chunk(n_ell, ell_chunk)
    n_k_pad = _pad_to_chunk(n_k, k_chunk)
    n_e_chunks = n_ell_pad // ell_chunk
    n_k_chunks = n_k_pad // k_chunk
    cd = compute_dtype
    use32 = jnp.dtype(cd) == jnp.float32
    coeffs_t = jnp.asarray(table["coeffs32_t"] if use32 else table["coeffs_t"])
    knots = jnp.asarray(table["knots"], dtype=jnp.float64)
    xlims = jnp.asarray(table["xlims"], dtype=jnp.float64)

    st_pad = jnp.pad(sources_t, ((0, n_k_pad - n_k), (0, 0)), constant_values=0.0)
    se_pad = jnp.pad(sources_e, ((0, n_k_pad - n_k), (0, 0)), constant_values=0.0)
    k_pad = jnp.pad(k, (0, n_k_pad - n_k), constant_values=k[-1])
    w_pad = jnp.pad(p_r * dlnk, (0, n_k_pad - n_k), constant_values=0.0)
    last_ix = jnp.int32(max(n_ell - 1, 0))
    dtau_c = dtau.astype(cd)

    def k_chunk_body(acc, i_ix):
        i0 = i_ix * jnp.int32(k_chunk)
        kk = jax.lax.dynamic_slice(k_pad, (i0,), (k_chunk,))
        st = jax.lax.dynamic_slice(st_pad, (i0, jnp.int32(0)), (k_chunk, st_pad.shape[1])).astype(cd)
        se = jax.lax.dynamic_slice(se_pad, (i0, jnp.int32(0)), (k_chunk, se_pad.shape[1])).astype(cd)
        w = jax.lax.dynamic_slice(w_pad, (i0,), (k_chunk,))
        x = kk[:, None] * chi[None, :]
        x_clip = jnp.clip(x, knots[0], knots[-1])
        it = jnp.clip(jnp.searchsorted(knots, x_clip, side="right") - 1, 0, knots.shape[0] - 2)
        dx = (x_clip - knots[it]).astype(cd)

        def ell_chunk_body(_, e_ix):
            e0 = e_ix * jnp.int32(ell_chunk)
            idx = jnp.minimum(jnp.arange(ell_chunk, dtype=jnp.int32) + e0, last_ix)
            c = coeffs_t[idx[:, None, None], it[None, :, :]]  # (ℓ, k, τ, 4)
            jl = ((c[..., 0] * dx + c[..., 1]) * dx + c[..., 2]) * dx + c[..., 3]
            jl = jnp.where(x[None] < xlims[idx][:, None, None], 0.0, jl)
            jl = jnp.where(jnp.isfinite(jl), jl, 0.0)
            jl = jl * dtau_c[None, None, :]
            dt = jnp.sum(st[None, :, :] * jl, axis=-1).astype(jnp.float64)
            de = jnp.sum(se[None, :, :] * jl, axis=-1).astype(jnp.float64)
            return None, (jnp.sum(w[None, :] * dt * dt, axis=-1),
                          jnp.sum(w[None, :] * de * de, axis=-1),
                          jnp.sum(w[None, :] * dt * de, axis=-1))

        _, (tt_c, ee_c, te_c) = jax.lax.scan(ell_chunk_body, None,
                                             jnp.arange(n_e_chunks, dtype=jnp.int32))
        tt, ee, te = acc
        return (tt + tt_c.reshape(n_ell_pad), ee + ee_c.reshape(n_ell_pad),
                te + te_c.reshape(n_ell_pad)), None

    z = jnp.zeros(n_ell_pad, dtype=jnp.float64)
    (tt, ee, te), _ = jax.lax.scan(k_chunk_body, (z, z, z), jnp.arange(n_k_chunks, dtype=jnp.int32))
    ell_ok = jnp.arange(n_ell_pad) < n_ell
    fac = 4.0 * jnp.pi
    return tuple(fac * jnp.where(ell_ok, a, 0.0)[:n_ell] for a in (tt, ee, te))


# ---- curved space ------------------------------------------------------------------------------
# With curvature K [Mpc⁻²] the projection kernel is the hyperspherical Bessel function
# Φ_ν^ℓ(χ/R), ν = qR, R = |K|^-½ (CAMB cmbmain.f90). Every evaluator used here writes it as
# Φ = w · j_ℓ(x_eff) — the flat spherical Bessel (the same Horner tables) at a curvature-mapped
# argument times an amplitude — so x and the table interval depend on ℓ, not only on qχ.
# ``curved_bessel_map`` supplies (x_eff, w[, Φ_direct, use_direct]) per (ℓ, q, χ).
# Flat-equivalent argument stretch of open models: tables reach CURVED_XMAX_FACTOR × max qχ.
CURVED_XMAX_FACTOR = 2.0
# The Olver map has ~50 elementwise intermediates per (ℓ, k, τ): smaller chunks than the flat scan.
CURVED_K_CHUNK = 64
CURVED_ELL_CHUNK = 8


def curved_bessel_map(ell, q, chi, K):
    """(x_eff, w) with Φ_ν^ℓ(χ) = w·j_ℓ(x_eff): ``hyperspherical_jax.olver_map_phys`` (CAMB 2.0.4's
    Olver map with its near-flat small-χ series; continuous through K = 0). Broadcasts ℓ (float
    array, ℓ ≥ 3), q [1/Mpc], χ [Mpc]; K traced (any sign).

    The map's fallback corners (closed ν − ℓ ≲ 128, open ν ≲ 0.1–0.4 ℓ) keep the leading map
    (≤ 3e-2 of the peak there): their turning point lies beyond χ* for |Ω_k| ≲ 0.1, where Φ is
    evanescent at every χ the sources reach. ℓ ≤ 2 uses the exact forms (``_curved_phi(exact=True)``)."""
    from gpucosmo.jax.eb.hyperspherical_jax import olver_map_phys

    x_eff, w, _ok = olver_map_phys(ell, q, chi, K)
    return x_eff, w


def _curved_phi(table, idx, ell_vals, q, chi, K, cd, exact: bool = False):
    """Φ_ν^ℓ for the table's ℓ indices ``idx`` (values ``ell_vals``) on a (k, τ) chunk: (ℓ, k, τ).
    Closed models: Φ = 0 for ν = qR ≤ ℓ (CAMB's cut). ``exact``: ``hyperspherical_jax.phi_phys``
    (the exact ℓ ≤ 2 forms) instead of the table lookup."""
    lv = ell_vals[:, None, None]
    if exact:
        from gpucosmo.jax.eb.hyperspherical_jax import phi_phys

        phi = phi_phys(lv, q[None, :, None], chi[None, None, :], K, recursion=False, gegenbauer=False)
        return jnp.where(jnp.isfinite(phi), phi, 0.0).astype(cd)
    knots = jnp.asarray(table["knots"], dtype=jnp.float64)
    use32 = jnp.dtype(cd) == jnp.float32
    coeffs_t = jnp.asarray(table["coeffs32_t"] if use32 else table["coeffs_t"])
    xlims = jnp.asarray(table["xlims"], dtype=jnp.float64)
    x_eff, w = curved_bessel_map(lv, q[None, :, None], chi[None, None, :], K)
    x_clip = jnp.clip(x_eff, knots[0], knots[-1])
    it = jnp.clip(jnp.searchsorted(knots, x_clip, side="right") - 1, 0, knots.shape[0] - 2)
    dx = (x_clip - knots[it]).astype(cd)
    c = coeffs_t[idx[:, None, None], it]  # (ℓ, k, τ, 4)
    jl = ((c[..., 0] * dx + c[..., 1]) * dx + c[..., 2]) * dx + c[..., 3]
    jl = jnp.where(x_eff < xlims[idx][:, None, None], 0.0, jl)
    phi = w * jl
    nu = q[None, :, None] * jax.lax.rsqrt(jnp.maximum(jnp.abs(K), 1e-300))
    phi = jnp.where((K > 0) & (nu <= lv + 0.5), 0.0, phi)
    return jnp.where(jnp.isfinite(phi), phi, 0.0).astype(cd)


def _los_tt_ee_te_scan_curved(sources_t, sources_e, dtau, chi, k, p_r, dlnk, table, ell_list,
                              K, *, k_chunk=LOS_K_CHUNK, ell_chunk=LOS_ELL_CHUNK,
                              compute_dtype=jnp.float64):
    """``_los_tt_ee_te_scan_k_outer`` with the hyperspherical kernel (``_curved_phi``): ℓ ≤ 2 by the
    exact forms (a separate small pass), the rest by the Olver map and the j_ℓ tables."""
    low = [i for i, e in enumerate(ell_list) if e <= 2]
    if low and len(low) < len(ell_list):
        hi_ix = [i for i in range(len(ell_list)) if i not in set(low)]
        a = _los_tt_ee_te_scan_curved_part(sources_t, sources_e, dtau, chi, k, p_r, dlnk, table,
                                           [ell_list[i] for i in low], K, k_chunk=k_chunk, ell_chunk=len(low),
                                           compute_dtype=compute_dtype, exact=True, table_ix=low)
        b = _los_tt_ee_te_scan_curved_part(sources_t, sources_e, dtau, chi, k, p_r, dlnk, table,
                                           [ell_list[i] for i in hi_ix], K, k_chunk=k_chunk, ell_chunk=ell_chunk,
                                           compute_dtype=compute_dtype, table_ix=hi_ix)
        order = np.argsort(np.asarray(low + hi_ix))
        return tuple(jnp.concatenate([x, y])[order] for x, y in zip(a, b))
    return _los_tt_ee_te_scan_curved_part(sources_t, sources_e, dtau, chi, k, p_r, dlnk, table, ell_list, K,
                                          k_chunk=k_chunk, ell_chunk=ell_chunk, compute_dtype=compute_dtype,
                                          exact=bool(low))


def _los_tt_ee_te_scan_curved_part(sources_t, sources_e, dtau, chi, k, p_r, dlnk, table, ell_list,
                                   K, *, k_chunk=LOS_K_CHUNK, ell_chunk=LOS_ELL_CHUNK,
                                   compute_dtype=jnp.float64, exact=False, table_ix=None):
    """One pass of ``_los_tt_ee_te_scan_curved`` over ``ell_list`` (table rows ``table_ix``)."""
    n_ell = len(ell_list)
    n_k = int(k.shape[0])
    k_chunk = max(int(k_chunk), 1)
    ell_chunk = max(int(ell_chunk), 1)
    n_ell_pad = _pad_to_chunk(n_ell, ell_chunk)
    n_k_pad = _pad_to_chunk(n_k, k_chunk)
    cd = compute_dtype
    ell_vals = jnp.asarray(list(ell_list) + [ell_list[-1]] * (n_ell_pad - n_ell), dtype=jnp.float64)
    rows = list(range(n_ell)) if table_ix is None else list(table_ix)
    rows = jnp.asarray(rows + [rows[-1]] * (n_ell_pad - n_ell), dtype=jnp.int32)
    st_pad = jnp.pad(sources_t, ((0, n_k_pad - n_k), (0, 0)), constant_values=0.0)
    se_pad = jnp.pad(sources_e, ((0, n_k_pad - n_k), (0, 0)), constant_values=0.0)
    k_pad = jnp.pad(k, (0, n_k_pad - n_k), constant_values=k[-1])
    w_pad = jnp.pad(p_r * dlnk, (0, n_k_pad - n_k), constant_values=0.0)
    dtau_c = dtau.astype(cd)

    def k_chunk_body(acc, i_ix):
        i0 = i_ix * jnp.int32(k_chunk)
        kk = jax.lax.dynamic_slice(k_pad, (i0,), (k_chunk,))
        st = jax.lax.dynamic_slice(st_pad, (i0, jnp.int32(0)), (k_chunk, st_pad.shape[1])).astype(cd)
        se = jax.lax.dynamic_slice(se_pad, (i0, jnp.int32(0)), (k_chunk, se_pad.shape[1])).astype(cd)
        w = jax.lax.dynamic_slice(w_pad, (i0,), (k_chunk,))

        def ell_chunk_body(_, e_ix):
            e0 = e_ix * jnp.int32(ell_chunk)
            idx = jax.lax.dynamic_slice(rows, (e0,), (ell_chunk,))
            lv = jax.lax.dynamic_slice(ell_vals, (e0,), (ell_chunk,))
            phi = _curved_phi(table, idx, lv, kk, chi, K, cd, exact=exact) * dtau_c[None, None, :]
            dt = jnp.sum(st[None, :, :] * phi, axis=-1).astype(jnp.float64)
            de = jnp.sum(se[None, :, :] * phi, axis=-1).astype(jnp.float64)
            return None, (jnp.sum(w[None, :] * dt * dt, axis=-1),
                          jnp.sum(w[None, :] * de * de, axis=-1),
                          jnp.sum(w[None, :] * dt * de, axis=-1))

        _, (tt_c, ee_c, te_c) = jax.lax.scan(ell_chunk_body, None,
                                             jnp.arange(n_ell_pad // ell_chunk, dtype=jnp.int32))
        tt, ee, te = acc
        return (tt + tt_c.reshape(n_ell_pad), ee + ee_c.reshape(n_ell_pad),
                te + te_c.reshape(n_ell_pad)), None

    z = jnp.zeros(n_ell_pad, dtype=jnp.float64)
    (tt, ee, te), _ = jax.lax.scan(k_chunk_body, (z, z, z), jnp.arange(n_k_pad // k_chunk, dtype=jnp.int32))
    return tuple(4.0 * jnp.pi * a[:n_ell] for a in (tt, ee, te))


def _bessel_xmax_host(k: jnp.ndarray, chi: jnp.ndarray) -> float:
    """Host xmax for Horner table build; falls back when arrays are traced."""
    try:
        k_max = float(np.max(np.asarray(k)))
    except (TypeError, ValueError):
        k_max = float(0.45)
    try:
        chi_max = float(np.max(np.asarray(chi)))
    except (TypeError, ValueError):
        # Under vmap / device_grids, τ0 is traced — use a conservative χ bound.
        chi_max = 1.5e4
    return max(k_max * chi_max * 1.05, 50.0)


def los_cl_from_sources(
    sources_k: jnp.ndarray,
    tau: jnp.ndarray,
    tau0: jnp.ndarray,
    k: jnp.ndarray,
    p_r: jnp.ndarray,
    ell: jnp.ndarray,
    *,
    bessel_lmax: int,
    ell_list: list[int] | None = None,
) -> jnp.ndarray:
    """Full flat Bessel LOS + CAMB ``dlnk`` Cls sum (device Horner tables).

    Builds one CAMB-style ``j_ℓ`` spline table for the sparse ℓ set, then
    evaluates fused LOS in (k, ℓ) chunks via ``lax.scan`` (bounded VRAM,
    no Python ``for e0`` graph unroll).
    """
    del bessel_lmax
    dlnk = camb_dlnk(k)
    if ell_list is None:
        ell_list = [int(e) for e in np.asarray(ell).tolist()]
    k_chunk = max(LOS_K_CHUNK, 1)
    ell_chunk = max(LOS_ELL_CHUNK, 1)

    t0 = jnp.asarray(tau0, dtype=jnp.float64)
    tau_a = jnp.asarray(tau, dtype=jnp.float64)
    dtau = camb_dpoints(tau_a, half_ends=True)
    chi = jnp.maximum(t0 - tau_a, 0.0)
    table = build_bessel_horner_table(ell_list, _bessel_xmax_host(k, chi))

    n_ell = len(ell_list)
    delta_all = _los_delta_ell_k_scan(
        sources_k,
        dtau,
        chi,
        k,
        table,
        n_ell,
        k_chunk=k_chunk,
        ell_chunk=ell_chunk,
    )
    # Broadcast Cls sum over ℓ: shape (n_ell,)
    return cl_from_delta_l_k(delta_all, p_r, dlnk)


def los_cl_te_from_sources(
    sources_t: jnp.ndarray,
    sources_e: jnp.ndarray,
    tau: jnp.ndarray,
    tau0: jnp.ndarray,
    k: jnp.ndarray,
    p_r: jnp.ndarray,
    ell: jnp.ndarray,
    *,
    bessel_lmax: int,
    ell_list: list[int] | None = None,
) -> jnp.ndarray:
    """C_ℓ^{TE} via flat Bessel LOS + CAMB dlnk; shared Horner j_ℓ for T/E."""
    del bessel_lmax
    dlnk = camb_dlnk(k)
    if ell_list is None:
        ell_list = [int(e) for e in np.asarray(ell).tolist()]
    k_chunk = max(LOS_K_CHUNK, 1)
    ell_chunk = max(LOS_ELL_CHUNK, 1)

    t0 = jnp.asarray(tau0, dtype=jnp.float64)
    tau_a = jnp.asarray(tau, dtype=jnp.float64)
    dtau = camb_dpoints(tau_a, half_ends=True)
    chi = jnp.maximum(t0 - tau_a, 0.0)
    table = build_bessel_horner_table(ell_list, _bessel_xmax_host(k, chi))

    n_ell = len(ell_list)
    cross = _los_te_cross_scan(
        sources_t,
        sources_e,
        dtau,
        chi,
        k,
        p_r,
        dlnk,
        table,
        n_ell,
        k_chunk=k_chunk,
        ell_chunk=ell_chunk,
    )
    return 4.0 * jnp.pi * cross


def los_cls_tt_ee_te_from_sources(
    sources_t: jnp.ndarray,
    sources_e: jnp.ndarray,
    tau: jnp.ndarray,
    tau0: jnp.ndarray,
    k: jnp.ndarray,
    p_r: jnp.ndarray,
    *,
    ell_list: list[int],
    compute_dtype=None,
    curv=None,
    k_max_host: float | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """(C_ℓ^TT, C_ℓ^EE, C_ℓ^TE) from one shared Bessel LOS pass (``curv`` = K [Mpc⁻²], traced:
    the hyperspherical kernel; ``p_r`` must then carry CAMB's curved measure, see cls_jax).
    ``k_max_host``: host bound on ``k`` for the Bessel-table range when ``k`` is traced (the curved
    grids are); without it a traced ``k`` assumes k ≤ 0.45/Mpc."""
    dlnk = camb_dlnk(k)
    t0 = jnp.asarray(tau0, dtype=jnp.float64)
    tau_a = jnp.asarray(tau, dtype=jnp.float64)
    dtau = camb_dpoints(tau_a, half_ends=True)
    chi = jnp.maximum(t0 - tau_a, 0.0)
    cd = _resolve_los_dtype(LOS_COMPUTE_DTYPE if compute_dtype is None else compute_dtype)
    if curv is not None:
        k_b = k if k_max_host is None else jnp.asarray([float(k_max_host)])
        table = build_bessel_horner_table(ell_list, CURVED_XMAX_FACTOR * _bessel_xmax_host(k_b, chi))
        return _los_tt_ee_te_scan_curved(sources_t, sources_e, dtau, chi, k, p_r, dlnk, table, ell_list,
                                         curv, k_chunk=CURVED_K_CHUNK, ell_chunk=CURVED_ELL_CHUNK,
                                         compute_dtype=cd)
    table = build_bessel_horner_table(ell_list, _bessel_xmax_host(k, chi))
    scan = _los_tt_ee_te_scan_k_outer if LOS_LOOP_ORDER == "k_outer" else _los_tt_ee_te_scan
    return scan(
        sources_t, sources_e, dtau, chi, k, p_r, dlnk, table, len(ell_list),
        k_chunk=max(LOS_K_CHUNK, 1), ell_chunk=max(LOS_ELL_CHUNK, 1),
        compute_dtype=_resolve_los_dtype(
            LOS_COMPUTE_DTYPE if compute_dtype is None else compute_dtype
        ),
    )


def lensing_source_from_weyl(
    weyl_k: jnp.ndarray,
    tau: jnp.ndarray,
    tau0: jnp.ndarray,
    chi_star: jnp.ndarray,
    curv=None,
) -> jnp.ndarray:
    """CAMB's lensing-potential source ``OutputSources(3)``.

    ``equations.f90``::

        sources(3) = -2*phi*f_K(tau-tau_maxvis)/(f_K(tau0-tau_maxvis)*f_K(tau0-tau))

    which for a flat model is ``-weyl*(chi_star-chi)/(chi_star*chi)`` given
    ``weyl = 2*phi``. ``chi_star = tau0 - tau_maxvis`` (visibility peak).
    CAMB zeroes it for ``tau <= tau_maxvis`` and ``tau0-tau <= 0.1``; folding
    the kernel into the source here lets the same Bessel LOS machinery used
    for T/E project it with no special-casing.
    """
    tau = jnp.asarray(tau, dtype=jnp.float64)
    t0 = jnp.asarray(tau0, dtype=jnp.float64)
    chi = t0 - tau
    chi_s = jnp.maximum(jnp.asarray(chi_star, dtype=jnp.float64), 1.0)
    if curv is None:
        geom = (chi_s - chi) / jnp.maximum(chi_s * chi, 1e-30)
    else:  # f_K(χ* − χ)/(f_K(χ*) f_K(χ))
        from gpucosmo.jax.eb.hierarchy_jax import f_k

        geom = f_k(chi_s - chi, curv) / jnp.maximum(f_k(chi_s, curv) * f_k(chi, curv), 1e-30)
    geom = jnp.where((chi > 0.1) & (chi < chi_s), geom, 0.0)
    return jnp.nan_to_num(-weyl_k * geom[None, :], nan=0.0, posinf=0.0, neginf=0.0)


def _interp_logp(p_r: jnp.ndarray, k: jnp.ndarray, kq: jnp.ndarray) -> jnp.ndarray:
    """P(kq) by log-log interpolation: exact for the power-law primordial P."""
    lk = jnp.log(jnp.maximum(k, 1e-300))
    lp = jnp.log(jnp.maximum(p_r, 1e-300))
    return jnp.exp(jnp.interp(jnp.log(jnp.maximum(kq, 1e-300)), lk, lp))


def spline_d2_in_k(src: jnp.ndarray, k: jnp.ndarray) -> jnp.ndarray:
    """Natural-cubic-spline second derivatives along k, per τ column.

    CAMB's ``ddScaledSrc``: the Limber source is read off ``Evolve_q`` by cubic
    spline, and ``Evolve_q`` is coarse (~200 points), so linear interpolation here
    costs a few tenths of a percent in C_L^{φφ}.
    """
    src = jnp.asarray(src, dtype=jnp.float64)
    k = jnp.asarray(k, dtype=jnp.float64)
    n = k.shape[0]
    h = jnp.diff(k)
    # Thomas algorithm on the natural-spline tridiagonal system, vectorized
    # over the τ columns.
    rhs = 6.0 * ((src[2:] - src[1:-1]) / h[1:, None] - (src[1:-1] - src[:-2]) / h[:-1, None])
    a = h[:-1]
    b = 2.0 * (k[2:] - k[:-2])
    c = h[1:]

    def fwd(carry, i):
        cp_prev, dp_prev = carry
        m = b[i] - a[i] * cp_prev
        cp = c[i] / m
        dp = (rhs[i] - a[i] * dp_prev) / m
        return (cp, dp), (cp, dp)

    z0 = jnp.zeros((), dtype=jnp.float64)
    _, (cp, dp) = jax.lax.scan(
        fwd, (z0, jnp.zeros(src.shape[1], dtype=jnp.float64)), jnp.arange(n - 2)
    )

    def bwd(x_next, i):
        x = dp[i] - cp[i] * x_next
        return x, x

    _, x_rev = jax.lax.scan(
        bwd, jnp.zeros(src.shape[1], dtype=jnp.float64), jnp.arange(n - 3, -1, -1)
    )
    interior = x_rev[::-1]
    zero = jnp.zeros((1, src.shape[1]), dtype=jnp.float64)
    return jnp.concatenate([zero, interior, zero], axis=0)


def _interp_source_in_k(
    src: jnp.ndarray, k: jnp.ndarray, kq: jnp.ndarray, d2: jnp.ndarray | None = None
):
    """``src[:, n]`` interpolated to ``kq[n]`` for every τ index ``n``.

    Cubic when ``d2`` (from ``spline_d2_in_k``) is given, matching CAMB's Limber
    source lookup; linear otherwise.
    """
    n_k = k.shape[0]
    n_tau = src.shape[1]
    i = jnp.clip(jnp.searchsorted(k, kq, side="right") - 1, 0, n_k - 2)
    h = k[i + 1] - k[i]
    hs = jnp.where(h > 0, h, 1.0)
    b = (kq - k[i]) / hs
    a = 1.0 - b
    n = jnp.arange(n_tau)
    lin = a * src[i, n] + b * src[i + 1, n]
    if d2 is None:
        return lin
    return lin + ((a**3 - a) * d2[i, n] + (b**3 - b) * d2[i + 1, n]) * hs**2 / 6.0


def cl_pp_limber_camb(
    source_phi: jnp.ndarray,
    tau: jnp.ndarray,
    tau0: jnp.ndarray,
    chi_star: jnp.ndarray,
    k: jnp.ndarray,
    p_r: jnp.ndarray,
    ell: jnp.ndarray,
    curv=None,
) -> jnp.ndarray:
    """C_ℓ^{φφ} by CAMB's Limber quadrature, summed over χ rather than k.

    ``CalcLimberScalCls``/``GetLimberTransfers`` integrate over the τ grid with
    ``Source = sqrt(chi*dchi)*S`` and ``fac = 2 pi^2 / (l+0.5)^3``, i.e.

        C_ℓ = 2π²/ν³ Σ_n χ_n dχ_n S(ν/χ_n, τ_n)² P(ν/χ_n),   ν = ℓ+1/2.

    Summing over χ (not k) is what makes this match CAMB: the k grid is tuned for
    the T/E Bessel integrals, and reusing it here samples χ = ν/k badly at low ℓ.

    ``curv`` (K [Mpc⁻²]): CAMB's curved transfer Limber (``IntegrateSourcesBessels``,
    cmbmain.f90), Δ_ℓ(q) = S(q, τ0 − χ) √(π/2/ν/√(1 − Kℓ²/q²))/q at f_K(χ) = ν/q, integrated
    with the curved measure dq q/k_s² P(k_s); in χ that is

        C_ℓ = 2π²/ν³ Σ_n f_K cos_K dχ_n (q²/k_s²) P(k_s) S(q, τ_n)² / √(1 − Kℓ²/q²),   q = ν/f_K(χ_n),

    (the near branch of f_K⁻¹: cos_K > 0), equal to the flat sum for K = 0.
    """
    tau = jnp.asarray(tau, dtype=jnp.float64)
    t0 = jnp.asarray(tau0, dtype=jnp.float64)
    chi = t0 - tau
    dchi = camb_dpoints(tau, half_ends=True)
    chi_s = jnp.maximum(jnp.asarray(chi_star, dtype=jnp.float64), 1.0)
    k_min, k_max = k[0], k[-1]
    inside = (chi > 0.1) & (chi < chi_s)
    d2 = spline_d2_in_k(source_phi, k)
    if curv is None:
        r_chi, jac = chi, chi  # f_K(χ), f_K cos_K
    else:
        from gpucosmo.jax.eb.hierarchy_jax import cot_k, f_k

        r_chi = f_k(chi, curv)
        jac = r_chi * r_chi * cot_k(chi, curv)
        inside = inside & (jac > 0.0)

    def one_ell(l):
        l = jnp.asarray(l, dtype=jnp.float64)
        nu = l + 0.5
        kq = nu / jnp.maximum(r_chi, 1e-30)
        s = _interp_source_in_k(source_phi, k, kq, d2)
        # CAMB drops k outside the solved range (LimbRec%Source = 0 for k > qmax).
        ok = inside & (kq >= k_min) & (kq <= k_max)
        if curv is None:
            pw = _interp_logp(p_r, k, kq)
        else:  # p_r is the flat-measure primordial P on k: evaluate it at k_s
            ks2 = jnp.maximum(kq * kq - curv, 1e-300)
            sq = 1.0 - curv * l * l / jnp.maximum(kq * kq, 1e-300)
            ok = ok & (sq > 0.0)
            pw = _interp_logp(p_r, k, jnp.sqrt(ks2)) * kq * kq / ks2 / jnp.sqrt(jnp.where(ok, sq, 1.0))
        w = jnp.where(ok, jac * dchi * pw, 0.0)
        return 2.0 * jnp.pi**2 / nu**3 * jnp.sum(w * jnp.where(ok, s, 0.0) ** 2)

    return jax.vmap(one_ell)(jnp.asarray(ell, dtype=jnp.float64))


def cl_pp_from_weyl(
    weyl_k: jnp.ndarray,
    tau: jnp.ndarray,
    tau0: jnp.ndarray,
    chi_star: jnp.ndarray,
    k: jnp.ndarray,
    p_r: jnp.ndarray,
    ell: jnp.ndarray,
    *,
    limber_ell: int = 100,
    ell_list: list[int] | None = None,
    weyl_limber: jnp.ndarray | None = None,
    k_limber: jnp.ndarray | None = None,
    p_r_limber: jnp.ndarray | None = None,
    curv=None,
    k_max_host: float | None = None,
) -> jnp.ndarray:
    """C_ℓ^{φφ}: exact Bessel LOS below ``limber_ell``, CAMB Limber above.

    ``curv`` (K [Mpc⁻²]): the lensing kernel with f_K, the hyperspherical LOS below
    ``limber_ell`` and CAMB's curved transfer Limber above (``cl_pp_limber_camb``). CAMB runs
    without source windows (CMB only) switch ``limber_windows`` off and use that transfer Limber,
    not ``GetLimberTransfers``; with K = 0 the two are the same integral. ``p_r_limber`` must
    be the flat-measure P(k) when ``curv`` is given.

    ``limber_ell`` mirrors CAMB's ``SourceTerms%limber_phi_lmin`` (default 100),
    so this reproduces CAMB's own choice of method at every ℓ.

    The two branches want different k ranges, which is why they can take different
    grids. The Bessel branch is bounded by ``max_bessels_etak/tau0``, but the Limber
    branch only ever needs the source at the single k = (ℓ+1/2)/χ per χ, so CAMB
    reads it off ``Evolve_q`` out to ``Max_eta_k/tau0`` -- 5.1 against 0.18 Mpc^-1
    for ``lens_potential_accuracy=4``. Passing the (wider) evolve grid as
    ``weyl_limber``/``k_limber``/``p_r_limber`` reproduces that; omitting them falls
    back to the Bessel grid and silently truncates high ℓ.
    """
    if ell_list is None:
        ell_list = [int(e) for e in np.asarray(ell).tolist()]
    src = lensing_source_from_weyl(weyl_k, tau, tau0, chi_star, curv)
    lim = int(limber_ell)

    out = jnp.zeros(len(ell_list), dtype=jnp.float64)
    los_idx = [i for i, e in enumerate(ell_list) if e < lim]
    lim_idx = [i for i, e in enumerate(ell_list) if e >= lim]
    if los_idx:
        los_list = [ell_list[i] for i in los_idx]
        if curv is None:
            cl_los = los_cl_from_sources(src, tau, tau0, k, p_r, jnp.asarray(los_list, dtype=jnp.float64),
                                         bessel_lmax=max(los_list), ell_list=los_list)
        else:
            cl_los = los_cls_tt_ee_te_from_sources(src, jnp.zeros_like(src), tau, tau0, k, p_r,
                                                   ell_list=los_list, compute_dtype=jnp.float64,
                                                   curv=curv, k_max_host=k_max_host)[0]
        out = out.at[jnp.asarray(los_idx)].set(cl_los)
    if lim_idx:
        lim_list = [ell_list[i] for i in lim_idx]
        if weyl_limber is None:
            if curv is not None:  # p_r carries the curved measure here, the Limber branch adds it itself
                raise ValueError("curved Limber φφ needs weyl_limber / k_limber / p_r_limber (flat-measure P)")
            src_l, k_l, p_l = src, k, p_r
        else:
            src_l = lensing_source_from_weyl(weyl_limber, tau, tau0, chi_star, curv)
            k_l = k if k_limber is None else k_limber
            p_l = p_r if p_r_limber is None else p_r_limber
        out = out.at[jnp.asarray(lim_idx)].set(
            cl_pp_limber_camb(
                src_l,
                tau,
                tau0,
                chi_star,
                k_l,
                p_l,
                jnp.asarray(lim_list, dtype=jnp.float64),
                curv,
            )
        )
    return out


def limber_cl_pp_from_psi(
    weyl_k: jnp.ndarray,
    tau: jnp.ndarray,
    tau0: jnp.ndarray,
    chi_star: jnp.ndarray,
    k: jnp.ndarray,
    p_r: jnp.ndarray,
    ell: jnp.ndarray,
) -> jnp.ndarray:
    """C_ℓ^{φφ} via Weyl Limber (device).

    ``weyl_k`` is φ+ψ along τ. Kernel ``f(χ)=(χ_*-χ)/(χ_* χ)·(φ+ψ)``;
    Limber ``Δ ≈ f(χ_*)/k √(π/(2ν))`` with ``χ_*=ν/k``, then CAMB dlnk sum.
    """
    dlnk = camb_dlnk(k)
    chi_star = jnp.maximum(chi_star, 1.0)

    def one_ell(l):
        nu = l.astype(jnp.float64) + 0.5

        def one_k(i):
            chi = nu / jnp.maximum(k[i], 1e-30)
            active = (chi > 1.0) & (chi < chi_star)
            tau_at = tau0 - chi
            weyl = jnp.interp(tau_at, tau, weyl_k[i], left=0.0, right=0.0)
            geom = (chi_star - chi) / jnp.maximum(chi_star * chi, 1e-30)
            delta = (
                weyl
                * geom
                / jnp.maximum(k[i], 1e-30)
                * jnp.sqrt(jnp.pi / (2.0 * nu))
            )
            return jnp.where(active, delta, 0.0)

        delta = jax.vmap(one_k)(jnp.arange(k.shape[0]))
        return cl_from_delta_l_k(delta, p_r, dlnk)

    return jax.vmap(one_ell)(ell)


def hybrid_cl_from_sources(
    sources_k: jnp.ndarray,
    tau: jnp.ndarray,
    tau0: jnp.ndarray,
    k: jnp.ndarray,
    p_r: jnp.ndarray,
    ell: jnp.ndarray,
    *,
    limber_ell: int,
    bessel_lmax: int,
    ell_list: list[int] | None = None,
) -> jnp.ndarray:
    """Bessel LOS for ℓ < limber_ell, Limber above — fully on-device.

    Only multipoles that use each method are evaluated (no discarded work).
    """
    bl = int(max(bessel_lmax, 1))
    lim = int(limber_ell)
    if ell_list is None:
        ell_list = [int(e) for e in np.asarray(ell).tolist()]
    los_idx = [i for i, e in enumerate(ell_list) if e < lim]
    lim_idx = [i for i, e in enumerate(ell_list) if e >= lim]
    out = jnp.zeros(len(ell_list), dtype=jnp.float64)
    if los_idx:
        los_list = [ell_list[i] for i in los_idx]
        cl_los = los_cl_from_sources(
            sources_k,
            tau,
            tau0,
            k,
            p_r,
            jnp.asarray(los_list, dtype=jnp.float64),
            bessel_lmax=bl,
            ell_list=los_list,
        )
        out = out.at[jnp.asarray(los_idx)].set(cl_los)
    if lim_idx:
        lim_list = [ell_list[i] for i in lim_idx]
        cl_limber = limber_cl_from_sources(
            sources_k,
            tau,
            tau0,
            k,
            p_r,
            jnp.asarray(lim_list, dtype=jnp.float64),
        )
        out = out.at[jnp.asarray(lim_idx)].set(cl_limber)
    return out


def hybrid_cl_te_from_sources(
    sources_t: jnp.ndarray,
    sources_e: jnp.ndarray,
    tau: jnp.ndarray,
    tau0: jnp.ndarray,
    k: jnp.ndarray,
    p_r: jnp.ndarray,
    ell: jnp.ndarray,
    *,
    limber_ell: int,
    bessel_lmax: int,
    ell_list: list[int] | None = None,
) -> jnp.ndarray:
    bl = int(max(bessel_lmax, 1))
    lim = int(limber_ell)
    if ell_list is None:
        ell_list = [int(e) for e in np.asarray(ell).tolist()]
    los_idx = [i for i, e in enumerate(ell_list) if e < lim]
    lim_idx = [i for i, e in enumerate(ell_list) if e >= lim]
    out = jnp.zeros(len(ell_list), dtype=jnp.float64)
    if los_idx:
        los_list = [ell_list[i] for i in los_idx]
        cl_los = los_cl_te_from_sources(
            sources_t,
            sources_e,
            tau,
            tau0,
            k,
            p_r,
            jnp.asarray(los_list, dtype=jnp.float64),
            bessel_lmax=bl,
            ell_list=los_list,
        )
        out = out.at[jnp.asarray(los_idx)].set(cl_los)
    if lim_idx:
        lim_list = [ell_list[i] for i in lim_idx]
        cl_limber = limber_cl_te_from_sources(
            sources_t,
            sources_e,
            tau,
            tau0,
            k,
            p_r,
            jnp.asarray(lim_list, dtype=jnp.float64),
        )
        out = out.at[jnp.asarray(lim_idx)].set(cl_limber)
    return out


def hybrid_cls_tt_ee_te_from_sources(
    sources_t: jnp.ndarray,
    sources_e: jnp.ndarray,
    tau: jnp.ndarray,
    tau0: jnp.ndarray,
    k: jnp.ndarray,
    p_r: jnp.ndarray,
    ell: jnp.ndarray,
    *,
    limber_ell: int,
    ell_list: list[int] | None = None,
    compute_dtype=None,
    curv=None,
    k_max_host: float | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Fused TT/EE/TE: shared Bessel LOS for ℓ < limber_ell, Limber above (``curv``: K, curved
    Bessel LOS; the T/E Limber branch is flat-only).

    Same split and results as ``hybrid_cl_from_sources`` (TT, EE) plus
    ``hybrid_cl_te_from_sources``, with the Bessel part done once.
    """
    lim = int(limber_ell)
    if ell_list is None:
        ell_list = [int(e) for e in np.asarray(ell).tolist()]
    los_idx = [i for i, e in enumerate(ell_list) if e < lim]
    lim_idx = [i for i, e in enumerate(ell_list) if e >= lim]
    n = len(ell_list)
    tt = jnp.zeros(n, dtype=jnp.float64)
    ee = jnp.zeros(n, dtype=jnp.float64)
    te = jnp.zeros(n, dtype=jnp.float64)
    if los_idx:
        los_list = [ell_list[i] for i in los_idx]
        c_tt, c_ee, c_te = los_cls_tt_ee_te_from_sources(
            sources_t, sources_e, tau, tau0, k, p_r, ell_list=los_list,
            compute_dtype=compute_dtype, curv=curv, k_max_host=k_max_host,
        )
        ix = jnp.asarray(los_idx)
        tt, ee, te = tt.at[ix].set(c_tt), ee.at[ix].set(c_ee), te.at[ix].set(c_te)
    if lim_idx:
        if curv is not None:
            raise NotImplementedError("the T/E Limber branch is flat-only")
        lim_ell = jnp.asarray([ell_list[i] for i in lim_idx], dtype=jnp.float64)
        ix = jnp.asarray(lim_idx)
        tt = tt.at[ix].set(limber_cl_from_sources(sources_t, tau, tau0, k, p_r, lim_ell))
        ee = ee.at[ix].set(limber_cl_from_sources(sources_e, tau, tau0, k, p_r, lim_ell))
        te = te.at[ix].set(
            limber_cl_te_from_sources(sources_t, sources_e, tau, tau0, k, p_r, lim_ell)
        )
    return tt, ee, te


__all__ = [
    "LOS_COMPUTE_DTYPE",
    "LOS_ELL_CHUNK",
    "LOS_HOST_BESSEL_K_MIN",
    "LOS_K_CHUNK",
    "LOS_LOOP_ORDER",
    "bessel_table_vs_scipy_rel_err",
    "build_bessel_horner_table",
    "camb_dlnk",
    "camb_dpoints",
    "cl_from_delta_l_k",
    "cl_pp_from_weyl",
    "cl_pp_limber_camb",
    "eval_bessel_horner",
    "eval_bessel_horner_many",
    "hybrid_cl_from_sources",
    "hybrid_cl_te_from_sources",
    "hybrid_cls_tt_ee_te_from_sources",
    "lensing_source_from_weyl",
    "limber_cl_from_sources",
    "limber_cl_pp_from_psi",
    "limber_cl_te_from_sources",
    "limber_project_source",
    "los_cl_from_sources",
    "los_cl_te_from_sources",
    "los_cls_tt_ee_te_from_sources",
    "los_delta_ell",
    "spherical_bessel_jl_scan",
    "spherical_bessel_jn",
]
