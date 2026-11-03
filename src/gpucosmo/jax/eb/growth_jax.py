"""Linear P(k, z), σ8(z) and f σ8(z) from the linear spectra (``linear_pk_jax``), CAMB conventions.

* ``P_m`` = CAMB ``delta_tot`` × ``delta_tot`` (CDM + baryons + massive ν), ``P_cb`` =
  ``delta_nonu`` × ``delta_nonu`` [Mpc³];
* ``σ8(z)`` = CAMB ``get_sigma8``: top-hat R = 8 h⁻¹ Mpc on ``P_m``;
* ``f σ8(z)`` = CAMB ``get_fsigma8``: σ²_vd / σ_dd (Planck 2015 XIII, eq. 33), the cross
  spectrum of ``delta_tot`` with the Newtonian-gauge CDM velocity divergence
  ``v_newtonian_cdm`` = −k v_N/ℋ (``linear_pk_jax``'s ``v_c``).

The integrals run in ln k over the solver's grid (k ≤ 2/Mpc) with a cubic interpolation of
ln P, extended to k = 20/Mpc with the end slope; the extension adds < 1e-4 to σ8.
"""

from __future__ import annotations

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

K_EXT = 20.0  # 1/Mpc
N_QUAD = 2048


def _quad_nodes(k_lo: float, k_hi: float, n: int = N_QUAD):
    x, w = np.polynomial.legendre.leggauss(n)
    a, b = np.log(k_lo), np.log(k_hi)
    lnk = 0.5 * (b - a) * x + 0.5 * (a + b)
    return lnk, 0.5 * (b - a) * w


@lru_cache(maxsize=8)
def _interp_matrix(lnk_grid: tuple, lnk_q: tuple, extrap: str = "linear") -> np.ndarray:
    """(n_q, n_k) linear operator: natural cubic spline in ln k inside the grid; beyond it linear
    with the end slope (``extrap="linear"``) or constant ("const") (static; ln P_q = A @ ln P)."""
    from scipy.interpolate import CubicSpline

    g, q = np.asarray(lnk_grid), np.asarray(lnk_q)
    eye = np.eye(g.size)
    A = CubicSpline(g, eye, bc_type="natural", axis=0)(np.minimum(q, g[-1]))
    slope = (eye[-1] - eye[-2]) / (g[-1] - g[-2])
    out = q > g[-1]
    if extrap == "linear":
        A[out] = eye[-1][None, :] + (q[out] - g[-1])[:, None] * slope[None, :]
    return A


@lru_cache(maxsize=8)
def _static_spline(lnk_grid: tuple):
    from gpucosmo.jax.fullshape.interp import StaticSpline

    return StaticSpline(np.asarray(lnk_grid), "natural")


def tophat(x):
    return 3.0 * (jnp.sin(x) - x * jnp.cos(x)) / x**3


def sigma_r_table(k_grid: np.ndarray, pk: jnp.ndarray, r_mpc, pk_cross: jnp.ndarray | None = None):
    """σ_R² per row of ``pk`` (n_z, n_k) [Mpc³] on ``k_grid`` [1/Mpc]; ``r_mpc`` scalar or (n_z,).
    With ``pk_cross`` (same shape, may change sign): ∫ P_cross W² instead, interpolated linearly
    in the ratio P_cross / P (smooth) so the sign is kept."""
    lnk_grid = np.log(np.asarray(k_grid, dtype=np.float64))
    lnk, w = _quad_nodes(float(k_grid[0]), K_EXT)
    k = np.exp(lnk)
    A = jnp.asarray(_interp_matrix(tuple(lnk_grid), tuple(lnk)))
    p = jnp.exp(jnp.log(pk) @ A.T)  # (n_z, n_q)
    if pk_cross is not None:  # the ratio is smooth and flat at high k: held at its last value
        Ac = jnp.asarray(_interp_matrix(tuple(lnk_grid), tuple(lnk), "const"))
        p = p * ((pk_cross / pk) @ Ac.T)
    x = jnp.asarray(k)[None, :] * jnp.reshape(jnp.asarray(r_mpc, dtype=jnp.float64), (-1, 1))
    integ = jnp.asarray(k**3 * w / (2 * np.pi**2))[None, :] * tophat(x) ** 2
    return jnp.sum(integ * p, axis=-1)


def growth_block(spec: dict, h, k_grid: np.ndarray | None = None) -> dict:
    """{"pk_m", "pk_cb", "sigma8", "fsigma8"} from ``linear_cb_velocity_spectra`` output ``spec``
    (its ``pk_m``, ``pk_cb`` and the transfers' cross ``pk_m_vc``) on the static ``k_grid``
    (default ``linpk_k_grid()``: ``spec["k"]`` is traced under jit)."""
    if k_grid is None:
        from gpucosmo.jax.eb.linear_pk_jax import linpk_k_grid

        k_grid = linpk_k_grid()
    k = np.asarray(k_grid)
    r8 = 8.0 / h
    s_dd = sigma_r_table(k, spec["pk_m"], r8)
    s_vd = sigma_r_table(k, spec["pk_m"], r8, pk_cross=spec["pk_m_vc"])
    return {"pk_m": spec["pk_m"], "pk_cb": spec["pk_cb"], "sigma8": jnp.sqrt(s_dd),
            "fsigma8": s_vd / jnp.sqrt(s_dd)}




def _eh_shape(k, k_last, params):
    """ln[P_EH(k) / P_EH(k_last)] with P_EH ∝ k^{n_s} T_EH(k)² (Eisenstein–Hu no-wiggle)."""
    from gpucosmo.jax.eb.nonlinear_sources_jax import eh98_nowiggle_transfer

    return (params["n_s"] * jnp.log(k / k_last)
            + 2.0 * jnp.log(eh98_nowiggle_transfer(k, params) / eh98_nowiggle_transfer(k_last, params)))


def _hik_correction(k_last, pk_last, k_hi, pk_hi, params):
    """(ln k nodes, c (n_z, n_nodes)): c = ln of the evolved high-k modes ``pk_hi`` (n_z, n_hi) over the
    Eisenstein–Hu continuation from (k_last, ``pk_last`` (n_z, 1)); 0 at k_last."""
    k_hi = jnp.asarray(k_hi, dtype=jnp.float64)
    c = jnp.log(pk_hi) - (jnp.log(pk_last) + _eh_shape(k_hi, k_last, params)[None, :])
    lnk = jnp.concatenate([jnp.log(jnp.asarray(k_last, dtype=jnp.float64))[None], jnp.log(k_hi)])
    return lnk, jnp.concatenate([jnp.zeros_like(c[:, :1]), c], axis=1)


def extended_lnpk(k_grid: np.ndarray, pk: jnp.ndarray, k_out: np.ndarray, params: dict,
                  k_hi: np.ndarray | None = None, pk_hi: jnp.ndarray | None = None) -> jnp.ndarray:
    """ln P (n_z, n_out) on ``k_out`` [1/Mpc]: the solver's spectrum interpolated in ln k inside its
    grid and continued past the last k with P ∝ k^{n_s} T_EH(k)² (Eisenstein–Hu no-wiggle; as the
    CMB path's HMcode input, ``nonlinear_sources_jax``). With the evolved modes ``pk_hi`` (n_z, n_hi)
    at ``k_hi`` past the grid (``linear_pk_jax.hik_k_nodes``), their ratio to that continuation is
    interpolated linearly in ln k and held beyond the last node."""
    lnk_grid = np.log(np.asarray(k_grid, dtype=np.float64))
    lnk_out = np.log(np.asarray(k_out, dtype=np.float64))
    inside = lnk_out <= lnk_grid[-1]
    A = jnp.asarray(_interp_matrix(tuple(lnk_grid), tuple(np.where(inside, lnk_out, lnk_grid[-1])), "const"))
    lnp = jnp.log(pk) @ A.T
    k_last = float(k_grid[-1])
    kj = jnp.asarray(k_out)
    ext = jnp.log(pk[:, -1:]) + _eh_shape(kj, jnp.asarray(k_last), params)[None, :]
    if k_hi is not None and len(k_hi):
        lnk_c, c = _hik_correction(k_last, pk[:, -1:], k_hi, pk_hi, params)
        ext = ext + jax.vmap(lambda row: jnp.interp(jnp.asarray(lnk_out), lnk_c, row))(c)
    return jnp.where(jnp.asarray(inside)[None, :], lnp, ext)


def nonlinear_lnpk(k_grid: np.ndarray, pk_m: jnp.ndarray, z: np.ndarray, k_out: np.ndarray, params: dict,
                   version: str = "mead2020", k_hi: np.ndarray | None = None,
                   pk_hi: jnp.ndarray | None = None, logT_AGN=7.8) -> tuple[jnp.ndarray, jnp.ndarray]:
    """(ln P_lin, ln P_nl) of total matter on ``k_out`` [1/Mpc] at the static redshifts ``z``:
    HMcode ``version`` (CAMB ``halofit_version`` "mead2020" / "mead2016" / "mead2020_feedback"
    with ``logT_AGN`` = CAMB ``HMCode_logT_AGN``, may be traced; ``hmcode2020_jax``) on
    the linear spectrum ``pk_m`` (n_z, n_k) [Mpc³] extended with ``extended_lnpk`` (and its evolved
    high-k modes ``k_hi``, ``pk_hi``)."""
    from gpucosmo.jax.eb.hmcode2020_jax import hmcode2020_ratio, linear_table_k_h

    h = jnp.asarray(params["h"], dtype=jnp.float64)
    kt_h = linear_table_k_h()
    # Δ²(k) = k³ P / 2π² on HMcode's table (k in h/Mpc there; P extended in 1/Mpc)
    kt = jnp.asarray(kt_h) * h
    hik = k_hi is not None and len(k_hi)
    lnp_t = jnp.stack([extended_lnpk_traced(k_grid, pk_m[i], kt, params, *((k_hi, pk_hi[i]) if hik else ()))
                       for i in range(pk_m.shape[0])])
    log_d2 = lnp_t + 3.0 * jnp.log(kt)[None, :] - np.log(2 * np.pi**2)
    cosmo = dict(Omega_m=params["Omega_b"] + params["Omega_c"] + params.get("Omega_nu_massive_today", 0.0),
                 Omega_b=params["Omega_b"], Omega_c=params["Omega_c"],
                 Omega_nu=params.get("Omega_nu_massive_today", 0.0), Omega_v=params["Omega_de"], h=h,
                 n_s=params["n_s"], w0=params.get("w0", -1.0), wa=params.get("wa", 0.0),
                 T_cmb=params.get("T_CMB", 2.7255),
                 N_nu=jnp.asarray(params.get("num_massive_nu", 1), dtype=jnp.float64))
    a = 1.0 / (1.0 + np.asarray(z, dtype=np.float64))
    ratio = hmcode2020_ratio(jnp.asarray(k_out) / h, a, log_d2, cosmo, version=version,
                             logT_AGN=logT_AGN)  # (n_z, n_out)
    lnp_lin = extended_lnpk(k_grid, pk_m, k_out, params, *((k_hi, pk_hi) if hik else ()))
    return lnp_lin, lnp_lin + 2.0 * jnp.log(ratio)


def extended_lnpk_traced(k_grid: np.ndarray, pk_row: jnp.ndarray, k_q: jnp.ndarray, params: dict,
                         k_hi: np.ndarray | None = None, pk_hi_row: jnp.ndarray | None = None) -> jnp.ndarray:
    """``extended_lnpk`` for one spectrum at traced wavenumbers ``k_q`` (HMcode's table k depends
    on h): natural cubic spline in ln k inside the grid (as ``extended_lnpk``; a linear one damped
    the BAO wiggles on HMcode's table: P_nl/P_lin −5e-3 vs CAMB at k = 0.15/Mpc), Eisenstein–Hu
    continuation outside (times the evolved high-k modes' ratio to it, ``k_hi``, ``pk_hi_row``)."""
    lnk_np = np.log(np.asarray(k_grid, dtype=np.float64))
    lnk_grid = jnp.asarray(lnk_np)
    lnp = jnp.log(pk_row)
    lq = jnp.log(k_q)
    inner = _static_spline(tuple(lnk_np))(lnp, jnp.clip(lq, lnk_np[0], lnk_np[-1]))
    k_last = jnp.asarray(float(k_grid[-1]))
    ext = lnp[-1] + _eh_shape(k_q, k_last, params)
    if k_hi is not None and len(k_hi):
        lnk_c, c = _hik_correction(k_last, pk_row[None, -1:], k_hi, pk_hi_row[None, :], params)
        ext = ext + jnp.interp(lq, lnk_c, c[0])
    lo = lnp[0] + (params["n_s"]) * (lq - lnk_grid[0])  # below k_min: primordial slope (P ∝ k^{n_s})
    return jnp.where(lq > lnk_grid[-1], ext, jnp.where(lq < lnk_grid[0], lo, inner))


__all__ = ["extended_lnpk", "growth_block", "nonlinear_lnpk", "sigma_r_table", "tophat"]
