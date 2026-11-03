#!/usr/bin/env python3
"""Tracer kernels for JAX-native Limber GGL theory."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from gpucosmo.jax.limber_quad import _GL25_W_J, _GL25_X_J
from gpucosmo.jax.power import CosmologyParams
from gpucosmo.jax.splines import akima_interp_jax

jax.config.update("jax_enable_x64", True)

_C_LIGHT = 299792.458  # km/s
# pyccl.physical_constants.RHO_CRITICAL [h^2 M_sun / Mpc^3]
_RHO_CRITICAL = 277536627203.6856
_C1_IA = 5.0e-14
_CCL_TRACER_N_CHI = 256
# The legacy WL kernel below carries H0^2/c (km/s/Mpc) instead of CCL's
# (H0/c)^2, i.e. it is CCL x c/2, and the Limber integrators divide by c/2.
# Density-type kernels (p(z) H(z)/c) sharing that integrator need the same
# c/2 so that IA x density matches CCL exactly. This used to be a fitted
# "geometry scale" (~1.505e5); it is the analytic unit conversion.
_IA_LIMBER_UNIT_SCALE = 0.5 * _C_LIGHT


def ia_limber_geometry_scale_cached(
    background_backend: str = "ccl",
) -> float:
    """Dimensionless residual IA scale (exactly 1: normalisation is analytic)."""
    del background_backend
    return 1.0


def normalized_nz(z: jnp.ndarray, nz: jnp.ndarray) -> jnp.ndarray:
    """Normalize n(z) to unit integral using composite Gauss-Legendre (CCL Limber grid)."""
    nz_nonneg = jnp.maximum(nz, 0.0)
    norm = _gk_integrate_z(nz_nonneg, z)
    return jnp.where(norm > 0.0, nz_nonneg / norm, nz_nonneg)


def interpolate_nz(z_grid: jnp.ndarray, nz_grid: jnp.ndarray, z_eval: jnp.ndarray) -> jnp.ndarray:
    """Linear interpolation of n(z) onto evaluation redshifts."""
    return jnp.interp(z_eval, z_grid, nz_grid, left=0.0, right=0.0)


def build_background_grids_ccl(
    z_grid: np.ndarray,
    params: CosmologyParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tabulate chi(z), E(z), H(z) from CCL/CAMB on a fixed redshift grid."""
    import pyccl as ccl

    from gpucosmo.jax.power import _rounded_cosmology_key, _warmed_ccl_linear_cosmo

    oc, ob, h, sigma8, n_s = _rounded_cosmology_key(params)
    cosmo = _warmed_ccl_linear_cosmo(oc, ob, h, sigma8, n_s)
    z_vals = np.asarray(z_grid, dtype=float)
    a_grid = 1.0 / (1.0 + z_vals)
    chi = np.asarray(
        [ccl.comoving_radial_distance(cosmo, float(a_val)) for a_val in a_grid],
        dtype=float,
    )
    ez = np.asarray(
        [ccl.h_over_h0(cosmo, float(a_val)) for a_val in a_grid],
        dtype=float,
    )
    hz = params.h * 100.0 * ez
    return chi, ez, hz


def build_background_grids(
    z_max: float,
    *,
    h: float,
    Omega_c: float,
    Omega_b: float,
    n_z: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute flat-LCDM background quantities on a uniform z grid."""
    z_grid = np.linspace(0.0, z_max, n_z)
    Omega_m = Omega_c + Omega_b
    Omega_de = 1.0 - Omega_m
    ez = np.sqrt(Omega_m * (1.0 + z_grid) ** 3 + Omega_de)
    # Comoving distance in Mpc/h: (c / H0) ∫ dz / E(z), with H0 = 100 h km/s/Mpc.
    chi = (_C_LIGHT / (100.0 * h)) * np.insert(
        np.cumsum(
            0.5 * (1.0 / ez[:-1] + 1.0 / ez[1:]) * np.diff(z_grid)
        ),
        0,
        0.0,
    )
    hz = h * 100.0 * ez
    return z_grid, chi, ez, hz


def build_growth_grid(z_grid: np.ndarray, params: CosmologyParams) -> np.ndarray:
    """Tabulate CCL linear growth factor D(z) on a redshift grid."""
    import pyccl as ccl

    cosmo = ccl.Cosmology(
        Omega_c=params.Omega_c,
        Omega_b=params.Omega_b,
        h=params.h,
        sigma8=params.sigma8,
        n_s=params.n_s,
        transfer_function="boltzmann_camb",
        matter_power_spectrum="halofit",
    )
    a_grid = 1.0 / (1.0 + np.asarray(z_grid, dtype=float))
    return np.asarray(
        [ccl.growth_factor(cosmo, float(a_val)) for a_val in a_grid],
        dtype=float,
    )


def _chi_at_z(z_eval: jnp.ndarray, z_grid: jnp.ndarray, chi_grid: jnp.ndarray) -> jnp.ndarray:
    return jnp.interp(z_eval, z_grid, chi_grid, left=0.0, right=chi_grid[-1])


def _z_at_chi(chi_eval: jnp.ndarray, chi_grid: jnp.ndarray, z_grid: jnp.ndarray) -> jnp.ndarray:
    """Akima z(chi) from CCL background tabulation (monotone in chi)."""
    return akima_interp_jax(chi_grid, z_grid, jnp.clip(chi_eval, chi_grid[0], chi_grid[-1]))


def build_chi_kernel_grid(chi_max: jnp.ndarray | float, n_chi: int = _CCL_TRACER_N_CHI) -> jnp.ndarray:
    """Equi-spaced comoving-distance grid matching CCL tracer sampling."""
    return jnp.linspace(0.0, chi_max, int(n_chi))


def _gk_integrate_chi(y: jnp.ndarray, chi: jnp.ndarray) -> jnp.ndarray:
    """Composite 25-point GL on chi (CCL tracer integral variable)."""

    def seg(y0, y1, c0, c1):
        half = 0.5 * (c1 - c0)
        mid = 0.5 * (c0 + c1)
        nodes = mid + half * _GL25_X_J

        def f(cp):
            t = (cp - c0) / jnp.where(c1 > c0, c1 - c0, 1.0)
            return y0 + t * (y1 - y0)

        vals = jax.vmap(f)(nodes)
        return half * jnp.sum(_GL25_W_J * vals)

    return jnp.sum(jax.vmap(seg)(y[:-1], y[1:], chi[:-1], chi[1:]))


def build_background_grids_jax(
    z_max: float,
    *,
    h: float,
    Omega_c: float,
    Omega_b: float,
    n_z: int = 512,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """JAX version of build_background_grids."""
    z_grid = jnp.linspace(0.0, z_max, n_z)
    Omega_m = Omega_c + Omega_b
    Omega_de = 1.0 - Omega_m
    ez = jnp.sqrt(Omega_m * (1.0 + z_grid) ** 3 + Omega_de)
    integrand = 0.5 * (1.0 / ez[:-1] + 1.0 / ez[1:]) * jnp.diff(z_grid)
    chi = (_C_LIGHT / (100.0 * h)) * jnp.concatenate([jnp.array([0.0]), jnp.cumsum(integrand)])
    hz = h * 100.0 * ez
    return z_grid, chi, ez, hz


def _spin2_ell_factor_wl(ell) -> jnp.ndarray:
    """Spin-2 Limber prefactor for the lensing kernel (der_angles=2 in CCL)."""
    return jnp.sqrt((ell - 1.0) * ell * (ell + 1.0) * (ell + 2.0)) / (ell + 0.5) ** 2


def _convergence_ell_factor(ell) -> jnp.ndarray:
    """Convergence / magnification Limber prefactor (CCL der_angles=1, der_bessel=-1)."""
    return ell * (ell + 1.0) / (ell + 0.5) ** 2


def _rsd_limber_ell_factor(ell) -> jnp.ndarray:
    """CCL Limber weight for der_bessel=2 (RSD), local χ_{ℓ+1}≈χ_ℓ approximation.

    Matches ``transfer_limber_single`` in CCL ``ccl_cls.c`` with ``pk_ratio=1``:
    ``2√((ℓ+½)/(ℓ+3/2))/(ℓ+3/2) − (¼+2ℓ)/(ℓ+½)²``.
    Approaches 0 at high ℓ (unlike naive Kaiser → 1).
    """
    lp1h = ell + 0.5
    lp3h = ell + 1.5
    return 2.0 * jnp.sqrt(lp1h / lp3h) / lp3h - (0.25 + 2.0 * ell) / (lp1h * lp1h)


def alpha_mag_to_ccl_s(alpha_mag: float) -> float:
    """HSC α_mag → CCL mag slope s (cobaya_ggl_theory.alpha_mag_to_ccl_s)."""
    return float(alpha_mag) / 2.5


def number_counts_kernel(
    z: jnp.ndarray,
    *,
    z_nz: jnp.ndarray,
    nz: jnp.ndarray,
    bias: float,
    hz: jnp.ndarray,
    has_rsd: bool = False,
    f_growth: jnp.ndarray | None = None,
    ell: jnp.ndarray | float | None = None,
    chi: jnp.ndarray | None = None,
    mag_s: float | None = None,
    Omega_m: float | None = None,
    h: float | None = None,
    z_grid: jnp.ndarray | None = None,
    chi_grid: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """CCL-compatible number-counts kernel with optional RSD + magnification.

    Density term: n(z) b H(z)/c  (CCL NumberCountsTracer).
    RSD: n(z) (−f) H(z)/c × CCL ``der_bessel=2`` Limber weight (see
    ``_rsd_limber_ell_factor``). When ``ell`` is None the RSD term is omitted
    (high-ℓ Limber weight → 0); pass ``ell`` for a correct (n_ell, n_z) kernel.
    Magnification: (5s−2) lensing kernel when ``mag_s`` is set (CCL mag_bias;
    s = α_mag/2.5). Callers that Limber-project mag must still apply
    ``_convergence_ell_factor`` (der_angles=1, der_bessel=-1).
    """
    nz_norm = normalized_nz(z_nz, nz)
    n_z = interpolate_nz(z_nz, nz_norm, z)
    dens = n_z * bias * hz / _C_LIGHT
    out = dens
    if has_rsd and ell is not None:
        if f_growth is None:
            f_use = jnp.ones_like(z)
        else:
            f_use = f_growth
        ell_arr = jnp.asarray(ell, dtype=float)
        ell_fac = _rsd_limber_ell_factor(ell_arr)
        base_rsd = n_z * f_use * hz / _C_LIGHT  # transfer supplies (−f); fac has sign
        if jnp.ndim(ell_arr) > 0:
            # dens: transfer b already in dens; RSD: kernel×(−f)×ell_weight
            rsd = -base_rsd[None, :] * ell_fac[:, None]
            out = dens[None, :] + rsd
            if mag_s is not None and Omega_m is not None and h is not None:
                mag = _magnification_kernel(
                    z,
                    z_nz=z_nz,
                    nz=nz_norm,
                    mag_s=mag_s,
                    Omega_m=Omega_m,
                    h=h,
                    z_grid=z if z_grid is None else z_grid,
                    chi_grid=chi if chi_grid is None else chi_grid,
                )
                mag_fac = _convergence_ell_factor(ell_arr)
                out = out + mag[None, :] * mag_fac[:, None]
            return out
        rsd = -base_rsd * float(ell_fac)
        out = dens + rsd
    if mag_s is not None and Omega_m is not None and h is not None:
        mag = _magnification_kernel(
            z,
            z_nz=z_nz,
            nz=nz_norm,
            mag_s=mag_s,
            Omega_m=Omega_m,
            h=h,
            z_grid=z if z_grid is None else z_grid,
            chi_grid=chi if chi_grid is None else chi_grid,
        )
        if ell is not None:
            ell_arr = jnp.asarray(ell, dtype=float)
            mag_fac = _convergence_ell_factor(ell_arr)
            if jnp.ndim(ell_arr) > 0:
                return dens[None, :] + mag[None, :] * mag_fac[:, None]
            mag = mag * float(mag_fac)
        out = out + mag
    return out


def _magnification_kernel(
    z: jnp.ndarray,
    *,
    z_nz: jnp.ndarray,
    nz: jnp.ndarray,
    mag_s: float,
    Omega_m: float,
    h: float,
    z_grid: jnp.ndarray,
    chi_grid: jnp.ndarray,
) -> jnp.ndarray:
    """CCL magnification bias kernel ∝ (5s−2) × lensing efficiency for lenses."""
    # Reuse WL geometry with source=lens n(z); prefactor matches CCL mag term.
    prefactor = (5.0 * mag_s - 2.0) * 1.5 * (h * 100.0) ** 2 * Omega_m / (2.0 * _C_LIGHT)

    def kernel_at_z(z_lens: jnp.ndarray) -> jnp.ndarray:
        chi_lens = _chi_at_z(z_lens, z_grid, chi_grid)

        def integrand(z_prime: jnp.ndarray) -> jnp.ndarray:
            chi_prime = _chi_at_z(z_prime, z_grid, chi_grid)
            n_prime = interpolate_nz(z_nz, nz, z_prime)
            return jnp.where(
                z_prime >= z_lens,
                n_prime * jnp.clip(chi_prime - chi_lens, 0.0) / jnp.clip(chi_prime, 1.0),
                0.0,
            )

        radial = _gk_integrate_z(integrand(z_grid), z_grid)
        return prefactor * (1.0 + z_lens) * chi_lens * radial

    return jax.vmap(kernel_at_z)(z)


def growth_rate_f_grid(
    z: jnp.ndarray,
    *,
    Omega_m: float,
) -> jnp.ndarray:
    """Approximate growth rate f(z) ≈ Ω_m(z)^0.55 (Linder)."""
    a = 1.0 / (1.0 + z)
    Omega_de = 1.0 - Omega_m
    ez2 = Omega_m * a ** (-3) + Omega_de
    om_z = Omega_m * a ** (-3) / jnp.clip(ez2, 1.0e-30)
    return om_z**0.55


def _gk_integrate_z(y: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
    """Composite 25-point GL on z (matches Limber integrator in fast.py)."""

    def seg(y0, y1, z0, z1):
        half = 0.5 * (z1 - z0)
        mid = 0.5 * (z0 + z1)
        nodes = mid + half * _GL25_X_J

        def f(zp):
            t = (zp - z0) / jnp.where(z1 > z0, z1 - z0, 1.0)
            return y0 + t * (y1 - y0)

        vals = jax.vmap(f)(nodes)
        return half * jnp.sum(_GL25_W_J * vals)

    return jnp.sum(jax.vmap(seg)(y[:-1], y[1:], z[:-1], z[1:]))


def weak_lensing_kernel_base(
    z: jnp.ndarray,
    *,
    z_nz: jnp.ndarray,
    nz: jnp.ndarray,
    z_grid: jnp.ndarray,
    chi_grid: jnp.ndarray,
    Omega_m: float,
    h: float,
) -> jnp.ndarray:
    """Ell-independent WL kernel on the Limber z grid; shape (n_z,)."""
    nz_norm = normalized_nz(z_nz, nz)
    prefactor = 1.5 * (h * 100.0) ** 2 * Omega_m / (2.0 * _C_LIGHT)

    def kernel_at_z(z_lens: jnp.ndarray) -> jnp.ndarray:
        chi_lens = _chi_at_z(z_lens, z_grid, chi_grid)

        def integrand(z_prime: jnp.ndarray) -> jnp.ndarray:
            chi_prime = _chi_at_z(z_prime, z_grid, chi_grid)
            n_prime = interpolate_nz(z_nz, nz_norm, z_prime)
            return jnp.where(
                z_prime >= z_lens,
                n_prime * jnp.clip(chi_prime - chi_lens, 0.0) / jnp.clip(chi_prime, 1.0),
                0.0,
            )

        radial = _gk_integrate_z(integrand(z_grid), z_grid)
        return prefactor * (1.0 + z_lens) * chi_lens * radial

    return jax.vmap(kernel_at_z)(z)


def weak_lensing_kernel(
    z: jnp.ndarray,
    *,
    z_nz: jnp.ndarray,
    nz: jnp.ndarray,
    ell,
    z_grid: jnp.ndarray,
    chi_grid: jnp.ndarray,
    Omega_m: float,
    h: float,
) -> jnp.ndarray:
    """Standard Limber weak-lensing kernel matching CCL conventions."""
    base = weak_lensing_kernel_base(
        z,
        z_nz=z_nz,
        nz=nz,
        z_grid=z_grid,
        chi_grid=chi_grid,
        Omega_m=Omega_m,
        h=h,
    )
    ell_factor = _spin2_ell_factor_wl(ell)
    if jnp.ndim(ell) == 0:
        return ell_factor * base
    return ell_factor[:, None] * base[None, :]


def ia_density_kernel(
    z: jnp.ndarray,
    *,
    z_nz: jnp.ndarray,
    nz: jnp.ndarray,
    ell,
    A_IA: float,
    alpha_IA: float,
    z_pivot_IA: float,
    Omega_m: float,
    hz: jnp.ndarray,
    growth: jnp.ndarray,
    omega_m_scaling: bool = False,
) -> jnp.ndarray:
    """CCL-compatible IA density tracer kernel (WeakLensingTracer ia_bias)."""
    return ia_density_kernel_batched(
        z,
        z_nz=z_nz,
        nz=nz,
        ell=ell,
        A_IA=A_IA,
        alpha_IA=alpha_IA,
        z_pivot_IA=z_pivot_IA,
        Omega_m=Omega_m,
        hz=hz,
        growth=growth,
        omega_m_scaling=omega_m_scaling,
    )


def ia_density_kernel_batched(
    z: jnp.ndarray,
    *,
    z_nz: jnp.ndarray,
    nz: jnp.ndarray,
    ell: jnp.ndarray,
    A_IA: float,
    alpha_IA: float,
    z_pivot_IA: float,
    Omega_m: float,
    hz: jnp.ndarray,
    growth: jnp.ndarray,
    geometry_scale: float | None = None,
    omega_m_scaling: bool = False,
) -> jnp.ndarray:
    """IA density kernel for all ell; shape (n_ell, n_z).

    ``geometry_scale`` is a dimensionless extra factor (default 1).
    ``omega_m_scaling=True``: historical ``A_IA * Omega_m`` amplitude (CCL
    convention otherwise).
    """
    nz_norm = normalized_nz(z_nz, nz)
    n_z = interpolate_nz(z_nz, nz_norm, z)
    z_factor = ((1.0 + z) / (1.0 + z_pivot_IA)) ** alpha_IA
    tmp_a = (A_IA * Omega_m if omega_m_scaling else A_IA) * z_factor
    rho_m = _RHO_CRITICAL * Omega_m
    transfer = -tmp_a * _C1_IA * rho_m / jnp.clip(growth, 1.0e-30)
    base = n_z * hz / _C_LIGHT
    scale = _IA_LIMBER_UNIT_SCALE * (1.0 if geometry_scale is None else geometry_scale)
    kernel = scale * base * transfer
    ell_factor = _spin2_ell_factor_wl(ell)
    if jnp.ndim(ell) == 0:
        return ell_factor * kernel
    return ell_factor[:, None] * kernel[None, :]


__all__ = [
    "build_background_grids",
    "build_background_grids_ccl",
    "build_background_grids_jax",
    "build_chi_kernel_grid",
    "build_growth_grid",
    "ia_density_kernel",
    "ia_density_kernel_batched",
    "ia_limber_geometry_scale_cached",
    "normalized_nz",
    "number_counts_kernel",
    "weak_lensing_kernel",
    "weak_lensing_kernel_base",
]
