#!/usr/bin/env python3
"""CAMB/CCL matter power spectra and cosmology bundles for JAX GGL theory."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Tuple


from gpucosmo.jax.nonlinear import (
    DEFAULT_NONLINEAR_PRESCRIPTION,
    GGL_HMCODE_CAMB_VERSION,
    NonlinearPrescription,
)

BoltzmannBackend = Literal["ccl", "clax", "gcamb", "cosmopower", "jax_eb"]
BackgroundBackend = Literal["analytic", "ccl"]
# jax_eb: the package's own Boltzmann solver (CAMB-equivalent, differentiable, no emulator);
# the others are opt-in (CosmoPower emulator, CCL/CAMB on the CPU, clax, gCAMB)
DEFAULT_BOLTZMANN_BACKEND: BoltzmannBackend = "jax_eb"
DEFAULT_BACKGROUND_BACKEND: BackgroundBackend = "ccl"

import jax
import jax.numpy as jnp
import numpy as np

from gpucosmo.jax.splines import akima_interp_jax, ccl_spline_params, pk2d_eval_from_z_jax


def interp_log_pk_bilinear(
    log_k: jnp.ndarray,
    z_tab: jnp.ndarray,
    log_pk: jnp.ndarray,
    k: jnp.ndarray,
    z_eval: jnp.ndarray,
) -> jnp.ndarray:
    """Bilinear interpolation in log(k) and z (pure JAX, arbitrary broadcast shape)."""
    log_k_eval = jnp.log(jnp.maximum(k, 1.0e-30))
    z_c = jnp.clip(z_eval, z_tab[0], z_tab[-1])
    lk_c = jnp.clip(log_k_eval, log_k[0], log_k[-1])

    iz = jnp.clip(jnp.searchsorted(z_tab, z_c, side="right") - 1, 0, z_tab.size - 2)
    ik = jnp.clip(jnp.searchsorted(log_k, lk_c, side="right") - 1, 0, log_k.size - 2)

    z0 = z_tab[iz]
    z1 = z_tab[iz + 1]
    tz = jnp.where(z1 > z0, (z_c - z0) / (z1 - z0), 0.0)

    k0 = log_k[ik]
    k1 = log_k[ik + 1]
    tk = jnp.where(k1 > k0, (lk_c - k0) / (k1 - k0), 0.0)

    p00 = log_pk[iz, ik]
    p01 = log_pk[iz, ik + 1]
    p10 = log_pk[iz + 1, ik]
    p11 = log_pk[iz + 1, ik + 1]
    p0 = p00 + tk * (p01 - p00)
    p1 = p10 + tk * (p11 - p10)
    return p0 + tz * (p1 - p0)


@dataclass(frozen=True)
class CosmologyParams:
    """Flat-LCDM parameters used by the HSC GGL likelihoods."""

    Omega_c: float
    Omega_b: float
    h: float
    sigma8: float
    n_s: float

    @property
    def Omega_m(self) -> float:
        return self.Omega_c + self.Omega_b


@dataclass(frozen=True)
class CosmologyBundle:
    """Cached cosmology-dependent tables for jitted Limber integration."""

    params: CosmologyParams
    log_k: np.ndarray
    z_pk: np.ndarray
    log_pk: np.ndarray
    z_grid: np.ndarray
    chi_grid: np.ndarray
    ez_grid: np.ndarray
    hz_grid: np.ndarray
    growth_grid: np.ndarray

    def as_jax(self) -> dict[str, jnp.ndarray]:
        return {
            "log_k": jnp.asarray(self.log_k),
            "z_pk": jnp.asarray(self.z_pk),
            "log_pk": jnp.asarray(self.log_pk),
            "z_grid": jnp.asarray(self.z_grid),
            "chi_grid": jnp.asarray(self.chi_grid),
            "ez_grid": jnp.asarray(self.ez_grid),
            "hz_grid": jnp.asarray(self.hz_grid),
            "growth_grid": jnp.asarray(self.growth_grid),
        }


@dataclass
class MatterPowerTable:
    """Log-spaced P(k, z) lookup table sampled from CAMB/CCL."""

    log_k: np.ndarray
    z: np.ndarray
    log_pk: np.ndarray  # shape (n_z, n_k)

    def pk(self, k: np.ndarray | float, z: np.ndarray | float) -> np.ndarray:
        """Interpolate nonlinear P(k, z) in NumPy."""
        k_arr = np.asarray(k, dtype=float)
        z_arr = np.asarray(z, dtype=float)
        log_k_eval = np.log(np.maximum(k_arr, 1.0e-30))
        z_clipped = np.clip(z_arr, self.z[0], self.z[-1])
        log_k_clipped = np.clip(log_k_eval, self.log_k[0], self.log_k[-1])

        iz = np.searchsorted(self.z, z_clipped, side="right") - 1
        iz = np.clip(iz, 0, len(self.z) - 2)
        z0 = self.z[iz]
        z1 = self.z[iz + 1]
        tz = np.where(z1 > z0, (z_clipped - z0) / (z1 - z0), 0.0)

        ik = np.searchsorted(self.log_k, log_k_clipped, side="right") - 1
        ik = np.clip(ik, 0, len(self.log_k) - 2)
        k0 = self.log_k[ik]
        k1 = self.log_k[ik + 1]
        tk = np.where(k1 > k0, (log_k_clipped - k0) / (k1 - k0), 0.0)

        p00 = self.log_pk[iz, ik]
        p01 = self.log_pk[iz, ik + 1]
        p10 = self.log_pk[iz + 1, ik]
        p11 = self.log_pk[iz + 1, ik + 1]
        p0 = p00 + tk * (p01 - p00)
        p1 = p10 + tk * (p11 - p10)
        return np.exp(p0 + tz * (p1 - p0))


def ccl_dense_pk_table_shape() -> tuple[int, int, float, float]:
    """Return (n_k, n_z, k_min, k_max) matching CCL spline resolution."""
    params = ccl_spline_params()
    k_min = float(params["K_MIN"])
    k_max = float(params["K_MAX"])
    n_k = int(np.ceil(np.log10(k_max / k_min) * float(params["N_K"]))) + 1
    return n_k, 80, k_min, k_max


def ccl_pk_k_bounds() -> tuple[float, float]:
    """Return (k_min, k_max) from CCL spline parameters."""
    params = ccl_spline_params()
    return float(params["K_MIN"]), float(params["K_MAX"])


def upsample_matter_table_to_ccl_dense(table: MatterPowerTable) -> MatterPowerTable:
    """Upsample a MatterPowerTable onto the CCL-dense log-k grid."""
    from gpucosmo.jax.splines import resample_logx_logy_numpy

    n_k, _, k_min, k_max = ccl_dense_pk_table_shape()
    k_dst = np.geomspace(k_min, k_max, n_k)
    k_src = np.exp(table.log_k)
    log_pk_dense = np.empty((table.z.size, n_k), dtype=float)
    for iz in range(table.z.size):
        log_pk_dense[iz, :] = np.log(
            resample_logx_logy_numpy(k_src, np.exp(table.log_pk[iz, :]), k_dst)
        )
    return MatterPowerTable(
        log_k=np.log(k_dst),
        z=table.z,
        log_pk=log_pk_dense,
    )


def build_dense_pk_table_from_ccl(
    params: CosmologyParams,
    *,
    z_max: float,
    n_z: int | None = None,
    nonlinear: bool = True,
    n_k: int | None = None,
    nonlinear_prescription: str = "halofit",
    hmcode_logT_AGN: float = 7.7,
    hmcode_version: str = "mead2020_feedback",
) -> MatterPowerTable:
    """Sample P(k, z) from CCL Pk2D at CCL-like (or coarser) k resolution.

    Uses vectorized per-``a`` Pk2D evaluation (no Python ``nonlin_matter_power``
    per-z wrapper beyond one call per redshift). ``nonlinear_prescription=
    'hmcode'`` requests CAMB ``halofit_version=hmcode_version`` through
    ``matter_power_spectrum='camb'``; the default ``'mead2020_feedback'`` (with
    ``hmcode_logT_AGN``) is the DES/hybrid baseline, while the gG/gI pipelines
    pass :data:`gpucosmo.jax.nonlinear.GGL_HMCODE_CAMB_VERSION` (``'mead2020'``)
    to match :class:`gpucosmo.ccl.theory.GGLTheory`.
    """
    import pyccl as ccl

    from gpucosmo.jax.nonlinear import ccl_nonlinear_config

    n_k_dense, default_n_z, k_min, k_max = ccl_dense_pk_table_shape()
    n_z_use = int(n_z if n_z is not None else default_n_z)
    n_k_use = int(n_k if n_k is not None else n_k_dense)
    matter_power_spectrum, extra_parameters = ccl_nonlinear_config(
        nonlinear_prescription if nonlinear else "linear",
        hmcode_version=hmcode_version,
        hmcode_logT_AGN=hmcode_logT_AGN,
    )
    cosmo = ccl.Cosmology(
        Omega_c=params.Omega_c,
        Omega_b=params.Omega_b,
        h=params.h,
        sigma8=params.sigma8,
        n_s=params.n_s,
        transfer_function="boltzmann_camb",
        matter_power_spectrum=matter_power_spectrum,
        extra_parameters=extra_parameters,
    )
    z_grid = np.linspace(0.0, z_max, n_z_use)
    k_grid = np.geomspace(k_min, k_max, n_k_use)
    a_grid = 1.0 / (1.0 + z_grid)
    if nonlinear:
        pk2d = cosmo.get_nonlin_power()
    else:
        pk2d = cosmo.get_linear_power()
    pk_grid = np.vstack(
        [np.asarray(pk2d(k_grid, float(a_val)), dtype=float) for a_val in a_grid]
    )
    return MatterPowerTable(
        log_k=np.log(k_grid),
        z=z_grid,
        log_pk=np.log(np.maximum(pk_grid, 1.0e-30)),
    )


def jax_eb_dense_log_pk(
    Omega_c, Omega_b, h, n_s, sigma8, z_grid, log_k_grid, *, cfg=None, nonlinear: bool = True,
):
    """(log P_lin, log P_nl) on (z_grid, log_k_grid) from the jax_eb Pk2D pair.

    Jittable / differentiable in the cosmological parameters. The Pk2D is the
    CAMB transfer run + CCL assembly of :mod:`gpucosmo.jax.eb.matter_pk_jax`
    (HMcode-2020 = CAMB ``mead2020``), evaluated exactly as CCL evaluates its
    splines; the grid is the CCL-dense table the ``ccl`` backend also uses.
    """
    from gpucosmo.jax.eb.matter_pk_jax import MatterPkConfig, matter_pk2d_pair

    cfg = cfg or MatterPkConfig()
    pk_lin, pk_nl = matter_pk2d_pair(Omega_c, Omega_b, h, n_s, sigma8, cfg=cfg)
    lk = jnp.asarray(log_k_grid, dtype=jnp.float64)
    a = 1.0 / (1.0 + jnp.asarray(z_grid, dtype=jnp.float64))
    lin = jax.vmap(lambda aa: jnp.log(pk_lin.eval_lk(lk, aa)))(a)
    if not nonlinear:
        return lin, lin
    nl = jax.vmap(lambda aa: jnp.log(pk_nl.eval_lk(lk, aa)))(a)
    return lin, nl


def build_dense_pk_tables_from_jax_eb(
    params: CosmologyParams,
    *,
    z_max: float,
    n_z: int | None = None,
    n_k: int | None = None,
    cfg=None,
) -> tuple[MatterPowerTable, MatterPowerTable]:
    """``(pk_lin, pk_nl)`` MatterPowerTables on the CCL-dense grid (jax_eb)."""
    n_k_dense, default_n_z, k_min, k_max = ccl_dense_pk_table_shape()
    z_grid = np.linspace(0.0, z_max, int(n_z if n_z is not None else default_n_z))
    log_k = np.log(np.geomspace(k_min, k_max, int(n_k if n_k is not None else n_k_dense)))
    lin, nl = jax_eb_dense_log_pk(
        params.Omega_c, params.Omega_b, params.h, params.n_s, params.sigma8,
        z_grid, log_k, cfg=cfg,
    )
    mk = lambda lp: MatterPowerTable(log_k=log_k, z=z_grid, log_pk=np.asarray(lp))  # noqa: E731
    return mk(lin), mk(nl)


def build_dense_pk_table_from_camb(
    params: CosmologyParams,
    *,
    z_max: float,
    n_z: int | None = None,
    n_k: int | None = None,
    nonlinear: bool = True,
    hmcode_logT_AGN: float = 7.7,
    k_max_compute: float = 100.0,
) -> MatterPowerTable:
    """Direct CAMB ``mead2020_feedback`` matter P(k,z) (Hybrid baseline).

    Avoids CCL cosmology wrapper overhead; samples CAMB's matter power
    spectrum on the requested grid and rescales to the target ``sigma8``.
    """
    from camb import model

    n_k_dense, default_n_z, k_min, k_max = ccl_dense_pk_table_shape()
    n_z_use = int(n_z if n_z is not None else default_n_z)
    n_k_use = int(n_k if n_k is not None else n_k_dense)
    z_asc = np.linspace(0.0, float(z_max), n_z_use)
    # CAMB wants redshifts descending for set_matter_power.
    z_desc = z_asc[::-1]
    import camb

    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=100.0 * params.h,
        ombh2=params.Omega_b * params.h**2,
        omch2=params.Omega_c * params.h**2,
        omk=0.0,
    )
    pars.InitPower.set_params(As=2.0e-9, ns=params.n_s)
    pars.set_matter_power(
        redshifts=z_desc,
        kmax=float(min(k_max_compute, k_max)),
    )
    if nonlinear:
        pars.NonLinear = model.NonLinear_both
        pars.NonLinearModel.set_params(
            halofit_version="mead2020_feedback",
            HMCode_logT_AGN=float(hmcode_logT_AGN),
        )
    else:
        pars.NonLinear = model.NonLinear_none
    results = camb.get_results(pars)
    results.calc_power_spectra(pars)
    kh, z_out, pk = results.get_matter_power_spectrum(
        minkh=float(k_min),
        maxkh=float(min(k_max_compute, k_max)),
        npoints=n_k_use,
    )
    s8 = float(results.get_sigma8_0())
    if s8 > 0.0:
        pk = np.asarray(pk, dtype=float) * (params.sigma8 / s8) ** 2
    else:
        pk = np.asarray(pk, dtype=float)
    # CAMB returns z descending matching input; reorder to ascending.
    z_out = np.asarray(z_out, dtype=float)
    order = np.argsort(z_out)
    return MatterPowerTable(
        log_k=np.log(np.asarray(kh, dtype=float)),
        z=z_out[order],
        log_pk=np.log(np.maximum(pk[order], 1.0e-30)),
    )


def pk_ccl_spline_jax(
    log_k: jnp.ndarray,
    z_tab: jnp.ndarray,
    log_pk: jnp.ndarray,
    k: jnp.ndarray,
    z_eval: jnp.ndarray,
) -> jnp.ndarray:
    """P(k, z) via CCL-matched Pk2D spline (Akima tensor product in ln k, a)."""
    if use_bilinear_pk_tables():
        return jnp.exp(
            interp_log_pk_bilinear(log_k, z_tab, log_pk, k, z_eval)
        )
    return pk2d_eval_from_z_jax(k, z_eval, log_k, z_tab, log_pk)


def use_bilinear_pk_tables() -> bool:
    import os

    return os.environ.get("GPUCOSMO_BILINEAR_PK", "").lower() in ("1", "true", "yes")


def use_pk2d_knots() -> bool:
    """Use coarse Pk2D knot tables (default); skip dense k upsampling unless forced."""
    import os

    if os.environ.get("GPUCOSMO_DENSE_PK", "").lower() in ("1", "true", "yes"):
        return False
    return True


def use_legacy_limber() -> bool:
    import os

    return os.environ.get("GPUCOSMO_LEGACY_LIMBER", "").lower() in ("1", "true", "yes")


def use_legacy_z_limber() -> bool:
    """Use z-grid Limber (composite GL25 / trapezoid) instead of default log-k."""
    import os

    if os.environ.get("GPUCOSMO_LEGACY_Z_LIMBER", "").lower() in ("1", "true", "yes"):
        return True
    if os.environ.get("GPUCOSMO_LOGK_LIMBER", "").lower() in ("0", "false", "no"):
        return True
    return False


def use_logk_limber() -> bool:
    """Log-k Limber is the default (CCL-matched); disabled by GPUCOSMO_LEGACY_Z_LIMBER."""
    return not use_legacy_z_limber()


def pk_z_on_limber_grid(
    ell: jnp.ndarray,
    log_k: jnp.ndarray,
    z_pk: jnp.ndarray,
    log_pk: jnp.ndarray,
    z_grid: jnp.ndarray,
    chi_grid: jnp.ndarray,
) -> jnp.ndarray:
    """P(k=(ell+0.5)/chi, z) on the Limber z grid; ell scalar or batched."""
    if ell.ndim == 0:
        k = (ell + 0.5) / jnp.clip(chi_grid, 1.0)
        return jnp.exp(interp_log_pk_bilinear(log_k, z_pk, log_pk, k, z_grid))
    k = (ell[:, None] + 0.5) / jnp.clip(chi_grid[None, :], 1.0)
    z_b = jnp.broadcast_to(z_grid, k.shape)
    return jnp.exp(interp_log_pk_bilinear(log_k, z_pk, log_pk, k, z_b))


def _sigma8_for_log_as(pars: camb.CAMBparams, log_as: float) -> float:
    pars.InitPower.set_params(As=np.exp(log_as), ns=pars.InitPower.ns)
    import camb

    return float(camb.get_results(pars).get_sigma8_0())


def _find_log_as_for_sigma8(
    pars: camb.CAMBparams,
    target_sigma8: float,
) -> float:
    """Bracket and refine log(As) so CAMB matches the requested sigma8."""
    log_lo, log_hi = np.log(1.0e-10), np.log(1.0e-8)
    sigma_lo = _sigma8_for_log_as(pars, log_lo)
    sigma_hi = _sigma8_for_log_as(pars, log_hi)
    if not (sigma_lo <= target_sigma8 <= sigma_hi):
        raise RuntimeError(
            "Unable to bracket sigma8 with CAMB: "
            f"target={target_sigma8}, range=({sigma_lo}, {sigma_hi})"
        )

    for _ in range(60):
        log_mid = 0.5 * (log_lo + log_hi)
        sigma_mid = _sigma8_for_log_as(pars, log_mid)
        if sigma_mid < target_sigma8:
            log_lo = log_mid
        else:
            log_hi = log_mid
    return 0.5 * (log_lo + log_hi)


def _rounded_cosmology_key(params: CosmologyParams) -> tuple[float, float, float, float, float]:
    """Stable cache key for CCL cosmology reuse (8-decimal rounding)."""
    return (
        round(params.Omega_c, 8),
        round(params.Omega_b, 8),
        round(params.h, 8),
        round(params.sigma8, 8),
        round(params.n_s, 8),
    )


@lru_cache(maxsize=64)
def _warmed_ccl_linear_cosmo(
    omega_c: float,
    omega_b: float,
    h: float,
    sigma8: float,
    n_s: float,
):
    """Return a CCL cosmology with CAMB Boltzmann splines already built."""
    import pyccl as ccl

    cosmo = ccl.Cosmology(
        Omega_c=omega_c,
        Omega_b=omega_b,
        h=h,
        sigma8=sigma8,
        n_s=n_s,
        transfer_function="boltzmann_camb",
        matter_power_spectrum="linear",
    )
    # One call triggers boltzmann_camb; later lookups use CCL's internal Pk2D splines.
    ccl.linear_matter_power(cosmo, 1.0e-4, 1.0)
    return cosmo


def _linear_pk_table_from_ccl_cosmo(
    cosmo,
    *,
    z_max: float,
    n_z: int,
    k_min: float,
    k_max: float,
    n_k: int,
) -> MatterPowerTable:
    """Sample linear P(k, z) via CCL's warmed Pk2D interpolator (~3 ms)."""
    z_grid = np.linspace(0.0, z_max, n_z)
    k_grid = np.logspace(np.log10(k_min), np.log10(k_max), n_k)
    pk2d = cosmo.get_linear_power()
    pk_grid = np.empty((n_z, n_k), dtype=float)
    for iz, z_val in enumerate(z_grid):
        pk_grid[iz, :] = np.asarray(
            pk2d(k_grid, 1.0 / (1.0 + z_val), cosmo),
            dtype=float,
        )
    return MatterPowerTable(
        log_k=np.log(k_grid),
        z=z_grid,
        log_pk=np.log(np.maximum(pk_grid, 1.0e-30)),
    )


@lru_cache(maxsize=128)
def get_linear_pk_table(
    omega_c: float,
    omega_b: float,
    h: float,
    sigma8: float,
    n_s: float,
    z_max: float,
    n_z: int,
    n_k: int,
    k_min: float,
    k_max: float,
) -> MatterPowerTable:
    """Cached linear P(k, z) at exact cosmology (CCL splines, not a cosmo grid)."""
    cosmo = _warmed_ccl_linear_cosmo(omega_c, omega_b, h, sigma8, n_s)
    return _linear_pk_table_from_ccl_cosmo(
        cosmo,
        z_max=z_max,
        n_z=n_z,
        k_min=k_min,
        k_max=k_max,
        n_k=n_k,
    )


def clear_linear_pk_caches() -> None:
    """Clear warmed CCL cosmologies and linear P(k, z) LRU caches."""
    _warmed_ccl_linear_cosmo.cache_clear()
    get_linear_pk_table.cache_clear()
    try:
        from gpucosmo.jax.cosmopower.linear_pk import clear_cosmopower_pk_caches

        clear_cosmopower_pk_caches()
    except ImportError:
        pass


def build_linear_pk(
    params: CosmologyParams,
    *,
    backend: BoltzmannBackend = DEFAULT_BOLTZMANN_BACKEND,
    z_max: float = 3.0,
    n_z: int = 80,
    k_min: float = 1.0e-4,
    k_max: float = 50.0,
    n_k: int = 256,
) -> MatterPowerTable:
    """Linear P(k, z) from the selected Boltzmann backend."""
    if backend == "ccl":
        return build_linear_pk_cpu(
            params,
            z_max=z_max,
            n_z=n_z,
            k_min=k_min,
            k_max=k_max,
            n_k=n_k,
        )
    if backend == "jax_eb":
        z_grid = np.linspace(0.0, z_max, n_z)
        log_k = np.log(np.geomspace(k_min, k_max, n_k))
        lin, _ = jax_eb_dense_log_pk(
            params.Omega_c, params.Omega_b, params.h, params.n_s, params.sigma8,
            z_grid, log_k, nonlinear=False,
        )
        return MatterPowerTable(log_k=log_k, z=z_grid, log_pk=np.asarray(lin))
    if backend == "cosmopower":
        from gpucosmo.jax.cosmopower.linear_pk import build_linear_pk_cosmopower

        return build_linear_pk_cosmopower(
            params,
            z_max=z_max,
            n_z=n_z,
            k_min=k_min,
            k_max=k_max,
            n_k=n_k,
        )
    raise ValueError(f"Unknown boltzmann backend: {backend!r}")


def build_linear_pk_cpu(
    params: CosmologyParams,
    *,
    z_max: float = 3.0,
    n_z: int = 80,
    k_min: float = 1.0e-4,
    k_max: float = 50.0,
    n_k: int = 256,
) -> MatterPowerTable:
    """Linear P(k, z) from CCL boltzmann_camb with per-cosmology spline reuse."""
    omega_c, omega_b, h, sigma8, n_s = _rounded_cosmology_key(params)
    return get_linear_pk_table(
        omega_c,
        omega_b,
        h,
        sigma8,
        n_s,
        float(z_max),
        int(n_z),
        int(n_k),
        float(k_min),
        float(k_max),
    )


def build_matter_power_table_from_ccl(
    params: CosmologyParams,
    *,
    z_max: float = 3.0,
    n_z: int = 80,
    k_min: float = 1.0e-4,
    k_max: float = 50.0,
    n_k: int = 256,
) -> MatterPowerTable:
    """Construct a nonlinear P(k, z) table from CCL (boltzmann_camb + halofit)."""
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
    z_grid = np.linspace(0.0, z_max, n_z)
    k_grid = np.logspace(np.log10(k_min), np.log10(k_max), n_k)
    pk_grid = np.empty((n_z, n_k), dtype=float)
    for iz, z_val in enumerate(z_grid):
        scale_factor = 1.0 / (1.0 + z_val)
        pk_grid[iz, :] = np.asarray(
            ccl.power.nonlin_matter_power(cosmo, k_grid, scale_factor),
            dtype=float,
        )
    return MatterPowerTable(
        log_k=np.log(k_grid),
        z=z_grid,
        log_pk=np.log(np.maximum(pk_grid, 1.0e-30)),
    )


def build_matter_power_table(
    params: CosmologyParams,
    *,
    z_max: float = 3.0,
    n_z: int = 80,
    k_min: float = 1.0e-4,
    k_max: float = 50.0,
    n_k: int = 256,
) -> MatterPowerTable:
    """Construct a nonlinear P(k, z) table from CAMB for the given cosmology."""
    import camb

    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=100.0 * params.h,
        ombh2=params.Omega_b * params.h**2,
        omch2=params.Omega_c * params.h**2,
        mnu=0.0,
    )
    z_grid = np.linspace(0.0, z_max, n_z)
    pars.set_matter_power(redshifts=z_grid, kmax=k_max)
    pars.InitPower.set_params(As=2.0e-9, ns=params.n_s)
    log_as = _find_log_as_for_sigma8(pars, params.sigma8)
    pars.InitPower.set_params(As=np.exp(log_as), ns=params.n_s)
    results = camb.get_results(pars)
    pk_interp = results.get_matter_power_interpolator(nonlinear=True)
    k_grid = np.logspace(np.log10(k_min), np.log10(k_max), n_k)
    pk_grid = np.empty((n_z, n_k), dtype=float)
    for iz, z_val in enumerate(z_grid):
        pk_grid[iz] = pk_interp.P(k_grid, z_val)
    return MatterPowerTable(
        log_k=np.log(k_grid),
        z=z_grid,
        log_pk=np.log(np.maximum(pk_grid, 1.0e-30)),
    )


@lru_cache(maxsize=64)
def _cached_matter_power_table(
    omega_c: float,
    omega_b: float,
    h: float,
    sigma8: float,
    n_s: float,
    use_ccl: bool,
) -> MatterPowerTable:
    params = CosmologyParams(
        Omega_c=omega_c,
        Omega_b=omega_b,
        h=h,
        sigma8=sigma8,
        n_s=n_s,
    )
    builder = build_matter_power_table_from_ccl if use_ccl else build_matter_power_table
    return builder(params)


def get_matter_power_table(
    params: CosmologyParams,
    *,
    use_ccl: bool = True,
) -> MatterPowerTable:
    """Return a cached P(k, z) table for the given cosmology."""
    return _cached_matter_power_table(
        round(params.Omega_c, 8),
        round(params.Omega_b, 8),
        round(params.h, 8),
        round(params.sigma8, 8),
        round(params.n_s, 8),
        use_ccl,
    )


@lru_cache(maxsize=64)
def get_cosmology_bundle(
    omega_c: float,
    omega_b: float,
    h: float,
    sigma8: float,
    n_s: float,
    z_max: float,
    *,
    n_z_limber: int = 512,
    use_ccl: bool = True,
    background_backend: BackgroundBackend = DEFAULT_BACKGROUND_BACKEND,
) -> CosmologyBundle:
    """Return cached Limber grids, growth, and P(k, z) for a cosmology."""
    params = CosmologyParams(
        Omega_c=omega_c,
        Omega_b=omega_b,
        h=h,
        sigma8=sigma8,
        n_s=n_s,
    )
    from gpucosmo.jax.kernels import (
        build_background_grids,
        build_background_grids_ccl,
        build_growth_grid,
    )

    pk_table = get_matter_power_table(params, use_ccl=use_ccl)
    z_grid = np.linspace(0.0, z_max, n_z_limber)
    if background_backend == "ccl":
        chi_grid, ez_grid, hz_grid = build_background_grids_ccl(z_grid, params)
    else:
        z_grid, chi_grid, ez_grid, hz_grid = build_background_grids(
            z_max,
            h=params.h,
            Omega_c=params.Omega_c,
            Omega_b=params.Omega_b,
            n_z=n_z_limber,
        )
    growth_grid = build_growth_grid(z_grid, params)
    return CosmologyBundle(
        params=params,
        log_k=pk_table.log_k,
        z_pk=pk_table.z,
        log_pk=pk_table.log_pk,
        z_grid=z_grid,
        chi_grid=chi_grid,
        ez_grid=ez_grid,
        hz_grid=hz_grid,
        growth_grid=growth_grid,
    )


def bundle_for_params(params: CosmologyParams, z_max: float) -> CosmologyBundle:
    """Convenience wrapper with rounded cache keys."""
    return get_cosmology_bundle(
        round(params.Omega_c, 8),
        round(params.Omega_b, 8),
        round(params.h, 8),
        round(params.sigma8, 8),
        round(params.n_s, 8),
        float(z_max),
    )


def pk_jax(table: MatterPowerTable, k, z):
    """JAX-friendly wrapper around the NumPy P(k, z) interpolator."""
    k_np = np.asarray(k, dtype=float)
    z_np = np.asarray(z, dtype=float)
    return jnp.asarray(table.pk(k_np, z_np))


@dataclass(frozen=True)
class DirectTheoryInputs:
    """CPU-built P(k, z) tables and fixed Limber grids for the exact GPU path."""

    params: CosmologyParams
    log_k: np.ndarray
    z_pk: np.ndarray
    log_pk_lin: np.ndarray
    z_grid: np.ndarray
    chi_grid: np.ndarray
    ez_grid: np.ndarray
    hz_grid: np.ndarray
    n_z_limber: int
    log_pk_nl: np.ndarray | None = None
    growth_grid: np.ndarray | None = None
    nonlinear_prescription: NonlinearPrescription = DEFAULT_NONLINEAR_PRESCRIPTION
    background_backend: BackgroundBackend = DEFAULT_BACKGROUND_BACKEND

    @property
    def z_max(self) -> float:
        return float(self.z_grid[-1])


def build_direct_theory_inputs(
    params: CosmologyParams,
    z_max: float,
    *,
    n_z_limber: int = 512,
    n_z_pk: int = 80,
    n_k: int = 256,
    backend: BoltzmannBackend = DEFAULT_BOLTZMANN_BACKEND,
    nonlinear_prescription: NonlinearPrescription = DEFAULT_NONLINEAR_PRESCRIPTION,
    background_backend: BackgroundBackend = DEFAULT_BACKGROUND_BACKEND,
    ccl_match: bool | None = None,
) -> DirectTheoryInputs:
    """Build P(k, z) tables and Limber z grid for the exact GPU theory path."""
    use_coarse = use_bilinear_pk_tables()
    if ccl_match is None:
        ccl_match = not use_coarse
    if backend == "jax_eb":
        # On-GPU CAMB transfer run + HMcode-2020 (mead2020); the linear table is
        # the same Pk2D's, as for the CCL/CAMB path.
        pk_lin, pk_nl = build_dense_pk_tables_from_jax_eb(params, z_max=z_max, n_z=n_z_pk)
        return _direct_inputs_from_tables(
            params, pk_lin, pk_nl if nonlinear_prescription == "hmcode" else None,
            z_max=z_max, n_z_limber=n_z_limber,
            nonlinear_prescription=nonlinear_prescription,
            background_backend=background_backend,
        )
    if use_coarse or (backend == "cosmopower" and not ccl_match):
        pk_lin = build_linear_pk(
            params,
            backend=backend,
            z_max=z_max,
            n_z=n_z_pk,
            n_k=n_k,
        )
    else:
        pk_lin = build_dense_pk_table_from_ccl(
            params,
            z_max=z_max,
            n_z=n_z_pk,
            nonlinear=False,
        )
        n_z_pk = int(pk_lin.z.size)
        n_k = int(pk_lin.log_k.size)
    log_pk_nl: np.ndarray | None = None
    if nonlinear_prescription == "hmcode":
        if use_coarse or (backend == "cosmopower" and not ccl_match):
            from gpucosmo.jax.cosmopower.linear_pk import build_nonlinear_pk_cosmopower

            pk_nl = build_nonlinear_pk_cosmopower(
                params,
                z_max=z_max,
                n_z=n_z_pk,
                n_k=n_k,
            )
            log_pk_nl = pk_nl.log_pk
            if ccl_match and not use_coarse and not use_pk2d_knots():
                pk_lin = upsample_matter_table_to_ccl_dense(pk_lin)
                pk_nl = upsample_matter_table_to_ccl_dense(pk_nl)
                log_pk_nl = pk_nl.log_pk
        else:
            # Same CCL/CAMB model as GGLTheory(nonlinear_prescription='hmcode'):
            # CAMB HMcode-2020 without feedback (previously this silently used
            # the builder default, i.e. CCL Halofit).
            pk_nl = build_dense_pk_table_from_ccl(
                params,
                z_max=z_max,
                n_z=n_z_pk,
                nonlinear=True,
                nonlinear_prescription="hmcode",
                hmcode_version=GGL_HMCODE_CAMB_VERSION,
            )
            log_pk_nl = pk_nl.log_pk
    return _direct_inputs_from_tables(
        params, pk_lin, None if log_pk_nl is None else MatterPowerTable(
            log_k=pk_lin.log_k, z=pk_lin.z, log_pk=log_pk_nl),
        z_max=z_max, n_z_limber=n_z_limber,
        nonlinear_prescription=nonlinear_prescription,
        background_backend=background_backend,
    )


def _direct_inputs_from_tables(
    params: CosmologyParams,
    pk_lin: MatterPowerTable,
    pk_nl: MatterPowerTable | None,
    *,
    z_max: float,
    n_z_limber: int,
    nonlinear_prescription: NonlinearPrescription,
    background_backend: BackgroundBackend,
) -> DirectTheoryInputs:
    """Background grids + DirectTheoryInputs around a (lin, nl) table pair."""
    log_pk_nl = None if pk_nl is None else pk_nl.log_pk
    z_grid = np.linspace(0.0, z_max, n_z_limber)
    from gpucosmo.jax.kernels import (
        build_background_grids,
        build_background_grids_ccl,
        build_growth_grid,
    )

    if background_backend == "ccl":
        chi_grid, ez_grid, hz_grid = build_background_grids_ccl(z_grid, params)
        growth_grid = build_growth_grid(z_grid, params)
    else:
        _, chi_grid, ez_grid, hz_grid = build_background_grids(
            z_max,
            h=params.h,
            Omega_c=params.Omega_c,
            Omega_b=params.Omega_b,
            n_z=n_z_limber,
        )
        growth_grid = None
    return DirectTheoryInputs(
        params=params,
        log_k=pk_lin.log_k,
        z_pk=pk_lin.z,
        log_pk_lin=pk_lin.log_pk,
        z_grid=z_grid,
        chi_grid=chi_grid,
        ez_grid=ez_grid,
        hz_grid=hz_grid,
        n_z_limber=n_z_limber,
        log_pk_nl=log_pk_nl,
        growth_grid=growth_grid,
        nonlinear_prescription=nonlinear_prescription,
        background_backend=background_backend,
    )


def direct_inputs_as_jax(inputs: DirectTheoryInputs) -> dict[str, jnp.ndarray]:
    """Upload direct-path CPU tables to the default JAX device."""
    out = {
        "log_k": jnp.asarray(inputs.log_k),
        "z_pk": jnp.asarray(inputs.z_pk),
        "log_pk_lin": jnp.asarray(inputs.log_pk_lin),
        "z_grid": jnp.asarray(inputs.z_grid),
        "chi_grid": jnp.asarray(inputs.chi_grid),
        "ez_grid": jnp.asarray(inputs.ez_grid),
        "hz_grid": jnp.asarray(inputs.hz_grid),
    }
    if inputs.log_pk_nl is not None:
        out["log_pk_nl"] = jnp.asarray(inputs.log_pk_nl)
    if inputs.growth_grid is not None:
        out["growth_grid"] = jnp.asarray(inputs.growth_grid)
    return out


__all__ = [
    "BackgroundBackend",
    "BoltzmannBackend",
    "DEFAULT_BACKGROUND_BACKEND",
    "DEFAULT_BOLTZMANN_BACKEND",
    "DEFAULT_NONLINEAR_PRESCRIPTION",
    "NonlinearPrescription",
    "CosmologyBundle",
    "CosmologyParams",
    "DirectTheoryInputs",
    "MatterPowerTable",
    "build_dense_pk_table_from_ccl",
    "ccl_dense_pk_table_shape",
    "ccl_pk_k_bounds",
    "ccl_spline_params",
    "build_linear_pk",
    "build_linear_pk_cpu",
    "build_matter_power_table",
    "bundle_for_params",
    "clear_linear_pk_caches",
    "direct_inputs_as_jax",
    "get_cosmology_bundle",
    "get_linear_pk_table",
    "get_matter_power_table",
    "interp_log_pk_bilinear",
    "pk_ccl_spline_jax",
    "use_bilinear_pk_tables",
    "use_pk2d_knots",
    "use_logk_limber",
    "use_legacy_limber",
    "use_legacy_z_limber",
    "upsample_matter_table_to_ccl_dense",
    "pk_z_on_limber_grid",
]
