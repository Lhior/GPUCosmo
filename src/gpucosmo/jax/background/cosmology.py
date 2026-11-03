#!/usr/bin/env python3
"""
JAX w0waCDM background cosmology matching CAMB conventions.

Model: w0waCDM (CPL equation of state w(a) = w0 + wa (1 - a)) with spatial curvature
Ω_k and CAMB's neutrino models: T_CMB = 2.7255 K, N_eff (``nnu``), Σm_ν (``mnu``) over
the eigenstates of a static ``massive_nu.NeutrinoModel`` (CAMB ``set_cosmology``:
degenerate / normal / inverted, optional thermal massive sterile ``meffsterile``). The
default — one 0.06 eV eigenstate, N_eff = 3.044, flat — is the DESI DR2 baseline.

Accuracy design (accuracy first, speed second):

- The massive-neutrino density is CAMB's own ``ThermalNuBackground`` fit
  (``massive_nu.thermal_nu_rho``, ~2e-5), per eigenstate.
- Recombination redshifts z_star and z_drag are Chebyshev surfaces in (ombh2, omch2)
  fitted to CAMB's RECFAST outputs at the default neutrinos (median over a small H0
  grid to suppress RECFAST's O(1e-6) numerical H0 jitter; early-Universe quantities
  are independent of H0 and, under the w0+wa<0 prior, of dark energy), plus, for
  sampled Σm_ν / N_eff / m_eff,sterile, 4-D Chebyshev corrections per neutrino model
  (scripts/build_nu_background_tables.py). Ω_k moves them by < 3e-6 (|Ω_k| ≤ 0.1),
  below that jitter, and is ignored there.
- Transverse distances use f_K(χ) (sinh/sin for open/closed, with a series near
  Ω_k = 0 so values and derivatives are smooth through flat).
- All distance and sound-horizon integrals are evaluated with high-order
  Gauss-Legendre quadrature (spectrally convergent for these smooth
  integrands -- no trapezoid shortcuts), with the exact w0waCDM H(z), in
  float64.

Everything is pure JAX: jit-able, vmap-able and differentiable end to end.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

C_KM_S = 299792.458

from gpucosmo.paths import data_file, packaged_data, source_root  # noqa: E402

REPO_ROOT = source_root()  # None for an installed package
# <data dir>/background_tables (scripts/build_background_tables.py writes there), else the copy
# shipped with the package.
# v2: z*/z_drag fitted with the solver's BBN helium (PArthENoPE); v1 (CAMB's default PRIMAT) was
# 9e-6 / 8e-6 off the solver's recombination.
DEFAULT_TABLES_PATH = data_file("background_tables", "w0wa_background_tables_v2.npz",
                                fallbacks=(packaged_data("w0wa_background_tables_v2.npz"),))

# Gauss-Legendre nodes/weights on [-1, 1], computed once at import (float64).
_GL128_X, _GL128_W = np.polynomial.legendre.leggauss(128)
_GL256_X, _GL256_W = np.polynomial.legendre.leggauss(256)
# Concrete even if this module is first imported inside a jit trace (else they leak as tracers).
with jax.ensure_compile_time_eval():
    GL128_X = jnp.asarray(_GL128_X)
    GL128_W = jnp.asarray(_GL128_W)
    GL256_X = jnp.asarray(_GL256_X)
    GL256_W = jnp.asarray(_GL256_W)

# Sound-horizon integration lower limit in ln(a).
LN_A_EARLY = float(np.log(1.0e-9))


class BackgroundParams(NamedTuple):
    """Sampled background parameters (physical densities + H0, CPL dark energy, Σm_ν [eV],
    N_eff, Ω_k, sterile m_eff [eV]). Every field is required: ``BackgroundParams.build``
    fills the extended ones from an ``ext`` dict (``cls_jax.ext_params``)."""

    ombh2: jnp.ndarray
    omch2: jnp.ndarray
    H0: jnp.ndarray
    w0: jnp.ndarray
    wa: jnp.ndarray
    mnu: jnp.ndarray
    nnu: jnp.ndarray
    omk: jnp.ndarray
    meffsterile: jnp.ndarray
    # Active ω_ν h² instead of Σm_ν (CAMB's omnuh2_active route; None: use ``mnu``).
    omnuh2: jnp.ndarray | None = None

    @classmethod
    def build(cls, ombh2, omch2, H0, w0, wa, ext: dict | None = None) -> "BackgroundParams":
        """Parameters with the extended ones from ``ext`` (missing keys: CAMB's defaults, one
        0.06 eV eigenstate, N_eff = 3.044, flat, no sterile)."""
        from gpucosmo.jax.eb.cls_jax import ext_params

        e = ext_params(ext)
        return cls(ombh2, omch2, H0, w0, wa, e["mnu"], e["nnu"], e["omk"], e["meffsterile"], e.get("omnuh2"))


@dataclass(frozen=True)
class BackgroundTables:
    """CAMB-derived constants and tables (see scripts/build_background_tables.py and
    scripts/build_nu_background_tables.py) and the static neutrino model."""

    omega_gamma_h2: float
    ombh2_range: tuple[float, float]
    omch2_range: tuple[float, float]
    zstar_cheb: jnp.ndarray
    zdrag_cheb: jnp.ndarray
    nu_model: object = None  # massive_nu.NeutrinoModel (None: CAMB default)
    # 4-D Chebyshev corrections Δz(ombh2, omch2, x3, x4) for ``nu_model`` (None: not built);
    # (x3, x4) = (mnu, nnu), or (nnu, meffsterile) for a sterile model. Piecewise in x4 at
    # ``nu_corr_breaks`` (kinks of z, e.g. N_eff = 3.044 with three massive states): the
    # coefficient arrays have a leading piece axis.
    nu_corr_axes: tuple[str, str] | None = None
    nu_corr_ranges: tuple | None = None
    nu_corr_breaks: tuple = ()
    zstar_corr_cheb: jnp.ndarray | None = None
    zdrag_corr_cheb: jnp.ndarray | None = None

    def __hash__(self):  # identity: closed over by jitted functions
        return id(self)

    def __eq__(self, other):
        return self is other

    @property
    def model(self):
        from gpucosmo.jax.eb.massive_nu import DEFAULT_NEUTRINO_MODEL

        return DEFAULT_NEUTRINO_MODEL if self.nu_model is None else self.nu_model


def nu_corrections_name(nu_model) -> str:
    """File name of a neutrino model's z*/z_drag corrections (build_nu_background_tables.py)."""
    m = nu_model
    tag = (f"{m.hierarchy}{m.num_massive if m.hierarchy == 'degenerate' else ''}"
           f"{'_sterile' if m.sterile else ''}")
    return f"nu_corrections_{tag}.npz"


def nu_corrections_path(nu_model, base: Path | str = DEFAULT_TABLES_PATH) -> Path:
    """The model's corrections next to the base tables, else the copy shipped with the package."""
    local = Path(base).with_name(nu_corrections_name(nu_model))
    if local.is_file():
        return local
    return Path(packaged_data(nu_corrections_name(nu_model)))


def load_background_tables(
    path: Path | str = DEFAULT_TABLES_PATH, *, nu_model=None,
) -> BackgroundTables:
    """Background tables for ``nu_model`` (``massive_nu.NeutrinoModel``; None: CAMB's default
    one 0.06 eV eigenstate). The model's z*/z_drag corrections are attached when built
    (needed only when Σm_ν, N_eff or m_eff,sterile differ from the default)."""
    from gpucosmo.jax.eb.massive_nu import DEFAULT_NEUTRINO_MODEL

    data = np.load(Path(path))
    corr = {}
    # None: the default model's corrections (a sampled Σm_ν with the default neutrinos needs them)
    cpath = nu_corrections_path(DEFAULT_NEUTRINO_MODEL if nu_model is None else nu_model, path)
    if cpath.is_file():
        c = np.load(cpath)
        corr = dict(nu_corr_axes=tuple(str(a) for a in c["axes"]),
                    nu_corr_ranges=tuple(tuple(float(v) for v in r) for r in c["ranges"]),
                    nu_corr_breaks=tuple(float(b) for b in c["x4_breaks"]),
                    zstar_corr_cheb=jnp.asarray(c["zstar_corr_cheb"]),
                    zdrag_corr_cheb=jnp.asarray(c["zdrag_corr_cheb"]))
    return BackgroundTables(
        omega_gamma_h2=float(data["omega_gamma_h2"]),
        ombh2_range=(float(data["ombh2_range"][0]), float(data["ombh2_range"][1])),
        omch2_range=(float(data["omch2_range"][0]), float(data["omch2_range"][1])),
        zstar_cheb=jnp.asarray(data["zstar_cheb"]),
        zdrag_cheb=jnp.asarray(data["zdrag_cheb"]),
        nu_model=nu_model,
        **corr,
    )


class BackgroundState(NamedTuple):
    """Densities of one cosmology (ω = Ω h²), resolved once from ``BackgroundParams``."""

    h2: jnp.ndarray
    H0: jnp.ndarray
    ombh2: jnp.ndarray
    omch2: jnp.ndarray
    omega_rad_h2: jnp.ndarray  # photons + massless neutrinos (∝ a⁻⁴)
    nu_om_h2: jnp.ndarray  # per massive eigenstate block: ω if massless, (n_eig,)
    nu_am: jnp.ndarray  # per block: m/(k_B T_ν,block), (n_eig,)
    omnuh2: jnp.ndarray  # massive neutrinos today
    omk: jnp.ndarray
    omega_de_h2: jnp.ndarray
    w0: jnp.ndarray
    wa: jnp.ndarray
    mnu: jnp.ndarray
    nnu: jnp.ndarray
    meffsterile: jnp.ndarray


def background_state(tables: BackgroundTables, params) -> BackgroundState:
    """Resolve ``params`` (or pass a ``BackgroundState`` through)."""
    if isinstance(params, BackgroundState):
        return params
    from gpucosmo.jax.eb.massive_nu import omega_nu1_h2

    nu = tables.model.resolve(params.mnu, params.nnu, params.meffsterile, omnuh2=params.omnuh2)
    # Σm_ν of the active states (the z*/z_drag tables' axis): given, or from ω_ν's eigenstates
    n_act = tables.model.n_active_eig
    mnu = params.mnu if params.omnuh2 is None else jnp.sum((nu["number"] * nu["mass"])[..., :n_act], axis=-1)
    h2 = (params.H0 / 100.0) ** 2
    omega_rad_h2 = tables.omega_gamma_h2 + omega_nu1_h2() * nu["N_ur"]
    omk = jnp.asarray(params.omk, dtype=jnp.float64)
    omega_de_h2 = (h2 - params.ombh2 - params.omch2 - omega_rad_h2 - nu["omnuh2"] - omk * h2)
    # Clamp to keep E^2 > 0 (NaN-free) outside the prior support; the prior assigns -inf to
    # any point with omega_de_h2 <= 0, so this never affects accepted samples or gradients.
    omega_de_h2 = jnp.maximum(omega_de_h2, 1.0e-10)
    return BackgroundState(h2, params.H0, params.ombh2, params.omch2, omega_rad_h2,
                           nu["omh2_massless"], nu["am"], nu["omnuh2"], omk, omega_de_h2,
                           params.w0, params.wa, mnu, params.nnu, params.meffsterile)


def _cheb_basis(x: jnp.ndarray, degree: int) -> jnp.ndarray:
    """Chebyshev polynomials T_0..T_degree at x in [-1, 1]; shape (..., degree+1)."""
    t = [jnp.ones_like(x), x]
    for _ in range(degree - 1):
        t.append(2.0 * x * t[-1] - t[-2])
    return jnp.stack(t, axis=-1)


def eval_cheb2d(
    coef: jnp.ndarray,
    x: jnp.ndarray,
    y: jnp.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> jnp.ndarray:
    """Evaluate a 2D Chebyshev surface fitted by build_background_tables.py."""
    xu = (2.0 * x - (x_range[0] + x_range[1])) / (x_range[1] - x_range[0])
    yu = (2.0 * y - (y_range[0] + y_range[1])) / (y_range[1] - y_range[0])
    tx = _cheb_basis(xu, coef.shape[0] - 1)
    ty = _cheb_basis(yu, coef.shape[1] - 1)
    return jnp.einsum("...i,ij,...j->...", tx, coef, ty)


def eval_cheb4d(coef, xs, ranges) -> jnp.ndarray:
    """Evaluate a 4-D Chebyshev tensor (``build_nu_background_tables.py``) at ``xs``."""
    bases = []
    for x, rng, n in zip(xs, ranges, coef.shape):
        xu = (2.0 * jnp.asarray(x, dtype=jnp.float64) - (rng[0] + rng[1])) / (rng[1] - rng[0])
        bases.append(_cheb_basis(xu, n - 1))
    return jnp.einsum("...i,...j,...k,...l,ijkl->...", *bases, coef)


def _is_default_nu(tables: BackgroundTables, mnu, nnu, meffsterile) -> bool:
    """Concrete default neutrinos (the base surfaces apply as they are)."""
    from gpucosmo.jax.eb import constants as C
    from gpucosmo.jax.eb.massive_nu import DEFAULT_NEUTRINO_MODEL

    if tables.model.base != DEFAULT_NEUTRINO_MODEL:
        return False
    try:
        return (float(mnu) == C.DEFAULT_MNU and float(nnu) == C.DEFAULT_NNU
                and float(meffsterile) == 0.0)
    except (TypeError, jax.errors.ConcretizationTypeError, jax.errors.TracerArrayConversionError):
        return False


def _nu_correction(tables: BackgroundTables, which: str, ombh2, omch2, mnu, nnu, meffsterile):
    if tables.zstar_corr_cheb is None:
        raise ValueError(
            f"z*/z_drag for {tables.model} away from the default neutrinos need "
            f"{nu_corrections_path(tables.model).name} (scripts/build_nu_background_tables.py)")
    vals = {"mnu": mnu, "nnu": nnu, "meffsterile": meffsterile}
    x3, x4 = (vals[a] for a in tables.nu_corr_axes)
    coef = tables.zstar_corr_cheb if which == "zstar" else tables.zdrag_corr_cheb
    r_ob, r_oc, r3, r4 = tables.nu_corr_ranges
    edges = (r4[0], *tables.nu_corr_breaks, r4[1])
    out = None
    for i in range(coef.shape[0]):  # one Chebyshev tensor per x4 piece; pick by x4
        val = eval_cheb4d(coef[i], (ombh2, omch2, x3, x4), (r_ob, r_oc, r3, (edges[i], edges[i + 1])))
        out = val if out is None else jnp.where(x4 >= edges[i], val, out)
    return out


def _z_rec(tables, which, ombh2, omch2, mnu, nnu, meffsterile):
    from gpucosmo.jax.eb import constants as C

    base = eval_cheb2d(tables.zstar_cheb if which == "zstar" else tables.zdrag_cheb, ombh2, omch2,
                       tables.ombh2_range, tables.omch2_range)
    mnu = C.DEFAULT_MNU if mnu is None else mnu
    nnu = C.DEFAULT_NNU if nnu is None else nnu
    meffsterile = 0.0 if meffsterile is None else meffsterile
    if _is_default_nu(tables, mnu, nnu, meffsterile):
        return base
    return base + _nu_correction(tables, which, ombh2, omch2, mnu, nnu, meffsterile)


def z_star(tables: BackgroundTables, ombh2, omch2, mnu=None, nnu=None, meffsterile=None):
    """z* (CAMB ``zstar``) at (ω_b, ω_c) and the neutrinos (None: the defaults)."""
    return _z_rec(tables, "zstar", ombh2, omch2, mnu, nnu, meffsterile)


def z_drag(tables: BackgroundTables, ombh2, omch2, mnu=None, nnu=None, meffsterile=None):
    """z_drag (CAMB ``zdrag``) at (ω_b, ω_c) and the neutrinos (None: the defaults)."""
    return _z_rec(tables, "zdrag", ombh2, omch2, mnu, nnu, meffsterile)


def hubble_e2_of_ln_a(
    tables: BackgroundTables, params, ln_a: jnp.ndarray
) -> jnp.ndarray:
    """E^2(a) = (H(a)/H0)^2: radiation, matter, massive ν (per eigenstate), curvature and CPL
    dark energy (CAMB ``dtauda``). ``params``: ``BackgroundParams`` or ``BackgroundState``."""
    from gpucosmo.jax.eb.massive_nu import thermal_nu_rho

    st = background_state(tables, params)
    a = jnp.exp(ln_a)
    inv_a = jnp.exp(-ln_a)
    rad = st.omega_rad_h2 * inv_a**4
    mat = (st.ombh2 + st.omch2) * inv_a**3
    if st.nu_am.shape[-1]:
        nu = jnp.sum(st.nu_om_h2 * thermal_nu_rho(a[..., None] * st.nu_am), axis=-1) * inv_a**4
    else:
        nu = 0.0
    curv = st.omk * st.h2 * inv_a**2
    # rho_DE(a)/rho_DE0 for CPL: a^{-3(1+w0+wa)} exp(-3 wa (1-a)).
    de = st.omega_de_h2 * jnp.exp(
        -3.0 * (1.0 + st.w0 + st.wa) * ln_a - 3.0 * st.wa * (1.0 - a)
    )
    return (rad + mat + nu + curv + de) / st.h2


def hubble_km_s_mpc(
    tables: BackgroundTables, params, z: jnp.ndarray
) -> jnp.ndarray:
    ln_a = -jnp.log1p(z)
    return params.H0 * jnp.sqrt(hubble_e2_of_ln_a(tables, params, ln_a))


def comoving_distance(
    tables: BackgroundTables, params, z: jnp.ndarray
) -> jnp.ndarray:
    """
    Comoving radial (line-of-sight) distance D_C(z) [Mpc] for low/intermediate redshifts.

    128-point Gauss-Legendre on [0, z] per redshift (spectrally accurate for
    the smooth, analytic integrand 1/E; validated to ~1e-12 relative).
    """
    st = background_state(tables, params)
    z = jnp.atleast_1d(z)
    half = 0.5 * z[:, None]
    z_nodes = half * (GL128_X[None, :] + 1.0)
    ln_a = -jnp.log1p(z_nodes)
    inv_e = 1.0 / jnp.sqrt(hubble_e2_of_ln_a(tables, st, ln_a))
    integral = half[:, 0] * jnp.sum(GL128_W[None, :] * inv_e, axis=1)
    return (C_KM_S / st.H0) * integral


def _sinc_k(u: jnp.ndarray) -> jnp.ndarray:
    """sinh(√u)/√u (u > 0), sin(√−u)/√−u (u < 0), 1 at u = 0: f_K(χ) = χ·_sinc_k(Ω_k (χ/D_H)²).
    A series near 0 keeps values and derivatives smooth through flat; the other branches see
    safe arguments so their (discarded) gradients stay finite."""
    small = jnp.abs(u) < 1.0e-3
    series = 1.0 + u / 6.0 + u * u / 120.0 + u**3 / 5040.0
    up = jnp.sqrt(jnp.where(u > 0.0, jnp.where(small, 1.0, u), 1.0))
    um = jnp.sqrt(jnp.where(u < 0.0, jnp.where(small, 1.0, -u), 1.0))
    return jnp.where(small, series, jnp.where(u > 0.0, jnp.sinh(up) / up, jnp.sin(um) / um))


def transverse_distance(tables: BackgroundTables, params, d_c: jnp.ndarray) -> jnp.ndarray:
    """f_K(D_C) [Mpc]: the transverse comoving distance D_M of a line-of-sight distance D_C."""
    st = background_state(tables, params)
    d_h = C_KM_S / st.H0
    return d_c * _sinc_k(st.omk * (d_c / d_h) ** 2)


def transverse_comoving_distance(tables: BackgroundTables, params, z: jnp.ndarray) -> jnp.ndarray:
    """D_M(z) = f_K(D_C(z)) [Mpc] (= D_C when flat)."""
    st = background_state(tables, params)
    return transverse_distance(tables, st, comoving_distance(tables, st, z))


def angular_diameter_distance_2(tables: BackgroundTables, params, z1, z2) -> jnp.ndarray:
    """D_A(z1, z2) [Mpc] (CAMB ``angular_diameter_distance2``; z1 < z2): f_K(χ2 − χ1)/(1+z2).
    For closed models CAMB's formula is the same (it is ill-defined only past the antipode)."""
    st = background_state(tables, params)
    z1 = jnp.atleast_1d(z1)
    z2 = jnp.atleast_1d(z2)
    chi = comoving_distance(tables, st, jnp.concatenate([z1, z2]))
    d = chi[z1.shape[0]:] - chi[: z1.shape[0]]
    return transverse_distance(tables, st, d) / (1.0 + z2)


def cumulative_distance_nodes(z: np.ndarray, n_gl: int = 8):
    """Static quadrature layout for ``comoving_distance_cumulative``.

    Sorts the (fixed, data) redshifts, splits [0, z_max] at the unique values and
    places an ``n_gl``-point Gauss-Legendre rule on each interval. Returns
    ``(z_nodes (n_int, n_gl), w_nodes (n_int, n_gl), inverse (n,))`` where
    ``inverse`` maps each input redshift to its interval end.
    """
    z = np.asarray(z, dtype=float)
    uz, inverse = np.unique(z, return_inverse=True)
    edges = np.concatenate([[0.0], uz])
    xg, wg = np.polynomial.legendre.leggauss(n_gl)
    half = 0.5 * np.diff(edges)[:, None]
    mid = 0.5 * (edges[1:] + edges[:-1])[:, None]
    return mid + half * xg[None, :], half * wg[None, :], inverse


def comoving_distance_cumulative(
    tables: BackgroundTables, params, nodes
) -> jnp.ndarray:
    """D_C [Mpc] at many fixed redshifts (e.g. an SN sample) by cumulative quadrature.

    Same integrand as ``comoving_distance`` but one short Gauss-Legendre rule per
    interval between consecutive sorted redshifts, summed cumulatively: ~16x
    fewer E(z) evaluations for 1.6k SNe than 128 nodes per SN, at the same
    (≲1e-12 relative) accuracy, and just as smooth in the parameters.
    """
    st = background_state(tables, params)
    z_nodes, w_nodes, inverse = nodes
    inv_e = 1.0 / jnp.sqrt(hubble_e2_of_ln_a(tables, st, -jnp.log1p(jnp.asarray(z_nodes))))
    seg = jnp.sum(jnp.asarray(w_nodes) * inv_e, axis=-1)
    return (C_KM_S / st.H0) * jnp.cumsum(seg)[jnp.asarray(inverse)]


def comoving_distance_highz(
    tables: BackgroundTables, params, z_end: jnp.ndarray
) -> jnp.ndarray:
    """
    D_C to a high redshift (e.g. z_star ~ 1090) [Mpc].

    Integrates in u = ln(1+z), where the integrand (1+z)/E(z) varies gently
    across the matter and radiation eras; 256-point Gauss-Legendre.
    """
    st = background_state(tables, params)
    u_end = jnp.log1p(z_end)
    half = 0.5 * u_end
    u_nodes = half * (GL256_X + 1.0)
    ln_a = -u_nodes
    inv_e = jnp.exp(u_nodes) / jnp.sqrt(hubble_e2_of_ln_a(tables, st, ln_a))
    return (C_KM_S / st.H0) * half * jnp.sum(GL256_W * inv_e)


def sound_horizon(
    tables: BackgroundTables, params, z_end: jnp.ndarray,
    r_per_a: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """
    Sound horizon r_s(z_end) [Mpc]:

        r_s = int_0^{a_end} c_s da / (a^2 H),   c_s = c / sqrt(3 (1 + R)),
        R = 3 rho_b / (4 rho_gamma) = (3 ombh2 / 4 omega_gamma_h2) a,

    evaluated in ln(a) with 256-point Gauss-Legendre from a = 1e-9 (the
    truncated early-time contribution is O(1e-9) relative -- the integrand
    scales as a in the radiation era). ``r_per_a`` overrides R/a (CAMB's
    ``cosmomc_theta`` uses the approximation 3e4·ω_b).
    """
    st = background_state(tables, params)
    ln_a_end = -jnp.log1p(z_end)
    half = 0.5 * (ln_a_end - LN_A_EARLY)
    mid = 0.5 * (ln_a_end + LN_A_EARLY)
    ln_a = mid + half * GL256_X
    a = jnp.exp(ln_a)
    if r_per_a is None:
        r_per_a = 3.0 * st.ombh2 / (4.0 * tables.omega_gamma_h2)
    r_baryon_photon = r_per_a * a
    cs = C_KM_S / jnp.sqrt(3.0 * (1.0 + r_baryon_photon))
    a_h = a * st.H0 * jnp.sqrt(hubble_e2_of_ln_a(tables, st, ln_a))
    return half * jnp.sum(GL256_W * cs / a_h)


def z_star_of(tables: BackgroundTables, params) -> jnp.ndarray:
    st = background_state(tables, params)
    return z_star(tables, st.ombh2, st.omch2, st.mnu, st.nnu, st.meffsterile)


def z_drag_of(tables: BackgroundTables, params) -> jnp.ndarray:
    st = background_state(tables, params)
    return z_drag(tables, st.ombh2, st.omch2, st.mnu, st.nnu, st.meffsterile)


def rs_drag(tables: BackgroundTables, params) -> jnp.ndarray:
    """Sound horizon at the drag epoch, r_d [Mpc] (CAMB 'rdrag')."""
    st = background_state(tables, params)
    return sound_horizon(tables, st, z_drag_of(tables, st))


def theta_star(tables: BackgroundTables, params) -> jnp.ndarray:
    """Acoustic angular scale theta_* = r_s(z_*) / D_M(z_*) (CAMB 'thetastar'/100)."""
    st = background_state(tables, params)
    zs = z_star_of(tables, st)
    rs = sound_horizon(tables, st, zs)
    dm = transverse_distance(tables, st, comoving_distance_highz(tables, st, zs))
    return rs / dm


def cosmomc_theta(tables: BackgroundTables, params) -> jnp.ndarray:
    """CAMB/CosmoMC ``cosmomc_theta`` (``theta_MC``, *not* θ*): the Hu & Sugiyama
    z* fit with ω_dm = ω_c + ω_ν, and r_s with R = 3e4·a·ω_b, over D_M (CAMB
    ``CosmomcTheta``/``dsound_da_approx``). Differs from ``theta_star`` by
    ≈ −2e-6 (−2e-4 in 100θ); Cobaya/ISiTGR chains sample this ``theta_MC_100``.
    """
    st = background_state(tables, params)
    ob = st.ombh2
    om = st.omch2 + st.omnuh2 + ob
    g1 = 0.0783 * ob ** (-0.238) / (1.0 + 39.5 * ob**0.763)
    g2 = 0.560 / (1.0 + 21.1 * ob**1.81)
    zs = 1048.0 * (1.0 + 0.00124 * ob ** (-0.738)) * (1.0 + g1 * om**g2)
    rs = sound_horizon(tables, st, zs, r_per_a=3.0e4 * ob)
    return rs / transverse_distance(tables, st, comoving_distance_highz(tables, st, zs))


__all__ = [
    "angular_diameter_distance_2",
    "background_state",
    "BackgroundState",
    "cosmomc_theta",
    "eval_cheb4d",
    "nu_corrections_name",
    "nu_corrections_path",
    "transverse_comoving_distance",
    "transverse_distance",
    "z_drag_of",
    "z_star_of",
    "BackgroundParams",
    "BackgroundTables",
    "C_KM_S",
    "comoving_distance",
    "comoving_distance_highz",
    "eval_cheb2d",
    "hubble_e2_of_ln_a",
    "hubble_km_s_mpc",
    "load_background_tables",
    "rs_drag",
    "sound_horizon",
    "theta_star",
    "z_drag",
    "z_star",
]
