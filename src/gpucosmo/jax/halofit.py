#!/usr/bin/env python3
"""GPU Takahashi (2012) Halofit from tabulated linear P(k, z).

Thin wrapper around the exact CCL 3.2.1 port in
:mod:`gpucosmo.jax.ccl.halofit`: the input table is treated as a CCL linear
``Pk2D`` (natural bicubic spline in ln k, quadratic ln P extrapolation above
``k_max``), Halofit's R_sigma / n_eff / C are solved at every redshift node
with converged quadrature and Newton root finding, and P_NL is returned on the
same (z, k) nodes. Agreement with ``pyccl`` Halofit on the same linear table is
at the level of CCL's own sigma(R) tolerance (``INTEGRATION_SIGMAR_EPSREL =
1e-5``, i.e. <~2e-5); with that tolerance tightened in CCL the two agree to
~1e-12.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from gpucosmo.jax.ccl.halofit import (
    NEFF_DEFAULT,
    T_CMB_DEFAULT,
    background_params_from_dict,
    halofit_log_pk_rows,
)

jax.config.update("jax_enable_x64", True)

_DEFAULT_H = 0.7


def _background(Omega_m, h, Omega_b, w0, wa, m_nu, mass_split, Neff, T_CMB, Omega_k):
    params = {
        "Omega_m": Omega_m,
        "Omega_b": Omega_b,
        "h": _DEFAULT_H if h is None else h,
        "w0": w0,
        "wa": wa,
        "Neff": Neff,
        "T_CMB": T_CMB,
        "Omega_k": Omega_k,
    }
    if m_nu is not None:
        params["m_nu"] = m_nu
        params["mass_split"] = mass_split
    return background_params_from_dict(params)


@jax.jit
def _halofit_table(log_k, z_pk, log_pk_lin, bg):
    a = 1.0 / (1.0 + jnp.asarray(z_pk, dtype=float))
    return halofit_log_pk_rows(log_k, a, log_pk_lin, bg)


def halofit_log_pk_jax(
    log_k: jnp.ndarray,
    z_pk: jnp.ndarray,
    log_pk_lin: jnp.ndarray,
    *,
    Omega_m: float,
    h: float | None = None,
    Omega_b: float = 0.0,
    w0: float = -1.0,
    wa: float = 0.0,
    m_nu=None,
    mass_split: str = "normal",
    Neff: float = NEFF_DEFAULT,
    T_CMB: float = T_CMB_DEFAULT,
    Omega_k: float = 0.0,
) -> jnp.ndarray:
    """Apply CCL-exact Takahashi Halofit to a linear log P(k, z) table.

    Parameters
    ----------
    log_k : (n_k,) ln k [1/Mpc], ascending.
    z_pk : (n_z,) redshifts of the table rows (any order).
    log_pk_lin : (n_z, n_k) ln P_lin [Mpc^3].
    Omega_m : total matter density today (CCL convention, incl. massive nu).
    h, w0, wa, m_nu, Neff, T_CMB, Omega_k :
        Background used for Omega_m(z), Omega_de(z) (CCL ``ccl_omega_x``,
        including radiation) and the Bird et al. neutrino terms. ``h`` enters
        through Omega_g h^2 (radiation in Omega_m(z), Omega_de(z)) and f_nu;
        pass it for CCL parity (``None`` -> 0.7; |dh| = 0.05 shifts P_NL by
        ~2.5e-4 at z <= 3). ``m_nu`` is the
        sum (eV) split with ``mass_split`` as in pyccl, or a list of masses;
        ``None``/0 means massless neutrinos. When ``m_nu`` is traced it must
        be above the hierarchy minimum.

    Jittable and differentiable w.r.t. ``log_pk_lin`` and the background
    parameters. Returns ln P_NL with the shape of ``log_pk_lin``.
    """
    bg = _background(Omega_m, h, Omega_b, w0, wa, m_nu, mass_split, Neff, T_CMB, Omega_k)
    return _halofit_table(log_k, z_pk, log_pk_lin, bg)


__all__ = ["halofit_log_pk_jax"]
