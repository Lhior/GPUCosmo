# This file is part of GPUCosmo. Copyright (C) 2025-2026 Leonel Medina-Varela.
# SPDX-License-Identifier: GPL-3.0-or-later
# Derived in part from the following, translated to JAX and modified for GPUCosmo:
# - ISiTGR (https://github.com/mishakb/ISiTGR), Dossett, Ishak, Garcia-Quintero et al.,
#   built on CAMB, Copyright 2000-2018 Antony Lewis and Anthony Challinor.
#   Licence: LICENSES/ISiTGR.txt.
# See NOTICE.md.
"""μ–Σ modified gravity for the EB solver: a port of ISiTGR's synchronous-gauge branch.

Parametrization (ISiTGR ``ISiTGR_muSigma``, DES/DESI convention)::

    μ(a, k) = 1 + μ0 Ω_DE(a)/Ω_DE,0 · s_1(a, k)
    Σ(a, k) = 1 + Σ0 Ω_DE(a)/Ω_DE,0 · s_2(a, k)
    s_i = (1 + c_i x²)/(1 + x²),  x = λ H(a)/k   (λ = 0: time-only)

with ``k²Ψ = −4πG a² μ [ρΔ + 3(ρ+p)σ_aniso]`` and the Weyl potential
``k²(Φ+Ψ) = −8πG a² Σ [ρΔ + 3/2 (ρ+p)σ_aniso]``.

ISiTGR (``fortran/equations.f90`` ``derivs``) applies the modified Einstein
equations only for a ≥ 0.02 (``use_mg``). There the shear follows from the
Newtonian potential Φ (``σ = η_k/ℋ − kΦ/ℋ``), η̇_k from the time derivative of
the Σ constraint (needs Σρπ̇), and ``z = σ − 3η̇_k/k²`` replaces the 00
constraint. The functions below mirror ISiTGR's ``ISiTGR_mu``, ``ISiTGR_mu_dot``,
``ISiTGR_Sigma``, ``ISiTGR_Sigma_dot``, ``OmegaDE`` and ``OmegaDEdot`` and the
``use_mg`` block (flat universe, β_k = 1). Quantities use CAMB's units
(``grho = 8πG a² ρ``, conformal time in Mpc).
"""

from __future__ import annotations

import jax.numpy as jnp

# ISiTGR: ``use_mg = (CP%GR == 0) .and. (a >= 0.02d0)``.
MG_A_MIN = 0.02


def mg_active(a):
    return a >= MG_A_MIN


def omega_de_and_dot(grhov, grho, adotoa, Hdot, a, w0, wa):
    """Ω_DE(a) = ρ_DE/ρ_tot and its conformal-time derivative (ISiTGR ``OmegaDE[dot]``)."""
    om = grhov / grho
    omdot = -om * (2.0 * (Hdot - adotoa * adotoa) / adotoa
                   + 3.0 * (1.0 + w0) * adotoa + 3.0 * wa * adotoa * (1.0 - a))
    return om, omdot


def mu_sigma(*, k, a, adotoa, Hdot, grhov, grho, params):
    """(μ, μ̇, Σ, Σ̇) at (a, k)."""
    w0 = params.get("w0", -1.0)
    wa = params.get("wa", 0.0)
    om0 = params["Omega_de"]
    om, omdot = omega_de_and_dot(grhov, grho, adotoa, Hdot, a, w0, wa)
    lam = params.get("mg_lambda", 0.0)
    c1 = params.get("mg_c1", 1.0)
    c2 = params.get("mg_c2", 1.0)
    s2 = (lam * (adotoa / a) / k) ** 2
    s2dot = 2.0 * s2 * (Hdot - adotoa * adotoa) / adotoa
    s_mu = (1.0 + c1 * s2) / (1.0 + s2)
    s_mu_dot = s2dot * (c1 - 1.0) / (1.0 + s2) ** 2
    s_sig = (1.0 + c2 * s2) / (1.0 + s2)
    s_sig_dot = s2dot * (c2 - 1.0) / (1.0 + s2) ** 2
    mu0 = params.get("mu0", 0.0)
    sig0 = params.get("Sigma0", 0.0)
    mu = 1.0 + mu0 * (om / om0) * s_mu
    mudot = mu0 * (omdot / om0 * s_mu + om / om0 * s_mu_dot)
    sig = 1.0 + sig0 * (om / om0) * s_sig
    sigdot = sig0 * (omdot / om0 * s_sig + om / om0 * s_sig_dot)
    return mu, mudot, sig, sigdot


def rsa_radiation_mg(k, grhoc, grhob, clxc, clxb):
    """ISiTGR's radiation-streaming closure under MG: (δ_r, q_r) for photons and ν."""
    clx = 2.0 * (grhoc * clxc + grhob * clxb) / 3.0 / (k * k)
    q = clx * k / jnp.sqrt((grhoc + grhob) / 3.0) * (2.0 / 3.0)
    return clx, q


def newtonian_potentials(*, k, mu, sig, rho_delta, dgpi, etak, adotoa):
    """ISiTGR (Ψ, Φ, σ): ``Ψ = −μ(ρΔ + 2Π)/(2k²)``, ``Φ = −Σ(ρΔ + Π)/k² − Ψ``."""
    k2 = k * k
    psi = -0.5 * mu * (rho_delta + 2.0 * dgpi) / k2
    phi = -sig * (rho_delta + dgpi) / k2 - psi
    sigma = etak / adotoa - k * phi / adotoa
    return psi, phi, sigma


def etak_dot_mg(*, k, mu, mudot, sig, sigdot, adotoa, Hdot, rho_delta, dgq, dgpi,
                dgpi_3wplus1, pidot_sum, sigma, rho_plus_p):
    """ISiTGR μ–Σ ``etakdot`` (flat, β_k = 1)."""
    k2 = k * k
    two_s_mu = 2.0 * sig - mu
    f_ms = k2 + 1.5 * two_s_mu * rho_plus_p
    f_1 = 1.0 + 3.0 * (adotoa * adotoa - Hdot) / k2
    return k / (2.0 * f_ms) * (
        k * two_s_mu * f_1 * dgq
        + 2.0 * (mu - sig) * pidot_sum
        + rho_delta * ((mudot - 2.0 * sigdot) + 2.0 * adotoa * (sig - mu))
        + 2.0 * dgpi * (mudot - sigdot + adotoa * two_s_mu - adotoa * mu)
        + k * sigma * (-2.0 * (adotoa * adotoa - Hdot) + two_s_mu * rho_plus_p)
        + 2.0 * adotoa * (sig - mu) * dgpi_3wplus1
    )


def psi_dot_mg(*, k, mu, mudot, adotoa, rho_delta, dgq, dgpi, dgpi_3wplus1, pidot_sum,
               sigma, phidot, psi, Hdot, rho_plus_p):
    """ISiTGR ``TGR_Psidot`` (for the ISW source)."""
    k2 = k * k
    f_1 = 1.0 + 3.0 * (adotoa * adotoa - Hdot) / k2
    return (-mudot / (2.0 * k2) * (rho_delta + 2.0 * dgpi)
            + mu / (2.0 * k2) * (adotoa * rho_delta - 2.0 * pidot_sum + k * f_1 * dgq
                                 + 2.0 * adotoa * dgpi_3wplus1 + 2.0 * adotoa * dgpi
                                 + (f_1 * sigma * k - 3.0 * (phidot + adotoa * psi))
                                 * rho_plus_p))


__all__ = [
    "MG_A_MIN",
    "etak_dot_mg",
    "mg_active",
    "mu_sigma",
    "newtonian_potentials",
    "omega_de_and_dot",
    "psi_dot_mg",
    "rsa_radiation_mg",
]
