#!/usr/bin/env python3
# This file is part of GPUCosmo. Copyright (C) 2025-2026 Leonel Medina-Varela.
# SPDX-License-Identifier: GPL-3.0-or-later
# Derived in part from the following, translated to JAX and modified for GPUCosmo:
# - CAMB (https://github.com/cmbant/CAMB), Copyright 2000-2025 Antony Lewis and Anthony Challinor.
#   Licence: LICENSES/CAMB.txt.
# See NOTICE.md.
"""JAX-accelerated HMcode: full on-device path with ``vmap`` over redshift.

Growth ODE / wiggle init run once per cosmology; per-z σ(R), Bullock, NFW
(``jax.scipy.special.sici``), imead 3/4/5, and dewiggle run under ``jit`` +
``vmap``. Feedback ratio matches host: ``sqrt(p3/plin)*sqrt(p5/p4)``.

Enabled when the caller sets ``GPUCOSMO_EB_HMCODE_JAX`` (default on under
``GPUCOSMO_EB_DEVICE=1``).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from gpucosmo.jax.eb.hmcode import (
    assign_HM_cosmology,
    hmcode_mead2020_feedback,
    _grow,
    _init_wiggle_from_pk,
)
from gpucosmo.jax.power import MatterPowerTable

# Sheth–Tormen / dewiggle / growth (match hmcode.py)
_ST_A, _ST_Q, _ST_P = 0.21616, 0.707, 0.3
_KMIN_W, _KMAX_W, _NK_W = 5.0e-3, 5.0, 512
_WIGGLE_SIGMA, _KNORM_NW = 0.25, 0.03
_AINI, _AFIN, _N_ODE = 1.0e-4, 1.0, 801
_AMIN, _AMAX, _N_GROW = 1.0e-3, 1.0, 64
_N_MASS = 256

_BATCH_FN = None


def _try_jax():
    try:
        import jax
        import jax.numpy as jnp
        from jax.scipy.special import sici as jax_sici

        jax.config.update("jax_enable_x64", True)
        return jax, jnp, jax_sici
    except Exception:
        return None, None, None


def _cosm_floats(cosm) -> dict:
    """Floats (+ optional host growth/wiggle) from HMCosmology or mapping."""
    g = getattr
    if isinstance(cosm, dict):
        g = lambda d, k, default=None: d.get(k, default)  # noqa: E731
        om_m = float(g(cosm, "om_m"))
        om_b = float(g(cosm, "om_b"))
        om_c = float(g(cosm, "om_c", om_m - om_b))
        om_nu = float(g(cosm, "om_nu", 0.0))
        f_nu = float(g(cosm, "f_nu", om_nu / max(om_m, 1e-30)))
        f_b = float(g(cosm, "f_b", om_b / max(om_m, 1e-30)))
        f_c = float(g(cosm, "f_c", om_c / max(om_m, 1e-30)))
    else:
        om_m = float(cosm.om_m)
        om_b = float(cosm.om_b)
        om_c = float(cosm.om_c)
        om_nu = float(getattr(cosm, "om_nu", 0.0))
        f_nu = float(cosm.f_nu)
        f_b = float(cosm.f_b)
        f_c = float(cosm.f_c)
    out = dict(
        om_m=om_m,
        om_b=om_b,
        om_c=om_c,
        om_nu=om_nu,
        h=float(g(cosm, "h", 0.7)),
        ns=float(g(cosm, "ns", g(cosm, "n_s", 0.96))),
        logT_AGN=float(g(cosm, "logT_AGN", 7.7)),
        Tcmb=float(g(cosm, "Tcmb", 2.7255)),
        f_nu=f_nu,
        f_b=f_b,
        f_c=f_c,
    )
    for key in ("a_growth", "growth", "g_growth", "agrow", "log_k_wiggle", "pk_wiggle"):
        val = g(cosm, key, None)
        if val is not None:
            out[key] = np.asarray(val, dtype=np.float64)
    return out


# ---------------------------------------------------------------------------
# Pure jnp helpers (built once when JAX is available)
# ---------------------------------------------------------------------------


def _build_device_kernels(jax, jnp, jax_sici):
    """Return pure-jnp HMcode kernels + jitted batch fn."""

    def _hubble2(z, om_m):
        return om_m * (1.0 + z) ** 3 + (1.0 - om_m)

    def _ah(z, om_m):
        return -0.5 * (om_m * (1.0 + z) ** 3 + (1.0 - om_m))

    def _omega_m_hm(z, om_m):
        return om_m * (1.0 + z) ** 3 / jnp.maximum(_hubble2(z, om_m), 1e-30)

    def _growth_fv(d, v, a, om_m):
        z = -1.0 + 1.0 / a
        om = _omega_m_hm(z, om_m)
        return 3.0 * om * d / (2.0 * a * a) - (2.0 + _ah(z, om_m) / _hubble2(z, om_m)) * (
            v / a
        )

    def _fill_growtab(om_m):
        """Fixed-step RK4 growth ODE → 64-point tables (device)."""
        zinit = -1.0 + 1.0 / _AINI
        f = 1.0 - _omega_m_hm(zinit, om_m)
        d0 = _AINI ** (1.0 - 3.0 * f / 5.0)
        v0 = (1.0 - 3.0 * f / 5.0) * _AINI ** (-3.0 * f / 5.0)
        t = jnp.linspace(_AINI, _AFIN, _N_ODE)
        dt = (_AFIN - _AINI) / (_N_ODE - 1)

        def body(i, carry):
            x, v = carry
            a4 = t[i]
            d4, v4 = x[i], v[i]
            kx1 = dt * v4
            kv1 = dt * _growth_fv(d4, v4, a4, om_m)
            kx2 = dt * (v4 + 0.5 * kv1)
            kv2 = dt * _growth_fv(d4 + 0.5 * kx1, v4 + 0.5 * kv1, a4 + 0.5 * dt, om_m)
            kx3 = dt * (v4 + 0.5 * kv2)
            kv3 = dt * _growth_fv(d4 + 0.5 * kx2, v4 + 0.5 * kv2, a4 + 0.5 * dt, om_m)
            kx4 = dt * (v4 + kv3)
            kv4 = dt * _growth_fv(d4 + kx3, v4 + kv3, a4 + dt, om_m)
            x = x.at[i + 1].set(x[i] + (kx1 + 2 * kx2 + 2 * kx3 + kx4) / 6.0)
            v = v.at[i + 1].set(v[i] + (kv1 + 2 * kv2 + 2 * kv3 + kv4) / 6.0)
            return x, v

        x = jnp.zeros(_N_ODE).at[0].set(d0)
        v = jnp.zeros(_N_ODE).at[0].set(v0)
        x, _ = jax.lax.fori_loop(0, _N_ODE - 1, body, (x, v))
        gnorm = jnp.interp(1.0, t, x)
        d_norm = x / jnp.maximum(gnorm, 1e-30)
        a = jnp.linspace(_AMIN, _AMAX, _N_GROW)
        growth = jnp.interp(a, t, d_norm)
        ungrow = gnorm * growth
        integrand = ungrow / a
        da = a[1:] - a[:-1]
        mid = 0.5 * (integrand[1:] + integrand[:-1])
        agrow = jnp.concatenate(
            [jnp.array([gnorm * growth[0]]), gnorm * growth[0] + jnp.cumsum(mid * da)]
        )
        return a, growth, ungrow, agrow

    def _interp_grow(z, a_tab, y_tab):
        return jnp.interp(1.0 / (1.0 + z), a_tab, y_tab)

    def _f_mead(x, y, p0, p1, p2, p3):
        return p0 + p1 * (1.0 - x) + p2 * (1.0 - x) ** 2 + p3 * (1.0 - y)

    def _dc_mead(z, om_m, f_nu, a_g, g_g, ag):
        lg = _interp_grow(z, a_g, g_g)
        bG = _interp_grow(z, a_g, ag)
        Om = _omega_m_hm(z, om_m)
        aa = 1.0 / (1.0 + z)
        x, y = lg / aa, bG / aa
        dc0 = (3.0 / 20.0) * (12.0 * jnp.pi) ** (2.0 / 3.0)
        dc = 1.0
        dc = dc + _f_mead(x, y, -0.0069, -0.0208, 0.0312, 0.0021) * jnp.log10(Om)
        dc = dc + _f_mead(x, y, 0.0001, -0.0647, -0.0417, 0.0646)
        return dc * dc0 * (1.0 - 0.041 * f_nu)

    def _dv_mead(z, om_m, f_nu, a_g, g_g, ag):
        lg = _interp_grow(z, a_g, g_g)
        bG = _interp_grow(z, a_g, ag)
        Om = _omega_m_hm(z, om_m)
        aa = 1.0 / (1.0 + z)
        x, y = lg / aa, bG / aa
        Dv0 = 18.0 * jnp.pi**2
        Dv = 1.0
        Dv = Dv + _f_mead(x, y, -0.79, -10.17, 2.51, 6.51) * jnp.log10(Om)
        Dv = Dv + _f_mead(x, y, -1.89, 0.38, 18.8, -15.87) * jnp.log10(Om) ** 2
        return Dv * Dv0 * (1.0 + 0.763 * f_nu)

    def _tophat_w(x):
        return jnp.where(
            x < 1e-3,
            1.0 - x**2 / 10.0,
            3.0 * (jnp.sin(x) - x * jnp.cos(x)) / jnp.maximum(x**3, 1e-30),
        )

    def _tophat_w_deriv(x):
        return jnp.where(
            x < 1e-3,
            -x / 5.0 + x**3 / 70.0,
            (3.0 / jnp.maximum(x**4, 1e-30))
            * ((x**2 - 3.0) * jnp.sin(x) + 3.0 * x * jnp.cos(x)),
        )

    def _fine_pk_grid(k_h, pk_h):
        logk = jnp.log(k_h)
        n_fine = max(int(k_h.shape[0]) * 4, 1024)
        logk_f = jnp.linspace(logk[0], logk[-1], n_fine)
        pk_f = jnp.exp(jnp.interp(logk_f, logk, jnp.log(jnp.maximum(pk_h, 1e-30))))
        return logk_f, jnp.exp(logk_f), pk_f

    def _sigma_r_from_fine(logk_f, k_f, pk_f, R):
        delta2 = k_f**3 * pk_f / (2.0 * jnp.pi**2)
        w = _tophat_w(k_f[None, :] * R[:, None])
        integ = delta2[None, :] * w**2
        return jnp.sqrt(jnp.maximum(jnp.trapezoid(integ, logk_f, axis=1), 0.0))

    def _sigmav_from_fine(logk_f, k_f, pk_f):
        delta2 = k_f**3 * pk_f / (2.0 * jnp.pi**2)
        integ = delta2 / jnp.maximum(k_f**2, 1e-30) / 3.0
        return jnp.sqrt(jnp.maximum(jnp.trapezoid(integ, logk_f), 0.0))

    def _neff_at_r_fine(logk_f, k_f, pk_f, R, sig):
        kR = k_f * R
        w, wd = _tophat_w(kR), _tophat_w_deriv(kR)
        delta2 = k_f**3 * pk_f / (2.0 * jnp.pi**2)
        num = jnp.trapezoid(delta2 * w * wd * kR, logk_f)
        return -3.0 - 2.0 * num / jnp.maximum(sig**2, 1e-30)

    def _gnu_st(nu):
        return _ST_A * (1.0 + (_ST_Q * nu * nu) ** (-_ST_P)) * jnp.exp(
            -0.5 * _ST_Q * nu * nu
        )

    def _tk_nw(k, om_m, om_b, h, Tcmb):
        wm, wb = om_m * h**2, om_b * h**2
        rb = om_b / jnp.maximum(om_m, 1e-30)
        s = 44.5 * jnp.log(9.83 / wm) / jnp.sqrt(1.0 + 10.0 * wb**0.75)
        alpha = 1.0 - 0.328 * jnp.log(431.0 * wm) * rb + 0.38 * jnp.log(22.3 * wm) * rb**2
        Gamma = om_m * h * (alpha + (1.0 - alpha) / (1.0 + (0.43 * k * s * h) ** 4))
        q = k * (Tcmb / 2.7) ** 2 / jnp.maximum(Gamma, 1e-30)
        L = jnp.log(2.0 * jnp.e + 1.8 * q)
        C = 14.2 + 731.0 / (1.0 + 62.5 * q)
        return L / (L + C * q**2)

    def _smooth_gaussian_logk(logk, f, sigma):
        nsig = 3.0
        edge = (jnp.abs(logk - logk[0]) < nsig * sigma) | (
            jnp.abs(logk - logk[-1]) < nsig * sigma
        )
        d2 = (logk[:, None] - logk[None, :]) ** 2
        w = jnp.exp(-d2 / (2.0 * sigma**2))
        smoothed = (w @ f) / jnp.sum(w, axis=1)
        return jnp.where(edge, f, smoothed)

    def _init_wiggle(k_h, pk_h, growz, om_m, om_b, h, ns, Tcmb):
        logk = jnp.linspace(jnp.log(_KMIN_W), jnp.log(_KMAX_W), _NK_W)
        k = jnp.exp(logk)
        g2 = jnp.maximum(growz, 1e-30) ** 2
        pk0 = jnp.exp(jnp.interp(logk, jnp.log(k_h), jnp.log(jnp.maximum(pk_h, 1e-30)))) / g2
        pk_nw = (k**ns) * _tk_nw(k, om_m, om_b, h, Tcmb) ** 2
        pk_norm = jnp.exp(
            jnp.interp(jnp.log(_KNORM_NW), logk, jnp.log(jnp.maximum(pk0, 1e-30)))
        )
        pk_nw_norm = jnp.interp(_KNORM_NW, k, pk_nw)
        pk_nw = pk_nw * (pk_norm / jnp.maximum(pk_nw_norm, 1e-30))
        ratio = pk0 / jnp.maximum(pk_nw, 1e-30)
        pk_smooth = _smooth_gaussian_logk(logk, ratio, _WIGGLE_SIGMA) * pk_nw
        return logk, pk0 - pk_smooth

    def _p_dewiggle(k_h, plin, z, sigv, growz, log_k_w, pk_w):
        logk = jnp.log(k_h)
        pk_wi = jnp.interp(logk, log_k_w, pk_w, left=0.0, right=0.0)
        pk_wi = jnp.where((logk < log_k_w[0]) | (logk > log_k_w[-1]), 0.0, pk_wi)
        f = jnp.exp(-((k_h * sigv) ** 2))
        return plin + (f - 1.0) * pk_wi * growz**2

    def _nfw_wk_eta(k_h, nu, rv, c_m, eta):
        ke = k_h[:, None] * (nu[None, :] ** eta)
        rs = rv / jnp.maximum(c_m, 1e-6)
        krs = jnp.maximum(ke * rs[None, :], 1e-8)
        c1 = 1.0 + c_m
        f = jnp.log(1.0 + c_m) - c_m / (1.0 + c_m)
        si1, ci1 = jax_sici(krs)
        si2, ci2 = jax_sici(c1[None, :] * krs)
        w = (
            jnp.sin(krs) * (si2 - si1)
            + jnp.cos(krs) * (ci2 - ci1)
            - jnp.sin(c_m[None, :] * krs) / (c1[None, :] * krs)
        ) / jnp.maximum(f[None, :], 1e-30)
        return jnp.where(ke * rs[None, :] < 1e-4, 1.0, w)

    def _as_conc(z, imead, logT_AGN):
        if imead == 3:
            return 5.196
        if imead == 4:
            return 4.0
        theta = logT_AGN - 7.8
        return (3.44 - 0.496 * theta) * 10.0 ** (z * (-0.0671 - 0.0371 * theta))

    def _bullock(m, R, z, dc, growz, a_tab, g_tab, logk_f, k_f, pk_f, imead, logT_AGN):
        As = _as_conc(z, imead, logT_AGN)
        f_b = 0.01 ** (1.0 / 3.0)
        sigf = _sigma_r_from_fine(logk_f, k_f, pk_f, R * f_b)
        rhs = dc * growz / jnp.maximum(sigf, 1e-30)
        future = rhs > growz
        af = jnp.clip(jnp.interp(rhs, g_tab, a_tab), a_tab[0], a_tab[-1])
        zf = jnp.where(future, z, -1.0 + 1.0 / af)
        return jnp.clip(As * (1.0 + zf) / (1.0 + z), 1.0, 100.0)

    def _baryonify_wk(wk, m, z, logT_AGN, om_b, om_m, f_c, f_b):
        theta = logT_AGN - 7.8
        mb = (10.0 ** (13.87 + 1.81 * theta)) * 10.0 ** ((-0.108 + 0.195 * theta) * z)
        fs = jnp.minimum(
            (0.0201 - 0.003 * theta) * 10.0 ** ((0.409 + 0.0224 * theta) * z),
            om_b / jnp.maximum(om_m, 1e-30),
        )
        r = (m / mb) ** 2
        return wk * (f_c + (f_b - fs) * r / (1.0 + r)) + fs

    def _halomod(k_h, plin, z, sig8z, m, R, sig, nu, gnu, dc, Dv, fine, rho_m, cosm, imead):
        logk_f, k_f, pk_f = fine
        growz = _interp_grow(z, cosm["a_growth"], cosm["growth"])
        c_m = _bullock(
            m, R, z, dc, growz, cosm["a_growth"], cosm["growth"],
            logk_f, k_f, pk_f, imead, cosm["logT_AGN"],
        )
        rv = R / (jnp.maximum(Dv, 1.0) ** (1.0 / 3.0))
        eta = 0.0 if imead in (4, 5) else 0.1281 * sig8z ** (-0.3644)
        fd = 0.0 if imead in (4, 5) else jnp.minimum(0.2696 * sig8z**0.9403, 0.99)
        ks = 0.05618 * sig8z ** (-1.013)
        kdamp = 0.05699 * sig8z ** (-1.089)

        rnl = jnp.where(
            nu[0] > 1.0,
            R[0],
            jnp.exp(jnp.interp(0.0, jnp.log(nu), jnp.log(R))),
        )
        sig_rnl = jnp.interp(jnp.log(rnl), jnp.log(R), sig)
        neff = jnp.clip(
            _neff_at_r_fine(logk_f, k_f, pk_f, rnl, sig_rnl),
            cosm["ns"] - 4.0,
            cosm["ns"],
        )
        alpha = 1.875 * (1.603**neff) if imead == 3 else 1.0

        order = jnp.argsort(nu)
        wk = _nfw_wk_eta(k_h, nu, rv, c_m, eta)
        if imead == 5:
            wk = _baryonify_wk(
                wk, m, z, cosm["logT_AGN"], cosm["om_b"], cosm["om_m"],
                cosm["f_c"], cosm["f_b"],
            )
        wk = wk[:, order]
        integrand = gnu[order][None, :] * wk**2 * m[order][None, :]
        dnu = jnp.diff(nu[order])
        sum_nu = jnp.sum(0.5 * (integrand[:, 1:] + integrand[:, :-1]) * dnu[None, :], axis=1)
        p1h_d2 = (sum_nu / rho_m) * k_h**3 / (2.0 * jnp.pi**2)
        x4 = (k_h / ks) ** 4
        p1h_d2 = p1h_d2 * x4 / (1.0 + x4)
        if imead in (3, 4):
            p1h_d2 = p1h_d2 * (1.0 - cosm["f_nu"]) ** 2

        plin_d2 = k_h**3 * plin / (2.0 * jnp.pi**2)
        if imead == 3:
            sigv = _sigmav_from_fine(logk_f, k_f, pk_f)
            p2h = _p_dewiggle(
                k_h, plin, z, sigv, growz, cosm["log_k_wiggle"], cosm["pk_wiggle"]
            )
            p2h_d2 = k_h**3 * p2h / (2.0 * jnp.pi**2)
            x = (k_h / kdamp) ** 2.85
            p2h_d2 = jnp.maximum(p2h_d2 * (1.0 - fd * x / (1.0 + x)), 0.0)
            alpha = jnp.clip(alpha, 0.5, 2.0)
            pfull_d2 = (p2h_d2**alpha + p1h_d2**alpha) ** (1.0 / jnp.maximum(alpha, 1e-6))
        else:
            pfull_d2 = plin_d2 + p1h_d2
        return jnp.maximum(
            pfull_d2 * (2.0 * jnp.pi**2) / jnp.maximum(k_h**3, 1e-30), 1e-30
        )

    def _one_z(k_h, z, plin, cosm):
        rho_m = 2.775e11 * cosm["om_m"]
        m = jnp.logspace(0.0, 18.0, _N_MASS)
        R = (3.0 * m / (4.0 * jnp.pi * rho_m)) ** (1.0 / 3.0)
        fine = _fine_pk_grid(k_h, plin)
        sig = _sigma_r_from_fine(*fine, R)
        sig8z = _sigma_r_from_fine(*fine, jnp.array([8.0]))[0]
        dc = _dc_mead(
            z, cosm["om_m"], cosm["f_nu"], cosm["a_growth"], cosm["g_growth"], cosm["agrow"]
        )
        Dv = _dv_mead(
            z, cosm["om_m"], cosm["f_nu"], cosm["a_growth"], cosm["g_growth"], cosm["agrow"]
        )
        nu = dc / jnp.maximum(sig, 1e-30)
        gnu = _gnu_st(nu)
        kw = dict(
            k_h=k_h, plin=plin, z=z, sig8z=sig8z, m=m, R=R, sig=sig, nu=nu, gnu=gnu,
            dc=dc, Dv=Dv, fine=fine, rho_m=rho_m, cosm=cosm,
        )
        p3 = _halomod(**kw, imead=3)
        p4 = _halomod(**kw, imead=4)
        p5 = _halomod(**kw, imead=5)
        ratio = jnp.sqrt(p3 / jnp.maximum(plin, 1e-30)) * jnp.sqrt(
            p5 / jnp.maximum(p4, 1e-30)
        )
        return plin * ratio**2

    @jax.jit
    def batch_z(k_h, z_arr, plin_h_nz, cosm):
        return jax.vmap(lambda z, plin: _one_z(k_h, z, plin, cosm))(z_arr, plin_h_nz)

    return SimpleNamespace(
        fill_growtab=_fill_growtab,
        init_wiggle=_init_wiggle,
        one_z=_one_z,
        batch_z=batch_z,
        interp_grow=_interp_grow,
    )


def _get_kernels():
    global _BATCH_FN
    if _BATCH_FN is not None:
        return _BATCH_FN
    jax, jnp, jax_sici = _try_jax()
    if jax is None:
        return None
    _BATCH_FN = _build_device_kernels(jax, jnp, jax_sici)
    return _BATCH_FN


def _prepare_cosm_device(cosm, k_h, pk0_h, z0):
    """Build device cosm dict: growth on device, wiggle once from z≈0."""
    jax, jnp, _ = _try_jax()
    ker = _get_kernels()
    cf = _cosm_floats(cosm)
    a_g, growth, g_g, agrow = ker.fill_growtab(jnp.asarray(cf["om_m"]))
    growz = float(ker.interp_grow(jnp.asarray(z0), a_g, growth))
    # Prefer host wiggle if already filled (bit-stable vs NumPy path)
    if cf.get("log_k_wiggle") is not None and cf.get("pk_wiggle") is not None:
        log_k_w = jnp.asarray(cf["log_k_wiggle"])
        pk_w = jnp.asarray(cf["pk_wiggle"])
    else:
        log_k_w, pk_w = ker.init_wiggle(
            jnp.asarray(k_h),
            jnp.asarray(pk0_h),
            jnp.asarray(growz),
            jnp.asarray(cf["om_m"]),
            jnp.asarray(cf["om_b"]),
            jnp.asarray(cf["h"]),
            jnp.asarray(cf["ns"]),
            jnp.asarray(cf["Tcmb"]),
        )
    return {
        "om_m": jnp.asarray(cf["om_m"]),
        "om_b": jnp.asarray(cf["om_b"]),
        "om_c": jnp.asarray(cf["om_c"]),
        "h": jnp.asarray(cf["h"]),
        "ns": jnp.asarray(cf["ns"]),
        "logT_AGN": jnp.asarray(cf["logT_AGN"]),
        "Tcmb": jnp.asarray(cf["Tcmb"]),
        "f_nu": jnp.asarray(cf["f_nu"]),
        "f_b": jnp.asarray(cf["f_b"]),
        "f_c": jnp.asarray(cf["f_c"]),
        "a_growth": a_g,
        "growth": growth,
        "g_growth": g_g,
        "agrow": agrow,
        "log_k_wiggle": log_k_w,
        "pk_wiggle": pk_w,
    }


def hmcode_batch_z_jax(k_h, z_arr, plin_h_nz, cosm_dict):
    """Full feedback HMcode for all redshifts. Returns ``pk_nl_h`` (n_z, n_k).

    ``cosm_dict`` may be an ``HMCosmology`` or a float mapping; growth/wiggle
    tables are built on device as needed.
    """
    ker = _get_kernels()
    if ker is None:
        raise RuntimeError("JAX unavailable for hmcode_batch_z_jax")
    import jax.numpy as jnp

    k_h = np.asarray(k_h, dtype=np.float64)
    z_arr = np.asarray(z_arr, dtype=np.float64)
    plin_h_nz = np.asarray(plin_h_nz, dtype=np.float64)
    iz0 = int(np.argmin(np.abs(z_arr)))
    cosm_dev = _prepare_cosm_device(cosm_dict, k_h, plin_h_nz[iz0], float(z_arr[iz0]))
    out = ker.batch_z(
        jnp.asarray(k_h),
        jnp.asarray(z_arr),
        jnp.asarray(plin_h_nz),
        cosm_dev,
    )
    return np.asarray(out)


def hmcode_one_z_jax(k_h: np.ndarray, z: float, plin_h: np.ndarray, cosm) -> np.ndarray:
    """Single-z feedback; prefers full device path, else host HMcode."""
    ker = _get_kernels()
    if ker is None:
        return hmcode_mead2020_feedback(k_h, z, plin_h, cosm)
    try:
        import jax.numpy as jnp

        cosm_dev = _prepare_cosm_device(cosm, k_h, plin_h, float(z))
        return np.asarray(
            ker.one_z(jnp.asarray(k_h), jnp.asarray(z), jnp.asarray(plin_h), cosm_dev)
        )
    except Exception:
        return hmcode_mead2020_feedback(k_h, z, plin_h, cosm)


def nonlinear_pk_hmcode_feedback_jax(
    lin: MatterPowerTable,
    *,
    Omega_m: float,
    Omega_b: float,
    h: float = 0.7,
    n_s: float = 0.96,
    logT_AGN: float = 7.7,
) -> MatterPowerTable:
    """Batch-over-z device HMcode; falls back to per-z loop on failure."""
    cosm = assign_HM_cosmology(Omega_m, Omega_b, h, n_s, logT_AGN=logT_AGN)
    k_mpc = np.exp(lin.log_k)
    k_h = k_mpc / h
    plin_h_nz = np.exp(lin.log_pk) * (h**3)

    iz0 = int(np.argmin(np.abs(lin.z)))
    _init_wiggle_from_pk(k_h, plin_h_nz[iz0], cosm, _grow(float(lin.z[iz0]), cosm))

    try:
        pk_nl_h = hmcode_batch_z_jax(k_h, lin.z, plin_h_nz, cosm)
        log_pk_nl = np.log(np.maximum(pk_nl_h / (h**3), 1e-30))
        return MatterPowerTable(log_k=lin.log_k, z=lin.z, log_pk=log_pk_nl)
    except Exception:
        pass

    log_pk_nl = np.zeros_like(lin.log_pk)
    for iz, z in enumerate(lin.z):
        pk_nl_h = hmcode_one_z_jax(k_h, float(z), plin_h_nz[iz], cosm)
        log_pk_nl[iz] = np.log(np.maximum(pk_nl_h / (h**3), 1e-30))
    return MatterPowerTable(log_k=lin.log_k, z=lin.z, log_pk=log_pk_nl)


__all__ = [
    "hmcode_batch_z_jax",
    "hmcode_one_z_jax",
    "nonlinear_pk_hmcode_feedback_jax",
]
