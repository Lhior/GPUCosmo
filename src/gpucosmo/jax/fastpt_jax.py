#!/usr/bin/env python3
# This file is part of GPUCosmo. Copyright (C) 2025-2026 Leonel Medina-Varela.
# SPDX-License-Identifier: GPL-3.0-or-later
# Derived in part from the following, translated to JAX and modified for GPUCosmo:
# - FAST-PT (https://github.com/JoeMcEwen/FAST-PT), Copyright (c) 2018, the FAST-PT
#   developers.
#   Licence: LICENSES/FAST-PT.txt.
# See NOTICE.md.
"""Pure-JAX port of the FAST-PT (v4) terms used by CCL's EulerianPTCalculator.

A real ``fastpt.FASTPT`` object is built once at init; its k grids and all
P-independent tables (the ``X_*`` tensor/scalar coefficient matrices, the
direct-convolution kernels of ``P_IA_deltaE2`` / ``P_IA_B`` / ``P_13_reg`` /
``Y1_reg_NL``) are copied into JAX constants. Everything that depends on the
linear power spectrum (power-law extrapolation, biasing, padding, FFTs,
Fourier-space convolutions, windowing, prefactors, trimming) is written in
``jax.numpy`` so the methods are jittable and differentiable w.r.t. ``P``.
``P_window`` / ``C_window`` are static (trace-time) Python values.

CCL defaults (``EulerianPTCalculator``)::

    k_s = np.logspace(-4, 2, int(6 * 20))
    pt = FASTPTJax(k_s, low_extrap=-5, high_extrap=3, n_pad=int(1.0 * len(k_s)))
    tpl = ept_ia_templates(pt, cosmo.linear_matter_power(k_s, 1.0), C_window=0.75)
"""

from __future__ import annotations

import contextlib
import io
from typing import Any, Callable

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

_TENSOR_TERMS = (
    "X_IA_E",
    "X_IA_B",
    "X_IA_A",
    "X_IA_DEE",
    "X_IA_DBB",
    "X_IA_deltaE1",
    "X_IA_0E0E",
    "X_IA_0B0B",
)

# P22 Legendre weights of FASTPT._get_P_1loop (rows of X_spt).
_P22_COEF = np.array(
    [2 * 1219 / 1470.0, 2 * 671 / 1029.0, 2 * 32 / 1715.0, 2 * 1 / 3.0,
     2 * 62 / 35.0, 2 * 8 / 35.0, 1 / 3.0]
)


def _capture_kernel(module: Any, conv_name: str, func: Callable, k: np.ndarray) -> np.ndarray:
    """Return the log-k convolution kernel ``f`` built inside a FAST-PT helper.

    The helper's module-level ``convolve``/``fftconvolve`` is swapped for a
    recorder during one call, so ``f`` is bit-identical to FAST-PT's.
    """
    captured: dict[str, np.ndarray] = {}
    orig = getattr(module, conv_name)

    def _record(a, b, *args, **kwargs):
        captured["f"] = np.array(b, dtype=float)
        return orig(a, b, *args, **kwargs)

    setattr(module, conv_name, _record)
    try:
        func(k, np.ones_like(k))
    finally:
        setattr(module, conv_name, orig)
    return captured["f"]


def _toeplitz_from_kernel(f: np.ndarray, dL: float) -> np.ndarray:
    """Matrix T with ``T @ P == (np.convolve(P, f) * dL)[N-1:2N-1]``."""
    N = (f.size + 1) // 2
    i = np.arange(N)[:, None]
    j = np.arange(N)[None, :]
    return f[i - j + N - 1] * dL


def _fftconvolve(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Full linear convolution along the last axis (scipy ``fftconvolve``)."""
    n = a.shape[-1] + b.shape[-1] - 1
    nfft = 1 << (n - 1).bit_length()
    out = jnp.fft.ifft(jnp.fft.fft(a, nfft) * jnp.fft.fft(b, nfft))
    return out[..., :n]


class FASTPTJax:
    """JAX mirror of ``fastpt.FASTPT`` for the EPT/IA terms used by CCL."""

    def __init__(
        self,
        k: np.ndarray,
        low_extrap: float | None = None,
        high_extrap: float | None = None,
        n_pad: int | None = None,
        verbose: bool = False,
    ) -> None:
        import fastpt
        import fastpt.IA.IA_ABD as ia_abd
        import fastpt.IA.IA_ta as ia_ta
        import fastpt.utils.matter_power_spt as mps
        from fastpt.utils.fastpt_extr import c_window, p_window

        self._c_window_fn = c_window
        self._p_window_fn = p_window
        k = np.asarray(k, dtype=float)
        sink = io.StringIO() if not verbose else None
        with contextlib.redirect_stdout(sink) if sink else contextlib.nullcontext():
            fpt = fastpt.FASTPT(k, low_extrap=low_extrap, high_extrap=high_extrap,
                                n_pad=n_pad, verbose=verbose)
            tensor = {name: getattr(fpt, name) for name in _TENSOR_TERMS}
            x_spt = fpt.X_spt
        self.fpt = fpt

        self.k_original = np.asarray(fpt.k_original)
        self.k_extrap = np.asarray(fpt.k_extrap)
        self.k_final = np.asarray(fpt.k_final)
        self.N = int(fpt.N)
        self.n_pad = int(fpt.n_pad)
        self.m = np.asarray(fpt.m)
        self.l = np.asarray(fpt.l)
        self.low_extrap = low_extrap
        self.high_extrap = high_extrap
        self.extrap = bool(fpt.extrap)
        self.id_pad = (np.asarray(fpt.id_pad) if self.n_pad > 0
                       else np.arange(self.k_final.size))
        if self.extrap:
            ek = fpt.EK
            self._DL = float(ek.DL)
            self.id_extrap = np.asarray(ek.id_extrap)
            self._k_low = jnp.asarray(ek.k_low) if low_extrap is not None else None
            self._k_high = jnp.asarray(ek.k_high) if high_extrap is not None else None
            self._k_min = float(ek.k_min) if low_extrap is not None else None
            self._k_max = float(ek.k_max) if high_extrap is not None else None

        kf_pad = self.k_final[self.id_pad]
        self._tensor: dict[str, tuple[jnp.ndarray, ...]] = {}
        for name, X in tensor.items():
            # FAST-PT stores p/nu1/nu2 as object arrays; cast to float.
            pf, p, nu1, nu2 = (np.asarray(x, dtype=float) for x in X[:4])
            g_m, g_n, h_l = X[4:]
            self._tensor[name] = (
                jnp.asarray(self.k_extrap[None, :] ** (-nu1[:, None])),
                jnp.asarray(self.k_extrap[None, :] ** (-nu2[:, None])),
                jnp.asarray(g_m), jnp.asarray(g_n), jnp.asarray(h_l),
                jnp.asarray(pf[:, None] * kf_pad[None, :] ** p[:, None]),
            )
        pf, p, g_m, g_n, two_part_l, h_l = x_spt
        self._scalar = (
            jnp.asarray(g_m), jnp.asarray(g_n), jnp.asarray(h_l * two_part_l),
            jnp.asarray(pf[:, None] * kf_pad[None, :] ** (-p[:, None] - 2)),
        )
        self._k_extrap_j = jnp.asarray(self.k_extrap)
        self._kfinal_nu_pad = jnp.asarray(kf_pad ** (-2.0))

        # Direct log-k convolution kernels as Toeplitz matrices.
        ko, ke = self.k_original, self.k_extrap
        dLo, dLe = np.log(ko[1]) - np.log(ko[0]), np.log(ke[1]) - np.log(ke[0])
        self._T_deltaE2 = jnp.asarray(_toeplitz_from_kernel(
            _capture_kernel(ia_ta, "convolve", ia_ta.P_IA_deltaE2, ko), dLo))
        self._T_B = jnp.asarray(_toeplitz_from_kernel(
            _capture_kernel(ia_abd, "convolve", ia_abd.P_IA_B, ko), dLo))
        self._T_13 = jnp.asarray(_toeplitz_from_kernel(
            _capture_kernel(mps, "fftconvolve", mps.P_13_reg, ke), dLe))
        self._T_Y1 = jnp.asarray(_toeplitz_from_kernel(
            _capture_kernel(mps, "fftconvolve", mps.Y1_reg_NL, ke), dLe))
        self._ko3 = jnp.asarray(ko ** 3)
        self._ke3 = jnp.asarray(ke ** 3)
        self._log_ke = jnp.asarray(np.log(ke))
        self._cwin_cache: dict[float, jnp.ndarray] = {}

    # ---------------------------------------------------------------- helpers
    def _c_window(self, C_window: float | None) -> jnp.ndarray | None:
        if C_window is None:
            return None
        key = float(C_window)
        if key not in self._cwin_cache:
            with jax.ensure_compile_time_eval():  # cached: concrete even when first built inside a trace
                self._cwin_cache[key] = jnp.asarray(
                    self._c_window_fn(self.m, int(C_window * self.N // 2.0)))
        return self._cwin_cache[key]

    def _extend(self, P: jnp.ndarray) -> jnp.ndarray:
        """k_extend.extrap_P_low / extrap_P_high with P-derived slopes."""
        P = jnp.asarray(P, dtype=jnp.float64)
        if self.low_extrap is not None:
            ns = (jnp.log(P[1]) - jnp.log(P[0])) / self._DL
            amp = P[0] / self._k_min ** ns
            P = jnp.concatenate([self._k_low ** ns * amp, P])
        if self.high_extrap is not None:
            ns = (jnp.log(P[-1]) - jnp.log(P[-2])) / self._DL
            amp = P[-1] / self._k_max ** ns
            P = jnp.concatenate([P, self._k_high ** ns * amp])
        return P

    def _cut(self, x: jnp.ndarray) -> jnp.ndarray:
        """FASTPT._apply_extrapolation: trim back to the input k grid."""
        return x[..., self.id_extrap] if self.extrap else x

    def _pad(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.n_pad <= 0:
            return x
        widths = [(0, 0)] * (x.ndim - 1) + [(self.n_pad, self.n_pad)]
        return jnp.pad(x, widths)

    def _fourier(self, P_b: jnp.ndarray, C_window: float | None, scalar: bool):
        """FASTPT._cache_fourier_coefficients; returns (c_m, c_m[m >= 0])."""
        c_pos = jnp.fft.rfft(P_b, axis=-1)
        if scalar:
            c_pos = c_pos.at[..., -1].divide(2.0)
        c_neg = jnp.conj(c_pos[..., 1:])[..., ::-1]
        c_m = jnp.concatenate([c_neg, c_pos], axis=-1) / float(self.N)
        W = self._c_window(C_window)
        if W is not None:
            c_m = c_m * W
        return c_m

    def _conv_to_k(self, C_l: jnp.ndarray) -> jnp.ndarray:
        """Reorder l=0..N-1,-N..-1, ifft*size, keep even elements and id_pad."""
        N = self.N
        C_l = jnp.concatenate([C_l[..., N:2 * N], C_l[..., :N]], axis=-1)
        A_k = jnp.fft.ifft(C_l, axis=-1) * C_l.shape[-1]
        return jnp.real(A_k[..., ::2])[..., self.id_pad]

    def J_k_tensor(self, P: jnp.ndarray, name: str, P_window=None,
                   C_window: float | None = None) -> jnp.ndarray:
        """Summed tensor term ``P_fin`` on k_extrap (before trimming)."""
        bias1, bias2, g_m, g_n, h_l, kfac = self._tensor[name]
        P = self._extend(P)
        P_b1 = P[None, :] * bias1
        P_b2 = P[None, :] * bias2
        if P_window is not None:
            W = jnp.asarray(self._p_window_fn(self.k_extrap, P_window[0], P_window[1]))
            P_b1, P_b2 = P_b1 * W, P_b2 * W
        c_m = self._fourier(self._pad(P_b1), C_window, scalar=False)
        c_n = self._fourier(self._pad(P_b2), C_window, scalar=False)
        C_l = _fftconvolve(c_m * g_m, c_n * g_n) * h_l
        return jnp.sum(self._conv_to_k(C_l) * kfac, axis=0)

    def J_k_scalar(self, P: jnp.ndarray, C_window: float | None = None, P_window=None):
        """X_spt scalar terms (nu=-2): (Ps, mat) on k_extrap. ``P_window`` tapers the biased input
        as in J_k_tensor (FAST-PT 2.x/3.x and ≥ 4.1.1; 4.0–4.1.0 ignored it on the scalar path)."""
        g_m, g_n, h2_l, kfac = self._scalar
        P = self._extend(P)
        P_b = P * self._k_extrap_j ** 2
        if P_window is not None:
            P_b = P_b * jnp.asarray(self._p_window_fn(self.k_extrap, P_window[0], P_window[1]))
        c_m = self._fourier(self._pad(P_b), C_window, scalar=True)
        C_l = _fftconvolve(c_m[None, :] * g_m, c_m[None, :] * g_n) * h2_l
        mat = self._conv_to_k(C_l) * kfac
        P_out = jnp.fft.irfft(c_m[self.N // 2:], n=self.N) * float(self.N)
        return P_out[self.id_pad] * self._kfinal_nu_pad, mat

    def _tensor_term(self, P, name, P_window, C_window, factor=1.0):
        return factor * self._cut(self.J_k_tensor(P, name, P_window, C_window))

    # ------------------------------------------------------------ public API
    def IA_tt(self, P, P_window=None, C_window=None):
        """(P_E, P_B)."""
        return (self._tensor_term(P, "X_IA_E", P_window, C_window, 2.0),
                self._tensor_term(P, "X_IA_B", P_window, C_window, 2.0))

    def IA_ta(self, P, P_window=None, C_window=None):
        """(P_deltaE1, P_deltaE2, P_0E0E, P_0B0B)."""
        P = jnp.asarray(P, dtype=jnp.float64)
        # deltaE2 uses the raw input P on k_original (no extrapolation/windows).
        P_deltaE2 = 2.0 * self._ko3 / (896.0 * np.pi ** 2) * P * (self._T_deltaE2 @ P)
        return (self._tensor_term(P, "X_IA_deltaE1", P_window, C_window, 2.0),
                P_deltaE2,
                self._tensor_term(P, "X_IA_0E0E", P_window, C_window),
                self._tensor_term(P, "X_IA_0B0B", P_window, C_window))

    def IA_mix(self, P, P_window=None, C_window=None):
        """(P_A, P_Btype2, P_DEE, P_DBB)."""
        P = jnp.asarray(P, dtype=jnp.float64)
        P_Btype2 = 4.0 * self._ko3 / (2.0 * np.pi ** 2) * P * (self._T_B @ P)
        return (self._tensor_term(P, "X_IA_A", P_window, C_window, 2.0),
                P_Btype2,
                self._tensor_term(P, "X_IA_DEE", P_window, C_window, 2.0),
                self._tensor_term(P, "X_IA_DBB", P_window, C_window, 2.0))

    def _one_loop_ext(self, Ps, mat):
        P22 = jnp.sum(jnp.asarray(_P22_COEF)[:, None] * mat, axis=0)
        P13 = 1.0 / 252.0 * self._ke3 / (2 * np.pi) ** 2 * Ps * (self._T_13 @ Ps)
        return P22 + P13

    def one_loop_dd(self, P, P_window=None, C_window=None):
        """(P_1loop, Ps)."""
        Ps, mat = self.J_k_scalar(P, C_window, P_window)
        return self._cut(self._one_loop_ext(Ps, mat)), self._cut(Ps)

    def one_loop_dd_bias_b3nl(self, P, P_window=None, C_window=None):
        """(P_1loop, Ps, Pd1d2, Pd2d2, Pd1s2, Pd2s2, Ps2s2, sig4, sig3nl); sig4 is 0-d."""
        Ps, mat = self.J_k_scalar(P, C_window, P_window)
        P_1loop = self._one_loop_ext(Ps, mat)
        Pd1d2 = 2.0 * (17.0 / 21 * mat[0] + mat[4] + 4.0 / 21 * mat[1])
        Pd2d2 = 2.0 * mat[0]
        Pd1s2 = 2.0 * (8.0 / 315 * mat[0] + 4.0 / 15 * mat[4] + 254.0 / 441 * mat[1]
                       + 2.0 / 5 * mat[5] + 16.0 / 245 * mat[2])
        Pd2s2 = 2.0 * (2.0 / 3 * mat[1])
        Ps2s2 = 2.0 * (4.0 / 45 * mat[0] + 8.0 / 63 * mat[1] + 8.0 / 35 * mat[2])
        sig4 = jnp.trapezoid(self._ke3 * Ps ** 2, x=self._log_ke) / (2.0 * np.pi ** 2)
        sig3nl = self._ke3 / (2 * np.pi) ** 2 * Ps * (self._T_Y1 @ Ps)
        c = self._cut
        return (c(P_1loop), c(Ps), c(Pd1d2), c(Pd2d2), c(Pd1s2), c(Pd2s2),
                c(Ps2s2), sig4, c(sig3nl))


def ept_ia_templates(pt, pk_lin_z0, *, P_window=None,
                     C_window: float | None = 0.75) -> dict[str, tuple]:
    """FAST-PT templates as stored by ``EulerianPTCalculator.update_ingredients``.

    Keys: ``ia_ta``, ``ia_tt``, ``ia_mix``, ``dd_bias`` (b3nl tuple) and
    ``one_loop_dd`` (``dd_bias[0:1]``, as CCL sets it when ``with_NC``).
    """
    kw = {"P_window": P_window, "C_window": C_window}
    dd_bias = pt.one_loop_dd_bias_b3nl(pk_lin_z0, **kw)
    return {
        "ia_ta": pt.IA_ta(pk_lin_z0, **kw),
        "ia_tt": pt.IA_tt(pk_lin_z0, **kw),
        "ia_mix": pt.IA_mix(pk_lin_z0, **kw),
        "dd_bias": dd_bias,
        "one_loop_dd": dd_bias[0:1],
    }


_JAXPT_CACHE_OPTIONS = (
    "jax_compilation_cache_dir",
    "jax_persistent_cache_min_entry_size_bytes",
    "jax_persistent_cache_min_compile_time_secs",
)


def _import_jaxpt():
    """``jaxpt.JAXPT``, undoing the global JAX cache options it sets on import."""
    saved = {name: getattr(jax.config, name) for name in _JAXPT_CACHE_OPTIONS}
    from jaxpt import JAXPT

    for name, value in saved.items():
        if getattr(jax.config, name) != value:
            jax.config.update(name, value)
    return JAXPT


class JAXPTAdapter:
    """``jaxpt.JAXPT`` (JAX-PT, github.com/vschac/jax-pt) with FAST-PT signatures.

    JAX-PT 1.0 drops ``P_1loop, Ps`` from ``one_loop_dd_bias_b3nl``; they are
    prepended here so the tuple layout matches ``fastpt.FASTPT`` (and CCL's
    ``EulerianPTCalculator`` indexing).

    JAX-PT caches its ``X_*`` tables lazily on first access; they are filled
    here, eagerly, so a first call inside ``jit`` / ``grad`` cannot store
    tracers in that cache (which breaks every later transformed call).
    """

    _X_TERMS = ("X_spt",) + _TENSOR_TERMS

    def __init__(self, k, *, low_extrap=None, high_extrap=None, n_pad=None):
        JAXPT = _import_jaxpt()
        with contextlib.redirect_stdout(io.StringIO()):
            self._pt = JAXPT(np.asarray(k, dtype=float), low_extrap=low_extrap,
                             high_extrap=high_extrap, n_pad=n_pad)
            for name in self._X_TERMS:
                getattr(self._pt, name)

    def IA_tt(self, P, P_window=None, C_window=None):
        return self._pt.IA_tt(P, P_window=P_window, C_window=C_window)

    def IA_ta(self, P, P_window=None, C_window=None):
        return self._pt.IA_ta(P, P_window=P_window, C_window=C_window)

    def IA_mix(self, P, P_window=None, C_window=None):
        return self._pt.IA_mix(P, P_window=P_window, C_window=C_window)

    def one_loop_dd(self, P, P_window=None, C_window=None):
        return self._pt.one_loop_dd(P, P_window=P_window, C_window=C_window)

    def one_loop_dd_bias_b3nl(self, P, P_window=None, C_window=None):
        P_1loop, Ps = self.one_loop_dd(P, P_window=P_window, C_window=C_window)
        rest = self._pt.one_loop_dd_bias_b3nl(P, P_window=P_window, C_window=C_window)
        return (P_1loop, Ps, *rest)


FASTPT_BACKENDS = ("auto", "jaxpt", "gpucosmo")


def jaxpt_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("jaxpt") is not None


def make_fastpt(k, *, low_extrap=None, high_extrap=None, n_pad=None,
                backend: str = "auto", C_window: float | None = 0.75):
    """FAST-PT-compatible JAX calculator.

    ``backend``: ``"jaxpt"`` (JAX-PT package), ``"gpucosmo"`` (the port in this
    module) or ``"auto"``: JAX-PT when installed and the calculator will run
    with a ``C_window`` (CCL's default 0.75), else the in-repo port. With a
    ``C_window`` both reproduce FAST-PT 4.0 to <1e-6 (IA) / ~3e-6 (one-loop
    matter); without one, JAX-PT 1.0 is off by up to ~0.4% in the low-k
    ``P_deltaE1`` / ``P_A`` terms while the in-repo port stays at <1e-9.
    """
    if backend not in FASTPT_BACKENDS:
        raise ValueError(f"Unknown FAST-PT backend {backend!r}; choose from {FASTPT_BACKENDS}")
    kw = dict(low_extrap=low_extrap, high_extrap=high_extrap, n_pad=n_pad)
    if backend == "auto":
        backend = "jaxpt" if jaxpt_available() and C_window is not None else "gpucosmo"
    if backend == "jaxpt":
        return JAXPTAdapter(k, **kw)
    return FASTPTJax(k, **kw)


__all__ = [
    "FASTPTJax",
    "FASTPT_BACKENDS",
    "JAXPTAdapter",
    "ept_ia_templates",
    "jaxpt_available",
    "make_fastpt",
]
