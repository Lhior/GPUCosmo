#!/usr/bin/env python3
# This file is part of GPUCosmo. Copyright (C) 2025-2026 Leonel Medina-Varela.
# SPDX-License-Identifier: GPL-3.0-or-later
# Derived in part from the following, translated to JAX and modified for GPUCosmo:
# - CCL (https://github.com/LSSTDESC/CCL), Copyright (c) 2016-2018, the LSSTDESC CCL
#   contributors.
#   Licence: LICENSES/CCL.txt.
# See NOTICE.md.
"""Perturbation-theory tracers and Eulerian PT (TATT) in JAX (``pyccl.nl_pt``).

Mirrors ``pyccl/nl_pt/{tracers,ept}.py``: the IA expansion
``s^I_ij = c1 s_ij + c2 (s_ik s_kj - s^2 delta_ij / 3) + cdelta delta s_ij``
(tidal alignment + tidal torquing, Blazek et al. 2019) and the galaxy bias
expansion ``b1 delta + b2/2 delta^2 + bs/2 s^2 + b3nl/2 psi + bk2/2 nabla^2 delta``.
FAST-PT templates come from JAX-PT (``pip install jax-pt``) when installed,
else from the in-repo port :mod:`gpucosmo.jax.fastpt_jax` (``fastpt_backend``
selects), so the resulting ``Pk2D`` objects are differentiable with respect
to cosmology.

Typical use (identical to pyccl)::

    c1, cd, c2 = translate_IA_norm(cosmo, z=z, a1=A1, a1delta=A1d, a2=A2)
    ptt_i = PTIntrinsicAlignmentTracer(c1=(z, c1), c2=(z, c2), cdelta=(z, cd))
    ptt_g = PTNumberCountsTracer(b1=b)
    ept = EulerianPTCalculator(with_NC=True, with_IA=True, cosmo=cosmo)
    pk_gi = ept.get_biased_pk2d(ptt_g, tracer2=ptt_i)
    tr_i = WeakLensingTracer(cosmo, dndz=nz, has_shear=False,
                             ia_bias=(z, np.ones_like(z)), use_A_ia=False)
    cl = angular_cl(cosmo, tr_g, tr_i, ell, p_of_k_a=pk_gi)
"""

from __future__ import annotations

import copy
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from gpucosmo.jax.ccl.cosmology import IA_C1, RHO_CRITICAL, Cosmology
from gpucosmo.jax.ccl.pk2d import Pk2D

jax.config.update("jax_enable_x64", True)


def translate_IA_norm(
    cosmo: Cosmology,
    *,
    z,
    a1=1.0,
    a1delta=None,
    a2=None,
    Om_m2_for_c2: bool = False,
    Om_m_fid: float = 0.3,
):
    """A_ia amplitudes -> c_ia biases (Blazek et al. 2019 / DES conventions)."""
    om_m = cosmo["Omega_m"]
    z = jnp.asarray(z, dtype=float)
    gz = cosmo.growth_factor(1.0 / (1.0 + z))
    c1 = c1delta = c2 = None
    if a1 is not None:
        c1 = -1.0 * a1 * IA_C1 * RHO_CRITICAL * om_m / gz
    if a1delta is not None:
        c1delta = -1.0 * a1delta * IA_C1 * RHO_CRITICAL * om_m / gz
    if a2 is not None:
        if Om_m2_for_c2:  # Blazek et al. 2019
            c2 = a2 * 5.0 * IA_C1 * RHO_CRITICAL * om_m**2 / (Om_m_fid * gz**2)
        else:  # DES convention
            c2 = a2 * 5.0 * IA_C1 * RHO_CRITICAL * om_m / (gz**2)
    return c1, c1delta, c2


def _bias_function(b):
    """Constant, zero, or ``interp1d(z, b, fill_value=b[-1])`` like pyccl."""
    if b is None:
        b = 0.0
    if not isinstance(b, (tuple, list)):
        if jnp.ndim(b) != 0:
            raise ValueError("bias must be a number or a tuple (z, b(z))")
        return lambda z: b * jnp.ones_like(jnp.asarray(z, dtype=float))
    if len(b) != 2:
        raise ValueError("bias must be a number or a tuple (z, b(z))")
    zb = jnp.asarray(b[0], dtype=float)
    bb = jnp.asarray(b[1], dtype=float)
    return lambda z: jnp.interp(jnp.asarray(z, dtype=float), zb, bb, left=bb[-1], right=bb[-1])


class PTTracer:
    type: str | None = None

    def __init__(self):
        self.biases: dict = {}

    def get_bias(self, bias_name: str, z):
        if bias_name not in self.biases:
            raise KeyError(f"Bias {bias_name} not included in this tracer")
        return self.biases[bias_name](z)


class PTMatterTracer(PTTracer):
    type = "M"


class PTNumberCountsTracer(PTTracer):
    type = "NC"

    def __init__(self, b1, b2=None, bs=None, b3nl=None, bk2=None):
        self.biases = {
            "b1": _bias_function(b1),
            "b2": _bias_function(b2),
            "bs": _bias_function(bs),
            "b3nl": _bias_function(b3nl),
            "bk2": _bias_function(bk2),
        }
        self._has_nonlinear = any(v is not None for v in (b2, bs, b3nl, bk2))

    b1 = property(lambda self: self.biases["b1"])
    b2 = property(lambda self: self.biases["b2"])
    bs = property(lambda self: self.biases["bs"])
    b3nl = property(lambda self: self.biases["b3nl"])
    bk2 = property(lambda self: self.biases["bk2"])


class PTIntrinsicAlignmentTracer(PTTracer):
    type = "IA"

    def __init__(self, c1, c2=None, cdelta=None):
        self.biases = {
            "c1": _bias_function(c1),
            "c2": _bias_function(c2),
            "cdelta": _bias_function(cdelta),
        }

    c1 = property(lambda self: self.biases["c1"])
    c2 = property(lambda self: self.biases["c2"])
    cdelta = property(lambda self: self.biases["cdelta"])


def _default_pk_a() -> np.ndarray:
    import pyccl

    return np.asarray(pyccl.get_pk_spline_a(), dtype=float)


class EulerianPTCalculator:
    """JAX ``pyccl.nl_pt.EulerianPTCalculator`` (FAST-PT based).

    Construct once (FAST-PT tables are built in numpy); call
    :meth:`update_ingredients` (or :meth:`bind`, which returns a copy) with a
    JAX :class:`Cosmology` inside or outside ``jax.jit``.
    """

    def __init__(
        self,
        *,
        with_NC: bool = False,
        with_IA: bool = False,
        with_matter_1loop: bool = True,
        cosmo: Cosmology | None = None,
        log10k_min: float = -4,
        log10k_max: float = 2,
        nk_per_decade: float = 20,
        a_arr=None,
        k_cutoff: float | None = None,
        n_exp_cutoff: float = 4,
        b1_pk_kind: str = "nonlinear",
        bk2_pk_kind: str = "nonlinear",
        pad_factor: float = 1.0,
        low_extrap: float = -5.0,
        high_extrap: float = 3.0,
        P_window=None,
        C_window: float | None = 0.75,
        sub_lowk: bool = False,
        fastpt_backend: str = "auto",
    ):
        from gpucosmo.jax.fastpt_jax import make_fastpt

        self.with_matter_1loop = with_matter_1loop
        self.with_NC = with_NC
        self.with_IA = with_IA
        self.fastpt_par = {
            "pad_factor": pad_factor,
            "low_extrap": low_extrap,
            "high_extrap": high_extrap,
            "P_window": P_window,
            "C_window": C_window,
            "sub_lowk": sub_lowk,
        }
        nk_total = int((log10k_max - log10k_min) * nk_per_decade)
        self.k_s = np.logspace(log10k_min, log10k_max, nk_total)
        self.a_s = np.asarray(_default_pk_a() if a_arr is None else a_arr, dtype=float).copy()
        self.z_s = 1.0 / self.a_s - 1.0
        if k_cutoff is not None:
            self.exp_cutoff = jnp.asarray(np.exp(-(self.k_s / k_cutoff) ** n_exp_cutoff))[None, :]
        else:
            self.exp_cutoff = 1.0
        n_pad = int(pad_factor * len(self.k_s))
        self.pt = make_fastpt(
            self.k_s, low_extrap=low_extrap, high_extrap=high_extrap, n_pad=n_pad,
            backend=fastpt_backend, C_window=C_window,
        )
        for kind in (b1_pk_kind, bk2_pk_kind):
            if kind not in ("linear", "nonlinear", "pt"):
                raise ValueError(f"Unknown P(k) prescription {kind}")
        self.b1_pk_kind = b1_pk_kind
        self.bk2_pk_kind = bk2_pk_kind
        if "pt" in (b1_pk_kind, bk2_pk_kind):
            self.with_matter_1loop = True
        self._pk2d_temp: dict = {}
        if cosmo is not None:
            self.update_ingredients(cosmo)

    @property
    def initialised(self) -> bool:
        return hasattr(self, "pk_b1")

    def _check_init(self):
        if not self.initialised:
            raise RuntimeError(
                "PT templates have not been initialised for this calculator. "
                "Please do so using `update_ingredients`."
            )

    def bind(self, cosmo: Cosmology) -> "EulerianPTCalculator":
        """Copy of this calculator with ingredients for ``cosmo`` (jit-friendly)."""
        out = copy.copy(self)
        out.update_ingredients(cosmo)
        return out

    def update_ingredients(self, cosmo: Cosmology) -> None:
        k_s = jnp.asarray(self.k_s)
        a_s = jnp.asarray(self.a_s)
        pklz0 = cosmo.linear_matter_power(k_s, 1.0)
        g = cosmo.growth_factor(a_s)
        self._g4 = g**4
        self._g4T = self._g4[:, None]
        kw = {"P_window": self.fastpt_par["P_window"], "C_window": self.fastpt_par["C_window"]}
        if self.with_NC:
            self.dd_bias = self.pt.one_loop_dd_bias_b3nl(pklz0, **kw)
            self.one_loop_dd = self.dd_bias[0:1]
            self.with_matter_1loop = True
        elif self.with_matter_1loop:
            self.one_loop_dd = self.pt.one_loop_dd(pklz0, **kw)
        if self.with_IA:
            self.ia_ta = self.pt.IA_ta(pklz0, **kw)
            self.ia_tt = self.pt.IA_tt(pklz0, **kw)
            self.ia_mix = self.pt.IA_mix(pklz0, **kw)

        def grid(fn):
            return jax.vmap(lambda a: fn(k_s, a))(a_s)

        pks = {}
        kinds = (self.b1_pk_kind, self.bk2_pk_kind)
        if "nonlinear" in kinds:
            pks["nonlinear"] = grid(cosmo.nonlin_matter_power)
        if "linear" in kinds or "pt" in kinds:
            pks["linear"] = grid(cosmo.linear_matter_power)
        if "pt" in kinds:
            pks["pt"] = pks["linear"] + self._g4T * self.one_loop_dd[0]
        self.pk_b1 = pks[self.b1_pk_kind]
        self.pk_bk = pks[self.bk2_pk_kind]
        self._pk2d_temp = {}
        self._cosmo = cosmo

    # -- spectra (ept.py) -------------------------------------------------
    def _get_pgg(self, tr1, tr2):
        self._check_init()
        k2 = jnp.asarray(self.k_s**2)[None, :]
        Pd1d1 = self.pk_b1
        Pd1d2 = self._g4T * self.dd_bias[2]
        Pd2d2 = self._g4T * self.dd_bias[3]
        Pd1s2 = self._g4T * self.dd_bias[4]
        Pd2s2 = self._g4T * self.dd_bias[5]
        Ps2s2 = self._g4T * self.dd_bias[6]
        Pd1p3 = self._g4T * self.dd_bias[8]
        Pd1k2 = self.pk_bk * k2
        z = self.z_s
        b11, b21, bs1, bk21, b3nl1 = (tr1.b1(z), tr1.b2(z), tr1.bs(z), tr1.bk2(z), tr1.b3nl(z))
        b12, b22, bs2, bk22, b3nl2 = (tr2.b1(z), tr2.b2(z), tr2.bs(z), tr2.bk2(z), tr2.b3nl(z))
        s4 = self._g4T * self.dd_bias[7] if self.fastpt_par["sub_lowk"] else 0.0
        pgg = (
            (b11 * b12)[:, None] * Pd1d1
            + 0.5 * (b11 * b22 + b12 * b21)[:, None] * Pd1d2
            + 0.25 * (b21 * b22)[:, None] * (Pd2d2 - 2.0 * s4)
            + 0.5 * (b11 * bs2 + b12 * bs1)[:, None] * Pd1s2
            + 0.25 * (b21 * bs2 + b22 * bs1)[:, None] * (Pd2s2 - (4.0 / 3.0) * s4)
            + 0.25 * (bs1 * bs2)[:, None] * (Ps2s2 - (8.0 / 9.0) * s4)
            + 0.5 * (b12 * b3nl1 + b11 * b3nl2)[:, None] * Pd1p3
            + 0.5 * (b12 * bk21 + b11 * bk22)[:, None] * Pd1k2
        )
        return pgg * self.exp_cutoff

    def _get_pgi(self, trg, tri):
        self._check_init()
        a00e, c00e, _a0e0e, _a0b0b = self.ia_ta
        a0e2, b0e2, _d0ee2, _d0bb2 = self.ia_mix
        z = self.z_s
        if getattr(trg, "_has_nonlinear", False):
            warnings.warn(
                "EulerianPTCalculators assume linear galaxy bias "
                "when computing galaxy-IA cross-correlations."
            )
        b1 = trg.b1(z)
        c1, c2, cd = tri.c1(z), tri.c2(z), tri.cdelta(z)
        pgi = b1[:, None] * (
            c1[:, None] * self.pk_b1
            + (self._g4 * cd)[:, None] * (a00e + c00e)
            + (self._g4 * c2)[:, None] * (a0e2 + b0e2)
        )
        return pgi * self.exp_cutoff

    def _get_pgm(self, trg):
        self._check_init()
        z = self.z_s
        k2 = jnp.asarray(self.k_s**2)[None, :]
        pgm = (
            trg.b1(z)[:, None] * self.pk_b1
            + 0.5 * trg.b2(z)[:, None] * self._g4T * self.dd_bias[2]
            + 0.5 * trg.bs(z)[:, None] * self._g4T * self.dd_bias[4]
            + 0.5 * trg.b3nl(z)[:, None] * self._g4T * self.dd_bias[8]
            + 0.5 * trg.bk2(z)[:, None] * self.pk_bk * k2
        )
        return pgm * self.exp_cutoff

    def _get_pii(self, tr1, tr2, return_bb: bool = False):
        self._check_init()
        a00e, c00e, a0e0e, a0b0b = self.ia_ta
        ae2e2, ab2b2 = self.ia_tt
        a0e2, b0e2, d0ee2, d0bb2 = self.ia_mix
        z = self.z_s
        c11, c21, cd1 = tr1.c1(z), tr1.c2(z), tr1.cdelta(z)
        c12, c22, cd2 = tr2.c1(z), tr2.c2(z), tr2.cdelta(z)
        g4 = self._g4
        if return_bb:
            pii = (
                (cd1 * cd2 * g4)[:, None] * a0b0b
                + (c21 * c22 * g4)[:, None] * ab2b2
                + ((cd1 * c22 + c21 * cd2) * g4)[:, None] * d0bb2
            )
        else:
            pii = (
                (c11 * c12)[:, None] * self.pk_b1
                + ((c11 * cd2 + c12 * cd1) * g4)[:, None] * (a00e + c00e)
                + (cd1 * cd2 * g4)[:, None] * a0e0e
                + (c21 * c22 * g4)[:, None] * ae2e2
                + ((c11 * c22 + c21 * c12) * g4)[:, None] * (a0e2 + b0e2)
                + ((cd1 * c22 + cd2 * c21) * g4)[:, None] * d0ee2
            )
        return pii * self.exp_cutoff

    def _get_pim(self, tri):
        self._check_init()
        a00e, c00e, _a0e0e, _a0b0b = self.ia_ta
        a0e2, b0e2, _d0ee2, _d0bb2 = self.ia_mix
        z = self.z_s
        c1, c2, cd = tri.c1(z), tri.c2(z), tri.cdelta(z)
        pim = (
            c1[:, None] * self.pk_b1
            + (self._g4 * cd)[:, None] * (a00e + c00e)
            + (self._g4 * c2)[:, None] * (a0e2 + b0e2)
        )
        return pim * self.exp_cutoff

    def _get_pmm(self):
        self._check_init()
        p1loop = self._g4T * self.one_loop_dd[0] if self.b1_pk_kind == "linear" else 0.0
        return (self.pk_b1 + p1loop) * self.exp_cutoff

    def _to_pk2d(self, pk, extrap_order_lok: int, extrap_order_hik: int) -> Pk2D:
        return Pk2D(
            a_arr=self.a_s,
            lk_arr=np.log(self.k_s),
            pk_arr=pk,
            is_logp=False,
            extrap_order_lok=extrap_order_lok,
            extrap_order_hik=extrap_order_hik,
        )

    def get_biased_pk2d(
        self,
        tracer1: PTTracer,
        *,
        tracer2: PTTracer | None = None,
        return_ia_bb: bool = False,
        extrap_order_lok: int = 1,
        extrap_order_hik: int = 2,
    ) -> Pk2D:
        if tracer2 is None:
            tracer2 = tracer1
        t1, t2 = tracer1.type, tracer2.type
        if "NC" in (t1, t2) and not self.with_NC:
            raise ValueError(
                "Can't use number counts tracer in EulerianPTCalculator with 'with_NC=False'"
            )
        if "IA" in (t1, t2) and not self.with_IA:
            raise ValueError(
                "Can't use intrinsic alignment tracer in EulerianPTCalculator with 'with_IA=False'"
            )
        if t1 == "NC":
            if t2 == "NC":
                pk = self._get_pgg(tracer1, tracer2)
            elif t2 == "IA":
                pk = self._get_pgi(tracer1, tracer2)
            else:
                pk = self._get_pgm(tracer1)
        elif t1 == "IA":
            if t2 == "NC":
                pk = self._get_pgi(tracer2, tracer1)
            elif t2 == "IA":
                pk = self._get_pii(tracer1, tracer2, return_bb=return_ia_bb)
            else:
                pk = self._get_pim(tracer1)
        else:
            if t2 == "NC":
                pk = self._get_pgm(tracer2)
            elif t2 == "IA":
                pk = self._get_pim(tracer2)
            else:
                pk = self._get_pmm()
        return self._to_pk2d(pk, extrap_order_lok, extrap_order_hik)


__all__ = [
    "EulerianPTCalculator",
    "PTIntrinsicAlignmentTracer",
    "PTMatterTracer",
    "PTNumberCountsTracer",
    "PTTracer",
    "translate_IA_norm",
]
