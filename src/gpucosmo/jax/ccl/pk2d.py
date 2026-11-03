#!/usr/bin/env python3
# This file is part of GPUCosmo. Copyright (C) 2025-2026 Leonel Medina-Varela.
# SPDX-License-Identifier: GPL-3.0-or-later
# Derived in part from the following, translated to JAX and modified for GPUCosmo:
# - CCL (https://github.com/LSSTDESC/CCL), Copyright (c) 2016-2018, the LSSTDESC CCL
#   contributors.
#   Licence: LICENSES/CCL.txt.
# See NOTICE.md.
"""JAX ``Pk2D`` and generic 2D (k, a) functions matching ``ccl_f2d_t``."""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from gpucosmo.jax.gsl_splines import (
    Bicubic2D,
    Spline1D,
    bicubic_deriv_x,
    bicubic_deriv_xx,
    bicubic_eval,
    bicubic_spline,
    cspline_spline,
    spline_deriv,
    spline_deriv2,
    spline_eval,
)

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class F2D:
    """``ccl_f2d_t``: either a bicubic (lk, a) table or factorizable fk(lk)*fa(a).

    ``growth_exponent`` controls a-extrapolation below ``a_min``: the value at
    ``a_min`` is multiplied by ``(D(a)/D(a_min))**growth_exponent`` when a
    cosmology is supplied (Pk2D uses 2; tracer transfers use 0 = constant).
    """

    fka: Bicubic2D | None = None
    fk: Spline1D | None = None
    fa: Spline1D | None = None
    is_log: bool = field(default=False, metadata={"static": True})
    extrap_order_lok: int = field(default=1, metadata={"static": True})
    extrap_order_hik: int = field(default=2, metadata={"static": True})
    growth_exponent: int = field(default=0, metadata={"static": True})

    @property
    def is_factorizable(self) -> bool:
        return self.fka is None

    @property
    def is_k_constant(self) -> bool:
        return self.fka is None and self.fk is None

    @property
    def is_a_constant(self) -> bool:
        return self.fka is None and self.fa is None

    def _lk_range(self):
        grid = self.fka.x if self.fka is not None else self.fk.x
        return grid[0], grid[-1]

    def _a_range(self):
        grid = self.fka.y if self.fka is not None else self.fa.x
        return grid[0], grid[-1]

    def eval(self, lk, a, cosmo=None):
        """Evaluate at (ln k, a); broadcasts ``lk`` against ``a``."""
        lk = jnp.asarray(lk, dtype=float)
        a = jnp.asarray(a, dtype=float)
        lk, a = jnp.broadcast_arrays(lk, a)
        unit = 0.0 if self.is_log else 1.0

        if self.is_a_constant:
            a_ev = a
            is_hiz = jnp.zeros(a.shape, dtype=bool)
        else:
            amin, amax = self._a_range()
            a_ev = jnp.clip(a, amin, amax)
            is_hiz = a < amin

        if self.is_k_constant:
            lk_ev = lk
            is_hik = jnp.zeros(lk.shape, dtype=bool)
            is_lok = jnp.zeros(lk.shape, dtype=bool)
        else:
            lkmin, lkmax = self._lk_range()
            lk_ev = jnp.clip(lk, lkmin, lkmax)
            is_hik = lk > lkmax
            is_lok = lk < lkmin

        if self.is_factorizable:
            fk = unit if self.fk is None else spline_eval(self.fk, lk_ev)
            fa = unit if self.fa is None else spline_eval(self.fa, a_ev)
            pre = fk + fa if self.is_log else fk * fa
        else:
            pre = bicubic_eval(self.fka, lk_ev, a_ev)

        post = pre
        if not self.is_k_constant:
            dlk = lk - lk_ev

            def d1(x):
                if self.is_factorizable:
                    return spline_deriv(self.fk, x)
                return bicubic_deriv_x(self.fka, x, a_ev)

            def d2(x):
                if self.is_factorizable:
                    return spline_deriv2(self.fk, x)
                return bicubic_deriv_xx(self.fka, x, a_ev)

            def extrap(order):
                out = pre
                if order > 0:
                    out = out + d1(lk_ev) * dlk
                if order > 1:
                    out = out + 0.5 * d2(lk_ev) * dlk * dlk
                return out

            post = jnp.where(
                is_hik,
                extrap(self.extrap_order_hik),
                jnp.where(is_lok, extrap(self.extrap_order_lok), pre),
            )
        if self.is_log:
            post = jnp.exp(post)
        if self.growth_exponent != 0 and cosmo is not None and not self.is_a_constant:
            amin, _ = self._a_range()
            gz = cosmo.growth_factor(a) / cosmo.growth_factor(amin)
            post = jnp.where(is_hiz, post * gz**self.growth_exponent, post)
        return post


jax.tree_util.register_dataclass(
    F2D,
    data_fields=["fka", "fk", "fa"],
    meta_fields=["is_log", "extrap_order_lok", "extrap_order_hik", "growth_exponent"],
)


class Pk2D:
    """CCL-compatible power spectrum P(k, a) (``pyccl.Pk2D`` array constructor).

    ``pk_arr`` has shape ``(na, nk)``; ``a_arr`` and ``lk_arr`` increasing.
    """

    def __init__(
        self,
        *,
        a_arr,
        lk_arr,
        pk_arr,
        is_logp: bool = True,
        extrap_order_lok: int = 1,
        extrap_order_hik: int = 2,
        _f2d: F2D | None = None,
    ):
        if _f2d is not None:
            self.psp = _f2d
            return
        a_arr = jnp.asarray(a_arr, dtype=float)
        lk_arr = jnp.asarray(lk_arr, dtype=float)
        pk_arr = jnp.asarray(pk_arr, dtype=float)
        if pk_arr.shape != (a_arr.shape[0], lk_arr.shape[0]):
            raise ValueError("pk_arr must have shape (len(a_arr), len(lk_arr))")
        z = jnp.log(pk_arr) if is_logp else pk_arr
        self.psp = F2D(
            fka=bicubic_spline(lk_arr, a_arr, z.T),
            is_log=bool(is_logp),
            extrap_order_lok=int(extrap_order_lok),
            extrap_order_hik=int(extrap_order_hik),
            growth_exponent=2,
        )

    @property
    def extrap_order_lok(self) -> int:
        return self.psp.extrap_order_lok

    @property
    def extrap_order_hik(self) -> int:
        return self.psp.extrap_order_hik

    def __call__(self, k, a, cosmo=None):
        """P(k, a). ``k`` and ``a`` broadcast against each other."""
        lk = jnp.log(jnp.asarray(k, dtype=float))
        return self.psp.eval(lk, a, cosmo)

    def apply_halofit(self, cosmo, *, nodes_per_interval: int | None = None) -> "Pk2D":
        """``pyccl.Pk2D.apply_halofit``: CCL-exact Takahashi Halofit (JAX)."""
        from gpucosmo.jax.ccl.halofit import halofit_pk2d

        return halofit_pk2d(cosmo, self, nodes_per_interval=nodes_per_interval)

    def eval_lk(self, lk, a, cosmo=None):
        return self.psp.eval(lk, a, cosmo)

    def get_spline_arrays(self):
        """(a_arr, lk_arr, pk_arr) at the spline nodes, like pyccl."""
        s = self.psp.fka
        pk = jnp.exp(s.z.T) if self.psp.is_log else s.z.T
        return s.y, s.x, pk

    @classmethod
    def from_ccl(cls, pk2d_ccl) -> "Pk2D":
        """Copy the spline nodes of a ``pyccl.Pk2D`` (e.g. a halo-model IA Pk)."""
        a_arr, lk_arr, pk_arr = pk2d_ccl.get_spline_arrays()
        is_log = bool(pk2d_ccl.psp.is_log)
        pk_arr = np.asarray(pk_arr, dtype=float)
        return cls(
            a_arr=np.asarray(a_arr, dtype=float),
            lk_arr=np.asarray(lk_arr, dtype=float),
            pk_arr=pk_arr,
            is_logp=is_log,
            extrap_order_lok=int(pk2d_ccl.extrap_order_lok),
            extrap_order_hik=int(pk2d_ccl.extrap_order_hik),
        )

    @classmethod
    def from_z_table(
        cls,
        log_k,
        z,
        log_pk,
        *,
        extrap_order_lok: int = 1,
        extrap_order_hik: int = 2,
    ) -> "Pk2D":
        """Build from a GPUCosmo ``(n_z, n_k)`` log P table with ascending z."""
        z = jnp.asarray(z, dtype=float)
        a_arr = (1.0 / (1.0 + z))[::-1]
        log_pk = jnp.asarray(log_pk, dtype=float)[::-1, :]
        f2d = F2D(
            fka=bicubic_spline(jnp.asarray(log_k, dtype=float), a_arr, log_pk.T),
            is_log=True,
            extrap_order_lok=extrap_order_lok,
            extrap_order_hik=extrap_order_hik,
            growth_exponent=2,
        )
        return cls(a_arr=None, lk_arr=None, pk_arr=None, _f2d=f2d)


jax.tree_util.register_pytree_node(
    Pk2D,
    lambda p: ((p.psp,), None),
    lambda _aux, children: Pk2D(a_arr=None, lk_arr=None, pk_arr=None, _f2d=children[0]),
)


def transfer_f2d(
    *,
    transfer_ka=None,
    transfer_k=None,
    transfer_a=None,
    is_logt: bool = False,
    extrap_order_lok: int = 0,
    extrap_order_hik: int = 2,
) -> F2D | None:
    """Tracer transfer function exactly as ``Tracer.add_tracer`` builds it."""
    if transfer_ka is None and transfer_k is None and transfer_a is None:
        return None
    if transfer_ka is not None:
        a_s, lk_s, t_ka = transfer_ka
        return F2D(
            fka=bicubic_spline(
                jnp.asarray(lk_s, dtype=float),
                jnp.asarray(a_s, dtype=float),
                jnp.asarray(t_ka, dtype=float).T,
            ),
            is_log=bool(is_logt),
            extrap_order_lok=int(extrap_order_lok),
            extrap_order_hik=int(extrap_order_hik),
            growth_exponent=0,
        )
    fk = None if transfer_k is None else cspline_spline(*transfer_k)
    fa = None if transfer_a is None else cspline_spline(*transfer_a)
    return F2D(
        fk=fk,
        fa=fa,
        is_log=bool(is_logt),
        extrap_order_lok=int(extrap_order_lok),
        extrap_order_hik=int(extrap_order_hik),
        growth_exponent=0,
    )


__all__ = ["F2D", "Pk2D", "transfer_f2d"]
