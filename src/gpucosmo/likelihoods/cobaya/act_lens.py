# This file is part of GPUCosmo. Copyright (C) 2025-2026 Leonel Medina-Varela.
# SPDX-License-Identifier: GPL-3.0-or-later
# Derived in part from the following, translated to JAX and modified for GPUCosmo:
# - act_dr6_lenslike (https://github.com/ACTCollaboration/act_dr6_lenslike),
#   Copyright (c) 2023, Members of the Atacama Cosmology Telescope Collaboration.
#   Licence: LICENSES/act_dr6_lenslike.txt.
# See NOTICE.md.
"""Native port of the ACT DR6 (+ Planck PR4) CMB lensing Cobaya likelihood.

``ACTDR6LensLike`` serves ``act_dr6_lenslike.ACTDR6LensLike`` with the JAX likelihood of
:mod:`gpucosmo.jax.cmb.act_dr6_lens_jax`, after checking that the arrays Cobaya's instance loaded
are the ones the JAX loader reduced.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from gpucosmo.likelihoods.cobaya.base import CobayaPort

_TWO_PI = 2.0 * np.pi
_REQUIRED = object()


def _nan_to_neginf(ll):
    return jnp.where(jnp.isnan(ll), -jnp.inf, ll)


class _ACTPort(CobayaPort):
    """Parameter access shared by the ACT ports."""

    def _init_params(self):
        const = self.bridge._model.parameterization.constant_params()
        self.fixed = {p: float(v) for p, v in const.items() if isinstance(v, (int, float, np.integer, np.floating))}

    def _has(self, name) -> bool:
        return name in self.params or name in self.fixed

    def _p(self, inp, name, default=_REQUIRED):
        if name in inp.params:
            return jnp.asarray(inp.params[name], dtype=jnp.float64)
        if name in self.fixed:
            return self.fixed[name]
        if default is _REQUIRED:
            raise KeyError(f"{self.cobaya_name}: parameter {name!r} is neither sampled nor fixed")
        return default

    def _need(self, names) -> None:
        missing = [n for n in dict.fromkeys(names) if not self._has(n)]
        if missing:
            raise NotImplementedError(f"{self.cobaya_name}: parameters {missing} are neither sampled nor fixed")

    def _widen_cl(self, spectra, lmax: int) -> None:
        """Request ``spectra`` to ``lmax`` (what the package reads) on the bridge this port owns, so
        ``requirements()`` (the port's and the bridge's) and the bridge's ``loglike`` include them
        (CAMB serves multipoles past the request; the bridge passes exactly the request)."""
        cl = dict(self.bridge._req["Cl"])
        for s in spectra:
            cl[s] = max(int(cl.get(s, 0)), int(lmax))
        self.bridge._req["Cl"] = cl

    @staticmethod
    def _cl(inp, s):
        return jnp.asarray(inp.shared["cmb"]["cl"][s], dtype=jnp.float64)


class ACTDR6LensLike(_ACTPort):
    """ACT DR6 (+ Planck PR4) CMB lensing, ``act_dr6_lenslike.ACTDR6LensLike`` through ``act_dr6_lens_jax``."""

    cobaya_names = ("act_dr6_lenslike.ACTDR6LensLike", "act_dr6_lenslike.act_dr6_lenslike.ACTDR6LensLike",
                    "act_dr6_lenslike")

    _REFUSED = {"limber": False, "mock": False, "varying_cmb_alens": False, "act_cmb_rescale": False,
                "act_calib": False}

    def setup(self, like):
        from gpucosmo.jax.cmb import act_dr6_lens_jax as lj

        name = self.cobaya_name
        for k, v in self._REFUSED.items():
            if bool(getattr(like, k, v)) != v:
                raise NotImplementedError(f"{name}: {k}={getattr(like, k)!r} not ported")
        if getattr(like, "scale_cov", None) is not None:
            raise NotImplementedError(f"{name}: scale_cov not ported")
        if getattr(like, "version", None) not in (None, lj.VERSION):
            raise NotImplementedError(f"{name}: version {like.version!r} (ported {lj.VERSION})")
        if int(like.trim_lmax) != lj.TRIM_LMAX:
            raise NotImplementedError(f"{name}: trim_lmax {like.trim_lmax} (ported {lj.TRIM_LMAX})")
        variant = str(like.variant).lower().strip()
        if variant not in lj.VARIANTS:
            raise NotImplementedError(f"{name}: variant {like.variant!r} (ported {lj.VARIANTS})")
        self.lens = lj.load_act_dr6_lens(variant, bool(like.lens_only), nsims_act=float(like.nsims_act),
                                         nsims_planck=float(like.nsims_planck), apply_hartlap=bool(like.apply_hartlap),
                                         no_like_corrections=bool(like.no_like_corrections),
                                         no_actlike_cmb_corrections=bool(like.no_actlike_cmb_corrections),
                                         verbose=False)
        d = like.data  # the arrays Cobaya's instance loaded must be the ones the loader reduced
        ok = (self.lens.data.shape == np.shape(d["data_binned_clkk"])
              and np.allclose(self.lens.data, d["data_binned_clkk"], rtol=1e-12, atol=0.0)
              and np.allclose(self.lens.cinv, d["cinv"], rtol=1e-9, atol=1e-12 * np.max(np.abs(d["cinv"]))))
        if not ok or self.lens.needs_cmb != bool(d["likelihood_corrections"]):
            raise ValueError(f"{name}: the repository's act_dr6_lens data differ from the Cobaya package's")
        self.n_ell = lj.N_ELL
        self._ll = lj.build_act_dr6_lens_loglike(self.lens)
        # ``generic_lnlike`` standardizes TT/EE/TE/BB even when it requests only φφ (lens_only), all
        # to L = trim_lmax + 1: CAMB serves them, the bridge passes what is requested — widen
        self._widen_cl(("tt", "te", "ee", "bb", "pp"), max(int(like.lmax), lj.N_ELL - 1))

    def loglike(self, inp):
        n = self.n_ell
        cl = inp.shared["cmb"]["cl"]
        ell = jnp.arange(n, dtype=jnp.float64)
        # get_Cl(ell_factor=False) then pp_to_kk: C^κκ = [ℓ(ℓ+1)]² C^φφ / 4 (0 at ℓ = 0)
        kk = jnp.where(ell >= 1, jnp.asarray(cl["pp"], dtype=jnp.float64)[:n] * (_TWO_PI / 4.0), 0.0)
        dl = None
        if self.lens.needs_cmb:
            dl = jnp.stack([self._cl(inp, s)[:n] for s in ("tt", "ee", "bb", "te")])
        return _nan_to_neginf(self._ll(kk, dl))


__all__ = ["ACTDR6LensLike"]
