# This file is part of GPUCosmo. Copyright (C) 2025-2026 Leonel Medina-Varela.
# SPDX-License-Identifier: GPL-3.0-or-later
# Derived in part from the following, translated to JAX and modified for GPUCosmo:
# - Cobaya (https://github.com/CobayaSampler/cobaya), Copyright 2017-2026 Jesus Torrado and
#   Antony Lewis.
#   Licence: LICENSES/Cobaya.txt.
# See NOTICE.md.
"""Native port of Cobaya's supernova likelihoods (``sn.*``).

Ports ``cobaya.likelihoods.base_classes.SN`` (``alpha_beta_logp`` / ``logp``) and the data sets
of ``cobaya/likelihoods/sn``:

* Pantheon — Gaussian in the distance moduli with the offset (M_b) marginalized analytically
  (or sampled, ``use_abs_mag``);
* JLA — α, β sampled: covariance C(α, β) assembled and Cholesky-solved per point, two offsets
  (host-mass step, ``twoscriptmfit``) marginalized analytically;
* JLA_lite — the same, marginalized over Cobaya's (α, β) grid with its precomputed inverses
  (not registered, see ``SN_SETS``; ``SN("sn.jla_lite")`` builds it);
* the ``PantheonPlus`` family (Pantheon+, Pantheon+SH0ES with the Cepheid host distances of
  the calibrators as theory, Union3, DES-Y5, DES-Dovekie) — Gaussian with Cobaya's inverse
  covariance (M_b projected out unless ``use_abs_mag``).

Data, masks and covariances are those Cobaya loaded (``setup`` reads them off the instance).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from gpucosmo.likelihoods.cobaya.base import CobayaPort

_TWOPI = 2 * np.pi

# module → class of the Cobaya SN data sets served natively ("sn.<module>", "sn.<module>.<class>").
# jla_lite (JLA_lite) is ported but left to the bridge: its 149 precomputed 740² inverse
# covariances (0.65 GB) become a constant of every compiled executable (~3x that while compiling;
# 5 GB peak in ``testing.compare`` on the CPU), against 0.65 GB once in Cobaya.
SN_SETS = {"pantheon": "Pantheon", "pantheonplus": "PantheonPlus", "pantheonplusshoes": "PantheonPlusShoes",
           "union3": "Union3", "desy5": "DESy5", "desdovekie": "DESDovekie", "jla": "JLA"}


def _offsets_chi2(A, B, C, D, E, F, use_abs_mag):
    """χ² with the two offsets (``twoscriptmfit``) marginalized, as Cobaya writes it."""
    G = F - D * D / E
    chi2 = A + jnp.log(E / _TWOPI) + jnp.log(G / _TWOPI)
    if use_abs_mag:
        return chi2
    return chi2 - C * C / G - B * B * F / (E * G) + 2.0 * B * C * D / (E * G)


class SN(CobayaPort):
    """Cobaya ``base_classes.SN`` and its ``PantheonPlus`` subclasses (see the module docstring)."""

    cobaya_names = tuple(f"sn.{m}" for m in SN_SETS) + tuple(f"sn.{m}.{c}" for m, c in SN_SETS.items())

    def setup(self, like):
        from cobaya.likelihoods.base_classes.sn import SN as CobayaSN
        from cobaya.likelihoods.sn.pantheonplus import PantheonPlus
        from cobaya.likelihoods.sn.pantheonplusshoes import PantheonPlusShoes

        cls = type(like)
        if cls.logp is not CobayaSN.logp or cls.alpha_beta_logp not in (
                CobayaSN.alpha_beta_logp, PantheonPlus.alpha_beta_logp, PantheonPlusShoes.alpha_beta_logp):
            raise NotImplementedError(f"{self.cobaya_name}: {cls.__name__} overrides the SN likelihood")
        self._fixed = dict(self.bridge._model.parameterization.constant_params())
        self.use_abs_mag = bool(like.use_abs_mag)
        if self.use_abs_mag:
            self._need("Mb")
        self.idx = self.z_index("angular_diameter_distance", like.zcmb)
        self.zfac = (1 + np.asarray(like.zhel, dtype=float)) * (1 + np.asarray(like.zcmb, dtype=float))
        self.mag = np.array(like.mag, dtype=float)
        if isinstance(like, PantheonPlus):
            self.mode = "plus"
            self.invcov = np.array(like.invcov, dtype=float)
            self.calib = np.array(like.is_calibrator, dtype=bool) if isinstance(like, PantheonPlusShoes) else None
            if self.calib is not None:
                self.ceph = np.array(like.ceph_dist, dtype=float) - 25.0
            return
        self.mode = "marg" if like.marginalize else "sampled" if like.alphabeta_covmat else "fixed"
        self.alphabeta = bool(like.alphabeta_covmat)
        if self.mode == "marg" and not self.alphabeta:
            raise NotImplementedError(f"{self.cobaya_name}: α–β marginalization without α–β covariances")
        self.twoscript = bool(like.twoscriptmfit)
        if self.twoscript:
            self.A1, self.A2 = np.array(like.A1, dtype=float), np.array(like.A2, dtype=float)
        pre = np.asarray(like.pre_vars, dtype=float)
        if self.alphabeta:
            self.pre_vars = pre
            for k in ("stretch", "colour", "stretch_var", "colour_var", "cov_mag_stretch", "cov_mag_colour",
                      "cov_stretch_colour"):
                setattr(self, k, np.array(getattr(like, k), dtype=float))
        else:
            self.w = 1.0 / pre
            self.wtval = np.sum(self.w)
        if self.mode == "fixed":
            self.invcov = np.array(like.invcov, dtype=float)
            self.consts = self._consts(self.invcov)
        elif self.mode == "sampled":
            self.alpha_name, self.beta_name = like.alpha_name, like.beta_name
            self._need(self.alpha_name)
            self._need(self.beta_name)
            self.covs = {k: np.array(v, dtype=float) for k, v in like.covs.items()}
        else:  # α–β grid: Cobaya's inverse covariances
            self.alpha_grid = np.array(like.alpha_grid, dtype=float)
            self.beta_grid = np.array(like.beta_grid, dtype=float)
            self.steps = (float(like.step_width_alpha), float(like.step_width_beta))
            # ~0.65 GB for the default grid: one device array filled in place (no transient
            # copies; tracing does not copy it again), Cobaya's copies become views of it
            # (zero-copy on the CPU)
            n = len(self.mag)
            fill = jax.jit(lambda out, i, m: out.at[i].set(m), donate_argnums=0)
            invcovs, consts = jnp.zeros((like.int_points, n, n)), []
            for i, (a, b) in enumerate(zip(self.alpha_grid, self.beta_grid)):
                m = like.invcovs[i]
                m = np.array(like.inverse_covariance_matrix(a, b)) if m is None else m
                consts.append(self._consts(m))
                invcovs = fill(invcovs, i, m)
            self.invcovs = invcovs
            self.consts = [np.stack(c) for c in zip(*consts)]
            if like.precompute_covmats:
                host = np.asarray(invcovs)
                for i in range(len(host)):
                    like.invcovs[i] = host[i]
            if not self.use_abs_mag:  # the offset's inverse-variance weights per grid point
                a, b = self.alpha_grid[:, None], self.beta_grid[:, None]
                self.w = 1.0 / (pre + a * a * self.stretch_var + b * b * self.colour_var
                                + 2.0 * a * self.cov_mag_stretch - 2.0 * b * self.cov_mag_colour
                                - 2.0 * (a * b) * self.cov_stretch_colour)
                self.wtval = np.sum(self.w, axis=1)

    def _consts(self, invcov):
        """The data-independent products of ``alpha_beta_logp`` (Cobaya's operation order)."""
        if self.twoscript:
            v = invcov.dot(self.A1)
            D, E = v.dot(self.A2), v.dot(self.A1)
            F = invcov.dot(self.A2).dot(self.A2)
            return D, E, F
        return (np.sum(invcov),)

    def _need(self, p):
        if p not in self.params and p not in self._fixed:  # Cobaya's logp would fail
            raise NotImplementedError(f"{self.cobaya_name}: parameter {p!r} is neither sampled nor fixed")

    def _param(self, inp, p):
        return inp.params[p] if p in self.params else float(self._fixed[p])

    def _cov(self, alpha, beta):
        """C(α, β) of ``SN.inverse_covariance_matrix`` (before the inverse)."""
        c = self.covs
        alphasq, betasq, alphabeta = alpha * alpha, beta * beta, alpha * beta
        cov = jnp.asarray(c["mag"]) if "mag" in c else 0.0
        if "stretch" in c:
            cov = cov + alphasq * jnp.asarray(c["stretch"])
        if "colour" in c:
            cov = cov + betasq * jnp.asarray(c["colour"])
        if "mag_stretch" in c:
            cov = cov + 2 * alpha * jnp.asarray(c["mag_stretch"])
        if "mag_colour" in c:
            cov = cov - 2 * beta * jnp.asarray(c["mag_colour"])
        if "stretch_colour" in c:
            cov = cov - 2 * alphabeta * jnp.asarray(c["stretch_colour"])
        delta = (self.pre_vars + alphasq * self.stretch_var + betasq * self.colour_var
                 + 2.0 * alpha * self.cov_mag_stretch + -2.0 * beta * self.cov_mag_colour
                 + -2.0 * alphabeta * self.cov_stretch_colour)
        return cov.at[jnp.diag_indices(len(self.mag))].add(delta)

    def _diffmag(self, lum, alpha, beta, Mb, w=None, wtval=None):
        mag = jnp.asarray(self.mag)
        if self.use_abs_mag:
            sm = Mb + 25
        elif self.alphabeta and w is None:
            w = 1.0 / (self.pre_vars + alpha * alpha * self.stretch_var + beta * beta * self.colour_var
                       + 2.0 * alpha * self.cov_mag_stretch - 2.0 * beta * self.cov_mag_colour
                       - 2.0 * (alpha * beta) * self.cov_stretch_colour)
            sm = jnp.sum((mag - lum) * w) / jnp.sum(w)
        else:
            sm = jnp.sum((mag - lum) * w) / wtval
        if self.alphabeta:
            return mag - lum + alpha * jnp.asarray(self.stretch) - beta * jnp.asarray(self.colour) - sm
        return mag - lum - sm

    def _logp(self, d, Md, consts):
        """``alpha_beta_logp`` from the residual d, M d and the constant products."""
        A = Md @ d
        if self.twoscript:
            D, E, F = consts
            chi2 = _offsets_chi2(A, Md @ jnp.asarray(self.A1), Md @ jnp.asarray(self.A2), D, E, F,
                                 self.use_abs_mag)
        else:
            (E,) = consts
            chi2 = A + jnp.log(E / _TWOPI)
            if not self.use_abs_mag:
                chi2 = chi2 - jnp.sum(Md) ** 2 / E
        return -chi2 / 2

    def loglike(self, inp):
        da = inp.theory["angular_diameter_distance"][self.idx]
        lum = 5 * jnp.log10(jnp.asarray(self.zfac) * da)
        Mb = self._param(inp, "Mb") if self.use_abs_mag else 0.0
        if self.mode == "plus":
            if self.calib is not None:  # Cepheid host distances as theory for the calibrators
                lum = jnp.where(self.calib, jnp.asarray(self.ceph), lum)
            d = jnp.asarray(self.mag) - lum - (Mb + 25 if self.use_abs_mag else 0.0)
            return -(d @ jnp.asarray(self.invcov)) @ d / 2.0
        if self.mode == "fixed":
            d = self._diffmag(lum, 0.0, 0.0, Mb, self.w, self.wtval)
            return self._logp(d, jnp.asarray(self.invcov) @ d, self.consts)
        if self.mode == "sampled":
            alpha, beta = self._param(inp, self.alpha_name), self._param(inp, self.beta_name)
            d = self._diffmag(lum, alpha, beta, Mb)
            cf = jax.scipy.linalg.cho_factor(self._cov(alpha, beta), lower=True)
            if self.twoscript:
                X = jax.scipy.linalg.cho_solve(cf, jnp.stack([d, jnp.asarray(self.A1), jnp.asarray(self.A2)], 1))
                MA1, MA2 = X[:, 1], X[:, 2]
                consts = (MA1 @ jnp.asarray(self.A2), MA1 @ jnp.asarray(self.A1), MA2 @ jnp.asarray(self.A2))
            else:
                X = jax.scipy.linalg.cho_solve(cf, jnp.stack([d, jnp.ones_like(d)], 1))
                consts = (jnp.sum(X[:, 1]),)
            return self._logp(d, X[:, 0], consts)

        # marginalized over the (α, β) grid
        def grid_point(a, b, M, consts, w, wtval):
            d = self._diffmag(lum, a, b, Mb, w, wtval)
            return -self._logp(d, M @ d, consts)

        g = len(self.alpha_grid)
        w, wtval = (self.w, self.wtval) if not self.use_abs_mag else (np.zeros((g, 1)), np.zeros(g))
        marge = jax.vmap(grid_point)(jnp.asarray(self.alpha_grid), jnp.asarray(self.beta_grid),
                                     self.invcovs, tuple(jnp.asarray(c) for c in self.consts),
                                     jnp.asarray(w), jnp.asarray(wtval))
        best = jnp.min(marge)
        return -best + jnp.log(jnp.sum(jnp.exp(-marge + best)) * self.steps[0] * self.steps[1])


__all__ = ["SN", "SN_SETS"]
