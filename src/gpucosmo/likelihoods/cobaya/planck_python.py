# This file is part of GPUCosmo. Copyright (C) 2025-2026 Leonel Medina-Varela.
# SPDX-License-Identifier: GPL-3.0-or-later
# Derived in part from the following, translated to JAX and modified for GPUCosmo:
# - Cobaya (https://github.com/CobayaSampler/cobaya), Copyright 2017-2026 Jesus Torrado and
#   Antony Lewis.
#   Licence: LICENSES/Cobaya.txt.
# See NOTICE.md.
"""Native ports of Cobaya's pure-Python Planck likelihoods (no clik).

* ``CamSpec`` — ``base_classes.planck_2018_CamSpec_python.Planck2018CamSpecPython`` (2018
  CamSpec, foreground model with the SZ prior) and its subclasses
  ``planck_2018_highl_CamSpec2021`` / ``planck_NPIPE_highl_CamSpec`` (``get_foregrounds``
  switched by the fixed ``use_fg_residual_model``: power-law residuals, the 2018 model or
  both). Gaussian in the data vector of the used spectra / ℓ ranges (``dataset_params``
  ``use_cl``, ``use_range`` are Cobaya's): theory + foregrounds over the calibrations,
  χ² with Cobaya's inverse covariance. For ≥ 1500 data points Cobaya's ``chi_squared``
  goes through BLAS ``dsymv`` and reads only the lower triangle of the inverse (``_Chi2``;
  the cached inverses are not exactly symmetric).
* ``PlikLite`` — ``base_classes.planck_pliklite.PlanckPlikLite`` (the native plik-lite
  TT / TTTEEE): binned D_ℓ over A_planck², Gaussian.
* ``LowlTT`` — ``planck_2018_lowl.TT`` (Python Gibbs/Commander approximation): per-ℓ
  Gaussianizing FITPACK splines x(C_ℓ) (``splines._basis``, the B-spline of SciPy's
  ``InterpolatedUnivariateSpline`` and of its derivative), log-Jacobian, Gaussian in x;
  −inf outside the spline prior bounds or where dx/dC < 0, as Cobaya.
* ``LowlEE`` — ``planck_2018_lowl.EE`` / ``EE_sroll2`` (SimAll / SRoll2 tabulated
  likelihoods): table row ``int(D_ℓ / (A_planck² · step))`` per ℓ, summed. Piecewise
  constant in the spectrum, so its gradient is zero (exact port, the default);
  ``smooth=True`` interpolates the table linearly between bin centres instead (a
  differentiable variant, not Cobaya's value).

The calibration ``A_planck`` of ``LowlTT`` / ``LowlEE`` (Cobaya: supported, default 1) is
used when the likelihood has it as a parameter; the bridge never assigns it, so with the
default setup it is 1, as in the bridge.

Large inverse covariances (CamSpec: up to 11786², 1.1 GB in float64) are held as device
arrays. JAX's default embeds closed-over arrays as constants in every compiled program;
with ``JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS=true`` (set before importing JAX) they are passed
by reference instead (no per-executable copies).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from gpucosmo.likelihoods.cobaya.base import CobayaPort
from gpucosmo.likelihoods.cobaya.cmblikes import _sqrt0
from gpucosmo.likelihoods.cobaya.splines import _basis

_REQUIRED = object()


class _Chi2:
    """δᵀ C⁻¹ δ as Cobaya's ``chi_squared``: ``c_inv.dot(δ).dot(δ)`` below 1500 points, else
    BLAS ``dsymv`` on ``c_inv.T`` with ``lower=0``, which reads only the lower triangle L of
    ``c_inv`` (the cached inverses are not exactly symmetric): 2 δᵀLδ − Σ diag(L) δ². The
    matrix lives on the device (one copy: L is zeroed above the diagonal in place)."""

    def __init__(self, covinv):
        self.full = covinv.shape[0] < 1500
        c = jnp.array(covinv, copy=True)  # never an alias of Cobaya's array (donated below)
        if self.full:
            self.M, self.diag = c.astype(jnp.float64), None
        else:
            self.M = jax.jit(lambda m: jnp.tril(m.astype(jnp.float64)), donate_argnums=0)(c)
            self.diag = np.asarray(np.diag(covinv), dtype=np.float64)

        self.ref = None

    def set_reference(self, delta0) -> None:
        """Centre the χ² on the residual ``delta0`` (concrete, e.g. at the expansion
        point): χ² = c₀ + 2 g·e + eᵀ S e with e = δ − δ₀, g = S δ₀, c₀ = δ₀ᵀ S δ₀ in float64 and the
        quadratic term in float32 against S = the symmetric matrix Cobaya's product implies (the
        lower triangle mirrored, or (M + Mᵀ)/2). Algebraically exact; with a posterior-typical
        reference eᵀSe is O(the number of parameters), so float32 costs ~1e-6 in χ², while the
        batched product runs as an SGEMM (~35× the float64 GEMM's speed on a consumer GPU).
        S, g and c₀ are computed on the host: in float64 on the device, S and Mᵀ take two more copies
        of the matrix (0.8 GB each for CamSpec's 9915 points), more than the SMC stage's memory pool."""
        d0 = np.asarray(delta0, dtype=np.float64)
        M = np.asarray(jax.device_get(self.M), dtype=np.float64)
        S = 0.5 * (M + M.T) if self.full else M + M.T - np.diag(self.diag)
        del M
        g = S @ d0
        self.ref = dict(d0=jnp.asarray(d0), g=jnp.asarray(g), c0=jnp.asarray(float(g @ d0)),
                        S32=jnp.asarray(S.astype(np.float32)))
        del S

    def __call__(self, delta):
        if self.ref is not None:
            r = self.ref
            e = delta - r["d0"]
            e32 = e.astype(jnp.float32)
            q = jnp.dot(jnp.matmul(e32, r["S32"], precision=jax.lax.Precision.HIGHEST), e32)
            return r["c0"] + 2.0 * jnp.dot(r["g"], e) + q.astype(jnp.float64)
        q = jnp.dot(delta @ self.M, delta)  # = δᵀMδ; δ @ M is a plain GEMM when batched
        if self.full:
            return q
        return 2.0 * q - jnp.dot(jnp.asarray(self.diag) * delta, delta)


class _PlanckPort(CobayaPort):
    """Parameter access shared by the ports: sampled (``inp.params``), fixed in the Cobaya
    model, or a default (Cobaya's ``data_params.get(name, default)``)."""

    def _init_params(self, like):
        const = self.bridge._model.parameterization.constant_params()
        self.fixed = {p: float(const[p]) for p in like.input_params if p in const}

    def _has(self, name) -> bool:
        return name in self.params or name in self.fixed

    def _param(self, inp, name, default=_REQUIRED):
        if name in inp.params:
            return inp.params[name]
        if name in self.fixed:
            return self.fixed[name]
        if default is _REQUIRED:
            raise KeyError(f"{self.cobaya_name}: parameter {name!r} is neither sampled nor fixed")
        return default


# ---------------------------------------------------------------- CamSpec
_CAMSPEC = ("planck_2018_highl_CamSpec.TT", "planck_2018_highl_CamSpec.TTTEEE")
_CAMSPEC += tuple(f"{d}.{s}" for d in ("planck_2018_highl_CamSpec2021", "planck_NPIPE_highl_CamSpec")
                  for s in ("TT", "TE", "EE", "TTTE", "TEEE", "TTTEEE"))
_FG2018 = ("aps100", "aps143", "aps217", "psr", "dust100", "dust143", "dust217", "dust143x217", "acib217",
           "cibr", "ncib", "cibrun", "asz143", "xi", "aksz")
_POWERLAW = ("amp_100", "amp_143", "amp_217", "amp_143x217", "n_100", "n_143", "n_217", "n_143x217")


def _sz_prior(aksz, asz143):
    """``prior_SZ_CamSpec``: stats.norm.logpdf(aksz + 1.6 asz143, loc=9.5, scale=3)."""
    x = (aksz + 1.6 * asz143 - 9.5) / 3.0
    return -x**2 / 2.0 - np.log(np.sqrt(2 * np.pi)) - np.log(3.0)


class CamSpec(_PlanckPort):
    """Planck CamSpec high-ℓ TT/TE/EE (2018, 2021 12.6HMcl, NPIPE/PR4 12.6cl), native
    Python version (``Planck2018CamSpecPython`` and subclasses)."""

    cobaya_names = _CAMSPEC
    handles_external_priors = True

    def setup(self, like):
        from cobaya.likelihoods.base_classes.planck_2018_CamSpec_python import Planck2018CamSpecPython
        from cobaya.likelihoods.planck_2018_highl_CamSpec2021.planck_2018_CamSpec2021_python import (
            Planck2018CamSpec2021Python)

        if not isinstance(like, Planck2018CamSpecPython):
            raise NotImplementedError(f"{self.cobaya_name}: not a native CamSpec likelihood")
        self._init_params(like)
        # data-vector segments: (spectrum index 0..5, ℓ of each entry), in Cobaya's order
        self.segments = [(i, np.asarray(like.ell_ranges[i], dtype=int))
                         for i, n in enumerate(like.used_sizes) if n > 0]
        self.data = np.asarray(like.data_vector, dtype=np.float64)
        self.use_tt = bool(np.any(like.cl_used[:4]))
        # foreground model: 2018 (``Planck2018CamSpecPython.get_foregrounds``) and/or power law
        self.fg_2018, self.fg_powerlaw = self.use_tt, False
        if self.use_tt and isinstance(like, Planck2018CamSpec2021Python):
            if "use_fg_residual_model" not in self.fixed:
                raise NotImplementedError(f"{self.cobaya_name}: use_fg_residual_model must be fixed")
            model = int(self.fixed["use_fg_residual_model"])
            if model not in (0, 1, 2):
                raise ValueError(f"{self.cobaya_name}: use_fg_residual_model {model} (Cobaya: 0, 1 or 2)")
            self.fg_2018, self.fg_powerlaw = model in (1, 2), model in (0, 2)
        need = (_FG2018 if self.fg_2018 else ()) + (_POWERLAW if self.fg_powerlaw else ())
        missing = [p for p in need if not self._has(p)]
        if missing:
            raise NotImplementedError(f"{self.cobaya_name}: foreground parameters {missing} neither sampled nor fixed")
        if self.fg_2018:  # templates at the ℓ of each TT segment (lnrat is integer in Cobaya: kept)
            self.tmpl = {i: dict(sz=like.sz_143[ls], ksz=like.ksz[ls], tszxcib=like.tszxcib[ls], cib=like.cib_217[ls],
                                 dust=like.dust[i][ls], lnrat=np.asarray(like.lnrat[ls], dtype=float),
                                 llp1=np.asarray(like.llp1[ls], dtype=float))
                         for i, ls in self.segments if i <= 3}
        self.spectra = sorted({("tt", "tt", "tt", "tt", "te", "ee")[i] for i, _ in self.segments})
        # external priors: the 2018 SZ prior only
        ext = self.bridge._model.prior.external
        if ext:
            if set(ext) != {"SZ"} or sorted(ext["SZ"].params) != ["aksz", "asz143"]:
                raise NotImplementedError(f"{self.cobaya_name}: prior terms {list(ext)}")
            for a, s in ((1.0, 4.0), (3.5, 0.7)):
                if not np.isclose(ext["SZ"].logp(aksz=a, asz143=s), _sz_prior(a, s), rtol=1e-12, atol=0):
                    raise NotImplementedError(f"{self.cobaya_name}: SZ prior differs from prior_SZ_CamSpec")
        self.sz_prior = bool(ext)
        self.chi2 = _Chi2(like.covinv)

    def cals(self, inp):
        """``get_cals``: (cal0, 1, cal2, √cal2, calTE, calEE) · A_planck²."""
        p = lambda n: self._param(inp, n, 1.0)  # noqa: E731
        cal2 = p("cal2")
        return [c * p("A_planck") ** 2 for c in (p("cal0"), 1.0, cal2, jnp.sqrt(cal2), p("calTE"), p("calEE"))]

    def foregrounds(self, inp, i, ls):
        """TT foregrounds of spectrum ``i`` (100, 143, 217, 143×217) at the multipoles ``ls``."""
        out = 0.0
        if self.fg_2018:
            p = lambda n: self._param(inp, n)  # noqa: E731
            t = self.tmpl[i]
            aps = [p("aps100"), p("aps143"), p("aps217")]
            aps.append(p("psr") * _sqrt0(aps[1] * aps[2]))
            aps = aps[i] * (1e-6 / 9)
            adust = (p("dust100"), p("dust143"), p("dust217"), p("dust143x217"))[i]
            acib217 = p("acib217")
            acib143 = 0.094 * acib217 / 1.134 * 1.33  # 143 fixed from 217 (Cobaya's default acib143 = −1)
            if self._has("acib143"):
                a = p("acib143")
                acib143 = jnp.where(a < 0, acib143, a)
            cl_cib = jnp.exp(p("ncib") * t["lnrat"] + p("cibrun") * t["lnrat"] ** 2 / 2) * t["cib"]
            asz143, xi = p("asz143"), p("xi")
            ksz = p("aksz") * t["ksz"]
            a_sz143, a_cib143, a_cib217 = asz143 * 0.95, acib143 * 1.134, acib217 * 1.33
            if i == 0:
                fg = ksz + asz143 * 2.022 * t["sz"]
            elif i == 1:
                fg = (a_cib143 * cl_cib + ksz + a_sz143 * t["sz"]
                      - 2.0 * _sqrt0(a_cib143 * a_sz143) * xi * t["tszxcib"])
            elif i == 2:
                fg = a_cib217 * cl_cib + ksz
            else:
                fg = (p("cibr") * (_sqrt0(a_cib143 * a_cib217) * cl_cib) + ksz
                      - _sqrt0(a_cib217 * a_sz143) * xi * t["tszxcib"])
            out = fg + (adust * t["dust"] + aps * t["llp1"])
        if self.fg_powerlaw:
            f = ("100", "143", "217", "143x217")[i]
            out = out + self._param(inp, f"amp_{f}") * (ls / 1500) ** self._param(inp, f"n_{f}")
        return out

    def residual(self, inp):
        """Data minus the calibrated, foreground-added model band powers."""
        cl = inp.shared["cmb"]["cl"]
        C = {s: jnp.asarray(cl[s], dtype=jnp.float64) for s in self.spectra}
        cals = self.cals(inp)
        model = []
        for i, ls in self.segments:
            if i <= 3:
                model.append((C["tt"][ls] + self.foregrounds(inp, i, ls)) / cals[i])
            else:
                model.append(C["te" if i == 4 else "ee"][ls] / cals[i])
        return jnp.asarray(self.data) - jnp.concatenate(model)

    def set_reference(self, inp) -> None:
        """Centre the float32 χ² on the residual at ``inp`` (the bundle's ``set_cmb_reference``,
        called with the DALI expansion's expansion point and the MAP nuisances)."""
        self.chi2.set_reference(jax.device_get(self.residual(inp)))

    def loglike(self, inp):
        delta = self.residual(inp)
        ll = -0.5 * self.chi2(delta)
        if self.sz_prior:
            ll = ll + _sz_prior(self._param(inp, "aksz"), self._param(inp, "asz143"))
        return ll


# ---------------------------------------------------------------- plik-lite
class PlikLite(_PlanckPort):
    """Planck 2018 plik-lite (foreground-marginalized, binned), native Python version
    (``PlanckPlikLite``): TTTEEE and TT."""

    cobaya_names = ("planck_2018_highl_plik.TTTEEE_lite_native", "planck_2018_highl_plik.TT_lite_native")

    def setup(self, like):
        self._init_params(like)
        self.calib = like.calibration_param
        if not self._has(self.calib):
            raise NotImplementedError(f"{self.cobaya_name}: calibration {self.calib!r} neither sampled nor fixed")
        n_l = int(np.max([like.blmax[i] for b in like.used_bins for i in b])) + 1
        self.bins = []  # (spectrum, W (n_bins, n_ℓ)): band = W @ D_ℓ, the weights already over ℓ(ℓ+1)/2π
        for s, used in zip(("tt", "te", "ee"), like.used_bins):
            if len(used):
                W = np.zeros((len(used), n_l))
                for r, i in enumerate(used):
                    W[r, like.blmin[i]:like.blmax[i] + 1] = like.weights[like.blmin[i]:like.blmax[i] + 1]
                self.bins.append((s, W))
        self.n_l = n_l
        self.X = np.asarray(like.X_data, dtype=np.float64)
        self.chi2 = _Chi2(like.invcov)

    def loglike(self, inp):
        cl = inp.shared["cmb"]["cl"]
        band = jnp.concatenate([jnp.asarray(W) @ jnp.asarray(cl[s], dtype=jnp.float64)[:self.n_l]
                                for s, W in self.bins])
        band = band / self._param(inp, self.calib) ** 2
        return -0.5 * self.chi2(jnp.asarray(self.X) - band)


# ---------------------------------------------------------------- low-ℓ
def _splev(t, c, k, x):
    """FITPACK ``splev`` (ext=0) of the B-spline (t, c, k) at scalar ``x``."""
    n = t.shape[0] - k - 1
    i0, N = _basis(t, k, n, x)
    return jnp.dot(jax.lax.dynamic_slice(c, (i0,), (k + 1,)), N)



class LowlTT(_PlanckPort):
    """Planck 2018 low-ℓ TT, Cobaya's native Python version (``planck_2018_lowl.TT``)."""

    cobaya_names = ("planck_2018_lowl.TT",)

    def setup(self, like):
        self._init_params(like)
        self.lmin, self.lmax = int(like.lmin), int(like.lmax)

        def stack(splines):
            t, c, k = zip(*(s._eval_args for s in splines))
            if len({len(x) for x in t}) != 1 or len(set(k)) != 1:
                raise NotImplementedError(f"{self.cobaya_name}: splines of different sizes")
            n = len(t[0]) - k[0] - 1
            return np.stack(t).astype(float), np.stack([np.asarray(x)[:n] for x in c]).astype(float), int(k[0])

        if any(s.ext != 0 for s in like._spline + like._spline_derivative):
            raise NotImplementedError(f"{self.cobaya_name}: spline extrapolation mode")
        self.x_spl = stack(like._spline)
        self.dx_spl = stack(like._spline_derivative)
        self.bounds = np.asarray(like._prior_bounds, dtype=float)
        self.covinv = np.asarray(like._covinv, dtype=float)
        self.mu = np.asarray(like._mu, dtype=float)
        self.offset = float(like._offset)

    def loglike(self, inp):
        cl = jnp.asarray(inp.shared["cmb"]["cl"]["tt"], dtype=jnp.float64)
        th = cl[self.lmin:self.lmax + 1] / self._param(inp, "A_planck", 1.0) ** 2
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        out = jnp.any(th < lo) | jnp.any(th > hi)
        th_ = jnp.clip(th, lo, hi)  # finite values / gradients where the point is rejected

        def ev(spl):
            t, c, k = spl
            return jax.vmap(lambda t_, c_, x_: _splev(t_, c_, k, x_))(jnp.asarray(t), jnp.asarray(c), th_)

        x, dx = ev(self.x_spl), ev(self.dx_spl)
        pos = dx > 0
        logl = jnp.sum(jnp.where(pos, jnp.log(jnp.where(pos, dx, 1.0)), -jnp.inf))
        delta = x - jnp.asarray(self.mu)
        logl = logl - 0.5 * jnp.dot(jnp.asarray(self.covinv) @ delta, delta) - self.offset
        return jnp.where(out | jnp.any(dx < 0), -jnp.inf, logl)


class LowlEE(_PlanckPort):
    """Planck 2018 low-ℓ EE, Cobaya's native Python versions of SimAll
    (``planck_2018_lowl.EE``) and SRoll2 (``planck_2018_lowl.EE_sroll2``).

    ``smooth=True`` (default False): linear interpolation of the table between the bin
    centres (x − ½, x = D_ℓ / (A² step)), constant beyond the first/last centre, −inf
    outside 0 ≤ x < n_steps — differentiable, but not Cobaya's value (which is piecewise
    constant: zero gradient, and negative x wraps around the table as NumPy indexing)."""

    cobaya_names = ("planck_2018_lowl.EE", "planck_2018_lowl.EE_sroll2")

    def __init__(self, cobaya_name: str | None = None, block: dict | None = None, *, smooth: bool = False, **kw):
        block = dict(block or {})
        self.smooth = bool(block.pop("smooth", smooth))
        super().__init__(cobaya_name, block, **kw)

    def setup(self, like):
        self._init_params(like)
        self.lmin, self.lmax = int(like._lmin), int(like._lmax)
        self.step = float(like._stepEE)
        self.table = np.asarray(like.probEE, dtype=float)  # (n_steps, n_ℓ)
        if self.table.shape[1] != self.lmax - self.lmin + 1:
            raise NotImplementedError(f"{self.cobaya_name}: table shape {self.table.shape}")

    def loglike(self, inp):
        cl = jnp.asarray(inp.shared["cmb"]["cl"]["ee"], dtype=jnp.float64)
        x = cl[self.lmin:self.lmax + 1] / (self._param(inp, "A_planck", 1.0) ** 2 * self.step)
        n, cols = self.table.shape[0], np.arange(self.table.shape[1])
        T = jnp.asarray(self.table)
        if self.smooth:
            ok = (x >= 0) & (x < n)
            u = jnp.clip(x - 0.5, 0.0, n - 1.0)
            i0 = jnp.clip(jnp.floor(u).astype(int), 0, n - 2)
            w = u - i0
            vals = (1 - w) * T[i0, cols] + w * T[i0 + 1, cols]
        else:
            idx = jnp.trunc(x)  # NumPy's astype(int)
            ok = (idx >= -n) & (idx < n)  # IndexError otherwise; negative indices wrap
            vals = T[jnp.where(ok, idx, 0).astype(int) % n, cols]
        return jnp.where(jnp.all(ok), jnp.sum(vals), -jnp.inf)


__all__ = ["CamSpec", "LowlEE", "LowlTT", "PlikLite"]
