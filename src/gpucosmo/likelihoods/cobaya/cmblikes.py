# This file is part of GPUCosmo. Copyright (C) 2025-2026 Leonel Medina-Varela.
# SPDX-License-Identifier: GPL-3.0-or-later
# Derived in part from the following, translated to JAX and modified for GPUCosmo:
# - Cobaya (https://github.com/CobayaSampler/cobaya), Copyright 2017-2026 Jesus Torrado and
#   Antony Lewis.
#   Licence: LICENSES/Cobaya.txt.
# See NOTICE.md.
"""Native ports of Cobaya's CMBlikes likelihoods (``cobaya.likelihoods.base_classes.cmblikes``).

``CMBLikes`` re-implements ``CMBlikes.log_likelihood`` on the arrays Cobaya reads from the
``.dataset`` files: theory spectra per required map pair (a spectrum Cobaya's provider would
not hold — not requested — is zero, as in ``get_theory_map_cls``), aberration, foregrounds and
calibration (``adapt_theory_for_maps``), binning windows plus the linear correction
(``get_binned_map_cls``: the lensing renormalization / N1 dependence on TT, EE, TE, PP), noise,
and the three likelihood forms: Gaussian in the bandpowers, Hamimeche–Lewis (``transform``,
matrix square roots / functions through ``jnp.linalg.eigh``) and the exact full-sky form
(``exact_chi_sq``, binned or unbinned). For ``bicep_keck_2018`` the BK18 dust + synchrotron
model of ``bicep_keck_2018.add_foregrounds`` is added (bandpass integrals, decorrelation, band
centre errors).

Served: Planck 2018 lensing (``planck_2018_lensing.native`` / ``CMBMarged`` and the standalone
``plancklensing`` package), Planck PR4 lensing (``planckpr4lensing``, class ``PR4Lensing``) and
BICEP/Keck 2018. BK18 is validated with our theory's BB (lensed scalar, r = 0: the theory has
no tensor modes yet). Not served: ``planckpr4lensing.ISWLensingTT`` / ``ISWLensingTTPP``,
which need C_ℓ^{φT} (not computed by our theory, so the bridge cannot serve them either) and
do not build in Cobaya with the installed NumPy (a ``str * numpy.bool`` in the package's
``iswlens_jtliks/lik.py``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from gpucosmo.likelihoods.cobaya.base import CobayaPort

_SPECTRA = ("tt", "te", "ee", "bb", "pp")  # spectra of ``shared["cmb"]["cl"]`` Cobaya can receive


def _tri(n: int) -> np.ndarray:
    """(n, n) index of element (i, j) in CMBlikes' packed order (``elements_to_matrix``):
    row by row, lower triangle (0,0), (1,0), (1,1), (2,0), ..."""
    idx = np.empty((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1):
            idx[i, j] = idx[j, i] = i * (i + 1) // 2 + j
    return idx


def _lower(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Row/column indices of the packed elements (``matrix_to_elements``: lower triangle)."""
    r, c = zip(*[(i, j) for i in range(n) for j in range(i + 1)])
    return np.asarray(r), np.asarray(c)


def _hl_g(x):
    """sign(x − 1) √(2 (x − ln x − 1)), its first order (x − 1) where the root's argument
    vanishes (Cobaya: max(0, ·)), so the gradient stays finite at x = 1."""
    a = x - jnp.log(x) - 1
    ok = a > 1e-300
    g = jnp.where(ok, jnp.sign(x - 1) * jnp.sqrt(2 * jnp.where(ok, a, 1.0)), x - 1)
    return jnp.where(jnp.isnan(a), jnp.nan, g)


class CMBLikes(CobayaPort):
    """Cobaya ``CMBlikes`` (``.dataset`` bandpower likelihoods): Planck 2018 lensing (native,
    CMB-marginalized; also the ``plancklensing`` package) and BICEP/Keck 2018 (HL likelihood
    with the BK18 foreground model)."""

    cobaya_names = ("planck_2018_lensing.native", "planck_2018_lensing.CMBMarged", "bicep_keck_2018",
                    "plancklensing.PlanckLensing", "plancklensing.PlanckLensingMarged")

    def setup(self, like):
        self.approx = like.like_approx
        if self.approx not in ("gaussian", "HL", "exact"):
            raise NotImplementedError(f"{self.cobaya_name}: like_approx {self.approx!r}")
        self.lmin, self.lmax = int(like.pcl_lmin), int(like.pcl_lmax)
        self.binned = bool(like.binned)
        self.nmaps, self.nreq = int(like.nmaps), int(like.nmaps_required)
        self.nbins = int(like.nbins_used)
        self.bin_min, self.bin_max = int(like.bin_min), int(like.bin_max)
        # required map pairs (i ≥ j, packed order) and the theory spectrum each reads
        self.pairs = [(i, j) for i in range(self.nreq) for j in range(i + 1)]
        have = set(self.requirements()["cmb"]["spectra"])
        self.pair_spec, theory_ij = [], []
        for i, j in self.pairs:
            tij = tuple(int(k) for k in like.map_cls[i, j].theory_ij)
            s = "".join(like.field_names[k] for k in tij).lower()
            self.pair_spec.append(s if s in have and s in _SPECTRA else None)
            theory_ij.append(tij)
        self.theory_ij = theory_ij
        self.tp_pairs = np.asarray([a <= 2 and b <= 2 for a, b in theory_ij])  # T/E/B (calibration, aberration)
        self.calib = like.calibration_param
        self.aberration = float(like.aberration_coeff)
        # parameter values: sampled (``inp.params``) or fixed in the Cobaya model
        const = self.bridge._model.parameterization.constant_params()
        self.fixed = {p: float(const[p]) for p in like.input_params if p in const}
        bad = [p for p in like.input_params if p not in self.params and p not in self.fixed]
        if bad:
            raise NotImplementedError(f"{self.cobaya_name}: input parameters {bad} are neither sampled nor fixed")
        if self.calib not in like.input_params:
            self.calib = None  # Cobaya calibrates only if the parameter is passed to logp
        self.fg = _BK18Foregrounds(like, self) if _is_bk(like) else None
        pair_index = {p: k for k, p in enumerate(self.pairs)}
        self.fold = None
        if self.binned:
            self.bins = self._windows(like.bins, pair_index, like.ncl)
            if like.linear_correction is not None:
                self.lin = self._windows(like.linear_correction, pair_index, like.ncl)
                self.fid_corr = np.asarray(like.fid_correction, dtype=float).T.copy()
            else:
                self.lin = None
            if self.fg is None and not self.aberration:
                self.fold = self._fold([self.bins] + ([self.lin] if self.lin is not None else []), like.ncl)
        else:
            # unbinned (exact only): C_b from the map-pair spectra at ℓ = bin_min + b
            self.unb_pair = np.asarray([[pair_index[(max(i, j), min(i, j))] for j in range(self.nmaps)]
                                        for i in range(self.nmaps)])
        self.tri = _tri(self.nmaps)
        self.low = _lower(self.nmaps)
        self.noise = None if like.cl_noise is None else np.asarray(like.cl_noise, dtype=float).T.copy()
        self.bp = np.asarray(like.bandpowers, dtype=float).T.copy()  # (n_bins, ncl): Ĉ elements
        self.Chat = np.asarray(like.bandpower_matrix, dtype=float)
        self.noise_mat = np.asarray(like.noise_matrix, dtype=float)
        if self.approx == "HL":
            self.Cfhalf = np.real(np.asarray(like.fiducial_sqrt_matrix)).astype(float)
        if self.approx == "exact":
            self.fsky = float(like.fsky)
            self.Ls = np.arange(self.nbins) + self.bin_min
        else:
            self.used = np.asarray(like.cl_used_index, dtype=int)
            self.covinv = np.asarray(like.covinv, dtype=float)

    def _windows(self, bw, pair_index, ncl):
        """``BinWindows`` → (W (n_in, n_bins, n_ℓ), map pair of each input, (n_in, ncl) one-hot
        of the output column) for the inputs ``BinWindows.bin`` uses: output used and
        ``map_cls[x, y]`` not None (indices as NumPy reads them: −1 for a map not required
        wraps around)."""
        n = self.nreq
        keep, pin, out = [], [], []
        for i, ((x, y), o) in enumerate(zip(bw.cols_in.T, bw.cols_out)):
            x, y = int(x) % n, int(y) % n
            if o < 0 or (x, y) not in pair_index:
                continue
            keep.append(i)
            pin.append(pair_index[(x, y)])
            out.append(np.eye(ncl)[o])
        W = np.asarray(bw.binning_matrix, dtype=float)[keep]
        return W, np.asarray(pin, dtype=int), np.asarray(out).reshape(len(keep), ncl)

    def _fold(self, wins, ncl):
        """Without foregrounds or aberration the binned theory is linear in the spectra: the
        windows summed per input spectrum, split into calibrated (T/E/B) and uncalibrated
        groups — {calibrated?: (spectra, W (n_spectra, n_bins, ncl, n_ℓ))}."""
        acc = {}
        for W, pin, out in wins:
            for i, k in enumerate(pin):
                s, o = self.pair_spec[k], np.flatnonzero(out[i])
                if s is None or not len(o):
                    continue  # zero spectrum or unused output
                key = (bool(self.calib is not None and self.tp_pairs[k]), s)
                a = acc.setdefault(key, np.zeros((W.shape[1], ncl, W.shape[2])))
                a[:, o[0], :] += W[i]
        fold = {}
        for cal in (False, True):
            specs = sorted(s for c, s in acc if c == cal)
            if specs:
                fold[cal] = (specs, np.stack([acc[(cal, s)] for s in specs]))
        return fold

    def _param(self, inp, name):
        return inp.params[name] if name in inp.params else self.fixed[name]

    # ---- theory map spectra (``get_theory_map_cls`` + ``adapt_theory_for_maps``) ----
    def map_cls(self, cl, inp):
        """(n_pairs, n_ℓ) D_ℓ of the required map pairs, ℓ = cl_lmin … cl_lmax."""
        n = self.lmax - self.lmin + 1
        rows = [jnp.asarray(cl[s], dtype=jnp.float64)[self.lmin:self.lmax + 1] if s else jnp.zeros(n)
                for s in self.pair_spec]
        CL = jnp.stack(rows)
        if self.aberration:
            CL = self._aberration(CL)
        if self.fg is not None:
            CL = self.fg.add(CL, inp)
        if self.calib is not None:
            a2 = self._param(inp, self.calib) ** 2
            CL = jnp.where(self.tp_pairs[:, None], CL / a2, CL)
        return CL

    def _aberration(self, CL):
        ells = np.arange(self.lmin, self.lmax + 1, dtype=float)
        norm = ells * (ells + 1)
        c = CL / norm
        d = (c[:, 2:] - c[:, :-2]) / 2
        d = jnp.concatenate([d[:, :1], d, d[:, -1:]], axis=1) * norm * ells
        return jnp.where(self.tp_pairs[:, None], CL + self.aberration * d, CL)

    @staticmethod
    def _bin(win, CL):
        W, pin, out = win
        contrib = jnp.einsum("ibl,il->ib", jnp.asarray(W), CL[pin])  # (n_in, n_bins)
        return contrib.T @ jnp.asarray(out)  # (n_bins, ncl)

    def binned_cls(self, CL):
        """``get_binned_map_cls``: (n_bins, ncl) binned theory (+ linear correction)."""
        band = self._bin(self.bins, CL)
        if self.lin is not None:
            band = band + (self._bin(self.lin, CL) - jnp.asarray(self.fid_corr))
        return band

    def binned_theory(self, cl, inp):
        """Binned theory from the spectra: through the folded windows when linear (``_fold``),
        by binning spectra + foreground templates with one matrix product when the foreground
        coefficients are ℓ-independent (``_bin_templates``), else through the map spectra.
        (Calibration is applied after binning; equal to Cobaya's order up to rounding.)"""
        if self.fold is None:
            if self.fg is not None and not self.aberration and self.fg.separable(inp):
                return self._bin_templates(cl, inp)
            return self.binned_cls(self.map_cls(cl, inp))
        band = jnp.zeros(self.bp.shape)
        for cal, (specs, W) in self.fold.items():
            S = jnp.stack([jnp.asarray(cl[s], dtype=jnp.float64)[self.lmin:self.lmax + 1] for s in specs])
            b = jnp.einsum("sbol,sl->bo", jnp.asarray(W), S)
            band = band + (b / self._param(inp, self.calib) ** 2 if cal else b)
        return band - jnp.asarray(self.fid_corr) if self.lin is not None else band

    def _bin_templates(self, cl, inp):
        """Map spectrum k = Σ_t A[k, t] X_t(ℓ), X = the theory spectra and the foreground
        templates: each window is applied to every X_t in one (n_in·n_bins, n_ℓ) × (n_ℓ, n_t)
        product, then combined with the pair coefficients."""
        specs = sorted({s for s in self.pair_spec if s})
        n = self.lmax - self.lmin + 1
        tmpl, coefs = self.fg.terms(inp)
        X = jnp.stack([jnp.asarray(cl[s], dtype=jnp.float64)[self.lmin:self.lmax + 1] for s in specs]
                      + [jnp.broadcast_to(t, (n,)) for t in tmpl])
        rows = []
        for k, c in enumerate(coefs):
            r = [1.0 if self.pair_spec[k] == s else 0.0 for s in specs]
            rows.append(jnp.stack([jnp.asarray(v, dtype=jnp.float64) for v in r + list(c or (0.0, 0.0, 0.0))]))
        A = jnp.stack(rows)
        if self.calib is not None:
            A = jnp.where(self.tp_pairs[:, None], A / self._param(inp, self.calib) ** 2, A)

        def bin_(win):
            W, pin, out = win
            Y = (jnp.asarray(W.reshape(-1, n)) @ X.T).reshape(W.shape[0], W.shape[1], -1)
            return jnp.einsum("ibt,it->ib", Y, A[pin]).T @ jnp.asarray(out)

        band = bin_(self.bins)
        if self.lin is not None:
            band = band + (bin_(self.lin) - jnp.asarray(self.fid_corr))
        return band

    # ---- likelihood forms ----
    def _hl(self, C, Chat, Cfhalf):
        """``CMBlikes.transform`` for one bin."""
        if self.nmaps == 1:
            rat = Chat[0, 0] / C[0, 0]
            return (_hl_g(rat) * Cfhalf[0, 0] ** 2).reshape(1, 1)
        d, U = jnp.linalg.eigh(C)
        r = jnp.sqrt(d)
        rot = (U.T @ Chat @ U) / jnp.outer(r, r)
        rot = U @ rot @ U.T  # C^{-1/2} Ĉ C^{-1/2}
        d, V = jnp.linalg.eigh(rot)
        W = Cfhalf @ V
        return (W * _hl_g(d)) @ W.T

    def _exact(self, C, Chat, L):
        if self.nmaps == 1:
            rat = Chat[0, 0] / C[0, 0]
            return (2 * L + 1) * self.fsky * (rat - 1 - jnp.log(rat))
        M = jnp.linalg.solve(C, Chat)
        return (2 * L + 1) * self.fsky * (jnp.trace(M) - self.nmaps - jnp.linalg.slogdet(M)[1])

    def loglike(self, inp):
        cl = inp.shared["cmb"]["cl"]
        if self.binned:
            band = self.binned_theory(cl, inp)  # (n_bins, ncl)
            C = band[:, self.tri]
        else:
            CL = self.map_cls(cl, inp)
            sl = slice(self.bin_min - self.lmin, self.bin_max - self.lmin + 1)
            C = jnp.moveaxis(CL[:, sl][self.unb_pair], -1, 0)  # (n_bins, n, n)
        if self.noise is not None:
            C = C + jnp.asarray(self.noise_mat)
        if self.approx == "exact":
            chi2 = jnp.sum(jax.vmap(self._exact)(C, jnp.asarray(self.Chat), jnp.asarray(self.Ls, dtype=float)))
        else:
            if self.approx == "HL":
                C = jax.vmap(self._hl)(C, jnp.asarray(self.Chat), jnp.asarray(self.Cfhalf))
                x = C[:, self.low[0], self.low[1]]
            else:
                x = (band + jnp.asarray(self.noise) if self.noise is not None else band) - jnp.asarray(self.bp)
            x = x[:, self.used].ravel()
            chi2 = jnp.asarray(self.covinv) @ x @ x
        ll = -0.5 * chi2
        return jnp.where(jnp.isfinite(ll), ll, -jnp.inf)  # the bridge maps NaN (failed HL / exact) to −inf


class PR4Lensing(CMBLikes):
    """Planck PR4 (NPIPE) lensing, package ``planckpr4lensing`` (a CMBlikes data set: Gaussian
    in the binned C_ℓ^φφ with the linear lensing-response correction; ``Marged``: CMB
    marginalized)."""

    cobaya_names = ("planckpr4lensing.PlanckPR4Lensing", "planckpr4lensing.PlanckPR4LensingMarged")


def _sqrt0(x):
    """√x with a zero (not NaN) gradient at x = 0 (A_dust or A_sync at its prior bound)."""
    pos = x > 0
    return jnp.where(pos, jnp.sqrt(jnp.where(pos, x, 1.0)), jnp.where(x == 0, 0.0, jnp.nan))


def _is_one(x) -> bool:
    """A fixed (non-traced) band-centre error of exactly 1: no correction (Cobaya's branch)."""
    return isinstance(x, (int, float)) and x == 1


def _need_decorr(delta):
    """Cobaya's |Δ − 1| > 1e-5 switch: a Python bool for fixed Δ, a traced one otherwise."""
    if isinstance(delta, (int, float)):
        return bool(abs(delta - 1) > 1e-5)
    return jnp.abs(delta - 1) > 1e-5


def _is_bk(like) -> bool:
    try:
        from cobaya.likelihoods.bicep_keck_2018 import bicep_keck_2018
    except ImportError:
        return False
    return isinstance(like, bicep_keck_2018)


class _BK18Foregrounds:
    """``bicep_keck_2018.add_foregrounds``: dust (modified blackbody) + synchrotron (power law)
    integrated over each map's bandpass, their correlation, frequency decorrelation and band
    centre errors, added to the EE/BB map pairs.

    The band-centre-error correction of the dust scaling follows CosmoMC (and the synchrotron
    branch): Cobaya's dust branch raises (``bandpass % nu_bar``) whenever a γ parameter is
    non-zero; with the default (fixed γ = 0) both coincide."""

    def __init__(self, like, port):
        import importlib

        mod = importlib.import_module(type(like).__module__)
        self.GK, self.T_CMB = float(mod.Ghz_Kelvin), float(mod.T_CMB_K)
        if like.nmaps != like.nmaps_required:
            raise NotImplementedError(f"{port.cobaya_name}: BK foregrounds with maps_required beyond maps_use")
        self.nu = [np.asarray(b.R[:, 0], dtype=float) for b in like.bandpasses]
        self.R = [np.asarray(b.R[:, 1], dtype=float) for b in like.bandpasses]
        self.dnu = [np.asarray(b.dnu, dtype=float) for b in like.bandpasses]
        self.th_dust = [float(b.th_dust) for b in like.bandpasses]
        self.th_sync = [float(b.th_sync) for b in like.bandpasses]
        self.nu_bar = [float(b.nu_bar) for b in like.bandpasses]
        self.fpivot_dust, self.fpivot_sync = float(like.fpivot_dust), float(like.fpivot_sync)
        self.decorr_dust = tuple(map(float, like.fpivot_dust_decorr))
        self.decorr_sync = tuple(map(float, like.fpivot_sync_decorr))
        self.lform_dust, self.lform_sync = like.lform_dust_decorr, like.lform_sync_decorr
        # band-centre-error parameter of each map (None: no error)
        self.gamma = []
        for name in like.used_map_order:
            g = next((f"gamma_{f}" for f in ("95", "150", "220") if f in name), None)
            self.gamma.append(g)
        self.rat = np.arange(like.pcl_lmin, like.pcl_lmax + 1) / 80.0
        self.port = port

    def dust_scaling(self, i, beta, Tdust, bce):
        GK, T = self.GK, self.T_CMB
        nu, R, dnu = self.nu[i], self.R[i], self.dnu[i]
        gb_int = jnp.sum(dnu * R * nu ** (3 + beta) / (jnp.exp(GK * nu / Tdust) - 1))
        nu0 = self.fpivot_dust
        gb0 = nu0 ** (3 + beta) / (jnp.exp(GK * nu0 / Tdust) - 1)
        out = (gb_int / gb0) / self.th_dust[i]
        if _is_one(bce):
            return out
        nb = GK * self.nu_bar[i]
        th_err = bce**4 * jnp.exp(nb * (bce - 1) / T) * (jnp.exp(nb / T) - 1) ** 2 / (jnp.exp(nb * bce / T) - 1) ** 2
        gb_err = bce ** (3 + beta) * (jnp.exp(nb / Tdust) - 1) / (jnp.exp(nb * bce / Tdust) - 1)
        return out * (gb_err / th_err)

    def sync_scaling(self, i, beta, bce):
        GK, T = self.GK, self.T_CMB
        nu, R, dnu = self.nu[i], self.R[i], self.dnu[i]
        pl_int = jnp.sum(dnu * R * nu ** (2 + beta))
        pl0 = self.fpivot_sync ** (2 + beta)
        out = (pl_int / pl0) / self.th_sync[i]
        if _is_one(bce):
            return out
        nb = GK * self.nu_bar[i]
        th_err = bce**4 * jnp.exp(nb * (bce - 1) / T) * (jnp.exp(nb / T) - 1) ** 2 / (jnp.exp(nb * bce / T) - 1) ** 2
        pl_err = bce ** (2 + beta)
        return out * (pl_err / th_err)

    def decorrelation(self, delta, nu0, nu1, nupivot, lform):
        scl_nu = (jnp.log(nu0 / nu1) ** 2) / (np.log(nupivot[0] / nupivot[1]) ** 2)
        scl_ell = {"lin": self.rat, "quad": self.rat**2}.get(lform, 1.0)
        up = delta > 1
        v = jnp.exp(jnp.log(jnp.where(up, 2.0 - delta, delta)) * scl_nu * scl_ell)
        return jnp.where(up, 2.0 - v, v)

    def separable(self, inp) -> bool:
        """True when every pair's coefficients are ℓ-independent: decorrelation off (fixed
        Δ = 1) or ℓ-flat (static, so ``CMBLikes.binned_theory`` can bin the templates)."""
        return all(_need_decorr(self.port._param(inp, d)) is False or lf not in ("lin", "quad")
                   for d, lf in (("delta_dust", self.lform_dust), ("delta_sync", self.lform_sync)))

    def terms(self, inp):
        """ℓ templates (dust, sync, dust×sync power; (n_ℓ,) each) and, per required map pair,
        their coefficients (scalars, or (n_ℓ,) arrays with ℓ-dependent decorrelation; None
        for pairs other than EE/BB)."""
        p = lambda n: self.port._param(inp, n)  # noqa: E731
        Adust, Async = p("BBdust"), p("BBsync")
        alphadust, betadust, Tdust = p("BBalphadust"), p("BBbetadust"), p("BBTdust")
        alphasync, betasync = p("BBalphasync"), p("BBbetasync")
        dustsync_corr = p("BBdustsynccorr")
        EEtoBB_dust, EEtoBB_sync = p("EEtoBB_dust"), p("EEtoBB_sync")
        delta_dust, delta_sync, gamma_corr = p("delta_dust"), p("delta_sync"), p("gamma_corr")
        n = len(self.nu)
        bce = [gamma_corr + p(g) + 1 if g else 1.0 for g in self.gamma]
        fdust = [self.dust_scaling(i, betadust, Tdust, bce[i]) for i in range(n)]
        fsync = [self.sync_scaling(i, betasync, bce[i]) for i in range(n)]
        rat = self.rat
        dustpow = Adust * rat**alphadust
        syncpow = Async * rat**alphasync
        dustsyncpow = dustsync_corr * _sqrt0(Adust * Async) * rat ** ((alphadust + alphasync) / 2)
        need_dust, need_sync = _need_decorr(delta_dust), _need_decorr(delta_sync)
        coefs = []
        for k, (i, j) in enumerate(self.port.pairs):
            tij = self.port.theory_ij[k]
            EE, BB = tij == (1, 1), tij == (2, 2)
            if not (EE or BB):
                coefs.append(None)
                continue
            dust = fdust[i] * fdust[j]
            sync = fsync[i] * fsync[j]
            dustsync = fdust[i] * fsync[j] + fsync[i] * fdust[j]
            if EE:
                dust = dust * EEtoBB_dust
                sync = sync * EEtoBB_sync
                dustsync = dustsync * jnp.sqrt(EEtoBB_sync * EEtoBB_dust)
            if i != j:  # decorrelation (not applied to the dust×sync term, as in BK)
                nu_i, nu_j = self.nu_bar[i] * bce[i], self.nu_bar[j] * bce[j]
                if need_dust is not False:
                    dust = dust * jnp.where(need_dust, self.decorrelation(delta_dust, nu_i, nu_j, self.decorr_dust,
                                                                          self.lform_dust), 1.0)
                if need_sync is not False:
                    sync = sync * jnp.where(need_sync, self.decorrelation(delta_sync, nu_i, nu_j, self.decorr_sync,
                                                                          self.lform_sync), 1.0)
            coefs.append((dust, sync, dustsync))
        return (dustpow, syncpow, dustsyncpow), coefs

    def add(self, CL, inp):
        """``add_foregrounds`` on the (n_pairs, n_ℓ) map spectra."""
        (dp, sp, dsp), coefs = self.terms(inp)
        return jnp.stack([CL[k] if c is None else CL[k] + (c[0] * dp + c[1] * sp + c[2] * dsp)
                          for k, c in enumerate(coefs)])


__all__ = ["CMBLikes", "PR4Lensing"]
