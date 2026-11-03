# This file is part of GPUCosmo. Copyright (C) 2025-2026 Leonel Medina-Varela.
# SPDX-License-Identifier: GPL-3.0-or-later
# Derived in part from the following, translated to JAX and modified for GPUCosmo:
# - HiLLiPoP (https://github.com/planck-npipe/hillipop) and LoLLiPoP
#   (https://github.com/planck-npipe/lollipop), Copyright (C) 2020 Matthieu Tristram.
#   Licence: GPL-3.0-or-later (COPYING).
# See NOTICE.md.
"""Planck PR4 likelihoods in JAX: HiLLiPoP TTTEEE and LoLLiPoP lowlE.

Ports of ``planck_2020_hillipop`` (v4.2, unbinned) and ``planck_2020_lollipop``
(M. Tristram et al.), evaluated on the official classes' own data products
exported by ``scripts/ref_planck_pr4_dump.py`` (``data/planck_pr4``). Units:
D_ℓ in μK² (HiLLiPoP), C_ℓ in μK² (LoLLiPoP).

HiLLiPoP: for each of the 15 map cross-spectra the residual
R = D_data − cal·(D_CMB + Σ foregrounds) is averaged over cross-frequencies with
the data's inverse-variance weights, cut/binned per (mode, cross-frequency) and
contracted with the inverse covariance (29758², kept in float32 as the official
code does: 3.5 GB). Foregrounds (``foregrounds.py``, TT: dust model, tSZ, kSZ,
clustered CIB, tSZ×CIB, radio and dusty point sources; TE/EE: dust model) use
the same templates and spectral scalings.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gpucosmo.paths import data_dir

jax.config.update("jax_enable_x64", True)

DATA = data_dir("planck_pr4")
MAPNAMES = ["100A", "100B", "143A", "143B", "217A", "217B"]
FREQS = [100, 100, 143, 143, 217, 217]
PAIRS = list(combinations(range(6), 2))  # 15 map cross-spectra
T_CMB, K_B, H_PL = 2.72548, 1.3806503e-23, 6.626068e-34
F0 = 143
FSZ = {100: 100.24, 143: 143, 217: 222.044}
FDUST = {100: 105.2, 143: 147.5, 217: 228.1, 353: 370.5}
FCIB = FDUST
FRADIO = {100: 100.4, 143: 140.5, 217: 218.6}

# Sampled HiLLiPoP nuisances (params_*.yaml) with (lo, hi, ref, gauss(μ, σ) | None).
HILLIPOP_PARAMS = {
    "A_planck": (0.9, 1.1, 1.0, (1.0, 0.0025)),
    "cal100A": (0.9, 1.1, 1.0, None), "cal100B": (0.9, 1.1, 1.0, None),
    "cal143B": (0.9, 1.1, 1.0, None), "cal217A": (0.9, 1.1, 1.0, None),
    "cal217B": (0.9, 1.1, 1.0, None),
    "Aradio": (0.0, 150.0, 60.0, None), "Adusty": (0.0, 100.0, 6.0, None),
    "AdustT": (0.5, 1.5, 1.0, (1.0, 0.1)), "beta_dustT": (1.4, 1.6, 1.51, (1.51, 0.01)),
    "Acib": (0.0, 20.0, 4.0, None), "beta_cib": (1.4, 2.1, 1.75, (1.75, 0.06)),
    "Atsz": (0.0, 50.0, 3.0, None), "Aksz": (0.0, 50.0, 1.0, None),
    "xi": (-1.0, 1.0, 0.1, None),
    "AdustP": (0.5, 1.5, 1.0, (1.0, 0.1)), "beta_dustP": (1.4, 1.8, 1.59, (1.59, 0.02)),
}
HILLIPOP_FIXED = {"cal143A": 1.0, "pe100A": 1.0, "pe100B": 1.0, "pe143A": 1.0, "pe143B": 1.0,
                  "pe217A": 0.975, "pe217B": 0.975, "beta_radio": -0.8}


def _quad_form_sym(C, x):
    """xᵀCx for the symmetric float32 ``C``, with the derivative 2(Cx)·dx.

    Plain autodiff of ``C @ x`` transposes the matmul and upcast the 29758² matrix to
    float64 (6.6 GiB) in the reverse pass, which does not fit an 8 GB GPU next to C.
    The custom JVP reuses the forward product, so gradients cost no extra C-sized buffer.
    On an 8 GB GPU also run with ``XLA_FLAGS="--xla_gpu_autotune_level=0
    --xla_gpu_enable_triton_gemm=false"``: Triton GEMMs materialize a copy of C and the
    autotuners allocate C-sized scratch operands.
    """

    def cx(x):  # HIGHEST: GPUs default to TF32 for float32 matmuls (log L error ~0.4)
        return jnp.matmul(C, x.astype(jnp.float32),
                          precision=jax.lax.Precision.HIGHEST).astype(jnp.float64)

    @jax.custom_jvp
    def q(x):
        return jnp.dot(x, cx(x))

    @q.defjvp
    def q_jvp(primals, tangents):
        (x,), (dx,) = primals, tangents
        y = cx(x)
        return jnp.dot(x, y), 2.0 * jnp.dot(y, dx)

    return q(x)


def _f_tsz(f):
    xx = H_PL * f * 1e9 / (K_B * T_CMB)
    return xx / jnp.tanh(xx / 2.0) - 4


def _f_planck(f, T):
    nu = f * 1e9
    return nu**3 / (jnp.exp(H_PL * nu / (K_B * T)) - 1.0)


def _dbdt(f):
    nu = f * 1e9
    xx = H_PL * nu / (K_B * T_CMB)
    return nu**4 * jnp.exp(xx) / (jnp.exp(xx) - 1.0) ** 2


def _tsz_ratio(f, f0):
    return _f_tsz(f) / _f_tsz(f0)


def _cib_ratio(f, f0, beta, T=25.0):
    return (f / f0) ** beta * (_f_planck(f, T) / _f_planck(f0, T)) / (_dbdt(f) / _dbdt(f0))


def _dust_ratio(f, f0, beta, T=19.6):
    return (f / f0) ** beta * (_f_planck(f, T) / _f_planck(f0, T)) / (_dbdt(f) / _dbdt(f0))


def _radio_ratio(f, f0, beta):
    return (f / f0) ** beta / (_dbdt(f) / _dbdt(f0))


class HiLLiPoP:
    """JAX HiLLiPoP TTTEEE (``loglike(dl_tt, dl_ee, dl_te, pars)``, D_ℓ in μK², ℓ = 0..)."""

    def __init__(self, data_dir: Path = DATA):
        s = np.load(data_dir / "hillipop_TTTEEE_small.npz")
        self.lmax = int(s["lmax"])
        self.xspec2xfreq = np.asarray(s["xspec2xfreq"])
        self.data = {m: jnp.asarray(s[f"dldata_{m}"]) for m in ("TT", "EE", "TE", "ET")}
        self.w = {m: np.asarray(s[f"dlweight_{m}"]) for m in ("TT", "EE", "TE", "ET")}
        # The binning matrices (bins × ℓ, 422 MiB dense in total) have one nonzero per row:
        # bin i is w_i·R[ℓ_i]. Stored as (ℓ_i, w_i) the product is a gather, identical to
        # the dense one for finite residuals.
        self.binp = {}
        for m in ("TT", "EE", "TE"):
            for xf in range(6):
                pm = np.asarray(s[f"binp_{m}_{xf}"])
                nz = pm != 0
                if not np.all(nz.sum(axis=1) == 1):
                    raise ValueError(f"binp_{m}_{xf}: expected one nonzero per row")
                col = np.argmax(nz, axis=1)
                self.binp[(m, xf)] = (jnp.asarray(col), jnp.asarray(pm[np.arange(pm.shape[0]), col]))
        # cross-frequency averaging operators: Rl_xf = Σ_xs W[xf, xs, ℓ] R_xs
        self.avg = {}
        for m in ("TT", "EE"):
            self.avg[m] = jnp.asarray(self._avg_op(self.w[m]))
        w_te = self.w["TE"] + 0.0
        self.avg_te = jnp.asarray(self._avg_op_pair(self.w["TE"], self.w["ET"]))
        del w_te
        self.fg = {k: jnp.asarray(s[k]) for k in s.files if k.startswith("fg_") and
                   not k.endswith("_present")}
        self.ll2pi = jnp.asarray(s["fg_TT_ps_radio_ll2pi"])
        self._invkll_path = data_dir / "hillipop_TTTEEE_invkll_f32.npy"
        self._invkll = None
        self.cross = [(FREQS[a], FREQS[b]) for a, b in PAIRS]

    @property
    def invkll(self):
        """The 3.3 GB float32 inverse covariance on the default device, loaded on first use
        (compressed-dali samplers never need it)."""
        if self._invkll is None:
            mm = np.load(self._invkll_path, mmap_mode="r")
            # Direct transfer: ``jnp.asarray`` stages the 3.3 GB matrix through a second
            # device buffer, which exceeds the allocator limit on 8 GB GPUs (the array is
            # then left in an error state and every consumer fails with a 3.3 GiB OOM).
            self._invkll = jax.device_put(mm)  # float32 (as the official likelihood)
        return self._invkll

    def _avg_op(self, w):
        W = np.zeros((6, 15, self.lmax + 1))
        tot = np.zeros((6, self.lmax + 1))
        for xs in range(15):
            tot[self.xspec2xfreq[xs]] += w[xs]
        tot[tot == 0] = np.inf
        for xs in range(15):
            W[self.xspec2xfreq[xs], xs] = w[xs] / tot[self.xspec2xfreq[xs]]
        return W

    def _avg_op_pair(self, w1, w2):
        """TE and ET residuals averaged together: (Σ w_TE R_TE + Σ w_ET R_ET)/(Σ w_TE + Σ w_ET)."""
        W = np.zeros((2, 6, 15, self.lmax + 1))
        tot = np.zeros((6, self.lmax + 1))
        for xs in range(15):
            tot[self.xspec2xfreq[xs]] += w1[xs] + w2[xs]
        tot[tot == 0] = np.inf
        for xs in range(15):
            W[0, self.xspec2xfreq[xs], xs] = w1[xs] / tot[self.xspec2xfreq[xs]]
            W[1, self.xspec2xfreq[xs], xs] = w2[xs] / tot[self.xspec2xfreq[xs]]
        return W

    def _cal(self, p, mode):
        out = []
        for a, b in PAIRS:
            m1, m2 = MAPNAMES[a], MAPNAMES[b]
            c1, c2 = p[f"cal{m1}"], p[f"cal{m2}"]
            if mode == "EE":
                c1, c2 = c1 * p[f"pe{m1}"], c2 * p[f"pe{m2}"]
            elif mode == "TE":
                c2 = c2 * p[f"pe{m2}"]
            elif mode == "ET":
                c1 = c1 * p[f"pe{m1}"]
            out.append(c1 * c2 / p["A_planck"] ** 2)
        return jnp.stack(out)[:, None]

    def _dust(self, p, mode):
        b = {"TT": ("beta_dustT", "beta_dustT"), "TE": ("beta_dustT", "beta_dustP"),
             "ET": ("beta_dustP", "beta_dustT"), "EE": ("beta_dustP", "beta_dustP")}[mode]
        a = {"TT": ("AdustT", "AdustT"), "TE": ("AdustT", "AdustP"),
             "ET": ("AdustP", "AdustT"), "EE": ("AdustP", "AdustP")}[mode]
        dlg = self.fg[f"fg_{mode}_dust_model_dlg"]
        fac = jnp.stack([_dust_ratio(FDUST[f1], FDUST[353], p[b[0]])
                         * _dust_ratio(FDUST[f2], FDUST[353], p[b[1]]) for f1, f2 in self.cross])
        return p[a[0]] * p[a[1]] * dlg * fac[:, None]

    def _foregrounds_tt(self, p):
        fg = self._dust(p, "TT")
        fg = fg + p["Atsz"] * self.fg["fg_TT_tsz_dl_sz"]
        fg = fg + p["Aksz"] * self.fg["fg_TT_ksz_dl_ksz"]
        cib_r = jnp.stack([_cib_ratio(FCIB[f1], F0, p["beta_cib"]) * _cib_ratio(FCIB[f2], F0, p["beta_cib"])
                           for f1, f2 in self.cross])
        fg = fg + p["Acib"] * self.fg["fg_TT_cib_dl_cib"][None, :] * cib_r[:, None]
        sx = jnp.stack([_tsz_ratio(FSZ[f2], F0) * _cib_ratio(FCIB[f1], F0, p["beta_cib"])
                        + _tsz_ratio(FSZ[f1], F0) * _cib_ratio(FCIB[f2], F0, p["beta_cib"])
                        for f1, f2 in self.cross])
        fg = fg - p["xi"] * jnp.sqrt(p["Acib"] * p["Atsz"]) * self.fg["fg_TT_szxcib_x_tmpl"][None, :] \
            * sx[:, None]
        rr = jnp.stack([_radio_ratio(FRADIO[f1], F0, p["beta_radio"])
                        * _radio_ratio(FRADIO[f2], F0, p["beta_radio"]) for f1, f2 in self.cross])
        fg = fg + p["Aradio"] * self.ll2pi[None, :] * rr[:, None]
        dr = jnp.stack([_cib_ratio(FCIB[f1], F0, p["beta_cib"]) * _cib_ratio(FCIB[f2], F0, p["beta_cib"])
                        for f1, f2 in self.cross])  # beta_dusty = beta_cib
        fg = fg + p["Adusty"] * self.ll2pi[None, :] * dr[:, None]
        return fg

    def residual_vector(self, dl_tt, dl_ee, dl_te, p):
        p = {**HILLIPOP_FIXED, **p}
        L = self.lmax + 1
        th = {"TT": dl_tt[:L], "EE": dl_ee[:L], "TE": dl_te[:L], "ET": dl_te[:L]}
        fgs = {"TT": self._foregrounds_tt(p), "EE": self._dust(p, "EE"),
               "TE": self._dust(p, "TE"), "ET": self._dust(p, "ET")}
        R = {m: self.data[m] - self._cal(p, m) * (th[m][None, :] + fgs[m])
             for m in ("TT", "EE", "TE", "ET")}
        xl = []
        for m in ("TT", "EE"):
            Rl = jnp.einsum("fsl,sl->fl", self.avg[m], R[m])
            for xf in range(6):
                col, w = self.binp[(m, xf)]
                xl.append(w * Rl[xf, col])
        Rl = jnp.einsum("fsl,sl->fl", self.avg_te[0], R["TE"]) \
            + jnp.einsum("fsl,sl->fl", self.avg_te[1], R["ET"])
        for xf in range(6):
            col, w = self.binp[("TE", xf)]
            xl.append(w * Rl[xf, col])
        return jnp.concatenate(xl)

    def reference(self, dl_tt, dl_ee, dl_te, p) -> dict:
        """Centring reference for ``loglike(..., ref=...)``: the residual x0 at the given
        spectra and nuisances, C x0 and x0ᵀ C x0, computed on the host in float64."""
        x0 = np.asarray(jax.jit(self.residual_vector)(dl_tt, dl_ee, dl_te, p), dtype=np.float64)
        mm = np.load(self._invkll_path, mmap_mode="r")
        cx0 = np.zeros_like(x0)
        for a in range(0, mm.shape[0], 2048):  # C is symmetric: C x0 = Σ_b C_bᵀ x0_b
            cx0 += np.asarray(mm[a:a + 2048], dtype=np.float64).T @ x0[a:a + 2048]
        return {"x0": jnp.asarray(x0), "cx0": jnp.asarray(cx0), "q0": jnp.asarray(float(x0 @ cx0))}

    def loglike(self, dl_tt, dl_ee, dl_te, p, invkll=None, ref=None):
        """``invkll``: pass the matrix as a (jit) argument; closing over it embeds
        3.5 GB of constants in every compiled program.

        ``ref`` (``reference``): χ² = x0ᵀCx0 + 2 (C x0)·δx + δxᵀ C δx with δx = x − x0. The
        float32 product with C then only sees δx: χ² ≈ 3·10⁴ is a ~200× cancellation of
        terms (Σ|x||C||x| ≈ 6·10⁶), so float32 accumulation of xᵀCx was off by
        0.005–0.03 in log L depending on the summation order (one point vs a batched
        GEMM, XLA's fusion), which entered the DALI expansion/exact weights as noise and
        moved log Z by ~0.01. The reference terms are float64 (``reference``)."""
        x = self.residual_vector(dl_tt, dl_ee, dl_te, p)
        C = self.invkll if invkll is None else invkll
        if ref is None:
            return -0.5 * _quad_form_sym(C, x)
        dx = x - ref["x0"]
        return -0.5 * (ref["q0"] + 2.0 * jnp.dot(ref["cx0"], dx) + _quad_form_sym(C, dx))


class LoLLiPoPlowlE:
    """JAX LoLLiPoP lowlE (offset Hamimeche–Lewis); ``loglike(cl_ee, A_planck)``, C_ℓ in μK²."""

    def __init__(self, data_dir: Path = DATA):
        s = np.load(data_dir / "lollipop_lowlE.npz")
        self.binop = jnp.asarray(s["binop"])
        self.lmax = int(s["lmax"])
        self.cldata = jnp.asarray(s["cldata"][0])
        self.clfid = jnp.asarray(s["clfid"][0])
        self.cloff = jnp.asarray(s["cloff"][0])
        self.invcov = jnp.asarray(s["invclcov"])
        self.nsim = float(s["Nsim"])
        self.marg = bool(s["marg"])

    def loglike(self, cl_ee, a_planck):
        clth = self.binop @ cl_ee[: self.lmax + 1]
        x = (self.cldata * a_planck**2 + self.cloff) / (clth + self.cloff)
        ax = jnp.abs(x)
        g = jnp.sign(x) * jnp.sign(ax - 1) * jnp.sqrt(2.0 * (ax - jnp.log(ax) - 1))
        sf = jnp.sqrt(self.clfid + self.cloff)
        X = sf * g * sf
        q = X @ self.invcov @ X
        chi2 = self.nsim * jnp.log(1 + q / (self.nsim - 1)) if self.marg else q
        return -0.5 * chi2


@lru_cache(maxsize=1)
def load_pr4(data_dir: str | None = None):
    d = Path(data_dir) if data_dir else DATA
    return HiLLiPoP(d), LoLLiPoPlowlE(d)


__all__ = ["HILLIPOP_FIXED", "HILLIPOP_PARAMS", "HiLLiPoP", "LoLLiPoPlowlE", "load_pr4"]
