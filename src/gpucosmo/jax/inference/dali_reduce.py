"""Compressed DALI expansion: the likelihoods' linear algebra applied once to the coefficients.

The DALI-expanded theory is an order-p DALI expansion of the theory block about the MAP
(``inference.dali_expansion``): y(d) = Σ_t φ_t(d) T_t with the monomials φ = [1, d_i, d_i d_j (i ≤ j),
…] of degree ≤ p in d = x_theory − x0. Several likelihoods are linear (or quadratic) in y, so
their expensive operations can be applied to the n_terms coefficient arrays T_t once, instead
of to y at every evaluation. The result is exact for the *unclipped* expansion: the
default DALI also clips each output's deviation far from the expansion point
(``mg_bundle._bounded``), the compressed one does not, so it is a different DALI
function — close to it in the posterior bulk (the clipping moves log L by ~0.02 there) — and
must be used consistently (``gpucosmo-dali-ns --compressed-dali``). A second-order log L +
gradient costs ~50 µs per point at batch 4096 on an RTX 3070 (the HiLLiPoP form is no longer the bottleneck: the FS marginalization, low-ℓ TT and
the θ → H0 solve are), with a few kB of state per point — what GPU-native samplers (tempered
SMC: ``mcmc.dali_smc``) need.

* **HiLLiPoP** (``HillipopCompressed``): the binned residual is x = Σ_k a_k(d, ν) b_k over
  K fixed vectors — the data, and per mode m ∈ {TT, EE, TE, ET} and cross-spectrum s the
  expansion terms T_t times −cal_{m,s}(ν) φ_t(d), and the foreground templates times
  −cal_{m,s}(ν) × amplitude × frequency scaling. χ² = xᵀCx = aᵀMa with M = B C Bᵀ, built once
  on the host in float64 (C is HiLLiPoP's float32 inverse covariance, exact in float64;
  each b_k lives in one (mode, cross-frequency) block, so M costs ~1/18 of a dense product).
  K = 60·n_terms + 151: at second order 1831 for ΛCDM (6 theory parameters), 2851 for μ–Σ (8);
  at third order 5191 and 10051. It is
  evaluated centred on a reference a0 (expansion point, reference nuisances):
  χ² = q0 + 2 g·δa + δaᵀ M δa, q0 and g = M a0 in float64, the last term a float32 GEMV
  (``HiLLiPoP.loglike(ref=...)``'s cancellation-safe form).
* **ACT DR6 (+ Planck) lensing** with the CMB-dependent corrections: the bandpowers are linear
  in κκ and the lensed D_ℓ, so only their expansion coefficients are kept.
* **DESI FS**: the model multipoles and the analytically marginalized templates are linear in
  the REPT tables, so the window is applied to the tables' coefficients (per tracer: n_data ×
  19 instead of 3 × 349 × 19).
* **Commander low-ℓ TT and LoLLiPoP**: only ℓ ≤ their lmax of the spectra is kept.
"""

from __future__ import annotations

import hashlib
import time

import jax
import jax.numpy as jnp
import numpy as np

from gpucosmo.jax.inference.dali_expansion import build_dali_fd, build_dali_fwd, monomials, n_terms, expansion_coord

jax.config.update("jax_enable_x64", True)

MODES = ("TT", "EE", "TE", "ET")


# ---------------------------------------------------------------- HiLLiPoP
def hillipop_fg_terms(hil, p: dict) -> dict:
    """Per mode: [(coef (15,), template (15, L) or (L,)), ...] with Σ coef·template equal to
    ``HiLLiPoP._foregrounds_tt`` (TT) / ``HiLLiPoP._dust`` (EE, TE, ET); ``p`` must include
    ``HILLIPOP_FIXED``."""
    from gpucosmo.jax.cmb import pr4_jax as P

    fg = hil.fg

    def dust(mode):
        b = {"TT": ("beta_dustT", "beta_dustT"), "TE": ("beta_dustT", "beta_dustP"),
             "ET": ("beta_dustP", "beta_dustT"), "EE": ("beta_dustP", "beta_dustP")}[mode]
        a = {"TT": ("AdustT", "AdustT"), "TE": ("AdustT", "AdustP"),
             "ET": ("AdustP", "AdustT"), "EE": ("AdustP", "AdustP")}[mode]
        fac = jnp.stack([P._dust_ratio(P.FDUST[f1], P.FDUST[353], p[b[0]])
                         * P._dust_ratio(P.FDUST[f2], P.FDUST[353], p[b[1]]) for f1, f2 in hil.cross])
        return p[a[0]] * p[a[1]] * fac, fg[f"fg_{mode}_dust_model_dlg"]

    ones = jnp.ones(len(hil.cross))
    cib_r = jnp.stack([P._cib_ratio(P.FCIB[f1], P.F0, p["beta_cib"]) * P._cib_ratio(P.FCIB[f2], P.F0, p["beta_cib"])
                       for f1, f2 in hil.cross])
    sx = jnp.stack([P._tsz_ratio(P.FSZ[f2], P.F0) * P._cib_ratio(P.FCIB[f1], P.F0, p["beta_cib"])
                    + P._tsz_ratio(P.FSZ[f1], P.F0) * P._cib_ratio(P.FCIB[f2], P.F0, p["beta_cib"])
                    for f1, f2 in hil.cross])
    rr = jnp.stack([P._radio_ratio(P.FRADIO[f1], P.F0, p["beta_radio"])
                    * P._radio_ratio(P.FRADIO[f2], P.F0, p["beta_radio"]) for f1, f2 in hil.cross])
    tt = [dust("TT"),
          (p["Atsz"] * ones, fg["fg_TT_tsz_dl_sz"]),
          (p["Aksz"] * ones, fg["fg_TT_ksz_dl_ksz"]),
          (p["Acib"] * cib_r, fg["fg_TT_cib_dl_cib"]),
          (-p["xi"] * jnp.sqrt(p["Acib"] * p["Atsz"]) * sx, fg["fg_TT_szxcib_x_tmpl"]),
          (p["Aradio"] * rr, hil.ll2pi),
          (p["Adusty"] * cib_r, hil.ll2pi)]  # beta_dusty = beta_cib (as _foregrounds_tt)
    return {"TT": tt, "EE": [dust("EE")], "TE": [dust("TE")], "ET": [dust("ET")]}


def _mv32(M32, v):
    return jnp.matmul(M32, v.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST).astype(jnp.float64)


@jax.custom_jvp
def _quad32(M32, v):
    """vᵀ M v with a float32 GEMV (full precision, no TF32); derivative 2 (M v)·dv, so the
    reverse pass reuses the forward product instead of a transposed one."""
    return jnp.dot(v, _mv32(M32, v))


@_quad32.defjvp
def _quad32_jvp(primals, tangents):
    M32, v = primals
    _, dv = tangents
    y = _mv32(M32, v)
    return jnp.dot(v, y), 2.0 * jnp.dot(y, dv)


class HillipopCompressed:
    """HiLLiPoP χ² of an unclipped quadratic DALI expansion as a K × K quadratic form.

    ``terms``: {"TT", "EE", "TE": (n_terms, ≥ lmax + 1)} expansion coefficients of D_ℓ^TT/EE/TE
    (μK², ℓ = 0 …); ET uses TE. ``build`` makes M (host, float64); ``reference`` the centring
    arrays passed to ``chi2_centred``.
    """

    def __init__(self, hil, terms: dict):
        from gpucosmo.jax.cmb.pr4_jax import HILLIPOP_FIXED, HILLIPOP_PARAMS

        self.hil = hil
        self.L = L = hil.lmax + 1
        self.terms = {m: np.asarray(terms["TE" if m == "ET" else m], dtype=np.float64)[:, :L] for m in MODES}
        self.nt = self.terms["TT"].shape[0]
        self.fixed = dict(HILLIPOP_FIXED)
        p_any = {**self.fixed, **{k: v[2] for k, v in HILLIPOP_PARAMS.items()}}
        self.templates = {m: [np.asarray(t) for _, t in v] for m, v in hillipop_fg_terms(hil, p_any).items()}
        # residual layout of HiLLiPoP.residual_vector: TT xf 0..5, EE xf 0..5, TE xf 0..5
        self.blocks = []
        off = 0
        for m in ("TT", "EE", "TE"):
            for xf in range(6):
                col, w = (np.asarray(a) for a in hil.binp[(m, xf)])
                self.blocks.append((m, xf, off, col, w))
                off += len(col)
        self.n_x = off
        self.avg = {"TT": np.asarray(hil.avg["TT"]), "EE": np.asarray(hil.avg["EE"]),
                    "TE": np.asarray(hil.avg_te[0]), "ET": np.asarray(hil.avg_te[1])}  # (6, 15, L)
        self.xf_of = np.asarray(hil.xspec2xfreq)
        self.K = 1 + 4 * 15 * self.nt + 15 * sum(len(v) for v in self.templates.values())
        self.M = None

    def _row(self, mode: str, s: int, V: np.ndarray) -> tuple[int, np.ndarray]:
        """(block index, binned residual contributions (n_v, block length)) of the ℓ-vectors
        V (n_v, L) in row (mode, s): only the block of s's cross-frequency is non-zero."""
        bm = "TE" if mode in ("TE", "ET") else mode
        xf = int(self.xf_of[s])
        q = next(i for i, (m, x, *_r) in enumerate(self.blocks) if m == bm and x == xf)
        col, w = self.blocks[q][3], self.blocks[q][4]
        A = self.avg[mode][xf, s, col]
        return q, w[None, :] * A[None, :] * np.atleast_2d(V)[:, col]

    def _basis(self):
        """Data row (dense, length n_x) and, per block, the basis rows living in it."""
        hil = self.hil
        data = np.zeros(self.n_x)
        rows = {q: [] for q in range(len(self.blocks))}  # q -> [(basis indices, values)]
        for m in MODES:
            dm = np.asarray(hil.data[m])
            for s in range(15):
                q, v = self._row(m, s, dm[s])
                off = self.blocks[q][2]
                data[off:off + v.shape[1]] += v[0]
        k = 1
        for m in MODES:
            for s in range(15):
                q, v = self._row(m, s, self.terms[m])
                rows[q].append((np.arange(k, k + self.nt), v))
                k += self.nt
        for m in MODES:
            for tmpl in self.templates[m]:
                for s in range(15):
                    q, v = self._row(m, s, tmpl[s] if tmpl.ndim == 2 else tmpl)
                    rows[q].append((np.asarray([k]), v))
                    k += 1
        assert k == self.K, (k, self.K)
        return data, {q: (np.concatenate([i for i, _ in r]), np.concatenate([v for _, v in r]))
                      for q, r in rows.items()}

    def basis_dense(self) -> np.ndarray:
        """B (K, n_x) as a dense array (tests)."""
        data, rows = self._basis()
        B = np.zeros((self.K, self.n_x))
        B[0] = data
        for q, (idx, V) in rows.items():
            off, n = self.blocks[q][2], len(self.blocks[q][3])
            B[idx, off:off + n] = V
        return B

    def cache_key(self) -> tuple:
        h = hashlib.sha256()
        for m in MODES:
            h.update(np.ascontiguousarray(self.terms[m]).tobytes())
        st = self.hil._invkll_path.stat()
        return (h.hexdigest()[:24], self.nt, self.L, st.st_size, int(st.st_mtime))

    def build(self, *, log=print, use_cache: bool = True, C=None) -> None:
        """M = B C Bᵀ (float64) from C's row slabs (C symmetric: a row slab is the block's
        columns); cached under ``data/cache`` by the coefficients' hash. ``C``: the inverse
        covariance (default: HiLLiPoP's float32 file, memory-mapped)."""
        from gpucosmo.jax.host_table_cache import cached_npz

        def make():
            t0 = time.perf_counter()
            data, rows = self._basis()
            Cm = np.load(self.hil._invkll_path, mmap_mode="r") if C is None else C
            if Cm.shape != (self.n_x, self.n_x):
                raise ValueError(f"inverse covariance {Cm.shape} vs residual length {self.n_x}")
            Y = np.zeros((self.n_x, self.K))  # C Bᵀ
            for q, (_, _, off, col, _) in enumerate(self.blocks):
                slab = np.asarray(Cm[off:off + len(col)], dtype=np.float64)  # (block, n_x) = C[:, block]ᵀ
                idx, V = rows[q]
                Y[:, idx] = slab.T @ V.T
                Y[:, 0] += slab.T @ data[off:off + len(col)]
            M = np.zeros((self.K, self.K))
            M[0] = data @ Y
            for q, (_, _, off, col, _) in enumerate(self.blocks):
                idx, V = rows[q]
                M[idx] = V @ Y[off:off + len(col)]
            log(f"HiLLiPoP compression: K = {self.K} basis vectors (n_terms {self.nt}), M built in "
                f"{time.perf_counter() - t0:.0f} s")
            return {"M": 0.5 * (M + M.T)}

        if C is not None or not use_cache:
            self.M = make()["M"]
        else:
            self.M = cached_npz("hillipop_compressed", self.cache_key(), make, module=__name__)["M"]

    def _cal_fg(self, p: dict):
        """(calibrations (4, 15), foreground coefficients (K − 1 − 60·n_terms,))."""
        p = {**self.fixed, **p}
        cal = jnp.stack([self.hil._cal(p, m)[:, 0] for m in MODES])  # (4, 15)
        terms = hillipop_fg_terms(self.hil, p)
        return cal, jnp.concatenate([-(cal[i] * coef) for i, m in enumerate(MODES) for coef, _ in terms[m]])

    def coefficients(self, phi: jnp.ndarray, p: dict) -> jnp.ndarray:
        """a (K,) for expansion monomials ``phi`` and HiLLiPoP nuisances ``p`` (sampled ones):
        [1, −vec(cal ⊗ φ), foreground coefficients]."""
        cal, fgc = self._cal_fg(p)
        expansion = -(cal[:, :, None] * phi[None, None, :]).reshape(-1)
        return jnp.concatenate([jnp.ones(1, dtype=expansion.dtype), expansion, fgc])

    def chi2_split(self, phi: jnp.ndarray, p: dict, ref: dict):
        """``chi2_centred(coefficients(phi, p), ref)`` (float32 M) without per-point float64
        K-vectors: the offset da = a − a₀ is formed in float32 from float64 differences,
        −vec(Δcal ⊗ φ + cal₀ ⊗ Δφ) (no cancellation of rounded a and a₀), and the linear
        term 2 g·da as a float64 contraction of the (4, 15, n_terms) block of g with φ."""
        cal, fgc = self._cal_fg(p)
        nt = self.nt
        dcal, dphi = cal - ref["cal0"], phi - ref["phi0"]
        da_t = -(dcal.astype(jnp.float32)[:, :, None] * phi.astype(jnp.float32)[None, None, :]
                 + ref["cal0"].astype(jnp.float32)[:, :, None] * dphi.astype(jnp.float32)[None, None, :])
        dfg = fgc - ref["fgc0"]
        da = jnp.concatenate([jnp.zeros(1, jnp.float32), da_t.reshape(-1), dfg.astype(jnp.float32)])
        lin = -(jnp.sum(cal * (ref["gT"] @ phi)) - ref["c0"]) + jnp.dot(ref["g"][1 + 60 * nt:], dfg)
        return ref["q0"] + 2.0 * lin + _quad32(ref["M32"], da)

    def reference(self, phi_ref, p_ref, *, float64: bool = False) -> dict:
        """Centring arrays for ``chi2_centred``: M (float32, or float64 with ``float64``),
        a0, g = M a0, q0 = a0ᵀ M a0."""
        a0 = np.asarray(jax.jit(self.coefficients)(jnp.asarray(phi_ref), p_ref), dtype=np.float64)
        g = self.M @ a0
        out = {"a0": jnp.asarray(a0), "g": jnp.asarray(g), "q0": jnp.asarray(float(a0 @ g))}
        cal0, fgc0 = (np.asarray(v, dtype=np.float64) for v in jax.jit(self._cal_fg)(p_ref))
        gT = g[1:1 + 60 * self.nt].reshape(4, 15, self.nt)
        phi0 = np.asarray(phi_ref, dtype=np.float64)
        out.update(phi0=jnp.asarray(phi0), cal0=jnp.asarray(cal0), fgc0=jnp.asarray(fgc0), gT=jnp.asarray(gT),
                   c0=jnp.asarray(float(np.sum(cal0 * (gT @ phi0)))))
        if float64:
            out["M64"] = jnp.asarray(self.M)
        else:
            out["M32"] = jnp.asarray(self.M, dtype=jnp.float32)
        return out

    @staticmethod
    def chi2_centred(a, ref: dict, M64=None):
        """χ² from the coefficients ``a``; the last term in float64 if ``M64`` is given or
        ``ref`` holds it, else a float32 GEMV."""
        da = a - ref["a0"]
        M64 = ref.get("M64") if M64 is None else M64
        quad = _quad32(ref["M32"], da) if M64 is None else jnp.dot(da, M64 @ da)
        return ref["q0"] + 2.0 * jnp.dot(ref["g"], da) + quad


# ---------------------------------------------------------------- DESI FS on windowed tables
def fs_loglike_windowed(fs_like, inp, nuis: dict, tw: dict, sigma8, fsigma8):
    """``DESIDR1FullShape.loglike`` with the window applied to the tables beforehand:
    ``tw[name]`` (n_data, 19) = window @ pkpoles[..., c, iz] per component c. The bias
    polynomial and the marginalized-template gradients are the official code's
    (``_poles`` on the identity table)."""
    from gpucosmo.jax.fullshape.desi_dr1 import MARG_SCALES, marginalize_linear

    prior_hess = -jnp.diag(jnp.asarray(MARG_SCALES ** -2.0))
    f_all = fsigma8 / sigma8
    eye = jnp.eye(19)
    total = 0.0
    for b in fs_like.bins:
        name = b["name"]
        theo = []
        if b["has_fs"]:
            iz = b["iz"]
            pv, gm = fs_like._poles(eye, sigma8[iz], f_all[iz], *nuis[name], b["shotnoise"], b["fsat"],
                                    b["sigv"])
            theo.append(tw[name] @ pv)
            g_fs = tw[name] @ gm
        bao = fs_like.bao_theory(inp, b["bao_index"], b["bao_params"])
        theo.append(bao)
        diff = jnp.concatenate(theo) - b["flatdata"]
        P = b["precision"]
        ll = -0.5 * diff @ P @ diff
        mlp = 0.0
        if b["has_fs"]:
            G = jnp.concatenate([g_fs, jnp.zeros((bao.shape[0], 4))])
            ll, mlp = marginalize_linear(diff, P, G, prior_hess)
        total = total + ll + mlp
    return total


def fs_whitening(fs_like) -> dict:
    """Per tracer: Lᵀ with P = L Lᵀ (Cholesky of the FS+BAO precision), split into the FS
    columns (applied to the table coefficients), the BAO columns and Lᵀ·data."""
    out = {}
    for b in fs_like.bins:
        P = np.asarray(b["precision"], dtype=np.float64)
        Lt = np.linalg.cholesky(P).T
        n_fs = np.asarray(b["window"]).shape[0] if b["has_fs"] else 0
        out[b["name"]] = dict(Lt_fs=Lt[:, :n_fs], Lt_bao=Lt[:, n_fs:], Lt_d=Lt @ np.asarray(b["flatdata"]))
    return out


def _chol_unrolled(A):
    """Lower Cholesky factor of a small SPD matrix as nested lists of scalars (unrolled:
    elementwise ops that fuse under vmap, instead of batched LAPACK-style kernels), and
    whether every pivot was positive. A non-positive pivot is replaced by 1 so values and
    gradients stay finite (the caller masks the result; √ of a negative pivot would make the
    gradient NaN through the mask)."""
    n = A.shape[-1]
    L = [[None] * n for _ in range(n)]
    ok = jnp.asarray(True)
    for j in range(n):
        piv = A[j, j] - sum((L[j][k] ** 2 for k in range(j)), jnp.zeros((), A.dtype))
        ok = ok & (piv > 0)
        d = jnp.sqrt(jnp.where(piv > 0, piv, 1.0))
        L[j][j] = d
        for i in range(j + 1, n):
            L[i][j] = (A[i, j] - sum((L[i][k] * L[j][k] for k in range(j)), jnp.zeros((), A.dtype))) / d
    return L, ok


def marginalize_whitened(r, G, prior_hess):
    """``desi_dr1.marginalize_linear`` for a whitened residual r = Lᵀ diff and templates
    G = Lᵀ G_raw (P = L Lᵀ): rᵀ P r = |r|², Gᵀ P G = Gᵀ G — the same expressions without the
    n × n products. A = Gᵀ G − prior_hess is SPD (prior + Gram): solve and log det by an
    unrolled Cholesky (on the GPU, batched LU solves and slogdet were ~200 tiny library
    kernels per call for the six tracers); not positive definite in floating point → −inf."""
    A = prior_hess * -1.0 + G.T @ G
    b = -(G.T @ r)  # post_grad
    L, pd = _chol_unrolled(A)
    n = len(L)
    z = []
    for i in range(n):  # L z = b
        z.append((b[i] - sum((L[i][k] * z[k] for k in range(i)), jnp.zeros((), b.dtype))) / L[i][i])
    dx = [None] * n
    for i in reversed(range(n)):  # Lᵀ dx = z
        dx[i] = (z[i] - sum((L[k][i] * dx[k] for k in range(i + 1, n)), jnp.zeros((), b.dtype))) / L[i][i]
    dx = jnp.stack(dx)
    diag = jnp.stack([L[i][i] for i in range(n)])
    res = r + G @ dx
    ll = -0.5 * res @ res - jnp.sum(jnp.log(diag))
    return jnp.where(pd, ll, -jnp.inf), 0.5 * dx @ prior_hess @ dx


def _table_times(tw, v):
    """``tw @ v`` for a table given whole or as (T0, deviation). float32 T0 and deviation:
    one product of their sum (4 % faster than two). float64 T0 with a float32 deviation:
    T0 (shared by every point) @ v is one GEMM over the batch and the deviation @ v a
    float32 batched product — no per-point float64 (n_data × 19) tensors forward or backward."""
    if isinstance(tw, tuple):
        t0, dev = tw
        if t0.dtype == dev.dtype:
            return (t0 + dev) @ v.astype(dev.dtype)
        return t0 @ v + (dev @ v.astype(dev.dtype)).astype(jnp.float64)
    return tw @ v


def fs_loglike_whitened(fs_like, inp, nuis: dict, tw: dict, sigma8, fsigma8, white: dict, *, dtype=jnp.float64):
    """``fs_loglike_windowed`` with the tracers' precision whitening folded into the tables:
    ``tw[name]`` (n_data, 19) = Lᵀ_fs · window · pkpoles[..., c, iz] (``white`` = ``fs_whitening``).

    ``dtype``: precision of the bias polynomial, the table products and the 4 × 4
    marginalization. Whitened, every entry is O(1–100) (|T·p| ≤ 30, Σ|T||p| ≤ 140 for the DR1
    tracers), so float32 rounds the residual by ≲ 1e-5; the BAO/data part of the residual is
    formed in float64. Measured on the MG DALI (1280 posterior and pilot points):
    float32 − float64 FS log L ≤ 1.6e-5, although −post_hess reaches condition 2.4e5 (its
    small eigenvalues are the prior's, in directions the templates barely constrain).
    float32 products run at full precision (``default_matmul_precision("highest")``: no
    TF32 rounding on Ampere+)."""
    with jax.default_matmul_precision("highest"):
        return _fs_loglike_whitened(fs_like, inp, nuis, tw, sigma8, fsigma8, white, dtype)


def _fs_loglike_whitened(fs_like, inp, nuis, tw, sigma8, fsigma8, white, dtype):
    from gpucosmo.jax.fullshape.desi_dr1 import MARG_SCALES

    prior_hess = -jnp.diag(jnp.asarray(MARG_SCALES ** -2.0))
    f_all = fsigma8 / sigma8
    eye = jnp.eye(19, dtype=dtype)
    total = 0.0
    for b in fs_like.bins:
        name, wh = b["name"], white[b["name"]]
        bao = fs_like.bao_theory(inp, b["bao_index"], b["bao_params"])
        r = jnp.asarray(wh["Lt_bao"]) @ bao - jnp.asarray(wh["Lt_d"])
        if b["has_fs"]:
            iz = b["iz"]
            cast = lambda v: jnp.asarray(v).astype(dtype)  # noqa: E731
            pv, gm = fs_like._poles(eye, cast(sigma8[iz]), cast(f_all[iz]), *map(cast, nuis[name]),
                                    *map(cast, (b["shotnoise"], b["fsat"], b["sigv"])))
            r = r.astype(dtype) + _table_times(tw[name], pv).astype(dtype)
            ll, mlp = marginalize_whitened(r, _table_times(tw[name], gm).astype(dtype), prior_hess.astype(dtype))
            ll, mlp = ll.astype(jnp.float64), mlp.astype(jnp.float64)
        else:
            ll, mlp = -0.5 * r @ r, 0.0
        total = total + ll + mlp
    return total


def psd_whitening(A) -> np.ndarray:
    """W with W Wᵀ = (A + Aᵀ)/2 for a positive semi-definite A (e.g. an inverse covariance with an
    analytically marginalized offset, which is singular), so xᵀ A x = |Wᵀx|²."""
    A = np.asarray(A, dtype=np.float64)
    lam, U = np.linalg.eigh(0.5 * (A + A.T))
    if lam.min() < -1e-8 * lam.max():
        raise ValueError(f"matrix not positive semi-definite (eigenvalue {lam.min():.3g})")
    return U * np.sqrt(np.clip(lam, 0.0, None))


# ---------------------------------------------------------------- ACT DR6 lensing (Cobaya port)
def act_port_projectable(L, cmb_terms: dict) -> bool:
    """The ``act_dr6_lenslike`` port (``likelihoods.cobaya.act_lens``) on an expansion with the extended
    spectra (``cmb_x``) to its multipoles: its bandpowers are linear in κκ and the lensed D_ℓ."""
    from gpucosmo.likelihoods.cobaya.act_lens import ACTDR6LensLike

    x = cmb_terms.get("cmb_x")
    return isinstance(L, ACTDR6LensLike) and x is not None and np.shape(x)[-1] >= L.n_ell


def act_port_terms(L, cmb_terms: dict) -> np.ndarray:
    """(n_terms, n_bins) expansion coefficients of the port's binned theory without its offset:
    a_kk C^κκ + Σ_X a_cmb,X D_ℓ^X, with C^κκ the ``cmb_x`` κκ row for ℓ ≥ 1 (the port's
    φφ → κκ conversion, 0 at ℓ = 0) and D_ℓ the TT, EE, BB, TE rows (the port's order)."""
    x = np.asarray(cmb_terms["cmb_x"], dtype=np.float64)
    n = L.n_ell
    kk = x[:, 0, :n] * (np.arange(n) >= 1)[None, :]
    b = kk @ np.asarray(L.lens.a_kk, dtype=np.float64).T
    if L.lens.needs_cmb:
        b = b + np.einsum("xbl,txl->tb", np.asarray(L.lens.a_cmb, dtype=np.float64), x[:, 1:5, :n])
    return b


# ---------------------------------------------------------------- CamSpec
def _cmb_spectra_terms(cmb_terms: dict) -> dict:
    """{"tt", "te", "ee": (n_terms, ℓ)} expansion coefficients of the lensed D_ℓ as CMB ports read them
    (``mg_bundle.shared_block``: ``cmb_x`` rows κκ, TT, EE, BB, TE, else ``dl_tt`` / ``dl_te`` / ``dl_ee``)."""
    if "cmb_x" in cmb_terms:
        x = np.asarray(cmb_terms["cmb_x"], dtype=np.float64)
        return {"tt": x[:, 1], "ee": x[:, 2], "te": x[:, 4]}
    return {s: np.asarray(cmb_terms[f"dl_{s}"], dtype=np.float64) for s in ("tt", "te", "ee")}


class _Inp:  # the parameter view of ``LikelihoodInputs`` that CamSpec's ``cals`` / ``_param`` read
    def __init__(self, params):
        self.params = params


class CamSpecCompressed:
    """CamSpec's high-ℓ χ² on the DALI expansion in the expansion basis; the nuisances stay exact.

    The residual is δ = d − Σ_k c_k v_k over fixed vectors v_k:

    * theory: per calibration group (data segments — TT 100/143/217/143×217, TE, EE — whose
      calibrations differ by fixed factors, e.g. the three TT spectra when cal0 = cal2 = 1), the
      expansion coefficients of the segments' spectra at their multipoles, with coefficients ψ_t / cal;
    * foregrounds: for the power laws amp_i (ℓ/1500)^{n_i}, an orthonormal basis W_i over the prior range
      of n_i (an SVD, residual < ``tol``), with coefficients amp_i β_i(n_i) / cal_i, β = W_iᵀ(ℓ/1500)^{n_i}
      tabulated on the n grid with dβ/dn and interpolated (cubic Hermite, ~1e-10 of the vector).

    ψ are the expansion's monomials in decorrelated coordinates z = A d (AᵀA = CamSpec's Fisher matrix of
    the theory parameters, regularized by the expansion steps): the same polynomial (φ = Bψ, B
    from the exact change of basis, degree by degree, so truncating the order is unchanged), but no
    single column's χ² contribution is large and cancelled by others. Then

        χ² = χ²_ref − 2 h·e + eᵀ M̃ e,   e = (c − c_ref) / s,   M̃ = s (VᵀSV) s,

    centred at a reference point (expansion point, reference nuisances), Jacobi-scaled
    (s_k = (VᵀSV)_kk^{−1/2}), the quadratic term a float32 product (``f64``: float64) with K ≈ (number
    of calibration groups) × n_terms + Σ dim W_i columns instead of the 9915 data points.
    Supported: the power-law foreground model (``use_fg_residual_model`` 0: NPIPE / 2021 defaults)."""

    SPEC = ("tt", "tt", "tt", "tt", "te", "ee")
    FREQ = ("100", "143", "217", "143x217")

    @staticmethod
    def supports(L) -> bool:
        from gpucosmo.likelihoods.cobaya.planck_python import CamSpec

        # (a Cobaya port's own ``theory``, the bridge's background products, is not read by CamSpec)
        return isinstance(L, CamSpec) and not L.fg_2018 and tuple(L.requires) == ("cmb",)

    def __init__(self, L, cmb_terms: dict, bundle, x_ref, d_ref, order: int, steps, *, log=print,
                 tol: float = 1e-12, n_grid: int = 4001, f64: bool = False):
        I = bundle.internals
        self.L, self.idx, self.order = L, I.idx, int(order)
        spec = _cmb_spectra_terms(cmb_terms)
        nt = spec["tt"].shape[0]
        d_ref = np.asarray(d_ref, dtype=np.float64)
        n = d_ref.size
        self.n = n
        if nt != n_terms(n, self.order):
            raise ValueError(f"CMB coefficients have {nt} terms, expected {n_terms(n, self.order)}")
        data = np.asarray(L.data, dtype=np.float64)
        lo, hi = np.asarray(bundle.prior_lo, float), np.asarray(bundle.prior_hi, float)
        x_ref = np.asarray(x_ref, dtype=np.float64)
        M = np.asarray(jax.device_get(L.chi2.M), dtype=np.float64)
        S = 0.5 * (M + M.T) if L.chi2.full else M + M.T - np.diag(L.chi2.diag)
        del M
        t0 = time.perf_counter()
        # data segments and calibration groups (calibration ratios constant over the nuisances)
        rows, off = {}, 0
        for i, ls in L.segments:
            rows[i] = (off, np.asarray(ls))
            off += ls.size
        if off != data.size:
            raise ValueError(f"CamSpec segments cover {off} of {data.size} data points")
        rng = np.random.default_rng(0)
        pidx = [I.idx[p_] for p_ in L.params]
        xs = [x_ref]
        for _ in range(4):
            xx = x_ref.copy()
            xx[pidx] = lo[pidx] + rng.uniform(0.2, 0.8, len(pidx)) * (hi[pidx] - lo[pidx])
            xs.append(xx)
        cal = np.array([[float(c) for c in L.cals(self._params(jnp.asarray(xx)))] for xx in xs])
        self.groups = []
        for i, _ in L.segments:
            for g in self.groups:
                r = cal[:, i] / cal[:, g["lead"]]
                if np.allclose(r, r[0], rtol=1e-12, atol=0):
                    g["members"].append((i, float(r[0])))
                    break
            else:
                self.groups.append(dict(lead=i, members=[(i, 1.0)]))
        Vth = []
        for g in self.groups:  # model of member i = ψ·T_i / cal_i = (ψ·T_i · cal_lead/cal_i) / cal_lead
            v = np.zeros((data.size, nt))
            for i, ratio in g["members"]:
                o, ls = rows[i]
                v[o:o + ls.size] = spec[self.SPEC[i]][:, ls].T / ratio
            Vth.append(v)
        # decorrelated coordinates z = A d from CamSpec's Fisher matrix of the linear terms
        J = sum(v[:, 1:n + 1] for v in Vth) / cal[0, self.groups[0]["lead"]]
        F = J.T @ S @ J + np.diag(1.0 / np.asarray(steps, dtype=np.float64) ** 2)
        A = np.linalg.cholesky(F).T  # AᵀA = F
        Z = rng.standard_normal((3 * nt, n))
        Psi = np.asarray(monomials(jnp.asarray(Z), self.order))
        Phi = np.asarray(monomials(jnp.asarray(np.linalg.solve(A, Z.T).T), self.order))
        Bt, *_ = np.linalg.lstsq(Psi, Phi, rcond=None)  # φ(d) = B ψ(A d), Bᵀ = Bt
        self.basis_change_err = float(np.max(np.abs(Phi - Psi @ Bt)) / np.max(np.abs(Phi)))
        cols = [v @ Bt.T for v in Vth]
        # foreground power laws: orthonormal bases over the prior range of the index
        self.fg = []
        for i, _ in L.segments:
            if i > 3 or not L.fg_powerlaw:
                continue
            f = self.FREQ[i]
            o, ls = rows[i]
            seg = dict(i=i, ls=ls.astype(np.float64), amp=f"amp_{f}", n=f"n_{f}")
            if seg["n"] in I.idx:
                j = I.idx[seg["n"]]
                grid = np.linspace(lo[j], hi[j], n_grid)
            else:
                grid = np.asarray([float(L.fixed[seg["n"]])])
            G = (ls / 1500.0)[:, None] ** grid[None, :]
            U, _, _ = np.linalg.svd(G, full_matrices=False)
            for k in range(1, U.shape[1] + 1):
                W = U[:, :k]
                rel = np.linalg.norm(G - W @ (W.T @ G), axis=0) / np.linalg.norm(G, axis=0)
                if rel.max() < tol:
                    break
            seg.update(W=W, basis_rel_err=float(rel.max()))
            # β(n) = Wᵀ(ℓ/1500)^n tabulated with dβ/dn on the grid (cubic Hermite per point, error ~1e-10)
            lnr = np.log(ls / 1500.0)
            seg.update(n_lo=float(grid[0]), n_step=float(grid[1] - grid[0]) if grid.size > 1 else 1.0,
                       B=(W.T @ G).T, dB=(W.T @ (G * lnr[:, None])).T)
            v = np.zeros((data.size, W.shape[1]))
            v[o:o + ls.size] = W
            cols.append(v)
            self.fg.append(seg)
        V = np.concatenate(cols, axis=1)
        self.K = V.shape[1]
        SV = S @ V
        Q = V.T @ SV
        Q = 0.5 * (Q + Q.T)
        self.A = A
        self.fg_tables = [dict(B=jnp.asarray(seg["B"]), dB=jnp.asarray(seg["dB"])) for seg in self.fg]
        c_ref = np.asarray(self.coefficients(jnp.asarray(self.psi(d_ref)), jnp.asarray(x_ref)))
        dres = data - V @ c_ref
        g = SV.T @ dres
        q0 = float(dres @ S @ dres)
        del S, SV
        sc = 1.0 / np.sqrt(np.clip(np.diag(Q), 1e-300, None))
        Ms = Q * sc[:, None] * sc[None, :]
        self.arrays = dict(A=jnp.asarray(A), c_ref=jnp.asarray(c_ref), sc=jnp.asarray(sc), hs=jnp.asarray(g * sc),
                           q0=jnp.asarray(q0), fg=self.fg_tables)
        self.f64 = bool(f64)
        self.arrays["M64" if self.f64 else "M32"] = jnp.asarray(Ms if self.f64 else Ms.astype(np.float32))
        # check: the compressed residual at the reference is the port's own
        phi_ref = np.asarray(monomials(jnp.asarray(d_ref), self.order))
        delta_port = np.asarray(jax.device_get(L.residual(self._port_inputs(spec, phi_ref, x_ref))))
        self.ref_check = float(np.max(np.abs(delta_port - dres)) / np.max(np.abs(dres)))
        log(f"compressed CamSpec: K = {self.K} ({len(self.groups)} calibration groups of "
            f"{[len(g_['members']) for g_ in self.groups]} segments × {nt} terms + {self.K - len(self.groups) * nt} "
            f"foreground basis vectors, basis error ≤ {max([s_['basis_rel_err'] for s_ in self.fg] or [0.0]):.1e}; "
            f"monomial basis change {self.basis_change_err:.1e}); built in {time.perf_counter() - t0:.1f} s; "
            f"reference residual vs the port {self.ref_check:.1e}")
        if not (self.ref_check < 1e-8 and self.basis_change_err < 1e-9):
            raise ValueError(f"compressed CamSpec: reference residual {self.ref_check:.2e}, basis change "
                             f"{self.basis_change_err:.2e} (relative)")

    def psi(self, d, order: int | None = None):
        """Monomials of the decorrelated coordinates A d (higher orders zeroed beyond ``order``)."""
        ps = monomials(jnp.asarray(self.A) @ jnp.asarray(d), self.order)
        if order is not None and order < self.order:
            ps = jnp.where(jnp.arange(ps.shape[-1]) < n_terms(self.n, order), ps, 0.0)
        return ps

    def _params(self, x):
        return _Inp({n: x[self.idx[n]] for n in self.L.params})

    def _port_inputs(self, spec, phi, x):
        from gpucosmo.likelihoods.base import LikelihoodInputs

        phi = np.asarray(phi, dtype=np.float64)
        cl = {s: jnp.asarray(phi @ spec[s]) for s in ("tt", "te", "ee")}
        return LikelihoodInputs(theory=None, params=self._params(jnp.asarray(x)).params, cosmo=None,
                                shared={"cmb": {"cl": cl}})

    @staticmethod
    def _beta(seg, tab, n_):
        """β(n) = Wᵀ(ℓ/1500)^n by cubic Hermite interpolation of the tabulated β and dβ/dn."""
        B, dB = tab["B"], tab["dB"]
        if B.shape[0] == 1:  # a fixed index
            return B[0]
        h = seg["n_step"]
        u = (n_ - seg["n_lo"]) / h
        j = jnp.clip(jnp.floor(u), 0, B.shape[0] - 2).astype(jnp.int32)
        t = u - j
        t2, t3 = t * t, t * t * t
        return ((2 * t3 - 3 * t2 + 1) * B[j] + (t3 - 2 * t2 + t) * h * dB[j]
                + (-2 * t3 + 3 * t2) * B[j + 1] + (t3 - t2) * h * dB[j + 1])

    def coefficients(self, psi, x, tables=None):
        """c (K,): per calibration group ψ / cal_lead, then per power law amp_i β_i(n_i) / cal_i."""
        inp = self._params(x)
        cals = self.L.cals(inp)
        out = [psi / cals[g["lead"]] for g in self.groups]
        for k, seg in enumerate(self.fg):
            tab = (self.fg_tables if tables is None else tables)[k]
            beta = self._beta(seg, tab, self.L._param(inp, seg["n"]))
            out.append(self.L._param(inp, seg["amp"]) / cals[seg["i"]] * beta)
        return jnp.concatenate(out)

    def loglike(self, d, x, arrays, order: int | None = None):
        """log L at the expansion-coordinate offset ``d`` and parameters ``x``."""
        from gpucosmo.likelihoods.cobaya.planck_python import _sz_prior

        ps = monomials(arrays["A"] @ d, self.order)
        if order is not None and order < self.order:
            ps = jnp.where(jnp.arange(ps.shape[-1]) < n_terms(self.n, order), ps, 0.0)
        c = self.coefficients(ps, x, tables=arrays["fg"])
        e = (c - arrays["c_ref"]) / arrays["sc"]
        quad = jnp.dot(e, arrays["M64"] @ e) if "M64" in arrays else _quad32(arrays["M32"], e)
        ll = -0.5 * (arrays["q0"] - 2.0 * jnp.dot(arrays["hs"], e) + quad)
        if self.L.sz_prior:
            inp = self._params(x)
            ll = ll + _sz_prior(self.L._param(inp, "aksz"), self.L._param(inp, "asz143"))
        return ll


# ---------------------------------------------------------------- the compressed DALI likelihood
class CompressedDALI:
    """DALI log-likelihood of ``bundle`` on the unclipped DALI expansion ``expansion``,
    with every likelihood's linear algebra applied to the coefficients.

    ``loglike(x, data)`` is jittable and differentiable (``data`` = ``self.data``, a pytree of
    device arrays: expansion coefficient matrix, HiLLiPoP centring arrays). ``x_ref``: point
    whose nuisances (and theory offset) centre the HiLLiPoP float32 term (the MAP).
    Supported: Planck PR4 (HiLLiPoP + LoLLiPoP + Commander TT), ACT DR6 lensing, DESI FS+BAO
    with ``fs_tables`` DALI expansions, SN. ``hillipop_f64``: HiLLiPoP's quadratic term as a float64
    GEMV (exact to ~1e-8 in log L; on the GPU the float32 one scatters by 3–7e-3 at
    posterior points, max 0.03–0.06, and ~0.1 in the tails, with a mean offset ≲ 3e-4, which
    shifts log Z by about as much; float64 costs 2–3.6× per log L + gradient on an RTX 3070).

    The reduced outputs are y = T₀ + Σ_{t≥1} φ_t T_t: T₀ (the expansion point) in float64 and
    the deviation as a full-precision float32 product (``expansion_f32``; its rounding, ~1e-7 of
    the deviation, is ≲ 1e-9 of y in the posterior — float64 GEMMs run at 1/64 of the float32
    rate on consumer GPUs). The DESI FS tables carry each tracer's precision whitening, so the
    χ² and the marginalization need no n × n products.

    ``sn``: "expanded" (default) evaluates the supernova χ² on the expansion's distance moduli with
    the covariance folded into the coefficients (χ² = |Wᵀm − Σ_t φ_t Wᵀμ_t|², WWᵀ = C⁻¹); "exact"
    computes the distances from the exact background at every point (float64 integrals and a dense
    float64 product, the GPU-time majority of a DESI + CMB + SN analysis). ``camspec``: "compressed"
    (default) evaluates CamSpec's χ² in the expansion basis (``CamSpecCompressed``); "direct" applies
    its 9915 × 9915 inverse covariance at every point. ``act``: "projected" (default) folds the
    ``act_dr6_lenslike`` port's binning and corrections into the coefficients of its bandpowers;
    "direct" rebuilds the spectra to ℓ 3000 and calls the port. ``trim_shared``: evaluate the
    built-in blocks the remaining likelihoods read (``requires``) only up to the largest multipole they
    declare (``Likelihood.cmb_lmax``; e.g. ℓ ≤ 29 for the Planck low-ℓ ports).
    """

    def __init__(self, bundle, expansion, x_ref, *, log=print, use_cache: bool = True, hillipop_f64: bool = False,
                 expansion_f32: bool = True, background: str = "expansion", background_method: str = "fd",
                 fs_f32: bool = True, sn: str = "expanded", camspec: str = "compressed", camspec_f64: bool = False,
                 act: str = "projected", trim_shared: bool = True):
        cfg = bundle.config
        I = bundle.internals
        if cfg.cmb not in (None, "pr4"):
            raise NotImplementedError("compressed DALI likelihood: Planck PR3 (plik) is not supported")
        if cfg.desi_fs and not cfg.fs_tables:
            raise NotImplementedError("compressed DALI likelihood: DESI FS needs an fs_tables DALI "
                                      "(the one-loop model is not linear in the linear spectra)")
        self.bundle, self.I = bundle, I
        self.n = n = int(expansion.x0.shape[0])
        self.log_idx = tuple(getattr(expansion, "log_idx", ()))  # the expansion coordinates (DALIExpansion.log_idx)
        self.order = int(expansion.order)
        self.nt = nt = n_terms(n, self.order)
        terms = jax.tree_util.tree_map(np.asarray, expansion.coef)
        cols, self.slices = [], {}

        def add(name, arr):
            arr = np.asarray(arr, dtype=np.float64)
            off = sum(c.shape[1] for c in cols)
            cols.append(arr.reshape(nt, -1))
            self.slices[name] = (off, arr.shape[1:])

        x_ref = np.asarray(x_ref, dtype=np.float64)
        d_ref = expansion_coord(x_ref[:n], self.log_idx) - expansion_coord(np.asarray(expansion.x0), self.log_idx)
        self.hil = None
        hil_ref = None
        if cfg.cmb == "pr4":
            tc = terms["cmb"]
            self.L_low = max(int(I.lowl_lmax), int(I.lol.lmax)) + 1
            for k in ("dl_tt", "dl_ee", "dl_te"):
                add(k, tc[k][:, :self.L_low])
            self.hil = HillipopCompressed(I.hil, {"TT": tc["dl_tt"], "EE": tc["dl_ee"], "TE": tc["dl_te"]})
            self.hil.build(log=log, use_cache=use_cache)
            hil_ref = self.hil.reference(monomials(d_ref, self.order), {k: jnp.asarray(x_ref[I.idx[k]]) for k in I.plik_names},
                                         float64=hillipop_f64)
        self.act = None
        if I.lens_ll is not None:
            act = I.act
            if act is not None and act.needs_cmb and "cmb_ext" in terms.get("cmb", {}):
                from gpucosmo.jax.cmb.act_dr6_lens_jax import N_ELL

                ext = terms["cmb"]["cmb_ext"]  # (nt, 5, ≥ N_ELL): κκ, TT, EE, BB, TE
                add("act_b", np.einsum("bl,tl->tb", act.a_kk, ext[:, 0, :N_ELL])
                    + np.einsum("xbl,txl->tb", act.a_cmb, ext[:, 1:, :N_ELL]))
                self.act = act
            else:  # candl / κκ-only configurations: keep the input spectra
                for k in ("cmb_ext", "dl_kk"):
                    if k in terms.get("cmb", {}):
                        add(k, terms["cmb"][k])
        if cfg.desi_fs:
            tf = terms["fs"]
            pk = tf["pkpoles"]  # (nt, 3, nk, 19, nz)
            self.fs_white = fs_whitening(I.fs_like)
            for b in I.fs_like.bins:
                if b["has_fs"]:
                    W = self.fs_white[b["name"]]["Lt_fs"] @ np.asarray(b["window"])  # whitened window
                    tab = pk[..., b["iz"]].reshape(nt, -1, pk.shape[3])
                    add(f"fs:{b['name']}", np.einsum("wq,tqc->twc", W, tab))
            add("sigma8", tf["sigma8"])
            add("fsigma8", tf["fsigma8"])
        # Background read by the compressed FS+BAO likelihood (r_drag, H and D_A at the BAO
        # redshifts) as expanded outputs too: the exact per-point θ → H0 Newton solve and its
        # quadratures were a third of the GPU time. Same design as the theory expansion
        # (``background_method`` "fd": 1σ finite-difference design; "fwd": forward mode).
        if sn not in ("expanded", "exact"):
            raise ValueError(f"sn must be 'expanded' or 'exact', not {sn!r}")
        if camspec not in ("compressed", "direct"):
            raise ValueError(f"camspec must be 'compressed' or 'direct', not {camspec!r}")
        self.sn_mode = None
        if I.sn_like is not None:
            self.sn_mode = sn
            if sn == "expanded":
                if "sn" not in terms:
                    raise ValueError("the DALI expansion has no supernova outputs: rebuild it (gpucosmo-dali-hmc "
                                     "--stage build), or pass sn='exact'")
                Wsn = psd_whitening(I.sn_like.invcov)
                self.sn_wdata = jnp.asarray(np.asarray(I.sn_like.mag, dtype=np.float64) @ Wsn)
                add("snw", np.asarray(terms["sn"]["mu"], dtype=np.float64) @ Wsn)
        self.bg_expansion = bool(cfg.desi_fs and background == "expansion"
                                 and (I.sn_like is None or self.sn_mode == "expanded"))
        if self.bg_expansion:
            for k, v in self._background_terms(bundle, expansion, nt, background_method).items():
                add(f"bg:{k}", v)
        # User likelihoods (``gpucosmo.likelihoods``): a compressible Gaussian's model vector
        # with its whitening folded in (χ² = |Lᵀd − Σ_t φ_t Lᵀm_t|², exact); otherwise every
        # ``theory`` output, reassembled per point for the likelihood's own ``loglike``; plus
        # the full built-in blocks (``requires``) it reads.
        self.extra = tuple(getattr(I, "extra", ()))
        self.ext_tree = {}
        te = terms.get("ext", {})
        nondiff = [L for L in self.extra if not getattr(L, "differentiable", True)]
        if act not in ("projected", "direct"):
            raise ValueError(f"act must be 'projected' or 'direct', not {act!r}")
        self.camspec, self.actp, needs = {}, {}, {}
        for L in self.extra:
            lab = L.label
            if not getattr(L, "differentiable", True):
                continue  # a quadratic proxy (``_fit_proxies``)
            if camspec == "compressed" and CamSpecCompressed.supports(L) and "cmb" in terms:
                steps_u = np.asarray(bundle.dali.steps, dtype=np.float64).copy()
                for i in self.log_idx:  # steps in the expansion coordinates
                    steps_u[i] = steps_u[i] / float(expansion.x0[i])
                self.camspec[lab] = CamSpecCompressed(L, terms["cmb"], bundle, x_ref, np.asarray(d_ref), self.order,
                                                      steps_u, log=log, f64=camspec_f64)
                continue
            if act == "projected" and "cmb" in terms and act_port_projectable(L, terms["cmb"]):
                add(f"actp:{lab}", act_port_terms(L, terms["cmb"]))
                self.actp[lab] = dict(data=jnp.asarray(L.lens.data), offset=jnp.asarray(L.lens.offset),
                                      cinv=jnp.asarray(L.lens.cinv))
                self._check_act_port(L, terms["cmb"], np.asarray(monomials(jnp.asarray(d_ref), self.order)),
                                     x_ref, log)
                continue
            if L.has_theory():
                if lab not in te:
                    raise ValueError(f"the DALI expansion has no outputs for likelihood {lab!r}: build it "
                                     "with this likelihood in the analysis")
                if getattr(L, "compressible", lambda: False)():
                    add(f"extw:{lab}", np.asarray(te[lab]["model"]) @ np.asarray(L.whitening).T)
                else:
                    leaves, treedef = jax.tree_util.tree_flatten(te[lab])
                    for i, leaf in enumerate(leaves):
                        add(f"ext:{lab}:{i}", leaf)
                    self.ext_tree[lab] = (treedef, len(leaves))
            for r in L.requires:
                needs.setdefault(r, []).append(L.cmb_lmax() if (r == "cmb" and trim_shared) else None)
        # the built-in blocks the remaining likelihoods read, to the largest multipole they declare
        self.shared_lmax = {}
        for r, lms in needs.items():
            lmax = None if any(v is None for v in lms) else int(max(lms))
            self.shared_lmax[r] = lmax
            for k, v in terms[r].items():
                v = np.asarray(v)
                if lmax is not None and v.ndim >= 2 and v.shape[-1] > lmax + 1:
                    v = v[..., :lmax + 1]
                add(f"shared:{r}:{k}", v)
        if not cols and not nondiff:
            raise ValueError("compressed DALI likelihood: the analysis has no DALI-expanded outputs")
        if not cols:  # only proxies: a dummy block keeps the layout uniform
            add("_none", np.zeros((nt, 1)))
        T = np.concatenate(cols, axis=1)
        self.n_out = T.shape[1]
        self.expansion_f32 = bool(expansion_f32)
        self.fs_dtype = jnp.float32 if (expansion_f32 and fs_f32) else jnp.float64
        # One coefficient block per reduced output: per-block products keep the backward pass
        # from assembling one (points × all outputs) cotangent (a slow, memory-bound pad).
        dt1 = jnp.float32 if expansion_f32 else jnp.float64
        blocks = {}
        for name, (off, shape) in self.slices.items():
            size = int(np.prod(shape)) if shape else 1
            dt0 = self.fs_dtype if name.startswith("fs:") else jnp.float64
            blocks[name] = {"T0": jnp.asarray(T[0, off:off + size], dtype=dt0),
                            "T1": jnp.asarray(T[1:, off:off + size], dtype=dt1)}
        self.data = {"x0": jnp.asarray(expansion.x0), "blocks": blocks}
        if hil_ref is not None:
            self.data["hil"] = hil_ref
        if self.camspec:
            self.data["camspec"] = {lab: c.arrays for lab, c in self.camspec.items()}
        if self.actp:
            self.data["actp"] = self.actp
        self.proxies = {}
        if nondiff:
            self._fit_proxies(bundle, expansion, x_ref, nondiff, log)
            self.data["proxy"] = {lab: p.arrays for lab, p in self.proxies.items()}
        log(f"compressed DALI likelihood: {self.n_out} reduced outputs × {nt} expansion terms"
            + (f"; HiLLiPoP K = {self.hil.K}" if self.hil is not None else ""))

    @staticmethod
    def _check_act_port(L, cmb_terms, phi_ref, x_ref, log):
        """The projected bandpowers' log L against the port's own at the reference spectra."""
        from gpucosmo.jax.inference.mg_bundle import shared_block
        from gpucosmo.likelihoods.base import LikelihoodInputs

        x = np.asarray(cmb_terms["cmb_x"], dtype=np.float64)
        b = phi_ref @ act_port_terms(L, cmb_terms) + np.asarray(L.lens.offset)
        r = np.asarray(L.lens.data) - b
        ll_proj = float(-0.5 * r @ np.asarray(L.lens.cinv) @ r)
        shared = {"cmb": shared_block("cmb", {"cmb": {"cmb_x": jnp.asarray(np.einsum("t,txl->xl", phi_ref, x))}})}
        ll_port = float(L.loglike(LikelihoodInputs(theory=None, params={}, cosmo=None, shared=shared)))
        log(f"projected {L.label}: {x.shape[0]} terms → {r.size} bandpowers; log L at the reference "
            f"{ll_proj:.6f} (port {ll_port:.6f})")
        if not abs(ll_proj - ll_port) < 1e-6 * max(1.0, abs(ll_port)):
            raise ValueError(f"projected {L.label} differs from the port at the reference: {ll_proj} vs {ll_port}")

    def _fit_proxies(self, bundle, expansion, x_ref, nondiff, log, chunk: int = 128):
        """Quadratic proxies (``cobaya_bridge.nondiff``) of the non-differentiable likelihoods in
        (theory parameters, own parameters), fitted at ``x_ref`` on the DALI-expanded theory (an
        approximation of log L outside the DALI hierarchy, see ``cobaya_bridge.nondiff``)."""
        from gpucosmo.cobaya_bridge.nondiff import fit_quadratic

        I, n = bundle.internals, self.n
        steps = np.asarray(bundle.dali.steps, dtype=np.float64)
        for L in nondiff:
            sel = list(range(n)) + [I.idx[p] for p in L.params]
            pr = [L.params[p] for p in L.params]
            h = np.concatenate([steps, [p.gauss[1] if p.gauss else 0.05 * (p.hi - p.lo) for p in pr]])
            h_max = np.concatenate([2.0 * steps, [0.25 * (p.hi - p.lo) for p in pr]])
            f = jax.jit(jax.vmap(lambda xx, L=L: L.loglike(I.ext_inputs(L, xx, expansion(xx[:n])))))

            def f_batch(X, f=f):
                X = np.asarray(X, dtype=np.float64)
                return np.concatenate([np.asarray(f(jnp.asarray(X[i:i + chunk]))) for i in range(0, len(X), chunk)])

            log(f"compressed DALI likelihood: fitting the quadratic proxy of {L.label} (non-differentiable)")
            self.proxies[L.label] = fit_quadratic(f_batch, x_ref, sel, h, h_max=h_max, log=log)

    def _get_split(self, y, name):
        """(T₀, deviation) of reduced output ``name``; the deviation in the expansion precision."""
        phi, blocks = y
        blk = blocks[name]
        shape = self.slices[name][1]
        with jax.named_scope("expansion"):
            dev = jnp.matmul(phi[1:].astype(blk["T1"].dtype), blk["T1"], precision=jax.lax.Precision.HIGHEST)
        if blk["T1"].dtype == jnp.float64:
            return (blk["T0"] + dev).reshape(shape)
        return blk["T0"].reshape(shape), dev.reshape(shape)

    def _background_terms(self, bundle, expansion, nt, method):
        """expansion terms (n_terms, ...) of r_drag, H(z) and D_A(z) at the BAO redshifts."""
        I = bundle.internals
        n = self.n
        rest = jnp.asarray(np.asarray(bundle.fiducial)[n:])

        def bg(xt):
            c = bundle.cosmo(jnp.concatenate([xt, rest]))
            inp = I.fs_inputs(c, {})
            return {"rdrag": inp.rdrag, "H_bao": inp.H_bao, "DA_bao": inp.DA_bao}

        x0 = np.asarray(expansion.x0)
        if method == "fwd":
            if self.log_idx:
                raise NotImplementedError("forward-mode background terms in log coordinates")
            tb = build_dali_fwd(bg, x0, order=self.order, verbose=False)
        else:
            tb = build_dali_fd(jax.jit(jax.vmap(bg)), x0, np.asarray(bundle.dali.steps), order=self.order,
                                 verbose=False, log_idx=self.log_idx)
        terms = jax.tree_util.tree_map(np.asarray, tb.coef)
        assert all(np.all(np.isfinite(np.asarray(v))) for v in terms.values())
        return terms

    def _get(self, y, name):
        """Reduced output ``name`` = T₀ + φ[1:] · T₁ (``y`` = (φ, data["blocks"]))."""
        phi, blocks = y
        blk = blocks[name]
        with jax.named_scope("expansion"):
            dev = jnp.matmul(phi[1:].astype(blk["T1"].dtype), blk["T1"], precision=jax.lax.Precision.HIGHEST)
            return (blk["T0"] + dev.astype(jnp.float64)).reshape(self.slices[name][1])

    def components(self, x, data, *, exact_hillipop: bool = False, order: int | None = None) -> dict:
        """Per-likelihood DALI log L at ``x`` (``exact_hillipop``: float64 M, validation). ``order``:
        the expansion truncated to a lower order (its higher monomials set to zero)."""
        from gpucosmo.jax.background.cosmology import BackgroundParams

        I, b = self.I, self.bundle
        x = jnp.asarray(x, dtype=jnp.float64)
        dx = expansion_coord(x[:self.n], self.log_idx) - expansion_coord(data["x0"], self.log_idx)
        phi = monomials(dx, self.order)
        if order is not None and order < self.order:
            phi = jnp.where(jnp.arange(self.nt) < n_terms(self.n, order), phi, 0.0)
        y = (phi, data["blocks"])
        parts = {}
        if self.hil is not None:
            pars = {k: x[I.idx[k]] for k in I.plik_names}
            with jax.named_scope("hillipop"):
                if exact_hillipop or "M32" not in data["hil"]:
                    a = self.hil.coefficients(phi, pars)
                    parts["hillipop"] = -0.5 * self.hil.chi2_centred(
                        a, data["hil"], M64=jnp.asarray(self.hil.M) if exact_hillipop else None)
                else:
                    parts["hillipop"] = -0.5 * self.hil.chi2_split(phi, pars, data["hil"])
            dl_tt, dl_ee, dl_te = (self._get(y, k) for k in ("dl_tt", "dl_ee", "dl_te"))
            ell = jnp.arange(self.L_low, dtype=jnp.float64)
            cl_ee = jnp.where(ell >= 2, dl_ee * 2 * jnp.pi / jnp.maximum(ell * (ell + 1), 1.0), 0.0)
            parts["lollipop"] = I.lol.loglike(cl_ee, pars["A_planck"])
            parts["lowl_tt"] = I.lowl(dl_tt, dl_te, dl_ee, pars["A_planck"])
        if I.lens_ll is not None:
            if self.act is not None:
                r = jnp.asarray(self.act.data) - (self._get(y, "act_b") + jnp.asarray(self.act.offset))
                parts["act_lens"] = -0.5 * r @ (jnp.asarray(self.act.cinv) @ r)
            elif "cmb_ext" in self.slices:
                ext = self._get(y, "cmb_ext")
                parts["act_lens"] = I.lens_ll(ext[0], ext[1:])
            else:
                parts["act_lens"] = I.lens_ll(self._get(y, "dl_kk"))
        with jax.named_scope("background"):
            need_c = self.sn_mode == "exact" or (b.config.desi_fs and not self.bg_expansion)
            c = b.cosmo(x) if need_c else None
        if b.config.desi_fs:
            from gpucosmo.jax.fullshape.desi_dr1 import FS_TRACERS

            with jax.named_scope("background"):
                if self.bg_expansion:
                    from types import SimpleNamespace

                    inp = SimpleNamespace(rdrag=self._get(y, "bg:rdrag"), H_bao=self._get(y, "bg:H_bao"),
                                          DA_bao=self._get(y, "bg:DA_bao"))
                else:
                    inp = I.fs_inputs(c, {})
            nuis = {t: (x[I.idx[f"{t}.b1p"]], x[I.idx[f"{t}.b2p"]], x[I.idx[f"{t}.bsp"]]) for t in FS_TRACERS}
            tw = {bb["name"]: self._get_split(y, f"fs:{bb['name']}") for bb in I.fs_like.bins if bb["has_fs"]}
            s8, f8 = self._get(y, "sigma8"), self._get(y, "fsigma8")
            with jax.named_scope("desi_fs"):
                parts["desi_fs"] = fs_loglike_whitened(I.fs_like, inp, nuis, tw, s8, f8, self.fs_white,
                                                       dtype=self.fs_dtype)
        if self.sn_mode == "expanded":
            with jax.named_scope("sn"):
                r = self.sn_wdata - self._get(y, "snw")
                parts["sn"] = -0.5 * r @ r
        elif self.sn_mode == "exact":
            parts["sn"] = I.sn_like(BackgroundParams.build(c["ombh2"], c["omch2"], 100.0 * c["h"], c["w0"],
                                                           c["wa"], ext=c.get("ext")))
        if self.extra:
            from gpucosmo.jax.inference.mg_bundle import shared_block
            from gpucosmo.likelihoods.base import Cosmology, LikelihoodInputs

            # exact per-point background; XLA drops it when no ``loglike`` reads ``cosmo``
            cosmo_ = Cosmology(c if c is not None else b.cosmo(x), b.tables)
            for L in self.extra:
                lab = L.label
                with jax.named_scope(f"ext_{lab}"):
                    if lab in self.camspec:
                        parts[lab] = self.camspec[lab].loglike(dx, x, data["camspec"][lab], order=order)
                        continue
                    if lab in self.actp:
                        a = data["actp"][lab]
                        r = a["data"] - (self._get(y, f"actp:{lab}") + a["offset"])
                        ll = -0.5 * r @ (a["cinv"] @ r)
                        parts[lab] = jnp.where(jnp.isnan(ll), -jnp.inf, ll)
                        continue
                    if lab in self.proxies:
                        parts[lab] = self.proxies[lab](x, data["proxy"][lab])
                        continue
                    if f"extw:{lab}" in self.slices:
                        r = jnp.asarray(np.asarray(L.whitening) @ L.data) - self._get(y, f"extw:{lab}")
                        parts[lab] = -0.5 * r @ r
                        continue
                    th = None
                    if lab in self.ext_tree:
                        treedef, n_leaves = self.ext_tree[lab]
                        th = jax.tree_util.tree_unflatten(treedef, [self._get(y, f"ext:{lab}:{i}")
                                                                    for i in range(n_leaves)])
                    blk = {r: {k.split(":", 2)[2]: self._get(y, k) for k in self.slices
                               if k.startswith(f"shared:{r}:")} for r in L.requires}
                    shared = {r: shared_block(r, blk, getattr(I, "block_static", None)) for r in L.requires}
                    inp = LikelihoodInputs(theory=th, params={n: x[I.idx[n]] for n in L.params}, cosmo=cosmo_,
                                           shared=shared)
                    parts[lab] = jnp.asarray(L.loglike(inp), dtype=jnp.float64)
        return parts

    def loglike(self, x, data, order: int | None = None):
        ll = sum(self.components(x, data, order=order).values(), jnp.asarray(0.0))
        return jnp.where(jnp.isfinite(ll), ll, -jnp.inf)


def direct_components(bundle, tt, x, *, clipped: bool = False) -> dict:
    """The bundle's own likelihood code on the DALI-expanded theory at ``x`` (unclipped unless
    ``clipped``: the default DALI), components merged as in ``CompressedDALI``."""
    th = bundle.dali.theory(tt, x) if clipped else {**tt[0](x[:tt[0].x0.shape[0]]), "aux": tt[1]}
    parts = bundle.log_like_from_theory_fn(x, th, components=True)
    fs = [k for k in parts if k.startswith("fs:") or k == "fs_mlp"]
    if fs:
        parts["desi_fs"] = sum(parts.pop(k) for k in fs)
    return parts


__all__ = ["CamSpecCompressed", "CompressedDALI", "HillipopCompressed", "act_port_projectable",
           "act_port_terms", "psd_whitening", "MODES", "direct_components", "fs_loglike_whitened",
           "fs_loglike_windowed", "fs_whitening", "marginalize_whitened",
           "hillipop_fg_terms"]
