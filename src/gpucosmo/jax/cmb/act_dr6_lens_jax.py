# This file is part of GPUCosmo. Copyright (C) 2025-2026 Leonel Medina-Varela.
# SPDX-License-Identifier: GPL-3.0-or-later
# Derived in part from the following, translated to JAX and modified for GPUCosmo:
# - act_dr6_lenslike (https://github.com/ACTCollaboration/act_dr6_lenslike),
#   Copyright (c) 2023, Members of the Atacama Cosmology Telescope Collaboration.
#   Licence: LICENSES/act_dr6_lenslike.txt.
# See NOTICE.md.
"""ACT DR6 (+ Planck PR4) CMB lensing likelihood in JAX: a port of ``act_dr6_lenslike``.

Ports ``act_dr6_lenslike.load_data`` / ``generic_lnlike`` (v1.2 data; Madhavacheril et al.
2023, Qu et al. 2023; Planck PR4 lensing: Carron, Mirmelstein & Lewis 2022) for the
variants ``act_baseline``, ``act_extended``, ``actplanck_baseline``, ``actplanck_extended``
with ``lens_only`` True (CMB-marginalized covariance, no corrections) or False (the
default with primary CMB: the reconstruction normalization and N1 bias are corrected
to first order in the departure of the true CMB and lensing spectra from the fiducial
ones, separately for ACT and Planck, with the joint covariance).

Every term of the binned theory is linear in the input spectra:

    b = B C^κκ + M_κκ (C^κκ − C^κκ_fid) + Σ_X M_X (C^X − C^X_fid),   X ∈ {TT, EE, BB, TE},
    M_κκ = B dN1_κκ,   M_X = B [diag(C^κκ_fid / A_fid) (−2 dA/dC^X) + dN1_X],

so ``load_act_dr6_lens`` reduces the 2.8 GB of v1.2 products to (n_bins × 3000)
matrices once (cached in ``data/act_dr6_lens/reduced_*.npz``) and the likelihood is a
few small mat-vecs. Inputs are lensed CMB D_ℓ (μK², ℓ(ℓ+1)C_ℓ/2π) and
C_L^κκ = [L(L+1)]² C_L^φφ / 4 (the ``TheorySpectra.dl_kk`` convention), ℓ = 0…2999.
``scripts/validate_act_dr6_lens.py`` checks it against the reference code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gpucosmo.paths import data_dir

jax.config.update("jax_enable_x64", True)

DATA = data_dir("act_dr6_lens")
VERSION = "v1.2"
VARIANTS = ("act_baseline", "act_extended", "actplanck_baseline", "actplanck_extended")
SPECS = ("tt", "ee", "bb", "te")  # order of the dA/dC (norm_correction) matrices
TRIM_LMAX = 2998
N_ELL = TRIM_LMAX + 2  # theory multipoles 0 … 2999 (``standardize`` with lbuffer 2)


def _standardize(ls, cls, trim_lmax=TRIM_LMAX, lbuffer=2, extra_dims="y"):
    """``act_dr6_lenslike.standardize``: place ``cls`` (first multipole ``ls[0]`` ≤ 2) on
    multipoles 0 … trim_lmax + lbuffer − 1, zero-padded; ``x`` dims are not ℓ axes.
    Works on memory-mapped arrays (only the kept block is read)."""
    cstart = int(ls[0])
    if not np.allclose(np.diff(ls), 1.0):
        raise ValueError("Multipoles are not spaced by 1")
    if cstart > 2:
        raise ValueError("Multipoles start at value greater than 2")
    nlen = trim_lmax + lbuffer
    cend = nlen - cstart
    if extra_dims == "xyy":
        out = np.zeros((cls.shape[0], nlen, nlen))
        out[:, cstart:, cstart:] = cls[:, :cend, :cend]
    elif extra_dims == "yy":
        out = np.zeros((nlen, nlen))
        out[cstart:, cstart:] = cls[:cend, :cend]
    elif extra_dims == "xy":
        out = np.zeros((cls.shape[0], nlen))
        out[:, cstart:] = cls[:, :cend]
    elif extra_dims == "y":
        out = np.zeros(nlen)
        out[cstart:] = cls[:cend]
    else:
        raise ValueError(extra_dims)
    return out


def _parse_variant(variant: str):
    v = variant.lower().strip()
    if v not in VARIANTS:
        raise ValueError(f"ACT DR6 lensing variant {variant!r}: ported {VARIANTS}")
    return "_baseline" in v, "actplanck" in v


def _fiducials(ddir: Path):
    """Fiducial lensed CMB and κκ C_ℓ of the likelihood corrections (μK², no ℓ factors)."""
    f_ls, f_tt, f_ee, f_bb, f_te = np.loadtxt(ddir / "like_corrs/cosmo2017_10K_acc3_lensedCls.dat",
                                              unpack=True)
    fac = 2.0 * np.pi / (f_ls * (f_ls + 1.0))
    fid = {s: _standardize(f_ls, c * fac) for s, c in zip(SPECS, (f_tt, f_ee, f_bb, f_te))}
    fd_ls, f_dd = np.loadtxt(ddir / "like_corrs/cosmo2017_10K_acc3_lenspotentialCls.dat",
                             unpack=True, usecols=[0, 5])
    fid["kk"] = _standardize(fd_ls, f_dd * 2.0 * np.pi / 4.0)
    return fid


def _response(ddir: Path, binmat, fid, planck: bool, cmb_corrections: bool, verbose: bool):
    """(M_κκ, {X: M_X}) of one experiment's rows (``binmat``: n_bins × N_ELL)."""
    if planck:
        norm_file = "P18_norm_correction_matrix_Lmin0_Lmax3000.npy"
        fal_file = "PLANCK_n0mv_fiducial_lmin600_lmax3000_Lmin0_Lmax3000.txt"
        n1_file = "N1_planck_der_{}_lmin100_lmax2048.txt"
    else:
        norm_file = "norm_correction_matrix_Lmin0_Lmax4000.npy"
        fal_file = "n0mv_fiducial_lmin600_lmax3000_Lmin0_Lmax4000.txt"
        n1_file = "N1der_{}_lmin600_lmax3000_full.txt"
    fal_ls, fal = np.loadtxt(ddir / "like_corrs" / fal_file)
    fal = _standardize(fal_ls, fal)
    ls = np.arange(N_ELL)
    g = np.ones(N_ELL)
    g[ls >= 2] = 1.0 / fal[ls >= 2]
    if not np.all(np.isfinite(g)):
        raise ValueError(f"{fal_file}: zero fiducial normalization at L ≥ 2")
    bw = binmat * (fid["kk"] * g)[None, :]  # B diag(C^κκ_fid / A_fid)

    def n1(spec):
        if verbose:
            print(f"  act_dr6_lens: {n1_file.format(spec.upper())}", flush=True)
        m = np.loadtxt(ddir / "like_corrs" / n1_file.format(spec.upper()))
        # act_dr6_lenslike standardizes every N1 matrix with the multipoles of the last
        # loaded A_fid file; both start at L = 0, so this is the plain leading block.
        return binmat @ _standardize(fal_ls, m, extra_dims="yy")

    m_kk = n1("kk")
    m_x = {}
    if cmb_corrections:
        cmat = np.load(ddir / "like_corrs" / norm_file, mmap_mode="r")
        for i, s in enumerate(SPECS):
            d_norm = _standardize(np.arange(cmat.shape[1]), cmat[i:i + 1], extra_dims="xyy")[0]
            m_x[s] = -2.0 * (bw @ d_norm) + n1(s)
    return m_kk, m_x


@dataclass(frozen=True)
class ACTDR6Lens:
    """Reduced ACT DR6 (+ Planck PR4) lensing likelihood (numpy float64 arrays).

    ``a_kk`` (n_bins × N_ELL) multiplies C^κκ, ``a_cmb`` (4 × n_bins × N_ELL) the lensed
    TT/EE/BB/TE D_ℓ (2π/ℓ(ℓ+1) folded in), ``offset`` collects the fiducial terms; the
    binned theory is ``a_kk C^κκ + Σ a_cmb D + offset``.
    """

    variant: str
    lens_only: bool
    data: np.ndarray
    cinv: np.ndarray
    a_kk: np.ndarray
    a_cmb: np.ndarray | None
    offset: np.ndarray
    n_act: int  # leading ACT bins of ``data``; Planck bins follow

    @property
    def needs_cmb(self) -> bool:
        return self.a_cmb is not None


def _reduce(variant, lens_only, ddir: Path, *, nsims_act, nsims_planck, apply_hartlap,
            no_like_corrections, no_actlike_cmb_corrections, verbose) -> dict:
    baseline, planck = _parse_variant(variant)
    corrections = not (lens_only or no_like_corrections)
    start, end = (2, -6) if baseline else (2, -3)
    y = np.loadtxt(ddir / "clkk_bandpowers_act.txt")
    n_tot_act = y.size
    data = y[start:end].copy()
    bm = np.loadtxt(ddir / "binning_matrix_act.txt")
    b_act = _standardize(np.arange(bm.shape[1]), bm[start:end, :], extra_dims="xy")
    if lens_only:
        fcov = np.loadtxt(ddir / ("covmat_actplanck_cmbmarg.txt" if planck else "covmat_act_cmbmarg.txt"))
    else:
        fcov = np.loadtxt(ddir / ("covmat_actplanck.txt" if planck else "covmat_act.txt"))
    sel = np.s_[n_tot_act + end:n_tot_act]
    cov = np.delete(np.delete(fcov, sel, 0), sel, 1)
    cov = np.delete(np.delete(cov, np.s_[:start], 0), np.s_[:start], 1)
    blocks = [("act", b_act, False)]
    if planck:
        data = np.append(data, np.loadtxt(ddir / "clkk_bandpowers_planck.txt"))
        bm = np.loadtxt(ddir / "binning_matrix_planck.txt")
        blocks.append(("planck", _standardize(np.arange(bm.shape[1]), bm, extra_dims="xy"), True))
    nbins = data.size
    nsims = min(nsims_act, nsims_planck) if planck else nsims_act
    hartlap = (nsims - nbins - 2.0) / (nsims - 1.0) if apply_hartlap else 1.0
    cinv = np.linalg.inv(cov) * hartlap

    ell = np.arange(N_ELL, dtype=np.float64)
    to_cl = np.where(ell >= 2, 2.0 * np.pi / np.maximum(ell * (ell + 1.0), 1.0), 0.0)
    kk_mask = (ell >= 2).astype(np.float64)  # CAMB's C_ℓ are 0 below ℓ = 2
    a_kk = np.concatenate([b for _, b, _ in blocks]) * kk_mask[None, :]
    a_cmb, offset = None, np.zeros(nbins)
    if corrections:
        fid = _fiducials(ddir)
        rows_kk, rows_x = [], {s: [] for s in SPECS}
        for name, b, is_planck in blocks:
            cmb_corr = is_planck or not no_actlike_cmb_corrections
            m_kk, m_x = _response(ddir, b, fid, is_planck, cmb_corr, verbose)
            rows_kk.append(m_kk)
            for s in SPECS:
                rows_x[s].append(m_x.get(s, np.zeros_like(b)))
        m_kk = np.concatenate(rows_kk)
        m_x = {s: np.concatenate(rows_x[s]) for s in SPECS}
        a_kk = a_kk + m_kk * kk_mask[None, :]
        offset = -m_kk @ fid["kk"] - sum(m_x[s] @ fid[s] for s in SPECS)
        a_cmb = np.stack([m_x[s] * to_cl[None, :] for s in SPECS])
    return dict(data=data, cinv=cinv, a_kk=a_kk, offset=offset,
                n_act_bins=np.int64(b_act.shape[0]),
                **({} if a_cmb is None else {"a_cmb": a_cmb}))


def load_act_dr6_lens(variant: str = "act_baseline", lens_only: bool = False, *,
                      data_dir: str | Path | None = None, nsims_act: float = 792.0,
                      nsims_planck: float = 400.0, apply_hartlap: bool = True,
                      no_like_corrections: bool = False,
                      no_actlike_cmb_corrections: bool = False,
                      cache: bool = True, verbose: bool = True) -> ACTDR6Lens:
    """Reduce (or load the cached reduction of) the v1.2 data for one configuration.

    Arguments mirror the Cobaya ``ACTDR6LensLike`` options (``trim_lmax`` 2998,
    ``version`` v1.2, no mock / Limber / covariance scaling / ACT recalibration).
    """
    root = Path(data_dir) if data_dir else DATA
    ddir = root / VERSION
    if not (ddir / "covmat_act.txt").exists():
        raise FileNotFoundError(f"{ddir}: run scripts/fetch_act_dr6_lens_data.sh")
    if lens_only:
        no_like_corrections = True  # as ACTDR6LensLike.initialize
    tag = (f"{variant}_{'lensonly' if lens_only else 'withcmb'}_s{int(nsims_act)}-{int(nsims_planck)}"
           f"_h{int(apply_hartlap)}_c{int(not no_like_corrections)}{int(not no_actlike_cmb_corrections)}")
    path = root / f"reduced_{VERSION}_{tag}.npz"
    if cache and path.exists():
        z = dict(np.load(path))
    else:
        z = _reduce(variant, lens_only, ddir, nsims_act=nsims_act, nsims_planck=nsims_planck,
                    apply_hartlap=apply_hartlap, no_like_corrections=no_like_corrections,
                    no_actlike_cmb_corrections=no_actlike_cmb_corrections, verbose=verbose)
        if cache:
            np.savez(path, **z)
    return ACTDR6Lens(variant=variant, lens_only=lens_only, data=z["data"], cinv=z["cinv"],
                      a_kk=z["a_kk"], a_cmb=z.get("a_cmb"), offset=z["offset"],
                      n_act=int(z["n_act_bins"]))


def build_act_dr6_lens_loglike(like: ACTDR6Lens):
    """``loglike(dl_kk, dl_cmb=None)`` → ln L.

    ``dl_kk``: C_L^κκ for L = 0 … ≥ 2999; ``dl_cmb``: lensed D_ℓ (4, ≥ 3000) in the order
    TT, EE, BB, TE (μK²), required unless ``like.lens_only``. Pure jax.numpy.
    """
    data, cinv = jnp.asarray(like.data), jnp.asarray(like.cinv)
    a_kk, offset = jnp.asarray(like.a_kk), jnp.asarray(like.offset)
    a_cmb = None if like.a_cmb is None else jnp.asarray(like.a_cmb)

    def loglike(dl_kk, dl_cmb=None):
        b = a_kk @ dl_kk[:N_ELL] + offset
        if a_cmb is not None:
            if dl_cmb is None:
                raise ValueError("this ACT DR6 lensing configuration needs the lensed CMB D_ℓ")
            b = b + jnp.einsum("xbl,xl->b", a_cmb, dl_cmb[:, :N_ELL])
        r = data - b
        return -0.5 * r @ (cinv @ r)

    return loglike


__all__ = ["ACTDR6Lens", "N_ELL", "VARIANTS", "build_act_dr6_lens_loglike", "load_act_dr6_lens"]
