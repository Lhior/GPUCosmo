#!/usr/bin/env python3
"""
Neutrino corrections to the recombination redshifts of the JAX background.

``build_background_tables.py`` fits z_star(ombh2, omch2) and z_drag(ombh2, omch2) to CAMB at
the default neutrinos (one 0.06 eV eigenstate, N_eff = 3.044). With Σm_ν, N_eff or a massive
sterile neutrino sampled, the expansion rate around recombination and the BBN helium fraction
move them by up to ~1e-3 (5e-4 for Σm_ν = 1 eV in one state, 4e-4 per 0.5 in N_eff). This
script tabulates, per neutrino model (``massive_nu.NeutrinoModel``), the correction

    Δz(ombh2, omch2, x3, x4) = z_CAMB − z_base(ombh2, omch2)

as a 4-D Chebyshev tensor on first-kind nodes, (x3, x4) = (mnu, nnu), or (nnu, meffsterile)
for a sterile model (Σm_ν fixed at 0.06 eV in one state, as the Planck sterile chains).
CAMB caps the massive states' share of N_eff at its standard value (num_massive·3.044/3), so
z has a kink there in N_eff: the N_eff axis is split at that point into separately fitted
pieces (``x4_breaks``).
CAMB values are medians over the same H0 grid as the base table (z* depends on H0 only
through RECFAST's O(1e-6) numerical jitter). Curvature moves z*, z_drag by < 3e-6 for
|Ω_k| ≤ 0.1 and is not tabulated.

    python scripts/build_nu_background_tables.py --model degenerate1 [--workers 12]
    python scripts/build_nu_background_tables.py --model all

Writes src/gpucosmo/_data/nu_corrections_<model>.npz (shipped with the package; ``--out-dir``
to write elsewhere, e.g. next to the base tables, which takes precedence) and prints the fit
residuals and a held-out check at random interior points.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from build_background_tables import (  # noqa: E402
    BBN_PREDICTOR,
    FID_H0,
    OMBH2_RANGE,
    OMCH2_RANGE,
    TCMB_K,
    ZSTAR_H0_GRID,
    cheb_nodes,
)

DEFAULT_BASE = REPO_ROOT / "src" / "gpucosmo" / "_data" / "w0wa_background_tables_v2.npz"

# Model tag -> (hierarchy, num_massive, sterile, axes, ranges of the two neutrino axes)
MNU_MAX = 1.5
NNU_RANGE = (1.0, 6.0)
MODELS = {
    "degenerate1": ("degenerate", 1, False, ("mnu", "nnu"), ((0.0, MNU_MAX), NNU_RANGE)),
    "degenerate3": ("degenerate", 3, False, ("mnu", "nnu"), ((0.0, MNU_MAX), NNU_RANGE)),
    "normal": ("normal", 3, False, ("mnu", "nnu"), ((0.0602, MNU_MAX), NNU_RANGE)),
    "inverted": ("inverted", 3, False, ("mnu", "nnu"), ((0.1007, MNU_MAX), NNU_RANGE)),
    "degenerate1_sterile": ("degenerate", 1, True, ("nnu", "meffsterile"), ((3.044, 6.0), (0.0, 3.0))),
}
STERILE_MNU = 0.06
# Physically relevant (ω_b, ω_c) for the held-out accuracy report (≥ 10σ beyond any posterior).
PHYS_OMBH2 = (0.019, 0.026)
PHYS_OMCH2 = (0.08, 0.16)
# Nodes per axis (ombh2, omch2, x3, x4): Δz is smooth and ≲ 1e-3 of z, so these leave fit
# errors ≪ RECFAST's jitter (checked on held-out points below).
N_NODES = (14, 14, 16, 12)
# Least squares below the node counts: CAMB's z_drag jitters by ~3e-6 (isolated slope jumps from its
# root-finding, e.g. near ω_c ≈ 0.071), which an interpolant would propagate (2e-5 between nodes);
# z* is smoother. The kink of CAMB's N_eff sharing is a piece boundary (``x4_breaks``).
FIT_DEGREES = {"zstar": (11, 11, 13, 9), "zdrag": (8, 8, 10, 8)}


def _camb_z(task):
    """Median over ZSTAR_H0_GRID of CAMB's (zstar, zdrag) at one node."""
    from gpucosmo.jax.cpu_threads import configure_worker_threads

    configure_worker_threads()
    import camb

    hierarchy, num_massive, sterile, ombh2, omch2, vals = task
    zs, zd = [], []
    for h0 in ZSTAR_H0_GRID:
        kw = dict(H0=h0, ombh2=ombh2, omch2=omch2, TCMB=TCMB_K, tau=0.054, As=2.1e-9, ns=0.965,
                  neutrino_hierarchy=hierarchy, num_massive_neutrinos=num_massive, bbn_predictor=BBN_PREDICTOR)
        if sterile:
            kw.update(mnu=STERILE_MNU, nnu=vals["nnu"], meffsterile=vals["meffsterile"])
        else:
            kw.update(mnu=vals["mnu"], nnu=vals["nnu"])
        try:
            d = camb.get_background(camb.set_params(**kw)).get_derived_params()
        except Exception:  # noqa: BLE001 (unphysical corner at this H0: use the others)
            continue
        zs.append(float(d["zstar"]))
        zd.append(float(d["zdrag"]))
    if not zs:
        raise RuntimeError(f"CAMB failed at every H0 for {task}")
    return float(np.median(zs)), float(np.median(zd))


def _unit(v, rng):
    return (2.0 * np.asarray(v) - (rng[0] + rng[1])) / (rng[1] - rng[0])


def fit_cheb_nd(values: np.ndarray, nodes: list[np.ndarray], ranges: list[tuple],
                degrees: tuple[int, ...]) -> np.ndarray:
    """Least-squares Chebyshev tensor of ``degrees`` on a tensor grid. The design matrix is a
    Kronecker product, so its pseudo-inverse is the product of the 1-D ones along each axis."""
    from numpy.polynomial import chebyshev as Cb

    coef = values
    for ax, (x, rng, d) in enumerate(zip(nodes, ranges, degrees)):
        V = Cb.chebvander(_unit(x, rng), d)
        coef = np.moveaxis(np.tensordot(np.linalg.pinv(V), np.moveaxis(coef, ax, 0), axes=1), 0, ax)
    return coef


def eval_points(coef: np.ndarray, pts: np.ndarray, ranges: list[tuple]) -> np.ndarray:
    """Evaluate at points ``pts`` (n, 4)."""
    from numpy.polynomial import chebyshev as Cb

    Ts = [Cb.chebvander(_unit(pts[:, i], ranges[i]), coef.shape[i] - 1) for i in range(4)]
    return np.einsum("ni,nj,nk,nl,ijkl->n", *Ts, coef)


def base_z(base, ombh2, omch2):
    from numpy.polynomial import chebyshev as Cb

    xu, yu = _unit(ombh2, OMBH2_RANGE), _unit(omch2, OMCH2_RANGE)
    zs = Cb.chebval2d(xu, yu, base["zstar_cheb"])
    zd = Cb.chebval2d(xu, yu, base["zdrag_cheb"])
    return zs, zd


def x4_breaks(hierarchy: str, num_massive: int, sterile: bool, r4: tuple) -> list[float]:
    """Kinks of z in the x4 axis inside ``r4``: N_eff = the massive states' standard share."""
    if sterile:
        return []
    std = 3.044
    b = num_massive * std / 3.0 if hierarchy == "degenerate" else std
    return [b] if r4[0] < b < r4[1] else []


def build(tag: str, base_path: Path, workers: int, n_check: int, out_dir: Path) -> Path:
    from gpucosmo.jax.background.cosmology import nu_corrections_name
    from gpucosmo.jax.eb.massive_nu import NeutrinoModel

    hierarchy, num_massive, sterile, axes, (r3, r4) = MODELS[tag]
    model = NeutrinoModel(hierarchy, num_massive, sterile=sterile)
    breaks = x4_breaks(hierarchy, num_massive, sterile, r4)
    edges = [r4[0], *breaks, r4[1]]
    pieces = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    piece_ranges = [[OMBH2_RANGE, OMCH2_RANGE, r3, pr] for pr in pieces]
    grids = []
    for ranges in piece_ranges:
        nodes = [cheb_nodes(*rng, n) for rng, n in zip(ranges, N_NODES)]
        grids.append((nodes, np.stack(np.meshgrid(*nodes, indexing="ij"), axis=-1).reshape(-1, 4)))
    rng = np.random.default_rng(12345)
    full = [OMBH2_RANGE, OMCH2_RANGE, r3, r4]
    lo = np.array([r[0] for r in full]); hi = np.array([r[1] for r in full])
    pad = 0.02 * (hi - lo)
    check = lo + pad + rng.random((n_check, 4)) * (hi - lo - 2 * pad)
    # half of the checks in the physically relevant (ω_b, ω_c) region
    n_phys = n_check // 2
    check[:n_phys, 0] = rng.uniform(*PHYS_OMBH2, n_phys)
    check[:n_phys, 1] = rng.uniform(*PHYS_OMCH2, n_phys)
    mk = lambda pts: [(hierarchy, num_massive, sterile, ob, oc, {axes[0]: x3, axes[1]: x4})  # noqa: E731
                      for ob, oc, x3, x4 in pts]
    tasks = sum((mk(g) for _, g in grids), []) + mk(check)
    print(f"[{tag}] {model}: {len(pieces)} piece(s) in {axes[1]} (breaks {breaks}), "
          f"{sum(len(g) for _, g in grids)} nodes + {n_check} checks × {len(ZSTAR_H0_GRID)} H0 "
          f"with {workers} workers", flush=True)
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        out = np.array(list(pool.map(_camb_z, tasks, chunksize=32)))
    base = np.load(base_path)
    shape = tuple(N_NODES)
    cs_all, cd_all, start = [], [], 0
    for (nodes, grid), ranges in zip(grids, piece_ranges):
        z_nodes = out[start: start + len(grid)]
        start += len(grid)
        bzs, bzd = base_z(base, grid[:, 0], grid[:, 1])
        cs_all.append(fit_cheb_nd((z_nodes[:, 0] - bzs).reshape(shape), nodes, ranges, FIT_DEGREES["zstar"]))
        cd_all.append(fit_cheb_nd((z_nodes[:, 1] - bzd).reshape(shape), nodes, ranges, FIT_DEGREES["zdrag"]))
    z_check = out[start:]
    # held-out accuracy (relative to z), each point on its piece
    piece_of = np.searchsorted(np.asarray(breaks), check[:, 3], side="right")
    cbzs, cbzd = base_z(base, check[:, 0], check[:, 1])
    ps, pd = np.empty(n_check), np.empty(n_check)
    for i, ranges in enumerate(piece_ranges):
        sel = piece_of == i
        ps[sel] = cbzs[sel] + eval_points(cs_all[i], check[sel], ranges)
        pd[sel] = cbzd[sel] + eval_points(cd_all[i], check[sel], ranges)
    es = np.abs(ps / z_check[:, 0] - 1)
    ed = np.abs(pd / z_check[:, 1] - 1)
    ph = slice(0, n_check // 2)
    print(f"[{tag}] held-out z* error max {es.max():.2e} (99% {np.quantile(es, 0.99):.2e}), "
          f"z_drag max {ed.max():.2e} (99% {np.quantile(ed, 0.99):.2e}); physical (ω_b, ω_c): "
          f"z* max {es[ph].max():.2e}, z_drag max {ed[ph].max():.2e}", flush=True)
    out_path = out_dir / nu_corrections_name(model)
    np.savez(out_path, axes=np.array(axes), ranges=np.array([OMBH2_RANGE, OMCH2_RANGE, r3, r4]),
             x4_breaks=np.array(breaks, dtype=float),
             zstar_corr_cheb=np.stack(cs_all), zdrag_corr_cheb=np.stack(cd_all), n_nodes=np.array(N_NODES),
             h0_grid=np.array(ZSTAR_H0_GRID), sterile_mnu=STERILE_MNU, fid_h0=FID_H0,
             heldout_zstar_max=es.max(), heldout_zdrag_max=ed.max())
    # raw CAMB values (refits, diagnostics) next to the outputs, not in the shipped tables
    diag = REPO_ROOT / "outputs" / "nu_background_tables" / out_path.name.replace(".npz", "_camb.npz")
    diag.parent.mkdir(parents=True, exist_ok=True)
    np.savez(diag, check_points=check, check_z=z_check, node_z=out[: start])
    print(f"[{tag}] wrote {out_path}", flush=True)
    return out_path


def main() -> int:
    from gpucosmo.jax.cpu_threads import DEFAULT_CPU_THREADS, default_worker_count

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="all", choices=["all", *MODELS])
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--workers", type=int, default=default_worker_count(DEFAULT_CPU_THREADS))
    parser.add_argument("--n-check", type=int, default=400)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "src" / "gpucosmo" / "_data")
    args = parser.parse_args()
    for tag in (MODELS if args.model == "all" else [args.model]):
        build(tag, args.base, args.workers, args.n_check, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
