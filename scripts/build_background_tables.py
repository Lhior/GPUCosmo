#!/usr/bin/env python3
"""
Build CAMB-derived tables for the JAX w0waCDM background module.

Products (saved to src/gpucosmo/_data/w0wa_background_tables_v2.npz; Y_He from the solver's BBN
table, PArthENoPE):

- Radiation / neutrino density constants extracted from a fiducial CAMB run
  (fixed T_CMB = 2.7255 K, N_eff = 3.044, one massive neutrino with
  sum m_nu = 0.06 eV -- the DESI DR2 baseline).
- Massive-neutrino density shape rho_nu(a) / rho_nu(a=1) on a dense ln(a)
  grid, taken directly from CAMB so no Fermi-Dirac convention can diverge.
- z_star(ombh2, omch2) and z_drag(ombh2, omch2) recombination redshifts on a
  2D Chebyshev-node grid spanning the sampling priors. These are
  early-Universe quantities: at fixed physical densities they are independent
  of H0 and of (w0, wa) under the w0+wa<0 prior (dark energy is negligible
  before recombination), which is verified by the validation script.

Everything downstream (distances, sound horizons r_s(z_star), r_drag) is
computed exactly in JAX with the full w0waCDM H(z), so the dark-energy
dependence of the BAO/CMB observables is not approximated.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from gpucosmo.jax.cpu_threads import (  # noqa: E402
    DEFAULT_CPU_THREADS,
    configure_cpu_threads,
    configure_worker_threads,
    default_worker_count,
)

import numpy as np

configure_cpu_threads(DEFAULT_CPU_THREADS, xla=False)
# v2 (2026-10-22): built with the solver's BBN table (PArthENoPE, as the DESI chains); v1 used CAMB
# 2.0.4's default PRIMAT 2024 Y_He, which put z* / z_drag 9e-6 / 8e-6 off the solver's recombination.
DEFAULT_OUT = REPO_ROOT / "src" / "gpucosmo" / "_data" / "w0wa_background_tables_v2.npz"

# DESI DR2 baseline neutrino treatment.
MNU_EV = 0.06
NNU = 3.044
TCMB_K = 2.7255
BBN_PREDICTOR = "PArthENoPE_880.2_standard.dat"  # = gpucosmo.jax.eb.bbn_jax.DESI_BBN_TABLE

# Table domain: must cover the sampling priors on (ombh2, omch2).
OMBH2_RANGE = (0.005, 0.04)
OMCH2_RANGE = (0.05, 0.30)
N_NODES = 32
CHEB_DEGREE = 31

# Average z_star/z_drag over a few H0 values to suppress RECFAST's
# O(1e-4) numerical H0 jitter (physically z_* depends only on ombh2, omch2).
ZSTAR_H0_GRID = (55.0, 67.36, 80.0)


FID_H0 = 67.36
FID_OMBH2 = 0.02237
FID_OMCH2 = 0.12


def _camb_pars(ombh2: float, omch2: float, h0: float = FID_H0):
    import camb

    return camb.set_params(
        H0=h0,
        ombh2=ombh2,
        omch2=omch2,
        mnu=MNU_EV,
        nnu=NNU,
        TCMB=TCMB_K,
        tau=0.054,
        As=2.1e-9,
        ns=0.965,
        bbn_predictor=BBN_PREDICTOR,
    )


def _derived_zs(args: tuple[float, float]) -> tuple[float, float]:
    configure_worker_threads()
    import camb

    ombh2, omch2 = args
    zstars = []
    zdrags = []
    for h0 in ZSTAR_H0_GRID:
        results = camb.get_background(_camb_pars(ombh2, omch2, h0=h0))
        derived = results.get_derived_params()
        zstars.append(float(derived["zstar"]))
        zdrags.append(float(derived["zdrag"]))
    return float(np.median(zstars)), float(np.median(zdrags))


def cheb_nodes(lo: float, hi: float, n: int) -> np.ndarray:
    """First-kind Chebyshev points mapped to [lo, hi]."""
    k = np.arange(n)
    x = np.cos(np.pi * (2 * k + 1) / (2 * n))
    return 0.5 * (lo + hi) + 0.5 * (hi - lo) * np.sort(x)


def fit_cheb2d(
    x_nodes: np.ndarray,
    y_nodes: np.ndarray,
    values: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    degree: int,
) -> np.ndarray:
    """Least-squares 2D Chebyshev coefficients on the given tensor grid."""
    from numpy.polynomial import chebyshev as C

    def unit(v: np.ndarray, rng: tuple[float, float]) -> np.ndarray:
        return (2.0 * v - (rng[0] + rng[1])) / (rng[1] - rng[0])

    xu = unit(x_nodes, x_range)
    yu = unit(y_nodes, y_range)
    xg, yg = np.meshgrid(xu, yu, indexing="ij")
    vander = C.chebvander2d(xg.ravel(), yg.ravel(), [degree, degree])
    coef, *_ = np.linalg.lstsq(vander, values.ravel(), rcond=None)
    coef = coef.reshape(degree + 1, degree + 1)
    fitted = vander @ coef.ravel()
    resid = np.abs(fitted - values.ravel())
    print(
        f"  cheb fit degree {degree}: max |resid| = {resid.max():.3e} "
        f"(relative {(resid / np.abs(values.ravel())).max():.3e})"
    )
    return coef


def build_density_constants() -> dict[str, np.ndarray | float]:
    """Extract radiation and neutrino density constants from a fiducial CAMB run."""
    import camb

    pars = _camb_pars(FID_OMBH2, FID_OMCH2)
    results = camb.get_background(pars)
    h2 = (FID_H0 / 100.0) ** 2

    omega_gamma_h2 = float(results.get_Omega("photon", z=0.0)) * h2
    omega_nu_massless_h2 = float(results.get_Omega("neutrino", z=0.0)) * h2
    omega_nu_massive_h2 = float(results.get_Omega("nu", z=0.0)) * h2
    omega_de_h2 = float(results.get_Omega("de", z=0.0)) * h2
    print(
        f"  omega_gamma_h2={omega_gamma_h2:.10e}\n"
        f"  omega_nu_massless_h2={omega_nu_massless_h2:.10e}\n"
        f"  omega_nu_massive_h2(a=1)={omega_nu_massive_h2:.10e}\n"
        f"  omega_de_h2={omega_de_h2:.10e}  (CAMB omnuh2={pars.omnuh2:.10e})"
    )

    # Massive neutrino density shape rho_nu(a)/rho_nu(1) from CAMB itself:
    # rho_nu(z) proportional to Omega_nu(z) * H(z)^2.
    ln_a = np.linspace(np.log(1.0e-10), 0.0, 4096)
    z = 1.0 / np.exp(ln_a) - 1.0
    omega_nu_z = np.array([results.get_Omega("nu", z=zi) for zi in z])
    hub = results.hubble_parameter(z)  # km/s/Mpc
    rho = omega_nu_z * hub**2
    shape = rho / rho[-1]  # ln_a[-1] == 0 -> a = 1
    return {
        "omega_gamma_h2": omega_gamma_h2,
        "omega_nu_massless_h2": omega_nu_massless_h2,
        "omega_nu_massive_h2": omega_nu_massive_h2,
        "nu_ln_a": ln_a,
        "nu_ln_shape": np.log(shape),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--workers",
        type=int,
        default=default_worker_count(DEFAULT_CPU_THREADS),
        help=f"Parallel CAMB processes (default {DEFAULT_CPU_THREADS})",
    )
    args = parser.parse_args()

    print("Extracting density constants from fiducial CAMB run...")
    consts = build_density_constants()

    ombh2_nodes = cheb_nodes(*OMBH2_RANGE, N_NODES)
    omch2_nodes = cheb_nodes(*OMCH2_RANGE, N_NODES)
    tasks = [(ob, oc) for ob in ombh2_nodes for oc in omch2_nodes]
    print(f"Running CAMB recombination on {len(tasks)} grid points "
          f"with {args.workers} workers (spawn)...")
    # spawn, not fork: parent already imported numpy/CAMB above; forked workers
    # deadlock on pthread locks and sit at 0% CPU in htop.
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        out = list(pool.map(_derived_zs, tasks, chunksize=8))
    zstar = np.array([o[0] for o in out]).reshape(N_NODES, N_NODES)
    zdrag = np.array([o[1] for o in out]).reshape(N_NODES, N_NODES)

    print("Fitting z_star Chebyshev surface:")
    zstar_coef = fit_cheb2d(
        ombh2_nodes, omch2_nodes, zstar, OMBH2_RANGE, OMCH2_RANGE, CHEB_DEGREE
    )
    print("Fitting z_drag Chebyshev surface:")
    zdrag_coef = fit_cheb2d(
        ombh2_nodes, omch2_nodes, zdrag, OMBH2_RANGE, OMCH2_RANGE, CHEB_DEGREE
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        mnu_ev=MNU_EV,
        nnu=NNU,
        tcmb_k=TCMB_K,
        ombh2_range=np.array(OMBH2_RANGE),
        omch2_range=np.array(OMCH2_RANGE),
        ombh2_nodes=ombh2_nodes,
        omch2_nodes=omch2_nodes,
        zstar_grid=zstar,
        zdrag_grid=zdrag,
        zstar_cheb=zstar_coef,
        zdrag_cheb=zdrag_coef,
        **consts,
    )
    print(f"Saved tables to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
