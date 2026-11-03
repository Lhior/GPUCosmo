#!/usr/bin/env python3
"""Print ``--sampling-box`` arguments: ±Nσ of a previous posterior for uniform-prior parameters.

For every parameter with a uniform prior in the Cobaya yaml whose ±``--nsigma`` range around
a previous run's posterior (``naut.ns_posterior.npz``) is narrower than the prior, emits
``--sampling-box NAME=LO,HI`` (intersected with the prior). ``run_dali_ns.py`` samples
the narrowed box, corrects log Z by the volume ratio and reports the posterior's distance to
the moved edges.

    python scripts/sampling_boxes.py --config <yaml> --posterior outputs/sns_mg_v2/naut.ns_posterior.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--posterior", type=Path, required=True)
    ap.add_argument("--nsigma", type=float, default=25.0)
    args = ap.parse_args()
    from gpucosmo.jax.inference.cobaya_yaml import resolve_cobaya

    pri = resolve_cobaya(args.config).priors
    z = np.load(args.posterior, allow_pickle=True)
    names = [str(n) for n in z["names"]]
    out = []
    for j, n in enumerate(names):
        p = pri.get(n)
        if p is None or p.gauss is not None:
            continue
        m, s = float(z["pos"][:, j].mean()), float(z["pos"][:, j].std())
        lo, hi = max(p.lo, m - args.nsigma * s), min(p.hi, m + args.nsigma * s)
        if lo > p.lo or hi < p.hi:
            out.append(f"--sampling-box {n}={lo:.6g},{hi:.6g}")
    print(" ".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
