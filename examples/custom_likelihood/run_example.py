#!/usr/bin/env python3
"""The example analysis from Python: bundle with user likelihoods, exact log posterior,
DALI expansion and compressed DALI likelihood. Runs on a CPU in a few seconds."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gpucosmo.likelihoods import load_plugin_module
from gpucosmo.pipeline import build_analysis_bundle

HERE = Path(__file__).resolve().parent
load_plugin_module(str(HERE / "my_likelihoods.py"))  # registers cosmic_chronometers

bundle, resolved, diffs = build_analysis_bundle(config=HERE / "analysis.yaml", verbose=True)
x0 = np.asarray(bundle.fiducial)
print("parameters:", bundle.param_names)

# exact log posterior (batched) and its components at the starting point
print("exact log posterior:", float(bundle.log_post_batch_fn(jnp.asarray(x0[None]))[0]))
th = bundle.theory_exact(jnp.asarray(x0[:bundle.n_theory]))
for k, v in bundle.log_like_from_theory_fn(jnp.asarray(x0), th, components=True).items():
    print(f"  {k:22s} {float(v):10.4f}")

# second-order DALI expansion of every theory output (including the user likelihoods')
expansion = bundle.dali.build(x0[:bundle.n_theory], verbose=False)
x1 = jnp.asarray(x0 * (1 + 0.002))
exact = bundle.log_like_from_theory_fn(x1, bundle.theory_exact(x1[:bundle.n_theory]))
surr = bundle.log_like_from_theory_fn(x1, bundle.dali.theory(expansion, x1))
print(f"log L at a nearby point: exact {float(exact):.6f}, DALI {float(surr):.6f}")

# compressed DALI likelihood (what the SMC engine samples) + its gradient
from gpucosmo.jax.inference.dali_reduce import CompressedDALI

cs = CompressedDALI(bundle, expansion[0], x0, log=lambda m: None)
ll, g = jax.value_and_grad(lambda x: cs.loglike(x, cs.data))(x1)
print(f"compressed DALI likelihood log L {float(ll):.6f}; gradient {np.asarray(g).round(3)}")
