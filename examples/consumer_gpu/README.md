# Consumer-GPU example: CMB spectra checked against CAMB

`run_example.py` computes lensed CMB spectra (TT, TE, EE and κκ, ℓ ≤ 2508) for the two
cosmologies in [settings.json](settings.json) with GPUCosmo's Einstein–Boltzmann solver. It
times a cold and a warm start, compares the spectra with CAMB, and reports whether any
evaluation failed numerically. It needs **no data products**: the only inputs are the
cosmological parameters.

```bash
pip install ".[gpu]"     # from the repository root: GPUCosmo with JAX for CUDA 12
python examples/consumer_gpu/run_example.py --out out_consumer_gpu
```

The script writes `out_consumer_gpu/result_<device>.json` and exits non-zero if a check fails.

## What it measures

- **Cold**: the first call in a fresh process, which covers tracing, XLA compilation, the host
  tables and the run. The persistent compilation cache is off unless `--compile-cache DIR` is
  given. With a warm cache the first call mostly loads instead of compiling.
- **Warm**: the median of 5 further calls of the compiled program, for the whole batch and per
  cosmology. The default batch is 8 cosmologies on a GPU (the two in `settings.json`, repeated)
  and 2 on a CPU.
- **Accuracy**: the relative difference from CAMB, per ℓ band, for TT, EE, TE (relative to
  √(TT·EE)) and κκ. One documented claim is checked (docs/cobaya.rst): lensed TT and EE at
  ℓ 1000–2508 within 4e-4 of CAMB, plus 6e-6 for the float32 evolution on GPUs; the limit is
  5e-4. The other bands are reported, not checked.
  The recorded runs gave 3.4e-4 (ΛCDM) and 3.5e-4 (w0wa) there. Below ℓ = 1000 the
  differences from this CAMB reference reach about 1e-3 for TT and EE (1.8e-3 for EE at ℓ < 30).
  The documented accuracy claim does not cover those bands, and this example does not determine
  how much of the difference is the CAMB reference's own error at AccuracyBoost 2.
- **Validity**: `solver_status` per cosmology; 0 means no failed mode
  ([numerical failures](../../docs/research_preview.rst)).

## Recorded run

[reference/result_nvidia_geforce_rtx_5070_laptop_gpu.json](reference/result_nvidia_geforce_rtx_5070_laptop_gpu.json):
RTX 5070 Laptop GPU (8 GB, driver 617.14, WSL2), JAX 0.10.1, float32 evolution, commit
3efac5d, batch of 8, 2026-11-01. The machine was lightly loaded by a concurrent CPU test run
(1-min load average 4–5 on 20 threads, recorded in the file).

| | |
|---|---|
| cold first call (compile + host tables + run, no persistent cache) | 136.9 s |
| warm, per batch of 8 (median of 5; 1.88–1.92 s) | 1.91 s |
| warm, per cosmology | 0.24 s |
| lensed TT/EE at ℓ 1000–2508, max relative difference from CAMB | 3.39e-4 (ΛCDM), 3.48e-4 (w0wa) |
| H0 from θ*, GPUCosmo − CAMB | −0.0007, −0.0001 km/s/Mpc |
| `solver_status` | 0 for all 8 |

For comparison, [reference/result_cpu.json](reference/result_cpu.json) is the same check on the
20-thread CPU of that laptop (float64 evolution, batch of 2, commit 94d74cc). It gave the same
accuracy (3.39e-4, 3.48e-4) and `solver_status` 0. It took 63.9 s cold and 11.5 s warm per batch,
5.7 s per cosmology, which is 24× the GPU's per-cosmology time.

A run on another GPU will give different times. The cold time includes XLA compilation. A
persistent cache (`--compile-cache`) can shorten the first call of later processes; that was not
measured here.

## Pinned reference

- `settings.json` pins the cosmologies, the solver settings and CAMB's settings: CAMB 2.0,
  AccuracyBoost 2, lens_potential_accuracy 8, linear lensing, θ* input, one 0.06 eV neutrino,
  PArthENoPE BBN, PPF dark energy. These follow `scripts/validate_nu_curv_vs_camb.py`.
- `reference/camb_reference.npz` holds the CAMB spectra. `--camb` recomputes them with your
  CAMB, which takes a few minutes on a CPU.
- `reference/result_*.json` holds the recorded runs. Each file records the device, driver, JAX
  and NumPy versions, the git commit, the timings and the accuracy table. A file comes from one
  run on the hardware it names. The CPU file checks correctness only: on a CPU the evolution runs
  in float64, and its timing is not a GPU number.

This example checks the theory against CAMB at a few points. It does not validate the
evidence pipeline: see [examples/custom_likelihood](../custom_likelihood) for a small end-to-end
evidence run, and [docs/research_preview.rst](../../docs/research_preview.rst) for the
validation status.
