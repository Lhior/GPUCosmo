Command-line tools
==================

All tools take ``--help``. The pipeline tools accept ``--config`` (a Cobaya yaml) or
``--analysis NAME`` (one of the yamls shipped in ``gpucosmo/configs/cobaya``: DESI DR2 BAO +
Planck 2018 low-ℓ + CamSpec NPIPE + ACT DR6 lensing + one SN set, for ΛCDM and w0waCDM). They also take ``--plugin MODULE`` (repeatable)
to import modules that register user likelihoods.

``gpucosmo-evidence``
    The whole pipeline (map → DALI build → SMC and order check) in one command, resumable. Exact
    stages share one process and one compiled theory program; the SMC runs as a subprocess.
    ``--dali-order`` (default 3, the triplet) sets the expansion's order and ``--smc-target-err``
    (default 0.009) the stopping error. ``--sampling-box``
    narrows uniform priors for the SMC (log Z corrected by the volume ratio); ``--auto-boxes NSIGMA``
    derives them from the run itself (±NSIGMA σ of the DALI expansion's Laplace approximation at the MAP,
    e.g. 10), so a first run needs no earlier posterior. Use it whenever the priors are much wider
    than the posterior (see :doc:`usage`). ``--ns-args`` passes further options to the
    SMC stage.

``gpucosmo-dali-hmc``
    ``--stage map``: maximum a posteriori by iterated linearization. ``--stage build``:
    DALI expansion at the MAP to order ``--dali-order`` (default 3; ``dali.npz``; ``--dali-method fd|fwd``).
    ``--stage sample`` / ``converge``: DALI-trajectory HMC. ``--accept dali`` (default) accepts or
    rejects on the DALI posterior with the official likelihoods, without an exact likelihood call
    (the posterior the evidence pipeline samples); ``--accept exact`` on the exact posterior, one
    batched exact theory call per iteration (about 6× slower per iteration for a CMB analysis), so the chains
    target the exact posterior whatever the DALI expansion's error.

``gpucosmo-dali-ns``
    ``--stage ns``: tempered SMC on the DALI posterior (``--compressed-dali``, GPU-native),
    started from a pilot-fitted reference by default (``--smc-reference prior|pilot|FILE.npz``) and
    stopped at ``--smc-target-err``: log Z_DALI and posterior draws (its ± is Monte Carlo
    precision; see :ref:`evidence-error-budget`). Then the DALI order check: the expansion
    truncated one order lower at the posterior draws, giving the last order's effect on log Z and
    on the posterior means (``--no-order-check`` skips it). Run by hand, the exact stages (map,
    build) want ``XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"`` next
    to large CMB covariances (``gpucosmo-evidence`` sets them); the SMC stage runs fastest without them.
    ``--stage laplace`` writes the Laplace approximation at the MAP that ``--auto-boxes`` uses.

``gpucosmo-likelihoods``
    List the built-in and registered likelihoods.

``gpucosmo-paths``
    Show the data, Cobaya-packages and cache locations.

``scripts/`` in the source checkout holds the same pipeline programs
(``run_dali_hmc.py``, ``run_dali_ns.py``), the background-table builders, the ACT DR6
lensing data download, the generator of the shipped DESI DR2 + CMB + SN yamls, and two
helpers the tests import (``validate_cobaya_bridge.py``, ``bench_evolve.py``).
