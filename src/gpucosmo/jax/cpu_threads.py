"""Consistent CPU thread / worker configuration for GPUCosmo scripts.

Fairness note
-------------
``DEFAULT_CPU_THREADS`` (20) is aimed at **multi-threaded CPU reference** arms
(validation, Cobaya CPU chains, BLAS-heavy host work). Comparing that to a
**single GPU stream** of jax_eb FP64 kernels is apples-to-oranges: the CPU arm
gets ~20-way OpenMP/XLA parallelism while the GPU arm is one latency-bound
wavefront. For GPU-vs-CPU latency bakeoffs use ``fair_cpu_threads_for_gpu_compare()``
(typically 1). ``configure_cpu_threads`` also caps at ``os.cpu_count()`` so a
20-thread default does not oversubscribe small hosts (e.g. WSL / ARM laptops).
"""

from __future__ import annotations

import os
from typing import Final

DEFAULT_CPU_THREADS: Final[int] = 20

_THREAD_ENV_VARS: Final[tuple[str, ...]] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def host_cpu_count() -> int:
    """Logical CPU count available to this process (at least 1)."""
    return max(1, int(os.cpu_count() or 1))


def effective_cpu_threads(n_threads: int | None = None) -> int:
    """Clamp requested threads to ``[1, host_cpu_count()]``.

    ``None`` uses ``DEFAULT_CPU_THREADS`` (still capped by host count).
    """
    requested = DEFAULT_CPU_THREADS if n_threads is None else int(n_threads)
    return max(1, min(requested, host_cpu_count()))


def fair_cpu_threads_for_gpu_compare() -> int:
    """Threads for latency comparisons against a single GPU stream.

    Returns 1 so CPU XLA/OpenMP do not win via 20-way unfairness. Throughput
    comparisons that intentionally use a saturated CPU should keep
    ``DEFAULT_CPU_THREADS`` / ``effective_cpu_threads()``.
    """
    return 1


def configure_cpu_threads(
    n_threads: int | None = DEFAULT_CPU_THREADS,
    *,
    xla: bool = True,
    fair_vs_gpu: bool = False,
) -> dict[str, str]:
    """
    Pin BLAS/OpenMP and (optionally) XLA CPU pools.

    Call before importing numpy/jax/camb in single-process workloads that should
    use many CPU threads (validation, cobaya CPU reference chains).

    ``fair_vs_gpu=True`` forces a single host thread (see module docstring).
    Requested counts are always capped at ``host_cpu_count()``.
    """
    if fair_vs_gpu:
        n = fair_cpu_threads_for_gpu_compare()
    else:
        n = effective_cpu_threads(n_threads)
    applied: dict[str, str] = {}
    for var in _THREAD_ENV_VARS:
        os.environ[var] = str(n)
        applied[var] = str(n)
    if xla:
        xla_flags = (
            f"--xla_cpu_multi_thread_eigen=true "
            f"--xla_force_host_platform_device_count=1 "
            f"intra_op_parallelism_threads={n}"
        )
        os.environ["XLA_FLAGS"] = xla_flags
        applied["XLA_FLAGS"] = xla_flags
    return applied


def configure_worker_threads() -> None:
    """One thread per process in multiprocessing pools (avoid oversubscription)."""
    for var in _THREAD_ENV_VARS:
        os.environ[var] = "1"


def default_worker_count(n_threads: int | None = DEFAULT_CPU_THREADS) -> int:
    """Process-pool size for embarrassingly parallel CPU work."""
    return effective_cpu_threads(n_threads)


__all__ = [
    "DEFAULT_CPU_THREADS",
    "configure_cpu_threads",
    "configure_worker_threads",
    "default_worker_count",
    "effective_cpu_threads",
    "fair_cpu_threads_for_gpu_compare",
    "host_cpu_count",
]
