"""Configure JAX to use the NVIDIA GPU in WSL (firecrown_developer env)."""

from __future__ import annotations

import os
import site
import sys
from pathlib import Path
from typing import Iterable


def nvidia_pip_lib_dirs() -> list[Path]:
    """Collect ``lib`` directories from pip ``nvidia-*`` wheels."""
    dirs: list[Path] = []
    for base in site.getsitepackages():
        nvidia_root = Path(base) / "nvidia"
        if not nvidia_root.is_dir():
            continue
        for lib_dir in nvidia_root.rglob("lib"):
            if lib_dir.is_dir():
                dirs.append(lib_dir)
    # De-duplicate while preserving order.
    seen: set[Path] = set()
    unique: list[Path] = []
    for d in dirs:
        if d not in seen:
            seen.add(d)
            unique.append(d)
    return unique


def wsl_nvidia_lib_dir() -> Path:
    return Path("/usr/lib/wsl/lib")


def build_ld_library_path(extra: Iterable[Path] | None = None) -> str:
    """Prefer pip CUDA/cuDNN libs and the WSL NVIDIA driver stub over system libs."""
    parts: list[str] = []
    for d in extra or ():
        parts.append(str(d))
    parts.extend(str(d) for d in nvidia_pip_lib_dirs())
    wsl = wsl_nvidia_lib_dir()
    if wsl.is_dir():
        parts.append(str(wsl))
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    if existing:
        parts.append(existing)
    return ":".join(parts)


def default_compilation_cache_dir() -> Path:
    """Default on-disk JAX compilation cache for GPUCosmo / CLAX warmups."""
    return Path.home() / ".cache" / "gpucosmo_jax_compilation"


def configure_jax_compilation_cache(
    cache_dir: Path | str | None = None,
    *,
    min_compile_time_secs: float = 0.0,
) -> Path:
    """
    Enable JAX persistent compilation cache on disk.

    Must run before ``import jax`` in the current process. Child processes
    spawned afterwards inherit the updated environment.
    """
    resolved = Path(cache_dir or default_compilation_cache_dir()).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    os.environ["JAX_COMPILATION_CACHE_DIR"] = str(resolved)
    os.environ["JAX_ENABLE_COMPILATION_CACHE"] = "true"
    os.environ.setdefault(
        "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES",
        "xla_gpu_per_fusion_autotune_cache_dir",
    )
    if min_compile_time_secs >= 0.0:
        os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = str(
            min_compile_time_secs
        )
    return resolved


def configure_jax_gpu_env(*, require_gpu: bool = False) -> None:
    """
    Set ``LD_LIBRARY_PATH`` so JAX loads pip CUDA wheels + WSL ``libcuda``.

    Must run before ``import jax`` in the current process. Child processes
  inherit the updated environment when spawned afterwards.
    """
    os.environ["LD_LIBRARY_PATH"] = build_ld_library_path()
    if os.environ.get("JAX_PLATFORMS", "").strip().lower() == "cpu":
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        return
    os.environ.pop("JAX_PLATFORMS", None)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    if not require_gpu:
        return

    import jax

    devices = jax.devices()
    if not any(getattr(d, "platform", "") == "gpu" for d in devices):
        raise RuntimeError(
            f"JAX GPU not available (devices={devices}). "
            "On WSL, remove conflicting apt packages such as libnvidia-compute-535 "
            "and use jax[cuda12] with pip nvidia wheels."
        )


def nvidia_smi_snapshot() -> dict[str, int | str] | None:
    """Return current GPU utilization and memory from nvidia-smi, or None if unavailable."""
    import shutil
    import subprocess

    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        name, util_s, mem_used_s, mem_total_s = [part.strip() for part in out.split(",")]
        return {
            "name": name,
            "util_gpu_pct": int(util_s),
            "memory_used_mib": int(mem_used_s),
            "memory_total_mib": int(mem_total_s),
        }
    except (subprocess.CalledProcessError, OSError, ValueError):
        return None


class GpuUtilMonitor:
    """Background nvidia-smi poller for JAX benchmark verification."""

    def __init__(self, *, interval_s: float = 0.15) -> None:
        import threading

        self._interval_s = interval_s
        self._samples: list[dict[str, int | str]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        import threading

        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, float | int]:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=2.0)
            self._thread = None
        return self.summary()

    def _poll_loop(self) -> None:
        import time

        while not self._stop.is_set():
            snap = nvidia_smi_snapshot()
            if snap is not None:
                self._samples.append(snap)
            time.sleep(self._interval_s)

    def summary(self) -> dict[str, float | int]:
        if not self._samples:
            return {
                "n_samples": 0,
                "max_util_gpu_pct": 0,
                "mean_util_gpu_pct": 0.0,
                "max_memory_used_mib": 0,
            }
        utils = [int(s["util_gpu_pct"]) for s in self._samples]
        mems = [int(s["memory_used_mib"]) for s in self._samples]
        return {
            "n_samples": len(self._samples),
            "max_util_gpu_pct": max(utils),
            "mean_util_gpu_pct": float(sum(utils) / len(utils)),
            "max_memory_used_mib": max(mems),
        }


def assert_gpu_active(
    summary: dict[str, float | int],
    *,
    min_max_util_pct: int = 5,
    min_memory_mib: int = 64,
) -> None:
    """Raise if nvidia-smi never saw meaningful GPU use during a JAX section."""
    max_util = int(summary.get("max_util_gpu_pct", 0))
    max_mem = int(summary.get("max_memory_used_mib", 0))
    if max_util < min_max_util_pct and max_mem < min_memory_mib:
        raise RuntimeError(
            "JAX benchmark did not activate the GPU according to nvidia-smi: "
            f"max util {max_util}%, max memory {max_mem} MiB. "
            "Check JAX devices, LD_LIBRARY_PATH / WSL CUDA setup, and that "
            "configure_jax_gpu_env() runs before import jax."
        )


def main() -> int:
    """CLI helper: ``python jax_gpu_env.py`` prints detected JAX devices."""
    configure_jax_gpu_env(require_gpu=True)
    import jax

    print("JAX devices:", jax.devices())
    import jax.numpy as jnp

    x = jnp.ones(4)
    print("test array:", x, "on", x.devices())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
