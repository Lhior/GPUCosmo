"""Choose GPU batch sizes for the exact theory from the device's memory.

The exact log-posterior batch evaluates ``n_chains`` cosmologies. How many are
vmapped together (``MGConfig.cmb_walker_batch`` / ``pk_walker_batch``) trades
memory for throughput: on an 8 GB laptop GPU next to Planck PR4's 3.3 GB
HiLLiPoP matrix only one CMB walker fits, while an 80 GB HPC GPU can batch all
chains. ``autosize_batches`` compiles the theory program for candidate batch
sizes (largest first), keeps those whose compiled footprint (XLA memory
analysis: temporaries + arguments + outputs + embedded constants) fits the
budget, times the best fitting candidates, and returns the fastest. Results are
cached per (device, memory, analysis, chain count, JAX version) in
``~/.cache/gpucosmo/autosize.json`` so the compiles are paid once per machine.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

CACHE = Path.home() / ".cache" / "gpucosmo" / "autosize.json"
BATCH_FIELDS = ("cmb_walker_batch", "pk_walker_batch", "cmb_layout")
# CMB layouts tried per batch size: memory-lean first, then the faster-but-hungrier one.
LAYOUTS = ("flat", "w_outer")


FRAG = 0.03  # allocator fragmentation allowance (fraction of the pool)


def device_budget_bytes(frag: float = FRAG) -> int:
    """Allocator limit (``XLA_PYTHON_CLIENT_MEM_FRACTION`` of the GPU) less a
    fragmentation allowance. Other resident programs and arrays are subtracted by
    the caller from their measured footprints."""
    stats = jax.devices()[0].memory_stats() or {}
    limit = stats.get("bytes_limit")
    if limit is None:  # CPU: no meaningful limit
        return 1 << 62
    return int(limit * (1.0 - frag))


def candidate_batches(n_chains: int, cmb: bool, fs: bool) -> list[tuple[int, int, str]]:
    """(cmb_walker_batch, pk_walker_batch, cmb_layout), largest batch first. Batches
    divide the chain count; P(k) gets at least 4 walkers when the CMB batch is
    memory-starved (it batches well and needs much less memory)."""
    divs = [d for d in range(n_chains, 0, -1) if n_chains % d == 0]
    if not cmb:
        return [(1, d, "flat") for d in divs]
    pk_floor = min(4, n_chains) if fs else 1
    cands = [(d, max(d, pk_floor) if fs else 1, lay) for d in divs for lay in LAYOUTS]
    # One cosmology per CMB call: the per-walker single-cosmology pipeline
    # (layout None) is ~5× faster than a sequential map over size-1 groups.
    cands.append((1, pk_floor if fs else 1, None))
    return list(dict.fromkeys(cands))


def _key(analysis: str, cfg, n_chains: int, budget: int) -> str:
    dev = jax.devices()[0]
    base = {k: v for k, v in dataclasses.asdict(cfg).items() if k not in BATCH_FIELDS}
    blob = json.dumps(dict(device=dev.device_kind, budget_256mb=budget >> 28,
                           analysis=analysis, cfg=base, n=n_chains, jax=jax.__version__),
                      sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def _load_cache() -> dict:
    try:
        return json.loads(CACHE.read_text())
    except (OSError, ValueError):
        return {}


def footprint_bytes(compiled, *, arguments: bool = True) -> int:
    """Device bytes of a compiled program: temporaries, outputs, embedded constants
    (generated code) and, with ``arguments``, its inputs. Programs whose large
    inputs are already resident (e.g. the likelihood's HiLLiPoP matrix) are
    counted with ``arguments=False``."""
    m = compiled.memory_analysis()
    return int(m.temp_size_in_bytes + m.output_size_in_bytes + m.generated_code_size_in_bytes
               + (m.argument_size_in_bytes if arguments else 0))


def autosize_batches(build_bundle, cfg, *, analysis: str, n_chains: int,
                     resident_bytes: int = 0, reserved_bytes: int = 0, frag: float = FRAG,
                     n_time: int = 3, verbose: bool = True):
    """Return ``cfg`` with the fastest fitting (cmb_walker_batch, pk_walker_batch).

    ``build_bundle(cfg)`` builds an MG bundle (``mg_bundle.build_mg_bundle``) whose
    ``theory_batch`` is the exact theory program. ``resident_bytes``: arrays held on
    the device (e.g. the PR4 HiLLiPoP matrix); ``reserved_bytes``: measured
    footprints of the other programs that stay loaded (likelihood, trajectory).
    Budget = allocator limit × (1 − ``frag``) − resident − reserved.
    """
    budget = device_budget_bytes(frag) - int(resident_bytes) - int(reserved_bytes)
    key = _key(analysis, cfg, n_chains, budget)
    cache = _load_cache()
    if key in cache:
        hit = cache[key]
        if verbose:
            print(f"autosize: cached {hit['choice']} for {analysis} "
                  f"({hit['device']}, {n_chains} chains)", flush=True)
        return dataclasses.replace(cfg, **dict(zip(BATCH_FIELDS, hit["choice"])))
    if verbose:
        lim = (jax.devices()[0].memory_stats() or {}).get("bytes_limit", 0)
        print(f"autosize: pool {lim / 2**30:.2f} GB − {frag:.0%} fragmentation − resident "
              f"{resident_bytes / 2**30:.2f} GB − other programs {reserved_bytes / 2**30:.2f} GB "
              f"= budget {budget / 2**30:.2f} GB", flush=True)
    from gpucosmo.jax.inference.mg_bundle import cmb_on

    cands = candidate_batches(n_chains, cmb_on(cfg), bool(cfg.desi_fs))

    def fallback():
        smallest = cands[-1]
        print(f"autosize: WARNING no configuration fits the budget "
              f"({budget / 2**30:.2f} GB); using the smallest, cmb {smallest[0]} "
              f"({smallest[2]}) / pk {smallest[1]}", flush=True)
        return dataclasses.replace(cfg, **dict(zip(BATCH_FIELDS, smallest)))

    if budget <= 0:
        return fallback()
    log, fitting, too_big, compiled = [], [], set(), {}

    def compile_cand(cb, pb, lay):
        c = dataclasses.replace(cfg, cmb_walker_batch=cb, pk_walker_batch=pb, cmb_layout=lay)
        b = build_bundle(c)
        X = jnp.asarray(np.tile(np.asarray(b.fiducial)[: b.n_theory], (n_chains, 1)))
        t0 = time.perf_counter()
        comp = jax.jit(b.theory_batch).lower(X).compile()
        return comp, X, time.perf_counter() - t0

    # Memory model from the two smallest "flat" programs (cheap to compile):
    # footprint ≈ base + per_walker · cmb_batch. Larger candidates predicted not to
    # fit (with 10% slack) are skipped without compiling them: large batches are
    # also the slowest to compile (30+ min for 8 walkers on a laptop CPU).
    flat = sorted({(cb, pb) for cb, pb, lay in cands if lay == "flat"})
    predict = None
    if len(flat) >= 2 and flat[0][0] != flat[1][0]:
        need_lo = []
        for cb, pb in flat[:2]:
            comp, X, t_c = compile_cand(cb, pb, "flat")
            compiled[(cb, pb, "flat")] = (comp, X, t_c)
            need_lo.append(footprint_bytes(comp))
        (c1, _), (c2, _) = flat[:2]
        slope = (need_lo[1] - need_lo[0]) / (c2 - c1)
        base = need_lo[0] - slope * c1
        predict = lambda cb: base + slope * cb  # noqa: E731
        if verbose:
            print(f"autosize: memory model {base / 2**30:.2f} GB + {slope / 2**30:.2f} GB per "
                  f"CMB walker; budget {budget / 2**30:.2f} GB", flush=True)
    for cb, pb, lay in cands:
        if (cb, pb) in too_big:  # "w_outer" needs more memory than "flat" at a given batch
            continue
        if predict is not None and (cb, pb, lay) not in compiled and predict(cb) > 1.1 * budget:
            too_big.add((cb, pb))
            log.append(dict(cmb=cb, pk=pb, layout=lay, need_gb=predict(cb) / 2**30,
                            predicted=True, fits=False))
            continue
        comp, X, t_c = compiled.pop((cb, pb, lay), None) or compile_cand(cb, pb, lay)
        need = footprint_bytes(comp)
        fits = need <= budget
        log.append(dict(cmb=cb, pk=pb, layout=lay, need_gb=need / 2**30, compile_s=t_c,
                        fits=fits))
        if verbose:
            print(f"autosize: cmb {cb} ({lay}) / pk {pb}: {need / 2**30:.2f} GB "
                  f"(budget {budget / 2**30:.2f} GB) {'fits' if fits else 'too big'}", flush=True)
        if not fits:
            too_big.add((cb, pb))
        else:
            fitting.append(((cb, pb, lay), comp, X))
            if len(fitting) >= n_time:
                break
    if not fitting:
        return fallback()
    best, best_t = None, np.inf
    for choice, comp, X in fitting:
        jax.block_until_ready(comp(X))  # warm
        t0 = time.perf_counter()
        jax.block_until_ready(comp(X * (1 + 1e-9)))
        dt = time.perf_counter() - t0
        if verbose:
            print(f"autosize: cmb {choice[0]} ({choice[2]}) / pk {choice[1]}: {dt:.2f} s per "
                  f"batch of {n_chains}", flush=True)
        for row in log:
            if (row["cmb"], row["pk"], row["layout"]) == choice:
                row["time_s"] = dt
        if dt < best_t:
            best, best_t = choice, dt
    del fitting
    dev = jax.devices()[0]
    cache[key] = dict(choice=list(best), device=dev.device_kind, analysis=analysis,
                      n_chains=n_chains, time_s=best_t, candidates=log)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache, indent=1))
    if verbose:
        print(f"autosize: chose cmb {best[0]} ({best[2]}) / pk {best[1]} ({best_t:.2f} s)",
              flush=True)
    return dataclasses.replace(cfg, **dict(zip(BATCH_FIELDS, best)))


__all__ = ["FRAG", "autosize_batches", "candidate_batches", "device_budget_bytes", "footprint_bytes"]
