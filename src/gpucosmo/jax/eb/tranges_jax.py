#!/usr/bin/env python3
# This file is part of GPUCosmo. Copyright (C) 2025-2026 Leonel Medina-Varela.
# SPDX-License-Identifier: GPL-3.0-or-later
# Derived in part from the following, translated to JAX and modified for GPUCosmo:
# - CAMB (https://github.com/cmbant/CAMB), Copyright 2000-2025 Antony Lewis and Anthony Challinor.
#   Licence: LICENSES/CAMB.txt.
# See NOTICE.md.
"""CAMB forutils ``TRanges`` (RangeUtils.f90) — piecewise log/linear sampling.

Faithful enough for ``SetkValuesForSources`` and ``SetTimeSteps``: ``Add_delta``,
``Add`` (non-overlapping sequential + simple overlap skip), ``GetArray``,
``Getdpoints``. Full Fortran merge/split for log↔linear overlaps is not needed
for CAMB's sequential CMB source/time construction.
"""

from __future__ import annotations

import numpy as np


class TRanges:
    """Piecewise ranges → points / dpoints (CAMB ``TRanges``)."""

    def __init__(self, range_tol: float = 0.1):
        self.range_tol = float(range_tol)
        self.ranges: list[dict] = []
        self.points: np.ndarray | None = None
        self.dpoints: np.ndarray | None = None
        self.lowest = 0.0
        self.highest = 0.0

    def init(self) -> None:
        self.ranges.clear()
        self.points = None
        self.dpoints = None
        self.lowest = 0.0
        self.highest = 0.0

    def add_delta(
        self,
        t_start: float,
        t_end: float,
        t_approx_delta: float,
        *,
        is_log: bool = False,
    ) -> None:
        """CAMB ``TRanges_Add_delta`` step count, then ``Add``."""
        t_start = float(t_start)
        t_end = float(t_end)
        t_approx_delta = float(t_approx_delta)
        if t_end <= t_start or t_approx_delta <= 0.0:
            return
        if is_log:
            n = max(1, int(np.log(t_end / t_start) / t_approx_delta + 1.0 - self.range_tol))
        else:
            n = max(1, int((t_end - t_start) / t_approx_delta + 1.0 - self.range_tol))
        self.add(t_start, t_end, n, is_log=is_log)

    def add(
        self,
        t_start: float,
        t_end: float,
        nstep: int,
        *,
        is_log: bool = False,
    ) -> None:
        """Append a sampling range (CAMB ``TRanges_Add`` simplified for CMB use)."""
        t_start = float(t_start)
        t_end = float(t_end)
        nstep = int(nstep)
        if t_end <= t_start or nstep <= 0:
            return
        if is_log:
            delta = np.log(t_end / t_start) / nstep
        else:
            delta = (t_end - t_start) / nstep

        # Skip if completely inside an existing finer-or-equal range (CAMB early return)
        for r in self.ranges:
            if t_start >= r["low"] and t_end <= r["high"]:
                if is_log == r["is_log"] and delta >= r["delta"] - 1e-30:
                    return
                if (not is_log) and (not r["is_log"]) and delta >= r["delta"] - 1e-30:
                    return

        # Drop existing ranges completely covered by a denser new range
        kept: list[dict] = []
        for r in self.ranges:
            if t_start <= r["low"] and t_end >= r["high"]:
                denser = (is_log == r["is_log"] and delta <= r["delta"] + 1e-30) or (
                    (not is_log) and (not r["is_log"]) and delta <= r["delta"] + 1e-30
                )
                if denser:
                    continue
            kept.append(r)
        self.ranges = kept
        self.ranges.append(
            {
                "low": t_start,
                "high": t_end,
                "steps": nstep,
                "delta": float(delta),
                "is_log": bool(is_log),
            }
        )
        self.points = None
        self.dpoints = None

    def get_array(self, *, want_dpoints: bool = True) -> np.ndarray:
        """CAMB ``TRanges_GetArray``: per-range j=0..steps-1, then Highest."""
        if not self.ranges:
            self.points = np.array([], dtype=np.float64)
            self.dpoints = np.array([], dtype=np.float64)
            return self.points

        # Sort ranges by Low (CAMB rebuilds endpoints; sequential CMB adds are ordered)
        regs = sorted(self.ranges, key=lambda r: (r["low"], r["high"]))
        self.lowest = regs[0]["low"]
        self.highest = max(r["high"] for r in regs)

        pts: list[float] = []
        for r in regs:
            for j in range(r["steps"]):
                if r["is_log"]:
                    pts.append(r["low"] * np.exp(j * r["delta"]))
                else:
                    pts.append(r["low"] + j * r["delta"])
        pts.append(self.highest)
        arr = np.asarray(pts, dtype=np.float64)
        # Sort+unique: CAMB merges overlapping Add (e.g. reion on late TimeSteps);
        # sequential Evolve_q ranges share endpoints — unique removes duplicates.
        self.points = np.unique(arr)
        if want_dpoints:
            self.get_dpoints(half_ends=True)
        return self.points

    def get_dpoints(self, *, half_ends: bool = True) -> np.ndarray:
        """CAMB ``TRanges_Getdpoints`` on the flattened point array."""
        if self.points is None:
            self.get_array(want_dpoints=False)
        x = self.points
        assert x is not None
        d = np.zeros_like(x)
        if x.size >= 3:
            d[1:-1] = 0.5 * (x[2:] - x[:-2])
        if x.size >= 2:
            if half_ends:
                d[0] = 0.5 * (x[1] - x[0])
                d[-1] = 0.5 * (x[-1] - x[-2])
            else:
                d[0] = x[1] - x[0]
                d[-1] = x[-1] - x[-2]
        self.dpoints = d
        return d


def dlnk_from_points(q: np.ndarray) -> np.ndarray:
    """CAMB Cls ``dlnk = q%dpoints/q`` with ``half_ends=.false.``."""
    r = TRanges()
    r.points = np.asarray(q, dtype=np.float64)
    d = r.get_dpoints(half_ends=False)
    q = np.maximum(r.points, 1e-30)
    return d / q


__all__ = ["TRanges", "dlnk_from_points"]
