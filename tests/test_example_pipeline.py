#!/usr/bin/env python3
"""The command-line evidence pipeline on examples/custom_likelihood (user likelihoods via --plugin)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "examples" / "custom_likelihood"


@pytest.mark.slow
def test_example_evidence_pipeline(tmp_path):
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(ROOT / "src")}
    common = ["--config", str(EX / "analysis.yaml"), "--plugin", str(EX / "my_likelihoods.py"), "--outdir", str(tmp_path)]
    hmc = [sys.executable, "-m", "gpucosmo.cli.dali_hmc", *common, "--no-autosize"]
    ns = [sys.executable, "-m", "gpucosmo.cli.dali_ns", *common, "--tag", "smc", "--compressed-dali"]
    for cmd in (hmc + ["--stage", "map"], hmc + ["--stage", "build", "--expansion-only"],
                ns + ["--stage", "ns", "--smc-particles", "1024", "--smc-runs", "4", "--smc-ess", "0.8",
                      "--smc-mcmc", "3"]):
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-4000:]
    out = json.loads((tmp_path / "smc.ns_summary.json").read_text())
    assert out["dali_order"] == 3
    assert abs(out["logz_dali"] - (-14.75)) < 0.15  # 4 × 1024 particles: σ ≈ 0.03
    oc = out["order_check"]  # triplet vs doublet: converged
    assert oc["ess_fraction"] > 0.9 and abs(oc["delta_logz"]) < 0.05 and oc["max_abs_mean_shift_sigma"] < 0.1
    resolved = (tmp_path / "config.updated.smc.ns.yaml").read_text()
    assert "cosmic_chronometers" in resolved and "bao_distances" in resolved


@pytest.mark.slow
def test_evidence_runner_on_the_example(tmp_path):
    """gpucosmo-evidence: all stages in one command (exact stages in one process, SMC in a
    subprocess), with the error target; resumes without redoing finished stages."""
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(ROOT / "src")}
    cmd = [sys.executable, "-m", "gpucosmo.cli.evidence", "--config", str(EX / "analysis.yaml"),
           "--plugin", str(EX / "my_likelihoods.py"), "--outdir", str(tmp_path), "--smc-target-err", "0.02",
           "--ns-args", "--smc-particles 1024 --smc-pilot-particles 1024"]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=2400)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    ev = json.loads((tmp_path / "smc.evidence.json").read_text())
    assert set(ev["stage_wall_sec"]) == {"map", "build", "ns"}
    assert ev["dali_order"] == 3 and abs(ev["logz_dali"] - (-14.75)) < 0.1
    assert abs(ev["order_check"]["delta_logz"]) < 0.05
    r2 = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)  # everything exists: no stage reruns
    assert r2.returncode == 0 and json.loads((tmp_path / "smc.evidence.json").read_text())["stage_wall_sec"] == {}
