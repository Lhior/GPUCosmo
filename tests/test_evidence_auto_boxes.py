"""``gpucosmo-evidence --auto-boxes``: uniform-prior boxes from a Laplace approximation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gpucosmo.cli.evidence import sampling_boxes_from_moments


def _p(lo, hi, gauss=None):
    return SimpleNamespace(lo=lo, hi=hi, gauss=gauss)


def test_boxes_narrow_only_uniform_priors_and_stay_inside_them():
    priors = {"theta": _p(0.5, 10.0), "ns": _p(0.8, 1.2), "tau": _p(0.01, 0.8), "A": _p(0.0, 2.0, gauss=(1.0, 0.1)),
              "wide": _p(-1.0, 1.0)}
    names = ["theta", "ns", "tau", "A", "wide", "fixed"]
    mean = [1.041, 0.97, 0.06, 1.0, 0.0, 3.0]
    sd = [3e-4, 0.004, 0.006, 0.01, 0.5, 1.0]
    b = sampling_boxes_from_moments(priors, names, mean, sd, 30.0)
    assert set(b) == {"theta", "ns", "tau"}  # Gaussian prior, already-narrow and unknown parameters untouched
    assert b["theta"] == pytest.approx((1.032, 1.05))
    assert b["ns"] == pytest.approx((0.85, 1.09))
    assert b["tau"] == pytest.approx((0.01, 0.24))  # clipped at the prior's lower edge


def test_degenerate_widths_are_skipped():
    assert sampling_boxes_from_moments({"x": _p(0.0, 1.0)}, ["x"], [0.5], [0.0], 30.0) == {}
