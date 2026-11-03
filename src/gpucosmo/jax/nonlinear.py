#!/usr/bin/env python3
"""Nonlinear matter-power prescriptions for the JAX GGL theory path.

The option string ``nonlinear_prescription`` means the same model on the CCL
(:class:`gpucosmo.ccl.theory.GGLTheory`) and JAX (``boltzmann_backend='ccl'``)
sides of the gG/gI pipelines:

- ``'halofit'``: CCL's Takahashi (2012) Halofit applied to the linear CAMB
  P(k) (``matter_power_spectrum='halofit'``). On the JAX side the same
  algorithm runs on device (:mod:`gpucosmo.jax.halofit`, a CCL-exact port).
- ``'hmcode'``: CAMB HMcode-2020 *without* baryonic feedback,
  ``matter_power_spectrum='camb'`` with
  ``extra_parameters={'camb': {'halofit_version': 'mead2020'}}``
  (:data:`GGL_HMCODE_CAMB_VERSION`). CCL's ``'halofit'`` ignores
  ``halofit_version``, so HMcode needs ``matter_power_spectrum='camb'``.

The DES w2x2pt / hybrid pipelines use ``'mead2020_feedback'`` with an explicit
``HMCode_logT_AGN`` instead; they pass ``hmcode_version`` explicitly.
"""

from __future__ import annotations

from typing import Literal

NonlinearPrescription = Literal["hmcode", "halofit"]
DEFAULT_NONLINEAR_PRESCRIPTION: NonlinearPrescription = "hmcode"

# CAMB ``halofit_version`` behind ``nonlinear_prescription='hmcode'`` (gG/gI).
GGL_HMCODE_CAMB_VERSION = "mead2020"


def ccl_nonlinear_config(
    prescription: str,
    *,
    hmcode_version: str = GGL_HMCODE_CAMB_VERSION,
    hmcode_logT_AGN: float | None = None,
) -> tuple[str, dict | None]:
    """``(matter_power_spectrum, extra_parameters)`` for ``pyccl.Cosmology``.

    ``hmcode_logT_AGN`` is only forwarded for ``hmcode_version=
    'mead2020_feedback'`` (it has no effect on the other CAMB versions).
    """
    p = str(prescription).strip().lower()
    if p == "halofit":
        return "halofit", None
    if p == "linear":
        return "linear", None
    if p != "hmcode":
        raise ValueError(
            f"nonlinear_prescription must be 'hmcode' or 'halofit', got {prescription!r}"
        )
    camb_extra: dict = {"halofit_version": str(hmcode_version)}
    if hmcode_version == "mead2020_feedback" and hmcode_logT_AGN is not None:
        camb_extra["HMCode_logT_AGN"] = float(hmcode_logT_AGN)
    return "camb", {"camb": camb_extra}


__all__ = [
    "DEFAULT_NONLINEAR_PRESCRIPTION",
    "GGL_HMCODE_CAMB_VERSION",
    "NonlinearPrescription",
    "ccl_nonlinear_config",
]
