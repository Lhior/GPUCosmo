"""Cobaya interoperability.

* ``bridge.CobayaLikelihood`` — any Cobaya likelihood evaluated (through Cobaya, on the host)
  on the GPU theory's products, usable in every gpucosmo pipeline;
* ``provider.BridgeTheory`` — the Cobaya theory component serving those products.

Submodules import Cobaya; nothing is imported until used.
"""

from __future__ import annotations

__all__ = ["BridgeTheory", "BridgeUnsupported", "CobayaLikelihood"]


def __getattr__(name):
    if name in ("CobayaLikelihood", "BridgeUnsupported"):
        from gpucosmo.cobaya_bridge import bridge

        return getattr(bridge, name)
    if name == "BridgeTheory":
        from gpucosmo.cobaya_bridge.provider import BridgeTheory

        return BridgeTheory
    raise AttributeError(name)
