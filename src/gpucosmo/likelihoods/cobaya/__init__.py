"""Native (JAX) ports of Cobaya likelihoods, registered under their Cobaya names.

A yaml ``likelihood:`` key such as ``bao.desi_dr2`` selects the port when one exists (the
resolver tries ports before the Cobaya bridge), so the likelihood runs jitted, vmapped and
differentiated on the GPU. Each port class lists the Cobaya likelihoods it serves in
``cobaya_names``; ``PORTS`` below are the modules that define them (imported on first use).
See ``base.CobayaPort`` for how ports are written.
"""

from __future__ import annotations

import importlib

# module → port classes (each registered under its ``cobaya_names``)
PORTS = {
    "gpucosmo.likelihoods.cobaya.background": ("BAO", "H0", "Mb"),
    "gpucosmo.likelihoods.cobaya.supernovae": ("SN",),
    "gpucosmo.likelihoods.cobaya.cmblikes": ("CMBLikes", "PR4Lensing"),
    "gpucosmo.likelihoods.cobaya.planck_python": ("CamSpec", "PlikLite", "LowlTT", "LowlEE"),
    "gpucosmo.likelihoods.cobaya.planck_clik": ("PlikClik", "LowlClik", "LensingClik"),
    "gpucosmo.likelihoods.cobaya.act_lens": ("ACTDR6LensLike",),
}
_DONE = False


def register_ports() -> dict:
    """Register every port class; returns {Cobaya name: class}."""
    global _DONE
    from gpucosmo.likelihoods.registry import register_likelihood

    out = {}
    for mod, classes in PORTS.items():
        try:
            m = importlib.import_module(mod)
        except ModuleNotFoundError as e:
            if e.name == mod:  # family not written (yet)
                continue
            import warnings

            warnings.warn(f"native Cobaya ports in {mod} unavailable: {e!r}", stacklevel=2)
            continue
        except ImportError as e:  # an optional dependency of one family must not hide the others
            import warnings

            warnings.warn(f"native Cobaya ports in {mod} unavailable: {e!r}", stacklevel=2)
            continue
        for cname in classes:
            cls = getattr(m, cname, None)
            if cls is None:
                continue
            if not _DONE:
                register_likelihood(cls, name=cls.cobaya_names[0], aliases=tuple(cls.cobaya_names[1:]))
            out.update(dict.fromkeys(cls.cobaya_names, cls))
    _DONE = True
    return out


def ported() -> dict:
    """{Cobaya name: port class}."""
    return register_ports()


__all__ = ["PORTS", "ported", "register_ports"]
