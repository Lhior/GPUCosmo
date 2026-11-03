#!/usr/bin/env python3
"""Optional clipy low-ell Planck likelihood wrapper.

Commander TT + SimAll EE via ``clipy.clik_candl``. Call
:func:`ensure_clipy_jax` (through :func:`load_clipy_lowl`) before use.

Jitting under ``jax_enable_x64=True``: works on CPU (Commander/SimAll return
finite logL when wrapped in ``jax.jit``). On GPU, prefer the int32 patches in
``clipy_jax_compat``; an extra outer ``jit`` on top of clipy's own JAX path can
still produce NaNs depending on the platform — call ``log_like`` without a
second outer jit if that happens.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import jax.numpy as jnp

from gpucosmo.jax.cmb.clipy_jax_compat import ensure_clipy_jax


@dataclass(frozen=True)
class ClipyLowl:
    """clipy clik_candl instances for low-ell TT and EE (if data present)."""

    tt_like: object | None
    ee_like: object | None
    lmax: int

    @property
    def active(self) -> bool:
        return self.tt_like is not None or self.ee_like is not None


def _default_lowl_dir(packages_path: str | None) -> Path | None:
    env = os.environ.get("GPUCOSMO_CLIPY_LOWL_DIR")
    if env:
        return Path(env)
    if packages_path is None:
        from gpucosmo.paths import packages_path as _packages_path

        packages_path = _packages_path()
        if packages_path is None:
            return None
    # Full baseline package layout (commander + simall siblings)
    baseline = (
        Path(packages_path)
        / "data"
        / "planck_2018"
        / "baseline"
        / "plc_3.0"
        / "low_l"
    )
    if baseline.is_dir():
        return baseline
    cand = Path(packages_path) / "data" / "planck_2018_lowl" / "clik"
    return cand if cand.is_dir() else None


def without_internal_priors(like):
    """Drop the Gaussian priors clipy adds inside ``log_like`` (``all_priors=True``) and keep
    the parameter values it fixes. The bundle applies every parameter prior itself (from the
    yaml, as Cobaya does, or ``params.PLIK_2018_PRIORS`` / ``SZ_PRIOR``); with clipy's copies
    they counted twice — e.g. A_planck: posterior σ 0.0017 instead of the official chains'
    0.0023, and Commander TT off the official χ² column by ((A_planck − 1)/0.0025)² (0.02 after)."""
    from collections import OrderedDict

    like._prior = OrderedDict()
    return like


def load_clipy_lowl(packages_path: str | None = None) -> ClipyLowl:
    """Load commander TT + simall EE clipy likelihoods when clik files exist."""
    root = _default_lowl_dir(packages_path)
    if root is None:
        return ClipyLowl(tt_like=None, ee_like=None, lmax=29)

    ensure_clipy_jax()
    import clipy

    tt_path = root / "commander" / "commander_dx12_v3_2_29.clik"
    ee_path = root / "simall" / "simall_100x143_offlike5_EE_Aplanck_B.clik"
    if not tt_path.is_dir():
        tt_path = root / "planck2018_lowl_TT.clik"
    if not ee_path.is_dir():
        ee_path = root / "planck2018_lowl_EE.clik"
    tt_like = without_internal_priors(clipy.clik_candl(str(tt_path), all_priors=True)) if tt_path.is_dir() else None
    ee_like = without_internal_priors(clipy.clik_candl(str(ee_path), all_priors=True)) if ee_path.is_dir() else None
    lmax = 29
    if tt_like is not None:
        lmax = max(lmax, int(max(tt_like.lmax)))
    if ee_like is not None:
        lmax = max(lmax, int(max(ee_like.lmax)))
    return ClipyLowl(tt_like=tt_like, ee_like=ee_like, lmax=lmax)


def _lowl_params(
    dl_tt: jnp.ndarray,
    dl_ee: jnp.ndarray,
    a_planck: jnp.ndarray,
    lmax: int,
    *,
    tt: bool,
    ee: bool,
) -> dict:
    params: dict = {"A_planck": a_planck}
    dl: dict = {}
    if tt:
        dl["TT"] = dl_tt[2 : lmax + 1]
    if ee:
        dl["EE"] = dl_ee[2 : lmax + 1]
    if dl:
        params["Dl"] = dl
    return params


class _SmoothSimall:
    """C¹ twin of clipy's SimAll EE likelihood (``clipy/simall.py``).

    SimAll reads its log-probability table with ``floor(C_ℓ ℓ(ℓ+1)/step)``: the
    value is piecewise constant and its gradient is identically zero, so
    gradient-based samplers never see the low-ℓ EE (τ) constraint. Here each
    table entry is placed at its bin centre and read with Catmull-Rom cubic
    interpolation: equal to the official value to within the in-bin variation
    (≲1e-4 per multipole for the Planck 2018 table, step 1e-4 μK²) and
    differentiable. Intended for DALI/gradient paths; keep the official
    stepwise likelihood as the exact target.
    """

    def __init__(self, internal):
        self._s = internal

    def __getattr__(self, name):
        return getattr(self._s, name)

    @staticmethod
    def _read(prob, u):
        n = prob.shape[1]
        t = u - 0.5
        i = jnp.clip(jnp.floor(t), 1, n - 3)
        f = t - i
        i = i.astype(jnp.int32)
        rows = jnp.arange(prob.shape[0])
        p0, p1, p2, p3 = (prob[rows, i + k] for k in (-1, 0, 1, 2))
        return (p1 + 0.5 * f * (p2 - p0)
                + f**2 * (p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3)
                + f**3 * (-0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3))

    def __call__(self, cls, nuisance_dict, chi2=False):
        s = self._s
        cls = s._calib(cls, nuisance_dict)
        res = 0.0
        for k, name in ((1, "EE"), (2, "BB"), (3, "TE")):
            if s.has_cl[k]:
                prob = getattr(s, "prob" + name)
                step = getattr(s, "step" + name)
                res = res + jnp.sum(self._read(prob, cls[k, s.lmin:] * s.llp1 / step))
        return res


def smooth_simall(ee_like):
    """Shallow copy of a clipy SimAll ``clik_candl`` using ``_SmoothSimall``."""
    # Manual shallow copy: clik_candl.__getattr__ forwards to _internal, which
    # sends copy.copy into infinite recursion on a half-built instance.
    out = object.__new__(type(ee_like))
    out.__dict__.update(ee_like.__dict__)
    out._internal = _SmoothSimall(ee_like.__dict__["_internal"])
    return out


def build_clipy_lowl_loglike(
    lowl: ClipyLowl,
    smooth_ee: bool = False,
) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """loglike(dl_tt, dl_te, dl_ee, A_planck) — returns 0 if clipy data missing.

    ``smooth_ee``: read the SimAll EE table with ``_SmoothSimall`` (differentiable)
    instead of clipy's stepwise lookup.
    """
    if not lowl.active:

        def _zero(*_a, **_k):
            return jnp.asarray(0.0)

        return _zero

    tt_like = lowl.tt_like
    ee_like = lowl.ee_like
    if smooth_ee and ee_like is not None:
        ee_like = smooth_simall(ee_like)
    lmax = lowl.lmax

    def loglike(
        dl_tt: jnp.ndarray,
        dl_te: jnp.ndarray,
        dl_ee: jnp.ndarray,
        a_planck: jnp.ndarray,
    ) -> jnp.ndarray:
        llike = jnp.asarray(0.0)
        if tt_like is not None:
            params = _lowl_params(dl_tt, dl_ee, a_planck, lmax, tt=True, ee=False)
            llike = llike + tt_like.log_like(params)
        if ee_like is not None:
            params = _lowl_params(dl_tt, dl_ee, a_planck, lmax, tt=False, ee=True)
            llike = llike + ee_like.log_like(params)
        return llike

    return loglike
