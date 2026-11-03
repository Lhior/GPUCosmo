#!/usr/bin/env python3
"""clipy full plik (TTTEEE) high-ell likelihood wrapper."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp

from gpucosmo.jax.cmb.clipy_jax_compat import ensure_clipy_jax

_DEFAULT_REL = (
    "baseline/plc_3.0/hi_l/plik/plik_rd12_HM_v22b_TTTEEE.clik"
)


@dataclass(frozen=True)
class ClipyPlik:
    """Loaded clipy clik_candl instance for high-ell plik TTTEEE."""

    like: object
    nuisance_names: tuple[str, ...]
    lmax: int
    path: str


def default_plik_clik_path(packages_path: str | None = None) -> Path:
    env = os.environ.get("GPUCOSMO_PLIK_CLIK")
    if env:
        return Path(env)
    from gpucosmo.paths import resolve_packages_path

    return Path(resolve_packages_path(packages_path)) / "data" / "planck_2018" / _DEFAULT_REL


def load_clipy_plik(
    packages_path: str | None = None,
    *,
    clik_path: str | Path | None = None,
) -> ClipyPlik | None:
    """Load full plik TTTEEE via clipy if the .clik directory exists."""
    path = Path(clik_path) if clik_path else default_plik_clik_path(packages_path)
    if not path.is_dir():
        return None
    ensure_clipy_jax()
    import clipy

    from gpucosmo.jax.cmb.clipy_likelihood import without_internal_priors

    # all_priors=True for clipy's fixed baseline values (cib_index, galf_EE_*, calib_*P, ...);
    # its Gaussian priors (gal545, galf_TE, calib_100T/217T, SZ, A_planck) are dropped: the
    # bundle applies the same ones (``params.PLIK_2018_PRIORS``, ``SZ_PRIOR`` or the yaml).
    like = without_internal_priors(clipy.clik_candl(str(path), all_priors=True))
    nuis = tuple(like.extra_parameter_names)
    lmax = int(max(like.lmax))
    return ClipyPlik(like=like, nuisance_names=nuis, lmax=lmax, path=str(path))


def build_clipy_plik_loglike(
    plik: ClipyPlik,
) -> Callable[..., jnp.ndarray]:
    """loglike(dl_tt, dl_te, dl_ee, nuisance_vector) with nuisances in plik.nuisance_names order."""
    like = plik.like
    names = plik.nuisance_names
    n_nuis = len(names)
    lmax = plik.lmax

    def loglike(
        dl_tt: jnp.ndarray,
        dl_te: jnp.ndarray,
        dl_ee: jnp.ndarray,
        nuis: jnp.ndarray,
    ) -> jnp.ndarray:
        params: dict = {
            "Dl": {
                "TT": dl_tt[2 : lmax + 1],
                "TE": dl_te[2 : lmax + 1],
                "EE": dl_ee[2 : lmax + 1],
            }
        }
        for i, name in enumerate(names):
            params[name] = nuis[..., i] if nuis.ndim else nuis[i]
        return like.log_like(params)

    loglike_jit = jax.jit(loglike)

    loglike_jit.n_nuisance = n_nuis  # type: ignore[attr-defined]
    loglike_jit.nuisance_names = names  # type: ignore[attr-defined]
    return loglike_jit
