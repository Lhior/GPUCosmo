"""Base classes for user likelihoods.

A likelihood has two parts, matching how the GPU bundles split the work:

* ``theory(cosmo)`` — quantities that depend on the cosmology only (distances, a model
  vector, spectra you compute yourself). Exact evaluations call it once per cosmology; the
  DALI expansions expand its outputs together with the built-in theory, so a costly
  ``theory`` costs nothing per DALI evaluation.
* ``loglike(inp)`` — the log-likelihood from those outputs and the likelihood's own
  nuisance parameters, evaluated at every point (keep it cheap).

Both run inside ``jax.jit`` / ``jax.vmap`` / ``jax.grad``: write them with ``jax.numpy``,
fixed array shapes and no Python control flow on traced values. See
``docs/adding_likelihoods.rst`` for the full guide.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np

C_KM_S = 299792.458


@dataclass(frozen=True)
class Param:
    """A sampled nuisance parameter: uniform box [lo, hi], starting value ``ref``
    (default: the box centre or the Gaussian mean) and an optional Gaussian prior
    ``gauss = (mean, sd)``. A Cobaya yaml's ``params`` block overrides these defaults."""

    lo: float
    hi: float
    ref: float | None = None
    gauss: tuple[float, float] | None = None
    latex: str | None = None

    def __post_init__(self):
        if not self.lo < self.hi:
            raise ValueError(f"Param box [{self.lo}, {self.hi}] is empty")

    @property
    def start(self) -> float:
        if self.ref is not None:
            return float(self.ref)
        if self.gauss is not None:
            return float(self.gauss[0])
        return 0.5 * (float(self.lo) + float(self.hi))


class Cosmology:
    """Background cosmology at one parameter point (traceable: use inside ``theory`` and
    ``loglike``). Distances in Mpc, H in km/s/Mpc; ``z`` may be a scalar or an array.

    Parameters (attributes): ``h``, ``H0``, ``ombh2``, ``omch2``, ``ns``, ``As``, ``logA``,
    ``tau``, ``w0``, ``wa``, ``mu0``, ``Sigma0`` (None when not sampled), and the extended
    ``mnu``, ``nnu``, ``omk``, ``meffsterile`` (CAMB's defaults when not sampled). w0waCDM with
    curvature and the analysis' neutrino model (``tables.model``), as the built-in theory.
    """

    def __init__(self, params: dict, tables):
        from gpucosmo.jax.background.cosmology import BackgroundParams, background_state

        self.params = params
        self._tables = tables
        self._bp = background_state(tables, BackgroundParams.build(
            params["ombh2"], params["omch2"], 100.0 * params["h"], params["w0"], params["wa"],
            ext=params.get("ext")))

    def __getattr__(self, name):
        p = self.__dict__.get("params", {})
        if name in p:
            return p[name]
        if name == "H0":
            return 100.0 * p["h"]
        if name == "logA":
            return jnp.log(p["As"] * 1e10)
        if name in ("mnu", "nnu", "omk", "meffsterile"):
            return getattr(self.__dict__["_bp"], name)
        raise AttributeError(name)

    @property
    def omnuh2(self):
        """Massive-neutrino density today, ω_ν (CAMB ``omnuh2``)."""
        return self._bp.omnuh2

    @property
    def Omega_m(self):
        """Total matter density today (baryons + CDM + massive neutrinos); the sampled value in
        bases that sample it (DES, HSC)."""
        if "Omega_m" in self.params:
            return self.params["Omega_m"]
        return (self.ombh2 + self.omch2 + self._bp.omnuh2) / self.h**2

    @property
    def Omega_k(self):
        return self._bp.omk

    def H(self, z):
        """Hubble rate H(z) [km/s/Mpc]."""
        from gpucosmo.jax.background.cosmology import hubble_km_s_mpc

        return hubble_km_s_mpc(self._tables, self._bp, jnp.asarray(z, dtype=jnp.float64))

    def comoving_distance(self, z):
        """Line-of-sight comoving distance D_C(z) [Mpc], z ≲ 10."""
        from gpucosmo.jax.background.cosmology import comoving_distance

        z = jnp.asarray(z, dtype=jnp.float64)
        return comoving_distance(self._tables, self._bp, z).reshape(z.shape)

    def transverse_comoving_distance(self, z):
        """D_M(z) = f_K(D_C) [Mpc] (= D_C when flat)."""
        from gpucosmo.jax.background.cosmology import transverse_distance

        return transverse_distance(self._tables, self._bp, self.comoving_distance(z))

    def angular_diameter_distance(self, z):
        """D_A(z) = D_M / (1 + z) [Mpc]."""
        return self.transverse_comoving_distance(z) / (1.0 + jnp.asarray(z, dtype=jnp.float64))

    def luminosity_distance(self, z):
        """D_L(z) = (1 + z) D_M [Mpc]."""
        return self.transverse_comoving_distance(z) * (1.0 + jnp.asarray(z, dtype=jnp.float64))

    def hubble_distance(self, z):
        """D_H(z) = c / H(z) [Mpc]."""
        return C_KM_S / self.H(z)

    def volume_distance(self, z):
        """D_V(z) = [z D_M² D_H]^(1/3) [Mpc]."""
        z = jnp.asarray(z, dtype=jnp.float64)
        return (z * self.transverse_comoving_distance(z) ** 2 * self.hubble_distance(z)) ** (1.0 / 3.0)

    @property
    def rdrag(self):
        """Sound horizon at the drag epoch r_d [Mpc] (CAMB ``rdrag``)."""
        from gpucosmo.jax.background.cosmology import rs_drag

        return rs_drag(self._tables, self._bp)

    @property
    def theta_star(self):
        """θ* = r_s(z*) / D_M(z*)."""
        from gpucosmo.jax.background.cosmology import theta_star

        return theta_star(self._tables, self._bp)


class LikelihoodInputs(NamedTuple):
    """What ``Likelihood.loglike`` receives.

    ``theory``: the output of ``Likelihood.theory`` at this point (exact, or the
    DALI expansion's); None if the likelihood defines no theory. ``params``: {name: value} of the
    likelihood's own parameters. ``cosmo``: ``Cosmology`` at this point (exact background).
    ``shared``: the built-in theory blocks named in ``Likelihood.requires`` —
    ``shared["cmb"]`` holds the lensed CMB ``dl_tt``, ``dl_te``, ``dl_ee`` (D_ℓ in μK²,
    ℓ = 0 … 2508; plus ``dl_kk`` or ``cmb_ext`` with CMB lensing) and, when ``requirements()``
    asks for more, ``cl`` = {"ell", "tt", "te", "ee", "bb" (lensed D_ℓ, μK²), "pp"
    ([ℓ(ℓ+1)]² C_ℓ^φφ/2π), "kk" (C_ℓ^κκ)} to the requested ℓ; ``shared["fs"]`` the DESI
    full-shape block.
    """

    theory: Any
    params: dict
    cosmo: Cosmology
    shared: dict


class Likelihood:
    """Base class of user likelihoods. Subclass it, then make it known with
    ``@register_likelihood`` (or a ``gpucosmo.likelihoods`` entry point, or name its import
    path in the yaml).

    Class attributes:

    * ``name`` — component label and Cobaya-yaml key (default: the class name);
    * ``aliases`` — further yaml keys that select this class;
    * ``params`` — {name: Param} nuisance parameters sampled with it (names must be unique
      across the analysis);
    * ``requires`` — built-in theory blocks ``loglike`` reads (``"cmb"``, ``"fs"``). "cmb" is
      computed on demand (no built-in CMB likelihood needed); "fs" needs the DESI full-shape
      likelihood in the analysis. Options per block (e.g. the CMB ℓ range and spectra)
      come from ``requirements()``;
    * ``differentiable`` — False for likelihoods JAX cannot differentiate (host callbacks, such
      as the Cobaya bridge): the pipelines evaluate them exactly where values suffice and use
      finite differences or a fitted quadratic where gradients are needed;
    * ``allow_shared_params`` — True lets parameters with names already in the analysis be
      shared (Cobaya semantics) instead of raising.

    ``__init__(**options)`` receives the options of the yaml block (or Python keywords):
    load data there, as NumPy arrays.
    """

    name: str | None = None
    aliases: tuple[str, ...] = ()
    params: dict[str, Param] = {}
    requires: tuple[str, ...] = ()
    differentiable: bool = True
    allow_shared_params: bool = False
    # Extended parameters (``mnu``, ``nnu``, ``omk``, ``meffsterile``) that the likelihood's own
    # ``theory`` holds fixed: an analysis sampling or changing one of them is refused.
    fixed_ext: tuple[str, ...] = ()
    # (hierarchy, num_massive) of the neutrinos the own ``theory`` uses, when it has its own; an
    # analysis changing Σm_ν with another ``NeutrinoModel`` is refused.
    neutrino_model: tuple[str, int] | None = None

    def __init__(self, **options):
        self.options = dict(options)
        if "name" in options:
            self.name = str(options["name"])

    # ---- the interface ----
    def theory(self, cosmo: Cosmology):
        """Cosmology-only outputs (a dict of arrays with fixed shapes), or None."""
        return None

    def loglike(self, inp: LikelihoodInputs):
        """log L at one point (scalar)."""
        raise NotImplementedError(f"{type(self).__name__}.loglike")

    def requirements(self) -> dict:
        """Theory blocks with options: {"cmb": {"lmax": ℓ_max, "spectra": ["tt", "te", "ee", "bb",
        "pp"]}, "fs": {}}. Default: the ``requires`` blocks with the built-ins' defaults (CMB
        TT/TE/EE to ℓ 2508). Asking for ℓ > 2508 or BB / φφ adds ``LikelihoodInputs.shared["cmb"]["cl"]``."""
        return {r: {} for r in self.requires}

    def log_prior(self, params: dict):
        """Extra prior terms on the likelihood's parameters beyond the per-parameter
        boxes / Gaussians of ``params`` (e.g. a correlated prior). Default 0. The values may
        carry leading batch axes (the samplers evaluate priors on batches of points): write
        it elementwise."""
        return 0.0

    # ---- helpers ----
    @property
    def label(self) -> str:
        return self.name or type(self).__name__

    def has_theory(self) -> bool:
        return type(self).theory is not Likelihood.theory

    def has_log_prior(self) -> bool:
        return type(self).log_prior is not Likelihood.log_prior

    def describe(self) -> dict:
        """Summary written to ``config.updated.*.yaml``."""
        return {"name": self.label, "class": f"{type(self).__module__}.{type(self).__qualname__}",
                "params": list(self.params), "requires": list(self.requires)}

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.label!r}>"


class GaussianLikelihood(Likelihood):
    """χ² = (d − m)ᵀ C⁻¹ (d − m) for a data vector ``d`` and a model vector ``m``.

    Implement ``model(cosmo)`` (cosmology only; expanded by the DALI expansions) and pass
    ``data`` with ``cov`` (or ``precision``) to ``__init__``, e.g. from a subclass that
    reads them from a file. Override ``vector(model, params)`` to apply nuisance
    parameters to the model (e.g. a calibration); while ``vector`` is not overridden the
    compressed DALI likelihood folds the covariance into the expansion coefficients (exact, no
    n × n product per point).
    """

    def __init__(self, data=None, cov=None, precision=None, **options):
        super().__init__(**options)
        if data is None:
            raise ValueError(f"{type(self).__name__}: no data vector")
        self.data = np.atleast_1d(np.asarray(data, dtype=np.float64))
        n = self.data.size
        if precision is None:
            if cov is None:
                raise ValueError(f"{type(self).__name__}: give cov or precision")
            cov = np.asarray(cov, dtype=np.float64)
            cov = np.diag(cov) if cov.ndim == 1 else cov.reshape(n, n)
            precision = np.linalg.inv(cov)
        precision = np.asarray(precision, dtype=np.float64).reshape(n, n)
        self.precision = 0.5 * (precision + precision.T)
        # Lᵀ with P = L Lᵀ: χ² = |Lᵀ (d − m)|²
        self.whitening = np.linalg.cholesky(self.precision).T

    def model(self, cosmo: Cosmology):
        """Model vector (length ``data.size``) from the cosmology."""
        raise NotImplementedError(f"{type(self).__name__}.model")

    def vector(self, model, params: dict):
        """Model vector with the nuisance parameters applied (default: unchanged)."""
        return model

    def theory(self, cosmo: Cosmology):
        return {"model": jnp.atleast_1d(self.model(cosmo))}

    def loglike(self, inp: LikelihoodInputs):
        r = jnp.asarray(self.whitening) @ (jnp.asarray(self.data) - self.vector(inp.theory["model"], inp.params))
        return -0.5 * r @ r

    def compressible(self) -> bool:
        """True if the compressed DALI likelihood may use the whitened-table form (the χ² is a
        quadratic in the model alone)."""
        cls = type(self)
        return (cls.vector is GaussianLikelihood.vector and cls.loglike is GaussianLikelihood.loglike
                and cls.theory is GaussianLikelihood.theory)


__all__ = ["C_KM_S", "Cosmology", "GaussianLikelihood", "Likelihood", "LikelihoodInputs", "Param"]
