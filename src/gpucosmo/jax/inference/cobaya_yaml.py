"""Cobaya yaml input for the μ–Σ / DESI DR1 full-shape bundles (``inference.mg_bundle``).

``resolve_cobaya(info)`` translates a Cobaya input (or ``updated``) yaml, e.g. the
official DESI DR1 ``chain.input.yaml`` files, into an ``MGConfig`` plus a
per-parameter prior table for ``build_mg_bundle``. Everything the yaml states
is either reproduced or rejected: unknown likelihoods, unsupported theory
options, parameters our solver cannot take (e.g. Ω_k with CMB likelihoods before the
curved perturbation solver), prior densities other than uniform/normal, and ``prior:`` lambdas
other than the MG stability cut, Planck's SZ prior and the DES basis's Ω_b h² cut all raise
``UnsupportedCobayaInput``. Accuracy settings (``AccuracyBoost``,
``lens_potential_accuracy``, ...) and the ``sampler`` block have no counterpart
and are listed in ``ResolvedCobaya.notes``.

``audit_bundle(bundle, resolved)`` diffs a built bundle (parameters, boxes,
Gaussian priors, fixed values, likelihood options) against a resolved yaml.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml


class UnsupportedCobayaInput(ValueError):
    """The yaml asks for something the GPU pipeline does not reproduce."""


# Likelihood keys (exact Cobaya names, or the DESI bindings' dotted path suffix).
_LIKES = {
    "desi_fs": ("fs_bao_likelihoods.desi_fs_bao_all", "desi_fs_bao_all"),
    "bbn": ("bbn_likelihoods.schoneberg2024",),
    "ns10": ("cmb_likelihoods.ns10_planck2018",),
    "lowl_tt": ("planck_2018_lowl.TT_clik", "planck_2018_lowl.TT"),
    "lowl_ee": ("planck_2018_lowl.EE_clik", "planck_2018_lowl.EE"),
    "plik": ("planck_2018_highl_plik.TTTEEE",),
    "hillipop": ("planck_2020_hillipop.TTTEEE",),
    "lollipop": ("planck_2020_lollipop.lowlE",),
    "act_lens": ("act_dr6_lenslike.ACTDR6LensLike", "ACTDR6LensLike"),
    # DESI's internal DES-Y5 binding: same data, M_B marginalized analytically (as ours).
    "sn:desy5": ("sn.desy5", "sn_likelihoods_internal.desy5"),
    "sn:pantheonplus": ("sn.pantheonplus",),
    "sn:union3": ("sn.union3",),
}
# The DESI DR1 FS+BAO likelihood we port (fullshape.desi_dr1): window-convolved
# P0/P2 at 0.02 < k < 0.2 h/Mpc for 6 tracers + post-recon BAO, Lyα BAO, analytic
# marginalization of counterterms / stochastic terms, physical bias priors.
_FS_TRACERS = {"bgs_z0", "lrg_z0", "lrg_z1", "lrg_z2", "elg_z1", "qso_z0", "lya_z0"}
_FS_OBSERVABLES = {"power+bao-recon", "spectrum-poles-rotated+bao-recon"}
# Extended parameters and the solver's defaults (CAMB's): sampled or fixed away from these through
# ``MGConfig.ext_sampled`` / ``ext_fixed``.
SOLVER_FIXED = {"mnu": 0.06, "nnu": 3.044, "omk": 0.0, "meffsterile": 0.0}
# Planck plik_rd12_HM_v22b_TTTEEE parameters clipy holds fixed (``like._default``).
PLIK_FIXED = {
    "cib_index": -1.3, "galf_EE_A_100": 0.055, "galf_EE_A_100_143": 0.04,
    "galf_EE_A_100_217": 0.094, "galf_EE_A_143": 0.086, "galf_EE_A_143_217": 0.21,
    "galf_EE_A_217": 0.7, "galf_EE_index": -2.4, "galf_TE_index": -2.4, "calib_100P": 1.021,
    "calib_143P": 0.966, "calib_217P": 1.04, "A_pol": 1.0,
    **{f"A_cnoise_e2e_{f}_{f}_EE": 1.0 for f in (100, 143, 217)},
    **{f"A_sbpx_{a}_TT": 1.0 for a in ("100_100", "143_143", "143_217", "217_217")},
    **{f"A_sbpx_{a}_EE": 1.0 for a in ("100_100", "100_143", "100_217", "143_143", "143_217",
                                        "217_217")},
}
# HiLLiPoP terms our port does not model; only their null value is reproducible.
HILLIPOP_ABSENT = {"AsyncT": 0.0, "AsyncP": 0.0}
_MG_PRIOR = "lambdamu0,Sigma0:(0ifmu0<2*Sigma0+1else-np.inf)"
_SZ_PRIOR = "lambdaksz_norm,A_sz:stats.norm.logpdf(ksz_norm+1.6*A_sz,loc=9.5,scale=3.0)"
# DES basis: Ω_b h² ≤ c (``MGConfig.ombh2_max``)
_OMBH2_CUT = re.compile(r"lambdaOmega_b,h:\(?0ifOmega_b\*h\*\*2<=([0-9.eE+-]+)else-np\.inf\)?")
# Gaussian priors without bounds get a ±GAUSS_BOX σ box (the sampler needs finite ranges).
GAUSS_BOX = 10.0
_NOTE_THEORY = {"AccuracyBoost", "lSampleBoost", "lAccuracyBoost", "lens_potential_accuracy",
                "lens_margin", "lmax", "theta_H0_range"}
# CAMB halofit_version → our HMcode port ("mead" is CAMB's alias of mead2016); CAMB's default
# (absent key) is mead2020. Only used when the CMB is lensed with the non-linear φφ.
_HALOFIT = {"mead2020": "mead2020", "mead2016": "mead2016", "mead": "mead2016"}


@dataclass(frozen=True)
class ParamPrior:
    lo: float
    hi: float
    ref: float
    gauss: tuple[float, float] | None = None


@dataclass
class ResolvedCobaya:
    config: object  # mg_bundle.MGConfig
    priors: dict[str, ParamPrior]
    fixed: dict[str, float]
    derived: list[str]
    likelihoods: dict[str, str]  # our component -> yaml key
    notes: list[str] = field(default_factory=list)
    source: str = ""
    # user likelihoods (gpucosmo.likelihoods.Likelihood instances), from the yaml or added
    extra_likelihoods: list = field(default_factory=list)

    def add_likelihoods(self, likelihoods, yaml_keys=None) -> None:
        """Add user likelihood instances; their parameters missing from ``priors`` get the
        class defaults (``Likelihood.params``)."""
        for i, L in enumerate(likelihoods):
            self.extra_likelihoods.append(L)
            self.likelihoods[f"ext:{L.label}"] = yaml_keys[i] if yaml_keys else "<python>"
            defaults = [n for n in L.params if n not in self.priors]
            for n in defaults:
                p = L.params[n]
                self.priors[n] = ParamPrior(float(p.lo), float(p.hi), p.start,
                                            None if p.gauss is None else tuple(map(float, p.gauss)))
            if defaults:
                self.notes.append(f"{L.label}: default priors for {defaults} (not in the yaml's params)")


# Cobaya likelihood-block options that configure Cobaya itself, not the likelihood.
_COBAYA_GENERIC = {"class", "python_path", "stop_at_error", "speed", "input_params", "output_params",
                   "version", "aliases"}


def _norm(s) -> str:
    return re.sub(r"\s+", "", str(s))


def load_yaml(path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def our_name(name: str) -> str:
    """Cobaya → bundle parameter names (the DESI bindings prefix FS biases with ``pre_``)."""
    return name[4:] if name.startswith("pre_") else name


def _ref_value(ref, prior):
    if isinstance(ref, (int, float)):
        return float(ref)
    if isinstance(ref, dict) and "loc" in ref:
        return float(ref["loc"])
    if prior.gauss is not None:
        return prior.gauss[0]
    return 0.5 * (prior.lo + prior.hi)


def _parse_prior(name, pr) -> ParamPrior:
    dist = pr.get("dist", "uniform")
    if dist == "uniform":
        lo = pr.get("min", pr.get("loc"))
        hi = pr.get("max", None if "loc" not in pr else pr["loc"] + pr.get("scale", 0))
        if lo is None or hi is None:
            raise UnsupportedCobayaInput(f"{name}: uniform prior needs min/max ({pr})")
        return ParamPrior(float(lo), float(hi), 0.5 * (float(lo) + float(hi)))
    if dist == "norm":
        mu, sd = float(pr["loc"]), float(pr["scale"])
        lo = float(pr.get("min", mu - GAUSS_BOX * sd))
        hi = float(pr.get("max", mu + GAUSS_BOX * sd))
        return ParamPrior(lo, hi, mu, (mu, sd))
    raise UnsupportedCobayaInput(f"{name}: prior dist '{dist}' not supported (uniform, norm)")


def cobaya_resolvable(key: str, block: dict | None = None) -> bool:
    """Cobaya can load this likelihood (internal, an installed external package, or a class path)."""
    try:
        from cobaya.component import get_component_class
    except ImportError:
        return False
    try:
        get_component_class(key, kind="likelihood", class_name=(block or {}).get("class"))
        return True
    except Exception:  # noqa: BLE001 (ComponentNotFoundError, import errors of the package)
        return False


def _match_likelihoods(likes: dict, *, bridge: bool = True) -> dict[str, str]:
    """{component: yaml key}: built-in components, then user likelihoods (``ext:<key>``) the
    ``gpucosmo.likelihoods`` registry (native ports of Cobaya likelihoods register under their
    Cobaya names) or an import path resolves, then (``bridge``) any other likelihood Cobaya can
    load, through the Cobaya bridge (``cobaya:<key>``)."""
    from gpucosmo.likelihoods.registry import get_likelihood_class

    found, unknown = {}, []
    for key in likes:
        block = likes[key] if isinstance(likes[key], dict) else None
        comp = next((c for c, pats in _LIKES.items()
                     if any(key == p or key.endswith("." + p) or key.endswith(p) for p in pats)), None)
        if comp is None and get_likelihood_class(key, block):
            comp = f"ext:{key}"
        if comp is None and bridge and cobaya_resolvable(key, block):
            comp = f"cobaya:{key}"
        if comp is None:
            unknown.append(key)
        else:
            found[comp] = key
    if unknown:
        raise UnsupportedCobayaInput(
            f"likelihoods not available on the GPU pipeline: {unknown} (a user likelihood must be registered — "
            "import its module with --plugin, install it with a 'gpucosmo.likelihoods' entry point — or be "
            "named by its import path, e.g. my_package.module.MyLikelihood or class: in its block; Cobaya "
            "likelihoods go through the Cobaya bridge when Cobaya can load them)")
    return found


def _reroute(likes: dict, comps, notes: list, why: str) -> None:
    """Move built-in components ``comps`` to the native-port / Cobaya-bridge path."""
    from gpucosmo.likelihoods.registry import get_likelihood_class

    for c in comps:
        if c in likes:
            key = likes.pop(c)
            new = f"ext:{key}" if get_likelihood_class(key) else f"cobaya:{key}"
            likes[new] = key
            notes.append(f"{key}: {why}; evaluated {'natively' if new.startswith('ext:') else 'through the Cobaya bridge'}")


def _instantiate_extra(info: dict, likes: dict, helpers: dict | None = None) -> tuple[list, list[str]]:
    """User likelihood instances for the ``ext:`` components, with the yaml block's options;
    the instance name is the yaml key, except for an import path / alias key (class name).
    ``cobaya:`` components become ``cobaya_bridge.CobayaLikelihood`` instances (the yaml's
    ``params`` entries override the likelihood's default parameter settings)."""
    from gpucosmo.likelihoods.registry import get_likelihood_class

    out, keys = [], []
    for comp, key in likes.items():
        if comp.startswith("cobaya:"):
            from gpucosmo.cobaya_bridge.bridge import BridgeUnsupported, CobayaLikelihood

            try:
                L = CobayaLikelihood(key, dict(info["likelihood"][key] or {}), params_info=info.get("params") or {},
                                     packages_path=info.get("packages_path"), theories=helpers or None)
            except BridgeUnsupported as e:
                raise UnsupportedCobayaInput(str(e)) from e
            out.append(L)
            keys.append(key)
            continue
        if not comp.startswith("ext:"):
            continue
        block = dict(info["likelihood"][key] or {})
        cls = get_likelihood_class(key, block)
        opts = {k: v for k, v in block.items() if k not in _COBAYA_GENERIC}
        from gpucosmo.likelihoods.cobaya.base import CobayaPort

        if issubclass(cls, CobayaPort):  # native port: Cobaya options and parameter settings as for the bridge
            try:  # the full block: Cobaya itself reads input_params & co.
                L = cls(key if key in cls.cobaya_names else None, block, params_info=info.get("params") or {},
                        packages_path=info.get("packages_path"))
            except NotImplementedError as e:  # an option the port does not cover: the bridge does
                from gpucosmo.cobaya_bridge.bridge import BridgeUnsupported, CobayaLikelihood

                try:
                    L = CobayaLikelihood(key, block, params_info=info.get("params") or {},
                                         packages_path=info.get("packages_path"), theories=helpers or None)
                except BridgeUnsupported as e2:
                    raise UnsupportedCobayaInput(f"{key}: {e}; bridge: {e2}") from e2
                L._fallback_note = f"{key}: native port declined ({e}); evaluated through the Cobaya bridge"
            out.append(L)
            keys.append(key)
            continue
        try:
            L = cls(**opts)
        except TypeError as e:
            raise UnsupportedCobayaInput(f"likelihood {key!r} ({cls.__module__}.{cls.__qualname__}): {e}") from e
        if block.get("class") or key == (cls.name or cls.__name__) or "name" in opts:
            L.name = str(opts.get("name", key))
        else:
            L.name = cls.name or cls.__name__
        out.append(L)
        keys.append(key)
    return out, keys


def _check_fs_options(opts: dict, notes: list):
    opts = opts or {}
    klim = opts.get("klim")
    if klim is not None and not np.allclose(klim, [0.02, 0.2]):
        raise UnsupportedCobayaInput(f"desi_fs_bao_all klim {klim}: only [0.02, 0.2] is ported")
    if opts.get("solve", "marg") != "marg":
        raise UnsupportedCobayaInput(f"desi_fs_bao_all solve={opts['solve']}: only 'marg' is ported")
    obs = opts.get("observable_name")
    if obs is not None and obs not in _FS_OBSERVABLES:
        raise UnsupportedCobayaInput(f"desi_fs_bao_all observable_name '{obs}' not ported")
    tr = opts.get("tracers")
    if tr is not None and set(map(str.lower, tr)) != _FS_TRACERS:
        raise UnsupportedCobayaInput(f"desi_fs_bao_all tracers {tr}: only all 6 FS + Lyα ported")
    if obs == "power+bao-recon":
        notes.append("desi_fs_bao_all: observable 'power+bao-recon' (internal desi-y1-kp naming) "
                     "→ public 'spectrum-poles-rotated+bao-recon' data files")


def _cobaya_theory_resolvable(key: str, block) -> bool:
    try:
        from cobaya.component import get_component_class

        get_component_class(key, kind="theory", class_name=(block or {}).get("class") if isinstance(block, dict) else None)
        return True
    except Exception:  # noqa: BLE001
        return False


def _check_theory(theory: dict, notes: list, des_basis: bool = False, hsc_basis: bool = False) -> dict:
    """Returns {'mg': bool, 'nonlinear_lens': bool}. ``des_basis``: the DES parameter basis (three
    degenerate massive neutrinos and the DES likelihood's Takahashi halofit);
    ``hsc_basis``: the HSC Y3 bases (one massive neutrino; HMcode-2016 with the sampled A_bary, 3x2pt,
    or Takahashi halofit, 2x2pt — ``out["hsc_halofit"]``)."""
    out = {"mg": False, "nonlinear_lens": False, "helpers": {}}
    boltz = [(k, v or {}) for k, v in (theory or {}).items()
             if "camb" in k.lower() or "isitgr" in k.lower()]
    for k, v in (theory or {}).items():
        if not (("camb" in k.lower()) or ("isitgr" in k.lower()) or ("reptvelocileptors" in k)):
            # other Cobaya theory components (e.g. mflike.BandpowerForeground): run with the
            # Cobaya-bridged likelihoods that read them
            if not _cobaya_theory_resolvable(k, v):
                raise UnsupportedCobayaInput(f"theory '{k}' not available")
            out["helpers"][k] = v
            notes.append(f"theory component {k}: evaluated with the Cobaya likelihoods through the bridge")
        if "reptvelocileptors" in k and (v or {}).get("is_physical_prior", True) is not True:
            raise UnsupportedCobayaInput("reptvelocileptors: only is_physical_prior=True is ported")
    if len(boltz) != 1:
        raise UnsupportedCobayaInput(f"expected one CAMB/ISiTGR theory block, got {[k for k, _ in boltz]}")
    key, block = boltz[0]
    ea = dict(block.get("extra_args") or {})
    n_nu = ea.pop("num_massive_neutrinos", 3 if des_basis else 1)
    hierarchy = str(ea.pop("neutrino_hierarchy", "degenerate")).lower()
    if hierarchy in ("1", "2", "3"):  # CAMB's integer codes
        hierarchy = ("normal", "inverted", "degenerate")[int(hierarchy) - 1]
    if hierarchy not in ("degenerate", "normal", "inverted"):
        raise UnsupportedCobayaInput(f"neutrino_hierarchy {hierarchy!r}: degenerate, normal or inverted")
    if (des_basis or hsc_basis) and (hierarchy != "degenerate" or n_nu != (3 if des_basis else 1)):
        raise UnsupportedCobayaInput("only num_massive_neutrinos = 1 (3 in the DES basis: Omega_m, As_1e9, ...) "
                                     "and degenerate neutrinos in the DES / HSC bases")
    if int(n_nu) != n_nu or n_nu < 0:
        raise UnsupportedCobayaInput(f"num_massive_neutrinos = {n_nu}")
    out["nu_hierarchy"], out["num_massive_nu"] = hierarchy, int(n_nu)
    if ea.pop("dark_energy_model", "ppf") not in ("ppf",):
        raise UnsupportedCobayaInput("only dark_energy_model = ppf")
    bbn = ea.pop("bbn_predictor", "PArthENoPE_880.2_standard.dat")
    if "PArthENoPE_880.2_standard" not in str(bbn):
        raise UnsupportedCobayaInput(f"bbn_predictor {bbn}: only PArthENoPE_880.2_standard")
    mgp = ea.pop("MG_parameterization", None)
    if mgp not in (None, "muSigma"):
        raise UnsupportedCobayaInput(f"MG_parameterization {mgp}: only muSigma")
    out["mg"] = mgp == "muSigma"
    nl = ea.pop("NonLinear", "NonLinear_none")
    if nl not in ("NonLinear_none", "NonLinear_lens"):
        raise UnsupportedCobayaInput(f"NonLinear {nl}: only NonLinear_none / NonLinear_lens")
    out["nonlinear_lens"] = nl == "NonLinear_lens"
    out["halofit_version"] = str(ea.pop("halofit_version", "mead2020"))
    out["key"] = key
    out["nonlinear_explicit"] = "NonLinear" in (block.get("extra_args") or {})
    out["lens_potential_accuracy"] = float((block.get("extra_args") or {}).get("lens_potential_accuracy", 1))
    noted = {k: ea.pop(k) for k in list(ea) if k in _NOTE_THEORY}
    if noted:
        notes.append(f"{key}: accuracy settings {noted} have no counterpart (the JAX solver is "
                     "validated against CAMB/ISiTGR at higher accuracy)")
    if ea:
        raise UnsupportedCobayaInput(f"{key} extra_args not supported: {ea}")
    return out


# ACTDR6LensLike options and the values the port reproduces (class defaults otherwise).
# ``nz``/``kmax``/``zmax`` only enter with ``limber``; ``lmax`` ≥ ``trim_lmax`` is checked.
_ACT_LENS_FIXED = {"mock": False, "nsims_act": 792.0, "nsims_planck": 400.0,
                   "no_actlike_cmb_corrections": False, "trim_lmax": 2998, "apply_hartlap": True,
                   "limber": False, "scale_cov": None, "varying_cmb_alens": False,
                   "version": "v1.2", "act_cmb_rescale": False, "act_calib": False}
_ACT_LENS_IGNORED = {"nz", "kmax", "zmax", "lmax", "type", "speed", "stop_at_error",
                     "input_params", "output_params", "variant", "lens_only", "no_like_corrections",
                     "python_path", "path"}


def _check_act_lens(a: dict, notes: list) -> tuple[str, bool]:
    """(variant, lens_only) of an ``ACTDR6LensLike`` block; everything else must be the
    ported default (``act_dr6_lens_jax``)."""
    from gpucosmo.jax.cmb.act_dr6_lens_jax import VARIANTS

    variant = str(a.get("variant", "act_baseline")).lower().strip()
    lens_only = bool(a.get("lens_only", False))  # ACTDR6LensLike default
    if variant not in VARIANTS:
        raise UnsupportedCobayaInput(f"ACT DR6 lensing variant {variant!r}: ported {VARIANTS}")
    if bool(a.get("no_like_corrections", False)) and not lens_only:
        raise UnsupportedCobayaInput("ACT DR6 lensing no_like_corrections without lens_only "
                                     "(uncorrected, un-marginalized covariance) not ported")
    for k, v in _ACT_LENS_FIXED.items():
        got = a.get(k, v)
        if (got is None) != (v is None) or (v is not None and got != v and not (
                isinstance(v, float) and np.isclose(float(got), v))):
            if not (k == "version" and got is None):  # None = the package default, v1.2
                raise UnsupportedCobayaInput(f"ACT DR6 lensing {k}={got!r}: only {v!r} is ported")
    unknown = set(a) - set(_ACT_LENS_FIXED) - _ACT_LENS_IGNORED
    if unknown:
        raise UnsupportedCobayaInput(f"ACT DR6 lensing options not ported: {sorted(unknown)}")
    if int(a.get("lmax", 4000)) < int(a.get("trim_lmax", 2998)):
        raise UnsupportedCobayaInput("ACT DR6 lensing lmax < trim_lmax")
    if variant == "act_baseline" and lens_only:
        notes.append("ACT DR6 lensing: act_baseline, lens_only=True (candl ACT_DR6_Lens)")
    else:
        notes.append(f"ACT DR6 lensing: {variant}, lens_only={lens_only} (act_dr6_lens_jax, "
                     "port of act_dr6_lenslike v1.2" + ("" if lens_only else
                     "; likelihood corrections from lensed TT/EE/BB/TE and κκ to L = 2999") + ")")
    return variant, lens_only


def resolve_cobaya(info: dict | str | Path, *, walker_batch: int = 0) -> ResolvedCobaya:
    """Translate a Cobaya yaml (dict or path) into ``MGConfig`` + prior table."""
    from gpucosmo.jax.inference.mg_bundle import MGConfig

    source = ""
    if not isinstance(info, dict):
        source = str(info)
        info = load_yaml(info)
    notes: list[str] = []
    likes = _match_likelihoods(info.get("likelihood") or {})
    if "desi_fs" in likes:
        raise UnsupportedCobayaInput("the DESI DR1 full-shape likelihood (desi_fs_bao_all) is not part of this "
                                     "preview release")
    from gpucosmo.jax.inference.mg_bundle import DES_THEORY_NAMES, HSC_TAKAHASHI_THEORY_NAMES, HSC_THEORY_NAMES

    hsc_basis = any(isinstance(v, dict) and "prior" in v for k, v in (info.get("params") or {}).items()
                    if k in ("log1e10As", "A_bary"))
    des_basis = not hsc_basis and any(isinstance(v, dict) and "prior" in v
                                      for k, v in (info.get("params") or {}).items()
                                      if k in ("Omega_m", "Omega_b", "As_1e9"))
    if des_basis or hsc_basis:
        raise UnsupportedCobayaInput("the DES Y3 / HSC Y3 parameter bases (Omega_m, As_1e9 / log1e10As, A_bary "
                                     "priors) and their likelihoods are not part of this preview release")
    th = _check_theory(info.get("theory") or {}, notes, des_basis=des_basis, hsc_basis=hsc_basis)
    if "desi_fs" in likes:
        _check_fs_options(info["likelihood"][likes["desi_fs"]], notes)

    # The built-in CMB likelihoods come as the official PR3 / PR4 combinations; any other
    # combination of their keys runs them as Cobaya likelihoods (native port or bridge).
    cmb = None
    if "plik" in likes and {"lowl_tt", "lowl_ee"} <= set(likes) and not {"hillipop", "lollipop"} & set(likes):
        cmb = "pr3"
    elif "hillipop" in likes and {"lowl_tt", "lollipop"} <= set(likes) and "lowl_ee" not in likes:
        cmb = "pr4"
    else:
        _reroute(likes, ("plik", "hillipop", "lollipop", "lowl_tt", "lowl_ee"), notes,
                 "not part of a built-in PR3 (plik + low-ℓ TT + EE) / PR4 (HiLLiPoP + LoLLiPoP + low-ℓ TT) set")
    act_lens = "act_lens" in likes
    act_variant, act_lens_only = "act_baseline", True
    if act_lens and cmb is None:
        _reroute(likes, ("act_lens",), notes, "built-in ACT DR6 lensing needs a built-in primary CMB set")
        act_lens = False
    if act_lens:
        a = info["likelihood"][likes["act_lens"]] or {}
        act_variant, act_lens_only = _check_act_lens(a, notes)
    if len([c for c in likes if c.startswith("sn:")]) > 1:
        _reroute(likes, [c for c in likes if c.startswith("sn:")][1:], notes, "one built-in SN dataset at a time")
    user_likes, user_keys = _instantiate_extra(info, likes, th.get("helpers"))
    notes += [L._fallback_note for L in user_likes if getattr(L, "_fallback_note", None)]
    from gpucosmo.jax.inference.mg_bundle import extra_requirements

    cmb_theory = cmb is None and "cmb" in extra_requirements(user_likes)
    if (cmb or cmb_theory) and th["key"].split(".")[-1].lower() == "camb" and th["lens_potential_accuracy"] >= 1:
        # Cobaya's CAMB wrapper sets NonLinear = NonLinear_lens for any lensed-Cl request with
        # lens_potential_accuracy >= 1 (default 1), *after* applying extra_args, so the yaml's
        # NonLinear (or its absence) does not apply. DESI's ISiTGR binding keeps the yaml value
        # (its NonLinear_none matches the official MG chains' per-likelihood χ²; see
        # scripts/compare_official_likelihoods.py).
        if not th["nonlinear_lens"]:
            notes.append("camb: Cobaya lenses the CMB with the non-linear φφ (NonLinear_lens) whenever "
                         "lens_potential_accuracy >= 1, overriding the yaml's NonLinear; reproduced "
                         "(cmb_nonlinear_lens=True)")
        th["nonlinear_lens"] = True
    # The yaml's HMcode version, also kept for linear-lensing yamls (used if a run forces the
    # non-linear φφ, e.g. --cmb-nonlinear-lens).
    nl_model = _HALOFIT.get(th["halofit_version"])
    if (cmb or cmb_theory) and th["nonlinear_lens"]:
        if nl_model is None:
            raise UnsupportedCobayaInput(f"halofit_version {th['halofit_version']!r} for the lensing φφ: "
                                         f"ported {sorted(_HALOFIT)}")
        notes.append(f"non-linear lensing φφ with HMcode {nl_model} (halofit_version "
                     f"{th['halofit_version']!r})")
    nl_model = nl_model or "mead2020"
    sn = [c.split(":")[1] for c in likes if c.startswith("sn:")]

    # ---- parameters ----
    priors: dict[str, ParamPrior] = {}
    fixed: dict[str, float] = {}
    derived: list[str] = []
    for name, p in (info.get("params") or {}).items():
        if isinstance(p, (int, float)):
            fixed[name] = float(p)
        elif isinstance(p, str):
            derived.append(name)
        elif isinstance(p, dict) and "prior" in p:
            pp = _parse_prior(name, p["prior"])
            priors[our_name(name)] = dataclasses.replace(pp, ref=_ref_value(p.get("ref"), pp))
        elif isinstance(p, dict) and "value" in p:
            v = p["value"]
            if isinstance(v, (int, float)):
                fixed[name] = float(v)
            else:
                derived.append(name)  # 'lambda ...' functions of other parameters
        else:
            derived.append(name)
    ext_sampled, ext_fixed = [], {}
    for k, v in SOLVER_FIXED.items():
        if des_basis and k == "mnu":
            continue  # Σm_ν from the sampled Ω_ν h² (DES likelihood theory)
        if k in priors:
            if des_basis or hsc_basis:
                raise UnsupportedCobayaInput(f"{k} is sampled; the {'DES' if des_basis else 'HSC'} basis fixes "
                                             f"{k} = {v}")
            ext_sampled.append(k)
        elif not np.isclose(fixed.get(k, v), v):
            if des_basis or hsc_basis:
                raise UnsupportedCobayaInput(f"{k} = {fixed[k]}: the {'DES' if des_basis else 'HSC'} basis fixes "
                                             f"{k} = {v}")
            ext_fixed[k] = float(fixed[k])
    sterile = "meffsterile" in ext_sampled or "meffsterile" in ext_fixed
    if sterile and ("mnu" in ext_sampled or "mnu" in ext_fixed):
        raise UnsupportedCobayaInput("a massive sterile neutrino (meffsterile) with Σm_ν other than 0.06 eV "
                                     "is not tabulated (Planck's sterile chains fix mnu = 0.06)")
    if sterile and th["nu_hierarchy"] != "degenerate":
        raise UnsupportedCobayaInput("meffsterile with a normal/inverted hierarchy is not tabulated")
    curved = "omk" in ext_sampled or "omk" in ext_fixed
    if curved and th["mg"]:
        raise UnsupportedCobayaInput("spatial curvature with μ–Σ modified gravity: the ISiTGR port is flat")
    if hsc_basis:
        takahashi = th["hsc_halofit"] != "mead2016"
        need = HSC_TAKAHASHI_THEORY_NAMES if takahashi else HSC_THEORY_NAMES
        missing = [n for n in need if n not in priors]
        if missing:
            raise UnsupportedCobayaInput(f"HSC basis: sample all of {need} (missing {missing})")
        if {"H0", "h", "logA", "As"} & set(priors):
            raise UnsupportedCobayaInput("HSC basis: h, H0, logA, As are derived, not sampled")
        if takahashi and "A_bary" in priors:
            raise UnsupportedCobayaInput("HSC basis with Takahashi halofit: A_bary has no effect; do not sample it")
        for L in user_likes:
            hf = getattr(L, "halofit", None)
            if hf is not None and hf != th["hsc_halofit"]:
                raise UnsupportedCobayaInput(f"{L.label}: its theory uses halofit {hf!r}, the yaml's camb "
                                             f"halofit_version is {th['hsc_halofit']!r}")
    if des_basis:
        missing = [n for n in DES_THEORY_NAMES if n not in priors]
        if missing:
            raise UnsupportedCobayaInput(f"DES basis: sample all of {DES_THEORY_NAMES} (missing {missing})")
        if {"H0", "ombh2", "omch2", "logA", "As"} & set(priors):
            raise UnsupportedCobayaInput("DES basis: H0 / ombh2 / omch2 / logA are derived, not sampled")
    cmb_basis = bool(cmb) or cmb_theory
    if "As" in priors or "H0" in priors and cmb_basis:
        raise UnsupportedCobayaInput("sample logA (not As) and, with CMB, theta_MC_100 (not H0)")
    if cmb_basis and "theta_MC_100" not in priors:
        raise UnsupportedCobayaInput("with CMB, theta_MC_100 must be sampled")
    from gpucosmo.jax.eb import cls_jax as _cls

    pk_likes = [L.label for L in user_likes if "pk" in getattr(L, "requires", ())]
    if curved and not _cls.CURVATURE_PERTURBATIONS and (cmb_basis or "desi_fs" in likes or pk_likes):
        raise UnsupportedCobayaInput("spatial curvature (omk) is supported at the background level only "
                                     "(BAO, SN, H0, ...) with GPUCOSMO_CURVATURE=0: not with CMB, full-shape or "
                                     "matter-power likelihoods")
    if curved and pk_likes:
        notes.append(f"Ω_k with {pk_likes}: the curved P(k) (linear and HMcode) is CAMB's; the projection is the "
                     "likelihood's own (Cobaya's, e.g. flat-kernel Limber in χ), as with Cobaya + CAMB")
    if curved and "r" in priors:
        raise UnsupportedCobayaInput("tensor modes (r) with spatial curvature are not supported")
    if cmb_basis and "theta_MC_100" in priors:
        notes.append("theta_MC_100 is sampled as 100θ* (r_s(z*)/D_M, ``theta_star``); the yaml's CAMB "
                     "cosmomc_theta is ≈ 2.2e-4 lower in 100θ at fixed H0. Same physics, uniform prior "
                     "on θ* instead of θ_MC; convert with background.cosmomc_theta before comparing chains")
    if not cmb_basis and not des_basis and not hsc_basis and "H0" not in priors:
        raise UnsupportedCobayaInput("without CMB, H0 must be sampled")
    w0wa = "w" in priors or "wa" in priors
    if w0wa and not {"w", "wa"} <= set(priors):
        raise UnsupportedCobayaInput("sample both w and wa, or neither")
    if not w0wa and (not np.isclose(fixed.get("w", -1.0), -1.0) or
                     not np.isclose(fixed.get("wa", 0.0), 0.0)):
        raise UnsupportedCobayaInput("fixed w ≠ −1 or wa ≠ 0 not supported")
    if cmb_basis and "tau" not in priors:
        raise UnsupportedCobayaInput("with CMB, tau must be sampled")
    tau = fixed.get("tau", 0.0544)
    if not cmb_basis and "tau" in priors:
        raise UnsupportedCobayaInput("without CMB, tau must be fixed")
    tensors = "r" in priors
    if tensors and not cmb_basis:
        raise UnsupportedCobayaInput("r is sampled but no likelihood needs the CMB (tensor modes are a CMB-block output)")
    if not tensors and not np.isclose(fixed.get("r", 0.0), 0.0):
        raise UnsupportedCobayaInput(f"r = {fixed['r']} fixed: sample r (tensor modes) or fix it to 0")
    if th["mg"] != ({"mu0", "Sigma0"} <= set(priors)):
        raise UnsupportedCobayaInput("MG_parameterization=muSigma needs mu0 and Sigma0 sampled")
    for name, v in fixed.items():
        if name.endswith(".b3p") and v != 0.0:
            raise UnsupportedCobayaInput(f"{name} = {v}: our REPT bias model has b3p = 0")
        if cmb == "pr3" and name in PLIK_FIXED and not np.isclose(v, PLIK_FIXED[name]):
            raise UnsupportedCobayaInput(f"plik {name} = {v} ≠ clipy fixed {PLIK_FIXED[name]}")
        if cmb == "pr4":
            from gpucosmo.jax.cmb.pr4_jax import HILLIPOP_FIXED

            ref = {**HILLIPOP_FIXED, **HILLIPOP_ABSENT}
            if name in ref and not np.isclose(v, ref[name]):
                raise UnsupportedCobayaInput(f"HiLLiPoP {name} = {v} ≠ ported value {ref[name]}")
    if cmb == "pr4" and any(k in fixed for k in HILLIPOP_ABSENT):
        notes.append("HiLLiPoP AsyncT/AsyncP fixed at 0: terms not modelled by the port (≡ 0)")

    # ---- extra prior terms ----
    mg_hard = sz = False
    ombh2_max = None
    extra = dict(info.get("prior") or {})
    if "plik" in likes:
        extra.update((info["likelihood"][likes["plik"]] or {}).get("prior") or {})
    for k, v in extra.items():
        if _norm(v) == _MG_PRIOR:
            mg_hard = True
        elif _norm(v) == _SZ_PRIOR and cmb == "pr3":
            sz = True
        elif _norm(v) == _SZ_PRIOR:
            notes.append(f"prior '{k}' (Planck SZ): applied by the bridged/ported plik likelihood itself")
        elif des_basis and (m := _OMBH2_CUT.fullmatch(_norm(v))):
            ombh2_max = float(m.group(1))
        else:
            raise UnsupportedCobayaInput(f"prior '{k}': {v} not supported")
    bbn_gauss = ns_gauss = bbn_gauss2d = None
    if "bbn" in likes:
        b = info["likelihood"][likes["bbn"]] or {}
        if list(b.get("quantities", ["omegabh2", "nnu"])) != ["omegabh2", "nnu"]:
            raise UnsupportedCobayaInput(f"schoneberg2024 quantities {b.get('quantities')}")
        from gpucosmo.jax.inference.mg_bundle import bbn_conditional

        if "nnu" in ext_sampled:  # the full (ω_b, N_eff) Gaussian: a joint prior (MGConfig.bbn_gauss2d)
            bbn_gauss2d = (tuple(float(v) for v in b["mean"]), tuple(tuple(float(v) for v in r) for r in b["cov"]))
            bbn_gauss = bbn_conditional(b["mean"], b["cov"], SOLVER_FIXED["nnu"])  # marks the BBN prior as on
            notes.append("BBN prior: the 2-D Gaussian in (ω_b, N_eff) (N_eff sampled)")
        else:
            bbn_gauss = bbn_conditional(b["mean"], b["cov"], fixed.get("nnu", SOLVER_FIXED["nnu"]))
    if "ns10" in likes:
        b = info["likelihood"][likes["ns10"]] or {}
        ns_gauss = (float(b.get("ns_mean", 0.9649)), float(b.get("ns_std", 0.042)))
    if (bbn_gauss is None) != (ns_gauss is None):
        raise UnsupportedCobayaInput("BBN and ns10 priors are only supported together")
    if info.get("sampler"):
        notes.append(f"sampler {list(info['sampler'])} ignored: DALI-trajectory HMC "
                     "(run_dali_hmc.py flags)")

    likes = {k: v for k, v in likes.items() if not k.startswith(("ext:", "cobaya:"))}
    cfg = MGConfig(desi_fs="desi_fs" in likes, cmb=cmb, sn=sn[0] if sn else None, mg=th["mg"],
                   w0wa=w0wa, bbn_ns_priors=bbn_gauss is not None,
                   bbn_gauss=bbn_gauss, ns_gauss=ns_gauss, tau_fixed=float(tau),
                   cmb_nonlinear_lens=th["nonlinear_lens"], cmb_nonlinear_model=nl_model,
                   cmb_lensing=act_lens,
                   cmb_lensing_variant=act_variant, cmb_lensing_lens_only=act_lens_only,
                   mg_hard_prior=mg_hard, cmb_theory=cmb_theory, tensors=tensors,
                   sz_prior=sz, ombh2_max=ombh2_max, walker_batch=walker_batch,
                   # Next to the 3.3 GB HiLLiPoP matrix, 8 walkers per CMB and per P(k) group
                   # fit the 0.80 pool (exact log posterior peak 5.6 GiB ΛCDM, 5.3 GiB μ–Σ;
                   # 16 does not): reduced knot storage + float32 per-mode evolution.
                   cmb_walker_batch=8 if cmb == "pr4" else MGConfig.cmb_walker_batch,
                   pk_walker_batch=8 if cmb == "pr4" else MGConfig.pk_walker_batch,
                   basis="des" if des_basis else (("hsc_takahashi" if th.get("hsc_halofit") == "takahashi" else "hsc")
                                                  if hsc_basis else "default"),
                   nu_hierarchy=th["nu_hierarchy"],
                   num_massive_nu=3 if des_basis else th["num_massive_nu"],
                   nu_sterile=sterile, ext_sampled=tuple(ext_sampled),
                   ext_fixed=tuple(sorted(ext_fixed.items())), bbn_gauss2d=bbn_gauss2d)
    res = ResolvedCobaya(config=cfg, priors=priors, fixed=fixed, derived=derived,
                         likelihoods=likes, notes=notes, source=source)
    if user_likes:
        res.add_likelihoods(user_likes, user_keys)
    return res


def audit_bundle(bundle, resolved: ResolvedCobaya, *, rtol: float = 1e-6) -> list[str]:
    """Differences between a built bundle and a resolved yaml ([] = identical)."""
    diffs = []
    ours = list(bundle.param_names)
    theirs = list(resolved.priors)
    if set(ours) != set(theirs):
        diffs.append(f"sampled parameters: only ours {sorted(set(ours) - set(theirs))}, "
                     f"only yaml {sorted(set(theirs) - set(ours))}")
    lo, hi = np.asarray(bundle.prior_lo), np.asarray(bundle.prior_hi)
    g_mu, g_sd = bundle.gauss_mu, bundle.gauss_sd
    # Cobaya's BBN / ns10 likelihoods are Gaussian priors on ω_b / n_s in the bundle.
    expect_g = {"ombh2": resolved.config.bbn_gauss, "ns": resolved.config.ns_gauss}
    for i, n in enumerate(ours):
        if n not in resolved.priors:
            continue
        p = resolved.priors[n]
        if expect_g.get(n) is not None:
            p = dataclasses.replace(p, gauss=expect_g[n])
        if not (np.isclose(lo[i], p.lo, rtol=rtol) and np.isclose(hi[i], p.hi, rtol=rtol)):
            diffs.append(f"{n}: box ours [{lo[i]:.6g}, {hi[i]:.6g}] vs yaml [{p.lo:.6g}, {p.hi:.6g}]")
        ours_g = None if not np.isfinite(g_sd[i]) else (float(g_mu[i]), float(g_sd[i]))
        if (ours_g is None) != (p.gauss is None) or (
                ours_g and not np.allclose(ours_g, p.gauss, rtol=1e-4)):
            diffs.append(f"{n}: Gaussian ours {ours_g} vs yaml {p.gauss}")
    c, r = bundle.config, resolved.config
    flags = ["desi_fs", "cmb", "sn", "mg", "w0wa"]
    flags += ["cmb_nonlinear_lens", "cmb_lensing"] if r.cmb else []
    flags += ["cmb_nonlinear_model"] if r.cmb and r.cmb_nonlinear_lens else []
    flags += ["cmb_lensing_variant", "cmb_lensing_lens_only"] if r.cmb_lensing else []
    flags += ["mg_hard_prior"] if r.mg else []
    flags += ["sz_prior"] if r.cmb == "pr3" else []  # plik only
    flags += ["ombh2_max"] if r.basis == "des" else []
    for f in flags:
        if getattr(c, f) != getattr(r, f):
            diffs.append(f"config.{f}: ours {getattr(c, f)} vs yaml {getattr(r, f)}")
    if not r.cmb and not np.isclose(c.tau_fixed, r.tau_fixed):
        diffs.append(f"tau fixed: ours {c.tau_fixed} vs yaml {r.tau_fixed}")
    ours_fixed = {**SOLVER_FIXED, **dict(c.ext_fixed)}
    for k, v in ours_fixed.items():
        if k in resolved.fixed and not np.isclose(resolved.fixed[k], v):
            diffs.append(f"{k}: ours {v} vs yaml {resolved.fixed[k]}")
    for f in ("nu_hierarchy", "num_massive_nu", "nu_sterile", "ext_sampled", "bbn_gauss2d"):
        if getattr(c, f) != getattr(r, f):
            diffs.append(f"config.{f}: ours {getattr(c, f)} vs yaml {getattr(r, f)}")
    return diffs


def dump_resolved(bundle, resolved: ResolvedCobaya | None, path, extra: dict | None = None):
    """Write our analogue of Cobaya's ``updated.yaml``: what the run actually used."""
    import subprocess

    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=Path(__file__).parent, check=False).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                    text=True, cwd=Path(__file__).parent, check=False).stdout)
    except OSError:
        git, dirty = "", False
    lo, hi, fid = (np.asarray(a) for a in (bundle.prior_lo, bundle.prior_hi, bundle.fiducial))
    params = {}
    for i, n in enumerate(bundle.param_names):
        d = {"prior": {"min": float(lo[i]), "max": float(hi[i])}, "ref": float(fid[i])}
        if np.isfinite(bundle.gauss_sd[i]):
            d["prior"]["gauss"] = [float(bundle.gauss_mu[i]), float(bundle.gauss_sd[i])]
        params[n] = d
    out = {
        "source": resolved.source if resolved else None,
        "config": dataclasses.asdict(bundle.config),
        "likelihoods": resolved.likelihoods if resolved else bundle.likelihood_label,
        "params": params,
        "fixed": {**{k: v for k, v in {**SOLVER_FIXED, **dict(bundle.config.ext_fixed)}.items()
                     if k not in bundle.config.ext_sampled},
                  **({"tau": bundle.config.tau_fixed} if not (bundle.config.cmb or bundle.config.cmb_theory) else {}),
                  **(resolved.fixed if resolved else {})},
        "bundle_notes": list(getattr(bundle, "notes", [])),
        **({"extra_likelihoods": [L.describe() for L in bundle.extra_likelihoods]}
           if getattr(bundle, "extra_likelihoods", ()) else {}),
        "notes": resolved.notes if resolved else [],
        "git": {"commit": git, "dirty": dirty},
        **(extra or {}),
    }
    Path(path).write_text(yaml.safe_dump(out, sort_keys=False, allow_unicode=True))


__all__ = ["GAUSS_BOX", "PLIK_FIXED", "ParamPrior", "ResolvedCobaya", "SOLVER_FIXED",
           "UnsupportedCobayaInput", "audit_bundle", "dump_resolved", "load_yaml", "our_name",
           "resolve_cobaya"]
