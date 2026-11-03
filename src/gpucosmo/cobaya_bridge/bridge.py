"""Any Cobaya likelihood on the GPU theory: ``CobayaLikelihood``.

``CobayaLikelihood("bao.desi_dr2")`` builds a Cobaya model holding that likelihood (with its
yaml options and parameter settings) and ``provider.BridgeTheory`` as its Boltzmann code,
reads what the likelihood requires, and exposes it as a ``gpucosmo.likelihoods.Likelihood``:

* ``params`` — the likelihood's sampled nuisance parameters with Cobaya's priors
  (uniform / normal);
* ``requirements()`` — the GPU theory blocks it needs: ``{"cmb": {"lmax", "spectra"}}``;
* ``theory(cosmo)`` — background products (H(z), distances, r_drag, θ*, ...) in JAX, so the
  DALI approximation expands them;
* ``loglike(inp)`` — a ``jax.pure_callback`` into Cobaya: the products are copied into the
  bridge theory and ``model.loglikes`` evaluates the likelihood on the host (NumPy). The
  value includes the likelihood's own ``prior:`` terms (e.g. Planck's SZ prior).

The callback has no derivative (``differentiable = False``): the pipelines evaluate it
exactly where values suffice (exact log posterior, MAP, DALI build) and
use a finite-difference gradient or a fitted quadratic where gradients are needed
(``cobaya_bridge.nondiff``). Natively ported likelihoods (``gpucosmo.likelihoods.cobaya``)
are JAX throughout and much faster; the bridge is the reference they are validated against.
"""

from __future__ import annotations

import logging
import math
import os

import jax
import jax.numpy as jnp
import numpy as np

from gpucosmo.likelihoods.base import Likelihood, Param

C_KM_S = 299792.458
# Our sampled cosmological parameters as the Cobaya theory's input parameters (values set
# from the GPU cosmology at each evaluation; the priors here are never used).
_THEORY_INPUTS = ("H0", "ombh2", "omch2", "logA", "As", "ns", "tau", "w", "wa", "mu0", "Sigma0",
                  "mnu", "nnu", "omk", "meffsterile")
# Derived parameters computed from the background (``theory``); "sigma8"/"S8" need P(k).
_BACKGROUND_DERIVED = ("H0", "h", "ombh2", "omch2", "omegam", "omegab", "omegac", "omega_de",
                       "omegamh2", "omnuh2", "omeganuh2", "rdrag", "zdrag", "rstar", "zstar",
                       "thetastar", "DAstar", "YHe", "Y_p", "Neff", "TCMB", "mnu", "nnu", "omk", "meffsterile",
                       "As", "logA", "ns", "tau", "w", "wa", "mu0", "Sigma0", "cosmomc_theta",
                       "theta_MC_100")
_CL_PROVIDED = ("tt", "te", "ee", "bb", "pp")
# Matter power spectra the linear "pk" block provides (CAMB variable names) and what needs it.
_PK_VARS = {"delta_tot": "pk_m", "delta_nonu": "pk_cb"}
_PK_DERIVED = ("sigma8", "S8", "s8h5", "s8omegamp5", "s8omegamp25")
_Z_BACKGROUND = ("Hubble", "angular_diameter_distance", "comoving_radial_distance", "Omega_b",
                 "Omega_cdm", "Omega_nu_massive")
# Flat sampling box for parameters with improper / huge Cobaya priors.
_BIG = 1e30
# Multipoles passed beyond a likelihood's C_ℓ request (when the solver has them), as CAMB does.
CL_MARGIN = 50
# C_ℓ keys some packages request as single letters (wmaplike's ``set.update("tt")``).
_CL_ALIAS = {"t": "tt", "e": "ee", "b": "bb", "p": "pp", "et": "te"}
# What some external likelihoods expect the user's yaml to add (used when it does not): helper
# theory components and nuisance parameters without defaults in the package.
KNOWN_SETUP = {
    # MFLike-based: the foreground model is a separate Cobaya theory component (provides fg_totals)
    # ACT DR6: the foreground block of the package's own Cobaya test (DR6 arrays, their band centres
    # — so the bandint_shift_dr6_* parameters act — and the chromatic beams)
    "act_dr6_mflike.": {"theories": {"mflike.BandpowerForeground": {
        "experiments": ["dr6_pa4_f220", "dr6_pa5_f090", "dr6_pa5_f150", "dr6_pa6_f090", "dr6_pa6_f150"],
        "bandint_freqs": [220, 90, 150, 90, 150], "beam_profile": {"beam_from_file": None}}}},
    # SO MFLike is a forecast: one of the installed simulations must be chosen (the package's
    # example: realization 44 with the shared covariance / bandpower-window file)
    "mflike.": {"theories": {"mflike.BandpowerForeground": None},
                "block": {"input_file": "LAT_simu_sacc_00044.fits", "cov_Bbl_file": "data_sacc_w_covar_and_Bbl.fits"}},
    # ACTPol DR4 lite: polarization efficiency y_p (Choi et al. 2020: uniform 0.9–1.1)
    "pyactlike.": {"params": {"yp2": {"prior": {"min": 0.9, "max": 1.1}, "ref": 1.0, "latex": "y_p"}}},
    # TDCOSMO: no lens sample is on by default (the 7 TDCOSMO lenses here); hierarc marginalizes by
    # Monte Carlo, so evaluations use common random numbers (fixed seed) to be a smooth function
    "tdcosmo.": {"block": {"tdcosmo_lenses": True}, "seed": 0},
}


def _norm_logpdf(x, mu, sd):
    return -0.5 * ((x - mu) / sd) ** 2 - np.log(sd * np.sqrt(2 * np.pi))


# Cobaya ``prior:`` lambdas with a JAX equivalent (whitespace-free source → (params, log density)),
# applied through ``log_prior``; any other external prior stays in the host callback.
KNOWN_EXTERNAL_PRIORS = {
    "lambdaksz_norm,A_sz:stats.norm.logpdf(ksz_norm+1.6*A_sz,loc=9.5,scale=3.0)":
        (("ksz_norm", "A_sz"), lambda k, a: _norm_logpdf(k + 1.6 * a, 9.5, 3.0)),
}


def known_setup(cobaya_name: str) -> dict:
    out = next((v for k, v in KNOWN_SETUP.items() if cobaya_name.startswith(k)), {})
    if cobaya_name.startswith("mflike.") and cobaya_name != "mflike.TTTEEE":
        # the foreground component must request the same spectra as the data set (TT, TTTE, ...)
        v = cobaya_name.split(".")[1]
        cls = [v[i:i + 2].lower() for i in range(0, len(v), 2)]
        out = {**out, "theories": {"mflike.BandpowerForeground": {"requested_cls": cls}}}
    return out


class BridgeUnsupported(ValueError):
    """The Cobaya likelihood needs something the GPU theory does not compute."""


def _clipy_jax_first():
    """Import clipy with its JAX backend before Cobaya does: Cobaya's clik wrapper sets
    ``CLIPY_NOJAX=1`` process-wide before importing clipy, which would leave the built-in and
    ported Planck likelihoods (JAX clipy) without a backend. Cobaya's clik objects run on the
    JAX build as well."""
    try:
        from gpucosmo.jax.cmb.clipy_jax_compat import ensure_clipy_jax

        ensure_clipy_jax()
    except Exception:  # noqa: BLE001 (clipy absent: nothing to protect)
        pass
    from gpucosmo.jax.cmb.candl_compat import ensure_candl_compat

    ensure_candl_compat()  # candl's jnp.atleast_1d(list) under JAX ≥ 0.10


def _theory_attr(model) -> None:
    """Likelihoods written for older Cobaya call ``self.theory.get_Cl`` (pyactlike 0.1); current
    Cobaya only sets ``self.provider``, so they return −inf. Point ``theory`` at the provider."""
    for like in model.likelihood.values():
        if getattr(like, "theory", None) is None and getattr(like, "provider", None) is not None:
            like.theory = like.provider


def _numpy_clik(model) -> None:
    """Cobaya's clik wrapper expects clipy's NumPy build (a 1-element array); with the JAX build
    (``_clipy_jax_first``) the call returns a 0-d JAX array — convert it on the instance."""
    for like in model.likelihood.values():
        f = getattr(like, "clik_likelihood", None)
        if f is not None and not getattr(f, "_gpucosmo_numpy", False):
            def g(vec, f=f):
                return np.atleast_1d(np.asarray(f(vec), dtype=float))
            g._gpucosmo_numpy = True
            g._gpucosmo_orig = f  # the clipy object (native ports call it on JAX arrays)
            like.clik_likelihood = g


def _quiet_cobaya():
    for n in ("cobaya", "model", "prior", "BridgeTheory", "gpucosmo_bridge"):
        logging.getLogger(n).setLevel(logging.WARNING)


def _param_from_cobaya(name: str, info: dict) -> Param:
    pr = info.get("prior")
    ref = info.get("ref")
    latex = info.get("latex")
    if isinstance(pr, (list, tuple)) and len(pr) == 2:
        pr = {"min": pr[0], "max": pr[1]}
    if not isinstance(pr, dict):
        raise BridgeUnsupported(f"{name}: prior {pr!r} not supported (uniform or normal)")
    dist = pr.get("dist", "uniform")
    if dist == "uniform":
        lo = pr.get("min", pr.get("loc"))
        hi = pr.get("max", None if "loc" not in pr else pr["loc"] + pr.get("scale", 0))
        lo, hi = float(lo), float(hi)
        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise BridgeUnsupported(f"{name}: improper uniform prior {pr}")
        gauss = None
    elif dist == "norm":
        mu, sd = float(pr["loc"]), float(pr["scale"])
        lo, hi = float(pr.get("min", mu - 10 * sd)), float(pr.get("max", mu + 10 * sd))
        gauss = (mu, sd)
    elif dist in ("loguniform", "reciprocal"):  # the box here, the 1/x density in ``log_prior``
        lo, hi = float(pr.get("a", pr.get("min"))), float(pr.get("b", pr.get("max")))
        gauss = None
    else:
        raise BridgeUnsupported(f"{name}: prior dist {dist!r} not supported (uniform, norm, loguniform)")
    r = None
    if isinstance(ref, (int, float)):
        r = float(ref)
    elif isinstance(ref, dict) and "loc" in ref:
        r = float(ref["loc"])
    if r is not None and not lo < r < hi:
        r = None
    return Param(lo, hi, r, gauss, latex)


class CobayaLikelihood(Likelihood):
    """One Cobaya likelihood (by its Cobaya name, e.g. ``"sn.pantheonplus"``,
    ``"planck_2018_lensing.native"``, ``"spt3g_2022.TTTEEE"``) evaluated through Cobaya on the
    GPU theory's products.

    ``block``: the yaml block's options (as in a Cobaya input). ``params_info``: the yaml's
    ``params`` entries (priors, fixed values) for this likelihood's parameters, overriding
    its defaults. ``packages_path``: Cobaya packages (default ``gpucosmo.paths.packages_path``).
    ``check=False`` skips the check that the GPU theory provides every requirement.
    """

    differentiable = False
    allow_shared_params = True

    def __init__(self, cobaya_name: str, block: dict | None = None, *, params_info: dict | None = None,
                 packages_path: str | None = None, name: str | None = None, check: bool = True,
                 theories: dict | None = None, seed: int | None = None):
        super().__init__(**({"name": name} if name else {}))
        if not name:
            self.name = cobaya_name
        self.cobaya_name = cobaya_name
        self.block = dict(block or {})
        _quiet_cobaya()
        setup = known_setup(cobaya_name)
        self.block = {**setup.get("block", {}), **self.block}
        # stochastic likelihoods: NumPy's global RNG reseeded before every evaluation
        self.seed = setup.get("seed") if seed is None else seed
        self.theories = dict(setup.get("theories", {})) if theories is None else dict(theories)
        params_info = {**setup.get("params", {}), **(params_info or {})}
        self._model, self._like_name = self._build(params_info, packages_path)
        th = self._theory
        req = th.requests()
        # parameters requested from the theory that the model samples / fixes itself (candl asks for
        # its nuisances this way) are not theory products
        given = set(self._model.parameterization.sampled_params()) | set(self._model.parameterization.constant_params())
        req["params"] = [p for p in req["params"] if p not in given or p in _THEORY_INPUTS]
        self._req = req
        if check:  # native ports (``likelihoods.cobaya``) compute what they need themselves
            self._check(req)
        sampled = self._model.parameterization.sampled_params_info()
        # the likelihood's parameters and those of helper theory components (e.g. foregrounds)
        own = [p for p in sampled if p not in _THEORY_INPUTS]
        self.params = {p: _param_from_cobaya(p, sampled[p]) for p in own}
        self._own = tuple(own)
        # log-uniform priors: {name: (a, b)} (``log_prior`` turns their uniform box into 1/(x ln(b/a)))
        self._loguniform = {p: (self.params[p].lo, self.params[p].hi) for p in own
                            if (sampled[p].get("prior") or {}).get("dist") in ("loguniform", "reciprocal")}
        self._theory_in = tuple(p for p in _THEORY_INPUTS if p in sampled)
        # external prior terms: known ones in JAX (``log_prior``), the rest on the host (``host_loglike``)
        src = self._model.info().get("prior") or {}
        self._jax_priors, self._host_priors = [], []
        for name in self._model.prior.external:
            key = "".join(str(src.get(name, "")).split())
            (self._jax_priors if key in KNOWN_EXTERNAL_PRIORS else self._host_priors).append(
                KNOWN_EXTERNAL_PRIORS.get(key, name))
        self._ext_prior = bool(self._host_priors)
        self.requires = tuple(self.requirements())
        self.n_calls = 0

    # ---- set-up ----
    def _build(self, params_info, packages_path):
        """The Cobaya model; built twice when the yaml overrides some of the likelihood's own
        parameters (the first build tells which parameters are its own)."""
        from cobaya.model import get_model

        from gpucosmo.cobaya_bridge.provider import BridgeTheory

        if packages_path is None:
            from gpucosmo.paths import packages_path as _pp

            packages_path = _pp()

        def build(over):
            params = {p: {"prior": {"min": -_BIG, "max": _BIG}} for p in _THEORY_INPUTS}
            params.update(over)
            info = {"likelihood": {self.cobaya_name: self.block or None},
                    "theory": {"gpucosmo_bridge": {"external": BridgeTheory}, **self.theories},
                    "params": params, "debug": False}
            if packages_path:
                info["packages_path"] = str(packages_path)
            _clipy_jax_first()
            try:
                m = get_model(info)
                _numpy_clik(m)
                _theory_attr(m)
                return m
            except Exception as e:  # noqa: BLE001 (Cobaya raises LoggedError and friends)
                raise BridgeUnsupported(f"{self.cobaya_name}: Cobaya could not build it: "
                                        f"{type(e).__name__}: {e}") from e
            finally:
                os.environ.pop("CLIPY_NOJAX", None)

        # parameters the likelihood requires but no component defines (e.g. pyactlike's yp2, or Mb
        # for SN use_abs_mag declared in the yaml's top-level params): added and rebuilt
        need = {k: v for k, v in params_info.items() if k in known_setup(self.cobaya_name).get("params", {})}
        import re

        for _ in range(64):
            try:
                lg = logging.getLogger("model")
                level, lg.level = lg.level, logging.CRITICAL  # the retries' "Requirement ... not provided"
                try:
                    model = build(need)
                finally:
                    lg.setLevel(level)
                break
            except BridgeUnsupported as e:
                miss = [m for m in re.findall(r"Requirement (\S+) of", str(e)) if m in params_info and m not in need]
                if not miss:
                    raise
                need.update({m: params_info[m] for m in miss})
        name = list(model.likelihood)[0]
        mine = set(model.parameterization.sampled_params()) | set(model.parameterization.constant_params())
        for comp in [model.likelihood[name], *[model.theory[t] for t in model.theory if t != "gpucosmo_bridge"]]:
            mine |= set(getattr(comp, "params", None) or {})
        mine -= set(_THEORY_INPUTS)
        over = {k: v for k, v in params_info.items() if k in mine and need.get(k) != v}
        if over:
            model = build({**need, **over})
            name = list(model.likelihood)[0]
        self._theory = model.theory["gpucosmo_bridge"]
        return model, name

    def _check(self, req):
        bad = []
        if req["unknown"]:
            bad.append(f"products {req['unknown']}")
        if req["sigma_R"]:
            bad.append("sigma_R")
        if req["unlensed_Cl"]:
            bad.append("unlensed_Cl")
        if req["z_pairs"] is not None and np.any(req["z_pairs"] > 1100):
            bad.append("angular_diameter_distance_2 beyond z 1100")
        for which in ("Cl", "lensed_scal_Cl"):
            for s in req[which]:
                if _CL_ALIAS.get(s.lower(), s.lower()) not in _CL_PROVIDED:
                    bad.append(f"{which}[{s}]")
        from gpucosmo.jax.inference.mg_bundle import PK_KMAX_EXT

        for pk in req["Pk_grid"]:
            if pk["k_max"] > PK_KMAX_EXT:
                bad.append(f"Pk_grid to k = {pk['k_max']}/Mpc (provided to {PK_KMAX_EXT}/Mpc)")
            elif any(v not in _PK_VARS for v in pk["vars"]) or len(set(pk["vars"])) > 1:
                bad.append(f"Pk_grid of {pk['vars']}")
            elif pk["nonlinear"] and pk["vars"][0] != "delta_tot":
                bad.append(f"non-linear Pk_grid of {pk['vars']} (HMcode gives total matter)")
        for p in req["params"]:
            if p not in _BACKGROUND_DERIVED + _PK_DERIVED:
                bad.append(f"parameter {p}")
        if bad:
            raise BridgeUnsupported(f"{self.cobaya_name} needs {', '.join(bad)}, which the GPU theory does not "
                                    "provide")

    def _pk_z(self) -> list[float]:
        req = self._req
        z = set()
        for pk in req["Pk_grid"]:
            z |= {float(v) for v in pk["z"]}
        for q in ("sigma8_z", "fsigma8"):
            if q in req["z"]:
                z |= {float(v) for v in req["z"][q]}
        if set(req["params"]) & set(_PK_DERIVED):
            z.add(0.0)
        return sorted(z)

    def _pk_kmax(self) -> float:
        """The extended P(k) grid's k_max this likelihood implies (0: the solver grid suffices)."""
        from gpucosmo.jax.eb.linear_pk_jax import LINPK_K_MAX

        ks = [pk["k_max"] for pk in self._req["Pk_grid"] if pk["nonlinear"] or pk["k_max"] > LINPK_K_MAX]
        return max(max(ks), LINPK_K_MAX) if ks else 0.0

    def requirements(self) -> dict:
        """GPU theory blocks: {"cmb": {"lmax": int, "spectra": [...]}, "pk": {"z": [...], "k_max": float}}
        (empty if background only)."""
        out = {}
        zpk = self._pk_z()
        if zpk:
            out["pk"] = {"z": zpk, "k_max": self._pk_kmax(),
                         "nonlinear": any(pk["nonlinear"] for pk in self._req["Pk_grid"])}
        cl = {**self._req["Cl"], **self._req["lensed_scal_Cl"]}
        if cl:
            from gpucosmo.jax.inference.mg_bundle import cmb_lmax_supported

            # as CAMB: TT/TE/EE always and a few multipoles past the request (spt3g_2020 reads ℓ_max + 1,
            # sptpol_2017.TE/EE read both TE and EE)
            spectra = {_CL_ALIAS.get(s.lower(), s.lower()) for s in cl} | {"tt", "te", "ee"}
            if "pp" in spectra:  # lensing likelihoods' CMB-dependent corrections read BB too (act_dr6_lenslike)
                spectra.add("bb")
            spectra = sorted(spectra)
            lmax = int(max(cl.values()))
            out["cmb"] = {"lmax": max(lmax, min(lmax + CL_MARGIN, cmb_lmax_supported())), "spectra": spectra}
        return out

    def describe(self) -> dict:
        return {**super().describe(), "cobaya": self.cobaya_name, "bridge": True,
                "requirements": self.requirements()}

    def log_prior(self, params: dict):
        """Log-uniform priors as a normalized correction to their uniform box:
        log[(b − a) / (x ln(b/a))], so the box times this is Cobaya's density exactly; plus the
        likelihood's ``prior:`` terms with a JAX form (``KNOWN_EXTERNAL_PRIORS``, e.g. Planck's SZ)."""
        lp = 0.0
        for p, (a, b) in self._loguniform.items():
            lp = lp + (np.log(b - a) - np.log(np.log(b / a))) - jnp.log(params[p])
        for names, f in self._jax_priors:
            lp = lp + f(*(params[n] for n in names))
        return lp

    # ---- JAX side ----
    def theory(self, cosmo):
        """Background products at the requested redshifts + background-derived parameters."""
        from gpucosmo.jax.background.cosmology import (
            comoving_distance_highz,
            rs_drag,
            sound_horizon,
            transverse_distance,
            z_drag_of,
            z_star_of,
        )

        req = self._req
        out = {}

        def dc(z):
            z = np.asarray(z, dtype=float)
            lo = z <= 10.0
            res = []
            if lo.any():
                res.append((np.flatnonzero(lo), cosmo.comoving_distance(z[lo])))
            if (~lo).any():
                zz = jnp.asarray(z[~lo])
                res.append((np.flatnonzero(~lo),
                            jax.vmap(lambda q: comoving_distance_highz(cosmo._tables, cosmo._bp, q))(zz)))
            v = jnp.zeros(z.shape, dtype=jnp.float64)
            for i, r in res:
                v = v.at[i].set(r)
            return v

        t = cosmo._tables
        h2 = cosmo.h**2
        for q, z in req["z"].items():
            if q in ("sigma8_z", "fsigma8"):
                continue  # from the "pk" block (``loglike``)
            if q == "Hubble":
                out[q] = cosmo.H(jnp.asarray(z))
            elif q == "comoving_radial_distance":
                out[q] = dc(z)
            elif q == "angular_diameter_distance":
                out[q] = transverse_distance(t, cosmo._bp, dc(z)) / (1.0 + jnp.asarray(z))
            elif q in ("Omega_b", "Omega_cdm", "Omega_nu_massive"):
                e2 = (cosmo.H(jnp.asarray(z)) / cosmo.H0) ** 2
                zp = 1.0 + jnp.asarray(z)
                if q == "Omega_nu_massive":
                    from gpucosmo.jax.eb.massive_nu import thermal_nu_rho

                    st = cosmo._bp
                    rho = (jnp.sum(st.nu_om_h2 * thermal_nu_rho(st.nu_am / zp[..., None]), axis=-1) * zp**4
                           if st.nu_am.shape[-1] else jnp.zeros_like(zp))
                else:
                    rho = (cosmo.ombh2 if q == "Omega_b" else cosmo.omch2) * zp**3
                out[q] = rho / h2 / e2
        if req["z_pairs"] is not None:
            zp = np.asarray(req["z_pairs"])
            u, inv = np.unique(zp.ravel(), return_inverse=True)
            d = dc(u)[inv].reshape(zp.shape)
            # CAMB angular_diameter_distance2: f_K(χ2 − χ1)/(1 + z2)
            out["angular_diameter_distance_2"] = (transverse_distance(t, cosmo._bp, d[:, 1] - d[:, 0])
                                                  / (1.0 + jnp.asarray(zp[:, 1])))
        want = set(req["params"])
        if want & set(_PK_DERIVED):
            want |= {"omegam", "h"}  # for S8 & co. (``products``)
        if want:
            out["params"] = self._derived(cosmo, want, dc, rs_drag, sound_horizon, z_drag_of, z_star_of)
        return out

    @staticmethod
    def _derived(cosmo, want, dc, rs_drag, sound_horizon, z_drag_of, z_star_of):
        from gpucosmo.jax.eb.bbn_jax import delta_neff_for, yhe_from_bbn, yp_bbn

        t = cosmo._tables
        h = cosmo.h
        st = cosmo._bp
        p = {}
        zs = z_star_of(t, st)
        dneff = delta_neff_for(st.nnu)
        for n in want:
            if n == "H0":
                p[n] = 100.0 * h
            elif n == "h":
                p[n] = h
            elif n in ("ombh2", "omch2", "ns", "tau", "As"):
                p[n] = cosmo.params[n]
            elif n == "logA":
                p[n] = jnp.log(1e10 * cosmo.As)
            elif n in ("w", "wa"):
                p[n] = cosmo.params["w0" if n == "w" else "wa"]
            elif n in ("mu0", "Sigma0"):
                v = cosmo.params.get(n)
                p[n] = jnp.asarray(0.0) if v is None else v
            elif n == "omegam":
                p[n] = cosmo.Omega_m
            elif n == "omegab":
                p[n] = cosmo.ombh2 / h**2
            elif n == "omegac":
                p[n] = cosmo.omch2 / h**2
            elif n == "omegamh2":
                p[n] = cosmo.Omega_m * h**2
            elif n in ("omnuh2", "omeganuh2"):
                p[n] = st.omnuh2
            elif n == "omega_de":
                p[n] = 1.0 - cosmo.Omega_m - st.omega_rad_h2 / h**2 - st.omk
            elif n == "rdrag":
                p[n] = cosmo.rdrag
            elif n == "zdrag":
                p[n] = z_drag_of(t, st)
            elif n == "zstar":
                p[n] = zs
            elif n == "rstar":
                p[n] = sound_horizon(t, st, zs)
            elif n == "thetastar":
                p[n] = 100.0 * cosmo.theta_star
            elif n == "theta_MC_100":
                p[n] = 100.0 * cosmo.theta_star  # our sampled 100θ* (``cobaya_yaml`` note)
            elif n == "cosmomc_theta":
                from gpucosmo.jax.background.cosmology import cosmomc_theta

                p[n] = cosmomc_theta(t, cosmo._bp)
            elif n == "DAstar":
                p[n] = _dastar(cosmo, zs) / 1000.0
            elif n == "YHe":
                p[n] = yhe_from_bbn(cosmo.ombh2, delta_neff=dneff)
            elif n == "Y_p":
                p[n] = yp_bbn(cosmo.ombh2, delta_neff=dneff)
            elif n in ("Neff", "nnu"):
                p[n] = st.nnu  # CAMB N_eff = massless + Σ degeneracies = nnu
            elif n in ("mnu", "omk", "meffsterile"):
                p[n] = getattr(st, n)
            elif n == "TCMB":
                p[n] = jnp.asarray(2.7255)
        return {k: jnp.asarray(v, dtype=jnp.float64) for k, v in p.items()}

    def loglike(self, inp):
        th = dict(inp.theory or {})
        cmb = None
        if "cmb" in self.requires:
            c = inp.shared["cmb"].get("cl")
            if c is None:
                raise BridgeUnsupported(f"{self.label}: the CMB block has no 'cl' spectra (bundle too old?)")
            cl_req = self.requirements()["cmb"]
            n = cl_req["lmax"] + 1
            cmb = {s: c[s][:n] for s in cl_req["spectra"]}
        nuis = jnp.stack([jnp.asarray(inp.params[p], dtype=jnp.float64) for p in self._own]) if self._own else \
            jnp.zeros(0)
        cp = inp.cosmo.params
        tin = jnp.stack([jnp.asarray(v, dtype=jnp.float64) for v in (
            100.0 * cp["h"], cp["ombh2"], cp["omch2"], jnp.log(1e10 * cp["As"]), cp["As"], cp["ns"], cp["tau"],
            cp["w0"], cp["wa"], 0.0 if cp.get("mu0") is None else cp["mu0"],
            0.0 if cp.get("Sigma0") is None else cp["Sigma0"], inp.cosmo.mnu, inp.cosmo.nnu, inp.cosmo.omk,
            inp.cosmo.meffsterile)])
        pk = self.pk_rows(inp.shared["pk"]) if "pk" in self.requires else None
        args = (th, cmb, nuis, tin, pk)
        return jax.pure_callback(self._host, jax.ShapeDtypeStruct((), jnp.float64), *args,
                                 vmap_method="sequential")

    def pk_rows(self, blk: dict) -> dict:
        """The rows of the shared "pk" block (``mg_bundle.shared_block``) at this likelihood's redshifts."""
        zall = np.asarray(blk["z"], dtype=float)
        zmine = np.asarray(self._pk_z())
        sel = np.asarray([int(np.argmin(np.abs(zall - z))) for z in zmine])
        if np.max(np.abs(zall[sel] - zmine)) > 1e-8:
            raise BridgeUnsupported(f"{self.label}: P(k) block lacks redshifts {zmine} (has {zall})")
        self._k = None if blk.get("k") is None else np.asarray(blk["k"], dtype=float)  # static: the solver grid
        self._k_x = None if blk.get("k_x") is None else np.asarray(blk["k_x"])  # static: the extended grid
        return {k: blk[k][sel] for k in ("pk_m", "pk_cb", "sigma8", "fsigma8", "pk_m_x", "pk_m_nl") if k in blk}

    # ---- host side ----
    def products(self, th, cmb, pk=None) -> dict:
        """Cobaya products (``BridgeTheory.set_products``) from the JAX outputs (NumPy)."""
        from gpucosmo.jax.cmb.params import T_CMB_MUK

        prods = {k: np.asarray(v, dtype=float) for k, v in th.items() if k != "params"}
        prods["params"] = {k: float(v) for k, v in (th.get("params") or {}).items()}
        if cmb is not None:
            cl = {}
            for s, v in cmb.items():
                v = np.asarray(v, dtype=float)
                cl[s] = v if s == "pp" else v / T_CMB_MUK**2  # pp: [ℓ(ℓ+1)]²C^φφ/2π already (shared_block)
            prods["Cl"] = cl
        if pk is not None:
            from gpucosmo.jax.eb.linear_pk_jax import linpk_k_grid

            from gpucosmo.jax.eb.linear_pk_jax import LINPK_K_MAX

            zp = np.asarray(self._pk_z())
            k = linpk_k_grid() if getattr(self, "_k", None) is None else self._k  # the block's static axis
            for g in self._req["Pk_grid"]:
                v = g["vars"][0]
                key = tuple(sorted(g["vars"]))
                if g["nonlinear"] or (g["k_max"] > LINPK_K_MAX and v == "delta_tot"):
                    kx = self._k_x
                    prods[("Pk_grid", False) + key] = (kx, zp, np.asarray(pk["pk_m_x"], dtype=float))
                    if g["nonlinear"]:
                        prods[("Pk_grid", True) + key] = (kx, zp, np.asarray(pk["pk_m_nl"], dtype=float))
                else:
                    prods[("Pk_grid", False) + key] = (k, zp, np.asarray(pk[_PK_VARS[v]], dtype=float))
            for q, key in (("sigma8_z", "sigma8"), ("fsigma8", "fsigma8")):
                if q in self._req["z"]:
                    zq = self._req["z"][q]
                    idx = [int(np.argmin(np.abs(zp - z))) for z in zq]
                    prods[q] = np.asarray(pk[key], dtype=float)[idx]
            if 0.0 in zp:
                s8 = float(np.asarray(pk["sigma8"])[int(np.argmin(np.abs(zp)))])
                om, h = prods["params"].get("omegam"), prods["params"].get("h")
                prods["params"]["sigma8"] = s8
                if om is not None:
                    prods["params"].update(S8=s8 * (om / 0.3) ** 0.5, s8omegamp5=s8 * om**0.5,
                                           s8omegamp25=s8 * om**0.25)
                if h is not None:
                    prods["params"]["s8h5"] = s8 / h**0.5
        return prods

    def point(self, nuis, tin) -> dict:
        tin = np.asarray(tin)
        if tin.shape[-1] != len(_THEORY_INPUTS):
            raise ValueError(f"theory inputs: {tin.shape[-1]} values for {len(_THEORY_INPUTS)} names {_THEORY_INPUTS}")
        pt = dict(zip(_THEORY_INPUTS, map(float, tin)))
        pt = {k: v for k, v in pt.items() if k in self._theory_in}
        pt.update(zip(self._own, map(float, np.asarray(nuis))))
        return pt

    def _host(self, th, cmb, nuis, tin, pk=None):
        self.n_calls += 1
        prods = self.products(th, cmb, pk)
        pt = self.point(nuis, tin)
        prods["params"].update({k: v for k, v in pt.items() if k in _THEORY_INPUTS})
        return np.asarray(self.host_loglike(prods, pt), dtype=np.float64)

    def host_loglike(self, prods: dict, pt: dict) -> float:
        """log L (+ the likelihood's own prior terms) for Cobaya products and a parameter point."""
        self._theory.set_products(prods)
        if self.seed is not None:
            np.random.seed(self.seed)
        try:
            ll = float(np.sum(self._model.loglikes(pt, return_derived=False, cached=False)))
            if self._ext_prior:  # the prior terms without a JAX form (``log_prior`` has the others)
                inp = self._model.parameterization.to_input(self._model.parameterization.check_sampled(
                    {**{k: 0.0 for k in self._model.parameterization.sampled_params()}, **pt}))
                ext = self._model.prior.external
                ll += float(sum(ext[n].logp(*(inp[q] for q in ext[n].params)) for n in self._host_priors))
        except Exception:  # noqa: BLE001 (any failure inside the likelihood = zero likelihood)
            return -np.inf
        return ll if np.isfinite(ll) else -np.inf


def _dastar(cosmo, zs):
    from gpucosmo.jax.background.cosmology import comoving_distance_highz, transverse_distance

    st = cosmo._bp
    return transverse_distance(cosmo._tables, st, comoving_distance_highz(cosmo._tables, st, zs)) / (1.0 + zs)


__all__ = ["BridgeUnsupported", "CobayaLikelihood"]
