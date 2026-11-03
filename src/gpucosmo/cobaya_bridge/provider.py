"""A Cobaya theory component whose products come from outside: the GPU theory.

``BridgeTheory`` is what a Cobaya likelihood sees as its Boltzmann code inside the bridge
(``cobaya_bridge.bridge.CobayaLikelihood``): Cobaya hands it the likelihood's requirements
(``must_provide``), from which ``requests()`` derives the arrays the GPU theory must compute;
before each evaluation the bridge stores those arrays with ``set_products``, and
``calculate`` copies them into Cobaya's state. The getters follow CAMB's wrapper
(``cobaya.theories.camb``) conventions: units, ``ell_factor``, z pools, ``Pk_grid``
layout.

Products (``set_products`` keys, NumPy, one cosmology):

* ``params`` — {name: value}: input and derived parameters (``H0``, ``ombh2``, ``rdrag``, ...);
* ``Hubble``, ``angular_diameter_distance``, ``comoving_radial_distance``, ``Omega_b``,
  ``Omega_cdm``, ``Omega_nu_massive``, ``sigma8_z``, ``fsigma8`` — arrays on the requested
  z pool (``requests()["z"][quantity]``, ascending); ``angular_diameter_distance_2`` on the
  pair pool;
* ``Cl`` / ``unlensed_Cl`` — {"tt", "ee", "bb", "te", "pp"?: dimensionless
  ℓ(ℓ+1)C_ℓ/2π (``pp``: [ℓ(ℓ+1)]² C_ℓ^φφ/2π)}, ℓ = 0 … lmax (CAMB's ``CL`` arrays);
* ``("Pk_grid", nonlinear, var1, var2)`` — (k [1/Mpc], z ascending, P[z, k] [Mpc³]).
"""

from __future__ import annotations

import numpy as np
from cobaya.log import LoggedError
from cobaya.theories.cosmo.boltzmannbase import BoltzmannBase
from cobaya.tools import Pool1D, Pool2D

T_CMB = 2.7255  # K, the solver's (eb.constants)

# Parameters the GPU theory supplies (inputs of our basis and derived quantities).
INPUT_PARAMS = ("H0", "ombh2", "omch2", "logA", "As", "ns", "tau", "w", "wa", "mu0", "Sigma0",
                "theta_MC_100", "cosmomc_theta", "mnu", "nnu", "omk", "meffsterile", "r")
DERIVED_PARAMS = ("h", "omegam", "omegab", "omegac", "omega_de", "omegamh2", "omeganuh2", "omnuh2",
                  "rdrag", "zdrag", "rstar", "zstar", "thetastar", "theta_s_100", "DAstar",
                  "sigma8", "S8", "s8h5", "s8omegamp5", "s8omegamp25", "YHe", "Y_p", "Neff", "TCMB",
                  "age", "zrei")
_Z_QUANTITIES = ("Hubble", "angular_diameter_distance", "comoving_radial_distance", "Omega_b",
                 "Omega_cdm", "Omega_nu_massive", "sigma8_z", "fsigma8")
_H_UNITS = {"km/s/Mpc": 1.0, "1/Mpc": 1.0 / 299792.458}


class BridgeTheory(BoltzmannBase):
    """Cobaya theory serving products computed elsewhere (``set_products``)."""

    # Cobaya class options
    _products: dict | None = None
    file_base_name = "gpucosmo_bridge"
    extra_args: dict = {}
    renames: dict = {}

    def initialize(self):
        super().initialize()
        self._products = None
        self._pools: dict = {}

    def get_allow_agnostic(self):
        return True

    def get_can_provide_params(self):
        return list(DERIVED_PARAMS) + list(INPUT_PARAMS)

    def get_can_support_params(self):
        return list(INPUT_PARAMS)

    def must_provide(self, **requirements):
        super().must_provide(**requirements)
        self._pools = {}
        for k, v in self._must_provide.items():
            if k in _Z_QUANTITIES:
                self._pools[k] = Pool1D(v["z"])
            elif k == "angular_diameter_distance_2":
                self._pools[k] = Pool2D(v["z_pairs"])
        return {}

    # ---- what the GPU theory must compute ----
    def requests(self) -> dict:
        """The requirements in the GPU theory's terms::

            {"z": {quantity: z array}, "z_pairs": array (n, 2) | None,
             "params": [names], "Cl": {spectrum: lmax}, "unlensed_Cl": {...},
             "Pk_grid": [{"nonlinear", "vars", "z", "k_max"}, ...]}
        """
        mp = self._must_provide or {}
        out = {"z": {}, "z_pairs": None, "params": [], "Cl": {}, "unlensed_Cl": {}, "lensed_scal_Cl": {},
               "Pk_grid": [], "sigma_R": [], "unknown": []}
        for k, v in mp.items():
            if k in _Z_QUANTITIES:
                out["z"][k] = np.asarray(self._pools[k].values, dtype=float)
            elif k == "angular_diameter_distance_2":
                out["z_pairs"] = np.asarray(self._pools[k].values, dtype=float)
            elif k in ("Cl", "unlensed_Cl", "lensed_scal_Cl"):
                out[k] = {s: int(lm) for s, lm in v.items()}
            elif isinstance(k, tuple) and k[0] == "Pk_grid":
                out["Pk_grid"].append(dict(nonlinear=bool(k[1]), vars=tuple(k[2:]),
                                           z=np.asarray(v["z"], dtype=float), k_max=float(v["k_max"])))
            elif isinstance(k, tuple) and k[0] == "sigma_R":
                out["sigma_R"].append(dict(vars=tuple(k[1:]), z=np.asarray(v["z"]), R=np.asarray(v["R"])))
            elif v is None:
                out["params"].append(k)
            else:
                out["unknown"].append(k)
        return out

    # ---- products ----
    def set_products(self, products: dict) -> None:
        self._products = products

    def calculate(self, state, want_derived=True, **params_values_dict):
        p = self._products
        if p is None:
            raise LoggedError(self.log, "BridgeTheory: no products set (call set_products first)")
        for k, v in p.items():
            if k != "params":
                state[k] = v
        pars = dict(p.get("params", {}))
        state["derived_extra"] = {"TCMB": T_CMB}
        if want_derived:
            state["derived"] = {k: pars.get(k) for k in self.output_params}
        state["params_all"] = pars
        return True

    def get_param(self, p):
        st = self.current_state
        for pool in ("params_all", "derived", "params"):
            d = st.get(pool) or {}
            if p in d and d[p] is not None:
                return d[p]
        raise LoggedError(self.log, "Parameter not provided by the GPU theory: '%s'", p)

    def _get_z_dependent(self, quantity, z, pool=None):
        pool = self._pools.get(quantity) if pool is None else pool
        if pool is None:
            raise LoggedError(self.log, f"{quantity} was not requested")
        try:
            i = pool.find_indices(z)
        except ValueError:
            raise LoggedError(self.log, f"{quantity} not computed for all z requested: {z} vs {pool.values}")
        return np.array(self.current_state[quantity], copy=True)[i]

    def get_Hubble(self, z, units="km/s/Mpc"):
        if units not in _H_UNITS:
            raise LoggedError(self.log, "Units not known for H: '%s'", units)
        return self._get_z_dependent("Hubble", z) * _H_UNITS[units]

    def _get_Cl(self, which, ell_factor=False, units="FIRASmuK2"):
        try:
            cls_in = self.current_state[which]
        except KeyError:
            raise LoggedError(self.log, "No %s were computed; were they requested?", which)
        uf = self._cmb_unit_factor(units, T_CMB)
        n = len(next(iter(cls_in.values())))
        ls = np.arange(n, dtype=np.int64)
        out = {"ell": ls}
        l1 = ls[1:].astype(float)
        llp1 = l1 * (l1 + 1)
        for sp, cl in cls_in.items():
            cl = np.array(cl, dtype=float, copy=True)
            if sp == "pp":
                if not ell_factor:
                    cl[1:] /= llp1**2 / (2 * np.pi)
            elif sp in ("pt", "pe", "tp", "ep"):
                cl *= uf
                if not ell_factor:
                    cl[1:] /= llp1**1.5 / (2 * np.pi)
            else:
                if not ell_factor:
                    cl[1:] /= llp1 / (2 * np.pi)
                cl *= uf**2
            out[sp] = cl
        if "te" in out:
            out["et"] = out["te"]
        for a, b in (("pt", "tp"), ("pe", "ep")):
            if a in out or b in out:
                out[a] = out[b] = out.get(a, out.get(b))
        return out

    def get_Cl(self, ell_factor=False, units="FIRASmuK2"):
        return self._get_Cl("Cl", ell_factor, units)

    def get_unlensed_Cl(self, ell_factor=False, units="FIRASmuK2"):
        return self._get_Cl("unlensed_Cl", ell_factor, units)

    def get_lensed_scal_Cl(self, ell_factor=False, units="FIRASmuK2"):
        return self._get_Cl("lensed_scal_Cl" if "lensed_scal_Cl" in self.current_state else "Cl",
                            ell_factor, units)

    def get_source_Cl(self):
        raise LoggedError(self.log, "source_Cl (CAMB sources) is not provided by the GPU theory")


def products_template(req: dict) -> dict:
    """Zero-filled products with the shapes ``requests()`` implies (used to check a likelihood
    initializes and to size the JAX outputs)."""
    out: dict = {"params": {}}
    for q, z in req["z"].items():
        out[q] = np.zeros(len(z))
    if req["z_pairs"] is not None:
        out["angular_diameter_distance_2"] = np.zeros(len(req["z_pairs"]))
    for which in ("Cl", "unlensed_Cl"):
        if req[which]:
            lmax = max(req[which].values())
            out[which] = {s: np.zeros(lmax + 1) for s in _cl_keys(req[which])}
    return out


def _cl_keys(spec: dict) -> list[str]:
    """CAMB array columns needed for the requested spectra ({"tt": l, "te": l, "pp": l, ...})."""
    keys = set()
    for s in spec:
        s = s.lower()
        if s in ("tt", "ee", "bb", "te", "pp"):
            keys.add(s)
        elif s == "et":
            keys.add("te")
        elif s in ("tp", "pt", "ep", "pe"):
            keys.add("tp" if "t" in s else "ep")
        else:
            keys.add(s)
    return sorted(keys)


__all__ = ["DERIVED_PARAMS", "INPUT_PARAMS", "BridgeTheory", "T_CMB", "products_template"]
