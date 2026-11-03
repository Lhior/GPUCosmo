"""Base class of the native (JAX) ports of Cobaya likelihoods.

A port reuses Cobaya for everything but the likelihood evaluation: ``CobayaPort.__init__``
builds the Cobaya likelihood through the bridge (``cobaya_bridge.CobayaLikelihood``) — so the
data files, yaml options, data-set selection and parameter priors are Cobaya's own — and
hands the loaded Cobaya instance to ``setup``, where the port copies the arrays it needs.
``loglike`` then re-implements Cobaya's ``logp`` in ``jax.numpy`` on the same products the
bridge would pass (``theory``: the bridge's background products; ``shared["cmb"]["cl"]``:
the CMB spectra), so the two can be compared value for value
(``likelihoods.cobaya.testing.compare``) and the port runs jitted, vmapped and differentiated
on the GPU.

A subclass sets ``cobaya_names`` (the Cobaya likelihoods it serves) and implements ``setup``
and ``loglike``; when its Cobaya likelihood has ``prior:`` terms it also implements
``log_prior`` (or adds them to ``loglike``) and sets ``handles_external_priors = True``.
"""

from __future__ import annotations

import numpy as np

from gpucosmo.likelihoods.base import Likelihood


class CobayaPort(Likelihood):
    """Native port of one or more Cobaya likelihoods (``cobaya_names``)."""

    cobaya_names: tuple[str, ...] = ()
    allow_shared_params = True
    handles_external_priors = False

    def __init__(self, cobaya_name: str | None = None, block: dict | None = None, *,
                 params_info: dict | None = None, packages_path: str | None = None, name: str | None = None,
                 **options):
        from gpucosmo.cobaya_bridge.bridge import CobayaLikelihood

        cobaya_name = cobaya_name or (self.cobaya_names[0] if self.cobaya_names else None)
        if cobaya_name is None:
            raise ValueError(f"{type(self).__name__}: no Cobaya likelihood name")
        block = {**(block or {}), **options}
        super().__init__(**({"name": name} if name else {}))
        self.name = name or cobaya_name
        self.cobaya_name = cobaya_name
        self.block = block
        self.bridge = CobayaLikelihood(cobaya_name, block, params_info=params_info, packages_path=packages_path,
                                       name=self.name, check=False)
        self.cobaya = self.bridge._model.likelihood[self.bridge._like_name]
        self.params = dict(self.bridge.params)
        self.requires = self.bridge.requires
        # ``prior:`` terms with a JAX form come through ``log_prior`` (the bridge's); others need the port
        if self.bridge._host_priors and not self.handles_external_priors:
            raise NotImplementedError(f"{type(self).__name__}: {cobaya_name} has prior terms "
                                      f"{self.bridge._host_priors} without a JAX form")
        self.setup(self.cobaya)

    # ---- for subclasses ----
    def setup(self, like) -> None:
        """Copy what ``loglike`` needs from the loaded Cobaya likelihood ``like`` (NumPy)."""
        raise NotImplementedError

    def requirements(self) -> dict:
        return self.bridge.requirements()

    def theory(self, cosmo):
        """The bridge's background products (same keys, so ``loglike`` mirrors Cobaya's getters)."""
        return self.bridge.theory(cosmo)

    def log_prior(self, params: dict):
        """The Cobaya priors the parameter boxes do not express (log-uniform, known ``prior:`` terms;
        ``bridge.log_prior``)."""
        return self.bridge.log_prior(params)

    def describe(self) -> dict:
        return {**super().describe(), "cobaya": self.cobaya_name, "native": True,
                "requirements": self.requirements()}

    # ---- helpers mirroring Cobaya's provider getters on the products ----
    def z_index(self, quantity: str, z) -> np.ndarray:
        """Indices of the redshifts ``z`` in the product array ``quantity`` (static)."""
        pool = self.bridge._theory._pools[quantity]
        return np.asarray(pool.find_indices(np.atleast_1d(np.asarray(z, dtype=float))), dtype=int)


__all__ = ["CobayaPort"]
