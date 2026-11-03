"""Native ports of Cobaya's clik-based Planck likelihoods (``base_classes.planck_clik``).

Cobaya evaluates these with clipy, which has a JAX build: the port calls the very clipy object
Cobaya loaded (the bridge imports clipy's JAX backend first, ``cobaya_bridge.bridge``) on the
vector Cobaya assembles — C_ℓ in μK² from ℓ = 0 for each requested spectrum (``pp``
dimensionless, first for lensing; ``tb``/``eb`` zero), then the nuisance parameters in clik's
order — so the value is Cobaya's, jitted, vmapped and differentiable.

Covers the plik TT/TE/EE/TTTEEE (binned, unbinned, lite), the Commander/SimAll low-ℓ and the
Planck 2018 lensing clik files. (The built-in PR3 set, ``jax.cmb.clipy_plik``, uses clipy's
candl interface on the same files.)
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from gpucosmo.likelihoods.cobaya.base import CobayaPort

# The unbinned files stay on the bridge: their covariances, embedded as constants in each compiled
# program, exceed the host memory here (TTTEEE_unbinned killed at 7 GB, TT_unbinned at 4 GB while
# compiling; TT_unbinned loads only with clipy's JAX build, which the bridge imports first).
_PLIK = tuple(f"planck_2018_highl_plik.{v}" for v in ("TT", "TE", "EE", "TTTEEE", "TT_lite", "TTTEEE_lite"))


class PlikClik(CobayaPort):
    """``planck_2018_highl_plik.*`` (clik files)."""

    cobaya_names = _PLIK

    def setup(self, like):
        from gpucosmo.jax.cmb.clipy_jax_compat import sanitize_clipy_templates

        f = like.clik_likelihood
        self._clik = getattr(f, "_gpucosmo_orig", f)  # the clipy object (unwrapped, see bridge._numpy_clik)
        sanitize_clipy_templates(self._clik)  # 0/0 template entries: same log L, finite gradients
        self.spectra = [(s, int(l)) for s, l in like.requested_cls_lmax.items()]
        self.expected = list(like.expected_params)
        missing = [p for p in self.expected if p not in self.params and p not in self.fixed_params()]
        if missing:
            raise NotImplementedError(f"clik parameters {missing} are neither sampled nor fixed")

    def cmb_lmax(self) -> int:
        """``loglike`` reads each requested spectrum to its requested ℓ_max."""
        return max(lmax for _, lmax in self.spectra)

    def fixed_params(self) -> dict:
        """The likelihood's parameters fixed in the Cobaya model (values passed to clik)."""
        model = self.bridge._model
        const = model.parameterization.constant_params()
        return {p: float(const[p]) for p in self.expected if p in const}

    def loglike(self, inp):
        cl = inp.shared["cmb"]["cl"]
        parts = []
        for s, lmax in self.spectra:
            if s in ("tb", "eb"):
                parts.append(jnp.zeros(lmax + 1))
                continue
            ell = np.arange(lmax + 1, dtype=float)
            if s == "pp":  # [ℓ(ℓ+1)]² C/2π → C^φφ (provider.get_Cl, no ℓ factor)
                fac = np.where(ell > 0, 2 * np.pi / np.maximum(ell * (ell + 1), 1.0) ** 2, 0.0)
            else:  # D_ℓ → C_ℓ, μK² (FIRAS)
                fac = np.where(ell > 0, 2 * np.pi / np.maximum(ell * (ell + 1), 1.0), 0.0)
            parts.append(cl[s][: lmax + 1] * jnp.asarray(fac))
        fixed = self.fixed_params()
        nuis = [jnp.asarray(inp.params[p], dtype=jnp.float64) if p in inp.params else jnp.asarray(fixed[p])
                for p in self.expected]
        vec = jnp.concatenate(parts + [jnp.stack(nuis)] if nuis else parts)
        ll = jnp.asarray(self._clik(vec), dtype=jnp.float64).reshape(())
        bad = jnp.isnan(jnp.dot(vec, vec)) | jnp.isnan(ll) | (ll <= -1e30)
        return jnp.where(bad, -jnp.inf, ll)


class LowlClik(PlikClik):
    """``planck_2018_lowl.TT_clik`` (Commander) and ``EE_clik`` (SimAll)."""

    cobaya_names = ("planck_2018_lowl.TT_clik", "planck_2018_lowl.EE_clik")


class LensingClik(PlikClik):
    """``planck_2018_lensing.clik``."""

    cobaya_names = ("planck_2018_lensing.clik",)


__all__ = ["LensingClik", "LowlClik", "PlikClik"]
