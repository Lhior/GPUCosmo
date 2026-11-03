"""Two user likelihoods, loaded with ``--plugin examples/custom_likelihood/my_likelihoods.py``.

* ``cosmic_chronometers`` — a Gaussian likelihood (H(z) measurements): implement ``model``.
* ``tt_band_power`` — a general likelihood with a nuisance parameter that reads the built-in
  lensed CMB spectra (``requires = ("cmb",)``; needs an analysis with a primary CMB
  likelihood). A toy, to show the interface.
"""

import jax.numpy as jnp
import numpy as np

from gpucosmo.likelihoods import GaussianLikelihood, Likelihood, Param, register_likelihood

# Moresco et al. 2016 / Ratsimbazafy et al. 2017-like H(z) points [km/s/Mpc] (illustrative)
CC_Z = [0.1791, 0.3519, 0.5929, 0.6797, 0.7812, 0.8754, 1.037, 1.363, 1.965]
CC_H = [75.0, 83.0, 104.0, 92.0, 105.0, 125.0, 154.0, 160.0, 186.5]
CC_SIGMA = [4.0, 14.0, 13.0, 8.0, 12.0, 17.0, 20.0, 33.6, 50.4]


@register_likelihood
class CosmicChronometers(GaussianLikelihood):
    """H(z) from cosmic chronometers (diagonal covariance by default)."""

    name = "cosmic_chronometers"

    def __init__(self, z=CC_Z, H=CC_H, sigma=CC_SIGMA, cov=None, **options):
        self.z = np.asarray(z, dtype=float)
        cov = np.diag(np.asarray(sigma, dtype=float) ** 2) if cov is None else cov
        super().__init__(data=H, cov=cov, **options)

    def model(self, cosmo):
        # cosmology only: the DALI approximation expands this vector; exact runs evaluate it
        return cosmo.H(self.z)


@register_likelihood
class TTBandPower(Likelihood):
    """Toy: mean D_ℓ^TT over a multipole band with a calibration nuisance ``tt_cal``."""

    name = "tt_band_power"
    requires = ("cmb",)
    params = {"tt_cal": Param(0.9, 1.1, gauss=(1.0, 0.01), latex=r"y_{TT}")}

    def __init__(self, lmin=200, lmax=250, value=5700.0, sigma=60.0, **options):
        super().__init__(**options)
        self.lmin, self.lmax, self.value, self.sigma = int(lmin), int(lmax), float(value), float(sigma)

    def loglike(self, inp):
        dl = inp.shared["cmb"]["dl_tt"][self.lmin:self.lmax + 1]  # μK², exact or DALI
        band = inp.params["tt_cal"] ** 2 * jnp.mean(dl)
        return -0.5 * ((band - self.value) / self.sigma) ** 2
