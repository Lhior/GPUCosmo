# This file is part of GPUCosmo. Copyright (C) 2025-2026 Leonel Medina-Varela.
# SPDX-License-Identifier: GPL-3.0-or-later
# Derived in part from the following, translated to JAX and modified for GPUCosmo:
# - Cobaya (https://github.com/CobayaSampler/cobaya), Copyright 2017-2026 Jesus Torrado and
#   Antony Lewis.
#   Licence: LICENSES/Cobaya.txt.
# See NOTICE.md.
"""Native ports of Cobaya's background likelihoods: BAO (``bao.*``) and H0 / M_b (``H0.*``)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from gpucosmo.likelihoods.cobaya.base import CobayaPort
from gpucosmo.likelihoods.cobaya.splines import RegularGrid, Spline1D, Spline2D

C_KM_S = 299792.458

# Cobaya BAO data sets (those with f σ8 read the linear "pk" block's f σ8(z)).
BAO_NAMES = (
    "bao.desi_2024_bao_all", "bao.desi_2024_bao_bgs_z1", "bao.desi_2024_bao_elg_z2", "bao.desi_2024_bao_lrg_z1",
    "bao.desi_2024_bao_lrg_z2", "bao.desi_2024_bao_lrgpluselg_z1", "bao.desi_2024_bao_lya",
    "bao.desi_2024_bao_qso_z1", "bao.desi_2024_eboss_bao_lya", "bao.desi_dr2", "bao.desi_dr2.desi_bao_all",
    "bao.desi_dr2.desi_bao_bgs", "bao.desi_dr2.desi_bao_elg2", "bao.desi_dr2.desi_bao_lrg1",
    "bao.desi_dr2.desi_bao_lrg2", "bao.desi_dr2.desi_bao_lrg3pluselg1", "bao.desi_dr2.desi_bao_lya",
    "bao.desi_dr2.desi_bao_qso", "bao.sdss_dr12_consensus_bao", "bao.sdss_dr12_lrg_bao_dmdh", "bao.sdss_dr16_bao_elg",
    "bao.sdss_dr16_baoplus_lyauto", "bao.sdss_dr16_baoplus_lyxqso", "bao.sdss_dr16_lrg_bao_dmdh",
    "bao.sdss_dr16_qso_bao_dmdh", "bao.sdss_dr7_mgs", "bao.sixdf_2011_bao", "bao.generic",
    "bao.sdss_dr12_consensus_final", "bao.sdss_dr12_consensus_full_shape", "bao.sdss_dr16_baoplus_elg",
    "bao.sdss_dr16_baoplus_lrg", "bao.sdss_dr16_baoplus_qso",
)


class BAO(CobayaPort):
    """Cobaya ``base_classes.BAO``: Gaussian on D_V/r_d, D_M/r_d, D_H/r_d, ... (``measurements_file``
    + covariance), or a tabulated likelihood (``grid_file`` 1–3D, ``prob_dist``)."""

    cobaya_names = BAO_NAMES

    def setup(self, like):
        self.rs_rescale = float(like.rs_rescale)
        if getattr(like, "prob_dist", None):
            self.mode = "prob"
            self.obs = [str(like.data["observable"].iloc[0])]
            self.z = np.asarray([float(like.data["z"].iloc[0])])
            # the spline is a closure in like.logpdf: rebuild it as Cobaya does
            import os

            from scipy.interpolate import UnivariateSpline

            path = os.path.normpath(getattr(like, "path", None) or os.path.join(like.packages_path, "data"))
            chi2 = np.loadtxt(os.path.join(path, like.prob_dist))
            alpha = np.linspace(like.prob_dist_bounds[0], like.prob_dist_bounds[1], len(chi2))
            self.spline = Spline1D(UnivariateSpline(alpha, -chi2 / 2, s=0, ext=2), ext=2)
            self.bounds = tuple(map(float, like.prob_dist_bounds))
        elif getattr(like, "use_grid_1d", False):
            self.mode, self.obs = "grid1", [like.observable_1]
            self.z = np.asarray([like.redshift])
            self.spline = Spline1D(like.interpolator, ext=2)
        elif getattr(like, "use_grid_2d", False):
            self.mode, self.obs = "grid2", [like.observable_1, like.observable_2]
            self.z = np.asarray([like.redshift])
            self.spline = Spline2D(like.interpolator)
        elif getattr(like, "use_grid_3d", False):
            self.mode, self.obs = "grid3", [like.observable_1, like.observable_2, like.observable_3]
            self.z = np.asarray([like.redshift])
            self.grid = RegularGrid(like.interpolator3D)
        else:
            self.mode = "gauss"
            self.obs = [str(o) for o in like.data["observable"]]
            self.z = np.asarray(like.data["z"], dtype=float)
            self.value = np.asarray(like.data["value"], dtype=float)
            self.invcov = np.asarray(like.invcov, dtype=float)
        req = self.bridge._req["z"]
        self._iDA = {z: int(self.z_index("angular_diameter_distance", z)[0]) for z in self.z} \
            if "angular_diameter_distance" in req else {}
        self._iH = {z: int(self.z_index("Hubble", z)[0]) for z in self.z} if "Hubble" in req else {}

    def _value(self, th, z, obs, pk=None):
        rs = th["params"]["rdrag"] * self.rs_rescale if "params" in th else None
        da = th["angular_diameter_distance"][self._iDA[z]] if z in self._iDA else None
        h = th["Hubble"][self._iH[z]] if z in self._iH else None
        if obs == "DV_over_rs":
            return jnp.cbrt(((1 + z) * da) ** 2 * C_KM_S * z / h) / rs
        if obs == "rs_over_DV":
            return rs / jnp.cbrt(((1 + z) * da) ** 2 * C_KM_S * z / h)
        if obs == "DM_over_rs":
            return (1 + z) * da / rs
        if obs == "DA_over_rs":
            return da / rs
        if obs == "DH_over_rs":
            return C_KM_S / h / rs
        if obs == "Hz_rs":
            return h * rs
        if obs == "F_AP":
            return (1 + z) * da * h / C_KM_S
        if obs == "f_sigma8":
            zs = np.asarray(pk["z"], dtype=float)
            return pk["fsigma8"][int(np.argmin(np.abs(zs - z)))]
        raise NotImplementedError(obs)

    def loglike(self, inp):
        th = inp.theory
        pk = inp.shared.get("pk")
        if self.mode == "gauss":
            t = jnp.stack([self._value(th, z, o, pk) for z, o in zip(self.z, self.obs)])
            r = t - jnp.asarray(self.value)
            return -0.5 * r @ (jnp.asarray(self.invcov) @ r)
        z = float(self.z[0])
        v = [self._value(th, z, o, pk) for o in self.obs]
        if self.mode in ("prob", "grid1"):
            return self.spline(v[0])
        if self.mode == "grid2":
            return self.spline(v[0], v[1])
        return self.grid(v)


class H0(CobayaPort):
    """Cobaya ``base_classes.H0``: Gaussian on H0 (``H0_mean``, ``H0_std``)."""

    cobaya_names = ("H0.riess2020", "H0.riess2018a", "H0.riess2018b", "H0.riess201903", "H0.freedman2020")

    def setup(self, like):
        self.mean, self.std = float(like.H0_mean), float(like.H0_std)

    def loglike(self, inp):
        return -0.5 * ((inp.cosmo.H0 - self.mean) / self.std) ** 2


class Mb(CobayaPort):
    """Cobaya ``base_classes.Mb``: Gaussian on the SN absolute magnitude M_b (a sampled parameter)."""

    cobaya_names = ("H0.riess2020Mb",)

    def setup(self, like):
        self.mean, self.std = float(like.Mb_mean), float(like.Mb_std)

    def loglike(self, inp):
        return -0.5 * ((inp.params["Mb"] - self.mean) / self.std) ** 2


__all__ = ["BAO", "BAO_NAMES", "H0", "Mb"]
