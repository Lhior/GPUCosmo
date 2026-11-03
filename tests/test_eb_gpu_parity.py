#!/usr/bin/env python3
"""Regression tests for the GPU / CAMB-parity pieces of the JAX EB pipeline."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

_COSMO = (0.02237, 0.1193, 0.676463, 0.965, 2.1e-9, 0.055, -0.836, -0.701)


def test_taurend_identical_eager_and_jit():
    """``sync_thermo_taurend_camb_like`` must not change under tracing.

    The old traced fallback kept the reionization bump in the cumulative
    visibility and returned taurend ≈ 1e4 instead of ≈ 600 under jit, which
    silently moved the photon RSA switch in every batched run.
    """
    from gpucosmo.jax.eb.cls_jax import _recfast_thermo_device

    eager = _recfast_thermo_device(*_COSMO)
    traced = jax.jit(lambda *p: _recfast_thermo_device(*p))(*_COSMO)
    for key in ("taurend", "taurst_camb_like", "tau_maxvis_camb_like"):
        np.testing.assert_allclose(float(eager[key]), float(traced[key]), rtol=1e-10,
                                   err_msg=key)
    assert 450.0 < float(eager["taurend"]) < 800.0


def test_lensing_jax_matches_numpy_reference_and_differentiates():
    from gpucosmo.jax.cmb import lensing_jax as L

    lmax = 1200
    ell = np.arange(lmax + 1, dtype=float)
    shape = np.exp(-((ell / 1500.0) ** 2))
    tt = 5000.0 * shape * (1 + 0.3 * np.sin(ell / 45.0))
    ee = 30.0 * shape * (1 + 0.5 * np.cos(ell / 45.0)) * (ell / 1000.0)
    te = 100.0 * shape * np.sin(ell / 60.0)
    pp = 1e-7 * (ell + 1.0) ** -0.5
    for a in (tt, ee, te, pp):
        a[:2] = 0.0
    kw = dict(lmax=lmax, lmax_lensed=lmax, use_lensing_template=False)
    got = np.stack([np.asarray(x) for x in L.lens_cmb_cls(tt, ee, te, pp, **kw)], axis=1)
    cls = np.stack([tt, ee, np.zeros_like(tt), te], axis=1)
    ref = cls + L._lens_delta_numpy(cls, pp, lmax=lmax)
    np.testing.assert_allclose(got[2:], ref[2:], rtol=1e-10, atol=1e-12 * np.abs(ref).max())

    f = jax.jit(lambda a: jnp.sum(L.lens_cmb_cls(a, ee, te, pp, **kw)[0][2:1000]))
    g = jax.grad(f)(jnp.asarray(tt))
    assert np.all(np.isfinite(np.asarray(g)))


def test_hmcode2020_limits_jit_vmap_and_grad():
    from gpucosmo.jax.eb.hmcode2020_jax import hmcode2020_ratio, linear_table_k_h
    from gpucosmo.jax.eb.nonlinear_sources_jax import eh98_nowiggle_transfer

    kt = linear_table_k_h()
    h = 0.6765
    params = dict(h=h, Omega_b=0.0489, Omega_c=0.2607, Omega_nu_massive_today=0.0014)
    t = np.asarray(eh98_nowiggle_transfer(jnp.asarray(kt * h), params))
    a = jnp.asarray([0.5, 1.0])
    base = np.log((kt * h / 0.05) ** -0.035 * (kt * h) ** 4 * t**2)
    base = base - np.interp(0.0, np.log(kt), base) + np.log(3.0)  # Δ²_lin(1 h/Mpc) ≈ 3
    tab = jnp.asarray(np.stack([base + 2 * np.log(0.61), base]))

    def cosmo(w0, wa):
        return dict(Omega_m=0.311, Omega_b=0.0489, Omega_c=0.2607, Omega_nu=0.0014,
                    Omega_v=0.689, h=h, n_s=0.965, w0=w0, wa=wa, T_cmb=2.7255, N_nu=1.0)

    kq = jnp.asarray([1e-3, 0.1, 1.0, 5.0])
    batched = jax.jit(jax.vmap(lambda w0, wa: hmcode2020_ratio(kq, a, tab, cosmo(w0, wa))))
    r = np.asarray(batched(jnp.asarray([-1.0, -0.836]), jnp.asarray([0.0, -0.701])))
    assert r.shape == (2, 2, 4) and np.all(np.isfinite(r))
    assert np.all(np.abs(r[..., 0] - 1.0) < 2e-3)  # linear at large scales
    assert np.all(r[..., 2] > 1.05)  # nonlinear boost at k = 1 h/Mpc
    g = jax.grad(lambda w: jnp.sum(jnp.log(hmcode2020_ratio(kq, a, tab, cosmo(w, 0.0)))))(-1.0)
    assert np.isfinite(float(g))


@pytest.mark.parametrize("version", ["mead2020", "mead2016"])
def test_hmcode_matches_camb(version):
    """sqrt(P_nl/P_lin) of the JAX HMcode vs CAMB's (``halofit_version``) on CAMB's own linear
    spectrum (scripts/validate_hmcode2020_vs_camb.py: ≤ 7e-5 for k ≤ 10 h/Mpc, z ≤ 4)."""
    camb = pytest.importorskip("camb")
    from gpucosmo.jax.eb.hmcode2020_jax import hmcode2020_ratio, linear_table_k_h

    zs = [0.0, 1.0]
    pars = camb.set_params(H0=67.65, ombh2=0.02237, omch2=0.1193, ns=0.965, As=2.1e-9, mnu=0.06,
                           num_massive_neutrinos=1, halofit_version=version,
                           NonLinear=camb.model.NonLinear_both)
    pars.set_matter_power(redshifts=zs, kmax=20.0, nonlinear=True)
    res = camb.get_results(pars)
    lin = res.get_matter_power_interpolator(nonlinear=False, hubble_units=True, k_hunit=True,
                                            log_interp=True, extrap_kmax=1e3)
    kh, z_c, pk_nl = res.get_nonlinear_matter_power_spectrum(hubble_units=True, k_hunit=True)
    _, _, pk_l = res.get_linear_matter_power_spectrum(hubble_units=True, k_hunit=True)
    h = 0.6765
    cosmo = dict(Omega_m=(0.1193 + 0.02237 + pars.omnuh2) / h**2, Omega_b=0.02237 / h**2,
                 Omega_c=0.1193 / h**2, Omega_nu=pars.omnuh2 / h**2, Omega_v=res.omega_de, h=h,
                 n_s=0.965, w0=-1.0, wa=0.0, T_cmb=pars.TCMB, N_nu=1.0)
    kt = linear_table_k_h()
    order = np.argsort(z_c)
    z_sorted = np.asarray(z_c)[order]
    tab = np.stack([np.log(lin.P(z, kt) * kt**3 / (2 * np.pi**2)) for z in z_sorted])
    sel = (kh >= 1e-3) & (kh <= 5.0)
    r = np.asarray(hmcode2020_ratio(kh[sel], 1.0 / (1.0 + z_sorted), tab, cosmo, version=version))
    ref = np.sqrt(pk_nl[order][:, sel] / pk_l[order][:, sel])
    assert np.max(np.abs(r / ref - 1.0)) < 2e-4


@pytest.mark.parametrize("mnu,n_massive,logt_agn", [(0.06, 1, 7.7), (0.3, 3, 7.3), (0.6, 3, 8.0)])
def test_hmcode_feedback_matches_camb(mnu, n_massive, logt_agn):
    """HMcode-2020 feedback (CAMB ``mead2020_feedback``, ``HMCode_logT_AGN``) vs CAMB for k ≤ 30
    h/Mpc, z ≤ 2 on CAMB's own linear spectrum (degenerate massive ν for n_massive = 3)."""
    camb = pytest.importorskip("camb")
    from gpucosmo.jax.eb.hmcode2020_jax import hmcode2020_ratio, linear_table_k_h

    zs = [0.0, 0.5, 1.0, 2.0]
    h, ombh2, omch2, ns = 0.6736, 0.02237, 0.12, 0.9649
    pars = camb.set_params(H0=100 * h, ombh2=ombh2, omch2=omch2, ns=ns, As=2.1e-9, mnu=mnu,
                           num_massive_neutrinos=n_massive, halofit_version="mead2020_feedback",
                           HMCode_logT_AGN=logt_agn, NonLinear=camb.model.NonLinear_pk)
    pars.set_matter_power(redshifts=zs, kmax=150.0, nonlinear=True)
    res = camb.get_results(pars)
    lin = res.get_matter_power_interpolator(nonlinear=False, hubble_units=True, k_hunit=True,
                                            log_interp=True, extrap_kmax=1e3)
    kh, z_c, pk_nl = res.get_nonlinear_matter_power_spectrum(hubble_units=True, k_hunit=True)
    _, _, pk_l = res.get_linear_matter_power_spectrum(hubble_units=True, k_hunit=True)
    cosmo = dict(Omega_m=(omch2 + ombh2 + pars.omnuh2) / h**2, Omega_b=ombh2 / h**2,
                 Omega_c=omch2 / h**2, Omega_nu=pars.omnuh2 / h**2, Omega_v=res.omega_de, h=h,
                 n_s=ns, w0=-1.0, wa=0.0, T_cmb=pars.TCMB, N_nu=float(pars.num_nu_massive))
    kt = linear_table_k_h()
    order = np.argsort(z_c)
    z_sorted = np.asarray(z_c)[order]
    tab = np.stack([np.log(lin.P(z, kt) * kt**3 / (2 * np.pi**2)) for z in z_sorted])
    sel = (kh >= 1e-3) & (kh <= 30.0)
    r = np.asarray(hmcode2020_ratio(kh[sel], 1.0 / (1.0 + z_sorted), tab, cosmo,
                                    version="mead2020_feedback", logT_AGN=logt_agn))
    ref = np.sqrt(pk_nl[order][:, sel] / pk_l[order][:, sel])
    assert np.max(np.abs(r / ref - 1.0)) < 2e-4  # measured ≤ 6e-5 (sqrt ratio)


def test_hmcode_feedback_jit_vmap_grad_logt_agn():
    """logT_AGN may be traced: jit + vmap over it, grad vs finite differences, the response
    → 1 at large scales and suppresses P(k = 5 h/Mpc) more for hotter AGN."""
    from gpucosmo.jax.eb.hmcode2020_jax import hmcode2020_ratio, linear_table_k_h
    from gpucosmo.jax.eb.nonlinear_sources_jax import eh98_nowiggle_transfer

    kt = linear_table_k_h()
    h = 0.6765
    params = dict(h=h, Omega_b=0.0489, Omega_c=0.2607, Omega_nu_massive_today=0.0014)
    t = np.asarray(eh98_nowiggle_transfer(jnp.asarray(kt * h), params))
    a = jnp.asarray([0.5, 1.0])
    base = np.log((kt * h / 0.05) ** -0.035 * (kt * h) ** 4 * t**2)
    base = base - np.interp(0.0, np.log(kt), base) + np.log(3.0)
    tab = jnp.asarray(np.stack([base + 2 * np.log(0.61), base]))
    cosmo = dict(Omega_m=0.311, Omega_b=0.0489, Omega_c=0.2607, Omega_nu=0.0014, Omega_v=0.689,
                 h=h, n_s=0.965, w0=-1.0, wa=0.0, T_cmb=2.7255, N_nu=1.0)
    kq = jnp.asarray([1e-3, 0.1, 1.0, 5.0])

    def fb(lt):
        return hmcode2020_ratio(kq, a, tab, cosmo, version="mead2020_feedback", logT_AGN=lt)

    dmo = np.asarray(hmcode2020_ratio(kq, a, tab, cosmo))
    resp = np.asarray(jax.jit(jax.vmap(fb))(jnp.asarray([7.3, 7.7, 8.0]))) / dmo[None]
    assert resp.shape == (3, 2, 4) and np.all(np.isfinite(resp))
    assert np.all(np.abs(resp[..., 0] - 1.0) < 1e-4)  # no response at k = 1e-3 h/Mpc
    assert np.all(np.diff(resp[:, :, 3], axis=0) < 0)  # hotter AGN → more suppression
    obj = lambda lt: jnp.sum(jnp.log(fb(lt)))  # noqa: E731
    g = float(jax.jit(jax.grad(obj))(7.7))
    eps = 1e-3
    fd = (float(obj(7.7 + eps)) - float(obj(7.7 - eps))) / (2 * eps)
    assert np.isfinite(g) and g < 0 and abs(g - fd) < 1e-4 * abs(fd)


def test_recfast_high_z_split_is_exact():
    from gpucosmo.jax.eb.cls_jax import _params_dict_cls
    from gpucosmo.jax.eb.recfast_jax import run_recfast_jax
    from gpucosmo.jax.eb.thermo_jax import build_background_jax

    bg = build_background_jax(_params_dict_cls(0.02237, 0.1193, 0.676, 0.965, 2.1e-9,
                                               0.055, -0.836, -0.701))
    keys = ("Omega_r", "Omega_m", "Omega_de", "ombh2", "h", "T_CMB", "YHe", "f_He",
            "H0_mpc", "tau_reio", "Omega_b", "Omega_c", "Omega_g", "Omega_nu",
            "Omega_nu_mass", "nu_mass_am", "w0", "wa")
    pf = {k: bg[k] for k in keys}
    _, xe_fast, tm_fast = run_recfast_jax(pf)
    _, xe_ref, tm_ref = run_recfast_jax(pf, split_high_z=False)
    np.testing.assert_allclose(np.asarray(xe_fast), np.asarray(xe_ref), rtol=1e-10, atol=0)
    np.testing.assert_allclose(np.asarray(tm_fast), np.asarray(tm_ref), rtol=1e-10, atol=0)


def test_interp_cubic_reproduces_nodes_quadratics_and_is_c1():
    from gpucosmo.jax.eb.interp_jax import interp_cubic

    x = jnp.linspace(0.0, 2.0, 21)
    y = 1.0 - 0.5 * x + 0.25 * x**2
    xq = jnp.linspace(-0.1, 2.1, 401)
    got = np.asarray(interp_cubic(xq, x, y))
    xc = np.clip(np.asarray(xq), 0.0, 2.0)
    np.testing.assert_allclose(got, 1.0 - 0.5 * xc + 0.25 * xc**2, atol=1e-12)
    np.testing.assert_allclose(np.asarray(interp_cubic(x, x, jnp.sin(x))),
                               np.sin(np.asarray(x)), atol=1e-14)
    # C¹: the derivative is continuous across a node.
    d = jax.vmap(jax.grad(lambda t: interp_cubic(t, x, jnp.sin(x))))
    left, right = np.asarray(d(jnp.asarray([1.0 - 1e-9, 1.0 + 1e-9])))
    assert abs(left - right) < 1e-6


def test_continuous_crossing_and_peak_helpers():
    from gpucosmo.jax.eb.kgrid_jax import first_crossing, refined_peak

    x = jnp.linspace(0.0, 1.0, 11)
    for level in (0.23, 0.2301, 0.2999, 0.3):
        assert abs(float(first_crossing(x, x, level)) - level) < 1e-12
    lx = jnp.linspace(0.0, 1.0, 101)
    for c in (0.4, 0.4031, 0.4049, 0.41):
        y = -((lx - c) ** 2)
        assert abs(float(jnp.log(refined_peak(lx, y))) - c) < 1e-12


@pytest.mark.slow
def test_fwd_jacobian_grad_matches_forward_mode():
    """jax.grad through theory_spectra_fwd_jacobian == direct jvp (ℓmax 300)."""
    from gpucosmo.jax.background.cosmology import load_background_tables
    from gpucosmo.jax.cmb.jax_eb_theory import (
        THEORY_PARAM_NAMES,
        get_theory_spectra_jax_eb,
        theory_spectra_fwd_jacobian,
    )
    from gpucosmo.jax.eb.cls_jax import CLS_BATCH_K_CHUNK

    tables = load_background_tables()
    lmax = 300
    # Unchunked k on the CPU (float64: equal to 1e-10). On an 8 GB GPU the unchunked 8-tangent
    # program needs > 6 GB: chunk k there; the sources are float32 (agreement to float32 accuracy)
    cpu = jax.default_backend() == "cpu"
    k_chunk, rtol = (0, 1e-10) if cpu else (CLS_BATCH_K_CHUNK, 1e-5)
    x0 = jnp.asarray([3.0304, 0.9673, 1.04064, 0.022201, 0.11987, 0.05205, -0.8363, -0.7014])
    w = jnp.linspace(0.5, 1.5, lmax + 1)

    def summary(t):
        return jnp.sum(w * t.dl_tt[: lmax + 1]) + jnp.sum(w * t.dl_ee[: lmax + 1])

    g = jax.grad(lambda x: summary(theory_spectra_fwd_jacobian(tables, x, lmax=lmax, k_chunk=k_chunk)))(x0)

    def plain(x):
        return summary(get_theory_spectra_jax_eb(
            tables, **dict(zip(THEORY_PARAM_NAMES, (x[i] for i in range(8)))), lmax=lmax,
            k_chunk=k_chunk))

    ref = jnp.stack([jax.jvp(plain, (x0,), (jnp.eye(8)[i],))[1] for i in range(8)])
    np.testing.assert_allclose(np.asarray(g), np.asarray(ref), rtol=rtol)


def test_plik_2018_nuisance_priors_match_cobaya():
    from gpucosmo.jax.cmb.params import (
        PLIK_2018_PRIORS,
        SZ_PRIOR,
        build_desi_full_params,
        nuisance_gaussian_priors,
    )

    names = ("A_planck", "calib_217T", "gal545_A_217", "ps_A_100_100", "xi_sz_cib", "A_sz")
    _, lo, hi, fid, _ = build_desi_full_params(names)
    mu, sd = nuisance_gaussian_priors(names)
    assert (mu[0], sd[0]) == (1.0, 0.0025)
    assert (mu[1], sd[1]) == (0.99805, 0.00065)
    assert (mu[2], sd[2]) == (91.9, 20.0)
    assert np.isinf(sd[3]) and (lo[8 + 3], hi[8 + 3]) == (0.0, 400.0)
    assert (lo[8 + 4], hi[8 + 4]) == (0.0, 1.0)
    assert (lo[8 + 5], hi[8 + 5]) == (0.0, 10.0)
    assert SZ_PRIOR == (1.6, 9.5, 3.0)
    assert len(PLIK_2018_PRIORS) == 21


def test_cumulative_sn_distances_match_per_sn_quadrature():
    import jax
    import jax.numpy as jnp
    import numpy as np

    from gpucosmo.jax.background.cosmology import (
        BackgroundParams, comoving_distance, comoving_distance_cumulative,
        cumulative_distance_nodes, load_background_tables)

    jax.config.update("jax_enable_x64", True)
    tables = load_background_tables()
    z = np.random.default_rng(0).uniform(0.001, 2.3, 1600)
    z[5] = z[7]  # duplicated redshifts share an interval end
    nodes = cumulative_distance_nodes(z)

    def both(v):
        p = BackgroundParams.build(ombh2=v[0], omch2=v[1], H0=v[2], w0=v[3], wa=v[4])
        return comoving_distance(tables, p, jnp.asarray(z)), comoving_distance_cumulative(tables, p, nodes)

    v = jnp.asarray([0.0224, 0.119, 66.0, -0.84, -0.6])
    a, b = both(v)
    np.testing.assert_allclose(b, a, rtol=1e-11)
    ja = jax.jacfwd(lambda v: both(v)[0])(v)
    jb = jax.jacfwd(lambda v: both(v)[1])(v)
    np.testing.assert_allclose(jb, ja, rtol=1e-9, atol=1e-9 * float(jnp.abs(ja).max()))


def test_smooth_simall_matches_stepwise_and_has_gradient():
    import jax
    import jax.numpy as jnp
    import numpy as np
    import pytest

    from gpucosmo.jax.cmb.clipy_likelihood import build_clipy_lowl_loglike, load_clipy_lowl

    jax.config.update("jax_enable_x64", True)
    lowl = load_clipy_lowl()
    if not lowl.active or lowl.ee_like is None:
        pytest.skip("Planck low-ell clipy data not installed")
    ell = np.arange(2501.0)
    # Reionization-bump-like EE and a smooth TT, in D_ell (muK^2).
    dl_ee = jnp.asarray(0.04 * np.exp(-((ell - 4.0) / 3.0) ** 2) + 0.01 + 1e-5 * ell)
    dl_tt = jnp.asarray(1000.0 + 0.0 * ell)
    zero = jnp.zeros_like(dl_ee)
    step = build_clipy_lowl_loglike(lowl)
    smooth = build_clipy_lowl_loglike(lowl, smooth_ee=True)
    for s in (0.9, 1.0, 1.1):
        a = float(step(dl_tt, zero, s * dl_ee, 1.0))
        b = float(smooth(dl_tt, zero, s * dl_ee, 1.0))
        assert abs(a - b) < 0.05, (s, a, b)
    g_step = jax.grad(lambda s: step(dl_tt, zero, s * dl_ee, 1.0))(1.0)
    g_smooth = jax.grad(lambda s: smooth(dl_tt, zero, s * dl_ee, 1.0))(1.0)
    assert float(g_step) == 0.0  # clipy's floor() lookup: why the smooth twin exists
    fd = (float(smooth(dl_tt, zero, 1.001 * dl_ee, 1.0))
          - float(smooth(dl_tt, zero, 0.999 * dl_ee, 1.0))) / 0.002
    np.testing.assert_allclose(float(g_smooth), fd, rtol=1e-4)
