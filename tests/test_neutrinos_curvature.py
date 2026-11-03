"""Extended cosmologies against CAMB 2.0.4: neutrino models (Σm_ν, N_eff, hierarchies, sterile),
the 2-D BBN helium, the background with curvature, and the z*/z_drag neutrino corrections."""

from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

camb = pytest.importorskip("camb")

from gpucosmo.jax.background import cosmology as cb  # noqa: E402
from gpucosmo.jax.eb.massive_nu import NeutrinoModel  # noqa: E402

# (hierarchy, num_massive, sterile, mnu, nnu, meffsterile)
NU_CASES = [
    ("degenerate", 1, False, 0.06, 3.044, 0.0),
    ("degenerate", 1, False, 0.4, 3.5, 0.0),
    ("degenerate", 3, False, 0.3, 2.6, 0.0),
    ("degenerate", 3, False, 0.06, 3.6, 0.0),
    ("normal", 3, False, 0.1, 3.044, 0.0),
    ("normal", 3, False, 0.059, 3.3, 0.0),  # CAMB's one-eigenstate branch below the minimum
    ("inverted", 3, False, 0.15, 2.9, 0.0),
    ("inverted", 3, False, 0.09, 3.044, 0.0),  # one degenerate pair below the minimum
    ("degenerate", 1, True, 0.06, 3.5, 0.3),
    ("normal", 3, True, 0.08, 3.4, 1.0),
]


def _camb(h, nm, mnu, nnu, ms, **kw):
    p = camb.set_params(H0=66.0, ombh2=0.0223, omch2=0.119, mnu=mnu, nnu=nnu, num_massive_neutrinos=nm,
                        neutrino_hierarchy=h, meffsterile=ms, **kw)
    return p, camb.get_background(p)


@pytest.mark.parametrize("h, nm, sterile, mnu, nnu, ms", NU_CASES)
def test_neutrino_model_matches_camb(h, nm, sterile, mnu, nnu, ms):
    p, r = _camb(h, nm, mnu, nnu, ms)
    d = NeutrinoModel(h, nm, sterile=sterile).resolve(mnu, nnu, ms)
    deg = np.asarray(d["deg"])
    keep = deg > 0
    e = p.nu_mass_eigenstates
    assert keep.sum() == e
    np.testing.assert_allclose(float(d["N_ur"]) + deg.sum(), p.num_nu_massless
                               + sum(p.nu_mass_degeneracies[:e]), rtol=1e-12)
    np.testing.assert_allclose(float(d["N_ur"]), p.num_nu_massless, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.sort(np.asarray(d["am"])[keep]), np.sort(np.asarray(r.nu_masses[:e])),
                               rtol=1e-6)
    np.testing.assert_allclose(float(d["omnuh2"]), p.omnuh2, rtol=1e-9)


def test_massless_limit_moves_states_to_massless():
    d = NeutrinoModel("degenerate", 3).resolve(0.0, 3.044)
    assert float(d["N_ur"]) == pytest.approx(3.044) and float(np.sum(d["deg"])) == 0.0
    assert float(d["omnuh2"]) == 0.0


def test_resolve_is_differentiable():
    for model in (NeutrinoModel("normal"), NeutrinoModel("inverted"), NeutrinoModel("degenerate", 3)):
        g = jax.grad(lambda m: model.resolve(m, 3.2)["omnuh2"])(0.2)
        assert np.isfinite(g) and g > 0


def test_bbn_helium_two_dimensional_matches_camb():
    from gpucosmo.jax.eb.bbn_jax import DESI_BBN_TABLE, yhe_from_bbn

    pred = camb.bbn.BBN_table_interpolator(DESI_BBN_TABLE)
    rng = np.random.default_rng(0)
    omb, dn = rng.uniform(0.006, 0.039, 50), rng.uniform(-2.9, 6.9, 50)
    ours = np.asarray(jax.vmap(lambda o, d: yhe_from_bbn(o, delta_neff=d))(jnp.asarray(omb), jnp.asarray(dn)))
    ref = np.array([pred.Y_He(o, d) for o, d in zip(omb, dn)])
    np.testing.assert_allclose(ours, ref, rtol=1e-12)


@pytest.mark.parametrize("h, nm, ext", [
    ("degenerate", 1, dict(mnu=0.06, nnu=3.044, omk=0.0)),
    ("degenerate", 3, dict(mnu=0.3, nnu=2.8, omk=0.05)),
    ("normal", 3, dict(mnu=0.12, nnu=3.044, omk=-0.05)),
    ("inverted", 3, dict(mnu=0.15, nnu=3.3, omk=0.02)),
    ("degenerate", 1, dict(mnu=0.06, nnu=3.044, omk=-0.1)),
])
def test_background_with_curvature_and_neutrinos_matches_camb(h, nm, ext):
    H0, ob, oc, w0, wa = 66.0, 0.0223, 0.119, -0.8, -0.5
    p = camb.set_params(H0=H0, ombh2=ob, omch2=oc, w=w0, wa=wa, dark_energy_model="ppf",
                        neutrino_hierarchy=h, num_massive_neutrinos=nm, **ext)
    r = camb.get_background(p)
    t = cb.load_background_tables(nu_model=NeutrinoModel(h, nm))
    bp = cb.BackgroundParams.build(ob, oc, H0, w0, wa, ext=ext)
    z = np.array([0.1, 0.5, 1.0, 2.33, 5.0])
    np.testing.assert_allclose(np.asarray(cb.hubble_km_s_mpc(t, bp, jnp.asarray(z))), r.hubble_parameter(z),
                               rtol=1e-10)
    np.testing.assert_allclose(np.asarray(cb.transverse_comoving_distance(t, bp, jnp.asarray(z))),
                               r.angular_diameter_distance(z) * (1 + z), rtol=1e-10)
    np.testing.assert_allclose(float(cb.angular_diameter_distance_2(t, bp, 0.5, 2.0)[0]),
                               r.angular_diameter_distance2(0.5, 2.0), rtol=1e-10)
    d = r.get_derived_params()
    np.testing.assert_allclose(float(cb.sound_horizon(t, bp, d["zdrag"])), d["rdrag"], rtol=1e-7)


@pytest.mark.parametrize("h, nm, sterile, axes", [
    ("degenerate", 1, False, ("mnu", "nnu")), ("degenerate", 3, False, ("mnu", "nnu")),
    ("normal", 3, False, ("mnu", "nnu")), ("inverted", 3, False, ("mnu", "nnu")),
    ("degenerate", 1, True, ("nnu", "meffsterile")),
])
def test_recombination_redshift_tables_match_camb(h, nm, sterile, axes):
    model = NeutrinoModel(h, nm, sterile=sterile)
    t = cb.load_background_tables(nu_model=model)
    assert t.nu_corr_axes == axes
    pts = ([dict(mnu=0.2, nnu=3.2), dict(mnu=0.65, nnu=2.7), dict(mnu=0.12, nnu=3.044)] if not sterile else
           [dict(nnu=3.3, meffsterile=0.4), dict(nnu=3.9, meffsterile=1.5)])
    for ext in pts:
        ob, oc = 0.0224, 0.121
        mnu = ext.get("mnu", 0.06)
        zs, zd = [], []
        for H0 in (55.0, 67.36, 80.0):  # median over H0 like the tables (RECFAST jitter)
            p = camb.set_params(H0=H0, ombh2=ob, omch2=oc, mnu=mnu, nnu=ext["nnu"], neutrino_hierarchy=h,
                                num_massive_neutrinos=nm, meffsterile=ext.get("meffsterile", 0.0),
                                bbn_predictor="PArthENoPE_880.2_standard.dat")  # the solver's BBN table
            dd = camb.get_background(p).get_derived_params()
            zs.append(dd["zstar"]), zd.append(dd["zdrag"])
        bp = cb.BackgroundParams.build(ob, oc, 67.0, -1.0, 0.0, ext=ext)
        assert float(cb.z_star_of(t, bp)) == pytest.approx(np.median(zs), rel=3e-6)
        assert float(cb.z_drag_of(t, bp)) == pytest.approx(np.median(zd), rel=5e-5)


def test_default_neutrinos_keep_the_base_tables():
    t = cb.load_background_tables()
    bp = cb.BackgroundParams.build(0.0224, 0.12, 67.0, -1.0, 0.0)
    assert float(cb.z_star_of(t, bp)) == float(cb.z_star(t, 0.0224, 0.12))


def test_transverse_distance_smooth_through_flat():
    t = cb.load_background_tables()

    def dm(omk):
        bp = cb.BackgroundParams.build(0.0224, 0.12, 67.0, -1.0, 0.0, ext={"omk": omk})
        return cb.transverse_comoving_distance(t, bp, jnp.asarray([1.0, 1090.0 / 1000]))[0]

    g = [float(jax.grad(dm)(x)) for x in (-1e-6, 0.0, 1e-6)]
    assert np.all(np.isfinite(g)) and np.ptp(g) < 1e-3 * abs(g[1])


def test_bao_sn_bundle_with_extended_parameters_matches_cobaya_camb():
    """End to end: a yaml sampling Σm_ν (three degenerate states), N_eff, Ω_k and w0wa with DESI DR2
    BAO + DES-Y5 SN, evaluated by the bundle and by Cobaya + CAMB at an off-fiducial point."""
    pytest.importorskip("cobaya")
    from cobaya.model import get_model

    from gpucosmo.jax.inference.cobaya_yaml import resolve_cobaya
    from gpucosmo.jax.inference.mg_bundle import build_mg_bundle
    from gpucosmo.paths import packages_path

    pt = dict(H0=69.0, ombh2=0.0223, omch2=0.118, logA=3.05, ns=0.965, mnu=0.25, nnu=3.3, omk=0.03,
              w=-0.85, wa=-0.6)
    like = {"bao.desi_dr2": None, "sn.desy5": None}
    camb_block = {"camb": {"extra_args": {"num_massive_neutrinos": 3, "dark_energy_model": "ppf",
                                          "bbn_predictor": "PArthENoPE_880.2_standard.dat"}}}
    try:
        m = get_model({"likelihood": like, "theory": camb_block, "packages_path": str(packages_path()),
                       "params": {**{k: v for k, v in pt.items() if k != "logA"},
                                  "As": float(np.exp(pt["logA"]) * 1e-10), "tau": 0.0544}})
    except Exception as e:  # noqa: BLE001 (data not installed)
        pytest.skip(f"Cobaya model: {e}")
    ref = dict(zip(m.likelihood, m.loglikes({}, return_derived=False)))
    info = {"likelihood": like, "theory": camb_block,
            "params": {**{k: {"prior": {"min": v - 0.5 * abs(v) - 0.01, "max": v + 0.5 * abs(v) + 0.01}}
                          for k, v in pt.items()}, "tau": 0.0544}}
    res = resolve_cobaya(info)
    assert set(res.config.ext_sampled) == {"mnu", "nnu", "omk"} and res.config.num_massive_nu == 3
    b = build_mg_bundle(res.config, priors=res.priors, likelihoods=res.extra_likelihoods)
    x = jnp.asarray([pt[n] for n in b.param_names])
    th = b.theory_exact(x[: b.n_theory])
    comps = b.log_like_from_theory_fn(x, {**th, "aux": b.aux}, components=True)
    ours = {k: float(v) for k, v in comps.items()}
    np.testing.assert_allclose(ours["sn"], ref["sn.desy5"], atol=1e-6)
    bao_key = next(k for k in ours if "desi_dr2" in k)
    # r_d from the z_drag tables (~1e-6; DESI DR2's 0.3 % points turn that into ~2e-3 in log L, 0.002σ)
    np.testing.assert_allclose(ours[bao_key], ref["bao.desi_dr2"], atol=5e-3)


@pytest.mark.parametrize("omk", [0.0, 0.05, -0.05])
def test_curved_limber_phi_matches_camb_transfer_limber(omk):
    """``cl_pp_limber_camb(curv=K)`` (a χ sum) against CAMB's curved transfer Limber done literally:
    Δ_ℓ(q) = S(q, τ0 − χ) √(π/2/ν/√(1 − Kℓ²/q²))/q at f_K(χ) = ν/q, C_ℓ = 4π ∫ dq/q (q²/k_s²) P(k_s) Δ²."""
    from gpucosmo.jax.eb.hierarchy_jax import f_k
    from gpucosmo.jax.eb.los_jax import cl_pp_limber_camb

    H0 = 70.0 / 299792.458
    K = -omk * H0**2
    tau0, chi_s = 14000.0, 13700.0
    tau = np.linspace(tau0 - chi_s + 1.0, tau0 - 0.05, 4000)
    k = np.geomspace(1e-4, 5.0, 600)
    chi = tau0 - tau
    src = np.exp(-((np.log(k)[:, None] + 3.0) ** 2) / 4.0) * (chi / chi_s * (1 - chi / chi_s))[None, :] / (1 + (k[:, None] * 30) ** 2)
    As, ns = 2.1e-9, 0.965
    p_r = As * (k / 0.05) ** (ns - 1)
    ells = np.array([100.0, 400.0, 1500.0])
    ours = np.asarray(cl_pp_limber_camb(jnp.asarray(src), jnp.asarray(tau), tau0, chi_s, jnp.asarray(k),
                                        jnp.asarray(p_r), jnp.asarray(ells), None if omk == 0 else K))
    chi_f = np.linspace(0.2, chi_s - 1e-3, 200000)
    fk = np.asarray(f_k(jnp.asarray(chi_f), K))
    from scipy.interpolate import RectBivariateSpline

    S = RectBivariateSpline(k, tau, src, kx=3, ky=1)
    for l, c in zip(ells, ours):
        nu = l + 0.5
        q = np.geomspace(max(nu / fk.max(), k[0]), k[-1], 20000)
        chi_x = np.interp(nu / q, fk, chi_f)  # f_K⁻¹ (near branch; f_K increasing here)
        sq = 1 - K * l**2 / q**2
        delta = S.ev(q, tau0 - chi_x) * np.sqrt(np.pi / 2 / nu / np.sqrt(sq)) / q
        ks = np.sqrt(q**2 - K)
        integrand = (q**2 / ks**2) * As * (ks / 0.05) ** (ns - 1) * delta**2
        ref = 4 * np.pi * np.trapezoid(integrand, np.log(q))
        assert c == pytest.approx(ref, rel=1e-5), (l, c / ref - 1)  # the flat k = ν/χ is off by ~3e-3


@pytest.mark.parametrize("R", [9000.0, 35500.0])
def test_closed_q_grid_sums_over_integer_nu(R):
    """The closed-model integration grid (``curved_int_grid`` + ``closed_dq_fix`` on CAMB's dq) against
    the exact sum over ν = 3, 4, ... of an integrand oscillating in ν like the transfer functions
    (period 2πR/χ*); rounding every node to an integer (the first version) was off by ~3 %."""
    from gpucosmo.jax.eb.cls_jax import _int_q_grid, closed_dq_fix, curved_int_grid
    from gpucosmo.jax.eb.los_jax import camb_dpoints

    K = 1.0 / R**2
    q = curved_int_grid(_int_q_grid(k_evolve=jnp.geomspace(1e-5, 0.45, 600), tau0=14000.0, chi_star=13720.0,
                                    k_max=0.35), K)
    w = np.asarray(camb_dpoints(q, half_ends=False) * closed_dq_fix(q, K))
    nu_q = np.asarray(q) * R
    period = 2 * np.pi * R / 13720.0
    for f in (lambda nu: (1 + 0.8 * np.cos(2 * np.pi * nu / period + 0.3)) * np.exp(-nu / 3000.0) / nu,
              lambda nu: np.exp(-(((nu - 8) / 4.0) ** 2))):
        exact = f(np.arange(3, int(0.35 * R) + 1)).sum() / R
        assert (w * f(nu_q)).sum() == pytest.approx(exact, rel=2e-4)


def _bao_sn_info(params=None, extra_args=None):
    base = dict(H0=(60.0, 80.0), ombh2=(0.02, 0.025), omch2=(0.1, 0.14), logA=(2.9, 3.2), ns=(0.9, 1.0))
    return {"likelihood": {"bao.desi_dr2": None, "sn.desy5": None},
            "theory": {"camb": {"extra_args": dict(extra_args or {})}},
            "params": {**{k: {"prior": {"min": a, "max": b}} for k, (a, b) in base.items()}, "tau": 0.0544,
                       **(params or {})}}


def _build(info, likelihoods=()):
    from _cobaya_data import require_cobaya_data

    from gpucosmo.jax.inference.cobaya_yaml import resolve_cobaya

    require_cobaya_data(*info["likelihood"])
    from gpucosmo.jax.inference.mg_bundle import build_mg_bundle

    res = resolve_cobaya(info)
    try:
        return build_mg_bundle(res.config, priors=res.priors, likelihoods=(*res.extra_likelihoods, *likelihoods))
    except FileNotFoundError as e:  # data not installed
        pytest.skip(str(e))


@pytest.mark.parametrize("params, extra_args, match", [
    ({"mnu": {"prior": {"min": 0.0, "max": 5.0}}}, {}, "exceeds"),  # beyond the z* tables (Σm_ν ≤ 1.5)
    ({"mnu": {"prior": {"min": 0.0, "max": 1.0}}}, {"neutrino_hierarchy": "normal", "num_massive_neutrinos": 3},
     "exceeds"),  # CAMB's one-eigenstate branch below 0.0601 eV
    ({}, {"num_massive_neutrinos": 0}, "non-zero neutrino mass"),  # CAMB raises too
    ({"meffsterile": {"prior": {"min": 0.0, "max": 1.0}}, "nnu": {"value": 3.0}}, {}, "outside the range"),
])
def test_extended_parameter_refusals(params, extra_args, match):
    pytest.importorskip("cobaya")
    with pytest.raises(ValueError, match=match):
        _build(_bao_sn_info(params, extra_args))


def test_zero_massive_neutrinos_with_zero_mass_builds():
    pytest.importorskip("cobaya")
    b = _build(_bao_sn_info({"mnu": {"value": 0.0}}, {"num_massive_neutrinos": 0}))
    assert "mnu" not in b.param_names


def test_plugin_holding_an_extended_parameter_fixed_is_refused():
    pytest.importorskip("cobaya")
    from gpucosmo.likelihoods.base import Likelihood

    class FlatOnly(Likelihood):
        name = "flat_only"
        fixed_ext = ("omk",)

        def theory(self, cosmo):
            return {"x": cosmo.H(0.5)}

        def loglike(self, inp):
            return 0.0 * inp.theory["x"]

    with pytest.raises(ValueError, match="holds"):
        _build(_bao_sn_info({"omk": {"prior": {"min": -0.05, "max": 0.05}}}), likelihoods=(FlatOnly(),))


def test_momentum_rule_follows_camb():
    """CAMB 2.0.4's q-rule promotion (results.f90): 4 bins above nu_masses = 600, 5 above 4300."""
    import dataclasses

    from gpucosmo.jax.eb.cls_jax import CLS_HIERARCHY
    from gpucosmo.jax.eb.hierarchy_jax import hierarchy_for_model, nu_nqmax
    from gpucosmo.jax.eb.massive_nu import NU_Q4_AM, NU_Q5_AM, camb_nu_nqmax

    assert [camb_nu_nqmax(am) for am in (NU_Q4_AM * 0.999, NU_Q4_AM * 1.001, NU_Q5_AM * 0.999,
                                         NU_Q5_AM * 1.001)] == [3, 4, 4, 5]
    one, deg3 = NeutrinoModel(), NeutrinoModel("degenerate", 3)
    assert nu_nqmax(one, {}) == nu_nqmax(deg3, {"mnu": 0.06}) == 3  # default masses
    assert nu_nqmax(one, {"mnu": jnp.asarray(0.06)}) == 5  # traced: any mass
    assert nu_nqmax(NeutrinoModel(sterile=True), {}) == 5
    pinned = dataclasses.replace(one, nqmax=4)
    assert pinned.base == one and hierarchy_for_model(CLS_HIERARCHY, pinned, {"mnu": 1.0}).nqmax == 4
    assert hierarchy_for_model(dataclasses.replace(CLS_HIERARCHY, nqmax=10), pinned, {}).nqmax == 10


@pytest.mark.parametrize("params, extra_args, nq", [
    ({}, {}, 3),
    ({"mnu": {"prior": {"min": 0.0, "max": 0.09}}}, {}, 3),
    ({"mnu": {"prior": {"min": 0.0, "max": 0.5}}}, {}, 4),
    ({"mnu": {"prior": {"min": 0.0, "max": 1.0}}}, {}, 5),
    ({"mnu": {"value": 0.3}}, {}, 4),
    ({"mnu": {"prior": {"min": 0.0, "max": 0.9}}}, {"num_massive_neutrinos": 3}, 4),
    ({"mnu": {"prior": {"min": 0.0, "max": 0.3}}}, {"num_massive_neutrinos": 3}, 3),
    ({"mnu": {"prior": {"min": 0.0, "max": 0.3}}, "nnu": {"prior": {"min": 2.0, "max": 4.0}}},
     {"num_massive_neutrinos": 3}, 4),  # N_eff < 3.044 cools the states (m/T 594 → 660)
    ({"mnu": {"prior": {"min": 0.07, "max": 0.3}}}, {"neutrino_hierarchy": "normal", "num_massive_neutrinos": 3}, 4),
    ({"meffsterile": {"prior": {"min": 0.0, "max": 1.0}}, "nnu": {"value": 3.4}}, {}, 5),
])
def test_bundle_momentum_rule_from_the_mass_range(params, extra_args, nq):
    """The analysis' heaviest eigenstate (box edge or fixed value) sets the quadrature, statically
    for the staged and exact theories (``tables.model``)."""
    pytest.importorskip("cobaya")
    b = _build(_bao_sn_info(params, extra_args))
    assert b.tables.model.nqmax == nq


def test_high_k_weyl_shape_carries_neutrino_free_streaming():
    """The Limber φφ source past the evolved k (``extend_weyl_high_k``) follows CAMB's transfer
    shape with a heavy state: EH98 alone misses the free-streaming suppression still building
    above k = 0.45 (2.6 % at k = 1/Mpc for 1 eV); with EH99's growth it is ≤ 6e-3."""
    camb = pytest.importorskip("camb")
    from gpucosmo.jax.eb.cls_jax import _params_dict_cls, ext_params
    from gpucosmo.jax.eb.nonlinear_sources_jax import eh98_nowiggle_transfer, eh99_nu_growth

    h, ombh2, omch2, ns, mnu = 0.62, 0.02237, 0.12, 0.965, 1.0
    cp = camb.set_params(H0=100 * h, ombh2=ombh2, omch2=omch2, mnu=mnu, ns=ns, As=2.1e-9)
    cp.set_matter_power(redshifts=[0.0], kmax=3.0)
    pk = camb.get_results(cp).get_matter_power_interpolator(nonlinear=False, hubble_units=False, k_hunit=False)
    k = np.array([0.45, 0.7, 1.0])
    t_camb = np.sqrt(pk.P(0.0, k) / pk.P(0.0, k[0])) * (k[0] / k) ** (ns / 2)
    p = _params_dict_cls(ombh2, omch2, h, ns, 2.1e-9, 0.06, -1.0, 0.0, **ext_params({"mnu": jnp.asarray(mnu)}))
    t_eh = np.asarray(eh98_nowiggle_transfer(jnp.asarray(k), p) / eh98_nowiggle_transfer(jnp.asarray(k[0]), p))
    g = np.asarray(eh99_nu_growth(jnp.asarray(k), 1.0, p) / eh99_nu_growth(jnp.asarray(k[0]), 1.0, p))
    err98, err99 = np.abs(t_camb / t_eh - 1), np.abs(t_camb / (t_eh * g) - 1)
    assert err98[-1] > 2e-2  # EH98 alone: 1.6 % at k = 0.7, 2.6 % at 1
    assert np.all(err99 <= 7e-3) and np.all(err99[1:] < err98[1:] / 4)  # 2.6e-3, 6.0e-3
