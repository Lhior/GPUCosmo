"""The exact theory block of ``mg_bundle`` as a chain of separately compiled stages.

``theory_stages(bundle)`` returns ``[(name, f), ...]`` with ``f(state) -> state`` on pytrees;
the composition maps the theory parameters x_th to the same tree as ``bundle.theory_exact``
({"cmb": {...}, "fs": {...}}). Used by ``staged_fwd`` to differentiate the theory in forward
mode without compiling it as one program (whose forward-mode compile exceeded the host
memory). CMB: prep (background, Recfast, θ → H0) → sources (Boltzmann hierarchy) → C_ℓ (line of
sight, φφ) → lensing; DESI FS: linear P(k) → REPT tables.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np


def theory_stages(b, *, layout: str = "flat", fs_split: bool = True, compute_dtype=None):
    """``compute_dtype``: precision of the per-mode evolutions (CMB sources, P(k) bands); None
    = production (float32 on GPUs). Second-order forward mode needs float64: float32 tangent
    products overflow (every second derivative of the P(k) transfers came out NaN)."""
    cfg = b.config
    names = list(b.param_names)
    n_th = int(b.n_theory)
    fid_rest = np.asarray(b.fiducial)[n_th:]
    stages = []

    def full_x(xt):
        return jnp.concatenate([xt, jnp.asarray(fid_rest)])[None]

    from gpucosmo.jax.inference.mg_bundle import cmb_on

    if cfg.tensors:
        raise NotImplementedError("staged forward-mode DALI builds do not cover tensor modes; use the "
                                  "finite-difference build (--dali-method fd)")
    if cfg.pk_z:
        raise NotImplementedError("staged forward-mode DALI builds do not cover the user likelihoods' P(k) "
                                  "block (pk_z); use the finite-difference build (--dali-method fd)")
    if cmb_on(cfg):
        from gpucosmo.jax.cmb.jax_eb_theory import (
            LENS_MARGIN,
            _check_lmax_extra,
            _nl_lens,
            _spectra_from_cls,
            solve_lmax_extra,
            uses_highl,
        )
        from gpucosmo.jax.cmb.theory import solve_h0_from_theta_mc
        from gpucosmo.jax.eb.cls_jax import (
            CLS_HIERARCHY,
            CLS_K_MAX,
            _batch_k_grids,
            cmb_params,
            compute_cmb_cls,
            hierarchy_for_model,
            highl_plan,
            recfast_thermo_batched,
            source_grids_batched,
        )

        i_logA, i_th = names.index("logA"), names.index("theta_MC_100")
        nl_lens = _nl_lens(cfg.cmb_nonlinear_model if cfg.cmb_nonlinear_lens else False)
        lmax = 2508
        lmax_extra = int(b.cmb_lmax_extra)
        l_cls = lmax + LENS_MARGIN
        # the bundle's solve: the accurate high-ℓ path unless MGConfig.cmb_legacy_lensing (as get_theory_spectra_*)
        l_solve = solve_lmax_extra(lmax, lmax_extra, accurate=not cfg.cmb_legacy_lensing)
        highl = uses_highl(lmax, l_solve)
        if highl:
            _check_lmax_extra(l_solve)
            plan = highl_plan(l_cls, l_solve, 1.0)
            k_ev, k_int = plan["k_ev"], plan["k_int"]
            grid_kw = dict(n_base=plan["n_base"], hi_mult=plan["hi_mult"])
            cls_extra = dict(lmax_highl=l_solve)
        else:
            k_ev, k_int = _batch_k_grids(None, float(CLS_K_MAX), 1.0, l_cls)
            grid_kw, cls_extra = {}, {}
        mg = bool(cfg.mg)
        tables = b.tables
        model = tables.model
        # static: any given Σm_ν may be light (``nu_may_be_light``), as in the batched exact theory
        ext_static = {**dict(cfg.ext_fixed), **{n: object() for n in cfg.ext_sampled}}
        hcfg = hierarchy_for_model(dataclasses.replace(CLS_HIERARCHY, mg=True) if mg else CLS_HIERARCHY, model,
                                   ext_static)
        if "omk" in cfg.ext_sampled or dict(cfg.ext_fixed).get("omk", 0.0) != 0.0:
            hcfg = dataclasses.replace(hcfg, curved=True)

        def prep(xt):
            X = full_x(xt)
            c = jax.vmap(b.cosmo)(X)
            ob, oc, w0, wa, ext = c["ombh2"], c["omch2"], c["w0"], c["wa"], c["ext"]
            h0 = jax.vmap(lambda o1, o2, w, a, th, e: solve_h0_from_theta_mc(tables, o1, o2, w, a, th, ext=e))(
                ob, oc, w0, wa, X[:, i_th], ext)
            h = h0 / 100.0
            As = jnp.exp(X[:, i_logA]) * 1.0e-10
            zeros = jnp.zeros_like(ob)
            mu = c["mu0"] if mg else zeros
            sg = c["Sigma0"] if mg else zeros

            def params_one(o1, o2, hh, n, a, t, w, wa_, m, s, e):
                return cmb_params(o1, o2, hh, n, a, t, w, wa_, *((m, s) if mg else (None, None)), ext=e,
                                  nu_model=model)[0]

            params_b = jax.vmap(params_one)(ob, oc, h, c["ns"], As, c["tau"], w0, wa, mu, sg, ext)
            thermo_b = jax.vmap(lambda *a, e: recfast_thermo_batched(*a, ext=e, nu_model=model))(
                ob, oc, h, c["ns"], As, c["tau"], w0, wa, e=ext)
            scal = dict(ob=ob, oc=oc, h=h, ns=c["ns"], As=As, tau=c["tau"], w0=w0, wa=wa, mu=mu, sg=sg, h0=h0,
                        ext=ext)
            return dict(x=xt, scal=scal, params=params_b, thermo=thermo_b)

        def sources(st):
            return {**st, "grid": source_grids_batched(k_ev, st["params"], st["thermo"], hcfg, layout=layout,
                                                       compute_dtype=compute_dtype, **grid_kw)}

        def cls(st):
            def one(s, p, th, g):
                return compute_cmb_cls(
                    ombh2=s["ob"], omch2=s["oc"], h=s["h"], tau_reio=s["tau"], ns=s["ns"], As=s["As"],
                    w0=s["w0"], wa=s["wa"], ext=s["ext"], nu_model=model, lmax=l_cls, use_recfast_thermo=True,
                    source_accuracy_boost=1.0,
                    device_grids=True, k_evolve=k_ev, k_int=k_int, nonlinear_lens=nl_lens,
                    **(dict(mu0=s["mu"], Sigma0=s["sg"]) if mg else {}), **cls_extra,
                    precomputed=dict(params=p, thermo=th, grid=g))
            return dict(x=st["x"], h0=st["scal"]["h0"],
                        cls=jax.vmap(one)(st["scal"], st["params"], st["thermo"], st["grid"]))

        def lens(st):
            spec = jax.vmap(lambda c, h: _spectra_from_cls(c, h, lmax=lmax, lens_cmb=True, lmax_extra=lmax_extra,
                                                           highl=highl))(st["cls"], st["h0"])
            out = b.cmb_out(jax.tree_util.tree_map(lambda a: a[0], spec))
            return dict(x=st["x"], cmb=out)

        stages += [("cmb_prep", prep), ("cmb_sources", sources), ("cmb_cls", cls), ("cmb_lens", lens)]

    if cfg.desi_fs and fs_split:
        from gpucosmo.jax.eb.linear_pk_jax import (
            LINPK_SINGLE_BAND,
            linpk_band_transfers,
            linpk_bands,
            linpk_k_grid,
            linpk_setup,
            linpk_spectra,
        )

        I = b.internals
        pk_keys = ("pk_cb", "pk_vc_vc", "pk_vb_vc", "pk_vb_vb")
        z_out = tuple(float(z) for z in I.z_fs)
        k_np = np.asarray(linpk_k_grid(), dtype=np.float64)
        bands = linpk_bands(k_np, LINPK_SINGLE_BAND if cfg.pk_single_band else None)
        static = {}  # cfg / redshift order from the setup (static, recorded at trace time)

        def as_dict(st):
            return st if isinstance(st, dict) else {"x": st}

        def fs_prep(st):  # cosmology (θ → H0), Recfast, per-k evolution inputs
            st = as_dict(st)
            c = b.cosmo(full_x(st["x"])[0])
            mg_kw = dict(mu0=c["mu0"], Sigma0=c["Sigma0"]) if cfg.mg else {}
            pst, pcfg, order, _k = linpk_setup(ombh2=c["ombh2"], omch2=c["omch2"], h=c["h"], ns=c["ns"], As=c["As"],
                                              w0=c["w0"], wa=c["wa"], z_out=z_out, ext=c["ext"],
                                              nu_model=b.tables.model, **mg_kw)
            static.update(cfg=pcfg, order=order)
            c = {k: v for k, v in c.items() if v is not None}
            return {**st, "fs_c": c, "fs_setup": pst}

        def band_stage(i, sel, mult):
            def f(st):
                if "cfg" not in static:  # set by fs_prep's trace; derive it again if this stage is traced first
                    fs_prep(st if "fs_setup" not in st else {"x": st["x"]})
                tr = linpk_band_transfers(st["fs_setup"], k_np[sel], mult, cfg=static["cfg"], k_chunk=cfg.pk_k_chunk,
                                          compute_dtype=compute_dtype)
                return {**st, f"fs_tr{i}": tr}
            return f

        def fs_spec(st):
            tr = jnp.concatenate([st[f"fs_tr{i}"] for i in range(len(bands))], axis=0)
            spec = linpk_spectra(st["fs_setup"], tr, static["order"], k_np, st["fs_c"]["As"], st["fs_c"]["ns"])
            keep = {k: v for k, v in st.items() if not (k.startswith("fs_tr") or k == "fs_setup")}
            return {**keep, "fs_pk": {k: spec[k] for k in pk_keys}}

        def fs_tab(st):  # REPT multipole tables (one loop) from the spectra
            c = {**st["fs_c"], "mu0": None, "Sigma0": None}
            tab = I.fs_like.theory_tables(I.fs_inputs(c, st["fs_pk"]))
            out = {"fs": {k: tab[k] for k in ("pkpoles", "sigma8", "fsigma8")}}
            if "cmb" in st:
                out["cmb"] = st["cmb"]
            return out

        stages += ([("fs_prep", fs_prep)] + [(f"fs_band{i}", band_stage(i, sel, m)) for i, (sel, m, _dt) in enumerate(bands)]
                   + [("fs_spec", fs_spec), ("fs_tables", fs_tab)])
    elif cfg.desi_fs:
        def fs(st):
            xt = st["x"] if isinstance(st, dict) else st
            out = {"fs": b.theory_fs(xt)}
            if isinstance(st, dict) and "cmb" in st:
                out["cmb"] = st["cmb"]
            return out
        stages.append(("fs", fs))
    elif cmb_on(cfg):
        stages.append(("out", lambda st: {"cmb": st["cmb"]}))
    return stages


__all__ = ["theory_stages"]
