"""Port-vs-bridge comparison and timing (tests, ``scripts/bench_cobaya_ports.py``).

``compare(port)`` evaluates a native port and its Cobaya bridge on identical inputs — the
background of the reference cosmology (optionally perturbed, for background-only
likelihoods) and the reference CMB spectra of ``tests/data/cmb_reference_lcdm.npz`` (our
solver's lensed spectra to ℓ 3258 at the ΛCDM fiducial) — at points that vary the
likelihood's own parameters, and times both: the port jitted and vmapped over a batch, the
bridge one point at a time through Cobaya.
"""

from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

REFERENCE = Path(__file__).resolve().parents[4] / "tests" / "data" / "cmb_reference_lcdm.npz"


REFERENCE_PK = REFERENCE.with_name("pk_reference_lcdm.npz")


def reference(path: str | Path | None = None) -> dict:
    """{"cosmo": {h, ombh2, ...}, "cmb": theory block, "pk": theory block, "static": shared-block
    axes} of the reference files (the linear-P(k) block from ``pk_reference_lcdm.npz``, same
    cosmology)."""
    z = np.load(path or REFERENCE)
    cosmo = {k[6:]: jnp.asarray(z[k]) for k in z.files if k.startswith("cosmo_")}
    cosmo.setdefault("mu0", None)
    cosmo.setdefault("Sigma0", None)
    cmb = {k[4:]: jnp.asarray(z[k]) for k in z.files if k.startswith("cmb_")}
    out = {"cosmo": cosmo, "cmb": cmb, "static": {}}
    if path is None and REFERENCE_PK.exists():
        p = np.load(REFERENCE_PK)
        out["pk"] = {k: jnp.asarray(p[k]) for k in ("lnpk_m", "lnpk_cb", "sigma8", "fsigma8", "lnpk_m_x", "lnpk_m_nl")
                     if k in p.files}
        out["static"]["pk"] = {"z": np.asarray(p["z"]), "k": np.asarray(p["k"])}
        if "k_x" in p.files:  # extended grid with the non-linear P_m (HMcode-2020)
            out["static"]["pk"]["k_x"] = np.asarray(p["k_x"])
    return out


def points(L, n: int, seed: int = 0, spread: float = 0.3) -> dict:
    """{param: (n,) values}: the start point, then draws within ``spread`` of each prior's width
    (Gaussian: ``spread``·σ·3) around it."""
    rng = np.random.default_rng(seed)
    out = {}
    for p, v in L.params.items():
        w = 3.0 * v.gauss[1] if v.gauss else (v.hi - v.lo)
        vals = v.start + spread * w * rng.uniform(-0.5, 0.5, size=n)
        vals[0] = v.start
        out[p] = np.clip(vals, v.lo, v.hi)
    return out


def inputs(L, ref: dict, params: dict, cosmo_shift: dict | None = None):
    """``LikelihoodInputs`` of ``L`` on the reference products."""
    from gpucosmo.jax.background.cosmology import load_background_tables
    from gpucosmo.jax.inference.mg_bundle import shared_block
    from gpucosmo.likelihoods.base import Cosmology, LikelihoodInputs

    c = dict(ref["cosmo"])
    for k, dv in (cosmo_shift or {}).items():
        c[k] = c[k] + dv
    cosmo = Cosmology(c, load_background_tables())
    th = L.theory(cosmo) if L.has_theory() else None
    shared = {r: shared_block(r, {r: ref[r]}, ref.get("static")) for r in L.requires}
    return LikelihoodInputs(theory=th, params={k: jnp.asarray(v) for k, v in params.items()}, cosmo=cosmo,
                            shared=shared)


def _shared(raw: dict, ref: dict) -> dict:
    from gpucosmo.jax.inference.mg_bundle import shared_block

    return {r: shared_block(r, raw, ref.get("static")) for r in raw}


def compare(port, n: int = 8, seed: int = 0, batch: int = 1024, cosmo_shifts=None, ref_path=None,
            n_bridge_timing: int = 20) -> dict:
    """Native vs bridge log L at ``n`` points (and at each ``cosmo_shifts`` entry) + timings."""
    ref = reference(ref_path)
    bridge = port.bridge
    pts = points(port, n, seed)
    shifts = [None] + list(cosmo_shifts or [])
    rows = []
    for sh in shifts:  # one compiled port function per cosmology (the theory products differ)
        inp_s = inputs(port, ref, {k: v[0] for k, v in pts.items()}, sh)
        raw = {r: ref[r] for r in port.requires}
        f_s = jax.jit(lambda th, p_, rw, base=inp_s: port.loglike(base._replace(theory=th, params=p_,
                                                                                shared=_shared(rw, ref))))
        for i in range(n):
            par = {k: v[i] for k, v in pts.items()}
            inp = inp_s._replace(params={k: jnp.asarray(v) for k, v in par.items()})
            nat = float(f_s(inp.theory, inp.params, raw))
            brg = float(bridge.loglike(inp))
            if port.has_log_prior():  # prior terms: the port's vs the bridge's (both JAX, the same code
                nat += float(port.log_prior(inp.params))  # path unless the port overrides it)
            if bridge.has_log_prior():
                brg += float(bridge.log_prior(inp.params))
            rows.append((nat, brg))
    rows = np.asarray(rows)
    diff = rows[:, 0] - rows[:, 1]
    # timing: native batched over the parameters at the reference products
    inp0 = inputs(port, ref, {k: v[0] for k, v in pts.items()})
    bpar = {k: jnp.asarray(np.resize(v, batch)) for k, v in pts.items()}

    # batched over the theory products (copies of the reference ones) and the parameters
    bth = jax.tree_util.tree_map(lambda a: jnp.broadcast_to(a, (batch,) + jnp.shape(a)), inp0.theory)
    raw0 = {r: ref[r] for r in port.requires}
    bsh = jax.tree_util.tree_map(lambda a: jnp.broadcast_to(a, (batch,) + jnp.shape(a)), raw0)

    def f(th, par, rw, dummy, c=inp0.cosmo):
        return port.loglike(inp0._replace(theory=th, params=par, shared=_shared(rw, ref), cosmo=c)) + 0.0 * dummy

    fb = jax.jit(jax.vmap(f))
    dummy = jnp.zeros(batch)
    jax.block_until_ready(fb(bth, bpar, bsh, dummy))
    reps = 5
    t0 = time.perf_counter()
    for _ in range(reps):
        jax.block_until_ready(fb(bth, bpar, bsh, dummy))
    t_native = (time.perf_counter() - t0) / reps / batch
    th0 = jax.tree_util.tree_map(np.asarray, inp0.theory) if inp0.theory is not None else {}
    cmb0 = None
    if "cmb" in bridge.requires:
        cl_req = bridge.requirements()["cmb"]
        cmb0 = {s: np.asarray(inp0.shared["cmb"]["cl"][s][:cl_req["lmax"] + 1]) for s in cl_req["spectra"]}
    pk0 = (jax.tree_util.tree_map(np.asarray, bridge.pk_rows(inp0.shared["pk"])) if "pk" in bridge.requires
           else None)
    nuis = np.asarray([pts[p][0] for p in bridge._own])
    cp = inp0.cosmo.params
    tin = np.asarray([100 * float(cp["h"]), float(cp["ombh2"]), float(cp["omch2"]), np.log(1e10 * float(cp["As"])),
                      float(cp["As"]), float(cp["ns"]), float(cp["tau"]), float(cp["w0"]), float(cp["wa"]), 0.0, 0.0,
                      float(inp0.cosmo.mnu), float(inp0.cosmo.nnu), float(inp0.cosmo.omk),
                      float(inp0.cosmo.meffsterile)])
    # the likelihood's own parameters change every call (Cobaya caches per parameter point)
    nuis_rows = [np.asarray([pts[p][i % n] for p in bridge._own]) for i in range(n_bridge_timing)]
    t0 = time.perf_counter()
    for i in range(n_bridge_timing):
        bridge._host(th0, cmb0, nuis_rows[i], tin, pk0)
    t_bridge = (time.perf_counter() - t0) / max(n_bridge_timing, 1) if n_bridge_timing else float("nan")
    return dict(name=port.cobaya_name, n=len(rows), max_abs_diff=float(np.max(np.abs(diff))),
                native=rows[:, 0].tolist(), bridge=rows[:, 1].tolist(), t_native_us=1e6 * t_native,
                t_bridge_us=1e6 * t_bridge, speedup=t_bridge / t_native if t_native > 0 else None,
                device=jax.devices()[0].platform, batch=batch)


__all__ = ["compare", "inputs", "points", "reference"]
