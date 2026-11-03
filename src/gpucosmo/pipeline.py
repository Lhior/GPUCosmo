"""Analysis set-up shared by the command-line pipelines: Cobaya yaml → likelihood bundle.

``build_analysis_bundle`` resolves a Cobaya input (``inference.cobaya_yaml``), adds any
user likelihoods (``gpucosmo.likelihoods``) and builds the GPU bundle
(``inference.mg_bundle``). The ``gpucosmo-dali-hmc`` / ``gpucosmo-dali-ns``
commands (``gpucosmo.cli``) start from it, and so can user code::

    from gpucosmo.pipeline import build_analysis_bundle
    bundle, resolved, diffs = build_analysis_bundle(config="my_analysis.yaml")
    bundle.log_post_batch_fn(x)          # exact log posterior, batched over rows of x
"""

from __future__ import annotations

from pathlib import Path

import jax

from gpucosmo.jax_cache import enable_jax_cache, stable_cache_key  # re-exported (__all__)

jax.config.update("jax_enable_x64", True)


def analysis_configs() -> dict[str, Path]:
    """Named analyses (``--analysis NAME``): the Cobaya yamls shipped with the package
    (``gpucosmo/configs/cobaya``, the official DESI DR1 μ–Σ / ΛCDM chain configs)."""
    from gpucosmo.paths import config_dir

    return {p.stem: p for p in sorted(config_dir().glob("*.yaml"))}


def lens_override(args) -> bool | None:
    """``--cmb-nonlinear-lens`` → True, ``--cmb-linear-lens`` → False, neither → None (yaml rule).

    Nested models compared by evidence must use the same choice: the resolver gives
    camb+CMB yamls non-linear lensing (Cobaya's NonLinear_lens rule) but ISiTGR yamls
    keep NonLinear_none, so a ΛCDM/MG pair needs an explicit flag on one of them.
    """
    nl, lin = bool(getattr(args, "cmb_nonlinear_lens", False)), bool(getattr(args, "cmb_linear_lens", False))
    if nl and lin:
        raise SystemExit("--cmb-nonlinear-lens and --cmb-linear-lens are mutually exclusive")
    return True if nl else (False if lin else None)


def load_plugins(modules=()) -> None:
    """Import ``modules`` (names or ``path/to/file.py``) so the likelihoods they register are
    known, and load the ``gpucosmo.likelihoods`` entry points of installed packages."""
    from gpucosmo.likelihoods.registry import load_entry_points, load_plugin_module

    for m in modules or ():
        load_plugin_module(m)
    load_entry_points()


# In-process sharing (gpucosmo-evidence): every stage of one process gets the same bundle, so its
# compiled theory program (~85 s to trace, lower and load per process) is reused across stages.
_SHARED: dict = {"on": False, "walker_batch": 1, "batch_overrides": None, "cache": {}}


class shared_bundle:
    """Context manager: ``build_analysis_bundle`` returns one bundle per analysis inside it, built
    with ``walker_batch`` / ``batch_overrides`` whatever a stage asks for (stages differ only in
    batching, not in the likelihood)."""

    def __init__(self, walker_batch: int = 1, batch_overrides: dict | None = None):
        self.kw = dict(walker_batch=walker_batch, batch_overrides=batch_overrides)

    def __enter__(self):
        _SHARED.update(on=True, cache={}, **self.kw)
        return self

    def __exit__(self, *exc):
        _SHARED.update(on=False, cache={})
        return False


def build_analysis_bundle(analysis: str | None = None, config: str | Path | dict | None = None, *,
                          walker_batch: int = 0, bbn_prior: str | None = None,
                          cmb_nonlinear_lens: bool | None = None, cmb_lensing: bool | None = None,
                          fs_tables: bool = False, verbose: bool = True,
                          autosize: bool = False, n_chains: int = 8,
                          batch_overrides: dict | None = None, reserved_fn=None,
                          frag: float | None = None, likelihoods=(), plugins=(), cmb_legacy_lensing: bool = False):
    """Bundle from a Cobaya yaml (``config``: path or dict, or ``analysis``: a name from
    ``analysis_configs()``).

    ``likelihoods``: extra ``gpucosmo.likelihoods.Likelihood`` instances added to the ones
    the yaml lists; ``plugins``: modules to import first (they register likelihood classes
    the yaml may name). ``bbn_prior="paper"`` / ``cmb_nonlinear_lens`` override the yaml (to
    continue runs started with those settings); ``cmb_legacy_lensing``: lens the CMB on the solver's
    base path as runs before 2026-10-17 (``MGConfig.cmb_legacy_lensing``; default: the accurate
    high-ℓ path). Returns (bundle, resolved yaml, audit diffs).
    """
    import dataclasses

    from gpucosmo.jax.inference.cobaya_yaml import audit_bundle, resolve_cobaya
    from gpucosmo.jax.inference.mg_bundle import BBN_PRIOR_PAPER, build_mg_bundle

    if _SHARED["on"]:
        key = (str(config) if not isinstance(config, dict) else id(config), analysis, bbn_prior, cmb_nonlinear_lens,
               cmb_lensing, fs_tables, tuple(plugins or ()), tuple(id(L) for L in likelihoods or ()), cmb_legacy_lensing)
        if key in _SHARED["cache"]:
            return _SHARED["cache"][key]
        walker_batch, batch_overrides = _SHARED["walker_batch"], _SHARED["batch_overrides"]
        autosize = False
    load_plugins(plugins)
    if config is None:
        configs = analysis_configs()
        if analysis not in configs:
            raise SystemExit(f"unknown analysis {analysis!r}; known: {sorted(configs)} (or pass a yaml)")
        config = configs[analysis]
    path = Path(config) if not isinstance(config, dict) else Path("<dict>")
    res = resolve_cobaya(config, walker_batch=walker_batch)
    if likelihoods:
        res.add_likelihoods(likelihoods)
    over = {}
    if bbn_prior == "paper" and res.config.bbn_gauss is not None:
        over["bbn_gauss"] = BBN_PRIOR_PAPER
    if cmb_nonlinear_lens is not None and cmb_nonlinear_lens != res.config.cmb_nonlinear_lens:
        over["cmb_nonlinear_lens"] = cmb_nonlinear_lens
    if cmb_lensing is not None and cmb_lensing != res.config.cmb_lensing:
        over["cmb_lensing"] = cmb_lensing
    if cmb_legacy_lensing and not res.config.cmb_legacy_lensing:
        over["cmb_legacy_lensing"] = True
    if fs_tables and res.config.desi_fs:
        over["fs_tables"] = True  # DALI layout only; the likelihood is unchanged
    if over:
        res.notes.append(f"command-line overrides of the yaml: {over}")
        res.config = dataclasses.replace(res.config, **over)
    if batch_overrides:
        res.config = dataclasses.replace(res.config, **batch_overrides)

    def build(cfg):
        return build_mg_bundle(cfg, priors=res.priors, likelihoods=res.extra_likelihoods)

    b = build(res.config)
    if autosize and not batch_overrides:
        from gpucosmo.jax.inference.autosize import FRAG, autosize_batches

        resident = sum(int(a.nbytes) for a in jax.tree_util.tree_leaves(b.aux))
        reserved = int(reserved_fn(b)) if reserved_fn is not None else 0
        cfg = autosize_batches(build, res.config, analysis=str(path.stem), n_chains=n_chains,
                               resident_bytes=resident, reserved_bytes=reserved,
                               frag=FRAG if frag is None else frag, verbose=verbose)
        if cfg != res.config:
            res.config = cfg
            b = build(res.config)
    diffs = audit_bundle(b, res)
    if _SHARED["on"]:
        _SHARED["cache"][key] = (b, res, diffs)
    if verbose:
        print(f"config {path}: {b.likelihood_label}; {len(b.param_names)} params", flush=True)
        for n in res.notes + b.notes:
            print("  note:", n, flush=True)
        for d in diffs:
            print("  differs from yaml:", d, flush=True)
    return b, res, diffs


__all__ = ["analysis_configs", "build_analysis_bundle", "enable_jax_cache", "lens_override", "load_plugins", "stable_cache_key",
           "shared_bundle"]
