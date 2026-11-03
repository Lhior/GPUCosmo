"""Where GPUCosmo looks for its data, external likelihood packages and caches.

Every location can be set by an environment variable (read when first used, so set it
before the first call). Defaults:

=========================  ===========================  =================================================
what                       variable                     default
=========================  ===========================  =================================================
data products (likelihood  ``GPUCOSMO_DATA_DIR``        ``<source checkout>/data`` when running from a
exports, tables)                                        checkout, else ``$XDG_DATA_HOME/gpucosmo``
                                                        (``~/.local/share/gpucosmo``)
host-table cache           ``GPUCOSMO_TABLE_CACHE``     ``<data dir>/cache`` (``0`` disables it)
JAX compilation cache      ``GPUCOSMO_JAX_CACHE_DIR``   ``~/.cache/gpucosmo_jax``
Cobaya packages (Planck    ``GPUCOSMO_PACKAGES_PATH``,  Cobaya's own setting (``cobaya-install`` writes
clik files, SN data)       ``COBAYA_PACKAGES_PATH``     ``packages_path`` to its config file)
=========================  ===========================  =================================================

``gpucosmo-paths`` prints the resolved locations and which data sets are present.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_DATA = "GPUCOSMO_DATA_DIR"
ENV_TABLE_CACHE = "GPUCOSMO_TABLE_CACHE"
ENV_JAX_CACHE = "GPUCOSMO_JAX_CACHE_DIR"
ENV_PACKAGES = ("GPUCOSMO_PACKAGES_PATH", "COBAYA_PACKAGES_PATH")

# Data sets: sub-directory of the data dir → (what needs it, how it is produced).
DATASETS = {
    "background_tables": ("background (θ* → H0, distances, r_drag) for every analysis",
                          "scripts/build_background_tables.py"),
    "bbn": ("BBN helium fraction (CMB analyses)", "PArthENoPE / PRIMAT tables from CAMB"),
    "camb": ("high-ℓ lensing template (CMB analyses)", "CAMB's HighLExtrapTemplate_lenspotentialCls.dat"),
    "planck_pr4": ("Planck PR4 HiLLiPoP + LoLLiPoP", "scripts/ref_planck_pr4_dump.py"),
    "act_dr6_lens": ("ACT DR6 (+ Planck PR4) CMB lensing", "act_dr6_lenslike v1.2 products (reduced on first use)"),
    "desi_dr1": ("DESI DR1 full-shape + BAO", "scripts/ref_desi_fs_dump.py"),
    "des_y3": ("DES Y3 3x2pt (des_y3_3x2pt)", "the CosmoSIS standard library's likelihood/des-y3: "
               "2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate.fits and shear_ratio/…_sr.pkl"),
    "hsc_y3": ("HSC Y3 × SDSS 3x2pt / 2x2pt (hsc_y3_3x2pt, hsc_y3_2x2pt)", "github.com/ztq1996/hscy3-data-sharing cloned as "
               "hsc_y3/hscy3-data-sharing; PSF templates (systematics.npz, coorT_cat0.npz) in hsc_y3/psf"),
}


def source_root() -> Path | None:
    """Root of the source checkout this package runs from (``src/gpucosmo`` layout with a
    ``pyproject.toml``), or None for an installed package."""
    root = Path(__file__).resolve().parents[2]
    if (root / "pyproject.toml").is_file() and (root / "src" / "gpucosmo").is_dir():
        return root
    return None


def _xdg(var: str, fallback: str) -> Path:
    v = os.environ.get(var, "")
    return Path(v) if v else Path.home() / fallback


def data_dir(*parts: str) -> Path:
    """Data directory (``GPUCOSMO_DATA_DIR``; a checkout's ``data/``; else
    ``$XDG_DATA_HOME/gpucosmo``), optionally joined with ``parts``."""
    env = os.environ.get(ENV_DATA, "")
    if env:
        base = Path(env).expanduser()
    else:
        root = source_root()
        base = root / "data" if root is not None else _xdg("XDG_DATA_HOME", ".local/share") / "gpucosmo"
    return base.joinpath(*parts)


def sibling_checkout(name: str, env: str) -> Path | None:
    """A companion checkout used by the legacy galaxy–galaxy-lensing and CAMB-reference code:
    ``$<env>``, else ``<source root>/../<name>`` if it exists, else None."""
    v = os.environ.get(env, "")
    if v:
        return Path(v).expanduser()
    root = source_root()
    if root is not None and (root.parent / name).is_dir():
        return root.parent / name
    return None


def hsc_root() -> Path | None:
    """HSC galaxy–galaxy-lensing checkout (``GPUCOSMO_HSC_ROOT``; legacy GGL likelihoods only)."""
    return sibling_checkout("HSC", "GPUCOSMO_HSC_ROOT")


def packaged_data(name: str) -> Path:
    """A small data file shipped inside the package (``gpucosmo/_data``)."""
    return Path(__file__).resolve().parent / "_data" / name


def camb_package_file(name: str) -> Path | None:
    """``name`` from the installed ``camb`` Python package (it ships CAMB's BBN tables and
    the high-ℓ lensing template), or None without camb. Located without importing camb."""
    import importlib.util

    try:
        spec = importlib.util.find_spec("camb")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    p = Path(next(iter(spec.submodule_search_locations))) / name
    return p if p.is_file() else None


def data_file(*parts: str, fallbacks=()) -> Path:
    """``data_dir(*parts)`` if it exists, else the first existing of ``fallbacks`` (paths or
    None), else ``data_dir(*parts)`` (so a missing-file error names the expected place)."""
    p = data_dir(*parts)
    if p.exists():
        return p
    for f in fallbacks:
        if f is not None and Path(f).exists():
            return Path(f)
    return p


def table_cache_dir() -> Path | None:
    """Host-table cache (``GPUCOSMO_TABLE_CACHE``; ``0`` → disabled; default
    ``<data dir>/cache``)."""
    env = os.environ.get(ENV_TABLE_CACHE, "")
    if env == "0":
        return None
    return Path(env).expanduser() if env else data_dir("cache")


def jax_cache_dir() -> Path:
    """Persistent XLA compilation cache (``GPUCOSMO_JAX_CACHE_DIR``; ``~/.cache/gpucosmo_jax``)."""
    env = os.environ.get(ENV_JAX_CACHE, "")
    return Path(env).expanduser() if env else _xdg("XDG_CACHE_HOME", ".cache") / "gpucosmo_jax"


def _cobaya_config_packages_path() -> str | None:
    # Cobaya's config file at its Linux location first (importing cobaya costs ~5 s), then
    # Cobaya's own resolution (other platforms' config directories).
    path = _xdg("XDG_CONFIG_HOME", ".config") / "cobaya" / "config.yaml"
    if path.is_file():
        try:
            import yaml

            p = (yaml.safe_load(path.read_text()) or {}).get("packages_path")
            if p:
                return str(p)
        except Exception:  # noqa: BLE001
            pass
    try:
        from cobaya.tools import load_packages_path_from_config_file  # type: ignore

        p = load_packages_path_from_config_file()
        return str(p) if p else None
    except Exception:  # noqa: BLE001 - Cobaya absent or API changed
        return None


def packages_path(required: bool = False) -> Path | None:
    """Cobaya packages directory (Planck clik files, SN data, …): ``GPUCOSMO_PACKAGES_PATH``,
    ``COBAYA_PACKAGES_PATH``, then Cobaya's config file. ``required``: raise if unset."""
    for var in ENV_PACKAGES:
        if os.environ.get(var):
            return Path(os.environ[var]).expanduser()
    p = _cobaya_config_packages_path()
    if p:
        return Path(p).expanduser()
    if required:
        raise FileNotFoundError(
            "no Cobaya packages path: set GPUCOSMO_PACKAGES_PATH (or COBAYA_PACKAGES_PATH), or run "
            "`cobaya-install <yaml> --packages-path <dir>`, which records it in Cobaya's config")
    return None


def resolve_packages_path(packages_path_: str | os.PathLike | None) -> str:
    """``packages_path_`` if given, else ``packages_path(required=True)`` (as str)."""
    return str(packages_path_) if packages_path_ else str(packages_path(required=True))


def config_dir() -> Path:
    """The Cobaya yamls shipped with the package (``gpucosmo/configs/cobaya``)."""
    return Path(__file__).resolve().parent / "configs" / "cobaya"


def describe() -> dict:
    """Resolved locations and data-set presence (``gpucosmo-paths``)."""
    d = data_dir()
    return {
        "source_root": str(source_root()) if source_root() else None,
        "data_dir": str(d),
        "table_cache": str(table_cache_dir()) if table_cache_dir() else None,
        "jax_cache": str(jax_cache_dir()),
        "packages_path": str(packages_path()) if packages_path() else None,
        "configs": str(config_dir()),
        "datasets": {k: {"present": (d / k).is_dir() and any((d / k).iterdir()), "used_by": v[0], "source": v[1]}
                     for k, v in DATASETS.items()},
    }


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(prog="gpucosmo-paths", description="Show where GPUCosmo looks for data, "
                                 "Cobaya packages and caches (see gpucosmo.paths for the variables).")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    info = describe()
    if args.json:
        print(json.dumps(info, indent=2))
        return 0
    for k in ("source_root", "data_dir", "table_cache", "jax_cache", "packages_path", "configs"):
        print(f"{k:14s} {info[k]}")
    print("data sets:")
    for k, v in info["datasets"].items():
        print(f"  {'✓' if v['present'] else '✗'} {k:18s} {v['used_by']}  [{v['source']}]")
    return 0


__all__ = ["DATASETS", "ENV_DATA", "camb_package_file", "data_file", "hsc_root", "packaged_data",
           "sibling_checkout", "ENV_JAX_CACHE", "ENV_PACKAGES", "ENV_TABLE_CACHE", "config_dir", "data_dir",
           "describe", "jax_cache_dir", "packages_path", "resolve_packages_path", "source_root", "table_cache_dir"]


if __name__ == "__main__":
    raise SystemExit(main())
