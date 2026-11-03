"""Skip helper for tests that build Cobaya likelihoods from installed data (installs without the
Cobaya packages, e.g. CI's no-data jobs, must skip them; #17)."""

from __future__ import annotations

import pytest


def require_cobaya_data(*likelihoods: str) -> None:
    """Skip unless the Cobaya packages path holds the data of these likelihoods."""
    pytest.importorskip("cobaya")
    from cobaya.component import get_component_class

    from gpucosmo.paths import packages_path

    path = packages_path()
    if path is None:
        pytest.skip("no Cobaya packages path (GPUCOSMO_PACKAGES_PATH / COBAYA_PACKAGES_PATH)")
    for name in likelihoods:
        if not get_component_class(name, kind="likelihood").is_installed(path=str(path), data=True):
            pytest.skip(f"{name} data not installed under {path}")
