#!/usr/bin/env python3
"""Same program as ``gpucosmo-dali-hmc`` (``gpucosmo.cli.dali_hmc``), runnable from a source checkout
without installing the package. Other scripts import its names from here."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gpucosmo.cli.dali_hmc import *  # noqa: E402,F401,F403
from gpucosmo.cli.dali_hmc import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
