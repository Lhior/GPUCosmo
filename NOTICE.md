# Notices

GPUCosmo — Copyright (C) 2025-2026 Leonel Medina-Varela.

GPUCosmo is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version ([COPYING](COPYING)). It is distributed in the
hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

## Third-party code

Parts of GPUCosmo are JAX translations of other codes: the algorithms, equations, constants and
often the code structure follow the original, modified to run under `jit`, `vmap` and `grad`.
Each such file starts with a header naming its sources. Their copyright notices and licences are
in [LICENSES/](LICENSES):

| Source | Licence | Notice | GPUCosmo modules (under `src/gpucosmo/`) |
| --- | --- | --- | --- |
| [CAMB](https://github.com/cmbant/CAMB) (incl. Recfast, HMcode, Halofit, forutils) | LGPL-3.0-or-later with additional conditions | [CAMB.txt](LICENSES/CAMB.txt) | `jax/eb/` (the Einstein–Boltzmann solver), `jax/cmb/lensing_jax.py` |
| [ISiTGR](https://github.com/mishakb/ISiTGR) | as CAMB | [ISiTGR.txt](LICENSES/ISiTGR.txt) | `jax/eb/mg_jax.py` and the μ–Σ terms of the `jax/eb/` hierarchy, sources and spectra |
| [Cobaya](https://github.com/CobayaSampler/cobaya) | LGPL-3.0-or-later with additional conditions | [Cobaya.txt](LICENSES/Cobaya.txt) | `likelihoods/cobaya/` (BAO, H0, SN, CMBlikes, Planck CamSpec / plik-lite / low-ℓ), `jax/cmb/plik_lite.py`, `jax/background/likelihoods.py` |
| [CCL](https://github.com/LSSTDESC/CCL) | BSD-3-Clause | [CCL.txt](LICENSES/CCL.txt) | `jax/ccl/`, `jax/fftlog.py`, Eisenstein–Hu in `jax/eb/transfer.py`, `jax/eb/matter_pk_jax.py` |
| [FAST-PT](https://github.com/JoeMcEwen/FAST-PT) | MIT | [FAST-PT.txt](LICENSES/FAST-PT.txt) | `jax/fastpt_jax.py` |
| [act_dr6_lenslike](https://github.com/ACTCollaboration/act_dr6_lenslike) | BSD-2-Clause | [act_dr6_lenslike.txt](LICENSES/act_dr6_lenslike.txt) | `jax/cmb/act_dr6_lens_jax.py`, `likelihoods/cobaya/act_lens.py` |
| [HiLLiPoP](https://github.com/planck-npipe/hillipop), [LoLLiPoP](https://github.com/planck-npipe/lollipop), Copyright (C) 2020 Matthieu Tristram | GPL-3.0-or-later | [COPYING](COPYING) | `jax/cmb/pr4_jax.py` |

The CAMB, ISiTGR and Cobaya licence files include the full GPL-3.0 and LGPL-3.0 texts. LGPL-3.0
code may be distributed under the GPL-3.0 (LGPL-3.0 section 2), as GPUCosmo is.

### CAMB's additional conditions

The CAMB licence (and ISiTGR's, the same text) adds conditions to the LGPL. They apply to the
CAMB-derived parts of GPUCosmo, and so to anyone who uses or redistributes them:

- Any publication using results of the code must be submitted to arXiv.org at the same time as,
  or before, submitting to a journal; arXiv must be updated with a version equivalent to that
  accepted by the journal on journal acceptance.
- If you identify any bugs you report them as soon as confirmed.
- The CAMB licence, its copyright notices and disclaimer appear on all copies of the software and
  documentation, including modifications made for internal use or for distribution.

Cobaya's licence adds the first two conditions for the Cobaya-derived parts.

## Optional dependencies

GPUCosmo imports, but does not include, many other packages (JAX, NumPy, SciPy, Cobaya, CAMB,
CCL, candl, clipy, cosmopower-jax, …). Each is under its own licence; `cosmopower-jax` is GPL-3.0.
