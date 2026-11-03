Data and caches
===============

GPUCosmo keeps the large likelihood products (exports of the official likelihoods' data
and covariances) outside the Python package. ``gpucosmo.paths`` resolves every location.
Set the environment variables before running:

==========================  ===========================  ==============================================
what                        variable                     default
==========================  ===========================  ==============================================
data products               ``GPUCOSMO_DATA_DIR``        ``<checkout>/data`` from a source checkout,
                                                         else ``$XDG_DATA_HOME/gpucosmo``
                                                         (``~/.local/share/gpucosmo``)
host-table cache            ``GPUCOSMO_TABLE_CACHE``     ``<data dir>/cache`` (``0`` disables it)
JAX compilation cache       ``GPUCOSMO_JAX_CACHE_DIR``   ``~/.cache/gpucosmo_jax``
Cobaya packages             ``GPUCOSMO_PACKAGES_PATH``,  Cobaya's own setting (``cobaya-install``
(Planck clik files, SN)     ``COBAYA_PACKAGES_PATH``     records ``packages_path`` in its config)
==========================  ===========================  ==============================================

``gpucosmo-paths`` prints the resolved locations and which data sets are present.

Data sets (sub-directories of the data directory):

=====================  ======================================  ==========================================
directory              needed by                               produced by
=====================  ======================================  ==========================================
``background_tables``  every analysis (θ* → H0, distances)     ``scripts/build_background_tables.py``;
                                                               a copy ships with the package
``bbn``                CMB analyses (helium fraction)          CAMB's BBN tables; read from an installed
                                                               ``camb`` package if absent
``camb``               CMB analyses (high-ℓ lensing template)  CAMB's
                                                               ``HighLExtrapTemplate_lenspotentialCls.dat``;
                                                               read from ``camb`` if absent
``planck_pr4``         Planck PR4 HiLLiPoP + LoLLiPoP          exports of the official likelihood
                                                               packages (not part of this preview)
``act_dr6_lens``       ACT DR6 (+ Planck PR4) lensing          the act_dr6_lenslike v1.2 products
                                                               (``scripts/fetch_act_dr6_lens_data.sh``),
                                                               reduced to small matrices on first use
=====================  ======================================  ==========================================

The Planck 2018 likelihoods (plik, Commander low-ℓ TT, SimAll low-ℓ EE) are read from the
Cobaya packages directory through clipy: install the clik versions with ``cobaya-install``
(``planck_2018_highl_plik.TTTEEE``, ``planck_2018_lowl.TT_clik``, ``planck_2018_lowl.EE_clik``).

Data for the DESI DR2 + CMB + SN demo
-------------------------------------

The shipped ``w0wa_desidr2_cmb_*`` and ``lcdm_desidr2_cmb_*`` analyses read every likelihood's
data through Cobaya, plus the ACT DR6 lensing products for its JAX port:

.. code-block:: bash

    pip install ".[desi-full]" \
        "act_dr6_lenslike @ git+https://github.com/ACTCollaboration/act_dr6_lenslike"
    cobaya-install -p PACKAGES bao.desi_dr2.desi_bao_all sn.desy5 sn.pantheonplus sn.union3 \
        planck_2018_lowl.TT_clik planck_2018_lowl.EE_clik planck_NPIPE_highl_CamSpec.TTTEEE \
        act_dr6_lenslike.ACTDR6LensLike
    export GPUCOSMO_PACKAGES_PATH=PACKAGES
    # the JAX port reads the same v1.2 products from the data directory: link them
    # (or download a second copy with scripts/fetch_act_dr6_lens_data.sh)
    mkdir -p data && ln -s "$PACKAGES/data/ACT_dr6_likelihood" data/act_dr6_lens
    gpucosmo-paths                      # check what was found

User likelihoods load their own data, typically in ``__init__`` from paths given as yaml
options (see :doc:`adding_likelihoods`).
