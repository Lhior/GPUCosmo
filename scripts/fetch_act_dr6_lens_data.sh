#!/bin/bash
# Download the ACT DR6 lensing likelihood data (act_dr6_lenslike v1.2: ACT and Planck PR4
# lensing bandpowers, binning matrices, covariances, likelihood-correction matrices; 2.8 GB
# unpacked) into data/act_dr6_lens/v1.2, as used by gpucosmo.jax.cmb.act_dr6_lens_jax.
#
#   bash scripts/fetch_act_dr6_lens_data.sh
set -euo pipefail
cd "$(dirname "$0")/.."
URL=https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr6/likelihood/data/ACT_dr6_likelihood_v1.2.tgz
OUT=data/act_dr6_lens
mkdir -p "$OUT"
if [ ! -f "$OUT/v1.2/covmat_actplanck.txt" ]; then
  curl -sfL -C - -o "$OUT/ACT_dr6_likelihood_v1.2.tgz" "$URL"
  tar xzf "$OUT/ACT_dr6_likelihood_v1.2.tgz" -C "$OUT"
  rm "$OUT/ACT_dr6_likelihood_v1.2.tgz"
fi
echo "ACT DR6 lensing data in $OUT/v1.2"
