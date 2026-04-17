#!/usr/bin/env bash
# Run **on the cluster** (after: ssh qzha@login.rc.fas.harvard.edu), once.
# Packs every runs/*/models/model.pt into one tarball in $HOME for a single scp.
#
#   bash cluster_pack_all_model_pt.sh
#   bash cluster_pack_all_model_pt.sh /n/home03/qzha/nnef/runs ~/my_models.tgz
#
# Then on your **Mac** (one password + one 2FA round-trip for scp):
#
#   cd /Library/Camille/FYP/nnef
#   mkdir -p runs
#   scp qzha@login.rc.fas.harvard.edu:nnef_all_model_pt.tgz .
#   tar xzf nnef_all_model_pt.tgz -C runs
#
# No git required — copy this file to the cluster with scp/nano if needed.

set -euo pipefail

RUNS="${1:-${HOME}/nnef/runs}"
OUT="${2:-${HOME}/nnef_all_model_pt.tgz}"

if [[ ! -d "${RUNS}" ]]; then
  echo "ERROR: runs dir not found: ${RUNS}"
  exit 1
fi

cd "${RUNS}"
# Paths in archive are relative to ${RUNS}, e.g. ./v2_dihedral_.../models/model.pt
count="$(find . -type f -path '*/models/model.pt' | wc -l | tr -d ' ')"
if [[ "${count}" -eq 0 ]]; then
  echo "ERROR: no model.pt under ${RUNS}"
  exit 1
fi

echo "==> Packing ${count} file(s) -> ${OUT}"
find . -type f -path '*/models/model.pt' -print0 | tar czf "${OUT}" --null -T -
ls -lh "${OUT}"
echo "==> On laptop: scp qzha@login.rc.fas.harvard.edu:~/${OUT##*/} ."
