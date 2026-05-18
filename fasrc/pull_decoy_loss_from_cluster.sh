#!/usr/bin/env bash
# Run on your **laptop** from the nnef repo root.
# Pulls per-target decoy CSVs from FASRC: nnef/data/decoys/<decoy_set>/<decoy_loss_dir>/
#
# layout on cluster (same locally after sync):
#   nnef/data/decoys/casp14/decoy_loss_<exp_tag>/T1053_decoy_loss.csv
#   nnef/data/decoys/3DRobot_set/decoy_loss_<exp_tag>/1BYIA_decoy_loss.csv
#
# Usage:
#   # Entire nnef/data/decoys/ tree (can be large)
#   bash fasrc/pull_decoy_loss_from_cluster.sh --all
#
#   # Only specific subdirs (path relative to nnef/data/decoys/)
#   bash fasrc/pull_decoy_loss_from_cluster.sh \
#     casp14/decoy_loss_yang_exp1 \
#     casp14/decoy_loss_v1_pure_rama_v2_6228201 \
#     3DRobot_set/decoy_loss_v1_pure_rama_v2_6228201
#
# Override: REMOTE_HOST, REMOTE_REPO

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-qzha@login.rc.fas.harvard.edu}"
REMOTE_REPO="${REMOTE_REPO:-/n/home03/qzha/nnef}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_BASE="${REMOTE_HOST}:${REMOTE_REPO}/nnef/data/decoys"
LOCAL_BASE="${ROOT}/nnef/data/decoys"

if [[ $# -eq 0 ]]; then
  echo "usage: $0 --all"
  echo "   or: $0 <relative_path_under_decoys> [<path> ...]"
  echo "example paths:"
  echo "  casp14/decoy_loss_yang_exp1"
  echo "  casp14/decoy_loss_v1_pure_rama_v2_6228201"
  exit 1
fi

if [[ "$1" == "--all" ]]; then
  mkdir -p "${LOCAL_BASE}"
  echo "==> rsync ${REMOTE_BASE}/ -> ${LOCAL_BASE}/"
  rsync -avh --progress "${REMOTE_BASE}/" "${LOCAL_BASE}/"
  echo "Done."
  exit 0
fi

mkdir -p "${LOCAL_BASE}"

for rel in "$@"; do
  rel="${rel#/}"
  rel="${rel#nnef/data/decoys/}"
  if [[ -z "${rel}" || "${rel}" == *".."* ]]; then
    echo "[pull_decoys] skip unsafe path: ${rel}" >&2
    exit 1
  fi
  echo "==> rsync ${REMOTE_BASE}/${rel}/ -> ${LOCAL_BASE}/${rel}/"
  mkdir -p "${LOCAL_BASE}/$(dirname "${rel}")"
  rsync -avh --progress "${REMOTE_BASE}/${rel}/" "${LOCAL_BASE}/${rel}/"
done

echo "Done. Local: ${LOCAL_BASE}/"
