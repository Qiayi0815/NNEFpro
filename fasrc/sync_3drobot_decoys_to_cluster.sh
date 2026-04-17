#!/usr/bin/env bash
# Laptop → FASRC: ship unpacked 3DRobot bead CSVs (large tree under nnef/data/decoys).
#
#   bash fasrc/sync_3drobot_decoys_to_cluster.sh
#
# Prereq: unpack + *_bead.csv locally (see nnef/data_prep_scripts/unpack_zhang_3drobot_decoys.py
# and regenerate_decoy_beads.py). Excludes prior decoy_loss_* outputs to save space.
#
# Remote layout must match paths.data_path:  ~/nnef/nnef/data/decoys/3DRobot_set/<TARGET>/...

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-qzha@login.rc.fas.harvard.edu}"
REMOTE_REPO="${REMOTE_REPO:-/n/home03/qzha/nnef}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${ROOT}/nnef/data/decoys/3DRobot_set"

cd "$ROOT"
if [[ ! -d "$SRC" ]]; then
  echo "ERROR: missing $SRC" >&2
  exit 1
fi

echo "==> rsync 3DRobot_set -> ${REMOTE_HOST}:${REMOTE_REPO}/nnef/data/decoys/3DRobot_set"
rsync -avh --progress \
  --exclude='decoy_loss_*/' \
  "${SRC}/" "${REMOTE_HOST}:${REMOTE_REPO}/nnef/data/decoys/3DRobot_set/"

echo "Done. On cluster, quick check:"
echo "  ssh ${REMOTE_HOST} 'ls ${REMOTE_REPO}/nnef/data/decoys/3DRobot_set | head'"
