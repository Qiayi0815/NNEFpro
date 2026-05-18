#!/usr/bin/env bash
# Sync nnef repo (code + fasrc + scripts) to FASRC. Does NOT copy decoys, runs,
# params, or HDF5 — use sync_code_and_rama_v2_to_cluster.sh when you also need
# local hhsuite_*.h5 pushed to nnef_data, or sync_{3drobot,casp14}_decoys_to_cluster.sh
# for decoy beads.
#
#   bash fasrc/sync_code_only_to_cluster.sh
#
# Override destination:
#   REMOTE_HOST=user@host REMOTE_REPO=/path/to/nnef bash fasrc/sync_code_only_to_cluster.sh

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-qzha@login.rc.fas.harvard.edu}"
REMOTE_REPO="${REMOTE_REPO:-/n/home03/qzha/nnef}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$ROOT"

echo "==> rsync code -> ${REMOTE_HOST}:${REMOTE_REPO}"
rsync -avh --progress \
  --exclude='.git/' \
  --exclude='__pycache__/' \
  --exclude='.DS_Store' \
  --exclude='runs/' \
  --exclude='params/' \
  --exclude='nnef/data/*.h5' \
  --exclude='nnef/data/decoys/' \
  --exclude='data_hh/' \
  --exclude='_archive/' \
  --exclude='.claude/' \
  --exclude='.claude.json' \
  --exclude='.omx/' \
  --exclude='eval/' \
  --exclude='*.pdf' \
  ./ "${REMOTE_HOST}:${REMOTE_REPO}/"

echo "Done. On cluster: cd ${REMOTE_REPO}"
