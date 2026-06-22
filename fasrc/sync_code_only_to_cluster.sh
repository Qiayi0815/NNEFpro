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
# eval/ handling: push *.py analysis/driver scripts under eval/decoys/ and
# eval/md_eval/, but DON'T push any data (csv, pdb, png, dcd, h5, json,
# subdirs of mode2_*, yang_*, 3dr_*, casp14_*, advisor_meeting/, etc.).
# The include rules come BEFORE the catch-all `eval/**` exclude so they win.
rsync -avh --progress \
  --include='eval/' \
  --include='eval/decoys/' \
  --include='eval/decoys/*.py' \
  --include='eval/md_eval/' \
  --include='eval/md_eval/*.py' \
  --exclude='eval/**' \
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
  --exclude='*.pdf' \
  ./ "${REMOTE_HOST}:${REMOTE_REPO}/"

echo "Done. On cluster: cd ${REMOTE_REPO}"
echo "  Pushed: code, fasrc/, nnef/scripts/, eval/decoys/*.py, eval/md_eval/*.py"
echo "  NOT pushed: runs/, params/, *.h5, eval data, figures, decoys data"
