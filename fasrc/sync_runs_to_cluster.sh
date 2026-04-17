#!/usr/bin/env bash
# Laptop → FASRC: copy finished run folders (checkpoints) needed for decoy eval.
# Default code rsync excludes runs/; use this after pulling model.pt locally or when
# cluster ~/nnef/runs is missing those experiments.
#
# Usage:
#   bash fasrc/sync_runs_to_cluster.sh
#   bash fasrc/sync_runs_to_cluster.sh exp1 v1_pure_6171704 v2_run_6160264
#
# Each name is a directory under ./runs/<name>/models/model.pt

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-qzha@login.rc.fas.harvard.edu}"
REMOTE_REPO="${REMOTE_REPO:-/n/home03/qzha/nnef}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ $# -gt 0 ]]; then
  NAMES=("$@")
else
  NAMES=(exp1 v1_pure_6171704 v2_dihedral_6172503 v2_run_6160264)
fi

for n in "${NAMES[@]}"; do
  if [[ ! -f "runs/${n}/models/model.pt" ]]; then
    echo "WARN: skip ${n} (no runs/${n}/models/model.pt)" >&2
    continue
  fi
  echo "==> rsync runs/${n} -> ${REMOTE_HOST}:${REMOTE_REPO}/runs/"
  rsync -avh --progress "runs/${n}/" "${REMOTE_HOST}:${REMOTE_REPO}/runs/${n}/"
done

echo "Done."
