#!/usr/bin/env bash
# Pull ``runs/yang_retrain_<SLURM_JOB_ID>/`` from FASRC into local ``$REPO/runs/``.
#
# Usage (from repo root):
#   bash fasrc/pull_yang_retrain_from_cluster.sh 6234567
#   ONLY_PT=1 bash fasrc/pull_yang_retrain_from_cluster.sh 6234567   # models/*.pt only
#
# List remote retrains (no download):
#   bash fasrc/pull_yang_retrain_from_cluster.sh --list
#
# Same REMOTE_HOST / REMOTE_REPO as fasrc/pull_models_from_cluster.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-qzha@login.rc.fas.harvard.edu}"
REMOTE_REPO="${REMOTE_REPO:-/n/home03/qzha/nnef}"

if [[ "${1:-}" == '-h' || "${1:-}" == '--help' ]]; then
  echo "Usage: $0 [--list | SLURM_JOB_ID]"
  echo "  Pulls runs/yang_retrain_<JOBID>/ from ${REMOTE_HOST}:${REMOTE_REPO}"
  exit 0
fi

if [[ "${1:-}" == '--list' ]]; then
  echo "Remote yang_retrain_* under ${REMOTE_HOST}:${REMOTE_REPO}/runs/:"
  ssh -o ConnectTimeout=25 "${REMOTE_HOST}" \
    "cd '${REMOTE_REPO}/runs' 2>/dev/null && ls -d yang_retrain_*/ 2>/dev/null | sed 's|/||' | sort -V" \
    || {
      echo "[pull_yang_retrain] ERROR: ssh or path failed." >&2
      exit 1
    }
  exit 0
fi

if [[ -z "${1:-}" || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 [--list | SLURM_JOB_ID]" >&2
  echo "  example: ONLY_PT=1 $0 6234567" >&2
  exit 1
fi

exec bash "${ROOT}/fasrc/pull_models_from_cluster.sh" "yang_retrain_$1"
