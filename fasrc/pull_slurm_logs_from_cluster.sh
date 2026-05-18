#!/usr/bin/env bash
# Run on your **laptop** from the nnef repo root.
# Training stdout/stderr live at runs/slurm-<JOBID>.{out,err} on the cluster (not inside
# runs/<exp_id>/). Use this after pull_models_from_cluster.sh so you have both checkpoints
# and Slurm text logs for grep / plotting.
#
# Usage:
#   bash fasrc/pull_slurm_logs_from_cluster.sh 6223467 6228517 6229240
#
# Override: REMOTE_HOST, REMOTE_REPO

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-qzha@login.rc.fas.harvard.edu}"
REMOTE_REPO="${REMOTE_REPO:-/n/home03/qzha/nnef}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_RUNS="${ROOT}/runs"
mkdir -p "${LOCAL_RUNS}"

if [[ $# -eq 0 ]]; then
  echo "usage: $0 <slurm_job_id> [<slurm_job_id> ...]"
  echo "  Pulls runs/slurm-NNNN.out and runs/slurm-NNNN.err (missing files are skipped with a warning)."
  exit 1
fi

for j in "$@"; do
  for ext in out err; do
    f="slurm-${j}.${ext}"
    src="${REMOTE_HOST}:${REMOTE_REPO}/runs/${f}"
    if rsync -avh --progress "${src}" "${LOCAL_RUNS}/" 2>/dev/null; then
      :
    else
      echo "[pull_slurm_logs] WARN: missing or failed: ${f}"
    fi
  done
done

echo "Done. Local: ${LOCAL_RUNS}/slurm-<jobid>.{out,err}"
