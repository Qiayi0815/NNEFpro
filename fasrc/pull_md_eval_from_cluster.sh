#!/usr/bin/env bash
# Run on your **laptop** from the nnef repo root.
# Pulls md_eval outputs from FASRC. Scopes narrowly to
#   eval/md_eval/       (full sweeps)
#   eval/md_eval_smoke/ (smoke-test outputs)
# so it does not re-sync the rest of eval/.
#
# Usage:
#   bash fasrc/pull_md_eval_from_cluster.sh
#   bash fasrc/pull_md_eval_from_cluster.sh --with-slurm-logs
#   bash fasrc/pull_md_eval_from_cluster.sh --smoke-only
#
# Override: REMOTE_HOST, REMOTE_REPO
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-qzha@login.rc.fas.harvard.edu}"
REMOTE_REPO="${REMOTE_REPO:-/n/home03/qzha/nnef}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_EVAL="${ROOT}/eval"
LOCAL_RUNS="${ROOT}/runs"

SMOKE_ONLY=0
WITH_SLURM_LOGS=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke-only) SMOKE_ONLY=1 ;;
    --with-slurm-logs) WITH_SLURM_LOGS=1 ;;
    -h|--help)
      echo "usage: $0 [--smoke-only] [--with-slurm-logs]"
      exit 0
      ;;
    *)
      echo "unknown arg: $1" >&2
      exit 1
      ;;
  esac
  shift
done

mkdir -p "${LOCAL_EVAL}/md_eval" "${LOCAL_EVAL}/md_eval_smoke"

echo "==> rsync ${REMOTE_HOST}:${REMOTE_REPO}/eval/md_eval_smoke/ -> ${LOCAL_EVAL}/md_eval_smoke/"
rsync -avh --progress \
  "${REMOTE_HOST}:${REMOTE_REPO}/eval/md_eval_smoke/" \
  "${LOCAL_EVAL}/md_eval_smoke/" || true

if [[ "${SMOKE_ONLY}" == "0" ]]; then
  echo "==> rsync ${REMOTE_HOST}:${REMOTE_REPO}/eval/md_eval/ -> ${LOCAL_EVAL}/md_eval/"
  rsync -avh --progress \
    "${REMOTE_HOST}:${REMOTE_REPO}/eval/md_eval/" \
    "${LOCAL_EVAL}/md_eval/" || true
fi

if [[ "${WITH_SLURM_LOGS}" == "1" ]]; then
  mkdir -p "${LOCAL_RUNS}"
  echo "==> rsync slurm-md-*.out from ${REMOTE_HOST}:${REMOTE_REPO}/runs/"
  rsync -avh --progress \
    --include='slurm-md-*.out' \
    --include='slurm-md-*.err' \
    --exclude='*' \
    "${REMOTE_HOST}:${REMOTE_REPO}/runs/" \
    "${LOCAL_RUNS}/"
fi

echo "Done. Local md_eval outputs: ${LOCAL_EVAL}/md_eval/<mode>/<run_base>/<target>_seed*/meta.json"
