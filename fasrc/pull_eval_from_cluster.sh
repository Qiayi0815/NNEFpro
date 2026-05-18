#!/usr/bin/env bash
# Run on your **laptop** from the nnef repo root.
# Pulls evaluation outputs from FASRC (summary.csv, plots, comparison.csv under eval/).
#
# Main outputs live in:
#   <repo>/eval/<run_basename>_casp14_3drobot/
# from evaluate_decoys.py / eval_one_run_casp14_3drobot.sh.
#
# Per-target decoy CSVs may also be under nnef/data/decoys/<set>/decoy_loss_<exp_tag>/;
# use --with-decoys only if you need those (larger transfer).
#
# Usage:
#   bash fasrc/pull_eval_from_cluster.sh
#   bash fasrc/pull_eval_from_cluster.sh --with-slurm-logs
#   bash fasrc/pull_eval_from_cluster.sh --with-decoys
#
# Override: REMOTE_HOST, REMOTE_REPO

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-qzha@login.rc.fas.harvard.edu}"
REMOTE_REPO="${REMOTE_REPO:-/n/home03/qzha/nnef}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_EVAL="${ROOT}/eval"
LOCAL_RUNS="${ROOT}/runs"
LOCAL_DATA_DECOYS="${ROOT}/nnef/data/decoys"

WITH_DECOYS=0
WITH_SLURM_LOGS=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-decoys) WITH_DECOYS=1 ;;
    --with-slurm-logs) WITH_SLURM_LOGS=1 ;;
    -h|--help)
      echo "usage: $0 [--with-decoys] [--with-slurm-logs]"
      exit 0
      ;;
    *)
      echo "unknown arg: $1" >&2
      exit 1
      ;;
  esac
  shift
done

mkdir -p "${LOCAL_EVAL}"

echo "==> rsync ${REMOTE_HOST}:${REMOTE_REPO}/eval/ -> ${LOCAL_EVAL}/"
rsync -avh --progress "${REMOTE_HOST}:${REMOTE_REPO}/eval/" "${LOCAL_EVAL}/"

if [[ "${WITH_SLURM_LOGS}" == "1" ]]; then
  mkdir -p "${LOCAL_RUNS}"
  echo "==> rsync slurm-eval-*.out from ${REMOTE_HOST}:${REMOTE_REPO}/runs/"
  rsync -avh --progress \
    --include='slurm-eval-*.out' \
    --include='slurm-eval-*.err' \
    --exclude='*' \
    "${REMOTE_HOST}:${REMOTE_REPO}/runs/" \
    "${LOCAL_RUNS}/"
fi

if [[ "${WITH_DECOYS}" == "1" ]]; then
  mkdir -p "${LOCAL_DATA_DECOYS}"
  echo "==> rsync ${REMOTE_HOST}:${REMOTE_REPO}/nnef/data/decoys/ (may be large)"
  rsync -avh --progress "${REMOTE_HOST}:${REMOTE_REPO}/nnef/data/decoys/" "${LOCAL_DATA_DECOYS}/"
fi

echo "Done. Local summaries: ${LOCAL_EVAL}/*/summary.csv"
