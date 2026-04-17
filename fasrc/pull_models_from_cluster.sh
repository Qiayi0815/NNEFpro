#!/usr/bin/env bash
# Run on your **laptop** from the nnef repo root.
# Pulls checkpoints (and optional full run dir) from FASRC via rsync.
#
# Usage
# -----
#   # One experiment folder (recommended): weights live in runs/<exp>/models/
#   bash fasrc/pull_models_from_cluster.sh v2_dihedral_rama_v2_6228517
#
#   # Only *.pt files under that experiment (smaller than full tensorboard tree)
#   ONLY_PT=1 bash fasrc/pull_models_from_cluster.sh v3_full_rama_v2_6223467
#
#   # All of runs/ (large — includes events + many epochs)
#   bash fasrc/pull_models_from_cluster.sh --all-runs
#
# Override host/paths if needed:
#   REMOTE_HOST=user@login.rc.fas.harvard.edu REMOTE_REPO=/n/home03/user/nnef \
#     bash fasrc/pull_models_from_cluster.sh my_exp_id

REMOTE_HOST="${REMOTE_HOST:-qzha@login.rc.fas.harvard.edu}"
REMOTE_REPO="${REMOTE_REPO:-/n/home03/qzha/nnef}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_RUNS="${ROOT}/runs"

ONLY_PT="${ONLY_PT:-0}"

pull_exp() {
  local exp="$1"
  local src="${REMOTE_HOST}:${REMOTE_REPO}/runs/${exp}/"
  local dst="${LOCAL_RUNS}/${exp}/"
  mkdir -p "${LOCAL_RUNS}"
  echo "==> rsync ${src} -> ${dst}"
  if [[ "${ONLY_PT}" == "1" ]]; then
    mkdir -p "${dst}/models"
    rsync -avh --progress \
      "${REMOTE_HOST}:${REMOTE_REPO}/runs/${exp}/models/" \
      "${dst}/models/"
  else
    rsync -avh --progress "${src}" "${dst}"
  fi
}

if [[ "${1:-}" == '--all-runs' ]]; then
  mkdir -p "${LOCAL_RUNS}"
  echo "==> rsync full runs/ (may be large)"
  rsync -avh --progress "${REMOTE_HOST}:${REMOTE_REPO}/runs/" "${LOCAL_RUNS}/"
  exit 0
fi

if [[ -z "${1:-}" ]]; then
  echo "usage: $0 <exp_id_under_runs>   e.g. v2_dihedral_rama_v2_6228517"
  echo "       $0 --all-runs"
  exit 1
fi

pull_exp "$1"
echo "Done. Local: ${LOCAL_RUNS}/$1/models/"
