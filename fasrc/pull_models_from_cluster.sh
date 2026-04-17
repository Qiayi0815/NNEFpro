#!/usr/bin/env bash
# Run on your **laptop** from the nnef repo root.
# Pulls checkpoints from FASRC via ssh + rsync.
#
# Usage
# -----
#   # One experiment (full run dir: tensorboard, logs, …)
#   bash fasrc/pull_models_from_cluster.sh v2_dihedral_rama_v2_6228517
#
#   # One experiment: only models/*.pt
#   ONLY_PT=1 bash fasrc/pull_models_from_cluster.sh v3_full_rama_v2_6223467
#
#   # Every experiment under runs/: only models/model.pt (latest weights each job)
#   bash fasrc/pull_models_from_cluster.sh --all-final
#
#   # Every experiment: entire models/ (model.pt + model_epoch_*.pt, etc.)
#   bash fasrc/pull_models_from_cluster.sh --all-models
#
#   # Entire runs/ tree (very large)
#   bash fasrc/pull_models_from_cluster.sh --all-runs
#
# Override: REMOTE_HOST, REMOTE_REPO

set -euo pipefail

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

# Prints one exp dir per line on stdout; stderr for errors. Returns 1 if ssh/path fails.
# Note: do NOT use BatchMode=yes here — it breaks password / keyboard-interactive logins.
list_remote_exp_dirs() {
  if ! ssh -o ConnectTimeout=20 "${REMOTE_HOST}" "test -d '${REMOTE_REPO}/runs'"; then
    echo "[pull_models] ERROR: cannot reach ${REMOTE_HOST} or ${REMOTE_REPO}/runs does not exist." >&2
    echo "            Try: ssh ${REMOTE_HOST} 'ls ${REMOTE_REPO}/runs'" >&2
    return 1
  fi
  ssh "${REMOTE_HOST}" "cd '${REMOTE_REPO}/runs' 2>/dev/null && ls -d */ 2>/dev/null | sed 's|/||'"
}

pull_all_final() {
  mkdir -p "${LOCAL_RUNS}"
  echo "==> Pull models/model.pt for each run under ${REMOTE_HOST}:${REMOTE_REPO}/runs/"
  local exp_list
  if ! exp_list="$(list_remote_exp_dirs)"; then
    exit 1
  fi
  local n=0 ok=0
  while IFS= read -r exp; do
    [[ -z "${exp}" ]] && continue
    n=$((n + 1))
    mkdir -p "${LOCAL_RUNS}/${exp}/models"
    if rsync -avh --progress \
      "${REMOTE_HOST}:${REMOTE_REPO}/runs/${exp}/models/model.pt" \
      "${LOCAL_RUNS}/${exp}/models/"; then
      ok=$((ok + 1))
    else
      echo "[pull_models] WARN: missing or failed: ${exp}/models/model.pt"
    fi
  done <<< "${exp_list}"
  if [[ "${n}" -eq 0 ]]; then
    echo "[pull_models] ERROR: no experiment directories under runs/."
    exit 1
  fi
  echo "==> Done: ${ok}/${n} model.pt files synced (see warnings for skips)."
}

pull_all_models_dir() {
  mkdir -p "${LOCAL_RUNS}"
  echo "==> Pull entire models/ for each run (all .pt snapshots)"
  local exp_list
  if ! exp_list="$(list_remote_exp_dirs)"; then
    exit 1
  fi
  local n=0
  while IFS= read -r exp; do
    [[ -z "${exp}" ]] && continue
    n=$((n + 1))
    mkdir -p "${LOCAL_RUNS}/${exp}/models"
    rsync -avh --progress \
      "${REMOTE_HOST}:${REMOTE_REPO}/runs/${exp}/models/" \
      "${LOCAL_RUNS}/${exp}/models/" || echo "[pull_models] WARN: ${exp}/models/ rsync failed"
  done <<< "${exp_list}"
  if [[ "${n}" -eq 0 ]]; then
    echo "[pull_models] ERROR: no experiment directories under runs/."
    exit 1
  fi
  echo "==> Done: processed ${n} experiment dirs."
}

case "${1:-}" in
  --all-runs)
    mkdir -p "${LOCAL_RUNS}"
    echo "==> rsync full runs/ (may be very large)"
    rsync -avh --progress "${REMOTE_HOST}:${REMOTE_REPO}/runs/" "${LOCAL_RUNS}/"
    ;;
  --all-final)
    pull_all_final
    ;;
  --all-models)
    pull_all_models_dir
    ;;
  '')
    echo "usage: $0 <exp_id_under_runs>"
    echo "       $0 --all-final      # every runs/*/models/model.pt"
    echo "       $0 --all-models    # every runs/*/models/ (all checkpoints)"
    echo "       $0 --all-runs      # entire runs/"
    exit 1
    ;;
  *)
    pull_exp "$1"
    echo "Done. Local: ${LOCAL_RUNS}/$1/models/"
    ;;
esac
