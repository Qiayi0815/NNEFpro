#!/usr/bin/env bash
# Local: full v3 eval (**CASP14 + 3DRobot_set**) for both checkpoints (same as cluster Slurm).
#
# Needs:
#   runs/v3_full_6223467/models/model.pt
#   runs/v3_full_rama_v2_6229240/models/model.pt
#   $DATA_DIR/hhsuite_esm_v2.h5  (large; rsync from cluster nnef_data if missing)
#
# Usage (from repo root, conda env with torch+h5py):
#   bash fasrc/eval_v3_pair_casp14_3drobot_local.sh
#
# Mac (Apple Silicon):
#   DEVICE=mps DATA_DIR=/Library/Camille/FYP/nnef_data bash fasrc/eval_v3_pair_casp14_3drobot_local.sh
#
# Linux + NVIDIA:
#   DEVICE=cuda DATA_DIR=$HOME/nnef_data bash fasrc/eval_v3_pair_casp14_3drobot_local.sh
#
# CPU only (very slow):
#   DEVICE=cpu bash fasrc/eval_v3_pair_casp14_3drobot_local.sh
#
# Env:
#   REPO          default: directory containing this script/..
#   DATA_DIR      default: $HOME/nnef_data
#   ESM_H5        default: $DATA_DIR/hhsuite_esm_v2.h5  (with fallbacks like eval_casp14_all_local.sh)
#   V3_RUN        default: runs/v3_full_6223467
#   V3_RAMA_RUN   default: runs/v3_full_rama_v2_6229240
#   ONLY=1|2      run only first or second model (optional)

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
export REPO
cd "$REPO"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

DEVICE="${DEVICE:-cuda}"
DATA_DIR="${DATA_DIR:-$HOME/nnef_data}"
export DATA_DIR

ESM_H5="${ESM_H5:-$DATA_DIR/hhsuite_esm_v2.h5}"
if [[ ! -f "$ESM_H5" ]]; then
  for _try in \
    "${DATA_DIR}/hhsuite_esm_v2.h5" \
    "$(dirname "$REPO")/nnef_data/hhsuite_esm_v2.h5" \
    "${REPO}/../nnef_data/hhsuite_esm_v2.h5" \
    "${REPO}/nnef/data/hhsuite_esm_v2.h5"
  do
    if [[ -f "$_try" ]]; then
      ESM_H5="$_try"
      echo "[eval_v3_pair] Resolved ESM_H5=$ESM_H5"
      break
    fi
  done
fi
export ESM_H5

V3_RUN="${V3_RUN:-runs/v3_full_6223467}"
V3_RAMA_RUN="${V3_RAMA_RUN:-runs/v3_full_rama_v2_6229240}"

ONLY="${ONLY:-}"

_run_one() {
  local load="$1"
  EVAL_MODE=v3_full LOAD_EXP="$load" DEVICE="$DEVICE" \
    bash fasrc/eval_one_run_casp14_3drobot.sh
}

echo "[eval_v3_pair] REPO=$REPO DEVICE=$DEVICE DATA_DIR=$DATA_DIR ESM_H5=$ESM_H5"

if [[ "$ONLY" == "2" ]]; then
  _run_one "$V3_RAMA_RUN"
elif [[ "$ONLY" == "1" ]]; then
  _run_one "$V3_RUN"
else
  _run_one "$V3_RUN"
  _run_one "$V3_RAMA_RUN"
fi

echo "[eval_v3_pair] Done. Outputs: eval/$(basename "$V3_RUN")_casp14_3drobot/  eval/$(basename "$V3_RAMA_RUN")_casp14_3drobot/"
