#!/usr/bin/env bash
# Run **on the Cannon login node** from ~/nnef. Submits **two parallel** GPU jobs:
# v3_full + v3_full_rama_v2, **CASP14 decoys only** (no 3DRobot_set).
#
# Outputs: eval/<run_basename>_casp14_only/summary.csv (+ plots/, decoy_loss under nnef/data/decoys/casp14/...)
#
# Usage:
#   cd ~/nnef
#   bash fasrc/submit_v3_casp14_only.sh
#
# Override:
#   V3_NO_RAMA_RUN=runs/v3_full_6223467 V3_RAMA_RUN=runs/v3_full_rama_v2_6229240 \\
#     DATA_DIR=$HOME/nnef_data bash fasrc/submit_v3_casp14_only.sh

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

DATA_DIR="${DATA_DIR:-$HOME/nnef_data}"
export DATA_DIR

V3_NO_RAMA_RUN="${V3_NO_RAMA_RUN:-runs/v3_full_6223467}"
V3_RAMA_RUN="${V3_RAMA_RUN:-runs/v3_full_rama_v2_6229240}"

for path in "$V3_NO_RAMA_RUN/models/model.pt" "$V3_RAMA_RUN/models/model.pt"; do
  if [[ ! -f "$path" ]]; then
    echo "[submit_v3_c14] ERROR: missing $path"
    exit 1
  fi
done

if [[ ! -f "$DATA_DIR/hhsuite_esm_v2.h5" ]]; then
  echo "[submit_v3_c14] ERROR: ESM cache not found: $DATA_DIR/hhsuite_esm_v2.h5"
  exit 1
fi

echo "[submit_v3_c14] DATA_DIR=$DATA_DIR"
echo "[submit_v3_c14] LOAD_EXP (1) $V3_NO_RAMA_RUN"
echo "[submit_v3_c14] LOAD_EXP (2) $V3_RAMA_RUN"

J1="$(sbatch --parsable --export=ALL,LOAD_EXP="$V3_NO_RAMA_RUN",DATA_DIR="$DATA_DIR" fasrc/eval_v3_full_casp14_only.slurm)"
J2="$(sbatch --parsable --export=ALL,LOAD_EXP="$V3_RAMA_RUN",DATA_DIR="$DATA_DIR" fasrc/eval_v3_full_rama_v2_casp14_only.slurm)"

echo "[submit_v3_c14] job_ids: $J1 -> runs/slurm-eval-v3-full-c14only-${J1}.out"
echo "[submit_v3_c14]           $J2 -> runs/slurm-eval-v3-rama-c14only-${J2}.out"
echo "[submit_v3_c14] eval dirs: eval/${V3_NO_RAMA_RUN#runs/}_casp14_only/  eval/${V3_RAMA_RUN#runs/}_casp14_only/"
