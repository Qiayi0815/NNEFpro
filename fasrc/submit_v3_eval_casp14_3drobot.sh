#!/usr/bin/env bash
# Run **on the Cannon login / submit node** from the nnef repo root (after rsync code).
# Submits **two parallel** GPU jobs: v3_full (no-rama name) + v3_full_rama_v2.
# Each runs CASP14 + 3DRobot_set via fasrc/eval_one_run_casp14_3drobot.sh (EVAL_MODE=v3_full).
#
# Needs on cluster: runs/<...>/models/model.pt, $DATA_DIR/hhsuite_esm_v2.h5
#
# Usage:
#   ssh qzha@login.rc.fas.harvard.edu
#   cd ~/nnef
#   bash fasrc/submit_v3_eval_casp14_3drobot.sh
#
# Override run dirs:
#   V3_NO_RAMA_RUN=runs/v3_full_6223467 V3_RAMA_RUN=runs/v3_full_rama_v2_6229240 \\
#     DATA_DIR=$HOME/nnef_data bash fasrc/submit_v3_eval_casp14_3drobot.sh

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

DATA_DIR="${DATA_DIR:-$HOME/nnef_data}"
export DATA_DIR

V3_NO_RAMA_RUN="${V3_NO_RAMA_RUN:-runs/v3_full_6223467}"
V3_RAMA_RUN="${V3_RAMA_RUN:-runs/v3_full_rama_v2_6229240}"

for path in "$V3_NO_RAMA_RUN/models/model.pt" "$V3_RAMA_RUN/models/model.pt"; do
  if [[ ! -f "$path" ]]; then
    echo "[submit_v3] ERROR: missing $path"
    exit 1
  fi
done

if [[ ! -f "$DATA_DIR/hhsuite_esm_v2.h5" ]]; then
  echo "[submit_v3] ERROR: ESM cache not found: $DATA_DIR/hhsuite_esm_v2.h5"
  echo "            Run precompute on cluster first: sbatch fasrc/precompute_esm.slurm"
  exit 1
fi

echo "[submit_v3] DATA_DIR=$DATA_DIR"
echo "[submit_v3] v3 (no rama in name): $V3_NO_RAMA_RUN"
echo "[submit_v3] v3 + Rama v2:        $V3_RAMA_RUN"

J1="$(sbatch --parsable --export=ALL,LOAD_EXP="$V3_NO_RAMA_RUN",DATA_DIR="$DATA_DIR" fasrc/eval_v3_full_casp14_3drobot.slurm)"
J2="$(sbatch --parsable --export=ALL,LOAD_EXP="$V3_RAMA_RUN",DATA_DIR="$DATA_DIR" fasrc/eval_v3_full_rama_v2_casp14_3drobot.slurm)"

echo "[submit_v3] submitted: nnef-eval-v3  job_id=$J1  -> runs/slurm-eval-v3-full-${J1}.out"
echo "[submit_v3] submitted: nnef-eval-v3rama job_id=$J2  -> runs/slurm-eval-v3-rama-v2-${J2}.out"
echo "[submit_v3] watch: squeue -u \"\$USER\""
echo "[submit_v3] logs when done: eval/${V3_NO_RAMA_RUN#runs/}_casp14_3drobot/ eval/${V3_RAMA_RUN#runs/}_casp14_3drobot/"
