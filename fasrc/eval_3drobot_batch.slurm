#!/usr/bin/env bash
#SBATCH --job-name=nnef-3dr-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=runs/slurm-3dr-eval-%j.out
#SBATCH --error=runs/slurm-3dr-eval-%j.err
#
# Batch 3DRobot_set scoring for every runs/*/models/model.pt (CUDA).
# Avoids Apple MPS crashes on long decoy jobs.
#
# Prereqs on cluster:
#   - fasrc/env_setup.sh conda env
#   - ~/nnef_data: hhsuite_CB_v2.h5, seq, rama (training data; decoy script may not open them)
#   - ~/nnef/nnef/data/decoys/3DRobot_set/ (bash fasrc/sync_3drobot_decoys_to_cluster.sh from laptop)
#   - ~/nnef/runs/<exp>/models/model.pt   (train there or bash fasrc/sync_runs_to_cluster.sh)
#
# Submit:
#   cd ~/nnef && sbatch fasrc/eval_3drobot_batch.slurm
#
# Smoke (pass env into the job; plain EXTRA_ARGS=... may be dropped by Slurm):
#   sbatch --export=ALL,EXTRA_ARGS='--targets 1C5EA' fasrc/eval_3drobot_batch.slurm

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/nnef}"
DATA_DIR="${DATA_DIR:-$HOME/nnef_data}"

module purge
module load python/3.10.12-fasrc01

PYTHON_BIN="$(command -v python)"
CONDA_HOOK="$(dirname "$(dirname "$PYTHON_BIN")")/etc/profile.d/conda.sh"
[ -f "$CONDA_HOOK" ] || { echo "[eval_3drobot_batch] conda.sh not found at $CONDA_HOOK"; exit 1; }
# shellcheck disable=SC1090
source "$CONDA_HOOK"
set +u
conda activate "$ENV_PREFIX"
set -u

PY="${ENV_PREFIX}/bin/python"
[ -x "$PY" ] || { echo "[eval_3drobot_batch] no $PY"; exit 1; }

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

if [[ -f "$DATA_DIR/hhsuite_esm_v2.h5" ]]; then
  export NNEF_ESM_H5="$DATA_DIR/hhsuite_esm_v2.h5"
fi

cd "$SLURM_SUBMIT_DIR"
mkdir -p runs eval

echo "----- 3DRobot batch eval -----"
hostname
nvidia-smi || true
echo "ENV_PREFIX=$ENV_PREFIX"
echo "DATA_DIR=$DATA_DIR"
echo "NNEF_ESM_H5=${NNEF_ESM_H5:-}"
echo "EXTRA_ARGS=${EXTRA_ARGS:-}"
echo "------------------------------"

# EXTRA_ARGS: optional, e.g. EXTRA_ARGS='--targets 1C5EA' for smoke
# shellcheck disable=SC2086
"$PY" -u nnef/scripts/batch_eval_3drobot.py --device cuda ${EXTRA_ARGS:-}

echo "[eval_3drobot_batch] finished with exit code $?"
