#!/usr/bin/env bash
#SBATCH --job-name=nnef-md-mode1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=runs/slurm-md-mode1-%A_%a.out
#SBATCH --error=runs/slurm-md-mode1-%A_%a.out
#SBATCH --array=0-35
#
# Mode 1 (folding from extended) over CASP14.
# DRAFT: do NOT submit until Mode 2 results have been reviewed.
#
#   models  = (yang_retrain, v1_rama, v1_rama_esm)
#   targets = (T1024, T1025, T1026, T1053)
#   seeds   = (0, 1, 2)
#
# Folding is much harder than native stability -- uses higher noise
# (beta ~ 3e-2) by default (see md_eval.py defaults_by_mode['fold']).
# Time budget is higher than Mode 2 accordingly.

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/nnef}"
DATA_DIR="${DATA_DIR:-$HOME/nnef_data}"

module purge
module load python/3.10.12-fasrc01

PYTHON_BIN="$(command -v python)"
CONDA_HOOK="$(dirname "$(dirname "$PYTHON_BIN")")/etc/profile.d/conda.sh"
[ -f "$CONDA_HOOK" ] || { echo "[md_mode1] conda.sh not found at $CONDA_HOOK"; exit 1; }
# shellcheck disable=SC1090
source "$CONDA_HOOK"
set +u
conda activate "$ENV_PREFIX"
set -u

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

PY="${ENV_PREFIX}/bin/python"
[ -x "$PY" ] || { echo "[md_mode1] ERROR: no $PY"; exit 1; }
"$PY" -s -c "import torch; print('[md_mode1] cuda', torch.cuda.is_available())" || exit 1

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TOKENIZERS_PARALLELISM=false

cd "$SLURM_SUBMIT_DIR"
mkdir -p runs eval/md_eval

MODELS=(yang_retrain v1_rama v1_rama_esm)
TARGETS=(T1024 T1025 T1026 T1053)
SEEDS=(0 1 2)

N_TARGETS=${#TARGETS[@]}
N_SEEDS=${#SEEDS[@]}

idx=${SLURM_ARRAY_TASK_ID:-0}
model_i=$(( idx / (N_TARGETS * N_SEEDS) ))
target_i=$(( (idx / N_SEEDS) % N_TARGETS ))
seed_i=$(( idx % N_SEEDS ))

export REPO="$SLURM_SUBMIT_DIR"
export DATA_DIR="$DATA_DIR"
export DEVICE=cuda
export MD_MODE=fold
export MODEL_KEY="${MODELS[$model_i]}"
export TARGET="${TARGETS[$target_i]}"
export SEED="${SEEDS[$seed_i]}"
export L="${L:-200000}"
export TRJ_LOG="${TRJ_LOG:-200}"
export OUT_ROOT="${OUT_ROOT:-eval/md_eval/mode1}"

echo "----- md_eval mode1 (folding) array task -----"
hostname
nvidia-smi
echo "ARRAY_ID=$SLURM_ARRAY_JOB_ID TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "MODEL=$MODEL_KEY TARGET=$TARGET SEED=$SEED L=$L"
echo "----------------------------------------------"

bash fasrc/md_eval_one.sh

echo "[md_mode1] task $SLURM_ARRAY_TASK_ID OK"
