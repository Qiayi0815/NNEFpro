#!/usr/bin/env bash
#SBATCH --job-name=nnef-eval-v1rama
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm-eval-v1-rama-v2-%j.out
#SBATCH --error=runs/slurm-eval-v1-rama-v2-%j.out
#
# v1_pure + Rama v2 checkpoint only (unique exp_tag / out_dir — safe vs nnef-eval-y1v2).
#
# Avoid GPU contention with your main eval job (e.g. JOBID 6259187): chain this AFTER it:
#   sbatch --dependency=afterok:6259187 fasrc/eval_v1_pure_rama_v2_only.slurm
#
# If the other job is already done, submit without dependency:
#   sbatch fasrc/eval_v1_pure_rama_v2_only.slurm
#
# Override run dir (must match Slurm training exp_id):
#   sbatch --export=ALL,V1_RAMA_RUN=runs/v1_pure_rama_v2_6228201 fasrc/eval_v1_pure_rama_v2_only.slurm

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/nnef}"

module purge
module load python/3.10.12-fasrc01

PYTHON_BIN="$(command -v python)"
CONDA_HOOK="$(dirname "$(dirname "$PYTHON_BIN")")/etc/profile.d/conda.sh"
[ -f "$CONDA_HOOK" ] || { echo "[eval_v1_rama_slurm] conda.sh not found at $CONDA_HOOK"; exit 1; }
# shellcheck disable=SC1090
source "$CONDA_HOOK"
set +u
conda activate "$ENV_PREFIX"
set -u

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

PY="${ENV_PREFIX}/bin/python"
[ -x "$PY" ] || { echo "[eval_v1_rama_slurm] ERROR: no $PY"; exit 1; }

if ! "$PY" -s -c "import torch; print('[eval_v1_rama_slurm] torch', torch.__version__, 'cuda', torch.cuda.is_available())"; then
  exit 1
fi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TOKENIZERS_PARALLELISM=false

echo "----- v1_pure Rama v2 only eval -----"
hostname
nvidia-smi
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "V1_RAMA_RUN=${V1_RAMA_RUN:-runs/v1_pure_rama_v2_6228201}"
echo "--------------------------------------"

cd "$SLURM_SUBMIT_DIR"
mkdir -p runs eval

bash fasrc/eval_v1_pure_rama_v2_only.sh

echo "[eval_v1_rama_slurm] OK"
