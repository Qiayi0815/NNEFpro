#!/usr/bin/env bash
#SBATCH --job-name=nnef-eval-v3rama
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm-eval-v3-rama-v2-%j.out
#SBATCH --error=runs/slurm-eval-v3-rama-v2-%j.out
#
# v3_full + Rama v2 (same flags as train_v3_full.slurm). Default dir matches exp_id
# v3_full_rama_v2_<JOBID>. If training saved under v3_full_<JOBID> by mistake, rename:
#   mv runs/v3_full_6229240 runs/v3_full_rama_v2_6229240
#
#   sbatch --export=ALL,LOAD_EXP=runs/v3_full_rama_v2_6229240,DATA_DIR=$HOME/nnef_data fasrc/eval_v3_full_rama_v2_casp14_3drobot.slurm

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/nnef}"
DATA_DIR="${DATA_DIR:-$HOME/nnef_data}"

module purge
module load python/3.10.12-fasrc01

PYTHON_BIN="$(command -v python)"
CONDA_HOOK="$(dirname "$(dirname "$PYTHON_BIN")")/etc/profile.d/conda.sh"
[ -f "$CONDA_HOOK" ] || { echo "[eval_slurm] conda.sh not found"; exit 1; }
# shellcheck disable=SC1090
source "$CONDA_HOOK"
set +u
conda activate "$ENV_PREFIX"
set -u

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

PY="${ENV_PREFIX}/bin/python"
[ -x "$PY" ] || { echo "[eval_slurm] ERROR: no $PY"; exit 1; }
"$PY" -s -c "import torch; print('[eval_slurm] cuda', torch.cuda.is_available())" || exit 1

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TOKENIZERS_PARALLELISM=false

export EVAL_MODE=v3_full
export DATA_DIR="$DATA_DIR"
export ESM_H5="${ESM_H5:-$DATA_DIR/hhsuite_esm_v2.h5}"

export LOAD_EXP="${LOAD_EXP:-runs/v3_full_rama_v2_6229240}"

echo "----- v3_full_rama_v2 eval -----"
echo "LOAD_EXP=$LOAD_EXP ESM_H5=$ESM_H5"
hostname
nvidia-smi

cd "$SLURM_SUBMIT_DIR"
mkdir -p runs eval

bash fasrc/eval_one_run_casp14_3drobot.sh
echo "[eval_slurm] OK"
