#!/usr/bin/env bash
#SBATCH --job-name=nnef-eval-v1esm-3dr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=runs/slurm-eval-v1esm-3dr-%j.out
#SBATCH --error=runs/slurm-eval-v1esm-3dr-%j.err
#
# 3DRobot_set eval for v1_esm_rama_v2_6642146 on cluster GPU.
# CASP14 side was already scored locally on MPS — skip it here.
#
# Submit:  sbatch fasrc/eval_v1_esm_3drobot_cluster.slurm

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/nnef}"
DATA_DIR="${DATA_DIR:-$HOME/nnef_data}"

module purge
module load python/3.10.12-fasrc01

PYTHON_BIN="$(command -v python)"
CONDA_HOOK="$(dirname "$(dirname "$PYTHON_BIN")")/etc/profile.d/conda.sh"
[ -f "$CONDA_HOOK" ] || { echo "[eval_v1esm_3dr] conda.sh not found at $CONDA_HOOK"; exit 1; }
# shellcheck disable=SC1090
source "$CONDA_HOOK"
set +u
conda activate "$ENV_PREFIX"
set -u

cd "$SLURM_SUBMIT_DIR"

export CONDA_PREFIX="$ENV_PREFIX"
export DEVICE=cuda
export EVAL_MODE=v1_esm
export LOAD_EXP=runs/v1_esm_rama_v2_6642146
export DECOY_SETS=3DRobot_set
export DATA_DIR="$DATA_DIR"
export OUT_DIR=eval/v1_esm_rama_v2_6642146_3drobot_cluster
export TAG=v1_esm_rama_v2_6642146
export REPO="$SLURM_SUBMIT_DIR"
export SKIP_EXISTING=1

bash fasrc/eval_one_run_casp14_3drobot.sh

echo "[eval_v1esm_3dr] finished with exit code $?"
