#!/usr/bin/env bash
#SBATCH --job-name=nnef-eval-v2drama
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm-eval-v2-dihedral-rama-%j.out
#SBATCH --error=runs/slurm-eval-v2-dihedral-rama-%j.out
#
# v2_dihedral + Rama v2 (train_v2_dihedral.slurm). Parallel-safe vs other eval jobs.
#
# Default run dir; override if your JOBID differs:
#   sbatch --export=ALL,LOAD_EXP=runs/v2_dihedral_rama_v2_6228517 fasrc/eval_v2_dihedral_rama_casp14_3drobot.slurm

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

export EVAL_MODE=v2_dihedral
export LOAD_EXP="${LOAD_EXP:-runs/v2_dihedral_rama_v2_6228517}"
export DATA_DIR="$DATA_DIR"

echo "----- v2_dihedral_rama eval -----"
echo "LOAD_EXP=$LOAD_EXP"
hostname
nvidia-smi

cd "$SLURM_SUBMIT_DIR"
mkdir -p runs eval

bash fasrc/eval_one_run_casp14_3drobot.sh
echo "[eval_slurm] OK"
