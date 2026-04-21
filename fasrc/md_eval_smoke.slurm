#!/usr/bin/env bash
#SBATCH --job-name=nnef-md-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=runs/slurm-md-smoke-%j.out
#SBATCH --error=runs/slurm-md-smoke-%j.out
#
# Smoke test for Langevin-MD evaluation: 1 model (v1_rama) × 1 target (T1053)
# × short L=10000 × 1 seed × mode=native. Confirms the md_eval.py driver
# produces valid PDB + CSV outputs on the cluster before launching the
# full array in fasrc/md_eval_mode2_casp14.slurm.
#
# Submit:
#   sbatch fasrc/md_eval_smoke.slurm
#
# Override (e.g. different model or target):
#   sbatch --export=ALL,MODEL_KEY=yang_retrain,TARGET=T1024 fasrc/md_eval_smoke.slurm

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/nnef}"
DATA_DIR="${DATA_DIR:-$HOME/nnef_data}"

module purge
module load python/3.10.12-fasrc01

PYTHON_BIN="$(command -v python)"
CONDA_HOOK="$(dirname "$(dirname "$PYTHON_BIN")")/etc/profile.d/conda.sh"
[ -f "$CONDA_HOOK" ] || { echo "[md_smoke] conda.sh not found at $CONDA_HOOK"; exit 1; }
# shellcheck disable=SC1090
source "$CONDA_HOOK"
set +u
conda activate "$ENV_PREFIX"
set -u

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

PY="${ENV_PREFIX}/bin/python"
[ -x "$PY" ] || { echo "[md_smoke] ERROR: no $PY"; exit 1; }
"$PY" -s -c "import torch; print('[md_smoke] cuda', torch.cuda.is_available())" || exit 1

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TOKENIZERS_PARALLELISM=false

cd "$SLURM_SUBMIT_DIR"
mkdir -p runs eval/md_eval

export REPO="$SLURM_SUBMIT_DIR"
export DATA_DIR="$DATA_DIR"
export DEVICE=cuda

# Smoke-test defaults (overridable via --export):
export MODEL_KEY="${MODEL_KEY:-v1_rama}"
export TARGET="${TARGET:-T1053}"
export MD_MODE="${MD_MODE:-native}"
export SEED="${SEED:-0}"
export L="${L:-10000}"
export TRJ_LOG="${TRJ_LOG:-50}"
export X_TYPE="${X_TYPE:-mix_fast}"
export OUT_ROOT="${OUT_ROOT:-eval/md_eval_smoke}"

echo "----- md_eval smoke test -----"
hostname
nvidia-smi
echo "MODEL_KEY=$MODEL_KEY TARGET=$TARGET MD_MODE=$MD_MODE L=$L"
echo "------------------------------"

bash fasrc/md_eval_one.sh

echo "[md_smoke] OK"
