#!/usr/bin/env bash
#SBATCH --job-name=nnef-md-yang-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=runs/slurm-md-yang-smoke-%j.out
#SBATCH --error=runs/slurm-md-yang-smoke-%j.out
#
# Smoke test for Yang-small MD reproduction: 1 model (yang_retrain) ×
# 1 protein (3KXT, 56 res) × short L=10000 × seed 0 × mode=native.
# Confirms the native_bead override path works before launching the full
# 45-task array in fasrc/md_eval_yang_small.slurm.
#
# Submit:
#   sbatch fasrc/md_eval_yang_smoke.slurm
#
# Override (different protein or model):
#   sbatch --export=ALL,MODEL_KEY=v1_rama,TARGET=1ZZK fasrc/md_eval_yang_smoke.slurm

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/nnef}"
DATA_DIR="${DATA_DIR:-$HOME/nnef_data}"

module purge
module load python/3.10.12-fasrc01

PYTHON_BIN="$(command -v python)"
CONDA_HOOK="$(dirname "$(dirname "$PYTHON_BIN")")/etc/profile.d/conda.sh"
[ -f "$CONDA_HOOK" ] || { echo "[md_yang_smoke] conda.sh not found at $CONDA_HOOK"; exit 1; }
# shellcheck disable=SC1090
source "$CONDA_HOOK"
set +u
conda activate "$ENV_PREFIX"
set -u

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

PY="${ENV_PREFIX}/bin/python"
[ -x "$PY" ] || { echo "[md_yang_smoke] ERROR: no $PY"; exit 1; }
"$PY" -s -c "import torch; print('[md_yang_smoke] cuda', torch.cuda.is_available())" || exit 1

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TOKENIZERS_PARALLELISM=false

cd "$SLURM_SUBMIT_DIR"
mkdir -p runs eval/md_eval

export REPO="$SLURM_SUBMIT_DIR"
export DATA_DIR="$DATA_DIR"
export DEVICE=cuda

# Smoke-test defaults (overridable via --export):
export MODEL_KEY="${MODEL_KEY:-yang_retrain}"
export TARGET="${TARGET:-3KXT}"
export MD_MODE="${MD_MODE:-native}"
export SEED="${SEED:-0}"
export L="${L:-10000}"
export TRJ_LOG="${TRJ_LOG:-50}"
export DECOY_SET="${DECOY_SET:-yang_small}"
export OUT_ROOT="${OUT_ROOT:-eval/md_eval_yang_smoke}"
export NATIVE_BEAD="${NATIVE_BEAD:-nnef/data/yang_small/${TARGET}/${TARGET}_bead.csv}"

if [[ ! -f "$NATIVE_BEAD" ]]; then
  echo "[md_yang_smoke] ERROR: bead CSV not found at $NATIVE_BEAD"
  exit 1
fi

echo "----- md_eval yang_small smoke test -----"
hostname
nvidia-smi
echo "MODEL_KEY=$MODEL_KEY TARGET=$TARGET NATIVE_BEAD=$NATIVE_BEAD L=$L"
echo "-----------------------------------------"

bash fasrc/md_eval_one.sh

echo "[md_yang_smoke] OK"
