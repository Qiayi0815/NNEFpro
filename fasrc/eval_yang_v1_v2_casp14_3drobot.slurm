#!/usr/bin/env bash
#SBATCH --job-name=nnef-eval-y1v2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm-eval-yang-v1-v2-%j.out
#SBATCH --error=runs/slurm-eval-yang-v1-v2-%j.out
#
# Batch driver for: Yang exp1 + v1_pure + v2_run [+ optional v3_full] on CASP14 + 3DRobot_set
# (see fasrc/eval_yang_v1_v2_casp14_3drobot.sh).
#
# Submit from repo root (directory containing nnef/ and fasrc/):
#   cd ~/nnef
#   sbatch fasrc/eval_yang_v1_v2_casp14_3drobot.slurm
#
# Override run dirs / checkpoint (optional):
#   V1_RUN=runs/v1_pure_6171704 V2_RUN=runs/v2_run_6160264 sbatch fasrc/...
#   YANG_CKPT=$HOME/nnef/params/exp1/models/model.pt sbatch fasrc/...
#
# After v3_full training finishes (runs/v3_full_rama_v2_<JOBID>/models/model.pt exists):
#   V3_RUN=runs/v3_full_rama_v2_6223467 DATA_DIR=$HOME/nnef_data sbatch fasrc/eval_yang_v1_v2_casp14_3drobot.slurm
# Requires $DATA_DIR/hhsuite_esm_v2.h5 (same as train_v3_full.slurm).
#
# Logs: runs/slurm-eval-yang-v1-v2-<JOBID>.out (stdout+stderr; PYTHONUNBUFFERED set before import torch).
# ----------------------------------------------------------------------------

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/nnef}"

module purge
module load python/3.10.12-fasrc01

PYTHON_BIN="$(command -v python)"
CONDA_HOOK="$(dirname "$(dirname "$PYTHON_BIN")")/etc/profile.d/conda.sh"
[ -f "$CONDA_HOOK" ] || { echo "[eval_slurm] conda.sh not found at $CONDA_HOOK"; exit 1; }
# shellcheck disable=SC1090
source "$CONDA_HOOK"
set +u
conda activate "$ENV_PREFIX"
set -u

# Before any ``import torch``: ignore ~/.local site-packages (breaks many clusters).
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

PY="${ENV_PREFIX}/bin/python"
[ -x "$PY" ] || { echo "[eval_slurm] ERROR: no $PY"; exit 1; }

# -s: omit user site-packages (~/.local); avoids libquadmath / numpy shadowing torch.
if ! "$PY" -s -c "import torch; print('[eval_slurm] torch', torch.__version__, 'cuda', torch.cuda.is_available())"; then
  echo "[eval_slurm] ERROR: import torch failed. PY=$PY"
  "$PY" -s -c "import torch" 2>&1 || true
  exit 1
fi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TOKENIZERS_PARALLELISM=false

echo "----- nnef eval Yang + v1 + v2 [+ v3] -----"
hostname
nvidia-smi
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "ENV_PREFIX=$ENV_PREFIX"
echo "PY=$PY"
echo "V1_RUN=${V1_RUN:-runs/v1_pure_6171704}"
echo "V2_RUN=${V2_RUN:-runs/v2_run_6160264}"
echo "V3_RUN=${V3_RUN:-}"
echo "DATA_DIR=${DATA_DIR:-$HOME/nnef_data}"
echo "------------------------------------"

cd "$SLURM_SUBMIT_DIR"
mkdir -p runs eval

# CONDA_PREFIX is set → eval_yang_v1_v2 script picks ENV python. Optional: PY=... V1_RUN=... V2_RUN=...
bash fasrc/eval_yang_v1_v2_casp14_3drobot.sh

echo "[eval_slurm] OK"
