#!/usr/bin/env bash
#SBATCH --job-name=nnef-md-yang-small
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=runs/slurm-md-yang-small-%A_%a.out
#SBATCH --error=runs/slurm-md-yang-small-%A_%a.out
#SBATCH --array=0-44
#
# Mode 2 (native-basin stability MD) over Yang et al. 2022's small-protein
# benchmark, restricted to 5 representative proteins that span the size
# range (56 - 289 residues):
#
#   models   = (yang_retrain, v1_rama, v1_rama_esm)
#   proteins = (3KXT, 1ZZK, 2MCM, 6H8O, 4M1X)
#   seeds    = (0, 1, 2)
#   => 3 * 5 * 3 = 45 array tasks (indices 0..44).
#
# Bead CSVs live at nnef/data/yang_small/<PDB>/<PDB>_bead.csv (used as
# both RMSD reference and MD starting structure for native-mode runs).
#
# Submit ONLY after the smoke test (fasrc/md_eval_yang_smoke.slurm) has
# finished cleanly and its outputs look sensible. Optionally chain:
#   sbatch --dependency=afterok:<SMOKE_JOBID> fasrc/md_eval_yang_small.slurm
#
# Override step budget:
#   sbatch --export=ALL,L=50000 fasrc/md_eval_yang_small.slurm

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/nnef}"
DATA_DIR="${DATA_DIR:-$HOME/nnef_data}"

module purge
module load python/3.10.12-fasrc01

PYTHON_BIN="$(command -v python)"
CONDA_HOOK="$(dirname "$(dirname "$PYTHON_BIN")")/etc/profile.d/conda.sh"
[ -f "$CONDA_HOOK" ] || { echo "[md_yang_small] conda.sh not found at $CONDA_HOOK"; exit 1; }
# shellcheck disable=SC1090
source "$CONDA_HOOK"
set +u
conda activate "$ENV_PREFIX"
set -u

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

PY="${ENV_PREFIX}/bin/python"
[ -x "$PY" ] || { echo "[md_yang_small] ERROR: no $PY"; exit 1; }
"$PY" -s -c "import torch; print('[md_yang_small] cuda', torch.cuda.is_available())" || exit 1

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TOKENIZERS_PARALLELISM=false

cd "$SLURM_SUBMIT_DIR"
mkdir -p runs eval/md_eval

MODELS=(yang_retrain v1_rama v1_rama_esm)
TARGETS=(3KXT 1ZZK 2MCM 6H8O 4M1X)
SEEDS=(0 1 2)

N_TARGETS=${#TARGETS[@]}
N_SEEDS=${#SEEDS[@]}

idx=${SLURM_ARRAY_TASK_ID:-0}
model_i=$(( idx / (N_TARGETS * N_SEEDS) ))
target_i=$(( (idx / N_SEEDS) % N_TARGETS ))
seed_i=$(( idx % N_SEEDS ))

TARGET="${TARGETS[$target_i]}"
NATIVE_BEAD="nnef/data/yang_small/${TARGET}/${TARGET}_bead.csv"

if [[ ! -f "$NATIVE_BEAD" ]]; then
  echo "[md_yang_small] ERROR: bead CSV not found: $NATIVE_BEAD"
  exit 1
fi

export REPO="$SLURM_SUBMIT_DIR"
export DATA_DIR="$DATA_DIR"
export DEVICE=cuda
export MD_MODE=native
export MODEL_KEY="${MODELS[$model_i]}"
export TARGET="$TARGET"
export SEED="${SEEDS[$seed_i]}"
export L="${L:-100000}"
export TRJ_LOG="${TRJ_LOG:-100}"
export DECOY_SET="${DECOY_SET:-yang_small}"
export OUT_ROOT="${OUT_ROOT:-eval/md_eval/yang_small}"
export NATIVE_BEAD="$NATIVE_BEAD"

echo "----- md_eval yang_small array task -----"
hostname
nvidia-smi
echo "ARRAY_ID=$SLURM_ARRAY_JOB_ID TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "MODEL=$MODEL_KEY TARGET=$TARGET SEED=$SEED L=$L NATIVE_BEAD=$NATIVE_BEAD"
echo "-----------------------------------------"

bash fasrc/md_eval_one.sh

echo "[md_yang_small] task $SLURM_ARRAY_TASK_ID OK"
