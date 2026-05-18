#!/usr/bin/env bash
#SBATCH --job-name=nnef-md-mode2-lrabl-T1026
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=runs/slurm-md-mode2-lrabl-T1026-%A_%a.out
#SBATCH --error=runs/slurm-md-mode2-lrabl-T1026-%A_%a.out
#SBATCH --array=0-11
#
# Mode 2 lr ablation on CASP14 T1026 (L=172, smallest target).
#
# Goal: diagnose the "Rg collapses 39 -> 28 A across all models" signal.
#   * If smaller lr brings Rg back near 38 A, the collapse is a sampler
#     step-size artefact and we can re-run paired Mode 2 at the rescued lr.
#   * If Rg stays collapsed regardless of lr, it is an energy-function
#     bias in both v1_rama and v1_rama_esm; more seeds / targets will not
#     disentangle the models, and we need to revisit the training signal.
#
# Grid (12 tasks):
#   models = (v1_rama, v1_rama_esm)       # yang_retrain intentionally dropped;
#                                           # yang_legacy retrain still running.
#   lrs    = (1e-3, 3e-4)                 # T_noise = lr; baseline 3e-3 skipped
#                                           # since it's already the previously
#                                           # observed "collapse" regime.
#   seeds  = (0, 1, 2)
# => 2 * 2 * 3 = 12 array tasks.
#
# L=50000 (half of the full-run default) because we only need the Rg and
# RMSD trajectory shapes, not full equilibration statistics.
#
# Prereq: T1026 list.csv + decoy bead CSVs already on cluster (previous
# md_eval_mode2_T1026 array synced them). v1_esm also needs the ESM cache
# $DATA_DIR/hhsuite_esm_v2.h5.
#
# Submit:
#   sbatch fasrc/md_eval_mode2_lr_ablation_T1026.slurm
# Longer run:
#   sbatch --export=ALL,L=100000 fasrc/md_eval_mode2_lr_ablation_T1026.slurm

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/nnef}"
DATA_DIR="${DATA_DIR:-$HOME/nnef_data}"

module purge
module load python/3.10.12-fasrc01

PYTHON_BIN="$(command -v python)"
CONDA_HOOK="$(dirname "$(dirname "$PYTHON_BIN")")/etc/profile.d/conda.sh"
[ -f "$CONDA_HOOK" ] || { echo "[md_mode2_lrabl] conda.sh not found at $CONDA_HOOK"; exit 1; }
# shellcheck disable=SC1090
source "$CONDA_HOOK"
set +u
conda activate "$ENV_PREFIX"
set -u

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

PY="${ENV_PREFIX}/bin/python"
[ -x "$PY" ] || { echo "[md_mode2_lrabl] ERROR: no $PY"; exit 1; }
"$PY" -s -c "import torch; print('[md_mode2_lrabl] cuda', torch.cuda.is_available())" || exit 1

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TOKENIZERS_PARALLELISM=false

cd "$SLURM_SUBMIT_DIR"
mkdir -p runs eval/md_eval

_T1026_LIST="nnef/data/decoys/casp14/T1026/list.csv"
if [[ ! -f "$_T1026_LIST" ]]; then
  echo "[md_mode2_lrabl] ERROR: $_T1026_LIST not found on cluster."
  echo "    rsync from laptop before submitting:"
  echo "      rsync -av nnef/data/decoys/casp14/T1026/ qzha@cluster:~/nnef/nnef/data/decoys/casp14/T1026/"
  exit 1
fi

MODELS=(v1_rama v1_rama_esm)
LRS=(1e-3 3e-4)
SEEDS=(0 1 2)

N_LRS=${#LRS[@]}
N_SEEDS=${#SEEDS[@]}
N_PER_MODEL=$(( N_LRS * N_SEEDS ))   # 6

idx=${SLURM_ARRAY_TASK_ID:-0}
model_i=$(( idx / N_PER_MODEL ))
rem=$(( idx % N_PER_MODEL ))
lr_i=$(( rem / N_SEEDS ))
seed_i=$(( rem % N_SEEDS ))

export REPO="$SLURM_SUBMIT_DIR"
export DATA_DIR="$DATA_DIR"
export DEVICE=cuda
export MD_MODE=native
export MODEL_KEY="${MODELS[$model_i]}"
export TARGET=T1026
export SEED="${SEEDS[$seed_i]}"
export LR="${LRS[$lr_i]}"
export T_NOISE="$LR"
export L="${L:-50000}"
export TRJ_LOG="${TRJ_LOG:-100}"
export OUT_ROOT="${OUT_ROOT:-eval/md_eval/mode2_lr_ablation}"

# Disambiguate runs by lr so the analysis script can tell the 9 per-model
# trajectories apart. md_eval_one.sh honours SAVE_DIR if set.
case "$MODEL_KEY" in
  v1_rama)     RUN_BASE="v1_pure_rama_v2_6228201" ;;
  v1_rama_esm) RUN_BASE="v1_esm_rama_v2_6642146" ;;
  *) echo "[md_mode2_lrabl] unexpected MODEL_KEY=$MODEL_KEY"; exit 1 ;;
esac
export SAVE_DIR="$OUT_ROOT/$MD_MODE/$RUN_BASE/${TARGET}_lr${LR}_seed${SEED}"

echo "----- md_eval mode2 lr-ablation T1026 -----"
hostname
nvidia-smi
echo "ARRAY_ID=$SLURM_ARRAY_JOB_ID TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "MODEL=$MODEL_KEY LR=$LR T_NOISE=$T_NOISE SEED=$SEED L=$L"
echo "SAVE_DIR=$SAVE_DIR"
echo "-------------------------------------------"

bash fasrc/md_eval_one.sh

echo "[md_mode2_lrabl] task $SLURM_ARRAY_TASK_ID OK -> $SAVE_DIR"
