#!/usr/bin/env bash
#SBATCH --job-name=nnef-v1-esm-abl-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=runs/slurm-v1-esm-abl-eval-%A_%a.out
#SBATCH --error=runs/slurm-v1-esm-abl-eval-%A_%a.out
#SBATCH --array=0-7
#
# ----------------------------------------------------------------------------
# Eval the 8 ESM-ablation checkpoints produced by fasrc/train_v1_esm.slurm
# (in --array mode). Each task takes one cell and runs:
#   1) decoy-correlation eval on CASP14 T1053 + T1026
#   2) mode-2 native-basin MD on T1026 with 3 seeds
#
# Reuses fasrc/md_eval_one.sh's v1_rama_esm branch via the env-passthrough
# wired up in patch E. Runs strictly AFTER the training array finishes.
#
# Submit (after training array done):
#   sbatch fasrc/eval_v1_esm_ablation_array.slurm
# ----------------------------------------------------------------------------

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/nnef}"
DATA_DIR="${DATA_DIR:-$HOME/nnef_data}"

# Same lookup tables as fasrc/train_v1_esm.slurm. Keep in sync if you change one.
WIDTHS=(32 32 32 32 128 128 128 128)
LAYERS=(33 33 32 32 33 33 32 32)
POOLS=(per_residue attn_pool per_residue attn_pool per_residue attn_pool per_residue attn_pool)

idx=${SLURM_ARRAY_TASK_ID:-0}
W=${WIDTHS[$idx]}
LAYER=${LAYERS[$idx]}
P=${POOLS[$idx]}
TAG="w${W}_l${LAYER}_${P}"

module purge
module load python/3.10.12-fasrc01
PYTHON_BIN="$(command -v python)"
CONDA_HOOK="$(dirname "$(dirname "$PYTHON_BIN")")/etc/profile.d/conda.sh"
[ -f "$CONDA_HOOK" ] || { echo "[abl_eval] conda.sh not found at $CONDA_HOOK"; exit 1; }
# shellcheck disable=SC1090
source "$CONDA_HOOK"
set +u
conda activate "$ENV_PREFIX"
set -u

PY="${ENV_PREFIX}/bin/python"
[ -x "$PY" ] || { echo "[abl_eval] ERROR: no $PY"; exit 1; }
"$PY" -s -c "import torch; print('[abl_eval] cuda', torch.cuda.is_available())" || exit 1

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

cd "$SLURM_SUBMIT_DIR"

# Locate the matching training run (latest job-id for this cell's tag).
RUN_DIR=$(ls -dt runs/v1_esm_abl_${TAG}_* 2>/dev/null | head -1)
[ -n "$RUN_DIR" ] || { echo "[abl_eval] ERROR: no runs/v1_esm_abl_${TAG}_* found"; exit 1; }
[ -f "$RUN_DIR/models/model.pt" ] || { echo "[abl_eval] ERROR: $RUN_DIR/models/model.pt missing"; exit 1; }

ESM_H5="$DATA_DIR/hhsuite_esm2_650m_v2.h5"
[ -f "$ESM_H5" ] || { echo "[abl_eval] ERROR: missing $ESM_H5"; exit 1; }

echo "----- v1_esm ablation eval -----"
hostname; nvidia-smi
echo "cell=$idx tag=$TAG  W=$W LAYER=$LAYER P=$P"
echo "RUN_DIR=$RUN_DIR"
echo "ESM_H5=$ESM_H5  dataset=esm_l${LAYER}"
echo "--------------------------------"

ARCH_SHARED=(
  --seq_len 14 --seq_type residue --residue_type_num 20
  --embed_size 32 --dim 128 --n_layers 4 --attn_heads 4
  --mixture_r 2 --mixture_angle 3 --mixture_rama 10
  --smooth_gaussian --smooth_r 0.3 --smooth_angle 45
  --coords_angle_loss_lamda 1 --profile_loss_lamda 10 --coords_rama_loss_lamda 1
  --use_position_weights --cen_seg_loss_lamda 1 --oth_seg_loss_lamda 3
)
ESM_FLAGS=(
  --use_esm
  --esm_h5_path "$ESM_H5"
  --esm_dataset_name "esm_l${LAYER}"
  --esm_dim_in 1280
  --esm_dim_out "$W"
  --esm_pool "$P"
)

# 1) Decoy correlation: T1053 + T1026
for TGT in T1053 T1026; do
  OUT_DIR="eval/$(basename "$RUN_DIR")_casp14_${TGT}"
  mkdir -p "$OUT_DIR"
  echo "----- decoy $TGT -----"
  "$PY" -s nnef/scripts/evaluate_decoys.py \
    --load_exp "$RUN_DIR" \
    --decoy_sets casp14 --targets "$TGT" \
    --device cuda --plot --no_skip_if_exists \
    --exp_tag "$(basename "$RUN_DIR")_${TGT}" \
    --out_dir "$OUT_DIR" \
    "${ARCH_SHARED[@]}" "${ESM_FLAGS[@]}"
done

# 2) Mode-2 native-basin MD: T1026, 3 seeds, matching mode2_yang_paper sampler.
mkdir -p runs eval/md_eval/esm_ablation
for SEED in 0 1 2; do
  echo "----- mode2 MD seed=$SEED -----"
  export REPO="$SLURM_SUBMIT_DIR"
  export DATA_DIR="$DATA_DIR"
  export DEVICE=cuda
  export MD_MODE=native
  export MODEL_KEY=v1_rama_esm
  export LOAD_EXP="$RUN_DIR"
  export TARGET=T1026
  export SEED="$SEED"
  export L=50000             # MD step count (md_eval_one.sh's L)
  export TRJ_LOG=100
  export LR=3e-4
  export T_NOISE=3e-4
  # ESM env -- consumed by the patched v1_rama_esm branch in md_eval_one.sh
  export ESM_H5_OVERRIDE="$ESM_H5"
  export ESM_DIM_IN=1280
  export ESM_DIM_OUT="$W"
  export ESM_DATASET_NAME="esm_l${LAYER}"
  export ESM_POOL="$P"
  export OUT_ROOT="eval/md_eval/esm_ablation"
  bash fasrc/md_eval_one.sh
done

echo "[abl_eval] cell $idx ($TAG) OK"
