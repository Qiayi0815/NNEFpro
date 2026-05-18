#!/usr/bin/env bash
#SBATCH --job-name=nnef-v1-esm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm-v1-esm-%A_%a.out
#SBATCH --error=runs/slurm-v1-esm-%A_%a.out
#
# ----------------------------------------------------------------------------
# v1 + ESM with three configurable knobs (esm_dim_out / esm layer / esm_pool).
#
# Two submission modes (one file, no fork):
#
#   1) SINGLE CELL (back-compat, reproduces the original v1_rama_esm_v2):
#        cd ~/nnef && sbatch fasrc/train_v1_esm.slurm
#      Uses defaults: ESM-C 600M, dim_out=32, dataset 'esm', pool per_residue.
#
#   2) 2^3 ABLATION ARRAY (width x layer x pool, 8 cells on 8 A100s):
#        cd ~/nnef && sbatch --array=0-7 \\
#            --export=ALL,ABL_ESM_H5=hhsuite_esm2_650m_v2.h5,ESM_DIM_IN=1280 \\
#            fasrc/train_v1_esm.slurm
#      Each task_id maps to one (width, layer, pool) via the lookup tables
#      below. The h5 must already be built with `--repr_layers 32 33` -- see
#      fasrc/precompute_esm.slurm (override --model esm2_t33_650M_UR50D
#      --repr_layers 32 33 --out_h5 .../hhsuite_esm2_650m_v2.h5).
#
# Output (single):  runs/v1_esm_rama_v2_<JOBID>/models/model.pt
# Output (array):   runs/v1_esm_abl_w<W>_l<L>_<POOL>_<JOBID>/models/model.pt
# ----------------------------------------------------------------------------

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/nnef}"
DATA_DIR="${DATA_DIR:-$HOME/nnef_data}"

# Ablation lookup tables: 8 cells = 2 widths * 2 layers * 2 pools.
WIDTHS=(32 32 32 32 128 128 128 128)
LAYERS=(33 33 32 32 33 33 32 32)
POOLS=(per_residue attn_pool per_residue attn_pool per_residue attn_pool per_residue attn_pool)

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    # ARRAY MODE
    idx=${SLURM_ARRAY_TASK_ID}
    [ "$idx" -ge 0 ] && [ "$idx" -lt "${#WIDTHS[@]}" ] || {
        echo "[train_v1_esm] ERROR: SLURM_ARRAY_TASK_ID=$idx out of range 0..${#WIDTHS[@]}";
        exit 1; }
    W=${WIDTHS[$idx]}
    L=${LAYERS[$idx]}
    P=${POOLS[$idx]}
    ESM_H5_NAME="${ABL_ESM_H5:-hhsuite_esm2_650m_v2.h5}"
    ESM_DIM_IN="${ESM_DIM_IN:-1280}"
    ESM_DATASET="esm_l${L}"
    EXP_ID="v1_esm_abl_w${W}_l${L}_${P}_${SLURM_JOB_ID}"
    JOB_LABEL="abl[$idx] w=$W layer=$L pool=$P"
else
    # SINGLE-CELL BACK-COMPAT (reproduces the original v1_rama_esm_v2_6642146)
    W="${ESM_DIM_OUT:-32}"
    L="${ESM_LAYER_LABEL:-final}"
    P="${ESM_POOL:-per_residue}"
    ESM_H5_NAME="${ABL_ESM_H5:-hhsuite_esm_v2.h5}"
    ESM_DIM_IN="${ESM_DIM_IN:-1152}"
    ESM_DATASET="${ESM_DATASET_NAME:-esm}"
    EXP_ID="v1_esm_rama_v2_${SLURM_JOB_ID}"
    JOB_LABEL="single-cell"
fi

module purge
module load python/3.10.12-fasrc01

PYTHON_BIN="$(command -v python)"
CONDA_HOOK="$(dirname "$(dirname "$PYTHON_BIN")")/etc/profile.d/conda.sh"
[ -f "$CONDA_HOOK" ] || { echo "[train_v1_esm] conda.sh not found at $CONDA_HOOK"; exit 1; }
# shellcheck disable=SC1090
source "$CONDA_HOOK"
set +u
conda activate "$ENV_PREFIX"
set -u

PY="${ENV_PREFIX}/bin/python"
[ -x "$PY" ] || { echo "[train_v1_esm] ERROR: no interpreter at $PY"; exit 1; }
"$PY" -c "import torch; print('[train_v1_esm] torch', torch.__version__, 'cuda', torch.cuda.is_available())" || exit 1

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

ESM_H5="$DATA_DIR/$ESM_H5_NAME"
if [ ! -f "$ESM_H5" ]; then
    echo "[train_v1_esm] ERROR: ESM cache not found at $ESM_H5"
    echo "[train_v1_esm] For single-cell: sbatch fasrc/precompute_esm.slurm"
    echo "[train_v1_esm] For ablation array, build the multi-layer ESM2 cache:"
    echo "                 sbatch fasrc/precompute_esm.slurm   # with esm2 + --repr_layers 32 33 overrides"
    exit 1
fi

echo "----- node info -----"
hostname
nvidia-smi
echo "ENV_PREFIX=$ENV_PREFIX DATA_DIR=$DATA_DIR"
echo "ESM_H5=$ESM_H5 ESM_DIM_IN=$ESM_DIM_IN ESM_DIM_OUT=$W ESM_DATASET=$ESM_DATASET POOL=$P"
echo "EXP_ID=$EXP_ID  ($JOB_LABEL)"
echo "---------------------"

cd "$SLURM_SUBMIT_DIR"
mkdir -p runs

"$PY" -u nnef/train_chimeric.py \
    --exp_id "$EXP_ID" \
    --save_path runs \
    --seq_type residue \
    --residue_type_num 20 \
    --chimeric \
    --seq_len 14 \
    --batch_size 1000 \
    --embed_size 32 --dim 128 --n_layers 4 --attn_heads 4 \
    --mixture_r 2 --mixture_angle 3 --mixture_rama 10 \
    --smooth_gaussian --smooth_r 0.3 --smooth_angle 45 \
    --lr 5e-5 --n_warmup_steps 5000 --steps_decay_scale 30000 \
    --coords_angle_loss_lamda 1 --profile_loss_lamda 10 --coords_rama_loss_lamda 1 \
    --use_position_weights --cen_seg_loss_lamda 1 --oth_seg_loss_lamda 3 \
    --log_interval 30 --save_interval 50 --num_workers 4 \
    --epochs 200 --total_num_samples 1000000 \
    --pdb_h5_path "$DATA_DIR/hhsuite_CB_v2.h5" \
    --seq_h5_path "$DATA_DIR/hhsuite_pdb_seq_v2.h5" \
    --rama_h5_path "$DATA_DIR/hhsuite_rama_v2.h5" \
    --use_esm \
    --esm_h5_path "$ESM_H5" \
    --esm_dataset_name "$ESM_DATASET" \
    --esm_dim_in "$ESM_DIM_IN" \
    --esm_dim_out "$W" \
    --esm_pool "$P"

echo "[train_v1_esm] finished with exit code $?"
