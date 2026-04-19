#!/usr/bin/env bash
#SBATCH --job-name=nnef-v2-cart-offset
#SBATCH --partition=gpu              # `gpu` = full A100 40GB (~20 min-2h queue).
                                     # `gpu_h200` = H200 (faster, queue varies).
                                     # `gpu_requeue` = backfill, preemptible.
                                     # `gpu_test` uses MIG slices (debug only).
#SBATCH --gres=gpu:1                 # one GPU is plenty for LocalTransformer (~3-6 GB).
#SBATCH --cpus-per-task=4            # data loader + a bit of headroom.
#SBATCH --mem=24G                    # batches are small; 24 GB is generous.
#SBATCH --time=24:00:00              # raise if you need longer runs.
#SBATCH --output=runs/slurm-%j.out   # both stdout and stderr land here
#SBATCH --error=runs/slurm-%j.err
#
# Submit with:
#
#   sbatch fasrc/train.slurm
#
# Monitor with:
#
#   squeue -u $USER
#   tail -f runs/slurm-<JOBID>.out
#
# Cancel with:
#
#   scancel <JOBID>
#
# ----------------------------------------------------------------------------
# Customize ENV_PREFIX + DATA_DIR for your setup.
# ----------------------------------------------------------------------------

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/nnef}"
DATA_DIR="${DATA_DIR:-$HOME/nnef_data}"

# ----------------------------------------------------------------------------
# 1. Activate the conda env built by fasrc/env_setup.sh.
# ----------------------------------------------------------------------------
module purge
module load python/3.10.12-fasrc01

# Robustly source conda's shell hook: locate it relative to the `conda`
# binary shipped by the python module. Works whether mamba is present or not.
# `conda`/`mamba` are shell functions on FASRC; derive the install prefix
# from the python binary shipped by the python module instead.
PYTHON_BIN="$(command -v python)"
CONDA_HOOK="$(dirname "$(dirname "$PYTHON_BIN")")/etc/profile.d/conda.sh"
[ -f "$CONDA_HOOK" ] || { echo "[train.slurm] conda.sh not found at $CONDA_HOOK"; exit 1; }
# shellcheck disable=SC1090
source "$CONDA_HOOK"
# conda hook can touch unset vars; `set -u` breaks `conda activate` on some builds.
set +u
conda activate "$ENV_PREFIX"
set -u

PY="${ENV_PREFIX}/bin/python"
[ -x "$PY" ] || { echo "[train.slurm] ERROR: no interpreter at $PY (run: bash fasrc/env_setup.sh)"; exit 1; }
"$PY" -c "import torch; print('[train.slurm] torch', torch.__version__, 'cuda', torch.cuda.is_available())" || {
    echo "[train.slurm] ERROR: PyTorch missing in $ENV_PREFIX. Re-run fasrc/env_setup.sh or: $PY -m pip install torch ..."
    exit 1
}

# ----------------------------------------------------------------------------
# 2. Helpful runtime knobs.
# ----------------------------------------------------------------------------
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

# ----------------------------------------------------------------------------
# 3. Print node / GPU info to the log for later debugging.
# ----------------------------------------------------------------------------
echo "----- node info -----"
hostname
nvidia-smi
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
echo "ENV_PREFIX=$ENV_PREFIX"
echo "PY=$PY"
echo "DATA_DIR=$DATA_DIR"
echo "---------------------"

# ----------------------------------------------------------------------------
# 4. Launch training.
#
# Baseline (bit-identical to pre-thesis NNEF, no ESM / cart / offset):
#   python nnef/train_chimeric.py --exp_id baseline_v2 ...
#
# With the thesis-v2 block extractor + cart coords + signed offset:
#   add  --use_cart_coords --use_seq_offset
#
# With ESM context features (slow; needs the precomputed h5):
#   add  --use_esm --esm_h5_path $DATA_DIR/hhsuite_esm_v2.h5
#
# NOTE: launch as a script (not `-m nnef.xxx`). train_chimeric.py uses bare
# sibling imports (`import options`, `from dataset import ...`), which only
# resolve when the script's directory (nnef/) is on sys.path[0].
# ----------------------------------------------------------------------------
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs

"$PY" -u nnef/train_chimeric.py \
    --exp_id v2_run_${SLURM_JOB_ID} \
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
    --use_cart_coords --use_seq_offset

echo "[train.slurm] finished with exit code $?"
