#!/usr/bin/env bash
#SBATCH --job-name=nnef-v3-full
#SBATCH --partition=gpu              # Same partition as v1/v2; no extra memory needed
#SBATCH --gres=gpu:1                 # because ESM is precomputed & looked up as a tensor.
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G                    # Slightly more than v1/v2 (24G) because every
                                     # batch now carries a (B, 15, 1152) fp16 tensor
                                     # of ESM features on top of the usual payload.
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm-%j.out
#SBATCH --error=runs/slurm-%j.err
#
# ----------------------------------------------------------------------------
# nnef-v3-full: NNEF with all four feature-extraction upgrades.
#
#   --use_cart_coords      block-local (x,y,z) in addition to spherical (r,theta,phi)
#   --use_seq_offset       signed chain-distance embedding for every block position
#   --use_esm              ESM-C 600M per-residue embeddings, frozen, per-chain
#                          cached in $DATA_DIR/hhsuite_esm_v2.h5
#   Rama supervision: same as v1/v2 — coords_rama_loss + $DATA_DIR/hhsuite_rama_v2.h5
#   --use_dihedral         explicit phi/psi sin/cos 4-d per-residue input. v2's
#                          local frame only preserves the *central* residue's
#                          N/CA/C orientation; neighbors are CB-only so real
#                          backbone twist is otherwise invisible. Adds ~512
#                          params (linear_x_dihedral) with zero-init, so still
#                          bit-identical at step 0.
#
# Dependency: $DATA_DIR/hhsuite_esm_v2.h5 must exist. Build it with:
#   sbatch fasrc/precompute_esm.slurm
# then submit this job with --dependency=afterok:<precompute_jobid>:
#   sbatch --dependency=afterok:<JOBID> fasrc/train_v3_full.slurm
#
# Everything else (optimizer, batch, epochs, dataset) matches train_v1_pure
# and train.slurm (v2_cart_offset) so the four-way ablation is clean.
# ----------------------------------------------------------------------------

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/nnef}"
DATA_DIR="${DATA_DIR:-$HOME/nnef_data}"

module purge
module load python/3.10.12-fasrc01

PYTHON_BIN="$(command -v python)"
CONDA_HOOK="$(dirname "$(dirname "$PYTHON_BIN")")/etc/profile.d/conda.sh"
[ -f "$CONDA_HOOK" ] || { echo "[train_v3_esm] conda.sh not found at $CONDA_HOOK"; exit 1; }
# shellcheck disable=SC1090
source "$CONDA_HOOK"
# conda hook can touch unset vars; `set -u` breaks `conda activate` on some builds.
set +u
conda activate "$ENV_PREFIX"
set -u

PY="${ENV_PREFIX}/bin/python"
[ -x "$PY" ] || { echo "[train_v3_full] ERROR: no interpreter at $PY (run: bash fasrc/env_setup.sh)"; exit 1; }
"$PY" -c "import torch; print('[train_v3_full] torch', torch.__version__, 'cuda', torch.cuda.is_available())" || {
    echo "[train_v3_full] ERROR: PyTorch missing in $ENV_PREFIX. Re-run fasrc/env_setup.sh or: $PY -m pip install torch ..."
    exit 1
}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

ESM_H5="$DATA_DIR/hhsuite_esm_v2.h5"
if [ ! -f "$ESM_H5" ]; then
    echo "[train_v3_full] ERROR: ESM cache not found at $ESM_H5"
    echo "[train_v3_full] run 'sbatch fasrc/precompute_esm.slurm' first."
    exit 1
fi

echo "----- node info -----"
hostname
nvidia-smi
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
echo "ENV_PREFIX=$ENV_PREFIX"
echo "DATA_DIR=$DATA_DIR"
echo "ESM_H5=$ESM_H5"
echo "---------------------"

cd "$SLURM_SUBMIT_DIR"
mkdir -p runs

"$PY" -u nnef/train_chimeric.py \
    --exp_id v3_full_rama_v2_${SLURM_JOB_ID} \
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
    --use_cart_coords --use_seq_offset \
    --use_esm \
    --esm_h5_path "$ESM_H5" \
    --esm_dim_in 1152 --esm_dim_out 32 \
    --use_dihedral

echo "[train_v3_full] finished with exit code $?"
