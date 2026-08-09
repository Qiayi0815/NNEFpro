#!/usr/bin/env bash
#SBATCH --job-name=nnef-esm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm-esm-%j.out
#SBATCH --error=runs/slurm-esm-%j.out
# ----------------------------------------------------------------------------
# esm model = ribbon (v2 frame + Ramachandran) + ESM-C adapter, retrained on the
# new 28k-chain dataset. Formerly v1_esm_rama_v2 (run 6642146, trained WITH the
# ESM zero-init bug -> ESM was effectively dead; that bug is now fixed and
# verified: linear_x_esm.grad > 0).
#
# Same as train_ribbon.slurm plus:  --use_esm --esm_h5_path <esm cache>
#   --esm_dim_in 1152 (ESM-C 600M) --esm_dim_out 32 --esm_pool per_residue
#
# Needs: v2 struct h5 + rama h5 + seq h5 + ESM h5 (precompute_esm_new.slurm) +
#        weighted pdb_list (computed inline). EPOCHS defaults 200 (smoke).
# ----------------------------------------------------------------------------
set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/nnef}"
DATA_DIR="${DATA_DIR:-$HOME/nnef_data}"
NS="${NS:-/n/netscratch/atchan_lab/qzha/nnef_data}"
PDB_H5="${PDB_H5:-$NS/hhsuite_CB_v2_new.h5}"
SEQ_H5="${SEQ_H5:-$NS/hhsuite_pdb_seq_yang.h5}"
RAMA_H5="${RAMA_H5:-$NS/hhsuite_rama_v2_new.h5}"
ESM_H5="${ESM_H5:-$NS/hhsuite_esm_v2_new.h5}"
PDB_LIST="${PDB_LIST:-$DATA_DIR/hhsuite_CB_v2_new_pdb_list.csv}"
EPOCHS="${EPOCHS:-200}"

module purge; module load python/3.10.12-fasrc01
PYTHON_BIN="$(command -v python)"
CONDA_HOOK="$(dirname "$(dirname "$PYTHON_BIN")")/etc/profile.d/conda.sh"
# shellcheck disable=SC1090
source "$CONDA_HOOK"; set +u; conda activate "$ENV_PREFIX"; set -u
PY="$ENV_PREFIX/bin/python"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}; export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"; mkdir -p runs; nvidia-smi || true
for f in "$PDB_H5" "$SEQ_H5" "$RAMA_H5" "$ESM_H5" "$PDB_LIST"; do
    [[ -e "$f" ]] || { echo "[esm] ERROR: missing $f"; exit 1; }
done
echo "[esm] PDB_H5=$PDB_H5 RAMA_H5=$RAMA_H5 ESM_H5=$ESM_H5 EPOCHS=$EPOCHS"

echo "[esm] computing weights into $PDB_LIST"
"$PY" -u nnef/data_prep_scripts/compute_weights.py \
    --struct_h5 "$PDB_H5" --seq_h5 "$SEQ_H5" --pdb_list "$PDB_LIST"

"$PY" -u nnef/train_chimeric.py \
    --exp_id "${EXP_ID:-esm}_${SLURM_JOB_ID}" \
    --save_path runs \
    --data_flag "$PDB_LIST" \
    --seq_type residue --residue_type_num 20 --chimeric \
    --seq_len 14 \
    --batch_size 1000 \
    --embed_size 32 --dim 128 --n_layers 4 --attn_heads 4 \
    --mixture_r 2 --mixture_angle 3 --mixture_seq 1 \
    --mixture_rama 10 --coords_rama_loss_lamda 1 \
    --use_esm --esm_h5_path "$ESM_H5" --esm_dim_in 1152 --esm_dim_out 32 --esm_pool per_residue \
    --grad_clip_norm 1.0 \
    --smooth_gaussian --smooth_r 0.3 --smooth_angle 45 \
    --lr 5e-5 --n_warmup_steps 5000 --steps_decay_scale 30000 \
    --coords_angle_loss_lamda 1 --profile_loss_lamda 10 \
    --use_position_weights --cen_seg_loss_lamda 1 --oth_seg_loss_lamda 3 \
    --log_interval 30 --save_interval 50 --num_workers 4 \
    --epochs "$EPOCHS" --total_num_samples 1000000 \
    --pdb_h5_path "$PDB_H5" \
    --seq_h5_path "$SEQ_H5" \
    --rama_h5_path "$RAMA_H5"

echo "[esm] finished exit=$?"
