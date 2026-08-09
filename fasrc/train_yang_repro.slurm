#!/usr/bin/env bash
#SBATCH --job-name=nnef-yang-repro
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm-yangrepro-%j.out
#SBATCH --error=runs/slurm-yangrepro-%j.out
# -----------------------------------------------------------------------------
# Faithful Yang 2022 reproduction on the REBUILT, frame-correct dataset.
#
# vs the old train_yang_legacy.slurm this fixes the two things the audit found:
#   1. DATA: trains on hhsuite_CB_yang.h5 (Yang neighbour-bead frame, central
#      bead at origin) instead of the v2 N-CA-C-frame h5. This matches the
#      --legacy_local_frame inference used at decoy/MD time -> no train/infer
#      frame mismatch (the root cause of the native-attractor MD failure).
#   2. TRAINER: gradient clipping OFF (--grad_clip_norm 0) to match Yang, who
#      uses none. (The old trainer hard-coded clip_grad_norm_(1.0).)
#
# All architecture/optim flags match Yang's README exp1 command exactly.
# EPOCHS defaults to 200 for a smoke; set EPOCHS=1000 to match the paper.
#
# Requires stage 6 (hhsuite_CB_yang.h5) + stage 7 (hhsuite_pdb_seq_yang.h5 +
# weighted pdb_list). Run fasrc/smoke_train_yang.py first to sanity-check load.
# -----------------------------------------------------------------------------
set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/nnef}"
DATA_DIR="${DATA_DIR:-$HOME/nnef_data}"
NS="${NS:-/n/netscratch/atchan_lab/qzha/nnef_data}"
PDB_H5="${PDB_H5:-$NS/hhsuite_CB_yang.h5}"
SEQ_H5="${SEQ_H5:-$NS/hhsuite_pdb_seq_yang.h5}"
PDB_LIST="${PDB_LIST:-$DATA_DIR/hhsuite_CB_yang_pdb_list.csv}"
EPOCHS="${EPOCHS:-200}"

module purge
module load python/3.10.12-fasrc01
PYTHON_BIN="$(command -v python)"
CONDA_HOOK="$(dirname "$(dirname "$PYTHON_BIN")")/etc/profile.d/conda.sh"
# shellcheck disable=SC1090
source "$CONDA_HOOK"; set +u; conda activate "$ENV_PREFIX"; set -u
PY="$ENV_PREFIX/bin/python"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"; mkdir -p runs
nvidia-smi || true
for f in "$PDB_H5" "$SEQ_H5" "$PDB_LIST"; do
    [[ -e "$f" ]] || { echo "[yang-repro] ERROR: missing $f"; exit 1; }
done
echo "[yang-repro] PDB_H5=$PDB_H5"
echo "[yang-repro] SEQ_H5=$SEQ_H5"
echo "[yang-repro] PDB_LIST=$PDB_LIST  EPOCHS=$EPOCHS"

# Angle/radial output distribution ablation (see options.py). Default keeps the
# Yang-faithful gaussian. Override: sbatch --export=ALL,ANGLE_DIST=vonmises,...
ANGLE_DIST="${ANGLE_DIST:-gaussian}"
R_DIST="${R_DIST:-gaussian}"
echo "[yang_repro] angle_dist=$ANGLE_DIST r_dist=$R_DIST"

"$PY" -u nnef/train_chimeric.py \
    --exp_id "${EXP_ID:-yang_repro}_${SLURM_JOB_ID}" \
    --save_path runs \
    --data_flag "$PDB_LIST" \
    --seq_type residue --residue_type_num 20 --chimeric \
    --seq_len 14 \
    --batch_size 1000 \
    --embed_size 32 --dim 128 --n_layers 4 --attn_heads 4 \
    --mixture_r 2 --mixture_angle 3 --mixture_seq 1 \
    --mixture_rama 0 \
    --legacy_local_frame \
    --angle_dist "$ANGLE_DIST" --r_dist "$R_DIST" \
    --grad_clip_norm 0 \
    --smooth_gaussian --smooth_r 0.3 --smooth_angle 45 \
    --lr 5e-5 --n_warmup_steps 5000 --steps_decay_scale 30000 \
    --coords_angle_loss_lamda 1 --profile_loss_lamda 10 \
    --use_position_weights --cen_seg_loss_lamda 1 --oth_seg_loss_lamda 3 \
    --log_interval 30 --save_interval 50 --num_workers 4 \
    --epochs "$EPOCHS" --total_num_samples 1000000 \
    --pdb_h5_path "$PDB_H5" \
    --seq_h5_path "$SEQ_H5" \
    --rama_h5_path "$DATA_DIR/__NO_RAMA_SENTINEL__.h5"

echo "[yang-repro] finished exit=$?"
