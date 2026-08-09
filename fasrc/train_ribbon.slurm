#!/usr/bin/env bash
#SBATCH --job-name=nnef-ribbon
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm-ribbon-%j.out
#SBATCH --error=runs/slurm-ribbon-%j.out
# -----------------------------------------------------------------------------
# "ribbon" = the FYP modified model (v2 N-CA-C frame + Ramachandran head),
# retrained on the new 28k-chain dataset. Formerly v1_pure_rama_v2 (run 6228201).
#
# vs the Yang reproduction (train_yang_repro.slurm) the differences are:
#   * v2 frame: NO --legacy_local_frame  (thesis's modified coordinate system)
#   * Ramachandran head ON: --mixture_rama 10 --coords_rama_loss_lamda 1
#     (the rama term is now also scored at DECOY inference after the protein_os
#      get_energy fix; it stays absent in CB-only MD by necessity.)
#   * grad_clip_norm 1.0 (preserve the modified model's original training recipe;
#     only the DATA and the rama-inference fix are the intended changes).
#
# Needs: v2 structure h5 (build_v2_h5.slurm) + rama h5 (build_rama_h5.slurm) +
#        seq h5 (hhsuite_pdb_seq_yang.h5, frame-independent) + weighted pdb_list.
# EPOCHS defaults 200 (smoke); set EPOCHS=1000 to match the paper budget.
# -----------------------------------------------------------------------------
set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/nnef}"
DATA_DIR="${DATA_DIR:-$HOME/nnef_data}"
NS="${NS:-/n/netscratch/atchan_lab/qzha/nnef_data}"
PDB_H5="${PDB_H5:-$NS/hhsuite_CB_v2_new.h5}"
SEQ_H5="${SEQ_H5:-$NS/hhsuite_pdb_seq_yang.h5}"
RAMA_H5="${RAMA_H5:-$NS/hhsuite_rama_v2_new.h5}"
PDB_LIST="${PDB_LIST:-$DATA_DIR/hhsuite_CB_v2_new_pdb_list.csv}"
# Optional held-out validation split (nnef/data_prep_scripts/split_pdb_list_val.py),
# used ONLY for hyperparameter selection (advisor suggestion #1: tune ribbon's
# params). Purely diagnostic -- train_chimeric.py's Val pass runs with
# torch.set_grad_enabled(False), it never affects the optimizer. Leave unset
# for ordinary (non-ablation) training runs.
VAL_PDB_LIST="${VAL_PDB_LIST:-}"
EPOCHS="${EPOCHS:-200}"
# Coords-head mixture counts. Yang/baseline use r=2, angle=3; raise them to test
# whether the plateaued radius/angle NLL is limited by output-distribution
# expressiveness (diagnostic lever #1). Eval must pass the SAME values.
MIXR="${MIXTURE_R:-2}"
MIXA="${MIXTURE_ANGLE:-3}"
# Ramachandran head params -- OUR addition (not Yang's), so these are the
# primary tuning targets for the ribbon hyperparameter ablation.
MIXRAMA="${MIXTURE_RAMA:-10}"
RAMALAMDA="${COORDS_RAMA_LAMDA:-1}"

module purge; module load python/3.10.12-fasrc01
PYTHON_BIN="$(command -v python)"
CONDA_HOOK="$(dirname "$(dirname "$PYTHON_BIN")")/etc/profile.d/conda.sh"
# shellcheck disable=SC1090
source "$CONDA_HOOK"; set +u; conda activate "$ENV_PREFIX"; set -u
PY="$ENV_PREFIX/bin/python"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}; export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"; mkdir -p runs; nvidia-smi || true
for f in "$PDB_H5" "$SEQ_H5" "$RAMA_H5" "$PDB_LIST"; do
    [[ -e "$f" ]] || { echo "[ribbon] ERROR: missing $f"; exit 1; }
done
echo "[ribbon] PDB_H5=$PDB_H5 SEQ_H5=$SEQ_H5 RAMA_H5=$RAMA_H5 EPOCHS=$EPOCHS MIXR=$MIXR MIXA=$MIXA MIXRAMA=$MIXRAMA RAMALAMDA=$RAMALAMDA VAL_PDB_LIST=${VAL_PDB_LIST:-<none>}"

# Yang-style sampling weights into the pdb_list (idempotent; needs struct+seq h5).
# Skips if 'weight' column already present -- compute_weights.py writes IN
# PLACE, and this script may run as CONCURRENT array tasks sharing the SAME
# pdb_list (see train_ribbon_ablation.slurm's 27-way sweep); without this
# guard every task would race to overwrite the same file simultaneously.
if head -1 "$PDB_LIST" | tr ',' '\n' | grep -qx weight; then
  echo "[ribbon] $PDB_LIST already weighted, skipping compute_weights.py"
else
  echo "[ribbon] computing weights into $PDB_LIST"
  "$PY" -u nnef/data_prep_scripts/compute_weights.py \
      --struct_h5 "$PDB_H5" --seq_h5 "$SEQ_H5" --pdb_list "$PDB_LIST"
fi

VAL_ARGS=()
if [[ -n "$VAL_PDB_LIST" ]]; then
  [[ -e "$VAL_PDB_LIST" ]] || { echo "[ribbon] ERROR: missing $VAL_PDB_LIST"; exit 1; }
  if head -1 "$VAL_PDB_LIST" | tr ',' '\n' | grep -qx weight; then
    echo "[ribbon] $VAL_PDB_LIST already weighted, skipping compute_weights.py"
  else
    echo "[ribbon] computing weights into $VAL_PDB_LIST"
    "$PY" -u nnef/data_prep_scripts/compute_weights.py \
        --struct_h5 "$PDB_H5" --seq_h5 "$SEQ_H5" --pdb_list "$VAL_PDB_LIST"
  fi
  VAL_ARGS=( --val_data_flag "$VAL_PDB_LIST" )
fi

# Output distribution ablation (see options.py). Defaults = Yang-faithful gaussian.
#   ANGLE_DIST: gaussian | vonmises | vmf     R_DIST: gaussian | lognormal
# Override at submit: sbatch --export=ALL,ANGLE_DIST=vmf,R_DIST=lognormal,EXP_ID=ribbon_vmf ...
ANGLE_DIST="${ANGLE_DIST:-gaussian}"
R_DIST="${R_DIST:-gaussian}"
echo "[ribbon] angle_dist=$ANGLE_DIST r_dist=$R_DIST"

"$PY" -u nnef/train_chimeric.py \
    --exp_id "${EXP_ID:-ribbon}_${SLURM_JOB_ID}" \
    --save_path runs \
    --data_flag "$PDB_LIST" \
    --seq_type residue --residue_type_num 20 --chimeric \
    --seq_len 14 \
    --batch_size 1000 \
    --embed_size 32 --dim 128 --n_layers 4 --attn_heads 4 \
    --mixture_r "$MIXR" --mixture_angle "$MIXA" --mixture_seq 1 \
    --mixture_rama "$MIXRAMA" --coords_rama_loss_lamda "$RAMALAMDA" \
    --angle_dist "$ANGLE_DIST" --r_dist "$R_DIST" \
    --grad_clip_norm 1.0 \
    --smooth_gaussian --smooth_r 0.3 --smooth_angle 45 \
    --lr 5e-5 --n_warmup_steps 5000 --steps_decay_scale 30000 \
    --coords_angle_loss_lamda 1 --profile_loss_lamda 10 \
    --use_position_weights --cen_seg_loss_lamda 1 --oth_seg_loss_lamda 3 \
    --log_interval 30 --save_interval 50 --num_workers 4 \
    --epochs "$EPOCHS" --total_num_samples 1000000 \
    --pdb_h5_path "$PDB_H5" \
    --seq_h5_path "$SEQ_H5" \
    --rama_h5_path "$RAMA_H5" \
    "${VAL_ARGS[@]}"

echo "[ribbon] finished exit=$?"
