#!/usr/bin/env bash
#SBATCH --job-name=nnef-diag-yx
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=runs/slurm-diag-yx-%j.out
#SBATCH --error=runs/slurm-diag-yx-%j.out
#
# Root-cause diagnostic for yang_legacy_7317784 underperformance.
#
# Runs nnef/scripts/diagnose_train_loss.py on TWO checkpoints against the
# same cullpdb training h5, in a single job so their losses are strictly
# comparable (same sampler RNG seed, same nodes, same worker pool).
#
#   A. params/exp1/models/model.pt        (paper's released checkpoint)
#   B. runs/yang_legacy_7317784/models/model.pt   (our retrain)
#
# Expected outcomes:
#   losses(A) ≈ losses(B) (~985 total)    -> cullpdb h5 is paper-compatible;
#                                            training code diverged. Next:
#                                            diff our trainer/loss against
#                                            yang's original repo.
#   losses(A) << losses(B)                -> cullpdb h5 differs from paper's
#                                            training data (rebuild drifted).
#                                            Next: compare our build_* scripts
#                                            to yang's preprocessing.
#   losses(A) >> losses(B)                -> params/exp1 is incompatible with
#                                            our current inference code path
#                                            (layer rename, flag mismatch).
#                                            Next: inspect strict-load keys.
#
# Submit:
#   sbatch fasrc/diagnose_yang_exp1_on_cullpdb.slurm

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/nnef}"
DATA_DIR="${DATA_DIR:-$HOME/nnef_data}"
N_BATCHES="${N_BATCHES:-30}"

module purge
module load python/3.10.12-fasrc01

PYTHON_BIN="$(command -v python)"
CONDA_HOOK="$(dirname "$(dirname "$PYTHON_BIN")")/etc/profile.d/conda.sh"
[ -f "$CONDA_HOOK" ] || { echo "[diag_yx] conda.sh not found at $CONDA_HOOK"; exit 1; }
# shellcheck disable=SC1090
source "$CONDA_HOOK"
set +u
conda activate "$ENV_PREFIX"
set -u

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

PY="${ENV_PREFIX}/bin/python"
[ -x "$PY" ] || { echo "[diag_yx] ERROR: no $PY"; exit 1; }
"$PY" -s -c "import torch; print('[diag_yx] cuda', torch.cuda.is_available())" || exit 1

cd "$SLURM_SUBMIT_DIR"

PDB_H5="$DATA_DIR/hhsuite_CB_cullpdb.h5"
SEQ_H5="$DATA_DIR/hhsuite_pdb_seq_v2.h5"
NO_RAMA="$DATA_DIR/__NO_RAMA_SENTINEL__.h5"

[[ -f "$PDB_H5" ]] || { echo "[diag_yx] ERROR: missing $PDB_H5"; exit 1; }
[[ -f "params/exp1/models/model.pt" ]] || { echo "[diag_yx] ERROR: missing params/exp1/models/model.pt"; exit 1; }
[[ -f "runs/yang_legacy_7317784/models/model.pt" ]] || { echo "[diag_yx] ERROR: missing yang_legacy ckpt"; exit 1; }

# Shared architecture flags — must match train_yang_legacy.slurm exactly.
ARCH=(
  --data_flag hhsuite_CB_v2_pdb_list.csv
  --seq_type residue --residue_type_num 20
  --chimeric
  --seq_len 14 --batch_size 1000
  --embed_size 32 --dim 128 --n_layers 4 --attn_heads 4
  --mixture_r 2 --mixture_angle 3
  --mixture_seq 1 --mixture_rama 0
  --legacy_local_frame
  --smooth_gaussian --smooth_r 0.3 --smooth_angle 45
  --coords_angle_loss_lamda 1 --profile_loss_lamda 10
  --use_position_weights --cen_seg_loss_lamda 1 --oth_seg_loss_lamda 3
  --pdb_h5_path "$PDB_H5"
  --seq_h5_path "$SEQ_H5"
  --rama_h5_path "$NO_RAMA"
  --num_workers 4
  --device cuda
  --n_batches "$N_BATCHES"
)

echo "================================================================"
echo "[diag_yx] A. PAPER CHECKPOINT params/exp1/models/model.pt"
echo "================================================================"
"$PY" -s nnef/scripts/diagnose_train_loss.py \
  --load_checkpoint params/exp1/models/model.pt \
  "${ARCH[@]}"

echo
echo "================================================================"
echo "[diag_yx] B. OUR RETRAIN runs/yang_legacy_7317784/models/model.pt"
echo "================================================================"
"$PY" -s nnef/scripts/diagnose_train_loss.py \
  --load_checkpoint runs/yang_legacy_7317784/models/model.pt \
  "${ARCH[@]}"

echo "[diag_yx] Done. Compare the two per-component tables above."
