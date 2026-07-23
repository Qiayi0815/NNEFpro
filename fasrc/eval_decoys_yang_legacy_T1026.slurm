#!/usr/bin/env bash
#SBATCH --job-name=nnef-decoys-yang-legacy
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=runs/slurm-decoys-yang-legacy-%j.out
#SBATCH --error=runs/slurm-decoys-yang-legacy-%j.out
#
# Decoy correlation (energy vs GDT_TS) for yang_legacy on CASP14 T1026.
# Complements the mode2 (native-basin stability) evaluation; this asks
# whether the learned energy RANKS decoys correctly, which is the
# primary metric from the Yang-2022 paper.
#
# Architecture flags must match fasrc/train_yang_legacy.slurm exactly:
#   --mixture_seq 1, --mixture_rama 0, --legacy_local_frame.
# A strict-load mismatch here means you edited one file but not the other.
#
# Submit:
#   sbatch fasrc/eval_decoys_yang_legacy_T1026.slurm
#
# Override (other target or another yang_legacy checkpoint):
#   sbatch --export=ALL,TARGET=T1053,LOAD_EXP=runs/yang_legacy_XXXXXXX \
#       fasrc/eval_decoys_yang_legacy_T1026.slurm

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/nnef}"
DATA_DIR="${DATA_DIR:-$HOME/nnef_data}"
LOAD_EXP="${LOAD_EXP:-runs/yang_legacy_7317784}"
TARGET="${TARGET:-T1026}"
# Angle/radial distribution ablation (see options.py --angle_dist/--r_dist).
# MUST match how LOAD_EXP was trained -- a mismatch silently scores decoys
# with the wrong distribution family, no error.
ANGLE_DIST="${ANGLE_DIST:-gaussian}"
R_DIST="${R_DIST:-gaussian}"

module purge
module load python/3.10.12-fasrc01

PYTHON_BIN="$(command -v python)"
CONDA_HOOK="$(dirname "$(dirname "$PYTHON_BIN")")/etc/profile.d/conda.sh"
[ -f "$CONDA_HOOK" ] || { echo "[decoys_yl] conda.sh not found at $CONDA_HOOK"; exit 1; }
# shellcheck disable=SC1090
source "$CONDA_HOOK"
set +u
conda activate "$ENV_PREFIX"
set -u

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

PY="${ENV_PREFIX}/bin/python"
[ -x "$PY" ] || { echo "[decoys_yl] ERROR: no $PY"; exit 1; }
"$PY" -s -c "import torch; print('[decoys_yl] cuda', torch.cuda.is_available())" || exit 1

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TOKENIZERS_PARALLELISM=false

cd "$SLURM_SUBMIT_DIR"

if [[ ! -f "$LOAD_EXP/models/model.pt" ]]; then
  echo "[decoys_yl] ERROR: missing $LOAD_EXP/models/model.pt"
  exit 1
fi

RUN_BASE="$(basename "$LOAD_EXP")"
OUT_DIR="${OUT_DIR:-eval/${RUN_BASE}_casp14_${TARGET}}"
TAG="${TAG:-${RUN_BASE}_${TARGET}}"

mkdir -p "$OUT_DIR"

echo "----- decoy correlation: yang_legacy / CASP14 / $TARGET -----"
hostname
nvidia-smi
echo "LOAD_EXP=$LOAD_EXP  TARGET=$TARGET  OUT_DIR=$OUT_DIR  ANGLE_DIST=$ANGLE_DIST  R_DIST=$R_DIST"
echo "-------------------------------------------------------------"

"$PY" -s nnef/scripts/evaluate_decoys.py \
  --load_exp "$LOAD_EXP" \
  --mixture_seq 1 \
  --mixture_rama 0 \
  --legacy_local_frame \
  --decoy_sets casp14 \
  --targets "$TARGET" \
  --device cuda \
  --plot \
  --no_skip_if_exists \
  --exp_tag "$TAG" \
  --out_dir "$OUT_DIR" \
  --seq_len 14 \
  --seq_type residue \
  --residue_type_num 20 \
  --embed_size 32 \
  --dim 128 \
  --n_layers 4 \
  --attn_heads 4 \
  --mixture_r 2 \
  --mixture_angle 3 \
  --angle_dist "$ANGLE_DIST" \
  --r_dist "$R_DIST" \
  --smooth_gaussian \
  --smooth_r 0.3 \
  --smooth_angle 45 \
  --coords_angle_loss_lamda 1 \
  --profile_loss_lamda 10 \
  --coords_rama_loss_lamda 1 \
  --use_position_weights \
  --cen_seg_loss_lamda 1 \
  --oth_seg_loss_lamda 3

echo "[decoys_yl] Done. summary + scatter under $OUT_DIR/"
