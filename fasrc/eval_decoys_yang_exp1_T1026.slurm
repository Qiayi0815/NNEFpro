#!/usr/bin/env bash
#SBATCH --job-name=nnef-decoys-yang-exp1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=runs/slurm-decoys-yang-exp1-%j.out
#SBATCH --error=runs/slurm-decoys-yang-exp1-%j.out
#
# Baseline: score the PAPER'S released Yang-2022 checkpoint (params/exp1)
# on CASP14 T1026. Same architecture as yang_legacy_7317784:
#   --mixture_seq 1  --mixture_rama 0  --legacy_local_frame
# but uses --load_checkpoint instead of --load_exp (params/exp1 has no
# training log tree, just a bare models/model.pt).
#
# Purpose: fills the missing cell in the T1026 table —
#   yang_legacy_7317784 on T1026 gave Pearson=-0.424, which is odd given
#   yang_retrain_6594199 (extra rama head) got -0.704 on the same target.
#   And yang_legacy on T1053 = -0.447 vs paper's -0.902, so the retrain
#   is likely undertrained / data-weak rather than the target being hard.
#   This job confirms by scoring the paper ckpt on T1026.
#
# Submit:
#   sbatch fasrc/eval_decoys_yang_exp1_T1026.slurm
#
# Override (other target or non-default checkpoint):
#   sbatch --export=ALL,TARGET=T1053 fasrc/eval_decoys_yang_exp1_T1026.slurm
#   sbatch --export=ALL,YANG_CKPT=/path/to/model.pt fasrc/eval_decoys_yang_exp1_T1026.slurm

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/nnef}"
DATA_DIR="${DATA_DIR:-$HOME/nnef_data}"
YANG_CKPT="${YANG_CKPT:-params/exp1/models/model.pt}"
TARGET="${TARGET:-T1026}"

module purge
module load python/3.10.12-fasrc01

PYTHON_BIN="$(command -v python)"
CONDA_HOOK="$(dirname "$(dirname "$PYTHON_BIN")")/etc/profile.d/conda.sh"
[ -f "$CONDA_HOOK" ] || { echo "[decoys_yx] conda.sh not found at $CONDA_HOOK"; exit 1; }
# shellcheck disable=SC1090
source "$CONDA_HOOK"
set +u
conda activate "$ENV_PREFIX"
set -u

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

PY="${ENV_PREFIX}/bin/python"
[ -x "$PY" ] || { echo "[decoys_yx] ERROR: no $PY"; exit 1; }
"$PY" -s -c "import torch; print('[decoys_yx] cuda', torch.cuda.is_available())" || exit 1

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TOKENIZERS_PARALLELISM=false

cd "$SLURM_SUBMIT_DIR"

if [[ ! -f "$YANG_CKPT" ]]; then
  echo "[decoys_yx] ERROR: missing $YANG_CKPT"
  echo "[decoys_yx]   expected: the paper's released exp1 weights."
  echo "[decoys_yx]   if absent on cluster, sync from local:"
  echo "[decoys_yx]     rsync -av params/exp1/ qzha@login.rc.fas.harvard.edu:/n/home03/qzha/nnef/params/exp1/"
  exit 1
fi

OUT_DIR="${OUT_DIR:-eval/yang_exp1_casp14_${TARGET}}"
TAG="${TAG:-yang_exp1_${TARGET}}"

mkdir -p "$OUT_DIR"

echo "----- decoy correlation: yang_exp1 (paper) / CASP14 / $TARGET -----"
hostname
nvidia-smi
echo "YANG_CKPT=$YANG_CKPT  TARGET=$TARGET  OUT_DIR=$OUT_DIR"
echo "-------------------------------------------------------------------"

"$PY" -s nnef/scripts/evaluate_decoys.py \
  --load_checkpoint "$YANG_CKPT" \
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
  --smooth_gaussian \
  --smooth_r 0.3 \
  --smooth_angle 45 \
  --coords_angle_loss_lamda 1 \
  --profile_loss_lamda 10 \
  --coords_rama_loss_lamda 1 \
  --use_position_weights \
  --cen_seg_loss_lamda 1 \
  --oth_seg_loss_lamda 3

echo "[decoys_yx] Done. summary + scatter under $OUT_DIR/"
