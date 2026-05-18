#!/usr/bin/env bash
# Local: score **3DRobot_set only** with the Yang checkpoint resolved like
# ``eval_casp14_all_local.sh`` (retrain: legacy frame, mixture_seq=1, mixture_rama auto).
#
# Default device is **mps** (Apple Silicon). Override: DEVICE=cpu|cuda
#
# Usage (repo root):
#   bash fasrc/eval_yang_retrain_3drobot_local.sh
#   YANG_RUN=yang_retrain_6594199 bash fasrc/eval_yang_retrain_3drobot_local.sh
#   TAG_YANG=my_tag OUT_YANG=eval/my_3drobot DEVICE=mps bash fasrc/eval_yang_retrain_3drobot_local.sh

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

DEVICE="${DEVICE:-mps}"

# shellcheck disable=SC1091
source "$(cd "$(dirname "$0")" && pwd)/include/resolve_yang_checkpoint.sh"

_RUN_ID="$(basename "$(dirname "$(dirname "$YANG_CKPT")")")"
TAG_YANG="${TAG_YANG:-${_RUN_ID}}"
OUT_YANG="${OUT_YANG:-eval/${_RUN_ID}_3drobot_local}"

echo "[eval_yang_3drobot] checkpoint: $YANG_CKPT | mixture_rama=$YANG_MIXTURE_RAMA"
echo "[eval_yang_3drobot] out: $OUT_YANG | tag: $TAG_YANG | device: $DEVICE"

if [[ ! -f "$YANG_CKPT" ]]; then
  echo "[eval_yang_3drobot] ERROR: missing $YANG_CKPT"
  exit 1
fi

ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/nnef}"
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  _PY="${CONDA_PREFIX}/bin/python"
elif [[ -x "${ENV_PREFIX}/bin/python" ]]; then
  _PY="${ENV_PREFIX}/bin/python"
else
  _PY="python3"
fi
PY_IM=( "$_PY" -s )

if ! "${PY_IM[@]}" -c "import numpy, pandas, scipy, torch" 2>/dev/null; then
  echo "[eval_yang_3drobot] ERROR: $_PY cannot import numpy/pandas/scipy/torch."
  exit 1
fi

if [[ "$DEVICE" == cuda* ]]; then
  if ! "${PY_IM[@]}" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "[eval_yang_3drobot] ERROR: DEVICE=cuda but CUDA not available."
    exit 1
  fi
fi

if [[ "$DEVICE" == mps ]]; then
  if ! "${PY_IM[@]}" -c "import torch; raise SystemExit(0 if torch.backends.mps.is_available() else 1)" 2>/dev/null; then
    echo "[eval_yang_3drobot] ERROR: DEVICE=mps but MPS not available — use DEVICE=cpu or a PyTorch build with MPS."
    exit 1
  fi
fi

DECOY_COMMON=(
  --decoy_sets 3DRobot_set
  --device "$DEVICE"
  --plot
  --no_skip_if_exists
)

ARCH_SHARED=(
  --seq_len 14
  --seq_type residue
  --residue_type_num 20
  --embed_size 32
  --dim 128
  --n_layers 4
  --attn_heads 4
  --mixture_r 2
  --mixture_angle 3
  --smooth_gaussian
  --smooth_r 0.3
  --smooth_angle 45
  --coords_angle_loss_lamda 1
  --profile_loss_lamda 10
  --coords_rama_loss_lamda 1
  --use_position_weights
  --cen_seg_loss_lamda 1
  --oth_seg_loss_lamda 3
)

mkdir -p "$(dirname "$OUT_YANG")"

echo "========== Yang → 3DRobot_set (mixture_rama=$YANG_MIXTURE_RAMA) =========="
"${PY_IM[@]}" nnef/scripts/evaluate_decoys.py \
  "${DECOY_COMMON[@]}" \
  --load_checkpoint "$YANG_CKPT" \
  --mixture_seq 1 \
  --mixture_rama "$YANG_MIXTURE_RAMA" \
  --legacy_local_frame \
  --exp_tag "$TAG_YANG" \
  --out_dir "$OUT_YANG" \
  "${ARCH_SHARED[@]}"
