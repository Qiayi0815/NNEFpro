#!/usr/bin/env bash
# Single-checkpoint eval: v1_pure + Rama v2 HDF5 (train_v1_pure on rama_v2 data).
# Uses its own --exp_tag / --out_dir / decoy_loss_* so it does not clash with
# eval_yang_v1_v2_casp14_3drobot.sh (v1_pure_ckpt, v2_cart_offset_ckpt, etc.).
#
# Usage (GPU node):
#   cd ~/nnef && bash fasrc/eval_v1_pure_rama_v2_only.sh
#
# Optional:
#   V1_RAMA_RUN=runs/v1_pure_rama_v2_6228201 bash fasrc/eval_v1_pure_rama_v2_only.sh
set -euo pipefail

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

REPO="${REPO:-$HOME/nnef}"
ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/nnef}"
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  _PY="${CONDA_PREFIX}/bin/python"
elif [[ -x "${ENV_PREFIX}/bin/python" ]]; then
  _PY="${ENV_PREFIX}/bin/python"
else
  _PY="python"
fi
PY_IM=( "$_PY" -s )

DEVICE="${DEVICE:-cuda}"
V1_RAMA_RUN="${V1_RAMA_RUN:-runs/v1_pure_rama_v2_6228201}"
_RUN_BASE="$(basename "$V1_RAMA_RUN")"
OUT_DIR="${OUT_DIR:-eval/${_RUN_BASE}_casp14_3drobot}"
TAG="${TAG:-${_RUN_BASE}}"

cd "$REPO"

if [[ ! -f "$V1_RAMA_RUN/models/model.pt" ]]; then
  echo "[eval_v1_rama] ERROR: missing $V1_RAMA_RUN/models/model.pt"
  exit 1
fi

if [[ "$DEVICE" == cuda* ]]; then
  if ! "${PY_IM[@]}" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "[eval_v1_rama] ERROR: need a GPU node for DEVICE=cuda"
    exit 1
  fi
fi

DECOY_COMMON=(
  --decoy_sets casp14,3DRobot_set
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

echo "========== v1_pure + Rama v2 only =========="
echo "[eval_v1_rama] $(date '+%Y-%m-%d %H:%M:%S') load_exp=$V1_RAMA_RUN out_dir=$OUT_DIR tag=$TAG"

"${PY_IM[@]}" nnef/scripts/evaluate_decoys.py \
  "${DECOY_COMMON[@]}" \
  --load_exp "$V1_RAMA_RUN" \
  --mixture_rama 10 \
  --exp_tag "$TAG" \
  --out_dir "$OUT_DIR" \
  "${ARCH_SHARED[@]}"

echo "[eval_v1_rama] Wrote $OUT_DIR/summary.csv"
