#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Single-checkpoint eval: v1_pure + Rama v2 HDF5 (train_v1_pure on rama_v2 data).
# Same inference flags as step (2/3) in eval_yang_v1_v2_casp14_3drobot.sh.
#
#   cd ~/nnef && bash fasrc/eval_v1_pure_rama_v2_casp14_3drobot.sh
#   sbatch fasrc/eval_v1_pure_rama_v2_casp14_3drobot.slurm
#
# Override:
#   V1_RAMA_RUN=runs/v1_pure_rama_v2_OTHERJOBID OUT_DIR=eval/my_tag bash fasrc/...
# ----------------------------------------------------------------------------
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
PY="${PY:-$_PY}"
PY_IM=( "$PY" -s )

V1_RAMA_RUN="${V1_RAMA_RUN:-runs/v1_pure_rama_v2_6228201}"
OUT_DIR="${OUT_DIR:-eval/v1_pure_rama_v2_6228201_casp14_3drobot}"
TAG="${TAG:-v1_pure_rama_v2_ckpt}"
DEVICE="${DEVICE:-cuda}"

cd "$REPO"

if ! "${PY_IM[@]}" -c "import numpy, pandas, scipy, torch" 2>/dev/null; then
  echo "[eval_v1_rama] ERROR: $PY cannot import deps (try PYTHONNOUSERSITE=1, fix ~/.local numpy)."
  exit 1
fi
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

echo "========== v1_pure + Rama v2: $V1_RAMA_RUN =========="
echo "[eval_v1_rama] $(date '+%Y-%m-%d %H:%M:%S') -> $OUT_DIR"
"${PY_IM[@]}" nnef/scripts/evaluate_decoys.py \
  "${DECOY_COMMON[@]}" \
  --load_exp "$V1_RAMA_RUN" \
  --mixture_rama 10 \
  --exp_tag "$TAG" \
  --out_dir "$OUT_DIR" \
  "${ARCH_SHARED[@]}"

echo "[eval_v1_rama] Done: $OUT_DIR/summary.csv"
