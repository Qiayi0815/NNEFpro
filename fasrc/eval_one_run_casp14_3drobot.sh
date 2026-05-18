#!/usr/bin/env bash
# Single checkpoint: CASP14 + 3DRobot_set (unique out_dir / exp_tag per LOAD_EXP).
#
# Modes (EVAL_MODE):
#   v1_esm       — v1 + ESM + Rama head (train_v1_esm.slurm); no cart/offset/dihedral
#   v2_dihedral  — v2 cart + offset + dihedral + Rama head (train_v2_dihedral.slurm)
#   v3_full      — v3 + ESM + dihedral + cart + offset (train_v3_full.slurm)
#
# Usage:
#   EVAL_MODE=v1_esm      LOAD_EXP=runs/v1_esm_rama_v2_6642146      bash fasrc/eval_one_run_casp14_3drobot.sh
#   EVAL_MODE=v2_dihedral LOAD_EXP=runs/v2_dihedral_rama_v2_6228517 bash fasrc/eval_one_run_casp14_3drobot.sh
#   EVAL_MODE=v3_full     LOAD_EXP=runs/v3_full_6223467             bash fasrc/eval_one_run_casp14_3drobot.sh
#   EVAL_MODE=v3_full     LOAD_EXP=runs/v3_full_rama_v2_6300000      bash fasrc/eval_one_run_casp14_3drobot.sh
#
set -euo pipefail

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

EVAL_MODE="${EVAL_MODE:?set EVAL_MODE to v1_esm, v2_dihedral or v3_full}"
LOAD_EXP="${LOAD_EXP:?set LOAD_EXP e.g. runs/v2_dihedral_rama_v2_6228517}"
DATA_DIR="${DATA_DIR:-$HOME/nnef_data}"
ESM_H5="${ESM_H5:-$DATA_DIR/hhsuite_esm_v2.h5}"

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
_RUN_BASE="$(basename "$LOAD_EXP")"
OUT_DIR="${OUT_DIR:-eval/${_RUN_BASE}_casp14_3drobot}"
TAG="${TAG:-${_RUN_BASE}}"

cd "$REPO"

if [[ ! -f "$LOAD_EXP/models/model.pt" ]]; then
  echo "[eval_one_run] ERROR: missing $LOAD_EXP/models/model.pt"
  exit 1
fi

if [[ "$EVAL_MODE" == v3_full || "$EVAL_MODE" == v1_esm ]]; then
  if [[ ! -f "$ESM_H5" ]]; then
    echo "[eval_one_run] ERROR: $EVAL_MODE needs ESM cache: $ESM_H5"
    exit 1
  fi
fi

if [[ "$DEVICE" == cuda* ]]; then
  if ! "${PY_IM[@]}" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "[eval_one_run] ERROR: need a GPU node for DEVICE=cuda"
    exit 1
  fi
fi

DECOY_SETS="${DECOY_SETS:-casp14,3DRobot_set}"
DECOY_COMMON=(
  --decoy_sets "$DECOY_SETS"
  --device "$DEVICE"
  --plot
)
# SKIP_EXISTING=1 → reuse prior per-target CSVs (e.g. after a TIMEOUT re-run).
if [[ "${SKIP_EXISTING:-0}" != "1" ]]; then
  DECOY_COMMON+=( --no_skip_if_exists )
fi

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

CMD=(
  "${PY_IM[@]}" nnef/scripts/evaluate_decoys.py
  "${DECOY_COMMON[@]}"
  --load_exp "$LOAD_EXP"
  --mixture_rama 10
  --exp_tag "$TAG"
  --out_dir "$OUT_DIR"
  "${ARCH_SHARED[@]}"
)

case "$EVAL_MODE" in
  v1_esm)
    CMD+=(
      --use_esm
      --esm_h5_path "$ESM_H5"
      --esm_dim_in 1152
      --esm_dim_out 32
    )
    ;;
  v2_dihedral)
    CMD+=( --use_cart_coords --use_seq_offset --use_dihedral )
    ;;
  v3_full)
    CMD+=(
      --use_cart_coords --use_seq_offset --use_dihedral
      --use_esm
      --esm_h5_path "$ESM_H5"
      --esm_dim_in 1152
      --esm_dim_out 32
    )
    ;;
  *)
    echo "[eval_one_run] ERROR: EVAL_MODE must be v1_esm, v2_dihedral or v3_full, got: $EVAL_MODE"
    exit 1
    ;;
esac

echo "========== eval_one_run CASP14+3DRobot =========="
echo "[eval_one_run] $(date '+%Y-%m-%d %H:%M:%S') EVAL_MODE=$EVAL_MODE load_exp=$LOAD_EXP"
echo "[eval_one_run] out_dir=$OUT_DIR tag=$TAG"

"${CMD[@]}"

echo "[eval_one_run] Wrote $OUT_DIR/summary.csv"
