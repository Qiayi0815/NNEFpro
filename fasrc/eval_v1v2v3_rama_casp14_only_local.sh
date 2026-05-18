#!/usr/bin/env bash
# Local: **CASP14 only** (no 3DRobot) for six checkpoints — same inference flags as
# ``eval_casp14_all_local.sh`` / ``eval_yang_v1_v2_casp14_3drobot.sh``:
#
#   (no "rama" in folder name — ablation naming)
#     v1_pure:     mixture_rama 10, no cart/offset/dihedral/esm
#     v2_run:      + cart + offset (no dihedral)
#     v3_full:     + cart + offset + dihedral + ESM
#
#   (Rama v2 / dihedral rama families)
#     v1_pure_rama_v2
#     v2_dihedral_rama_v2
#     v3_full_rama_v2
#
# Outputs: ``eval/<run_basename>_casp14_only/`` (summary.csv, plots/, decoy_loss_* under data/decoys/casp14/…)
#
# Usage (repo root, conda env with torch + h5py):
#   DEVICE=mps DATA_DIR=/Library/Camille/FYP/nnef_data bash fasrc/eval_v1v2v3_rama_casp14_only_local.sh
#   DEVICE=cuda DATA_DIR=$HOME/nnef_data bash fasrc/eval_v1v2v3_rama_casp14_only_local.sh
#
# Skip subsets: SKIP_V1=1 SKIP_V2=1 … or SKIP_NO_RAMA=1 | SKIP_RAMA_TIER=1
#
# Override runs (defaults match this repo’s usual job ids):
#   V1_RUN=runs/v1_pure_6171704 V2_RUN=runs/v2_run_6160264 V3_RUN=runs/v3_full_6223467 \\
#   V1_RAMA_RUN=runs/v1_pure_rama_v2_6228201 \\
#   V2_RAMA_RUN=runs/v2_dihedral_rama_v2_6228517 \\
#   V3_RAMA_RUN=runs/v3_full_rama_v2_6229240 \\
#     bash fasrc/eval_v1v2v3_rama_casp14_only_local.sh

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

DEVICE="${DEVICE:-cuda}"
DATA_DIR="${DATA_DIR:-$HOME/nnef_data}"
ESM_H5="${ESM_H5:-$DATA_DIR/hhsuite_esm_v2.h5}"

if [[ ! -f "$ESM_H5" ]]; then
  for _try in \
    "${DATA_DIR}/hhsuite_esm_v2.h5" \
    "${REPO}/../nnef_data/hhsuite_esm_v2.h5" \
    "$(dirname "$REPO")/nnef_data/hhsuite_esm_v2.h5" \
    "${REPO}/nnef/data/hhsuite_esm_v2.h5"
  do
    if [[ -f "$_try" ]]; then
      ESM_H5="$_try"
      echo "[eval_v123_c14] Resolved ESM_H5=$ESM_H5"
      break
    fi
  done
fi

V1_RUN="${V1_RUN:-runs/v1_pure_6171704}"
V2_RUN="${V2_RUN:-runs/v2_run_6160264}"
V3_RUN="${V3_RUN:-runs/v3_full_6223467}"

V1_RAMA_RUN="${V1_RAMA_RUN:-runs/v1_pure_rama_v2_6228201}"
V2_RAMA_RUN="${V2_RAMA_RUN:-runs/v2_dihedral_rama_v2_6228517}"
V3_RAMA_RUN="${V3_RAMA_RUN:-runs/v3_full_rama_v2_6229240}"

ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/nnef}"
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  _PY="${CONDA_PREFIX}/bin/python"
elif [[ -x "${ENV_PREFIX}/bin/python" ]]; then
  _PY="${ENV_PREFIX}/bin/python"
else
  _PY="python3"
fi
PY_IM=( "$_PY" -s )

DECOY_COMMON=(
  --decoy_sets casp14
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

if ! "${PY_IM[@]}" -c "import numpy, pandas, scipy, torch" 2>/dev/null; then
  echo "[eval_v123_c14] ERROR: ${_PY} cannot import numpy/pandas/scipy/torch."
  exit 1
fi

if [[ "$DEVICE" == cuda* ]]; then
  if ! "${PY_IM[@]}" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "[eval_v123_c14] ERROR: DEVICE=cuda but CUDA not available — use DEVICE=mps or DEVICE=cpu."
    exit 1
  fi
fi

run_eval() {
  echo "========== $1 =========="
  shift
  "${PY_IM[@]}" nnef/scripts/evaluate_decoys.py "$@"
}

_out_c14() {
  local run_path="$1"
  echo "eval/$(basename "$run_path")_casp14_only"
}

# ---- tier: v1 / v2 / v3 (folder names without *_rama* in the ablation) ----
if [[ "${SKIP_NO_RAMA:-0}" != "1" ]]; then
  if [[ "${SKIP_V1:-0}" != "1" ]]; then
    if [[ ! -f "$V1_RUN/models/model.pt" ]]; then
      echo "[eval_v123_c14] SKIP v1_pure: missing $V1_RUN/models/model.pt"
    else
      _b="$(basename "$V1_RUN")"
      run_eval "v1_pure ($_b)" \
        "${DECOY_COMMON[@]}" \
        --load_exp "$V1_RUN" \
        --mixture_rama 10 \
        --exp_tag "$_b" \
        --out_dir "$(_out_c14 "$V1_RUN")" \
        "${ARCH_SHARED[@]}"
    fi
  fi

  if [[ "${SKIP_V2:-0}" != "1" ]]; then
    if [[ ! -f "$V2_RUN/models/model.pt" ]]; then
      echo "[eval_v123_c14] SKIP v2_run (cart+offset): missing $V2_RUN/models/model.pt"
    else
      _b="$(basename "$V2_RUN")"
      run_eval "v2_run cart+offset ($_b)" \
        "${DECOY_COMMON[@]}" \
        --load_exp "$V2_RUN" \
        --mixture_rama 10 \
        --use_cart_coords \
        --use_seq_offset \
        --exp_tag "$_b" \
        --out_dir "$(_out_c14 "$V2_RUN")" \
        "${ARCH_SHARED[@]}"
    fi
  fi

  if [[ "${SKIP_V3:-0}" != "1" ]]; then
    if [[ ! -f "$V3_RUN/models/model.pt" ]]; then
      echo "[eval_v123_c14] SKIP v3_full: missing $V3_RUN/models/model.pt"
    elif [[ ! -f "$ESM_H5" ]]; then
      echo "[eval_v123_c14] SKIP v3_full: ESM cache missing: $ESM_H5"
    else
      _b="$(basename "$V3_RUN")"
      run_eval "v3_full ($_b)" \
        "${DECOY_COMMON[@]}" \
        --load_exp "$V3_RUN" \
        --mixture_rama 10 \
        --use_cart_coords \
        --use_seq_offset \
        --use_dihedral \
        --use_esm \
        --esm_h5_path "$ESM_H5" \
        --esm_dim_in 1152 \
        --esm_dim_out 32 \
        --exp_tag "$_b" \
        --out_dir "$(_out_c14 "$V3_RUN")" \
        "${ARCH_SHARED[@]}"
    fi
  fi
else
  echo "[eval_v123_c14] SKIP_NO_RAMA=1 (skip v1/v2/v3 non-rama tier)"
fi

# ---- tier: *rama* / dihedral+rama families ----
if [[ "${SKIP_RAMA_TIER:-0}" != "1" ]]; then
  if [[ "${SKIP_V1R:-0}" != "1" ]]; then
    if [[ ! -f "$V1_RAMA_RUN/models/model.pt" ]]; then
      echo "[eval_v123_c14] SKIP v1_pure_rama_v2: missing $V1_RAMA_RUN/models/model.pt"
    else
      _b="$(basename "$V1_RAMA_RUN")"
      run_eval "v1_pure + Rama v2 ($_b)" \
        "${DECOY_COMMON[@]}" \
        --load_exp "$V1_RAMA_RUN" \
        --mixture_rama 10 \
        --exp_tag "$_b" \
        --out_dir "$(_out_c14 "$V1_RAMA_RUN")" \
        "${ARCH_SHARED[@]}"
    fi
  fi

  if [[ "${SKIP_V2R:-0}" != "1" ]]; then
    if [[ ! -f "$V2_RAMA_RUN/models/model.pt" ]]; then
      echo "[eval_v123_c14] SKIP v2_dihedral+Rama: missing $V2_RAMA_RUN/models/model.pt"
    else
      _b="$(basename "$V2_RAMA_RUN")"
      run_eval "v2_dihedral + Rama v2 ($_b)" \
        "${DECOY_COMMON[@]}" \
        --load_exp "$V2_RAMA_RUN" \
        --mixture_rama 10 \
        --use_cart_coords \
        --use_seq_offset \
        --use_dihedral \
        --exp_tag "$_b" \
        --out_dir "$(_out_c14 "$V2_RAMA_RUN")" \
        "${ARCH_SHARED[@]}"
    fi
  fi

  if [[ "${SKIP_V3R:-0}" != "1" ]]; then
    if [[ ! -f "$V3_RAMA_RUN/models/model.pt" ]]; then
      echo "[eval_v123_c14] SKIP v3_full_rama: missing $V3_RAMA_RUN/models/model.pt"
    elif [[ ! -f "$ESM_H5" ]]; then
      echo "[eval_v123_c14] SKIP v3_full_rama: ESM cache missing: $ESM_H5"
    else
      _b="$(basename "$V3_RAMA_RUN")"
      run_eval "v3_full + Rama v2 ($_b)" \
        "${DECOY_COMMON[@]}" \
        --load_exp "$V3_RAMA_RUN" \
        --mixture_rama 10 \
        --use_cart_coords \
        --use_seq_offset \
        --use_dihedral \
        --use_esm \
        --esm_h5_path "$ESM_H5" \
        --esm_dim_in 1152 \
        --esm_dim_out 32 \
        --exp_tag "$_b" \
        --out_dir "$(_out_c14 "$V3_RAMA_RUN")" \
        "${ARCH_SHARED[@]}"
    fi
  fi
else
  echo "[eval_v123_c14] SKIP_RAMA_TIER=1 (skip v1r/v2r/v3r)"
fi

echo "[eval_v123_c14] Done. Summaries: eval/*_casp14_only/summary.csv"
