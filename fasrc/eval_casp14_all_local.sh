#!/usr/bin/env bash
# Local / laptop: re-score **CASP14 only** (no 3DRobot) for Yang vs your v1–v3 runs.
# Uses the same **two preprocessing families** as fasrc/eval_yang_v1_v2_casp14_3drobot.sh:
#
#   (A) Yang baseline (legacy local frame, --mixture_seq 1):
#       • Retrained (train_yang_style.slurm): runs/yang_retrain_<JOBID>/models/model.pt
#         → --mixture_rama 10 (auto if path is not params/exp1).
#       • Released paper ckpt: params/exp1/models/model.pt → --mixture_rama 0.
#       Resolve order: YANG_CKPT, or YANG_RUN, or newest runs/yang_retrain_*, else params/exp1.
#       Pull from cluster:  bash fasrc/pull_yang_retrain_from_cluster.sh <JOBID>
#
#   (B) Your trained runs (--load_exp runs/.../models/model.pt):
#       v1_pure + Rama v2:        --mixture_rama 10  (no cart / offset / dihedral / esm)
#       v2_dihedral + Rama v2:    + --use_cart_coords --use_seq_offset --use_dihedral
#       v3_full (± Rama in name): + --use_esm --esm_h5_path … (requires hhsuite_esm_v2.h5)
#
# Usage (from nnef repo root, conda env with torch activated):
#   cd /path/to/nnef
#   chmod +x fasrc/eval_casp14_all_local.sh
#   # CPU smoke / laptop without GPU:
#   DEVICE=cpu bash fasrc/eval_casp14_all_local.sh
#   # With GPU + data under ~/nnef_data:
#   DEVICE=cuda DATA_DIR=$HOME/nnef_data bash fasrc/eval_casp14_all_local.sh
#
# Optional: skip models (1 = skip)
#   SKIP_YANG=1 bash fasrc/eval_casp14_all_local.sh
#   SKIP_V1=1 SKIP_V2=1 ...
#
# Override Yang (optional):
#   YANG_RUN=yang_retrain_6234567 bash fasrc/eval_casp14_all_local.sh
#   YANG_CKPT=$PWD/runs/yang_retrain_6234567/models/model.pt \\
#   YANG_MIXTURE_RAMA=10 bash fasrc/eval_casp14_all_local.sh
# Other checkpoints:
#   V1_RAMA_RUN=runs/v1_pure_rama_v2_6228201 bash fasrc/eval_casp14_all_local.sh

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

DEVICE="${DEVICE:-cuda}"
DATA_DIR="${DATA_DIR:-$HOME/nnef_data}"
ESM_H5="${ESM_H5:-$DATA_DIR/hhsuite_esm_v2.h5}"
# `hhsuite_esm_v2.h5` is ~5–8 GB (precompute_esm on cluster); laptops often keep it elsewhere or only on FASRC.
# If the default path is missing, try common locations before skipping v3.
if [[ ! -f "$ESM_H5" ]]; then
  for _try in \
    "${DATA_DIR}/hhsuite_esm_v2.h5" \
    "${REPO}/../nnef_data/hhsuite_esm_v2.h5" \
    "$(dirname "$REPO")/nnef_data/hhsuite_esm_v2.h5" \
    "${REPO}/nnef/data/hhsuite_esm_v2.h5"
  do
    if [[ -f "$_try" ]]; then
      ESM_H5="$_try"
      echo "[eval_casp14_local] Resolved ESM_H5=$ESM_H5"
      break
    fi
  done
fi
if [[ ! -f "$ESM_H5" ]]; then
  echo "[eval_casp14_local] WARN: no hhsuite_esm_v2.h5 found. v3 eval needs it."
  echo "         Set:  export ESM_H5=/path/to/hhsuite_esm_v2.h5"
  echo "         Or:  rsync from cluster: rsync -avh qzha@login.rc.fas.harvard.edu:/n/home03/qzha/nnef_data/hhsuite_esm_v2.h5 \"\$HOME/nnef_data/\""
fi

# shellcheck disable=SC1091
source "$(cd "$(dirname "$0")" && pwd)/include/resolve_yang_checkpoint.sh"
echo "[eval_casp14_local] Yang: $YANG_CKPT | mixture_rama=$YANG_MIXTURE_RAMA"

V1_RAMA_RUN="${V1_RAMA_RUN:-runs/v1_pure_rama_v2_6228201}"
V2D_RAMA_RUN="${V2D_RAMA_RUN:-runs/v2_dihedral_rama_v2_6228517}"
V3_RUN="${V3_RUN:-runs/v3_full_6223467}"
V3_RAMA_RUN="${V3_RAMA_RUN:-runs/v3_full_rama_v2_6229240}"

TAG_YANG="${TAG_YANG:-yang_exp1}"
OUT_YANG="${OUT_YANG:-eval/yang_exp1_casp14_local}"

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
  echo "[eval_casp14_local] ERROR: $PY cannot import numpy/pandas/scipy/torch."
  exit 1
fi

if [[ "$DEVICE" == cuda* ]]; then
  if ! "${PY_IM[@]}" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "[eval_casp14_local] WARN: DEVICE=cuda but CUDA not available — use DEVICE=cpu or a GPU machine."
    exit 1
  fi
fi

if [[ "$DEVICE" == mps ]]; then
  if ! "${PY_IM[@]}" -c "import torch; raise SystemExit(0 if torch.backends.mps.is_available() else 1)" 2>/dev/null; then
    echo "[eval_casp14_local] ERROR: DEVICE=mps but MPS is not available (need Apple Silicon + PyTorch with MPS)."
    exit 1
  fi
fi

run_eval() {
  echo "========== $1 =========="
  shift
  "${PY_IM[@]}" nnef/scripts/evaluate_decoys.py "$@"
}

if [[ "${SKIP_YANG:-0}" != "1" ]]; then
  if [[ ! -f "$YANG_CKPT" ]]; then
    echo "[eval_casp14_local] SKIP Yang: checkpoint not found: $YANG_CKPT"
  else
    mkdir -p "$(dirname "$OUT_YANG")"
    run_eval "Yang (legacy frame, mixture_rama=$YANG_MIXTURE_RAMA)" \
      "${DECOY_COMMON[@]}" \
      --load_checkpoint "$YANG_CKPT" \
      --mixture_seq 1 \
      --mixture_rama "$YANG_MIXTURE_RAMA" \
      --legacy_local_frame \
      --exp_tag "$TAG_YANG" \
      --out_dir "$OUT_YANG" \
      "${ARCH_SHARED[@]}"
  fi
else
  echo "[eval_casp14_local] SKIP_YANG=1"
fi

_v1_base="$(basename "$V1_RAMA_RUN")"
if [[ "${SKIP_V1:-0}" != "1" ]]; then
  if [[ ! -f "$V1_RAMA_RUN/models/model.pt" ]]; then
    echo "[eval_casp14_local] SKIP v1+Rama: missing $V1_RAMA_RUN/models/model.pt"
  else
    run_eval "v1_pure + Rama v2 ($_v1_base)" \
      "${DECOY_COMMON[@]}" \
      --load_exp "$V1_RAMA_RUN" \
      --mixture_rama 10 \
      --exp_tag "$_v1_base" \
      --out_dir "eval/${_v1_base}_casp14_local" \
      "${ARCH_SHARED[@]}"
  fi
fi

_v2_base="$(basename "$V2D_RAMA_RUN")"
if [[ "${SKIP_V2:-0}" != "1" ]]; then
  if [[ ! -f "$V2D_RAMA_RUN/models/model.pt" ]]; then
    echo "[eval_casp14_local] SKIP v2 dihedral+Rama: missing $V2D_RAMA_RUN/models/model.pt"
  else
    run_eval "v2_dihedral + Rama v2 ($_v2_base)" \
      "${DECOY_COMMON[@]}" \
      --load_exp "$V2D_RAMA_RUN" \
      --mixture_rama 10 \
      --use_cart_coords \
      --use_seq_offset \
      --use_dihedral \
      --exp_tag "$_v2_base" \
      --out_dir "eval/${_v2_base}_casp14_local" \
      "${ARCH_SHARED[@]}"
  fi
fi

_v3_base="$(basename "$V3_RUN")"
if [[ "${SKIP_V3:-0}" != "1" ]]; then
  if [[ ! -f "$V3_RUN/models/model.pt" ]]; then
    echo "[eval_casp14_local] SKIP v3 (no-Rama name): missing $V3_RUN/models/model.pt"
  elif [[ ! -f "$ESM_H5" ]]; then
    echo "[eval_casp14_local] SKIP v3: ESM cache missing: $ESM_H5"
  else
    run_eval "v3_full ($_v3_base)" \
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
      --exp_tag "$_v3_base" \
      --out_dir "eval/${_v3_base}_casp14_local" \
      "${ARCH_SHARED[@]}"
  fi
fi

_v3r_base="$(basename "$V3_RAMA_RUN")"
if [[ "${SKIP_V3_RAMA:-0}" != "1" ]]; then
  if [[ ! -f "$V3_RAMA_RUN/models/model.pt" ]]; then
    echo "[eval_casp14_local] SKIP v3+Rama: missing $V3_RAMA_RUN/models/model.pt"
  elif [[ ! -f "$ESM_H5" ]]; then
    echo "[eval_casp14_local] SKIP v3+Rama: ESM cache missing: $ESM_H5"
  else
    run_eval "v3_full + Rama v2 ($_v3r_base)" \
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
      --exp_tag "$_v3r_base" \
      --out_dir "eval/${_v3r_base}_casp14_local" \
      "${ARCH_SHARED[@]}"
  fi
fi

echo "[eval_casp14_local] Done. summary.csv under eval/*_casp14_local/"
echo "[eval_casp14_local] decoy_loss_* CSVs: nnef/data/decoys/casp14/decoy_loss_<exp_tag>/"
