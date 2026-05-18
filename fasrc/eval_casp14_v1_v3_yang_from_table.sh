#!/usr/bin/env bash
# Local CASP14 decoy eval for exactly the three models listed in:
#   eval/casp14_only_summary_merged.csv
#
#   • v1   — v1_pure_rama_v2_6228201  (--mixture_rama 10, no cart/ESM)
#   • v3   — v3_full_rama_v2_6229240  (+ cart + dihedral + ESM + Rama)
#   • NNEF — Yang retrain 6594199    (--mixture_seq 1, legacy frame, mixture_rama 10)
#
# Scores **all** targets in nnef/data/decoys/casp14/pdb_no_missing_residue.csv
# (e.g. T1024, T1025, T1026, T1053 after fetch_casp14_targets).  Does not run v2
# or the non-Rama v3 checkpoint.
#
# Usage (repo root):
#   chmod +x fasrc/eval_casp14_v1_v3_yang_from_table.sh
#   DEVICE=cpu DATA_DIR=~/nnef_data bash fasrc/eval_casp14_v1_v3_yang_from_table.sh
#
# On some Macs ``DEVICE=mps`` can raise a bus error mid-eval; use ``DEVICE=cpu``
# if that happens (slower but stable).
#
# Override checkpoints (paths under repo or absolute):
#   V1_RAMA_RUN=runs/v1_pure_rama_v2_6228201 \
#   V3_RAMA_RUN=runs/v3_full_rama_v2_6229240 \
#   YANG_RUN=yang_retrain_6594199 \
#     bash fasrc/eval_casp14_v1_v3_yang_from_table.sh
#
# Outputs (default): eval/<run>_casp14_multi/summary.csv
# Merge Pearson table after all three finish:
#   PYTHONPATH=nnef python nnef/scripts/evaluate_decoys.py \
#     --compare_exps eval/v1_pure_rama_v2_6228201_casp14_multi,eval/v3_full_rama_v2_6229240_casp14_multi,eval/yang_retrain_6594199_casp14_multi \
#     --out_dir eval/casp14_three_models_multi_compare

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export REPO

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

DEVICE="${DEVICE:-cpu}"
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
      echo "[casp14_3] Resolved ESM_H5=$ESM_H5"
      break
    fi
  done
fi

V1_RAMA_RUN="${V1_RAMA_RUN:-runs/v1_pure_rama_v2_6228201}"
V3_RAMA_RUN="${V3_RAMA_RUN:-runs/v3_full_rama_v2_6229240}"
export YANG_RUN="${YANG_RUN:-yang_retrain_6594199}"

# shellcheck disable=SC1091
source "$(cd "$(dirname "$0")" && pwd)/include/resolve_yang_checkpoint.sh"

_v1_base="$(basename "$V1_RAMA_RUN")"
_v3r_base="$(basename "$V3_RAMA_RUN")"
_yang_tag="$(basename "$YANG_RUN")"

OUT_V1="${OUT_V1:-eval/${_v1_base}_casp14_multi}"
OUT_V3R="${OUT_V3R:-eval/${_v3r_base}_casp14_multi}"
OUT_YANG="${OUT_YANG:-eval/${_yang_tag}_casp14_multi}"

TAG_YANG="${TAG_YANG:-$_yang_tag}"

# Prefer an explicit interpreter (must have torch + h5py + scipy):
#   EVAL_PYTHON=/path/to/python bash ...
ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/nnef}"
if [[ -n "${EVAL_PYTHON:-}" ]]; then
  _PY="$EVAL_PYTHON"
elif [[ -x "${ENV_PREFIX}/bin/python" ]]; then
  _PY="${ENV_PREFIX}/bin/python"
elif [[ -x "${HOME}/miniconda3/envs/nnef/bin/python" ]]; then
  _PY="${HOME}/miniconda3/envs/nnef/bin/python"
elif [[ -x "${HOME}/miniconda3/envs/INT303/bin/python" ]]; then
  _PY="${HOME}/miniconda3/envs/INT303/bin/python"
elif [[ -x "${HOME}/miniconda3/envs/muse/bin/python" ]]; then
  _PY="${HOME}/miniconda3/envs/muse/bin/python"
elif [[ -x "/Users/mac/miniconda3/envs/INT303/bin/python" ]]; then
  _PY="/Users/mac/miniconda3/envs/INT303/bin/python"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  _PY="${CONDA_PREFIX}/bin/python"
else
  _PY="python3"
fi
# Note: avoid ``python -s`` here — on some laptops ``h5py`` only exists in the
# user site-packages tree, and ``-s`` would hide it and break evaluate_decoys.
PY_IM=( "$_PY" )

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

if ! "${PY_IM[@]}" -c "import numpy, pandas, scipy, torch, h5py" 2>/dev/null; then
  echo "[casp14_3] ERROR: $_PY cannot import numpy/pandas/scipy/torch/h5py."
  echo "         Install e.g.:  pip install h5py   or set EVAL_PYTHON to a full nnef env."
  exit 1
fi

if [[ "$DEVICE" == cuda* ]]; then
  if ! "${PY_IM[@]}" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "[casp14_3] ERROR: DEVICE=cuda but CUDA not available."
    exit 1
  fi
fi

if [[ "$DEVICE" == mps ]]; then
  if ! "${PY_IM[@]}" -c "import torch; raise SystemExit(0 if torch.backends.mps.is_available() else 1)" 2>/dev/null; then
    echo "[casp14_3] ERROR: DEVICE=mps but MPS not available (use DEVICE=cpu or cuda)."
    exit 1
  fi
fi

run_eval() {
  echo "========== $1 =========="
  shift
  "${PY_IM[@]}" nnef/scripts/evaluate_decoys.py "$@"
}

# Re-export YANG_RUN for resolve script default path (already sourced).
if [[ ! -f "$YANG_CKPT" ]]; then
  echo "[casp14_3] ERROR: Yang checkpoint not found: $YANG_CKPT"
  echo "         Set YANG_CKPT=... or YANG_RUN=${YANG_RUN} with runs under $REPO/runs/"
  exit 1
fi

if [[ ! -f "$REPO/$V1_RAMA_RUN/models/model.pt" && ! -f "$V1_RAMA_RUN/models/model.pt" ]]; then
  echo "[casp14_3] ERROR: missing v1 checkpoint: $REPO/$V1_RAMA_RUN/models/model.pt"
  exit 1
fi
if [[ ! -f "$REPO/$V3_RAMA_RUN/models/model.pt" && ! -f "$V3_RAMA_RUN/models/model.pt" ]]; then
  echo "[casp14_3] ERROR: missing v3+Rama checkpoint: $REPO/$V3_RAMA_RUN/models/model.pt"
  exit 1
fi
if [[ ! -f "$ESM_H5" ]]; then
  echo "[casp14_3] ERROR: v3 needs ESM cache at ESM_H5 (e.g. hhsuite_esm_v2.h5). Not found."
  exit 1
fi

_v1_path="$V1_RAMA_RUN"
_v3_path="$V3_RAMA_RUN"
[[ "$_v1_path" != /* ]] && _v1_path="$REPO/$_v1_path"
[[ "$_v3_path" != /* ]] && _v3_path="$REPO/$_v3_path"

run_eval "v1_pure + Rama ($_v1_base)" \
  "${DECOY_COMMON[@]}" \
  --load_exp "$_v1_path" \
  --mixture_rama 10 \
  --exp_tag "$_v1_base" \
  --out_dir "$OUT_V1" \
  "${ARCH_SHARED[@]}"

run_eval "v3_full + Rama ($_v3r_base)" \
  "${DECOY_COMMON[@]}" \
  --load_exp "$_v3_path" \
  --mixture_rama 10 \
  --use_cart_coords \
  --use_seq_offset \
  --use_dihedral \
  --use_esm \
  --esm_h5_path "$ESM_H5" \
  --esm_dim_in 1152 \
  --esm_dim_out 32 \
  --exp_tag "$_v3r_base" \
  --out_dir "$OUT_V3R" \
  "${ARCH_SHARED[@]}"

run_eval "Yang / NNEF retrain ($TAG_YANG)" \
  "${DECOY_COMMON[@]}" \
  --load_checkpoint "$YANG_CKPT" \
  --mixture_seq 1 \
  --mixture_rama "$YANG_MIXTURE_RAMA" \
  --legacy_local_frame \
  --exp_tag "$TAG_YANG" \
  --out_dir "$OUT_YANG" \
  "${ARCH_SHARED[@]}"

echo ""
echo "[casp14_3] Done."
echo "  $OUT_V1/summary.csv"
echo "  $OUT_V3R/summary.csv"
echo "  $OUT_YANG/summary.csv"
echo ""
echo "Merge (optional):"
echo "  cd $REPO && PYTHONPATH=nnef $_PY nnef/scripts/evaluate_decoys.py \\"
echo "    --compare_exps ${OUT_V1},${OUT_V3R},${OUT_YANG} \\"
echo "    --out_dir eval/casp14_three_models_multi_compare"
