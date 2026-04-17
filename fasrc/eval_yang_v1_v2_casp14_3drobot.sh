#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# CASP14 + 3DRobot_set decoy inference for:
#   (1) Yang et al. pretrained exp1  (no Rama head, legacy local frame)
#   (2) Your finished v1_pure run (v2 local frame, Rama head)
#   (3) Your finished v2_run (v2 local frame + cart + offset, Rama head)
#   (4) Optional v3_full (cart + offset + ESM + dihedral) if V3_RUN is set and ESM cache exists
#
# Writes per-checkpoint:
#   eval/<tag>/summary.csv
#   eval/<tag>/plots/<decoy_set>_<pdb>_scatter.pdf  (Pearson/Spearman on figure)
#   eval/<tag>/plots/boxplot_pearson.pdf
# And merged:
#   eval/compare_yang_v1_v2_casp14_3drobot/comparison.csv  (adds v3 column when step (4) runs)
#
# Usage (edit variables in "CONFIG" then):
#   cd "$HOME/nnef" # repo root: contains nnef/ and usually runs/
#   bash fasrc/eval_yang_v1_v2_casp14_3drobot.sh
#
# Run on a GPU node (login nodes have no NVIDIA driver). Examples (FASRC):
#   salloc --partition=gpu --gres=gpu:1 --time=8:00:00
#   cd ~/nnef && bash fasrc/eval_yang_v1_v2_casp14_3drobot.sh
# Optional: CPU-only smoke test on login:  DEVICE=cpu bash fasrc/eval_yang_v1_v2_casp14_3drobot.sh
#
# v3 (after train_v3_full.slurm finishes — check: squeue no longer lists that JOBID; then):
#   V3_RUN=runs/v3_full_rama_v2_<JOBID> DATA_DIR=$HOME/nnef_data bash fasrc/eval_yang_v1_v2_casp14_3drobot.sh
#
# Slurm (recommended — logs in runs/slurm-eval-yang-v1-v2-<JOBID>.out):
#   cd ~/nnef && sbatch fasrc/eval_yang_v1_v2_casp14_3drobot.slurm
# ----------------------------------------------------------------------------
set -euo pipefail

# Do not load ~/.local/lib/python*/site-packages (user ``pip install --user``).
# Those wheels often shadow the conda env and break on FASRC (e.g. numpy +
# missing libquadmath). Override with PYTHONNOUSERSITE=0 if you truly need it.
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

# ========================= CONFIG (edit) =====================================
# Repo root (directory that contains the nnef/ package and runs/).
REPO="${REPO:-$HOME/nnef}"

# Python: prefer explicit env binary so ``bash this.sh`` always matches training
# (subshells sometimes resolve bare ``python`` to the module stack, not nnef).
ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/nnef}"
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  _PY_CAND="${CONDA_PREFIX}/bin/python"
elif [[ -x "${ENV_PREFIX}/bin/python" ]]; then
  _PY_CAND="${ENV_PREFIX}/bin/python"
else
  _PY_CAND="python"
fi
PY="${PY:-$_PY_CAND}"
# ``python -s``: never add ~/.local site-packages (broken user numpy breaks ``import torch``).
PY_IM=( "$PY" -s )

# Inference device: default cuda (requires GPU node). Use DEVICE=cpu only for debugging.
DEVICE="${DEVICE:-cuda}"

# Finished training runs (must contain models/model.pt).
V1_RUN="${V1_RUN:-runs/v1_pure_6171704}"
V2_RUN="${V2_RUN:-runs/v2_run_6160264}"

# Yang pretrained exp1: README / fold_one use repo-root params/ (not under nnef/).
# If yours lives elsewhere: YANG_CKPT=... bash fasrc/eval_yang_v1_v2_casp14_3drobot.sh
YANG_CKPT="${YANG_CKPT:-$REPO/params/exp1/models/model.pt}"

OUT_YANG="${OUT_YANG:-eval/yang_exp1_casp14_3drobot}"
TAG_YANG="${TAG_YANG:-yang_exp1}"

OUT_V1="${OUT_V1:-eval/v1_pure_casp14_3drobot}"
TAG_V1="${TAG_V1:-v1_pure_ckpt}"

OUT_V2="${OUT_V2:-eval/v2_run_casp14_3drobot}"
TAG_V2="${TAG_V2:-v2_cart_offset_ckpt}"

# Optional v3: leave empty to skip. Example: runs/v3_full_rama_v2_6223467
V3_RUN="${V3_RUN:-}"
DATA_DIR="${DATA_DIR:-$HOME/nnef_data}"
ESM_H5="${ESM_H5:-$DATA_DIR/hhsuite_esm_v2.h5}"

OUT_V3="${OUT_V3:-eval/v3_full_casp14_3drobot}"
TAG_V3="${TAG_V3:-v3_full_ckpt}"

OUT_COMPARE="${OUT_COMPARE:-eval/compare_yang_v1_v2_casp14_3drobot}"
# =============================================================================

cd "$REPO"

if ! "${PY_IM[@]}" -c "import numpy, pandas, scipy, torch" 2>/dev/null; then
  echo "[eval] ERROR: $PY cannot import numpy/pandas/scipy/torch (try: rm -rf ~/.local/lib/python3.10/site-packages/numpy*)."
  echo "        Interpreter: $("${PY_IM[@]}" -c 'import sys; print(sys.executable)')"
  echo "        This script sets PYTHONNOUSERSITE=1 so ~/.local does not shadow the env."
  echo "        If numpy still fails: remove broken user packages, e.g."
  echo "          rm -rf ~/.local/lib/python3.10/site-packages/numpy*"
  echo "        Or reinstall env: bash fasrc/env_setup.sh"
  exit 1
fi

if [[ ! -f "$YANG_CKPT" ]]; then
  echo "[eval] ERROR: Yang checkpoint not found: $YANG_CKPT"
  echo "        Set YANG_CKPT=... or place model.pt at the path above."
  exit 1
fi
if [[ ! -f "$V1_RUN/models/model.pt" ]]; then
  echo "[eval] ERROR: v1 run missing model: $V1_RUN/models/model.pt"
  exit 1
fi
if [[ ! -f "$V2_RUN/models/model.pt" ]]; then
  echo "[eval] ERROR: v2 run missing model: $V2_RUN/models/model.pt"
  exit 1
fi

if [[ "$DEVICE" == cuda* ]]; then
  if ! "${PY_IM[@]}" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "[eval] ERROR: DEVICE=$DEVICE but no GPU is visible here (you are likely on a login node, e.g. holylogin*)."
    echo "        Do not run this eval on login nodes. Example:"
    echo "          salloc --partition=gpu --gres=gpu:1 --time=8:00:00"
    echo "          cd \"$REPO\" && module load python/3.10.12-fasrc01 && source \"\$(dirname \"\$(dirname \"\$(which python)\")\")/etc/profile.d/conda.sh\""
    echo "          conda activate \"\$HOME/envs/nnef\" && bash fasrc/eval_yang_v1_v2_casp14_3drobot.sh"
    echo "        CPU-only (very slow): DEVICE=cpu bash fasrc/eval_yang_v1_v2_casp14_3drobot.sh"
    exit 1
  fi
fi

DECOY_COMMON=(
  --decoy_sets casp14,3DRobot_set
  --device "$DEVICE"
  --plot
  --no_skip_if_exists
)

# Shared architecture / loss knobs (match fasrc/train_v1_pure.slurm & train.slurm).
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

echo "========== (1/3) Yang exp1: legacy local frame, mixture_rama=0 =========="
echo "[eval] $(date '+%Y-%m-%d %H:%M:%S') starting Python; first run may sit silent 30–120s while torch/CUDA loads."
"${PY_IM[@]}" nnef/scripts/evaluate_decoys.py \
  "${DECOY_COMMON[@]}" \
  --load_checkpoint "$YANG_CKPT" \
  --mixture_seq 1 \
  --mixture_rama 0 \
  --legacy_local_frame \
  --exp_tag "$TAG_YANG" \
  --out_dir "$OUT_YANG" \
  "${ARCH_SHARED[@]}"

echo "========== (2/3) v1_pure: v2 local frame, no cart/offset =========="
echo "[eval] $(date '+%Y-%m-%d %H:%M:%S') starting Python (v1)."
"${PY_IM[@]}" nnef/scripts/evaluate_decoys.py \
  "${DECOY_COMMON[@]}" \
  --load_exp "$V1_RUN" \
  --mixture_rama 10 \
  --exp_tag "$TAG_V1" \
  --out_dir "$OUT_V1" \
  "${ARCH_SHARED[@]}"

echo "========== (3/3) v2_run: v2 local frame + cart + offset =========="
echo "[eval] $(date '+%Y-%m-%d %H:%M:%S') starting Python (v2)."
"${PY_IM[@]}" nnef/scripts/evaluate_decoys.py \
  "${DECOY_COMMON[@]}" \
  --load_exp "$V2_RUN" \
  --mixture_rama 10 \
  --use_cart_coords \
  --use_seq_offset \
  --exp_tag "$TAG_V2" \
  --out_dir "$OUT_V2" \
  "${ARCH_SHARED[@]}"

COMPARE_EXPS="$OUT_YANG,$OUT_V1,$OUT_V2"

if [[ -n "$V3_RUN" ]]; then
  if [[ ! -f "$V3_RUN/models/model.pt" ]]; then
    echo "[eval] WARN: V3_RUN set but missing $V3_RUN/models/model.pt — still training? Skip v3."
    echo "[eval]        squeue -u \"\$USER\"; when done: ls \"$V3_RUN/models/model.pt\""
  elif [[ ! -f "$ESM_H5" ]]; then
    echo "[eval] WARN: ESM cache missing: $ESM_H5 — skip v3 (build on cluster: sbatch fasrc/precompute_esm.slurm)."
  else
    echo "========== (4/4) v3_full: cart + offset + ESM + dihedral =========="
    echo "[eval] $(date '+%Y-%m-%d %H:%M:%S') starting Python (v3). ESM_H5=$ESM_H5"
    "${PY_IM[@]}" nnef/scripts/evaluate_decoys.py \
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
      --exp_tag "$TAG_V3" \
      --out_dir "$OUT_V3" \
      "${ARCH_SHARED[@]}"
    COMPARE_EXPS="$COMPARE_EXPS,$OUT_V3"
  fi
else
  echo "[eval] Skip v3 (V3_RUN unset). To include: V3_RUN=runs/v3_full_rama_v2_<JOBID> bash $0"
fi

echo "========== Compare summaries =========="
"${PY_IM[@]}" nnef/scripts/evaluate_decoys.py \
  --compare_exps "$COMPARE_EXPS" \
  --out_dir "$OUT_COMPARE"

echo "[eval] Done. Plots under $OUT_YANG/plots, $OUT_V1/plots, $OUT_V2/plots${V3_RUN:+, $OUT_V3/plots}"
echo "[eval] Merged Pearson columns: $OUT_COMPARE/comparison.csv"
