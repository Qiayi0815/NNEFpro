#!/usr/bin/env bash
# Run one Langevin-MD evaluation (one model, one target, one seed, one mode).
# Dispatches architecture flags by MODEL_KEY so the slurm wrappers stay simple.
#
# Required env:
#   MODEL_KEY   yang_retrain | yang_legacy | v1_rama | v1_rama_esm
#   TARGET      CASP14 target id (e.g. T1053)
#   MD_MODE     native | fold | decoy
#   SEED        integer seed
#
# Optional env:
#   L                Langevin step count          (default 100000; smoke: set 10000)
#   LR               lr (= alpha)                 (default: mode-specific in md_eval.py)
#   T_NOISE          noise scale (= beta)         (default: mode-specific)
#   X_TYPE           cart|internal|int_fast|mixed|mix_fast (default mix_fast)
#   TRJ_LOG          trj_log_interval             (default 100)
#   DEVICE           cuda|cpu|mps                 (default cuda)
#   DECOY_SET        casp14 | 3DRobot_set         (default casp14)
#   INIT_BEAD        only for MD_MODE=decoy
#   NATIVE_BEAD      override native bead CSV for RMSD reference
#   OUT_ROOT         output root                  (default eval/md_eval)
#   REPO             repo root                    (default $HOME/nnef)
#   ENV_PREFIX       conda env prefix             (default $HOME/envs/nnef)
#   DATA_DIR         data dir (for ESM)           (default $HOME/nnef_data)
#   ESM_H5           esm cache                    (default $DATA_DIR/hhsuite_esm_v2.h5)
#
set -euo pipefail

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

: "${MODEL_KEY:?set MODEL_KEY to yang_retrain|v1_rama|v1_rama_esm}"
: "${TARGET:?set TARGET e.g. T1053}"
: "${MD_MODE:?set MD_MODE to native|fold|decoy}"
: "${SEED:?set SEED (integer)}"

REPO="${REPO:-$HOME/nnef}"
ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/nnef}"
DATA_DIR="${DATA_DIR:-$HOME/nnef_data}"
ESM_H5="${ESM_H5:-$DATA_DIR/hhsuite_esm_v2.h5}"

if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  _PY="${CONDA_PREFIX}/bin/python"
elif [[ -x "${ENV_PREFIX}/bin/python" ]]; then
  _PY="${ENV_PREFIX}/bin/python"
else
  _PY="python"
fi
PY_IM=( "$_PY" -s )

DEVICE="${DEVICE:-cuda}"
DECOY_SET="${DECOY_SET:-casp14}"
L="${L:-100000}"
TRJ_LOG="${TRJ_LOG:-100}"
OUT_ROOT="${OUT_ROOT:-eval/md_eval}"

# Mode-aware sampler defaults. DynamicsMixFast has a cart_scale=50
# amplifier that explodes on any but the most conservative step sizes:
#   * Mode 2 (native): used mix_fast originally -> chain diffused out
#     of the basin within a few thousand steps.
#   * Mode 3 (decoy): used mix_fast originally -> chain Rg blew up to
#     ~1800 A via accumulated dihedral whip (smoke job 7079609).
# So native and decoy both switch to the plain ``cart`` sampler. Fold
# keeps mix_fast to retain the 50x exploration factor needed to leave
# the extended-chain basin; its NaN problem is handled by lowering lr
# in md_eval.py's defaults_by_mode['fold'].
if [[ "$MD_MODE" == "native" ]]; then
  X_TYPE="${X_TYPE:-cart}"
  LR="${LR:-3e-3}"
  T_NOISE="${T_NOISE:-3e-3}"
elif [[ "$MD_MODE" == "decoy" ]]; then
  X_TYPE="${X_TYPE:-cart}"
  # lr / t_noise inherit md_eval.py's defaults_by_mode['decoy'] (1e-2)
else
  X_TYPE="${X_TYPE:-mix_fast}"
fi

cd "$REPO"

echo "[md_eval_one] entered: model=$MODEL_KEY target=$TARGET mode=$MD_MODE seed=$SEED cwd=$(pwd)"

# Auto-resolve a "proxy native" bead CSV for CASP14 targets when the user
# does not supply NATIVE_BEAD: pick the highest-GDT_TS decoy from the
# target's list.csv. CASP14 does not ship an explicit native bead CSV, so
# for Mode 2 (native-basin stability) this gives a high-quality starting
# structure that stands in for the true native. Override by setting
# NATIVE_BEAD=<path> explicitly.
#
# Single-pass awk (no pipes) to avoid SIGPIPE aborts under pipefail+errexit.
if [[ -z "${NATIVE_BEAD:-}" && "$DECOY_SET" == "casp14" ]]; then
  _list_csv="nnef/data/decoys/${DECOY_SET}/${TARGET}/list.csv"
  if [[ -f "$_list_csv" ]]; then
    _top_decoy="$(awk -F, 'NR>1 && $2!="" && ($2+0 > best) {best=$2+0; name=$1} END{print name}' "$_list_csv")"
    if [[ -n "$_top_decoy" ]]; then
      NATIVE_BEAD="nnef/data/decoys/${DECOY_SET}/${TARGET}/${_top_decoy}_bead.csv"
      echo "[md_eval_one] auto-resolved NATIVE_BEAD (top-GDT proxy): $NATIVE_BEAD"
    else
      echo "[md_eval_one] WARN: no top-GDT decoy parsed from $_list_csv"
    fi
  else
    echo "[md_eval_one] WARN: list.csv not found: $_list_csv"
  fi
fi

if [[ "$DEVICE" == cuda* ]]; then
  if ! "${PY_IM[@]}" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "[md_eval_one] ERROR: DEVICE=cuda but no GPU visible"
    exit 1
  fi
fi

# Shared architecture flags (seq_len=14, k=10) — matches eval_one_run_casp14_only.sh.
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

MODEL_ARGS=()
RUN_BASE=""
case "$MODEL_KEY" in
  yang_retrain)
    LOAD_EXP="${LOAD_EXP:-runs/yang_retrain_6594199}"
    if [[ ! -f "$LOAD_EXP/models/model.pt" ]]; then
      echo "[md_eval_one] ERROR: missing $LOAD_EXP/models/model.pt"
      exit 1
    fi
    RUN_BASE="$(basename "$LOAD_EXP")"
    MODEL_ARGS=(
      --load_exp "$LOAD_EXP"
      --mixture_seq 1
      --mixture_rama 10
      --legacy_local_frame
    )
    ;;
  yang_legacy)
    # Faithful Yang-2022 architecture: --mixture_rama 0 (no rama head).
    # Trained by fasrc/train_yang_legacy.slurm; loading with --mixture_rama 10
    # would fail state_dict strict load since this checkpoint has no rama head.
    LOAD_EXP="${LOAD_EXP:-runs/yang_legacy_7317784}"
    if [[ ! -f "$LOAD_EXP/models/model.pt" ]]; then
      echo "[md_eval_one] ERROR: missing $LOAD_EXP/models/model.pt"
      exit 1
    fi
    RUN_BASE="$(basename "$LOAD_EXP")"
    MODEL_ARGS=(
      --load_exp "$LOAD_EXP"
      --mixture_seq 1
      --mixture_rama 0
      --legacy_local_frame
    )
    ;;
  v1_rama)
    LOAD_EXP="${LOAD_EXP:-runs/v1_pure_rama_v2_6228201}"
    if [[ ! -f "$LOAD_EXP/models/model.pt" ]]; then
      echo "[md_eval_one] ERROR: missing $LOAD_EXP/models/model.pt"
      exit 1
    fi
    RUN_BASE="$(basename "$LOAD_EXP")"
    MODEL_ARGS=(
      --load_exp "$LOAD_EXP"
      --mixture_rama 10
    )
    ;;
  v1_rama_esm)
    LOAD_EXP="${LOAD_EXP:-runs/v1_esm_rama_v2_6642146}"
    if [[ ! -f "$LOAD_EXP/models/model.pt" ]]; then
      echo "[md_eval_one] ERROR: missing $LOAD_EXP/models/model.pt"
      exit 1
    fi
    if [[ ! -f "$ESM_H5" ]]; then
      echo "[md_eval_one] ERROR: v1_rama_esm needs ESM cache: $ESM_H5"
      exit 1
    fi
    RUN_BASE="$(basename "$LOAD_EXP")"
    MODEL_ARGS=(
      --load_exp "$LOAD_EXP"
      --mixture_rama 10
      --use_esm
      --esm_h5_path "$ESM_H5"
      --esm_dim_in 1152
      --esm_dim_out 32
    )
    ;;
  *)
    echo "[md_eval_one] ERROR: MODEL_KEY must be yang_retrain|yang_legacy|v1_rama|v1_rama_esm, got: $MODEL_KEY"
    exit 1
    ;;
esac

SAVE_DIR="${SAVE_DIR:-$OUT_ROOT/$MD_MODE/$RUN_BASE/${TARGET}_seed${SEED}}"
mkdir -p "$SAVE_DIR"

CMD=(
  "${PY_IM[@]}" nnef/scripts/md_eval.py
  "${MODEL_ARGS[@]}"
  "${ARCH_SHARED[@]}"
  --device "$DEVICE"
  --md_mode "$MD_MODE"
  --decoy_set "$DECOY_SET"
  --target "$TARGET"
  --x_type "$X_TYPE"
  --fold_engine dynamics
  --L "$L"
  --trj_log_interval "$TRJ_LOG"
  --seed "$SEED"
  --save_dir "$SAVE_DIR"
)

# Mode-specific defaults are picked inside md_eval.py when CLI leaves them
# at argparse defaults; honour explicit overrides only.
if [[ -n "${LR:-}" ]]; then
  CMD+=( --lr "$LR" )
fi
if [[ -n "${T_NOISE:-}" ]]; then
  CMD+=( --T_max "$T_NOISE" )
fi
if [[ -n "${INIT_BEAD:-}" ]]; then
  CMD+=( --init_bead "$INIT_BEAD" )
fi
if [[ -n "${NATIVE_BEAD:-}" ]]; then
  CMD+=( --native_bead "$NATIVE_BEAD" )
fi

echo "========== md_eval_one =========="
echo "[md_eval_one] $(date '+%Y-%m-%d %H:%M:%S')"
echo "[md_eval_one] model=$MODEL_KEY ($RUN_BASE) target=$TARGET mode=$MD_MODE seed=$SEED"
echo "[md_eval_one] L=$L trj_log=$TRJ_LOG x_type=$X_TYPE device=$DEVICE"
echo "[md_eval_one] save_dir=$SAVE_DIR"

"${CMD[@]}"

echo "[md_eval_one] OK -> $SAVE_DIR"
