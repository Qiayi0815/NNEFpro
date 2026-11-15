#!/usr/bin/env bash
#SBATCH --job-name=nnef-decomp
#SBATCH --partition=gpu_requeue
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=runs/slurm-decomp-%A_%a.out
#SBATCH --error=runs/slurm-decomp-%A_%a.out
#
# Per-term decoy energy decomposition (--decompose_energy) for any checkpoint
# family, over a chunk of targets. One array task = CHUNK targets from
# TARGET_FILE (one target id per line); task i takes lines [i*CHUNK, (i+1)*CHUNK).
#
#   FAMILY=yang    -> --legacy_local_frame --mixture_rama 0          (Yang repro + its dist ablations)
#   FAMILY=ribbon  -> v2 frame, --mixture_rama 10
#   FAMILY=esm     -> ribbon + ESM-C per-residue adapter (needs ESM_H5 covering the targets)
#
# Submit (e.g. 3DRobot, 20 targets/task):
#   N=$(wc -l < fasrc/targets_3drobot.txt); sbatch --array=0-$(( (N+19)/20 - 1 )) \
#     --export=ALL,FAMILY=yang,LOAD_EXP=runs/yang_full1000_27329319,DECOY_SET=3DRobot_set,\
#TARGET_FILE=fasrc/targets_3drobot.txt,CHUNK=20 fasrc/decomp_decoys_array.slurm
#
# ANGLE_DIST / R_DIST MUST match how LOAD_EXP was trained (silent mismatch otherwise).
# Re-queued tasks resume: per-target decoy_loss csvs are cached (skip_if_exists).
set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/nnef}"
FAMILY="${FAMILY:?set FAMILY=yang|ribbon|esm}"
LOAD_EXP="${LOAD_EXP:?set LOAD_EXP=runs/<checkpoint>}"
DECOY_SET="${DECOY_SET:?set DECOY_SET=casp14|3DRobot_set}"
TARGET_FILE="${TARGET_FILE:?set TARGET_FILE=<one target per line>}"
CHUNK="${CHUNK:-20}"
ANGLE_DIST="${ANGLE_DIST:-gaussian}"
R_DIST="${R_DIST:-gaussian}"
ESM_H5="${ESM_H5:-}"
# Extra evaluate_decoys.py flags (e.g. "--energy_profile_center_only"); give such
# runs their own TAG so they do not overwrite the default decomposition cache.
EXTRA_ARGS="${EXTRA_ARGS:-}"

module purge
module load python/3.10.12-fasrc01
PYTHON_BIN="$(command -v python)"
CONDA_HOOK="$(dirname "$(dirname "$PYTHON_BIN")")/etc/profile.d/conda.sh"
# shellcheck disable=SC1090
source "$CONDA_HOOK"; set +u; conda activate "$ENV_PREFIX"; set -u
export PYTHONUNBUFFERED=1
PY="${ENV_PREFIX}/bin/python"
"$PY" -s -c "import torch; assert torch.cuda.is_available()" || { echo "[decomp] need GPU"; exit 1; }

cd "$SLURM_SUBMIT_DIR"
[[ -f "$LOAD_EXP/models/model.pt" ]] || { echo "[decomp] ERROR: missing $LOAD_EXP/models/model.pt"; exit 1; }

TASK="${SLURM_ARRAY_TASK_ID:-0}"
START=$(( TASK * CHUNK + 1 ))
END=$(( START + CHUNK - 1 ))
TARGETS="$(sed -n "${START},${END}p" "$TARGET_FILE" | paste -sd, -)"
[[ -n "$TARGETS" ]] || { echo "[decomp] task $TASK: no targets (lines $START-$END)"; exit 0; }

case "$FAMILY" in
    yang)   MODEL_ARGS=( --mixture_seq 1 --mixture_rama 0 --legacy_local_frame ) ;;
    ribbon) MODEL_ARGS=( --mixture_seq 1 --mixture_rama 10 ) ;;
    esm)
        [[ -f "$ESM_H5" ]] || { echo "[decomp] ERROR: esm needs ESM_H5 covering the targets: '$ESM_H5'"; exit 1; }
        MODEL_ARGS=( --mixture_seq 1 --mixture_rama 10 --use_esm --esm_h5_path "$ESM_H5"
                     --esm_dim_in 1152 --esm_dim_out 32 --esm_pool per_residue ) ;;
    *) echo "[decomp] unknown FAMILY=$FAMILY"; exit 1 ;;
esac

RUN_BASE="$(basename "$LOAD_EXP")"
TAG="${TAG:-decomp_${RUN_BASE}}"
OUT_DIR="${OUT_DIR:-eval/decomp/${RUN_BASE}/${DECOY_SET}/task${TASK}}"
mkdir -p "$OUT_DIR"

echo "----- decomp: $FAMILY / $DECOY_SET / task $TASK -----"
hostname; nvidia-smi --query-gpu=name --format=csv,noheader
echo "LOAD_EXP=$LOAD_EXP ANGLE_DIST=$ANGLE_DIST R_DIST=$R_DIST TARGETS=$TARGETS"

"$PY" -s nnef/scripts/evaluate_decoys.py \
  --load_exp "$LOAD_EXP" \
  "${MODEL_ARGS[@]}" --decompose_energy \
  --decoy_sets "$DECOY_SET" --targets "$TARGETS" \
  --device cuda --plot \
  --exp_tag "$TAG" --out_dir "$OUT_DIR" \
  --seq_len 14 --seq_type residue --residue_type_num 20 \
  --embed_size 32 --dim 128 --n_layers 4 --attn_heads 4 \
  --mixture_r 2 --mixture_angle 3 \
  --angle_dist "$ANGLE_DIST" --r_dist "$R_DIST" \
  --smooth_gaussian --smooth_r 0.3 --smooth_angle 45 \
  --coords_angle_loss_lamda 1 --profile_loss_lamda 10 --coords_rama_loss_lamda 1 \
  --use_position_weights --cen_seg_loss_lamda 1 --oth_seg_loss_lamda 3 \
  $EXTRA_ARGS

echo "[decomp] Done -> $OUT_DIR/"
