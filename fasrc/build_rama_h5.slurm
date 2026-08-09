#!/usr/bin/env bash
#SBATCH --job-name=nnef-ramah5
#SBATCH --partition=shared
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=runs/slurm-ramah5-%j.out
#SBATCH --error=runs/slurm-ramah5-%j.out
# -----------------------------------------------------------------------------
# Ramachandran h5 for the ribbon model (mixture_rama=10), aligned 1:1 with the
# v2 structure h5 by group_num. Computes phi/psi from the N/CA/C columns in the
# bead CSVs (build_rama_h5_v2.py). Run AFTER build_v2_h5.slurm.
#   inputs : $CB_H5 (v2 structure h5) , $BEAD_DIR/*_bead.csv
#   output : $OUT_H5 (hhsuite_rama_v2_new.h5)
# -----------------------------------------------------------------------------
set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/nnef}"
DATA_DIR="${DATA_DIR:-$HOME/nnef_data}"
NS="${NS:-/n/netscratch/atchan_lab/qzha/nnef_data}"
CB_H5="${CB_H5:-$NS/hhsuite_CB_v2_new.h5}"
BEAD_DIR="${BEAD_DIR:-$DATA_DIR/bead_csvs}"
OUT_H5="${OUT_H5:-$NS/hhsuite_rama_v2_new.h5}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-16}}"

module purge; module load python/3.10.12-fasrc01
PYTHON_BIN="$(command -v python)"
CONDA_HOOK="$(dirname "$(dirname "$PYTHON_BIN")")/etc/profile.d/conda.sh"
# shellcheck disable=SC1090
source "$CONDA_HOOK"; set +u; conda activate "$ENV_PREFIX"; set -u
PY="$ENV_PREFIX/bin/python"

cd "${SLURM_SUBMIT_DIR:-$PWD}"; mkdir -p "$(dirname "$OUT_H5")" runs
[[ -e "$CB_H5" ]] || { echo "[ramah5] ERROR: missing $CB_H5 (run build_v2_h5.slurm first)"; exit 1; }
echo "[ramah5] CB_H5=$CB_H5  BEAD_DIR=$BEAD_DIR -> $OUT_H5"

"$PY" -u nnef/data_prep_scripts/build_rama_h5_v2.py build \
    --cb_h5 "$CB_H5" --bead_dir "$BEAD_DIR" --out_h5 "$OUT_H5" --num_workers "$WORKERS"
echo "[ramah5] done -> $OUT_H5"
