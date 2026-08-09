#!/usr/bin/env bash
#SBATCH --job-name=nnef-v2h5
#SBATCH --partition=shared
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=runs/slurm-v2h5-%j.out
#SBATCH --error=runs/slurm-v2h5-%j.out
# -----------------------------------------------------------------------------
# Stage 6: build the v2 N-CA-C-frame structure h5 (thesis modified frame,
#   inputs : $BEAD_DIR/*_bead.csv , $CHIM_DIR/*.chimeric
#   outputs: $OUT_H5 (hhsuite_CB_v2_new.h5) + $PDB_LIST csv
#
# Reuses local_extractor_v2's chain matching + IO; only the per-block geometry
# is Yang's. group_num convention == v2, so this is a drop-in for the existing
# DatasetLocalGenCM (its group_num-1 indexing stays correct) once paired with a
# matching seq h5 (stage 7).
#
# Requires BOTH stage 3 (chimeric) and stage 4+5 (beads) complete.
# -----------------------------------------------------------------------------
set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/nnef}"
DATA_DIR="${DATA_DIR:-$HOME/nnef_data}"
NS="${NS:-/n/netscratch/atchan_lab/qzha}"
BEAD_DIR="${BEAD_DIR:-$DATA_DIR/bead_csvs}"
CHIM_DIR="${CHIM_DIR:-$NS/chimeric}"
OUT_H5="${OUT_H5:-$NS/nnef_data/hhsuite_CB_v2_new.h5}"
PDB_LIST="${PDB_LIST:-$DATA_DIR/hhsuite_CB_v2_new_pdb_list.csv}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-16}}"

module purge 2>/dev/null || true
module load python/3.10.12-fasrc01 2>/dev/null || true
PYTHON_BIN="$(command -v python)"
CONDA_HOOK="$(dirname "$(dirname "$PYTHON_BIN")")/etc/profile.d/conda.sh"
if [[ -f "$CONDA_HOOK" ]]; then
    # shellcheck disable=SC1090
    source "$CONDA_HOOK"; set +u; conda activate "$ENV_PREFIX"; set -u
fi
PY="$ENV_PREFIX/bin/python"; [[ -x "$PY" ]] || PY="$(command -v python)"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p "$(dirname "$OUT_H5")" runs

echo "[yangh5] BEAD_DIR=$BEAD_DIR  CHIM_DIR=$CHIM_DIR"
echo "[yangh5] beads=$(ls "$BEAD_DIR"/*_bead.csv 2>/dev/null | wc -l)  chimeric=$(ls "$CHIM_DIR"/*.chimeric 2>/dev/null | wc -l)"

"$PY" -u nnef/data_prep_scripts/local_extractor_v2.py build-h5 \
    --bead_dir "$BEAD_DIR" \
    --chim_dir "$CHIM_DIR" \
    --out_h5 "$OUT_H5" \
    --pdb_list "$PDB_LIST" \
    --num_workers "$WORKERS"

echo "[yangh5] done -> $OUT_H5"
echo "[yangh5] chains: $("$PY" -c "import h5py; print(len(h5py.File('$OUT_H5','r').keys()))")"
