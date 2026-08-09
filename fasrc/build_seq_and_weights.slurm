#!/usr/bin/env bash
#SBATCH --job-name=nnef-seqh5
#SBATCH --partition=shared
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=runs/slurm-seqh5-%j.out
#SBATCH --error=runs/slurm-seqh5-%j.out
# -----------------------------------------------------------------------------
# Stage 7: chimeric MSAs -> seq h5 (build_seq_h5.py), then write Yang-style
# sampling weights into the pdb_list (compute_weights.py).
#   inputs : $CHIM_DIR/*.chimeric , $STRUCT_H5 (from stage 6), $PDB_LIST
#   outputs: $SEQ_H5 (hhsuite_pdb_seq_yang.h5) + updated weights in $PDB_LIST
#
# Run AFTER stage 6 (build_yang_h5) so the pdb_list + structure h5 exist.
# -----------------------------------------------------------------------------
set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/nnef}"
DATA_DIR="${DATA_DIR:-$HOME/nnef_data}"
NS="${NS:-/n/netscratch/atchan_lab/qzha}"
CHIM_DIR="${CHIM_DIR:-$NS/chimeric}"
SEQ_H5="${SEQ_H5:-$NS/nnef_data/hhsuite_pdb_seq_yang.h5}"
STRUCT_H5="${STRUCT_H5:-$NS/nnef_data/hhsuite_CB_yang.h5}"
PDB_LIST="${PDB_LIST:-$DATA_DIR/hhsuite_CB_yang_pdb_list.csv}"
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
mkdir -p "$(dirname "$SEQ_H5")" runs

echo "[seqh5] chimeric=$(ls "$CHIM_DIR"/*.chimeric 2>/dev/null | wc -l) -> $SEQ_H5"
"$PY" -u nnef/data_prep_scripts/build_seq_h5.py \
    --chim_dir "$CHIM_DIR" --out_h5 "$SEQ_H5" --num_workers "$WORKERS"

echo "[seqh5] computing Yang-style weights into $PDB_LIST"
"$PY" -u nnef/data_prep_scripts/compute_weights.py \
    --struct_h5 "$STRUCT_H5" --seq_h5 "$SEQ_H5" --pdb_list "$PDB_LIST"

echo "[seqh5] done -> $SEQ_H5"
