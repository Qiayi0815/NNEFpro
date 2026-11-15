#!/usr/bin/env bash
#SBATCH --job-name=nnef-beads
#SBATCH --partition=shared
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --array=0-15
#SBATCH --output=runs/slurm-beads-%A_%a.out
#SBATCH --error=runs/slurm-beads-%A_%a.out
# -----------------------------------------------------------------------------
# Stage 4+5, runnable IN PARALLEL with align_hhsuite.slurm: download every
# unique PISCES PDB and extract its bead CSV. Independent of HHsuite output, so
# doing it now pre-stages all beads; stage 6 later uses (aligned ∩ has-bead).
#
# Reuses data_hh/q_extract_beads.extract_beads -> identical bead-CSV schema.
# RESUMABLE: a PDB whose <PDB>_bead.csv already exists is skipped, so the ~13k
# beads we already have are not re-fetched; only the ~17k missing ones download.
#
# Raw .pdb files go to netscratch (large, transient); bead CSVs go to
# $DATA_DIR/bead_csvs (small, consumed by stage 6).
#
# LOCAL SMOKE: SMOKE=1 bash fasrc/fetch_and_beads.slurm   (shard 0, 1 shard, needs network)
# -----------------------------------------------------------------------------
set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/nnef}"
DATA_DIR="${DATA_DIR:-$HOME/nnef_data}"
FASTA="${FASTA:-$DATA_DIR/cullpdb_pc50.0_res0.0-3.0_len40-10000_R1.0_Xray_d2024_11_18_chains32021.fasta}"
PDB_DIR="${PDB_DIR:-/n/netscratch/atchan_lab/qzha/pdb_files}"
BEAD_DIR="${BEAD_DIR:-$DATA_DIR/bead_csvs}"
NSHARDS="${NSHARDS:-16}"                                # keep == #SBATCH --array width
SMOKE="${SMOKE:-0}"

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
mkdir -p "$PDB_DIR" "$BEAD_DIR" runs

SHARD="${SLURM_ARRAY_TASK_ID:-0}"
[[ "$SMOKE" == "1" ]] && { SHARD=0; NSHARDS=1; }

echo "[beads] shard $SHARD/$NSHARDS  FASTA=$FASTA  PDB_DIR=$PDB_DIR  BEAD_DIR=$BEAD_DIR"
"$PY" -u nnef/data_prep_scripts/fetch_and_beads.py \
    --fasta "$FASTA" \
    --pdb_dir "$PDB_DIR" \
    --bead_dir "$BEAD_DIR" \
    --shard "$SHARD" --nshards "$NSHARDS"
echo "[beads] shard $SHARD done"
