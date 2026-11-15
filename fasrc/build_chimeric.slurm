#!/usr/bin/env bash
#SBATCH --job-name=nnef-chimeric
#SBATCH --partition=shared
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=6:00:00
#SBATCH --array=0-7
#SBATCH --output=runs/slurm-chimeric-%A_%a.out
#SBATCH --error=runs/slurm-chimeric-%A_%a.out
# -----------------------------------------------------------------------------
# Stage 3: a3m MSAs -> chimeric CSVs (data_prep_scripts/build_chimeric.py).
# Reads the alignment output, writes one .chimeric per chain. Fast (pure parse),
# 8-way array is plenty. Resumable.
#
# A3M_DIR defaults to the home path the (old) align job wrote to; override if
# your a3m are on netscratch:  A3M_DIR=/n/netscratch/.../hhwork/aligned_a3m
# -----------------------------------------------------------------------------
set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/nnef}"
DATA_DIR="${DATA_DIR:-$HOME/nnef_data}"
A3M_DIR="${A3M_DIR:-$DATA_DIR/aligned_a3m}"
OUT_DIR="${OUT_DIR:-/n/netscratch/atchan_lab/qzha/chimeric}"
NSHARDS="${NSHARDS:-8}"

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
mkdir -p "$OUT_DIR" runs
SHARD="${SLURM_ARRAY_TASK_ID:-0}"

echo "[chimeric] shard $SHARD/$NSHARDS  A3M_DIR=$A3M_DIR  OUT_DIR=$OUT_DIR"
"$PY" -u nnef/data_prep_scripts/build_chimeric.py \
    --a3m_dir "$A3M_DIR" --out_dir "$OUT_DIR" \
    --shard "$SHARD" --nshards "$NSHARDS"
echo "[chimeric] shard $SHARD done"
