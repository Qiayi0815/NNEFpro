#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# One-time setup for the HHsuite alignment stage (pipeline stage 2) on FASRC.
#
# Installs HH-suite3 (hhblits + hhfilter) into the existing nnef conda env and
# downloads the PDB70 database that align_hhsuite.slurm searches against.
#
# Mirrors the conventions in fasrc/env_setup.sh (same module + conda hook).
# Run ONCE from a short interactive allocation (NOT the login node — the PDB70
# download is ~56 GB and the conda solve is heavy):
#
#   salloc -p test --time=4:00:00 --mem=16G --cpus-per-task=8
#   bash fasrc/setup_hhsuite.sh
#
# After it finishes, fasrc/align_hhsuite.slurm can run the alignment.
# -----------------------------------------------------------------------------
set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/nnef}"
DATA_DIR="${DATA_DIR:-$HOME/nnef_data}"
PDB70_DIR="${PDB70_DIR:-$DATA_DIR/pdb70}"
# HH-suite PDB70 (mmCIF-derived, latest). ~56 GB unpacked.
PDB70_URL="${PDB70_URL:-https://wwwuser.gwdg.de/~compbiol/data/hhsuite/databases/hhsuite_dbs/pdb70_from_mmcif_latest.tar.gz}"

module purge
module load python/3.10.12-fasrc01

PYTHON_BIN="$(command -v python)"
CONDA_HOOK="$(dirname "$(dirname "$PYTHON_BIN")")/etc/profile.d/conda.sh"
[ -f "$CONDA_HOOK" ] || { echo "[setup_hhsuite] conda.sh not found at $CONDA_HOOK"; exit 1; }
# shellcheck disable=SC1090
source "$CONDA_HOOK"
set +u
conda activate "$ENV_PREFIX"
set -u

# -----------------------------------------------------------------------------
# 1. Install HH-suite3 from bioconda into the nnef env (provides hhblits,
#    hhfilter, hhsearch). Skips if already present.
# -----------------------------------------------------------------------------
# NOTE: do NOT probe with `hhblits -h` — it prints usage and exits non-zero,
# which under `set -o pipefail` looks like a failure. Test the binary instead.
if [ -x "$ENV_PREFIX/bin/hhblits" ]; then
    echo "[setup_hhsuite] hhblits already installed at $ENV_PREFIX/bin/hhblits"
else
    echo "[setup_hhsuite] installing hh-suite from bioconda ..."
    conda install -y -p "$ENV_PREFIX" -c conda-forge -c bioconda hhsuite
fi
[ -x "$ENV_PREFIX/bin/hhblits" ]  || { echo "ERROR: hhblits not found in env"; exit 1; }
[ -x "$ENV_PREFIX/bin/hhfilter" ] || { echo "ERROR: hhfilter not found in env"; exit 1; }
echo "[setup_hhsuite] hhblits + hhfilter OK"

# -----------------------------------------------------------------------------
# 2. Download + unpack PDB70 (skips if the prefix files already exist).
# -----------------------------------------------------------------------------
mkdir -p "$PDB70_DIR"
if ls "$PDB70_DIR"/pdb70_*.ffdata >/dev/null 2>&1; then
    echo "[setup_hhsuite] PDB70 already present in $PDB70_DIR:"
    ls -lh "$PDB70_DIR"/pdb70_*.ff{data,index} 2>/dev/null | head
else
    echo "[setup_hhsuite] downloading PDB70 (~56 GB) to $PDB70_DIR ..."
    TARBALL="$PDB70_DIR/pdb70.tar.gz"
    # -c resumes a partial download if the job was re-run.
    wget -c -O "$TARBALL" "$PDB70_URL"
    echo "[setup_hhsuite] unpacking ..."
    tar -xzf "$TARBALL" -C "$PDB70_DIR"
    rm -f "$TARBALL"
fi

echo "[setup_hhsuite] DONE."
echo "  hhblits : $ENV_PREFIX/bin/hhblits"
echo "  PDB70   : $PDB70_DIR  (prefix: $PDB70_DIR/pdb70)"
echo "  next    : sbatch fasrc/align_hhsuite.slurm"
