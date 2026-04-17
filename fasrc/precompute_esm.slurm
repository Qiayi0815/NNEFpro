#!/usr/bin/env bash
#SBATCH --job-name=nnef-esm-cache
#SBATCH --partition=gpu_h200         # H200 is fastest for ESM-C 600M fp16.
                                     # Fall back to `gpu` (A100) if gpu_h200
                                     # queue is long; both work, A100 is ~1.5x slower.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G                    # ESM-C 600M fp16 weights ~1.2 GB +
                                     # activations for L<=1024 -> a few GB max.
#SBATCH --time=06:00:00              # ~12k chains * ~0.5-1s/chain on H200 -> 1.5-3.5h.
                                     # Bumped to 6h so long tails don't get killed.
#SBATCH --output=runs/slurm-%j.out
#SBATCH --error=runs/slurm-%j.err
#
# ----------------------------------------------------------------------------
# Precompute ESM-C 600M per-residue embeddings for every v2 chain and write
# them to $DATA_DIR/hhsuite_esm_v2.h5. Resume-safe: re-submitting picks up
# where it left off (see --overwrite in precompute_esm.py to force a rebuild).
#
# Submit with:   sbatch fasrc/precompute_esm.slurm
#
# Monitor with:  tail -f runs/slurm-<JOBID>.out
#
# One-time env setup (only if the `esm` package isn't installed yet):
#
#   conda activate $HOME/envs/nnef
#   pip install esm                  # EvolutionaryScale: ESM-C, ESM3 open
#   # NOT fair-esm -- that's a different package that only has ESM1/2.
#
# The first run downloads the ESM-C 600M weights (~2.4 GB) into
# ~/.cache/huggingface/ and ~/.cache/esm/; subsequent runs reuse the cache.
# ----------------------------------------------------------------------------

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/nnef}"
DATA_DIR="${DATA_DIR:-$HOME/nnef_data}"

module purge
module load python/3.10.12-fasrc01

PYTHON_BIN="$(command -v python)"
CONDA_HOOK="$(dirname "$(dirname "$PYTHON_BIN")")/etc/profile.d/conda.sh"
[ -f "$CONDA_HOOK" ] || { echo "[precompute_esm] conda.sh not found at $CONDA_HOOK"; exit 1; }
# shellcheck disable=SC1090
source "$CONDA_HOOK"
# conda hook can touch unset vars; `set -u` breaks `conda activate` on some builds.
set +u
conda activate "$ENV_PREFIX"
set -u

PY="${ENV_PREFIX}/bin/python"
[ -x "$PY" ] || { echo "[precompute_esm] ERROR: no interpreter at $PY (run: bash fasrc/env_setup.sh)"; exit 1; }

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# Verify the `esm` package (EvolutionaryScale) is importable; fail early with
# a clear message if the user still has only `fair-esm`.
"$PY" -c "from esm.models.esmc import ESMC; print('esm package ok')" 2>/dev/null || {
    echo "[precompute_esm] ERROR: 'esm' package (EvolutionaryScale) not importable."
    echo "[precompute_esm] Install with: pip install esm"
    echo "[precompute_esm] (This is different from 'fair-esm', which is Meta's ESM1/2 package.)"
    exit 1
}

echo "----- node info -----"
hostname
nvidia-smi
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
echo "ENV_PREFIX=$ENV_PREFIX"
echo "PY=$PY"
echo "DATA_DIR=$DATA_DIR"
echo "---------------------"

cd "$SLURM_SUBMIT_DIR"
mkdir -p runs

"$PY" -u nnef/data_prep_scripts/precompute_esm.py \
    --seq_h5   "$DATA_DIR/hhsuite_pdb_seq_v2.h5" \
    --pdb_list "$DATA_DIR/hhsuite_CB_v2_pdb_list.csv" \
    --out_h5   "$DATA_DIR/hhsuite_esm_v2.h5" \
    --model    esmc_600m \
    --device   cuda:0 \
    --dtype    float16 \
    --max_len  1024 \
    --flush_every 50

echo "[precompute_esm] finished with exit code $?"
