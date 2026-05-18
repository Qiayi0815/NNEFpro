#!/usr/bin/env bash
#SBATCH --job-name=nnef-esm2-cache
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=runs/slurm-esm2-precompute-%j.out
#SBATCH --error=runs/slurm-esm2-precompute-%j.out
#
# ----------------------------------------------------------------------------
# Precompute ESM2-650M per-residue embeddings for layers 32 (penultimate) +
# 33 (final), in a single forward pass per chain. Writes to
# $DATA_DIR/hhsuite_esm2_650m_v2.h5 with two datasets per PDB:
#   <pdb_chain>/esm_l32   (L, 1280)  float16
#   <pdb_chain>/esm_l33   (L, 1280)  float16
#
# Companion to fasrc/precompute_esm.slurm (which builds the ESM-C 600M
# single-layer cache `hhsuite_esm_v2.h5`). This one is needed for the
# ESM-ablation array launched by fasrc/train_v1_esm.slurm --array=0-7.
#
# Why ESM2 here (and not ESM-C)? EvolutionaryScale's ESM-C `logits()` API
# does not expose intermediate transformer layers; fair-esm's ESM2 forward
# accepts `repr_layers=[32, 33]` so we can get both layers in one pass.
#
# Prereq: pip install fair-esm  (different package from `esm`).
#         The first run downloads ESM2-650M weights (~2.5 GB) into
#         ~/.cache/torch/hub/ or fair-esm's default cache.
#
# Submit (single job, resume-safe):
#   sbatch fasrc/precompute_esm2_multilayer.slurm
#
# Or chain it before train + eval (one ssh, three sbatch commands -- see end
# of this file's header comment block for the dependency-chain recipe).
# ----------------------------------------------------------------------------
# DEPENDENCY CHAIN RECIPE (run from the cluster login node):
#
#   JOBA=$(sbatch --parsable fasrc/precompute_esm2_multilayer.slurm)
#   echo "precompute jobid: $JOBA"
#
#   JOBB=$(sbatch --parsable --dependency=afterok:$JOBA --array=0-7 \
#                 fasrc/train_v1_esm.slurm)
#   echo "training array jobid: $JOBB"
#
#   JOBC=$(sbatch --parsable --dependency=afterok:$JOBB \
#                 fasrc/eval_v1_esm_ablation_array.slurm)
#   echo "eval array jobid: $JOBC"
#
# squeue -u $USER will show all three queued; SLURM only starts a stage once
# its dependency is satisfied (afterok = previous job finished with exit 0).
# ----------------------------------------------------------------------------

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/nnef}"
DATA_DIR="${DATA_DIR:-$HOME/nnef_data}"

module purge
module load python/3.10.12-fasrc01

PYTHON_BIN="$(command -v python)"
CONDA_HOOK="$(dirname "$(dirname "$PYTHON_BIN")")/etc/profile.d/conda.sh"
[ -f "$CONDA_HOOK" ] || { echo "[precompute_esm2] conda.sh not found at $CONDA_HOOK"; exit 1; }
# shellcheck disable=SC1090
source "$CONDA_HOOK"
# conda hook can touch unset vars; `set -u` breaks `conda activate` on some builds.
set +u
conda activate "$ENV_PREFIX"
set -u

PY="${ENV_PREFIX}/bin/python"
[ -x "$PY" ] || { echo "[precompute_esm2] ERROR: no interpreter at $PY"; exit 1; }

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

# Verify fair-esm wins the `import esm` lookup. EvolutionaryScale's `esm` package
# uses the same module name and will shadow fair-esm if both are installed -- in
# that case `import esm` succeeds but `esm.pretrained` does not exist.
"$PY" -c "
import esm, sys
ok = hasattr(esm, 'pretrained') and hasattr(esm.pretrained, 'esm2_t33_650M_UR50D')
if not ok:
    print('[precompute_esm2] ERROR: import esm succeeded but it is NOT fair-esm.', file=sys.stderr)
    print('[precompute_esm2]   module file: ' + esm.__file__, file=sys.stderr)
    print('[precompute_esm2]   has pretrained: ' + str(hasattr(esm, 'pretrained')), file=sys.stderr)
    print('[precompute_esm2]   Likely EvolutionaryScale esm is shadowing fair-esm.', file=sys.stderr)
    print('[precompute_esm2]   Fix: $PY -m pip uninstall -y esm esm-sdk && $PY -m pip install fair-esm', file=sys.stderr)
    sys.exit(1)
print('[precompute_esm2] fair-esm OK:', esm.__file__)
" || exit 1

echo "----- node info -----"
hostname
nvidia-smi
echo "ENV_PREFIX=$ENV_PREFIX"
echo "DATA_DIR=$DATA_DIR"
echo "PY=$PY"
echo "---------------------"

cd "$SLURM_SUBMIT_DIR"
mkdir -p runs

"$PY" -u nnef/data_prep_scripts/precompute_esm.py \
    --seq_h5     "$DATA_DIR/hhsuite_pdb_seq_v2.h5" \
    --pdb_list   "$DATA_DIR/hhsuite_CB_v2_pdb_list.csv" \
    --out_h5     "$DATA_DIR/hhsuite_esm2_650m_v2.h5" \
    --model      esm2_t33_650M_UR50D \
    --repr_layers 32 33 \
    --device     cuda:0 \
    --dtype      float16 \
    --max_len    1024 \
    --flush_every 50

echo "[precompute_esm2] finished with exit code $?"
