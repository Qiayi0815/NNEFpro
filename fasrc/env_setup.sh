#!/usr/bin/env bash
# One-time environment bootstrap for running NNEF on FASRC / Cannon.
#
# Run this ONCE from inside a short interactive compute allocation, not on
# the login node (conda/pip compilation can be heavy):
#
#   salloc -p test --time=1:00:00 --mem=8G --cpus-per-task=4
#   bash fasrc/env_setup.sh
#
# After it finishes, re-use the env from any SLURM job with:
#
#   module load python/3.10.12-fasrc01
#   source /n/sw/Mambaforge-<ver>/etc/profile.d/conda.sh
#   conda activate $HOME/envs/nnef
set -euo pipefail

# -----------------------------------------------------------------------------
# Env lives under $HOME. 100 GB quota is plenty for torch+CUDA (~8 GB) plus
# the raw data and checkpoints for this project.
# -----------------------------------------------------------------------------
ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/nnef}"

# -----------------------------------------------------------------------------
# 1. Load Python. FASRC's `python/3.10.12-fasrc01` chains in Mambaforge,
#    so after this `which python` points into /n/sw/Mambaforge-<ver>/bin/.
# -----------------------------------------------------------------------------
module purge
module load python/3.10.12-fasrc01

# -----------------------------------------------------------------------------
# 2. Locate conda.sh.
#
#    `conda` and `mamba` are shell FUNCTIONS on FASRC (not binaries), so
#    `command -v conda` returns the function body, not a path. We instead
#    derive the install prefix from the python binary shipped by the module:
#      /n/sw/Mambaforge-<ver>/bin/python
#          -> prefix  /n/sw/Mambaforge-<ver>
#          -> hook    /n/sw/Mambaforge-<ver>/etc/profile.d/conda.sh
#    We also honour $CONDA_EXE / $CONDA_SHLVL if the user already has conda
#    active in this shell.
# -----------------------------------------------------------------------------
PYTHON_BIN="$(command -v python || true)"
if [ -z "$PYTHON_BIN" ]; then
    echo "[env_setup] ERROR: no 'python' on PATH after loading module." >&2
    exit 1
fi

CONDA_PREFIX_GUESS="$(dirname "$(dirname "$PYTHON_BIN")")"
CONDA_HOOK=""
for cand in \
    "${CONDA_EXE:-__none__}/../../etc/profile.d/conda.sh" \
    "$CONDA_PREFIX_GUESS/etc/profile.d/conda.sh"
do
    if [ -f "$cand" ]; then
        CONDA_HOOK="$(cd "$(dirname "$cand")" && pwd)/$(basename "$cand")"
        break
    fi
done

if [ -z "$CONDA_HOOK" ]; then
    echo "[env_setup] ERROR: could not locate conda.sh."                       >&2
    echo "[env_setup] Tried: $CONDA_PREFIX_GUESS/etc/profile.d/conda.sh"       >&2
    echo "[env_setup] Run: find /n/sw/Mambaforge-* -maxdepth 4 -name conda.sh" >&2
    exit 1
fi

echo "[env_setup] sourcing $CONDA_HOOK"
# shellcheck disable=SC1090
source "$CONDA_HOOK"

# Prefer mamba when available: much faster solves on large deps.
if type -P mamba >/dev/null 2>&1 || declare -F mamba >/dev/null 2>&1; then
    CONDA_CMD="mamba"
else
    CONDA_CMD="conda"
fi
echo "[env_setup] solver: $CONDA_CMD"

# -----------------------------------------------------------------------------
# 3. Create env under $HOME.
# -----------------------------------------------------------------------------
mkdir -p "$(dirname "$ENV_PREFIX")"
if [ ! -d "$ENV_PREFIX" ]; then
    "$CONDA_CMD" create -y -p "$ENV_PREFIX" python=3.10
fi
# conda hook can touch unset vars; `set -u` breaks `conda activate` on some builds.
set +u
conda activate "$ENV_PREFIX"
set -u

# -----------------------------------------------------------------------------
# 4. Install PyTorch matching the CUDA build on Cannon GPU nodes. As of
#    2026-Q2 Cannon GPUs (A100 / H100 / H200) run CUDA 12.1-12.4 drivers;
#    cu121 wheels work on all of them. If you are stuck on older V100, use
#    cu118 instead.
# -----------------------------------------------------------------------------
pip install --upgrade pip
pip install torch==2.3.1 torchvision --index-url https://download.pytorch.org/whl/cu121

# -----------------------------------------------------------------------------
# 5. NNEF runtime deps. biotite is pulled in transitively by fair-esm.
# -----------------------------------------------------------------------------
pip install \
    numpy pandas h5py tqdm biopython \
    tensorboard matplotlib scipy

# -----------------------------------------------------------------------------
# 5b. ESM package for --use_esm. We primarily use ESM-C 600M (EvolutionaryScale):
#     `pip install esm`    -> esm.models.esmc.ESMC, open weights for 300M/600M
#     `pip install fair-esm` -> Meta's ESM1/2 (kept as a fallback only).
#
# The two packages share the top-level `esm` namespace in a confusing way;
# installing both works but `from esm.models.esmc import ESMC` only resolves
# when EvolutionaryScale's package is present.
#
# `httpx` is imported by esm.sdk (forge client) at package import time; some
# esm releases omit it from install_requires — install explicitly.
# -----------------------------------------------------------------------------
pip install esm fair-esm httpx

# -----------------------------------------------------------------------------
# 6. Sanity check. We intentionally do NOT `pip install -e .`: the repo has
#    no setup.py / pyproject.toml and train.slurm runs train_chimeric.py as a
#    script with sibling imports (`import options`, `from dataset import ...`).
# -----------------------------------------------------------------------------
python -c "
import torch, h5py, Bio, pandas, numpy
print('torch   :', torch.__version__, 'cuda avail:', torch.cuda.is_available())
print('devices :', torch.cuda.device_count())
print('h5py    :', h5py.__version__)
print('env path:', '$ENV_PREFIX')
from esm.models.esmc import ESMC
print('esm-c   : ESMC import OK')
"

cat <<EOF

[env_setup] Done.

To use this env in any SLURM job, put these lines near the top of your
sbatch script (or source them in an interactive session):

  module load python/3.10.12-fasrc01
  source "$CONDA_HOOK"
  set +u
  conda activate $ENV_PREFIX
  set -u
  export PYTHONNOUSERSITE=1

Avoid ``pip install --user`` for numpy/scipy/torch: packages under
``~/.local/lib/python3.10/site-packages`` can shadow the conda env and fail
with missing shared libraries (e.g. libquadmath). Use ``PYTHONNOUSERSITE=1``
or remove the broken trees under ~/.local.

EOF
