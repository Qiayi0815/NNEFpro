#!/usr/bin/env bash
#SBATCH --job-name=nnef-md-yang-legacy
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=runs/slurm-md-yang-legacy-%A_%a.out
#SBATCH --error=runs/slurm-md-yang-legacy-%A_%a.out
#SBATCH --array=0-2
#
# Mode 2 (native-basin stability MD) — yang_legacy only, T1026 at lr=3e-4.
#   model   = yang_legacy (runs/yang_legacy_7317784, mixture_rama=0,
#             mixture_seq=1, legacy_local_frame)
#   targets = (T1026)        # only target with matching lr=3e-4 v1 baseline
#                            # (see eval/md_eval/mode2_lr_ablation/native/).
#   seeds   = (0, 1, 2)
#   => 1 * 1 * 3 = 3 array tasks.
#
# lr=3e-4 + T_noise=3e-4 chosen from the mode2_lr_ablation result: at
# lr=3e-3 every model drifts ~4 A off native within 100k steps (sampler-
# dominated), while lr=3e-4 keeps RMSD-to-native ~1.2 A over the same
# budget -- that is the regime where the energy landscape is actually
# being probed rather than the integrator. L halved to 50k since the
# per-step displacement is smaller and trajectories equilibrate faster.
#
# Outputs land under eval/md_eval/mode2_yang_legacy/native/ so the
# existing mode2/native heatmap (built at lr=3e-3) is not contaminated
# with apples-to-oranges data. The plot script picks up the new sweep
# dir automatically.
#
# Submit:
#   sbatch fasrc/md_eval_mode2_yang_legacy.slurm

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/nnef}"
DATA_DIR="${DATA_DIR:-$HOME/nnef_data}"

module purge
module load python/3.10.12-fasrc01

PYTHON_BIN="$(command -v python)"
CONDA_HOOK="$(dirname "$(dirname "$PYTHON_BIN")")/etc/profile.d/conda.sh"
[ -f "$CONDA_HOOK" ] || { echo "[md_yang_legacy] conda.sh not found at $CONDA_HOOK"; exit 1; }
# shellcheck disable=SC1090
source "$CONDA_HOOK"
set +u
conda activate "$ENV_PREFIX"
set -u

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

PY="${ENV_PREFIX}/bin/python"
[ -x "$PY" ] || { echo "[md_yang_legacy] ERROR: no $PY"; exit 1; }
"$PY" -s -c "import torch; print('[md_yang_legacy] cuda', torch.cuda.is_available())" || exit 1

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TOKENIZERS_PARALLELISM=false

cd "$SLURM_SUBMIT_DIR"
mkdir -p runs eval/md_eval

TARGETS=(T1026)
SEEDS=(0 1 2)

N_TARGETS=${#TARGETS[@]}
N_SEEDS=${#SEEDS[@]}

idx=${SLURM_ARRAY_TASK_ID:-0}
target_i=$(( idx / N_SEEDS ))
seed_i=$(( idx % N_SEEDS ))

export REPO="$SLURM_SUBMIT_DIR"
export DATA_DIR="$DATA_DIR"
export DEVICE=cuda
export MD_MODE=native
export MODEL_KEY=yang_legacy
export TARGET="${TARGETS[$target_i]}"
export SEED="${SEEDS[$seed_i]}"
export L="${L:-50000}"
export TRJ_LOG="${TRJ_LOG:-100}"
# Sampler: lr=3e-4 + T_noise=3e-4 (see ablation rationale in header).
export LR="${LR:-3e-4}"
export T_NOISE="${T_NOISE:-3e-4}"
export OUT_ROOT="${OUT_ROOT:-eval/md_eval/mode2_yang_legacy}"

echo "----- md_eval yang_legacy array task -----"
hostname
nvidia-smi
echo "ARRAY_ID=$SLURM_ARRAY_JOB_ID TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "MODEL=$MODEL_KEY TARGET=$TARGET SEED=$SEED L=$L LR=$LR T_NOISE=$T_NOISE"
echo "------------------------------------------"

bash fasrc/md_eval_one.sh

echo "[md_yang_legacy] task $SLURM_ARRAY_TASK_ID OK"
