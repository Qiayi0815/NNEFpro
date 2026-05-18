#!/usr/bin/env bash
#SBATCH --job-name=nnef-md-yang-paper
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=runs/slurm-md-yang-paper-%A_%a.out
#SBATCH --error=runs/slurm-md-yang-paper-%A_%a.out
#SBATCH --array=0-2
#
# Mode 2 (native-basin stability MD) -- PAPER CHECKPOINT (params/exp1) on T1026.
#
# This is the apples-to-apples companion to fasrc/md_eval_mode2_yang_legacy.slurm.
#   model   = Yang 2022 released ckpt (params/exp1/models/model.pt)
#             architecture: --mixture_seq 1 --mixture_rama 0 --legacy_local_frame
#             (strictly identical to yang_legacy's arch, only the weights differ).
#   target  = T1026  (matched to the mode2_yang_legacy run so trajectories are
#                     directly comparable seed-for-seed).
#   seeds   = (0, 1, 2)
#   sampler = lr=3e-4, T_noise=3e-4, L=50000  (same as mode2_yang_legacy; the
#             regime where the energy landscape -- not the integrator -- is
#             being probed. See mode2_lr_ablation results for rationale.)
#
# Outputs land under eval/md_eval/mode2_yang_paper/ so plot_md_eval.py picks
# them up as a separate sweep without contaminating mode2_yang_legacy's heatmap.
# Run _base will be 'exp1' (= basename of params/exp1); plot_md_eval falls
# through to the literal name, which is fine for the figure.
#
# We REUSE md_eval_one.sh's yang_legacy branch via MODEL_KEY=yang_legacy and
# override LOAD_EXP=params/exp1 -- the arch flags happen to match exactly, and
# params/exp1 is a --load_exp-compatible layout (params/exp1/models/model.pt).
#
# Submit:
#   sbatch fasrc/md_eval_mode2_yang_paper.slurm

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/nnef}"
DATA_DIR="${DATA_DIR:-$HOME/nnef_data}"

module purge
module load python/3.10.12-fasrc01

PYTHON_BIN="$(command -v python)"
CONDA_HOOK="$(dirname "$(dirname "$PYTHON_BIN")")/etc/profile.d/conda.sh"
[ -f "$CONDA_HOOK" ] || { echo "[md_yang_paper] conda.sh not found at $CONDA_HOOK"; exit 1; }
# shellcheck disable=SC1090
source "$CONDA_HOOK"
set +u
conda activate "$ENV_PREFIX"
set -u

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

PY="${ENV_PREFIX}/bin/python"
[ -x "$PY" ] || { echo "[md_yang_paper] ERROR: no $PY"; exit 1; }
"$PY" -s -c "import torch; print('[md_yang_paper] cuda', torch.cuda.is_available())" || exit 1

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TOKENIZERS_PARALLELISM=false

cd "$SLURM_SUBMIT_DIR"
mkdir -p runs eval/md_eval

if [[ ! -f params/exp1/models/model.pt ]]; then
  echo "[md_yang_paper] ERROR: missing params/exp1/models/model.pt on this cluster."
  echo "[md_yang_paper]   sync from local:"
  echo "[md_yang_paper]     rsync -av params/exp1/ qzha@login.rc.fas.harvard.edu:/n/home03/qzha/nnef/params/exp1/"
  exit 1
fi

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
# Reuse yang_legacy's arch-flag branch, override the checkpoint path.
export MODEL_KEY=yang_legacy
export LOAD_EXP=params/exp1
export TARGET="${TARGETS[$target_i]}"
export SEED="${SEEDS[$seed_i]}"
export L="${L:-50000}"
export TRJ_LOG="${TRJ_LOG:-100}"
# Sampler: lr=3e-4 + T_noise=3e-4 -- matches mode2_yang_legacy for seed-wise comparison.
export LR="${LR:-3e-4}"
export T_NOISE="${T_NOISE:-3e-4}"
export OUT_ROOT="${OUT_ROOT:-eval/md_eval/mode2_yang_paper}"

echo "----- md_eval yang_paper (params/exp1) array task -----"
hostname
nvidia-smi
echo "ARRAY_ID=$SLURM_ARRAY_JOB_ID TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "MODEL=$MODEL_KEY LOAD_EXP=$LOAD_EXP TARGET=$TARGET SEED=$SEED L=$L LR=$LR T_NOISE=$T_NOISE"
echo "-------------------------------------------------------"

bash fasrc/md_eval_one.sh

echo "[md_yang_paper] task $SLURM_ARRAY_TASK_ID OK"
