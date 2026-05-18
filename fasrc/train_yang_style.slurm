#!/usr/bin/env bash
#SBATCH --job-name=nnef-yang-retrain
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm-%j.out
#SBATCH --error=runs/slurm-%j.err
#
# Retrain a **Yang-style** LocalTransformer on your v2 HDF5 (same files as
# train_v1_pure / train.slurm): legacy neighbor-bead local frame + mixture_seq=1,
# no cart coords / seq offset / ESM / dihedral.
#
# Checkpoints: ``runs/yang_retrain_<SLURM_JOB_ID>/models/model.pt``
#
# Important (architecture vs released ``params/exp1``):
#   The paper / ``inference_paper2022.py`` checkpoint uses ``--mixture_rama 0``
#   (no Ramachandran head). This codebase's training path applies φ/ψ NLL when
#   rama HDF5 targets are present; ``mixture_rama=0`` is not supported for that
#   path. This job therefore uses ``--mixture_rama 10`` like ``train_v1_pure``.
#   For decoy eval, use ``--mixture_rama 10`` (default in evaluate_decoys for your
#   runs), not the paper2022 ``--mixture_rama 0`` preset, unless you add a
#   dedicated no-rama training branch.
#
# Data (override with env):
#   DATA_DIR       default: $HOME/nnef_data
#   DATA_FLAG      default: hhsuite_CB_v2_pdb_list.csv  (under nnef/data/)
#
# Submit:
#   cd ~/nnef && sbatch fasrc/train_yang_style.slurm
#   DATA_DIR=$HOME/nnef_data sbatch fasrc/train_yang_style.slurm
#   DATA_FLAG=my_train_weights.csv sbatch fasrc/train_yang_style.slurm
#
# After: pull checkpoints to laptop (repo root):
#   ONLY_PT=1 bash fasrc/pull_yang_retrain_from_cluster.sh <JOBID>
# CASP14 local eval (matches legacy frame + mixture_seq=1 + mixture_rama=10):
#   YANG_RUN=yang_retrain_<JOBID> DEVICE=cpu bash fasrc/eval_casp14_all_local.sh

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/nnef}"
DATA_DIR="${DATA_DIR:-$HOME/nnef_data}"
DATA_FLAG="${DATA_FLAG:-hhsuite_CB_v2_pdb_list.csv}"

module purge
module load python/3.10.12-fasrc01

PYTHON_BIN="$(command -v python)"
CONDA_HOOK="$(dirname "$(dirname "$PYTHON_BIN")")/etc/profile.d/conda.sh"
[ -f "$CONDA_HOOK" ] || { echo "[train_yang_style] conda.sh not found at $CONDA_HOOK"; exit 1; }
# shellcheck disable=SC1090
source "$CONDA_HOOK"
set +u
conda activate "$ENV_PREFIX"
set -u

PY="${ENV_PREFIX}/bin/python"
[ -x "$PY" ] || { echo "[train_yang_style] ERROR: no $PY"; exit 1; }
"$PY" -c "import torch; print('[train_yang_style] torch', torch.__version__, 'cuda', torch.cuda.is_available())" || exit 1

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

echo "----- node info -----"
hostname
nvidia-smi
echo "ENV_PREFIX=$ENV_PREFIX DATA_DIR=$DATA_DIR DATA_FLAG=$DATA_FLAG"
echo "---------------------"

cd "$SLURM_SUBMIT_DIR"
mkdir -p runs

"$PY" -u nnef/train_chimeric.py \
    --exp_id "yang_retrain_${SLURM_JOB_ID}" \
    --save_path runs \
    --data_flag "$DATA_FLAG" \
    --seq_type residue \
    --residue_type_num 20 \
    --chimeric \
    --seq_len 14 \
    --batch_size 1000 \
    --embed_size 32 --dim 128 --n_layers 4 --attn_heads 4 \
    --mixture_r 2 --mixture_angle 3 \
    --mixture_seq 1 \
    --mixture_rama 10 \
    --legacy_local_frame \
    --smooth_gaussian --smooth_r 0.3 --smooth_angle 45 \
    --lr 5e-5 --n_warmup_steps 5000 --steps_decay_scale 30000 \
    --coords_angle_loss_lamda 1 --profile_loss_lamda 10 --coords_rama_loss_lamda 1 \
    --use_position_weights --cen_seg_loss_lamda 1 --oth_seg_loss_lamda 3 \
    --log_interval 30 --save_interval 50 --num_workers 4 \
    --epochs 200 --total_num_samples 1000000 \
    --pdb_h5_path "$DATA_DIR/hhsuite_CB_v2.h5" \
    --seq_h5_path "$DATA_DIR/hhsuite_pdb_seq_v2.h5" \
    --rama_h5_path "$DATA_DIR/hhsuite_rama_v2.h5"

echo "[train_yang_style] finished with exit code $?"
