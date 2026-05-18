#!/usr/bin/env bash
#SBATCH --job-name=nnef-v1-esm-peptide-k6
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm-%j.out
#SBATCH --error=runs/slurm-%j.err
#
# v1 + ESM + Rama trained on peptide-width blocks (k=6, block_size=11, seq_len=10).
# ESM h5 is keyed per-chain so it is k-independent and reuses hhsuite_esm_v2.h5.
# --seq_len auto-detected from the structure h5.
#
# Submit:  sbatch fasrc/train_v1_esm_peptide_k6.slurm

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/nnef}"
DATA_DIR="${DATA_DIR:-$HOME/nnef_data}"

module purge
module load python/3.10.12-fasrc01

PYTHON_BIN="$(command -v python)"
CONDA_HOOK="$(dirname "$(dirname "$PYTHON_BIN")")/etc/profile.d/conda.sh"
[ -f "$CONDA_HOOK" ] || { echo "[train_v1_esm_peptide_k6] conda.sh not found at $CONDA_HOOK"; exit 1; }
# shellcheck disable=SC1090
source "$CONDA_HOOK"
set +u
conda activate "$ENV_PREFIX"
set -u

PY="${ENV_PREFIX}/bin/python"
[ -x "$PY" ] || { echo "[train_v1_esm_peptide_k6] ERROR: no $PY"; exit 1; }
"$PY" -c "import torch; print('[train_v1_esm_peptide_k6] torch', torch.__version__, 'cuda', torch.cuda.is_available())" || exit 1

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

ESM_H5="$DATA_DIR/hhsuite_esm_v2.h5"
if [ ! -f "$ESM_H5" ]; then
    echo "[train_v1_esm_peptide_k6] ERROR: ESM cache not found at $ESM_H5"
    exit 1
fi

echo "----- node info -----"
hostname
nvidia-smi
echo "ENV_PREFIX=$ENV_PREFIX DATA_DIR=$DATA_DIR ESM_H5=$ESM_H5"
echo "---------------------"

cd "$SLURM_SUBMIT_DIR"
mkdir -p runs

"$PY" -u nnef/train_chimeric.py \
    --exp_id "v1_esm_rama_peptide_k6_${SLURM_JOB_ID}" \
    --save_path runs \
    --data_flag hhsuite_CB_peptide_k6_pdb_list.csv \
    --seq_type residue \
    --residue_type_num 20 \
    --chimeric \
    --batch_size 1000 \
    --embed_size 32 --dim 128 --n_layers 4 --attn_heads 4 \
    --mixture_r 2 --mixture_angle 3 --mixture_rama 10 \
    --smooth_gaussian --smooth_r 0.3 --smooth_angle 45 \
    --lr 5e-5 --n_warmup_steps 5000 --steps_decay_scale 30000 \
    --coords_angle_loss_lamda 1 --profile_loss_lamda 10 --coords_rama_loss_lamda 1 \
    --use_position_weights --cen_seg_loss_lamda 1 --oth_seg_loss_lamda 3 \
    --log_interval 30 --save_interval 50 --num_workers 4 \
    --epochs 200 --total_num_samples 1000000 \
    --pdb_h5_path "$DATA_DIR/hhsuite_CB_peptide_k6.h5" \
    --seq_h5_path "$DATA_DIR/hhsuite_pdb_seq_v2.h5" \
    --rama_h5_path "$DATA_DIR/hhsuite_rama_peptide_k6.h5" \
    --use_esm \
    --esm_h5_path "$ESM_H5" \
    --esm_dim_in 1152 \
    --esm_dim_out 32

echo "[train_v1_esm_peptide_k6] finished with exit code $?"
