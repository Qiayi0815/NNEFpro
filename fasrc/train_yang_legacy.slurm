#!/usr/bin/env bash
#SBATCH --job-name=nnef-yang-legacy
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=36:00:00
#SBATCH --output=runs/slurm-%j.out
#SBATCH --error=runs/slurm-%j.err
#
# ----------------------------------------------------------------------------
# nnef-yang-legacy: faithful Yang 2022-style LocalTransformer baseline.
#
# This replaces the old train_yang_style.slurm (which trained on v2 data with
# mixture_rama=10 and ended up bit-identical to v1_pure_rama_v2). Changes vs
# train_yang_style.slurm:
#   * pdb_h5_path = hhsuite_CB_cullpdb.h5   (legacy v1-era extraction;
#                                            ~2.388x more blocks per PDB than
#                                            hhsuite_CB_v2.h5, but we keep the
#                                            same --total_num_samples so both
#                                            runs see the same 200M-sample
#                                            gradient budget.)
#   * mixture_rama = 0                      (no Ramachandran head, matches the
#                                            Yang 2022 paper exactly; the CE
#                                            loss path already guards on
#                                            m_rama>0, so loss_rama == 0.)
#   * rama_h5_path = <nonexistent>          (DatasetLocalGenCM catches the
#                                            OSError and falls back to zero
#                                            rama tensors + zero mask; combined
#                                            with mixture_rama=0 this skips
#                                            the rama path entirely.)
#   * drop --coords_rama_loss_lamda         (no effect when loss_rama == 0.)
#
# Unchanged (= matches v1_pure_rama_v2 for a clean architectural ablation):
#   * same training csv (hhsuite_CB_v2_pdb_list.csv) = same 12,445 PDBs
#   * same seq h5, same seq_len/batch/optim/seed/epochs/total_num_samples
#   * --legacy_local_frame + --mixture_seq 1 (Yang-paper architecture flags;
#     note --legacy_local_frame only affects inference protein_os.py, not the
#     training pipeline.)
#
# NOTE on seed: --mixture_rama 0 vs 10 changes linear_out_x output dim by 60
# units, so PyTorch's RNG consumes fewer numbers during init. All subsequent
# layer inits differ between this run and v1_pure_rama_v2 even with the same
# seed=42. That is intended: we want a faithful Yang-paper architecture, not
# a partial-weight sibling of v1_pure.
#
# ----------------------------------------------------------------------------
# Before submitting: rsync the legacy h5 to the cluster once (~1.9 GB)
#
#   rsync -av --progress nnef/data/hhsuite_CB_cullpdb.h5 \
#     qzha@login.rc.fas.harvard.edu:/n/home03/qzha/nnef_data/hhsuite_CB_cullpdb.h5
#
# Then on the cluster:
#
#   ssh qzha@login.rc.fas.harvard.edu
#   cd ~/nnef && sbatch fasrc/train_yang_legacy.slurm
#
# After training, pull the checkpoint:
#
#   ONLY_PT=1 bash fasrc/pull_yang_retrain_from_cluster.sh <JOBID>
#
# For MD eval (legacy frame + mixture_seq=1 + mixture_rama=0), md_eval_one.sh
# currently hardcodes LOAD_EXP=runs/yang_retrain_6594199 with mixture_rama=10;
# you'll need to add a new MODEL_KEY or override LOAD_EXP + drop mixture_rama.
# ----------------------------------------------------------------------------

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/nnef}"
DATA_DIR="${DATA_DIR:-$HOME/nnef_data}"
DATA_FLAG="${DATA_FLAG:-hhsuite_CB_v2_pdb_list.csv}"

module purge
module load python/3.10.12-fasrc01

PYTHON_BIN="$(command -v python)"
CONDA_HOOK="$(dirname "$(dirname "$PYTHON_BIN")")/etc/profile.d/conda.sh"
[ -f "$CONDA_HOOK" ] || { echo "[train_yang_legacy] conda.sh not found at $CONDA_HOOK"; exit 1; }
# shellcheck disable=SC1090
source "$CONDA_HOOK"
set +u
conda activate "$ENV_PREFIX"
set -u

PY="${ENV_PREFIX}/bin/python"
[ -x "$PY" ] || { echo "[train_yang_legacy] ERROR: no $PY"; exit 1; }
"$PY" -c "import torch; print('[train_yang_legacy] torch', torch.__version__, 'cuda', torch.cuda.is_available())" || exit 1

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

# Fail fast if the legacy cullpdb h5 wasn't synced to this cluster.
if [[ ! -f "$DATA_DIR/hhsuite_CB_cullpdb.h5" ]]; then
    echo "[train_yang_legacy] ERROR: $DATA_DIR/hhsuite_CB_cullpdb.h5 not found."
    echo "    Run (from laptop, repo root) before submitting:"
    echo "      rsync -av --progress nnef/data/hhsuite_CB_cullpdb.h5 \\"
    echo "        qzha@login.rc.fas.harvard.edu:$DATA_DIR/hhsuite_CB_cullpdb.h5"
    exit 1
fi

"$PY" -u nnef/train_chimeric.py \
    --exp_id "yang_legacy_${SLURM_JOB_ID}" \
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
    --mixture_rama 0 \
    --legacy_local_frame \
    --smooth_gaussian --smooth_r 0.3 --smooth_angle 45 \
    --lr 5e-5 --n_warmup_steps 5000 --steps_decay_scale 30000 \
    --coords_angle_loss_lamda 1 --profile_loss_lamda 10 \
    --use_position_weights --cen_seg_loss_lamda 1 --oth_seg_loss_lamda 3 \
    --log_interval 30 --save_interval 50 --num_workers 4 \
    --epochs 200 --total_num_samples 1000000 \
    --pdb_h5_path "$DATA_DIR/hhsuite_CB_cullpdb.h5" \
    --seq_h5_path "$DATA_DIR/hhsuite_pdb_seq_v2.h5" \
    --rama_h5_path "$DATA_DIR/__NO_RAMA_SENTINEL__.h5"

echo "[train_yang_legacy] finished with exit code $?"
