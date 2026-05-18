#!/usr/bin/env bash
#SBATCH --job-name=nnef-yang-legacy-nl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm-yang-legacy-nl-%j.out
#SBATCH --error=runs/slurm-yang-legacy-nl-%j.out
#
# yang_legacy retrain on hhsuite_CB_newlogic_v2keys.h5.
#
# Motivation (confirmed by diagnose_yang_exp1_on_cullpdb.slurm + check3):
#   * runs/yang_legacy_7317784 was trained on hhsuite_CB_cullpdb.h5, which was
#     built by the OLD local_extractor_hh.py (pure 15-nearest-neighbor blocks).
#   * Our current architecture uses the MODIFIED building-block construction
#     (5 sequence neighbors [gc-2..gc+2] + 10 topological nearest = 15 per block).
#   * The modified extractor is data_hh/qlocal_ex_modify_add4_hh.py, which
#     produced data_hh/hhsuite_CB_newlogic.h5.
#   * Key-renamed copy (12ASA -> 12AS_A to match seq_v2) lives at
#     nnef/data/hhsuite_CB_newlogic_v2keys.h5 (built by
#     nnef/data_prep_scripts/rename_newlogic_keys.py).
#
# All arch flags unchanged from train_yang_legacy.slurm. Only difference:
#   --pdb_h5_path   $DATA_DIR/hhsuite_CB_newlogic_v2keys.h5   (was cullpdb.h5)
#   --data_flag     hhsuite_CB_newlogic_v2keys_pdb_list.csv   (was v2 list)
#   (--seq_h5_path, --rama_h5_path unchanged)
#   (--legacy_local_frame, --mixture_seq 1, --mixture_rama 0 unchanged)
#
# Before submitting:
#   1. Build the key-renamed h5 locally (~10 s):
#        python nnef/data_prep_scripts/rename_newlogic_keys.py
#   2. Sync the h5 to the cluster's data dir (~600 MB):
#        rsync -av --progress nnef/data/hhsuite_CB_newlogic_v2keys.h5 \\
#          qzha@login.rc.fas.harvard.edu:/n/home03/qzha/nnef_data/
#   3. Sync the code + pdb_list (csv travels with code sync):
#        bash fasrc/sync_code_only_to_cluster.sh
#   4. Submit:
#        ssh qzha@login.rc.fas.harvard.edu 'cd nnef && sbatch fasrc/train_yang_legacy_newlogic.slurm'
#
# After training, decoy eval (T1053 + T1026) + mode2 MD on T1026 to compare
# against params/exp1 (paper: -0.90 / -0.84 Pearson, MD RMSD 1.12 Å).
#
# Expected outcome: retrain lands closer to paper numbers on T1053 and T1026
# decoy Pearson, and tighter native basin in mode2 MD than the cullpdb retrain.
# If it still underperforms paper, the remaining gap is in the upstream
# hhsuite/bead-csv stage (not our extractor), which is a bigger investigation.
# ----------------------------------------------------------------------------

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/nnef}"
DATA_DIR="${DATA_DIR:-$HOME/nnef_data}"
DATA_FLAG="${DATA_FLAG:-hhsuite_CB_newlogic_v2keys_pdb_list.csv}"
PDB_H5="${PDB_H5:-$DATA_DIR/hhsuite_CB_newlogic_v2keys.h5}"

module purge
module load python/3.10.12-fasrc01

PYTHON_BIN="$(command -v python)"
CONDA_HOOK="$(dirname "$(dirname "$PYTHON_BIN")")/etc/profile.d/conda.sh"
[ -f "$CONDA_HOOK" ] || { echo "[train_yang_nl] conda.sh not found at $CONDA_HOOK"; exit 1; }
# shellcheck disable=SC1090
source "$CONDA_HOOK"
set +u
conda activate "$ENV_PREFIX"
set -u

PY="${ENV_PREFIX}/bin/python"
[ -x "$PY" ] || { echo "[train_yang_nl] ERROR: no $PY"; exit 1; }
"$PY" -c "import torch; print('[train_yang_nl] torch', torch.__version__, 'cuda', torch.cuda.is_available())" || exit 1

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

echo "----- node info -----"
hostname
nvidia-smi
echo "ENV_PREFIX=$ENV_PREFIX DATA_DIR=$DATA_DIR DATA_FLAG=$DATA_FLAG PDB_H5=$PDB_H5"
echo "---------------------"

cd "$SLURM_SUBMIT_DIR"
mkdir -p runs

if [[ ! -f "$PDB_H5" ]]; then
    echo "[train_yang_nl] ERROR: $PDB_H5 not found on this cluster."
    echo "    Run (from laptop, repo root) before submitting:"
    echo "      python nnef/data_prep_scripts/rename_newlogic_keys.py"
    echo "      rsync -av --progress nnef/data/hhsuite_CB_newlogic_v2keys.h5 \\"
    echo "        qzha@login.rc.fas.harvard.edu:$DATA_DIR/"
    exit 1
fi

if [[ ! -f "nnef/data/$DATA_FLAG" ]]; then
    echo "[train_yang_nl] ERROR: nnef/data/$DATA_FLAG not found."
    echo "    Run the rename script locally to produce it, then re-sync code."
    exit 1
fi

"$PY" -u nnef/train_chimeric.py \
    --exp_id "yang_legacy_nl_${SLURM_JOB_ID}" \
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
    --pdb_h5_path "$PDB_H5" \
    --seq_h5_path "$DATA_DIR/hhsuite_pdb_seq_v2.h5" \
    --rama_h5_path "$DATA_DIR/__NO_RAMA_SENTINEL__.h5"

echo "[train_yang_nl] finished with exit code $?"
