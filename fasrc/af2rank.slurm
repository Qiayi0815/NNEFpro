#!/bin/bash
#SBATCH -J af2rank
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -o runs/pack/af2rank_%j.out
# AF2Rank baseline on pep_af2.  env: TARGETS (comma list, optional), CHUNK (i/n, default 0/1)
#   sbatch -p gpu_test -t 1:00:00 --export=ALL,TARGETS=1ERD,2LE7 fasrc/af2rank.slurm      (smoke)
#   sbatch -p gpu,gpu_requeue -t 12:00:00 fasrc/af2rank.slurm                             (full)
L=/n/holylabs/LABS/atchan_lab/Lab/qzha/envs/localcolabfold
export PYTHONPATH=/n/holylabs/LABS/atchan_lab/Lab/qzha/envs/af2rank_pylib
export LD_LIBRARY_PATH=$L/colabfold-conda/lib:$LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd ~/nnef
$L/colabfold-conda/bin/python -u eval/baselines_pep/af2rank_score.py --data_dir $L/colabfold \
  ${TARGETS:+--targets $TARGETS} --chunk ${CHUNK:-0/1} --set ${SET:-pep_af2} ${AF2WORK:+--work $AF2WORK} 2>&1 | grep -v -i warn
