#!/bin/bash
#SBATCH -J pep_esmf
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --constraint="cc8.0|cc8.6|cc9.0"
#SBATCH --mem=48G
#SBATCH -t 3:00:00
#SBATCH -o /n/holylabs/LABS/atchan_lab/Lab/qzha/pep_nmr_work/logs/esmfold_%j.log
module load python/3.10.12-fasrc01 >/dev/null 2>&1
source $(dirname $(dirname $(command -v python)))/etc/profile.d/conda.sh
conda activate /n/holylabs/LABS/atchan_lab/Lab/qzha/envs/pepbench
export HF_HOME=/n/holylabs/LABS/atchan_lab/Lab/qzha/pep_nmr_work/hf
cd ~/nnef
nvidia-smi -L
python -s nnef/data_prep_scripts/build_pep_nmr.py esmfold --work /n/holylabs/LABS/atchan_lab/Lab/qzha/pep_nmr_work ${IDS:+--ids $IDS}
