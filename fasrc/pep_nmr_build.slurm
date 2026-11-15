#!/bin/bash
#SBATCH -J pep_build
#SBATCH -p serial_requeue,sapphire
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 3:00:00
#SBATCH -o /n/holylabs/LABS/atchan_lab/Lab/qzha/pep_nmr_work/logs/build_%j.log
module load python/3.10.12-fasrc01 >/dev/null 2>&1
source $(dirname $(dirname $(command -v python)))/etc/profile.d/conda.sh
conda activate /n/holylabs/LABS/atchan_lab/Lab/qzha/envs/pepbench
cd ~/nnef
python -s nnef/data_prep_scripts/build_pep_nmr.py build --work /n/holylabs/LABS/atchan_lab/Lab/qzha/pep_nmr_work --nproc 16
