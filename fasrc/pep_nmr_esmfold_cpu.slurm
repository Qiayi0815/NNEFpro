#!/bin/bash
#SBATCH -J pep_esmf
#SBATCH -p serial_requeue,sapphire,shared
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o /n/holylabs/LABS/atchan_lab/Lab/qzha/pep_nmr_work/logs/esmfold_cpu_%j.log
module load python/3.10.12-fasrc01 >/dev/null 2>&1
source $(dirname $(dirname $(command -v python)))/etc/profile.d/conda.sh
conda activate /n/holylabs/LABS/atchan_lab/Lab/qzha/envs/pepbench
export HF_HOME=/n/holylabs/LABS/atchan_lab/Lab/qzha/pep_nmr_work/hf
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
cd ~/nnef
python -s nnef/data_prep_scripts/build_pep_nmr.py esmfold --work /n/holylabs/LABS/atchan_lab/Lab/qzha/pep_nmr_work ${IDS:+--ids $IDS}
