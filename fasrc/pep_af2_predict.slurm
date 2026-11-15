#!/bin/bash
#SBATCH -J pep_af2_pred
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --constraint="cc8.0|cc8.6|cc9.0"
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o /n/holylabs/LABS/atchan_lab/Lab/qzha/pep_af2_work/logs/af2_pred_%x_%j.log
# Step 2 of pep_af2: AF2 monomer (alphafold2_ptm), 5 models x 5 seeds, 3 recycles, from the
# server MSAs in af2_a3m/. SET=msa (default) or SET=drop (adds --use-dropout).
#   sbatch --export=ALL,SET=msa  fasrc/pep_af2_predict.slurm
#   sbatch --export=ALL,SET=drop fasrc/pep_af2_predict.slurm   (only after the msa job finished: 1 GPU job at a time)
# colabfold_batch skips queries whose outputs exist, so a re-submission resumes.
SET=${SET:-msa}
CF=/n/holylabs/LABS/atchan_lab/Lab/qzha/envs/localcolabfold/colabfold-conda/bin
export LD_LIBRARY_PATH=/n/holylabs/LABS/atchan_lab/Lab/qzha/envs/localcolabfold/colabfold-conda/lib:$LD_LIBRARY_PATH  # system libstdc++ too old
W=${WORK:-/n/holylabs/LABS/atchan_lab/Lab/qzha/pep_af2_work}  # WORK=<dir> for other sets (e.g. pep_new); also pass sbatch -o for the log
EXTRA=""; [[ "$SET" == "drop" ]] && EXTRA="--use-dropout"
nvidia-smi -L
A3M=${A3M:-af2_a3m}; OUT=${OUT:-af2_$SET}
$CF/colabfold_batch --model-type alphafold2_ptm --num-models 5 --num-seeds 5 --num-recycle 3 \
    --random-seed 0 $EXTRA $W/$A3M $W/$OUT
n=$(ls $W/$OUT/*.pdb 2>/dev/null | wc -l); echo "pdb files: $n"
[ "$n" -gt 0 ] || exit 1
