#!/usr/bin/env bash
#SBATCH -p shared,sapphire -c 1 --mem=6G -t 12:00:00 -J t0_3dr -o /n/holylabs/LABS/atchan_lab/Lab/qzha/nnef_casp14full_view/runs/recb/t0_3dr/logs/%a.out
cd /n/holylabs/LABS/atchan_lab/Lab/qzha/nnef_casp14full_view; sed -n "${SLURM_ARRAY_TASK_ID}p" runs/recb/t0_3dr/cmds.txt | bash
