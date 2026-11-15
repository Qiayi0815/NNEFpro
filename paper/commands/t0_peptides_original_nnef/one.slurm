#!/usr/bin/env bash
#SBATCH -p shared,sapphire -c 1 --mem=5G -t 6:00:00 -J recb_exp1 -o /dev/null
cd ~/nnef; sed -n "${SLURM_ARRAY_TASK_ID}p" runs/recb/exp1/cmds.txt | bash
