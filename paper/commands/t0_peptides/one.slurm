#!/usr/bin/env bash
#SBATCH -p shared,sapphire -c 1 --mem=5G -t 3:00:00 -J recb_t0p -o /dev/null
cd ~/nnef; sed -n "${SLURM_ARRAY_TASK_ID}p" runs/recb/t0pep/cmds.txt | bash
