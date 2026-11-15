#!/bin/bash
#SBATCH -p serial_requeue -c 8 --mem=16G -t 1-00:00:00 --requeue -J singleT
#SBATCH -o /n/home03/qzha/nnef/runs/recb/logs/singleT_%A_%a.out
cd ~/nnef; hostname; bash runs/recb/remc/singleT/one.sh $SLURM_ARRAY_TASK_ID
