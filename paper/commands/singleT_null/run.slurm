#!/bin/bash
#SBATCH -p serial_requeue -c 2 --mem=6G -t 0-08:00:00 --requeue -J singleTnull
#SBATCH -o /n/home03/qzha/nnef/runs/recb/logs/singleTnull_%A_%a.out
cd ~/nnef; hostname; bash runs/recb/remc/singleT_null/one.sh $SLURM_ARRAY_TASK_ID
