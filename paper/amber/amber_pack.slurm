#!/usr/bin/env bash
#SBATCH --job-name=amber_pep
#SBATCH --partition=gpu_test,gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=runs/sim_pep/logs/%x_%j.out
# Runs every line of CMD_FILE concurrently on one GPU (small GBn2 peptides), waits for all.
cd ~/nnef
nvidia-smi -L
while read -r line; do [ -z "$line" ] && continue; echo "[pack] $line"; bash -c "$line" & sleep 2; done < "$CMD_FILE"
wait
echo "[pack] all done"
