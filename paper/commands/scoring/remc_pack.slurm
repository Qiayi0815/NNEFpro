#!/usr/bin/env bash
#SBATCH --job-name=remc_pack
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=runs/sim_remc/logs/%x_%j.out
# Runs every line of CMD_FILE concurrently on one GPU (one process per line), from ~/nnef.
# Lines whose --save_dir already has meta.json are skipped, so resubmitting is safe.
set -uo pipefail
cd ~/nnef
mkdir -p runs/sim_remc/logs
n=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  sd=$(echo "$line" | sed -E 's/.*--save_dir ([^ ]+).*/\1/')
  if [ -f "$sd/meta.json" ]; then echo "[pack] done already: $sd"; continue; fi
  mkdir -p "$sd"
  echo "[pack] start $sd"
  OMP_NUM_THREADS=1 bash -c "$line" > "$sd/run.log" 2>&1 &
  n=$((n+1))
done < "$CMD_FILE"
wait
echo "[pack] all $n finished"
