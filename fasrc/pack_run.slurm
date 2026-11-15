#!/usr/bin/env bash
#SBATCH --job-name=nnef-pack
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --constraint=cc8.0|cc8.6|cc9.0
#SBATCH --output=runs/slurm-pack-%j.out
#SBATCH --error=runs/slurm-pack-%j.out
#
# Pack many small single-protein GPU tasks into ONE allocation. These NNEF
# tasks (decoy scoring, Langevin MD on one chain) use a small fraction of a
# GPU, so running PER_GPU of them concurrently on each GPU multiplies
# throughput -- and sidesteps queue backlogs (gpu_test allows 2 jobs/user,
# 8 GPUs total). --constraint excludes Blackwell (cc12.0) nodes: our torch 2.3.1+cu121
# has no kernels for them (fails with "no kernel image", silently inside decoy scoring).
#
# CMD_FILE: one shell command per line. Each line runs as
#   CUDA_VISIBLE_DEVICES=<slot % n_gpu> bash -c "<line>"  > <LOG_DIR>/task_<n>.log
# Lines must be self-contained and idempotent (re-runs skip finished work).
#
#   sbatch --export=ALL,CMD_FILE=runs/pack/e1_cmds.txt,PER_GPU=8 fasrc/pack_run.slurm
set -euo pipefail

CMD_FILE="${CMD_FILE:?set CMD_FILE=<one command per line>}"
PER_GPU="${PER_GPU:-8}"
LOG_DIR="${LOG_DIR:-runs/pack/logs_${SLURM_JOB_ID}}"
cd "$SLURM_SUBMIT_DIR"
mkdir -p "$LOG_DIR"

# Use the GPUs Slurm actually allocated (nvidia-smi -L lists the whole node).
IFS=, read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:?no GPUs allocated}"
N_GPU=${#GPUS[@]}
SLOTS=$(( N_GPU * PER_GPU ))
N_CMD=$(grep -cv '^\s*$' "$CMD_FILE")
echo "[pack] $N_CMD commands, $N_GPU GPUs (${GPUS[*]}) x $PER_GPU = $SLOTS concurrent slots"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

run_one() {
    local n="$1" line="$2"
    local gpu="${GPUS[$(( n % N_GPU ))]}"
    echo "[pack] start task $n on GPU $gpu: $line"
    if CUDA_VISIBLE_DEVICES=$gpu bash -c "$line" > "$LOG_DIR/task_${n}.log" 2>&1; then
        echo "[pack] done  task $n"
    else
        echo "[pack] FAIL  task $n (exit $?) -> $LOG_DIR/task_${n}.log"
    fi
}

n=0
while IFS= read -r line; do
    [[ -z "${line// }" ]] && continue
    while (( $(jobs -rp | wc -l) >= SLOTS )); do sleep 5; done
    run_one "$n" "$line" &
    n=$(( n + 1 ))
done < "$CMD_FILE"
wait
echo "[pack] all $n tasks finished"
