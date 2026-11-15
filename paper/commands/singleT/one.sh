#!/usr/bin/env bash
# one.sh N [extra args]: run line N of singleT/cmds.txt (resumes from state.pkl; skips finished chains)
cd ~/nnef; line=$(sed -n "${1}p" runs/recb/remc/singleT/cmds.txt); [ -z "$line" ] && exit 0
shift; extra="$*"
if [ -n "$extra" ]; then line=$(echo "$line" | sed -E "s#--save_dir runs/recb/remc/singleT/#--save_dir runs/recb/remc/singleT/_smoke/#")" $extra"; fi
sd=$(echo "$line" | sed -E "s/.*--save_dir ([^ ]+).*/\1/"); mkdir -p $sd
[ -f $sd/meta.json ] && { echo "$sd done"; exit 0; }
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
echo "$line" > $sd/cmd.txt
bash -c "$line" >> $sd/run.log 2>&1
