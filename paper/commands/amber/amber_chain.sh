#!/usr/bin/env bash
# gpu_test allows 2 submitted jobs: each link submits the next two Amber packs, then queues itself after them.
cd ~/nnef; k=${1:?first list}; J=runs/recb/jobs.txt; ids=()
for n in $k $((k+1)); do f=runs/recb/t1/cmds_test/AT_$n.txt; [ -f $f ] || continue
  id=$(sbatch --parsable -J recb_t1At -p gpu_test --gres=gpu:1 -c 4 --mem=16G -t 12:00:00 --export=ALL,CMD_FILE=$f runs/sim_pep/code/amber_pack.slurm) || { echo "submit failed $n"; exit 1; }
  echo "t1At_$n $id" >> $J; ids+=($id)
done
[ ${#ids[@]} -gt 0 ] || { echo "chain finished"; exit 0; }
dep=$(IFS=:; echo "${ids[*]}")
sbatch --parsable --dependency=afterany:$dep -J recb_t1Ac -p test -t 00:10:00 -c 1 --mem=1G -o runs/recb/t1/chain_%j.out runs/recb/t1/amber_chain.sh $((k+2))
