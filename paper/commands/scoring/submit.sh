#!/usr/bin/env bash
# Ideal-CB re-evaluation (Monte Carlo only). Writes job ids to runs/recb/jobs.txt
set -euo pipefail
cd ~/nnef
J=runs/recb/jobs.txt; : > $J
SM="--seq_head masked_center --smooth_nbr --smooth_tau 1.0 --energy_profile_weight 10"
ORIG=runs/smooth_rib_k10_54636530; CTL=runs/smooth_rib_k10_s7_56982226; C42=runs/cbrib_k10_s42_56982226; C7=runs/cbrib_k10_s7_56982226
sc() { # SET TAG RUN FAMILY FLAGS cb(ideal|orig)
  local set=$1 tag=$2 run=$3 fam=$4 fl=$5 cb=$6
  [ "$cb" = ideal ] && export NNEF_CB=ideal || unset NNEF_CB
  echo "$set $tag $(SET=$set bash fasrc/pep_score_cpu.sh $tag $run $fam 10 "$fl")" >> $J
}
export CHUNK=27
for set in pep_nmr pep_term pep_norm; do
  sc $set rib_ic_w10 $ORIG ribbon "$SM" ideal; sc $set cb42_w10 $C42 ribbon "$SM" ideal; sc $set cb7_w10 $C7 ribbon "$SM" ideal
done
sc pep_nmr ctl7_w10 $CTL ribbon "$SM" orig; sc pep_nmr ctl7_ic_w10 $CTL ribbon "$SM" ideal
sc pep_nmr yang_smooth_ic_w10 runs/yang_smooth_k10_55128199 yang "$SM" ideal
export CHUNK=10
for set in pep_af2 pep_new_af2; do
  sc $set rib_ic_w10_c1000 $ORIG ribbon "$SM --struct_dist_cutoff 1000" ideal
  sc $set cb42_w10_c1000 $C42 ribbon "$SM --struct_dist_cutoff 1000" ideal
  sc $set cb7_w10_c1000 $C7 ribbon "$SM --struct_dist_cutoff 1000" ideal
done
unset NNEF_CB
export OUT=eval/per_residue TIME=12:00:00
for spec in rib_ic_w10:$ORIG cb42_w10:$C42 cb7_w10:$C7; do sc 3DRobot_set ${spec%%:*} ${spec#*:} ribbon "$SM" ideal; done
unset OUT TIME
GC="cc8.0|cc8.6|cc9.0"
gp() { # file partition time
  echo "$(basename $1) $(sbatch --parsable -J recb_$(basename $1 .txt) -p $2 -t $3 --constraint="$GC" --export=ALL,CMD_FILE=$1 runs/sim_remc/remc_pack.slurm)" >> $J
}
gp runs/recb/cmds/A_00.txt gpu_test 12:00:00
echo "t0_fig4 $(sbatch --parsable -J recb_t0 --export=ALL,CMD_FILE=runs/recb/t0_fig4.txt,PER_GPU=8 fasrc/pack_run.slurm)" >> $J
for f in runs/recb/cmds/C_*.txt runs/recb/cmds/H_*.txt; do gp $f gpu 18:00:00; done
gp runs/recb/cmds/F_00.txt gpu 2-00:00:00
for f in runs/recb/cmds/L_*.txt; do gp $f gpu 2-12:00:00; done
wc -l $J
