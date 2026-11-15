#!/usr/bin/env bash
# After pep_af2_build: score pep_af2 with the NNEF models (CPU arrays), then af2_select analysis.
#   sbatch --dependency=afterok:<build> -p serial_requeue,shared,sapphire -c 1 --mem=2G -t 0:30:00 fasrc/pep_af2_score_chain.sh
set -euo pipefail
cd ~/nnef
SM="--seq_head masked_center --smooth_nbr --smooth_tau 1.0 --energy_profile_weight 10"
export SET=pep_af2 CHUNK=10
# AF2 models often have extended termini: a 20 A block cutoff drops ribbon terminal blocks -> score ribbon with cutoff 1000
J1=$(bash fasrc/pep_score_cpu.sh rib_k10_smooth_w10_c1000 runs/smooth_rib_k10_54636530 ribbon 10 "$SM --struct_dist_cutoff 1000")
J2=$(bash fasrc/pep_score_cpu.sh yang_smooth_k10_w10 runs/yang_smooth_k10_55128199 yang 10 "$SM")
J3=$(bash fasrc/pep_score_cpu.sh yang_k10_l10 runs/ho_yang_k10_l10_54636554 yang 10 "" 10)
echo "score arrays: $J1 $J2 $J3"
sbatch --parsable --dependency=afterany:$J1:$J2:$J3 -p serial_requeue,shared,sapphire -c 2 --mem=16G -t 1:00:00 \
  --job-name=af2_select -o runs/pack/af2_select.out --wrap "export PYTHONWARNINGS=ignore; ~/envs/nnef/bin/python -s eval/decoys/af2_select.py \
  --models rib_k10_smooth_w10_c1000,yang_smooth_k10_w10,yang_k10_l10 --hybrid rib_k10_smooth_w10_c1000:yang_smooth_k10_w10"
