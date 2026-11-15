#!/usr/bin/env bash
#SBATCH -p shared -c 4 --mem=32G -t 3:00:00 -J recb_an2 -o runs/recb/logs/an2_%j.out
cd ~/nnef; export PYTHONWARNINGS=ignore
PY="$HOME/envs/nnef/bin/python -s"; O=runs/recb/analysis; mkdir -p $O
echo "=== PEP_COMPARE"; $PY eval/decoys/pep_compare.py --models rib_k10_smooth_w10,rib_ic_w10,ctl7_w10,ctl7_ic_w10,cb42_w10,cb7_w10,rib_k10_smooth_w10_s2,yang_smooth_k10_w10,yang_smooth_ic_w10,yang_k10_l10 --baselines rama_aa,rwplus --out $O/pep_compare.csv 2>&1 | cut -c1-150
echo "=== AF2 pep_af2"; $PY eval/decoys/af2_select.py --models rib_k10_smooth_w10_c1000,rib_ic_w10_c1000,cb42_w10_c1000,cb7_w10_c1000 --out $O/af2_pep_af2 2>&1 | cut -c1-150 | head -50
echo "=== AF2 pep_new_af2 (POST-HOC, not the pre-registered test)"; $PY eval/decoys/af2_select.py --set pep_new_af2 --models rib_k10_smooth_w10_c1000,rib_ic_w10_c1000,cb42_w10_c1000,cb7_w10_c1000 --out $O/af2_pep_new_posthoc 2>&1 | cut -c1-150 | head -50
