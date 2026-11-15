#!/bin/bash
export PYTHONWARNINGS=ignore
P=~/envs/nnef/bin/python
cd ~/nnef
echo "##### 1. pep_new: reference baselines (selection + confidence trend)"
$P -s eval/decoys/af2_select.py --set pep_new_af2 --models rib_k10_smooth_w10_c1000 --out eval/decoys/af2_select_pep_new_ext
$P -s eval/decoys/af2_conf_trend.py --dir eval/decoys/af2_select_pep_new_ext --scorers rib_k10_smooth_w10_c1000,af2rank,rosetta_min,rosetta_raw
echo "##### 2. pep_new: pre-registered test + homolog split (descriptive)"
$P -s eval/decoys/prereg_pep_new.py --dir eval/decoys/af2_select_pep_new_ext --split_col homolog_in_train
echo "##### 3. robustness: medoid GDT_TS as quality"
$P -s eval/decoys/af2_select.py --set pep_new_af2 --quality GDT_TS --models rib_k10_smooth_w10_c1000 --out eval/decoys/af2_select_pep_new_medoid > /dev/null
$P -s eval/decoys/prereg_pep_new.py --dir eval/decoys/af2_select_pep_new_medoid
$P -s eval/decoys/af2_select.py --set pep_af2 --quality GDT_TS --models rib_k10_smooth_w10_c1000 --out eval/decoys/af2_select_pep_af2_medoid > /dev/null
$P -s eval/decoys/af2_conf_trend.py --dir eval/decoys/af2_select_pep_af2_medoid --scorers rib_k10_smooth_w10_c1000,af2rank,rosetta_min
