#!/usr/bin/env bash
#SBATCH -p shared -c 4 --mem=32G -t 3:00:00 -J recb_an1 -o runs/recb/logs/an1_%j.out
cd ~/nnef; export PYTHONWARNINGS=ignore
PY="$HOME/envs/nnef/bin/python -s"
echo "=== T0"; $PY eval/md_eval/arch_t0_summary.py 2>&1 | grep -E "samp|model|rib_k10_smooth_w10 |rib_ic|cb42|cb7|yang_k10_l10|yang_smooth_k10_w10"
echo "=== SHUFFLE"; $PY nnef/scripts/seq_shuffle_test.py --analyze --out_dir eval/seq_shuffle_cb > runs/recb/shuf_summary.txt 2>&1; cat runs/recb/shuf_summary.txt | cut -c1-260
echo "=== PROT"; $PY eval/decoys/prot_analysis.py > runs/recb/prot_summary.txt 2>&1; grep -E "model|rib_ic|cb42|cb7|rib_k10_smooth_w10 |yang_k10_l10 |3DRobot" runs/recb/prot_summary.txt | head -30
