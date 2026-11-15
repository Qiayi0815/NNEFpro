#!/bin/bash
#SBATCH -J pep_af2_check
#SBATCH -p test
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 0:10:00
#SBATCH -o /n/holylabs/LABS/atchan_lab/Lab/qzha/pep_af2_work/logs/af2_check_%j.log
# Gate between the 2-peptide smoke run and the full run: each smoke peptide must have
# 25 predicted models (5 models x 5 seeds); otherwise exit 1 so the dependent full chain never starts.
W=${WORK:-/n/holylabs/LABS/atchan_lab/Lab/qzha/pep_af2_work}  # WORK=<dir> for other sets (e.g. pep_new); also pass sbatch -o for the log
ok=1
for id in $(tail -n +2 $W/queries_smoke.csv | cut -d, -f1); do
  n=$(ls $W/af2_smoke/${id}*_unrelaxed_*.pdb 2>/dev/null | wc -l); echo "$id: $n models"
  [ "$n" -ge 25 ] || ok=0
done
[ $ok = 1 ] && echo SMOKE_OK || { echo SMOKE_FAILED; exit 1; }
