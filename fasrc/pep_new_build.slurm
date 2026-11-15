#!/bin/bash
#SBATCH -J pepnew_build
#SBATCH -p test
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 3:00:00
#SBATCH -o /n/holylabs/LABS/atchan_lab/Lab/qzha/pep_new_work/nmr/logs/build_%j.log
# pep_new (eval/decoys/PREREG_pep_new.md): NMR decoy set nnef/data/decoys/pep_new for the peptides
# that survive curation + pep_nmr-similarity removal (members of pep_new/seq_clusters_id40.tsv,
# from build_pep_new_clusters.py), then training-set leakage annotation and finalize.
# No ESMFold stage. IDS=<comma list> restricts the build (smoke) and skips leakage/finalize.
set -e
PY=/n/holylabs/LABS/atchan_lab/Lab/qzha/envs/pepbench/bin/python
W=/n/holylabs/LABS/atchan_lab/Lab/qzha/pep_new_work/nmr
cd ~/nnef
D=nnef/data/decoys/pep_new
B="$PY -s nnef/data_prep_scripts/build_pep_nmr.py"
if [[ -n "$IDS" ]]; then
  $B build --work $W --out $W/smoke_out --ids $IDS --nproc 2
  exit 0
fi
ids=$(cut -f2 $D/seq_clusters_id40.tsv | sort -u | paste -sd,)
$B build --work $W --out $D --ids $ids --nproc 16
$B leakage --work $W --nproc 16
$B finalize --work $W --out $D --id_table $W/raw/pep_new_ids.csv
