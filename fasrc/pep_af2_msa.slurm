#!/bin/bash
#SBATCH -J pep_af2_msa
#SBATCH -p test
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 6:00:00
#SBATCH -o /n/holylabs/LABS/atchan_lab/Lab/qzha/pep_af2_work/logs/af2_msa_%j.log
# Step 1 of pep_af2: query the ColabFold MMseqs2 server once for all 206 peptides
# (CPU only; colabfold_batch handles the queries sequentially -> polite). Writes <id>.a3m
# into af2_a3m/, which both prediction runs (msa / dropout) reuse.
# Check first that `colabfold_batch --help` lists --msa-only (ColabFold >= 1.5.3).
CF=/n/holylabs/LABS/atchan_lab/Lab/qzha/envs/localcolabfold/colabfold-conda/bin
export LD_LIBRARY_PATH=/n/holylabs/LABS/atchan_lab/Lab/qzha/envs/localcolabfold/colabfold-conda/lib:$LD_LIBRARY_PATH  # system libstdc++ too old
W=${WORK:-/n/holylabs/LABS/atchan_lab/Lab/qzha/pep_af2_work}  # WORK=<dir> for other sets (e.g. pep_new); also pass sbatch -o for the log
Q=${QUERIES:-queries.csv}; A3M=${A3M:-af2_a3m}
$CF/colabfold_batch --help > $W/logs/cf_help.txt 2>&1
grep -q -- "--msa-only" $W/logs/cf_help.txt || { echo "colabfold_batch has no --msa-only (or failed to import); see logs/cf_help.txt"; exit 1; }
$CF/colabfold_batch --msa-only --msa-mode mmseqs2_uniref_env $W/$Q $W/$A3M
n=$(ls $W/$A3M/*.a3m 2>/dev/null | wc -l); echo "a3m files: $n"
[ "$n" -ge "$(( $(wc -l < $W/$Q) - 1 ))" ] || exit 1
