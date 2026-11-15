#!/bin/bash
# Submit the whole pep_af2 pipeline as one dependency chain (survives ssh drops):
# smoke (2 peptides) msa -> predict -> check  ==afterok==>  full msa -> predict(msa) -> predict(drop) -> build
set -e
cd ~/nnef
W=/n/holylabs/LABS/atchan_lab/Lab/qzha/pep_af2_work
head -3 $W/queries.csv > $W/queries_smoke.csv
a=$(sbatch --parsable --export=ALL,QUERIES=queries_smoke.csv,A3M=af2_a3m_smoke fasrc/pep_af2_msa.slurm)
b=$(sbatch --parsable --dependency=afterok:$a --export=ALL,SET=msa,A3M=af2_a3m_smoke,OUT=af2_smoke fasrc/pep_af2_predict.slurm)
c=$(sbatch --parsable --dependency=afterok:$b fasrc/pep_af2_check.slurm)
d=$(sbatch --parsable --dependency=afterok:$c fasrc/pep_af2_msa.slurm)
e=$(sbatch --parsable --dependency=afterok:$d --export=ALL,SET=msa fasrc/pep_af2_predict.slurm)
# gpu_test allows 2 submitted jobs per user (b and e); the optional dropout set goes to the main GPU queue
f=$(sbatch --parsable --dependency=afterok:$e -p gpu,gpu_requeue --export=ALL,SET=drop fasrc/pep_af2_predict.slurm)
# build/score as soon as the main (MSA) set exists; re-run the build after the dropout set to add it
g=$(sbatch --parsable --dependency=afterok:$e fasrc/pep_af2_build.slurm)
echo "smoke msa $a  smoke predict $b  check $c  | full msa $d  predict-msa $e  predict-drop $f  build $g"
