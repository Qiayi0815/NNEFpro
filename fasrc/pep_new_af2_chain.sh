#!/bin/bash
# ColabFold for pep_new (eval/decoys/PREREG_pep_new.md), exactly as pep_af2: the same slurm
# scripts with WORK pointed at pep_new_work. Prerequisite: $W/queries.csv from
#   build_pep_af2.py fasta --work $W --nmr_set pep_new --targets nnef/data/decoys/pep_new/pep_new_reps.txt
#   bash fasrc/pep_new_af2_chain.sh smoke   # 2 peptides: msa -> predict(msa) -> check
#   bash fasrc/pep_new_af2_chain.sh full    # msa -> {predict(msa) on gpu_test, predict(drop) on gpu}
#                                           #     -> build (afterok on BOTH predicts) -> nnef/data/decoys/pep_new_af2
# Build only (BASELINES=0, --blind): no scoring / analysis of the held-out set.
set -e
cd ~/nnef
export WORK=/n/holylabs/LABS/atchan_lab/Lab/qzha/pep_new_work
W=$WORK; L=$W/logs
mkdir -p $L
case "${1:-}" in
smoke)
  head -3 $W/queries.csv > $W/queries_smoke.csv
  a=$(sbatch --parsable -J pepnew_msa_smoke -o $L/af2_msa_smoke_%j.log \
        --export=ALL,QUERIES=queries_smoke.csv,A3M=af2_a3m_smoke fasrc/pep_af2_msa.slurm)
  b=$(sbatch --parsable -J pepnew_pred_smoke -o $L/af2_pred_smoke_%j.log --dependency=afterok:$a \
        --export=ALL,SET=msa,A3M=af2_a3m_smoke,OUT=af2_smoke fasrc/pep_af2_predict.slurm)
  c=$(sbatch --parsable -J pepnew_check -o $L/af2_check_%j.log --dependency=afterok:$b fasrc/pep_af2_check.slurm)
  echo "smoke msa $a  predict $b  check $c"
  ;;
full)
  export BASELINES=0
  export BUILD_ARGS="--work $W --nmr_work $W/nmr --nmr_set pep_new --out_set pep_new_af2 --targets $HOME/nnef/nnef/data/decoys/pep_new/pep_new_reps.txt --blind"
  d=$(sbatch --parsable -J pepnew_msa -o $L/af2_msa_%j.log fasrc/pep_af2_msa.slurm)
  e=$(sbatch --parsable -J pepnew_pred_msa -o $L/af2_pred_msa_%j.log --dependency=afterok:$d \
        --export=ALL,SET=msa fasrc/pep_af2_predict.slurm)
  # gpu_test allows 2 jobs per user -> the dropout set runs in parallel on the main GPU queues
  f=$(sbatch --parsable -J pepnew_pred_drop -o $L/af2_pred_drop_%j.log --dependency=afterok:$d \
        -p gpu,gpu_requeue --export=ALL,SET=drop fasrc/pep_af2_predict.slurm)
  g=$(sbatch --parsable -J pepnew_af2_build -o $L/af2_build_%j.log --dependency=afterok:$e:$f fasrc/pep_af2_build.slurm)
  echo "full msa $d  predict-msa $e  predict-drop $f  build $g (afterok:$e:$f)"
  ;;
*) echo "usage: $0 smoke|full"; exit 1 ;;
esac
