#!/bin/bash
#SBATCH -J pep_af2_build
#SBATCH -p serial_requeue,shared,sapphire
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 4:00:00
#SBATCH -o /n/holylabs/LABS/atchan_lab/Lab/qzha/pep_af2_work/logs/af2_build_%j.log
# Step 3: beads + quality labels -> nnef/data/decoys/pep_af2, then baselines and top-1 analysis.
PY=/n/holylabs/LABS/atchan_lab/Lab/qzha/envs/pepbench/bin/python
cd /n/home03/qzha/nnef
# Other sets: BUILD_ARGS="--work W --nmr_work NW --nmr_set S --out_set O --targets T --blind" and
# BASELINES=0 (build only; no scoring/analysis). Defaults reproduce the pep_af2 run.
$PY -s nnef/data_prep_scripts/build_pep_af2.py build --nproc 16 $BUILD_ARGS
$PY -s nnef/data_prep_scripts/build_pep_af2.py finalize $BUILD_ARGS
rc=$?; [[ "${BASELINES:-1}" == "1" ]] || exit $rc
$PY -s eval/baselines_pep/baselines_pep.py score --set pep_af2 --nproc 16 \
    --baselines rama_aa,rama_generic,rwplus,dssp_agree,rg,plddt,ptm
$PY -s eval/baselines_pep/baselines_pep.py af2_top1
