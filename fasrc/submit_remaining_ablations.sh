#!/usr/bin/env bash
# Submit v1_pure + ESM cache + v3_full while v2_cart_offset keeps running.
#
# From the cluster repo root (e.g. ~/nnef), after rsync of latest code:
#
#   bash fasrc/submit_remaining_ablations.sh
#
# Notes
# -----
# * Your current job (e.g. 6160264) holds one ``gpu`` A100. Extra ``gpu``
#   training jobs may show as PD until another GPU is free — this is normal.
# * ``precompute_esm.slurm`` uses ``gpu_h200`` by default, so it can start in
#   parallel with v2 on ``gpu`` if your account has H200 access and quota.
# * ``train_v3_full.slurm`` is chained with ``--dependency=afterok:$ESM_JOB`` so
#   it only runs after the ESM h5 exists.
# * To submit only a subset, comment out blocks below or run the individual
#   ``sbatch`` lines by hand.

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p runs

echo "==> Repo: $(pwd)"
echo "==> Submitting jobs (check: squeue -u \$USER) ..."

V1_JOB="$(sbatch --parsable fasrc/train_v1_pure.slurm)"
echo "    v1_pure          JOBID=$V1_JOB   (partition: gpu, see train_v1_pure.slurm)"

ESM_JOB="$(sbatch --parsable fasrc/precompute_esm.slurm)"
echo "    precompute_esm   JOBID=$ESM_JOB   (partition: gpu_h200)"

V3_JOB="$(sbatch --parsable --dependency=afterok:"$ESM_JOB" fasrc/train_v3_full.slurm)"
echo "    v3_full          JOBID=$V3_JOB   (starts after ESM job $ESM_JOB succeeds)"

echo ""
echo "Done. Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f runs/slurm-<JOBID>.out"
echo ""
echo "If v3_full should not wait for ESM, cancel it and run:"
echo "  sbatch fasrc/train_v3_full.slurm"
