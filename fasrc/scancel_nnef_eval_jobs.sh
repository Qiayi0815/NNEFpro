#!/usr/bin/env bash
# Run **on the Cannon login node** (or via ssh) to cancel Slurm jobs by ID.
#
#   scancel 6450883 6450884
#
# Or:
#   bash fasrc/scancel_nnef_eval_jobs.sh 6450883 6450884
#
set -euo pipefail
if [[ $# -eq 0 ]]; then
  echo "usage: $0 <jobid> [<jobid> ...]"
  exit 1
fi
exec scancel "$@"
