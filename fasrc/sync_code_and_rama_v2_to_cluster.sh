#!/usr/bin/env bash
# Run on your laptop from the nnef repo root (where this file lives under fasrc/).
# Requires SSH/rsync access to FASRC as qzha (see fasrc/README.md).
#
#   bash fasrc/sync_code_and_rama_v2_to_cluster.sh
#
# Then SSH in and submit (pick one):
#   cd ~/nnef && sbatch fasrc/train_v2_dihedral.slurm   # v2 cart+offset + dihedral + rama
#   cd ~/nnef && sbatch fasrc/train_v1_pure.slurm # baseline + rama only

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-qzha@login.rc.fas.harvard.edu}"
REMOTE_REPO="${REMOTE_REPO:-/n/home03/qzha/nnef}"
REMOTE_DATA="${REMOTE_DATA:-/n/home03/qzha/nnef_data}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$ROOT"

echo "==> [1/2] rsync code -> ${REMOTE_HOST}:${REMOTE_REPO}"
echo "    (decoys are NOT included — use fasrc/sync_3drobot_decoys_to_cluster.sh etc.)"
rsync -avh --progress \
  --exclude='.git/' \
  --exclude='__pycache__/' \
  --exclude='runs/' \
  --exclude='params/' \
  --exclude='nnef/data/*.h5' \
  --exclude='nnef/data/decoys/' \
  --exclude='data_hh/' \
  --exclude='*.pdf' \
  ./ "${REMOTE_HOST}:${REMOTE_REPO}/"

echo "==> [2/2] rsync v2 h5 + rama v2 -> ${REMOTE_HOST}:${REMOTE_DATA}"
for f in \
  nnef/data/hhsuite_CB_v2.h5 \
  nnef/data/hhsuite_CB_v2_pdb_list.csv \
  nnef/data/hhsuite_pdb_seq_v2.h5 \
  nnef/data/hhsuite_rama_v2.h5
do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing $f (build or fix path)" >&2
    exit 1
  fi
done

rsync -avh --progress \
  nnef/data/hhsuite_CB_v2.h5 \
  nnef/data/hhsuite_CB_v2_pdb_list.csv \
  nnef/data/hhsuite_pdb_seq_v2.h5 \
  nnef/data/hhsuite_rama_v2.h5 \
  "${REMOTE_HOST}:${REMOTE_DATA}/"

echo ""
echo "Done. On cluster:"
echo "  ssh ${REMOTE_HOST}"
echo "  cd ${REMOTE_REPO} && sbatch fasrc/train_v2_dihedral.slurm"
echo "  squeue -u qzha"
