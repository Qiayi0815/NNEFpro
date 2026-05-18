#!/usr/bin/env bash
# Run on your **Mac** from the nnef repo root (same layout as fasrc/pull_decoy_loss_from_cluster.sh).
# Pulls v3 eval outputs for CASP14 **T1053** from FASRC for the two standard v3 runs:
#   v3_full_6223467  and  v3_full_rama_v2_6229240
#
# After rsync, writes eval/T1053_v3_dual_compare.csv (two rows: T1053 summary per model).
#
# Prereq: stop long-running GPU jobs first if you do not want them to keep writing:
#   ssh qzha@login.rc.fas.harvard.edu 'scancel JOBID1 JOBID2'
#
# Usage:
#   bash fasrc/pull_T1053_v3_pair_from_cluster.sh
#
# Env:
#   REMOTE_HOST   (default: qzha@login.rc.fas.harvard.edu)
#   REMOTE_REPO   (default: /n/home03/qzha/nnef)

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-qzha@login.rc.fas.harvard.edu}"
REMOTE_REPO="${REMOTE_REPO:-/n/home03/qzha/nnef}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE="${REMOTE_HOST}:${REMOTE_REPO}"

RUN1="v3_full_6223467"
RUN2="v3_full_rama_v2_6229240"
EV1="eval/${RUN1}_casp14_3drobot"
EV2="eval/${RUN2}_casp14_3drobot"
D1="nnef/data/decoys/casp14/decoy_loss_${RUN1}"
D2="nnef/data/decoys/casp14/decoy_loss_${RUN2}"

echo "==> Pull eval dirs (summary + plots/) for both runs"
mkdir -p "${ROOT}/${EV1}" "${ROOT}/${EV2}"
rsync -avh --progress "${REMOTE}/${EV1}/" "${ROOT}/${EV1}/"
rsync -avh --progress "${REMOTE}/${EV2}/" "${ROOT}/${EV2}/"

echo "==> Pull T1053 per-decoy tables (casp14)"
mkdir -p "${ROOT}/${D1}" "${ROOT}/${D2}"
rsync -avh --progress "${REMOTE}/${D1}/T1053_decoy_loss.csv" "${ROOT}/${D1}/"
rsync -avh --progress "${REMOTE}/${D2}/T1053_decoy_loss.csv" "${ROOT}/${D2}/"

OUT_MERGE="${ROOT}/eval/T1053_v3_dual_compare.csv"
{
  echo "model,pdb,decoy_set,metric,n_decoys,pearson_r,pearson_p,spearman_r,spearman_p"
  awk -v m="$RUN1" -F, 'NR>1 && $1=="T1053" && $2=="casp14" {print m "," $0}' "${ROOT}/${EV1}/summary.csv"
  awk -v m="$RUN2" -F, 'NR>1 && $1=="T1053" && $2=="casp14" {print m "," $0}' "${ROOT}/${EV2}/summary.csv"
} > "${OUT_MERGE}"

echo ""
echo "Done."
echo "  Summary merge:  ${OUT_MERGE}"
echo "  Scatter plots:  ${EV1}/plots/casp14_T1053_scatter.pdf  ${EV2}/plots/casp14_T1053_scatter.pdf"
echo "  Decoy CSVs:     ${D1}/T1053_decoy_loss.csv  ${D2}/T1053_decoy_loss.csv"
