#!/usr/bin/env bash
# Laptop → FASRC: ship CASP14 decoy **beads** tree under ``nnef/data/decoys/casp14/``.
#
# Scoring reads ``<TARGET>/list.csv`` and ``<TARGET>/<decoy_id>_bead.csv`` (see
# ``evaluate_decoys.py``). Raw ``*.pdb`` are optional for NNEF eval.
#
#   bash fasrc/sync_casp14_decoys_to_cluster.sh
#
# If your **full** tree lives outside the repo (common when the git checkout
# only contains a subset like T1053), set the source explicitly:
#
#   CASP14_SRC=/path/to/nnef/data/decoys/casp14 \
#     bash fasrc/sync_casp14_decoys_to_cluster.sh
#
# Large / first-time sync: run from a machine on fast internet to FASRC.
# Override: ``REMOTE_HOST``, ``REMOTE_REPO`` (same defaults as ``sync_3drobot``).

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-qzha@login.rc.fas.harvard.edu}"
REMOTE_REPO="${REMOTE_REPO:-/n/home03/qzha/nnef}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${CASP14_SRC:-$ROOT/nnef/data/decoys/casp14}"
DST="${REMOTE_HOST}:${REMOTE_REPO}/nnef/data/decoys/casp14"

RSYNC_OPTS=( -avh --progress -z )

cd "$ROOT"
if [[ ! -d "$SRC" ]]; then
  echo "[sync_casp14] ERROR: missing directory: $SRC" >&2
  exit 1
fi

if [[ ! -f "${SRC}/pdb_no_missing_residue.csv" ]]; then
  echo "[sync_casp14] WARN: no pdb_no_missing_residue.csv under $SRC (expected for full eval)"
fi

echo "==> rsync casp14 (beads + lists; excludes decoy_loss_* output dirs)"
echo "    SRC=$SRC"
echo "    DST=$DST/"
rsync "${RSYNC_OPTS[@]}" \
  --exclude='decoy_loss_*/' \
  "${SRC}/" "${DST}/"

echo ""
echo "Done. On cluster, quick sanity checks:"
echo "  ssh ${REMOTE_HOST} 'wc -l ${REMOTE_REPO}/nnef/data/decoys/casp14/pdb_no_missing_residue.csv; ls ${REMOTE_REPO}/nnef/data/decoys/casp14 | grep -E \"^T[0-9]\" | wc -l'"
echo "  (second line ≈ number of T* target folders; should match your full CASP14 list.)"
