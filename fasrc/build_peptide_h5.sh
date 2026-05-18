#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# build_peptide_h5.sh -- rebuild the v2 structure + Rama HDF5 with a custom
# ``k`` (block_size = 5 + k). Used to sweep building-block width for the
# peptide-energy experiments on top of v1_rama:
#
#   k=6  -> block_size=11, seq_len=10  (train_v1_rama_peptide_k6.slurm)
#   k=4  -> block_size=9,  seq_len=8   (train_v1_rama_peptide_k4.slurm)
#   k=10 -> block_size=15, seq_len=14  (original v1_pure_rama_v2; default)
#
# Inputs
#   BEAD_DIR   per-chain N/CA/C/CB bead CSVs (defaults to data_hh/bead_csvs)
#   CHIM_DIR   per-chain .chimeric MSA files (defaults to data_hh/chimeric)
#
# Outputs (written under $OUT_DIR, default: nnef/data)
#   hhsuite_CB_peptide_k${K}.h5              structure + group_num + start_id
#   hhsuite_CB_peptide_k${K}_pdb_list.csv    PDB ids + WeightedRandomSampler weight
#   hhsuite_rama_peptide_k${K}.h5            per-block φ/ψ aligned 1:1
#
# Re-uses the existing seq h5 (hhsuite_pdb_seq_v2.h5); its keys are per-chain
# so they are independent of k.
#
# Usage
#   K=6  bash fasrc/build_peptide_h5.sh
#   K=4  bash fasrc/build_peptide_h5.sh
#   # cluster: add --num_workers $(nproc) behind the scenes
#
# The shell env var ``PY`` overrides the python binary (useful under
# conda / Slurm). Defaults to the interpreter on PATH.
# ---------------------------------------------------------------------------

set -euo pipefail

K="${K:-6}"
BEAD_DIR="${BEAD_DIR:-data_hh/bead_csvs}"
CHIM_DIR="${CHIM_DIR:-data_hh/chimeric}"
OUT_DIR="${OUT_DIR:-nnef/data}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PY="${PY:-python}"

if [ ! -d "$BEAD_DIR" ]; then
    echo "[build_peptide_h5] BEAD_DIR not found: $BEAD_DIR" >&2
    exit 1
fi
if [ ! -d "$CHIM_DIR" ]; then
    echo "[build_peptide_h5] CHIM_DIR not found: $CHIM_DIR" >&2
    exit 1
fi
mkdir -p "$OUT_DIR"

CB_H5="$OUT_DIR/hhsuite_CB_peptide_k${K}.h5"
CB_LIST="$OUT_DIR/hhsuite_CB_peptide_k${K}_pdb_list.csv"
RAMA_H5="$OUT_DIR/hhsuite_rama_peptide_k${K}.h5"

echo "[build_peptide_h5] k=$K  block_size=$((5 + K))  seq_len=$((5 + K - 1))"
echo "[build_peptide_h5] BEAD_DIR=$BEAD_DIR"
echo "[build_peptide_h5] CHIM_DIR=$CHIM_DIR"
echo "[build_peptide_h5] OUT_DIR=$OUT_DIR"
echo "[build_peptide_h5] num_workers=$NUM_WORKERS"

# 1) structure / block h5
"$PY" -m nnef.data_prep_scripts.local_extractor_v2 build-h5 \
    --bead_dir  "$BEAD_DIR" \
    --chim_dir  "$CHIM_DIR" \
    --out_h5    "$CB_H5" \
    --pdb_list  "$CB_LIST" \
    --k         "$K" \
    --num_workers "$NUM_WORKERS"

# 2) Ramachandran h5 aligned 1:1 with the new CB h5
"$PY" -m nnef.data_prep_scripts.build_rama_h5_v2 build \
    --cb_h5      "$CB_H5" \
    --bead_dir   "$BEAD_DIR" \
    --out_h5     "$RAMA_H5" \
    --num_workers "$NUM_WORKERS"

# 3) quick shape sanity
"$PY" - <<PYEOF
import h5py, sys
with h5py.File("$CB_H5") as cb, h5py.File("$RAMA_H5") as rm:
    k0 = sorted(cb.keys())[0]
    cb_shape = cb[k0]['coords'].shape
    rm_shape = rm[k0]['rama'].shape
    expect_w = $((5 + K))
    ok = (cb_shape[1] == expect_w and rm_shape[1] == expect_w)
    print(f'[verify] sample={k0}  coords={cb_shape}  rama={rm_shape}  '
          f'expected_block_size={expect_w}  {"OK" if ok else "MISMATCH"}')
    sys.exit(0 if ok else 1)
PYEOF

echo "[build_peptide_h5] wrote:"
echo "    $CB_H5"
echo "    $CB_LIST"
echo "    $RAMA_H5"
