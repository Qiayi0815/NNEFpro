#!/usr/bin/env bash
#SBATCH --job-name=nnef-hhalign
#SBATCH --partition=shared
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=72:00:00
#SBATCH --array=0-31
#SBATCH --output=runs/slurm-hhalign-%A_%a.out
#SBATCH --error=runs/slurm-hhalign-%A_%a.out
# -----------------------------------------------------------------------------
# Pipeline stage 1+2: HHsuite alignment (hhblits vs PDB70 + hhfilter) over the
# 32,021-chain PISCES set, faithful to data_hh/align.sh (hhblits -n 3;
# hhfilter -id 50 -cov 50).
#
# Design note: we do NOT explode the FASTA into 32k per-chain files (creating
# 32k tiny files on the home NFS in one burst is pathologically slow — each
# close() forces a network flush). Instead:
#   * shard 0 splits the FASTA into N_SHARDS chunk files (32 files -> fast),
#     round-robin by record so every shard gets a mix of short/long proteins.
#   * each array task streams its own chunk, reuses ONE node-local temp query
#     file per record, and writes the filtered MSA to $OUT_DIR/<PDBCHAIN>.a3m
#     on netscratch (fast parallel FS, ample inodes).
# RESUMABLE: a chain whose a3m already exists is skipped.
#
# PREREQ: bash fasrc/setup_hhsuite.sh   (hhblits installed + PDB70 downloaded)
#
# LOCAL SMOKE (no SLURM/PDB70): SMOKE=1 SMOKE_N=5 bash fasrc/align_hhsuite.slurm
# -----------------------------------------------------------------------------
set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/nnef}"
DATA_DIR="${DATA_DIR:-$HOME/nnef_data}"
FASTA="${FASTA:-$DATA_DIR/cullpdb_pc50.0_res0.0-3.0_len40-10000_R1.0_Xray_d2024_11_18_chains32021.fasta}"
PDB70="${PDB70:-/n/netscratch/atchan_lab/qzha/pdb70/pdb70}"
WORK="${WORK:-/n/netscratch/atchan_lab/qzha/hhwork}"   # chunks + a3m on fast FS
CHUNK_DIR="${CHUNK_DIR:-$WORK/chunks}"
OUT_DIR="${OUT_DIR:-$WORK/aligned_a3m}"
N_SHARDS="${N_SHARDS:-32}"                              # keep == #SBATCH --array width
CPU="${CPU:-${SLURM_CPUS_PER_TASK:-16}}"
SMOKE="${SMOKE:-0}"
SMOKE_N="${SMOKE_N:-5}"
TMPDIR="${TMPDIR:-/tmp}/hhq_${SLURM_ARRAY_TASK_ID:-0}_$$"
mkdir -p "$CHUNK_DIR" "$OUT_DIR" "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

HHBLITS="$ENV_PREFIX/bin/hhblits"; HHFILTER="$ENV_PREFIX/bin/hhfilter"
if [[ "$SMOKE" == "0" ]]; then
    module purge 2>/dev/null || true
    module load python/3.10.12-fasrc01
    PYTHON_BIN="$(command -v python)"
    CONDA_HOOK="$(dirname "$(dirname "$PYTHON_BIN")")/etc/profile.d/conda.sh"
    # shellcheck disable=SC1090
    source "$CONDA_HOOK"; set +u; conda activate "$ENV_PREFIX"; set -u
fi
[[ -x "$HHBLITS" ]]  || HHBLITS="$(command -v hhblits  || true)"
[[ -x "$HHFILTER" ]] || HHFILTER="$(command -v hhfilter || true)"

# ---- stage 1: split FASTA into N_SHARDS chunks (shard 0 only; fast) ----------
SENTINEL="$CHUNK_DIR/.chunks_ready"
make_chunks() {
    [[ -f "$FASTA" ]] || { echo "[hhalign] ERROR: FASTA not found: $FASTA"; exit 1; }
    if [[ -f "$SENTINEL" ]]; then echo "[hhalign] chunks already present"; return; fi
    local want; want=$(grep -c '^>' "$FASTA")
    echo "[hhalign] splitting $want chains into $N_SHARDS chunks -> $CHUNK_DIR ..."
    # Round-robin records across N_SHARDS chunk files. Only N_SHARDS handles are
    # ever open (well under the ulimit), and each stays open — no per-file flush.
    awk -v nsh="$N_SHARDS" -v dir="$CHUNK_DIR" '
        /^>/ { cur = dir "/shard_" ((++rec - 1) % nsh) ".fasta"; print > cur; next }
        rec > 0 { print >> cur }
    ' "$FASTA"
    touch "$SENTINEL"
    echo "[hhalign] chunks ready ($(ls "$CHUNK_DIR"/shard_*.fasta | wc -l | tr -d ' ') files)"
}

ensure_chunks() {
    if [[ "${SLURM_ARRAY_TASK_ID:-0}" == "0" ]]; then
        make_chunks
    else
        local w=0
        while [[ ! -f "$SENTINEL" ]]; do
            sleep 15; w=$((w+15))
            [[ $w -ge 1800 ]] && { echo "[hhalign] shard timed out waiting for chunks"; exit 1; }
        done
    fi
}

# ---- align one record (id + its single-sequence query file) ------------------
align_record() {
    local id="$1" qfile="$2"
    local out="$OUT_DIR/${id}.a3m"
    [[ -s "$out" ]] && return 0                        # resume: already done
    local raw="$TMPDIR/${id}_raw.a3m"
    if [[ -z "$HHBLITS" || -z "$HHFILTER" ]]; then
        echo "[hhalign][SMOKE] hhblits absent; WOULD run for $id"
        { echo ">$id"; grep -v '^>' "$qfile" | head -1; } > "$out"   # stub for wiring test
        return 0
    fi
    if "$HHBLITS" -i "$qfile" -d "$PDB70" -oa3m "$raw" -n 3 -cpu "$CPU" >/dev/null 2>&1 \
       && "$HHFILTER" -i "$raw" -id 50 -cov 50 -o "$out" >/dev/null 2>&1; then
        rm -f "$raw"
    else
        echo "[hhalign] FAILED $id"; rm -f "$raw"
    fi
}

# ---- stream a chunk file record-by-record (one reused temp query) ------------
process_chunk() {
    local chunk="$1" limit="${2:-0}" n=0
    local id="" q="$TMPDIR/q.fasta"
    local done_n=0 skip_n=0
    while IFS= read -r line || [[ -n "$line" ]]; do
        if [[ "$line" == ">"* ]]; then
            if [[ -n "$id" ]]; then
                if [[ -s "$OUT_DIR/${id}.a3m" ]]; then skip_n=$((skip_n+1)); else align_record "$id" "$q"; done_n=$((done_n+1)); fi
                n=$((n+1)); [[ "$limit" -gt 0 && "$n" -ge "$limit" ]] && { id=""; break; }
            fi
            id="${line#>}"; id="${id%% *}"
            printf '>%s\n' "$id" > "$q"
        else
            [[ -n "$id" ]] && printf '%s\n' "$line" >> "$q"
        fi
    done < "$chunk"
    if [[ -n "$id" ]]; then
        if [[ -s "$OUT_DIR/${id}.a3m" ]]; then skip_n=$((skip_n+1)); else align_record "$id" "$q"; done_n=$((done_n+1)); fi
    fi
    echo "[hhalign] chunk $(basename "$chunk"): aligned $done_n, skipped(resume) $skip_n"
}

# ---- driver ------------------------------------------------------------------
cd "${SLURM_SUBMIT_DIR:-$PWD}"

if [[ "$SMOKE" == "1" ]]; then
    make_chunks
    echo "[hhalign][SMOKE] processing first $SMOKE_N records of shard_0"
    process_chunk "$CHUNK_DIR/shard_0.fasta" "$SMOKE_N"
    echo "[hhalign][SMOKE] outputs:"; ls -1 "$OUT_DIR" | head
    exit 0
fi

ensure_chunks
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
CHUNK="$CHUNK_DIR/shard_${SHARD}.fasta"
[[ -f "$CHUNK" ]] || { echo "[hhalign] no chunk for shard $SHARD ($CHUNK)"; exit 0; }
echo "[hhalign] shard $SHARD -> $CHUNK ($(grep -c '^>' "$CHUNK") chains)"
process_chunk "$CHUNK"
echo "[hhalign] shard $SHARD done"
