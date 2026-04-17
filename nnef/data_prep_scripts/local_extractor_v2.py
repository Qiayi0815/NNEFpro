"""V2 building-block extractor (thesis: Qiayi Zha, BIO303 2024/25 §2.3).

Differences vs the original Yang 2022 / ``local_extractor_hh.py`` pipeline
--------------------------------------------------------------------------

1. **Centers include the 4 terminal residues.** The original scheme requires
   ``gc-2, gc-1, gc+1, gc+2`` to all exist, which makes residues at positions
   ``1, 2, n-1, n`` impossible centers. This version promotes them to valid
   centers so peptides (where terminal fraction is large) are explicitly
   supervised. See Figure 1 in the thesis.

2. **Local coordinate frame is now intra-residue (backbone-atom defined).**
   Old frame used the previous/next residues' Cα positions to orient the
   axes, which is undefined at termini. New frame, defined solely from the
   central residue's own backbone atoms (Figure 2b):

       origin      = CA_cen
       x-axis      = (N  - CA) / ||N  - CA||
       y-axis      = component of (C - CA) orthogonal to x, normalized
       z-axis      = x x y

   This is the standard protein backbone frame (same convention used by e.g.
   AlphaFold's IPA). It is rigid-body invariant and well-defined everywhere.

3. **Two block layouts, picked by whether the center is terminal:**

   * **Interior center** (gc not in ``first_two ∪ last_two``): the classic
     NNEF layout of the peptide ±2 window (5 residues) concatenated with the
     ``k`` nearest non-window neighbors by Cβ distance.

   * **Terminal center**: no contiguous peptide window is available, so the
     block is simply the ``5 + k`` nearest residues by Cβ distance
     (default ``k=10`` -> 15-residue block, same total size as interior).
     All 15 positions carry ``segment_info = 5`` ("other"), and
     ``re_order_df_g`` fills in ``seg`` based on group_num continuity.

HDF5 schema
-----------

Matches the baseline `DatasetLocalGenCM` reader (same keys as the original
extractor writes), so training code reads v2 files via ``--pdb_h5_path`` with
zero dataset-side changes:

    <pdb_id>/seq         (num_blocks, 15)   int8
    <pdb_id>/group_num   (num_blocks, 15)   int
    <pdb_id>/coords      (num_blocks, 15, 3) float32   # in the local frame
    <pdb_id>/seg         (num_blocks, 15)   int16
    <pdb_id>/start_id    (num_blocks, 15)   int8
    <pdb_id>/res_counts  (num_blocks, 3)    int16      # counts at 8/10/12 A

CLI
---

    # PDB -> per-chain bead CSV (populates N, CA, C, CB for every residue)
    python -m data_prep_scripts.local_extractor_v2 extract-beads \
        --pdb PATH/TO/input.pdb --out PATH/TO/input_bead.csv

    # bead_csvs/ directory -> consolidated v2 HDF5
    python -m data_prep_scripts.local_extractor_v2 build-h5 \
        --bead_dir  bead_csvs \
        --chim_dir  chimeric \
        --out_h5    hhsuite_CB_cullpdb_v2.h5 \
        --pdb_list  hhsuite_CB_cullpdb_v2_list.csv

The resulting h5 is a drop-in replacement for the baseline cullpdb h5: point
``--pdb_h5_path`` at it on the training side and everything else is
untouched.

Attribution
-----------

Block layout + intra-residue frame come from the thesis. The prototype lives
in ``data_hh/qlocal_ex_modify_add4_hh.py`` and was used to produce
``data_hh/hhsuite_CB_newlogic.h5``. This module is a cleaned and packaged
port: same math, proper CLI, no absolute/relative path surprises, no prints
in the hot loop.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from typing import Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, Selection
from tqdm import tqdm

# Make ``paths`` importable whether we run as ``python -m data_prep_scripts...``
# from the ``nnef/`` folder or as ``python -m nnef.data_prep_scripts...`` from
# the repo root.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_DIR = os.path.dirname(_THIS_DIR)
if _PACKAGE_DIR not in sys.path:
    sys.path.insert(0, _PACKAGE_DIR)

from paths import data_path  # noqa: E402

# ---------------------------------------------------------------------------
# 1. PDB -> bead CSV (populates N, CA, C, CB per residue; required by v2).
# ---------------------------------------------------------------------------

_BEAD_V2_COLS = [
    'chain_id', 'group_num', 'group_name',
    'xn',  'yn',  'zn',
    'xca', 'yca', 'zca',
    'xc',  'yc',  'zc',
    'xcb', 'ycb', 'zcb',
]


def extract_beads_v2(pdb_path: str,
                     out_csv: Optional[str] = None,
                     amino_acids_csv: Optional[str] = None) -> pd.DataFrame:
    """Parse a PDB and emit a bead CSV carrying backbone (N, CA, C) and CB.

    Glycine has no Cβ -> we fall back to Cα for the Cβ column, matching the
    Yang 2022 convention (and matching how ``utils.extract_beads`` handles
    Gly).

    Any residue missing any of ``N``, ``CA``, ``C`` is skipped: without them
    the local frame for a block centered there is undefined, so it cannot
    contribute to the dataset anyway.
    """
    if amino_acids_csv is None:
        amino_acids_csv = data_path('amino_acids.csv')
    aa = pd.read_csv(amino_acids_csv)
    three = [x.upper() for x in aa.AA3C]
    three_to_one = {x.upper(): y for x, y in zip(aa.AA3C, aa.AA)}

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('X', pdb_path)
    residues = Selection.unfold_entities(structure, 'R')

    rows = []
    for res in residues:
        rname = res.get_resname()
        if rname not in three:
            continue
        try:
            n = res['N'].get_coord()
            ca = res['CA'].get_coord()
            c = res['C'].get_coord()
        except KeyError:
            # Missing any backbone atom -> can't build the local frame if this
            # residue ever becomes a center; drop it cleanly.
            continue
        if rname == 'GLY':
            cb = ca
        else:
            try:
                cb = res['CB'].get_coord()
            except KeyError:
                continue

        rows.append((
            res.parent.id,
            res.id[1],
            three_to_one[rname],
            n[0], n[1], n[2],
            ca[0], ca[1], ca[2],
            c[0], c[1], c[2],
            cb[0], cb[1], cb[2],
        ))

    df = pd.DataFrame(rows, columns=_BEAD_V2_COLS)
    if out_csv is not None:
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or '.', exist_ok=True)
        df.to_csv(out_csv, index=False, float_format='%.3f')
    return df


# ---------------------------------------------------------------------------
# 2. Local-frame math.
# ---------------------------------------------------------------------------

def compute_local_frame(n: np.ndarray,
                        ca: np.ndarray,
                        c: np.ndarray,
                        eps: float = 1e-6) -> Optional[np.ndarray]:
    """Return the 3x3 rotation matrix R such that ``R @ (p - CA)`` gives the
    position of ``p`` in the central residue's intra-residue frame.

    Convention (thesis Figure 2b):

        x-hat =  (N  - CA) / || . ||
        y-hat =  component of (C - CA) orthogonal to x-hat, normalized
        z-hat =  x-hat x y-hat

    Returns ``None`` if the geometry is degenerate (collinear backbone,
    atoms on top of each other, etc.), so the caller can skip this residue.
    """
    x_vec = n - ca
    x_norm = np.linalg.norm(x_vec)
    if x_norm < eps:
        return None
    x_hat = x_vec / x_norm

    c_vec = c - ca
    c_perp = c_vec - np.dot(c_vec, x_hat) * x_hat
    y_norm = np.linalg.norm(c_perp)
    if y_norm < eps:
        return None
    y_hat = c_perp / y_norm

    z_hat = np.cross(x_hat, y_hat)
    z_norm = np.linalg.norm(z_hat)
    if z_norm < eps:
        return None
    z_hat = z_hat / z_norm

    return np.vstack([x_hat, y_hat, z_hat])


# ---------------------------------------------------------------------------
# 3. Block extractor.
# ---------------------------------------------------------------------------

def _re_order_block(df_block: pd.DataFrame) -> pd.DataFrame:
    """Assign ``seg`` (group-num connectivity id) and sort block rows.

    Positions 0..4 are the 'central segment' (peptide ±2 for interior blocks,
    or simply the five closest by Cβ distance for terminal blocks). Positions
    5..14 are the top-k 'others'; within these we start a new ``seg`` id
    whenever two consecutive rows are NOT adjacent in group_num. The final
    sort ``(segment, seg_dist, group_num)`` groups chain-contiguous neighbors
    together, which is what the Yang 2022 model expects as input.
    """
    df_block = df_block.sort_values(by=['segment', 'group_num']).reset_index(drop=True)
    group_num = df_block['group_num'].values
    distance = df_block['distance'].values

    seg = np.ones(len(df_block), dtype=int)
    if len(seg) >= 6:
        seg[5] = 2
    for i in range(6, len(seg)):
        seg[i] = seg[i - 1] if group_num[i] == group_num[i - 1] + 1 else seg[i - 1] + 1

    seg_dist = np.zeros(len(seg))
    for s in np.unique(seg):
        seg_dist[seg == s] = distance[seg == s].mean()

    df_block['seg'] = seg
    df_block['seg_dist'] = seg_dist
    return df_block.sort_values(by=['segment', 'seg_dist', 'group_num']).reset_index(drop=True)


def _build_block_df(center_num: int,
                    gnum_seg: np.ndarray,
                    gname_seg: np.ndarray,
                    local_xyz: np.ndarray,
                    dist_local: np.ndarray,
                    segment_info: np.ndarray,
                    counts: Tuple[int, int, int]) -> pd.DataFrame:
    c8, c10, c12 = counts
    return pd.DataFrame({
        'center_num': center_num,
        'group_num': gnum_seg,
        'group_name': gname_seg,
        'local_x': local_xyz[:, 0],
        'local_y': local_xyz[:, 1],
        'local_z': local_xyz[:, 2],
        'distance': dist_local,
        'segment': segment_info,
        'count8a': c8,
        'count10a': c10,
        'count12a': c12,
    })


def extract_blocks_v2(df_beads: pd.DataFrame, k: int = 10) -> Optional[pd.DataFrame]:
    """Compute v2 blocks for a single chain's bead CSV.

    Returns a long DataFrame with ``15 * num_valid_blocks`` rows, or ``None``
    if the chain has fewer than 15 residues or no block passes validity
    checks.
    """
    if df_beads.shape[0] < 15:
        return None

    missing = [c for c in _BEAD_V2_COLS if c not in df_beads.columns]
    if missing:
        raise ValueError(f"bead CSV missing columns: {missing}")

    df_beads = df_beads.sort_values('group_num').reset_index(drop=True)
    group_num = df_beads['group_num'].values.astype(int)
    group_name = df_beads['group_name'].values

    unique_g = np.unique(group_num)
    if unique_g.size < 4:
        return None
    first_two = set(unique_g[:2].tolist())
    last_two = set(unique_g[-2:].tolist())

    N_all = df_beads[['xn', 'yn', 'zn']].values
    CA_all = df_beads[['xca', 'yca', 'zca']].values
    C_all = df_beads[['xc', 'yc', 'zc']].values
    CB_all = df_beads[['xcb', 'ycb', 'zcb']].values

    block_size = 5 + k  # 15 by default
    blocks = []

    for gc in group_num:
        cen_idx_arr = np.where(group_num == gc)[0]
        if cen_idx_arr.size == 0:
            continue
        cen_idx = int(cen_idx_arr[0])

        R = compute_local_frame(N_all[cen_idx], CA_all[cen_idx], C_all[cen_idx])
        if R is None:
            continue
        CA_cen = CA_all[cen_idx]
        CB_cen = CB_all[cen_idx]
        dist_glob = np.linalg.norm(CB_all - CB_cen, axis=1)

        is_terminal = (gc in first_two) or (gc in last_two)

        if is_terminal:
            # --- Terminal center: 15 nearest residues by Cβ distance ------
            idx_combined = np.argsort(dist_glob)[:block_size]
            if idx_combined.size < block_size:
                continue
            mask_others = ~np.isin(np.arange(len(dist_glob)), idx_combined)
            segment_info = np.full(block_size, 5, dtype=int)

        else:
            # --- Interior center: peptide ±2 window + top-k others --------
            surrounding = [gc - 2, gc - 1, gc + 1, gc + 2]
            if not all(s in group_num for s in surrounding):
                continue

            window_mask = np.isin(group_num, [gc - 2, gc - 1, gc, gc + 1, gc + 2])
            if np.count_nonzero(window_mask) != 5:
                continue

            mask_others = ~window_mask
            dist_others_for_sort = np.where(mask_others, dist_glob, np.inf)
            topk_idx = np.argsort(dist_others_for_sort)[:k]
            idx_window = np.where(window_mask)[0]
            idx_combined = np.concatenate([idx_window, topk_idx])
            if idx_combined.size != block_size:
                continue

            gnum_block = group_num[idx_combined]
            segment_info = np.full(block_size, 5, dtype=int)
            segment_info[gnum_block == gc]     = 0
            segment_info[gnum_block == gc - 1] = 1
            segment_info[gnum_block == gc + 1] = 2
            segment_info[gnum_block == gc - 2] = 3
            segment_info[gnum_block == gc + 2] = 4

        # --- Coordinates in the local frame -------------------------------
        cb_local = (R @ (CB_all[idx_combined] - CA_cen).T).T            # (15, 3)
        cb_center_local = (R @ (CB_cen - CA_cen)).reshape(3)
        dist_local = np.linalg.norm(cb_local - cb_center_local, axis=1)  # (15,)

        # --- Ambient residue counts (8 / 10 / 12 A) -----------------------
        dist_others = dist_glob[mask_others]
        c8  = int((dist_others <  8.0).sum())
        c10 = int((dist_others < 10.0).sum())
        c12 = int((dist_others < 12.0).sum())

        df_block = _build_block_df(
            center_num=int(gc),
            gnum_seg=group_num[idx_combined],
            gname_seg=group_name[idx_combined],
            local_xyz=cb_local,
            dist_local=dist_local,
            segment_info=segment_info,
            counts=(c8, c10, c12),
        )
        if df_block.shape[0] != block_size:
            continue

        blocks.append(_re_order_block(df_block))

    if not blocks:
        return None
    return pd.concat(blocks, ignore_index=True)


# ---------------------------------------------------------------------------
# 4. Batch driver: bead CSVs -> consolidated HDF5.
# ---------------------------------------------------------------------------

def _write(grp: 'h5py.Group', name: str, arr: np.ndarray, dtype: str,
           compression: Optional[str], level: int, shuffle: bool) -> None:
    """Chunked + compressed dataset write with sane defaults.

    Chunks are sized to ~256 KiB along the first (num_blocks) axis so
    per-chunk filter overhead stays low. Compression is skipped for
    degenerate (num_blocks == 0) datasets since HDF5 refuses empty
    chunked datasets without a maxshape.
    """
    if arr.shape[0] == 0 or compression is None:
        grp.create_dataset(name, data=arr, dtype=dtype)
        return
    per_row = int(np.prod(arr.shape[1:]) if arr.ndim > 1 else 1) \
        * np.dtype(dtype).itemsize
    rows = max(1, (256 * 1024) // max(per_row, 1))
    rows = min(rows, arr.shape[0])
    chunks = (rows,) + tuple(arr.shape[1:])
    grp.create_dataset(
        name,
        data=arr,
        dtype=dtype,
        chunks=chunks,
        compression=compression,
        compression_opts=level if compression == 'gzip' else None,
        shuffle=shuffle,
    )


def _load_index_vocab(amino_acids_csv: str) -> dict:
    aa = pd.read_csv(amino_acids_csv)
    vocab = {}
    for _, row in aa.iterrows():
        sl = row['AA'].upper()
        tl = row['AA3C'].upper()
        idx = int(row['idx']) - 1
        vocab[sl] = idx
        vocab[tl] = idx
    return vocab


def _candidate_chain_letters(pdb_id: str, default_chain: str) -> list:
    """Order in which to try chain ids extracted from a chimeric pdb_id.

    The cullpdb chimeric corpus uses two naming conventions interchangeably:

      * Underscore form: ``2G2C_A``  -> chain is the suffix after ``_``.
      * Concatenated form: ``12ASA`` -> chain is the trailing letter
        (any single-character chain id, including digits like ``1``).

    We also fall back to ``default_chain`` and finally to ``None`` to mean
    "any chain present in the bead CSV", which the caller will iterate
    over. This lets us cover all 13.8k chimeric files instead of only the
    ~12.6k that have chain id ``A``.
    """
    candidates = []
    if '_' in pdb_id:
        candidates.append(pdb_id.split('_', 1)[1])
    elif len(pdb_id) >= 5:
        # Last char of a 5+ char id is the chain letter.
        candidates.append(pdb_id[-1])
    if default_chain not in candidates:
        candidates.append(default_chain)
    return candidates


def _seq_to_letters(group_name_series: pd.Series) -> str:
    """Bead ``group_name`` is already 1-letter (e.g. 'A', 'C', ...);
    just join. Defined as a helper so the matching path stays readable.
    """
    return ''.join(group_name_series.values.astype(str))


def _best_offset_identity(seq_hh: str,
                          gn: np.ndarray,
                          bead_seq: str,
                          min_coverage: float = 0.5,
                          ) -> Tuple[Optional[int], float, float]:
    """Find the integer offset ``o`` maximising fractional residue identity
    between ``seq_hh[gn + o]`` and ``bead_seq`` over the positions that
    land inside the chimeric range.

    This is the GENERAL offset search -- not restricted to ``{-1, 0}``.
    PDB ATOM records use wildly varying numbering schemes:

      * chimeric has a leading His-tag / cloning fragment / MSA prefix
        that is absent from the crystal (offset e.g. +4, +7)
      * the bead is a single domain of a larger protein; ``group_num``
        starts in the hundreds or thousands (offset e.g. -260, -790)
      * ``group_num`` includes negative values for engineered tags

    Trying only ``{-1, 0}`` rejects >50% of legitimate chains. Scanning
    all offsets ``[ -gn_max, L_hh - 1 - gn_min ]`` is fully vectorised
    in numpy and completes in well under a millisecond per chain.

    For each offset ``o`` we compute:

      * ``n_valid`` = number of ``gn[i] + o`` in ``[0, L_hh)``
      * ``coverage`` = ``n_valid / L_bead`` -- rejected if below
        ``min_coverage`` (guards against accidental matches from a
        handful of positions)
      * ``identity`` = fraction of valid positions where the chimeric
        letter equals the bead letter

    Returns ``(offset_or_None, best_identity, best_coverage)``. If no
    offset meets ``min_coverage``, ``offset`` is ``None``.
    """
    arr = np.frombuffer(seq_hh.encode('ascii'), dtype=np.uint8)
    bead_arr = np.frombuffer(bead_seq.encode('ascii'), dtype=np.uint8)
    L_hh = int(arr.shape[0])
    L_bd = int(bead_arr.shape[0])
    if L_hh == 0 or L_bd == 0:
        return None, 0.0, 0.0

    # Offsets where at least one bead position is in range.
    o_lo = int(-gn.max())
    o_hi = int(L_hh - 1 - gn.min())
    if o_hi < o_lo:
        return None, 0.0, 0.0

    offsets = np.arange(o_lo, o_hi + 1, dtype=np.int64)
    # idx_mat[o_index, i] = gn[i] + offsets[o_index]
    idx_mat = gn[None, :].astype(np.int64) + offsets[:, None]
    valid_mat = (idx_mat >= 0) & (idx_mat < L_hh)
    n_valid = valid_mat.sum(axis=1)
    coverage = n_valid / L_bd

    # Clip out-of-range indices so the fancy index doesn't throw;
    # valid_mat masks them out of the match count below.
    idx_clip = np.clip(idx_mat, 0, L_hh - 1)
    match_mat = (arr[idx_clip] == bead_arr[None, :]) & valid_mat
    n_match = match_mat.sum(axis=1)

    # identity over valid positions; guarded against zero-divide.
    identity = np.zeros_like(coverage, dtype=np.float64)
    nz = n_valid > 0
    identity[nz] = n_match[nz] / n_valid[nz]

    ok = coverage >= min_coverage
    if not ok.any():
        return None, 0.0, 0.0

    # Pick the offset with highest identity; break ties by higher coverage.
    identity_filt = np.where(ok, identity, -1.0)
    best_i = int(np.argmax(identity_filt))
    return int(offsets[best_i]), float(identity[best_i]), float(coverage[best_i])


def _match_chain_and_offset(pdb_id: str,
                            df_beads_all: pd.DataFrame,
                            chimeric_dir: str,
                            default_chain: str,
                            min_identity: float = 0.9,
                            min_coverage: float = 0.5,
                            ) -> Optional[tuple]:
    """Find the (chain_id, idx_offset) whose bead residue sequence best
    matches the chimeric sequence at positions that fall inside the
    chimeric range, accepting the match if fractional identity >=
    ``min_identity`` AND the overlap covers at least ``min_coverage``
    of the bead sequence.

    Returns ``None`` if no candidate chain clears both thresholds. If
    the chimeric file is missing and the bead CSV has a single chain,
    we fall back to accepting that chain with offset=-1 (the 1-based
    default): no chimeric means we can't check identity, but we also
    can't be "wrong" about which chain to use. Such chains are kept
    because their residue labels come from the PDB itself.
    """
    chim_path = os.path.join(chimeric_dir, f'{pdb_id}.chimeric')
    chains_present = df_beads_all['chain_id'].astype(str).unique().tolist()

    if not os.path.exists(chim_path):
        # Unambiguous single-chain case: trust it.
        if len(chains_present) == 1:
            return chains_present[0], -1
        return None

    seq_hh = pd.read_csv(chim_path)['seq'].values[0]

    # Try chain ids implied by the pdb_id first (should be the right
    # one in the overwhelming majority of cases), then any other chain
    # in the bead CSV as a last resort. Best-identity wins on ties.
    candidates = _candidate_chain_letters(pdb_id, default_chain)
    other_chains = [c for c in chains_present if c not in candidates]

    best: Optional[tuple] = None  # (identity, coverage, chain, offset)
    tried = set()
    for chain in candidates + other_chains:
        chain = str(chain)
        if chain in tried:
            continue
        tried.add(chain)
        df_beads = df_beads_all[df_beads_all['chain_id'].astype(str) == chain]
        if df_beads.empty:
            continue
        df_beads = df_beads.sort_values('group_num').reset_index(drop=True)
        gn = df_beads['group_num'].values.astype(int)
        bead_seq = _seq_to_letters(df_beads['group_name'])
        offset, ident, cov = _best_offset_identity(
            seq_hh, gn, bead_seq, min_coverage=min_coverage,
        )
        if offset is None:
            continue
        # Prefer higher identity, break ties by higher coverage.
        if (best is None
                or ident > best[0]
                or (ident == best[0] and cov > best[1])):
            best = (ident, cov, chain, offset)

    if best is None or best[0] < min_identity:
        return None
    return best[2], best[3]


def _process_chain(task: tuple):
    """Worker for parallel extraction. Must be importable at module level
    so ``multiprocessing`` can pickle it.

    Input ``task`` is a self-contained tuple of primitives so each worker
    only needs what's on disk; the caller has already pre-loaded the
    amino-acid vocabulary (a small dict, cheap to pickle per task).

    Returns one of:
      * ``('ok', pdb_id, arrays_dict)`` — ready to write into the h5.
      * ``('drop', pdb_id, reason_key)`` — skip; reason aggregated upstream.
    """
    (pdb_id, bead_dir, chim_dir, default_chain,
     k, dist_cutoff, block_size, index_vocab,
     min_identity, min_coverage) = task

    pdb_core = pdb_id[:4]
    bead_file = os.path.join(bead_dir, f'{pdb_core}_bead.csv')
    if not os.path.exists(bead_file):
        return 'drop', pdb_id, 'no_bead_file'

    df_beads_all = pd.read_csv(bead_file)
    df_beads_all['chain_id'] = df_beads_all['chain_id'].astype(str)

    match = _match_chain_and_offset(
        pdb_id, df_beads_all, chim_dir, default_chain,
        min_identity=min_identity,
        min_coverage=min_coverage,
    )
    if match is None:
        return 'drop', pdb_id, 'no_chain_match'
    chain, _ = match
    df_beads = df_beads_all[df_beads_all['chain_id'] == chain].copy()

    try:
        df_final = extract_blocks_v2(df_beads, k=k)
    except ValueError:
        return 'drop', pdb_id, 'no_blocks'
    if df_final is None or df_final.shape[0] % block_size != 0:
        return 'drop', pdb_id, 'no_blocks'

    seq_idx = (
        df_final['group_name']
        .apply(lambda x: index_vocab.get(x.upper(), -1))
        .values.reshape(-1, block_size)
    )
    if (seq_idx < 0).any():
        return 'drop', pdb_id, 'unknown_residue'

    group_num = df_final['group_num'].values.reshape(-1, block_size)
    coords    = df_final[['local_x', 'local_y', 'local_z']].values.reshape(-1, block_size, 3)
    seg       = df_final['seg'].values.reshape(-1, block_size)
    distance  = df_final['distance'].values.reshape(-1, block_size)
    res_counts = (df_final[['count8a', 'count10a', 'count12a']]
                  .values.reshape(-1, block_size, 3)[:, 0, :])

    start_id = np.zeros_like(seg, dtype=np.int8)
    start_id[:, 1:] = (seg[:, 1:] == seg[:, :-1]).astype(np.int8)

    valid = distance.max(axis=1) < dist_cutoff
    if not np.any(valid):
        return 'drop', pdb_id, 'all_blocks_dropped_by_dist_cutoff'

    gn_block = group_num[valid]
    gn_dtype = 'i2' if (gn_block.size == 0 or gn_block.max() <= 32767) else 'i4'

    arrays = {
        'seq':        (seq_idx[valid].astype(np.int8),   'i1'),
        'group_num':  (gn_block.astype(gn_dtype),        gn_dtype),
        'coords':     (coords[valid].astype(np.float32), 'f4'),
        'seg':        (seg[valid].astype(np.int16),      'i2'),
        'start_id':   (start_id[valid].astype(np.int8),  'i1'),
        'res_counts': (res_counts[valid].astype(np.int16), 'i2'),
    }
    return 'ok', pdb_id, arrays


def build_h5_from_bead_csvs(bead_dir: str,
                            chim_dir: str,
                            out_h5: str,
                            pdb_list_csv: str,
                            amino_acids_csv: Optional[str] = None,
                            k: int = 10,
                            dist_cutoff: float = 20.0,
                            default_chain: str = 'A',
                            compression: Optional[str] = 'gzip',
                            compression_level: int = 4,
                            shuffle: bool = True,
                            num_workers: int = 1,
                            min_identity: float = 0.9,
                            min_coverage: float = 0.5) -> None:
    """Walk ``chim_dir/*.chimeric``, pair each with the matching bead CSV
    under ``bead_dir``, run the v2 extractor, and consolidate results into
    ``out_h5``. Also writes the surviving PDB ids to ``pdb_list_csv``.

    The output is chunked + gzip-compressed by default, which shaves
    roughly 40-55% off the uncompressed footprint that older NNEF h5
    files had (all integer / narrow-range float datasets). ``h5py``
    decompresses transparently on read, so the training pipeline is
    unaffected.
    """
    if amino_acids_csv is None:
        amino_acids_csv = data_path('amino_acids.csv')

    index_vocab = _load_index_vocab(amino_acids_csv)

    pdb_ids = sorted({
        os.path.splitext(f)[0]
        for f in os.listdir(chim_dir)
        if f.endswith('.chimeric')
    })

    used = []
    block_size = 5 + k

    drop_stats = {
        'no_bead_file':   0,
        'no_chain_match': 0,
        'no_blocks':      0,
        'unknown_residue': 0,
        'all_blocks_dropped_by_dist_cutoff': 0,
    }

    tasks = [
        (pdb_id, bead_dir, chim_dir, default_chain,
         k, dist_cutoff, block_size, index_vocab,
         min_identity, min_coverage)
        for pdb_id in pdb_ids
    ]

    def _pdb_key(pdb_id: str) -> str:
        """Normalize key so the v2 h5 plugs into the existing seq_h5 /
        rama_h5, which are keyed as ``{PDB4}_{CHAIN}``. Chimeric files are
        named ``12ASA.chimeric`` (5-char); convert to ``12AS_A``.
        """
        if len(pdb_id) == 5 and '_' not in pdb_id:
            return f'{pdb_id[:4]}_{pdb_id[4]}'
        return pdb_id

    def _write_result(h5f, pdb_id, arrays):
        grp = h5f.create_group(_pdb_key(pdb_id))
        for key, (arr, dtype) in arrays.items():
            _write(grp, key, arr, dtype, compression, compression_level, shuffle)

    nw = max(1, int(num_workers))
    with h5py.File(out_h5, 'w') as h5f:
        if nw == 1:
            # Serial path -- useful for debugging / profiling.
            results_iter = (_process_chain(t) for t in tasks)
        else:
            # Parallel path. 'spawn' rather than 'fork': on macOS forking
            # from a process that has ever touched Accelerate / ObjC
            # frameworks (which h5py + pandas both do) can leave child
            # workers stuck on framework locks, and in practice measured
            # ~10x slower than even serial. Spawn pays a one-time
            # per-worker startup cost (~5-10 s) but stays CPU-bound
            # afterward.
            #
            # Large chunksize (64) amortizes IPC: at chunksize 8 the
            # dispatcher round-trip through the main process was the
            # throughput ceiling.
            ctx = mp.get_context('spawn')
            pool = ctx.Pool(processes=nw)
            # chunksize=8 empirically sustained the best throughput on
            # macOS: larger values (64) left main idle for long stretches
            # before the first chunk flushed, and smaller values paid
            # more IPC overhead per task.
            results_iter = pool.imap_unordered(_process_chain, tasks, chunksize=8)

        # Periodic flush = cheap insurance. A kill between flushes loses
        # at most the last ``flush_every`` chains' worth of writes, but
        # the HDF5 file itself remains readable (without the flush, even
        # the groups already "written" can be unrecoverable if the file
        # header is mid-update at kill time).
        flush_every = 200
        try:
            for i, (status, pdb_id, payload) in enumerate(tqdm(
                results_iter, total=len(tasks), desc='v2 extract',
            )):
                if status == 'ok':
                    _write_result(h5f, pdb_id, payload)
                    used.append(pdb_id)
                else:  # 'drop'
                    drop_stats[payload] = drop_stats.get(payload, 0) + 1
                if (i + 1) % flush_every == 0:
                    h5f.flush()
        finally:
            if nw > 1:
                pool.close()
                pool.join()
            h5f.flush()

    # Also record block counts (as a cheap seq_len proxy) and a uniform
    # default sampling weight so this CSV is directly consumable by the
    # training pipeline's WeightedRandomSampler.
    used_sorted = sorted(_pdb_key(p) for p in used)
    with h5py.File(out_h5, 'r') as hf_read:
        seq_lens = [int(hf_read[p]['coords'].shape[0]) for p in used_sorted]
    pd.DataFrame({
        'pdb':     used_sorted,
        'weight':  [1.0] * len(used_sorted),
        'seq_len': seq_lens,
    }).to_csv(pdb_list_csv, index=False)
    print(
        f'\n[build-h5] kept {len(used)} / {len(pdb_ids)} chains '
        f'({100.0 * len(used) / max(len(pdb_ids), 1):.1f}%)\n'
        f'  drop reasons: {drop_stats}\n'
        f'  settings: k={k} dist_cutoff={dist_cutoff} min_identity={min_identity} '
        f'num_workers={nw} compression={compression}'
    )


# ---------------------------------------------------------------------------
# 5. CLI.
# ---------------------------------------------------------------------------

def _main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            'v2 building-block extractor (thesis: Qiayi Zha 2025 §2.3). '
            'See module docstring for layout details.'
        ),
    )
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_beads = sub.add_parser('extract-beads',
                             help='PDB -> bead CSV with N/CA/C/CB columns.')
    p_beads.add_argument('--pdb', required=True, help='Input PDB file.')
    p_beads.add_argument('--out', required=True, help='Output bead CSV path.')
    p_beads.add_argument('--amino_acids_csv', default=None,
                         help='Override the amino_acids.csv location.')

    p_h5 = sub.add_parser('build-h5',
                          help='bead_csvs/ + chimeric/ -> consolidated v2 HDF5.')
    p_h5.add_argument('--bead_dir', required=True)
    p_h5.add_argument('--chim_dir', required=True)
    p_h5.add_argument('--out_h5', required=True)
    p_h5.add_argument('--pdb_list', required=True,
                      help='CSV path to write the list of successfully '
                           'processed PDB ids.')
    p_h5.add_argument('--amino_acids_csv', default=None)
    p_h5.add_argument('--k', type=int, default=10,
                      help='Number of non-window neighbors (interior blocks).')
    p_h5.add_argument('--dist_cutoff', type=float, default=20.0,
                      help='Drop blocks whose farthest neighbor exceeds this '
                           'distance (A).')
    p_h5.add_argument('--default_chain', type=str, default='A',
                      help="Chain id to select when the PDB id has no '_X' "
                           'suffix.')
    p_h5.add_argument('--compression', choices=('gzip', 'lzf', 'none'),
                      default='gzip',
                      help='Per-chunk compression filter (default: gzip). '
                           'Pass "none" to reproduce the old uncompressed '
                           'layout.')
    p_h5.add_argument('--compression_level', type=int, default=4,
                      help='gzip level 0..9 (default 4).')
    p_h5.add_argument('--no_shuffle', action='store_true',
                      help='Disable the HDF5 byte-shuffle filter '
                           '(shuffle is on by default).')
    p_h5.add_argument('--num_workers', type=int, default=max(1, (os.cpu_count() or 2) - 1),
                      help='Parallel workers for per-chain extraction '
                           '(default: cpu_count - 1). Set 1 to disable.')
    p_h5.add_argument('--min_identity', type=float, default=0.9,
                      help='Minimum residue identity between the chimeric '
                           'sequence and the bead chain to accept the chain '
                           '(default 0.9). Identity is computed over the '
                           'positions where the bead group_num falls inside '
                           'the chimeric range (partial structure coverage '
                           'is allowed). Two unrelated proteins share ~0.05, '
                           'so 0.7-0.95 all give a near-zero false-positive '
                           'rate; lower values keep more samples with '
                           'engineered mutations / modified residues.')
    p_h5.add_argument('--min_coverage', type=float, default=0.5,
                      help='Minimum fraction of bead positions that must fall '
                           'inside the chimeric range (default 0.5). Protects '
                           'against accepting a match computed over only a '
                           'handful of residues; chains where crystal covers '
                           'only part of the canonical chain are still kept '
                           'as long as >= 50%% of bead positions overlap.')

    args = parser.parse_args(argv)

    if args.cmd == 'extract-beads':
        df = extract_beads_v2(args.pdb, out_csv=args.out,
                              amino_acids_csv=args.amino_acids_csv)
        print(f'[extract-beads] wrote {len(df)} residues to {args.out}')
        return 0

    if args.cmd == 'build-h5':
        compression = None if args.compression == 'none' else args.compression
        build_h5_from_bead_csvs(
            bead_dir=args.bead_dir,
            chim_dir=args.chim_dir,
            out_h5=args.out_h5,
            pdb_list_csv=args.pdb_list,
            amino_acids_csv=args.amino_acids_csv,
            k=args.k,
            dist_cutoff=args.dist_cutoff,
            default_chain=args.default_chain,
            compression=compression,
            compression_level=args.compression_level,
            shuffle=not args.no_shuffle,
            num_workers=args.num_workers,
            min_identity=args.min_identity,
            min_coverage=args.min_coverage,
        )
        return 0

    parser.error(f'Unknown command: {args.cmd}')
    return 2


if __name__ == '__main__':
    sys.exit(_main())
