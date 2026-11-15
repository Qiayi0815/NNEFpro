#!/usr/bin/env python3
"""Build a Yang-2022-faithful structure h5 (``hhsuite_CB_yang.h5``).

This is the frame-correct sibling of ``local_extractor_v2.build-h5``. It reuses
v2's chain matching + IO wholesale and swaps ONLY the per-block geometry for
Yang's original construction (verbatim from ``data_hh/local_extractor_hh.py``):

  * local frame  : centred on the central residue's Cβ bead (so the central
                   bead sits at the origin, r == 0), rotated by the neighbour
                   beads c1=Cβ(a-1), c2=Cβ(a+1)  -> Yang's ``_rotation_matrix``.
  * block        : central 5-residue segment (a, a±1, a±2) + k spatial-nearest
                   = 5+k beads; TERMINI ARE NOT CENTRES (interior only), exactly
                   like Yang. v2 differs on all three points (N–CA–C frame, Cβ
                   coords offset from CA, termini promoted to centres).

group_num convention is kept identical to v2 (raw residue numbers), so the
output is a drop-in for the EXISTING ``hhsuite_pdb_seq_v2.h5`` and the existing
``DatasetLocalGenCM`` (its ``group_num-1`` indexing stays correct). The ONLY
thing that changes vs ``hhsuite_CB_v2.h5`` is the coordinate frame — which is
the whole point (it makes ``--legacy_local_frame`` inference match training).

Usage (same flags as local_extractor_v2 build-h5):
    python -m data_prep_scripts.build_yang_h5 \
        --bead_dir  $DATA_DIR/bead_csvs \
        --chim_dir  $DATA_DIR/chimeric \
        --out_h5    $DATA_DIR/hhsuite_CB_yang.h5 \
        --pdb_list  $DATA_DIR/hhsuite_CB_yang_pdb_list.csv \
        --num_workers 16

Validate against Yang's ORIGINAL data (from nnef_data.zip) on overlapping chains:
    python -m data_prep_scripts.build_yang_h5 validate \
        --our  $DATA_DIR/hhsuite_CB_yang.h5 \
        --yang /path/to/hhsuite_CB_cullpdb.h5
"""
import argparse
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

# Reuse v2's matching + IO so we don't fork the alignment logic.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from local_extractor_v2 import (          # noqa: E402
    _load_index_vocab, _match_chain_and_offset, _write,
)

# `_pdb_key` is defined as a closure inside build_h5_from_bead_csvs; replicate it.
def _pdb_key(pdb_id: str) -> str:
    if len(pdb_id) == 5 and '_' not in pdb_id:
        return f'{pdb_id[:4]}_{pdb_id[4]}'
    return pdb_id


# ---------------------------------------------------------------------------
# Yang geometry — verbatim from data_hh/local_extractor_hh.py (extract_one_topk
# + re_order_df_g), restructured to return the same column schema that
# local_extractor_v2._process_chain consumes.
# ---------------------------------------------------------------------------
def _rotation_matrix(c1, c2):
    c1 = np.asarray(c1, float).reshape(3)
    c2 = np.asarray(c2, float).reshape(3)
    z = np.cross(c1, c2)
    x = c1
    y = np.cross(z, x)
    x = x / np.sqrt(np.sum(x ** 2))
    y = y / np.sqrt(np.sum(y ** 2))
    z = z / np.sqrt(np.sum(z ** 2))
    return np.vstack([x, y, z])


def _re_order_block(df_g: pd.DataFrame) -> pd.DataFrame:
    """Yang's re_order_df_g: group chain-contiguous runs into segments, order
    segments by their mean distance, then by group_num within a segment."""
    df_g = df_g.sort_values(by=['segment', 'group_num'])
    group_num = df_g['group_num'].values
    distance = df_g['distance'].values
    n = len(df_g)
    seg = np.ones(n, dtype=int)
    if n >= 6:
        seg[5] = 2
        for i in range(6, n):
            seg[i] = seg[i - 1] if group_num[i] == group_num[i - 1] + 1 else seg[i - 1] + 1
    seg_dist = np.zeros(n)
    for i in range(5, n):
        seg_dist[i] = distance[seg == seg[i]].mean()
    df_g = df_g.copy()
    df_g['seg'] = seg
    df_g['seg_dist'] = seg_dist
    return df_g.sort_values(by=['segment', 'seg_dist', 'group_num'])


def extract_blocks_yang(df_beads: pd.DataFrame, k: int = 10) -> Optional[pd.DataFrame]:
    """Yang neighbour-bead frame, interior-only centres, 5+k blocks.

    Returns a long df (one row per (block, position)) with columns:
    group_name, group_num, local_x, local_y, local_z, seg, distance,
    count8a, count10a, count12a  — matching local_extractor_v2._process_chain.
    """
    group_num = df_beads['group_num'].values
    group_name = df_beads['group_name'].values
    group_coords = df_beads[['xcb', 'ycb', 'zcb']].values
    gset = set(group_num.tolist())

    rows = []
    for gc in group_num:
        if (gc - 1 not in gset) or (gc + 1 not in gset) or (gc - 2 not in gset) or (gc + 2 not in gset):
            continue
        cen_i = (group_num == gc)
        ind = np.isin(group_num, [gc - 2, gc - 1, gc, gc + 1, gc + 2])
        if ind.sum() != 5:
            continue

        coords = group_coords - group_coords[cen_i]      # centre on central Cβ -> central r==0
        c1 = coords[group_num == gc - 1]
        c2 = coords[group_num == gc + 1]
        if np.sum(c1 ** 2) == 0 or np.sum(c2 ** 2) == 0:
            continue
        R = _rotation_matrix(c1, c2)

        coords_seg = (R[None, :, :] @ coords[ind][:, :, None]).squeeze(-1)
        others = ~ind
        gnum_others = group_num[others]
        coords_others = coords[others]
        dist_i = np.sqrt((coords_others ** 2).sum(axis=1))
        topk = np.argsort(dist_i)[:k]
        coords_topk = (R[None, :, :] @ coords_others[topk][:, :, None]).squeeze(-1)

        gnum = np.append(group_num[ind], gnum_others[topk])
        gname = np.append(group_name[ind], group_name[others][topk])
        allc = np.vstack((coords_seg, coords_topk))
        distance = np.sqrt((allc ** 2).sum(axis=1))

        segment_info = np.full(len(gnum), 5, dtype=int)
        segment_info[gnum == gc] = 0
        segment_info[gnum == gc - 1] = 1
        segment_info[gnum == gc + 1] = 2
        segment_info[gnum == gc - 2] = 3
        segment_info[gnum == gc + 2] = 4

        c8 = int(np.sum(dist_i < 8))
        c10 = int(np.sum(dist_i < 10))
        c12 = int(np.sum(dist_i < 12))

        df_g = pd.DataFrame({
            'group_num': gnum, 'group_name': gname,
            'local_x': allc[:, 0], 'local_y': allc[:, 1], 'local_z': allc[:, 2],
            'distance': distance, 'segment': segment_info,
            'count8a': c8, 'count10a': c10, 'count12a': c12,
        })
        rows.append(_re_order_block(df_g))

    if not rows:
        return None
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Worker — mirror of local_extractor_v2._process_chain but calling Yang geometry.
# Module-level for multiprocessing picklability.
# ---------------------------------------------------------------------------
def _process_chain_yang(task: tuple):
    (pdb_id, bead_dir, chim_dir, default_chain,
     k, dist_cutoff, block_size, index_vocab,
     min_identity, min_coverage) = task

    pdb_core = pdb_id[:4]
    bead_file = os.path.join(bead_dir, f'{pdb_core}_bead.csv')
    if not os.path.exists(bead_file):
        return 'drop', pdb_id, 'no_bead_file'

    df_beads_all = pd.read_csv(bead_file)
    df_beads_all['chain_id'] = df_beads_all['chain_id'].astype(str)

    match = _match_chain_and_offset(pdb_id, df_beads_all, chim_dir, default_chain,
                                    min_identity=min_identity, min_coverage=min_coverage)
    if match is None:
        return 'drop', pdb_id, 'no_chain_match'
    chain, idx_offset = match
    df_beads = df_beads_all[df_beads_all['chain_id'] == chain].copy()
    df_beads = df_beads.drop_duplicates('group_num')

    try:
        df_final = extract_blocks_yang(df_beads, k=k)
    except Exception:
        return 'drop', pdb_id, 'no_blocks'
    if df_final is None or df_final.shape[0] % block_size != 0:
        return 'drop', pdb_id, 'no_blocks'

    seq_idx = (df_final['group_name'].apply(lambda x: index_vocab.get(str(x).upper(), -1))
               .values.reshape(-1, block_size))
    if (seq_idx < 0).any():
        return 'drop', pdb_id, 'unknown_residue'

    group_num = df_final['group_num'].values.reshape(-1, block_size)
    # Sequence registration: the matcher found idx_offset such that the chimeric
    # sequence position = PDB group_num + idx_offset, while the dataset reads the
    # sequence at (group_num - 1). group_num stays the PDB numbering (the rama
    # builder and every inference path look residues up by it); the per-chain
    # correction is stored as 'seq_shift' = idx_offset + 1 and applied ONLY to
    # the sequence lookup in dataset/data_chimeric.py. Before 2026-10-19 the
    # offset was discarded -> ~60% of chains (numbering not starting at 1) were
    # trained on misregistered sequences.
    seq_shift = int(idx_offset) + 1
    coords = df_final[['local_x', 'local_y', 'local_z']].values.reshape(-1, block_size, 3)
    seg = df_final['seg'].values.reshape(-1, block_size)
    distance = df_final['distance'].values.reshape(-1, block_size)
    res_counts = df_final[['count8a', 'count10a', 'count12a']].values.reshape(-1, block_size, 3)[:, 0, :]

    start_id = np.zeros_like(seg, dtype=np.int8)
    start_id[:, 1:] = (seg[:, 1:] == seg[:, :-1]).astype(np.int8)

    valid = distance.max(axis=1) < dist_cutoff
    if not np.any(valid):
        return 'drop', pdb_id, 'all_blocks_dropped_by_dist_cutoff'

    gn_block = group_num[valid]
    gn_dtype = 'i2' if (gn_block.size == 0 or gn_block.max() <= 32767) else 'i4'
    arrays = {
        'seq':        (seq_idx[valid].astype(np.int8),    'i1'),
        'group_num':  (gn_block.astype(gn_dtype),         gn_dtype),
        'coords':     (coords[valid].astype(np.float32),  'f4'),
        'seg':        (seg[valid].astype(np.int16),       'i2'),
        'start_id':   (start_id[valid].astype(np.int8),   'i1'),
        'res_counts': (res_counts[valid].astype(np.int16),'i2'),
        'seq_shift':  (np.array([seq_shift], dtype=np.int32), 'i4'),
    }
    return 'ok', pdb_id, arrays


def build(bead_dir, chim_dir, out_h5, pdb_list_csv, amino_acids_csv=None,
          k=10, dist_cutoff=20.0, default_chain='A', num_workers=1,
          min_identity=0.9, min_coverage=0.5):
    import h5py
    from tqdm import tqdm
    if amino_acids_csv is None:
        here = os.path.dirname(os.path.abspath(__file__))
        amino_acids_csv = os.path.join(here, '..', 'data', 'amino_acids.csv')
    index_vocab = _load_index_vocab(amino_acids_csv)
    block_size = 5 + k

    pdb_ids = sorted({os.path.splitext(f)[0] for f in os.listdir(chim_dir) if f.endswith('.chimeric')})
    tasks = [(p, bead_dir, chim_dir, default_chain, k, dist_cutoff, block_size,
              index_vocab, min_identity, min_coverage) for p in pdb_ids]

    used, drop_stats = [], {}
    with h5py.File(out_h5, 'w') as h5f:
        if num_workers <= 1:
            it = (_process_chain_yang(t) for t in tasks)
        else:
            import multiprocessing as mp
            pool = mp.get_context('spawn').Pool(processes=num_workers)
            it = pool.imap_unordered(_process_chain_yang, tasks, chunksize=8)
        try:
            for i, (status, pdb_id, payload) in enumerate(tqdm(it, total=len(tasks), desc='yang extract')):
                if status == 'ok':
                    grp = h5f.create_group(_pdb_key(pdb_id))
                    for key, (arr, dtype) in payload.items():
                        _write(grp, key, arr, dtype, 'gzip', 4, True)
                    used.append(pdb_id)
                else:
                    drop_stats[payload] = drop_stats.get(payload, 0) + 1
                if (i + 1) % 200 == 0:
                    h5f.flush()
        finally:
            if num_workers > 1:
                pool.close(); pool.join()
            h5f.flush()

    used_sorted = sorted(_pdb_key(p) for p in used)
    with h5py.File(out_h5, 'r') as hf:
        seq_lens = [int(hf[p]['coords'].shape[0]) for p in used_sorted]
    pd.DataFrame({'pdb': used_sorted, 'weight': [1.0] * len(used_sorted),
                  'seq_len': seq_lens}).to_csv(pdb_list_csv, index=False)
    print(f'[build_yang_h5] kept {len(used)}/{len(pdb_ids)} chains; drops={drop_stats}')


# ---------------------------------------------------------------------------
# Validation against Yang's ORIGINAL h5 (frame check on overlapping chains).
# ---------------------------------------------------------------------------
def validate(our_h5, yang_h5, n=50):
    import h5py
    our = h5py.File(our_h5, 'r')
    yang = h5py.File(yang_h5, 'r')
    overlap = sorted(set(our.keys()) & set(yang.keys()))
    print(f'[validate] {len(overlap)} overlapping chains; checking up to {n}')
    if not overlap:
        print('  NO overlap (key format differs?)'); return
    central_ok = radii_ok = 0; checked = 0
    for pdb in overlap[:n]:
        oc = our[pdb]['coords'][()]
        yc = yang[pdb]['coords'][()]
        # central bead must sit at the origin in BOTH (the whole point)
        c_our = np.linalg.norm(oc, axis=2).min(axis=1).mean()
        c_yang = np.linalg.norm(yc, axis=2).min(axis=1).mean()
        if c_our < 0.05 and c_yang < 0.05:
            central_ok += 1
        # per-block radius multiset should match Yang for shared central residues
        our_by_r = sorted(np.round(np.sort(np.linalg.norm(oc, axis=2), axis=1).sum(1), 1))
        yang_by_r = sorted(np.round(np.sort(np.linalg.norm(yc, axis=2), axis=1).sum(1), 1))
        common = len(set(our_by_r) & set(yang_by_r))
        if common >= 0.8 * min(len(our_by_r), len(yang_by_r)):
            radii_ok += 1
        checked += 1
    print(f'  central-at-origin match: {central_ok}/{checked}')
    print(f'  per-block radii overlap >=80%: {radii_ok}/{checked}')
    print('  PASS' if central_ok == checked else '  CHECK central-origin failures')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd')

    b = sub.add_parser('build', help='build the Yang-frame h5')
    b.add_argument('--bead_dir', required=True)
    b.add_argument('--chim_dir', required=True)
    b.add_argument('--out_h5', required=True)
    b.add_argument('--pdb_list', required=True)
    b.add_argument('--amino_acids_csv', default=None)
    b.add_argument('--k', type=int, default=10)
    b.add_argument('--dist_cutoff', type=float, default=20.0)
    b.add_argument('--default_chain', default='A')
    b.add_argument('--num_workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    b.add_argument('--min_identity', type=float, default=0.9)
    b.add_argument('--min_coverage', type=float, default=0.5)

    v = sub.add_parser('validate', help='compare to Yang original h5')
    v.add_argument('--our', required=True)
    v.add_argument('--yang', required=True)
    v.add_argument('--n', type=int, default=50)

    # default to build when no subcommand (so plain flags still work)
    args, _ = p.parse_known_args(argv)
    if args.cmd == 'validate':
        validate(args.our, args.yang, args.n)
    else:
        a = b.parse_args([x for x in (argv or sys.argv[1:]) if x != 'build'])
        build(a.bead_dir, a.chim_dir, a.out_h5, a.pdb_list, a.amino_acids_csv,
              a.k, a.dist_cutoff, a.default_chain, a.num_workers,
              a.min_identity, a.min_coverage)


if __name__ == '__main__':
    main()
