"""Build ``hhsuite_rama_v2.h5`` aligned 1:1 with ``hhsuite_CB_v2.h5``.

For every PDB key and every building block row, we copy φ/ψ (radians) from the
same residue ``group_num`` entries used in the CB v2 file. Backbone dihedrals
are computed from N/CA/C columns in the v2 bead CSV (same source as ``extract-beads``).

Schema (matches ``DatasetLocalGenCM`` / ``data_chimeric.py``)::

    <PDB4>_<chain>/rama float32  (num_blocks, 15, 2)   # phi, psi; NaN where undefined

Usage (from repo root, with ``nnef/`` package on ``PYTHONPATH``)::

    python -m nnef.data_prep_scripts.build_rama_h5_v2 build \\
        --cb_h5 nnef/data/hhsuite_CB_v2.h5 \\
        --bead_dir data_hh/bead_csvs \\
        --out_h5 nnef/data/hhsuite_rama_v2.h5 \\
        --num_workers 8

    python -m nnef.data_prep_scripts.build_rama_h5_v2 verify \\
        --cb_h5 nnef/data/hhsuite_CB_v2.h5 \\
        --rama_h5 nnef/data/hhsuite_rama_v2.h5
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from typing import Dict, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_DIR = os.path.dirname(_THIS_DIR)
if _PACKAGE_DIR not in sys.path:
    sys.path.insert(0, _PACKAGE_DIR)

from utils import _load_dihedral_from_beads  # noqa: E402


def _pdb_key_to_parts(pdb_key: str) -> Tuple[str, str]:
    """``12AS_A`` -> (``12AS``, ``A``); ``12ASA`` -> (``12AS``, ``A``)."""
    if '_' in pdb_key:
        a, b = pdb_key.split('_', 1)
        return a[:4], b
    if len(pdb_key) >= 5:
        return pdb_key[:4], pdb_key[4]
    raise ValueError(f'cannot parse pdb key: {pdb_key!r}')


def _phi_psi_lut_from_bead_df(df_chain: pd.DataFrame) -> Optional[Dict[int, np.ndarray]]:
    """Map ``group_num`` -> float32 (2,) phi/psi radians. ``None`` if backbone cols missing."""
    df = df_chain.sort_values('group_num').reset_index(drop=True)
    arr = _load_dihedral_from_beads(df)
    if arr is None:
        return None
    gnums = df['group_num'].values.astype(int)
    lut: Dict[int, np.ndarray] = {}
    for i, g in enumerate(gnums):
        lut[int(g)] = arr[i].astype(np.float32)
    return lut


def _process_key(task: Tuple[str, str, str]) -> Tuple[str, str, Optional[np.ndarray]]:
    """Return (status, pdb_key, rama_or_none). status in ok, no_bead, no_backbone, bad_shape."""
    pdb_key, cb_h5_path, bead_dir = task
    pdb4, chain = _pdb_key_to_parts(pdb_key)
    bead_path = os.path.join(bead_dir, f'{pdb4}_bead.csv')
    if not os.path.isfile(bead_path):
        return 'no_bead', pdb_key, None

    with h5py.File(cb_h5_path, 'r', libver='latest', swmr=True) as h5:
        if pdb_key not in h5:
            return 'bad_shape', pdb_key, None
        gn = h5[pdb_key]['group_num'][()]
        if gn.ndim != 2 or gn.shape[1] != 15:
            return 'bad_shape', pdb_key, None

    df_all = pd.read_csv(bead_path)
    if 'chain_id' not in df_all.columns:
        return 'no_bead', pdb_key, None
    df_all['chain_id'] = df_all['chain_id'].astype(str)
    df_ch = df_all[df_all['chain_id'] == str(chain)]
    if df_ch.empty:
        return 'no_bead', pdb_key, None

    lut = _phi_psi_lut_from_bead_df(df_ch)
    if lut is None:
        return 'no_backbone', pdb_key, None

    n_b, w = gn.shape
    rama = np.full((n_b, w, 2), np.nan, dtype=np.float32)
    for bi in range(n_b):
        for j in range(w):
            g = int(gn[bi, j])
            if g in lut:
                rama[bi, j] = lut[g]
    return 'ok', pdb_key, rama


def _write_rama_dataset(
        grp: h5py.Group,
        rama: np.ndarray,
        compression: Optional[str],
        level: int,
        shuffle: bool,
) -> None:
    if rama.shape[0] == 0 or compression is None:
        grp.create_dataset('rama', data=rama, dtype='f4')
        return
    rows = max(1, min(rama.shape[0],256))
    chunks = (rows, rama.shape[1], rama.shape[2])
    grp.create_dataset(
        'rama', data=rama, dtype='f4',
        chunks=chunks,
        compression=compression,
        compression_opts=level if compression == 'gzip' else None,
        shuffle=shuffle,
    )


def cmd_build(args: argparse.Namespace) -> int:
    cb_h5 = os.path.expanduser(args.cb_h5)
    out_h5 = os.path.expanduser(args.out_h5)
    bead_dir = os.path.expanduser(args.bead_dir)
    if not os.path.isfile(cb_h5):
        print(f'[build] missing CB h5: {cb_h5}')
        return 1
    if not os.path.isdir(bead_dir):
        print(f'[build] bead_dir not found: {bead_dir}')
        return 1

    with h5py.File(cb_h5, 'r') as h5r:
        keys = sorted(h5r.keys())

    if args.limit is not None:
        keys = keys[: int(args.limit)]

    tasks = [(k, cb_h5, bead_dir) for k in keys]
    nw = max(1, int(args.num_workers))
    compression = None if args.compression == 'none' else args.compression

    stats = {s: 0 for s in ('ok', 'no_bead', 'no_backbone', 'bad_shape')}

    if os.path.exists(out_h5):
        os.remove(out_h5)

    if nw == 1:
        it = (_process_key(t) for t in tasks)
    else:
        ctx = mp.get_context('spawn')
        pool = ctx.Pool(processes=nw)
        it = pool.imap_unordered(_process_key, tasks, chunksize=8)

    try:
        with h5py.File(out_h5, 'w') as h5w:
            for status, pdb_key, rama in tqdm(it, total=len(tasks), desc='rama v2'):
                stats[status] = stats.get(status, 0) + 1
                if status != 'ok' or rama is None:
                    continue
                grp = h5w.create_group(pdb_key)
                _write_rama_dataset(
                    grp, rama,
                    compression=compression,
                    level=int(args.compression_level),
                    shuffle=not args.no_shuffle,
                )
                if stats['ok'] % 200 == 0:
                    h5w.flush()
            h5w.flush()
    finally:
        if nw > 1:
            pool.close()
            pool.join()

    with h5py.File(out_h5, 'r') as hf:
        n_written = len(hf.keys())
    print(
        f'\n[build] wrote {n_written} PDB groups to {out_h5}\n'
        f'  task status counts: {stats}\n'
        f'  (ok = chains processed successfully; others skipped without writing)',
    )
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    cb_h5 = os.path.expanduser(args.cb_h5)
    rama_h5 = os.path.expanduser(args.rama_h5)
    if not os.path.isfile(cb_h5) or not os.path.isfile(rama_h5):
        print('[verify] missing input h5')
        return 1

    failures = []
    nan_fracs = []
    deep = int(getattr(args, 'deep', 0) or 0)
    bead_dir = getattr(args, 'bead_dir', None)
    shared_keys: list = []

    with h5py.File(cb_h5, 'r') as cb, h5py.File(rama_h5, 'r') as rm:
        cb_keys = set(cb.keys())
        rm_keys = set(rm.keys())
        missing_rm = sorted(cb_keys - rm_keys)
        extra_rm = sorted(rm_keys - cb_keys)
        if missing_rm:
            failures.append(f'missing {len(missing_rm)} keys in rama h5 (show up to 5): {missing_rm[:5]}')
        if extra_rm:
            failures.append(f'extra {len(extra_rm)} keys in rama h5 (show up to 5): {extra_rm[:5]}')

        shared_keys = sorted(cb_keys & rm_keys)
        for k in shared_keys:
            gn = cb[k]['group_num'][()]
            if 'rama' not in rm[k]:
                failures.append(f'{k}: no rama dataset')
                continue
            ra = rm[k]['rama'][()]
            if ra.shape != (gn.shape[0], gn.shape[1], 2):
                failures.append(f'{k}: rama shape {ra.shape} vs group_num {gn.shape}')
                continue
            finite = np.isfinite(ra).all(axis=-1)
            nan_fracs.append(1.0 - float(finite.mean()))

    if bead_dir and deep > 0:
        bd = os.path.expanduser(bead_dir)
        with h5py.File(rama_h5, 'r') as rm:
            for k in shared_keys[:deep]:
                if 'rama' not in rm[k]:
                    continue
                st, _, rebuilt = _process_key((k, cb_h5, bd))
                on_disk = rm[k]['rama'][()]
                if st != 'ok' or rebuilt is None:
                    failures.append(f'deep {k}: rebuild status={st}')
                    continue
                if not np.allclose(rebuilt, on_disk, equal_nan=True, rtol=1e-5, atol=1e-4):
                    d = float(np.nanmax(np.abs(rebuilt - on_disk)))
                    failures.append(f'deep {k}: max abs diff vs bead recompute {d}')

    if nan_fracs:
        print(
            f'[verify] NaN fraction per residue (mean over PDBs): '
            f'{float(np.mean(nan_fracs)):.4f}  (median {float(np.median(nan_fracs)):.4f})',
        )

    if failures:
        print('[verify] FAILED:')
        for f in failures[:30]:
            print(' ', f)
        if len(failures) > 30:
            print(f'  ... and {len(failures) - 30} more')
        return 1

    print('[verify] OK: keys and rama shapes match CB v2 for all shared entries.')
    return 0


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)

    b = sub.add_parser('build', help='Create rama h5 from CB v2 + bead CSVs')
    b.add_argument('--cb_h5', required=True, help='Reference hhsuite_CB_v2.h5')
    b.add_argument('--bead_dir', required=True, help='Directory of *_bead.csv (v2 columns)')
    b.add_argument('--out_h5', required=True, help='Output path, e.g. hhsuite_rama_v2.h5')
    b.add_argument('--num_workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    b.add_argument('--compression', choices=('gzip', 'lzf', 'none'), default='gzip')
    b.add_argument('--compression_level', type=int, default=4)
    b.add_argument('--no_shuffle', action='store_true')
    b.add_argument('--limit', type=int, default=None, help='Process only first N PDB keys (debug)')
    b.set_defaults(func=cmd_build)

    v = sub.add_parser('verify', help='Check rama h5 vs CB v2')
    v.add_argument('--cb_h5', required=True)
    v.add_argument('--rama_h5', required=True)
    v.add_argument('--bead_dir', default=None,
                   help='If set with --deep, recompute φ/ψ from bead CSVs for a spot check.')
    v.add_argument('--deep', type=int, default=0,
                   help='Number of PDBs to spot-check against on-the-fly recompute (needs --bead_dir).')
    v.set_defaults(func=cmd_verify)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == '__main__':
    sys.exit(main())
