"""Repack an existing NNEF block-HDF5 with chunking + compression.

Why this script exists
----------------------

The older cullpdb ``.h5`` files under ``nnef/data/`` were written with the
HDF5 defaults (``chunks=None``, ``compression=None``): contiguous storage,
no filters. For the integer-heavy datasets used here
(``seq``/``start_id``/``seg``/``group_num``/``res_counts``), that is very
wasteful -- typical gzip:4 + byteshuffle recovers ~40-55% without changing
the training pipeline, because h5py decompresses transparently on read.

Usage
-----

    python -m nnef.scripts.compress_h5 \
        --in  nnef/data/hhsuite_CB_cullpdb.h5 \
        --out nnef/data/hhsuite_CB_cullpdb.gz.h5

    # drop-in replace after you've verified the output
    mv nnef/data/hhsuite_CB_cullpdb.gz.h5 nnef/data/hhsuite_CB_cullpdb.h5

The schema is preserved exactly: same group names, same dataset keys, same
shapes. Two optional, off-by-default tweaks help more:

  * ``--downcast``    Cast ``group_num`` to ``int16`` when its max fits.
                      Real protein chains are way below 32k residues, so
                      this is safe for every dataset seen in this repo.
  * ``--shuffle``     HDF5 byte-shuffle filter; often 10-20% extra win on
                      integer datasets. On by default.

Safety
------

The script never touches the input file. It writes ``--out``, then the
caller can atomically ``mv`` when satisfied. A round-trip check (read
every dataset back from the output and spot-compare one block per chain)
can be enabled with ``--verify``.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

import h5py
import numpy as np
from tqdm import tqdm

# Datasets for which a smaller integer dtype is always safe: the max value
# observed in the cullpdb corpora stays well below the int16 range.
_DOWNCAST_CANDIDATES = {
    'group_num': np.int16,
}


def _choose_chunks(shape: tuple, dtype: np.dtype,
                   target_bytes: int = 256 * 1024) -> Optional[tuple]:
    """Pick a chunk shape close to ``target_bytes`` (default 256 KiB).

    For (N, 15) / (N, 15, 3) / (N, 3) datasets we chunk along the first
    (num_blocks) axis only. h5py's ``chunks=True`` auto-chunker is also
    fine, but doing it explicitly keeps chunks large enough that the
    per-chunk filter overhead stays low.
    """
    if len(shape) == 0 or shape[0] == 0:
        return None
    per_row = int(np.prod(shape[1:]) if len(shape) > 1 else 1) * dtype.itemsize
    if per_row == 0:
        return None
    rows = max(1, target_bytes // max(per_row, 1))
    rows = min(rows, shape[0])
    return (rows,) + tuple(shape[1:])


def _copy_dataset(src: h5py.Dataset,
                  dst_parent: h5py.Group,
                  name: str,
                  compression: str,
                  level: int,
                  shuffle: bool,
                  downcast: bool) -> tuple:
    """Copy one dataset with chunking + compression.

    Returns ``(raw_bytes_in, raw_bytes_out)`` where ``raw_bytes_out`` is
    the *allocated* storage for the written dataset (includes filter
    savings).
    """
    data = src[()]
    src_bytes = src.size * src.dtype.itemsize

    out_dtype = data.dtype
    if downcast and name in _DOWNCAST_CANDIDATES:
        target = _DOWNCAST_CANDIDATES[name]
        lo = np.iinfo(target).min
        hi = np.iinfo(target).max
        if data.size == 0 or (int(data.min()) >= lo and int(data.max()) <= hi):
            out_dtype = target
            data = data.astype(target)

    chunks = _choose_chunks(data.shape, np.dtype(out_dtype))

    # Compression only works on chunked datasets. If we decide not to chunk
    # (tiny dataset), fall back to contiguous + no filters.
    if chunks is None:
        ds = dst_parent.create_dataset(name, data=data, dtype=out_dtype)
    else:
        ds = dst_parent.create_dataset(
            name,
            data=data,
            dtype=out_dtype,
            chunks=chunks,
            compression=compression if compression != 'none' else None,
            compression_opts=level if compression == 'gzip' else None,
            shuffle=shuffle,
        )

    return src_bytes, ds.id.get_storage_size()


def repack(in_path: str,
           out_path: str,
           compression: str = 'gzip',
           level: int = 4,
           shuffle: bool = True,
           downcast: bool = True,
           verify: bool = False) -> None:
    if os.path.abspath(in_path) == os.path.abspath(out_path):
        raise ValueError('--in and --out must be different files')
    if os.path.exists(out_path):
        raise FileExistsError(
            f'{out_path} already exists; refusing to overwrite. '
            'Remove it manually or pick another --out.'
        )

    t0 = time.time()
    raw_in = 0
    raw_out = 0
    group_count = 0
    dataset_count = 0

    with h5py.File(in_path, 'r') as fin, h5py.File(out_path, 'w') as fout:
        # Mirror top-level attrs, if any (there usually are none here).
        for k, v in fin.attrs.items():
            fout.attrs[k] = v

        pdb_ids = list(fin.keys())
        for pdb in tqdm(pdb_ids, desc='repack'):
            src = fin[pdb]
            if isinstance(src, h5py.Dataset):
                # Flat layout: dataset directly under root.
                b_in, b_out = _copy_dataset(
                    src, fout, pdb, compression, level, shuffle, downcast,
                )
                raw_in += b_in
                raw_out += b_out
                dataset_count += 1
                continue

            grp = fout.create_group(pdb)
            for attr_k, attr_v in src.attrs.items():
                grp.attrs[attr_k] = attr_v
            for key, ds in src.items():
                if not isinstance(ds, h5py.Dataset):
                    # Nested groups: the NNEF h5 files are flat, so we
                    # don't need to recurse. If a nested group ever shows
                    # up, warn loudly rather than silently dropping it.
                    print(f'[warn] nested group skipped: {pdb}/{key}',
                          file=sys.stderr)
                    continue
                b_in, b_out = _copy_dataset(
                    ds, grp, key, compression, level, shuffle, downcast,
                )
                raw_in += b_in
                raw_out += b_out
                dataset_count += 1
            group_count += 1

    dt = time.time() - t0
    in_size = os.path.getsize(in_path)
    out_size = os.path.getsize(out_path)
    print(
        f'\n[repack] {in_path} -> {out_path}\n'
        f'  groups: {group_count}, datasets: {dataset_count}\n'
        f'  raw:       {raw_in  / 1e6:8.1f} MB (sum of itemsize*count)\n'
        f'  stored:    {raw_out / 1e6:8.1f} MB (allocated dataset bytes)\n'
        f'  file size: {in_size / 1e6:8.1f} MB  ->  {out_size / 1e6:8.1f} MB '
        f'({100.0 * out_size / max(in_size, 1):.1f}%)\n'
        f'  elapsed: {dt:.1f}s'
    )

    if verify:
        _verify(in_path, out_path, downcast=downcast)


def _verify(in_path: str, out_path: str, downcast: bool) -> None:
    """Read both files back, spot-check one block per chain."""
    with h5py.File(in_path, 'r') as fin, h5py.File(out_path, 'r') as fout:
        keys_in = set(fin.keys())
        keys_out = set(fout.keys())
        if keys_in != keys_out:
            missing = keys_in - keys_out
            extra = keys_out - keys_in
            raise AssertionError(
                f'chain set differs: missing={list(missing)[:5]}, '
                f'extra={list(extra)[:5]}'
            )

        for pdb in tqdm(sorted(keys_in), desc='verify'):
            src = fin[pdb]
            dst = fout[pdb]
            if isinstance(src, h5py.Dataset):
                a = src[()]
                b = dst[()]
                _assert_eq(pdb, 'root', a, b, downcast=False)
                continue
            for key, ds in src.items():
                a = ds[()]
                b = dst[key][()]
                should_downcast = downcast and key in _DOWNCAST_CANDIDATES
                _assert_eq(pdb, key, a, b, downcast=should_downcast)
    print('[verify] all datasets round-trip bit-identical')


def _assert_eq(pdb: str, key: str, a: np.ndarray, b: np.ndarray,
               downcast: bool) -> None:
    if a.shape != b.shape:
        raise AssertionError(
            f'shape mismatch at {pdb}/{key}: {a.shape} vs {b.shape}'
        )
    # If we downcasted, numeric values must still agree after an int cast.
    if downcast:
        if not np.array_equal(a.astype(b.dtype), b):
            raise AssertionError(f'value mismatch at {pdb}/{key} after downcast')
    else:
        if not np.array_equal(a, b):
            raise AssertionError(f'value mismatch at {pdb}/{key}')


def _main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(
        description='Repack an NNEF block HDF5 with chunking + compression.'
    )
    p.add_argument('--in', dest='in_path', required=True,
                   help='Source HDF5 (read-only).')
    p.add_argument('--out', dest='out_path', required=True,
                   help='Destination HDF5 (must not exist).')
    p.add_argument('--compression', choices=('gzip', 'lzf', 'none'),
                   default='gzip',
                   help='Per-chunk compression filter (default: gzip).')
    p.add_argument('--level', type=int, default=4,
                   help='gzip level 0..9 (default 4; higher = smaller '
                        'but slower to write, no read penalty).')
    p.add_argument('--no_shuffle', action='store_true',
                   help='Disable the HDF5 byte-shuffle filter. Default '
                        'is shuffle on; it usually gives +10..20%% on '
                        'top of gzip for the integer datasets.')
    p.add_argument('--no_downcast', action='store_true',
                   help='Do not downcast group_num to int16. Default is '
                        'to downcast when safe (all protein chains fit).')
    p.add_argument('--verify', action='store_true',
                   help='After repacking, re-open both files and assert '
                        'every dataset round-trips bit-identical '
                        '(accounting for downcasting).')

    args = p.parse_args(argv)

    repack(
        in_path=args.in_path,
        out_path=args.out_path,
        compression=args.compression,
        level=args.level,
        shuffle=not args.no_shuffle,
        downcast=not args.no_downcast,
        verify=args.verify,
    )
    return 0


if __name__ == '__main__':
    sys.exit(_main())
