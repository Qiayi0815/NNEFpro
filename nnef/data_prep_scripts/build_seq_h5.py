"""Build ``hhsuite_pdb_seq_v2.h5`` from per-chain ``.chimeric`` MSA files.

The training pipeline (``DatasetLocalGenCM``) expects, for every PDB id in the
``data_flag`` CSV, a key in ``seq_h5`` with shape ``(n_msa_rows, L_chain)`` of
int8 amino acid indices (0..19, matching ``amino_acids.csv`` with ``idx - 1``).

Historically ``hhsuite_pdb_seq_cullpdb.h5`` was built by an external MSA
pipeline and only covered ~7.8k of our 12.4k v2 chains. This rebuild packs
all chimeric MSAs we have on disk so the v2 structure set is fully
consumable. Output keys use the ``{PDB4}_{CHAIN}`` naming convention to match
the v2 structure h5.

Usage
-----
::

    python -m nnef.data_prep_scripts.build_seq_h5 \
        --chim_dir     data_hh/chimeric \
        --out_h5       nnef/data/hhsuite_pdb_seq_v2.h5 \
        --pdb_list     nnef/data/hhsuite_CB_v2_pdb_list.csv \
        --num_workers  8

The output is compressed per-chain (``gzip`` + byte-shuffle) for disk
efficiency. Rows containing non-canonical letters (``X``, ``B``, ``-``, ...)
are dropped; the wild-type row (index 0) is always kept if it is canonical so
length invariants hold. If the WT has a non-canonical letter, the chain is
skipped entirely.
"""
from __future__ import annotations

import argparse
import os
from multiprocessing import Pool, get_context
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd

try:
    from nnef.paths import data_path
except ImportError:  # allow "python file.py" from the repo root
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from nnef.paths import data_path

# ---------------------------------------------------------------------------
# Amino-acid vocabulary (1-based idx in CSV, 0-based in the h5).
# ---------------------------------------------------------------------------
_AA_DF = pd.read_csv(data_path('amino_acids.csv'))
_AA2IDX: Dict[str, int] = {r.AA: int(r.idx) - 1 for _, r in _AA_DF.iterrows()}
_VALID = np.zeros(256, dtype=bool)
_ENC   = np.zeros(256, dtype=np.int16)
for aa, idx in _AA2IDX.items():
    _VALID[ord(aa)] = True
    _ENC[ord(aa)]   = idx


def _encode_seq(s: str) -> Optional[np.ndarray]:
    """Encode a sequence string into int8 indices 0..19, or ``None`` if any
    letter is outside the 20 canonical AAs."""
    arr = np.frombuffer(s.encode('ascii'), dtype=np.uint8)
    if not _VALID[arr].all():
        return None
    return _ENC[arr].astype(np.int8)


# ---------------------------------------------------------------------------
# Worker: parse a single .chimeric -> (pdb_key, (n_msa, L) int8 ndarray).
# ---------------------------------------------------------------------------
def _pdb_key(pdb5: str) -> str:
    return f'{pdb5[:4]}_{pdb5[4]}' if len(pdb5) == 5 and '_' not in pdb5 else pdb5


def _process_one(args: Tuple[str, str]) -> Tuple[str, Optional[np.ndarray], str]:
    pdb5, chim_dir = args
    path = os.path.join(chim_dir, f'{pdb5}.chimeric')
    try:
        df = pd.read_csv(path, usecols=['seq'])
    except FileNotFoundError:
        return pdb5, None, 'no_chim_file'
    except Exception as e:  # malformed CSV
        return pdb5, None, f'csv_error:{type(e).__name__}'

    if df.empty:
        return pdb5, None, 'empty'

    seqs: List[np.ndarray] = []
    wt = _encode_seq(df['seq'].iloc[0])
    if wt is None:
        return pdb5, None, 'wt_non_canonical'
    seqs.append(wt)
    L = wt.shape[0]

    for s in df['seq'].iloc[1:].values:
        if len(s) != L:
            continue  # length mismatch -> drop
        enc = _encode_seq(s)
        if enc is not None:
            seqs.append(enc)

    arr = np.stack(seqs, axis=0)
    return pdb5, arr, 'ok'


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------
def build(chim_dir: str,
          out_h5: str,
          pdb_list_csv: Optional[str],
          num_workers: int,
          compression: Optional[str] = 'gzip',
          compression_level: int = 4,
          shuffle: bool = True,
          ) -> None:
    # Discover chains: either from the pdb_list CSV, or from the chim_dir.
    if pdb_list_csv is not None and os.path.isfile(pdb_list_csv):
        df = pd.read_csv(pdb_list_csv)
        # Accept either ``{PDB4}_{C}`` or ``{PDB4}{C}`` ids.
        pdb5_list = [p.replace('_', '') for p in df['pdb'].astype(str)]
    else:
        pdb5_list = sorted(
            f[:-len('.chimeric')]
            for f in os.listdir(chim_dir)
            if f.endswith('.chimeric')
        )

    tasks = [(p, chim_dir) for p in pdb5_list]
    print(f'[build_seq_h5] {len(tasks)} chains, workers={num_workers}')

    drop_stats = {'no_chim_file': 0, 'empty': 0, 'wt_non_canonical': 0,
                  'csv_error': 0, 'ok': 0}
    n_written = 0

    def _write(h5f: h5py.File, key: str, arr: np.ndarray) -> None:
        kwargs = {}
        if compression:
            kwargs['compression'] = compression
            if compression == 'gzip':
                kwargs['compression_opts'] = int(compression_level)
            kwargs['shuffle'] = bool(shuffle)
            # One row per chunk = cheap random-row access for training.
            kwargs['chunks'] = (1, arr.shape[1])
        h5f.create_dataset(key, data=arr, dtype='i1', **kwargs)

    os.makedirs(os.path.dirname(os.path.abspath(out_h5)), exist_ok=True)
    with h5py.File(out_h5, 'w') as h5f:
        if num_workers <= 1:
            iterator = (_process_one(t) for t in tasks)
        else:
            ctx = get_context('spawn')  # macOS-safe; linux is fine too
            pool = ctx.Pool(num_workers)
            iterator = pool.imap_unordered(_process_one, tasks, chunksize=16)

        for i, (pdb5, arr, status) in enumerate(iterator, start=1):
            bucket = status.split(':', 1)[0]
            drop_stats[bucket] = drop_stats.get(bucket, 0) + 1
            if arr is not None:
                _write(h5f, _pdb_key(pdb5), arr)
                n_written += 1
            if i % 500 == 0:
                h5f.flush()
                print(f'  {i}/{len(tasks)}  written={n_written}  '
                      f'drop={ {k: v for k, v in drop_stats.items() if k != "ok"} }')

        if num_workers > 1:
            pool.close()
            pool.join()
        h5f.flush()

    print(f'\n[build_seq_h5] done: wrote {n_written} chains to {out_h5}')
    print(f'  drop_stats: {drop_stats}')


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--chim_dir',     required=True)
    p.add_argument('--out_h5',       required=True)
    p.add_argument('--pdb_list',     default=None,
                   help='Optional CSV with a `pdb` column to restrict output.')
    p.add_argument('--num_workers',  type=int, default=8)
    p.add_argument('--compression',       default='gzip',
                   choices=['gzip', 'lzf', 'none'])
    p.add_argument('--compression_level', type=int, default=4)
    p.add_argument('--shuffle',           type=lambda s: s.lower() != 'false',
                   default=True)
    args = p.parse_args()
    comp = None if args.compression == 'none' else args.compression
    build(args.chim_dir, args.out_h5, args.pdb_list,
          args.num_workers, comp, args.compression_level, args.shuffle)


if __name__ == '__main__':
    main()
