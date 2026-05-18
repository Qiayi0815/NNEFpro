"""Rename data_hh/hhsuite_CB_newlogic.h5 keys (12ASA -> 12AS_A) + filter.

Motivation: the "newlogic" h5 was built with `qlocal_ex_modify_add4_hh.py` which
uses our modified building-block construction (5 sequence neighbors +
10 topological nearest = 15 per block). This is the correct preprocessing for
our `--legacy_local_frame` + modified-block training code -- matches what
`yang_legacy`'s architecture expects.

BUT: newlogic uses concatenated PDB+chain keys (12ASA), while all the existing
sequence h5s (hhsuite_pdb_seq_v2.h5) use underscore keys (12AS_A). The
DatasetLocalGenCM loader uses the same key string to index into both h5s, so
key conventions must match. This script produces a key-renamed copy of newlogic
and a pdb_list CSV compatible with the existing seq_v2 h5.

Input:
  data_hh/hhsuite_CB_newlogic.h5                  (8054 proteins, 12ASA keys)
  nnef/data/hhsuite_pdb_seq_v2.h5                 (12394 proteins, 12AS_A keys)

Output:
  nnef/data/hhsuite_CB_newlogic_v2keys.h5         (underscore keys, intersect)
  nnef/data/hhsuite_CB_newlogic_v2keys_pdb_list.csv (pdb, weight=1.0, seq_len)

Usage:
  python nnef/data_prep_scripts/rename_newlogic_keys.py
"""
from __future__ import annotations

import os

import h5py
import pandas as pd
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))

SRC_PDB_H5 = os.path.join(REPO, 'data_hh/hhsuite_CB_newlogic.h5')
SRC_SEQ_H5 = os.path.join(REPO, 'nnef/data/hhsuite_pdb_seq_v2.h5')
OUT_PDB_H5 = os.path.join(REPO, 'nnef/data/hhsuite_CB_newlogic_v2keys.h5')
OUT_CSV = os.path.join(REPO, 'nnef/data/hhsuite_CB_newlogic_v2keys_pdb_list.csv')


def to_underscore_key(k: str, seq_keys: set[str]) -> str | None:
    # Try 1-char chain first (the common case), then 2-char (rare).
    for n_chain in (1, 2):
        if len(k) <= n_chain:
            continue
        cand = f'{k[:-n_chain]}_{k[-n_chain:]}'
        if cand in seq_keys:
            return cand
    return None


def main() -> None:
    for p in (SRC_PDB_H5, SRC_SEQ_H5):
        if not os.path.isfile(p):
            raise SystemExit(f'[rename] missing input: {p}')

    with h5py.File(SRC_SEQ_H5, 'r') as f:
        seq_keys = set(f.keys())
    print(f'[rename] seq_v2 keys: {len(seq_keys)}')

    with h5py.File(SRC_PDB_H5, 'r') as f_in:
        nl_keys = list(f_in.keys())
        print(f'[rename] newlogic keys: {len(nl_keys)}')

        mapping: list[tuple[str, str]] = []
        for k in nl_keys:
            u = to_underscore_key(k, seq_keys)
            if u is not None:
                mapping.append((k, u))
        hit = len(mapping)
        miss = len(nl_keys) - hit
        print(f'[rename] matched: {hit}   dropped (not in seq_v2): {miss}')

        # Write renamed h5.
        rows = []
        with h5py.File(OUT_PDB_H5, 'w') as f_out:
            for src, dst in tqdm(mapping, desc='copying groups'):
                g_src = f_in[src]
                g_dst = f_out.create_group(dst)
                for name in g_src.keys():
                    arr = g_src[name][()]
                    g_dst.create_dataset(name, data=arr, dtype=arr.dtype)
                n_blocks = int(g_src['coords'].shape[0])
                rows.append({'pdb': dst, 'weight': 1.0, 'seq_len': n_blocks})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    size_mb = os.path.getsize(OUT_PDB_H5) / 1e6
    print(f'[rename] wrote {OUT_PDB_H5} ({size_mb:.1f} MB, {len(df)} proteins)')
    print(f'[rename] wrote {OUT_CSV} ({len(df)} rows, columns=pdb,weight,seq_len)')
    print(f'[rename] total blocks across all proteins: {df.seq_len.sum():,}')


if __name__ == '__main__':
    main()
