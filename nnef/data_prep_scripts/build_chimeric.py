#!/usr/bin/env python3
"""Stage 3: turn HHsuite a3m MSAs into chimeric-sequence CSVs.

FASRC/parallel port of data_hh/qlocal_align_hh.py, with identical semantics:
for each aligned a3m, remove insertions (lowercase), take the first sequence as
the reference (the native PDB sequence), and for every homolog replace its gaps
('-') with the reference residue -> a "chimeric" sequence of the SAME length as
the reference that can be mapped onto the structure.

Output CSV columns match the existing corpus: protein_name, ins_frac, del_frac,
hamming_dist, seq. Row 0 is the reference. Downstream (build_seq_h5,
build_yang_h5._match_chain_and_offset) reads the ``seq`` column, row 0 = ref.

Reads  $A3M_DIR/<PDBCHAIN>.a3m   ->   writes  $OUT_DIR/<PDBCHAIN>.chimeric
Sharded by file index for parallel array jobs; resumable (skips existing).
"""
import argparse
import os

import numpy as np
import pandas as pd

_AA = '-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_AA2ID = {c: i for i, c in enumerate(_AA)}
_ID2AA = {i: c for i, c in enumerate(_AA)}


def a3m_to_chimeric(a3m_path):
    """Return a DataFrame (or None if the a3m is unusable)."""
    names, seqs, ins_num = [], [], []
    with open(a3m_path) as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            if line.startswith('>'):
                names.append(line.split()[0][1:])
            else:
                s = np.array([_AA2ID.get(c, 0) for c in line])
                seqs.append(s[s <= 26])            # drop lowercase = insertions
                ins_num.append(int((s > 26).sum()))
    if len(seqs) < 1:
        return None
    try:
        arr = np.vstack(seqs)                       # all match-state rows share length L
    except ValueError:
        return None                                 # ragged (malformed a3m) -> skip
    L = arr.shape[1]
    if L == 0:
        return None
    ref = arr[0]
    chimeric = [''.join(_ID2AA[x] for x in ref)]
    del_frac = [0.0]
    hamming = [0.0]
    for s in arr[1:]:
        dmask = (s == 0)
        del_frac.append(float(dmask.sum()) / L)
        s_mod = s.copy()
        s_mod[dmask] = ref[dmask]
        hamming.append(float((s_mod != ref).sum()) / L)
        chimeric.append(''.join(_ID2AA[x] for x in s_mod))
    return pd.DataFrame({
        'protein_name': names,
        'ins_frac': np.array(ins_num) / L,
        'del_frac': del_frac,
        'hamming_dist': hamming,
        'seq': chimeric,
    })


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--a3m_dir', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshards', type=int, default=1)
    a = p.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(a.a3m_dir) if f.endswith('.a3m'))
    mine = [f for i, f in enumerate(files) if i % a.nshards == a.shard]
    print(f'[build_chimeric] shard {a.shard}/{a.nshards}: {len(mine)} of {len(files)} a3m')

    n_ok = n_skip = n_bad = 0
    for i, fn in enumerate(mine):
        key = fn[:-4]                               # strip .a3m
        out = os.path.join(a.out_dir, f'{key}.chimeric')
        if os.path.exists(out):
            n_skip += 1
            continue
        df = a3m_to_chimeric(os.path.join(a.a3m_dir, fn))
        if df is None or len(df) == 0:
            n_bad += 1
            continue
        df.to_csv(out, index=False, float_format='%.3f')
        n_ok += 1
        if (i + 1) % 500 == 0:
            print(f'  [{a.shard}] {i+1}/{len(mine)} ok={n_ok} skip={n_skip} bad={n_bad}')
    print(f'[build_chimeric] shard {a.shard} DONE: wrote {n_ok}, skipped {n_skip}, bad {n_bad}')


if __name__ == '__main__':
    main()
