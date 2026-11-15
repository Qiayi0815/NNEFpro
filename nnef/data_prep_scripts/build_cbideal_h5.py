"""Copy a v2 structure h5 with every non-Gly CB rebuilt from N/CA/C (ideal CB geometry).

Why: the identity (profile) term of ribbon is very sensitive to sub-Angstrom CB
micro-placement (CASP14 decoys rescored with ideal CB: per-target r 0.55 -> 0.69;
FIP35 300 K Amber frames: +135 -> +37 kT*). Training on ideal CB removes that
nuisance variable; inference must then use ideal CB too.

Block membership, order (central residue = row 0), seq/seg/start_id/res_counts are copied unchanged, so the
existing seq h5 and rama h5 (aligned by group_num) stay valid. Only `coords` is
recomputed:  coords = R_cen @ (CB_ideal - CA_cen), with R_cen the central residue's
N-CA-C frame (local_extractor_v2.compute_local_frame). Gly keeps CB = CA.

Check per block: the ORIGINAL CB from the bead CSV must reproduce the stored coords
(max |d| < --tol); blocks that fail keep their original coords and are counted.

    python -s nnef/data_prep_scripts/build_cbideal_h5.py \
        --cb_h5 .../hhsuite_CB_v2_k10.h5 --bead_dir ~/nnef_data/bead_csvs \
        --out_h5 .../hhsuite_CBideal_v2_k10.h5 [--limit 200]
"""
import argparse
import os
import sys

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from local_extractor_v2 import compute_local_frame  # noqa: E402


def ideal_cb(n, ca, c):
    """Same coefficients as runs/q2_decoys/term_split_score_cb.py (decoy CB-ideal test)."""
    b, cc = ca - n, c - ca
    a = np.cross(b, cc)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * cc + ca


def convert_chain(key, grp, bead_dir, tol):
    """Keys are '12AS_A' or concatenated ('7QVDAAA'); try every chain in the bead CSV,
    implied ones first, and keep the one whose original CB reproduces the most blocks."""
    coords = grp['coords'][()]
    f = os.path.join(bead_dir, f'{key[:4]}_bead.csv')
    if not os.path.exists(f):
        return coords, 0, len(coords)
    df_all = pd.read_csv(f)
    chains = df_all['chain_id'].astype(str).unique().tolist()
    implied = [key.split('_')[-1]] if '_' in key else [key[4:], key[-1]]
    best = (coords, 0, len(coords))
    for ch in [c for c in implied if c in chains] + [c for c in chains if c not in implied]:
        res = _convert(df_all[df_all['chain_id'].astype(str) == ch], grp, coords, tol)
        if res[1] > best[1]:
            best = res
        if best[1] == len(coords):
            break
    return best


def _convert(df, grp, coords, tol):
    gn_all = df['group_num'].values.astype(int)
    row = {g: i for i, g in enumerate(gn_all)}
    N, CA, C, CB = (df[[f'x{a}', f'y{a}', f'z{a}']].values.astype(np.float64) for a in ('n', 'ca', 'c', 'cb'))
    gly = df['group_name'].astype(str).str.upper().isin(['G', 'GLY']).values
    CBi = np.where(gly[:, None], CB, ideal_cb(N, CA, C))

    gnum = grp['group_num'][()].astype(int)
    out, n_ok = coords.copy(), 0
    for b in range(len(coords)):
        idx = [row.get(g, -1) for g in gnum[b]]
        if min(idx) < 0:
            continue
        ic = idx[0]                     # _re_order_block puts the central residue first
        R = compute_local_frame(N[ic], CA[ic], C[ic])
        if R is None:
            continue
        orig = (R @ (CB[idx] - CA[ic]).T).T
        if np.abs(orig - coords[b]).max() > tol:
            continue
        out[b] = (R @ (CBi[idx] - CA[ic]).T).T
        n_ok += 1
    return out.astype(np.float32), n_ok, len(coords)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cb_h5', required=True)
    p.add_argument('--bead_dir', required=True)
    p.add_argument('--out_h5', required=True)
    p.add_argument('--tol', type=float, default=0.02)   # bead CSVs are written with 3 decimals
    p.add_argument('--limit', type=int, default=None)
    args = p.parse_args()

    tot_ok = tot = 0
    shift = []
    with h5py.File(args.cb_h5, 'r') as fi, h5py.File(args.out_h5, 'w') as fo:
        keys = list(fi.keys())[:args.limit]
        for i, k in enumerate(keys):
            g = fi[k]
            new, n_ok, n = convert_chain(k, g, args.bead_dir, args.tol)
            go = fo.create_group(k)
            for name in g:
                go.create_dataset(name, data=new if name == 'coords' else g[name][()], compression='gzip')
            tot_ok, tot = tot_ok + n_ok, tot + n
            shift.append(np.linalg.norm(new - g['coords'][()], axis=-1).ravel())
            if i % 2000 == 0:
                print(f'[cbideal] {i}/{len(keys)} {k}: {n_ok}/{n} blocks rebuilt', flush=True)
    s = np.concatenate(shift)
    print(f'[cbideal] rebuilt {tot_ok}/{tot} blocks ({tot_ok / max(tot, 1):.1%}); '
          f'CB shift median {np.median(s):.3f} A, p95 {np.percentile(s, 95):.3f} A -> {args.out_h5}')


if __name__ == '__main__':
    main()
