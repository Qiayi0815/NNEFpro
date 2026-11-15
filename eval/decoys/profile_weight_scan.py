#!/usr/bin/env python3
"""Decoy ranking of the re-weighted energy E_w = struct + w * profile (+ start_id).

Uses the per-term columns written by --decompose_energy (no rescoring needed).
The training-time weighting (profile_loss_lamda=10) is what the stored
loss_profile already includes, so w=1 reproduces the shipped energy; w<1
down-weights the sequence-profile term, w=0 is geometry(+start_id)-only.

    python eval/decoys/profile_weight_scan.py   # from repo root, after collect_decomp.py
"""
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = os.path.join('nnef', 'data', 'decoys')
SETS = {'3DRobot_set': ('RMSD', +1), 'casp14': ('GDT_TS', -1), 'casp14_full': ('GDT_TS', -1), 'pep_nmr': ('GDT_TS', -1)}
WEIGHTS = [0.0, 0.03, 0.1, 0.3, 0.5, 1.0, 2.0]


def main():
    rows = []
    for dset, (qcol, sign) in SETS.items():
        for ddir in sorted(glob.glob(os.path.join(ROOT, dset, 'decoy_loss_decomp_*'))):
            model = os.path.basename(ddir).replace('decoy_loss_decomp_', '')
            for csv in sorted(glob.glob(os.path.join(ddir, '*_decoy_loss.csv'))):
                d = pd.read_csv(csv)
                if 'loss_profile' not in d.columns:
                    continue
                pdb = os.path.basename(csv).replace('_decoy_loss.csv', '')
                if qcol not in d.columns:
                    lst = pd.read_csv(os.path.join(ROOT, dset, pdb, 'list.csv'))
                    d = d.merge(lst[['NAME', qcol]], on='NAME', how='left')
                d = d[(d['loss'] < 9000) & d[qcol].notna()]
                if len(d) < 20 or d[qcol].std() == 0:
                    continue
                geo = d['loss_r'] + d['loss_angle'] + d.get('loss_rama', 0.0)
                rest = d['loss_start_id'] + d.get('loss_res_counts', 0.0)
                for w in WEIGHTS:
                    e = geo + rest + w * d['loss_profile']
                    if e.std() == 0:
                        continue
                    rows.append(dict(set=dset, model=model, target=pdb, w=w,
                                     r=sign * pearsonr(e, d[qcol])[0]))
    df = pd.DataFrame(rows)
    os.makedirs('eval/decomp', exist_ok=True)
    df.to_csv('eval/decomp/profile_weight_scan.csv', index=False)
    for dset, g in df.groupby('set'):
        print(f'\n=== {dset}: median r of struct+start_id+w*profile (higher = better) ===')
        print(g.groupby(['model', 'w'])['r'].median().unstack('w').round(3).to_string())


if __name__ == '__main__':
    main()
