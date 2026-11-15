#!/usr/bin/env python3
"""Paired significance for decoy-ranking comparisons, and a train-overlap split.

Median Pearson differences of ~0.005 between models are within target-to-target
noise unless tested PAIRED (same target, two models). For each requested pair
A vs B this reports, over the targets both models scored:
  median r(A) - median r(B), median of per-target differences with a 95%
  bootstrap CI (targets resampled), fraction of targets where A > B, and the
  Wilcoxon signed-rank p-value.

It also splits every (model, w) by whether the target's PDB code is in the
training list, so a memorisation gap (in-train >> held-out) is visible.

Input: eval/decomp/profile_weight_scan.csv (set, model, target, w, r).

    python eval/decoys/paired_stats.py --train_list ~/nnef_data/hhsuite_CB_yang_pdb_list.csv \
        --pair yang_full1000_27329319:0.1,yang_full1000_27329319:1 \
        --pair ribbon_fix_xxx:0.3,yang_full1000_27329319:1
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def parse_side(s):
    m, w = s.rsplit(':', 1)
    return m, float(w)


def boot_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(x, size=(n, len(x)), replace=True), axis=1)
    return np.percentile(meds, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scan', default='eval/decomp/profile_weight_scan.csv')
    ap.add_argument('--train_list', default=None, help='pdb_list csv the models trained on (pdb = XXXX_C)')
    ap.add_argument('--pair', action='append', default=[], help='modelA:wA,modelB:wB')
    ap.add_argument('--out', default='eval/decomp/paired_stats.csv')
    args = ap.parse_args()

    df = pd.read_csv(args.scan)
    df['code'] = df['target'].str[:4].str.upper()
    rows = []
    for p in args.pair:
        a, b = (parse_side(s) for s in p.split(','))
        for dset, g in df.groupby('set'):
            ra = g[(g.model == a[0]) & np.isclose(g.w, a[1])].set_index('target')['r']
            rb = g[(g.model == b[0]) & np.isclose(g.w, b[1])].set_index('target')['r']
            common = ra.index.intersection(rb.index)
            if len(common) < 5:
                continue
            d = (ra[common] - rb[common]).to_numpy()
            lo, hi = boot_ci(d)
            rows.append(dict(set=dset, A=f'{a[0]}@w{a[1]:g}', B=f'{b[0]}@w{b[1]:g}', n=len(common),
                             med_A=ra[common].median(), med_B=rb[common].median(),
                             med_diff=float(np.median(d)), ci_lo=lo, ci_hi=hi,
                             frac_A_better=float((d > 0).mean()),
                             wilcoxon_p=float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0))
    out = pd.DataFrame(rows)
    pd.set_option('display.width', 250)
    if len(out):
        out.to_csv(args.out, index=False)
        print('=== paired comparisons (Pearson, higher = better) ===')
        print(out.round(4).to_string(index=False))

    if args.train_list:
        train = set(pd.read_csv(args.train_list)['pdb'].str[:4].str.upper())
        df['in_train'] = df['code'].isin(train)
        sp = (df.groupby(['set', 'model', 'w', 'in_train'])['r']
                .agg(['size', 'median']).unstack('in_train'))
        sp.columns = [f'{a}_{"train" if b else "heldout"}' for a, b in sp.columns]
        sp['gap_train_minus_heldout'] = sp['median_train'] - sp['median_heldout']
        sp = sp.reset_index()
        sp = sp[np.isclose(sp.w, 1.0) | np.isclose(sp.w, 0.1) | np.isclose(sp.w, 0.3)]
        sp.to_csv(args.out.replace('.csv', '_train_overlap.csv'), index=False)
        print('\n=== train-overlap split (median Pearson; targets whose PDB code is in the training list) ===')
        print(sp.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
