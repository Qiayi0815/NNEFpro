#!/usr/bin/env python3
"""Choose the inference identity weight w on a VALIDATION half of pep_nmr, report once on TEST.

Reviewers flagged that w = 10-15 was found by scanning the same peptides used to report
results. Here the 537 peptides are split by MMseqs2 40%-identity cluster (no cluster on
both sides) into validation / test halves, and w is chosen with a rule fixed in advance:

    among the scanned w with validation helix-control wins (non-helical natives) <= 5%,
    take the one with the highest validation median Spearman on all non-control decoys;
    ties -> the smaller w.

Input: eval/decoys/pep_compare_w_per_target.csv from pep_compare.py run over the w series.

    python eval/decoys/w_heldout.py --family rib_k10_smooth:1,3,10,15,30 --family rib_k4_smooth:1,3,6,10,15
"""
import argparse

import numpy as np
import pandas as pd

DEC = 'nnef/data/decoys/pep_nmr'


def tag(base, w):
    return base if w == 1 else f'{base}_w{w:g}'


def summarise(df):
    nonhel = df.ss != 'helix'
    return pd.Series(dict(n=len(df), noctrl=df.noctrl.median(), pert=df.pert.median(),
                          helix_wins_nonhelical=df.loc[nonhel, 'helix_wins'].mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per_target', default='eval/decoys/pep_compare_w_per_target.csv')
    ap.add_argument('--family', action='append', required=True, help='base_tag:w1,w2,...')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max_helix', type=float, default=0.05)
    a = ap.parse_args()
    df = pd.read_csv(a.per_target)
    clu = pd.read_csv(f'{DEC}/seq_clusters_id40.tsv', sep='\t', header=None, names=['rep', 'member'])
    reps = np.array(sorted(clu.rep.unique()))
    rng = np.random.default_rng(a.seed)
    val_reps = set(rng.choice(reps, size=len(reps) // 2, replace=False))
    split = clu.set_index('member').rep.map(lambda r: 'val' if r in val_reps else 'test')
    df['split'] = df.target.map(split)
    pd.set_option('display.width', 200)
    for fam in a.family:
        base, ws = fam.split(':')
        ws = [float(w) for w in ws.split(',')]
        rows = []
        for w in ws:
            sub = df[df.entity == tag(base, w)]
            for sp in ('val', 'test'):
                rows.append(dict(w=w, split=sp, **summarise(sub[sub.split == sp])))
        t = pd.DataFrame(rows)
        v = t[t.split == 'val']
        ok = v[v.helix_wins_nonhelical <= a.max_helix]
        chosen = (ok.sort_values(['noctrl', 'w'], ascending=[False, True]).iloc[0].w if len(ok)
                  else v.sort_values('helix_wins_nonhelical').iloc[0].w)
        print(f'\n=== {base}: w chosen on validation = {chosen:g} '
              f'(rule: helix wins <= {a.max_helix:.0%}, then max noctrl Spearman) ===')
        print(t.pivot(index='w', columns='split', values=['noctrl', 'pert', 'helix_wins_nonhelical', 'n'])
              .round(3).to_string())
        te = t[(t.split == 'test') & (t.w == chosen)].iloc[0]
        print(f'TEST (reported once): w={chosen:g}  noctrl {te.noctrl:.3f}  pert {te.pert:.3f}  '
              f'helix wins (non-helical) {te.helix_wins_nonhelical:.1%}  n={int(te.n)}')


if __name__ == '__main__':
    main()
