#!/usr/bin/env python3
"""Does an energy help more where AlphaFold2 is unsure? (pep_af2, analysis fixed before looking)

Input: <dir>/per_target.csv from af2_select.py (one row per peptide x scorer).
Per peptide: d = GDT(pick of scorer) - GDT(pick of pLDDT)   (> 0: better than pLDDT)
             conf = max pLDDT over the peptide's AF2 models
Primary test (pre-specified): Spearman(d, conf) < 0, one-sided, permutation p (10,000 shuffles).
Bins: conf < 70, 70-80, 80-90, >= 90; mean d with bootstrap 95% CI.
Controls: random pick (regression-to-the-mean reference: when the pLDDT pick is poor, any other
pick tends to look better), rwplus, rama_aa. A scorer's trend counts only if it is clearly stronger
than random's; reported as the difference in Spearman vs random with a paired bootstrap CI.

    python eval/decoys/af2_conf_trend.py --dir eval/decoys/af2_select --scorers rib_k10_smooth_w10_c1000,...
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BINS = [0, 70, 80, 90, 101]
LABELS = ['<70', '70-80', '80-90', '>=90']


def perm_p(d, c, rng, n=10000):
    r0 = spearmanr(d, c)[0]
    null = np.array([spearmanr(d, rng.permutation(c))[0] for _ in range(n)])
    return r0, float((null <= r0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='eval/decoys/af2_select')
    ap.add_argument('--scorers', required=True, help='comma list; random/rwplus/rama_aa added as controls')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    df = pd.read_csv(f'{a.dir}/per_target.csv')
    pl = df[df.scorer == 'plddt'].set_index('target').pick
    conf = df.groupby('target').plddt_max.first()
    scorers = [s for s in a.scorers.split(',') if s] + ['random', 'rwplus', 'rama_aa']
    D = {s: (df[df.scorer == s].set_index('target').pick - pl).dropna() for s in scorers if (df.scorer == s).any()}
    common = sorted(set.intersection(*[set(v.index) for v in D.values()]))
    D = pd.DataFrame({s: v.reindex(common) for s, v in D.items()})
    c = conf.reindex(common).values
    b = pd.cut(c, BINS, right=False, labels=LABELS)
    print(f'n peptides = {len(common)}; per bin: {pd.Series(b).value_counts().reindex(LABELS).to_dict()}')
    rows = []
    for s in D:
        d = D[s].values
        r, p = perm_p(d, c, rng)
        row = dict(scorer=s, spearman_d_vs_conf=r, p_one_sided=p)
        for lab in LABELS:
            x = d[b == lab]
            bs = [rng.choice(x, len(x)).mean() for _ in range(2000)] if len(x) > 1 else [np.nan]
            row[f'{lab} mean_d'] = x.mean() if len(x) else np.nan
            row[f'{lab} CI'] = f'[{np.percentile(bs, 2.5):+.2f},{np.percentile(bs, 97.5):+.2f}]'
        # trend relative to random (paired bootstrap over peptides)
        if s != 'random':
            dr = D['random'].values
            diffs = []
            for _ in range(2000):
                i = rng.integers(0, len(d), len(d))
                diffs.append(spearmanr(d[i], c[i])[0] - spearmanr(dr[i], c[i])[0])
            row['rho_minus_random'] = r - spearmanr(dr, c)[0]
            row['rho_minus_random_CI'] = f'[{np.percentile(diffs, 2.5):+.3f},{np.percentile(diffs, 97.5):+.3f}]'
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(f'{a.dir}/conf_trend.csv', index=False)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 30)
    print('d = GDT(pick) - GDT(pLDDT pick); negative Spearman = helps more where AF2 is unsure')
    print(out.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
