#!/usr/bin/env python3
"""Pre-registered test on pep_new_af2 (eval/decoys/PREREG_pep_new.md + Amendment 1).

Input: <dir>/per_target.csv from af2_select.py --set pep_new_af2 (per peptide x scorer, all peptides).
  conf = max pLDDT of the peptide's AF2 models
  gated pick: conf >= 80 -> pLDDT pick, else -> energy pick (ribbon k10 w10 cutoff 1000)
  d_gate = GDT(gated pick) - GDT(pLDDT pick); d_ribbon = GDT(ribbon pick) - GDT(pLDDT pick)
H1 (primary, length >= 15): mean d_gate > 0, one-sided Wilcoxon signed-rank on non-zero d_gate.
H2 (length >= 15): Spearman(d_ribbon, conf) < 0, one-sided permutation p (10,000).
Secondary: all peptides with fallback to the pLDDT pick where the energy did not score (d = 0).
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

THR = 80
E = 'rib_k10_smooth_w10_c1000'


def boot_ci(x, rng, n=10000):
    m = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return np.percentile(m, [2.5, 97.5])


def h1(d, rng, label):
    nz = d[np.abs(d) > 1e-9]
    p = wilcoxon(nz, alternative='greater')[1] if len(nz) >= 6 else np.nan
    lo, hi = boot_ci(d, rng)
    print(f'H1 {label}: n={len(d)}  mean d_gate={d.mean():+.3f} GDT  95% CI [{lo:+.3f},{hi:+.3f}]  '
          f'non-zero={len(nz)} (better {int((nz > 0).sum())} / worse {int((nz < 0).sum())})  '
          f'one-sided Wilcoxon p={p:.4f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='eval/decoys/af2_select_pep_new')
    ap.add_argument('--meta', default='nnef/data/decoys/pep_new/targets_summary.csv')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--split_col', default='', help='descriptive split column of --meta, e.g. homolog_in_train')
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    df = pd.read_csv(f'{a.dir}/per_target.csv')
    L = pd.read_csv(a.meta).set_index('pdb').length
    piv = df.pivot_table(index='target', columns='scorer', values='pick')
    conf = df.groupby('target').plddt_max.first()
    t = pd.DataFrame(dict(conf=conf, pl=piv['plddt'], rib=piv.get(E), L=L.reindex(piv.index)))
    t['scored'] = t.rib.notna()
    t['gate'] = np.where((t.conf >= THR) | ~t.scored, t.pl, t.rib)
    t['d_gate'] = t.gate - t.pl
    t['d_rib'] = t.rib - t.pl
    t.to_csv(f'{a.dir}/prereg_per_target.csv')
    prim = t[(t.L >= 15) & t.scored]
    print(f'peptides: all {len(t)}, length>=15 {int((t.L >= 15).sum())}, of which energy-scored {len(prim)}; '
          f'conf<{THR} among primary: {int((prim.conf < THR).sum())}')
    print('--- PRIMARY (length >= 15) ---')
    h1(prim.d_gate.values, rng, 'gated rule')
    r0 = spearmanr(prim.d_rib, prim.conf)[0]
    null = np.array([spearmanr(prim.d_rib, rng.permutation(prim.conf.values))[0] for _ in range(10000)])
    print(f'H2: Spearman(d_ribbon, conf) = {r0:+.3f}, one-sided permutation p = {(null <= r0).mean():.4f}')
    print('--- SECONDARY (all peptides, fallback to pLDDT where unscored) ---')
    h1(t.d_gate.values, rng, 'gated rule, all')
    print('--- DESCRIPTIVE: bins of conf (length >= 15) ---')
    b = pd.cut(prim.conf, [0, 70, 80, 90, 101], right=False, labels=['<70', '70-80', '80-90', '>=90'])
    print(prim.groupby(b, observed=False).agg(n=('d_rib', 'size'), mean_d_ribbon=('d_rib', 'mean'),
                                              mean_d_gate=('d_gate', 'mean')).round(3).to_string())
    if a.split_col:
        print(f'--- DESCRIPTIVE: split by {a.split_col} (length >= 15) ---')
        g = pd.read_csv(a.meta).set_index('pdb')[a.split_col].reindex(prim.index)
        for v, x in prim.groupby(g):
            h1(x.d_gate.values, rng, f'{a.split_col}={v}')


if __name__ == '__main__':
    main()
