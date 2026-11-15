#!/usr/bin/env python3
"""Summarise the per-term decoy decomposition (fasrc/decomp_decoys_array.slurm).

For every model x decoy set x target, correlate each energy term with decoy
quality WITHIN the target (energies of different proteins are not comparable:
the per-block mean energy has a composition-dependent baseline), then report
the distribution over targets.

Quality is GDT_TS for casp14 (good model -> NEGATIVE r) and RMSD for 3DRobot
(good model -> POSITIVE r). Everything is sign-flipped to "higher = better
ranking" so the two sets read the same way.

Terms: total, struct (= loss_r + loss_angle + loss_rama), profile, start_id.
casp14 additionally reports a modal-length-only correlation, to show whether
truncated decoys (fewer residues) bias the ranking.

    python eval/decoys/collect_decomp.py            # run from repo root on the cluster
"""
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = os.path.join('nnef', 'data', 'decoys')
SETS = {'3DRobot_set': ('RMSD', +1), 'casp14': ('GDT_TS', -1), 'casp14_full': ('GDT_TS', -1), 'pep_nmr': ('GDT_TS', -1)}
TERMS = ['total', 'struct', 'profile', 'start_id']


def n_lines(path):
    with open(path) as fh:
        return sum(1 for _ in fh) - 1


def target_rows(model, dset, csv):
    pdb = os.path.basename(csv).replace('_decoy_loss.csv', '')
    qcol, sign = SETS[dset]
    d = pd.read_csv(csv)
    if qcol not in d.columns:
        lst = pd.read_csv(os.path.join(ROOT, dset, pdb, 'list.csv'))
        d = d.merge(lst[['NAME', qcol]], on='NAME', how='left')
    d = d[(d['loss'] < 900 * 10) & d[qcol].notna()].copy()  # 999 = failed decoy
    if len(d) < 20 or d[qcol].std() == 0:
        return []
    d['total'] = d['loss']
    d['struct'] = d['loss_r'] + d['loss_angle'] + d.get('loss_rama', 0.0)
    d['profile'] = d['loss_profile']
    d['start_id'] = d['loss_start_id']
    subsets = {'all': d}
    if dset.startswith('casp14'):
        lens = {n: n_lines(os.path.join(ROOT, dset, pdb, f'{n}_bead.csv')) for n in d['NAME']}
        d['L'] = d['NAME'].map(lens)
        subsets['modal_len'] = d[d['L'] == d['L'].mode()[0]]
    rows = []
    for sub, s in subsets.items():
        for t in TERMS:
            if s[t].std() == 0:
                continue
            rows.append(dict(model=model, set=dset, target=pdb, subset=sub, term=t, n=len(s),
                             pearson=sign * pearsonr(s[t], s[qcol])[0],
                             spearman=sign * spearmanr(s[t], s[qcol])[0]))
    return rows


def main():
    rows = []
    for dset in SETS:
        for ddir in sorted(glob.glob(os.path.join(ROOT, dset, 'decoy_loss_decomp_*'))):
            model = os.path.basename(ddir).replace('decoy_loss_decomp_', '')
            for csv in sorted(glob.glob(os.path.join(ddir, '*_decoy_loss.csv'))):
                if 'loss_profile' not in pd.read_csv(csv, nrows=1).columns:
                    continue
                rows += target_rows(model, dset, csv)
    df = pd.DataFrame(rows)
    os.makedirs('eval/decomp', exist_ok=True)
    df.to_csv('eval/decomp/decomp_per_target.csv', index=False)

    summ = (df.groupby(['set', 'subset', 'model', 'term'])
              .agg(n_targets=('target', 'nunique'),
                   median_pearson=('pearson', 'median'),
                   frac_gt_075=('pearson', lambda x: float((x > 0.75).mean())),
                   median_spearman=('spearman', 'median'))
              .reset_index())
    summ.to_csv('eval/decomp/decomp_summary.csv', index=False)
    pd.set_option('display.width', 200)
    for (dset, sub), g in summ.groupby(['set', 'subset']):
        print(f'\n=== {dset} [{sub}]  (sign-flipped: higher = better ranking) ===')
        print(g.pivot(index='model', columns='term', values='median_pearson')
               .round(3)[TERMS].to_string())
        print('  frac targets with total r > 0.75:')
        print(g[g.term == 'total'].set_index('model')[['n_targets', 'frac_gt_075']].to_string())


if __name__ == '__main__':
    main()
