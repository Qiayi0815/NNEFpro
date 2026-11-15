#!/usr/bin/env python3
"""Protein decoy ranking (3DRobot_set: RMSD; casp14_full: GDT_TS) from the CPU
per-block scores in eval/per_residue/<model>/<set>/<target>.npz.

Decoy energy = mean block energy over residues covered in >= 90% of decoys
(pep_analysis.decoy_energies). Per target: Pearson/Spearman with quality
sign-flipped so higher = better ranking. Paired comparisons on identical decoy
sets: median difference, bootstrap CI over targets, Wilcoxon.

    python eval/decoys/prot_analysis.py --pair rib_k10_smooth_w10,yang_k10_l10
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pep_analysis import decoy_energies  # noqa: E402

SETS = {'3DRobot_set': ('RMSD', +1), 'casp14_full': ('GDT_TS', -1)}
ROOT = 'eval/per_residue'


def quality(dset, t):
    q, sign = SETS[dset]
    lst = pd.read_csv(f'nnef/data/decoys/{dset}/{t}/list.csv')
    lst['NAME'] = lst['NAME'].astype(str).str.replace(r'\.pdb$', '', regex=True)
    return lst.set_index('NAME')[q], sign


def per_target(model, dset):
    rows = []
    for f in sorted(glob.glob(f'{ROOT}/{model}/{dset}/*.npz')):
        t = os.path.basename(f)[:-4]
        e = decoy_energies(f)
        q, sign = quality(dset, t)
        d = pd.concat([e.rename('e'), q.rename('q')], axis=1).dropna()
        if len(d) < 20 or d.e.std() == 0:
            continue
        rows.append(dict(model=model, set=dset, target=t, n=len(d),
                         pearson=sign * pearsonr(d.e, d.q)[0], spearman=sign * spearmanr(d.e, d.q)[0]))
    return rows


def shared(A, B, dset):
    out = []
    for f in sorted(glob.glob(f'{ROOT}/{A}/{dset}/*.npz')):
        t = os.path.basename(f)[:-4]
        fb = f'{ROOT}/{B}/{dset}/{t}.npz'
        if not os.path.exists(fb):
            continue
        q, sign = quality(dset, t)
        d = pd.concat([decoy_energies(f).rename('a'), decoy_energies(fb).rename('b'), q.rename('q')], axis=1).dropna()
        if len(d) >= 20:
            out.append(sign * (pearsonr(d.a, d.q)[0] - pearsonr(d.b, d.q)[0]))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair', action='append', default=[])
    a = ap.parse_args()
    models = sorted({p.split(os.sep)[-2] for p in glob.glob(f'{ROOT}/*/*')})
    rows = []
    for m in models:
        for dset in SETS:
            rows += per_target(m, dset)
    df = pd.DataFrame(rows)
    df.to_csv('eval/decoys/prot_per_target.csv', index=False)
    s = df.groupby(['set', 'model']).agg(n=('target', 'size'), pearson=('pearson', 'median'),
                                         spearman=('spearman', 'median'),
                                         frac_gt_075=('pearson', lambda x: float((x > 0.75).mean())))
    pd.set_option('display.width', 200)
    print(s.round(3).to_string())
    for p in a.pair:
        A, B = p.split(',')
        for dset in SETS:
            d = shared(A, B, dset)
            if len(d) < 5:
                continue
            boot = np.median(np.random.default_rng(0).choice(d, (10000, len(d))), 1)
            print(f'[paired pearson, shared decoys] {dset:12s} {A} - {B}: n={len(d)} median {np.median(d):+.4f} '
                  f'CI [{np.percentile(boot, 2.5):+.4f}, {np.percentile(boot, 97.5):+.4f}] '
                  f'A better {np.mean(d > 0):.2f} p={wilcoxon(d).pvalue:.3g}')


if __name__ == '__main__':
    main()
