#!/usr/bin/env python3
"""Energy-vs-GDT decoy scatter grid for the retrained models.

Reads the per-decoy energies written by the scorer
(``nnef/data/decoys/casp14/decoy_loss_<run>_<target>/<target>_decoy_loss.csv``)
and each target's ``list.csv`` (GDT_TS), and draws a rows=targets x cols=models
scatter grid with the Pearson r annotated. This is the "does energy rank decoy
quality" figure -- and shows the esm collapse (flat cloud, r~0) next to
yang/ribbon (strong negative slope).

  python eval/decoys/make_decoy_figures.py --targets T1026 T1053 \
      --runs yang_full1000 ribbon_full1000 esm_full1000
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEC = os.path.join(ROOT, 'nnef', 'data', 'decoys', 'casp14')
COLOR = {'yang': '#1f77b4', 'ribbon': '#d62728', 'esm': '#2ca02c'}


def _family(run):
    return next((f for f in COLOR if run.startswith(f)), 'other')


def _find_loss_csv(run, target):
    # dir is decoy_loss_<RUN_BASE>_<target>; RUN_BASE may carry a jobid suffix.
    hits = glob.glob(os.path.join(DEC, f'decoy_loss_{run}*_{target}', f'{target}_decoy_loss.csv'))
    return sorted(hits)[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', nargs='+', default=['T1026', 'T1053'])
    ap.add_argument('--runs', nargs='+',
                    default=['yang_full1000', 'ribbon_full1000', 'esm_full1000'])
    ap.add_argument('--out', default=os.path.join(ROOT, 'eval', 'md_new', 'figures',
                                                  'decoy_scatter.png'))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    nT, nR = len(a.targets), len(a.runs)
    fig, axes = plt.subplots(nT, nR, figsize=(3.2 * nR, 3.0 * nT), squeeze=False)
    for i, t in enumerate(a.targets):
        lc = pd.read_csv(os.path.join(DEC, t, 'list.csv'))
        for j, run in enumerate(a.runs):
            ax = axes[i][j]
            csv = _find_loss_csv(run, t)
            if not csv:
                ax.set_title(f'{run}\n(no data)', fontsize=8); ax.axis('off'); continue
            dl = pd.read_csv(csv)
            key = 'NAME' if 'NAME' in dl.columns and 'NAME' in lc.columns else dl.columns[0]
            m = dl.merge(lc, on=key)
            gcol = next(c for c in m.columns if 'GDT' in c.upper())
            g = m[gcol].to_numpy(float); e = m['loss'].to_numpy(float)
            ok = np.isfinite(g) & np.isfinite(e)
            r = np.corrcoef(e[ok], g[ok])[0, 1] if ok.sum() > 2 else float('nan')
            ax.scatter(g[ok], e[ok], s=8, alpha=0.5, color=COLOR.get(_family(run), '#888'))
            ax.set_title(f'{run}\n{t}  r={r:+.3f}  N={ok.sum()}', fontsize=8)
            if j == 0:
                ax.set_ylabel('energy')
            if i == nT - 1:
                ax.set_xlabel('GDT_TS')
            ax.grid(alpha=0.3)
    fig.suptitle('Decoy energy vs quality (strong negative slope = good ranking)',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(a.out, dpi=150)
    print(f'[decoy figs] wrote {a.out}')


if __name__ == '__main__':
    main()
