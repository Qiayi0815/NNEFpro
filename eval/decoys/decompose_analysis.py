#!/usr/bin/env python3
"""Correlate each ENERGY TERM with decoy quality (GDT), to see which term drives
(or kills) the decoy ranking.

Run the decoy eval with --decompose_energy first so <pdb>_decoy_loss.csv carries
per-term columns (loss_r, loss_angle, loss_profile, ...). Then:

  python eval/decoys/decompose_analysis.py \
      --decoy_loss nnef/data/decoys/casp14/<dir>/T1026_decoy_loss.csv \
      --list_csv   nnef/data/decoys/casp14/T1026/list.csv

Hypothesis for the ESM collapse: the total 'loss' is dominated by loss_profile;
for yang/ribbon loss_profile tracks structure (strong -r vs GDT), but ESM makes
it structure-independent -> its correlation goes to ~0 and drags the total down,
while the structure-only term (loss_r + loss_angle) stays strongly negative.
"""
import argparse
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--decoy_loss', required=True)
    ap.add_argument('--list_csv', required=True)
    a = ap.parse_args()

    dl = pd.read_csv(a.decoy_loss)
    lc = pd.read_csv(a.list_csv)
    key = 'NAME' if 'NAME' in dl.columns and 'NAME' in lc.columns else dl.columns[0]
    m = dl.merge(lc, on=key)
    gdt_col = next((c for c in m.columns if 'GDT' in c.upper()), None)
    if gdt_col is None:
        raise SystemExit(f'no GDT column in {a.list_csv} (cols={list(lc.columns)})')
    gdt = m[gdt_col].to_numpy(dtype=float)

    def r(x):
        x = np.asarray(x, dtype=float)
        ok = np.isfinite(x) & np.isfinite(gdt)
        return np.corrcoef(x[ok], gdt[ok])[0, 1] if ok.sum() > 2 else float('nan')

    print(f'{a.decoy_loss}')
    print(f'n={len(m)} valid, vs {gdt_col}\n')
    print(f'{"term":18s} pearson_vs_GDT   mean_energy')
    for t in ['loss', 'loss_r', 'loss_angle', 'loss_profile',
              'loss_start_id', 'loss_res_counts', 'loss_rama']:
        if t in m.columns:
            print(f'  {t:16s} {r(m[t]):+7.3f}        {np.nanmean(m[t]):.2f}')
    if 'loss_r' in m and 'loss_angle' in m:
        s = m['loss_r'].to_numpy(float) + m['loss_angle'].to_numpy(float)
        print(f'  {"struct(r+angle)":16s} {r(s):+7.3f}        {np.nanmean(s):.2f}')


if __name__ == '__main__':
    main()
