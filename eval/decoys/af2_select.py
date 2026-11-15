#!/usr/bin/env python3
"""ColabFold model selection on pep_af2: can an energy pick a better AF2 model than pLDDT?

Per peptide, the AF2 models (native excluded) are scored by each scorer; the pick is the
model with the lowest score. Quality = gdt_ts_max_ens (GDT_TS to the closest NMR model).
  loss        = best possible quality - quality of the pick (0 = picked the best model)
  vs_plddt    = pick quality - pLDDT-pick quality (> 0: better than AF2's own confidence)
Scorers: NNEF models (whole-decoy energy, eval/per_residue_pep/<tag>/pep_af2), the block-typed
hybrid, baselines from eval/baselines_pep/<b>/pep_af2/scores.csv (plddt, ptm, rama_aa, rwplus, ...),
random (expected loss), and pLDDT + energy combinations z(-pLDDT) + c*z(E).
Subsets: all peptides; low-confidence (max pLDDT of the target < 70); informative
(best - worst model quality >= 10 GDT, i.e. the choice matters). Paired Wilcoxon vs plddt.

    python eval/decoys/af2_select.py --models rib_k10_smooth_w10,yang_smooth_k10_w10,yang_k10_l10 \
        --hybrid rib_k10_smooth_w10:yang_smooth_k10_w10
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pep_analysis import decoy_energies  # noqa: E402
from combo_eval import hybrid_energies  # noqa: E402

SET = 'pep_af2'
DEC = f'nnef/data/decoys/{SET}'
PER = 'eval/per_residue_pep'
BASE = 'eval/baselines_pep'
COMBO_C = (0.5, 1, 2)


def z(x):
    return (x - x.mean()) / (x.std() + 1e-9)


def scorers_for(t, lst, models, hybrids, baselines):
    """name -> pd.Series(score indexed by NAME); lower = better."""
    out = {}
    for m in models:
        f = f'{PER}/{m}/{SET}/{t}.npz'
        if os.path.exists(f):
            out[m] = decoy_energies(f)
    for h in hybrids:
        a, b = h.split(':')
        fa, fb = f'{PER}/{a}/{SET}/{t}.npz', f'{PER}/{b}/{SET}/{t}.npz'
        if os.path.exists(fa) and os.path.exists(fb):
            out[f'hybrid({a}|{b})'] = hybrid_energies(fa, fb)
    for b, sc in baselines.items():
        g = sc[sc.target == t]
        if len(g):
            out[b] = g.set_index('NAME').score
    out.setdefault('plddt', -lst.set_index('NAME').plddt_mean)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='rib_k10_smooth_w10,yang_smooth_k10_w10,yang_k10_l10')
    ap.add_argument('--hybrid', action='append', default=[], help='terminal_model:interior_model')
    ap.add_argument('--low_plddt', type=float, default=70)
    ap.add_argument('--min_spread', type=float, default=10)
    ap.add_argument('--out', default='eval/decoys/af2_select')
    ap.add_argument('--set', default='pep_af2', help='AF2 candidate set (pep_af2 | pep_new_af2)')
    ap.add_argument('--quality', default='gdt_ts_max_ens', help='gdt_ts_max_ens (closest NMR model) | GDT_TS (medoid)')
    a = ap.parse_args()
    global SET, DEC
    SET, DEC = a.set, f'nnef/data/decoys/{a.set}'
    models = [m for m in a.models.split(',') if m]
    baselines = {}
    for f in glob.glob(f'{BASE}/*/{SET}/scores.csv'):
        b = f.split(os.sep)[-3]
        baselines[b] = pd.read_csv(f)
    targets = pd.read_csv(f'{DEC}/pdb_no_missing_residue.csv').pdb
    rows = []
    for t in targets:
        lst = pd.read_csv(f'{DEC}/{t}/list.csv')
        af = lst[lst.source != 'native'].set_index('NAME')
        q = af[a.quality]
        if len(af) < 2:
            continue
        sc = scorers_for(t, lst, models, a.hybrid, baselines)
        pl = sc['plddt'].reindex(af.index)
        nnef = [k for k in sc if k in models or k.startswith('hybrid(')]
        for k in nnef:
            e = sc[k].reindex(af.index)
            for c in COMBO_C:
                sc[f'plddt+{c:g}{k}'] = z(pl) + c * z(e)
        info = dict(target=t, best=q.max(), worst=q.min(), plddt_max=af.plddt_mean.max())
        for k, s in sc.items():
            s = s.reindex(af.index)
            ok = s.notna()
            if ok.sum() < 0.9 * len(af):          # scorer failed on too many models of this target
                continue
            pick = s[ok].idxmin()
            rows.append(dict(info, scorer=k, n=int(ok.sum()), pick=q[pick], loss=q.max() - q[pick],
                             spearman=spearmanr(-s[ok], q[ok])[0] if s[ok].std() > 0 and q[ok].std() > 0 else np.nan))
        rows.append(dict(info, scorer='random', n=len(af), pick=q.mean(), loss=q.max() - q.mean(), spearman=0.0))
    df = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(f'{a.out}/per_target.csv', index=False)
    # compare every scorer on the SAME peptides: those all NNEF scorers covered
    need = [k for k in df.scorer.unique() if k in models or k.startswith('hybrid(')]
    cov = df[df.scorer.isin(need)].groupby('target').scorer.nunique()
    common = set(cov[cov == len(need)].index)
    print(f'[af2_select] {len(common)}/{df.target.nunique()} peptides covered by all of {need}')
    df = df[df.target.isin(common)]
    pl = df[df.scorer == 'plddt'].set_index('target').pick
    df['vs_plddt'] = df.pick - df.target.map(pl)
    subsets = {'all': df.target.notna(),
               f'low_conf(max pLDDT<{a.low_plddt:g})': df.plddt_max < a.low_plddt,
               f'informative(spread>={a.min_spread:g})': (df.best - df.worst) >= a.min_spread}
    pd.set_option('display.width', 220)
    pd.set_option('display.max_rows', 200)
    out = []
    for name, m in subsets.items():
        d = df[m]
        for k, g in d.groupby('scorer'):
            r = dict(subset=name, scorer=k, n_targets=len(g), mean_pick_gdt=g.pick.mean(), mean_loss=g.loss.mean(),
                     median_loss=g.loss.median(), frac_best=(g.loss <= 1e-6).mean(), mean_vs_plddt=g.vs_plddt.mean(),
                     frac_beats_plddt=(g.vs_plddt > 1e-6).mean(), frac_loses_plddt=(g.vs_plddt < -1e-6).mean(),
                     median_spearman=g.spearman.median())
            nz = g.vs_plddt[g.vs_plddt.abs() > 1e-6]
            r['p_vs_plddt'] = wilcoxon(nz)[1] if k != 'plddt' and len(nz) >= 6 else np.nan
            out.append(r)
    s = pd.DataFrame(out).sort_values(['subset', 'mean_loss'])
    s.to_csv(f'{a.out}/summary.csv', index=False)
    for name in subsets:
        x = s[s.subset == name].drop(columns='subset')
        print(f'\n=== {name}  (GDT units; loss lower = better; vs_plddt > 0 = better than pLDDT pick) ===')
        print(x.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
