#!/usr/bin/env python3
"""Models vs trivial baselines on the peptide sets, by DECOY TYPE.

The φ/ψ-noise ladder dominates pep_nmr and a residue-specific Ramachandran
histogram (eval/baselines_pep) ranks it about as well as the NNEF models, so a
single "all decoys" Spearman says little. This puts models (whole-decoy energy
from eval/per_residue_pep, same >=90% common-residue rule as pep_analysis.py)
and baselines (eval/baselines_pep/<b>/<set>/scores.csv, lower = better) on the
same footing and reports, per decoy subset, the median per-target Spearman of
-score vs quality:
  pep_nmr  : quality = GDT_TS; subsets realistic (native + NMR models + ESMFold),
             pert (σ ladder), noctrl (everything but the 4 artificial controls);
             plus specificity = fraction of targets whose lowest-score decoy is
             the ideal-helix control (non-helical natives only, where that is wrong)
  pep_term : quality = -term_rmsd among the rebuilt decoys (native_rebuild + σ ladder)

    python eval/decoys/pep_compare.py --models rib_k10_smooth_w10,yang_k10_l10,yang_old \
        --baselines rama_aa,rwplus,dssp_agree
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pep_analysis import decoy_energies  # noqa: E402

DEC = 'nnef/data/decoys'


def kind(name):
    n = name.split('_', 1)[1]
    if n.startswith('ctrl_'):
        return n
    for k in ('nmr', 'pert', 'term', 'esmfold', 'native_rebuild', 'ideal_rebuild', 'native'):
        if n.startswith(k):
            return k
    return n


def scores_for(entity, is_model, dset, target, cache):
    if is_model:
        f = f'eval/per_residue_pep/{entity}/{dset}/{target}.npz'
        return decoy_energies(f) if os.path.exists(f) else None
    key = (entity, dset)
    if key not in cache:
        f = f'eval/baselines_pep/{entity}/{dset}/scores.csv'
        cache[key] = pd.read_csv(f).set_index(['target', 'NAME'])['score'] if os.path.exists(f) else None
    s = cache[key]
    if s is None or target not in s.index.get_level_values(0):
        return None
    return s.loc[target]


def rho(s, q):
    d = pd.concat([s.rename('s'), q.rename('q')], axis=1).dropna()
    return spearmanr(-d.s, d.q)[0] if len(d) >= 8 and d.s.std() > 0 and d.q.std() > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='')
    ap.add_argument('--baselines', default='rama_aa,rama_generic,rwplus,dssp_agree,rg')
    ap.add_argument('--out', default='eval/decoys/pep_compare.csv')
    a = ap.parse_args()
    ents = [(m, True) for m in a.models.split(',') if m] + [(b, False) for b in a.baselines.split(',') if b]
    meta = pd.read_csv(f'{DEC}/pep_nmr/targets_summary.csv').set_index('pdb')
    rows, cache = [], {}
    for t in meta.index:
        lst = pd.read_csv(f'{DEC}/pep_nmr/{t}/list.csv').set_index('NAME')
        g = lst['GDT_TS']
        kinds = pd.Series({n: kind(n) for n in lst.index})
        term_f = f'{DEC}/pep_term/{t}/list.csv'
        tl = pd.read_csv(term_f).set_index('NAME') if os.path.exists(term_f) else None
        for ent, is_model in ents:
            r = dict(entity=ent, target=t, ss=meta.loc[t, 'ss_class'], clean=not bool(meta.loc[t, 'homolog_in_train']))
            s = scores_for(ent, is_model, 'pep_nmr', t, cache)
            if s is not None:
                s = s.reindex(lst.index)
                real = kinds.isin(['native', 'nmr', 'esmfold'])
                r['realistic'] = rho(s[real], g[real])
                r['pert'] = rho(s[kinds == 'pert'], g[kinds == 'pert'])
                r['noctrl'] = rho(s[~kinds.str.startswith('ctrl_')], g[~kinds.str.startswith('ctrl_')])
                if s.notna().any():
                    r['helix_wins'] = float(kinds[s.idxmin()] == 'ctrl_helix')
                    r['esm_vs_native'] = float(s.get(f'{t}_esmfold', np.nan) < s.get(f'{t}_native', np.nan)) \
                        if np.isfinite([s.get(f'{t}_esmfold', np.nan), s.get(f'{t}_native', np.nan)]).all() else np.nan
            if tl is not None:
                st = scores_for(ent, is_model, 'pep_term', t, cache)
                if st is not None:
                    st = st.reindex(tl.index)
                    reb = ~tl.index.str.endswith('_native')
                    r['term'] = rho(st[reb], -tl.loc[reb, 'term_rmsd'])
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(a.out.replace('.csv', '_per_target.csv'), index=False)
    pd.set_option('display.width', 220)
    nonhelix = df.ss != 'helix'
    out = []
    for name, m in [('all', np.ones(len(df), bool)), ('clean', df.clean)]:
        g = df[m].groupby('entity')
        tab = g[['realistic', 'pert', 'noctrl', 'term']].median()
        tab['helix_wins_nonhelical'] = df[m & nonhelix].groupby('entity')['helix_wins'].mean()
        tab['esmfold_below_native'] = g['esm_vs_native'].mean()
        tab['subset'] = name
        out.append(tab.reset_index())
        print(f'\n=== {name}: median per-target Spearman by decoy type (higher = better); '
              f'helix_wins_nonhelical: lower = more sequence-specific ===')
        print(tab.drop(columns='subset').round(3).to_string())
    pd.concat(out).to_csv(a.out, index=False)


if __name__ == '__main__':
    main()
