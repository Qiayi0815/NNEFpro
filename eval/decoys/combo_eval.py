#!/usr/bin/env python3
"""Would summing two energies (ribbon frame + Yang frame) get the best of both?

Uses the whole-decoy energies already scored (eval/per_residue*/<model>/<set>/<target>.npz,
pep_analysis.decoy_energies) and evaluates, per decoy set, the combinations
  raw:  E = E_A + alpha * E_B        (usable as an energy for minimisation)
  z:    E = z(E_A) + z(E_B)          (per-target z-scores; ranking-only upper reference)
against each model alone. Metrics as in prot_analysis.py / pep_compare.py.

    python eval/decoys/combo_eval.py --a rib_k10_smooth_w10 --b yang_smooth_k10_w10
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pep_analysis import decoy_energies  # noqa: E402

DEC = 'nnef/data/decoys'


def hybrid_energies(f_term, f_int):
    """Block-typed hybrid: SUM of the interior model's centre-block energies over interior
    residues + the terminal model's centre-block energies over the first/last two residues
    (residue columns of the per_residue npz). Decoys missing any required block -> NaN."""
    zt, zi = np.load(f_term), np.load(f_int)
    Et, Ei = zt['E_cen'], zi['E_cen']
    R = Et.shape[1]
    term = np.zeros(R, bool)
    term[[0, 1, R - 2, R - 1]] = True
    ei = Ei[:, ~term]
    keep = np.isfinite(ei).mean(0) >= 0.9          # same common-residue rule as decoy_energies
    e = ei[:, keep].sum(1) + Et[:, term].sum(1)
    e[~(np.isfinite(ei[:, keep]).all(1) & np.isfinite(Et[:, term]).all(1))] = np.nan
    return pd.Series(e, index=zt['names'].astype(str))


def rama_series(tag, dset, t):
    """Per-decoy Ramachandran term of the ribbon model (decomposition scoring:
    nnef/data/decoys/<set>/decoy_loss_decomp_<tag>/<t>_decoy_loss.csv, per-block mean)."""
    f = f'{DEC}/{dset}/decoy_loss_decomp_{tag}/{t}_decoy_loss.csv'
    if not tag or not os.path.exists(f):
        return None
    d = pd.read_csv(f)
    d['NAME'] = d.NAME.astype(str).str.replace(r'\.pdb$', '', regex=True)
    return d.set_index('NAME')['loss_rama']


def load(model, dset, root):
    out = {}
    for f in glob.glob(f'{root}/{model}/{dset}/*.npz'):
        out[os.path.basename(f)[:-4]] = decoy_energies(f)
    return out


def nres(model, dset, root, t):
    f = f'{root}/{model}/{dset}/{t}.npz'
    return np.load(f)['E_cen'].shape[1] if os.path.exists(f) else None


def hyb_for(a, b, dset, root, t):
    """A = terminal (ribbon) model, B = interior (Yang-frame) model."""
    fa, fb = f'{root}/{a}/{dset}/{t}.npz', f'{root}/{b}/{dset}/{t}.npz'
    return hybrid_energies(fa, fb) if os.path.exists(fa) and os.path.exists(fb) else None


BETAS = (0.5, 1, 2)
RAMA = {}


def combos(ea, eb, alphas, hyb=None, rama=None, n_res=None):
    d = pd.concat([ea.rename('a'), eb.rename('b')], axis=1).dropna()
    res = {'A': d.a, 'B': d.b}
    if hyb is not None:
        res['hybrid'] = hyb.reindex(d.index)
        if rama is not None and n_res:
            # hybrid + beta * (Rama term on every residue): mean per block x number of residues
            for be in BETAS:
                res[f'hybrid+{be:g}rama'] = res['hybrid'] + be * rama.reindex(d.index) * n_res
    for al in alphas:
        res[f'A+{al:g}B'] = d.a + al * d.b
    z = lambda x: (x - x.mean()) / (x.std() + 1e-9)
    res['z(A)+z(B)'] = z(d.a) + z(d.b)
    return res


def kind(name):
    n = name.split('_', 1)[1]
    if n.startswith('ctrl_'):
        return n
    for k in ('nmr', 'pert', 'term', 'esmfold', 'native_rebuild', 'ideal_rebuild', 'native'):
        if n.startswith(k):
            return k
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True)
    ap.add_argument('--b', required=True)
    ap.add_argument('--alphas', default='0.5,1,2')
    ap.add_argument('--rama_tag', default='', help='decomposition tag of the terminal (ribbon) model (dir decoy_loss_decomp_<tag>), e.g. smoothw10_rib10')
    a = ap.parse_args()
    alphas = [float(x) for x in a.alphas.split(',')]
    rows = []
    for dset, q, sign in (('3DRobot_set', 'RMSD', 1), ('casp14_full', 'GDT_TS', -1)):
        EA, EB = load(a.a, dset, 'eval/per_residue'), load(a.b, dset, 'eval/per_residue')
        for t in set(EA) & set(EB):
            lst = pd.read_csv(f'{DEC}/{dset}/{t}/list.csv')
            lst['NAME'] = lst.NAME.astype(str).str.replace(r'\.pdb$', '', regex=True)
            qq = lst.set_index('NAME')[q]
            for name, e in combos(EA[t], EB[t], alphas, hyb_for(a.a, a.b, dset, 'eval/per_residue', t), rama_series(a.rama_tag, dset, t), nres(a.a, dset, 'eval/per_residue', t)).items():
                d = pd.concat([e.rename('e'), qq.rename('q')], axis=1).dropna()
                if len(d) >= 20:
                    rows.append(dict(set=dset, combo=name, target=t, metric='pearson', v=sign * pearsonr(d.e, d.q)[0]))
    meta = pd.read_csv(f'{DEC}/pep_nmr/targets_summary.csv').set_index('pdb')
    EA, EB = load(a.a, 'pep_nmr', 'eval/per_residue_pep'), load(a.b, 'pep_nmr', 'eval/per_residue_pep')
    for t in set(EA) & set(EB):
        lst = pd.read_csv(f'{DEC}/pep_nmr/{t}/list.csv').set_index('NAME')
        g = lst.GDT_TS
        for name, e in combos(EA[t], EB[t], alphas, hyb_for(a.a, a.b, 'pep_nmr', 'eval/per_residue_pep', t), rama_series(a.rama_tag, 'pep_nmr', t), nres(a.a, 'pep_nmr', 'eval/per_residue_pep', t)).items():
            e = e.dropna()
            k = pd.Series({n: kind(n) for n in e.index})
            pe = e[k == 'pert']
            if len(pe) >= 8:
                rows.append(dict(set='pep_nmr', combo=name, target=t, metric='pert_spearman', v=spearmanr(-pe, g.reindex(pe.index))[0]))
            if meta.loc[t, 'ss_class'] != 'helix' and len(e):
                rows.append(dict(set='pep_nmr', combo=name, target=t, metric='helix_wins_nonhelical', v=float(k[e.idxmin()] == 'ctrl_helix')))
    EA, EB = load(a.a, 'pep_term', 'eval/per_residue_pep'), load(a.b, 'pep_term', 'eval/per_residue_pep')
    for t in set(EA) & set(EB):
        tl = pd.read_csv(f'{DEC}/pep_term/{t}/list.csv').set_index('NAME')
        for name, e in combos(EA[t], EB[t], alphas, hyb_for(a.a, a.b, 'pep_term', 'eval/per_residue_pep', t), rama_series(a.rama_tag, 'pep_term', t), nres(a.a, 'pep_term', 'eval/per_residue_pep', t)).items():
            e = e[~e.index.str.endswith('_native')]
            d = pd.concat([e.rename('e'), tl.term_rmsd.rename('q')], axis=1).dropna()
            if len(d) >= 8 and d.e.std() > 0:
                rows.append(dict(set='pep_term', combo=name, target=t, metric='term_spearman', v=spearmanr(-d.e, -d.q)[0]))
    df = pd.DataFrame(rows)
    agg = df.groupby(['set', 'metric', 'combo']).v.agg(lambda x: x.mean() if 'wins' in x.name else x.median())
    tab = df.groupby(['set', 'metric', 'combo']).v.median().unstack('combo')
    wins = df[df.metric == 'helix_wins_nonhelical'].groupby('combo').v.mean()
    tab.loc[('pep_nmr', 'helix_wins_nonhelical'), :] = wins
    order = ['A', 'B'] + [f'A+{al:g}B' for al in alphas] + ['z(A)+z(B)', 'hybrid'] + [c for c in tab.columns if c.startswith('hybrid+')]
    pd.set_option('display.width', 220)
    print(f'A = {a.a}   B = {a.b}   (median per target; helix_wins = mean fraction, lower is better)')
    print(tab[order].round(3).to_string())


if __name__ == '__main__':
    main()
