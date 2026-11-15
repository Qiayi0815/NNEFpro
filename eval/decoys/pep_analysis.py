#!/usr/bin/env python3
"""Peptide decoy benchmark (pep_nmr, 537 NMR peptides): whole-decoy ranking per model.

Reads eval/per_residue_pep/<model>/pep_nmr/<target>.npz (fasrc/pep_score_cpu.sh).
Decoy energy = mean of the block energies over the residues that EVERY scored
decoy of the target has a centred block for (so decoys losing blocks to the
20 Å span rule, and models with different terminal coverage, are compared on a
fixed residue set within a target). Per target and model:
  pearson / spearman of -E vs GDT_TS (higher = better), top1_gdt_loss
  (best GDT - GDT of the lowest-energy decoy), native_rank (0 = native lowest).
Subsets: all; clean (no training homolog, < 2 disulfides, no micelle/TFE/organic
solvent flag, solution NMR); length <= 20; secondary-structure classes.
Paired comparisons (Wilcoxon + bootstrap CI of the median difference) for the
pairs given with --pair A,B on the targets both models scored.

    python eval/decoys/pep_analysis.py --pair rib_k4_hard,yang_k4_mc --pair rib_k4_hard,rib_k4_termoff
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, wilcoxon

ROOT = 'nnef/data/decoys/pep_nmr'


def target_metrics(npz, lst):
    z = np.load(npz)
    names = list(z['names'])
    gdt = lst.set_index('NAME').reindex(names)['GDT_TS'].to_numpy()
    E = z['E_cen']
    scored = np.isfinite(E).any(1) & np.isfinite(gdt)
    if scored.sum() < 20:
        return None
    # residues with a centred block in >= 90% of scored decoys; decoys missing any
    # of them (e.g. extended controls whose terminal blocks exceed the 20 A span)
    # are excluded rather than shrinking every decoy's residue set
    common = np.isfinite(E[scored]).mean(0) >= 0.9
    if common.sum() == 0:
        return None
    full = scored & np.isfinite(E[:, common]).all(1)
    e = np.full(len(names), np.nan)
    e[full] = E[full][:, common].mean(1)
    ok = np.isfinite(e)
    g, e = gdt[ok], e[ok]
    nm = np.array(names)[ok].astype(str)
    nat = np.flatnonzero(np.char.endswith(nm, '_native'))
    ctrl = np.char.find(nm, '_ctrl_') >= 0
    helix = np.flatnonzero(np.char.endswith(nm, '_ctrl_helix'))
    out = dict(n=int(ok.sum()), n_excluded=int(scored.sum() - full.sum()), n_res=int(common.sum()),
               pearson=pearsonr(-e, g)[0], spearman=spearmanr(-e, g)[0],
               top1_gdt_loss=float(g.max() - g[np.argmin(e)]),
               native_rank=int((e < e[nat[0]]).sum()) if len(nat) else np.nan,
               helix_ctrl_wins=bool(len(helix) and np.argmin(e) == helix[0]),
               winner=nm[np.argmin(e)].split('_', 1)[1].split('_')[0] if '_' in nm[np.argmin(e)] else '?')
    # the same metrics without the artificial controls (ideal helix/hairpin/PPII/extended)
    gn, en = g[~ctrl], e[~ctrl]
    if len(gn) >= 20:
        out.update(spearman_noctrl=spearmanr(-en, gn)[0], top1_noctrl=float(gn.max() - gn[np.argmin(en)]))
    return out


def decoy_energies(npz):
    """name -> whole-decoy energy (same >=90% common-residue rule as target_metrics)."""
    z = np.load(npz)
    E = z['E_cen']
    scored = np.isfinite(E).any(1)
    common = np.isfinite(E[scored]).mean(0) >= 0.9
    full = scored & np.isfinite(E[:, common]).all(1)
    e = np.full(len(E), np.nan)
    e[full] = E[full][:, common].mean(1)
    return pd.Series(e, index=z['names'].astype(str))


def shared_pair(root, A, B, targets):
    """Per-target Spearman of A and B on EXACTLY the decoys both models scored."""
    out = []
    for t in targets:
        fa, fb = (os.path.join(root, m, 'pep_nmr', f'{t}.npz') for m in (A, B))
        if not (os.path.exists(fa) and os.path.exists(fb)):
            continue
        ea, eb = decoy_energies(fa), decoy_energies(fb)
        g = pd.read_csv(os.path.join(ROOT, t, 'list.csv')).set_index('NAME')['GDT_TS']
        d = pd.concat([ea.rename('a'), eb.rename('b'), g.rename('g')], axis=1).dropna()
        if len(d) >= 20:
            out.append(dict(target=t, a=spearmanr(-d.a, d.g)[0], b=spearmanr(-d.b, d.g)[0]))
    return pd.DataFrame(out).set_index('target')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='eval/per_residue_pep')
    ap.add_argument('--pair', action='append', default=[])
    ap.add_argument('--out', default='eval/decoys/pep_nmr_summary.csv')
    a = ap.parse_args()
    meta = pd.read_csv(os.path.join(ROOT, 'targets_summary.csv')).set_index('pdb')
    rows = []
    for d in sorted(glob.glob(os.path.join(a.root, '*', 'pep_nmr'))):
        model = d.split(os.sep)[-2]
        for f in sorted(glob.glob(os.path.join(d, '*.npz'))):
            t = os.path.basename(f)[:-4]
            m = target_metrics(f, pd.read_csv(os.path.join(ROOT, t, 'list.csv')))
            if m:
                rows.append(dict(model=model, target=t, **m))
    df = pd.DataFrame(rows).join(meta, on='target')
    df.to_csv(a.out.replace('.csv', '_per_target.csv'), index=False)
    sol = df['solvent_flag'].astype(str)
    subsets = {
        'all': np.ones(len(df), bool),
        'clean': (~df.homolog_in_train.astype(bool)) & (df.n_disulfides < 2) & ~df.solid_state.astype(bool)
                 & ~sol.str.contains('micelle|TFE|organic|HFIP|membrane', case=False),
        'aqueous': sol.str.fullmatch('aqueous', case=False) & ~df.solid_state.astype(bool),
        'no_disulfide': df.n_disulfides == 0,
        'disulfide_rich': df.disulfide_rich.astype(bool),
        'len<=20': df.length <= 20,
        'helix': df.ss_class == 'helix', 'hairpin': df.ss_class == 'hairpin',
        'mixed': df.ss_class == 'mixed', 'coil': df.ss_class == 'coil',
    }
    summ = []
    for sname, mask in subsets.items():
        g = df[mask].groupby('model')
        summ.append(g.agg(n_targets=('target', 'size'), excl=('n_excluded', 'median'), pearson=('pearson', 'median'),
                          spearman=('spearman', 'median'), top1_gdt_loss=('top1_gdt_loss', 'median'),
                          native_top1=('native_rank', lambda x: float((x == 0).mean())),
                          helix_ctrl_wins=('helix_ctrl_wins', 'mean'),
                          spearman_noctrl=('spearman_noctrl', 'median'), top1_noctrl=('top1_noctrl', 'median'))
                    .assign(subset=sname).reset_index())
    summ = pd.concat(summ)
    summ.to_csv(a.out, index=False)
    pd.set_option('display.width', 220)
    for sname, g in summ.groupby('subset', sort=False):
        print(f'\n=== {sname} ===')
        print(g.drop(columns='subset').set_index('model').round(3).to_string())
    # sequence clusters (MMseqs2, 40% identity): near-duplicate peptides are not
    # independent -> average within cluster, test/bootstrap over clusters
    clu_f = os.path.join(ROOT, 'seq_clusters_id40.tsv')
    clu = (pd.read_csv(clu_f, sep='\t', header=None, names=['rep', 'member']).set_index('member')['rep']
           if os.path.exists(clu_f) else None)
    tests = []
    for p in a.pair:
        A, B = p.split(',')
        sp = shared_pair(a.root, A, B, sorted(df.target.unique()))
        for sname, mask in subsets.items():
            ts = set(df[mask].target)
            x = sp[sp.index.isin(ts)].dropna()
            d = (x.a - x.b)
            if clu is not None:
                d = d.groupby(clu.reindex(d.index).fillna(pd.Series(d.index, index=d.index))).mean()
            dlt = d.to_numpy()
            if len(dlt) < 5:
                continue
            boot = np.median(np.random.default_rng(0).choice(dlt, (10000, len(dlt))), 1)
            tests.append(dict(A=A, B=B, subset=sname, n_targets=len(x), n_clusters=len(dlt),
                              median=np.median(dlt), ci_lo=np.percentile(boot, 2.5), ci_hi=np.percentile(boot, 97.5),
                              frac_A_better=np.mean(dlt > 0), p=wilcoxon(dlt).pvalue if np.any(dlt) else 1.0))
    if tests:
        t = pd.DataFrame(tests)
        # Holm correction over every (pair, subset) test reported
        order = np.argsort(t.p.to_numpy())
        adj = np.minimum(1, np.maximum.accumulate((len(t) - np.arange(len(t))) * t.p.to_numpy()[order]))
        t.loc[t.index[order], 'p_holm'] = adj
        t.to_csv(a.out.replace('.csv', '_paired.csv'), index=False)
        print('\n=== paired Spearman on shared decoys, cluster-level (40% id) Wilcoxon + cluster bootstrap, Holm ===')
        print(t.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
