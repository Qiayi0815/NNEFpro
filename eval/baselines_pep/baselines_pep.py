#!/usr/bin/env python3
"""Trivial / classical baselines for the peptide decoy sets (pep_nmr, pep_term).

Every baseline gives one score per decoy (LOWER = better), computed on CPU from the
bead CSVs, written to ``eval/baselines_pep/<baseline>/<set>/scores.csv`` (target,NAME,score).

Baselines
---------
rama_aa       -sum_i log p(phi_i, psi_i | aa_i): 10-degree 2D histogram per amino acid from
              the TRAINING blocks (hhsuite_rama_v2_k4.h5 slot 0 = centre residue, chains of
              hhsuite_CB_v2_k4_pdb_list_holdout.csv, aa from hhsuite_pdb_seq_v2_k4.h5 row 0 at
              index group_num - 1 + seq_shift, as in data_chimeric.py), +0.5 pseudocount per
              bin. Residues with only one defined angle (psi of residue 1, phi of residue L)
              use the matching 1D marginal, so terminal perturbations are visible.
rama_generic  same, amino-acid independent histogram.
rg            Calpha radius of gyration.
neg_contacts  -(# CB pairs with |i-j| >= 3 within 8 A)   (Gly CB = CA).
rwplus        calRWplus (Zhang lab 64-bit binary) on a minimal PDB written from the beads
              (N, CA, C, ideal carbonyl O, CB for non-Gly). No side chains beyond CB, so this
              is RW+ restricted to backbone+CB atoms.
dssp_agree    1 - fraction of residues whose 3-state DSSP (pydssp, ideal O) matches the
              single-sequence PSIPRED prediction (runpsipred_single equivalent:
              seq2mtx | psipred | psipass2). Only if the psipred env exists.

Dihedral sign: the training rama h5 was built with nnef.utils._compute_phi_psi_from_backbone,
whose formula returns the NEGATIVE of the IUPAC dihedral (alpha helix at about +60/+45).
Decoy angles here are computed with an exact copy of that function, so the histograms and
decoy angles share one convention.

Metrics (``summary``) mirror eval/decoys/pep_analysis.py: per target Pearson/Spearman of
-score vs GDT_TS, top1_gdt_loss, native_rank, helix_ctrl_wins, spearman/top1 without the
artificial _ctrl_ decoys; for pep_term additionally spearman_term = Spearman(-score,
-term_rmsd). Unlike the model evaluation, no decoys are excluded (baselines score every
decoy). Subsets: all, clean (same definition as pep_analysis), len<=20, ss classes.

Usage (repo root ~/nnef, env /n/holylabs/LABS/atchan_lab/Lab/qzha/envs/pepbench)::

    python -s eval/baselines_pep/baselines_pep.py rama_hist
    python -s eval/baselines_pep/baselines_pep.py score --set pep_nmr [--targets 1L2Y] [--nproc 16]
    python -s eval/baselines_pep/baselines_pep.py summary
    python -s eval/baselines_pep/baselines_pep.py esm_vs_native   # ESMFold vs native (and raw vs normalised native)
    python -s eval/baselines_pep/baselines_pep.py score --set pep_af2 --baselines rama_aa,...,plddt,ptm
    python -s eval/baselines_pep/baselines_pep.py af2_top1        # model-selection top-1 loss on pep_af2
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import tempfile
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DECOYS = os.path.join(REPO, 'nnef', 'data', 'decoys')
OUTROOT = os.path.join(REPO, 'eval', 'baselines_pep')
D3 = '/n/holylabs/LABS/atchan_lab/Lab/qzha/nnef_data_v3'
WORK = '/n/holylabs/LABS/atchan_lab/Lab/qzha/pep_nmr_work'
HIST = os.path.join(WORK, 'baselines', 'rama_hist_k4.npz')
RWPLUS_DIR = os.path.join(WORK, 'tools', 'rwplus')
PSIPRED_ENV = '/n/holylabs/LABS/atchan_lab/Lab/qzha/envs/psipred'
AA = 'ACDEFGHIKLMNPQRSTVWY'
THREE = dict(A='ALA', C='CYS', D='ASP', E='GLU', F='PHE', G='GLY', H='HIS', I='ILE', K='LYS', L='LEU',
             M='MET', N='ASN', P='PRO', Q='GLN', R='ARG', S='SER', T='THR', V='VAL', W='TRP', Y='TYR')
NB = 36
PSEUDO = 0.5
ALL_BASELINES = ('rama_aa', 'rama_generic', 'rg', 'neg_contacts', 'rwplus', 'dssp_agree')
# pep_af2 additionally: 'plddt' (-mean pLDDT) and 'ptm' (-pTM), read from list.csv


# --------------------------------------------------------------------------- #
def phi_psi_train_convention(n, ca, c):
    """Exact copy of nnef/utils.py::_compute_phi_psi_from_backbone (radians, NaN at ends)."""
    def _d(p1, p2, p3, p4):
        b1, b2, b3 = p2 - p1, p3 - p2, p4 - p3
        b2u = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-12)
        n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
        m1 = np.cross(n1, b2u)
        return np.arctan2((m1 * n2).sum(-1), (n1 * n2).sum(-1))
    L = len(n)
    phi, psi = np.full(L, np.nan), np.full(L, np.nan)
    phi[1:] = _d(c[:-1], n[1:], ca[1:], c[1:])
    psi[:-1] = _d(n[:-1], ca[:-1], c[:-1], n[1:])
    return phi, psi


def _bin(x):
    return np.clip(((x + np.pi) / (2 * np.pi) * NB).astype(int), 0, NB - 1)


def cmd_rama_hist(a):
    import h5py
    lst = pd.read_csv(os.path.join(D3, 'hhsuite_CB_v2_k4_pdb_list_holdout.csv'))
    H2 = np.zeros((21, NB, NB))
    Hphi = np.zeros((21, NB))
    Hpsi = np.zeros((21, NB))
    n_ok = n_bad = 0
    with h5py.File(os.path.join(D3, 'hhsuite_rama_v2_k4.h5'), 'r') as hr, \
            h5py.File(os.path.join(D3, 'hhsuite_CB_v2_k4.h5'), 'r') as hc, \
            h5py.File(os.path.join(D3, 'hhsuite_pdb_seq_v2_k4.h5'), 'r') as hs:
        for i, p in enumerate(lst.pdb):
            if p not in hr or p not in hc or p not in hs:
                n_bad += 1
                continue
            rama = hr[p]['rama'][:, 0, :]
            gn = hc[p]['group_num'][:, 0]
            sh = int(hc[p]['seq_shift'][0]) if 'seq_shift' in hc[p] else 0
            seq = hs[p][0]
            idx = gn - 1 + sh
            ok = (idx >= 0) & (idx < len(seq))
            aa = np.full(len(gn), -1)
            aa[ok] = seq[idx[ok]]
            ph, ps = rama[:, 0].astype(float), rama[:, 1].astype(float)
            fp, fs = np.isfinite(ph), np.isfinite(ps)
            bp, bs = _bin(np.nan_to_num(ph)), _bin(np.nan_to_num(ps))
            v = aa >= 0
            for t in (aa, np.full(len(aa), 20)):
                m = v & fp & fs
                np.add.at(H2, (t[m], bp[m], bs[m]), 1)
                m = v & fp
                np.add.at(Hphi, (t[m], bp[m]), 1)
                m = v & fs
                np.add.at(Hpsi, (t[m], bs[m]), 1)
            n_ok += 1
            if i % 2000 == 0:
                print(f'[rama_hist] {i}/{len(lst)}', flush=True)
    os.makedirs(os.path.dirname(HIST), exist_ok=True)
    np.savez(HIST, H2=H2, Hphi=Hphi, Hpsi=Hpsi)
    print(f'[rama_hist] chains {n_ok} (missing {n_bad}); residues {H2[20].sum():.0f}; '
          f'frac phi>0 (generic) {Hphi[20, NB // 2:].sum() / Hphi[20].sum():.3f} -> {HIST}')


def load_logp():
    z = np.load(HIST)
    L2 = np.log((z['H2'] + PSEUDO) / (z['H2'] + PSEUDO).sum((1, 2), keepdims=True))
    Lphi = np.log((z['Hphi'] + PSEUDO) / (z['Hphi'] + PSEUDO).sum(1, keepdims=True))
    Lpsi = np.log((z['Hpsi'] + PSEUDO) / (z['Hpsi'] + PSEUDO).sum(1, keepdims=True))
    return L2, Lphi, Lpsi


# --------------------------------------------------------------------------- #
def xyz(df, nm):
    return df[[f'x{nm}', f'y{nm}', f'z{nm}']].values.astype(float)


def ideal_O(N, CA, C):
    O = np.zeros_like(C)
    for i in range(len(C)):
        u = C[i] - CA[i]
        u /= np.linalg.norm(u)
        if i < len(C) - 1:
            v = C[i] - N[i + 1]
            u = u + v / np.linalg.norm(v)
            u /= np.linalg.norm(u)
        O[i] = C[i] + 1.231 * u
    return O


def write_pdb(df, fn):
    N, CA, C, CB = xyz(df, 'n'), xyz(df, 'ca'), xyz(df, 'c'), xyz(df, 'cb')
    O = ideal_O(N, CA, C)
    k = 1
    with open(fn, 'w') as fh:
        for i, (g, aa) in enumerate(zip(df.group_num, df.group_name)):
            atoms = [('N', N[i]), ('CA', CA[i]), ('C', C[i]), ('O', O[i])]
            if aa != 'G':
                atoms.append(('CB', CB[i]))
            for nm, x in atoms:
                fh.write(f'ATOM  {k:5d}  {nm:<3s} {THREE[aa]} A{int(g):4d}    '
                         f'{x[0]:8.3f}{x[1]:8.3f}{x[2]:8.3f}  1.00  0.00           {nm[0]}\n')
                k += 1
        fh.write('TER\nEND\n')


_RW_LOCAL = None


def _rw_dir():
    """Per-process node-local copy of calRWplus + its 20 MB tables (NFS reads dominate)."""
    global _RW_LOCAL
    if _RW_LOCAL is None:
        import shutil
        d = tempfile.mkdtemp(prefix='rwplus_', dir=os.environ.get('TMPDIR', '/tmp'))
        for f in ('calRWplus', 'rw.dat', 'scb.dat'):
            shutil.copy2(os.path.join(RWPLUS_DIR, f), d)
        _RW_LOCAL = d
    return _RW_LOCAL


def rwplus(df, tmp):
    fn = os.path.join(tmp, 'd.pdb')
    write_pdb(df, fn)
    out = subprocess.run([os.path.join(_rw_dir(), 'calRWplus'), fn], cwd=_rw_dir(),
                         capture_output=True, text=True).stdout
    for line in out.splitlines():
        if 'RW' in line and '=' in line:
            try:
                return float(line.split('=')[-1].split()[0])
            except ValueError:
                pass
    return np.nan


def dssp3(df):
    import pydssp
    N, CA, C = xyz(df, 'n'), xyz(df, 'ca'), xyz(df, 'c')
    coord = np.stack([N, CA, C, ideal_O(N, CA, C)], 1)[None].astype(np.float32)
    return ''.join(pydssp.assign(coord, out_type='c3')[0]).replace('-', 'C')


def psipred_ss(seq, tmp):
    b = os.path.join(PSIPRED_ENV, 'bin')
    data = glob.glob(os.path.join(PSIPRED_ENV, 'share', 'psipred*', 'data'))[0]
    fa = os.path.join(tmp, 'q.fasta')
    open(fa, 'w').write(f'>q\n{seq}\n')
    mtx = os.path.join(tmp, 'q.mtx')
    with open(mtx, 'w') as fh:
        subprocess.run([os.path.join(b, 'seq2mtx'), fa], stdout=fh, check=True)
    ss = os.path.join(tmp, 'q.ss')
    with open(ss, 'w') as fh:
        subprocess.run([os.path.join(b, 'psipred'), mtx] + [os.path.join(data, f'weights.dat{x}') for x in ('', '2', '3')],
                       stdout=fh, check=True, cwd=tmp)
    subprocess.run([os.path.join(b, 'psipass2'), os.path.join(data, 'weights_p2.dat'), '1', '1.0', '1.0',
                    os.path.join(tmp, 'q.ss2'), ss], check=True, cwd=tmp, capture_output=True)
    pred = ''.join(l.split()[2] for l in open(os.path.join(tmp, 'q.ss2')) if l.strip() and l.split()[0].isdigit())
    return pred


def score_target(task):
    dset, t, baselines = task
    L2, Lphi, Lpsi = load_logp()
    tdir = os.path.join(DECOYS, dset, t)
    lst = pd.read_csv(os.path.join(tdir, 'list.csv'))
    tmp = tempfile.mkdtemp(prefix='bl_', dir=os.environ.get('TMPDIR', '/tmp'))
    out = {b: [] for b in baselines}
    pred = None
    for name in lst.NAME:
        df = pd.read_csv(os.path.join(tdir, f'{name}_bead.csv')).sort_values('group_num').reset_index(drop=True)
        if 'dssp_agree' in baselines and pred is None:
            pred = psipred_ss(''.join(df.group_name), tmp)
        N, CA, C, CB = xyz(df, 'n'), xyz(df, 'ca'), xyz(df, 'c'), xyz(df, 'cb')
        aa = np.array([AA.index(x) for x in df.group_name])
        phi, psi = phi_psi_train_convention(N, CA, C)
        both = np.isfinite(phi) & np.isfinite(psi)
        bp, bs = _bin(np.nan_to_num(phi)), _bin(np.nan_to_num(psi))
        only_psi = ~np.isfinite(phi) & np.isfinite(psi)
        only_phi = np.isfinite(phi) & ~np.isfinite(psi)
        for b in baselines:
            if b in ('rama_aa', 'rama_generic'):
                ix = aa if b == 'rama_aa' else np.full(len(aa), 20)
                s = -(L2[ix[both], bp[both], bs[both]].sum() + Lpsi[ix[only_psi], bs[only_psi]].sum()
                      + Lphi[ix[only_phi], bp[only_phi]].sum())
            elif b == 'rg':
                s = float(np.sqrt(((CA - CA.mean(0)) ** 2).sum(1).mean()))
            elif b == 'neg_contacts':
                D = np.linalg.norm(CB[:, None] - CB[None], axis=-1)
                ii, jj = np.triu_indices(len(CB), 3)
                s = -float((D[ii, jj] < 8.0).sum())
            elif b == 'rwplus':
                s = rwplus(df, tmp)
            elif b in ('plddt', 'ptm'):  # AF2 confidence (pep_af2 only); NaN for the native
                col = 'plddt_mean' if b == 'plddt' else 'ptm'
                s = -float(lst.set_index('NAME').at[name, col]) if col in lst else np.nan
            elif b == 'dssp_agree':
                obs = dssp3(df)
                s = 1.0 - np.mean([x == y for x, y in zip(obs, pred)])
            out[b].append(float(s))
    subprocess.run(['rm', '-rf', tmp])
    return t, list(lst.NAME), out


def cmd_score(a):
    dset = a.set
    baselines = [b for b in a.baselines.split(',')]
    if 'dssp_agree' in baselines and not os.path.isdir(PSIPRED_ENV):
        print('[score] psipred env missing -> skipping dssp_agree')
        baselines.remove('dssp_agree')
    targets = (a.targets.split(',') if a.targets else
               list(pd.read_csv(os.path.join(DECOYS, dset, 'pdb_no_missing_residue.csv')).pdb))
    rows = {b: [] for b in baselines}
    with Pool(a.nproc) as pool:
        for k, (t, names, out) in enumerate(pool.imap_unordered(score_target, [(dset, t, baselines) for t in targets])):
            for b in baselines:
                rows[b] += [(t, n, s) for n, s in zip(names, out[b])]
            if k % 50 == 0:
                print(f'[score] {dset} {k + 1}/{len(targets)} {t}', flush=True)
    for b in baselines:
        d = os.path.join(OUTROOT, b, dset)
        os.makedirs(d, exist_ok=True)
        fn = os.path.join(d, 'scores.csv' if not a.targets else 'scores_smoke.csv')
        pd.DataFrame(rows[b], columns=['target', 'NAME', 'score']).sort_values(['target', 'NAME']).to_csv(fn, index=False)
        print(f'[score] wrote {fn}')


# --------------------------------------------------------------------------- #
def target_metrics(sc, lst):
    d = lst.set_index('NAME').join(sc.set_index('NAME')['score']).reset_index()
    d = d[np.isfinite(d.score) & np.isfinite(d.GDT_TS)]
    if len(d) < 20:
        return None
    nm = d.NAME.to_numpy().astype(str)
    g, e = d.GDT_TS.to_numpy(), d.score.to_numpy()
    nat = np.flatnonzero(np.char.endswith(nm, '_native'))
    ctrl = np.char.find(nm, '_ctrl_') >= 0
    helix = np.flatnonzero(np.char.endswith(nm, '_ctrl_helix'))
    sp = lambda x, y: spearmanr(x, y)[0] if np.std(x) > 0 and np.std(y) > 0 else np.nan  # noqa: E731
    out = dict(n=len(d), pearson=pearsonr(-e, g)[0] if np.std(e) > 0 and np.std(g) > 0 else np.nan,
               spearman=sp(-e, g), top1_gdt_loss=float(g.max() - g[np.argmin(e)]),
               native_rank=int((e < e[nat[0]]).sum()) if len(nat) else np.nan,
               helix_ctrl_wins=bool(len(helix) and np.argmin(e) == helix[0]))
    gn, en = g[~ctrl], e[~ctrl]
    if len(gn) >= 20:
        out.update(spearman_noctrl=sp(-en, gn), top1_noctrl=float(gn.max() - gn[np.argmin(en)]))
    if 'term_rmsd' in d:
        out['spearman_term'] = sp(-e, -d.term_rmsd.to_numpy())
    return out


def cmd_summary(a):
    rows = []
    for f in sorted(glob.glob(os.path.join(OUTROOT, '*', '*', 'scores.csv'))):
        b, dset = f.split(os.sep)[-3], f.split(os.sep)[-2]
        sc = pd.read_csv(f)
        for t, g in sc.groupby('target'):
            m = target_metrics(g, pd.read_csv(os.path.join(DECOYS, dset, t, 'list.csv')))
            if m:
                rows.append(dict(baseline=b, set=dset, target=t, **m))
    df = pd.DataFrame(rows)
    meta = pd.read_csv(os.path.join(DECOYS, 'pep_nmr', 'targets_summary.csv')).set_index('pdb')
    df = df.join(meta, on='target')
    df.to_csv(os.path.join(OUTROOT, 'summary_per_target.csv'), index=False)
    sol = df['solvent_flag'].astype(str)
    subsets = {
        'all': np.ones(len(df), bool),
        'clean': (~df.homolog_in_train.astype(bool)) & (df.n_disulfides < 2) & ~df.solid_state.astype(bool)
                 & ~sol.str.contains('micelle|TFE|organic|HFIP|membrane', case=False),
        'len<=20': df.length <= 20,
        'helix': df.ss_class == 'helix', 'hairpin': df.ss_class == 'hairpin',
        'mixed': df.ss_class == 'mixed', 'coil': df.ss_class == 'coil',
    }
    agg = dict(n_targets=('target', 'size'), pearson=('pearson', 'median'), spearman=('spearman', 'median'),
               top1_gdt_loss=('top1_gdt_loss', 'median'),
               native_top1=('native_rank', lambda x: float((x == 0).mean())),
               helix_ctrl_wins=('helix_ctrl_wins', 'mean'))
    if 'spearman_noctrl' in df:
        agg.update(spearman_noctrl=('spearman_noctrl', 'median'), top1_noctrl=('top1_noctrl', 'median'))
    if 'spearman_term' in df:
        agg.update(spearman_term=('spearman_term', 'median'))
    summ = pd.concat([df[m].groupby(['set', 'baseline']).agg(**agg).assign(subset=s).reset_index()
                      for s, m in subsets.items()])
    summ.to_csv(os.path.join(OUTROOT, 'summary.csv'), index=False)
    pd.set_option('display.width', 250)
    for (dset, s), g in summ.groupby(['set', 'subset'], sort=False):
        print(f'\n=== {dset} / {s} ===')
        print(g.drop(columns=['set', 'subset']).set_index('baseline').round(3).dropna(axis=1, how='all').to_string())


def cmd_esm_vs_native(a):
    """Fraction of targets where a baseline scores ESMFold below (better than) the native,
    overall and when ESMFold is wrong (RMSD_CA > 4 A); for pep_norm also native_raw vs native."""
    rows = []
    for f in sorted(glob.glob(os.path.join(OUTROOT, '*', '*', 'scores.csv'))):
        b, dset = f.split(os.sep)[-3], f.split(os.sep)[-2]
        if dset == 'pep_term':
            continue
        sc = pd.read_csv(f).set_index('NAME').score
        for t in sorted(set(pd.read_csv(f).target)):
            lst = pd.read_csv(os.path.join(DECOYS, dset, t, 'list.csv')).set_index('NAME')
            nat, esm, raw = f'{t}_native', f'{t}_esmfold', f'{t}_native_raw'
            if nat not in sc or esm not in sc:
                continue
            r = dict(baseline=b, set=dset, target=t, esm_rmsd=lst.RMSD_CA.get(esm, np.nan),
                     esm_below_native=bool(sc[esm] < sc[nat]), d_esm_minus_native=sc[esm] - sc[nat])
            if raw in sc:
                r.update(raw_below_norm_native=bool(sc[raw] < sc[nat]), d_raw_minus_native=sc[raw] - sc[nat])
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTROOT, 'esm_vs_native_per_target.csv'), index=False)
    agg = dict(n=('target', 'size'), esm_below_native=('esm_below_native', 'mean'))
    if 'raw_below_norm_native' in df:
        agg['raw_below_norm_native'] = ('raw_below_norm_native', 'mean')
    out = pd.concat([df.groupby(['baseline', 'set']).agg(**agg).assign(subset='all'),
                     df[df.esm_rmsd > 4].groupby(['baseline', 'set']).agg(**agg).assign(subset='esm_rmsd>4')])
    out.to_csv(os.path.join(OUTROOT, 'esm_vs_native.csv'))
    print(out.round(3).to_string())


def cmd_af2_top1(a):
    """pep_af2 model selection: among the AF2 models of each target (native excluded),
    top-1 loss = best quality - quality of the lowest-score model, for quality =
    gdt_ts_max_ens (vs NMR ensemble) and GDT_TS (vs medoid). Also the fraction of targets
    where the pick is the best model, and Spearman(-score, gdt_ts_max_ens)."""
    rows = []
    for f in sorted(glob.glob(os.path.join(OUTROOT, '*', 'pep_af2', 'scores.csv'))):
        b = f.split(os.sep)[-3]
        sc = pd.read_csv(f)
        for t, g in sc.groupby('target'):
            lst = pd.read_csv(os.path.join(DECOYS, 'pep_af2', t, 'list.csv'))
            d = lst.merge(g[['NAME', 'score']], on='NAME')
            for sub, m in (('all', d.source != 'native'), ('msa', d.source == 'af2_msa'), ('drop', d.source == 'af2_drop')):
                x = d[m & np.isfinite(d.score)]
                if len(x) < 2:
                    continue
                i = x.score.idxmin()
                r = dict(baseline=b, target=t, models=sub, n=len(x),
                         loss_ens=x.gdt_ts_max_ens.max() - x.gdt_ts_max_ens[i],
                         loss_medoid=x.GDT_TS.max() - x.GDT_TS[i],
                         pick_gdt_ens=x.gdt_ts_max_ens[i], best_gdt_ens=x.gdt_ts_max_ens.max(),
                         spearman_ens=spearmanr(-x.score, x.gdt_ts_max_ens)[0] if x.score.std() > 0 and x.gdt_ts_max_ens.std() > 0 else np.nan)
                rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTROOT, 'pep_af2_top1_per_target.csv'), index=False)
    s = df.groupby(['models', 'baseline']).agg(n_targets=('target', 'size'), median_loss_ens=('loss_ens', 'median'),
                                                mean_loss_ens=('loss_ens', 'mean'), median_loss_medoid=('loss_medoid', 'median'),
                                                mean_loss_medoid=('loss_medoid', 'mean'),
                                                frac_pick_best=('loss_ens', lambda v: float((v <= 1e-6).mean())),
                                                frac_loss_gt5=('loss_ens', lambda v: float((v > 5).mean())),
                                                mean_pick_gdt_ens=('pick_gdt_ens', 'mean'),
                                                median_spearman_ens=('spearman_ens', 'median'))
    s.to_csv(os.path.join(OUTROOT, 'pep_af2_top1.csv'))
    pd.set_option('display.width', 220)
    print(s.round(3).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['rama_hist', 'score', 'summary', 'esm_vs_native', 'af2_top1'])
    ap.add_argument('--set', default='pep_nmr')
    ap.add_argument('--targets', default='')
    ap.add_argument('--baselines', default=','.join(ALL_BASELINES))
    ap.add_argument('--nproc', type=int, default=8)
    a = ap.parse_args()
    dict(rama_hist=cmd_rama_hist, score=cmd_score, summary=cmd_summary, esm_vs_native=cmd_esm_vs_native, af2_top1=cmd_af2_top1)[a.cmd](a)


if __name__ == '__main__':
    main()
