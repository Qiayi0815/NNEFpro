"""Interpenetrating atoms in the structures after minimisation. Run from paper_plan/figures: python3 scripts/t0_clash.py -> nmr_bundle/t0_clash.csv
A pair counts when two backbone or Cβ atoms of residues at least two apart in sequence are closer than 2.0 Å. Same 206 peptides for both models
(t0_all/final = Ribbon, t0_all/exp1 = original NNEF, each <ID>/final_bb.npy as N, CA, C, CB); experimental conformer from pep_native."""
import glob, os, json
import numpy as np, pandas as pd
CUT = 2.0


def close(X):
    L = X.shape[1]; A = X.transpose(1, 0, 2).reshape(-1, 3); res = np.repeat(np.arange(L), X.shape[0]); D = np.linalg.norm(A[:, None] - A[None], axis=-1)
    m = np.abs(res[:, None] - res[None]) >= 2; return float(D[m].min()), int((D[m] < CUT).sum() // 2)


rows = []
for model in ('final', 'exp1'):
    for d in sorted(glob.glob(f't0_all/{model}/*')):
        T = os.path.basename(d); a, b = close(np.load(f'{d}/final_bb.npy').astype(float)); meta = json.load(open(f'{d}/meta.json'))
        nb = pd.read_csv(f'pep_native/{T}/{T}_native_bead.csv'); N = np.stack([nb[[f'x{k}', f'y{k}', f'z{k}']].values for k in ('n', 'ca', 'c', 'cb')]).astype(float); na, nn = close(N)
        rows.append(dict(model=model, target=T, dmin=a, n_close=b, exp_dmin=na, exp_n_close=nn, rmsd=meta['final_rmsd_to_native']))
r = pd.DataFrame(rows); r.to_csv('nmr_bundle/t0_clash.csv', index=False)
assert set(r[r.model == 'final'].target) == set(r[r.model == 'exp1'].target)
for m, g in r.groupby('model'):
    print(f'{m}: {len(g)} peptides; with a pair < {CUT} Å: {(g.n_close > 0).sum()} ({100 * (g.n_close > 0).mean():.0f}%); < 1.5 Å: {(g.dmin < 1.5).sum()}; median closest distance {g.dmin.median():.2f} Å; '
          f'experimental: {(g.exp_n_close > 0).sum()} with a pair < {CUT} Å, median closest {g.exp_dmin.median():.2f} Å; median RMSD {g.rmsd.median():.2f}, > 2 Å: {(g.rmsd > 2).sum()}')
