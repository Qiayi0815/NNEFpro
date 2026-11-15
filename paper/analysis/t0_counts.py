"""Canonical T = 0 false-minimum count for Fig 1. One script, one rule, every model and set.

False minimum (same rule as restored/stats.py): final RMSD to native > 2 A AND energy_best < energy_native.
Per-position deviation: CA distance to native after Kabsch superposition of the final structure (final_bb.npy).

    python t0_counts.py --out DIR --spec exp1:peptide_dev:'eval/md_eval/pep/t0/exp1_dev/*' ...
Writes DIR/counts.csv (model,set,n_false,n_total) and DIR/per_position.csv (model,set,target,L,pos,dev).
"""
import argparse, glob, json, os
import numpy as np, pandas as pd

RMSD_THR = 2.0


def native_ca(bead):
    df = pd.read_csv(bead).sort_values('group_num')
    return df[['xca', 'yca', 'zca']].to_numpy(float)


def kabsch_dev(P, Q):
    """per-residue distance after superposing P onto Q"""
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1, 1, d]) @ Vt
    return np.linalg.norm(Pc @ R - Qc, axis=1)


ap = argparse.ArgumentParser()
ap.add_argument('--spec', nargs='+', required=True, help='model:set:glob of run dirs')
ap.add_argument('--out', required=True)
ap.add_argument('--expect', nargs='*', default=[], help='model:set:N, stop if the count of finished runs differs')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
expect = {tuple(e.split(':')[:2]): int(e.split(':')[2]) for e in a.expect}

cnt, pos, rows = [], [], []
for s in a.spec:
    model, st, g = s.split(':', 2)
    dirs = sorted(d for d in glob.glob(g) if os.path.exists(os.path.join(d, 'meta.json')))
    nf = 0
    for d in dirs:
        m = json.load(open(os.path.join(d, 'meta.json')))
        false = (m['final_rmsd_to_native'] > RMSD_THR) and (m['energy_best'] < m['energy_native']) and not m.get('diverged')
        nf += int(false)
        rows.append(dict(model=model, set=st, target=m['target'], rmsd=m['final_rmsd_to_native'],
                         dE=m['energy_best'] - m['energy_native'], diverged=m.get('diverged'), false=false))
        X = np.load(os.path.join(d, 'final_bb.npy'))  # (4, L, 3) N, CA, C, CB
        nat = native_ca(m['native_bead'])
        if len(nat) != X.shape[1]:
            raise SystemExit(f'length mismatch {d}: native {len(nat)} vs final {X.shape[1]}')
        dev = kabsch_dev(X[1].astype(float), nat)
        pos += [dict(model=model, set=st, target=m['target'], L=len(dev), pos=i, dev=float(v)) for i, v in enumerate(dev)]
    if (model, st) in expect and expect[(model, st)] != len(dirs):
        raise SystemExit(f'{model}/{st}: {len(dirs)} finished runs, expected {expect[(model, st)]}')
    cnt.append(dict(model=model, set=st, n_false=nf, n_total=len(dirs)))
    print(f'{model:8s} {st:16s} {nf:4d}/{len(dirs):4d}  ({100 * nf / max(1, len(dirs)):.1f}%)')

pd.DataFrame(cnt).to_csv(os.path.join(a.out, 'counts.csv'), index=False)
pd.DataFrame(pos).to_csv(os.path.join(a.out, 'per_position.csv'), index=False)
pd.DataFrame(rows).to_csv(os.path.join(a.out, 'per_target.csv'), index=False)
