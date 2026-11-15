"""One definition of the ensemble observables for MC, null chains, comparators and Amber.

RMSF: CA, iterative superposition on the ensemble mean, fit on interior residues (2 per end
excluded), computed on all residues; interior and end residues are reported separately.
Q: CA, |i-j| > 3, d0 < 8 A, beta 5, lambda 1.2. P_ij: CA < 8 A, |i-j| > 3.
Primary score: Pearson r of interior log RMSF. Amplitude: R = median(model) / median(reference).

    python nnef/scripts/ensemble_metrics.py --kT_star runs/recb/kT_star.json --targets 2LMA,... \
        --model cb42=runs/recb/remc/t1/cb42/{t}/native --amber runs/sim_pep/amber_pep --out out.csv

Refuses to run without a frozen kT* file: temperatures are chosen before peptides are read.
A --model root may hold traj.npz (REMC, slot nearest kT*, last 50% of frames) or frames.npy (F, 4, L, 3).
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from analyze_remc import kabsch_fit, native_contacts, q_value


def rmsf(ca, trim=2, n_iter=3):
    """ca (F, L, 3) -> RMSF (L,), superposed on the mean of the interior residues."""
    L = ca.shape[1]
    idx = np.arange(trim, L - trim) if L > 2 * trim + 3 else np.arange(L)
    X = ca.astype(float)
    mean = X.mean(0)
    for _ in range(n_iter):
        out = []
        for x in X:
            cx, cm = x[idx].mean(0), mean[idx].mean(0)
            U, _, Vt = np.linalg.svd((x[idx] - cx).T @ (mean[idx] - cm))
            d = np.sign(np.linalg.det(U @ Vt))
            out.append((x - cx) @ (U @ np.diag([1, 1, d]) @ Vt) + cm)
        X = np.stack(out)
        mean = X.mean(0)
    return np.sqrt(((X - mean) ** 2).sum(-1).mean(0))


def contact_map(ca, cut=8.0, sep=3):
    d = np.linalg.norm(ca[:, None] - ca[:, :, None], axis=-1)
    i, j = np.triu_indices(ca.shape[1], k=sep + 1)
    return (d[:, i, j] < cut).mean(0)


def round_trips(walker):
    """Completed bottom-top-bottom cycles over all walkers; walker (R, S) = id at each slot."""
    R, S = walker.shape
    pos = np.zeros((walker.max() + 1, S), int)
    for s in range(S):
        pos[walker[:, s], s] = np.arange(R)
    n = 0
    for p in pos:
        state = 0
        for x in p:
            if state == 0 and x == 0:
                state = 1
            elif state == 1 and x == R - 1:
                state = 2
            elif state == 2 and x == 0:
                n += 1
                state = 1
    return n


def load_ensemble(path, kT):
    """-> ca (F, L, 3), info dict; REMC slot nearest kT, last half of the frames."""
    tp = os.path.join(path, 'traj.npz')
    if os.path.exists(tp):
        z = np.load(tp)
        if not kT and z['kT'].max() > z['kT'].min() * 2.5:  # E = 0 uses 2 replicas at kT 1-2 and has no kT*
            raise SystemExit(f'{path}: missing kT*; refusing to fall back to slot 0')
        s = int(np.argmin(np.abs(np.log(z['kT'] / kT)))) if kT else 0
        F = z['frames'].shape[1]
        ca = z['frames'][s, F // 2:, 1].astype(float)
        meta = [f for f in ('meta.json', 'meta_partial.json') if os.path.exists(os.path.join(path, f))]
        m = json.load(open(os.path.join(path, meta[0]))) if meta else {}
        return ca, dict(kT_used=float(z['kT'][s]), n_frames=len(ca), sweeps=m.get('n_sweeps_done'),
                        complete=m.get('complete'), round_trips=round_trips(z['walker']), native=z['native'][1].astype(float))
    f = np.load(os.path.join(path, 'frames.npy'))[:, 1].astype(float)
    if os.path.exists(os.path.join(path, 'frames_step.npy')):  # Langevin trajectory: same last-50% window as REMC
        f = f[len(f) // 2:]
    return f, dict(n_frames=len(f))


def position_null(L):
    """Position-only null: RMSF predicted from distance to the nearer chain end alone (log RMSF = -d)."""
    return np.exp(-np.minimum(np.arange(L), L - 1 - np.arange(L)).astype(float))


def anm_rmsf(ca, cutoff=15.0):
    """ANM null: harmonic network on native CA, unit spring constant, cutoff 15 A; RMSF up to one scale factor."""
    L = len(ca)
    H = np.zeros((3 * L, 3 * L))
    for i in range(L):
        for j in range(i + 1, L):
            d = ca[j] - ca[i]
            r2 = d @ d
            if r2 > cutoff ** 2:
                continue
            b = -np.outer(d, d) / r2
            H[3 * i:3 * i + 3, 3 * j:3 * j + 3] = b
            H[3 * j:3 * j + 3, 3 * i:3 * i + 3] = b
            H[3 * i:3 * i + 3, 3 * i:3 * i + 3] -= b
            H[3 * j:3 * j + 3, 3 * j:3 * j + 3] -= b
    w, v = np.linalg.eigh(H)
    keep = w > 1e-6 * w.max()  # drops the six rigid-body modes
    c = (v[:, keep] ** 2 / w[keep]).sum(1)
    return np.sqrt(c.reshape(L, 3).sum(1))


def rmsd_to(ca, ref):
    """Kabsch CA RMSD of each frame (F, L, 3) to ref (L, 3)."""
    r0 = ref - ref.mean(0)
    out = []
    for x in ca:
        y = x - x.mean(0)
        u, _, vt = np.linalg.svd(y.T @ r0)
        out.append(np.sqrt(((y @ (u @ np.diag([1, 1, np.sign(np.linalg.det(u @ vt))]) @ vt) - r0) ** 2).sum(1).mean()))
    return np.array(out)


RMSD_BAND = (0.5, 2.0)  # Fig 3 gate: median model RMSD to native / median Amber RMSD; set before REMC is read
CEILING_MIN = 0.5  # peptide enters the primary analysis only if Amber agrees with itself; set before REMC is read


def amber(root, t):
    """Pool seeds (frame 0 = native, dropped). half: first vs second half within a seed (diagnostic only)."""
    ens, half, native = [], [], None
    for s in sorted(glob.glob(f'{root}/{t}/s*/frames.npy')):
        if not os.path.exists(os.path.join(os.path.dirname(s), 'meta.json')):  # frames.npy is written every 5 ns
            continue
        f = np.load(s)
        ca = f[1:, 1].astype(float)
        if len(ca) < 100:
            continue
        native = f[0, 1].astype(float)
        ens.append(ca)
        half.append((rmsf(ca[: len(ca) // 2]), rmsf(ca[len(ca) // 2:])))
    return (np.concatenate(ens), half, ens, native) if ens else (None, [], [], None)


def seed_ceiling(ens):
    """Leave-one-seed-out ceiling: median r of each seed against the pool of the others; nan with one seed."""
    if len(ens) < 2:
        return float('nan')
    return float(np.median([compare(rmsf(e), rmsf(np.concatenate(ens[:k] + ens[k + 1:])))['r_log_rmsf']
                            for k, e in enumerate(ens)]))


def compare(a, b, trim=2):
    i = np.arange(trim, len(a) - trim)
    la, lb = np.log(np.maximum(a[i], 1e-3)), np.log(np.maximum(b[i], 1e-3))
    return dict(r_log_rmsf=float(np.corrcoef(la, lb)[0, 1]), R_amp=float(np.median(a[i]) / np.median(b[i])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kT_star', required=True, help='json {model: kT*}, frozen before peptides are read')
    ap.add_argument('--targets', required=True)
    ap.add_argument('--model', action='append', default=[], help='name=path with {t}')
    ap.add_argument('--amber', default='runs/sim_pep/amber_pep')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    kts = json.load(open(a.kT_star))
    rows = []
    for t in a.targets.split(','):
        ref, half, seeds, nat0 = amber(a.amber, t)
        if ref is None:
            continue
        rr, pr = rmsf(ref), contact_map(ref)
        nseed, ceil = len(seeds), seed_ceiling(seeds)
        within = float(np.median([compare(x, y)['r_log_rmsf'] for x, y in half]))
        amber_rmsd = float(np.median(rmsd_to(ref, nat0)))
        L0 = len(rr)
        for nm, pred in (('null_position', position_null(L0)), ('null_anm', anm_rmsf(nat0))):  # shape only: no scale
            rows.append(dict(target=t, model=nm, n_amber_seeds=nseed, ceiling_r=ceil, within_seed_r=within,
                             primary=bool(ceil >= CEILING_MIN), r_log_rmsf=compare(pred, rr)['r_log_rmsf']))
        for spec in a.model:
            name, path = spec.split('=')
            path = path.format(t=t)
            if not (os.path.exists(os.path.join(path, 'traj.npz')) or os.path.exists(os.path.join(path, 'frames.npy'))):
                continue
            ca, info = load_ensemble(path, kts.get(name.split('.')[0]))
            if ca.shape[1] != len(rr):
                continue
            pm = contact_map(ca)
            row = dict(target=t, model=name, n_amber_seeds=nseed, ceiling_r=ceil, within_seed_r=within, primary=bool(ceil >= CEILING_MIN),
                       **compare(rmsf(ca), rr),
                       P_r=float(np.corrcoef(pm, pr)[0, 1]) if pm.std() and pr.std() else np.nan, P_mae=float(np.abs(pm - pr).mean()),
                       end_ratio=float(np.median(rmsf(ca)[[0, 1, -2, -1]]) / np.median(rr[[0, 1, -2, -1]])))
            row['rmsd_ratio'] = float(np.median(rmsd_to(ca, nat0))) / amber_rmsd
            row['in_rmsd_band'] = bool(RMSD_BAND[0] <= row['rmsd_ratio'] <= RMSD_BAND[1])
            nat = info.pop('native', None)
            row.update(info)
            if nat is not None:
                i, j, d0 = native_contacts(nat)
                row['Q'] = float(np.mean([q_value(c, i, j, d0) for c in ca])) if len(i) else np.nan
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print(df.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
