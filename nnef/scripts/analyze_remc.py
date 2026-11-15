"""Per-temperature observables for sim_remc.py runs.

    python nnef/scripts/analyze_remc.py runs/sim_remc/<...>/traj.npz [--burn 0.2] [--ref_rmsf amber.csv]

Writes next to traj.npz:
    summary.csv   one row per temperature: kT, mean/sd Calpha RMSD to native, median Calpha RMSF,
                  <Q>, fraction of frames with RMSD < 2 A, integrated autocorrelation time of RMSD
                  (in sweeps), energy mean/sd, acceptance, swap rate to the next temperature
    rmsf.npy      (R, L) Calpha RMSF per temperature (frames superposed on the native)
    rmsd.npy      (R, F) Calpha RMSD to native per saved frame
    q.npy         (R, F) fraction of native contacts per saved frame
Q: Calpha pairs |i-j| > 3 within 8 A in the native; smooth Best-Hummer-Eaton switch
(beta 5 / A, lambda 1.2), the same definition as the FIP35 Amber analysis.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd


def kabsch_fit(P, Q):
    """Superpose P (n, 3) onto Q; return the moved P."""
    pc, qc = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    d = np.sign(np.linalg.det(U @ Vt))
    return (P - pc) @ (U @ np.diag([1, 1, d]) @ Vt) + qc


def native_contacts(ca, cut=8.0, sep=3):
    L = len(ca)
    i, j = np.triu_indices(L, k=sep + 1)
    d = np.linalg.norm(ca[i] - ca[j], axis=-1)
    m = d < cut
    return i[m], j[m], d[m]


def q_value(ca, i, j, d0, beta=5.0, lam=1.2):
    d = np.linalg.norm(ca[i] - ca[j], axis=-1)
    return float(np.mean(1.0 / (1.0 + np.exp(beta * (d - lam * d0)))))


def tau_int(x, c=5.0):
    """Integrated autocorrelation time (Sokal automatic window), in samples."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    if n < 10 or x.var() == 0:
        return float('nan')
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    acf /= acf[0]
    tau = 1.0
    for m in range(1, n):
        tau += 2 * acf[m]
        if m >= c * tau:
            break
    return float(tau)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('traj')
    ap.add_argument('--burn', type=float, default=0.2, help='fraction of production frames discarded')
    args = ap.parse_args()
    d = np.load(args.traj)
    out = os.path.dirname(args.traj)
    mp = os.path.join(out, 'meta.json')
    meta = json.load(open(mp if os.path.exists(mp) else os.path.join(out, 'meta_partial.json')))
    frames, kT, sweep = d['frames'], d['kT'], d['frame_sweep']   # (R, F, 4, L, 3)
    nat = d['native'][1].astype(float)
    R, F = frames.shape[:2]
    ci, cj, cd = native_contacts(nat)
    rmsd = np.zeros((R, F))
    q = np.zeros((R, F))
    fitted = np.zeros((R, F) + nat.shape)
    for r in range(R):
        for f in range(F):
            ca = frames[r, f, 1].astype(float)
            fc = kabsch_fit(ca, nat)
            fitted[r, f] = fc
            rmsd[r, f] = np.sqrt(((fc - nat) ** 2).sum(-1).mean())
            q[r, f] = q_value(ca, ci, cj, cd)
    b = int(args.burn * F)
    rmsf = np.zeros((R, nat.shape[0]))
    for r in range(R):
        X = fitted[r, b:]
        # iterate superposition onto the running mean so RMSF is not tied to the native
        mean = X.mean(0)
        for _ in range(3):
            X = np.stack([kabsch_fit(x, mean) for x in X])
            mean = X.mean(0)
        rmsf[r] = np.sqrt(((X - mean) ** 2).sum(-1).mean(0))
    dsw = float(np.median(np.diff(sweep))) if F > 1 else 1.0
    E = d['E']
    swap = meta.get('swap_rate', []) + [np.nan]
    rows = []
    for r in range(R):
        rows.append(dict(kT=kT[r], rmsd_mean=rmsd[r, b:].mean(), rmsd_sd=rmsd[r, b:].std(),
                         frac_rmsd_lt2=float((rmsd[r, b:] < 2.0).mean()),
                         rmsf_median=float(np.median(rmsf[r])), q_mean=q[r, b:].mean(),
                         tau_rmsd_sweeps=tau_int(rmsd[r, b:]) * dsw,
                         n_eff=(F - b) / max(tau_int(rmsd[r, b:]), 1.0),
                         E_mean=E[r, int(args.burn * E.shape[1]):].mean(),
                         E_sd=E[r, int(args.burn * E.shape[1]):].std(),
                         acc_small=meta['acceptance_small'][r], acc_shear=meta['acceptance_shear'][r],
                         swap_up=swap[r]))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, 'summary.csv'), index=False)
    np.save(os.path.join(out, 'rmsf.npy'), rmsf)
    np.save(os.path.join(out, 'rmsd.npy'), rmsd)
    np.save(os.path.join(out, 'q.npy'), q)
    pd.set_option('display.width', 200)
    print(df.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
