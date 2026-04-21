"""Deep-dive analysis of one md_eval run (designed for smoke inspection).

Unlike plot_md_eval.py (aggregates a whole sweep), this script drills into
a single run directory and produces the plots/numbers you need to decide
whether the Langevin sampler is behaving. Useful before launching a sweep.

Reads from <run_dir>:
    meta.json
    <target>_energy_rmsd.csv   (native/best/init refs + frame_* rows)
    <target>_rmsf.csv          (one row per residue)

Writes to <run_dir>/analysis/:
    traces.png          energy / RMSD / Rg vs step
    equilibration.png   first-half vs second-half histograms
    autocorr.png        ACF of energy & RMSD + integrated time
    distributions.png   energy & Rg histograms (single-peak check)
    rmsf_profile.png    per-residue fluctuations
    report.txt          numeric summary

Usage:
    python eval/md_eval/analyze_smoke.py \\
        --run_dir eval/md_eval_smoke/native/v1_pure_rama_v2_6228201/T1053_seed0
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd


def load_run(run_dir: str):
    with open(os.path.join(run_dir, 'meta.json')) as fh:
        meta = json.load(fh)
    target = meta['target']
    df = pd.read_csv(os.path.join(run_dir, f'{target}_energy_rmsd.csv'))
    rmsf = pd.read_csv(os.path.join(run_dir, f'{target}_rmsf.csv'))
    return meta, df, rmsf


def autocorr_fft(x: np.ndarray) -> np.ndarray:
    x = x - x.mean()
    n = len(x)
    f = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n].real
    acf /= acf[0] if acf[0] != 0 else 1.0
    return acf


def tau_int(acf: np.ndarray) -> float:
    # Sokal-style: sum until first sign change
    mask = acf > 0
    cutoff = int(np.argmax(~mask)) if (~mask).any() else len(acf)
    return 1.0 + 2.0 * acf[1:cutoff].sum()


def _frames(df: pd.DataFrame) -> pd.DataFrame:
    return df[df['label'].str.startswith('frame_')].reset_index(drop=True)


def plot_traces(df, meta, out):
    f = _frames(df)
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    axes[0].plot(f['step'], f['energy'], lw=0.8)
    axes[0].axhline(meta['energy_native'], color='k', ls='--', lw=0.7,
                    label=f"native={meta['energy_native']:.2f}")
    axes[0].set_ylabel('energy')
    axes[0].legend(fontsize=8, loc='best')
    axes[1].plot(f['step'], f['rmsd_to_native'], lw=0.8)
    axes[1].set_ylabel('RMSD to native (Å)')
    axes[2].plot(f['step'], f['rg'], lw=0.8)
    # CSV's "native" row has Rg from the aligned/centred reference; use that
    # as the visual baseline since meta.rg_native is pre-alignment.
    rg0 = df.loc[df['label'] == 'native', 'rg'].iloc[0]
    axes[2].axhline(rg0, color='k', ls='--', lw=0.7, label=f"Rg_ref={rg0:.2f}")
    axes[2].set_ylabel('Rg (Å)')
    axes[2].set_xlabel('MD step')
    axes[2].legend(fontsize=8, loc='best')
    fig.suptitle(
        f"{meta['target']} / {meta['md_mode']} / seed={meta['seed']} "
        f"(x_type={meta['x_type']}, lr={meta['lr']}, t_noise={meta['t_noise']})"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_equilibration(df, out):
    f = _frames(df)
    mid = len(f) // 2
    halves = [('first half', f.iloc[:mid]), ('second half', f.iloc[mid:])]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    for ax, col in zip(axes, ['energy', 'rmsd_to_native', 'rg']):
        for name, h in halves:
            ax.hist(h[col], bins=30, alpha=0.5,
                    label=f"{name} μ={h[col].mean():.3g}")
        ax.set_xlabel(col)
        ax.legend(fontsize=7)
    fig.suptitle('Equilibration check: first half vs second half distributions')
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_autocorr(df, out):
    f = _frames(df)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    summary = {}
    for ax, col in zip(axes, ['energy', 'rmsd_to_native']):
        acf = autocorr_fft(f[col].values)
        t = tau_int(acf)
        n_eff = len(f) / max(t, 1.0)
        summary[col] = (t, n_eff)
        show = min(len(acf), 100)
        ax.plot(np.arange(show), acf[:show])
        ax.axhline(0, color='k', lw=0.5)
        ax.set_title(f"{col}: τ_int≈{t:.1f}  N_eff≈{n_eff:.0f} / {len(f)}")
        ax.set_xlabel('lag (frames)')
        ax.set_ylabel('ACF')
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return summary


def plot_distributions(df, out):
    f = _frames(df)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    for ax, col, unit in zip(axes, ['energy', 'rg'], ['', 'Å']):
        ax.hist(f[col], bins=40)
        ax.set_xlabel(f"{col} {unit}".strip())
        ax.set_title(f"μ={f[col].mean():.3g}  σ={f[col].std():.3g}")
    fig.suptitle('Equilibrium distributions (should be single-peaked)')
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_rmsf(rmsf, meta, out):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(rmsf['residue'], rmsf['rmsf'], width=1.0)
    ax.axhline(rmsf['rmsf'].mean(), color='k', ls='--', lw=0.7,
               label=f"mean={rmsf['rmsf'].mean():.2f} Å")
    ax.set_xlabel('residue index')
    ax.set_ylabel('RMSF (Å)')
    ax.set_title(f"{meta['target']} per-residue fluctuations (L={meta['L']})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def write_report(meta, df, rmsf, acf, out):
    f = _frames(df)
    mid = len(f) // 2
    first, second = f.iloc[:mid], f.iloc[mid:]
    drift = abs(second['energy'].mean() - first['energy'].mean())
    pooled_std = f['energy'].std()
    equilibrated = drift < 0.2 * pooled_std
    lines = [
        f"md_eval analysis  —  {meta['target']} / {meta['md_mode']} / seed {meta['seed']}",
        f"run config: x_type={meta['x_type']}  lr={meta['lr']}  t_noise={meta['t_noise']}  L={meta['L']}",
        f"elapsed: {meta['elapsed_sec']:.1f} s",
        "",
        "[1] Energy relaxation",
        f"  native = {meta['energy_native']:.2f}",
        f"  init   = {meta['energy_init']:.2f}",
        f"  best   = {meta['energy_best']:.2f}  (drop from init: {meta['energy_init']-meta['energy_best']:+.2f})",
        f"  first-half  μ={first['energy'].mean():.2f}  σ={first['energy'].std():.2f}",
        f"  second-half μ={second['energy'].mean():.2f}  σ={second['energy'].std():.2f}",
        f"  equilibrated? {'YES' if equilibrated else 'NO'} "
        f"(|Δμ|={drift:.3f} vs 0.2σ={0.2*pooled_std:.3f})",
        "",
        "[2] Structural stability",
        f"  final_rmsd_to_native = {meta['final_rmsd_to_native']:.3f} Å",
        f"  Rg: traj μ={f['rg'].mean():.3f}  σ={f['rg'].std():.3f}",
        f"  RMSF: min={rmsf['rmsf'].min():.2f}  max={rmsf['rmsf'].max():.2f}  mean={rmsf['rmsf'].mean():.2f} Å",
        "",
        "[3] Sampling quality (on logged frames)",
        f"  energy: τ_int ≈ {acf['energy'][0]:.1f} frames   N_eff ≈ {acf['energy'][1]:.0f} / {len(f)}",
        f"  RMSD  : τ_int ≈ {acf['rmsd_to_native'][0]:.1f} frames   N_eff ≈ {acf['rmsd_to_native'][1]:.0f} / {len(f)}",
        f"  (frame spacing = {meta['trj_log_interval']} MD steps)",
    ]
    txt = "\n".join(lines)
    with open(out, 'w') as fh:
        fh.write(txt + "\n")
    print(txt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', required=True,
                    help='a single md_eval run dir (the one with meta.json)')
    args = ap.parse_args()

    meta, df, rmsf = load_run(args.run_dir)
    out_dir = os.path.join(args.run_dir, 'analysis')
    os.makedirs(out_dir, exist_ok=True)

    plot_traces(df, meta, os.path.join(out_dir, 'traces.png'))
    plot_equilibration(df, os.path.join(out_dir, 'equilibration.png'))
    acf = plot_autocorr(df, os.path.join(out_dir, 'autocorr.png'))
    plot_distributions(df, os.path.join(out_dir, 'distributions.png'))
    plot_rmsf(rmsf, meta, os.path.join(out_dir, 'rmsf_profile.png'))
    write_report(meta, df, rmsf, acf, os.path.join(out_dir, 'report.txt'))
    print(f"\n-> {out_dir}")


if __name__ == '__main__':
    main()
