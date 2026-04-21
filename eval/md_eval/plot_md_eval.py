"""Summarise and plot Langevin-MD evaluation outputs.

Expects the layout produced by ``fasrc/md_eval_one.sh``:

    eval/md_eval/<mode>/<run_base>/<target>_seed<N>/
        meta.json
        <target>_energy_rmsd.csv
        <target>_rmsf.csv
        <target>_trajectory_<mode>.pdb

Outputs (written alongside input root):
    summary.csv          -- one row per (mode, model, target, seed)
    plots/<mode>/<target>_energy.png      -- energy vs step, all models
    plots/<mode>/<target>_rmsd.png        -- rmsd_to_native vs step
    plots/<mode>/<target>_rmsf.png        -- per-residue RMSF
    plots/<mode>/summary_heatmap.png      -- mean final-RMSD per (model, target)

Usage:
    python eval/md_eval/plot_md_eval.py --root eval/md_eval
    python eval/md_eval/plot_md_eval.py --root eval/md_eval --mode mode2
"""
from __future__ import annotations

import argparse
import json
import os
from glob import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd


def _load_run(run_dir: str) -> dict | None:
    meta_path = os.path.join(run_dir, 'meta.json')
    if not os.path.isfile(meta_path):
        return None
    with open(meta_path) as fh:
        meta = json.load(fh)

    target = meta['target']
    energy_csv = os.path.join(run_dir, f'{target}_energy_rmsd.csv')
    rmsf_csv = os.path.join(run_dir, f'{target}_rmsf.csv')
    if not os.path.isfile(energy_csv):
        return None
    df = pd.read_csv(energy_csv)
    # Drop the native/best/init reference rows; keep only trajectory frames.
    traj = df[df['label'].str.startswith('frame_')].reset_index(drop=True)
    rmsf = None
    if os.path.isfile(rmsf_csv):
        rmsf = pd.read_csv(rmsf_csv)
    return {'meta': meta, 'traj': traj, 'rmsf': rmsf, 'refs': df.iloc[:3]}


def _walk_runs(root: str, mode_filter: str | None):
    runs = []
    mode_dirs = sorted(d for d in glob(os.path.join(root, '*')) if os.path.isdir(d))
    for mode_dir in mode_dirs:
        mode = os.path.basename(mode_dir)
        if mode_filter and mode != mode_filter:
            continue
        for model_dir in sorted(glob(os.path.join(mode_dir, '*'))):
            if not os.path.isdir(model_dir):
                continue
            run_base = os.path.basename(model_dir)
            for run_dir in sorted(glob(os.path.join(model_dir, '*_seed*'))):
                loaded = _load_run(run_dir)
                if loaded is None:
                    continue
                loaded['mode'] = mode
                loaded['run_base'] = run_base
                runs.append(loaded)
    return runs


def _run_base_to_model_key(run_base: str) -> str:
    if run_base.startswith('yang_retrain'):
        return 'yang_retrain'
    if run_base.startswith('v1_esm_rama_v2'):
        return 'v1_rama_esm'
    if run_base.startswith('v1_pure_rama_v2'):
        return 'v1_rama'
    return run_base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='eval/md_eval',
                        help='Root of the md_eval output tree')
    parser.add_argument('--mode', default=None,
                        help='Restrict to one mode dir (e.g. mode2)')
    parser.add_argument('--out_dir', default=None,
                        help='Where to put summary.csv and plots/ '
                             '(default: <root>)')
    args = parser.parse_args()

    out_dir = args.out_dir or args.root
    os.makedirs(out_dir, exist_ok=True)

    runs = _walk_runs(args.root, args.mode)
    if not runs:
        raise SystemExit(f'No md_eval runs found under {args.root!r}')

    # --- Summary CSV ---------------------------------------------------------
    rows = []
    for r in runs:
        m = r['meta']
        rows.append({
            'mode': r['mode'],
            'model_key': _run_base_to_model_key(r['run_base']),
            'run_base': r['run_base'],
            'target': m['target'],
            'seed': m['seed'],
            'n_residues': m['n_residues'],
            'L': m['L'],
            'lr': m['lr'],
            't_noise': m['t_noise'],
            'energy_native': m['energy_native'],
            'energy_init': m['energy_init'],
            'energy_best': m['energy_best'],
            'final_rmsd_to_native': m['final_rmsd_to_native'],
            'final_rmsd_to_start': m['final_rmsd_to_start'],
            'rg_native': m['rg_native'],
            'elapsed_sec': m['elapsed_sec'],
        })
    summary = pd.DataFrame(rows)
    summary_path = os.path.join(out_dir, 'summary.csv')
    summary.to_csv(summary_path, index=False)
    print(f'[plot_md_eval] wrote {summary_path} ({len(summary)} runs)')

    # --- Per-(mode, target) traces ------------------------------------------
    plots_root = os.path.join(out_dir, 'plots')
    os.makedirs(plots_root, exist_ok=True)

    mode_target_groups: dict[tuple[str, str], list[dict]] = {}
    for r in runs:
        key = (r['mode'], r['meta']['target'])
        mode_target_groups.setdefault(key, []).append(r)

    for (mode, target), grp in mode_target_groups.items():
        plot_dir = os.path.join(plots_root, mode)
        os.makedirs(plot_dir, exist_ok=True)

        fig_e, ax_e = plt.subplots(figsize=(7, 4))
        fig_r, ax_r = plt.subplots(figsize=(7, 4))
        model_colors: dict[str, str] = {}
        palette = plt.cm.tab10.colors
        for run in grp:
            model_key = _run_base_to_model_key(run['run_base'])
            if model_key not in model_colors:
                model_colors[model_key] = palette[len(model_colors) % len(palette)]
            color = model_colors[model_key]
            traj = run['traj']
            label = f"{model_key} s{run['meta']['seed']}"
            ax_e.plot(traj['step'], traj['energy'], color=color, alpha=0.7, label=label)
            ax_r.plot(traj['step'], traj['rmsd_to_native'], color=color, alpha=0.7, label=label)

        for ax, ylabel, fig, fname in [
            (ax_e, 'energy', fig_e, f'{target}_energy.png'),
            (ax_r, 'RMSD to native (Å)', fig_r, f'{target}_rmsd.png'),
        ]:
            ax.set_xlabel('MD step')
            ax.set_ylabel(ylabel)
            ax.set_title(f'{mode} / {target}')
            ax.legend(fontsize=7, ncol=2, loc='best')
            fig.tight_layout()
            fig.savefig(os.path.join(plot_dir, fname), dpi=150)
            plt.close(fig)

        # RMSF: mean across seeds per model
        fig_f, ax_f = plt.subplots(figsize=(7, 4))
        by_model: dict[str, list[np.ndarray]] = {}
        for run in grp:
            if run['rmsf'] is None:
                continue
            model_key = _run_base_to_model_key(run['run_base'])
            by_model.setdefault(model_key, []).append(run['rmsf']['rmsf'].values)
        for model_key, stacks in by_model.items():
            min_len = min(len(s) for s in stacks)
            arr = np.stack([s[:min_len] for s in stacks])
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            ax_f.plot(np.arange(min_len), mean,
                      color=model_colors.get(model_key), label=model_key)
            ax_f.fill_between(np.arange(min_len), mean - std, mean + std,
                              color=model_colors.get(model_key), alpha=0.2)
        ax_f.set_xlabel('residue')
        ax_f.set_ylabel('RMSF (Å)')
        ax_f.set_title(f'{mode} / {target} RMSF (mean ± std across seeds)')
        ax_f.legend(fontsize=8)
        fig_f.tight_layout()
        fig_f.savefig(os.path.join(plot_dir, f'{target}_rmsf.png'), dpi=150)
        plt.close(fig_f)

    # --- Cross-model heatmap per mode ---------------------------------------
    for mode in sorted(summary['mode'].unique()):
        sub = summary[summary['mode'] == mode]
        pivot = sub.groupby(['model_key', 'target'])['final_rmsd_to_native'].mean().unstack()
        if pivot.empty:
            continue
        fig, ax = plt.subplots(figsize=(1.2 * pivot.shape[1] + 2, 0.6 * pivot.shape[0] + 2))
        im = ax.imshow(pivot.values, aspect='auto', cmap='viridis')
        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(pivot.shape[0]))
        ax.set_yticklabels(pivot.index)
        ax.set_title(f'{mode}: mean final RMSD to native (Å)')
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            color='white' if val > pivot.values.mean() else 'black',
                            fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_root, mode, 'summary_heatmap.png'), dpi=150)
        plt.close(fig)
        print(f'[plot_md_eval] heatmap -> plots/{mode}/summary_heatmap.png')

    print(f'[plot_md_eval] done -> {out_dir}')


if __name__ == '__main__':
    main()
