"""Summarise and plot Langevin-MD evaluation outputs.

Expects the layout produced by ``fasrc/md_eval_one.sh``:

    eval/md_eval/<sweep>/<md_mode>/<run_base>/<target>_seed<N>/
        meta.json
        <target>_energy_rmsd.csv
        <target>_rmsf.csv
        <target>_trajectory_<md_mode>.pdb

Outputs (written alongside input root):
    summary.csv          -- one row per (sweep, md_mode, model, target, seed)
    plots/<sweep>/<md_mode>/<target>_energy.png
    plots/<sweep>/<md_mode>/<target>_rmsd.png
    plots/<sweep>/<md_mode>/<target>_rg.png
    plots/<sweep>/<md_mode>/<target>_rmsf.png
    plots/<sweep>/<md_mode>/summary_heatmap.png

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
    """Find every run_dir (one meta.json each) under root.

    Layout: ``root/<sweep>/<md_mode>/<run_base>/<target_seed>/meta.json``
    where sweep is e.g. ``mode2``/``mode2_lr_ablation`` and md_mode is
    ``native``/``fold``/``decoy``. ``md_mode`` is read from meta.json so
    the walker stays robust to layout tweaks.
    """
    runs = []
    meta_paths = sorted(glob(os.path.join(root, '**', 'meta.json'), recursive=True))
    for meta_path in meta_paths:
        run_dir = os.path.dirname(meta_path)
        rel_parts = os.path.relpath(run_dir, root).split(os.sep)
        if len(rel_parts) < 2:
            continue
        sweep = rel_parts[0]
        if mode_filter and sweep != mode_filter:
            continue
        run_base = rel_parts[-2]
        loaded = _load_run(run_dir)
        if loaded is None:
            continue
        loaded['mode'] = sweep
        loaded['md_mode'] = loaded['meta'].get('md_mode', 'unknown')
        loaded['run_base'] = run_base
        runs.append(loaded)
    return runs


def _run_base_to_model_key(run_base: str) -> str:
    if run_base.startswith('yang_retrain'):
        return 'yang_retrain'
    if run_base.startswith('yang_legacy'):
        return 'yang_legacy'
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
            'md_mode': r['md_mode'],
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

    # --- Per-(mode, md_mode, target) traces ---------------------------------
    plots_root = os.path.join(out_dir, 'plots')
    os.makedirs(plots_root, exist_ok=True)

    groups: dict[tuple[str, str, str], list[dict]] = {}
    for r in runs:
        key = (r['mode'], r['md_mode'], r['meta']['target'])
        groups.setdefault(key, []).append(r)

    for (mode, md_mode, target), grp in groups.items():
        plot_dir = os.path.join(plots_root, mode, md_mode)
        os.makedirs(plot_dir, exist_ok=True)
        # Include lr in the label only when a group spans multiple lrs (ablation).
        lrs_in_group = {float(run['meta']['lr']) for run in grp}
        show_lr = len(lrs_in_group) > 1

        fig_e, ax_e = plt.subplots(figsize=(7, 4))
        fig_r, ax_r = plt.subplots(figsize=(7, 4))
        fig_g, ax_g = plt.subplots(figsize=(7, 4))
        model_colors: dict[str, str] = {}
        palette = plt.cm.tab10.colors
        for run in grp:
            model_key = _run_base_to_model_key(run['run_base'])
            if model_key not in model_colors:
                model_colors[model_key] = palette[len(model_colors) % len(palette)]
            color = model_colors[model_key]
            traj = run['traj']
            seed = run['meta']['seed']
            if show_lr:
                label = f"{model_key} lr{run['meta']['lr']:.0e} s{seed}"
            else:
                label = f"{model_key} s{seed}"
            ax_e.plot(traj['step'], traj['energy'], color=color, alpha=0.7, label=label)
            ax_r.plot(traj['step'], traj['rmsd_to_native'], color=color, alpha=0.7, label=label)
            ax_g.plot(traj['step'], traj['rg'], color=color, alpha=0.7, label=label)

        rg_native = grp[0]['meta'].get('rg_native')
        if rg_native is not None:
            ax_g.axhline(rg_native, color='k', linestyle='--', alpha=0.5,
                         label=f'native Rg={rg_native:.2f}')

        for ax, ylabel, fig, fname in [
            (ax_e, 'energy', fig_e, f'{target}_energy.png'),
            (ax_r, 'RMSD to native (Å)', fig_r, f'{target}_rmsd.png'),
            (ax_g, 'Rg (Å)', fig_g, f'{target}_rg.png'),
        ]:
            ax.set_xlabel('MD step')
            ax.set_ylabel(ylabel)
            ax.set_title(f'{mode} / {md_mode} / {target}')
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
        ax_f.set_title(f'{mode} / {md_mode} / {target} RMSF (mean ± std across seeds)')
        ax_f.legend(fontsize=8)
        fig_f.tight_layout()
        fig_f.savefig(os.path.join(plot_dir, f'{target}_rmsf.png'), dpi=150)
        plt.close(fig_f)

    # --- Cross-model heatmap per (mode, md_mode) ----------------------------
    for (mode, md_mode), sub in summary.groupby(['mode', 'md_mode']):
        pivot = sub.groupby(['model_key', 'target'])['final_rmsd_to_native'].mean().unstack()
        if pivot.empty:
            continue
        fig, ax = plt.subplots(figsize=(1.2 * pivot.shape[1] + 2, 0.6 * pivot.shape[0] + 2))
        im = ax.imshow(pivot.values, aspect='auto', cmap='viridis')
        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(pivot.shape[0]))
        ax.set_yticklabels(pivot.index)
        ax.set_title(f'{mode}/{md_mode}: mean final RMSD to native (Å)')
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            color='white' if val > pivot.values.mean() else 'black',
                            fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        out_path = os.path.join(plots_root, mode, md_mode, 'summary_heatmap.png')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f'[plot_md_eval] heatmap -> plots/{mode}/{md_mode}/summary_heatmap.png')

    print(f'[plot_md_eval] done -> {out_dir}')


if __name__ == '__main__':
    main()
