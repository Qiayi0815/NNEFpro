"""One-shot driver for NNEF decoy-set evaluation.

This script wraps ``decoy_score.score_target`` and adds:

* Multi-decoy-set iteration in a single invocation (``--decoy_sets``).
* Optional target filtering (``--targets``).
* Automatic Pearson / Spearman correlation against the quality metric that
  ships in each target's ``list.csv`` (``GDT_TS`` for CASP14,
  ``RMSD`` for 3DRobot_set, auto-detected with a priority list).
* A single ``summary.csv`` covering every scored target.
* Optional per-target scatter plots and a per-decoy-set boxplot (``--plot``).
* A compare-mode (``--compare_exps``) that merges several already-written
  ``summary.csv`` files into one cross-checkpoint ``comparison.csv``.

CLI usage::

    # Score a single checkpoint on CASP14 + 3DRobot_set, with plots
    # Optional: score from a periodic snapshot without copying to model.pt:
    #   --load_checkpoint runs/v2_run_6160264/models/model_epoch_0100.pt

    python nnef/scripts/evaluate_decoys.py \
        --load_exp runs/v2_run_6160264 \
        --decoy_sets casp14,3DRobot_set \
        --exp_tag v2_run_6160264 \
        --seq_type residue --residue_type_num 20 \
        --embed_size 32 --dim 128 --n_layers 4 --attn_heads 4 \
        --mixture_r 2 --mixture_angle 3 --mixture_rama 10 \
        --smooth_gaussian --smooth_r 0.3 --smooth_angle 45 \
        --use_cart_coords --use_seq_offset \
        --out_dir eval/v2_run_6160264 --plot

    # Compare multiple checkpoints (each must already have summary.csv)
    python nnef/scripts/evaluate_decoys.py \
        --compare_exps eval/v1_pure_6160300,eval/v2_run_6160264,eval/v3_esm_6160400 \
        --out_dir eval/comparison_v1_v2_v3

The scoring-mode CLI reuses every model flag from ``options.get_decoy_parser``
so the invocation must exactly match the training-side config of the
checkpoint being loaded (``test_setup`` is shared with ``decoy_score.py``).
"""
from __future__ import annotations

import argparse
import os
import sys

print('[evaluate_decoys] importing numpy / scipy / pandas...', flush=True)
import h5py
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# Make sibling imports (``import options``) resolve the same way they do for
# ``train_chimeric.py`` and ``decoy_score.py`` when those are invoked as
# scripts instead of modules. This mirrors what the other scripts assume.
_HERE = os.path.abspath(os.path.dirname(__file__))
_NNEF_DIR = os.path.abspath(os.path.join(_HERE, '..'))
if _NNEF_DIR not in sys.path:
    sys.path.insert(0, _NNEF_DIR)

print('[evaluate_decoys] importing nnef + torch (cold start can take ~1–2 min)...', flush=True)
import options
from decoy_score import (  # noqa: E402
    load_target_list,
    score_target,
    _SUPPORTED_DECOY_SETS,
)
from utils import test_setup  # noqa: E402
from paths import ensure_dir  # noqa: E402


# --------------------------------------------------------------------------- #
# Quality-metric auto-detection                                               #
# --------------------------------------------------------------------------- #

# Columns searched for in each target's list.csv, in priority order.
# CASP{13,14} lists ship with GDT_TS. 3DRobot ships with RMSD. CASP11 ships
# with NAME only (no quality metric) -- correlations are skipped there.
_METRIC_PRIORITY = ('GDT_TS', 'TMscore', 'TM_score', 'TM', 'GDT_HA', 'LDDT', 'RMSD')


def pick_metric(df):
    for col in _METRIC_PRIORITY:
        if col in df.columns and df[col].notna().any():
            return col
    return None


def compute_correlations(df, pdb_id, decoy_set):
    """Return one summary row for a scored target."""
    metric = pick_metric(df)
    row = {
        'pdb': pdb_id,
        'decoy_set': decoy_set,
        'metric': metric,
        'n_decoys': int(len(df)),
        'pearson_r': np.nan,
        'pearson_p': np.nan,
        'spearman_r': np.nan,
        'spearman_p': np.nan,
    }
    if metric is None:
        return row
    sub = df[[metric, 'loss']].dropna()
    if len(sub) < 3:
        return row
    x = sub[metric].values.astype(float)
    y = sub['loss'].values.astype(float)
    # Guard against degenerate columns where all values are equal
    if np.std(x) == 0 or np.std(y) == 0:
        return row
    r, p = pearsonr(x, y)
    s, sp = spearmanr(x, y)
    row['pearson_r'] = float(r)
    row['pearson_p'] = float(p)
    row['spearman_r'] = float(s)
    row['spearman_p'] = float(sp)
    return row


# --------------------------------------------------------------------------- #
# Scoring mode                                                                #
# --------------------------------------------------------------------------- #

def run_scoring_mode(args):
    print('[evaluate_decoys] run_scoring_mode: building model and loading checkpoint...', flush=True)
    if args.out_dir in ('eval/default', 'eval\\default') and args.exp_tag == 'eval':
        print(
            '[evaluate_decoys] WARNING: generic --out_dir eval/default and '
            '--exp_tag=eval will reuse nnef/data/decoys/<set>/decoy_loss_eval/ '
            'and overwrite per-target CSVs. Prefer unique paths, e.g. '
            '--out_dir eval/v2_run_${JOBID}_casp14 --exp_tag v2_run_${JOBID}',
        )

    device, _model, energy_fn, _pbase = test_setup(args)

    # v3 (ESM) scoring opens the cache once and shares the handle across every
    # decoy/decoy_set. Missing cache silently falls back to baseline scoring.
    esm_h5 = None
    if getattr(args, 'use_esm', False) and getattr(args, 'esm_h5_path', None):
        if os.path.exists(args.esm_h5_path):
            esm_h5 = h5py.File(args.esm_h5_path, 'r')
            print(f'[evaluate_decoys] opened esm h5 {args.esm_h5_path} '
                  f'({len(esm_h5.keys())} entries)')
        else:
            print(f'[evaluate_decoys] --use_esm set but {args.esm_h5_path} '
                  f'missing; falling back to no-ESM scoring')

    decoy_sets = [s.strip() for s in args.decoy_sets.split(',') if s.strip()]
    target_filter = None
    if args.targets:
        target_filter = set(s.strip() for s in args.targets.split(',') if s.strip())

    ensure_dir(args.out_dir)

    rows = []
    for decoy_set in decoy_sets:
        if decoy_set not in _SUPPORTED_DECOY_SETS:
            print(f'[evaluate_decoys] skip unknown decoy_set {decoy_set!r} '
                  f'(expected one of {_SUPPORTED_DECOY_SETS})')
            continue
        try:
            pdb_list = load_target_list(decoy_set)
        except FileNotFoundError as exc:
            print(f'[evaluate_decoys] skip {decoy_set} (target list missing: {exc})')
            continue
        if target_filter is not None:
            pdb_list = [p for p in pdb_list if p in target_filter]
        if len(pdb_list) == 0:
            print(f'[evaluate_decoys] no targets to score for {decoy_set}')
            continue

        # Used by load_protein_decoy(...) which reads args.decoy_set directly.
        args.decoy_set = decoy_set
        # Per-exp subdir under nnef/data/decoys/<set>/, matches the
        # convention used by decoy_score.py (--decoy_loss_dir).
        decoy_loss_subdir = f'decoy_loss_{args.exp_tag}'

        print(f'[evaluate_decoys] scoring {len(pdb_list)} target(s) in {decoy_set} '
              f'-> decoy_loss_{args.exp_tag}/')
        for pdb_id in pdb_list:
            df = score_target(
                pdb_id, decoy_set, decoy_loss_subdir, args,
                device, energy_fn,
                trainer=None, skip_if_exists=args.skip_if_exists,
                esm_h5=esm_h5,
            )
            if df is None:
                continue
            row = compute_correlations(df, pdb_id, decoy_set)
            rows.append(row)
            print(f'  {pdb_id:12s} [{decoy_set:14s}] '
                  f'metric={row["metric"] or "-":7s} N={row["n_decoys"]:<4d} '
                  f'Pearson={row["pearson_r"]:+.4f}  Spearman={row["spearman_r"]:+.4f}')

            if args.plot:
                _plot_scatter(df, pdb_id, decoy_set, row, args.out_dir)

    if not rows:
        print('[evaluate_decoys] no targets scored.')
        return

    summary = pd.DataFrame(rows)
    summary_csv = os.path.join(args.out_dir, 'summary.csv')
    summary.to_csv(summary_csv, index=False)
    print(f'[evaluate_decoys] wrote {summary_csv} ({len(summary)} rows)')

    if args.plot and len(summary) >= 3:
        _plot_boxplot(summary, args.out_dir)

    # Short stdout digest so it's easy to grep from slurm logs
    med = summary.groupby('decoy_set')['pearson_r'].agg(['median', 'count'])
    print('[evaluate_decoys] median Pearson r per decoy_set:')
    print(med.to_string())

    if esm_h5 is not None:
        esm_h5.close()


# --------------------------------------------------------------------------- #
# Compare mode                                                                #
# --------------------------------------------------------------------------- #

def run_compare_mode(args):
    exp_paths = [s.strip() for s in args.compare_exps.split(',') if s.strip()]
    ensure_dir(args.out_dir)

    merged = None
    tags = []
    for exp_path in exp_paths:
        summary_csv = os.path.join(exp_path, 'summary.csv')
        if not os.path.exists(summary_csv):
            print(f'[evaluate_decoys] {summary_csv} missing, skip')
            continue
        df = pd.read_csv(summary_csv)
        tag = os.path.basename(os.path.normpath(exp_path)) or 'exp'
        tags.append(tag)
        df = df.rename(columns={
            'n_decoys': f'n_decoys_{tag}',
            'pearson_r': f'pearson_r_{tag}',
            'pearson_p': f'pearson_p_{tag}',
            'spearman_r': f'spearman_r_{tag}',
            'spearman_p': f'spearman_p_{tag}',
        })
        keep = ['pdb', 'decoy_set', 'metric'] + [c for c in df.columns if c.endswith(f'_{tag}')]
        df = df[keep]
        merged = df if merged is None else merged.merge(
            df, on=['pdb', 'decoy_set', 'metric'], how='outer',
        )

    if merged is None or not tags:
        print('[evaluate_decoys] nothing to compare.')
        return

    out_csv = os.path.join(args.out_dir, 'comparison.csv')
    merged.to_csv(out_csv, index=False)
    print(f'[evaluate_decoys] wrote {out_csv}')
    # Compact table: just Pearson r per exp
    cols = ['pdb', 'decoy_set', 'metric'] + [f'pearson_r_{t}' for t in tags]
    present = [c for c in cols if c in merged.columns]
    print(merged[present].to_string(index=False, float_format=lambda v: f'{v:+.4f}'))


# --------------------------------------------------------------------------- #
# Plotting helpers (lazy-imported matplotlib)                                 #
# --------------------------------------------------------------------------- #

def _plot_scatter(df, pdb_id, decoy_set, row, out_dir):
    metric = row['metric']
    if metric is None:
        return
    sub = df[[metric, 'loss']].dropna()
    if len(sub) < 2:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x = sub[metric].values.astype(float)
    y = sub['loss'].values.astype(float)
    n_plotted = len(sub)
    # Many decoys can share identical (metric, loss) → markers stack; counts explain "few dots".
    n_unique_xy = sub[[metric, 'loss']].drop_duplicates().shape[0]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, s=16, alpha=0.55, edgecolors='none')
    if len(sub) >= 2 and np.std(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        xf = np.linspace(x.min(), x.max(), 100)
        ax.plot(xf, slope * xf + intercept, color='red', lw=1.0)
    ax.set_xlabel(metric)
    ax.set_ylabel('Energy score (loss)')
    ax.set_title(
        f'{pdb_id}  ({decoy_set})  rows={row["n_decoys"]}  '
        f'plotted={n_plotted}  unique (x,y)={n_unique_xy}',
    )
    ax.text(
        0.02, 0.98,
        f'Pearson r = {row["pearson_r"]:+.3f}  p = {row["pearson_p"]:.2g}\n'
        f'Spearman ρ = {row["spearman_r"]:+.3f}  p = {row["spearman_p"]:.2g}',
        transform=ax.transAxes, va='top', fontsize=9,
    )
    plots_dir = os.path.join(out_dir, 'plots')
    ensure_dir(plots_dir)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, f'{decoy_set}_{pdb_id}_scatter.pdf'))
    plt.close(fig)


def _plot_boxplot(summary, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sets = list(summary['decoy_set'].unique())
    groups = [summary.loc[summary['decoy_set'] == s, 'pearson_r'].dropna().values
              for s in sets]
    if sum(len(g) for g in groups) == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(groups, labels=sets)
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_ylabel('Pearson r')
    ax.set_title('Per-target Pearson r by decoy set')
    plots_dir = os.path.join(out_dir, 'plots')
    ensure_dir(plots_dir)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'boxplot_pearson.pdf'))
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Argparse                                                                    #
# --------------------------------------------------------------------------- #

def build_parser():
    # Start from get_decoy_parser so scoring mode has every model flag; compare
    # mode simply ignores the model args. This avoids a second parser branch.
    parser = options.get_decoy_parser()

    group = parser.add_argument_group('evaluate_decoys')
    group.add_argument('--decoy_sets', type=str, default='casp14',
                       help='Comma-separated decoy sets to score '
                            '(choices: 3DRobot_set,casp11,casp13,casp14).')
    group.add_argument('--targets', type=str, default='',
                       help='Optional comma-separated subset of target pdb ids. '
                            'Default: every target in pdb_no_missing_residue.csv.')
    group.add_argument('--exp_tag', type=str, default='eval',
                       help='Short tag embedded into the decoy_loss subdirectory '
                            'and the eval output layout (one per checkpoint).')
    group.add_argument('--skip_if_exists', action='store_true', default=True,
                       help='Reuse previously written <pdb>_decoy_loss.csv. Default True.')
    group.add_argument('--no_skip_if_exists', action='store_false',
                       dest='skip_if_exists',
                       help='Force re-scoring even if the output csv already exists.')
    group.add_argument('--plot', action='store_true', default=False,
                       help='Produce a per-target scatter plot and a per-decoy_set boxplot.')
    group.add_argument('--out_dir', type=str, default='eval/default',
                       help='Directory for summary.csv, plots/, and comparison.csv.')
    group.add_argument('--compare_exps', type=str, default='',
                       help='If set, enter compare-mode: read summary.csv from each '
                            'listed eval dir (comma-separated) and write a combined '
                            'comparison.csv into --out_dir. Scoring is skipped.')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.compare_exps:
        run_compare_mode(args)
    else:
        run_scoring_mode(args)


if __name__ == '__main__':
    main()
