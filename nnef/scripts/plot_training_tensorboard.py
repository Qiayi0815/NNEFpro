#!/usr/bin/env python3
"""Plot NNEF training curves from TensorBoard ``events.out.tfevents*`` files.

Reads scalar tags logged by ``LocalGenTrainer`` (``Train/total_loss``, etc.),
draws:

* **Default:** one figure with **total loss** for all runs overlaid, plus one multi-panel figure with **each loss component**, all runs per panel.
* With ``--separate``: **one PDF per run** — ``train_total_loss_<label>.pdf`` and
  ``train_losses_by_component_<label>.pdf`` — no overlapping colors.

Scalars are logged with ``global_step = epoch * len(dataloader) + iter`` (see
``LocalGenTrainer``). Use ``--x_axis epoch`` (default) to plot against **epoch**
(``step / steps_per_epoch``); use ``--x_axis step`` for the raw TensorBoard step.

Usage::

    # Overlay — x-axis = epoch (auto-infer steps/epoch from step pattern, or set explicitly)
    python nnef/scripts/plot_training_tensorboard.py \\
        --runs runs/v1_pure_6171704 runs/v2_run_6160264 runs/v2_dihedral_6172503 \\
        --out_dir figures/training_compare

    # Match FASRC train slurms: batch_size=1000, total_num_samples=1e6 → 1000 iters/epoch
    python nnef/scripts/plot_training_tensorboard.py --runs ... --steps_per_epoch 1000

    # Different dataloader lengths per run (same order as --runs)
    python nnef/scripts/plot_training_tensorboard.py \\
        --runs runs/a runs/b --steps_per_epoch_list 1000 950

    # One figure per run (recommended if curves look hidden)
    python nnef/scripts/plot_training_tensorboard.py --separate \\
        --runs ... --labels v1_pure v2_cart_offset v2_dihedral \\
        --out_dir figures/training_each

Requires: ``pip install tensorboard matplotlib``
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np

# Sibling imports when run as script from repo root
_HERE = os.path.abspath(os.path.dirname(__file__))
_NNEF_DIR = os.path.abspath(os.path.join(_HERE, '..'))
if _NNEF_DIR not in sys.path:
    sys.path.insert(0, _NNEF_DIR)

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError as exc:
    raise SystemExit(
        'Need tensorboard: pip install tensorboard matplotlib\n'
        f'Original error: {exc}',
    ) from exc


TRAIN_TAGS = [
    'Train/total_loss',
    'Train/profile_loss',
    'Train/coords_radius_loss',
    'Train/coords_angle_loss',
    'Train/coords_rama_loss',
    'Train/start_id_loss',
    'Train/res_counts_loss',
]


def _pick_total_loss_tag(data: Dict[str, List[Tuple[int, float]]]) -> str | None:
    """Exact tag first, then any Train* total_loss (handles renames / typos)."""
    if 'Train/total_loss' in data and data['Train/total_loss']:
        return 'Train/total_loss'
    candidates = [
        k for k, v in data.items()
        if 'total_loss' in k and k.startswith('Train') and v
    ]
    return min(candidates) if candidates else None


def _align_train_tag(
    data: Dict[str, List[Tuple[int, float]]],
    canonical: str,
) -> str | None:
    """Map canonical tag name to what exists in this run (prefix must be Train/)."""
    if canonical in data and data[canonical]:
        return canonical
    suffix = canonical.split('/')[-1] if '/' in canonical else canonical
    matches = [k for k, v in data.items() if k.endswith('/' + suffix) and v]
    if len(matches) == 1:
        return matches[0]
    matches = [k for k, v in data.items() if k.endswith(suffix) and v]
    return matches[0] if len(matches) == 1 else None


def _slug_label(label: str) -> str:
    """Filesystem-safe stem from legend / run name."""
    s = re.sub(r'[^\w\-.]+', '_', label.strip())
    s = re.sub(r'_+', '_', s).strip('_')
    return (s[:120] if s else 'run')


def _load_scalars(logdir: str) -> Dict[str, List[Tuple[int, float]]]:
    """tag -> [(step, value), ...] sorted by step."""
    ea = EventAccumulator(logdir, size_guidance={'scalars': 0})
    ea.Reload()
    if 'scalars' not in ea.Tags():
        return {}
    out: Dict[str, List[Tuple[int, float]]] = {}
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        out[tag] = [(e.step, float(e.value)) for e in events]
    return out


def infer_steps_per_epoch(steps: Sequence[int]) -> int:
    """Infer ``len(dataloader)`` from logged global_step pattern.

    Trainer logs at ``i % log_freq == 0`` with ``global_step = epoch * L + i``.
    So every step ``s`` satisfies ``(s % L) % log_int == 0`` for log interval
    ``log_int`` and epoch length ``L``.
    """
    u = np.sort(np.unique(np.asarray(list(steps), dtype=np.int64)))
    if len(u) < 2:
        return max(1, int(u[0]) + 1 if len(u) else 1)
    d = np.diff(u)
    pos = d[d > 0]
    if len(pos) == 0:
        return 1
    log_int = int(max(1, round(float(np.median(pos)))))
    max_u = int(u[-1])
    for L in range(log_int, max_u + 2):
        ok = True
        for s in u:
            rem = int(s % L)
            if rem % log_int != 0:
                ok = False
                break
        if ok:
            return L
    return max(1, int(max_u) // 200)


def _steps_to_epochs(
    pairs: List[Tuple[int, float]],
    steps_per_epoch: int,
) -> List[Tuple[float, float]]:
    spe = max(1, int(steps_per_epoch))
    return [(float(s) / spe, v) for s, v in pairs]


def _x_label(x_axis: str) -> str:
    return 'Epoch' if x_axis == 'epoch' else 'Global step'


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        '--runs', nargs='+', required=True,
        help='Run directories containing events.out.tfevents* (e.g. runs/v1_pure_JOBID).',
    )
    p.add_argument(
        '--labels', nargs='*', default=None,
        help='Legend labels (same length as --runs). Default: directory basename.',
    )
    p.add_argument('--out_dir', type=str, default='figures/training_compare',
                   help='Output directory for PDFs.')
    p.add_argument('--verbose', action='store_true',
                   help='Print tags and point counts per run (debug missing curves).')
    p.add_argument('--separate', action='store_true',
                   help='Write one total-loss PDF and one component-grid PDF per run '
                        '(no overlay / no combined multi-run figure).')
    p.add_argument(
        '--x_axis', choices=('epoch', 'step'), default='epoch',
        help='Horizontal axis: epoch (= global_step / steps_per_epoch) or raw TB step.',
    )
    p.add_argument(
        '--steps_per_epoch', type=int, default=None,
        help='Dataloader length (iterations per epoch). Default: infer per run, '
             'or use 1000 when inference fails (batch 1000 × 1M samples in train slurms).',
    )
    p.add_argument(
        '--steps_per_epoch_list', nargs='*', type=int, default=None,
        help='One integer per --runs when lengths differ (same order as --runs).',
    )
    args = p.parse_args()

    if args.labels is not None and len(args.labels) != len(args.runs):
        p.error('--labels must have the same length as --runs')
    if args.steps_per_epoch_list is not None:
        if len(args.steps_per_epoch_list) != len(args.runs):
            p.error('--steps_per_epoch_list must have the same length as --runs')

    os.makedirs(args.out_dir, exist_ok=True)

    labels: List[str] = list(args.labels) if args.labels else [
        os.path.basename(os.path.normpath(r)) for r in args.runs
    ]

    series_per_run: List[Dict[str, List[Tuple[int, float]]]] = []
    for rd in args.runs:
        if not os.path.isdir(rd):
            raise SystemExit(f'Not a directory: {rd}')
        # Helpful hint when user synced only models/ but forgot tfevents
        tfev = [
            f for f in os.listdir(rd)
            if f.startswith('events.out.tfevents')
        ]
        if not tfev:
            print(
                f'[warn] {rd!r} has no events.out.tfevents* in the top-level '
                f'directory — rsync the full run dir from the cluster.',
                file=sys.stderr,
            )
        data = _load_scalars(rd)
        if not data:
            raise SystemExit(f'No scalar events in {rd!r} (missing tfevents?)')
        series_per_run.append(data)
        if args.verbose:
            tags = sorted(data.keys())
            tl = _pick_total_loss_tag(data)
            n_tl = len(data[tl]) if tl else 0
            print(
                f'[plot_training] {rd}: {len(tags)} tags, '
                f'Train/total_loss via {tl!r} -> {n_tl} points',
                file=sys.stderr,
            )
            print(
                f'  tags: {tags[:20]}{" ..." if len(tags) > 20 else ""}',
                file=sys.stderr,
            )

    # Steps per epoch for epoch x-axis (LocalGenTrainer: global_step = epoch * L + i).
    spe_list: List[int] = []
    for i, data in enumerate(series_per_run):
        if args.x_axis == 'step':
            spe_list.append(1)
            continue
        if args.steps_per_epoch_list is not None:
            spe_list.append(max(1, int(args.steps_per_epoch_list[i])))
            continue
        if args.steps_per_epoch is not None:
            spe_list.append(max(1, int(args.steps_per_epoch)))
            continue
        tag = _pick_total_loss_tag(data)
        steps_src = [s for s, _ in data[tag]] if tag else []
        if not steps_src:
            for _k, v in data.items():
                if v:
                    steps_src = [s for s, _ in v]
                    break
        inferred = infer_steps_per_epoch(steps_src) if steps_src else 1000
        if inferred <= 1 and steps_src and max(steps_src) > 5000:
            inferred = 1000
            if args.verbose:
                print(
                    f'[plot_training] {labels[i]}: inference weak; using fallback '
                    f'steps_per_epoch={inferred}',
                    file=sys.stderr,
                )
        spe_list.append(max(1, inferred))
        auto_spe = args.steps_per_epoch is None and args.steps_per_epoch_list is None
        if args.verbose or (auto_spe and args.x_axis == 'epoch'):
            print(
                f'[plot_training] {labels[i]}: steps_per_epoch={spe_list[-1]} '
                f'(x_axis={args.x_axis})',
                file=sys.stderr,
            )

    def _xy_from_pairs(
        pairs: List[Tuple[int, float]],
        spe: int,
    ) -> Tuple[List[float], List[float]]:
        if not pairs:
            return [], []
        if args.x_axis == 'step':
            xs, ys = zip(*pairs)
            return [float(x) for x in xs], list(ys)
        ep = _steps_to_epochs(pairs, spe)
        xs, ys = zip(*ep)
        return list(xs), list(ys)

    xl = _x_label(args.x_axis)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    if args.separate:
        for label, data, spe in zip(labels, series_per_run, spe_list):
            slug = _slug_label(label)
            tag = _pick_total_loss_tag(data)
            if tag:
                fig, ax = plt.subplots(figsize=(8, 5))
                xs, ys = _xy_from_pairs(data[tag], spe)
                ax.plot(xs, ys, color='C0', lw=2.0)
                ax.set_xlabel(xl)
                ax.set_ylabel(tag)
                ax.set_title(f'Total training loss — {label}')
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                out = os.path.join(args.out_dir, f'train_total_loss_{slug}.pdf')
                fig.savefig(out)
                plt.close(fig)
                print(f'Wrote {out}')
            else:
                print(
                    f'[warn] separate: skip total_loss for {label!r} (no tag)',
                    file=sys.stderr,
                )

            tags_one = [
                t for t in TRAIN_TAGS
                if _align_train_tag(data, t)
            ]
            if not tags_one:
                continue
            ncols = 2
            nrows = (len(tags_one) + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.2 * nrows), squeeze=False)
            for idx, ctag in enumerate(tags_one):
                r, c = divmod(idx, ncols)
                ax = axes[r][c]
                real = _align_train_tag(data, ctag)
                assert real is not None
                xs, ys = _xy_from_pairs(data[real], spe)
                ax.plot(xs, ys, color='C0', lw=1.3)
                ax.set_xlabel(xl)
                ax.set_ylabel(ctag.split('/')[-1])
                ax.set_title(real if real == ctag else f'{ctag}\n({real})')
                ax.grid(True, alpha=0.3)
            for j in range(len(tags_one), nrows * ncols):
                r, c = divmod(j, ncols)
                axes[r][c].set_visible(False)
            fig.suptitle(f'Training losses by component — {label}', y=1.02, fontsize=12)
            fig.tight_layout()
            out_g = os.path.join(args.out_dir, f'train_losses_by_component_{slug}.pdf')
            fig.savefig(out_g, bbox_inches='tight')
            plt.close(fig)
            print(f'Wrote {out_g}')
        return

    # --- Combined: total loss only ---
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = 0
    for i, (label, data, spe) in enumerate(zip(labels, series_per_run, spe_list)):
        tag = _pick_total_loss_tag(data)
        if not tag:
            print(
                f'[warn] skip overlay for {label!r}: no Train/*total_loss in run '
                f'(tags sample: {list(data.keys())[:8]})',
                file=sys.stderr,
            )
            continue
        xs, ys = _xy_from_pairs(data[tag], spe)
        color = colors[i % len(colors)]
        ax.plot(xs, ys, label=f'{label} ({tag})', lw=1.8, alpha=0.9, color=color)
        plotted += 1
    if plotted == 0:
        raise SystemExit('No Train/total_loss series found in any run; check --runs and tfevents.')
    if plotted < len(labels):
        print(
            f'[warn] overlay only has {plotted}/{len(labels)} curves — '
            f're-sync runs that are missing events.out.tfevents*',
            file=sys.stderr,
        )
    ax.set_xlabel(xl)
    ax.set_ylabel('Train / total_loss')
    ax.set_title('Total training loss (overlay)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p_combined = os.path.join(args.out_dir, 'train_total_loss_overlay.pdf')
    fig.savefig(p_combined)
    plt.close(fig)
    print(f'Wrote {p_combined}')

    # --- Separate panels: each tag, all runs ---
    tags_to_plot = [
        t for t in TRAIN_TAGS
        if any(_align_train_tag(d, t) for d in series_per_run)
    ]
    n = len(tags_to_plot)
    if n == 0:
        return
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.2 * nrows), squeeze=False)
    for idx, tag in enumerate(tags_to_plot):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        for i, (label, data, spe) in enumerate(zip(labels, series_per_run, spe_list)):
            real = _align_train_tag(data, tag)
            if not real:
                if args.verbose:
                    print(
                        f'[warn] {label}: missing {tag} (no single fuzzy match)',
                        file=sys.stderr,
                    )
                continue
            xs, ys = _xy_from_pairs(data[real], spe)
            color = colors[i % len(colors)]
            lbl = label if real == tag else f'{label} [{real}]'
            ax.plot(xs, ys, label=lbl, lw=1.3, alpha=0.9, color=color)
        ax.set_xlabel(xl)
        ax.set_ylabel(tag.split('/')[-1])
        ax.set_title(tag)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    for j in range(len(tags_to_plot), nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)
    fig.suptitle('Training losses by component', y=1.02, fontsize=12)
    fig.tight_layout()
    p_grid = os.path.join(args.out_dir, 'train_losses_by_component.pdf')
    fig.savefig(p_grid, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {p_grid}')


if __name__ == '__main__':
    main()
