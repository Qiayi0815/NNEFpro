#!/usr/bin/env python3
"""Run 3DRobot_set decoy evaluation for checkpoints under runs/*/models/model.pt.

Usage (repo root = parent of the ``nnef`` package)::

    # Apple Silicon: default device is mps if available, else cuda, else cpu
    python nnef/scripts/batch_eval_3drobot.py

    python nnef/scripts/batch_eval_3drobot.py --device mps
    python nnef/scripts/batch_eval_3drobot.py --targets 1C5EA   # smoke

**Defaults for your workflow**

- Skips run folders whose names match ``rama`` (case-insensitive) — unfinished rama-v2 jobs.
- Includes **Yang / README ``runs/exp1``** with ``mixture_rama=0`` (``--mixture_seq 1``, etc.).
- Other runs use the modern head (``mixture_rama=10``).

v3 needs ``hhsuite_esm_v2.h5`` (``--esm_h5`` or ``NNEF_ESM_H5``); otherwise v3 is skipped.

Compare summaries::

    python nnef/scripts/evaluate_decoys.py \\
        --compare_exps eval/3dr_exp1,eval/3dr_v2_run_6160264,... \\
        --out_dir eval/3dr_comparison"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parent.parent
_REPO = _PKG.parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))


def _default_device() -> str:
    import torch

    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def _modern_common(device: str) -> list[str]:
    return [
        '--seq_type', 'residue',
        '--residue_type_num', '20',
        '--seq_len', '14',
        '--embed_size', '32',
        '--dim', '128',
        '--n_layers', '4',
        '--attn_heads', '4',
        '--mixture_r', '2',
        '--mixture_angle', '3',
        '--mixture_seq', '1',
        '--mixture_rama', '10',
        '--smooth_gaussian',
        '--smooth_r', '0.3',
        '--smooth_angle', '45',
        '--coords_angle_loss_lamda', '1',
        '--profile_loss_lamda', '10',
        '--coords_rama_loss_lamda', '1',
        '--use_position_weights',
        '--cen_seg_loss_lamda', '1',
        '--oth_seg_loss_lamda', '3',
        '--device', device,
    ]


def _yang_exp1_common(device: str) -> list[str]:
    """README / paper decoy flags (no Ramachandran mixture head)."""
    return [
        '--seq_type', 'residue',
        '--residue_type_num', '20',
        '--seq_len', '14',
        '--embed_size', '32',
        '--dim', '128',
        '--n_layers', '4',
        '--attn_heads', '4',
        '--mixture_r', '2',
        '--mixture_angle', '3',
        '--mixture_seq', '1',
        '--mixture_rama', '0',
        '--smooth_gaussian',
        '--smooth_r', '0.3',
        '--smooth_angle', '45',
        '--coords_angle_loss_lamda', '1',
        '--profile_loss_lamda', '10',
        '--radius_loss_lamda', '1',
        '--start_id_loss_lamda', '1',
        '--res_counts_loss_lamda', '1',
        '--coords_rama_loss_lamda', '0',
        '--use_position_weights',
        '--cen_seg_loss_lamda', '1',
        '--oth_seg_loss_lamda', '3',
        '--device', device,
    ]


def classify_run(name: str) -> str:
    n = name.lower()
    if n == 'exp1':
        return 'yang_exp1'
    if 'v3_full' in n or re.match(r'^v3_', n):
        return 'v3'
    if 'v2_dihedral' in n or 'v2-dihedral' in n:
        return 'v2_dihedral'
    if 'v2_run' in n or 'v2_cart' in n:
        return 'v2_cart'
    if 'v1_pure' in n or n.startswith('v1_'):
        return 'v1_pure'
    return 'unknown'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runs_dir', type=Path, default=_REPO / 'runs')
    ap.add_argument(
        '--device', type=str, default='',
        help='mps | cuda | cpu (default: auto)',
    )
    ap.add_argument('--esm_h5', type=str, default='', help='hhsuite_esm_v2.h5 for v3')
    ap.add_argument('--decoy_sets', type=str, default='3DRobot_set')
    ap.add_argument('--targets', type=str, default='')
    ap.add_argument('--plot', action='store_true')
    ap.add_argument(
        '--skip_pattern',
        type=str,
        default=r'(gpu_smoke|eval_|init_checkpoint|rama|^_|\.)',
        help='Regex; skip if match (default also skips *rama* runs)',
    )
    args = ap.parse_args()

    device = args.device.strip() or _default_device()
    print(f'[batch_3dr] device={device}')

    runs_dir: Path = args.runs_dir.resolve()
    skip_re = re.compile(args.skip_pattern, re.IGNORECASE)
    esm_path = (args.esm_h5 or os.environ.get('NNEF_ESM_H5', '')).strip()
    if not esm_path:
        from paths import data_path as _dp  # noqa: E402

        _candidate = _dp('hhsuite_esm_v2.h5')
        if Path(_candidate).is_file():
            esm_path = _candidate
    if esm_path and not Path(esm_path).is_file():
        print(f'[batch_3dr] WARN: --esm_h5 not a file ({esm_path}); v3 will be skipped.')
        esm_path = ''

    exps: list[tuple[str, str]] = []
    for models_pt in sorted(runs_dir.glob('*/models/model.pt')):
        exp = models_pt.parent.parent.name
        if skip_re.search(exp):
            print(f'[batch_3dr] skip (pattern): {exp}')
            continue
        kind = classify_run(exp)
        if kind == 'unknown':
            print(f'[batch_3dr] skip (unknown kind): {exp}')
            continue
        if kind == 'v3' and not esm_path:
            print(f'[batch_3dr] skip (no ESM h5 for v3): {exp}')
            continue
        exps.append((exp, kind))

    if not exps:
        print('[batch_3dr] no experiments to score.')
        return 1

    eval_py = _PKG / 'scripts' / 'evaluate_decoys.py'
    out_eval = _REPO / 'eval'
    out_eval.mkdir(parents=True, exist_ok=True)

    for exp, kind in exps:
        load_exp = str((runs_dir / exp).resolve())
        tag = f'3dr_{exp}'
        out_dir = out_eval / tag
        cmd = [
            sys.executable,
            str(eval_py),
            '--load_exp', load_exp,
            '--decoy_sets', args.decoy_sets,
            '--exp_tag', tag,
            '--out_dir', str(out_dir),
            '--no_skip_if_exists',
        ]
        if args.targets:
            cmd.extend(['--targets', args.targets])
        if args.plot:
            cmd.append('--plot')

        if kind == 'yang_exp1':
            cmd.extend(_yang_exp1_common(device))
        else:
            cmd.extend(_modern_common(device))

        if kind == 'v1_pure':
            pass
        elif kind == 'v2_cart':
            cmd.extend(['--use_cart_coords', '--use_seq_offset'])
        elif kind == 'v2_dihedral':
            cmd.extend(['--use_cart_coords', '--use_seq_offset', '--use_dihedral'])
        elif kind == 'v3':
            cmd.extend([
                '--use_cart_coords', '--use_seq_offset', '--use_dihedral',
                '--use_esm',
                '--esm_h5_path', esm_path,
                '--esm_dim_in', '1152',
                '--esm_dim_out', '32',
            ])

        print('\n' + '=' * 72)
        print(f'[batch_3dr] {exp}  ({kind})')
        print(' '.join(cmd))
        r = subprocess.run(cmd, cwd=str(_REPO))
        if r.returncode != 0:
            print(f'[batch_3dr] FAILED {exp} exit={r.returncode}')
        else:
            print(f'[batch_3dr] OK -> {out_dir / "summary.csv"}')

    print('\n[batch_3dr] Compare (edit to successful dirs):')
    parts = ','.join(str(out_eval / f'3dr_{exp}') for exp, _ in exps)
    print(f'  python nnef/scripts/evaluate_decoys.py --compare_exps {parts} --out_dir eval/3dr_all_compare')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
