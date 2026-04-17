"""Unpack Zhang-lab ``3DRobot_set`` bundles into ``nnef/data/decoys/3DRobot_set``.

After downloading ``3DRobot_set.tar.bz2`` from
http://zhanglab.ccmb.med.umich.edu/3DRobot/decoys and extracting the **outer**
archive, you get ``downloads/3DRobot_set/<TARGET>.tar.bz2`` (200 targets).

This script extracts each inner archive into ``<out>/<TARGET>/`` (PDBs +
``native.pdb``), converts ``list.txt`` → ``list.csv`` (NAME, RMSD) for
``decoy_score.py``, and writes ``pdb_no_missing_residue.csv``.

Usage (from repo root, ``nnef`` on ``PYTHONPATH``)::

    # Full200 targets (~several GB uncompressed; long runtime)
    python -m nnef.data_prep_scripts.unpack_zhang_3drobot_decoys \\
        --bundle_dir downloads/3DRobot_set \\
        --out nnef/data/decoys/3DRobot_set

    # Smoke: two targets only
    python -m nnef.data_prep_scripts.unpack_zhang_3drobot_decoys \\
        --bundle_dir downloads/3DRobot_set --limit 2
"""
from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path

import pandas as pd

_NNEF_DIR = Path(__file__).resolve().parent.parent
if str(_NNEF_DIR) not in sys.path:
    sys.path.insert(0, str(_NNEF_DIR))


def _target_id(archive: Path) -> str:
    name = archive.name
    if not name.endswith('.tar.bz2'):
        raise ValueError(f'expected *.tar.bz2, got {archive}')
    return name[: -len('.tar.bz2')]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        '--bundle_dir',
        type=Path,
        default=Path('downloads/3DRobot_set'),
        help='Directory containing <TARGET>.tar.bz2 from Zhang 3DRobot_set',
    )
    p.add_argument(
        '--out',
        type=Path,
        default=_NNEF_DIR / 'data' / 'decoys' / '3DRobot_set',
        help='nnef decoys layout root (default: package data/decoys/3DRobot_set)',
    )
    p.add_argument('--limit', type=int, default=None, help='Process only first N archives (debug)')
    args = p.parse_args()

    bundle_dir: Path = args.bundle_dir.expanduser().resolve()
    out: Path = args.out.expanduser().resolve()
    if not bundle_dir.is_dir():
        print(f'[unpack_3drobot] missing bundle_dir: {bundle_dir}')
        return 1

    archives = sorted(bundle_dir.glob('*.tar.bz2'))
    if args.limit is not None:
        archives = archives[: int(args.limit)]
    if not archives:
        print(f'[unpack_3drobot] no *.tar.bz2 under {bundle_dir}')
        return 1

    out.mkdir(parents=True, exist_ok=True)
    targets: list[str] = []

    for arch in archives:
        tid = _target_id(arch)
        targets.append(tid)
        dest = out / tid
        if dest.is_dir() and (dest / 'list.csv').is_file():
            print(f'[unpack_3drobot] skip {tid} (already has list.csv)')
            continue
        print(f'[unpack_3drobot] extract {tid} ...')
        with tarfile.open(arch, 'r:bz2') as tar:
            tar.extractall(out)

        list_txt = dest / 'list.txt'
        list_csv = dest / 'list.csv'
        if not list_txt.is_file():
            print(f'[unpack_3drobot] WARN {tid}: no list.txt')
            continue
        df = pd.read_csv(list_txt, sep=r'\s+')
        if 'NAME' not in df.columns or 'RMSD' not in df.columns:
            print(f'[unpack_3drobot] WARN {tid}: bad columns {list(df.columns)}')
            continue
        df.to_csv(list_csv, index=False)
        list_txt.unlink(missing_ok=True)

    pdb_csv = out / 'pdb_no_missing_residue.csv'
    pd.DataFrame({'pdb': sorted(set(targets))}).to_csv(pdb_csv, index=False)
    print(f'[unpack_3drobot] wrote {pdb_csv} ({len(set(targets))} PDBs)')
    print(
        '[unpack_3drobot] Next: regenerate v2 bead CSVs for scoring, e.g.\n'
        '  python -m nnef.data_prep_scripts.regenerate_decoy_beads \\\n'
        f'    --decoy_set 3DRobot_set --pdb_root {out} --overwrite --num_workers 8',
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
