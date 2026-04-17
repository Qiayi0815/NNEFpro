#!/usr/bin/env python3
"""Regenerate decoy ``*_bead.csv`` files using thesis / v2 backbone extraction.

The shipped evaluation sets (CASP14, 3DRobot, …) store one bead CSV per decoy
under ``nnef/data/decoys/<decoy_set>/<target_id>/``. This script overwrites
those CSVs by parsing the corresponding **PDB** with``local_extractor_v2.extract_beads_v2``, which emits the canonical columns::

 chain_id, group_num, group_name,
    xn,yn,zn, xca,yca,zca, xc,yc,zc, xcb,ycb,zcb

That layout matches training HDF5 extraction and supplies N/C for phi/psi in
``utils.load_protein_decoy`` when ``--use_dihedral`` is enabled.

**This repository does not contain decoy PDBs** — only pre-built bead CSVs.
You must download or mirror the raw models locally and pass ``--pdb_root``.

Typical PDB layout (default search)::

    {pdb_root}/{target_id}/{decoy_stem}.pdb

where ``decoy_stem`` is the ``NAME`` field from each target's ``list.csv``
(stripping a trailing ``.pdb`` for CASP13/14 manifests that omit the extension).

Examples
--------
Dry-run (no writes)::

    python -m nnef.data_prep_scripts.regenerate_decoy_beads \\
        --decoy_set casp14 \\
        --pdb_root /data/casp14/models \\
        --targets T1053 \\
        --dry_run

Regenerate every target in the set (slow; use ``--num_workers``)::

    python -m nnef.data_prep_scripts.regenerate_decoy_beads \\
        --decoy_set 3DRobot_set \\
        --pdb_root /data/3DRobot/pdbs \\
        --overwrite \\
        --num_workers 8
"""
from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

# Resolve nnef/ on sys.path when invoked as ``python -m ...``.
_NNEF_DIR = Path(__file__).resolve().parent.parent
if str(_NNEF_DIR) not in sys.path:
    sys.path.insert(0, str(_NNEF_DIR))

from paths import data_path  # noqa: E402
from data_prep_scripts.local_extractor_v2 import extract_beads_v2  # noqa: E402

_SUPPORTED = ('3DRobot_set', 'casp11', 'casp13', 'casp14')


def _load_target_ids(decoy_set: str) -> List[str]:
    if decoy_set == '3DRobot_set':
        csv_path = data_path('decoys', decoy_set, 'pdb_no_missing_residue.csv')
        return pd.read_csv(csv_path)['pdb'].astype(str).tolist()
    if decoy_set == 'casp11':
        txt_path = data_path('decoys', decoy_set, 'no_missing_residue.txt')
        return pd.read_csv(txt_path)['pdb'].astype(str).tolist()
    if decoy_set in ('casp13', 'casp14'):
        csv_path = data_path('decoys', decoy_set, 'pdb_no_missing_residue.csv')
        return pd.read_csv(csv_path)['pdb'].astype(str).tolist()
    raise ValueError(f'decoy_set must be one of {_SUPPORTED}, got {decoy_set!r}')


def _load_decoy_names(pdb_id: str, decoy_set: str) -> pd.DataFrame:
    if decoy_set == '3DRobot_set':
        return pd.read_csv(data_path('decoys', decoy_set, pdb_id, 'list.csv'))
    if decoy_set == 'casp11':
        return pd.read_csv(
            data_path('decoys', decoy_set, pdb_id, 'list.txt'),
            header=None, names=['NAME'],
        )
    if decoy_set in ('casp13', 'casp14'):
        return pd.read_csv(data_path('decoys', decoy_set, pdb_id, 'list.csv'))
    raise ValueError(decoy_set)


def _decoy_stem(list_name: str, decoy_set: str) -> str:
    """Basename of the decoy model file without ``.pdb`` when appropriate."""
    if decoy_set in ('casp13', 'casp14'):
        return str(list_name)
    s = str(list_name)
    return s[:-4] if s.endswith('.pdb') else s


def _decoy_id_for_bead_csv(list_name: str, decoy_set: str) -> str:
    """Must match ``decoy_score._decoy_id_for`` → ``{id}_bead.csv``."""
    if decoy_set in ('casp13', 'casp14'):
        return str(list_name)
    s = str(list_name)
    return s[:-4] if s.endswith('.pdb') else s


def _resolve_pdb_path(
    pdb_root: str,
    pdb_id: str,
    list_name: str,
    decoy_set: str,
) -> List[str]:
    """Ordered candidate paths; first existing wins."""
    stem = _decoy_stem(list_name, decoy_set)
    raw = str(list_name)
    cands = [
        os.path.join(pdb_root, pdb_id, raw),
        os.path.join(pdb_root, pdb_id, f'{stem}.pdb'),
        os.path.join(pdb_root, pdb_id, stem),
        os.path.join(pdb_root, decoy_set, pdb_id, raw),
        os.path.join(pdb_root, decoy_set, pdb_id, f'{stem}.pdb'),
    ]
    # De-dupe preserving order
    seen = set()
    out = []
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _collect_jobs(
    decoy_set: str,
    pdb_root: str,
    targets: Optional[Iterable[str]],
    overwrite: bool,
    amino_acids_csv: Optional[str],
) -> List[Tuple[str, str, str, str, Optional[str]]]:
    """Each tuple: (pdb_id, decoy_list_name, pdb_path, out_csv, amino_acids_csv)."""
    tset = set(targets) if targets is not None else None
    all_targets = _load_target_ids(decoy_set)
    jobs: List[Tuple[str, str, str, str, Optional[str]]] = []

    for pdb_id in all_targets:
        if tset is not None and pdb_id not in tset:
            continue
        df = _load_decoy_names(pdb_id, decoy_set)
        if 'NAME' not in df.columns:
            raise ValueError(
                f'{pdb_id}: list manifest missing NAME column: {df.columns.tolist()}',
            )
        for list_name in df['NAME'].astype(str):
            decoy_id = _decoy_id_for_bead_csv(list_name, decoy_set)
            out_csv = data_path('decoys', decoy_set, pdb_id, f'{decoy_id}_bead.csv')
            if os.path.exists(out_csv) and not overwrite:
                continue
            pdb_path = None
            for cand in _resolve_pdb_path(pdb_root, pdb_id, list_name, decoy_set):
                if os.path.isfile(cand):
                    pdb_path = cand
                    break
            if pdb_path is None:
                # Placeholder path for error reporting
                pdb_path = _resolve_pdb_path(pdb_root, pdb_id, list_name, decoy_set)[0]
            jobs.append((pdb_id, list_name, pdb_path, out_csv, amino_acids_csv))
    return jobs


def _run_one(job: Tuple[str, str, str, str, Optional[str]]) -> Tuple[str, str]:
    pdb_id, list_name, pdb_path, out_csv, aa_csv = job
    if not os.path.isfile(pdb_path):
        return 'missing_pdb', f'{pdb_id} {list_name} -> {pdb_path}'
    try:
        extract_beads_v2(pdb_path, out_csv=out_csv, amino_acids_csv=aa_csv)
        return 'ok', out_csv
    except Exception as exc:  # noqa: BLE001
        return 'error', f'{pdb_id} {list_name}: {exc}'


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--decoy_set', required=True, choices=_SUPPORTED)
    p.add_argument(
        '--pdb_root', required=True,
        help='Root directory that contains per-target folders of decoy PDBs.',
    )
    p.add_argument(
        '--targets', default='',
        help='Comma-separated target ids (e.g. T1053,1A0A). Empty = all targets.',
    )
    p.add_argument('--overwrite', action='store_true',
                   help='Rebuild even when the bead CSV already exists.')
    p.add_argument('--dry_run', action='store_true',
                   help='Print planned actions only; do not write files.')
    p.add_argument('--num_workers', type=int, default=1,
                   help='Parallel workers (default 1 = serial).')
    p.add_argument('--amino_acids_csv', default=None,
                   help='Override path to amino_acids.csv (default: data_path).')
    args = p.parse_args(argv)

    targets = None
    if args.targets.strip():
        targets = {x.strip() for x in args.targets.split(',') if x.strip()}

    jobs = _collect_jobs(
        args.decoy_set,
        os.path.expanduser(args.pdb_root),
        targets,
        overwrite=args.overwrite,
        amino_acids_csv=args.amino_acids_csv,
    )

    if not jobs:
        print('[regenerate_decoy_beads] nothing to do (all CSVs exist? try --overwrite)')
        return 0

    print(f'[regenerate_decoy_beads] {len(jobs)} bead file(s) to process '
          f'for {args.decoy_set!r}')

    if args.dry_run:
        for pdb_id, list_name, pdb_path, out_csv, _ in jobs[:50]:
            print(f'  {pdb_id}  {list_name!r}  pdb={pdb_path!r}  -> {out_csv}')
        if len(jobs) > 50:
            print(f'  ... and {len(jobs) - 50} more')
        return 0

    if args.num_workers <= 1:
        stats = {'ok': 0, 'error': 0, 'missing_pdb': 0}
        for job in tqdm(jobs, desc='beads'):
            status, msg = _run_one(job)
            if status == 'ok':
                stats['ok'] += 1
            elif status == 'missing_pdb':
                stats['missing_pdb'] += 1
                tqdm.write(f'[skip] {msg}')
            else:
                stats['error'] += 1
                tqdm.write(f'[{status}] {msg}')
    else:
        stats = {'ok': 0, 'error': 0, 'missing_pdb': 0}
        with Pool(args.num_workers) as pool:
            for status, msg in tqdm(
                pool.imap_unordered(_run_one, jobs, chunksize=8),
                total=len(jobs), desc='beads',
            ):
                if status == 'ok':
                    stats['ok'] += 1
                elif status == 'missing_pdb':
                    stats['missing_pdb'] += 1
                else:
                    stats['error'] += 1
                    tqdm.write(f'[{status}] {msg}')

    print(
        f'[regenerate_decoy_beads] done: ok={stats["ok"]} '
        f'missing_pdb={stats["missing_pdb"]} error={stats["error"]}',
    )
    return 0 if stats['error'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
