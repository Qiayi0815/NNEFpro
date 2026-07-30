#!/usr/bin/env python3
"""Score your own PDB structure(s) with the NNEF energy function.

One command, one PDB (or a folder of PDBs) in, one CSV of energies out. The
per-checkpoint architecture flags (frame convention, angle/radial output
distribution, rama head, ESM adapter) are hard-coded below so you cannot
accidentally score a checkpoint with the wrong settings -- every checkpoint
in this project has, at some point, been silently mis-scored this way, so
this script does not expose those flags at all.

Checkpoints (see --checkpoint):
    yang              Yang et al. 2022 reproduction, retrained on 28k chains
    yang_vonmises     same backbone, circular von Mises angle distribution
    yang_vmf          same backbone, von Mises-Fisher direction distribution
    ribbon            yang + v2 (N-Ca-C) frame + Ramachandran head
    ribbon_vonmises   ribbon + von Mises angle distribution
    esm               ribbon + frozen ESM-C 600M sequence embedding
                       (needs the `esm` pip package and a GPU/CPU that can
                       run a 600M-parameter language model -- see the guide)

Usage:
    # one structure
    python nnef/scripts/score_for_collaborator.py \\
        --checkpoint yang --input my_protein.pdb --out_csv scores.csv

    # a folder of structures (e.g. decoys of one target)
    python nnef/scripts/score_for_collaborator.py \\
        --checkpoint ribbon --input decoys/T1026/ --out_csv scores.csv

    # also compute Pearson r against known quality labels (e.g. GDT_TS)
    python nnef/scripts/score_for_collaborator.py \\
        --checkpoint yang --input decoys/T1026/ --out_csv scores.csv \\
        --labels_csv decoys/T1026/gdt.csv   # columns: structure,label

Input PDB requirements: standard PDB format, single chain, every residue
needs backbone N/CA/C (+ CB, except glycine). Residues missing any of these
atoms are silently dropped (matches this project's training-data convention)
-- check the printed residue count against what you expect.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_NNEF_DIR = os.path.abspath(os.path.join(_HERE, '..'))
_REPO_ROOT = os.path.abspath(os.path.join(_NNEF_DIR, '..'))
if _NNEF_DIR not in sys.path:
    sys.path.insert(0, _NNEF_DIR)

import options  # noqa: E402
from protein_os import Protein  # noqa: E402
from utils import load_protein_bead, test_setup  # noqa: E402
from _collaborator_common import CHECKPOINTS, pdb_to_bead_df, resolve_args  # noqa: E402


def score_one(pdb_path: str, energy_fn, device) -> float:
    df, n_dropped = pdb_to_bead_df(pdb_path)
    if df is None:
        raise ValueError(f'{pdb_path}: no residue with a complete backbone (N/CA/C[/CB]) found')
    if n_dropped:
        print(f'  [{os.path.basename(pdb_path)}] warning: dropped {n_dropped} '
              f'residue(s) missing a backbone atom; scoring the remaining {len(df)}')
    tmp_csv = pdb_path + '.__bead_tmp.csv'
    df.to_csv(tmp_csv, index=False)
    try:
        seq, coords, profile = load_protein_bead(tmp_csv, mode='CB', device=device)
    finally:
        os.remove(tmp_csv)
    protein = Protein(seq, coords, profile)
    return float(protein.get_energy(energy_fn).item())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', required=True, choices=sorted(CHECKPOINTS))
    ap.add_argument('--input', required=True, help='a .pdb file, or a directory of .pdb files')
    ap.add_argument('--out_csv', required=True)
    ap.add_argument('--labels_csv', default=None,
                     help='optional CSV with columns "structure,label" (e.g. GDT_TS) '
                          '-- if given, also prints Pearson/Spearman r vs. energy')
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    args = ap.parse_args()

    ns, arch = resolve_args(args.checkpoint, _REPO_ROOT, options)
    ns.device = args.device

    if arch.get('use_esm'):
        # This script never builds/passes an esm_full tensor to Protein(), so
        # Protein's ESM adapter branch (`if self.use_esm and esm is not None`)
        # would silently take the esm=None path -- the checkpoint's ESM
        # weights contribute NOTHING, but the run completes and prints a
        # plausible-looking energy with no warning. Refuse instead of
        # producing a confidently wrong number; see TESTING.md sec. 5.
        raise SystemExit(
            "[score_for_collaborator] ERROR: --checkpoint esm needs a precomputed "
            "ESM-C embedding for your sequence, which this script does not build. "
            "Running it anyway would silently score with the ESM branch switched "
            "off (misleadingly plausible-looking energy). See TESTING.md section 5.")

    print(f'[score_for_collaborator] checkpoint={args.checkpoint}  '
          f'angle_dist={arch["angle_dist"]}  legacy_frame={arch["legacy_local_frame"]}  '
          f'device={args.device}')
    device, _model, energy_fn, _pb = test_setup(ns)

    if os.path.isdir(args.input):
        pdb_paths = sorted(glob.glob(os.path.join(args.input, '*.pdb')))
        if not pdb_paths:
            raise SystemExit(f'[score_for_collaborator] no .pdb files found under {args.input}')
    else:
        pdb_paths = [args.input]

    rows = []
    for p in pdb_paths:
        name = os.path.splitext(os.path.basename(p))[0]
        try:
            e = score_one(p, energy_fn, device)
        except Exception as exc:
            print(f'  [{name}] FAILED: {exc}')
            continue
        print(f'  [{name}] energy = {e:.3f}')
        rows.append({'structure': name, 'energy': e})

    if not rows:
        raise SystemExit('[score_for_collaborator] no structure scored successfully')
    out = pd.DataFrame(rows)

    if args.labels_csv:
        labels = pd.read_csv(args.labels_csv)
        out = out.merge(labels, on='structure', how='left')
        n_matched = out['label'].notna().sum()
        if n_matched >= 3:
            from scipy.stats import pearsonr, spearmanr
            sub = out.dropna(subset=['label'])
            r, _ = pearsonr(sub['energy'], sub['label'])
            rho, _ = spearmanr(sub['energy'], sub['label'])
            print(f'\nPearson(energy, label)  = {r:.3f}   (n={n_matched})')
            print(f'Spearman(energy, label) = {rho:.3f}')
            print('Note: a GOOD model has energy and quality NEGATIVELY correlated '
                  '(lower energy = higher quality/GDT_TS).')
        else:
            print(f'\n[score_for_collaborator] only {n_matched} structures matched --labels_csv, '
                  f'skipping correlation (need >= 3)')

    out.to_csv(args.out_csv, index=False)
    print(f'\nwrote {args.out_csv}  ({len(out)} structures)')


if __name__ == '__main__':
    main()
