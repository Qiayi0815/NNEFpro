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


# ----------------------------------------------------------------------
# Per-checkpoint architecture (DO NOT expose these as CLI flags -- see
# module docstring for why).
# ----------------------------------------------------------------------
_BASE_ARCH = dict(
    seq_len=14, seq_type='residue', residue_type_num=20,
    embed_size=32, dim=128, n_layers=4, attn_heads=4,
    mixture_r=2, mixture_angle=3,
    smooth_gaussian=True, smooth_r=0.3, smooth_angle=45,
    coords_angle_loss_lamda=1, profile_loss_lamda=10, coords_rama_loss_lamda=1,
    use_position_weights=True, cen_seg_loss_lamda=1, oth_seg_loss_lamda=3,
)

CHECKPOINTS = {
    'yang': dict(
        load_exp='checkpoints/yang', legacy_local_frame=True,
        mixture_seq=1, mixture_rama=0, angle_dist='gaussian', r_dist='gaussian',
    ),
    'yang_vonmises': dict(
        load_exp='checkpoints/yang_vonmises', legacy_local_frame=True,
        mixture_seq=1, mixture_rama=0, angle_dist='vonmises', r_dist='gaussian',
    ),
    'yang_vmf': dict(
        load_exp='checkpoints/yang_vmf', legacy_local_frame=True,
        mixture_seq=1, mixture_rama=0, angle_dist='vmf', r_dist='gaussian',
    ),
    'ribbon': dict(
        load_exp='checkpoints/ribbon', legacy_local_frame=False,
        mixture_seq=1, mixture_rama=10, angle_dist='gaussian', r_dist='gaussian',
    ),
    'ribbon_vonmises': dict(
        load_exp='checkpoints/ribbon_vonmises', legacy_local_frame=False,
        mixture_seq=1, mixture_rama=10, angle_dist='vonmises', r_dist='gaussian',
    ),
    'esm': dict(
        load_exp='checkpoints/esm', legacy_local_frame=False,
        mixture_seq=1, mixture_rama=10, angle_dist='gaussian', r_dist='gaussian',
        use_esm=True, esm_dim_in=1152, esm_dim_out=32, esm_pool='per_residue',
        # esm_h5_path is NOT set here -- 'esm' scoring needs a per-residue ESM-C
        # embedding computed for YOUR sequence; see the guide's "esm" section.
    ),
}

_THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}


def pdb_to_bead_df(pdb_path: str):
    """One PDB -> one bead DataFrame (chain_id, group_num, group_name, N/CA/C/CB
    xyz). Mirrors nnef/data_prep_scripts/fetch_and_beads.py's extract_beads()
    exactly, so this matches the training-data format bit-for-bit. Returns
    None if no residue has a complete backbone (+CB)."""
    from Bio.PDB import PDBParser, Selection

    structure = PDBParser(QUIET=True).get_structure(os.path.basename(pdb_path), pdb_path)
    cols = ['chain_id', 'group_num', 'group_name',
            'xn', 'yn', 'zn', 'xca', 'yca', 'zca', 'xc', 'yc', 'zc', 'xcb', 'ycb', 'zcb']
    bead = {c: [] for c in cols}
    n_dropped = 0
    for res in Selection.unfold_entities(structure, 'R'):
        if res.id[0] != ' ':
            continue  # skip waters / heteroatoms
        resname = res.get_resname().upper()
        if resname not in _THREE_TO_ONE:
            continue
        try:
            n, ca, c = (res[a].get_coord() for a in ('N', 'CA', 'C'))
        except KeyError:
            n_dropped += 1
            continue
        if resname == 'GLY':
            cb = ca
        else:
            try:
                cb = res['CB'].get_coord()
            except KeyError:
                n_dropped += 1
                continue
        bead['chain_id'].append(res.parent.id)
        bead['group_num'].append(res.id[1])
        bead['group_name'].append(_THREE_TO_ONE[resname])
        bead['xn'].append(n[0]); bead['yn'].append(n[1]); bead['zn'].append(n[2])
        bead['xca'].append(ca[0]); bead['yca'].append(ca[1]); bead['zca'].append(ca[2])
        bead['xc'].append(c[0]); bead['yc'].append(c[1]); bead['zc'].append(c[2])
        bead['xcb'].append(cb[0]); bead['ycb'].append(cb[1]); bead['zcb'].append(cb[2])
    if not bead['chain_id']:
        return None, n_dropped
    return pd.DataFrame(bead), n_dropped


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

    arch = CHECKPOINTS[args.checkpoint]
    parser = options.get_fold_parser()
    ns = argparse.Namespace(**vars(parser.parse_args([])))  # defaults for everything else
    for k, v in _BASE_ARCH.items():
        setattr(ns, k, v)
    for k, v in arch.items():
        setattr(ns, k, v)
    ns.device = args.device
    ns.load_exp = os.path.join(_REPO_ROOT, arch['load_exp'])
    if not os.path.isfile(os.path.join(ns.load_exp, 'models', 'model.pt')):
        raise SystemExit(
            f"[score_for_collaborator] ERROR: no checkpoint at {ns.load_exp}/models/model.pt -- "
            f"did you pull the checkpoints (see the guide's setup step)?")

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
