"""Download Yang 2022 small-protein test set PDBs and convert to bead CSVs.

Yang et al. 2022 (JCTC, doi:10.1021/acs.jctc.2c00069) used 18 small proteins
(<120 residues, 4 alpha + 7 beta + 7 alpha/beta) for their MD validation.
This script downloads the PDB files from RCSB and writes each to a bead CSV
in the format expected by fasrc/md_eval_yang_small.slurm:

  nnef/data/yang_small/<PDB>/<PDB>_bead.csv

Columns: chain_id, group_num, group_name, xn, yn, zn, xca, yca, zca,
         xc, yc, zc, xcb, ycb, zcb
(one row per residue; glycine CB is copied from CA.)

Usage:
  python nnef/data_prep_scripts/fetch_yang_small_beads.py
  python nnef/data_prep_scripts/fetch_yang_small_beads.py --only 3VI6 4O7Q

Requires: biopython, requests.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import pandas as pd
import requests
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa

# Standard 3-letter -> 1-letter amino acid code map. Hardcoded to avoid the
# Bio.PDB API churn around three_to_one (removed in newer biopython releases).
AA3_TO_AA1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}


# Yang 2022 full 18-protein small test set.
# Source: SI of Yang/Xiong/Zonta JCTC 2022 (confirmed via PMC9476656).
# Already on disk: 1ZZK, 2MCM, 3KXT, 4M1X, 6H8O.
YANG_18 = [
    ('1ZZK', 'A'),
    ('2MCM', 'A'),
    ('2VIM', 'A'),
    ('3FBL', 'A'),
    ('3IPZ', 'A'),
    ('3KXT', 'A'),
    ('3NGP', 'A'),
    ('3P0C', 'A'),
    ('3SNY', 'A'),
    ('3SOL', 'A'),
    ('3VI6', 'A'),
    ('4M1X', 'A'),
    ('4O7Q', 'A'),
    ('4QRL', 'A'),
    ('5CYB', 'A'),
    ('5JOE', 'A'),
    ('5ZGM', 'A'),
    ('6H8O', 'A'),
]


REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT_ROOT = os.path.join(REPO, 'nnef', 'data', 'yang_small')


def download_pdb(pdb_id: str, dest: str) -> None:
    """Download PDB file from RCSB if not already present."""
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        return
    url = f'https://files.rcsb.org/download/{pdb_id}.pdb'
    print(f'  fetching {url}')
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    with open(dest, 'w') as fh:
        fh.write(r.text)


def extract_bead_csv(pdb_path: str, target_chain: str) -> pd.DataFrame:
    """Extract N/CA/C/CB coordinates per residue for the requested chain.

    Glycine has no CB; we copy CA coordinates into CB columns to match the
    convention in the existing yang_small bead CSVs.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('s', pdb_path)
    model = next(structure.get_models())  # first model only (NMR ensembles)

    if target_chain not in {c.id for c in model}:
        raise ValueError(f'chain {target_chain!r} not found in {pdb_path}; '
                         f'available chains: {[c.id for c in model]}')

    chain = model[target_chain]
    rows = []
    for res in chain:
        # Skip waters, ligands, alt-conformers etc.
        if res.id[0] != ' ':
            continue
        if not is_aa(res, standard=True):
            continue
        try:
            n = res['N'].get_coord()
            ca = res['CA'].get_coord()
            c = res['C'].get_coord()
        except KeyError:
            # Backbone incomplete -- skip this residue.
            continue
        try:
            cb = res['CB'].get_coord()
        except KeyError:
            cb = ca  # glycine convention
        three = res.get_resname().upper()
        one_letter = AA3_TO_AA1.get(three)
        if one_letter is None:
            continue  # non-canonical residue, skip
        rows.append({
            'chain_id': target_chain,
            'group_num': res.id[1],
            'group_name': one_letter,
            'xn': round(float(n[0]), 3), 'yn': round(float(n[1]), 3), 'zn': round(float(n[2]), 3),
            'xca': round(float(ca[0]), 3), 'yca': round(float(ca[1]), 3), 'zca': round(float(ca[2]), 3),
            'xc': round(float(c[0]), 3), 'yc': round(float(c[1]), 3), 'zc': round(float(c[2]), 3),
            'xcb': round(float(cb[0]), 3), 'ycb': round(float(cb[1]), 3), 'zcb': round(float(cb[2]), 3),
        })
    return pd.DataFrame(rows)


def process_one(pdb_id: str, chain: str, force: bool = False) -> Optional[str]:
    pdb_id_u = pdb_id.upper()
    out_dir = os.path.join(OUT_ROOT, pdb_id_u)
    os.makedirs(out_dir, exist_ok=True)
    pdb_path = os.path.join(out_dir, f'{pdb_id_u}.pdb')
    bead_csv = os.path.join(out_dir, f'{pdb_id_u}_bead.csv')

    if os.path.isfile(bead_csv) and not force:
        n = sum(1 for _ in open(bead_csv)) - 1
        print(f'[{pdb_id_u}] already has bead CSV ({n} residues) -- skipping. Use --force to redo.')
        return bead_csv

    print(f'[{pdb_id_u}_{chain}] processing...')
    download_pdb(pdb_id_u, pdb_path)
    df = extract_bead_csv(pdb_path, chain)
    if len(df) == 0:
        print(f'[{pdb_id_u}_{chain}] ERROR: no standard amino acid residues found in chain {chain}')
        return None
    df.to_csv(bead_csv, index=False)
    print(f'[{pdb_id_u}_{chain}] -> {bead_csv} ({len(df)} residues)')
    return bead_csv


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--only', nargs='+', default=None,
                   help='Only process these PDB IDs (case-insensitive).')
    p.add_argument('--force', action='store_true',
                   help='Re-extract bead CSV even if it already exists.')
    args = p.parse_args()

    if args.only:
        wanted = {x.upper() for x in args.only}
        targets = [(p, c) for p, c in YANG_18 if p.upper() in wanted]
        if not targets:
            print(f'no match in YANG_18 for {wanted}', file=sys.stderr)
            sys.exit(1)
    else:
        targets = YANG_18

    print(f'==> processing {len(targets)} protein(s) into {OUT_ROOT}')
    ok, fail = 0, 0
    for pdb_id, chain in targets:
        try:
            out = process_one(pdb_id, chain, force=args.force)
            if out:
                ok += 1
            else:
                fail += 1
        except Exception as exc:
            print(f'[{pdb_id}_{chain}] FAILED: {type(exc).__name__}: {exc}', file=sys.stderr)
            fail += 1

    print(f'\n==> done: {ok} ok, {fail} failed')


if __name__ == '__main__':
    main()
