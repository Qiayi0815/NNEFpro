#!/usr/bin/env python3
"""Download a list of PDB IDs and extract bead CSVs (N/CA/C/CB) for MD.

Reuses ``fetch_and_beads.download_pdb`` + ``extract_beads`` so the output columns
match the existing corpus (chain_id,group_num,group_name,xn..zcb) -- i.e. exactly
what the v2-frame (ribbon/esm) and legacy (yang) MD both read. One folder per
protein so NATIVE_BEAD paths are predictable:

    <out_root>/<PDB>/<PDB>_bead.csv

Used to set up the Paper_1 Table-2 proteins (scorpion toxin 1AHO, lysozyme, ...):

    python data_prep_scripts/fetch_beads_for_list.py \
        --pdb_ids 1AHO 2LZT 3U7T 4YUO 5TOF 6B8X 3X2P 1STN \
        --out_root nnef/data/paper1_table2
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_and_beads import download_pdb, extract_beads  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pdb_ids', nargs='+', required=True)
    ap.add_argument('--out_root', default='nnef/data/paper1_table2')
    ap.add_argument('--amino_acids_csv',
                    default=os.path.join(_HERE, '..', 'data', 'amino_acids.csv'))
    a = ap.parse_args()

    ok, bad = [], []
    for pid in a.pdb_ids:
        pid = pid.strip()
        d = os.path.join(a.out_root, pid)
        os.makedirs(d, exist_ok=True)
        try:
            download_pdb(pid, d)                       # -> d/<pid>.pdb
            extract_beads(f'{pid}.pdb', d, d, a.amino_acids_csv)  # -> d/<pid>_bead.csv
            bead = os.path.join(d, f'{pid}_bead.csv')
            if os.path.isfile(bead):
                n = sum(1 for _ in open(bead)) - 1
                print(f'[beads] {pid}: {n} residues -> {bead}', flush=True)
                ok.append(pid)
            else:
                print(f'[beads] {pid}: NO bead csv (missing backbone atoms?)', flush=True)
                bad.append(pid)
        except Exception as e:
            print(f'[beads] {pid}: FAILED {e}', flush=True)
            bad.append(pid)
    print(f'\n[beads] done: {len(ok)} ok {ok}; {len(bad)} failed {bad}')


if __name__ == '__main__':
    main()
