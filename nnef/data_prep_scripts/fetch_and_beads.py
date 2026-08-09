#!/usr/bin/env python3
"""Download PISCES PDBs and extract bead CSVs — parallelizable, resumable.

Independent of the HHsuite alignment stage, so it can run concurrently with
fasrc/align_hhsuite.slurm to pre-stage all bead CSVs. Reuses the EXISTING
bead extractor (data_hh/q_extract_beads.extract_beads) so the output columns
(chain_id, group_num, group_name, xn..zcb) exactly match the current bead_csvs.

For each unique PDB id in the PISCES FASTA (round-robin sharded):
  * skip if <PDB>_bead.csv already exists (resume),
  * download <PDB>.pdb from RCSB if missing,
  * extract beads.

Usage (one shard):
    python data_prep_scripts/fetch_and_beads.py \
        --fasta   $DATA_DIR/cullpdb_...chains32021.fasta \
        --pdb_dir $DATA_DIR/pdb_files \
        --bead_dir $DATA_DIR/bead_csvs \
        --shard 0 --nshards 16
"""
import argparse
import os
import sys
import time

import requests
import pandas as pd
from Bio.PDB import PDBParser, Selection

_HERE = os.path.dirname(os.path.abspath(__file__))


def extract_beads(pdb_filename, pdb_dir, output_dir, amino_acids_file):
    """Inlined from data_hh/q_extract_beads.py (data_hh is not synced to the
    cluster). Writes <base>_bead.csv with columns
    chain_id,group_num,group_name,xn..zcb — identical to the existing corpus.
    Drops the whole file if any residue is missing N/CA/C (or CB for non-Gly).
    """
    base = os.path.splitext(pdb_filename)[0]
    attempts = [os.path.join(pdb_dir, pdb_filename),
                os.path.join(pdb_dir, pdb_filename + '.pdb')]
    pdb_path = next((p for p in attempts if os.path.isfile(p)), None)
    if pdb_path is None:
        return False
    aa_df = pd.read_csv(amino_acids_file)
    vocab_aa = aa_df.AA3C.str.upper().tolist()
    vocab_dict = {row.AA3C.upper(): row.AA for _, row in aa_df.iterrows()}
    try:
        structure = PDBParser(QUIET=True).get_structure(base, pdb_path)
    except Exception:
        return False
    cols = ['chain_id', 'group_num', 'group_name',
            'xn', 'yn', 'zn', 'xca', 'yca', 'zca', 'xc', 'yc', 'zc', 'xcb', 'ycb', 'zcb']
    bead = {c: [] for c in cols}
    for res in Selection.unfold_entities(structure, 'R'):
        if res.id[0] != ' ':
            continue
        resname = res.get_resname().upper()
        if resname not in vocab_aa:
            continue
        try:
            n, ca, c = (res[a].get_coord() for a in ('N', 'CA', 'C'))
        except KeyError:
            return False
        if resname != 'GLY':
            try:
                cb = res['CB'].get_coord()
            except KeyError:
                return False
        else:
            cb = ca
        bead['chain_id'].append(res.parent.id)
        bead['group_num'].append(res.id[1])
        bead['group_name'].append(vocab_dict.get(resname, resname))
        bead['xn'].append(n[0]); bead['yn'].append(n[1]); bead['zn'].append(n[2])
        bead['xca'].append(ca[0]); bead['yca'].append(ca[1]); bead['zca'].append(ca[2])
        bead['xc'].append(c[0]); bead['yc'].append(c[1]); bead['zc'].append(c[2])
        bead['xcb'].append(cb[0]); bead['ycb'].append(cb[1]); bead['zcb'].append(cb[2])
    if not bead['chain_id']:
        return False
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(bead).to_csv(os.path.join(output_dir, f'{base}_bead.csv'), index=False)
    return True


def pdb_ids_from_fasta(fasta):
    ids = []
    seen = set()
    with open(fasta) as f:
        for line in f:
            if line.startswith('>'):
                pdb4 = line[1:5].upper()             # header like ">16PKA ..."
                if pdb4 not in seen:
                    seen.add(pdb4)
                    ids.append(pdb4)
    return ids


def download_pdb(pdb_id, out_dir, retries=3):
    dst = os.path.join(out_dir, f'{pdb_id}.pdb')
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return True
    url = f'https://files.rcsb.org/download/{pdb_id}.pdb'
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                with open(dst, 'w') as fh:
                    fh.write(r.text)
                return True
            if r.status_code == 404:
                return False                          # obsolete / no PDB-format file
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fasta', required=True)
    p.add_argument('--pdb_dir', required=True)
    p.add_argument('--bead_dir', required=True)
    p.add_argument('--amino_acids_csv', default=os.path.join(_HERE, '..', 'data', 'amino_acids.csv'))
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshards', type=int, default=1)
    a = p.parse_args()

    os.makedirs(a.pdb_dir, exist_ok=True)
    os.makedirs(a.bead_dir, exist_ok=True)

    ids = pdb_ids_from_fasta(a.fasta)
    mine = [pid for i, pid in enumerate(ids) if i % a.nshards == a.shard]
    print(f'[fetch_and_beads] shard {a.shard}/{a.nshards}: {len(mine)} of {len(ids)} unique PDBs')

    n_skip = n_dl = n_bead = n_fail = 0
    for i, pid in enumerate(mine):
        bead_csv = os.path.join(a.bead_dir, f'{pid}_bead.csv')
        if os.path.exists(bead_csv):
            n_skip += 1
            continue
        if not download_pdb(pid, a.pdb_dir):
            n_fail += 1
            continue
        n_dl += 1
        try:
            ok = extract_beads(f'{pid}.pdb', pdb_dir=a.pdb_dir,
                               output_dir=a.bead_dir, amino_acids_file=a.amino_acids_csv)
            if ok:
                n_bead += 1
        except Exception as e:
            print(f'  bead extract failed {pid}: {e}')
            n_fail += 1
        if (i + 1) % 200 == 0:
            print(f'  [{a.shard}] {i+1}/{len(mine)}  dl={n_dl} bead={n_bead} skip={n_skip} fail={n_fail}')

    print(f'[fetch_and_beads] shard {a.shard} DONE: downloaded {n_dl}, beads {n_bead}, '
          f'skipped {n_skip}, failed {n_fail}')


if __name__ == '__main__':
    main()
