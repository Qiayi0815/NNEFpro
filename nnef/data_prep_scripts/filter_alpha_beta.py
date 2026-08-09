#!/usr/bin/env python3
"""Filter a pdb_list to CATH alpha/beta chains (Yang 2022 'radical partition').

Yang trained ONLY on CATH class-3 (alpha/beta) chains (~7,500) to test
transferability. This reproduces that filter on OUR rebuilt pdb_list, so we can
train an alpha/beta-only model and compare it to the all-classes model on the
SAME architecture + h5 (the alpha/beta list is just a subset — no new h5).

Classification (same rule as data_prep_scripts/cath.py): a chain's PDB is kept
if ALL its CATH domains share a single class digit and that digit is '3'
(1=mainly-alpha, 2=mainly-beta, 3=alpha/beta).

    python data_prep_scripts/filter_alpha_beta.py \
        --cath nnef/data/cath-b-newest-all \
        --pdb_list $DATA_DIR/hhsuite_CB_v2_new_pdb_list.csv \
        --out $DATA_DIR/hhsuite_CB_v2_new_alpha_beta.csv
"""
import argparse
import pandas as pd


def build_pdb_class_map(cath_path):
    """pdb4 (lower) -> set of CATH class digits across all its domains."""
    df = pd.read_csv(cath_path, sep=r'\s+', header=None, usecols=[0, 2],
                     names=['id', 'cath'])
    pdb4 = df['id'].str[:4].str.lower()
    cls = df['cath'].str.split('.').str[0]
    m = {}
    for p, c in zip(pdb4, cls):
        m.setdefault(p, set()).add(c)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cath', required=True)
    ap.add_argument('--pdb_list', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--class_digit', default='3',
                    help="CATH class to keep: 1=alpha 2=beta 3=alpha/beta")
    a = ap.parse_args()

    cmap = build_pdb_class_map(a.cath)
    df = pd.read_csv(a.pdb_list)
    keep = []
    n_nocath = n_multi = 0
    for pdb in df['pdb'].astype(str):
        pdb4 = pdb.split('_')[0].lower()
        classes = cmap.get(pdb4)
        if classes is None:
            n_nocath += 1
            keep.append(False)
        elif len(classes) == 1 and next(iter(classes)) == a.class_digit:
            keep.append(True)
        else:
            n_multi += 1
            keep.append(False)
    out = df[pd.Series(keep, index=df.index)].reset_index(drop=True)
    out.to_csv(a.out, index=False)
    print(f'[filter_alpha_beta] {len(df)} chains -> {len(out)} class-{a.class_digit} '
          f'(dropped: {n_nocath} no-CATH, {n_multi} multi/other-class)')
    print(f'  wrote {a.out}')


if __name__ == '__main__':
    main()
