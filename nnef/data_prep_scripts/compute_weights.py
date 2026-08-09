#!/usr/bin/env python3
"""Stage 7b: write Yang-style sampling weights into the pdb_list CSV.

Yang weights each chain by sqrt(n_homologous_seqs) * n_building_blocks / 100
(data_hh/get_weight.py), so the WeightedRandomSampler draws proportionally to
sequence diversity x structure size. build_yang_h5 writes a uniform weight=1.0;
this replaces it once both h5 files exist.

    python data_prep_scripts/compute_weights.py \
        --struct_h5 $NS/nnef_data/hhsuite_CB_yang.h5 \
        --seq_h5    $NS/nnef_data/hhsuite_pdb_seq_yang.h5 \
        --pdb_list  $DATA_DIR/hhsuite_CB_yang_pdb_list.csv
"""
import argparse
import numpy as np
import pandas as pd
import h5py


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--struct_h5', required=True)
    p.add_argument('--seq_h5', required=True)
    p.add_argument('--pdb_list', required=True)
    a = p.parse_args()

    df = pd.read_csv(a.pdb_list)
    struct = h5py.File(a.struct_h5, 'r')
    seq = h5py.File(a.seq_h5, 'r')

    weight = np.zeros(len(df))
    seq_len = np.zeros(len(df), dtype=int)
    kept = np.ones(len(df), dtype=bool)
    for i, pdb in enumerate(df['pdb'].astype(str)):
        if pdb not in struct or pdb not in seq:
            kept[i] = False
            continue
        n_seq = seq[pdb].shape[0]                       # homologs
        n_blk = struct[pdb]['group_num'].shape[0]       # building blocks
        weight[i] = np.sqrt(max(n_seq, 1)) * n_blk / 100.0
        seq_len[i] = n_blk
    df['weight'] = weight
    df['seq_len'] = seq_len
    n_drop = int((~kept).sum())
    df = df[kept].reset_index(drop=True)
    df.to_csv(a.pdb_list, index=False)
    print(f'[compute_weights] {len(df)} chains weighted '
          f'(dropped {n_drop} missing from a h5); '
          f'weight range [{df["weight"].min():.3f}, {df["weight"].max():.3f}]')


if __name__ == '__main__':
    main()
