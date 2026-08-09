#!/usr/bin/env python3
"""Split a pdb_list CSV (as consumed by train_chimeric.py --data_flag) into a
held-out validation split, for tuning ribbon's hyperparameters (mixture_rama,
coords_rama_loss_lamda, mixture_angle) WITHOUT looking at the eval benchmark
(3KXT MD / decoy Pearson). Pandas-only, no torch needed.

Usage:
    python nnef/data_prep_scripts/split_pdb_list_val.py \\
        --pdb_list $NS/hhsuite_CB_v2_new_pdb_list.csv \\
        --frac 0.05 --seed 0 \\
        --train_out $NS/hhsuite_CB_v2_new_pdb_list_train.csv \\
        --val_out   $NS/hhsuite_CB_v2_new_pdb_list_val.csv

Splits by row (each row = one pdb chain), so no chain appears in both splits.
Run this ONCE; reuse the same train/val CSVs across the whole ribbon ablation
grid so every combo is validated on identical held-out data.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pdb_list', required=True)
    p.add_argument('--frac', type=float, default=0.05, help='fraction held out for validation')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--train_out', required=True)
    p.add_argument('--val_out', required=True)
    args = p.parse_args()

    df = pd.read_csv(args.pdb_list)
    n = len(df)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(n)
    n_val = max(1, int(n * args.frac))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    df.iloc[train_idx].reset_index(drop=True).to_csv(args.train_out, index=False)
    df.iloc[val_idx].reset_index(drop=True).to_csv(args.val_out, index=False)
    print(f'[split_pdb_list_val] {n} chains -> train {len(train_idx)} / val {len(val_idx)} '
          f'(seed={args.seed}, frac={args.frac})')
    print(f'  train -> {args.train_out}')
    print(f'  val   -> {args.val_out}')


if __name__ == '__main__':
    main()
