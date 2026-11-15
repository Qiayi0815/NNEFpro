"""Which energy term drives Langevin drift below native? (diagnostic A1)

For harvested negatives (nnef/gen_negatives.py output) re-score the native and
the far frames (RMSD >= --rmsd_min) term by term and report the mean change
per term, Delta_t = E_t(frame) - E_t(native). If the total drop is carried by
loss_profile, MD is pushing structures toward ones that make the sequence
"more predictable" (over-confident inverse folding); if by loss_r/loss_angle,
the local geometric density itself is mis-specified off-manifold.

    python nnef/scripts/drift_term_analysis.py <model flags> --neg_dir runs/negatives/yang_r0 \
        --out eval/decomp/drift_terms_yang.csv
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch

_NNEF = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _NNEF)
import options  # noqa: E402
from protein_os import Protein  # noqa: E402
from utils import load_protein_bead, test_setup  # noqa: E402

TERMS = ['loss_r', 'loss_angle', 'loss_profile', 'loss_start_id', 'loss_res_counts', 'loss_rama']


def main():
    parser = options.get_local_gen_parser()
    parser.add_argument('--neg_dir', required=True)
    parser.add_argument('--rmsd_min', type=float, default=2.0)
    parser.add_argument('--max_chains', type=int, default=200)
    parser.add_argument('--out', required=True)
    args = options.parse_args_and_arch(parser)
    device, model, energy_fn, _pb = test_setup(args)

    beads = {os.path.basename(p)[:-len('_bead.csv')]: p
             for p in glob.glob(os.path.join(args.neg_dir, '_pool_shard*', 'pool_beads', '*_bead.csv'))}
    rows = []
    for f in sorted(glob.glob(os.path.join(args.neg_dir, '*.npz')))[:args.max_chains]:
        cid = os.path.basename(f)[:-4]
        if cid not in beads:
            continue
        z = np.load(f)
        far = np.where(z['rmsd'] >= args.rmsd_min)[0]
        if len(far) == 0:
            continue
        seq, coords, profile = load_protein_bead(beads[cid], args.mode, device)
        prot = Protein(seq, coords.clone(), profile)
        with torch.no_grad():
            _, t_nat = prot.get_energy(energy_fn, return_terms=True)
            t_nat = np.array([float(t) for t in t_nat])
            for i in far[-3:]:  # the three latest far frames
                prot.coords = torch.tensor(z['frames'][i], dtype=coords.dtype, device=device)
                _, t_f = prot.get_energy(energy_fn, return_terms=True)
                d = np.array([float(t) for t in t_f]) - t_nat
                rows.append(dict(chain=cid, frame=int(i), rmsd=float(z['rmsd'][i]),
                                 rg_ratio=float(z['rg'][i] / z['rg_native']),
                                 d_total=float(d.sum()), **{f'd_{k}': float(v) for k, v in zip(TERMS, d)}))
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    per_chain = df.groupby('chain').mean(numeric_only=True)
    print(f'[drift] {per_chain.shape[0]} chains, {len(df)} far frames (RMSD >= {args.rmsd_min})')
    print(f'[drift] frac chains with total below native: {(per_chain.d_total < 0).mean():.2f}')
    for k in TERMS:
        c = per_chain[f'd_{k}']
        share = (c / per_chain.d_total).median()
        print(f'[drift] {k:16s} mean Delta {c.mean():+8.2f}  median {c.median():+8.2f}  '
              f'frac<0 {(c < 0).mean():.2f}  median share of total drop {share:+.2f}')
    print(f'[drift] corr(d_total, rg_ratio) = {per_chain.d_total.corr(per_chain.rg_ratio):+.2f}')


if __name__ == '__main__':
    main()
