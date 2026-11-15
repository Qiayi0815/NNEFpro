"""Score backbone frames (F, 4, L, 3: N, CA, C, CB[Gly->CA]) with a backbone NNEF model (summed energy).

Used for Yang 2022 Fig. 4 style plots (energy vs RMSD along an external MD trajectory).

    python nnef/scripts/score_frames.py <model flags> --native_bead runs/sim_fip35/data/2F21_bead.csv \
        --frames runs/sim_fip35/hot/T390/bb_frames.npy --stride 1 --save_dir runs/sim_fip35/fig4/rib/T390

Writes energy.npy (per scored frame, frame index = stride * k), energy_native (in meta.json).
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..')))
sys.path.insert(0, _HERE)
import options  # noqa: E402
from md_eval_bb import load_backbone_bead  # noqa: E402
from protein_os import Protein, ProteinBase  # noqa: E402
from utils import ideal_cb, ideal_cb_on, test_setup  # noqa: E402


def main():
    parser = options.get_fold_parser()
    parser.add_argument('--native_bead', required=True)
    parser.add_argument('--frames', required=True)
    parser.add_argument('--stride', type=int, default=1)
    args = options.parse_args_and_arch(parser)
    os.makedirs(args.save_dir, exist_ok=True)

    device, model, energy_fn, _ = test_setup(args)
    energy_fn.residue_sum = True
    ProteinBase.dynamic_rama = True
    ProteinBase.v2_fast = True
    seq, profile, X0, gnum = load_backbone_bead(args.native_bead, device)
    prot = Protein(seq, X0[3].clone(), profile, n_coords=X0[0].clone(), ca_coords=X0[1].clone(),
                   c_coords=X0[2].clone(), chain_group_num=gnum)

    def energy(X):
        cb = ideal_cb(X, seq) if ideal_cb_on() else X[3]
        prot.coords, prot.n_coords, prot.ca_coords, prot.c_coords = cb, X[0], X[1], X[2]
        return float(prot.get_energy(energy_fn))

    fr = np.load(args.frames)[::args.stride]
    assert fr.shape[1:] == tuple(X0.shape), f'frames {fr.shape[1:]} vs native bead {tuple(X0.shape)}'
    t0 = time.time()
    with torch.no_grad():
        e_nat = energy(X0)
        E = []
        for k, x in enumerate(fr):
            E.append(energy(torch.tensor(x, dtype=torch.float32, device=device)))
            if (k + 1) % 500 == 0:
                print(f'[score] {k + 1}/{len(fr)} {time.time() - t0:.0f}s', flush=True)
    np.save(os.path.join(args.save_dir, 'energy.npy'), np.array(E))
    with open(os.path.join(args.save_dir, 'meta.json'), 'w') as fh:
        json.dump(dict(frames=args.frames, stride=args.stride, n=len(E), energy_native=e_nat,
                       load_exp=args.load_exp, native_bead=args.native_bead, elapsed_sec=time.time() - t0), fh, indent=2)
    print(f'[score] DONE {len(E)} frames, E_native={e_nat:.1f} -> {args.save_dir}')


if __name__ == '__main__':
    main()
