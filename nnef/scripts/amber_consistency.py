"""Score all-atom Amber14SB frames (projected to Cb) with the learned energy (T3.1).

Positive control for the self-consistency test: states that real physics
visits at 300 K. For each profile weight w we report, vs the native:
  * Delta E per term for Amber frames (thermal native-basin states),
  * fraction of Amber frames below native, corr(Delta E, RMSD),
  * cosine between the model force -dE/dx and the Amber Cb force (per residue,
    averaged) -- a force-matching style check of whether re-weighting makes the
    learned forces more physical.

Inputs: eval/md_eval/amber14sb_ref/<PDB>/cb_forces.npz (cb_coords, cb_forces)
and the native bead nnef/data/yang_small/<PDB>/<PDB>_bead.csv.

    python nnef/scripts/amber_consistency.py <model flags> --targets 3KXT 5ZGM 6H8O \
        --weights 1.0 0.3 0.1 --out eval/decomp/amber_consistency_yang.csv
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

_NNEF = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _NNEF)
import options  # noqa: E402
from protein_os import Protein  # noqa: E402
from train_contrastive import _kabsch_rmsd  # noqa: E402
from utils import load_protein_bead, test_setup  # noqa: E402

TERMS = ['loss_r', 'loss_angle', 'loss_profile', 'loss_start_id', 'loss_res_counts', 'loss_rama']


def energy_terms_and_force(prot, x, energy_fn):
    x = x.detach().clone().requires_grad_(True)
    prot.coords = x
    e, terms = prot.get_energy(energy_fn, return_terms=True)
    (g,) = torch.autograd.grad(e, x)
    return float(e), np.array([float(t) for t in terms]), (-g).detach()


def main():
    parser = options.get_local_gen_parser()
    parser.add_argument('--amber_root', default='eval/md_eval/amber14sb_ref')
    parser.add_argument('--bead_root', default='nnef/data/yang_small')
    parser.add_argument('--targets', nargs='+', required=True)
    parser.add_argument('--weights', nargs='+', type=float, default=[1.0, 0.3, 0.1])
    parser.add_argument('--out', required=True)
    args = options.parse_args_and_arch(parser)
    device, model, energy_fn, _pb = test_setup(args)
    for p in model.parameters():
        p.requires_grad_(False)

    rows = []
    for t in args.targets:
        z = np.load(os.path.join(args.amber_root, t, 'cb_forces.npz'))
        seq, native, profile = load_protein_bead(os.path.join(args.bead_root, t, f'{t}_bead.csv'),
                                                 args.mode, device)
        if z['cb_coords'].shape[1] != native.shape[0]:
            print(f'[amber] {t}: length mismatch {z["cb_coords"].shape[1]} vs {native.shape[0]}; skip')
            continue
        prot = Protein(seq, native.clone(), profile)
        for w in args.weights:
            energy_fn.profile_weight = w
            e_nat, t_nat, _ = energy_terms_and_force(prot, native, energy_fn)
            for i in range(z['cb_coords'].shape[0]):
                x = torch.tensor(z['cb_coords'][i], dtype=native.dtype, device=device)
                e, terms, f_model = energy_terms_and_force(prot, x, energy_fn)
                f_amb = torch.tensor(z['cb_forces'][i], dtype=native.dtype, device=device)
                cos = torch.nn.functional.cosine_similarity(f_model, f_amb, dim=-1)
                rows.append(dict(target=t, w=w, frame=i, rmsd=float(_kabsch_rmsd(x, native)),
                                 dE=e - e_nat, force_cos=float(cos.mean()),
                                 **{f'd_{k}': float(v) for k, v in zip(TERMS, terms - t_nat)}))
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    for (t, w), g in df.groupby(['target', 'w']):
        print(f'[amber] {t} w={w:<4} frames={len(g)} RMSD {g.rmsd.mean():.2f}  '
              f'dE mean {g.dE.mean():+7.2f}  frac<native {(g.dE < 0).mean():.2f}  '
              f'corr(dE,RMSD) {g.dE.corr(g.rmsd):+.2f}  d_profile {g.d_loss_profile.mean():+7.2f}  '
              f'd_geom {(g.d_loss_r + g.d_loss_angle).mean():+.2f}  force_cos {g.force_cos.mean():+.3f}')


if __name__ == '__main__':
    main()
