#!/usr/bin/env python3
"""Dump what the network sees and predicts for every block of one chain (figure data for fig_model.py).

One forward pass, CPU. Inputs are captured inside LocalEnergyCE.forward, i.e. after the smooth
neighbour layout, so slots, weights and offsets are exactly those of the scored energy.

    NNEF_CB=ideal python -s nnef/scripts/dump_block_mixture.py <model flags as in sim_remc.py> \
        --device cpu --csv 1PSV_medoid_backbone.csv --out 1PSV_block_dump.npz

--csv: columns res, aa, x/y/z for n, ca, c, cb (paper_plan/figures/prep_1psv_geometry.py).
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..')))
sys.path.insert(0, _HERE)
import options  # noqa: E402
import model.local_ss as ls  # noqa: E402
from paths import data_path  # noqa: E402
from protein_os import Protein, ProteinBase  # noqa: E402
from utils import test_setup  # noqa: E402


def main():
    parser = options.get_fold_parser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--out', required=True)
    args = options.parse_args_and_arch(parser)
    device, model, energy_fn, _ = test_setup(args)
    ProteinBase.dynamic_rama = True
    ProteinBase.v2_fast = True

    df = pd.read_csv(args.csv)
    aa = pd.read_csv(data_path('amino_acids.csv'))
    vocab = {x: y - 1 for x, y in zip(aa.AA, aa.idx)}
    seq = df.aa.values
    ids = torch.tensor([vocab[s] for s in seq])
    X = torch.tensor(np.stack([df[[f'x{a}', f'y{a}', f'z{a}']].values for a in ('n', 'ca', 'c', 'cb')]).astype(np.float32))
    prot = Protein(seq, X[3].clone(), ids, n_coords=X[0].clone(), ca_coords=X[1].clone(), c_coords=X[2].clone(),
                   chain_group_num=torch.tensor(df.res.values))

    cap = {}
    ce = energy_fn.energy_fn

    orig_layout = ls.smooth_layout

    def layout(coords_cart, offset, tau, *a, **k):
        perm, s = orig_layout(coords_cart, offset, tau, *a, **k)
        cap['perm'], cap['offset_raw'], cap['cart_raw'] = perm, offset, coords_cart
        return perm, s
    ls.smooth_layout = layout

    def hook(_m, a, kw, out):
        cap['seq_full'], cap['coords_in'] = kw['seq_full'], a[1]
        cap['nbr_s'], cap['nbr_bucket'] = kw['nbr_s'], kw['nbr_bucket']
        cap['out_x'], cap['out_s'] = out
    ce.model.register_forward_hook(hook, with_kwargs=True)

    orig_loss = ce.get_mixture_loss

    def loss(out, target_coords, rama=None, rama_mask=None, w=None):
        cap['target_coords'], cap['rama'], cap['rama_mask'] = target_coords, rama, rama_mask
        cap['coef'] = ce.get_mixture_coef(out)
        return orig_loss(out, target_coords, rama=rama, rama_mask=rama_mask, w=w)
    ce.get_mixture_loss = loss

    with torch.no_grad():
        E_block, g_local = prot.get_block_energies(energy_fn)
        cap_b = dict(cap)
        E_tot, terms = prot.get_energy(energy_fn, return_terms=True)
    cap = cap_b
    g_perm = ls.permute(g_local, cap['perm'])
    names = ('r_pi', 'r_mu', 'r_sigma', 'a_pi', 'a_mu1', 'a_mu2', 'a_s1', 'a_s2', 'a_rho',
             'rama_pi', 'rama_mu_phi', 'rama_mu_psi', 'rama_s_phi', 'rama_s_psi', 'rama_rho')
    out = {n: c.cpu().numpy() for n, c in zip(names, cap['coef'])}
    out.update(
        g=g_perm.cpu().numpy(), g_raw=g_local.cpu().numpy(), perm=cap['perm'].cpu().numpy(),
        offset=ls.permute(cap['offset_raw'], cap['perm']).cpu().numpy(), bucket=cap['nbr_bucket'].cpu().numpy(),
        cart=ls.permute(cap['cart_raw'], cap['perm']).cpu().numpy(),
        coords=cap['coords_in'].cpu().numpy(), target_coords=cap['target_coords'].cpu().numpy(),
        s=cap['nbr_s'].cpu().numpy(), seq_slots=cap['seq_full'].cpu().numpy(),
        rama=cap['rama'].cpu().numpy(), rama_mask=cap['rama_mask'].cpu().numpy(),
        p_id=F.softmax(cap['out_s'][:, 0], dim=-1).cpu().numpy(),
        E_block=E_block.cpu().numpy(), E_total=float(E_tot), terms=np.array([float(t) for t in terms]),
        aa_order=np.array(aa.AA.values, dtype=str), seq=np.array(seq, dtype=str),
        profile_weight=float(energy_fn.profile_weight),
    )
    np.savez(args.out, **out)
    print('blocks', out['g'].shape, 'E_total', out['E_total'], 'sum E_block', out['E_block'].sum())
    print('terms (r, angle, profile, start_id, res_counts, rama):', out['terms'])
    b = int(np.where(out['g'][:, 0] == 17)[0][0])
    print('block of residue 18: slots', (out['g'][b] + 1).tolist())
    print('offsets', out['offset'][b].tolist())
    print('s', np.round(out['s'][b], 3).tolist())


if __name__ == '__main__':
    main()
