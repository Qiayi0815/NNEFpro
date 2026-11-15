"""Test whether a backbone NNEF energy is continuous: fine line scans from saved frames.

Along x(s) = x0 + s * v (v random, unit RMS per atom, rigid-geometry not enforced; s up to --s_max A),
compare each step's energy change dE with the gradient prediction g . dx. Jumps (|dE - g.dx| > --jump_kT * kT)
are energy discontinuities that gradient-based dynamics cannot see (e.g. neighbour-set switches).

    python nnef/scripts/energy_jump_scan.py <model flags> --native_bead ... --frames runs/.../frames.npy \
        --kT 0.82 --n_frames 5 --n_dirs 4 --s_max 0.3 --n_s 600 --save_dir runs/sim_lang/jumps/rib_7JIY
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..')))
sys.path.insert(0, _HERE)
import options  # noqa: E402
from md_eval_bb import load_backbone_bead  # noqa: E402
from physics.mala import energy_and_grad  # noqa: E402
from protein_os import Protein, ProteinBase  # noqa: E402
from utils import test_setup  # noqa: E402


def main():
    parser = options.get_fold_parser()
    parser.add_argument('--native_bead', required=True)
    parser.add_argument('--frames', required=True, help='frames.npy (F, 4, L, 3); frame 0 = native')
    parser.add_argument('--kT', type=float, required=True)
    parser.add_argument('--n_frames', type=int, default=5)
    parser.add_argument('--n_dirs', type=int, default=4)
    parser.add_argument('--s_max', type=float, default=0.3)
    parser.add_argument('--n_s', type=int, default=600)
    parser.add_argument('--jump_kT', type=float, default=0.5)
    args = options.parse_args_and_arch(parser)
    os.makedirs(args.save_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    device, model, energy_fn, _ = test_setup(args)
    for p in model.parameters():
        p.requires_grad_(False)
    energy_fn.residue_sum = True
    ProteinBase.dynamic_rama = True
    ProteinBase.v2_fast = True
    seq, profile, X0, gnum = load_backbone_bead(args.native_bead, device)
    prot = Protein(seq, X0[3].clone(), profile, n_coords=X0[0].clone(), ca_coords=X0[1].clone(),
                   c_coords=X0[2].clone(), chain_group_num=gnum)

    def energy(X):
        prot.coords, prot.n_coords, prot.ca_coords, prot.c_coords = X[3], X[0], X[1], X[2]
        return prot.get_energy(energy_fn)

    ug = energy_and_grad(energy)
    fr = np.load(args.frames)
    idx = np.unique(np.linspace(0, len(fr) - 1, args.n_frames).astype(int))
    gly = np.linalg.norm(fr[0, 3] - fr[0, 1], axis=-1) < 1e-3
    s = np.linspace(0, args.s_max, args.n_s + 1)
    out = []
    for k in idx:
        x0 = torch.tensor(fr[k], dtype=torch.float32, device=device)
        for _ in range(args.n_dirs):
            v = rng.standard_normal(fr[k].shape)
            v[3, gly] = v[1, gly]                      # Gly CB rides on CA
            v /= np.sqrt((v ** 2).sum(-1).mean())
            v = torch.tensor(v, dtype=torch.float32, device=device)
            E, gv = [], []
            for si in s:
                u, g = ug(x0 + float(si) * v)
                E.append(float(u)); gv.append(float((g * v).sum()))
            E, gv = np.array(E), np.array(gv)
            ds = s[1] - s[0]
            resid = np.diff(E) - 0.5 * (gv[1:] + gv[:-1]) * ds    # trapezoid gradient prediction
            jumps = np.abs(resid) > args.jump_kT * args.kT
            out.append(dict(frame=int(k), n_jumps=int(jumps.sum()), max_jump=float(np.abs(resid).max()),
                            sum_abs_jump=float(np.abs(resid[jumps]).sum()), E_range=float(E.max() - E.min()),
                            E=E.tolist(), gv=gv.tolist()))
            print(f'[scan] frame {k}: jumps>{args.jump_kT}kT = {jumps.sum()} over {args.s_max} A, '
                  f'max |dE - g.dx| = {np.abs(resid).max():.2f} (kT={args.kT}), E range {E.max() - E.min():.1f}',
                  flush=True)
    meta = dict(frames=args.frames, kT=args.kT, s_max=args.s_max, n_s=args.n_s, ds=float(s[1] - s[0]),
                jump_kT=args.jump_kT, scans=out)
    with open(os.path.join(args.save_dir, 'scan.json'), 'w') as fh:
        json.dump(meta, fh)
    nj = np.array([o['n_jumps'] for o in out]); mj = np.array([o['max_jump'] for o in out])
    print(f'[scan] DONE {len(out)} scans: jumps per {args.s_max} A median {np.median(nj):.0f}, '
          f'max jump median {np.median(mj):.2f} = {np.median(mj) / args.kT:.1f} kT')


if __name__ == '__main__':
    main()
