"""Constrained underdamped Langevin dynamics for backbone NNEF models (physics/constrained_langevin.py).

Same model flags as md_eval_bb.py. The energy is SUMMED over blocks, so --kT is in
each model's summed-energy units (use the torsion-REMC calibrated kT*). Rigid
backbone distances are constraints (no bonded prior). Unit masses, reduced time.

    python nnef/scripts/sim_langevin.py <model flags> --target 7JIY \
        --native_bead nnef/data/decoys/pep_nmr/7JIY/7JIY_native_bead.csv \
        --kT 0.82 --dt 0.02 --n_steps 200000 --save_every 500 --save_dir runs/sim_lang/...

Outputs: frames.npy (F, 4, L, 3) with frame 0 = start, frames_step.npy, energy.npy (per saved frame),
meta.json (kinetic temperature, max constraint violation, timing). Partial outputs every --checkpoint frames.
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
from md_eval_bb import bond_pairs, load_backbone_bead  # noqa: E402
from physics.constrained_langevin import ConstrainedBAOAB  # noqa: E402
from physics.mala import energy_and_grad  # noqa: E402
from protein_os import Protein, ProteinBase  # noqa: E402
from utils import ideal_cb, ideal_cb_on, test_setup  # noqa: E402


def main():
    parser = options.get_fold_parser()
    parser.add_argument('--target', required=True)
    parser.add_argument('--native_bead', required=True)
    parser.add_argument('--kT', type=float, required=True)
    parser.add_argument('--dt', type=float, default=0.02)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--n_steps', type=int, default=100000)
    parser.add_argument('--save_every', type=int, default=500)
    parser.add_argument('--checkpoint', type=int, default=20, help='write partial outputs every N saved frames')
    parser.add_argument('--wall_k', type=float, default=0.0,
                        help='excluded-volume wall k*(d0-d)^2 on non-local CA-CA (|i-j|>=3) closer than --wall_d0, '
                             'k in units of kT per A^2 (0 = off). Mirrors the hard sphere of sim_remc --hs_dmin.')
    parser.add_argument('--wall_d0', type=float, default=3.0)
    parser.add_argument('--metropolis', type=int, default=0,
                        help='>0: generalized HMC -- this many RATTLE steps per proposal, then a Metropolis test on '
                             'U + KE (exact sampling of exp(-U/kT)); 0: plain BAOAB. --n_steps counts proposals.')
    args = options.parse_args_and_arch(parser)
    os.makedirs(args.save_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    gen = torch.Generator(device='cpu').manual_seed(args.seed)

    device, model, energy_fn, _ = test_setup(args)
    for p in model.parameters():
        p.requires_grad_(False)
    energy_fn.residue_sum = True
    ProteinBase.dynamic_rama = True
    ProteinBase.v2_fast = True
    seq, profile, X0, gnum = load_backbone_bead(args.native_bead, device)
    L = X0.shape[1]
    prot = Protein(seq, X0[3].clone(), profile, n_coords=X0[0].clone(), ca_coords=X0[1].clone(),
                   c_coords=X0[2].clone(), chain_group_num=gnum)

    wi, wj = torch.triu_indices(L, L, offset=3, device=device)

    def energy(X):
        cb = ideal_cb(X, seq) if ideal_cb_on() else X[3]  # ideal Cb = virtual site of N/CA/C, as in md_eval_bb
        prot.coords, prot.n_coords, prot.ca_coords, prot.c_coords = cb, X[0], X[1], X[2]
        u = prot.get_energy(energy_fn)
        if args.wall_k > 0:
            d = (X[1, wi] - X[1, wj]).norm(dim=-1)
            u = u + args.wall_k * args.kT * torch.relu(args.wall_d0 - d).pow(2).sum()
        return u

    ug = energy_and_grad(energy)
    sim = ConstrainedBAOAB(X0.cpu(), lambda x: tuple(t.cpu() for t in ug(x.to(device))),
                           bond_pairs(L), kT=args.kT, dt=args.dt, gamma=args.gamma, generator=gen)
    frames, steps, energies, temps = [X0.cpu().numpy()], [0], [float(sim.u)], []
    t0 = time.time()
    diverged = False

    def dump(final):
        np.save(os.path.join(args.save_dir, 'frames.npy'), np.stack(frames))
        np.save(os.path.join(args.save_dir, 'frames_step.npy'), np.array(steps))
        np.save(os.path.join(args.save_dir, 'energy.npy'), np.array(energies))
        np.save(os.path.join(args.save_dir, 'kinetic_T.npy'), np.array(temps))  # every 10 steps
        meta = dict(target=args.target, sampler='constrained_baoab', L=L, kT=args.kT, dt=args.dt,
                    gamma=args.gamma, n_steps=args.n_steps, wall_k=args.wall_k, wall_d0=args.wall_d0, steps_done=steps[-1], save_every=args.save_every,
                    seed=args.seed, load_exp=args.load_exp, native_bead=args.native_bead,
                    n_constraints=int(len(sim.a)), n_virtual_gly=int(len(sim.v_cb)),
                    kinetic_T_mean=float(np.mean(temps)) if temps else None,
                    kinetic_T_sd=float(np.std(temps)) if temps else None,
                    max_constraint_violation=sim.max_violation(), max_shake_iter=sim.max_shake_iter,
                    metropolis=args.metropolis, accept_rate=sim.n_accept / max(sim.n_step, 1) if args.metropolis else None,
                    diverged=diverged, elapsed_sec=time.time() - t0, complete=final)
        with open(os.path.join(args.save_dir, 'meta.json' if final else 'meta_partial.json'), 'w') as fh:
            json.dump(meta, fh, indent=2)

    for s in range(1, args.n_steps + 1):
        u = sim.step_ghmc(args.metropolis) if args.metropolis else sim.step()
        if not torch.isfinite(torch.as_tensor(u)) or not torch.isfinite(sim.x).all():
            diverged = True
            print(f'[lang] DIVERGED at step {s}', flush=True)
            break
        if s % 10 == 0:
            temps.append(sim.kinetic_temperature())
        if s % args.save_every == 0:
            frames.append(sim.X.numpy())
            steps.append(s)
            energies.append(float(u))
            if len(frames) % 20 == 0:
                print(f'[lang] step {s} E={float(u):.2f} T_kin={np.mean(temps[-200:]):.3f} '
                      f'acc={sim.n_accept / sim.n_step:.3f} '
                      f'viol={sim.max_violation():.1e} {time.time() - t0:.0f}s', flush=True)
            if len(frames) % args.checkpoint == 0:
                dump(final=False)
    dump(final=True)
    print(f'[lang] DONE {args.target} steps={steps[-1]} T_kin={np.mean(temps):.3f} (target {args.kT}) '
          f'{time.time() - t0:.0f}s -> {args.save_dir}')


if __name__ == '__main__':
    main()
