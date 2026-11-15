"""4-bead (N, CA, C, CB) Langevin MD for backbone-aware NNEF models (ribbon).

The ribbon model builds each block in the central residue's intra-residue
N-CA-C frame, adds blocks centred on the 4 terminal residues, and has a
Ramachandran head. None of that can act in CB-only MD (md_eval.py silently
fell back to the legacy frame). Here all four backbone beads move:

    E(X) = E_nnef(N, CA, C, CB)                      (v2 frame, all blocks, live phi/psi)
         + k_bond * sum_d (|d(X)| - d0)^2             (bonded prior)

The bonded prior holds each residue rigid (all 6 intra-residue distances of
N, CA, C, CB) and each peptide unit rigid (C_i-N_i+1, CA_i-N_i+1, C_i-CA_i+1,
CA_i-CA_i+1), with reference lengths taken from the starting native. The only
soft degrees of freedom left are the backbone (phi, psi) rotations -- the
classic rigid-peptide-plane model. RMSD/RMSF are reported on CB, so outputs are
directly comparable with md_eval.py (same meta.json / csv layout).

    python nnef/scripts/md_eval_bb.py <ribbon model flags> --target 3KXT \
        --native_bead nnef/data/yang_small/3KXT/3KXT_bead.csv --L 200000 \
        --lr 3e-4 --T_max 3e-4 --save_dir eval/md_eval/bb/3KXT_seed0
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_NNEF = os.path.abspath(os.path.join(_HERE, '..'))
sys.path.insert(0, _NNEF)
sys.path.insert(0, _HERE)
import options  # noqa: E402
from md_eval import _compute_rmsd_series, _compute_rmsf, _rg  # noqa: E402
from paths import data_path  # noqa: E402
from physics.mala import MALA, configurational_temperature, energy_and_grad  # noqa: E402
from protein_os import Protein, ProteinBase  # noqa: E402
from utils import ideal_cb, ideal_cb_on, test_setup  # noqa: E402

ATOMS = ('n', 'ca', 'c', 'cb')


def load_backbone_bead(path, device):
    """seq (1-letter array), profile ids, X (4, L, 3) in N/CA/C/CB order, group_num."""
    aa = pd.read_csv(data_path('amino_acids.csv'))
    vocab = {x: y - 1 for x, y in zip(aa.AA, aa.idx)}
    vocab3 = {x.upper(): y - 1 for x, y in zip(aa.AA3C, aa.idx)}
    df = pd.read_csv(path).sort_values('group_num')
    seq = df['group_name'].values
    ids = [vocab[s] if len(s) == 1 else vocab3[s] for s in seq]
    X = np.stack([df[[f'x{a}', f'y{a}', f'z{a}']].values for a in ATOMS]).astype(np.float32)
    if not np.isfinite(X).all():
        raise ValueError(f'{path}: missing backbone atoms (NaN)')
    if ideal_cb_on():
        X[3] = ideal_cb(X, seq)
    return (seq, torch.tensor(ids, device=device),
            torch.tensor(X, device=device), torch.tensor(df['group_num'].values, device=device))


def bond_pairs(L):
    """(atom_a, res_a, atom_b, res_b) index lists for the rigid-residue/peptide prior."""
    pairs = []
    for i in range(L):
        for a in range(4):
            for b in range(a + 1, 4):
                pairs.append((a, i, b, i))                       # intra-residue, 6 pairs
        if i + 1 < L:
            # C_i-N_i+1, CA_i-N_i+1, C_i-CA_i+1, CA_i-CA_i+1 (peptide unit)
            pairs += [(2, i, 0, i + 1), (1, i, 0, i + 1), (2, i, 1, i + 1), (1, i, 1, i + 1)]
    p = np.array(pairs)
    return [torch.tensor(p[:, j]) for j in range(4)]


def main():
    parser = options.get_fold_parser()
    parser.add_argument('--target', required=True)
    parser.add_argument('--native_bead', required=True)
    parser.add_argument('--k_bond', type=float, default=100.0)
    parser.add_argument('--sampler', choices=['langevin', 'mala', 'descent', 'minimize', 'adam'], default='langevin',
                        help="langevin = legacy x - lr*g + T_max*xi (kT = T_max^2/(2 lr), not exact); "
                             "mala = Metropolis-adjusted Langevin at --kT (exact Boltzmann); "
                             "descent = noise-free gradient descent at step --lr (T = 0 test for "
                             "spurious minima, temperature-independent); minimize = L-BFGS "
                             "(strong-Wolfe line search) to the nearest local minimum, --L iterations, "
                             "a frame logged every --trj_log_interval iterations; adam = T = 0 descent "
                             "with Adam (step ~ --lr in A per coordinate), robust to the energy's "
                             "discontinuities at neighbour-set switches where L-BFGS line search stalls")
    parser.add_argument('--kT', type=float, default=1.0, help='MALA temperature (summed-energy units)')
    parser.add_argument('--mala_eps', type=float, default=1e-4, help='initial MALA step (adapted in warm-up)')
    parser.add_argument('--warmup', type=int, default=2000, help='MALA warm-up steps (step adaptation, frames discarded)')
    parser.add_argument('--energy_sum', type=int, default=None,
                        help='1 = summed (extensive) NN energy, 0 = per-block mean. Default: 1 for '
                             'mala/descent, 0 for legacy langevin.')
    parser.add_argument('--hybrid_term_exp', default=None,
                        help='hybrid energy: a SECOND model (run dir) scores only the terminal blocks '
                             '(first/last two residues) while the main model scores the rest; the two '
                             'energies are summed. Each model keeps its own frame/k/smoothing flags.')
    parser.add_argument('--hybrid_term_flags', default='',
                        help='model flags for --hybrid_term_exp (as on its own command line)')
    parser.add_argument('--probe_path', type=int, default=0,
                        help='>0: no dynamics; evaluate energy and gradient at this many points on a '
                             'straight path from the start to a random 1 A perturbation and report '
                             'non-finite gradients and the largest energy steps (continuity check)')
    parser.add_argument('--ref_bead', default=None,
                        help='optional reference structure (e.g. the NMR native when starting from a '
                             'predicted model): adds rmsd_to_ref (Cb, residues matched by group_num)')
    parser.add_argument('--chain_break_max', type=float, default=4.2,
                        help='refuse chains whose CA(i)-CA(i+1) exceeds this (gaps)')
    parser.add_argument('--save_frames', type=int, default=0,
                        help='>0: also save the full (N,CA,C,CB) backbone every this many steps (and the '
                             'last step) to frames.npy (n,4,L,3) + frames_step.npy; 0 = off')
    parser.add_argument('--mala_fast_log', type=int, default=0,
                        help='1 = MALA only: skip the extra per-step energy re-evaluation and evaluate '
                             'E_nn only on logging / frame-saving steps (best-energy tracking then uses '
                             'those steps only). 0 = original behaviour (default).')
    args = options.parse_args_and_arch(parser)
    os.makedirs(args.save_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device, _model, energy_fn, _pb = test_setup(args)
    for p in _model.parameters():
        p.requires_grad_(False)
    if args.energy_sum is None:
        args.energy_sum = 0 if args.sampler == 'langevin' else 1
    energy_fn.residue_sum = bool(args.energy_sum)  # EnergyFun: mean over blocks x n_blocks
    ProteinBase.dynamic_rama = True
    ProteinBase.v2_fast = True  # vectorised v2 blocks (equivalence-tested); reference loop is ~300x slower

    # Hybrid energy: ProteinBase carries each model's block-building switches as
    # class attributes, so snapshot them per model and swap before each evaluation.
    def snapshot():
        return {k: v for k, v in vars(ProteinBase).items()
                if not k.startswith('_') and isinstance(v, (bool, int, float, str))}

    def apply(snap):
        for k, v in snap.items():
            setattr(ProteinBase, k, v)

    snap_main, energy_fn2, snap_term = snapshot(), None, None
    if args.hybrid_term_exp:
        import shlex
        args2 = options.get_local_gen_parser().parse_args(
            shlex.split(args.hybrid_term_flags) + ['--load_exp', args.hybrid_term_exp, '--device', args.device])
        _, model2, energy_fn2, _ = test_setup(args2)
        for p_ in model2.parameters():
            p_.requires_grad_(False)
        energy_fn2.residue_sum = bool(args.energy_sum)
        ProteinBase.dynamic_rama = True
        ProteinBase.v2_fast = True
        ProteinBase.terminal_blocks, ProteinBase.terminal_only = True, True
        snap_term = snapshot()
        apply(snap_main)

    seq, profile, X0, gnum = load_backbone_bead(args.native_bead, device)
    L = X0.shape[1]
    ca_gap = (X0[1, 1:] - X0[1, :-1]).norm(dim=-1).max().item()
    if ca_gap > args.chain_break_max:
        raise SystemExit(f'[md_bb] {args.target}: chain break (max CA-CA {ca_gap:.2f} A)')
    ia, ra, ib, rb = [t.to(device) for t in bond_pairs(L)]
    d0 = (X0[ia, ra] - X0[ib, rb]).norm(dim=-1)

    prot = Protein(seq, X0[3].clone(), profile, n_coords=X0[0].clone(), ca_coords=X0[1].clone(),
                   c_coords=X0[2].clone(), chain_group_num=gnum)

    def energy(X):
        cb = ideal_cb(X, seq) if ideal_cb_on() else X[3]         # ideal CB = virtual site of N/CA/C
        prot.coords, prot.n_coords, prot.ca_coords, prot.c_coords = cb, X[0], X[1], X[2]
        e_nn = prot.get_energy(energy_fn)
        if energy_fn2 is not None:
            apply(snap_term)
            e_nn = e_nn + prot.get_energy(energy_fn2)
            apply(snap_main)
        e_bond = args.k_bond * ((X[ia, ra] - X[ib, rb]).norm(dim=-1) - d0).pow(2).sum()
        return e_nn, e_bond

    if args.probe_path:
        gen = torch.Generator(device='cpu').manual_seed(args.seed)
        D = torch.randn(X0.shape, generator=gen).to(X0)
        D = D / D.pow(2).sum(-1).mean().sqrt()          # ~1 A RMS displacement per atom
        Es, bad = [], 0
        for t in np.linspace(0.0, 1.0, args.probe_path):
            Xt = (X0 + float(t) * D).detach().requires_grad_(True)
            e_nn, _ = energy(Xt)
            (g,) = torch.autograd.grad(e_nn, Xt)
            bad += int(not torch.isfinite(g).all())
            Es.append(float(e_nn))
        dE = np.abs(np.diff(Es))
        top = np.sort(dE)[-5:][::-1]
        print(f'[probe] {args.target} L={L} hybrid={bool(args.hybrid_term_exp)} points={len(Es)} '
              f'nonfinite_grad={bad} median|dE|={np.median(dE):.4g} max|dE|={dE.max():.4g} '
              f'max/median={dE.max() / max(np.median(dE), 1e-12):.1f} top5={np.round(top, 4).tolist()}')
        return

    with torch.no_grad():
        e_nat, eb_nat = energy(X0)
    print(f'[md_bb] {args.target} L={L} E_nn_native={float(e_nat):.3f} sampler={args.sampler} '
          f'energy_sum={args.energy_sum} alpha={args.lr} beta={args.T_max} kT={args.kT} k_bond={args.k_bond}')

    X = X0.clone()
    frames, energies, bond_rmsd = [], [], []
    bb_frames, bb_steps = [], []
    best_e, best_X = float(e_nat), X0[3].clone()
    diverged, t0 = False, time.time()
    chain = None
    conf_g2, conf_lap = 0.0, 0.0

    def u_total(Xv):
        e_nn_, e_bond_ = energy(Xv)
        return e_nn_ + e_bond_

    if args.sampler == 'mala':
        chain = MALA(X0, energy_and_grad(u_total), kT=args.kT, eps=args.mala_eps, warmup=args.warmup)
    adam_opt, x_flat_a = None, None
    if args.sampler == 'adam':
        x_flat_a = X0.reshape(-1).clone().contiguous().requires_grad_(True)
        adam_opt = torch.optim.Adam([x_flat_a], lr=args.lr)
    if args.sampler == 'minimize':
        # LBFGS flattens gradients with .view(-1); optimise a contiguous flat
        # copy and reshape inside the closure (indexed energies give strided grads).
        x_flat = X0.reshape(-1).clone().contiguous().requires_grad_(True)
        Xv = x_flat.view_as(X0)
        opt = torch.optim.LBFGS([x_flat], lr=1.0, max_iter=args.trj_log_interval, history_size=50,
                                line_search_fn='strong_wolfe', tolerance_grad=1e-6, tolerance_change=1e-10)

        def closure():
            opt.zero_grad()
            u = u_total(x_flat.view_as(X0))
            u.backward()
            return u
        for it in range(max(1, args.L // args.trj_log_interval)):
            opt.step(closure)
            with torch.no_grad():
                e_nn, e_bond = energy(Xv.detach())
            if not torch.isfinite(e_nn):
                diverged = True
                break
            e = float(e_nn)
            if e < best_e:
                best_e, best_X = e, Xv[3].detach().clone()
            frames.append(Xv[3].detach().cpu().clone())
            energies.append(e)
            with torch.no_grad():
                bond_rmsd.append(float(((Xv[ia, ra] - Xv[ib, rb]).norm(dim=-1) - d0).pow(2).mean().sqrt()))
            print(f'Iter:{(it + 1) * args.trj_log_interval}, E_nn:{e:.2f}, E_bond:{float(e_bond):.3f}, '
                  f'bond_rmsd:{bond_rmsd[-1]:.4f}')
        args.n_minimize_iter, args.L = args.L, 0  # keep the request; skip the stochastic loop
    for step in range(args.L):
        if chain is not None:
            chain.step()
            X = chain.x
            if step < args.warmup:
                continue  # warm-up frames are not part of the trajectory
            if args.mala_fast_log and not (step % args.trj_log_interval == 0 or (
                    args.save_frames and (step % args.save_frames == 0 or step == args.L - 1))):
                continue  # nothing is logged or saved at this step
            with torch.no_grad():
                e_nn, e_bond = energy(X)
        elif adam_opt is not None:
            adam_opt.zero_grad()
            X = x_flat_a.view_as(X0)
            e_nn, e_bond = energy(X)
            (e_nn + e_bond).backward()
            if not torch.isfinite(x_flat_a.grad).all():
                diverged = True
                print(f'[md_bb] DIVERGED: non-finite gradient at step {step}')
                break
            adam_opt.step()
            X = x_flat_a.detach().view_as(X0)
        else:
            X = X.detach().requires_grad_(True)
            e_nn, e_bond = energy(X)
            (g,) = torch.autograd.grad(e_nn + e_bond, X)
            if not torch.isfinite(g).all():
                diverged = True
                print(f'[md_bb] DIVERGED: non-finite gradient at step {step}')
                break
            with torch.no_grad():
                noise = 0.0 if args.sampler == 'descent' else args.T_max * torch.randn_like(X)
                X = X - args.lr * g + noise
        e = float(e_nn)
        if chain is not None and step % (args.trj_log_interval * 10) == 0:
            a_, b_ = configurational_temperature(u_total, X)
            conf_g2, conf_lap = conf_g2 + a_, conf_lap + b_
        if e < best_e:
            best_e, best_X = e, X[3].detach().clone()
        if args.save_frames and (step % args.save_frames == 0 or step == args.L - 1):
            bb_frames.append(X.detach().cpu().numpy().copy())
            bb_steps.append(step)
        if step % args.trj_log_interval == 0:
            frames.append(X[3].detach().cpu().clone())
            energies.append(e)
            with torch.no_grad():
                bond_rmsd.append(float(((X[ia, ra] - X[ib, rb]).norm(dim=-1) - d0).pow(2).mean().sqrt()))
            if step % (args.trj_log_interval * 50) == 0:
                acc = f', acc:{chain.acceptance:.2f}, eps:{chain.eps:.2e}' if chain is not None else ''
                print(f'Step:{step}, E_nn:{e:.2f}, E_bond:{float(e_bond):.3f}, '
                      f'bond_rmsd:{bond_rmsd[-1]:.4f}{acc}')
    elapsed = time.time() - t0

    native_cb = X0[3].detach().cpu()
    X_end = Xv if args.sampler == 'minimize' else X
    np.save(os.path.join(args.save_dir, 'final_bb.npy'), X_end.detach().cpu().numpy())  # (4,L,3) N,CA,C,CB
    if bb_frames:
        np.save(os.path.join(args.save_dir, 'frames.npy'), np.stack([X0.detach().cpu().numpy()] + bb_frames))
        np.save(os.path.join(args.save_dir, 'frames_step.npy'), np.array([-1] + bb_steps))  # -1 = native start
    sample = [native_cb, best_X.cpu(), native_cb.clone()] + frames
    xyz = torch.stack(sample).numpy()
    rmsd = _compute_rmsd_series(xyz, xyz[0])
    n_logged = len(frames)
    cols = {
        'label': ['native', 'best', 'init'] + [f'frame_{i}' for i in range(n_logged)],
        'step': [-1, -1, 0] + [i * args.trj_log_interval for i in range(n_logged)],
        'energy': [float(e_nat), best_e, float(e_nat)] + energies,
        'rmsd_to_native': rmsd, 'rmsd_to_start': rmsd,
        'rg': [_rg(f) for f in xyz],
    }
    ref_meta = {}
    if args.ref_bead:
        _, _, Xr, gr = load_backbone_bead(args.ref_bead, 'cpu')
        g0 = gnum.cpu().numpy() if torch.is_tensor(gnum) else np.asarray(gnum)
        gr = gr.cpu().numpy() if torch.is_tensor(gr) else np.asarray(gr)
        common = np.intersect1d(g0, gr)
        i0 = np.searchsorted(g0, common) if np.all(np.diff(g0) > 0) else np.array([np.where(g0 == c)[0][0] for c in common])
        ir = np.array([np.where(gr == c)[0][0] for c in common])
        ref_cb = Xr[3].numpy()[ir]
        rr = _compute_rmsd_series(xyz[:, i0], ref_cb)
        cols['rmsd_to_ref'] = rr
        ref_meta = dict(ref_bead=args.ref_bead, n_ref_residues=int(len(common)),
                        initial_rmsd_to_ref=float(rr[0]),
                        final_rmsd_to_ref=float(rr[3:][-max(1, n_logged // 5):].mean()) if n_logged >= 5 else float(rr[-1]))
    pd.DataFrame(cols).to_csv(os.path.join(args.save_dir, f'{args.target}_energy_rmsd.csv'), index=False)
    if n_logged >= 2:
        pd.DataFrame({'residue': np.arange(L), 'rmsf': _compute_rmsf(xyz[3:])}).to_csv(
            os.path.join(args.save_dir, f'{args.target}_rmsf.csv'), index=False)
    tail = max(1, n_logged // 5)
    final = float(rmsd[3:][-tail:].mean()) if n_logged >= 5 else float(rmsd[-1])
    meta = dict(target=args.target, md_mode='native', x_type='backbone4',
                L=getattr(args, 'n_minimize_iter', args.L), lr=args.lr,
                t_noise=args.T_max, k_bond=args.k_bond, trj_log_interval=args.trj_log_interval,
                seed=args.seed, load_exp=args.load_exp, native_bead=args.native_bead, n_residues=L, n_logged_frames=n_logged,
                energy_native=float(e_nat), energy_best=best_e, final_rmsd_to_native=final,
                final_rmsd_to_start=final, rg_native=float(_rg(xyz[0])),
                max_bond_rmsd=float(max(bond_rmsd)) if bond_rmsd else None,
                sampler=args.sampler, kT=args.kT if args.sampler == 'mala' else
                (0.0 if args.sampler in ('descent', 'minimize', 'adam') else args.T_max ** 2 / (2 * args.lr)),
                energy_sum=bool(args.energy_sum),
                mala_acceptance=chain.acceptance if chain is not None else None,
                mala_eps=chain.eps if chain is not None else None,
                kT_configurational=(conf_g2 / conf_lap) if conf_lap > 0 else None,
                elapsed_sec=elapsed, diverged=diverged, **ref_meta)
    with open(os.path.join(args.save_dir, 'meta.json'), 'w') as fh:
        json.dump(meta, fh, indent=2)
    print(f'[md_bb] DONE {args.target} final_rmsd={final:.2f}A energy_best={best_e:.2f} '
          f'max_bond_rmsd={meta["max_bond_rmsd"]} elapsed={elapsed:.0f}s -> {args.save_dir}')


if __name__ == '__main__':
    main()
