"""Torsion-space replica-exchange MC for backbone NNEF models (see physics/torsion_remc.py).

Same model flags as md_eval_bb.py. Energies are SUMMED over blocks (extensive),
so kT is in each model's summed-energy units, as for MALA.

    python nnef/scripts/sim_remc.py <model flags> --target 2MU9 \
        --native_bead nnef/data/decoys/pep_nmr/2MU9/2MU9_native_bead.csv \
        --kT_min 3 --kT_max 60 --n_rep 12 --n_sweeps 2000 --save_dir runs/sim_remc/...

Outputs (save_dir): traj.npz with
    frames (R, F, 4, L, 3) float32  -- configuration at each TEMPERATURE, every --save_every sweeps
    frame_sweep (F,), kT (R,), E (R, S) energy per sweep, walker (R, S) walker id per temperature,
    native (4, L, 3), start (4, L, 3)
and meta.json (acceptance per move type, swap rates, step sizes, timing).
"""
from __future__ import annotations

import json
import math
import os
import pickle
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..')))
sys.path.insert(0, _HERE)
import options  # noqa: E402
from md_eval_bb import load_backbone_bead  # noqa: E402
from physics.torsion_remc import REMC, HardSphere, Torsions  # noqa: E402
from protein_os import Protein, ProteinBase  # noqa: E402
from utils import test_setup  # noqa: E402


def _cat(xs):
    return None if xs[0] is None else torch.cat(xs, 0)


def make_batch_energy(prots, energy_fn, device):
    """list of (4, L, 3) numpy -> summed NNEF energy per structure, ONE network pass."""
    ce = energy_fn.energy_fn

    def batch_energy(Xs):
        parts, sizes = [], []
        with torch.no_grad():
            for prot, X in zip(prots, Xs):
                Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
                prot.coords, prot.n_coords, prot.ca_coords, prot.c_coords = Xt[3], Xt[0], Xt[1], Xt[2]
                profile_local, coords_local, start_id, res_counts, g_local = prot.get_local_struct()
                coords_cart, seq_offset, esm_local, dihedral_local = prot._build_extras(coords_local, g_local)
                rama, rama_mask = prot._gather_rama_local(g_local)
                coords_local = prot._local_cartesian_to_radian(coords_local)
                parts.append((profile_local, coords_local, start_id, res_counts, rama, rama_mask,
                              coords_cart, seq_offset, esm_local, dihedral_local))
                sizes.append(coords_local.shape[0])
            cols = list(zip(*parts))
            prev = (ce.per_block, energy_fn.residue_sum, energy_fn.return_loss_terms)
            ce.per_block, energy_fn.residue_sum, energy_fn.return_loss_terms = True, False, False
            try:
                e = energy_fn.forward(_cat(cols[0]), _cat(cols[1]), _cat(cols[2]), _cat(cols[3]),
                                      rama=_cat(cols[4]), rama_mask=_cat(cols[5]),
                                      coords_cart=_cat(cols[6]), seq_offset=_cat(cols[7]),
                                      esm=_cat(cols[8]), dihedral=_cat(cols[9]))
            finally:
                ce.per_block, energy_fn.residue_sum, energy_fn.return_loss_terms = prev
        return np.array([float(s.sum()) for s in torch.split(e.reshape(-1), sizes)])
    return batch_energy


def make_null_energy(seq, kind):
    """Null chains for Fig 4 (same sampler, geometry and hard sphere, no learned energy):
    zero = E 0 everywhere; rama_aa = -sum log p(phi, psi | aa) from the training histogram
    (same convention and smoothing as eval/baselines_pep/baselines_pep.py)."""
    if kind == 'zero':
        return lambda Xs: np.zeros(len(Xs))
    sys.path.insert(0, os.path.join(_HERE, '..', '..', 'eval', 'baselines_pep'))
    from baselines_pep import AA, _bin, load_logp, phi_psi_train_convention
    L2, Lphi, Lpsi = load_logp()
    aa = np.array([AA.index(x) for x in seq])

    def one(X):
        phi, psi = phi_psi_train_convention(X[0], X[1], X[2])
        both = np.isfinite(phi) & np.isfinite(psi)
        bp, bs = _bin(np.nan_to_num(phi)), _bin(np.nan_to_num(psi))
        only_psi, only_phi = ~np.isfinite(phi) & np.isfinite(psi), np.isfinite(phi) & ~np.isfinite(psi)
        return -(L2[aa[both], bp[both], bs[both]].sum() + Lpsi[aa[only_psi], bs[only_psi]].sum()
                 + Lphi[aa[only_phi], bp[only_phi]].sum())
    return lambda Xs: np.array([one(X) for X in Xs])


def main():
    parser = options.get_fold_parser()
    parser.add_argument('--target', required=True)
    parser.add_argument('--native_bead', required=True)
    parser.add_argument('--start_bead', default=None, help='start structure (default: native)')
    parser.add_argument('--start_extended', type=int, default=0,
                        help='1 = start from an extended chain (phi -150, psi 150) built on the native geometry')
    parser.add_argument('--kT_min', type=float, required=True)
    parser.add_argument('--kT_max', type=float, required=True)
    parser.add_argument('--n_rep', type=int, default=12)
    parser.add_argument('--kT_list', default='',
                        help='comma-separated explicit temperatures (overrides kT_min/kT_max/n_rep geometric ladder)')
    parser.add_argument('--n_sweeps', type=int, default=2000, help='1 sweep = 2(L-1) MC steps per replica')
    parser.add_argument('--warmup_sweeps', type=int, default=100, help='step-size adaptation (discarded)')
    parser.add_argument('--swap_every', type=int, default=10, help='MC steps between exchange attempts')
    parser.add_argument('--save_every', type=float, default=1, help='sweeps between saved frames')
    parser.add_argument('--hs_dmin', type=float, default=3.0, help='hard-sphere non-local CA-CA distance (A)')
    parser.add_argument('--checkpoint_sweeps', type=int, default=20,
                        help='write traj.npz + meta_partial.json every this many production sweeps')
    parser.add_argument('--resume', type=int, default=1,
                        help='1 = continue exactly from state.pkl if a checkpoint exists (requeue-safe)')
    parser.add_argument('--check_energy', type=int, default=1,
                        help='1 = assert batched per-block sum == Protein.get_energy (summed) at the start')
    parser.add_argument('--null_energy', default='', choices=['', 'zero', 'rama_aa'],
                        help='replace the model by a null energy (no network, CPU): see make_null_energy')
    args = options.parse_args_and_arch(parser)
    os.makedirs(args.save_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    kTs = np.geomspace(args.kT_min, args.kT_max, args.n_rep)
    if args.kT_list:
        kTs = np.array(sorted(float(x) for x in args.kT_list.split(',')))
        args.n_rep = len(kTs)
    if args.null_energy:
        seq, profile, Xn, gnum = load_backbone_bead(args.native_bead, 'cpu')
        L = Xn.shape[1]
        batch_energy = make_null_energy(seq, args.null_energy)
        args.check_energy = 0
    else:
        device, _model, energy_fn, _pb = test_setup(args)
        for p in _model.parameters():
            p.requires_grad_(False)
        energy_fn.residue_sum = True
        ProteinBase.dynamic_rama = True
        ProteinBase.v2_fast = True

        seq, profile, Xn, gnum = load_backbone_bead(args.native_bead, device)
        L = Xn.shape[1]
        prots = [Protein(seq, Xn[3].clone(), profile, n_coords=Xn[0].clone(), ca_coords=Xn[1].clone(),
                         c_coords=Xn[2].clone(), chain_group_num=gnum) for _ in range(args.n_rep)]
        batch_energy = make_batch_energy(prots, energy_fn, device)

    native = Xn.cpu().numpy().astype(np.float64)
    X0 = native.copy()
    if args.start_bead:
        X0 = load_backbone_bead(args.start_bead, 'cpu')[2].numpy().astype(np.float64)
    if args.start_extended:
        Torsions(L).set_all(X0, phi=math.radians(-150), psi=math.radians(150))

    hs = HardSphere(L, args.hs_dmin)
    if not hs.ok(native):
        hs.d_min = max(0.0, hs.min_dist(native) - 0.2)
        print(f'[remc] native violates hs_dmin; lowered to {hs.d_min:.2f} A')
    if not hs.ok(X0):
        raise SystemExit('[remc] start structure overlaps (hard sphere)')

    if args.check_energy:
        with torch.no_grad():
            e_ref = float(prots[0].get_energy(energy_fn))
        e_bat = batch_energy([native])[0]
        print(f'[remc] energy check: get_energy={e_ref:.4f} batched={e_bat:.4f}')
        assert abs(e_ref - e_bat) <= 1e-3 * max(1.0, abs(e_ref)), 'batched energy != get_energy'
    e_nat = float(batch_energy([native])[0])

    sim = REMC(X0, kTs, batch_energy, hs, rng)
    steps_per_sweep = 2 * (L - 1)
    save_every = max(1, int(round(args.save_every * steps_per_sweep)))
    n_warm = args.warmup_sweeps * steps_per_sweep
    n_prod = args.n_sweeps * steps_per_sweep
    print(f'[remc] {args.target} L={L} E_native={e_nat:.3f} kT={np.round(kTs, 3).tolist()} '
          f'steps/sweep={steps_per_sweep} warmup={n_warm} prod={n_prod}')

    state_path = os.path.join(args.save_dir, 'state.pkl')
    traj_path = os.path.join(args.save_dir, 'traj.npz')
    STATE = ('X', 'E', 'sigma', 'walker', '_win', 'acc', 'swap', 'n_clash')
    resumed = args.resume and os.path.exists(state_path) and os.path.exists(traj_path)
    t0 = time.time()
    for s in range(0 if resumed else n_warm):
        sim.step(adapt=True)
        if (s + 1) % 50 == 0:
            sim.adapt()
        if (s + 1) % args.swap_every == 0:
            sim.exchange(((s + 1) // args.swap_every) % 2)
    if not resumed:
        sim.swap[:] = 0
    t_warm = time.time() - t0

    def dump(frames, fsweep, E_tr, W_tr, final):
        """traj.npz + meta.json (final) or meta_partial.json (checkpoint; the pack runner skips only meta.json)."""
        elapsed = time.time() - t0
        with open(traj_path + '.tmp', 'wb') as fh:  # atomic: a kill mid-write keeps the last good file
            np.savez_compressed(fh, frames=np.stack(frames, 1), frame_sweep=np.array(fsweep), kT=kTs,
                                E=np.array(E_tr).T, walker=np.array(W_tr).T,
                                native=native.astype(np.float32), start=X0.astype(np.float32))
        os.replace(traj_path + '.tmp', traj_path)
        meta = dict(target=args.target, sampler='torsion_remc', L=L, n_rep=args.n_rep, kT=kTs.tolist(),
                    n_sweeps=args.n_sweeps, warmup_sweeps=args.warmup_sweeps, steps_per_sweep=steps_per_sweep,
                    swap_every=args.swap_every, hs_dmin=hs.d_min, start_extended=bool(args.start_extended),
                    start_bead=args.start_bead, native_bead=args.native_bead, load_exp=args.load_exp,
                    seed=args.seed, energy_native=e_nat,
                    acceptance_small=sim.acceptance[:, 0].tolist(), acceptance_shear=sim.acceptance[:, 1].tolist(),
                    sigma_small_deg=np.degrees(sim.sigma[:, 0]).tolist(),
                    sigma_shear_deg=np.degrees(sim.sigma[:, 1]).tolist(),
                    swap_rate=sim.swap_rate.tolist(), n_clash_rejected=sim.n_clash,
                    elapsed_sec=elapsed, warmup_sec=t_warm,
                    sec_per_step=elapsed / max(1, n_warm + len(E_tr) * steps_per_sweep),
                    n_sweeps_done=len(E_tr), complete=final)
        name = 'meta.json' if final else 'meta_partial.json'
        with open(os.path.join(args.save_dir, name), 'w') as fh:
            json.dump(meta, fh, indent=2)

    def save_state(s_next):
        """Everything needed to continue the chain exactly; frames stay in traj.npz (written first)."""
        st = {k: getattr(sim, k) for k in STATE}
        st.update(s_next=s_next, n_frames=len(frames), n_E=len(E_tr), rng=rng.bit_generator.state, t_warm=t_warm)
        with open(state_path + '.tmp', 'wb') as fh:
            pickle.dump(st, fh)
        os.replace(state_path + '.tmp', state_path)

    frames, fsweep, E_tr, W_tr = [], [], [], []
    s0 = 0
    if resumed:
        with open(state_path, 'rb') as fh:
            st = pickle.load(fh)
        for k in STATE:
            setattr(sim, k, st[k])
        rng.bit_generator.state = st['rng']
        s0, t_warm = st['s_next'], st['t_warm']
        with np.load(traj_path) as z:  # traj.npz can be one checkpoint ahead of state.pkl: truncate
            frames = [z['frames'][:, i] for i in range(st['n_frames'])]
            fsweep = z['frame_sweep'][:st['n_frames']].tolist()
            E_tr = [z['E'][:, i] for i in range(st['n_E'])]
            W_tr = [z['walker'][:, i] for i in range(st['n_E'])]
        print(f'[remc] resumed at sweep {s0 / steps_per_sweep:.0f}', flush=True)
    for s in range(s0, n_prod):
        sim.step(adapt=False)
        if (s + 1) % args.swap_every == 0:
            sim.exchange(((s + 1) // args.swap_every) % 2)
        if s % save_every == 0 or s == n_prod - 1:
            frames.append(np.stack(sim.X).astype(np.float32))
            fsweep.append(s / steps_per_sweep)
        if (s + 1) % steps_per_sweep == 0:
            E_tr.append(sim.E.copy())
            W_tr.append(sim.walker.copy())
        if (s + 1) % (steps_per_sweep * 10) == 0:
            el = time.time() - t0
            print(f'[remc] sweep {(s + 1) // steps_per_sweep}/{args.n_sweeps} '
                  f'E={np.round(sim.E, 1).tolist()} acc={np.round(sim.acceptance.mean(1), 2).tolist()} '
                  f'swap={np.round(sim.swap_rate, 2).tolist()} {el:.0f}s', flush=True)
        if (s + 1) % (steps_per_sweep * args.checkpoint_sweeps) == 0 and s + 1 < n_prod:
            dump(frames, fsweep, E_tr, W_tr, final=False)  # partial results survive a time limit
            save_state(s + 1)

    dump(frames, fsweep, E_tr, W_tr, final=True)
    if os.path.exists(state_path):
        os.remove(state_path)
    print(f'[remc] DONE {args.target} {time.time() - t0:.0f}s swap={np.round(sim.swap_rate, 2).tolist()} '
          f'-> {args.save_dir}')


if __name__ == '__main__':
    main()
