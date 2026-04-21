"""Langevin-MD evaluation driver for NNEF energy models.

Runs short MD trajectories driven by the learned energy (``Dynamics*`` in
``physics/dynamics.py``) and writes a standardised output bundle per run:

    <save_dir>/trajectory.pdb      -- logged frames (via write_pdb_sample2)
    <save_dir>/energy_rmsd.csv     -- per-frame energy, rmsd_to_native,
                                     rmsd_to_start, rg
    <save_dir>/rmsf.csv            -- per-residue RMSF over logged frames
    <save_dir>/meta.json           -- config + summary stats

Modes
-----
* ``--md_mode native`` (default): start at the native bead CSV, measure
  drift (RMSD vs native / vs start, energy vs t, RMSF). Answers "does the
  native sit at a local minimum of the learned energy?"
* ``--md_mode fold``: start from an extended chain (internal coords
  ``(r, theta, phi) = (5.367, 0.1, 0.0)``), measure whether the energy
  folds it back toward native. Hardest test; uses higher noise by default.
* ``--md_mode decoy``: start at a user-supplied decoy bead CSV, measure
  whether dynamics relaxes toward native.

Inputs (paths)
--------------
* ``--decoy_set`` + ``--target``: resolves automatic defaults for the
  start / native bead CSVs under ``data/decoys/<set>/<target>/``.
* ``--init_bead`` and/or ``--native_bead``: explicit overrides.

Model loading mirrors ``decoy_score.py`` / ``fold_one.py`` via
``utils.test_setup`` -- same architecture flags as the existing eval
slurm scripts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch


_HERE = os.path.dirname(os.path.abspath(__file__))
_NNEF_DIR = os.path.abspath(os.path.join(_HERE, '..'))
_REPO_ROOT = os.path.abspath(os.path.join(_NNEF_DIR, '..'))
if _NNEF_DIR not in sys.path:
    sys.path.insert(0, _NNEF_DIR)

import options  # noqa: E402
from paths import data_path  # noqa: E402
from protein_os import Protein  # noqa: E402
from utils import load_protein_bead, load_protein_pdb, test_setup, write_pdb_sample2  # noqa: E402
from physics.dynamics import (  # noqa: E402
    Dynamics,
    DynamicsIntFast,
    DynamicsInternal,
    DynamicsMixed,
    DynamicsMixFast,
)


_SAMPLER_BY_XTYPE = {
    'cart': Dynamics,
    'internal': DynamicsInternal,
    'int_fast': DynamicsIntFast,
    'mixed': DynamicsMixed,
    'mix_fast': DynamicsMixFast,
}


def _resolve_bead_path(explicit: str | None, decoy_set: str, target: str, filename: str) -> str:
    if explicit:
        path = os.path.expanduser(explicit)
    else:
        path = data_path('decoys', decoy_set, target, filename)
    return path


def _load_start_and_native(args, device):
    """Return (seq, coords_init, coords_native, profile).

    ``coords_native`` is used only as an RMSD reference. If the user does
    not provide a native bead CSV and we cannot find one on disk, we fall
    back to coords_init so RMSD is measured relative to the starting
    frame.
    """
    target = args.target

    # Native bead CSV (reference for RMSD).
    native_path = _resolve_bead_path(
        args.native_bead, args.decoy_set, target, f'{target}_bead.csv',
    )
    if os.path.isfile(native_path):
        seq_nat, coords_native, profile_nat = load_protein_bead(native_path, args.mode, device)
        is_proxy = 'TS' in os.path.basename(native_path)
        label = 'native-proxy (top-GDT decoy)' if is_proxy else 'native'
        print(f'[md_eval] {label} bead: {native_path} (L={len(seq_nat)})')
    else:
        seq_nat = None
        coords_native = None
        profile_nat = None
        print(f'[md_eval] WARN: native bead not found at {native_path}; '
              f'RMSD-to-native will equal RMSD-to-start.')

    # Starting structure.
    if args.md_mode == 'native':
        if coords_native is None:
            raise FileNotFoundError(
                f'--md_mode native requires a native bead CSV. Looked at: {native_path}. '
                f'Pass --native_bead PATH explicitly.'
            )
        seq = seq_nat
        coords_init = coords_native.clone()
        profile = profile_nat
    elif args.md_mode == 'fold':
        if coords_native is None:
            raise FileNotFoundError(
                f'--md_mode fold needs native (for RMSD ref + sequence). Looked at: {native_path}. '
                f'Pass --native_bead PATH.'
            )
        seq = seq_nat
        profile = profile_nat
        # Start from extended chain in internal coords (same init as fold_one.py).
        coords_tmp = coords_native.clone()
        proto = Protein(seq, coords_tmp, profile)
        extend_int = torch.tensor(
            [[5.367, 0.1, 0.0]], device=device,
        ).repeat((len(seq) - 3, 1))
        proto.update_coords_internal(extend_int)
        proto.update_cartesian_from_internal()
        coords_init = proto.coords.detach().clone()
    elif args.md_mode == 'decoy':
        if not args.init_bead:
            raise ValueError('--md_mode decoy requires --init_bead PATH')
        init_path = os.path.expanduser(args.init_bead)
        if not os.path.isfile(init_path):
            raise FileNotFoundError(f'--init_bead not found: {init_path}')
        seq_init, coords_init, profile_init = load_protein_bead(
            init_path, args.mode, device,
        )
        print(f'[md_eval] decoy init bead: {init_path} (L={len(seq_init)})')
        # Prefer the native sequence if available (sanity: lengths must match).
        if seq_nat is not None and len(seq_nat) == len(seq_init):
            seq = seq_nat
            profile = profile_nat
        else:
            seq = seq_init
            profile = profile_init
    else:
        raise ValueError(f'--md_mode must be native / fold / decoy, got {args.md_mode!r}')

    if coords_native is None:
        coords_native = coords_init.clone()

    return seq, coords_init, coords_native, profile


def _kabsch_align(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Return ``x`` rotated/translated onto ``ref`` (optimal RMSD).

    Both inputs are (N, 3) in Å. No reflection: fix via det(R)=+1 with a
    correcting diagonal. Standard Kabsch.
    """
    x_c = x - x.mean(axis=0)
    r_c = ref - ref.mean(axis=0)
    H = x_c.T @ r_c
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    return x_c @ R.T + ref.mean(axis=0)


def _compute_rmsd_series(sample_xyz_np: np.ndarray, ref_xyz_np: np.ndarray) -> np.ndarray:
    """RMSD (Å) of each frame in ``sample_xyz_np`` vs ``ref_xyz_np`` after
    Kabsch superposition. sample_xyz_np: (F, N, 3). ref_xyz_np: (N, 3)."""
    out = np.empty(sample_xyz_np.shape[0], dtype=np.float32)
    for i, frame in enumerate(sample_xyz_np):
        aligned = _kabsch_align(frame.astype(np.float64), ref_xyz_np.astype(np.float64))
        out[i] = float(np.sqrt(((aligned - ref_xyz_np) ** 2).sum(axis=1).mean()))
    return out


def _compute_rmsf(sample_xyz_np: np.ndarray) -> np.ndarray:
    """Per-residue RMSF (Å) over frames, after superposing each frame to
    frame 0. sample_xyz_np: (F, N, 3)."""
    ref = sample_xyz_np[0].astype(np.float64)
    aligned = np.stack(
        [_kabsch_align(frame.astype(np.float64), ref) for frame in sample_xyz_np],
        axis=0,
    )
    mean = aligned.mean(axis=0)
    deviations = aligned - mean[None]
    per_atom = (deviations ** 2).sum(axis=-1).mean(axis=0)
    return np.sqrt(per_atom).astype(np.float32)


def _rg(coords_np: np.ndarray) -> float:
    centered = coords_np - coords_np.mean(axis=0, keepdims=True)
    return float(np.sqrt((centered ** 2).sum(axis=1).mean()))


def _build_parser() -> argparse.ArgumentParser:
    parser = options.get_fold_parser()
    parser.add_argument('--md_mode', type=str, default='native',
                        help='native / fold / decoy')
    parser.add_argument('--decoy_set', type=str, default='casp14',
                        help='Used to auto-locate native/start bead CSVs.')
    parser.add_argument('--target', type=str, default=None,
                        help='Target pdb id (e.g. T1053). Required.')
    parser.add_argument('--init_bead', type=str, default=None,
                        help='Start bead CSV (for --md_mode decoy).')
    parser.add_argument('--native_bead', type=str, default=None,
                        help='Native bead CSV for RMSD reference (auto-located if omitted).')
    return parser


def main() -> None:
    parser = _build_parser()
    args = options.parse_args_and_arch(parser)

    if args.target is None:
        raise SystemExit('--target is required (e.g. T1053)')
    if args.save_dir is None:
        raise SystemExit('--save_dir is required')
    os.makedirs(args.save_dir, exist_ok=True)

    # Mode-specific defaults if the user did not override on the CLI.
    defaults_by_mode = {
        'native': dict(lr=1e-2, T_max=1e-2),
        'fold':   dict(lr=3e-2, T_max=3e-2),
        'decoy':  dict(lr=1e-2, T_max=1e-2),
    }
    d = defaults_by_mode[args.md_mode]
    # options.py defaults: lr=1e-3, T_max=0.1. Only override if CLI left them at default.
    if args.lr == 1e-3:
        args.lr = d['lr']
    if abs(args.T_max - 0.1) < 1e-9:
        args.T_max = d['T_max']

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device, _model, energy_fn, _pb = test_setup(args)

    seq, coords_init, coords_native, profile = _load_start_and_native(args, device)

    protein_native = Protein(seq, coords_native, profile)
    energy_native = float(protein_native.get_energy(energy_fn).item())
    rg2_native, _ = protein_native.get_rad_gyration(coords_native)
    print(f'[md_eval] energy_native={energy_native:.3f} rg_native={rg2_native.item() ** 0.5:.3f}')

    protein = Protein(seq, coords_init.clone(), profile.clone())
    energy_init = float(protein.get_energy(energy_fn).item())
    print(f'[md_eval] energy_init={energy_init:.3f}')

    sampler_cls = _SAMPLER_BY_XTYPE.get(args.x_type)
    if sampler_cls is None:
        raise SystemExit(
            f'--x_type must be one of {list(_SAMPLER_BY_XTYPE)}, got {args.x_type!r}'
        )

    minimizer = sampler_cls(
        energy_fn, protein,
        num_steps=args.L,
        lr=args.lr,
        t_noise=args.T_max,
        log_interval=args.trj_log_interval,
    )

    t0 = time.time()
    minimizer.run()
    elapsed = time.time() - t0

    coords_best = minimizer.x_best
    energy_best = float(minimizer.energy_best)
    # Sample layout mirrors fold_one.py: [native, best, init, *logged_frames].
    sample = [
        coords_native.detach().cpu().clone(),
        coords_best.detach().cpu().clone(),
        coords_init.detach().cpu().clone(),
    ] + list(minimizer.sample)
    sample_energy = [energy_native, energy_best, energy_init] + list(minimizer.sample_energy)

    write_pdb_sample2(seq, sample, args.target, f'trajectory_{args.md_mode}', args.save_dir)

    sample_xyz = torch.stack(sample, 0).cpu().detach().numpy()
    ref_native = sample_xyz[0].copy()
    ref_start = sample_xyz[2].copy()
    rmsd_native = _compute_rmsd_series(sample_xyz, ref_native)
    rmsd_start = _compute_rmsd_series(sample_xyz, ref_start)
    rg_vals = np.array([_rg(f) for f in sample_xyz])

    # Frame labels: -2 = native ref, -1 = best ever seen, 0 = starting
    # frame; subsequent entries are every ``trj_log_interval`` step.
    n_logged = len(minimizer.sample)
    labels = ['native', 'best', 'init'] + [f'frame_{i}' for i in range(n_logged)]
    steps = [-1, -1, 0] + [i * args.trj_log_interval for i in range(n_logged)]

    df = pd.DataFrame({
        'label': labels,
        'step': steps,
        'energy': sample_energy,
        'rmsd_to_native': rmsd_native,
        'rmsd_to_start': rmsd_start,
        'rg': rg_vals,
    })
    df.to_csv(os.path.join(args.save_dir, f'{args.target}_energy_rmsd.csv'), index=False)

    # RMSF over the logged-trajectory portion only (exclude native/best/init refs).
    if n_logged >= 2:
        rmsf_np = _compute_rmsf(sample_xyz[3:])
        pd.DataFrame({'residue': np.arange(len(rmsf_np)), 'rmsf': rmsf_np}).to_csv(
            os.path.join(args.save_dir, f'{args.target}_rmsf.csv'), index=False,
        )

    # Final-window RMSD (mean over last 20% of logged frames) is a cleaner
    # drift metric than the single last frame.
    if n_logged >= 5:
        tail_n = max(1, n_logged // 5)
        final_rmsd_native = float(rmsd_native[3:][-tail_n:].mean())
        final_rmsd_start = float(rmsd_start[3:][-tail_n:].mean())
    else:
        final_rmsd_native = float(rmsd_native[-1])
        final_rmsd_start = float(rmsd_start[-1])

    meta = {
        'target': args.target,
        'md_mode': args.md_mode,
        'x_type': args.x_type,
        'L': args.L,
        'lr': args.lr,
        't_noise': args.T_max,
        'trj_log_interval': args.trj_log_interval,
        'seed': args.seed,
        'load_exp': args.load_exp,
        'load_checkpoint': args.load_checkpoint,
        'n_residues': int(len(seq)),
        'n_logged_frames': n_logged,
        'energy_native': energy_native,
        'energy_init': energy_init,
        'energy_best': energy_best,
        'final_rmsd_to_native': final_rmsd_native,
        'final_rmsd_to_start': final_rmsd_start,
        'rg_native': float(rg2_native.item() ** 0.5),
        'elapsed_sec': elapsed,
    }
    with open(os.path.join(args.save_dir, 'meta.json'), 'w') as fh:
        json.dump(meta, fh, indent=2)

    print(
        f'[md_eval] DONE target={args.target} mode={args.md_mode} '
        f'final_rmsd_to_native={final_rmsd_native:.2f}Å '
        f'energy_best={energy_best:.2f} elapsed={elapsed:.1f}s -> {args.save_dir}'
    )


if __name__ == '__main__':
    main()
