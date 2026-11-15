#!/usr/bin/env python3
"""Stage 2 of T_eff calibration: regress the NNEF's Cbeta energy gradient
against the REAL (Amber14SB) force at matching frames, to estimate the
"effective temperature" implicitly baked into the NNEF potential.

Physics (see collaborator's slides, MD simulations 21.July.pptx):
    "real" energy:  P = (1/Z) exp(-E/(kB*T))  =>  E_real = kB*T*(-log P - log Z)
    NNEF:           P = exp(-E_NNEF)          =>  log P = -E_NNEF
    => grad(E_real) = kB*T_eff * grad(-log P - log Z) ... substituting grad(log P) = -grad(E_NNEF)
                     = kB*T_eff * grad(E_NNEF)                   (log Z term drops; constant)
    => F_amber       = -grad(E_real) = -kB*T_eff * grad(E_NNEF)  (NEGATIVE proportionality)
So T_eff = -slope(F_amber vs grad(E_NNEF)) / kB, fit through the origin, pooled
over every (residue, frame) pair. Once T_eff is known, the Brownian-dynamics
noise term should use beta = sqrt(6*kB*T_eff*alpha) (fluctuation-dissipation),
instead of the independently-grid-searched (lr, T_noise) pair currently used.

Run in the `nnef` conda env (needs torch); stage 1 (nnef/scripts/amber_cb_forces.py,
`openmm` env) must have already produced the --amber_npz file(s).

Usage:
    python nnef/scripts/calibrate_teff.py \\
        --load_exp runs/yang_full1000_27329319 --mode CB \\
        --amber_npz eval/md_eval/amber14sb_ref/3KXT/cb_forces.npz \\
                    eval/md_eval/amber14sb_ref/6H8O/cb_forces.npz \\
                    eval/md_eval/amber14sb_ref/5ZGM/cb_forces.npz \\
        --device cuda --out_csv eval/md_eval/teff_yang_full1000.csv

No --native_bead needed: the sequence comes straight from each npz's own
residue_names (saved by amber_cb_forces.py), which is guaranteed aligned 1:1
with that same npz's cb_coords/cb_forces since they were built in the same
loop. This deliberately avoids cross-referencing the training bead CSV, whose
residue count can differ by a few (PDBFixer may model in a disordered loop
residue that's simply absent from the bead CSV -- seen on 5ZGM: 109 vs 110).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_NNEF_DIR = os.path.abspath(os.path.join(_HERE, '..'))
if _NNEF_DIR not in sys.path:
    sys.path.insert(0, _NNEF_DIR)

import options  # noqa: E402
from paths import data_path  # noqa: E402
from protein_os import Protein  # noqa: E402
from utils import test_setup  # noqa: E402

KB_KJ_PER_MOL_K = 0.0083144626  # Boltzmann constant, kJ/(mol*K)


def _load_vocab():
    amino_acids = pd.read_csv(data_path('amino_acids.csv'))
    return {x.upper(): y - 1 for x, y in zip(amino_acids.AA3C, amino_acids.idx)}, \
           {x: y - 1 for x, y in zip(amino_acids.AA, amino_acids.idx)}


def per_protein_gradients(npz_path, energy_fn, device, vocab1):
    # Sequence comes from the npz's OWN residue_names (saved by
    # amber_cb_forces.py, one-letter codes, exactly aligned 1:1 with cb_coords/
    # cb_forces by construction -- same loop, same order, same count). This
    # deliberately does NOT cross-reference the training bead CSV: PDBFixer can
    # model in a disordered/missing loop residue that's simply absent from the
    # bead CSV (seen on 5ZGM: 109 bead-CSV residues vs 110 amber residues), and
    # a naive positional or count-based join would either crash or silently
    # introduce a false adjacency gap in the model's local-block neighbor
    # geometry. Scoring the sequence Amber ACTUALLY simulated (complete,
    # contiguous) sidesteps that whole class of bug.
    data = np.load(npz_path, allow_pickle=True)
    cb_coords = data['cb_coords']   # (n_frames, n_res, 3) Angstrom
    cb_forces = data['cb_forces']   # (n_frames, n_res, 3) kJ/mol/Angstrom
    pdb_id = str(data['pdb_id'])
    n_frames, n_res, _ = cb_coords.shape

    seq = [str(x) for x in data['residue_names']]
    seq_id = np.array([vocab1[x] for x in seq])
    profile = torch.tensor(seq_id, dtype=torch.long, device=device)
    seq_arr = np.array(seq)  # unused downstream except by Protein() signature

    grads, forces = [], []
    for i in range(n_frames):
        coords = torch.tensor(cb_coords[i], dtype=torch.float, device=device, requires_grad=True)
        protein = Protein(seq_arr, coords, profile)
        energy = protein.get_energy(energy_fn)
        energy.backward()
        g = protein.coords.grad
        if g is None or not torch.isfinite(g).all():
            continue
        grads.append(g.detach().cpu().numpy())
        forces.append(cb_forces[i])

    if not grads:
        raise RuntimeError(f'{pdb_id}: no frame produced a finite gradient')
    return pdb_id, np.concatenate(grads, axis=0), np.concatenate(forces, axis=0)


def main():
    parser = options.get_fold_parser()
    parser.add_argument('--amber_npz', nargs='+', required=True)
    # --mode already defined by options.get_fold_parser() (default 'CB', matches).
    parser.add_argument('--out_csv', type=str, default=None)
    args = options.parse_args_and_arch(parser)

    # SAFETY: this script only extracts CB coords/forces (stage 1 does not
    # capture N/CA/C). A v2-frame model (ribbon/esm, use_local_frame_v2=True,
    # i.e. --legacy_local_frame NOT set) would silently fall back to the wrong
    # (legacy) frame when Protein() gets no n_coords/ca_coords/c_coords --
    # _has_v2_backbone() just returns False, no crash, wrong gradient. Refuse.
    if not getattr(args, 'legacy_local_frame', False):
        raise SystemExit(
            "[calibrate_teff] ERROR: --legacy_local_frame is not set, but this "
            "script only has CB coords (no N/CA/C). A v2-frame model (ribbon/esm) "
            "would silently score in the WRONG frame. Only yang/yang_legacy "
            "checkpoints (trained with --legacy_local_frame) are safe to "
            "calibrate right now; extend amber_cb_forces.py to also extract "
            "N/CA/C atom forces before calibrating ribbon/esm.")

    device, _model, energy_fn, _pb = test_setup(args)
    _vocab3, vocab1 = _load_vocab()

    all_grad, all_force, per_protein_rows = [], [], []
    for npz_path in args.amber_npz:
        pdb_id, grad, force = per_protein_gradients(npz_path, energy_fn, device, vocab1)
        n = grad.shape[0]
        print(f'[calibrate_teff] {pdb_id}: {n} (residue,frame) samples')
        all_grad.append(grad); all_force.append(force)
        per_protein_rows.append((pdb_id, grad, force))

    grad = np.concatenate(all_grad, axis=0).reshape(-1)     # flatten (n,3) -> (3n,)
    force = np.concatenate(all_force, axis=0).reshape(-1)

    # F_amber = kB*T_eff * grad(E_NNEF)  -> through-origin least squares slope.
    slope = float(np.dot(grad, force) / np.dot(grad, grad))
    # F_amber = -kB*T_eff * grad(E_NNEF), so T_eff = -slope/kB (see module docstring).
    t_eff = -slope / KB_KJ_PER_MOL_K
    resid = force - slope * grad
    r2 = 1.0 - np.sum(resid ** 2) / np.sum((force - force.mean()) ** 2)

    print(f'\n=== T_eff calibration (pooled over {len(grad)} residue*frame*xyz samples) ===')
    print(f'  slope (kB*T_eff)  = {slope:.6g} kJ/mol per NNEF-gradient-unit')
    print(f'  T_eff             = {t_eff:.1f} K')
    print(f'  R^2 (through origin) = {r2:.3f}  (low R^2 = force/gradient poorly correlated -- '
          f'proportionality itself may be a bad model, not just the constant)')

    for alpha in (3e-4, 1e-3, 3e-3, 1e-2):
        beta_fdt = (6 * KB_KJ_PER_MOL_K * t_eff * alpha) ** 0.5
        print(f'  alpha={alpha:.0e}  ->  FDT-consistent beta = {beta_fdt:.4g}   '
              f'(commonly used beta so far: {alpha:.0e}, i.e. beta==alpha by convention)')

    if args.out_csv:
        rows = []
        for pdb_id, g, f in per_protein_rows:
            gm = np.linalg.norm(g, axis=-1)
            fm = np.linalg.norm(f, axis=-1)
            for gi, fi in zip(gm, fm):
                rows.append({'pdb_id': pdb_id, 'grad_norm': gi, 'force_norm': fi})
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)
        print(f'[calibrate_teff] wrote per-sample norms -> {args.out_csv}')


if __name__ == '__main__':
    main()
