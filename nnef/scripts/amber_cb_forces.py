#!/usr/bin/env python3
"""Stage 1 of T_eff calibration: recompute Amber14SB forces on the Cbeta atom
of every residue, at sampled frames of an existing amber14sb_reference_md.py
trajectory. Saves (cb_coords, cb_forces) per frame to an .npz -- no torch/NNEF
needed here, so this runs in the `openmm` conda env.

Why: amber14sb_reference_md.py only saved trajectory.dcd + rmsd/rmsf, not
forces. OpenMM lets us recompute the force at any saved frame cheaply (one
context.getState call), without rerunning the dynamics.

Rebuilds the EXACT same System as amber14sb_reference_md.py (implicit solvent
OBC2, no water -- confirmed via meta.json's use_water=false) so the force is
the same one that generated the trajectory.

Usage (openmm env, on the cluster):
    python nnef/scripts/amber_cb_forces.py \\
        --pdb_id 3KXT \\
        --ref_dir eval/md_eval/amber14sb_ref/3KXT \\
        --stride 200 \\
        --out eval/md_eval/amber14sb_ref/3KXT/cb_forces.npz

--stride 200 on a 20000-frame trajectory -> 100 sampled frames (3KXT);
on the 5000-frame ones (6H8O, 5ZGM) use --stride 50 -> 100 frames.
"""
from __future__ import annotations

import argparse
import os

import numpy as np


def three_to_one_map():
    # Same 20-aa table used across nnef/data_prep_scripts/local_extractor_v2.py.
    return {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    }


def main() -> None:
    from openmm.app import ForceField, Modeller, PDBFile, Simulation, NoCutoff, HBonds
    from openmm import LangevinMiddleIntegrator, Platform, unit
    import mdtraj as md

    p = argparse.ArgumentParser()
    p.add_argument('--pdb_id', required=True)
    p.add_argument('--ref_dir', required=True,
                    help='dir with <pdb_id>_fixed.pdb + trajectory.dcd (from amber14sb_reference_md.py)')
    p.add_argument('--stride', type=int, default=200, help='take every Nth frame')
    p.add_argument('--out', required=True)
    p.add_argument('--temperature', type=float, default=300.0)
    p.add_argument('--chain_id', type=str, default=None,
                    help='restrict to one PDB chain (e.g. "A") -- needed when the '
                         'amber system has more chains (e.g. a homodimer in the '
                         'crystal asymmetric unit) than the single-chain bead CSV '
                         'used for training/eval. Omit for single-chain proteins.')
    args = p.parse_args()

    fixed_pdb = os.path.join(args.ref_dir, f'{args.pdb_id}_fixed.pdb')
    dcd = os.path.join(args.ref_dir, 'trajectory.dcd')
    for f in (fixed_pdb, dcd):
        if not os.path.isfile(f):
            raise FileNotFoundError(f)

    print(f'[amber_cb_forces] {args.pdb_id}: rebuilding implicit-solvent OBC2 system '
          f'(matches amber14sb_reference_md.py use_water=false) ...')
    pdb = PDBFile(fixed_pdb)
    forcefield = ForceField('amber14-all.xml', 'implicit/obc2.xml')
    system = forcefield.createSystem(pdb.topology, nonbondedMethod=NoCutoff, constraints=HBonds)
    integrator = LangevinMiddleIntegrator(args.temperature * unit.kelvin, 1.0 / unit.picosecond, 2.0 * unit.femtoseconds)
    try:
        platform = Platform.getPlatformByName('CUDA')
    except Exception:
        platform = Platform.getPlatformByName('CPU')
    simulation = Simulation(pdb.topology, system, integrator, platform)

    # --- residue -> CB atom index map (GLY uses CA, matching
    # nnef/data_prep_scripts/local_extractor_v2.py's cb=ca convention) ---
    top = pdb.topology
    chain_ids_seen = sorted({c.id for c in top.chains()})
    if len(chain_ids_seen) > 1 and args.chain_id is None:
        print(f'[amber_cb_forces] WARNING: {len(chain_ids_seen)} chains in topology '
              f'{chain_ids_seen} but --chain_id not set -- using ALL of them. If the '
              f'training bead CSV is single-chain (common for a crystallographic '
              f'dimer), pass --chain_id to match, e.g. --chain_id {chain_ids_seen[0]}.')
    cb_atom_idx = []   # one atom index per residue, in topology order
    residue_names = []
    residue_numbers = []  # PDB resSeq (res.id) -- lets stage 2 align by residue
    # NUMBER instead of assuming positional 1:1 correspondence. PDBFixer's
    # findMissingResidues()/addMissingAtoms() can model in a disordered loop
    # residue that's simply absent from the training bead CSV (off-by-a-few
    # residue count), so a raw position/count match is not safe in general.
    three_to_one = three_to_one_map()
    for res in top.residues():
        if args.chain_id is not None and res.chain.id != args.chain_id:
            continue
        if res.name not in three_to_one:
            continue  # skip waters/ligands/HETATM residues, if any
        atom_by_name = {a.name: a.index for a in res.atoms()}
        if res.name == 'GLY':
            idx = atom_by_name.get('CA')
        else:
            idx = atom_by_name.get('CB', atom_by_name.get('CA'))
        if idx is None:
            continue
        cb_atom_idx.append(idx)
        residue_names.append(three_to_one[res.name])
        residue_numbers.append(int(res.id))
    cb_atom_idx = np.asarray(cb_atom_idx, dtype=np.int64)
    residue_numbers = np.asarray(residue_numbers, dtype=np.int64)
    n_res = len(cb_atom_idx)
    print(f'[amber_cb_forces] {n_res} residues with a CB (or CA-for-GLY) atom resolved '
          f'(resSeq range {residue_numbers.min()}-{residue_numbers.max()})')

    print(f'[amber_cb_forces] loading trajectory {dcd} (stride={args.stride}) ...')
    traj = md.load_dcd(dcd, top=fixed_pdb, stride=args.stride)
    n_frames = traj.n_frames
    print(f'[amber_cb_forces] {n_frames} frames sampled')

    cb_coords = np.zeros((n_frames, n_res, 3), dtype=np.float32)   # Angstrom
    cb_forces = np.zeros((n_frames, n_res, 3), dtype=np.float32)   # kJ/mol/Angstrom

    for i in range(n_frames):
        pos_nm = traj.xyz[i]  # mdtraj stores nm, (n_atoms, 3)
        simulation.context.setPositions(pos_nm)
        state = simulation.context.getState(getForces=True, getPositions=True)
        forces_kj_per_nm = state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer)
        positions_nm = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)

        cb_coords[i] = positions_nm[cb_atom_idx] * 10.0          # nm -> Angstrom
        cb_forces[i] = forces_kj_per_nm[cb_atom_idx] / 10.0       # kJ/mol/nm -> kJ/mol/Angstrom

        if i % 20 == 0:
            print(f'  frame {i}/{n_frames}')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    np.savez(args.out, cb_coords=cb_coords, cb_forces=cb_forces,
             residue_names=np.array(residue_names), residue_numbers=residue_numbers,
             pdb_id=args.pdb_id, temperature=args.temperature)
    print(f'[amber_cb_forces] wrote {args.out}  '
          f'(cb_coords {cb_coords.shape}, cb_forces {cb_forces.shape})')


if __name__ == '__main__':
    main()
