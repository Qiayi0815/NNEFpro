"""Amber14SB reference MD on a single protein, OpenMM driver.

Produces the per-residue RMSF profile that is the ground-truth judge for
the NNEF lr sweep (Yang 2022 Figure 5 protocol).

Inputs:
  --pdb_id      4-letter PDB id (will fetch from RCSB if --pdb_path absent)
  --pdb_path    optional explicit path to PDB file
  --out_dir     output directory (will be created)
  --ns          simulation time in nanoseconds (default 50)
  --equil_ns    equilibration time before recording (default 2)
  --temperature K (default 300)
  --seed        integer seed for the LangevinIntegrator

Outputs in <out_dir>:
  trajectory.dcd       raw OpenMM trajectory (recorded every 10 ps)
  rmsd.csv             time (ps), RMSD-to-equilibrated-start (Å), Rg (Å)
  rmsf.csv             residue index, RMSF (Å)         — Yang Figure 5 metric
  meta.json            params + summary stats

This script is the discrimination judge for fasrc/md_eval_yang_lr_big_sweep.slurm:
once both are done, eval/decoys/compare_amber_vs_nnef_rmsf.py computes
Pearson(NNEF_RMSF, Amber_RMSF) per (model, lr) cell.

Dependencies (load on the cluster — typically `conda activate openmm` env):
  openmm >= 8, pdbfixer, mdtraj, numpy, pandas
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np


def fetch_pdb(pdb_id: str, dest: Path) -> Path:
    pdb_id = pdb_id.lower()
    if dest.exists():
        return dest
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    print(f"[amber] fetching {url}")
    urllib.request.urlretrieve(url, dest)
    return dest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pdb_id", required=True)
    p.add_argument("--pdb_path", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--ns", type=float, default=50.0)
    p.add_argument("--equil_ns", type=float, default=2.0)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dt_fs", type=float, default=2.0,
                   help="integrator time step (fs)")
    p.add_argument("--record_ps", type=float, default=10.0,
                   help="record frame every N ps")
    p.add_argument("--use_water", action="store_true",
                   help="solvate in TIP3P water + add Na+/Cl- ions")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    from openmm.app import (
        PDBFile,
        ForceField,
        Modeller,
        Simulation,
        DCDReporter,
        StateDataReporter,
        PME,
        HBonds,
        NoCutoff,
    )
    from openmm import LangevinMiddleIntegrator, Platform, unit
    from pdbfixer import PDBFixer

    # ---- 1. load + fix PDB ----
    if args.pdb_path:
        src_pdb = Path(args.pdb_path)
    else:
        src_pdb = out / f"{args.pdb_id}.pdb"
        fetch_pdb(args.pdb_id, src_pdb)

    print(f"[amber] loading {src_pdb}")
    fixer = PDBFixer(filename=str(src_pdb))
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.removeHeterogens(keepWater=False)
    fixer.addMissingHydrogens(7.0)

    fixed_pdb = out / f"{args.pdb_id}_fixed.pdb"
    with open(fixed_pdb, "w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)
    print(f"[amber] wrote {fixed_pdb}")

    # ---- 2. force field + (optional) solvation ----
    if args.use_water:
        forcefield = ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
        modeller = Modeller(fixer.topology, fixer.positions)
        modeller.addSolvent(forcefield, padding=1.0 * unit.nanometers,
                            ionicStrength=0.15 * unit.molar)
        system = forcefield.createSystem(
            modeller.topology, nonbondedMethod=PME,
            nonbondedCutoff=1.0 * unit.nanometers,
            constraints=HBonds,
        )
        topology = modeller.topology
        positions = modeller.positions
    else:
        # implicit solvent (faster, no waters) — Yang's MD figures used this
        forcefield = ForceField("amber14-all.xml", "implicit/obc2.xml")
        system = forcefield.createSystem(
            fixer.topology, nonbondedMethod=NoCutoff,
            constraints=HBonds,
        )
        topology = fixer.topology
        positions = fixer.positions

    # ---- 3. integrator + minimisation ----
    integrator = LangevinMiddleIntegrator(
        args.temperature * unit.kelvin,
        1.0 / unit.picosecond,
        args.dt_fs * unit.femtoseconds,
    )
    integrator.setRandomNumberSeed(args.seed)

    try:
        platform = Platform.getPlatformByName("CUDA")
    except Exception:
        platform = Platform.getPlatformByName("CPU")
    print(f"[amber] platform = {platform.getName()}")

    simulation = Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(positions)
    print("[amber] minimising ...")
    simulation.minimizeEnergy(maxIterations=2000)

    # ---- 4. equilibrate ----
    equil_steps = int(args.equil_ns * 1000 / (args.dt_fs * 1e-3))
    print(f"[amber] equilibrating {args.equil_ns} ns = {equil_steps} steps ...")
    simulation.context.setVelocitiesToTemperature(args.temperature * unit.kelvin, args.seed)
    simulation.step(equil_steps)

    # ---- 5. production ----
    prod_steps = int(args.ns * 1000 / (args.dt_fs * 1e-3))
    rec_steps  = int(args.record_ps / (args.dt_fs * 1e-3))
    print(f"[amber] production {args.ns} ns = {prod_steps} steps, record every {args.record_ps} ps")

    dcd = out / "trajectory.dcd"
    state_log = out / "state.log"
    simulation.reporters.append(DCDReporter(str(dcd), rec_steps))
    simulation.reporters.append(StateDataReporter(
        str(state_log), rec_steps,
        step=True, potentialEnergy=True, kineticEnergy=True,
        temperature=True, speed=True, separator="\t",
    ))

    t0 = time.time()
    simulation.step(prod_steps)
    elapsed = time.time() - t0
    print(f"[amber] production done in {elapsed:.1f} s")

    # ---- 6. analyse trajectory: RMSF, RMSD, Rg ----
    import mdtraj as md
    traj = md.load_dcd(str(dcd), top=str(fixed_pdb))
    print(f"[amber] loaded trajectory: {traj.n_frames} frames × {traj.n_atoms} atoms")

    ca_idx = traj.topology.select("name CA")
    traj_ca = traj.atom_slice(ca_idx)
    traj_ca.superpose(traj_ca, frame=0)

    rmsd = md.rmsd(traj_ca, traj_ca, frame=0) * 10.0   # nm → Å
    rg   = md.compute_rg(traj_ca) * 10.0

    rmsf = md.rmsf(traj_ca, traj_ca, 0) * 10.0
    residues = [r.resSeq for r in traj_ca.topology.residues]

    import pandas as pd
    t_ps = np.arange(traj.n_frames) * args.record_ps
    pd.DataFrame({"time_ps": t_ps, "rmsd_A": rmsd, "rg_A": rg}).to_csv(
        out / "rmsd.csv", index=False)
    pd.DataFrame({"residue": residues, "rmsf": rmsf}).to_csv(
        out / "rmsf.csv", index=False)

    meta = dict(
        pdb_id=args.pdb_id, ns=args.ns, equil_ns=args.equil_ns,
        temperature=args.temperature, seed=args.seed,
        dt_fs=args.dt_fs, record_ps=args.record_ps,
        n_frames=int(traj.n_frames), n_residues=int(traj_ca.n_atoms),
        final_rmsd_to_start=float(rmsd[-1]),
        mean_rmsd=float(rmsd.mean()),
        rg_native=float(rg[0]), rg_mean=float(rg.mean()),
        rmsf_mean=float(rmsf.mean()),
        elapsed_sec=elapsed,
        use_water=args.use_water,
    )
    json.dump(meta, open(out / "meta.json", "w"), indent=2)
    print(f"[amber] wrote {out / 'meta.json'}")
    print(f"[amber] summary  rmsf_mean={rmsf.mean():.3f} Å  "
          f"final_rmsd={rmsd[-1]:.3f} Å  rg={rg.mean():.2f} Å")


if __name__ == "__main__":
    main()
