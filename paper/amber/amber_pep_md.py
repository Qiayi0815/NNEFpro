#!/usr/bin/env python3
"""Amber ff14SB + GBn2 implicit-solvent MD of a pep_nmr peptide from its NMR target model.

Preparation is identical to eval/baselines_pep/refine_amber.py / build_pep_norm.py
(heavy atoms of the pep_nmr target model + chain from the mmCIF, PDBFixer: no missing
residues, missing heavy atoms added, disulfides from geometry, H at pH 7), NO restraints.
Then: minimise (tol 10 kJ/mol/nm), Langevin (LangevinMiddle) 300 K, friction 1/ps, 2 fs,
HBonds constraints, NoCutoff; --equil_ns discarded; production --ns recorded every
--record_ps. Output (out_dir): frames.npy (n,4,L,3) N/CA/C/CB in pep_nmr bead order
(Gly CB = CA; frame 0 = the pep_nmr native bead coordinates), frames_time_ps.npy
(-1 = native), meta.json.

    envs/pepmm/bin/python -s amber_pep_md.py --pdb 2MU9 --ns 100 --seed 0 --out_dir ...
"""
import argparse, json, os, sys, time, gzip, io
import numpy as np
import pandas as pd

REPO = os.path.expanduser("~/nnef")
sys.path.insert(0, os.path.join(REPO, "nnef"))
sys.path.insert(0, os.path.join(REPO, "nnef", "data_prep_scripts"))
import build_pep_norm as N  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--ns", type=float, default=100.0)
    ap.add_argument("--equil_ns", type=float, default=1.0)
    ap.add_argument("--record_ps", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    import openmm
    from openmm import app, unit
    from pdbfixer import PDBFixer
    from Bio.PDB import MMCIFParser

    pdb = a.pdb
    cur = pd.read_csv(N.WORK / "curation.csv", keep_default_na=False).set_index("pdb")
    meta = pd.read_csv(N.NMR / "targets_summary.csv").set_index("pdb")
    chain, tmodel = cur.loc[pdb, "chain"], int(meta.loc[pdb, "target_model"])
    tmpl = pd.read_csv(N.NMR / pdb / f"{pdb}_native_bead.csv").sort_values("group_num").reset_index(drop=True)
    seq = "".join(tmpl.group_name)
    st = MMCIFParser(QUIET=True).get_structure(pdb, io.StringIO(gzip.open(N.WORK / "cif" / f"{pdb}.cif.gz", "rt").read()))
    model = [m for m in st if m.serial_num == tmodel][0]
    txt = N.model_chain_pdb(st, model, chain)

    fixer = PDBFixer(pdbfile=io.StringIO(txt))
    fixer.missingResidues = {}
    fixer.findNonstandardResidues()
    if fixer.nonstandardResidues:
        raise SystemExit(f"nonstandard residues {fixer.nonstandardResidues}")
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.topology.createDisulfideBonds(fixer.positions)
    mod = app.Modeller(fixer.topology, fixer.positions)
    ff = N.forcefield()  # amber14-all.xml + implicit/gbn2.xml
    mod.addHydrogens(ff, pH=7.0)
    system = ff.createSystem(mod.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)

    # bead atom indices in topology residue order (must reproduce the bead sequence)
    three = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'CYX': 'C', 'GLN': 'Q', 'GLU': 'E',
             'GLY': 'G', 'HIS': 'H', 'HID': 'H', 'HIE': 'H', 'HIP': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
             'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}
    idx, s = [], ""
    for r in mod.topology.residues():
        if r.name not in three:
            continue
        an = {at.name: at.index for at in r.atoms()}
        idx.append([an["N"], an["CA"], an["C"], an.get("CB", an["CA"])])
        s += three[r.name]
    if s != seq:
        raise SystemExit(f"sequence mismatch {s} vs {seq}")
    idx = np.array(idx)  # (L,4)

    integ = openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds)
    integ.setRandomNumberSeed(a.seed + 1)
    try:
        plat = openmm.Platform.getPlatformByName("CUDA"); props = {}
        ctx = openmm.Context(system, integ, plat, props)
    except Exception:
        plat = openmm.Platform.getPlatformByName("CPU")
        ctx = openmm.Context(system, integ, plat, {"Threads": str(a.threads)})
    print(f"[amber_pep] {pdb} L={len(seq)} atoms={system.getNumParticles()} platform={plat.getName()}", flush=True)
    ctx.setPositions(mod.positions)
    openmm.LocalEnergyMinimizer.minimize(ctx, 10.0 * unit.kilojoule_per_mole / unit.nanometer, 5000)
    ctx.setVelocitiesToTemperature(300 * unit.kelvin, a.seed + 1)
    t0 = time.time()
    integ.step(int(a.equil_ns * 500000))
    t_eq = time.time() - t0
    print(f"[amber_pep] equil {a.equil_ns} ns in {t_eq:.0f}s ({a.equil_ns / max(t_eq, 1e-9) * 86400:.0f} ns/day)", flush=True)
    rec = int(round(a.record_ps / 0.002))  # 2 fs steps per record
    nrec = int(round(a.ns * 1000 / a.record_ps))
    nat = np.stack([N.xyz(tmpl, k) for k in ("n", "ca", "c", "cb")])  # (4,L,3)
    frames, times = [nat.astype(np.float32)], [-1.0]
    for k in range(nrec):
        integ.step(rec)
        pos = ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        frames.append(np.transpose(pos[idx], (1, 0, 2)).astype(np.float32))
        times.append((k + 1) * a.record_ps)
        if (k + 1) % max(1, nrec // 20) == 0:
            np.save(os.path.join(a.out_dir, "frames.npy"), np.stack(frames))
            np.save(os.path.join(a.out_dir, "frames_time_ps.npy"), np.array(times))
            print(f"[amber_pep] {times[-1] / 1000:.1f} ns  elapsed {time.time() - t0:.0f}s", flush=True)
    np.save(os.path.join(a.out_dir, "frames.npy"), np.stack(frames))
    np.save(os.path.join(a.out_dir, "frames_time_ps.npy"), np.array(times))
    el = time.time() - t0
    json.dump(dict(pdb=pdb, ns=a.ns, equil_ns=a.equil_ns, record_ps=a.record_ps, seed=a.seed, temperature_K=300,
                   forcefield="amber14-all.xml (ff14SB) + implicit/gbn2.xml", integrator="LangevinMiddle 1/ps 2fs HBonds",
                   platform=plat.getName(), n_atoms=system.getNumParticles(), L=len(seq), elapsed_sec=el,
                   chain=chain, target_model=tmodel),
              open(os.path.join(a.out_dir, "meta.json"), "w"), indent=2)
    print(f"[amber_pep] DONE {pdb} {el:.0f}s", flush=True)


if __name__ == "__main__":
    main()
