#!/usr/bin/env python3
"""Amber refinement baseline: does physics-based relaxation move ESMFold toward the NMR target?

For each peptide in ~/nnef/runs/pack/pep_md_targets.txt the full-atom ESMFold PDB
(<work>/esmfold/<PDB>.pdb) is prepared exactly as in nnef/data_prep_scripts/build_pep_norm.py:
H stripped, then PDBFixer (missing heavy atoms incl. OXT, disulfides from geometry), then
hydrogens at pH 7. The force field is Amber ff14SB + GBn2, NoCutoff, HBonds constraints,
with NO restraints. Two runs:

  (a) min: LocalEnergyMinimizer, tolerance 10 kJ/mol/nm, max 5000 steps;
  (b) md : from (a), Langevin 300 K, friction 1/ps, 2 fs, 25,000 steps (50 ps; seed 1),
           then minimise the final frame as in (a).

Beads come from local_extractor_v2.extract_beads_v2 (Gly CB = CA). The reference is the
pep_nmr target (medoid NMR model, pep_nmr/<PDB>/<PDB>_native_bead.csv). RMSDs are
computed after superposing all residues (each atom type superposed on itself).

Output: eval/baselines_pep/refine_amber.csv (pdb, rmsd_cb_start/min/md, rmsd_ca_start/min/md,
plus energies, timings and status) and refine_amber_summary.csv: median dRMSD (after - start)
and the fraction improved / worsened by > 0.1 A, overall and per ss_class.

    /n/holylabs/LABS/atchan_lab/Lab/qzha/envs/pepmm/bin/python -s eval/baselines_pep/refine_amber.py run --nproc 32
    (or chunked: run --chunk K --nchunks 2 --nproc 48 per job; summary merges the parts)
    .../python -s eval/baselines_pep/refine_amber.py summary
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
import traceback
from multiprocessing import Pool

os.environ.setdefault("OPENMM_CPU_THREADS", "1")
import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "nnef"))
sys.path.insert(0, os.path.join(REPO, "nnef", "data_prep_scripts"))
from local_extractor_v2 import extract_beads_v2  # noqa: E402
import build_pep_norm as N  # noqa: E402  (prep / forcefield / rmsd helpers)

WORK = "/n/holylabs/LABS/atchan_lab/Lab/qzha/pep_nmr_work"
NMR = os.path.join(REPO, "nnef", "data", "decoys", "pep_nmr")
TARGETS = os.path.expanduser("~/nnef/runs/pack/pep_md_targets.txt")
OUT = os.path.join(REPO, "eval", "baselines_pep", "refine_amber.csv")
MAXIT = 5000
NSTEPS = 25000


def prep(pdb_text):
    import io
    from openmm import app
    from pdbfixer import PDBFixer
    fixer = PDBFixer(pdbfile=io.StringIO(pdb_text))
    fixer.missingResidues = {}
    fixer.findNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.topology.createDisulfideBonds(fixer.positions)
    mod = app.Modeller(fixer.topology, fixer.positions)
    mod.addHydrogens(N.forcefield(), pH=7.0)
    return mod


def beads_from_state(top, positions, fn):
    from openmm import app
    with open(fn, "w") as fh:
        app.PDBFile.writeFile(top, positions, fh, keepIds=True)
    return extract_beads_v2(fn, out_csv=None).sort_values("group_num").reset_index(drop=True)


def rmsds(df, ref):
    if "".join(df.group_name) != "".join(ref.group_name):
        raise RuntimeError("sequence mismatch")
    return (N.rmsd_sup(N.xyz(df, "cb"), N.xyz(ref, "cb")), N.rmsd_sup(N.xyz(df, "ca"), N.xyz(ref, "ca")))


def run_one(pdb):
    import openmm
    from openmm import app, unit
    t0 = time.time()
    row = dict(pdb=pdb, status="ok")
    tmp = tempfile.mkdtemp(prefix=f"ref_{pdb}_", dir=os.environ.get("TMPDIR", "/tmp"))
    try:
        ref = pd.read_csv(os.path.join(NMR, pdb, f"{pdb}_native_bead.csv")).sort_values("group_num").reset_index(drop=True)
        ef = os.path.join(WORK, "esmfold", f"{pdb}.pdb")
        start = extract_beads_v2(ef, out_csv=None).sort_values("group_num").reset_index(drop=True)
        row["rmsd_cb_start"], row["rmsd_ca_start"] = rmsds(start, ref)
        mod = prep(N.esm_pdb_text(ef))
        system = N.forcefield().createSystem(mod.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
        integ = openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds)
        integ.setRandomNumberSeed(1)
        ctx = openmm.Context(system, integ, openmm.Platform.getPlatformByName("CPU"), {"Threads": "1"})
        ctx.setPositions(mod.positions)
        tol = 10.0 * unit.kilojoule_per_mole / unit.nanometer
        E = lambda: ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)  # noqa: E731
        row["E_start"] = E()
        openmm.LocalEnergyMinimizer.minimize(ctx, tol, MAXIT)
        row["E_min"] = E()
        st = ctx.getState(getPositions=True)
        dmin = beads_from_state(mod.topology, st.getPositions(), os.path.join(tmp, "min.pdb"))
        row["rmsd_cb_min"], row["rmsd_ca_min"] = rmsds(dmin, ref)
        row["t_min_s"] = time.time() - t0
        ctx.setVelocitiesToTemperature(300 * unit.kelvin, 1)
        integ.step(NSTEPS)
        row["E_md_end"] = E()
        openmm.LocalEnergyMinimizer.minimize(ctx, tol, MAXIT)
        row["E_md_min"] = E()
        st = ctx.getState(getPositions=True)
        pos = st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        if not np.isfinite(pos).all():
            raise RuntimeError("non-finite coordinates after MD")
        dmd = beads_from_state(mod.topology, st.getPositions(), os.path.join(tmp, "md.pdb"))
        row["rmsd_cb_md"], row["rmsd_ca_md"] = rmsds(dmd, ref)
    except Exception:  # noqa: BLE001
        row["status"] = "fail: " + traceback.format_exc().strip().splitlines()[-1][:200]
    row["t_total_s"] = time.time() - t0
    os.system(f"rm -rf {tmp}")
    return row


def cmd_run(a):
    ids = a.ids.split(",") if a.ids else [l.strip() for l in open(TARGETS) if l.strip()]
    if a.nchunks:
        ids = ids[a.chunk::a.nchunks]
    rows = []
    with Pool(a.nproc) as pool:
        for r in pool.imap_unordered(run_one, ids):
            rows.append(r)
            print(f"[refine] {r['pdb']} {r['status']} cb start/min/md = "
                  f"{r.get('rmsd_cb_start', np.nan):.2f}/{r.get('rmsd_cb_min', np.nan):.2f}/{r.get('rmsd_cb_md', np.nan):.2f} "
                  f"t={r['t_total_s']:.0f}s", flush=True)
    cols = ["pdb", "rmsd_cb_start", "rmsd_cb_min", "rmsd_cb_md", "rmsd_ca_start", "rmsd_ca_min", "rmsd_ca_md"]
    df = pd.DataFrame(rows)
    df = df[cols + [c for c in df.columns if c not in cols]].sort_values("pdb")
    out = (OUT.replace(".csv", "_smoke.csv") if a.ids else
           OUT.replace(".csv", f"_part{a.chunk}.csv") if a.nchunks else OUT)
    df.to_csv(out, index=False, float_format="%.4f")
    print(f"[refine] wrote {out} ({(df.status == 'ok').sum()}/{len(df)} ok)")


def cmd_summary(a):
    import glob
    parts = sorted(glob.glob(OUT.replace(".csv", "_part*.csv")))
    if parts:  # merge chunked runs into refine_amber.csv
        pd.concat([pd.read_csv(f) for f in parts]).sort_values("pdb").to_csv(OUT, index=False, float_format="%.4f")
    df = pd.read_csv(OUT)
    print(f"[summary] {(df.status == 'ok').sum()}/{len(df)} ok")
    df = df[df.status == "ok"]
    meta = pd.read_csv(os.path.join(NMR, "targets_summary.csv")).set_index("pdb")
    df["ss_class"] = df.pdb.map(meta.ss_class)
    rows = []
    for grp, g in [("all", df)] + list(df.groupby("ss_class")):
        for atom in ("cb", "ca"):
            for stage in ("min", "md"):
                d = g[f"rmsd_{atom}_{stage}"] - g[f"rmsd_{atom}_start"]
                rows.append(dict(subset=grp, atom=atom, stage=stage, n=len(g),
                                 median_start=g[f"rmsd_{atom}_start"].median(),
                                 median_after=g[f"rmsd_{atom}_{stage}"].median(),
                                 median_delta=d.median(), mean_delta=d.mean(),
                                 frac_improved=(d < -0.1).mean(), frac_worsened=(d > 0.1).mean()))
    s = pd.DataFrame(rows)
    s.to_csv(OUT.replace(".csv", "_summary.csv"), index=False)
    pd.set_option("display.width", 200)
    print(s.round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "summary"])
    ap.add_argument("--ids", default=None)
    ap.add_argument("--nproc", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--nchunks", type=int, default=0)
    a = ap.parse_args()
    {"run": cmd_run, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    main()
