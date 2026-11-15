#!/usr/bin/env python3
"""Build ``nnef/data/decoys/pep_norm``: geometry-normalised copy of the realistic pep_nmr decoys.

Question: do energies prefer ESMFold over the NMR native because of local-geometry style
(X-ray-like bond geometry / CB placement in ESMFold vs NMR-refinement geometry)? Every
full-atom structure of a pep_nmr target -- all NMR models (from the mmCIF in
``<work>/cif``, chain chosen in ``<work>/curation.csv``) and the ESMFold PDB -- is passed
through ONE pipeline:

  1. single model/chain -> PDB, hydrogens stripped (NMR H naming varies);
  2. PDBFixer: no missing residues added; missing heavy atoms (incl. OXT) added;
     disulfides from geometry (Topology.createDisulfideBonds, SG-SG); hydrogens at pH 7
     via Modeller.addHydrogens(forcefield) so CYX/HIS variants follow the force field;
  3. OpenMM, Amber ff14SB (amber14-all.xml) + GBn2 implicit solvent (implicit/gbn2.xml),
     NoCutoff, HBonds constraints; harmonic Calpha restraints to the input positions,
     E = 0.5 * k * d^2 with k = 10 kcal/mol/A^2 (= 4184 kJ/mol/nm^2), separate force group;
  4. LocalEnergyMinimizer, tolerance 10 kJ/mol/nm, maxIterations 2000 (CPU platform);
  5. beads re-extracted with local_extractor_v2.extract_beads_v2 (same as pep_nmr) and
     given the pep_nmr target's chain_id / group_num (positional; sequence checked).

Decoys per target (NAMEs as in pep_nmr): ``<PDB>_native`` (normalised target model, same
model number as pep_nmr), ``<PDB>_nmrNN``, ``<PDB>_esmfold`` (all normalised), plus
``<PDB>_native_raw`` = the UN-normalised pep_nmr native bead file, so geometry-style
effects can be measured directly. Quality columns GDT_TS / RMSD_CA / LDDT_CA are vs the
normalised target, plus ``per_residue_lddt.csv``. Extra list.csv columns: source,
min_ca_rmsd (Calpha RMSD moved by minimisation, no superposition), min_ca_rmsd_sup (after
superposition), restraint_kJ (final restraint energy), energy_kJ (final total potential).
Structures that fail (parameterisation / minimisation / sequence mismatch) are left out
and listed in ``<work>/norm/failures.csv``; a target is dropped if its native fails.

Usage (env /n/holylabs/LABS/atchan_lab/Lab/qzha/envs/pepmm; repo ~/nnef)::

    python -s nnef/data_prep_scripts/build_pep_norm.py run --ids 1L2Y,2JOF,1A11     # smoke
    python -s nnef/data_prep_scripts/build_pep_norm.py run --chunk $SLURM_ARRAY_TASK_ID --nchunks 20 --nproc 8
    python -s nnef/data_prep_scripts/build_pep_norm.py finalize
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import shutil
import sys
import tempfile
import traceback
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OPENMM_CPU_THREADS", "1")

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

from paths import data_path  # noqa: E402
from local_extractor_v2 import extract_beads_v2  # noqa: E402

WORK = Path("/n/holylabs/LABS/atchan_lab/Lab/qzha/pep_nmr_work")
NMR = Path(data_path("decoys", "pep_nmr"))
OUT = Path(data_path("decoys", "pep_norm"))
K_KJ_NM2 = 10.0 * 4.184 * 100.0  # 10 kcal/mol/A^2 in kJ/mol/nm^2
TOL = 10.0                        # kJ/mol/nm
MAXIT = 2000


# --------------------------------------------------------------------------- #
# geometry metrics (same definitions as build_pep_nmr.py; copied so this script
# runs in the OpenMM env without importing that module's extras)
# --------------------------------------------------------------------------- #
def kabsch(P, Q):
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, qc - pc @ R.T


def rmsd_sup(P, Q):
    R, t = kabsch(P, Q)
    return float(np.sqrt(((P @ R.T + t - Q) ** 2).sum(1).mean()))


def gdt_ts(P, Q):
    L = len(Q)
    seeds = [np.arange(L)]
    for w in (3, 5, 7):
        if w <= L:
            seeds += [np.arange(s, s + w) for s in range(L - w + 1)]
    fr = []
    for cut in (1.0, 2.0, 4.0, 8.0):
        best = 0
        for sd in seeds:
            sel = sd
            for _ in range(20):
                if len(sel) < 3:
                    break
                R, t = kabsch(P[sel], Q[sel])
                d = np.linalg.norm(P @ R.T + t - Q, axis=1)
                new = np.where(d <= cut)[0]
                best = max(best, len(new))
                if len(new) < 3 or np.array_equal(new, sel):
                    break
                sel = new
            if best == L:
                break
        fr.append(best / L)
    return 100.0 * float(np.mean(fr))


def lddt_ca(P, Q, r0=15.0):
    dq = np.linalg.norm(Q[:, None] - Q[None], axis=-1)
    dp = np.linalg.norm(P[:, None] - P[None], axis=-1)
    mask = (dq < r0) & ~np.eye(len(Q), dtype=bool)
    diff = np.abs(dq - dp)
    pres = sum((diff < th).astype(float) for th in (0.5, 1.0, 2.0, 4.0)) / 4.0
    n = mask.sum(1)
    per = np.where(n > 0, (pres * mask).sum(1) / np.maximum(n, 1), np.nan)
    return float((pres * mask).sum() / max(mask.sum(), 1)), per


def xyz(df, nm):
    return df[[f"x{nm}", f"y{nm}", f"z{nm}"]].values.astype(float)


# --------------------------------------------------------------------------- #
# OpenMM normalisation
# --------------------------------------------------------------------------- #
_FF = None


def forcefield():
    global _FF
    if _FF is None:
        from openmm import app
        _FF = app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
    return _FF


def normalise_pdb(pdb_text, out_pdb):
    """PDB text (heavy atoms) -> minimised PDB file. Returns stats dict."""
    import openmm
    from openmm import app, unit
    from pdbfixer import PDBFixer

    fixer = PDBFixer(pdbfile=io.StringIO(pdb_text))
    fixer.missingResidues = {}
    fixer.findNonstandardResidues()
    if fixer.nonstandardResidues:
        raise RuntimeError(f"nonstandard {[r.name for r, _ in fixer.nonstandardResidues]}")
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.topology.createDisulfideBonds(fixer.positions)
    n_ss = sum(1 for b in fixer.topology.bonds()
               if b[0].name == "SG" and b[1].name == "SG")
    mod = app.Modeller(fixer.topology, fixer.positions)
    mod.addHydrogens(forcefield(), pH=7.0)
    system = forcefield().createSystem(mod.topology, nonbondedMethod=app.NoCutoff,
                                       constraints=app.HBonds)
    rest = openmm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    rest.addGlobalParameter("k", K_KJ_NM2)
    for p in ("x0", "y0", "z0"):
        rest.addPerParticleParameter(p)
    pos0 = mod.positions.value_in_unit(unit.nanometer)
    ca_idx = [a.index for a in mod.topology.atoms() if a.name == "CA"]
    for i in ca_idx:
        rest.addParticle(i, list(pos0[i]))
    rest.setForceGroup(1)
    system.addForce(rest)
    integ = openmm.VerletIntegrator(0.001)
    plat = openmm.Platform.getPlatformByName("CPU")
    ctx = openmm.Context(system, integ, plat, {"Threads": "1"})
    ctx.setPositions(mod.positions)
    e0 = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    openmm.LocalEnergyMinimizer.minimize(ctx, TOL * unit.kilojoule_per_mole / unit.nanometer, MAXIT)
    st = ctx.getState(getEnergy=True, getPositions=True)
    e1 = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    er = ctx.getState(getEnergy=True, groups={1}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    pos = st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    if not np.isfinite(pos).all() or not np.isfinite(e1):
        raise RuntimeError("non-finite after minimisation")
    with open(out_pdb, "w") as fh:
        app.PDBFile.writeFile(mod.topology, st.getPositions(), fh, keepIds=True)
    a0 = np.array([pos0[i] for i in ca_idx]) * 10
    a1 = pos[ca_idx] * 10
    return dict(n_ss_omm=n_ss, energy0_kJ=e0, energy_kJ=e1, restraint_kJ=er,
                min_ca_rmsd=float(np.sqrt(((a1 - a0) ** 2).sum(1).mean())),
                min_ca_rmsd_sup=rmsd_sup(a1, a0))


def model_chain_pdb(structure, model, chain_id):
    from Bio.PDB import PDBIO, Select

    class Sel(Select):
        def accept_model(self, m):
            return m is model

        def accept_chain(self, c):
            return c.id == chain_id

        def accept_atom(self, a):
            return a.element != "H" and a.element != "D" and not a.get_id().startswith("H")

    io_ = PDBIO()
    io_.set_structure(structure)
    buf = io.StringIO()
    io_.save(buf, Sel())
    return buf.getvalue()


def esm_pdb_text(fn):
    return "".join(l for l in open(fn) if l.startswith(("ATOM", "TER", "END")) and
                   (not l.startswith("ATOM") or l[76:78].strip() != "H"))


# --------------------------------------------------------------------------- #
def run_target(pdb):
    from Bio.PDB import MMCIFParser
    cur = pd.read_csv(WORK / "curation.csv", keep_default_na=False).set_index("pdb")
    meta = pd.read_csv(NMR / "targets_summary.csv").set_index("pdb")
    chain = cur.loc[pdb, "chain"]
    tmodel = int(meta.loc[pdb, "target_model"])
    tmpl = pd.read_csv(NMR / pdb / f"{pdb}_native_bead.csv").sort_values("group_num").reset_index(drop=True)
    seq = "".join(tmpl.group_name)
    txt = gzip.open(WORK / "cif" / f"{pdb}.cif.gz", "rt").read()
    st = MMCIFParser(QUIET=True).get_structure(pdb, io.StringIO(txt))
    jobs = []
    for m in st:
        name = f"{pdb}_native" if m.serial_num == tmodel else f"{pdb}_nmr{m.serial_num:02d}"
        jobs.append((name, "native" if m.serial_num == tmodel else "nmr", lambda m=m: model_chain_pdb(st, m, chain)))
    ef = WORK / "esmfold" / f"{pdb}.pdb"
    if ef.exists():
        jobs.append((f"{pdb}_esmfold", "esmfold", lambda: esm_pdb_text(ef)))
    tmp = tempfile.mkdtemp(prefix=f"norm_{pdb}_", dir=os.environ.get("TMPDIR", "/tmp"))
    rows, fails, beads = [], [], {}
    for name, src, get in jobs:
        try:
            fn = os.path.join(tmp, f"{name}.pdb")
            stats = normalise_pdb(get(), fn)
            df = extract_beads_v2(fn, out_csv=None).sort_values("group_num").reset_index(drop=True)
            if "".join(df.group_name) != seq:
                raise RuntimeError(f"sequence mismatch {''.join(df.group_name)} vs {seq}")
            df["chain_id"] = tmpl.chain_id.values
            df["group_num"] = tmpl.group_num.values
            beads[name] = df
            rows.append(dict(NAME=name, source=src, **stats))
        except Exception as exc:  # noqa: BLE001
            fails.append(dict(pdb=pdb, NAME=name, source=src, error=repr(exc)[:300]))
    shutil.rmtree(tmp, ignore_errors=True)
    nat = f"{pdb}_native"
    if nat not in beads:
        return dict(pdb=pdb, ok=False, fails=fails)
    tdir = OUT / pdb
    if tdir.exists():
        shutil.rmtree(tdir)
    tdir.mkdir(parents=True)
    Q = xyz(beads[nat], "ca")
    raw = tmpl.copy()
    beads[f"{pdb}_native_raw"] = raw
    rows.append(dict(NAME=f"{pdb}_native_raw", source="native_raw"))
    out, per = [], {}
    order = [nat, f"{pdb}_native_raw"] + [r["NAME"] for r in rows if r["NAME"] not in (nat, f"{pdb}_native_raw")]
    rmap = {r["NAME"]: r for r in rows}
    for name in order:
        df = beads[name]
        P = xyz(df, "ca")
        g, pr = lddt_ca(P, Q)
        r = rmap[name]
        out.append(dict(NAME=name, GDT_TS=round(gdt_ts(P, Q), 2), RMSD_CA=round(rmsd_sup(P, Q), 3),
                        LDDT_CA=round(g, 4), source=r["source"],
                        min_ca_rmsd=round(r.get("min_ca_rmsd", np.nan), 4),
                        min_ca_rmsd_sup=round(r.get("min_ca_rmsd_sup", np.nan), 4),
                        restraint_kJ=round(r.get("restraint_kJ", np.nan), 3),
                        energy_kJ=round(r.get("energy_kJ", np.nan), 2), n_ss_omm=r.get("n_ss_omm", np.nan)))
        per[name] = np.round(pr, 4)
        df.to_csv(tdir / f"{name}_bead.csv", index=False, float_format="%.3f")
    lst = pd.DataFrame(out)
    lst.to_csv(tdir / "list.csv", index=False)
    pr = pd.DataFrame.from_dict(per, orient="index", columns=[str(i + 1) for i in range(len(Q))])
    pr.index.name = "NAME"
    pr.to_csv(tdir / "per_residue_lddt.csv")
    norm = lst[lst.source != "native_raw"]
    info = dict(pdb=pdb, ok=True, n_decoys=len(lst), n_failed=len(fails),
                native_min_ca_rmsd=float(lst.min_ca_rmsd.iloc[0]),
                native_raw_rmsd_to_norm=float(lst.RMSD_CA[lst.source == "native_raw"].iloc[0]),
                esmfold_min_ca_rmsd=float(norm.min_ca_rmsd[norm.source == "esmfold"].iloc[0]) if (norm.source == "esmfold").any() else np.nan,
                esmfold_rmsd=float(norm.RMSD_CA[norm.source == "esmfold"].iloc[0]) if (norm.source == "esmfold").any() else np.nan,
                max_min_ca_rmsd=float(norm.min_ca_rmsd.max()), median_restraint_kJ=float(norm.restraint_kJ.median()),
                n_ss_omm=int(lst.n_ss_omm.iloc[0]), fails=fails)
    return info


def cmd_run(a):
    meta = pd.read_csv(NMR / "targets_summary.csv")
    ids = a.ids.split(",") if a.ids else list(meta.pdb)
    if a.nchunks:
        ids = ids[a.chunk::a.nchunks]
    (WORK / "norm" / "info").mkdir(parents=True, exist_ok=True)
    if not a.force:  # restart-safe (requeue): skip targets already finished OK
        def _done(p):
            f = WORK / "norm" / "info" / f"{p}.json"
            return f.exists() and json.load(open(f)).get("ok") and (OUT / p / "list.csv").exists()
        ids = [p for p in ids if not _done(p)]
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(a.nproc) as pool:
        for info in pool.imap_unordered(_safe_run, ids):
            json.dump(info, open(WORK / "norm" / "info" / f"{info['pdb']}.json", "w"))
            if info.get("ok"):
                print(f"[norm] {info['pdb']} n={info['n_decoys']} failed={info['n_failed']} "
                      f"native_moved={info['native_min_ca_rmsd']:.3f} raw_vs_norm={info['native_raw_rmsd_to_norm']:.3f} "
                      f"esm_moved={info['esmfold_min_ca_rmsd']:.3f} max_moved={info['max_min_ca_rmsd']:.3f} "
                      f"restr_med={info['median_restraint_kJ']:.2f} ss={info['n_ss_omm']}", flush=True)
            else:
                print(f"[norm] {info['pdb']} FAILED {info.get('fails', info.get('error'))}", flush=True)


def _safe_run(pdb):
    try:
        return run_target(pdb)
    except Exception:  # noqa: BLE001
        return dict(pdb=pdb, ok=False, error=traceback.format_exc()[-500:])


def cmd_finalize(a):
    meta = pd.read_csv(NMR / "targets_summary.csv")
    infos = [json.load(open(f)) for f in sorted((WORK / "norm" / "info").glob("*.json"))]
    fails = [f for i in infos for f in i.get("fails", [])] + \
            [dict(pdb=i["pdb"], NAME="(target)", source="", error=i.get("error", "native failed"))
             for i in infos if not i.get("ok")]
    pd.DataFrame(fails).to_csv(WORK / "norm" / "failures.csv", index=False)
    ok = pd.DataFrame([{k: v for k, v in i.items() if k != "fails"} for i in infos if i.get("ok")])
    ok = ok[ok.pdb.map(lambda p: (OUT / p / "list.csv").exists())]
    keep = [c for c in ["pdb", "length", "n_models", "target_model", "ss_class", "paper_class", "n_disulfides",
                        "disulfide_rich", "solvent_flag", "solid_state", "max_identity_to_train",
                        "homolog_in_train", "sequence"] if c in meta.columns]
    summ = meta[keep].merge(ok.drop(columns=["ok"]), on="pdb").sort_values("pdb")
    summ.to_csv(OUT / "targets_summary.csv", index=False)
    pd.DataFrame({"pdb": summ.pdb}).to_csv(OUT / "pdb_no_missing_residue.csv", index=False)
    print(f"[finalize] {len(summ)} targets ok, {sum(not i.get('ok') for i in infos)} targets failed, "
          f"{len(fails)} failure rows -> {WORK / 'norm' / 'failures.csv'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "finalize"])
    ap.add_argument("--ids", default=None)
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--nchunks", type=int, default=0)
    ap.add_argument("--nproc", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    {"run": cmd_run, "finalize": cmd_finalize}[a.cmd](a)


if __name__ == "__main__":
    main()
