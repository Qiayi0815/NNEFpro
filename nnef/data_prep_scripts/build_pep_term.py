#!/usr/bin/env python3
"""Build the terminal-perturbation peptide decoy set ``nnef/data/decoys/pep_term``.

Purpose: isolate whether a model can detect errors at the 2+2 chain-terminal
residues (Yang-style models have no blocks centred there; ribbon does).

For every pep_nmr target (same target = medoid NMR model, read from
``pep_nmr/<PDB>/<PDB>_native_bead.csv``) the decoys are:

  <PDB>_native            the target itself (NMR geometry; RMSD 0)
  <PDB>_native_rebuild    NeRF ideal-geometry rebuild from the target's own phi/psi/omega
                          (= the sigma=0 ideal rebuild; reference-quality decoy with the
                          SAME geometry model as every perturbed decoy)
  <PDB>_term_sSS_KK       same rebuild, with Gaussian noise (sigma SS deg, SS in 20/45/90,
                          KK = 01..10) added to phi AND psi of the first two and last two
                          residues only; every interior phi/psi and every omega is native.

Note: in a N->C NeRF build phi of residue 1 and psi of residue L do not move any
backbone atom, so the effective perturbed torsions are psi1, phi2, psi2,
phi(L-1), psi(L-1), phi(L). The rebuilt chains are superposed on the target by the
INTERIOR Calpha atoms (residues 3..L-2), so the bead files show the terminal error
directly.

Geometry, CB placement and Gly CB = CA are identical to build_pep_nmr.py (functions are
imported from it). Same on-disk format as pep_nmr: ``pdb_no_missing_residue.csv``,
``targets_summary.csv``, per-target ``list.csv`` + ``<NAME>_bead.csv`` +
``per_residue_lddt.csv``.

list.csv columns: NAME, GDT_TS, RMSD_CA, LDDT_CA (vs the target, as in pep_nmr),
term_rmsd (Calpha RMSD over the 2+2 terminal residues after superposing the interior
Calpha residues 3..L-2 on the target), interior_rmsd (Calpha RMSD of the interior after
that superposition), source, sigma, n_blocks_v2, n_blocks_v2_interior.

Usage (FASRC; env /n/holylabs/LABS/atchan_lab/Lab/qzha/envs/pepbench)::

    python -s nnef/data_prep_scripts/build_pep_term.py [--ids 1L2Y,2JOF] [--nproc 16]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

from paths import data_path  # noqa: E402
import build_pep_nmr as B  # noqa: E402

SIGMAS = (20, 45, 90)
N_REP = 10
NMR = Path(data_path("decoys", "pep_nmr"))
OUT = Path(data_path("decoys", "pep_term"))


def term_rmsd(P, Q):
    """Superpose P on Q by interior CA (2..L-3); return (terminal RMSD, interior RMSD)."""
    L = len(Q)
    inter = np.arange(2, L - 2)
    term = np.array([0, 1, L - 2, L - 1])
    R, t = B.kabsch(P[inter], Q[inter])
    X = P @ R.T + t
    d = np.linalg.norm(X - Q, axis=1)
    return float(np.sqrt((d[term] ** 2).mean())), float(np.sqrt((d[inter] ** 2).mean()))


def build_one(pdb):
    tgt = pd.read_csv(NMR / pdb / f"{pdb}_native_bead.csv").sort_values("group_num").reset_index(drop=True)
    Q = B._xyz(tgt, "ca")
    Nn, Cn = B._xyz(tgt, "n"), B._xyz(tgt, "c")
    L = len(Q)
    tdir = OUT / pdb
    if tdir.exists():
        shutil.rmtree(tdir)
    tdir.mkdir(parents=True)
    rows, per = [], {}

    def add(name, df, source, sigma=np.nan):
        P = B._xyz(df, "ca")
        g, pr = B.lddt_ca(P, Q)
        tr, ir = term_rmsd(P, Q)
        nbi, nbt = B.v2_block_counts(B._xyz(df, "cb"))
        rows.append(dict(NAME=name, GDT_TS=round(B.gdt_ts(P, Q), 2), RMSD_CA=round(B.rmsd_sup(P, Q), 3),
                         LDDT_CA=round(g, 4), term_rmsd=round(tr, 3), interior_rmsd=round(ir, 3),
                         source=source, sigma=sigma, n_blocks_v2=nbi + nbt, n_blocks_v2_interior=nbi))
        per[name] = np.round(pr, 4)
        df.to_csv(tdir / f"{name}_bead.csv", index=False, float_format="%.3f")

    phi, psi, omg = B.backbone_torsions(Nn, Q, Cn)
    inter = np.arange(2, L - 2)

    def rebuilt(ph, ps, name, source, sigma=np.nan):
        N, CA, C = B.nerf_build(ph, ps, omg)
        R, t = B.kabsch(CA[inter], Q[inter])
        N, CA, C = N @ R.T + t, CA @ R.T + t, C @ R.T + t
        add(name, B._bead_df(tgt, N, CA, C, B.ideal_cb(N, CA, C)), source, sigma)

    add(f"{pdb}_native", tgt, "native")
    rebuilt(phi, psi, f"{pdb}_native_rebuild", "native_rebuild", 0)
    rng = np.random.default_rng((int.from_bytes(pdb.encode(), "little") + 7) % (2 ** 32))
    term = np.array([0, 1, L - 2, L - 1])
    for s in SIGMAS:
        for k in range(N_REP):
            ph, ps = phi.copy(), psi.copy()
            ph[term] += rng.normal(0, s, 4)
            ps[term] += rng.normal(0, s, 4)
            rebuilt(ph, ps, f"{pdb}_term_s{s:02d}_{k + 1:02d}", "term", s)
    lst = pd.DataFrame(rows)
    lst.to_csv(tdir / "list.csv", index=False)
    pr = pd.DataFrame.from_dict(per, orient="index", columns=[str(i + 1) for i in range(L)])
    pr.index.name = "NAME"
    pr.to_csv(tdir / "per_residue_lddt.csv")
    t = lst[lst.source == "term"]
    return dict(pdb=pdb, n_decoys=len(lst),
                native_rebuild_rmsd=float(lst.RMSD_CA[lst.source == "native_rebuild"].iloc[0]),
                native_rebuild_term_rmsd=float(lst.term_rmsd[lst.source == "native_rebuild"].iloc[0]),
                term_rmsd_s20=float(t.term_rmsd[t.sigma == 20].median()),
                term_rmsd_s45=float(t.term_rmsd[t.sigma == 45].median()),
                term_rmsd_s90=float(t.term_rmsd[t.sigma == 90].median()),
                interior_rmsd_max=float(t.interior_rmsd.max()),
                gdt_min=float(lst.GDT_TS.min()),
                n_decoys_zero_v2_blocks=int((lst.n_blocks_v2 == 0).sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", default=None)
    ap.add_argument("--nproc", type=int, default=8)
    a = ap.parse_args()
    meta = pd.read_csv(NMR / "targets_summary.csv")
    ids = a.ids.split(",") if a.ids else list(meta.pdb)
    OUT.mkdir(parents=True, exist_ok=True)
    infos = []
    with Pool(a.nproc) as pool:
        for info in pool.imap_unordered(build_one, ids):
            infos.append(info)
            print(f"[term] {info['pdb']} n={info['n_decoys']} rebuild={info['native_rebuild_rmsd']:.2f} "
                  f"term_rmsd s20/45/90 = {info['term_rmsd_s20']:.2f}/{info['term_rmsd_s45']:.2f}/"
                  f"{info['term_rmsd_s90']:.2f} interior_max={info['interior_rmsd_max']:.3f}", flush=True)
    if a.ids:
        return
    keep = [c for c in ["pdb", "length", "n_models", "target_model", "ss_class", "paper_class", "n_disulfides",
                        "disulfide_rich", "solvent_flag", "solid_state", "max_identity_to_train", "best_hit",
                        "homolog_in_train", "pdb_code_in_train", "sequence"] if c in meta.columns]
    summ = meta[keep].merge(pd.DataFrame(infos), on="pdb").sort_values("pdb")
    summ.to_csv(OUT / "targets_summary.csv", index=False)
    pd.DataFrame({"pdb": summ.pdb}).to_csv(OUT / "pdb_no_missing_residue.csv", index=False)
    print(f"[term] {len(summ)} targets -> {OUT}")


if __name__ == "__main__":
    main()
