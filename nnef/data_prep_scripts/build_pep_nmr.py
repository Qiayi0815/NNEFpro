#!/usr/bin/env python3
"""Build the NMR peptide decoy benchmark ``nnef/data/decoys/pep_nmr``.

Same on-disk format as ``nnef/data/decoys/casp14_full``:

* set level   : ``pdb_no_missing_residue.csv`` (column ``pdb``) and
                ``targets_summary.csv`` (per-target metadata, see ``finalize``)
* per target  : ``<PDB>/list.csv`` (``NAME`` + quality columns) and one
                ``<PDB>/<NAME>_bead.csv`` per decoy (columns
                chain_id,group_num,group_name,xn..zcb), plus
                ``<PDB>/per_residue_lddt.csv`` (rows = decoy NAME, columns = 1-based
                residue position in bead order, Cα-lDDT per residue).

Beads are produced with the repo's canonical converter
``local_extractor_v2.extract_beads_v2`` (Gly CB = CA; caps / non-standard residues are
skipped by name) applied to a single-model, single-chain PDB written per NMR model,
so conventions match casp14_full. Generated decoys (NeRF perturbations, controls,
ESMFold) inherit the target's chain_id / group_num so all decoys of a target share
residue numbering.

Target list
-----------
McDonald, Hostetler, ... Gulsevin, "Benchmarking AlphaFold2 on peptide structure
prediction", Structure 31:111 (2023), Table S1 (Elsevier mmc2.xlsx, 588 PDB IDs,
classes AH_MP / AH_SL / BHPIN / DSRP / MIX_MP / MIX_SL). The table is copied to
``<work>/raw/mcdonald2023_tableS1.csv``.

Curation (``curate``; one row per entry in ``<work>/curation.csv``)
------------------------------------------------------------------
Chain = the L-polypeptide chain whose sequence best matches the paper's FASTA.
Hard filters (in order, first failing reason recorded in ``drop_reason``):
  parse_fail, no_protein_chain, nonstandard (any non-standard residue in the chain
  other than terminal caps ACE/NH2/NME/FOR/ACY..., incl. D-amino-acid comp ids),
  d_amino_acid (polypeptide(D) entity or inverted CA chirality in model 1),
  head_to_tail_cyclic (struct_conn covale N(first)-C(last) or N1-C_last < 2 A),
  insertion_or_dup_numbering, chain_break (consecutive C-N > 2 A),
  model_inconsistent (models with different residue sets), n_models<10,
  length outside 10..40 (bead residues).
Kept-with-flag columns: n_disulfides, disulfide_rich (>=2), other_crosslinks
(non-disulfide covalent links between chain residues), n_protein_chains (multichain),
ligands (non-polymer, non-water), caps, solvent_flag (micelle/TFE/HFIP/organic/
aqueous/unknown parsed from pdbx_nmr_sample_details + exptl_sample_conditions +
paper description), solid_state (exptl.method).

Target / decoys (``build``)
---------------------------
target = medoid NMR model (min mean pairwise Cα-RMSD after superposition);
``target_model`` records its model number. Decoys:
  <PDB>_native           target itself (RMSD 0)
  <PDB>_nmrNN            every other NMR model (NN = model number)
  <PDB>_esmfold          ESMFold (HF transformers EsmForProteinFolding, facebook/esmfold_v1)
                         if ``<work>/esmfold/<PDB>.pdb`` exists (stage ``esmfold``)
  <PDB>_ideal_rebuild    NeRF rebuild of the target from its own phi/psi/omega with ideal
                         bond lengths/angles (sigma=0 control)
  <PDB>_pert_sSS_KK      NeRF rebuild with Gaussian noise (sigma SS deg) added to every
                         phi and psi (omega kept native, so cis-Pro is preserved);
                         sigma in {5,10,20,30,45,60,90}, 8 each, seeded per target
  <PDB>_ctrl_{extended,helix,hairpin,ppii}
                         ideal-geometry chains with fixed phi/psi (180/180; -57/-47;
                         beta -120/130 strands with a type-I' turn (60/30, 90/0) at the
                         centre; -75/145)
NeRF geometry: N-CA 1.458, CA-C 1.525, C-N 1.329 A; N-CA-C 111.2, CA-C-N 116.2,
C-N-CA 121.7 deg; CB placed with the ideal-tetrahedral formula (trRosetta constants);
Gly CB = CA. Rebuilt decoys are Kabsch-superposed onto the target by Cα.

Quality columns (vs target, Cα only): RMSD_CA (Kabsch), GDT_TS (mean fraction under
1/2/4/8 A, x100; per cutoff the best superposition over iterative seeds: all
contiguous 3/5/7-residue windows + full chain, LGA-like), LDDT_CA (global Cα lDDT,
0..1, inclusion radius 15 A on the target, thresholds 0.5/1/2/4 A, all i!=j pairs),
plus ``source`` and ``sigma`` columns, and ``n_blocks_v2`` / ``n_blocks_v2_interior``: number of
local blocks that survive the v2 scorer's 20 A span rule (k=10; see ``v2_block_counts``) -- 0 means
the current ``protein_os`` raises "every block was dropped" (all peptides with L < 15, and
extended decoys). ``evaluate_decoys.pick_metric`` will pick GDT_TS.

Secondary structure: pydssp (3-state) on the target's N,CA,C,O.
ss_class: helix if fH>=0.3 and fE<0.1; hairpin if fE>=0.2 and fH<0.1;
mixed if fH>=0.15 and fE>=0.1; else coil (fH/fE fraction of residues).

Leakage (``leakage``): MMSeqs2 (static binary, --exhaustive-search, SW, no prefilter)
of each peptide against every training chain in
~/nnef_data/hhsuite_CB_yang_pdb_list.csv and nnef_data_v3/hhsuite_CB_v2_k10_pdb_list_holdout.csv
(sequences from hhsuite_pdb_seq_v2_k10.h5). max_identity_to_train = max nident / qlen
(query-length normalised so short local hits do not count as 100%), taking the max of
MMseqs2 and an exhaustive ungapped numpy scan; best_hit; best_hit_fident (identity over
aligned columns), best_hit_evalue, homolog_in_train (MMseqs2 E < 1e-3),
shuffled_null_identity (same ungapped scan on the shuffled peptide = chance floor, ~0.3-0.4
for 20-40-mers against ~28k chains) and pdb_code_in_train.

Usage (FASRC, env /n/holylabs/LABS/atchan_lab/Lab/qzha/envs/pepbench, repo ~/nnef)::

    python -s nnef/data_prep_scripts/build_pep_nmr.py curate  --work W [--ids 1A11,2MC7] [--nproc 8]
    python -s nnef/data_prep_scripts/build_pep_nmr.py train_fasta --work W
    python -s nnef/data_prep_scripts/build_pep_nmr.py leakage --work W
    python -s nnef/data_prep_scripts/build_pep_nmr.py esmfold --work W        # GPU or CPU (CPU: ~2 s/peptide
                                                                              # with 16 cores; fasrc/pep_nmr_esmfold_cpu.slurm)
    python -s nnef/data_prep_scripts/build_pep_nmr.py build   --work W [--ids ...] [--nproc 8]
    python -s nnef/data_prep_scripts/build_pep_nmr.py finalize --work W
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

_NNEF_DIR = Path(__file__).resolve().parent.parent
if str(_NNEF_DIR) not in sys.path:
    sys.path.insert(0, str(_NNEF_DIR))

from paths import data_path  # noqa: E402
from data_prep_scripts.local_extractor_v2 import extract_beads_v2  # noqa: E402

warnings.filterwarnings("ignore")

STD3 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G", "HIS": "H",
    "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q",
    "ARG": "R", "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}
CAPS = {"ACE", "NH2", "NME", "FOR", "ACY", "AME", "CH3", "PHQ"}  # terminal blocking groups
WATER = {"HOH", "DOD", "WAT"}
SIGMAS = (5, 10, 20, 30, 45, 60, 90)
N_PER_SIGMA = 8
MIN_MODELS, MIN_LEN, MAX_LEN = 10, 10, 40
TRAIN_LISTS = (
    os.path.expanduser("~/nnef_data/hhsuite_CB_yang_pdb_list.csv"),
    "/n/holylabs/LABS/atchan_lab/Lab/qzha/nnef_data_v3/hhsuite_CB_v2_k10_pdb_list_holdout.csv",
)
TRAIN_SEQ_H5 = "/n/holylabs/LABS/atchan_lab/Lab/qzha/nnef_data_v3/hhsuite_pdb_seq_v2_k10.h5"
MMSEQS = "/n/holylabs/LABS/atchan_lab/Lab/qzha/pep_nmr_work/tools/mmseqs/bin/mmseqs"
EXCLUDE_DEFAULT = os.path.expanduser("~/nnef/fasrc/peptide_exclude_ids.txt")

SOLVENT_PATTERNS = [
    ("micelle", r"\bDPC\b|dodecylphospho|\bSDS\b|dodecyl ?sulfate|micell|\bLPPG\b|\bDHPC\b|bicell|"
                r"\bDMPC\b|\bDPPC\b|\bPOPC\b|\bPOPG\b|\bDMPG\b|\bLMPG\b|liposom|vesic|detergent|"
                r"\bDPC-d38\b|\bd38-DPC\b|lipid|membrane mimetic"),
    ("TFE", r"\bTFE\b|trifluoroethanol|TFE-d\d"),
    ("HFIP", r"HFIP|hexafluoro"),
    ("organic", r"methanol|CD3OH|CD3OD|\bDMSO\b|acetonitrile|chloroform|CDCl3|(?<!fluoro)ethanol"),
]


# --------------------------------------------------------------------------- #
# geometry helpers                                                            #
# --------------------------------------------------------------------------- #

def kabsch(P, Q):
    """Return R, t minimising |(P @ R.T + t) - Q|."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    return R, qc - pc @ R.T


def rmsd_sup(P, Q):
    R, t = kabsch(P, Q)
    return float(np.sqrt(((P @ R.T + t - Q) ** 2).sum(1).mean()))


def gdt_ts(P, Q):
    """LGA-like GDT_TS (x100) of model P vs target Q (Cα, same length)."""
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
    """Cα lDDT of model P vs target Q. Returns (global, per_residue[L])."""
    dq = np.linalg.norm(Q[:, None] - Q[None], axis=-1)
    dp = np.linalg.norm(P[:, None] - P[None], axis=-1)
    mask = (dq < r0) & ~np.eye(len(Q), dtype=bool)
    diff = np.abs(dq - dp)
    pres = np.zeros_like(dq)
    for th in (0.5, 1.0, 2.0, 4.0):
        pres += (diff < th)
    pres /= 4.0
    n = mask.sum(1)
    per = np.where(n > 0, (pres * mask).sum(1) / np.maximum(n, 1), np.nan)
    glob = float((pres * mask).sum() / max(mask.sum(), 1))
    return glob, per


def dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / np.linalg.norm(b1)
    v = b0 - (b0 @ b1n) * b1n
    w = b2 - (b2 @ b1n) * b1n
    x = v @ w
    y = np.cross(b1n, v) @ w
    return float(np.degrees(np.arctan2(y, x)))


def backbone_torsions(N, CA, C):
    L = len(N)
    phi = np.full(L, np.nan)
    psi = np.full(L, np.nan)
    omg = np.full(L, 180.0)
    for i in range(L):
        if i > 0:
            phi[i] = dihedral(C[i - 1], N[i], CA[i], C[i])
        if i < L - 1:
            psi[i] = dihedral(N[i], CA[i], C[i], N[i + 1])
            omg[i] = dihedral(CA[i], C[i], N[i + 1], CA[i + 1])
    phi[0] = -60.0 if np.isnan(phi[0]) else phi[0]
    psi[-1] = 120.0 if np.isnan(psi[-1]) else psi[-1]
    return phi, psi, omg


B_NCA, B_CAC, B_CN = 1.458, 1.525, 1.329
A_NCAC, A_CACN, A_CNCA = np.radians(111.2), np.radians(116.2), np.radians(121.7)


def _place(a, b, c, bond, angle, tors):
    bc = c - b
    bc /= np.linalg.norm(bc)
    n = np.cross(b - a, bc)
    n /= np.linalg.norm(n)
    m = np.cross(n, bc)
    d2 = np.array([-bond * np.cos(angle), bond * np.sin(angle) * np.cos(tors),
                   bond * np.sin(angle) * np.sin(tors)])
    return c + d2[0] * bc + d2[1] * m + d2[2] * n


def nerf_build(phi, psi, omg):
    """Ideal-geometry backbone from torsions (deg). Returns N, CA, C (L,3)."""
    L = len(phi)
    ph, ps, om = np.radians(phi), np.radians(psi), np.radians(omg)
    N = np.zeros((L, 3)); CA = np.zeros((L, 3)); C = np.zeros((L, 3))
    N[0] = [0, 0, 0]
    CA[0] = [B_NCA, 0, 0]
    C[0] = CA[0] + B_CAC * np.array([-np.cos(A_NCAC), np.sin(A_NCAC), 0])
    for i in range(1, L):
        N[i] = _place(N[i - 1], CA[i - 1], C[i - 1], B_CN, A_CACN, ps[i - 1])
        CA[i] = _place(CA[i - 1], C[i - 1], N[i], B_NCA, A_CNCA, om[i - 1])
        C[i] = _place(C[i - 1], N[i], CA[i], B_CAC, A_NCAC, ph[i])
    return N, CA, C


def ideal_cb(N, CA, C):
    b = CA - N
    c = C - CA
    a = np.cross(b, c)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA


# --------------------------------------------------------------------------- #
# paths / io                                                                  #
# --------------------------------------------------------------------------- #

def out_root(a):
    return Path(a.out) if a.out else Path(data_path("decoys", "pep_nmr"))


def load_paper_table(w):
    df = pd.read_csv(Path(w) / "raw" / "mcdonald2023_tableS1.csv")
    df.columns = [c.strip() for c in df.columns]
    df = df[df["Name"].notna() & (df["Name"].astype(str).str.strip() != "")].copy()
    df["pdb"] = df["Name"].astype(str).str.strip().str.upper()
    df["paper_class"] = df["Type"].astype(str).str.strip()
    df["paper_seq"] = df["FASTA Sequence"].astype(str).str.strip()
    df["paper_desc"] = df["Description"].astype(str)
    return df.set_index("pdb")


def load_id_table(fn):
    """Target table for entries that are NOT in the McDonald 2023 table (e.g. pep_new:
    post-AF2-cutoff RCSB NMR peptides). CSV with columns ``pdb`` and ``seq`` (RCSB canonical
    sequence of the single protein entity). There is no paper annotation for these entries,
    so ``paper_class`` / ``paper_desc`` are empty and ``paper_seq`` = the RCSB sequence (only
    used to pick the best-matching chain; these entries have one chain). Used only when
    ``--id_table`` is given; the McDonald path (pep_nmr) is unchanged."""
    df = pd.read_csv(fn, keep_default_na=False)
    df["pdb"] = df["pdb"].astype(str).str.strip().str.upper()
    df["paper_class"] = ""
    df["paper_seq"] = df["seq"].astype(str).str.strip()
    df["paper_desc"] = ""
    return df[["pdb", "paper_class", "paper_seq", "paper_desc"]].set_index("pdb")


def _as_list(d, k):
    v = d.get(k)
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def _seqid(a, b):
    """Crude ungapped best-offset identity (fraction of shorter)."""
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    best = 0
    for off in range(-len(a) + 1, len(b)):
        m = sum(1 for i, x in enumerate(a) if 0 <= off + i < len(b) and b[off + i] == x)
        best = max(best, m)
    return best / len(a)


def beads_for_model(structure, model, chain_id, tmpdir):
    """Write one model/chain to PDB and run the canonical bead converter."""
    from Bio.PDB import PDBIO, Select

    class Sel(Select):
        def accept_model(self, m):
            return m is model

        def accept_chain(self, c):
            return c.id == chain_id

    io_ = PDBIO()
    io_.set_structure(structure)
    fn = os.path.join(tmpdir, f"m{model.serial_num}.pdb")
    io_.save(fn, Sel())
    df = extract_beads_v2(fn, out_csv=None)
    return df


def o_coords(chain_res):
    out = []
    for r in chain_res:
        out.append(r["O"].get_coord() if "O" in r else np.full(3, np.nan))
    return np.array(out, dtype=float)


# --------------------------------------------------------------------------- #
# curate                                                                      #
# --------------------------------------------------------------------------- #

def _solvent(text):
    flags = [name for name, pat in SOLVENT_PATTERNS if re.search(pat, text, re.I)]
    return flags


def curate_one(task):
    pdb, w, paper = task
    from Bio.PDB import MMCIFParser
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    import gzip

    row = dict(pdb=pdb, paper_class=paper.get("paper_class"), paper_seq=paper.get("paper_seq"),
               drop_reason="")
    fn = Path(w) / "cif" / f"{pdb}.cif.gz"
    try:
        txt = gzip.open(fn, "rt").read()
        d = MMCIF2Dict(io.StringIO(txt))
        st = MMCIFParser(QUIET=True).get_structure(pdb, io.StringIO(txt))
    except Exception as exc:  # noqa: BLE001
        row["drop_reason"] = f"parse_fail:{exc}"
        return row, None

    row["method"] = ";".join(_as_list(d, "_exptl.method"))
    row["solid_state"] = "SOLID" in row["method"].upper()
    row["release_date"] = (_as_list(d, "_pdbx_audit_revision_history.revision_date") or [""])[0]
    # ---- solvent / sample text
    stext = " ".join(_as_list(d, "_pdbx_nmr_sample_details.solution_contents")
                     + _as_list(d, "_pdbx_nmr_sample_details.solvent_system")
                     + _as_list(d, "_pdbx_nmr_sample_details.details")
                     + _as_list(d, "_pdbx_nmr_exptl_sample_conditions.details")
                     + _as_list(d, "_pdbx_nmr_exptl_sample.component"))
    stext = re.sub(r"\s+", " ", stext.replace("?", " ").replace(".", ". ")).strip()
    title = " ".join(_as_list(d, "_struct.title") + _as_list(d, "_struct_keywords.text"))
    flags = _solvent(stext)
    ttext = title + " " + str(paper.get("paper_desc", ""))
    flags_title = [f for f in _solvent(ttext) if f not in flags]
    if not flags and not flags_title and re.search(r"membrane|lipid|bilayer", ttext, re.I):
        flags_title = ["membrane"]
    row["solvent_text"] = stext[:300]
    if flags or flags_title:
        row["solvent_flag"] = ";".join(flags + [f + "(title)" for f in flags_title])
    elif stext:
        row["solvent_flag"] = "aqueous"
    else:
        row["solvent_flag"] = "unknown"
    row["pH"] = ";".join(sorted(set(x for x in _as_list(d, "_pdbx_nmr_exptl_sample_conditions.pH") if x not in "?.")))
    row["temperature"] = ";".join(sorted(set(x for x in _as_list(d, "_pdbx_nmr_exptl_sample_conditions.temperature") if x not in "?.")))

    # ---- entities
    ent_ids = _as_list(d, "_entity_poly.entity_id")
    ent_type = dict(zip(ent_ids, _as_list(d, "_entity_poly.type")))
    ent_seq = dict(zip(ent_ids, [s.replace("\n", "") for s in _as_list(d, "_entity_poly.pdbx_seq_one_letter_code_can")]))
    ent_seq_raw = dict(zip(ent_ids, [s.replace("\n", "") for s in _as_list(d, "_entity_poly.pdbx_seq_one_letter_code")]))
    ent_strand = dict(zip(ent_ids, _as_list(d, "_entity_poly.pdbx_strand_id")))
    row["entity_types"] = ";".join(sorted(set(ent_type.values())))
    row["has_nucleic"] = any("nucleotide" in t for t in ent_type.values())
    row["has_D_entity"] = any("polypeptide(D)" in t for t in ent_type.values())
    nonpoly = [n for n in _as_list(d, "_pdbx_entity_nonpoly.comp_id") if n not in WATER]
    row["ligands"] = ";".join(sorted(set(nonpoly)))

    models = list(st)
    row["n_models"] = len(models)
    m0 = models[0]
    # protein chains in model 1 (chains with >=1 standard residue)
    prot_chains = [c.id for c in m0 if any(r.get_resname() in STD3 for r in c)]
    row["n_protein_chains"] = len(prot_chains)
    if not prot_chains:
        row["drop_reason"] = "no_protein_chain"
        return row, None

    pseq = str(paper.get("paper_seq", "") or "")

    def chain_seq(c):
        return "".join(STD3.get(r.get_resname(), "X") for r in m0[c] if r.id[0] == " " or r.get_resname() in STD3)

    scored = sorted(prot_chains, key=lambda c: (-_seqid(chain_seq(c), pseq), prot_chains.index(c)))
    ch = scored[0]
    row["chain"] = ch
    row["paper_seq_match"] = round(_seqid(chain_seq(ch), pseq), 3)
    # entity of chain
    ent_of_chain = None
    for e, s in ent_strand.items():
        if ch in [x.strip() for x in s.split(",")]:
            ent_of_chain = e
    row["entity_seq_raw"] = ent_seq_raw.get(ent_of_chain, "")
    row["entity_type"] = ent_type.get(ent_of_chain, "")

    res = [r for r in m0[ch] if r.get_resname() not in WATER]
    names = [r.get_resname() for r in res]
    caps = [n for n in names if n in CAPS]
    nonstd = [n for n in names if n not in STD3 and n not in CAPS]
    row["caps"] = ";".join(caps)
    row["nonstandard"] = ";".join(sorted(set(nonstd)))
    # non-standard markers in the entity sequence, e.g. (DAL), (AIB)
    nonstd_seq = [x for x in re.findall(r"\(([A-Z0-9]+)\)", row["entity_seq_raw"]) if x not in CAPS]
    if nonstd_seq:
        row["nonstandard"] = ";".join(sorted(set(nonstd + nonstd_seq)))

    std_res = [r for r in res if r.get_resname() in STD3]
    # ---- chirality (model 1)
    n_d = 0
    for r in std_res:
        if r.get_resname() == "GLY" or not all(a in r for a in ("N", "CA", "C", "CB")):
            continue
        Nn, CA, Cc, CB = (r[a].get_coord().astype(float) for a in ("N", "CA", "C", "CB"))
        cbi = ideal_cb(Nn[None], CA[None], Cc[None])[0]
        vol = lambda x: np.dot(Nn - CA, np.cross(Cc - CA, x - CA))  # noqa: E731
        if np.sign(vol(CB)) != np.sign(vol(cbi)):
            n_d += 1
    row["n_D_residues"] = n_d

    # ---- struct_conn: disulfides & covalent links
    ct = _as_list(d, "_struct_conn.conn_type_id")
    c1 = _as_list(d, "_struct_conn.ptnr1_auth_asym_id")
    c2 = _as_list(d, "_struct_conn.ptnr2_auth_asym_id")
    r1 = _as_list(d, "_struct_conn.ptnr1_auth_seq_id")
    r2 = _as_list(d, "_struct_conn.ptnr2_auth_seq_id")
    n1 = _as_list(d, "_struct_conn.ptnr1_label_comp_id")
    n2 = _as_list(d, "_struct_conn.ptnr2_label_comp_id")
    a1 = _as_list(d, "_struct_conn.ptnr1_label_atom_id")
    a2 = _as_list(d, "_struct_conn.ptnr2_label_atom_id")
    first_num = std_res[0].id[1] if std_res else None
    last_num = std_res[-1].id[1] if std_res else None
    n_ss, other, h2t, inter_ss = 0, [], False, 0
    for k in range(len(ct)):
        t = ct[k].lower()
        same = c1[k] == ch and c2[k] == ch
        if t == "disulf":
            if same:
                n_ss += 1
            elif ch in (c1[k], c2[k]):
                inter_ss += 1
        elif t == "covale" and same:
            if n1[k] in CAPS or n2[k] in CAPS:
                continue
            try:
                s1, s2 = int(r1[k]), int(r2[k])
            except ValueError:
                continue
            pair = {(s1, a1[k]), (s2, a2[k])}
            if (first_num, "N") in pair and (last_num, "C") in pair:
                h2t = True
            else:
                other.append(f"{n1[k]}{s1}.{a1[k]}-{n2[k]}{s2}.{a2[k]}")
    if std_res and "N" in std_res[0] and "C" in std_res[-1]:
        dist = np.linalg.norm(std_res[0]["N"].get_coord() - std_res[-1]["C"].get_coord())
        if dist < 2.0:
            h2t = True
    # geometric disulfide count as fallback (SG-SG < 2.3 A)
    sg = [r["SG"].get_coord() for r in std_res if r.get_resname() == "CYS" and "SG" in r]
    n_ss_geo = 0
    for i in range(len(sg)):
        for j in range(i + 1, len(sg)):
            if np.linalg.norm(sg[i] - sg[j]) < 2.3:
                n_ss_geo += 1
    row["n_disulfides"] = max(n_ss, n_ss_geo)
    row["disulfide_rich"] = row["n_disulfides"] >= 2
    row["interchain_disulfides"] = inter_ss
    row["other_crosslinks"] = ";".join(other)
    row["head_to_tail_cyclic"] = h2t

    # ---- numbering checks
    nums = [(r.id[1], r.id[2]) for r in std_res]
    ins = any(ic.strip() for _, ic in nums)
    dup = len(set(n for n, _ in nums)) != len(nums)
    breaks = 0
    for i in range(len(std_res) - 1):
        if "C" in std_res[i] and "N" in std_res[i + 1]:
            if np.linalg.norm(std_res[i]["C"].get_coord() - std_res[i + 1]["N"].get_coord()) > 2.0:
                breaks += 1
    row["chain_breaks"] = breaks
    row["length_std"] = len(std_res)

    reason = ""
    if row["has_D_entity"] or n_d > 0:
        reason = "d_amino_acid"
    if nonstd:
        reason = "nonstandard"   # non-standard (incl. D comp ids) takes precedence
    elif nonstd_seq:
        reason = "nonstandard"
    if not reason and h2t:
        reason = "head_to_tail_cyclic"
    if not reason and (ins or dup):
        reason = "insertion_or_dup_numbering"
    if not reason and breaks:
        reason = "chain_break"
    if not reason and len(models) < MIN_MODELS:
        reason = "n_models<10"
    if not reason and not (MIN_LEN <= len(std_res) <= MAX_LEN):
        reason = "length"

    beads = None
    if not reason:
        tmp = tempfile.mkdtemp(prefix=f"pep_{pdb}_", dir=os.environ.get("TMPDIR", "/tmp"))
        try:
            per = []
            for m in models:
                df = beads_for_model(st, m, ch, tmp)
                per.append((m.serial_num, df))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        ref = per[0][1]
        if len(ref) != len(std_res):
            reason = "bead_missing_atoms"
        else:
            key = list(zip(ref.group_num, ref.group_name))
            if any(list(zip(df.group_num, df.group_name)) != key for _, df in per):
                reason = "model_inconsistent"
        if not reason and not (MIN_LEN <= len(ref) <= MAX_LEN):
            reason = "length"
        if not reason:
            O = []
            for m in models:
                rr = [r for r in m[ch] if r.get_resname() in STD3]
                O.append(o_coords(rr))
            beads = dict(models=[s for s, _ in per], dfs=[df for _, df in per], O=O)
    row["length"] = len(beads["dfs"][0]) if beads else len(std_res)
    row["sequence"] = "".join(beads["dfs"][0].group_name) if beads else "".join(
        STD3.get(r.get_resname(), "X") for r in std_res)
    row["drop_reason"] = reason
    return row, beads


def cmd_curate(a):
    w = Path(a.work)
    paper = load_id_table(a.id_table) if a.id_table else load_paper_table(w)
    ids = a.ids.split(",") if a.ids else list(paper.index)
    (w / "beads_cache").mkdir(exist_ok=True)
    tasks = [(p, str(w), paper.loc[p].to_dict()) for p in ids]
    rows = []
    with Pool(a.nproc) as pool:
        for row, beads in pool.imap_unordered(curate_one, tasks, chunksize=2):
            if beads is not None:
                pd.to_pickle(beads, w / "beads_cache" / f"{row['pdb']}.pkl")
            rows.append(row)
            print(f"[curate] {row['pdb']} len={row.get('length')} models={row.get('n_models')} "
                  f"drop={row['drop_reason'] or '-'}", flush=True)
    df = pd.DataFrame(rows).sort_values("pdb")
    out = w / ("curation.csv" if not a.ids else "curation_smoke.csv")
    df.to_csv(out, index=False)
    print(f"[curate] wrote {out}: {len(df)} entries, kept {int((df.drop_reason == '').sum())}")
    print(df.drop_reason.replace("", "KEPT").value_counts().to_string())


# --------------------------------------------------------------------------- #
# leakage                                                                     #
# --------------------------------------------------------------------------- #

def cmd_train_fasta(a):
    import h5py
    w = Path(a.work)
    aa = pd.read_csv(data_path("amino_acids.csv"))
    idx2aa = {i - 1: c for i, c in zip(aa.idx, aa.AA)}
    chains = set()
    for f in TRAIN_LISTS:
        chains |= set(pd.read_csv(f)["pdb"].astype(str))
    chains = sorted(chains)
    n_ok = 0
    with h5py.File(TRAIN_SEQ_H5, "r") as h5, open(w / "train_seqs.fasta", "w") as fh:
        for c in chains:
            if c not in h5:
                continue
            ds = h5[c]
            # (n_msa_rows, L) int8, row 0 = wild-type chain (build_seq_h5.py)
            arr = np.asarray(ds[0] if ds.ndim == 2 else ds[()]).ravel()
            if arr.dtype.kind in "SUO":
                s = "".join(x.decode() if isinstance(x, bytes) else str(x) for x in arr)
            else:
                s = "".join(idx2aa.get(int(x), "X") for x in arr)
            fh.write(f">{c}\n{s}\n")
            n_ok += 1
    (w / "train_codes.txt").write_text("\n".join(sorted({c[:4].upper() for c in chains})) + "\n")
    print(f"[train_fasta] {n_ok}/{len(chains)} chains written")


def _ungapped_identity(cur, fasta):
    """Exhaustive ungapped scan: for every peptide, max over training chains and
    offsets (incl. partial overhang) of #identical residues / peptide length."""
    names, seqs, cur_n = [], [], None
    for line in open(fasta):
        line = line.strip()
        if line.startswith(">"):
            cur_n = line[1:]
        elif cur_n:
            names.append(cur_n)
            seqs.append(line)
    pad = 40
    buf, owner = [], []
    for i, s in enumerate(seqs):
        buf.append("#" * pad + s)
        owner.append(np.full(pad + len(s), i, dtype=np.int32))
    buf.append("#" * pad)
    owner.append(np.full(pad, -1, dtype=np.int32))
    T = np.frombuffer("".join(buf).encode(), dtype=np.uint8)
    own = np.concatenate(owner)
    out = {}
    for p, q in zip(cur.pdb, cur.sequence):
        qa = np.frombuffer(q.encode(), dtype=np.uint8)
        L = len(qa)
        n = len(T) - L + 1
        cnt = np.zeros(n, dtype=np.int16)
        for i in range(L):
            cnt += (T[i:i + n] == qa[i])
        j = int(np.argmax(cnt))
        hit_owner = own[j:j + L]
        hit_owner = hit_owner[hit_owner >= 0]
        out[p] = (float(cnt[j]) / L, names[int(np.bincount(hit_owner).argmax())] if len(hit_owner) else "")
    return out


def cmd_leakage(a):
    w = Path(a.work)
    cur = pd.read_csv(w / "curation.csv", keep_default_na=False)
    cur = cur[cur.drop_reason == ""]
    q = w / "pep_query.fasta"
    with open(q, "w") as fh:
        for p, s in zip(cur.pdb, cur.sequence):
            fh.write(f">{p}\n{s}\n")
    mm = w / "mmseqs"
    if mm.exists():
        shutil.rmtree(mm)
    mm.mkdir()
    out = mm / "hits.m8"
    cmd = [MMSEQS, "easy-search", str(q), str(w / "train_seqs.fasta"), str(out),
           tempfile.mkdtemp(prefix="mmseqs_", dir=os.environ.get("TMPDIR", "/tmp")),  # node-local (NFS rm fails)
           "--exhaustive-search", "1", "-e", "1e30", "--min-seq-id", "0", "--max-seqs", "100000",
           "--min-ungapped-score", "0", "--seq-id-mode", "0",
           "--alignment-mode", "3", "--comp-bias-corr", "0", "--mask", "0",
           "--format-output", "query,target,fident,nident,alnlen,qlen,tlen,qstart,qend,evalue,bits",
           "--threads", str(a.nproc), "-v", "1"]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    hits = pd.read_csv(out, sep="\t", header=None,
                       names="query,target,fident,nident,alnlen,qlen,tlen,qstart,qend,evalue,bits".split(","))
    # mmseqs' nident column is 0 in this build -> matches = round(fident * alnlen)
    hits["qid"] = np.round(hits.fident * hits.alnlen) / hits.qlen
    best = hits.sort_values(["qid", "bits"], ascending=False).groupby("query").head(1).set_index("query")
    ung = _ungapped_identity(cur, w / "train_seqs.fasta")
    # chance baseline: same scan with each peptide's sequence shuffled (fixed seed)
    rs = np.random.default_rng(0)
    shuf = cur.copy()
    shuf["sequence"] = ["".join(rs.permutation(list(q))) for q in cur.sequence]
    ung_null = _ungapped_identity(shuf, w / "train_seqs.fasta")
    codes = set((w / "train_codes.txt").read_text().split())
    rows = []
    for p in cur.pdb:
        r = dict(pdb=p, mmseqs_identity=0.0, mmseqs_hit="", best_hit_fident=np.nan, best_hit_alnlen=0,
                 best_hit_evalue=np.nan)
        if p in best.index:
            b = best.loc[p]
            r.update(mmseqs_identity=round(float(b.qid), 4), mmseqs_hit=b.target,
                     best_hit_fident=float(b.fident), best_hit_alnlen=int(b.alnlen),
                     best_hit_evalue=float(b.evalue))
        ui, uh = ung[p]
        r.update(ungapped_identity=round(ui, 4), ungapped_hit=uh)
        if r["mmseqs_identity"] >= ui:
            r.update(max_identity_to_train=r["mmseqs_identity"], best_hit=r["mmseqs_hit"])
        else:
            r.update(max_identity_to_train=r["ungapped_identity"], best_hit=uh)
        r["shuffled_null_identity"] = round(ung_null[p][0], 4)
        r["homolog_in_train"] = bool(r["best_hit_evalue"] < 1e-3) if p in best.index else False
        r["pdb_code_in_train"] = p.upper() in codes
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(w / "leakage.csv", index=False)
    v = df.max_identity_to_train
    print(f"[leakage] n={len(df)} <=0.3: {(v <= 0.3).sum()} <=0.4: {(v <= 0.4).sum()} "
          f">=0.9: {(v >= 0.9).sum()} code_in_train: {df.pdb_code_in_train.sum()} "
          f"homolog(E<1e-3): {df.homolog_in_train.sum()} null median {df.shuffled_null_identity.median():.3f}")


# --------------------------------------------------------------------------- #
# ESMFold                                                                     #
# --------------------------------------------------------------------------- #

def cmd_esmfold(a):
    import torch
    from transformers import AutoTokenizer, EsmForProteinFolding
    w = Path(a.work)
    cur = pd.read_csv(w / "curation.csv", keep_default_na=False)
    cur = cur[cur.drop_reason == ""]
    if a.ids:
        cur = cur[cur.pdb.isin(a.ids.split(","))]
    od = w / "esmfold"
    od.mkdir(exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
    model = model.to(dev).eval()
    if dev == "cuda":
        model.esm = model.esm.half()
    print(f"[esmfold] device={dev} n={len(cur)}", flush=True)
    for p, s in zip(cur.pdb, cur.sequence):
        fn = od / f"{p}.pdb"
        if fn.exists():
            continue
        with torch.no_grad():
            ids = tok([s], return_tensors="pt", add_special_tokens=False)["input_ids"].to(dev)
            out = model(ids)
        pdb_txt = model.output_to_pdb(out)[0]
        plddt = float(out["plddt"][0, :, 1].mean().cpu()) if out["plddt"].dim() == 3 else float(out["plddt"].mean())
        fn.write_text(pdb_txt)
        print(f"[esmfold] {p} L={len(s)} mean_pLDDT={plddt:.3f}", flush=True)


# --------------------------------------------------------------------------- #
# build                                                                       #
# --------------------------------------------------------------------------- #

def v2_block_counts(CB, k=10, cutoff=20.0):
    """Replicates the block-survival rule of protein_os._get_local_struct_v2
    (5-residue window + k nearest by CB; block dropped if any member CB is
    >= ``cutoff`` A from the centre CB; terminal centres use window fillers).
    Returns (n_interior_blocks, n_terminal_blocks). A decoy with 0 total blocks
    makes the current scorer raise 'every block was dropped'."""
    L = len(CB)
    D = np.linalg.norm(CB[:, None] - CB[None], axis=-1)
    n_int = n_term = 0
    for i in range(L):
        offs = [o for o in (0, -1, 1, -2, 2) if 0 <= i + o < L]
        win = [i + o for o in offs]
        terminal = len(win) < 5 or i < 2 or i >= L - 2
        n_need = (5 - len(win)) + k
        others = np.array([j for j in range(L) if j not in win])
        if len(others) < n_need:
            continue
        top = others[np.argsort(D[i, others])[:n_need]]
        if D[i, np.concatenate([win, top])].max() >= cutoff:
            continue
        if terminal:
            n_term += 1
        else:
            n_int += 1
    return n_int, n_term


def _bead_df(template, N, CA, C, CB):
    df = template[["chain_id", "group_num", "group_name"]].copy().reset_index(drop=True)
    gly = (df.group_name == "G").values
    CB = CB.copy()
    CB[gly] = CA[gly]
    for nm, X in (("n", N), ("ca", CA), ("c", C), ("cb", CB)):
        df[f"x{nm}"], df[f"y{nm}"], df[f"z{nm}"] = X[:, 0], X[:, 1], X[:, 2]
    return df


def _xyz(df, nm):
    return df[[f"x{nm}", f"y{nm}", f"z{nm}"]].values.astype(float)


def _ss_class(N, CA, C, O):
    try:
        import pydssp
        coord = np.stack([N, CA, C, O], 1)[None].astype(np.float32)
        ss = pydssp.assign(coord, out_type="c3")[0]
        ss = "".join(ss)
    except Exception as exc:  # noqa: BLE001
        print(f"[dssp] fail {exc}")
        return "", "unknown"
    fH, fE = ss.count("H") / len(ss), ss.count("E") / len(ss)
    if fH >= 0.3 and fE < 0.1:
        cls = "helix"
    elif fE >= 0.2 and fH < 0.1:
        cls = "hairpin"
    elif fH >= 0.15 and fE >= 0.1:
        cls = "mixed"
    else:
        cls = "coil"
    return ss, cls


def build_one(task):
    pdb, w, out = task
    w, out = Path(w), Path(out)
    beads = pd.read_pickle(w / "beads_cache" / f"{pdb}.pkl")
    dfs, models = beads["dfs"], beads["models"]
    CAs = np.stack([_xyz(d, "ca") for d in dfs])
    M = len(dfs)
    R = np.zeros((M, M))
    for i in range(M):
        for j in range(i + 1, M):
            R[i, j] = R[j, i] = rmsd_sup(CAs[i], CAs[j])
    med = int(np.argmin(R.mean(1)))
    tgt = dfs[med].reset_index(drop=True)
    Q = _xyz(tgt, "ca")
    Nn, CAn, Cn = _xyz(tgt, "n"), Q, _xyz(tgt, "c")
    tdir = out / pdb
    if tdir.exists():
        shutil.rmtree(tdir)
    tdir.mkdir(parents=True)
    rows, per = [], {}

    def add(name, df, source, sigma=np.nan):
        P = _xyz(df, "ca")
        assert len(P) == len(Q)
        g, pr = lddt_ca(P, Q)
        nbi, nbt = v2_block_counts(_xyz(df, "cb"))
        rows.append(dict(NAME=name, GDT_TS=round(gdt_ts(P, Q), 2), RMSD_CA=round(rmsd_sup(P, Q), 3),
                         LDDT_CA=round(g, 4), source=source, sigma=sigma,
                         n_blocks_v2=nbi + nbt, n_blocks_v2_interior=nbi))
        per[name] = np.round(pr, 4)
        df.to_csv(tdir / f"{name}_bead.csv", index=False, float_format="%.3f")

    add(f"{pdb}_native", tgt, "native")
    for k, (mn, d) in enumerate(zip(models, dfs)):
        if k != med:
            add(f"{pdb}_nmr{mn:02d}", d.reset_index(drop=True), "nmr")
    ef = w / "esmfold" / f"{pdb}.pdb"
    esm_plddt = np.nan
    if ef.exists():
        de = extract_beads_v2(str(ef), out_csv=None).reset_index(drop=True)
        if len(de) == len(tgt) and "".join(de.group_name) == "".join(tgt.group_name):
            P = _xyz(de, "ca")
            Rr, t = kabsch(P, Q)
            dd = _bead_df(tgt, _xyz(de, "n") @ Rr.T + t, P @ Rr.T + t, _xyz(de, "c") @ Rr.T + t,
                          _xyz(de, "cb") @ Rr.T + t)
            add(f"{pdb}_esmfold", dd, "esmfold")
            bf = [float(l[60:66]) for l in open(ef) if l.startswith("ATOM") and l[12:16].strip() == "CA"]
            esm_plddt = float(np.mean(bf)) if bf else np.nan
        else:
            print(f"[build] {pdb}: esmfold length/seq mismatch, skipped")

    phi, psi, omg = backbone_torsions(Nn, CAn, Cn)
    rng = np.random.default_rng(int.from_bytes(pdb.encode(), "little") % (2 ** 32))

    def rebuilt(ph, ps, om, name, source, sigma=np.nan):
        N, CA, C = nerf_build(ph, ps, om)
        Rr, t = kabsch(CA, Q)
        N, CA, C = N @ Rr.T + t, CA @ Rr.T + t, C @ Rr.T + t
        add(name, _bead_df(tgt, N, CA, C, ideal_cb(N, CA, C)), source, sigma)

    rebuilt(phi, psi, omg, f"{pdb}_ideal_rebuild", "ideal_rebuild", 0)
    for s in SIGMAS:
        for k in range(N_PER_SIGMA):
            dphi = rng.normal(0, s, len(phi))
            dpsi = rng.normal(0, s, len(psi))
            rebuilt(phi + dphi, psi + dpsi, omg, f"{pdb}_pert_s{s:02d}_{k + 1:02d}", "pert", s)
    L = len(phi)
    om180 = np.where(np.abs(omg) < 90, omg, 180.0)  # keep native cis bonds
    rebuilt(np.full(L, 180.0), np.full(L, 180.0), om180, f"{pdb}_ctrl_extended", "ctrl")
    rebuilt(np.full(L, -57.0), np.full(L, -47.0), om180, f"{pdb}_ctrl_helix", "ctrl")
    rebuilt(np.full(L, -75.0), np.full(L, 145.0), om180, f"{pdb}_ctrl_ppii", "ctrl")
    hp_phi, hp_psi = np.full(L, -120.0), np.full(L, 130.0)
    c = L // 2 - 1
    hp_phi[c], hp_psi[c], hp_phi[c + 1], hp_psi[c + 1] = 60.0, 30.0, 90.0, 0.0
    rebuilt(hp_phi, hp_psi, om180, f"{pdb}_ctrl_hairpin", "ctrl")

    lst = pd.DataFrame(rows)
    lst.to_csv(tdir / "list.csv", index=False)
    pr = pd.DataFrame.from_dict(per, orient="index", columns=[str(i + 1) for i in range(L)])
    pr.index.name = "NAME"
    pr.to_csv(tdir / "per_residue_lddt.csv")
    O = beads["O"][med]
    O = O.copy()
    for i in np.where(~np.isfinite(O).all(1))[0]:   # missing carbonyl O -> ideal placement
        u = Cn[i] - CAn[i]
        u /= np.linalg.norm(u)
        if i < len(O) - 1:
            v = Cn[i] - Nn[i + 1]
            v /= np.linalg.norm(v)
            u = u + v
            u /= np.linalg.norm(u)
        O[i] = Cn[i] + 1.231 * u
    ss, cls = _ss_class(Nn, CAn, Cn, O)
    info = dict(pdb=pdb, target_model=int(models[med]), n_models=M, length=L, dssp=ss, ss_class=cls,
                medoid_mean_rmsd=round(float(R[med].sum() / max(M - 1, 1)), 3),
                nmr_rmsd_max=round(float(R[med].max()), 3), n_decoys=len(lst),
                has_esmfold=bool((lst.source == "esmfold").any()),
                esmfold_rmsd=float(lst.loc[lst.source == "esmfold", "RMSD_CA"].iloc[0]) if (lst.source == "esmfold").any() else np.nan,
                esmfold_plddt=esm_plddt,
                ideal_rebuild_rmsd=float(lst.loc[lst.source == "ideal_rebuild", "RMSD_CA"].iloc[0]),
                rmsd_min=float(lst.RMSD_CA[lst.source != "native"].min()), rmsd_max=float(lst.RMSD_CA.max()),
                gdt_min=float(lst.GDT_TS.min()),
                n_decoys_zero_v2_blocks=int((lst.n_blocks_v2 == 0).sum()),
                native_v2_blocks=int(lst.n_blocks_v2.iloc[0]))
    json.dump(info, open(w / "build_info" / f"{pdb}.json", "w"))
    return info


def cmd_build(a):
    w = Path(a.work)
    (w / "build_info").mkdir(exist_ok=True)
    cur = pd.read_csv(w / "curation.csv", keep_default_na=False)
    ids = a.ids.split(",") if a.ids else list(cur.pdb[cur.drop_reason == ""])
    out = out_root(a)
    out.mkdir(parents=True, exist_ok=True)
    with Pool(a.nproc) as pool:
        for info in pool.imap_unordered(build_one, [(p, str(w), str(out)) for p in ids]):
            print(f"[build] {info['pdb']} L={info['length']} M={info['n_models']} "
                  f"med=model{info['target_model']} ss={info['ss_class']} n={info['n_decoys']} "
                  f"ideal={info['ideal_rebuild_rmsd']:.2f} rmsd {info['rmsd_min']:.2f}-{info['rmsd_max']:.2f}",
                  flush=True)


def cmd_finalize(a):
    w = Path(a.work)
    out = out_root(a)
    cur = pd.read_csv(w / "curation.csv", keep_default_na=False)
    kept = cur[cur.drop_reason == ""].copy()
    infos = pd.DataFrame([json.load(open(f)) for f in sorted((w / "build_info").glob("*.json"))])
    infos = infos[infos.pdb.isin(kept.pdb) & infos.pdb.map(lambda p: (out / p / "list.csv").exists())]
    leak = pd.read_csv(w / "leakage.csv") if (w / "leakage.csv").exists() else pd.DataFrame(columns=["pdb"])
    df = (kept[["pdb", "chain", "sequence", "paper_class", "n_disulfides", "disulfide_rich",
                "interchain_disulfides", "other_crosslinks", "n_protein_chains", "ligands", "caps",
                "solvent_flag", "solid_state", "pH", "temperature", "method"]]
          .merge(infos.drop(columns=[]), on="pdb", how="inner")
          .merge(leak, on="pdb", how="left"))
    cols = ["pdb", "length", "n_models", "target_model", "ss_class", "paper_class", "n_disulfides",
            "disulfide_rich", "solvent_flag", "solid_state", "max_identity_to_train", "best_hit",
            "homolog_in_train", "best_hit_evalue", "mmseqs_identity", "ungapped_identity",
            "shuffled_null_identity", "best_hit_fident", "pdb_code_in_train", "n_decoys", "has_esmfold", "esmfold_rmsd", "esmfold_plddt",
            "ideal_rebuild_rmsd", "rmsd_min", "rmsd_max", "native_v2_blocks", "n_decoys_zero_v2_blocks", "medoid_mean_rmsd", "nmr_rmsd_max", "dssp",
            "chain", "sequence", "interchain_disulfides", "other_crosslinks", "n_protein_chains",
            "ligands", "caps", "pH", "temperature", "method"]
    df = df[[c for c in cols if c in df.columns]].sort_values("pdb")
    df.to_csv(out / "targets_summary.csv", index=False)
    pd.DataFrame({"pdb": df.pdb}).to_csv(out / "pdb_no_missing_residue.csv", index=False)
    if a.id_table:
        # non-McDonald set (pep_new): own curation file name; never touch the training
        # exclude list (fasrc/peptide_exclude_ids.txt) unless --exclude_out is given explicitly
        cur.to_csv(out / "curation_all.csv", index=False)
        print(f"[finalize] {len(df)} targets -> {out} (id_table set: curation_all.csv)")
        if a.exclude_out == EXCLUDE_DEFAULT:
            return
    else:
        cur.to_csv(out / "curation_all588.csv", index=False)
    ex = Path(a.exclude_out)
    ex.write_text("\n".join(sorted(set(cur.pdb.str.upper()))) + "\n")
    print(f"[finalize] {len(df)} targets -> {out}; exclude list ({cur.pdb.nunique()} codes, all 588 "
          f"paper entries) -> {ex}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["curate", "train_fasta", "leakage", "esmfold", "build", "finalize"])
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", default=None, help="default nnef/data/decoys/pep_nmr")
    ap.add_argument("--ids", default=None)
    ap.add_argument("--nproc", type=int, default=4)
    ap.add_argument("--exclude_out", default=EXCLUDE_DEFAULT)
    ap.add_argument("--id_table", default=None,
                    help="CSV (pdb,seq) of targets not in the McDonald table (pep_new); "
                         "default: <work>/raw/mcdonald2023_tableS1.csv (pep_nmr)")
    a = ap.parse_args()
    {"curate": cmd_curate, "train_fasta": cmd_train_fasta, "leakage": cmd_leakage,
     "esmfold": cmd_esmfold, "build": cmd_build, "finalize": cmd_finalize}[a.cmd](a)


if __name__ == "__main__":
    main()
