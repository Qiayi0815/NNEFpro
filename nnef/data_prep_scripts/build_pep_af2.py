#!/usr/bin/env python3
"""Build ``nnef/data/decoys/pep_af2``: AlphaFold2 model-selection benchmark for peptides.

Question: can an energy pick the best of many AF2 predictions of a peptide better than
AF2's own confidence (pLDDT / pTM)?

Targets: the peptides in ~/nnef/runs/pack/pep_md_targets.txt (pep_nmr targets; 40%-identity
cluster representatives, >= 15 aa). Sequence = the pep_nmr target bead sequence.

Predictions (LocalColabFold, /n/holylabs/LABS/atchan_lab/Lab/qzha/envs/localcolabfold;
fasrc/pep_af2_predict.slurm): alphafold2_ptm, 5 models x 5 seeds (--num-seeds 5),
--num-recycle 3, MSAs from the ColabFold MMseqs2 server (``msa`` set) and the same with
--use-dropout (``drop`` set). No templates, no relaxation. Output dirs
<work>/af2_<set>/ with ColabFold's standard ``<PDB>_unrelaxed_rank_*_model_M_seed_SSS.pdb``
and ``<PDB>_scores_rank_*_model_M_seed_SSS.json`` files.

Decoy set (pep_nmr format): per target ``list.csv`` + ``<NAME>_bead.csv`` (beads from
local_extractor_v2.extract_beads_v2, Gly CB = CA, chain_id/group_num copied from the pep_nmr
target, sequence checked) + ``per_residue_lddt.csv`` (Calpha lDDT vs the medoid target) +
``per_residue_plddt.csv``. NAMEs: ``<PDB>_af2_m{M}_s{S}`` (msa set) and
``<PDB>_af2_m{M}_s{S}_drop`` (dropout set), M = 1..5, S = seed; ``<PDB>_native`` = the pep_nmr
medoid NMR model (copied bead, source=native).

list.csv columns: NAME, GDT_TS, RMSD_CA, LDDT_CA (vs the medoid target, as pep_nmr),
gdt_ts_max_ens, rmsd_ca_min_ens, lddt_ca_max_ens (best over ALL deposited NMR models of the
entry, from <nmr_work>/beads_cache/<PDB>.pkl), plddt_mean (0-100), ptm, source
(native / af2_msa / af2_drop), model, seed.

Usage (repo ~/nnef; env /n/holylabs/LABS/atchan_lab/Lab/qzha/envs/pepbench)::

Other sets (e.g. pep_new) override the paths with --work / --nmr_work / --nmr_set / --out_set /
--targets (defaults = pep_af2); --blind keeps quality numbers (GDT, pLDDT) out of the stdout log
(pre-registered held-out sets).

    python -s nnef/data_prep_scripts/build_pep_af2.py fasta
    python -s nnef/data_prep_scripts/build_pep_af2.py build [--ids 1L2Y] [--nproc 16]
    python -s nnef/data_prep_scripts/build_pep_af2.py finalize
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
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
from local_extractor_v2 import extract_beads_v2  # noqa: E402
import build_pep_nmr as B  # noqa: E402

WORK = Path("/n/holylabs/LABS/atchan_lab/Lab/qzha/pep_af2_work")
NMR_WORK = Path("/n/holylabs/LABS/atchan_lab/Lab/qzha/pep_nmr_work")
NMR = Path(data_path("decoys", "pep_nmr"))
OUT = Path(data_path("decoys", "pep_af2"))
TARGETS = Path(os.path.expanduser("~/nnef/runs/pack/pep_md_targets.txt"))
SETS = {"msa": "", "drop": "_drop"}
PAT = re.compile(r"^(?P<pdb>\w+?)_unrelaxed_rank_\d+_alphafold2_ptm_model_(?P<m>\d)_seed_(?P<s>\d+)\.pdb$")


def target_ids():
    return [l.strip() for l in open(TARGETS) if l.strip()]


def cmd_fasta(a):
    meta = pd.read_csv(NMR / "targets_summary.csv").set_index("pdb")
    WORK.mkdir(parents=True, exist_ok=True)
    ids = target_ids()
    with open(WORK / "queries.fasta", "w") as fh:
        for p in ids:
            fh.write(f">{p}\n{meta.loc[p, 'sequence']}\n")
    # also a CSV (id,sequence), the input form colabfold_batch handles best for batches
    pd.DataFrame({"id": ids, "sequence": [meta.loc[p, "sequence"] for p in ids]}).to_csv(WORK / "queries.csv", index=False)
    print(f"[fasta] {len(ids)} queries -> {WORK / 'queries.fasta'}")


def build_one(pdb):
    tmpl = pd.read_csv(NMR / pdb / f"{pdb}_native_bead.csv").sort_values("group_num").reset_index(drop=True)
    seq = "".join(tmpl.group_name)
    Q = B._xyz(tmpl, "ca")
    ens = pd.read_pickle(NMR_WORK / "beads_cache" / f"{pdb}.pkl")
    E = [B._xyz(d.sort_values("group_num").reset_index(drop=True), "ca") for d in ens["dfs"]]
    tdir = OUT / pdb
    if tdir.exists():
        shutil.rmtree(tdir)
    tdir.mkdir(parents=True)
    rows, per, perp, bad = [], {}, {}, []

    def add(name, df, src, model=np.nan, seed=np.nan, plddt=None, ptm=np.nan):
        P = B._xyz(df, "ca")
        g, pr = B.lddt_ca(P, Q)
        ens_r = [B.rmsd_sup(P, X) for X in E]
        ens_g = [B.gdt_ts(P, X) for X in E]
        ens_l = [B.lddt_ca(P, X)[0] for X in E]
        rows.append(dict(NAME=name, GDT_TS=round(B.gdt_ts(P, Q), 2), RMSD_CA=round(B.rmsd_sup(P, Q), 3),
                         LDDT_CA=round(g, 4), gdt_ts_max_ens=round(max(ens_g), 2),
                         rmsd_ca_min_ens=round(min(ens_r), 3), lddt_ca_max_ens=round(max(ens_l), 4),
                         plddt_mean=round(float(np.mean(plddt)), 2) if plddt is not None else np.nan,
                         ptm=ptm, source=src, model=model, seed=seed))
        per[name] = np.round(pr, 4)
        if plddt is not None:
            perp[name] = np.round(plddt, 2)
        df.to_csv(tdir / f"{name}_bead.csv", index=False, float_format="%.3f")

    add(f"{pdb}_native", tmpl.copy(), "native")
    for sname, suf in SETS.items():
        d = WORK / f"af2_{sname}"
        for f in sorted(glob.glob(str(d / f"{pdb}_unrelaxed_rank_*.pdb"))):
            m = PAT.match(os.path.basename(f))
            if not m or m["pdb"] != pdb:
                continue
            M, S = int(m["m"]), int(m["s"])
            js = f.replace("_unrelaxed_", "_scores_").replace(".pdb", ".json")
            try:
                sc = json.load(open(js))
                df = extract_beads_v2(f, out_csv=None).sort_values("group_num").reset_index(drop=True)
                if "".join(df.group_name) != seq:
                    raise RuntimeError("sequence mismatch")
                df["chain_id"] = tmpl.chain_id.values
                df["group_num"] = tmpl.group_num.values
                add(f"{pdb}_af2_m{M}_s{S}{suf}", df, f"af2_{sname}", M, S,
                    np.asarray(sc["plddt"], float), float(sc.get("ptm", np.nan)))
            except Exception as exc:  # noqa: BLE001
                bad.append(f"{os.path.basename(f)}: {exc!r}"[:200])
    lst = pd.DataFrame(rows)
    lst.to_csv(tdir / "list.csv", index=False)
    cols = [str(i + 1) for i in range(len(Q))]
    pr = pd.DataFrame.from_dict(per, orient="index", columns=cols)
    pr.index.name = "NAME"
    pr.to_csv(tdir / "per_residue_lddt.csv")
    if perp:
        pp = pd.DataFrame.from_dict(perp, orient="index", columns=cols)
        pp.index.name = "NAME"
        pp.to_csv(tdir / "per_residue_plddt.csv")
    af = lst[lst.source != "native"]
    info = dict(pdb=pdb, length=len(Q), n_models=len(af), n_msa=int((af.source == "af2_msa").sum()),
                n_drop=int((af.source == "af2_drop").sum()), n_bad=len(bad), bad=bad)
    if len(af):
        top = af.loc[af.plddt_mean.idxmax()]
        info.update(gdt_ens_min=af.gdt_ts_max_ens.min(), gdt_ens_max=af.gdt_ts_max_ens.max(),
                    gdt_ens_median=af.gdt_ts_max_ens.median(),
                    rmsd_ens_min=af.rmsd_ca_min_ens.min(), rmsd_ens_max=af.rmsd_ca_min_ens.max(),
                    best_possible_gdt=af.gdt_ts_max_ens.max(), plddt_top1_gdt=float(top.gdt_ts_max_ens),
                    plddt_top1_name=top.NAME, plddt_max=af.plddt_mean.max(), plddt_min=af.plddt_mean.min())
    return info


def cmd_build(a):
    ids = a.ids.split(",") if a.ids else target_ids()
    OUT.mkdir(parents=True, exist_ok=True)
    (WORK / "build_info").mkdir(exist_ok=True)
    with Pool(a.nproc) as pool:
        for info in pool.imap_unordered(build_one, ids):
            json.dump(info, open(WORK / "build_info" / f"{info['pdb']}.json", "w"), default=float)
            if a.blind:
                print(f"[af2] {info['pdb']} n={info['n_models']} (msa {info['n_msa']}, drop {info['n_drop']}) "
                      f"bad={info['n_bad']}", flush=True)
                continue
            print(f"[af2] {info['pdb']} n={info['n_models']} (msa {info['n_msa']}, drop {info['n_drop']}) "
                  f"bad={info['n_bad']} best={info.get('best_possible_gdt', np.nan):.1f} "
                  f"plddt_top1={info.get('plddt_top1_gdt', np.nan):.1f}", flush=True)


def cmd_finalize(a):
    infos = [json.load(open(f)) for f in sorted((WORK / "build_info").glob("*.json"))]
    df = pd.DataFrame([{k: v for k, v in i.items() if k != "bad"} for i in infos])
    df = df[df.n_models > 0]
    modes = json.load(open(WORK / "msa_mode.json")) if (WORK / "msa_mode.json").exists() else {}
    df["msa_mode"] = df.pdb.map(lambda p: modes.get(p, modes.get("_default", "mmseqs2_server")))
    meta = pd.read_csv(NMR / "targets_summary.csv").set_index("pdb")
    for c in ("ss_class", "paper_class", "n_disulfides", "solvent_flag", "homolog_in_train", "max_identity_to_train"):
        df[c] = df.pdb.map(meta[c])
    df["plddt_pick_gap"] = df.best_possible_gdt - df.plddt_top1_gdt
    df.sort_values("pdb").to_csv(OUT / "targets_summary.csv", index=False)
    pd.DataFrame({"pdb": sorted(df.pdb)}).to_csv(OUT / "pdb_no_missing_residue.csv", index=False)
    if a.blind:
        print(f"[finalize] {len(df)} targets, {int(df.n_models.sum())} AF2 models -> {OUT}")
        return
    print(f"[finalize] {len(df)} targets, {int(df.n_models.sum())} AF2 models; "
          f"median plddt-pick gap {df.plddt_pick_gap.median():.2f}")


def main():
    global WORK, NMR_WORK, NMR, OUT, TARGETS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["fasta", "build", "finalize"])
    ap.add_argument("--ids", default=None)
    ap.add_argument("--nproc", type=int, default=8)
    # path overrides for other peptide sets (defaults = pep_af2, unchanged)
    ap.add_argument("--work", default=str(WORK), help="ColabFold work dir (af2_<set>/, queries, build_info)")
    ap.add_argument("--nmr_work", default=str(NMR_WORK), help="NMR work dir with beads_cache/")
    ap.add_argument("--nmr_set", default="pep_nmr", help="NMR decoy set under nnef/data/decoys")
    ap.add_argument("--out_set", default="pep_af2", help="output decoy set under nnef/data/decoys")
    ap.add_argument("--targets", default=str(TARGETS), help="target id list (one PDB per line)")
    ap.add_argument("--blind", action="store_true", help="do not print quality numbers (held-out sets)")
    a = ap.parse_args()
    WORK, NMR_WORK, TARGETS = Path(a.work), Path(a.nmr_work), Path(a.targets)
    NMR, OUT = Path(data_path("decoys", a.nmr_set)), Path(data_path("decoys", a.out_set))
    {"fasta": cmd_fasta, "build": cmd_build, "finalize": cmd_finalize}[a.cmd](a)


if __name__ == "__main__":
    main()
