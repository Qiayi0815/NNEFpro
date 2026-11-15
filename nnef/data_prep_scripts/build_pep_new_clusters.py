#!/usr/bin/env python3
"""Redundancy filtering for pep_new (post-AF2-cutoff NMR peptides; eval/decoys/PREREG_pep_new.md).

1. Remove pep_new peptides with >= 40% identity to any pep_nmr target sequence
   (nnef/data/decoys/pep_nmr/targets_summary.csv). Identity = MMseqs2 matches / query length,
   exactly the ``max_identity_to_train`` definition of build_pep_nmr.py ``leakage`` (exhaustive
   Smith-Waterman, alignment-mode 3, no E-value / coverage cut), so a short pep_new peptide
   contained in a longer pep_nmr one counts as identical, while short local hits do not count
   as 100%. Conservatively, a peptide is ALSO removed if any alignment satisfies the pep_nmr
   clustering rule (MMseqs2 fident >= 0.4 with query and target coverage >= 0.6), i.e. it would
   be called >= 40% identical in the sense of seq_clusters_id40.tsv (union of both definitions).
2. Cluster the remaining pep_new peptides at 40% identity with the same command that produced
   pep_nmr/seq_clusters_id40.tsv:
       mmseqs easy-cluster pep.fa clu tmp --min-seq-id 0.4 -c 0.6 --cov-mode 0 -s 7.5
   Representative = the MMseqs2 cluster representative (first column of the tsv), as for pep_nmr.

Outputs (in --out, default nnef/data/decoys/pep_new):
  seq_clusters_id40.tsv      rep<TAB>member (all non-redundant-vs-pep_nmr pep_new peptides)
  pep_new_reps.txt           one representative PDB per line (ColabFold targets)
  pep_nmr_similarity.csv     per curated pep_new peptide: best pep_nmr hit, identity, removed flag
Input: <work>/curation.csv (build_pep_nmr.py curate --id_table ...).

    python -s nnef/data_prep_scripts/build_pep_new_clusters.py --work W [--out DIR]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

_NNEF_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_NNEF_DIR))
from paths import data_path  # noqa: E402

MMSEQS = "/n/holylabs/LABS/atchan_lab/Lab/qzha/pep_nmr_work/tools/mmseqs/bin/mmseqs"
THR = 0.4
COLS = "query,target,fident,nident,alnlen,qlen,tlen,qstart,qend,tstart,tend,evalue,bits".split(",")


def write_fa(fn, ids, seqs):
    with open(fn, "w") as fh:
        for p, s in zip(ids, seqs):
            fh.write(f">{p}\n{s}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", default=data_path("decoys", "pep_new"))
    ap.add_argument("--nmr_set", default="pep_nmr")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    w, out = Path(a.work), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="pepnew_clu_", dir=os.environ.get("TMPDIR", "/tmp")))

    cur = pd.read_csv(w / "curation.csv", keep_default_na=False)
    cur = cur[cur.drop_reason == ""].sort_values("pdb").reset_index(drop=True)
    nmr = pd.read_csv(Path(data_path("decoys", a.nmr_set)) / "targets_summary.csv")
    write_fa(tmp / "new.fa", cur.pdb, cur.sequence)
    write_fa(tmp / "nmr.fa", nmr.pdb, nmr.sequence)

    # ---- 1. similarity to pep_nmr
    m8 = tmp / "vs_nmr.m8"
    subprocess.run([MMSEQS, "easy-search", str(tmp / "new.fa"), str(tmp / "nmr.fa"), str(m8), str(tmp / "s"),
                    "--exhaustive-search", "1", "-e", "1e30", "--min-seq-id", "0", "--max-seqs", "100000",
                    "--min-ungapped-score", "0", "--seq-id-mode", "0", "--alignment-mode", "3",
                    "--comp-bias-corr", "0", "--mask", "0", "--format-output", ",".join(COLS),
                    "--threads", str(a.threads), "-v", "1"], check=True)
    hits = pd.read_csv(m8, sep="\t", header=None, names=COLS) if m8.stat().st_size else pd.DataFrame(columns=COLS)
    # mmseqs' nident column is 0 in this build (see build_pep_nmr.leakage) -> matches = round(fident * alnlen)
    hits["qid"] = np.round(hits.fident * hits.alnlen) / hits.qlen
    best = hits.sort_values(["qid", "bits"], ascending=False).groupby("query").head(1).set_index("query")
    sim = pd.DataFrame({"pdb": cur.pdb, "sequence": cur.sequence, "length": cur.sequence.str.len()})
    sim["pep_nmr_best_hit"] = sim.pdb.map(best.target).fillna("")
    sim["pep_nmr_identity"] = sim.pdb.map(best.qid).fillna(0.0).round(4)
    sim["pep_nmr_hit_fident"] = sim.pdb.map(best.fident)
    sim["pep_nmr_hit_alnlen"] = sim.pdb.map(best.alnlen)
    # second definition: the pep_nmr clustering rule (fident>=0.4, both coverages>=0.6)
    hits["cov_q"] = (hits.qend - hits.qstart + 1) / hits.qlen
    hits["cov_t"] = (hits.tend - hits.tstart + 1) / hits.tlen
    clu_like = set(hits["query"][(hits.fident >= THR) & (hits.cov_q >= 0.6) & (hits.cov_t >= 0.6)])
    sim["clusterrule_link_to_pep_nmr"] = sim.pdb.isin(clu_like)
    sim["removed_sim_pep_nmr"] = (sim.pep_nmr_identity >= THR) | sim.clusterrule_link_to_pep_nmr
    sim.to_csv(out / "pep_nmr_similarity.csv", index=False)
    keep = sim[~sim.removed_sim_pep_nmr]
    qid_hit = sim.pep_nmr_identity >= THR
    print(f"[sim] curated {len(sim)}; removed {int(sim.removed_sim_pep_nmr.sum())} = query-norm identity >= {THR}: "
          f"{int(qid_hit.sum())} + cluster-rule link only: {int((sim.clusterrule_link_to_pep_nmr & ~qid_hit).sum())}; "
          f"remaining {len(keep)}")

    # ---- 2. 40% clusters within pep_new (pep_nmr recipe)
    write_fa(tmp / "pep.fa", keep.pdb, keep.sequence)
    subprocess.run([MMSEQS, "easy-cluster", str(tmp / "pep.fa"), str(tmp / "clu"), str(tmp / "c"),
                    "--min-seq-id", "0.4", "-c", "0.6", "--cov-mode", "0", "-s", "7.5", "-v", "1",
                    "--threads", str(a.threads)], check=True)
    clu = pd.read_csv(tmp / "clu_cluster.tsv", sep="\t", header=None, names=["rep", "member"])
    assert set(clu.member) == set(keep.pdb), "cluster members != kept peptides"
    (out / "seq_clusters_id40.tsv").write_text((tmp / "clu_cluster.tsv").read_text())
    reps = sorted(clu.rep.unique())
    (out / "pep_new_reps.txt").write_text("\n".join(reps) + "\n")
    sizes = clu.groupby("rep").size()
    print(f"[cluster] {len(keep)} peptides -> {len(reps)} clusters (singletons {int((sizes == 1).sum())}, "
          f"largest {int(sizes.max())}); reps -> {out / 'pep_new_reps.txt'}")


if __name__ == "__main__":
    main()
