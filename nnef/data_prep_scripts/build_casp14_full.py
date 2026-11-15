#!/usr/bin/env python3
"""Build the large held-out CASP14 decoy benchmark ``nnef/data/decoys/casp14_full``.

Same on-disk format as ``nnef/data/decoys/casp14`` (per-target ``list.csv`` with
NAME,GDT_TS + one ``<NAME>_bead.csv`` per decoy, set-level
``pdb_no_missing_residue.csv``). Beads are produced with the repo's canonical
converter ``local_extractor_v2.extract_beads_v2`` (the same one
``regenerate_decoy_beads.py`` used for the existing CASP14 beads; Gly CB = CA;
residues missing N/CA/C, or CB for non-Gly, are dropped).

Selection rules
---------------
* Evaluation unit (GDT_TS source, official ``casp14.res_tables.T``):
    - ``eval_unit=D1``    : target whose ONLY table is ``<T>-D1.txt`` (single-domain
                            evaluation unit; D1 is the whole evaluated region).
    - ``eval_unit=whole`` : target that has a whole-target table ``<T>.txt``
                            (multi-domain targets, plus T1054/T1095 whose single
                            domain is a trimmed part of the chain). GDT_TS here is
                            the official whole-chain LGA GDT_TS.
  Targets with no table (cancelled/not evaluated) are skipped.
* Every TS model ``<T>TS<grp>_<1..5>`` with a finite GDT_TS is kept, unless its bead
  residue count < 50% of the target sequence length (casp14.seq.txt) or it fails to
  parse. Chain breaks are kept (scorer segments by group_num adjacency).
* Targets with >= 50 surviving decoys are kept.

Usage (on FASRC, repo root ~/nnef, python -s with ~/envs/nnef)::

    python -s nnef/data_prep_scripts/build_casp14_full.py plan   --work W
    python -s nnef/data_prep_scripts/build_casp14_full.py build  --work W --target T1026
    python -s nnef/data_prep_scripts/build_casp14_full.py finalize --work W
    python -s nnef/data_prep_scripts/build_casp14_full.py verify --work W   # T1026 vs casp14

``W`` holds raw downloads/tables (raw tarballs + unpacked PDBs are deleted after
each target is converted). Output root: ``--out`` (default nnef/data/decoys/casp14_full).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

_NNEF_DIR = Path(__file__).resolve().parent.parent
if str(_NNEF_DIR) not in sys.path:
    sys.path.insert(0, str(_NNEF_DIR))

from paths import data_path  # noqa: E402
from data_prep_scripts.local_extractor_v2 import extract_beads_v2  # noqa: E402

BASE = "https://predictioncenter.org/download_area/CASP14"
RES_TABLES_URL = f"{BASE}/results/tables/casp14.res_tables.T.tar.gz"
PRED_URL = f"{BASE}/predictions/regular/{{t}}.tar.gz"
SEQ_URL = f"{BASE}/sequences/casp14.seq.txt"
TARGETLIST_URL = "https://predictioncenter.org/casp14/targetlist.cgi"

MIN_FRAC = 0.5
MIN_DECOYS = 50
MODEL_RE = re.compile(r"^(T\d{4}(?:s\d)?TS\d{3}_[1-5])$")


def _safe_extract(tgz, dest):
    with tarfile.open(tgz, "r:gz") as tf:
        if hasattr(tarfile, "data_filter"):
            tf.extractall(dest, filter="data")
        else:
            for m in tf.getmembers():
                if m.name.startswith("/") or ".." in m.name.split("/"):
                    raise RuntimeError(f"unsafe member {m.name}")
            tf.extractall(dest)


def _fetch(url, dest):
    if not os.path.isfile(dest):
        print(f"[get] {url}", flush=True)
        urllib.request.urlretrieve(url, dest + ".part")
        os.replace(dest + ".part", dest)


def _parse_res_table(path):
    """Official LGA table -> DataFrame(NAME, GDT_TS). Model col is field 2."""
    rows = []
    with open(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            f = line.split()
            if len(f) < 4:
                continue
            name = re.sub(r"-D\d+$", "", f[1])
            try:
                gdt = float(f[3])
            except ValueError:
                gdt = np.nan
            rows.append((name, gdt))
    df = pd.DataFrame(rows, columns=["NAME", "GDT_TS"]).drop_duplicates("NAME", keep="first")
    return df


def _parse_seqs(path):
    seqs, cur = {}, None
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            cur = line[1:].split()[0]
            if cur in seqs:      # duplicated header (T1085) -> keep first
                cur = None
            else:
                seqs[cur] = ""
        elif cur:
            seqs[cur] += line
    return seqs


def _parse_pdb_codes(path):
    if not os.path.isfile(path):
        return {}
    txt = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", open(path, errors="ignore").read()))
    parts = re.split(r"\b\d+\. (T1\d{3}(?:s\d)?) ", txt)
    out = {}
    for i in range(1, len(parts) - 1, 2):
        m = re.search(r"PDB code (\w{4})", parts[i + 1])
        out[parts[i]] = m.group(1).lower() if m else ""
    return out


def cmd_plan(a):
    w = Path(a.work)
    (w / "raw").mkdir(parents=True, exist_ok=True)
    _fetch(RES_TABLES_URL, str(w / "raw" / "res_tables.tar.gz"))
    tdir = w / "tables"
    if not tdir.is_dir():
        tdir.mkdir()
        _safe_extract(str(w / "raw" / "res_tables.tar.gz"), str(tdir))
    _fetch(SEQ_URL, str(w / "casp14.seq.txt"))
    try:
        _fetch(TARGETLIST_URL, str(w / "targetlist.html"))
    except Exception as exc:  # noqa: BLE001
        print(f"[plan] targetlist fetch failed: {exc}")
    seqs = _parse_seqs(str(w / "casp14.seq.txt"))
    codes = _parse_pdb_codes(str(w / "targetlist.html"))

    tables = sorted(p.name[:-4] for p in tdir.glob("T*.txt"))
    targets = sorted({re.sub(r"-D\d+$", "", t) for t in tables})
    plan = []
    for t in targets:
        doms = [x for x in tables if x.startswith(t + "-D")]
        if t in tables:
            unit, table = "whole", f"{t}.txt"
        elif doms == [f"{t}-D1"]:
            unit, table = "D1", f"{t}-D1.txt"
        else:
            print(f"[plan] skip {t}: domains {doms} and no whole-target table")
            continue
        if t not in seqs:
            print(f"[plan] skip {t}: no sequence")
            continue
        plan.append(dict(target=t, eval_unit=unit, table=table, n_domains=len(doms),
                         length=len(seqs[t]), pdb_code=codes.get(t, "")))
    json.dump(plan, open(w / "plan.json", "w"), indent=1)
    with open(w / "targets_all.txt", "w") as fh:
        fh.write("\n".join(p["target"] for p in plan) + "\n")
    print(f"[plan] {len(plan)} targets "
          f"({sum(p['eval_unit'] == 'D1' for p in plan)} D1, "
          f"{sum(p['eval_unit'] == 'whole' for p in plan)} whole) -> {w/'plan.json'}")


def cmd_build(a):
    w = Path(a.work)
    plan = {p["target"]: p for p in json.load(open(w / "plan.json"))}
    out_root = Path(a.out)
    for t in a.target.split(","):
        p = plan[t]
        tgz = w / "raw" / f"{t}.tar.gz"
        xdir = w / "raw" / f"{t}_x"
        _fetch(PRED_URL.format(t=t), str(tgz))
        if xdir.is_dir():
            shutil.rmtree(xdir)
        xdir.mkdir()
        _safe_extract(str(tgz), str(xdir))
        files = {os.path.basename(f): f for f in glob.glob(str(xdir / "**" / "*"), recursive=True)
                 if os.path.isfile(f)}
        gdt = _parse_res_table(str(w / "tables" / p["table"]))
        gdt = gdt[gdt.NAME.str.match(MODEL_RE)]
        tdir = out_root / t
        if tdir.is_dir():
            shutil.rmtree(tdir)
        tdir.mkdir(parents=True)
        stats = dict(table_rows=len(gdt), nan_gdt=0, no_pdb=0, parse_err=0, short=0, kept=0)
        keep = []
        min_res = MIN_FRAC * p["length"]
        for name, g in zip(gdt.NAME, gdt.GDT_TS):
            if not np.isfinite(g):
                stats["nan_gdt"] += 1
                continue
            if name not in files:
                stats["no_pdb"] += 1
                continue
            out_csv = str(tdir / f"{name}_bead.csv")
            try:
                df = extract_beads_v2(files[name], out_csv=None)
            except Exception as exc:  # noqa: BLE001
                stats["parse_err"] += 1
                print(f"[build] {name}: {exc}")
                continue
            if len(df) < min_res:
                stats["short"] += 1
                continue
            # identical to extract_beads_v2(..., out_csv=...) serialisation
            df.to_csv(out_csv, index=False, float_format="%.3f")
            keep.append((name, g))
            stats["kept"] += 1
        pd.DataFrame(keep, columns=["NAME", "GDT_TS"]).to_csv(tdir / "list.csv", index=False)
        (tdir / "score_domain.txt").write_text(p["table"] + "\n")
        stats.update(p)
        json.dump(stats, open(w / f"stats_{t}.json", "w"))
        if not a.keep_raw:
            shutil.rmtree(xdir)
            tgz.unlink()
        print(f"[build] {t}: {stats}", flush=True)


def cmd_finalize(a):
    w = Path(a.work)
    out_root = Path(a.out)
    rows = []
    for sf in sorted(w.glob("stats_*.json")):
        s = json.load(open(sf))
        t = s["target"]
        lst = out_root / t / "list.csv"
        if not lst.is_file():
            continue
        df = pd.read_csv(lst)
        ok = len(df) >= MIN_DECOYS
        rows.append(dict(
            target=t, eval_unit=s["eval_unit"], gdt_table=s["table"], n_domains=s["n_domains"],
            length=s["length"], n_decoys=len(df),
            gdt_min=df.GDT_TS.min() if len(df) else np.nan,
            gdt_median=df.GDT_TS.median() if len(df) else np.nan,
            gdt_max=df.GDT_TS.max() if len(df) else np.nan,
            n_table_rows=s["table_rows"], n_nan_gdt=s["nan_gdt"], n_no_pdb=s["no_pdb"],
            n_parse_err=s["parse_err"], n_short=s["short"],
            pdb_code=s.get("pdb_code", ""),
            held_out="yes (CASP14, released after training cutoff)",
            included=ok,
        ))
        if not ok:
            print(f"[finalize] drop {t}: {len(df)} decoys < {MIN_DECOYS}")
            shutil.rmtree(out_root / t)
    summ = pd.DataFrame(rows)
    summ.to_csv(out_root / "targets_summary.csv", index=False, float_format="%.2f")
    inc = summ[summ.included]
    pd.DataFrame({"pdb": inc.target}).to_csv(out_root / "pdb_no_missing_residue.csv", index=False)
    for unit in ("D1", "whole"):
        (out_root / f"targets_{unit}.txt").write_text(
            "\n".join(inc[inc.eval_unit == unit].target) + "\n")
    (out_root / "targets_all.txt").write_text("\n".join(inc.target) + "\n")
    print(f"[finalize] {len(inc)} targets, {int(inc.n_decoys.sum())} decoys "
          f"(D1={int((inc.eval_unit == 'D1').sum())}, whole={int((inc.eval_unit == 'whole').sum())})")


def cmd_verify(a):
    """Rebuild every T1026 bead in the existing casp14 set and byte-compare."""
    w = Path(a.work)
    tgz = w / "raw" / "T1026.tar.gz"
    xdir = w / "raw" / "T1026_verify"
    _fetch(PRED_URL.format(t="T1026"), str(tgz))
    if xdir.is_dir():
        shutil.rmtree(xdir)
    xdir.mkdir()
    _safe_extract(str(tgz), str(xdir))
    ref = Path(data_path("decoys", "casp14", "T1026"))
    names = pd.read_csv(ref / "list.csv").NAME
    same = diff = 0
    tmp = str(w / "verify_tmp.csv")
    for n in names:
        extract_beads_v2(str(xdir / "T1026" / n), out_csv=tmp)
        if open(tmp).read() == open(ref / f"{n}_bead.csv").read():
            same += 1
        else:
            diff += 1
    shutil.rmtree(xdir)
    print(f"[verify] T1026: {same} byte-identical, {diff} differ (of {len(names)})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["plan", "build", "finalize", "verify"])
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", default=data_path("decoys", "casp14_full"))
    ap.add_argument("--target", default="")
    ap.add_argument("--keep_raw", action="store_true")
    a = ap.parse_args()
    {"plan": cmd_plan, "build": cmd_build, "finalize": cmd_finalize, "verify": cmd_verify}[a.cmd](a)


if __name__ == "__main__":
    main()
