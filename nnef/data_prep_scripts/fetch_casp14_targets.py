#!/usr/bin/env python3
"""Download CASP14 regular-domain TS models + official GDT_TS tables, then write
``nnef/data/decoys/casp14/<TARGET>/list.csv`` for decoy evaluation.

The repository only shipped **T1053** under ``decoys/casp14/``. This script pulls
additional targets from the Prediction Center:

* Result tables: ``results/tables/casp14.res_tables.T.tar.gz`` (cached under
  ``--cache_dir``).
* Models: ``predictions/regular/<TARGET>.tar.gz`` extracted to ``--pdb_root/<TARGET>/``.

For multi-domain targets, use ``--pick_domain max`` (default) for the
highest ``-DN`` (e.g. ``T1053-D2.txt``), or ``min`` for the lowest
(usually ``D1``) — comparable to picking one evaluation unit per target.

After this script, build bead CSVs::

    PYTHONPATH=nnef python -m nnef.data_prep_scripts.regenerate_decoy_beads \\
        --decoy_set casp14 --pdb_root data_hh --overwrite --num_workers 8

Then run ``nnef/scripts/evaluate_decoys.py`` as usual (no ``--targets`` needed
if ``pdb_no_missing_residue.csv`` lists every prepared target).

Examples::

    # Fetch several targets (downloads + list.csv + updates pdb_no_missing_residue.csv)
    python -m nnef.data_prep_scripts.fetch_casp14_targets \\
        --targets T1024,T1025,T1026

    # Skip tarball download if raw models already live under data_hh/<TARGET>/
    python -m nnef.data_prep_scripts.fetch_casp14_targets \\
        --targets T1027 --skip_models_download
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Iterable, List, Optional, Set

import pandas as pd

_NNEF_DIR = Path(__file__).resolve().parent.parent
if str(_NNEF_DIR) not in sys.path:
    sys.path.insert(0, str(_NNEF_DIR))

from paths import REPO_ROOT, data_path, ensure_dir  # noqa: E402

RES_TABLES_URL = (
    "https://predictioncenter.org/download_area/CASP14/results/tables/"
    "casp14.res_tables.T.tar.gz"
)
PREDICTIONS_REGULAR = (
    "https://predictioncenter.org/download_area/CASP14/predictions/regular"
)

# Same column layout as ``nnef/data/decoys/casp14/T1053/list_combine.py``.
_RES_COLS = [
    "idx", "Model", "GR", "GDT_TS", "NP_P", "RANK", "Z_M1_GDT", "Z_MA_GDT", "GDT_HA",
    "GDC_SC", "GDC_ALL", "RMS_CA", "RMS_ALL", "NP", "err", "AL0_P", "AL4_P", "ALI_P",
    "LGA_S", "RMSD_L", "Z_score_M", "Z_score_D", "Al_Res", "RMSD_D", "MolPrb_Score",
    "LDDT", "SphGr", "CAD_AA", "RPF", "CODM", "DFM", "Handed", "SOV", "CE", "QCS",
    "CONTS", "TMscore", "Dali_raw", "FlexE", "QSE", "CAD_SS", "MolPrb_clash",
    "MolPrb_rotout", "MolPrb_ramout", "MolPrb_ramfv",
]


def _default_cache_dir() -> str:
    return os.path.join(REPO_ROOT, "data_hh", "casp14_cache")


def _default_pdb_root() -> str:
    return os.path.join(REPO_ROOT, "data_hh")


def _ensure_res_tables(cache_dir: str) -> Path:
    """Extract ``casp14.res_tables.T.tar.gz`` into ``cache_dir/casp14_res_tables_T``."""
    dest = Path(cache_dir) / "casp14_res_tables_T"
    marker = dest / ".extract_ok"
    tgz = Path(cache_dir) / "casp14.res_tables.T.tar.gz"
    ensure_dir(str(cache_dir))
    if not marker.is_file():
        if not tgz.is_file():
            print(f"[fetch_casp14] downloading {RES_TABLES_URL}")
            urllib.request.urlretrieve(RES_TABLES_URL, str(tgz))
        print(f"[fetch_casp14] extracting {tgz.name} -> {dest}")
        if dest.is_dir():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        with tarfile.open(tgz, "r:gz") as tf:
            tf.extractall(dest)
        marker.write_text("ok\n", encoding="utf-8")
    return dest


def _pick_domain_table(tables_dir: Path, target_id: str, pick: str) -> Path:
    matches = sorted(glob.glob(str(tables_dir / f"{target_id}-D*.txt")))
    if not matches:
        raise FileNotFoundError(
            f"No {target_id}-D*.txt under {tables_dir} — check target id "
            f"(use the same name as predictions/regular/{target_id}.tar.gz).",
        )
    if pick == "max":
        return Path(matches[-1])
    if pick == "min":
        return Path(matches[0])
    raise ValueError(f"pick_domain must be 'min' or 'max', got {pick!r}")


def _parse_domain_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        delim_whitespace=True,
        comment="#",
        names=_RES_COLS,
        na_values=["N/A"],
    )
    df["NAME"] = df["Model"].astype(str).str.replace(r"-D\d+$", "", regex=True)
    out = df[["NAME", "GDT_TS"]].copy()
    out = out.drop_duplicates(subset=["NAME"], keep="first")
    return out


def _download_file(url: str, dest: Path) -> None:
    print(f"[fetch_casp14] downloading {url}")
    urllib.request.urlretrieve(url, str(dest))


def _extract_prediction_tar(tgz: Path, dest_parent: Path) -> None:
    # Archives contain a single top-level directory Txxxx/ with model files.
    print(f"[fetch_casp14] extracting {tgz.name} -> {dest_parent}")
    with tarfile.open(tgz, "r:gz") as tf:
        tf.extractall(dest_parent)


def _write_list_and_sidecar(
    target_id: str,
    domain_table: Path,
    list_csv: Path,
) -> None:
    df = _parse_domain_table(domain_table)
    ensure_dir(str(list_csv.parent))
    df.to_csv(list_csv, index=False)
    side = list_csv.parent / "score_domain.txt"
    side.write_text(domain_table.name + "\n", encoding="utf-8")
    print(
        f"[fetch_casp14] wrote {list_csv} ({len(df)} decoys) "
        f"from {domain_table.name}",
    )


def _merge_pdb_list(target_ids: Iterable[str], path: Path) -> None:
    ids: Set[str] = set()
    if path.is_file():
        prev = pd.read_csv(path)
        if "pdb" in prev.columns:
            ids.update(prev["pdb"].astype(str).tolist())
    ids.update(target_ids)
    # CASP14 ids sort lexicographically; keeps T1024 before T1053.
    sorted_ids = sorted(ids)
    pd.DataFrame({"pdb": sorted_ids}).to_csv(path, index=False)
    print(f"[fetch_casp14] updated {path} ({len(sorted_ids)} targets)")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--targets",
        required=True,
        help="Comma-separated CASP14 target ids (e.g. T1024,T1053,T1045s1).",
    )
    p.add_argument(
        "--pdb_root",
        default=_default_pdb_root(),
        help=f"Where prediction archives are extracted (default: {_default_pdb_root()}).",
    )
    p.add_argument(
        "--cache_dir",
        default=_default_cache_dir(),
        help=f"Cache for res_tables tarball (default: {_default_cache_dir()}).",
    )
    p.add_argument(
        "--skip_models_download",
        action="store_true",
        help="Do not fetch predictions/regular/<T>.tar.gz; expect raw models under "
        "pdb_root/<T>/ already.",
    )
    p.add_argument(
        "--force_extract",
        action="store_true",
        help="When downloading models: remove existing pdb_root/<T>/ before extract. "
        "Default is to skip extraction if that directory already exists.",
    )
    p.add_argument(
        "--pick_domain",
        choices=("min", "max"),
        default="max",
        help="Which official domain table to use when multiple -DN.txt exist "
        "(default: max, same spirit as existing T1053 which uses D2 when present).",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print actions only.",
    )
    args = p.parse_args(argv)

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    if not targets:
        print("[fetch_casp14] no targets", file=sys.stderr)
        return 1

    pdb_root = Path(os.path.expanduser(args.pdb_root))
    cache_dir = Path(os.path.expanduser(args.cache_dir))
    decoys_root = Path(data_path("decoys", "casp14"))
    pdb_list_csv = decoys_root / "pdb_no_missing_residue.csv"

    if args.dry_run:
        print(f"[fetch_casp14] would process targets: {targets}")
        print(f"  pdb_root={pdb_root}")
        print(f"  cache_dir={cache_dir}")
        return 0

    tables_dir = _ensure_res_tables(str(cache_dir))

    for tid in targets:
        domain_path = _pick_domain_table(tables_dir, tid, args.pick_domain)
        list_out = decoys_root / tid / "list.csv"
        _write_list_and_sidecar(tid, domain_path, list_out)

        if args.skip_models_download:
            ensure_dir(str(pdb_root / tid))
            shutil.copy2(domain_path, pdb_root / tid / domain_path.name)
            print(f"[fetch_casp14] skip download for {tid}")
            continue

        tgt_dir = pdb_root / tid
        if tgt_dir.is_dir() and any(tgt_dir.iterdir()) and not args.force_extract:
            print(
                f"[fetch_casp14] keep existing {tgt_dir} (use --force_extract to replace)",
            )
        else:
            if tgt_dir.is_dir() and args.force_extract:
                shutil.rmtree(tgt_dir)
            url = f"{PREDICTIONS_REGULAR}/{tid}.tar.gz"
            tgz = cache_dir / f"{tid}.tar.gz"
            if not tgz.is_file():
                _download_file(url, tgz)
            else:
                print(f"[fetch_casp14] using cached {tgz.name}")
            ensure_dir(str(pdb_root))
            _extract_prediction_tar(tgz, pdb_root)

        ensure_dir(str(pdb_root / tid))
        shutil.copy2(domain_path, pdb_root / tid / domain_path.name)

    _merge_pdb_list(targets, pdb_list_csv)
    print(
        "[fetch_casp14] next: regenerate bead CSVs, e.g.\n"
        f"  cd {REPO_ROOT} && PYTHONPATH=nnef python -m nnef.data_prep_scripts.regenerate_decoy_beads \\\n"
        f"    --decoy_set casp14 --pdb_root {pdb_root} --targets {','.join(targets)} \\\n"
        f"    --overwrite --num_workers 8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
