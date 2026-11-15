"""Extract the LAST frame from yang_lr_big_sweep trajectory PDBs and write
them as bead CSVs that can be passed to md_eval_one.sh as INIT_BEAD for a
continuation run.

For each (model, protein, lr, seed) cell in yang_lr_big_sweep:
  1. Read <prot>_trajectory_native.pdb final MODEL
  2. Take the native bead CSV (has all columns: xn,yn,zn,xca,yca,zca,xc,yc,zc,xcb,ycb,zcb)
  3. Overwrite the xca/yca/zca columns with the final-frame Cα coordinates
  4. Keep xn/yn/zn/xc/yc/zc/xcb/ycb/zcb at their native values (they are
     not used in mode=CA inference anyway, just need to be present)
  5. Write to eval/md_eval/yang_lr_big_sweep_continue/init_beads/<RUN_BASE>/<PROT>_lr<LR>_seed<S>.csv

Run from repo root on the cluster (no GPU needed):
  python nnef/scripts/extract_final_frames_to_beads.py
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO     = Path(__file__).resolve().parents[2]
LR_DIR   = REPO / "eval" / "md_eval" / "yang_lr_big_sweep" / "native"
OUT_DIR  = REPO / "eval" / "md_eval" / "yang_lr_big_sweep_continue" / "init_beads"
NATIVE_DIR = REPO / "nnef" / "data" / "yang_small"

OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_last_frame(pdb_path: Path) -> np.ndarray:
    """Return (N, 3) array of Cα coords from the LAST MODEL in a multi-frame PDB."""
    frames, cur = [], []
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith("MODEL"):
                cur = []
            elif line.startswith("ATOM"):
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                cur.append((x, y, z))
            elif line.startswith("ENDMDL"):
                frames.append(cur); cur = []
    if cur:
        frames.append(cur)
    return np.asarray(frames[-1], dtype=np.float64)


def write_continue_bead(prot: str, final_ca: np.ndarray, out_csv: Path) -> None:
    """Overlay final-frame CA on the native bead CSV (keeps non-CA columns)."""
    native_csv = NATIVE_DIR / prot / f"{prot}_bead.csv"
    df = pd.read_csv(native_csv)
    if len(df) != final_ca.shape[0]:
        raise ValueError(
            f"{prot}: native bead has {len(df)} rows but trajectory final frame "
            f"has {final_ca.shape[0]} Cα — refusing to write"
        )
    df = df.copy()
    df["xca"] = final_ca[:, 0]
    df["yca"] = final_ca[:, 1]
    df["zca"] = final_ca[:, 2]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)


def main() -> None:
    if not LR_DIR.exists():
        raise SystemExit(f"[extract] {LR_DIR} not found — run on cluster after Priority 2 finishes")

    n_done = 0; n_skip = 0; n_err = 0
    pattern = re.compile(r"([A-Z0-9]+)_lr([0-9e\-\.]+)_seed(\d+)")
    for model_dir in sorted(LR_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        run_base = model_dir.name
        out_model = OUT_DIR / run_base
        for cell in sorted(model_dir.iterdir()):
            m = pattern.match(cell.name)
            if not m:
                continue
            prot, lr, seed = m.group(1), m.group(2), m.group(3)
            pdb = cell / f"{prot}_trajectory_native.pdb"
            if not pdb.exists():
                n_skip += 1
                continue
            out_csv = out_model / f"{prot}_lr{lr}_seed{seed}.csv"
            if out_csv.exists():
                n_skip += 1
                continue
            try:
                final = parse_last_frame(pdb)
                write_continue_bead(prot, final, out_csv)
                n_done += 1
            except Exception as e:
                print(f"[extract] ERR {cell.name}: {e}")
                n_err += 1

    print(f"\n[extract] wrote {n_done} bead CSVs, skipped {n_skip}, errored {n_err}")
    print(f"[extract] output dir: {OUT_DIR}")


if __name__ == "__main__":
    main()
