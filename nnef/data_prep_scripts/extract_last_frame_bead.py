"""Extract the last frame of an MD trajectory PDB and write a bead CSV
suitable for use as `--init_bead` to md_eval.py.

The MD trajectory PDB only stores Cβ positions (labelled "CA" by md_eval.py's
convention — see eval/md_eval/plot_md_eval.py docstring). We splice those Cβ
positions onto the native bead CSV's N/CA/C columns, which the model does NOT
read when running in --mode CB (the default). This produces a bead CSV that
"looks like" a real bead extraction but with the trajectory's last-frame Cβ.

Usage:
  python nnef/data_prep_scripts/extract_last_frame_bead.py \\
      --traj_pdb  PATH/<target>_trajectory_native.pdb \\
      --native_bead PATH/<target>_bead.csv \\
      --out_csv   PATH/<target>_last_frame_bead.csv
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


def parse_last_frame_cb(pdb_path: str) -> np.ndarray:
    """Return (N, 3) Cβ coords of the LAST MODEL in the trajectory."""
    frames = []
    current = []
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith('MODEL'):
                current = []
            elif line.startswith('ATOM'):
                current.append((float(line[30:38]),
                                float(line[38:46]),
                                float(line[46:54])))
            elif line.startswith('ENDMDL'):
                if current:
                    frames.append(current)
                current = []
    if current:
        frames.append(current)
    if not frames:
        raise RuntimeError(f'no MODEL frames in {pdb_path}')
    return np.asarray(frames[-1], dtype=np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--traj_pdb', required=True)
    p.add_argument('--native_bead', required=True,
                   help='native bead CSV (provides N/CA/C cols + chain + group_num)')
    p.add_argument('--out_csv', required=True)
    args = p.parse_args()

    cb_last = parse_last_frame_cb(args.traj_pdb)
    native = pd.read_csv(args.native_bead)
    if cb_last.shape[0] != len(native):
        raise RuntimeError(
            f'length mismatch: trajectory last frame has {cb_last.shape[0]} '
            f'residues, native bead has {len(native)}')

    # Splice last-frame Cβ into native bead CSV's CB columns. Keep N/CA/C as
    # native values (model in --mode CB does not read those).
    out = native.copy()
    out['xcb'] = cb_last[:, 0]
    out['ycb'] = cb_last[:, 1]
    out['zcb'] = cb_last[:, 2]
    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f'wrote {args.out_csv}  ({len(out)} residues; '
          f'CB from {os.path.basename(args.traj_pdb)})')


if __name__ == '__main__':
    main()
