"""Structural comparison figures for one Langevin-MD run.

Given an md_eval run dir (e.g. eval/md_eval/mode2_yang_legacy/native/
yang_legacy_7317784/T1026_seed0), produce thesis-quality figures that
compare the simulated trajectory against the proxy-native bead:

  figs/overlay_3d.png     -- 3D Cα trace: native + initial + mid + final
  figs/contact_map.png    -- Cα contact map (native | final | diff)
  figs/dist_shift.png     -- pairwise distance |Δ| heatmap (final - native)

CAVEAT: CASP14 has no native.pdb shipped. Mode 2 uses the top-GDT decoy
bead CSV as a proxy — same as what rmsd_to_native was measured against.
If you override NATIVE_BEAD at runtime, pass the same path here via
--native_bead so the figure matches the CSV numbers exactly.

Usage:
  python eval/md_eval/compare_vs_native.py \\
      --run_dir eval/md_eval/mode2_yang_legacy/native/yang_legacy_7317784/T1026_seed0

Batch (all runs for a target):
  for d in eval/md_eval/*/native/*/T1026_seed*; do
    python eval/md_eval/compare_vs_native.py --run_dir "$d"
  done
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, E402
import numpy as np
import pandas as pd


def _read_trajectory_pdb(path: str) -> np.ndarray:
    """Parse a multi-MODEL Cα-only PDB into (F, N, 3) float array."""
    frames: list[list[tuple[float, float, float]]] = []
    current: list[tuple[float, float, float]] = []
    with open(path) as fh:
        for line in fh:
            if line.startswith('MODEL'):
                current = []
            elif line.startswith('ATOM'):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                current.append((x, y, z))
            elif line.startswith('ENDMDL'):
                frames.append(current)
                current = []
    if current:
        frames.append(current)
    return np.asarray(frames, dtype=np.float32)


def _read_bead_csv(path: str) -> np.ndarray:
    df = pd.read_csv(path)
    xyz_cols = [c for c in df.columns if c.lower() in ('x', 'y', 'z')]
    if len(xyz_cols) == 3:
        return df[xyz_cols].to_numpy(dtype=np.float32)
    # Fallback: last three numeric columns.
    return df.select_dtypes('number').iloc[:, -3:].to_numpy(dtype=np.float32)


def _kabsch(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Align P onto Q (both (N,3)); returns aligned P."""
    Pc = P - P.mean(0)
    Qc = Q - Q.mean(0)
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    return Pc @ R.T + Q.mean(0)


def _contact_map(coords: np.ndarray, cutoff: float = 8.0) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    return (dist < cutoff).astype(np.float32)


def _auto_native_bead(target: str, decoy_set: str, repo_root: str) -> str | None:
    list_csv = os.path.join(repo_root, 'nnef/data/decoys', decoy_set, target, 'list.csv')
    if not os.path.isfile(list_csv):
        return None
    df = pd.read_csv(list_csv).dropna(subset=['GDT_TS'])
    if df.empty:
        return None
    top = df.sort_values('GDT_TS', ascending=False).iloc[0]['NAME']
    return os.path.join(repo_root, 'nnef/data/decoys', decoy_set, target, f'{top}_bead.csv')


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--run_dir', required=True)
    p.add_argument('--native_bead', default=None,
                   help='Override native bead CSV (default: auto-resolve top-GDT decoy)')
    p.add_argument('--contact_cutoff', type=float, default=8.0)
    p.add_argument('--repo_root', default='.')
    args = p.parse_args()

    meta_path = os.path.join(args.run_dir, 'meta.json')
    with open(meta_path) as fh:
        meta = json.load(fh)
    target = meta['target']
    md_mode = meta.get('md_mode', 'native')
    decoy_set = meta.get('decoy_set', 'casp14')

    traj_pdb = os.path.join(args.run_dir, f'{target}_trajectory_{md_mode}.pdb')
    traj = _read_trajectory_pdb(traj_pdb)  # (F, N, 3)
    n_frames, n_res, _ = traj.shape
    print(f'[compare] {n_frames} frames × {n_res} residues')

    native_bead = args.native_bead or _auto_native_bead(target, decoy_set, args.repo_root)
    if native_bead is None or not os.path.isfile(native_bead):
        raise SystemExit(f'[compare] native bead CSV not found: {native_bead!r}')
    native = _read_bead_csv(native_bead)
    assert native.shape[0] == n_res, f'length mismatch: native={native.shape[0]} traj={n_res}'

    # Align every frame onto native (Kabsch). Trajectory already starts from
    # native, but drift + tumbling still accumulates; aligning makes the
    # 3D overlay readable.
    aligned = np.stack([_kabsch(frame, native) for frame in traj])
    initial = aligned[0]
    midpoint = aligned[n_frames // 2]
    final = aligned[-1]

    fig_dir = os.path.join(args.run_dir, 'figs')
    os.makedirs(fig_dir, exist_ok=True)

    # --- 1. 3D Cα trace overlay ---------------------------------------------
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    traces = [
        ('native',  native,  'black', 2.5, 1.0),
        ('init',    initial, 'tab:green', 1.2, 0.9),
        ('mid',     midpoint, 'tab:orange', 1.2, 0.7),
        ('final',   final,   'tab:red', 1.6, 0.9),
    ]
    for label, coords, color, lw, alpha in traces:
        ax.plot(coords[:, 0], coords[:, 1], coords[:, 2],
                color=color, lw=lw, alpha=alpha, label=label)
    ax.set_title(f'{target} Cα trace — seed {meta["seed"]} '
                 f'(L={meta["L"]}, lr={meta["lr"]:.0e})')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_xlabel('x (Å)'); ax.set_ylabel('y (Å)'); ax.set_zlabel('z (Å)')
    fig.tight_layout()
    out = os.path.join(fig_dir, 'overlay_3d.png')
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f'[compare] wrote {out}')

    # --- 2. Contact map comparison ------------------------------------------
    cm_native = _contact_map(native, args.contact_cutoff)
    cm_final = _contact_map(final, args.contact_cutoff)
    cm_diff = cm_final - cm_native  # +1 = new contact, -1 = lost contact

    # Jaccard-like overlap stat.
    inter = np.logical_and(cm_native > 0, cm_final > 0).sum()
    union = np.logical_or(cm_native > 0, cm_final > 0).sum()
    jaccard = inter / max(union, 1)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].imshow(cm_native, cmap='Greys', origin='lower')
    axes[0].set_title(f'native contacts (<{args.contact_cutoff} Å)')
    axes[1].imshow(cm_final, cmap='Greys', origin='lower')
    axes[1].set_title('final-frame contacts')
    axes[2].imshow(cm_diff, cmap='RdBu_r', vmin=-1, vmax=1, origin='lower')
    axes[2].set_title(f'diff (red=new, blue=lost)  J={jaccard:.3f}')
    for ax in axes:
        ax.set_xlabel('residue j'); ax.set_ylabel('residue i')
    fig.suptitle(f'{target} contact map — seed {meta["seed"]}')
    fig.tight_layout()
    out = os.path.join(fig_dir, 'contact_map.png')
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f'[compare] wrote {out}  (Jaccard={jaccard:.3f})')

    # --- 3. Pairwise distance |Δ| heatmap -----------------------------------
    dmat = lambda X: np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    dd = np.abs(dmat(final) - dmat(native))
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    im = ax.imshow(dd, cmap='magma', origin='lower')
    ax.set_title(f'{target}  |Δd_ij|  final - native (Å)  '
                 f'mean={dd.mean():.2f}  max={dd.max():.2f}')
    ax.set_xlabel('residue j'); ax.set_ylabel('residue i')
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    out = os.path.join(fig_dir, 'dist_shift.png')
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f'[compare] wrote {out}')

    # --- 4. one-line summary you can paste into the thesis table ------------
    refs = pd.read_csv(os.path.join(args.run_dir, f'{target}_energy_rmsd.csv')).head(3)
    print()
    print(f'[summary] {target} seed={meta["seed"]}  '
          f'rmsd_final={meta["final_rmsd_to_native"]:.2f}Å  '
          f'jaccard_contacts={jaccard:.3f}  '
          f'mean_|Δd|={dd.mean():.2f}Å  max_|Δd|={dd.max():.2f}Å  '
          f'E_best={meta["energy_best"]:.1f} (native={meta["energy_native"]:.1f})')


if __name__ == '__main__':
    main()
