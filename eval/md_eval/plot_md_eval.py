"""Summarise and plot Langevin-MD evaluation outputs.

Expects the layout produced by ``fasrc/md_eval_one.sh``:

    eval/md_eval/<sweep>/<md_mode>/<run_base>/<target>_seed<N>/
        meta.json
        <target>_energy_rmsd.csv
        <target>_rmsf.csv
        <target>_trajectory_<md_mode>.pdb

Outputs (written alongside input root):
    summary.csv          -- one row per (sweep, md_mode, model, target, seed)
    plots/<sweep>/<md_mode>/<target>_energy.png
    plots/<sweep>/<md_mode>/<target>_rmsd.png
    plots/<sweep>/<md_mode>/<target>_rg.png
    plots/<sweep>/<md_mode>/<target>_rmsf.png
    plots/<sweep>/<md_mode>/summary_heatmap.png

Usage:
    python eval/md_eval/plot_md_eval.py --root eval/md_eval
    python eval/md_eval/plot_md_eval.py --root eval/md_eval --mode mode2
"""
from __future__ import annotations

import argparse
import json
import os
from glob import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Time-INdependent observables (RMSF, native-contacts %, secondary-structure
# content) — these don't depend on the arbitrary NNEF integrator step size,
# unlike RMSD-vs-step. See advisor note: NNEF Langevin has no physical time
# scale, so equilibrium-style averages are the only fair cross-model metric.
# --------------------------------------------------------------------------- #
def _read_ca_trajectory(pdb_path: str) -> 'np.ndarray | None':
    """Parse a multi-MODEL Cα-only PDB into (F, N, 3) float32. Returns None
    if the file is missing or unreadable."""
    if not os.path.isfile(pdb_path):
        return None
    frames = []
    current = []
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith('MODEL'):
                current = []
            elif line.startswith('ATOM'):
                try:
                    current.append((
                        float(line[30:38]), float(line[38:46]), float(line[46:54])
                    ))
                except ValueError:
                    return None
            elif line.startswith('ENDMDL'):
                if current:
                    frames.append(current)
                current = []
    if current:
        frames.append(current)
    if not frames:
        return None
    return np.asarray(frames, dtype=np.float32)


def _native_coords_from_bead(bead_csv: str, atom: str = 'CB') -> 'np.ndarray | None':
    """Read the native bead CSV and return (N, 3) coordinates for the requested
    atom. The MD trajectory PDB is mislabeled 'CA' but actually stores Cβ
    positions (NNEF runs with --mode CB by default), so atom='CB' is what we
    want for contact / RMSF references. atom='CA' is used only for static
    secondary-structure assignment on the native, where real Cα geometry is
    required (Cα-Cα = 3.8 Å vs Cβ-Cβ = 5-6 Å)."""
    if not os.path.isfile(bead_csv):
        return None
    df = pd.read_csv(bead_csv)
    cols = {
        'CA': ('xca', 'yca', 'zca'),
        'CB': ('xcb', 'ycb', 'zcb'),
    }[atom]
    return df[list(cols)].to_numpy(dtype=np.float32)


def _native_contacts_fraction(traj: 'np.ndarray', native: 'np.ndarray',
                               cutoff: float = 8.0,
                               seq_sep: int = 2) -> 'np.ndarray':
    """For each frame, fraction of native Cα-Cα contacts (|i-j|>=seq_sep and
    native distance < cutoff) that are preserved (distance < cutoff in frame).
    Returns (F,) array."""
    n = native.shape[0]
    diff = native[:, None, :] - native[None, :, :]
    dist_native = np.linalg.norm(diff, axis=-1)
    # Mask of native contacts (upper triangle only, |i-j| >= seq_sep)
    i_idx, j_idx = np.triu_indices(n, k=seq_sep)
    is_native_contact = dist_native[i_idx, j_idx] < cutoff
    pairs_i = i_idx[is_native_contact]
    pairs_j = j_idx[is_native_contact]
    if pairs_i.size == 0:
        return np.zeros(traj.shape[0], dtype=np.float32)
    # Per-frame: how many of those pairs are still < cutoff
    out = np.empty(traj.shape[0], dtype=np.float32)
    for k, frame in enumerate(traj):
        d = np.linalg.norm(frame[pairs_i] - frame[pairs_j], axis=-1)
        out[k] = float((d < cutoff).sum()) / float(pairs_i.size)
    return out


def _lddt_per_frame(traj: 'np.ndarray', native: 'np.ndarray',
                    cutoff: float = 15.0,
                    thresholds=(0.5, 1.0, 2.0, 4.0)) -> 'np.ndarray':
    """Per-frame lDDT (local Distance Difference Test) of trajectory vs native.

    For each residue i, lDDT_i = mean over interaction partners j (|i-j|>=1,
    d_native(i,j) < cutoff) of (fraction of error thresholds at which
    |d_md(i,j) - d_native(i,j)| < threshold). Per-frame lDDT = mean of lDDT_i.
    Returns (F,) array in [0, 1]; 1 = perfect distance preservation.

    Cross-protein comparable (length-normalized), insensitive to outlier
    residues (RMSD's main weakness). The metric used by AlphaFold (pLDDT) and
    CAMEO/CASP for quality assessment, here applied to MD trajectories."""
    N = native.shape[0]
    dn = np.linalg.norm(native[:, None, :] - native[None, :, :], axis=-1)
    iarr, jarr = np.indices((N, N))
    pair_mask = (np.abs(iarr - jarr) >= 1) & (dn < cutoff)
    n_partners = pair_mask.sum(axis=1).astype(np.float32)
    valid_res = n_partners > 0
    n_thr = float(len(thresholds))

    out = np.empty(traj.shape[0], dtype=np.float32)
    for k, frame in enumerate(traj):
        df = np.linalg.norm(frame[:, None, :] - frame[None, :, :], axis=-1)
        d_err = np.abs(df - dn)
        scores = np.zeros(N, dtype=np.float32)
        for t in thresholds:
            within = (d_err < t) & pair_mask  # (N, N) bool
            scores += within.sum(axis=1).astype(np.float32) / np.maximum(n_partners, 1)
        scores /= n_thr
        out[k] = float(scores[valid_res].mean()) if valid_res.any() else 0.0
    return out


def _ca_only_ss(coords: 'np.ndarray') -> str:
    """Cα-only secondary-structure assignment (PROSS-style, Srinivasan &
    Rose 1999). Uses two geometric features at residue i:

      - tau (3-Cα bond angle):  Cα(i-1)-Cα(i)-Cα(i+1)
      - alpha (4-Cα torsion):   Cα(i-1)-Cα(i)-Cα(i+1)-Cα(i+2)

    Thresholds (relaxed PROSS):
      Helix:  alpha ∈ [25°, 105°]   AND  tau ∈ [80°, 105°]
      Strand: |alpha| ≥ 105°         AND  tau ∈ [115°, 175°]

    Returns string of length L with chars 'H', 'E', 'C'. Then smooths so SS
    runs must be ≥ 3 residues (isolated H/E are demoted to coil)."""
    L = coords.shape[0]
    ss = ['C'] * L
    if L < 5:
        return ''.join(ss)

    def _bond_angle(a, b, c):
        v1 = a - b; v2 = c - b
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))

    def _torsion(p0, p1, p2, p3):
        b0 = p1 - p0; b1 = p2 - p1; b2 = p3 - p2
        b1n = b1 / (np.linalg.norm(b1) + 1e-9)
        v = b0 - np.dot(b0, b1n) * b1n
        w = b2 - np.dot(b2, b1n) * b1n
        x = float(np.dot(v, w))
        y = float(np.dot(np.cross(b1n, v), w))
        return float(np.degrees(np.arctan2(y, x)))

    for i in range(1, L - 2):
        tau = _bond_angle(coords[i - 1], coords[i], coords[i + 1])
        alpha = _torsion(coords[i - 1], coords[i], coords[i + 1], coords[i + 2])
        a_abs = abs(alpha)
        if 25 <= alpha <= 105 and 80 <= tau <= 105:
            ss[i] = 'H'
        elif a_abs >= 105 and 115 <= tau <= 175:
            ss[i] = 'E'

    # Smooth: SS runs must be ≥ 3 residues; demote shorter to coil.
    smoothed = list(ss)
    runs = []
    j = 0
    while j < L:
        k = j
        while k < L and smoothed[k] == smoothed[j]:
            k += 1
        runs.append((j, k, smoothed[j]))
        j = k
    for (a, b, lab) in runs:
        if lab in ('H', 'E') and (b - a) < 3:
            for k in range(a, b):
                smoothed[k] = 'C'
    return ''.join(smoothed)


def _ss_fractions(ss_string: str) -> tuple:
    L = len(ss_string)
    if L == 0:
        return 0.0, 0.0, 0.0
    h = ss_string.count('H') / L
    e = ss_string.count('E') / L
    c = 1.0 - h - e
    return h, e, c


def _ss_match_fraction(traj_ss: list, native_ss: str) -> 'np.ndarray':
    """For each frame, fraction of residues whose Cα-only SS label matches the
    native's. traj_ss is list of strings (one per frame). Returns (F,) array."""
    n = len(native_ss)
    out = np.empty(len(traj_ss), dtype=np.float32)
    for k, s in enumerate(traj_ss):
        if len(s) != n:
            out[k] = np.nan
        else:
            out[k] = sum(1 for a, b in zip(s, native_ss) if a == b) / n
    return out


def _load_run(run_dir: str) -> dict | None:
    meta_path = os.path.join(run_dir, 'meta.json')
    if not os.path.isfile(meta_path):
        return None
    with open(meta_path) as fh:
        meta = json.load(fh)

    target = meta['target']
    energy_csv = os.path.join(run_dir, f'{target}_energy_rmsd.csv')
    rmsf_csv = os.path.join(run_dir, f'{target}_rmsf.csv')
    if not os.path.isfile(energy_csv):
        return None
    df = pd.read_csv(energy_csv)
    # Drop the native/best/init reference rows; keep only trajectory frames.
    traj = df[df['label'].str.startswith('frame_')].reset_index(drop=True)
    rmsf = None
    if os.path.isfile(rmsf_csv):
        rmsf = pd.read_csv(rmsf_csv)

    # --- Time-independent observables ---------------------------------------
    # NNEF runs --mode CB by default, so the trajectory PDB stores Cβ positions
    # (labelled "CA" in the PDB ATOM records -- that's md_eval.py's convention).
    # We use Cβ from the native bead CSV as the contact / RMSF reference, and
    # real Cα from the native CSV for static secondary-structure context.
    md_mode = meta.get('md_mode', 'native')
    pdb_path = os.path.join(run_dir, f'{target}_trajectory_{md_mode}.pdb')
    traj_coords = _read_ca_trajectory(pdb_path)  # actually Cβ despite label

    # Resolve native bead CSV. Try (in order): meta-recorded path, yang_small
    # location, CASP14 top-GDT decoy auto-resolve, 3DRobot native.
    native_bead = None
    candidates = []
    if meta.get('native_bead'):
        candidates.append(meta['native_bead'])
    candidates.append(f'nnef/data/yang_small/{target}/{target}_bead.csv')
    # CASP14 top-GDT proxy: pick the highest-GDT_TS decoy from list.csv
    casp_list = f'nnef/data/decoys/casp14/{target}/list.csv'
    if os.path.isfile(casp_list):
        try:
            ll = pd.read_csv(casp_list).dropna(subset=['GDT_TS'])
            if len(ll):
                top = ll.sort_values('GDT_TS', ascending=False).iloc[0]['NAME']
                candidates.append(
                    f'nnef/data/decoys/casp14/{target}/{top}_bead.csv')
        except Exception:
            pass
    # 3DRobot native bead
    candidates.append(f'nnef/data/decoys/3DRobot_set/{target}/native_bead.csv')
    for c in candidates:
        if c and os.path.isfile(c):
            native_bead = c
            break
    native_cb = _native_coords_from_bead(native_bead, 'CB') if native_bead else None
    native_ca = _native_coords_from_bead(native_bead, 'CA') if native_bead else None

    contacts = None
    lddt = None
    native_ss = None
    if traj_coords is not None and native_cb is not None \
            and traj_coords.shape[1] == native_cb.shape[0]:
        contacts = _native_contacts_fraction(traj_coords, native_cb, cutoff=8.0)
        lddt = _lddt_per_frame(traj_coords, native_cb, cutoff=15.0)
    if native_ca is not None:
        native_ss = _ca_only_ss(native_ca)

    # Energy std (basin-width proxy) over post-equilibration frames.
    # Computed here on the full energy column so consumers don't need the raw
    # trajectory; _post_eq_mean is in main() so we keep the raw array here.
    energy_post_eq = None
    if 'energy' in traj.columns and len(traj) > 0:
        energy_post_eq = traj['energy'].to_numpy(dtype=np.float32)

    return {
        'meta': meta,
        'traj': traj,
        'rmsf': rmsf,
        'refs': df.iloc[:3],
        # Dynamic, time-independent (mean across post-equilibration frames):
        'native_contacts': contacts,            # (F,) Cβ-based fraction-preserved
        'lddt': lddt,                            # (F,) per-frame lDDT in [0, 1]
        'energy_trace': energy_post_eq,          # (F,) raw energy values
        # Static, per-protein context (not a model observable):
        'native_ss': native_ss,
        'native_h_frac': _ss_fractions(native_ss)[0] if native_ss else None,
        'native_e_frac': _ss_fractions(native_ss)[1] if native_ss else None,
        'native_c_frac': _ss_fractions(native_ss)[2] if native_ss else None,
    }


def _walk_runs(root: str, mode_filter: str | None):
    """Find every run_dir (one meta.json each) under root.

    Layout: ``root/<sweep>/<md_mode>/<run_base>/<target_seed>/meta.json``
    where sweep is e.g. ``mode2``/``mode2_lr_ablation`` and md_mode is
    ``native``/``fold``/``decoy``. ``md_mode`` is read from meta.json so
    the walker stays robust to layout tweaks.
    """
    runs = []
    meta_paths = sorted(glob(os.path.join(root, '**', 'meta.json'), recursive=True))
    for meta_path in meta_paths:
        run_dir = os.path.dirname(meta_path)
        rel_parts = os.path.relpath(run_dir, root).split(os.sep)
        if len(rel_parts) < 2:
            continue
        sweep = rel_parts[0]
        if mode_filter and sweep != mode_filter:
            continue
        run_base = rel_parts[-2]
        loaded = _load_run(run_dir)
        if loaded is None:
            continue
        loaded['mode'] = sweep
        loaded['md_mode'] = loaded['meta'].get('md_mode', 'unknown')
        loaded['run_base'] = run_base
        runs.append(loaded)
    return runs


def _run_base_to_model_key(run_base: str) -> str:
    if run_base.startswith('yang_retrain'):
        return 'yang_retrain'
    if run_base.startswith('yang_legacy'):
        return 'yang_legacy'
    if run_base.startswith('v1_esm_rama_v2'):
        return 'v1_rama_esm'
    if run_base.startswith('v1_pure_rama_v2'):
        return 'v1_rama'
    # ESM ablation cells: run_base like v1_esm_abl_w32_l32_attn_pool_13861188.
    # Strip the trailing JOBID to get a stable label across resubmits.
    if run_base.startswith('v1_esm_abl_'):
        import re
        m = re.match(r'v1_esm_abl_(w\d+_l\d+_\w+?)_\d+$', run_base)
        if m:
            return m.group(1)
        return run_base[len('v1_esm_abl_'):]
    # Yang 2022 released paper checkpoint lives at params/exp1, basename = 'exp1'.
    if run_base == 'exp1':
        return 'paper_exp1'
    return run_base


# Models considered "ESM ablation cells" for the paired ESM-vs-rama delta plot.
def _is_esm_ablation_key(model_key: str) -> bool:
    import re
    return bool(re.match(r'^w\d+_l\d+_(per_residue|center_only|attn_pool)$', model_key))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='eval/md_eval',
                        help='Root of the md_eval output tree')
    parser.add_argument('--mode', default=None,
                        help='Restrict to one mode dir (e.g. mode2)')
    parser.add_argument('--out_dir', default=None,
                        help='Where to put summary.csv and plots/ '
                             '(default: <root>)')
    args = parser.parse_args()

    out_dir = args.out_dir or args.root
    os.makedirs(out_dir, exist_ok=True)

    runs = _walk_runs(args.root, args.mode)
    if not runs:
        raise SystemExit(f'No md_eval runs found under {args.root!r}')

    # --- Summary CSV ---------------------------------------------------------
    # Equilibration: drop the first 10% of frames before averaging the
    # time-INdependent observables. NNEF Langevin step has no physical-time
    # interpretation, so we always report frame-averaged observables; the
    # equilibration cut just removes the obvious "still-near-init" portion.
    EQUIL_FRAC = 0.10

    def _post_eq_mean(arr):
        if arr is None or len(arr) == 0:
            return None
        k = max(1, int(len(arr) * EQUIL_FRAC))
        post = arr[k:]
        return float(np.nanmean(post)) if len(post) else None

    rows = []
    for r in runs:
        m = r['meta']
        row = {
            'mode': r['mode'],
            'md_mode': r['md_mode'],
            'model_key': _run_base_to_model_key(r['run_base']),
            'run_base': r['run_base'],
            'target': m['target'],
            'seed': m['seed'],
            'n_residues': m['n_residues'],
            'L': m['L'],
            'lr': m['lr'],
            't_noise': m['t_noise'],
            'energy_native': m['energy_native'],
            'energy_init': m['energy_init'],
            'energy_best': m['energy_best'],
            'final_rmsd_to_native': m['final_rmsd_to_native'],
            'final_rmsd_to_start': m['final_rmsd_to_start'],
            'rg_native': m['rg_native'],
            'elapsed_sec': m['elapsed_sec'],
            # Mean per-residue RMSF (time-independent, averaged over residues)
            'rmsf_mean': (float(r['rmsf']['rmsf'].mean()) if r['rmsf'] is not None
                          else None),
            # Time-INdependent dynamic observable (post-equilibration mean):
            # what fraction of native Cβ-Cβ contacts (<8 Å) remain in MD.
            'native_contacts_mean': _post_eq_mean(r['native_contacts']),
            # lDDT (length-normalized, outlier-robust structural similarity --
            # gold standard for MD/structure quality, used by AlphaFold pLDDT).
            'lddt_mean':            _post_eq_mean(r['lddt']),
            # Energy std post-equilibration: proxy for basin width.
            # Lower = tighter basin = "sharper" energy landscape.
            'energy_std_post_eq':   (float(np.std(r['energy_trace'][max(1, int(len(r['energy_trace']) * 0.10)):]))
                                     if r['energy_trace'] is not None and len(r['energy_trace']) > 10
                                     else None),
            # Static native SS reference (computed once from real Cα, for
            # context only -- we cannot reliably assign SS to MD Cβ frames).
            'native_h_frac':        r['native_h_frac'],
            'native_e_frac':        r['native_e_frac'],
            'native_c_frac':        r['native_c_frac'],
        }
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary_path = os.path.join(out_dir, 'summary.csv')
    summary.to_csv(summary_path, index=False)
    print(f'[plot_md_eval] wrote {summary_path} ({len(summary)} runs)')

    # --- Per-(mode, md_mode, target) traces ---------------------------------
    plots_root = os.path.join(out_dir, 'plots')
    os.makedirs(plots_root, exist_ok=True)

    groups: dict[tuple[str, str, str], list[dict]] = {}
    for r in runs:
        key = (r['mode'], r['md_mode'], r['meta']['target'])
        groups.setdefault(key, []).append(r)

    for (mode, md_mode, target), grp in groups.items():
        plot_dir = os.path.join(plots_root, mode, md_mode)
        os.makedirs(plot_dir, exist_ok=True)
        # Include lr in the label only when a group spans multiple lrs (ablation).
        lrs_in_group = {float(run['meta']['lr']) for run in grp}
        show_lr = len(lrs_in_group) > 1

        fig_e, ax_e = plt.subplots(figsize=(7, 4))
        fig_r, ax_r = plt.subplots(figsize=(7, 4))
        fig_g, ax_g = plt.subplots(figsize=(7, 4))
        model_colors: dict[str, str] = {}
        palette = plt.cm.tab10.colors
        for run in grp:
            model_key = _run_base_to_model_key(run['run_base'])
            if model_key not in model_colors:
                model_colors[model_key] = palette[len(model_colors) % len(palette)]
            color = model_colors[model_key]
            traj = run['traj']
            seed = run['meta']['seed']
            if show_lr:
                label = f"{model_key} lr{run['meta']['lr']:.0e} s{seed}"
            else:
                label = f"{model_key} s{seed}"
            ax_e.plot(traj['step'], traj['energy'], color=color, alpha=0.7, label=label)
            ax_r.plot(traj['step'], traj['rmsd_to_native'], color=color, alpha=0.7, label=label)
            ax_g.plot(traj['step'], traj['rg'], color=color, alpha=0.7, label=label)

        rg_native = grp[0]['meta'].get('rg_native')
        if rg_native is not None:
            ax_g.axhline(rg_native, color='k', linestyle='--', alpha=0.5,
                         label=f'native Rg={rg_native:.2f}')

        for ax, ylabel, fig, fname in [
            (ax_e, 'energy', fig_e, f'{target}_energy.png'),
            (ax_r, 'RMSD to native (Å)', fig_r, f'{target}_rmsd.png'),
            (ax_g, 'Rg (Å)', fig_g, f'{target}_rg.png'),
        ]:
            ax.set_xlabel('MD step')
            ax.set_ylabel(ylabel)
            ax.set_title(f'{mode} / {md_mode} / {target}')
            ax.legend(fontsize=7, ncol=2, loc='best')
            fig.tight_layout()
            fig.savefig(os.path.join(plot_dir, fname), dpi=150)
            plt.close(fig)

        # RMSF: mean across seeds per model
        fig_f, ax_f = plt.subplots(figsize=(7, 4))
        by_model: dict[str, list[np.ndarray]] = {}
        for run in grp:
            if run['rmsf'] is None:
                continue
            model_key = _run_base_to_model_key(run['run_base'])
            by_model.setdefault(model_key, []).append(run['rmsf']['rmsf'].values)
        for model_key, stacks in by_model.items():
            min_len = min(len(s) for s in stacks)
            arr = np.stack([s[:min_len] for s in stacks])
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            ax_f.plot(np.arange(min_len), mean,
                      color=model_colors.get(model_key), label=model_key)
            ax_f.fill_between(np.arange(min_len), mean - std, mean + std,
                              color=model_colors.get(model_key), alpha=0.2)
        ax_f.set_xlabel('residue')
        ax_f.set_ylabel('RMSF (Å)')
        ax_f.set_title(f'{mode} / {md_mode} / {target} RMSF (mean ± std across seeds)')
        ax_f.legend(fontsize=8)
        fig_f.tight_layout()
        fig_f.savefig(os.path.join(plot_dir, f'{target}_rmsf.png'), dpi=150)
        plt.close(fig_f)

    # --- Cross-model heatmap per (mode, md_mode) ----------------------------
    for (mode, md_mode), sub in summary.groupby(['mode', 'md_mode']):
        pivot = sub.groupby(['model_key', 'target'])['final_rmsd_to_native'].mean().unstack()
        if pivot.empty:
            continue
        fig, ax = plt.subplots(figsize=(1.2 * pivot.shape[1] + 2, 0.6 * pivot.shape[0] + 2))
        im = ax.imshow(pivot.values, aspect='auto', cmap='viridis')
        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(pivot.shape[0]))
        ax.set_yticklabels(pivot.index)
        ax.set_title(f'{mode}/{md_mode}: mean final RMSD to native (Å)')
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            color='white' if val > pivot.values.mean() else 'black',
                            fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        out_path = os.path.join(plots_root, mode, md_mode, 'summary_heatmap.png')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f'[plot_md_eval] heatmap -> plots/{mode}/{md_mode}/summary_heatmap.png')

    # --- Per-(mode, md_mode) model box plot ---------------------------------
    # One box per model_key showing the spread of final RMSD across all
    # (protein, seed) tasks. Useful as a one-glance "which model wins" figure.
    for (mode, md_mode), sub in summary.groupby(['mode', 'md_mode']):
        models = sorted(sub['model_key'].unique())
        if len(models) < 2 or len(sub) < 6:
            continue
        data = [sub.loc[sub.model_key == m, 'final_rmsd_to_native'].values
                for m in models]
        fig, ax = plt.subplots(figsize=(max(4.0, 1.2 * len(models) + 1.5), 4))
        bp = ax.boxplot(data, labels=models, patch_artist=True, showmeans=True,
                        widths=0.55,
                        meanprops={'marker': 'D', 'markerfacecolor': 'k',
                                   'markeredgecolor': 'k', 'markersize': 5})
        cmap = plt.cm.tab10.colors
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(cmap[i % len(cmap)])
            patch.set_alpha(0.55)
        for i, m in enumerate(models):
            y = sub.loc[sub.model_key == m, 'final_rmsd_to_native'].values
            x = np.full_like(y, i + 1, dtype=float) + \
                (np.random.RandomState(i).rand(len(y)) - 0.5) * 0.18
            ax.scatter(x, y, color=cmap[i % len(cmap)], alpha=0.7, s=16,
                       edgecolor='k', linewidth=0.3)
        ax.set_ylabel('final RMSD to native (Å)')
        ax.set_title(f'{mode}/{md_mode}: per-task final RMSD by model '
                     f'(n={len(sub) // len(models)} per box)')
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=20, ha='right')
        fig.tight_layout()
        out_path = os.path.join(plots_root, mode, md_mode, 'box_final_rmsd.png')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f'[plot_md_eval] box     -> plots/{mode}/{md_mode}/box_final_rmsd.png')

    # --- ESM-vs-rama paired delta bar per (mode, md_mode) -------------------
    # Only drawn when both 'v1_rama' (no ESM) and an ESM-ablation cell are
    # present in the same sweep. Negative bar = ESM helps that protein.
    for (mode, md_mode), sub in summary.groupby(['mode', 'md_mode']):
        if 'v1_rama' not in set(sub.model_key):
            continue
        esm_keys = [m for m in sub.model_key.unique() if _is_esm_ablation_key(m)]
        if not esm_keys:
            continue
        rama_per_target = (sub[sub.model_key == 'v1_rama']
                           .groupby('target')['final_rmsd_to_native'].mean())
        for esm_key in esm_keys:
            esm_per_target = (sub[sub.model_key == esm_key]
                              .groupby('target')['final_rmsd_to_native'].mean())
            common = sorted(set(rama_per_target.index) & set(esm_per_target.index))
            if not common:
                continue
            delta = (esm_per_target[common] - rama_per_target[common]).sort_values()
            colors = ['tab:green' if v < 0 else 'tab:red' for v in delta.values]
            fig, ax = plt.subplots(figsize=(8, 0.32 * len(delta) + 1.5))
            ax.barh(range(len(delta)), delta.values, color=colors, alpha=0.75)
            ax.set_yticks(range(len(delta)))
            ax.set_yticklabels(delta.index)
            ax.axvline(0, color='k', lw=0.8)
            ax.set_xlabel(f'ΔRMSD = RMSD({esm_key}) − RMSD(v1_rama)  (Å)')
            n_helps = int((delta < 0).sum()); n_hurts = int((delta > 0).sum())
            ax.set_title(f'{mode}/{md_mode}: ESM contribution per protein   '
                         f'(green={n_helps} helps, red={n_hurts} hurts, '
                         f'mean Δ={delta.mean():+.2f} Å)')
            ax.grid(axis='x', alpha=0.3)
            fig.tight_layout()
            out_path = os.path.join(plots_root, mode, md_mode,
                                    f'bar_esm_delta_{esm_key}.png')
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f'[plot_md_eval] esm_delta -> plots/{mode}/{md_mode}/bar_esm_delta_{esm_key}.png')

    # --- Time-INdependent observables: grouped bars per protein x model -----
    # These are the metrics the advisor wants emphasized (vs RMSD-vs-step,
    # which is misleading since NNEF Langevin has no physical time scale).
    def _grouped_bar(sub, value_col, ylabel, title, out_path,
                     native_col=None, ylim=None):
        """One grouped bar chart: x = target, color = model_key.
        sub: subset of summary for one (mode, md_mode).
        value_col: column name to plot (mean across seeds).
        native_col: optional column of native reference (drawn as horizontal
            black ticks per protein, ignoring model).
        """
        models = sorted(sub.model_key.unique())
        targets = sorted(sub.target.unique())
        if not targets or len(models) < 2:
            return
        agg = (sub.groupby(['target', 'model_key'])[value_col]
                  .agg(['mean', 'std']).reset_index())
        if native_col and native_col in sub.columns:
            native_per_target = (sub.groupby('target')[native_col]
                                    .first().to_dict())
        else:
            native_per_target = {}

        n_t = len(targets); n_m = len(models)
        bar_w = 0.8 / n_m
        fig, ax = plt.subplots(figsize=(max(7, 0.55 * n_t + 2.5), 4))
        palette = plt.cm.tab10.colors
        for mi, m in enumerate(models):
            xs = []; ys = []; es = []
            for ti, t in enumerate(targets):
                row = agg[(agg.target == t) & (agg.model_key == m)]
                if row.empty:
                    continue
                xs.append(ti + (mi - (n_m - 1) / 2) * bar_w)
                ys.append(float(row['mean'].iloc[0]))
                es.append(float(row['std'].iloc[0]) if pd.notna(row['std'].iloc[0]) else 0.0)
            ax.bar(xs, ys, width=bar_w * 0.95, yerr=es, capsize=2,
                   color=palette[mi % len(palette)], alpha=0.75,
                   edgecolor='black', linewidth=0.4, label=m)
        # Draw native reference (per-protein horizontal tick) if provided
        for ti, t in enumerate(targets):
            v = native_per_target.get(t)
            if v is not None and pd.notna(v):
                ax.plot([ti - 0.4, ti + 0.4], [v, v], color='black',
                        lw=1.5, alpha=0.8,
                        label='native' if ti == 0 else None)
        ax.set_xticks(range(n_t))
        ax.set_xticklabels(targets, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8, loc='best', ncol=min(n_m + 1, 4))
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f'[plot_md_eval] grouped_bar -> {os.path.relpath(out_path, out_dir)}')

    for (mode, md_mode), sub in summary.groupby(['mode', 'md_mode']):
        if sub.empty:
            continue
        bar_dir = os.path.join(plots_root, mode, md_mode)

        # Native contacts preserved (Cβ-Cβ, advisor-recommended observable)
        if sub['native_contacts_mean'].notna().any():
            _grouped_bar(
                sub, 'native_contacts_mean',
                ylabel='Fraction of native Cβ-Cβ contacts preserved',
                title=f'{mode}/{md_mode}: native contacts preserved (post-eq mean, cutoff 8 Å)',
                out_path=os.path.join(bar_dir, 'bar_native_contacts.png'),
                ylim=(0, 1.05),
            )

        # Mean per-residue RMSF (advisor-recommended observable, time-averaged)
        if sub['rmsf_mean'].notna().any():
            _grouped_bar(
                sub, 'rmsf_mean',
                ylabel='Mean per-residue RMSF (Å)',
                title=f'{mode}/{md_mode}: trajectory RMSF (mean across residues)',
                out_path=os.path.join(bar_dir, 'bar_rmsf_mean.png'),
            )

        # lDDT (CASP/AlphaFold standard, length-normalized, outlier-robust)
        if sub['lddt_mean'].notna().any():
            _grouped_bar(
                sub, 'lddt_mean',
                ylabel='Mean lDDT (post-eq, vs native)',
                title=f'{mode}/{md_mode}: lDDT (higher = closer to native, length-normalized)',
                out_path=os.path.join(bar_dir, 'bar_lddt.png'),
                ylim=(0, 1.05),
            )

        # Energy std (basin-width proxy; lower = sharper energy near native)
        if sub['energy_std_post_eq'].notna().any():
            _grouped_bar(
                sub, 'energy_std_post_eq',
                ylabel='Energy std (post-eq) — proxy for basin width',
                title=f'{mode}/{md_mode}: trajectory energy fluctuation (lower = sharper basin)',
                out_path=os.path.join(bar_dir, 'bar_energy_std.png'),
            )

    # --- Cross-model RMSF profile correlation -------------------------------
    # For each protein with ≥2 models, compute pairwise Pearson correlation
    # between each model's mean RMSF profile (averaged across seeds). High
    # correlation = "models agree on which residues are flexible" (a sign
    # that learned flexibility patterns are model-agnostic / capture real
    # physics rather than model-specific artifacts).
    for (mode, md_mode), grp_meta in summary.groupby(['mode', 'md_mode']):
        runs_in_grp = [r for r in runs if r['mode'] == mode and r['md_mode'] == md_mode]
        if not runs_in_grp:
            continue
        # Build per-(target, model_key) mean RMSF profile.
        prof: dict[tuple[str, str], 'np.ndarray'] = {}
        for r in runs_in_grp:
            if r['rmsf'] is None:
                continue
            target = r['meta']['target']
            model_key = _run_base_to_model_key(r['run_base'])
            arr = r['rmsf']['rmsf'].to_numpy(dtype=np.float32)
            prof.setdefault((target, model_key), []).append(arr)
        if not prof:
            continue
        mean_prof = {k: np.stack([a[:min(len(x) for x in v)] for a in v]).mean(0)
                     for k, v in prof.items()}
        # For each target, pairwise model RMSF correlation
        targets = sorted({t for (t, m) in mean_prof.keys()})
        models = sorted({m for (t, m) in mean_prof.keys()})
        if len(models) < 2 or not targets:
            continue
        corr_rows = []
        for t in targets:
            row = {'target': t}
            for mi, m1 in enumerate(models):
                for m2 in models[mi + 1:]:
                    a = mean_prof.get((t, m1)); b = mean_prof.get((t, m2))
                    if a is None or b is None:
                        row[f'{m1}__{m2}'] = np.nan
                        continue
                    L = min(len(a), len(b))
                    a, b = a[:L], b[:L]
                    if a.std() == 0 or b.std() == 0:
                        row[f'{m1}__{m2}'] = np.nan
                    else:
                        row[f'{m1}__{m2}'] = float(np.corrcoef(a, b)[0, 1])
            corr_rows.append(row)
        corr_df = pd.DataFrame(corr_rows).set_index('target')
        # Write CSV + heatmap
        bar_dir = os.path.join(plots_root, mode, md_mode)
        os.makedirs(bar_dir, exist_ok=True)
        corr_csv = os.path.join(bar_dir, 'rmsf_cross_model_correlation.csv')
        corr_df.to_csv(corr_csv)

        fig, ax = plt.subplots(figsize=(max(5.0, 0.8 * len(corr_df.columns) + 2),
                                         0.32 * len(corr_df) + 1.5))
        im = ax.imshow(corr_df.values, aspect='auto', cmap='RdBu_r',
                       vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr_df.columns)))
        ax.set_xticklabels(corr_df.columns, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(corr_df)))
        ax.set_yticklabels(corr_df.index, fontsize=9)
        for i in range(corr_df.shape[0]):
            for j in range(corr_df.shape[1]):
                v = corr_df.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                            color='white' if abs(v) > 0.6 else 'black',
                            fontsize=7)
        ax.set_title(f'{mode}/{md_mode}: pairwise RMSF profile correlation between models\n'
                     f'(high = models agree on which residues are flexible)')
        fig.colorbar(im, ax=ax, fraction=0.04)
        fig.tight_layout()
        out_path = os.path.join(bar_dir, 'rmsf_cross_model_correlation.png')
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f'[plot_md_eval] rmsf_corr -> {os.path.relpath(out_path, out_dir)}')
        print(f'[plot_md_eval] rmsf_corr -> {os.path.relpath(corr_csv, out_dir)}')

    print(f'[plot_md_eval] done -> {out_dir}')


if __name__ == '__main__':
    main()
