#!/usr/bin/env python3
"""Scatter plots: GDT_TS vs loss for CASP14 multi-target eval (Yang, RIBBON, RIBBON+ESM).

**Why T1053 looked "wrong" before:** RIBBON / RIBBON+ESM often return **non-finite
energy (NaN)** on many T1053 decoys (long, ~580-residue models) while **Yang
(legacy local frame)** scores almost all of them. That is evaluation behaviour,
not a plotting bug. Plotting each model's CSV with ``dropna()`` then showed
~495 Yang points vs ~24 RIBBON points.

**Default (paired mode):** inner-join decoys on ``NAME`` where **all three**
models have finite ``loss``, so the three panels use the **same** decoy set and
comparable *n*. Use ``--unpaired`` to plot every finite (GDT_TS, loss) pair
per model separately (mixing different *n* across panels).

Reads ``nnef/data/decoys/casp14/decoy_loss_<exp_tag>/<pdb>_decoy_loss.csv``.

Outputs::

    eval/figures/casp14_multi/casp14_<TARGET>_three_models_scatter.png
    eval/figures/casp14_multi/casp14_all_targets_4x3_scatter.png
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPO = EVAL_DIR.parent
MPL_DIR = EVAL_DIR / ".mplconfig"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, str(REPO / "nnef"))
from paths import data_path  # noqa: E402

OUT = EVAL_DIR / "figures" / "casp14_multi"

MODELS: list[tuple[str, str]] = [
    ("NNEF (Yang retrain)", "yang_retrain_6594199"),
    ("RIBBON (v1 + Rama)", "v1_pure_rama_v2_6228201"),
    ("RIBBON+ESM (v1 + Rama + ESM)", "v1_esm_rama_v2_6642146"),
]

TARGETS = ["T1024", "T1025", "T1026", "T1053"]
METRIC = "GDT_TS"

COLORS = ["#0173B2", "#029E73", "#DE8F05"]


def _read_decoy_loss(tag: str, pdb: str) -> pd.DataFrame:
    p = Path(data_path("decoys", "casp14", f"decoy_loss_{tag}", f"{pdb}_decoy_loss.csv"))
    if not p.is_file():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    if METRIC not in df.columns or "loss" not in df.columns:
        raise ValueError(f"{p}: need {METRIC} and loss")
    return df


def load_paired(pdb: str) -> pd.DataFrame:
    """Decoys where all three models have finite ``METRIC`` and ``loss`` (inner join on NAME)."""
    tbl: pd.DataFrame | None = None
    for _title, tag in MODELS:
        df = _read_decoy_loss(tag, pdb).set_index("NAME")
        if tbl is None:
            tbl = pd.DataFrame({METRIC: df[METRIC], f"loss__{tag}": df["loss"]})
        else:
            tbl = tbl.join(
                df["loss"].rename(f"loss__{tag}"),
                how="inner",
            )
    assert tbl is not None
    tbl = tbl.dropna(how="any")
    tbl = tbl[np.isfinite(tbl[METRIC].to_numpy(dtype=float))]
    for _t, tag in MODELS:
        c = f"loss__{tag}"
        tbl = tbl[np.isfinite(tbl[c].to_numpy(dtype=float))]
    return tbl.reset_index()


def load_unpaired_column(tag: str, pdb: str) -> pd.DataFrame:
    """Single model: finite METRIC and loss (may differ from other models)."""
    df = _read_decoy_loss(tag, pdb)
    n_in = len(df)
    out = df[[METRIC, "loss"]].dropna()
    out = out[np.isfinite(out[METRIC]) & np.isfinite(out["loss"])]
    if n_in > 50 and len(out) < n_in * 0.5:
        print(
            f"[plot_casp14_multi] note {pdb} {tag}: only {len(out)}/{n_in} finite "
            f"pairs — often NaN energies on long CASP decoys for non-legacy checkpoints.",
            file=sys.stderr,
        )
    return out


def scatter_ax(ax, x: np.ndarray, y: np.ndarray, color: str, title: str) -> None:
    ax.scatter(x, y, s=10, alpha=0.42, edgecolors="none", c=color)
    if len(x) >= 2 and np.std(x) > 0 and np.std(y) > 0:
        r, _ = pearsonr(x, y)
        xf = np.linspace(x.min(), x.max(), 80)
        coef = np.polyfit(x, y, 1)
        ax.plot(xf, np.poly1d(coef)(xf), color="#CC3311", lw=1.05, alpha=0.92)
        ax.set_title(f"{title}\nr = {r:+.4f}  (n={len(x)})", fontsize=9)
    else:
        ax.set_title(f"{title}\n(n={len(x)})", fontsize=9)
    ax.set_xlabel(METRIC)
    ax.set_ylabel("Energy (loss)")


def plot_row_paired(fig, axes, pdb: str) -> None:
    tbl = load_paired(pdb)
    n = len(tbl)
    x_common = tbl[METRIC].to_numpy(dtype=float)
    for ax, (title, tag), color in zip(axes, MODELS, COLORS):
        y = tbl[f"loss__{tag}"].to_numpy(dtype=float)
        scatter_ax(ax, x_common, y, color, title)
    fig.suptitle(
        f"{pdb} — CASP14 (paired: n={n} decoys)",
        fontsize=11,
        fontweight="600",
    )


def plot_row_unpaired(fig, axes, pdb: str) -> None:
    for ax, (title, tag), color in zip(axes, MODELS, COLORS):
        df = load_unpaired_column(tag, pdb)
        x = df[METRIC].to_numpy(dtype=float)
        y = df["loss"].to_numpy(dtype=float)
        scatter_ax(ax, x, y, color, title)
    fig.suptitle(f"{pdb} — CASP14 (unpaired: each model's finite points)", fontsize=11, fontweight="600")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--unpaired",
        action="store_true",
        help="Plot each model independently (different n per panel; T1053 Yang≈495 vs RIBBON≈24).",
    )
    args = p.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    paired = not args.unpaired
    suf = "" if paired else "_unpaired"

    plot_row = plot_row_paired if paired else plot_row_unpaired

    for pdb in TARGETS:
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), dpi=150)
        plot_row(fig, axes, pdb)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        out = OUT / f"casp14_{pdb}_three_models_scatter{suf}.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"wrote {out}")

    fig, axes = plt.subplots(4, 3, figsize=(11.5, 14), dpi=150, sharex=False, sharey=False)
    for ri, pdb in enumerate(TARGETS):
        if paired:
            tbl = load_paired(pdb)
            n = len(tbl)
            x_common = tbl[METRIC].to_numpy(dtype=float)
            for j, ((title, tag), color) in enumerate(zip(MODELS, COLORS)):
                y = tbl[f"loss__{tag}"].to_numpy(dtype=float)
                scatter_ax(axes[ri, j], x_common, y, color, title)
        else:
            for j, ((title, tag), color) in enumerate(zip(MODELS, COLORS)):
                df = load_unpaired_column(tag, pdb)
                x = df[METRIC].to_numpy(dtype=float)
                y = df["loss"].to_numpy(dtype=float)
                scatter_ax(axes[ri, j], x, y, color, title)
        axes[ri, 0].text(
            -0.28,
            0.5,
            pdb,
            transform=axes[ri, 0].transAxes,
            fontsize=10,
            fontweight="600",
            va="center",
            ha="right",
            rotation=90,
        )
    for ax in axes.flat:
        ax.label_outer()
    mode = "paired (same decoys in all panels)" if paired else "unpaired"
    fig.suptitle(
        f"CASP14 — loss vs GDT_TS (three models × four targets) [{mode}]",
        fontsize=12,
        y=0.995,
    )
    fig.tight_layout()
    out_all = OUT / f"casp14_all_targets_4x3_scatter{suf}.png"
    fig.savefig(out_all, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_all}")


if __name__ == "__main__":
    main()
