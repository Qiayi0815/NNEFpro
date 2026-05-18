#!/usr/bin/env python3
"""Compare NNEF, RIBBON, RIBBON+ESM: CASP14 T1053 + 3DRobot per-target analysis."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EVAL = Path(__file__).resolve().parent
OUT = EVAL / "figures" / "four_model_compare"
MPL_DIR = EVAL / ".mplconfig"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

MERGED_CASP = EVAL / "casp14_only_summary_merged.csv"

# Display order: NNEF, RIBBON, RIBBON+ESM (V2 dihedral excluded)
MODELS: list[tuple[str, str, Path]] = [
    ("NNEF", "nnef", EVAL / "yang_retrain_6594199_3drobot_local" / "summary.csv"),
    ("RIBBON", "ribbon", EVAL / "v1_pure_rama_v2_6228201_casp14_3drobot" / "summary.csv"),
    ("RIBBON+ESM", "ribbon_esm", EVAL / "v3_full_rama_v2_6229240_casp14_3drobot" / "summary.csv"),
]

COLORS = ["#0173B2", "#DE8F05", "#029E73"]

CASP_ORDER = ["NNEF_retrained", "v1_pure_Rama_v2", "v3_full_Rama_v2"]
CASP_LABEL = {
    "NNEF_retrained": "NNEF",
    "v1_pure_Rama_v2": "RIBBON",
    "v3_full_Rama_v2": "RIBBON+ESM",
}


def load_3drobot(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[df["decoy_set"] == "3DRobot_set"].copy()


def _per_target_winner_labels(mat: np.ndarray, names: list[str]) -> list[str]:
    """Each row: which model has highest r (loss vs RMSD). Ties → 'tie'."""
    out: list[str] = []
    for row in mat:
        mx = float(np.max(row))
        idx = np.nonzero(np.isclose(row, mx, rtol=0.0, atol=1e-9))[0]
        if len(idx) > 1:
            out.append("tie")
        else:
            out.append(names[int(idx[0])])
    return out


def _pairwise_wins(a: np.ndarray, b: np.ndarray) -> tuple[int, int, int]:
    d = a - b
    a_wins = int(np.sum(d > 1e-9))
    b_wins = int(np.sum(d < -1e-9))
    ties = int(len(d) - a_wins - b_wins)
    return a_wins, b_wins, ties


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    n_models = len(MODELS)

    # --- CASP14 T1053 (from merged casp14-only evals; omit V2 dihedral) ---
    casp = pd.read_csv(MERGED_CASP)
    casp = casp[casp["model_label"] != "v2_dihedral_Rama_v2"].copy()
    casp["_ord"] = casp["model_label"].map({m: i for i, m in enumerate(CASP_ORDER)})
    casp = casp.sort_values("_ord")
    labels_casp = [CASP_LABEL[x] for x in casp["model_label"]]
    pear_c = casp["pearson_r"].to_numpy()
    spear_c = casp["spearman_r"].to_numpy()

    x = np.arange(len(labels_casp))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=150)
    ax.bar(x - w / 2, pear_c, width=w, label="Pearson r", color="#332288", alpha=0.88)
    ax.bar(x + w / 2, spear_c, width=w, label="Spearman ρ", color="#88CCEE", alpha=0.88)
    ax.axhline(0, color="0.35", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_casp, rotation=22, ha="right")
    ax.set_ylabel("Correlation with GDT_TS (T1053 decoys)")
    ax.set_title("CASP14 T1053 — loss vs. GDT_TS")
    ax.legend(frameon=False, loc="lower left")
    ax.set_ylim(-1.0, 0.1)
    fig.tight_layout()
    fig.savefig(OUT / "01_casp14_T1053_pearson_spearman.png")
    plt.close(fig)

    # --- 3DRobot: merge on pdb ---
    parts = []
    for disp, key, path in MODELS:
        d = load_3drobot(path)
        d = d.rename(
            columns={
                "pearson_r": f"pearson_{key}",
                "spearman_r": f"spearman_{key}",
            }
        )[["pdb", f"pearson_{key}", f"spearman_{key}"]]
        parts.append(d)
    m = parts[0]
    for p in parts[1:]:
        m = m.merge(p, on="pdb", how="outer")

    pear_cols = [f"pearson_{k}" for _, k, _ in MODELS]
    spear_cols = [f"spearman_{k}" for _, k, _ in MODELS]
    # Only compare targets where every model has finite Pearson & Spearman (ignore missing / non-numeric).
    for c in pear_cols + spear_cols:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.replace([np.inf, -np.inf], np.nan)
    paired = m.dropna(subset=pear_cols + spear_cols, how="any").reset_index(drop=True)
    n_paired = len(paired)
    assert bool(paired[pear_cols + spear_cols].notna().all().all()), "paired frame must have no NaNs"

    stats_rows = []
    for disp, key, _ in MODELS:
        pc = paired[f"pearson_{key}"]
        sc = paired[f"spearman_{key}"]
        stats_rows.append(
            {
                "model": disp,
                "n_3drobot_paired": n_paired,
                "pearson_mean": pc.mean(),
                "pearson_median": pc.median(),
                "pearson_std": pc.std(ddof=1),
                "spearman_mean": sc.mean(),
                "spearman_median": sc.median(),
                "spearman_std": sc.std(ddof=1),
            }
        )
    pd.DataFrame(stats_rows).to_csv(OUT / "3drobot_paired_summary_stats.csv", index=False)

    disp_names = [d for d, _, _ in MODELS]
    pear_data = [paired[c].to_numpy() for c in pear_cols]
    spear_data = [paired[c].to_numpy() for c in spear_cols]

    # --- Benchmark scope (proteins / decoys) & per-target “who wins” ---
    scope_keyval: list[tuple[str, float | int | str]] = []
    raw_dr_rows = []
    for disp, slug, path in MODELS:
        raw = pd.read_csv(path)
        dr = raw[raw["decoy_set"] == "3DRobot_set"]
        raw_dr_rows.append(dr)
        scope_keyval.append((f"n_3drobot_rows_{slug}", len(dr)))
        scope_keyval.append(
            (f"n_3drobot_finite_pearson_{slug}", int(dr["pearson_r"].notna().sum()))
        )
    n_decoys = int(raw_dr_rows[0]["n_decoys"].dropna().iloc[0])
    scope_keyval.append(("n_decoys_per_target", n_decoys))
    n_union = int(m["pdb"].nunique())
    scope_keyval.append(("n_distinct_proteins_union_of_summaries", n_union))
    scope_keyval.append(("n_proteins_paired_all_models_finite", n_paired))
    scope_keyval.append(
        (
            "note",
            "All 3DRobot comparisons use only rows with finite Pearson & Spearman for every model "
            "(missing/blank/non-numeric coerced to NaN; ±inf dropped). Winner = higher r (loss vs RMSD).",
        )
    )
    pd.DataFrame(scope_keyval, columns=["key", "value"]).to_csv(
        OUT / "3drobot_benchmark_scope.csv", index=False
    )

    Pw = np.column_stack(pear_data)
    Sw = np.column_stack(spear_data)
    pear_winners = _per_target_winner_labels(Pw, disp_names)
    spear_winners = _per_target_winner_labels(Sw, disp_names)

    def _winner_table(metric: str, labels: list[str]) -> pd.DataFrame:
        vc = pd.Series(labels).value_counts()
        rows = []
        order = disp_names + ["tie"]
        for w in order:
            c = int(vc.get(w, 0))
            rows.append(
                {
                    "metric": metric,
                    "winner": w,
                    "n_proteins": c,
                    "pct_of_paired": round(100.0 * c / n_paired, 4),
                }
            )
        return pd.DataFrame(rows)

    win_tbl = pd.concat(
        [_winner_table("pearson", pear_winners), _winner_table("spearman", spear_winners)],
        ignore_index=True,
    )
    win_tbl.to_csv(OUT / "3drobot_per_target_winner_counts.csv", index=False)

    pair_ix = [(0, 1), (0, 2), (1, 2)]
    pr_rows = []
    for i, j in pair_ix:
        a, b, t = _pairwise_wins(Pw[:, i], Pw[:, j])
        pr_rows.append(
            {
                "metric": "pearson",
                "model_a": disp_names[i],
                "model_b": disp_names[j],
                "a_wins": a,
                "b_wins": b,
                "ties": t,
                "a_win_pct": round(100.0 * a / n_paired, 4),
                "b_win_pct": round(100.0 * b / n_paired, 4),
                "tie_pct": round(100.0 * t / n_paired, 4),
            }
        )
        a, b, t = _pairwise_wins(Sw[:, i], Sw[:, j])
        pr_rows.append(
            {
                "metric": "spearman",
                "model_a": disp_names[i],
                "model_b": disp_names[j],
                "a_wins": a,
                "b_wins": b,
                "ties": t,
                "a_win_pct": round(100.0 * a / n_paired, 4),
                "b_win_pct": round(100.0 * b / n_paired, 4),
                "tie_pct": round(100.0 * t / n_paired, 4),
            }
        )
    pd.DataFrame(pr_rows).to_csv(OUT / "3drobot_pairwise_win_rates.csv", index=False)

    # Stacked bar: fraction of proteins where each model has the best per-target r
    tie_color = "#BAB0AC"
    stack_order = disp_names + ["tie"]
    stack_colors = COLORS + [tie_color]

    def _stack_fracs(labels: list[str]) -> list[float]:
        vc = pd.Series(labels).value_counts()
        return [float(vc.get(w, 0)) / n_paired for w in stack_order]

    fr_pear = _stack_fracs(pear_winners)
    fr_spear = _stack_fracs(spear_winners)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.2), dpi=150, sharey=True)
    for ax, fr, title in zip(axes, (fr_pear, fr_spear), ("Pearson r", "Spearman ρ")):
        left = 0.0
        for frac, c, lab in zip(fr, stack_colors, stack_order):
            ax.barh([0], [frac], left=left, color=c, height=0.5, label=f"{lab} ({frac*100:.1f}%)")
            left += frac
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("Fraction of paired proteins")
        ax.set_title(f"{title} — best per-target correlation (n={n_paired})")
        ax.legend(bbox_to_anchor=(0.5, -0.28), loc="upper center", ncol=2, frameon=False, fontsize=8)
    fig.suptitle("3DRobot — which model wins on each protein? (higher r = better)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "07_3drobot_per_target_win_fractions.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), dpi=150, sharey=True)
    bp0 = axes[0].boxplot(
        pear_data,
        tick_labels=disp_names,
        patch_artist=True,
        medianprops=dict(color="0.15", linewidth=1.2),
    )
    bp1 = axes[1].boxplot(
        spear_data,
        tick_labels=disp_names,
        patch_artist=True,
        medianprops=dict(color="0.15", linewidth=1.2),
    )
    for bp, pal in ((bp0, COLORS), (bp1, COLORS)):
        for patch, c in zip(bp["boxes"], pal):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
    axes[0].set_ylabel("Per-target correlation with RMSD")
    axes[0].set_title(f"Pearson r (n={n_paired} paired targets)")
    axes[1].set_title(f"Spearman ρ (n={n_paired} paired targets)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=22)
        ax.set_ylim(0, 1.02)
        ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    fig.suptitle(
        "3DRobot — decoy-loss vs. RMSD correlations (paired PDBs, three models)",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT / "02_3drobot_boxplots_paired.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.8), dpi=150, sharey=True)
    positions = list(range(1, n_models + 1))
    vp0 = axes[0].violinplot(pear_data, positions=positions, showmeans=True, showmedians=False)
    vp1 = axes[1].violinplot(spear_data, positions=positions, showmeans=True, showmedians=False)
    for vp, pal in ((vp0, COLORS), (vp1, COLORS)):
        for b, c in zip(vp["bodies"], pal):
            b.set_facecolor(c)
            b.set_alpha(0.65)
    for ax, title in zip(axes, ("Pearson r", "Spearman ρ")):
        ax.set_xticks(positions)
        ax.set_xticklabels(disp_names, rotation=22, ha="right")
        ax.set_title(f"{title} (paired targets, n={n_paired})")
        ax.set_ylim(0, 1.02)
        ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    axes[0].set_ylabel("Correlation with RMSD")
    fig.tight_layout()
    fig.savefig(OUT / "03_3drobot_violins_paired.png", bbox_inches="tight")
    plt.close(fig)

    def _matrix_heatmap(M: np.ndarray, title: str, fname: str) -> None:
        lo, hi = float(np.nanmin(M)), float(np.nanmax(M))
        fig, ax = plt.subplots(figsize=(4.8, 4.4), dpi=150)
        im = ax.imshow(M, vmin=lo - 0.02, vmax=hi, cmap="viridis")
        ax.set_xticks(range(n_models))
        ax.set_yticks(range(n_models))
        ax.set_xticklabels(disp_names, rotation=22, ha="right")
        ax.set_yticklabels(disp_names)
        mid = (lo + hi) / 2
        for i in range(n_models):
            for j in range(n_models):
                ax.text(
                    j,
                    i,
                    f"{M[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color="w" if M[i, j] < mid else "0.1",
                    fontsize=9,
                )
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, label="Correlation")
        fig.tight_layout()
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)

    P = paired[pear_cols].to_numpy()
    S = paired[spear_cols].to_numpy()
    _matrix_heatmap(
        np.corrcoef(P.T),
        "Cross-model agreement (Pearson)\nPearson r between per-target Pearson r vectors",
        OUT / "04a_3drobot_cross_model_pearson_vectors.png",
    )
    _matrix_heatmap(
        np.corrcoef(S.T),
        "Cross-model agreement (Spearman)\nPearson r between per-target Spearman ρ vectors",
        OUT / "04b_3drobot_cross_model_spearman_vectors.png",
    )

    pairs = [(0, 1), (0, 2), (1, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), dpi=150)
    for ax, (i, j) in zip(axes, pairs):
        xi = paired[pear_cols[i]]
        xj = paired[pear_cols[j]]
        ax.scatter(xi, xj, s=12, alpha=0.45, c="#0173B2", edgecolors="none")
        ax.plot((0, 1), (0, 1), "k--", linewidth=0.8, alpha=0.6)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.set_xlabel(f"{disp_names[i]} (Pearson r)")
        ax.set_ylabel(f"{disp_names[j]} (Pearson r)")
        d = (xi - xj).to_numpy()
        ax.set_title(f"MAE |Δr| = {np.mean(np.abs(d)):.3f}")
        ax.grid(True, linestyle=":", alpha=0.4)
    fig.suptitle(f"3DRobot paired targets (n={n_paired}) — per-target Pearson r", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "05_3drobot_pairwise_scatter_pearson.png", bbox_inches="tight")
    plt.close(fig)

    rng = np.random.default_rng(0)
    B = 4000
    idx = np.arange(n_paired)
    boot_pear = {d: [] for d in disp_names}
    boot_spear = {d: [] for d in disp_names}
    for _ in range(B):
        samp = rng.choice(idx, size=n_paired, replace=True)
        sub = paired.iloc[samp]
        for i, d in enumerate(disp_names):
            boot_pear[d].append(float(sub[pear_cols[i]].mean()))
            boot_spear[d].append(float(sub[spear_cols[i]].mean()))

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), dpi=150, sharey=True)
    xb = np.arange(n_models)
    for ax, means, boot, ylab in (
        (
            axes[0],
            [paired[c].mean() for c in pear_cols],
            boot_pear,
            "Mean Pearson r",
        ),
        (
            axes[1],
            [paired[c].mean() for c in spear_cols],
            boot_spear,
            "Mean Spearman ρ",
        ),
    ):
        cis = [np.percentile(boot[d], [2.5, 97.5]) for d in disp_names]
        err = np.array([(m - lo, hi - m) for m, (lo, hi) in zip(means, cis)]).T
        ax.bar(xb, means, yerr=err, capsize=4, color=COLORS, alpha=0.85, ecolor="0.25")
        ax.set_xticks(xb)
        ax.set_xticklabels(disp_names, rotation=22, ha="right")
        ax.set_ylabel(f"{ylab} (bootstrap 95% CI)")
        ax.set_ylim(0, 1.02)
        ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    axes[0].set_title("Pearson")
    axes[1].set_title("Spearman")
    fig.suptitle(f"3DRobot — mean per-target correlation (paired PDBs, n={n_paired})", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "06_3drobot_bootstrap_mean_corr.png", bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
