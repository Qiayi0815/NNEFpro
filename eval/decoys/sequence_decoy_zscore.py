"""Yang 2022 Figure-6 style sequence-decoy energy gap (discrimination metric).

For each (model, protein) cell:
  - load native bead CSV (Cα coords + AA sequence)
  - compute E(native_seq, native_coords)
  - generate N decoy sequences of each type ON THE NATIVE BACKBONE:
      shuffle      — random permutation of native AA composition
      random       — each position iid uniform over 20 AAs (Yang type C)
      conservative — each position substituted within Yang chemical group
                     (Yang type A)
      polyAA       — 20 sequences, one per AA at every position (Yang type D)
  - compute E for each decoy
  - Z-score = (mean(E_decoy) − E_native) / std(E_decoy)
  - large positive Z-score = model favours native sequence on native backbone

Output:
  figures/advisor_meeting/sequence_decoy_energies.csv  (raw per-run rows)
  figures/advisor_meeting/sequence_decoy_zscores.csv   (per-cell summary)
  figures/advisor_meeting/fig7_sequence_decoy_zscore.png

Discrimination metric independent of MD parameters — orthogonal to lr/T_noise
debate. Only requires inference, no Langevin integration.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parents[1]
NNEF = REPO / "nnef"
sys.path.insert(0, str(NNEF))

import options  # noqa: E402

from protein_os import Protein  # noqa: E402
from utils import load_protein_bead, test_setup  # noqa: E402

PROTEINS = ["1ZZK", "3KXT", "5ZGM", "6H8O"]
BEAD_TPL = NNEF / "data" / "yang_small" / "{prot}" / "{prot}_bead.csv"

OUT_DIR = REPO / "figures" / "advisor_meeting"
OUT_DIR.mkdir(parents=True, exist_ok=True)

YANG_CONSERVATIVE_GROUPS = [
    "AVILM",
    "FWY",
    "ST",
    "DE",
    "KRH",
    "NQ",
    "GP",
    "C",
]

AA_LETTERS = list("ACDEFGHIKLMNPQRSTVWY")


def _aa_to_group_lookup() -> dict[str, str]:
    table: dict[str, str] = {}
    for g in YANG_CONSERVATIVE_GROUPS:
        for aa in g:
            table[aa] = g
    return table


def shared_arch_argv() -> list[str]:
    """Architecture flags shared by paper exp1 + v1_rama (per fasrc/md_eval_one.sh)."""
    return [
        "--seq_len", "14",
        "--seq_type", "residue",
        "--residue_type_num", "20",
        "--embed_size", "32",
        "--dim", "128",
        "--n_layers", "4",
        "--attn_heads", "4",
        "--mixture_r", "2",
        "--mixture_angle", "3",
        "--smooth_gaussian",
        "--smooth_r", "0.3",
        "--smooth_angle", "45",
        "--coords_angle_loss_lamda", "1",
        "--profile_loss_lamda", "10",
        "--coords_rama_loss_lamda", "1",
        "--use_position_weights",
        "--cen_seg_loss_lamda", "1",
        "--oth_seg_loss_lamda", "3",
    ]


_ESM_CKPT_GLOB = sorted((REPO / "runs").glob("v1_esm_abl_w32_l32_attn_pool_*"))
_ESM_CKPT_DIR = _ESM_CKPT_GLOB[-1] if _ESM_CKPT_GLOB else None


def _resolve_esm_h5() -> Path | None:
    """Find the ESM2-650M cache h5. Checks (in order):
      1. $NNEF_ESM_H5 env var (explicit override)
      2. <repo>/nnef/data/hhsuite_esm2_650m_v2.h5  (local convention)
      3. $DATA_DIR/hhsuite_esm2_650m_v2.h5         (cluster convention)
      4. ~/nnef_data/hhsuite_esm2_650m_v2.h5       (cluster default)
    """
    candidates = []
    if os.environ.get("NNEF_ESM_H5"):
        candidates.append(Path(os.environ["NNEF_ESM_H5"]))
    candidates.append(REPO / "nnef" / "data" / "hhsuite_esm2_650m_v2.h5")
    if os.environ.get("DATA_DIR"):
        candidates.append(Path(os.environ["DATA_DIR"]) / "hhsuite_esm2_650m_v2.h5")
    candidates.append(Path.home() / "nnef_data" / "hhsuite_esm2_650m_v2.h5")
    for c in candidates:
        if c.exists():
            return c
    return None

MODEL_RECIPES = {
    "paper_exp1": {
        "label": "Yang 2022 (paper exp1)",
        "color": "#1f77b4",
        "argv_extra": [
            "--load_exp", str(REPO / "params" / "exp1"),
            "--mixture_seq", "1",
            "--mixture_rama", "0",
            "--legacy_local_frame",
        ],
    },
    "v1_rama": {
        "label": "v1_rama (no ESM)",
        "color": "#2ca02c",
        "argv_extra": [
            "--load_exp", str(REPO / "runs" / "v1_pure_rama_v2_6228201"),
            "--mixture_rama", "10",
        ],
    },
}

if _ESM_CKPT_DIR is not None and (_ESM_CKPT_DIR / "models" / "model.pt").exists():
    _esm_h5 = _resolve_esm_h5()
    if _esm_h5 is not None:
        MODEL_RECIPES["w32_l32_attn_pool"] = {
            "label": "w32 l32 attn_pool (ESM2)",
            "color": "#d62728",
            "argv_extra": [
                "--load_exp", str(_ESM_CKPT_DIR),
                "--mixture_rama", "10",
                "--use_esm",
                "--esm_h5_path", str(_esm_h5),
                "--esm_dim_in", "1280",
                "--esm_dim_out", "32",
                "--esm_dataset_name", "esm_l32",
                "--esm_pool", "attn_pool",
            ],
        }
        print(f"[recipes] ESM model added: {_ESM_CKPT_DIR.name}  +  {_esm_h5}")
    else:
        print(f"[recipes] ESM ckpt found at {_ESM_CKPT_DIR} but no ESM h5 reachable.")
        print(f"[recipes]   set NNEF_ESM_H5=/path/to/hhsuite_esm2_650m_v2.h5")
        print(f"[recipes]   OR symlink ./nnef/data/hhsuite_esm2_650m_v2.h5 -> the real file")
        print(f"[recipes]   OR set DATA_DIR=/path/to/nnef_data on the cluster")
else:
    print("[recipes] No ESM checkpoint found under runs/v1_esm_abl_w32_l32_attn_pool_*; "
          "skipping ESM model.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu",
                   help="cpu | cuda | mps  (default cpu — Mac safe)")
    p.add_argument("--n_decoys", type=int, default=200,
                   help="decoys per (protein, model, type) for shuffle/random/conservative")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def setup_model(model_key: str, device_str: str):
    recipe = MODEL_RECIPES[model_key]
    argv = ["sequence_decoy_zscore.py"] + shared_arch_argv() + recipe["argv_extra"]
    argv += ["--device", device_str]
    sys.argv = argv
    parser = options.get_decoy_parser()
    model_args = options.parse_args_and_arch(parser)
    device, model, energy_fn, ProteinBase = test_setup(model_args)
    return device, model, energy_fn, ProteinBase, model_args


def load_one_protein(prot: str, mode: str, device):
    bead_path = str(BEAD_TPL).format(prot=prot)
    seq, coords, profile = load_protein_bead(bead_path, mode, device)
    return seq, coords, profile


def gen_decoy_profiles(native_seq_letters: np.ndarray,
                       vocab: dict[str, int],
                       n_decoys: int,
                       decoy_type: str,
                       rng: np.random.Generator) -> list[np.ndarray]:
    """Returns a list of (L,) int arrays (each = one decoy sequence as profile ids)."""
    L = len(native_seq_letters)

    if decoy_type == "shuffle":
        out = []
        for _ in range(n_decoys):
            perm = rng.permutation(L)
            shuffled = native_seq_letters[perm]
            out.append(np.array([vocab[a] for a in shuffled], dtype=np.int64))
        return out

    if decoy_type == "random":
        out = []
        for _ in range(n_decoys):
            seq = rng.choice(AA_LETTERS, size=L)
            out.append(np.array([vocab[a] for a in seq], dtype=np.int64))
        return out

    if decoy_type == "conservative":
        group = _aa_to_group_lookup()
        out = []
        for _ in range(n_decoys):
            seq = np.empty(L, dtype="<U1")
            for i, aa in enumerate(native_seq_letters):
                g = group.get(aa, AA_LETTERS)
                seq[i] = rng.choice(list(g))
            out.append(np.array([vocab[a] for a in seq], dtype=np.int64))
        return out

    if decoy_type == "polyAA":
        return [
            np.array([vocab[aa]] * L, dtype=np.int64)
            for aa in AA_LETTERS
        ]

    raise ValueError(f"unknown decoy_type {decoy_type!r}")


def compute_energy(seq_letters: np.ndarray,
                   coords_tensor: torch.Tensor,
                   profile_ids: np.ndarray,
                   energy_fn,
                   device) -> float:
    profile_tensor = torch.tensor(profile_ids, dtype=torch.long, device=device)
    protein = Protein(seq_letters, coords_tensor, profile_tensor)
    with torch.no_grad():
        e = protein.get_energy(energy_fn).item()
    return e


def amino_acids_vocab() -> dict[str, int]:
    df = pd.read_csv(NNEF / "data" / "amino_acids.csv")
    return {row.AA: row.idx - 1 for row in df.itertuples()}


def native_seq_letters_from_profile(profile_ids: np.ndarray,
                                    vocab: dict[str, int]) -> np.ndarray:
    inv = {v: k for k, v in vocab.items()}
    return np.array([inv[i] for i in profile_ids])


def run() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    vocab = amino_acids_vocab()

    raw_rows: list[dict] = []
    summary_rows: list[dict] = []

    for model_key, recipe in MODEL_RECIPES.items():
        print(f"\n========== model = {model_key} ==========")
        device, model, energy_fn, ProteinBase, _model_args = setup_model(
            model_key, args.device,
        )

        for prot in PROTEINS:
            print(f"  [{model_key} | {prot}] loading bead + computing native energy")
            seq_letters, coords, profile_native = load_one_protein(
                prot, mode="CA", device=device,
            )
            native_profile_np = profile_native.cpu().numpy()
            native_letters = native_seq_letters_from_profile(native_profile_np, vocab)

            e_native = compute_energy(seq_letters, coords, native_profile_np,
                                      energy_fn, device)
            raw_rows.append(dict(model=model_key, prot=prot,
                                 decoy_type="native", decoy_idx=-1,
                                 energy=e_native))

            for decoy_type in ("shuffle", "random", "conservative", "polyAA"):
                n_eff = 20 if decoy_type == "polyAA" else args.n_decoys
                decoys = gen_decoy_profiles(native_letters, vocab,
                                             n_eff, decoy_type, rng)
                energies = []
                for i, prof in enumerate(decoys):
                    e = compute_energy(seq_letters, coords, prof,
                                       energy_fn, device)
                    energies.append(e)
                    raw_rows.append(dict(model=model_key, prot=prot,
                                         decoy_type=decoy_type,
                                         decoy_idx=i,
                                         energy=e))
                energies = np.asarray(energies)
                mu, sd = float(energies.mean()), float(energies.std(ddof=1))
                z = (mu - e_native) / max(sd, 1e-9)
                summary_rows.append(dict(
                    model=model_key, prot=prot, decoy_type=decoy_type,
                    e_native=e_native,
                    e_decoy_mean=mu, e_decoy_std=sd,
                    z_score=z, n=int(len(energies)),
                ))
                print(f"    {prot:5s}  {decoy_type:13s}  "
                      f"E_nat={e_native:8.3f}  "
                      f"E_dec={mu:8.3f}±{sd:6.3f}  Z={z:+6.2f}  N={n_eff}")

        del model, energy_fn
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_csv(OUT_DIR / "sequence_decoy_energies.csv", index=False)
    sum_df = pd.DataFrame(summary_rows)
    sum_df.to_csv(OUT_DIR / "sequence_decoy_zscores.csv", index=False)

    print("\n========== Z-score summary (positive = native preferred) ==========")
    pivot = sum_df.pivot_table(index=["prot", "decoy_type"],
                               columns="model", values="z_score").round(2)
    print(pivot.to_string())

    make_figure(raw_df, sum_df, OUT_DIR / "fig7_sequence_decoy_zscore.png")


def make_figure(raw_df: pd.DataFrame, sum_df: pd.DataFrame, out_path: Path) -> None:
    decoy_types = ["shuffle", "random", "conservative", "polyAA"]
    n_prot = len(PROTEINS)

    model_keys = list(MODEL_RECIPES.keys())
    n_models = len(model_keys)
    fig, axes = plt.subplots(n_prot, n_models, figsize=(6.5 * n_models, 3.2 * n_prot),
                             constrained_layout=True, squeeze=False)

    for row, prot in enumerate(PROTEINS):
        for col, model_key in enumerate(model_keys):
            ax = axes[row, col]
            sub_raw = raw_df[(raw_df.model == model_key) & (raw_df.prot == prot)]
            sub_sum = sum_df[(sum_df.model == model_key) & (sum_df.prot == prot)]
            e_native = sub_raw.loc[sub_raw.decoy_type == "native", "energy"].iloc[0]

            x_positions = []
            x_labels = []
            for ti, dt in enumerate(decoy_types):
                es = sub_raw.loc[sub_raw.decoy_type == dt, "energy"].to_numpy()
                z = sub_sum.loc[sub_sum.decoy_type == dt, "z_score"].iloc[0]
                ax.scatter(np.full_like(es, ti, dtype=float),
                           es, alpha=0.35, s=14, color="#888")
                ax.scatter([ti], [np.median(es)], s=80,
                           color="black", marker="_", lw=2)
                x_positions.append(ti)
                x_labels.append(f"{dt}\nZ={z:+.1f}")

            ax.axhline(e_native, ls="--", color=MODEL_RECIPES[model_key]["color"],
                       lw=1.4, label=f"E(native)={e_native:.2f}")
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels, fontsize=8)
            ax.set_ylabel("Energy")
            ax.set_title(f"{prot}  ·  {MODEL_RECIPES[model_key]['label']}",
                         fontsize=10)
            ax.legend(loc="upper left", fontsize=8, frameon=False)
            ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        "Sequence-decoy energy gap on native backbone  ·  "
        "Yang 2022 Figure 6 reproduction (4 proteins × 2 local models)",
        fontsize=12, fontweight="bold", y=1.005,
    )
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    run()
