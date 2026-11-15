"""Measure ‖∇E(native)‖ for each model on each of the 4 small proteins.

This tests the physical intuition that different energy functions need
different lr in Langevin MD. The displacement per MD step is
   |Δx|  ≈  lr · ‖∇E‖
so to get the SAME per-step displacement we need
   lr_A · ‖∇E_A‖  ≈  lr_B · ‖∇E_B‖
Thus the ratio ‖∇E_v1_rama‖ / ‖∇E_exp1‖ directly tells us how much smaller
v1_rama's lr should be relative to paper_exp1's (or Yang's α=0.01).

Output:
  figures/advisor_meeting/native_gradient_norms.csv
  prints per-(model, protein) ‖∇E‖ mean / std (per-residue rms, total)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

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

_ESM_CKPT_GLOB = sorted((REPO / "runs").glob("v1_esm_abl_w32_l32_attn_pool_*"))
_ESM_CKPT_DIR = _ESM_CKPT_GLOB[-1] if _ESM_CKPT_GLOB else None


def _resolve_esm_h5() -> Path | None:
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
        "argv_extra": [
            "--load_exp", str(REPO / "params" / "exp1"),
            "--mixture_seq", "1",
            "--mixture_rama", "0",
            "--legacy_local_frame",
        ],
    },
    "v1_rama": {
        "label": "v1_rama (no ESM)",
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
        print(f"[recipes] ESM model added: {_ESM_CKPT_DIR.name} + {_esm_h5}")


def shared_arch_argv() -> list[str]:
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


def setup_model(model_key: str, device_str: str):
    recipe = MODEL_RECIPES[model_key]
    sys.argv = ["native_gradient_norm.py"] + shared_arch_argv() + recipe["argv_extra"] + [
        "--device", device_str,
    ]
    parser = options.get_decoy_parser()
    args = options.parse_args_and_arch(parser)
    device, model, energy_fn, ProteinBase = test_setup(args)
    return device, model, energy_fn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    rows = []
    for model_key in MODEL_RECIPES:
        print(f"\n========== model = {model_key} ==========")
        device, model, energy_fn = setup_model(model_key, args.device)

        for prot in PROTEINS:
            bead = str(BEAD_TPL).format(prot=prot)
            seq, coords_t, profile = load_protein_bead(bead, "CA", device)
            n_res = coords_t.shape[0]
            coords_var = coords_t.detach().clone().requires_grad_(True)

            torch.set_grad_enabled(True)
            protein = Protein(seq, coords_var, profile)
            e = protein.get_energy(energy_fn)
            (grad,) = torch.autograd.grad(e, coords_var, create_graph=False)
            torch.set_grad_enabled(False)

            grad_np = grad.detach().cpu().numpy()
            per_res_norm = np.linalg.norm(grad_np, axis=1)
            total_norm = float(np.linalg.norm(grad_np))
            per_res_rms = float(per_res_norm.mean())
            per_res_max = float(per_res_norm.max())

            rows.append(dict(
                model=model_key, prot=prot, n_res=n_res,
                e_native=float(e.detach().cpu().item()),
                grad_norm_total=total_norm,
                grad_per_res_rms=per_res_rms,
                grad_per_res_max=per_res_max,
            ))
            print(f"  {prot:5s}  N={n_res:3d}  ‖∇E‖={total_norm:9.3f}  "
                  f"per-res rms={per_res_rms:7.3f}  per-res max={per_res_max:7.3f}")

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "native_gradient_norms.csv", index=False)

    print("\n========== Ratio v1_rama / paper_exp1 (per protein) ==========")
    pv = df.pivot(index="prot", columns="model", values="grad_per_res_rms")
    pv["ratio"] = pv["v1_rama"] / pv["paper_exp1"]
    print(pv.round(3))

    print("\nMean ratio:", round(pv["ratio"].mean(), 3))
    print("If ratio > 1, v1_rama needs SMALLER lr than paper_exp1 by that factor.")


if __name__ == "__main__":
    main()
