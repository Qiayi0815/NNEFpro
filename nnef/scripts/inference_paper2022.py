"""Inference driver aligned with Yang et al. 2022 (NNEF / ``params/exp1``).

What the paper does (and how this repo implements it)
-----------------------------------------------------
1. **Representation** — Each interior residue *i* (with sequence neighbors
   *i*±2) defines a **local building block**: **Cβ** coordinates (glycine: Cα)
   are **centered on *i***. The five consecutive residues *i*−2…*i*+2 form a
   **peptide segment**; **k=10** additional residues are the **nearest in3D
   Euclidean distance** in that centered frame. A **rotation** puts the peptide
   into a **local canonical frame** (neighbors’ vectors expressed consistently).
   See ``Protein.get_local_struct`` in ``nnef/protein_os.py``.

2. **Features** — Each of the L = 5+k sites carries **spherical internal
   coordinates** (r, θ, φ) derived from local Cartesians, **amino-acid identity**
   (one-hot / index), and **connectivity** between segment vs. “other” sites
   (``start_id`` / segment masks in training). The released **exp1** checkpoint
   was trained **without** a Ramachandran mixture output head: you must set
   ``--mixture_rama 0`` so tensor shapes match ``linear_out_x``.

3. **Model** — A **causally masked Transformer** over the L sites predicts
   **mixture-of-Gaussian** distributions over geometric and sequence variables.
   Training is **unsupervised maximum likelihood** (NLL) on native local blocks
   mined from PDBs.

4. **Inference** — For a **full structure**, blocks are built the same way;
   ``EnergyFun.forward`` evaluates the **negative log-likelihood** per block;
   ``Protein.get_energy`` **sums** them into one scalar **energy / loss** used
   to **rank decoys**, score MD frames, or drive sampling (see ``decoy_score.py``,
   ``fold_os.py``, etc.).

Usage
-----
From the **repository root** (the folder that contains ``params/`` and the
inner ``nnef/`` package)::

    # Apple Silicon: use --device mps (Metal) for faster scoring when supported.
    python nnef/scripts/inference_paper2022.py \\
        --device mps \\
        --decoy_sets casp14 --targets T1053 \\
        --exp_tag paper2022_exp1 --out_dir eval/paper2022_T1053

Single-target quick test (writes under ``nnef/data/decoys/<set>/<decoy_loss_dir>/``)::

    python nnef/scripts/inference_paper2022.py \\
        --device mps --one_target T1053 \\
        --decoy_set casp14 --decoy_loss_dir decoy_loss_paper2022

Override checkpoint (must still be **no-Rama-head** width unless you change
``--mixture_rama`` to match)::

    python nnef/scripts/inference_paper2022.py --checkpoint path/to/model.pt ...
"""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_NNEF_DIR = os.path.abspath(os.path.join(_HERE, '..'))
_REPO_ROOT = os.path.abspath(os.path.join(_NNEF_DIR, '..'))
if _NNEF_DIR not in sys.path:
    sys.path.insert(0, _NNEF_DIR)

from decoy_score import score_target  # noqa: E402
from evaluate_decoys import build_parser, run_compare_mode, run_scoring_mode  # noqa: E402
from paths import data_path  # noqa: E402
from utils import test_setup  # noqa: E402

DEFAULT_CKPT = os.path.join(_REPO_ROOT, 'params', 'exp1', 'models', 'model.pt')


def _paper_model_argv(checkpoint: str):
    """Flags that must match ``params/exp1/models/model.pt`` (Yang 2022 NNEF)."""
    return [
        '--load_checkpoint', checkpoint,
        '--mode', 'CB',
        '--seq_type', 'residue',
        '--residue_type_num', '20',
        '--seq_len', '14',
        '--embed_size', '32',
        '--dim', '128',
        '--n_layers', '4',
        '--attn_heads', '4',
        '--mixture_r', '2',
        '--mixture_angle', '3',
        '--mixture_seq', '1',
        '--mixture_rama', '0',
        '--legacy_local_frame',
        '--smooth_gaussian',
        '--smooth_r', '0.3',
        '--smooth_angle', '45',
        '--coords_angle_loss_lamda', '1',
        '--profile_loss_lamda', '10',
        '--coords_rama_loss_lamda', '1',
        '--use_position_weights',
        '--cen_seg_loss_lamda', '1',
        '--oth_seg_loss_lamda', '3',
    ]


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--checkpoint', default=DEFAULT_CKPT,
                     help='Path to Yang2022-style state_dict (default: params/exp1/models/model.pt)')
    pre.add_argument('--one_target', default='',
                     help='If set, only score this target id (needs --decoy_set).')
    pre_args, rest = pre.parse_known_args(sys.argv[1:])

    if not os.path.isfile(pre_args.checkpoint):
        print(f'[inference_paper2022] ERROR: checkpoint not found: {pre_args.checkpoint}')
        sys.exit(1)

    parser = build_parser()
    args = parser.parse_args(_paper_model_argv(pre_args.checkpoint) + rest)

    if args.compare_exps:
        run_compare_mode(args)
        return

    if pre_args.one_target:
        device, _model, energy_fn, _pb = test_setup(args)
        df = score_target(
            pre_args.one_target,
            args.decoy_set,
            args.decoy_loss_dir,
            args,
            device,
            energy_fn,
            trainer=None,
            skip_if_exists=args.skip_if_exists,
            esm_h5=None,
        )
        if df is None:
            print(f'[inference_paper2022] scoring failed for {pre_args.one_target!r}')
            sys.exit(1)
        out = data_path(
            'decoys', args.decoy_set, args.decoy_loss_dir,
            f'{pre_args.one_target}_decoy_loss.csv',
        )
        print(f'[inference_paper2022] done -> {out}')
        return

    run_scoring_mode(args)


if __name__ == '__main__':
    main()
