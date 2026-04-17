"""Write a randomly-initialised ``models/model.pt`` for decoy smoke tests.

Training on FASRC will overwrite this with real weights. The hyper-parameters
passed here must match ``fasrc/train_v1_pure.slurm`` (or whatever checkpoint you
plan to load in ``evaluate_decoys.py`` / ``decoy_score.py``).

Example (from repo root; use ``--device cpu`` on machines without CUDA)::

    PYTHONPATH=nnef python nnef/scripts/save_init_checkpoint.py \\
        --out_dir runs/v1_pure_init_checkpoint --device cpu --seed 42 \\
        --seq_type residue --residue_type_num 20 --seq_len 14 \\
        --embed_size 32 --dim 128 --n_layers 4 --attn_heads 4 \\
        --mixture_r 2 --mixture_angle 3 --mixture_rama 10 \\
        --smooth_gaussian --smooth_r 0.3 --smooth_angle 45
"""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_NNEF_DIR = os.path.abspath(os.path.join(_HERE, '..'))
if _NNEF_DIR not in sys.path:
    sys.path.insert(0, _NNEF_DIR)

import torch  # noqa: E402

import options  # noqa: E402
from model import LocalTransformer  # noqa: E402


def build_parser():
    parser = options.get_decoy_parser()
    parser.description = __doc__
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Directory that will contain models/model.pt')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    os.makedirs(os.path.join(args.out_dir, 'models'), exist_ok=True)
    out_pt = os.path.join(args.out_dir, 'models', 'model.pt')

    model = LocalTransformer(args).eval()
    torch.save(model.state_dict(), out_pt)
    print(f'[save_init_checkpoint] wrote {out_pt} ({sum(p.numel() for p in model.parameters())} params)')


if __name__ == '__main__':
    main()
