"""Measure the training-time loss of a given checkpoint on a given h5.

Purpose: confirm whether `runs/yang_legacy_7317784` underperforms the paper's
released `params/exp1` ckpt because the training code has drifted (both
checkpoints should hit similar losses on the same data) or because our
`hhsuite_CB_cullpdb.h5` was built differently from the paper's training
data (the paper ckpt would hit much lower loss on the "correct" h5).

The script builds the exact same dataset + collate + trainer pipeline used
by `nnef/train_chimeric.py`, loads the requested checkpoint, iterates N
batches in eval mode (no grad), and reports per-component mean losses.
Invocation must use the training flags that match the ckpt's architecture
— typically pass the same flags as `fasrc/train_yang_legacy.slurm` plus
`--load_checkpoint <path>`.

Usage:
    python nnef/scripts/diagnose_train_loss.py \\
        --load_checkpoint params/exp1/models/model.pt \\
        --pdb_h5_path $DATA_DIR/hhsuite_CB_cullpdb.h5 \\
        --seq_h5_path $DATA_DIR/hhsuite_pdb_seq_v2.h5 \\
        --rama_h5_path $DATA_DIR/__NO_RAMA_SENTINEL__.h5 \\
        --data_flag hhsuite_CB_v2_pdb_list.csv \\
        --mixture_seq 1 --mixture_rama 0 --legacy_local_frame \\
        --seq_len 14 --batch_size 1000 \\
        --embed_size 32 --dim 128 --n_layers 4 --attn_heads 4 \\
        --mixture_r 2 --mixture_angle 3 \\
        --smooth_gaussian --smooth_r 0.3 --smooth_angle 45 \\
        --coords_angle_loss_lamda 1 --profile_loss_lamda 10 \\
        --use_position_weights \\
        --cen_seg_loss_lamda 1 --oth_seg_loss_lamda 3 \\
        --n_batches 30 \\
        --device cuda
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_HERE = os.path.abspath(os.path.dirname(__file__))
_NNEF_DIR = os.path.abspath(os.path.join(_HERE, '..'))
_REPO_ROOT = os.path.abspath(os.path.join(_NNEF_DIR, '..'))
if _NNEF_DIR not in sys.path:
    sys.path.insert(0, _NNEF_DIR)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import options  # noqa: E402
from model import LocalTransformer, LocalEnergyCE  # noqa: E402
from train_chimeric import make_loader  # noqa: E402


def _resolve_ckpt(args: argparse.Namespace) -> str:
    # --load_checkpoint takes precedence; otherwise --load_exp/models/model.pt.
    path = getattr(args, 'load_checkpoint', None)
    if path:
        return path if os.path.isabs(path) else os.path.join(_REPO_ROOT, path)
    load_exp = getattr(args, 'load_exp', None)
    if load_exp:
        p = os.path.join(load_exp, 'models', 'model.pt')
        return p if os.path.isabs(p) else os.path.join(_REPO_ROOT, p)
    raise SystemExit('pass --load_checkpoint or --load_exp')


def main() -> None:
    parser = options.get_local_gen_parser()
    parser.add_argument('--n_batches', type=int, default=30,
                        help='Number of batches to average over (default 30).')
    args = options.parse_args_and_arch(parser)

    if not hasattr(args, 'device') or args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(args.device)

    ckpt_path = _resolve_ckpt(args)
    if not os.path.isfile(ckpt_path):
        raise SystemExit(f'[diagnose] ckpt not found: {ckpt_path}')
    print(f'[diagnose] ckpt:        {ckpt_path}')
    print(f'[diagnose] pdb_h5_path: {args.pdb_h5_path}')
    print(f'[diagnose] seq_h5_path: {args.seq_h5_path}')
    print(f'[diagnose] rama_h5_path:{args.rama_h5_path}')
    print(f'[diagnose] data_flag:   {args.data_flag}')
    print(f'[diagnose] mixture_seq={args.mixture_seq}  mixture_rama={args.mixture_rama}  '
          f'legacy_local_frame={getattr(args, "legacy_local_frame", False)}')
    print(f'[diagnose] device={device}  n_batches={args.n_batches}  batch_size={args.batch_size}')

    # Model + strict-load so a silent shape mismatch explodes here instead
    # of being masked by random-init weights dominating the "loss".
    model = LocalTransformer(args).to(device)
    state = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state, strict=True)
    model.eval()

    energy_fn = LocalEnergyCE(model, args).to(device)

    # Loader: same shape as training, except num_samples = n_batches * batch_size
    # so the WeightedRandomSampler gives us a deterministic-size slice.
    num_samples = int(args.n_batches) * int(args.batch_size)
    loader = make_loader(args.data_flag, args, num_samples=num_samples, shuffle_weights=True)

    # Per-component accumulators. Match the order logged by LocalGenTrainer:
    #   loss_r (coords_radius), loss_angle, loss_profile, loss_start_id,
    #   loss_res_counts, loss_rama
    names = ['coords_radius_loss', 'coords_angle_loss', 'profile_loss',
             'start_id_loss', 'res_counts_loss', 'coords_rama_loss']
    totals = [0.0] * len(names)
    counts = 0
    t0 = time.time()

    with torch.no_grad():
        for i, data in enumerate(loader):
            if data is None:
                continue
            # Unpack mirrors LocalGenTrainer.step (lengths 6/9/10).
            esm_block = coords_cart = seq_offset = dihedral_local = None
            if len(data) == 10:
                (seq, coords, start_id, res_counts, rama_full, rama_mask,
                 esm_block, coords_cart, seq_offset, dihedral_local) = data
            elif len(data) == 9:
                (seq, coords, start_id, res_counts, rama_full, rama_mask,
                 esm_block, coords_cart, seq_offset) = data
            elif len(data) == 6:
                seq, coords, start_id, res_counts, rama_full, rama_mask = data
            else:
                seq, coords, start_id, res_counts = data
                N, L = seq.shape
                rama_full = torch.zeros((N, L, 2), dtype=torch.float32)
                rama_mask = torch.zeros((N, L), dtype=torch.float32)

            seq = seq.to(device); coords = coords.to(device)
            start_id = start_id.to(device); res_counts = res_counts.to(device)
            rama_full = rama_full.to(device); rama_mask = rama_mask.to(device)
            if esm_block is not None: esm_block = esm_block.to(device)
            if coords_cart is not None: coords_cart = coords_cart.to(device)
            if seq_offset is not None: seq_offset = seq_offset.to(device)
            if dihedral_local is not None: dihedral_local = dihedral_local.to(device)

            losses = energy_fn.forward(
                seq, coords, start_id, res_counts,
                rama=rama_full, rama_mask=rama_mask,
                esm=esm_block, coords_cart=coords_cart, seq_offset=seq_offset,
                dihedral=dihedral_local,
            )
            for k, l in enumerate(losses):
                totals[k] += float(l.item())
            counts += 1

            if (i + 1) % 10 == 0:
                print(f'[diagnose] batch {i+1}/{args.n_batches}  '
                      f'running total={sum(totals)/counts:.3f}')
            if counts >= args.n_batches:
                break

    if counts == 0:
        raise SystemExit('[diagnose] no batches consumed — empty loader?')

    means = [t / counts for t in totals]
    total_mean = sum(means)
    dt = time.time() - t0

    print()
    print(f'=== diagnose_train_loss ({counts} batches × {args.batch_size} samples, {dt:.1f}s) ===')
    print(f'ckpt: {ckpt_path}')
    print(f'h5:   {args.pdb_h5_path}')
    print()
    for name, m in zip(names, means):
        print(f'  {name:22s} = {m:10.4f}')
    print(f'  {"total_loss":22s} = {total_mean:10.4f}')
    print()
    print('Compare against the Train/* last values in runs/yang_legacy_7317784/'
          'events.out.tfevents.* (e.g. Train/total_loss last≈985).')


if __name__ == '__main__':
    main()
