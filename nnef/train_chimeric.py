import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

import options
from dataset import DatasetLocalGenCM
from model import LocalTransformer
from paths import data_path
from trainer.local_trainer import LocalGenTrainer
from utils import resolve_model_checkpoint_path


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_drop_none(batch):
    """
    Filters out None samples (your Dataset may return None if a chimeric index is OOB).
    Works for three variants, in order of baseline-compatibility:
      L=4 : (seq, coords, start_id, res_counts)                                     -- legacy
      L=6 : (seq, coords, start_id, res_counts, rama_full, rama_mask)               -- current baseline
      L=9 : L=6 fields + (esm_block, coords_cart, seq_offset)                       -- when any of
             --use_esm / --use_cart_coords / --use_seq_offset is enabled.
      L=10: L=9 fields + (dihedral_local)                                            -- when
             --use_dihedral is also enabled (always emitted by the dataset in the
             extras path, so L=10 is the canonical extras shape; L=9 is only a
             legacy fallback).
    """
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    # inspect length
    L = len(batch[0])
    if L == 4:
        (seq, coords, start_id, res_counts) = zip(*batch)
        return (torch.stack(seq, 0),
                torch.stack(coords, 0),
                torch.stack(start_id, 0),
                torch.stack(res_counts, 0))
    elif L == 6:
        (seq, coords, start_id, res_counts, rama_full, rama_mask) = zip(*batch)
        return (torch.stack(seq, 0),
                torch.stack(coords, 0),
                torch.stack(start_id, 0),
                torch.stack(res_counts, 0),
                torch.stack(rama_full, 0),
                torch.stack(rama_mask, 0))
    elif L == 9:
        (seq, coords, start_id, res_counts, rama_full, rama_mask,
         esm_block, coords_cart, seq_offset) = zip(*batch)
        return (torch.stack(seq, 0),
                torch.stack(coords, 0),
                torch.stack(start_id, 0),
                torch.stack(res_counts, 0),
                torch.stack(rama_full, 0),
                torch.stack(rama_mask, 0),
                torch.stack(esm_block, 0),
                torch.stack(coords_cart, 0),
                torch.stack(seq_offset, 0))
    elif L == 10:
        (seq, coords, start_id, res_counts, rama_full, rama_mask,
         esm_block, coords_cart, seq_offset, dihedral_local) = zip(*batch)
        return (torch.stack(seq, 0),
                torch.stack(coords, 0),
                torch.stack(start_id, 0),
                torch.stack(res_counts, 0),
                torch.stack(rama_full, 0),
                torch.stack(rama_mask, 0),
                torch.stack(esm_block, 0),
                torch.stack(coords_cart, 0),
                torch.stack(seq_offset, 0),
                torch.stack(dihedral_local, 0))
    else:
        raise ValueError(f"Unexpected batch element length: {L}")


def make_loader(csv_path: str, args, num_samples: int, shuffle_weights: bool = True):
    csv_abs = data_path(csv_path)
    dataset = DatasetLocalGenCM(
        csv_abs,
        args,
        pdb_h5_path=args.pdb_h5_path,
        seq_h5_path=args.seq_h5_path,
        rama_h5_path=args.rama_h5_path,
        rama_dataset_name=args.rama_dataset_name,
        esm_h5_path=(args.esm_h5_path if getattr(args, 'use_esm', False) else None),
        esm_dataset_name=getattr(args, 'esm_dataset_name', 'esm'),
    )

    weights = pd.read_csv(csv_abs)["weight"].values
    sampler = WeightedRandomSampler(weights=weights,
                                    num_samples=num_samples,
                                    replacement=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=getattr(args, "pin_memory", False),
        persistent_workers=(args.num_workers > 0),
        collate_fn=collate_drop_none
    )
    return loader


def main():
    parser = options.get_local_gen_parser()
    args = options.parse_args_and_arch(parser)

    # Device default if missing
    if not hasattr(args, "device") or args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    use_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    set_seed(args.seed)

    # Create output directories. Each run lives under save_path/exp_id so
    # parallel SLURM jobs do not overwrite each other's checkpoints/logs.
    root = os.path.join(args.save_path, args.exp_id)
    os.makedirs(root, exist_ok=True)
    os.makedirs(os.path.join(root, "models"), exist_ok=True)

    writer = SummaryWriter(root)

    # ---------------- Loaders ----------------
    train_data_loader = make_loader(args.data_flag, args, num_samples=args.total_num_samples)

    if args.val_data_flag is not None:
        # For val/test we usually just want ~1 epoch worth of batches; use batch_size as a small sample
        val_data_loader = make_loader(args.val_data_flag, args, num_samples=args.batch_size)
    else:
        val_data_loader = None

    if args.test_data_flag is not None:
        test_data_loader = make_loader(args.test_data_flag, args, num_samples=args.batch_size)
    else:
        test_data_loader = None

    # ---------------- Model ----------------
    device = torch.device(args.device)
    model = LocalTransformer(args)

    try:
        if getattr(args, 'load_checkpoint', None) or args.load_exp is not None:
            ckpt_path = resolve_model_checkpoint_path(args)
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt, strict=True)
            print(f"[train_chimeric] initialised from {ckpt_path}")
    except ValueError:
        # Fresh run: neither load_checkpoint nor load_exp
        pass

    model.to(device)

    trainer = LocalGenTrainer(writer, model, device, args)

    print("Training Start")
    for i, epoch in enumerate(range(args.epochs)):
        trainer.train(epoch, train_data_loader, flag="Train")

        # Latest weights (always) + periodic epoch snapshots for decoy pick /
        # early-stop analysis. ``ep`` is 1-based epoch index.
        state = model.state_dict()
        torch.save(state, f"{root}/models/model.pt")
        ep = i + 1
        interval = int(getattr(args, "save_interval", 0) or 0)
        if interval > 0 and (ep % interval == 0):
            torch.save(state, f"{root}/models/model_epoch_{ep:04d}.pt")

        # Validation
        if val_data_loader is not None:
            for j in range(10):
                trainer.test(epoch * 10 + j, val_data_loader, flag="Val")

        # Test
        if test_data_loader is not None:
            for j in range(10):
                trainer.test(epoch * 10 + j, test_data_loader, flag="Test")


if __name__ == "__main__":
    main()
