import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
from optim import ScheduledOptim
from model import LocalEnergyCE


class LocalGenTrainer:
    def __init__(self, writer, model, device, args):
        self.model = model
        self.energy_fn = LocalEnergyCE(model, args)

        self.device = device

        # Optimizer & scheduler
        self.optim = Adam(self.model.parameters(), lr=args.lr, betas=args.betas, weight_decay=args.weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, init_lr=args.lr,
                                             n_warmup_steps=args.n_warmup_steps,
                                             steps_decay_scale=args.steps_decay_scale)

        self.log_freq = args.log_interval
        self.writer = writer

        print("Total Parameters:", sum(p.nelement() for p in self.model.parameters()))

    def step(self, data):
        """
        Expects batches from the DataLoader in the form:
        seq, coords, start_id, res_counts, rama_full, rama_mask
        Shapes:
          seq:        (N, L)          long
          coords:     (N, L, 3)       float
          start_id:   (N, L)          long
          res_counts: (N, 3)          float (unused in CE head, kept for API compatibility)
          rama_full:  (N, L, 2)       float (phi, psi) radians
          rama_mask:  (N, L)          float/bool (1.0 valid, 0.0 invalid)
        """
        # Unpack; allow data to be a list/tuple produced by a custom collate.
        # Three shapes are supported so that baseline runs, rama runs, and
        # ESM/cart/seq_offset runs all share the same trainer code path.
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
            # Backward compatibility: older loaders without rama
            seq, coords, start_id, res_counts = data
            # Create dummy rama tensors so the call still works
            N, L = seq.shape
            rama_full = torch.zeros((N, L, 2), dtype=torch.float32)
            rama_mask = torch.zeros((N, L), dtype=torch.float32)

        # Move to device
        seq = seq.to(self.device)                # (N, L)
        coords = coords.to(self.device)          # (N, L, 3)
        start_id = start_id.to(self.device)      # (N, L)
        res_counts = res_counts.to(self.device)  # (N, 3)
        rama_full = rama_full.to(self.device)    # (N, L, 2)
        rama_mask = rama_mask.to(self.device)    # (N, L)
        if esm_block is not None:
            esm_block = esm_block.to(self.device)      # (N, L, d_esm_in)
        if coords_cart is not None:
            coords_cart = coords_cart.to(self.device)  # (N, L, 3)
        if seq_offset is not None:
            seq_offset = seq_offset.to(self.device)    # (N, L), long
        if dihedral_local is not None:
            dihedral_local = dihedral_local.to(self.device)  # (N, L, 4)

        # Forward through energy (CE variant): returns 6 losses with rama added
        loss_r, loss_angle, loss_seq, loss_start_id, loss_res_counts, loss_rama = \
            self.energy_fn.forward(
                seq, coords, start_id, res_counts,
                rama=rama_full, rama_mask=rama_mask,
                esm=esm_block, coords_cart=coords_cart, seq_offset=seq_offset,
                dihedral=dihedral_local,
            )

        return loss_r, loss_angle, loss_seq, loss_start_id, loss_res_counts, loss_rama

    def train(self, epoch, data_loader, flag='Train'):
        if flag == 'Train':
            self.model.train()
            torch.set_grad_enabled(True)
        else:
            self.model.eval()
            torch.set_grad_enabled(False)

        for i, data in tqdm(enumerate(data_loader)):
            # If your dataset might yield None for some samples, you may need a collate_fn that drops them.
            # Here we assume the DataLoader already handles that (recommended).
            losses = self.step(data)
            loss_r, loss_angle, loss_profile, loss_start_id, loss_res_counts, loss_rama = losses

            # Total loss now includes Ramachandran
            loss = loss_r + loss_angle + loss_profile + loss_start_id + loss_res_counts + loss_rama

            if flag == 'Train':
                self.optim_schedule.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optim_schedule.step_and_update_lr()

            # Logging
            len_data_loader = len(data_loader)
            log_freq = self.log_freq if flag == 'Train' else 1
            if i % log_freq == 0:
                global_step = epoch * len_data_loader + i
                self.writer.add_scalar(f'{flag}/profile_loss', loss_profile.item(), global_step)
                self.writer.add_scalar(f'{flag}/coords_radius_loss', loss_r.item(), global_step)
                self.writer.add_scalar(f'{flag}/coords_angle_loss', loss_angle.item(), global_step)
                self.writer.add_scalar(f'{flag}/coords_rama_loss', loss_rama.item(), global_step)   # NEW
                self.writer.add_scalar(f'{flag}/start_id_loss', loss_start_id.item(), global_step)
                self.writer.add_scalar(f'{flag}/res_counts_loss', loss_res_counts.item(), global_step)
                self.writer.add_scalar(f'{flag}/total_loss', loss.item(), global_step)

                print(f'{flag} epoch {epoch} Iter: {i} '
                      f'profile_loss: {loss_profile.item():.3f} '
                      f'coords_radius_loss: {loss_r.item():.3f} '
                      f'coords_angle_loss: {loss_angle.item():.3f} '
                      f'coords_rama_loss: {loss_rama.item():.3f} '   # NEW
                      f'start_id_loss: {loss_start_id.item():.3f} '
                      f'res_counts_loss: {loss_res_counts.item():.3f} '
                      f'total_loss: {loss.item():.3f} ')

        # Restore training mode after validation/test pass
        if flag != 'Train':
            self.model.train()
            torch.set_grad_enabled(True)

    def test(self, epoch, data_loader, flag='Test'):
        self.train(epoch, data_loader, flag=flag)
