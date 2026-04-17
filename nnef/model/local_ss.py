import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np
import pandas as pd

from paths import data_path

"""
Adds a Ramachandran (phi, psi) mixture head and loss.
- Pass rama: (N, L, 2) radians, wrapped to (-pi, pi] is fine (we also re-wrap to be safe)
- Pass rama_mask: (N, L) float/bool, 1 for valid positions, 0 for invalid
"""


# ---------------------- Helper: periodic wrap around predicted mean ----------------------
def _wrap_to_mu(x, mu):
    """
    Wrap x (angles) so it's the closest branch to mu within (-pi, pi].
    x, mu: broadcastable tensors
    """
    pi = np.pi
    # bring x to (-pi, pi]
    x0 = ((x + pi) % (2 * pi)) - pi
    # adjust to be closest to mu
    idx1 = (mu > -np.pi) & (mu < np.pi) & (x0 - mu > np.pi)
    idx2 = (mu > -np.pi) & (mu < np.pi) & (x0 - mu < -np.pi)
    x2 = torch.ones_like(x0)
    x2[:] = x0[:]
    x2[idx1] = x0[idx1] - 2 * np.pi
    x2[idx2] = x0[idx2] + 2 * np.pi
    return x2


class LocalTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.seq_type = args.seq_type
        self.residue_type_num = args.residue_type_num
        if self.seq_type == 'residue':
            self.aa_embedding = nn.Embedding(self.residue_type_num, embedding_dim=args.embed_size)
        else:
            self.linear_seq = nn.Linear(self.residue_type_num, args.embed_size)

        self.seq_start_embed = nn.Embedding(1, embedding_dim=args.embed_size)
        self.pos_x_embed = nn.Embedding(args.seq_len + 1, embedding_dim=args.embed_size // 4)
        self.pos_s_embed = nn.Embedding(args.seq_len + 1, embedding_dim=args.embed_size // 4)

        self.start_id_embed = nn.Embedding(2, embedding_dim=args.embed_size // 4)

        # ---- Optional feature-extraction enhancements (see options.py) ----
        # Design: keep the baseline linear_x unchanged and add one ADDITIVE
        # side layer per extra feature. Each side layer is zero-initialized,
        # so (i) a baseline checkpoint loads into the extended model via a
        # plain strict=False call (no shape mismatch), (ii) identity-at-zero
        # is automatic without any weight-copying gymnastics at load time,
        # and (iii) the model only picks up the extras when it learns to.
        self.use_esm = bool(getattr(args, 'use_esm', False))
        self.use_cart_coords = bool(getattr(args, 'use_cart_coords', False))
        self.use_seq_offset = bool(getattr(args, 'use_seq_offset', False))
        self.use_dihedral = bool(getattr(args, 'use_dihedral', False))

        self.linear_x = nn.Linear(2 * (args.embed_size // 4) + 3, args.dim)
        self.linear_s = nn.Linear(args.embed_size + args.embed_size // 4, args.dim)

        if self.use_esm:
            self.esm_dim_in = int(getattr(args, 'esm_dim_in', 1280))
            self.esm_dim_out = int(getattr(args, 'esm_dim_out', 32))
            hidden_e = max(4 * self.esm_dim_out, self.esm_dim_out)
            self.esm_proj = nn.Sequential(
                nn.Linear(self.esm_dim_in, hidden_e),
                nn.ReLU(),
                nn.Linear(hidden_e, self.esm_dim_out),
            )
            self.linear_x_esm = nn.Linear(self.esm_dim_out, args.dim)
            # Zero-init both the final projection and the additive side layer
            # so the ESM branch contributes nothing at step 0 regardless of
            # the ESM tensor magnitude.
            nn.init.zeros_(self.esm_proj[-1].weight)
            nn.init.zeros_(self.esm_proj[-1].bias)
            nn.init.zeros_(self.linear_x_esm.weight)
            nn.init.zeros_(self.linear_x_esm.bias)

        if self.use_cart_coords:
            # Raw block-local (x, y, z) complementing spherical (r, θ, φ).
            self.linear_x_cart = nn.Linear(3, args.dim)
            nn.init.zeros_(self.linear_x_cart.weight)
            nn.init.zeros_(self.linear_x_cart.bias)

        if self.use_seq_offset:
            self.seq_offset_max = int(getattr(args, 'seq_offset_max', 64))
            self.seq_offset_embed = nn.Embedding(
                num_embeddings=2 * self.seq_offset_max + 1,
                embedding_dim=args.embed_size // 4,
            )
            self.linear_x_offset = nn.Linear(args.embed_size // 4, args.dim)
            nn.init.zeros_(self.seq_offset_embed.weight)
            nn.init.zeros_(self.linear_x_offset.weight)
            nn.init.zeros_(self.linear_x_offset.bias)

        if self.use_dihedral:
            # Backbone dihedrals (phi, psi) injected as sin/cos 4-d per residue.
            # Zero-init -> baseline bit-identity under --use_dihedral at step 0.
            # Causal mask on the structure axis already guarantees the rama
            # prediction head cannot read phi/psi of the position it's asked
            # to predict, so no target leakage.
            self.linear_x_dihedral = nn.Linear(4, args.dim)
            nn.init.zeros_(self.linear_x_dihedral.weight)
            nn.init.zeros_(self.linear_x_dihedral.bias)

        hidden = args.dim
        out_hidden = 2 * args.dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=args.attn_heads,
            dim_feedforward=hidden * 4, dropout=args.dropout
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden, nhead=args.attn_heads,
            dim_feedforward=hidden * 4, dropout=args.dropout
        )

        mask = torch.tril(torch.ones((args.seq_len + 1, args.seq_len + 1), device=args.device))
        mask = mask.masked_fill((mask == 0), float('-inf')).masked_fill(mask == 1, float(0.0))
        self.register_buffer('mask', mask)

        self._encoder1 = nn.TransformerEncoder(encoder_layer, args.n_layers // 2)
        self._encoder2 = nn.TransformerEncoder(encoder_layer, args.n_layers // 2)
        self._decoder = nn.TransformerDecoder(decoder_layer, args.n_layers)

        # ---------------------- Output dimensions ----------------------
        self.connect_dim = 2
        self.out_r_dim = args.mixture_r * 3  # (pi, mu, sigma)
        self.out_angle_dim = args.mixture_angle * 6  # (pi, mu1, mu2, sigma1, sigma2, rho)
        self.out_rama_dim = args.mixture_rama * 6  # NEW: Ramachandran (phi, psi) mixture
        if self.seq_type == 'residue':
            self.out_profile_dim = self.residue_type_num
        else:
            self.out_profile_dim = args.mixture_seq * 3 * self.residue_type_num

        # Structure head now includes Ramachandran slice before connectivity logits
        self.out_x_dim = self.out_r_dim + self.out_angle_dim + self.out_rama_dim + self.connect_dim
        self.out_s_dim = self.out_profile_dim

        self.linear_out_x = nn.Sequential(
            nn.Linear(hidden, out_hidden), nn.ReLU(),
            nn.Linear(out_hidden, out_hidden), nn.ReLU(),
            nn.Linear(out_hidden, self.out_x_dim)
        )
        self.linear_out_s = nn.Sequential(
            nn.Linear(hidden, out_hidden), nn.ReLU(),
            nn.Linear(out_hidden, out_hidden), nn.ReLU(),
            nn.Linear(out_hidden, self.out_s_dim)
        )

    def forward(self, seq, coords, start_id,
                esm=None, coords_cart=None, seq_offset=None,
                dihedral=None):
        if self.seq_type == 'residue':
            seq_feature = self.aa_embedding(seq)
        else:
            seq_feature = self.linear_seq(seq)

        seq_start_feature = self.seq_start_embed.weight.expand((seq_feature.size(0), -1, -1))
        seq_feature = torch.cat((seq_start_feature, seq_feature), dim=1)  # (N, L+1, E)

        start_id_feature = self.start_id_embed(start_id)
        pos_x_feature = self.pos_x_embed.weight.expand((coords.size(0), -1, -1))
        pos_s_feature = self.pos_s_embed.weight.expand((seq_feature.size(0), -1, -1))

        s_feature = torch.cat((seq_feature, pos_s_feature), -1)
        x_feature = torch.cat((coords, start_id_feature, pos_x_feature), -1)

        s_feature = self.linear_s(s_feature)
        x_feature = self.linear_x(x_feature)

        # Optional additive extras (see options.py). Each branch is gated by
        # its flag AND by the presence of the corresponding input; when both
        # are off, this block is a no-op so the baseline path is bit-identical.
        # Every side layer is zero-initialized -> no contribution at step 0.
        if self.use_esm and esm is not None:
            e_proj = self.esm_proj(esm)                   # (N, L, esm_dim_out)
            x_feature = x_feature + self.linear_x_esm(e_proj)
        if self.use_cart_coords and coords_cart is not None:
            x_feature = x_feature + self.linear_x_cart(coords_cart)
        if self.use_seq_offset and seq_offset is not None:
            off_feature = self.seq_offset_embed(seq_offset)   # (N, L, E//4)
            x_feature = x_feature + self.linear_x_offset(off_feature)
        if self.use_dihedral and dihedral is not None:
            # dihedral: (N, L, 4) -- (sin phi, cos phi, sin psi, cos psi).
            x_feature = x_feature + self.linear_x_dihedral(dihedral)

        s_feature = s_feature.transpose(0, 1)  # (L, N, E)
        x_feature = x_feature.transpose(0, 1)  # (L, N, E)

        code_x = self._encoder1(x_feature, mask=self.mask)
        out_x = self._encoder2(code_x, mask=self.mask)
        out_x = out_x.transpose(0, 1)  # (N, L, E)

        out_s = self._decoder(s_feature, code_x, tgt_mask=self.mask)
        out_s = out_s.transpose(0, 1)  # (N, L, E)

        out_x = self.linear_out_x(out_x)
        out_s = self.linear_out_s(out_s)
        return out_x, out_s


class LocalEnergy(nn.Module):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.m_r = args.mixture_r
        self.m_angle = args.mixture_angle
        self.m_rama = args.mixture_rama  # NEW
        self.m_seq = args.mixture_seq
        self.m_res = args.mixture_res_counts
        self.residue_type_num = args.residue_type_num

        self.start_id_loss = nn.CrossEntropyLoss()
        self.profile_prob = args.profile_prob
        self.profile_loss_lamda = args.profile_loss_lamda
        self.coords_angle_loss_lamda = args.coords_angle_loss_lamda
        self.coords_rama_loss_lamda = args.coords_rama_loss_lamda  # NEW

    def forward(self, seq, coords, start_id, rama=None, rama_mask=None):
        seq_len = seq.shape[-1]

        input_seq = seq[:, :-1]
        input_coords = coords
        input_start_id = start_id

        target_seq = seq
        target_coords = coords[:, 1:, :]

        out_x, out_s = self.model(input_seq, input_coords, input_start_id)

        if seq_len > 6:
            target_start_id = start_id[:, 6:]  # ignore first 6
            out_start_id = out_x[:, 5:-1, -2:].transpose(1, 2)  # logits for connectivity
            loss_start_id = self.start_id_loss(out_start_id, target_start_id)
        else:
            loss_start_id = torch.tensor(0.0, device=seq.device)

        loss_r, loss_angle, loss_profile, loss_rama = self.get_mixture_loss(
            out_x[:, :-1, :-2], out_s, target_coords, target_seq, rama=rama, rama_mask=rama_mask
        )

        loss_angle *= self.coords_angle_loss_lamda
        loss_profile *= self.profile_loss_lamda
        loss_rama *= self.coords_rama_loss_lamda  # NEW

        return loss_r, loss_angle, loss_profile, loss_start_id, loss_rama

    def get_mixture_coef(self, out_x, out_s):
        m_r, m_angle, m_seq = self.m_r, self.m_angle, self.m_seq
        m_rama = self.m_rama

        N, L, _ = out_x.size()
        idx1 = 3 * m_r
        idx2 = idx1 + 6 * m_angle
        idx3 = idx2 + 6 * m_rama

        z_r = out_x[:, :, 0:idx1].reshape((N, L, m_r, 3))
        z_angle = out_x[:, :, idx1:idx2].reshape((N, L, m_angle, 6))
        z_rama = out_x[:, :, idx2:idx3].reshape((N, L, m_rama, 6))  # NEW

        # sequence/profile head
        z_profile = out_s.reshape((N, L + 1, self.residue_type_num, m_seq, 3))

        # mixture weights
        r_pi = F.softmax(z_r[:, :, :, 0], dim=-1)
        angle_pi = F.softmax(z_angle[:, :, :, 0], dim=-1)
        rama_pi = F.softmax(z_rama[:, :, :, 0], dim=-1)  # NEW
        profile_pi = F.softmax(z_profile[:, :, :, :, 0], dim=-1)

        # means
        r_mu = z_r[:, :, :, 1]
        angle_mu1 = z_angle[:, :, :, 1]
        angle_mu2 = z_angle[:, :, :, 2]
        rama_mu_phi = z_rama[:, :, :, 1]  # NEW
        rama_mu_psi = z_rama[:, :, :, 2]  # NEW
        profile_mu = z_profile[:, :, :, :, 1]

        # sigmas and correlations
        r_sigma = torch.exp(z_r[:, :, :, 2])
        angle_sigma1 = torch.exp(z_angle[:, :, :, 3])
        angle_sigma2 = torch.exp(z_angle[:, :, :, 4])
        angle_corr = torch.tanh(z_angle[:, :, :, 5])

        rama_sigma_phi = torch.exp(z_rama[:, :, :, 3])  # NEW
        rama_sigma_psi = torch.exp(z_rama[:, :, :, 4])  # NEW
        rama_corr = torch.tanh(z_rama[:, :, :, 5])  # NEW

        profile_sigma = torch.exp(z_profile[:, :, :, :, 2]) + 0.002

        m_coef = [
            r_pi, r_mu, r_sigma,
            angle_pi, angle_mu1, angle_mu2, angle_sigma1, angle_sigma2, angle_corr,
            rama_pi, rama_mu_phi, rama_mu_psi, rama_sigma_phi, rama_sigma_psi, rama_corr,
            profile_pi, profile_mu, profile_sigma
        ]
        return m_coef

    # model/local_ss.py  (only replace LocalEnergyCE.get_mixture_loss)

    def get_mixture_loss(self, out, target_coords, rama=None, rama_mask=None):
        r, theta, phi = target_coords[:, :, 0:1], target_coords[:, :, 1:2], target_coords[:, :, 2:3]

        (r_pi, r_mu, r_sigma,
         angle_pi, angle_mu1, angle_mu2, angle_sigma1, angle_sigma2, angle_corr,
         rama_pi, rama_mu_phi, rama_mu_psi, rama_sigma_phi, rama_sigma_psi, rama_corr) = self.get_mixture_coef(out)

        def normal_1d(x, mu, sigma):
            norm = (2 * np.pi) ** 0.5 * sigma
            exp = torch.exp(-0.5 * (x - mu) ** 2 / (sigma ** 2))
            return exp / norm

        def normal_2d(x1, x2, mu1, mu2, s1, s2, rho):
            z1 = (x1 - mu1) ** 2 / (s1 ** 2)
            z2 = (x2 - mu2) ** 2 / (s2 ** 2)
            z12 = (x1 - mu1) * (x2 - mu2) / (s1 * s2)
            z = z1 + z2 - 2 * rho * z12
            exp = torch.exp(-z / (2 * (1 - rho ** 2)))
            norm = 2 * np.pi * s1 * s2 * torch.sqrt(1 - rho ** 2)
            return exp / norm

        def log_weighted_sum(x, w):
            x = torch.sum(x * w, dim=-1)
            return -torch.log(x + 1e-9)

        # --- r loss ---
        loss_r = normal_1d(r, r_mu, r_sigma)
        loss_r = log_weighted_sum(loss_r, r_pi)
        loss_r = torch.mean(loss_r)

        # --- (theta, phi) angle loss (wrap phi) ---
        phi_rep = phi.repeat([1, 1, self.m_angle])
        phi_w = _wrap_to_mu(phi_rep, angle_mu2)

        loss_angle1 = normal_2d(theta[:, 2:], phi_w[:, 2:], angle_mu1[:, 2:], angle_mu2[:, 2:],
                                angle_sigma1[:, 2:], angle_sigma2[:, 2:], angle_corr[:, 2:])
        loss_angle1 = log_weighted_sum(loss_angle1, angle_pi[:, 2:])
        loss_angle2 = normal_1d(phi_w[:, 1], angle_mu2[:, 1], angle_sigma2[:, 1])
        loss_angle2 = log_weighted_sum(loss_angle2, angle_pi[:, 1])
        loss_angle = torch.mean(loss_angle1) + torch.mean(loss_angle2)

        # --- Ramachandran ---
        loss_rama = torch.tensor(0.0, device=out.device)
        if rama is not None:
            # Align to L-1
            rama = rama[:, 1:, :]  # (N, L-1, 2)
            if rama_mask is not None:
                rama_mask = rama_mask[:, 1:]  # (N, L-1)

            # Fallback if mask is missing or all-zero
            if (rama_mask is None) or (rama_mask.sum() == 0):
                rama_mask = torch.ones(rama.shape[:2], dtype=torch.float32, device=out.device)

            # Per-sample all-zero rows? Replace those rows with ones.
            # This avoids zeroing the entire sample’s Rama loss.
            row_sums = rama_mask.sum(dim=-1, keepdim=True)  # (N,1)
            fix_rows = (row_sums == 0)
            if torch.any(fix_rows):
                rama_mask = rama_mask.clone()
                rama_mask[fix_rows.squeeze(-1)] = 1.0

            phi_ram = rama[:, :, 0:1]
            psi_ram = rama[:, :, 1:2]
            phi_rep_r = phi_ram.repeat([1, 1, self.m_rama])
            psi_rep_r = psi_ram.repeat([1, 1, self.m_rama])

            phi_wr = _wrap_to_mu(phi_rep_r, rama_mu_phi)
            psi_wr = _wrap_to_mu(psi_rep_r, rama_mu_psi)

            # numerically safer pdf (tiny floor on sigma)
            eps_s = 1e-5
            pdf = normal_2d(phi_wr, psi_wr,
                            rama_mu_phi, rama_mu_psi,
                            torch.clamp(rama_sigma_phi, min=eps_s),
                            torch.clamp(rama_sigma_psi, min=eps_s),
                            torch.clamp(rama_corr, min=-0.99, max=0.99))

            mix = torch.sum(pdf * rama_pi, dim=-1)  # (N, L-1)
            rama_nll = -torch.log(mix + 1e-9)

            # Masked mean per sample
            rama_nll = rama_nll * rama_mask
            denom = torch.clamp(rama_mask.sum(dim=-1), min=1.0)  # (N,)
            rama_nll = rama_nll.sum(dim=-1) / denom
            loss_rama = rama_nll.mean()

            # Optional: tiny debug counters as buffers you can print occasionally
            if not hasattr(self, "_dbg_rama_all_zero_batches"):
                self.register_buffer("_dbg_rama_all_zero_batches", torch.zeros(1))
            if not hasattr(self, "_dbg_rama_seen"):
                self.register_buffer("_dbg_rama_seen", torch.zeros(1))

            # Batch had any zero rows originally?
            if torch.any(row_sums == 0):
                self._dbg_rama_all_zero_batches += 1
            self._dbg_rama_seen += 1

        return loss_r, loss_angle, loss_rama


class LocalEnergyCE(nn.Module):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.m_r = args.mixture_r
        self.m_angle = args.mixture_angle
        self.m_rama = args.mixture_rama  # NEW

        self.random_ref = args.random_ref
        self.smooth_gaussian = args.smooth_gaussian
        self.smooth_r = args.smooth_r
        self.smooth_angle = args.smooth_angle

        self.reduction = args.reduction

        self.seq_loss = nn.CrossEntropyLoss(reduction='none')

        if self.random_ref:
            df = pd.read_csv(data_path('aa_freq.csv'))
            self.aa_freq = df['freq'].values / df['freq'].sum()

        self.start_id_loss = nn.CrossEntropyLoss(reduction='none')

        self.profile_loss_lamda = args.profile_loss_lamda
        self.coords_angle_loss_lamda = args.coords_angle_loss_lamda
        self.coords_rama_loss_lamda = args.coords_rama_loss_lamda  # NEW

        if args.use_position_weights:
            device = torch.device(args.device)
            position_weights = torch.ones((1, args.seq_len + 1), device=device)
            position_weights[:, 0:5] *= args.cen_seg_loss_lamda
            position_weights[:, 5:] *= args.oth_seg_loss_lamda
            self.position_weights = position_weights
        else:
            self.position_weights = None

    def forward(self, seq, coords, start_id, res_counts,
                rama=None, rama_mask=None,
                esm=None, coords_cart=None, seq_offset=None,
                dihedral=None):
        seq_len = seq.shape[-1]

        input_seq = seq[:, :-1]
        input_coords = coords
        input_start_id = start_id

        target_seq = seq
        target_coords = coords[:, 1:, :]

        out_x, out_s = self.model(
            input_seq, input_coords, input_start_id,
            esm=esm, coords_cart=coords_cart, seq_offset=seq_offset,
            dihedral=dihedral,
        )

        # structure losses (r, angles, rama)
        loss_r, loss_angle, loss_rama = self.get_mixture_loss(out_x[:, :-1, :-2], target_coords,
                                                              rama=rama, rama_mask=rama_mask)

        # sequence CE
        loss_seq = self.seq_loss(out_s.transpose(1, 2), target_seq)  # (N, L)
        if self.random_ref:
            aa_freq = torch.tensor(self.aa_freq, dtype=torch.float, device=target_seq.device)
            seq_prob = aa_freq[target_seq]
            loss_seq_ref = torch.log(seq_prob)
            loss_seq = loss_seq + loss_seq_ref

        # connectivity/start_id
        if seq_len > 6:
            target_start_id = start_id[:, 6:]
            out_start_id = out_x[:, 5:-1, -2:].transpose(1, 2)
            loss_start_id = self.start_id_loss(out_start_id, target_start_id)
            if self.position_weights is not None:
                loss_start_id = loss_start_id * self.position_weights[:, 6:]
            loss_start_id = torch.sum(loss_start_id, dim=-1)
            if self.reduction != 'keep_batch_dim':
                loss_start_id = torch.mean(loss_start_id)
        else:
            loss_start_id = torch.tensor([0], dtype=torch.float, device=seq.device)

        # position-weighted sequence aggregation
        if self.position_weights is not None:
            loss_seq = loss_seq * self.position_weights  # (N, L)
        loss_seq = torch.sum(loss_seq, dim=-1)
        if self.reduction != 'keep_batch_dim':
            loss_seq = torch.mean(loss_seq)

        # weights
        loss_angle *= self.coords_angle_loss_lamda
        loss_seq *= self.profile_loss_lamda
        loss_rama *= self.coords_rama_loss_lamda  # NEW

        loss_res_counts = torch.tensor([0], dtype=torch.float, device=seq.device)

        return loss_r, loss_angle, loss_seq, loss_start_id, loss_res_counts, loss_rama

    def get_mixture_coef(self, out):
        m_r, m_angle, m_rama = self.m_r, self.m_angle, self.m_rama

        N, L, _ = out.size()
        idx1 = 3 * m_r
        idx2 = idx1 + 6 * m_angle
        idx3 = idx2 + 6 * m_rama

        z_r = out[:, :, 0:idx1].reshape((N, L, m_r, 3))
        z_angle = out[:, :, idx1:idx2].reshape((N, L, m_angle, 6))

        r_pi = F.softmax(z_r[:, :, :, 0], dim=-1)
        angle_pi = F.softmax(z_angle[:, :, :, 0], dim=-1)

        r_mu = z_r[:, :, :, 1]
        r_sigma = torch.exp(z_r[:, :, :, 2])

        angle_mu1 = z_angle[:, :, :, 1]
        angle_mu2 = z_angle[:, :, :, 2]
        angle_sigma1 = torch.exp(z_angle[:, :, :, 3])
        angle_sigma2 = torch.exp(z_angle[:, :, :, 4])

        if self.smooth_gaussian:
            angle_sigma1 = angle_sigma1 + np.pi / 180.0 * self.smooth_angle
            angle_sigma2 = angle_sigma2 + np.pi / 180.0 * self.smooth_angle
            r_sigma = r_sigma + self.smooth_r

        angle_corr = torch.tanh(z_angle[:, :, :, 5]).clamp(min=-0.99, max=0.99)

        # Ramachandran params (pre-rama checkpoints: m_rama == 0, no slice / no softmax on empty)
        if m_rama > 0:
            z_rama = out[:, :, idx2:idx3].reshape((N, L, m_rama, 6))
            rama_pi = F.softmax(z_rama[:, :, :, 0], dim=-1)
            rama_mu_phi = z_rama[:, :, :, 1]
            rama_mu_psi = z_rama[:, :, :, 2]
            rama_sigma_phi = torch.exp(z_rama[:, :, :, 3])
            rama_sigma_psi = torch.exp(z_rama[:, :, :, 4])
            rama_corr = torch.tanh(z_rama[:, :, :, 5]).clamp(min=-0.99, max=0.99)
        else:
            rama_pi = out.new_ones((N, L, 1))
            rama_mu_phi = out.new_zeros((N, L, 1))
            rama_mu_psi = out.new_zeros((N, L, 1))
            rama_sigma_phi = out.new_ones((N, L, 1))
            rama_sigma_psi = out.new_ones((N, L, 1))
            rama_corr = out.new_zeros((N, L, 1))

        return (r_pi, r_mu, r_sigma,
                angle_pi, angle_mu1, angle_mu2, angle_sigma1, angle_sigma2, angle_corr,
                rama_pi, rama_mu_phi, rama_mu_psi, rama_sigma_phi, rama_sigma_psi, rama_corr)

    def get_mixture_loss(self, out, target_coords, rama=None, rama_mask=None):
        # targets over L-1 positions (coords[:, 1:, :])
        r, theta, phi = target_coords[:, :, 0:1], target_coords[:, :, 1:2], target_coords[:, :, 2:3]

        (r_pi, r_mu, r_sigma,
         angle_pi, angle_mu1, angle_mu2, angle_sigma1, angle_sigma2, angle_corr,
         rama_pi, rama_mu_phi, rama_mu_psi, rama_sigma_phi, rama_sigma_psi, rama_corr) = self.get_mixture_coef(out)

        def normal_1d(x, mu, sigma):
            norm = (2 * np.pi) ** 0.5 * sigma
            exp = torch.exp(-0.5 * (x - mu) ** 2 / (sigma ** 2))
            return exp / norm

        def normal_2d(x1, x2, mu1, mu2, s1, s2, rho):
            z1 = (x1 - mu1) ** 2 / (s1 ** 2)
            z2 = (x2 - mu2) ** 2 / (s2 ** 2)
            z12 = (x1 - mu1) * (x2 - mu2) / (s1 * s2)
            z = z1 + z2 - 2 * rho * z12
            exp = torch.exp(-z / (2 * (1 - rho ** 2)))
            norm = 2 * np.pi * s1 * s2 * torch.sqrt(1 - rho ** 2)
            return exp / norm

        def log_weighted_sum(x, w):
            x = torch.sum(x * w, dim=-1)
            return -torch.log(x + 1e-9)

        # --- r loss ---
        loss_r = normal_1d(r, r_mu, r_sigma)
        loss_r = log_weighted_sum(loss_r, r_pi)
        loss_r = torch.mean(loss_r)

        # --- (theta, phi) angle loss (wrap phi to nearest mu) ---
        phi_rep = phi.repeat([1, 1, self.m_angle])
        phi_w = _wrap_to_mu(phi_rep, angle_mu2)

        loss_angle1 = normal_2d(theta[:, 2:], phi_w[:, 2:], angle_mu1[:, 2:], angle_mu2[:, 2:],
                                angle_sigma1[:, 2:], angle_sigma2[:, 2:], angle_corr[:, 2:])
        loss_angle1 = log_weighted_sum(loss_angle1, angle_pi[:, 2:])
        # keep your indexing convention for the second term
        loss_angle2 = normal_1d(phi_w[:, 1], angle_mu2[:, 1], angle_sigma2[:, 1])
        loss_angle2 = log_weighted_sum(loss_angle2, angle_pi[:, 1])
        loss_angle = torch.mean(loss_angle1) + torch.mean(loss_angle2)

        # --- Ramachandran loss alignment (FIX) ---
        # coords targets are length L-1; align Rama to the same by dropping the first position
        if rama is not None:
            rama = rama[:, 1:, :]  # (N, L-1, 2)
        if rama_mask is not None:
            rama_mask = rama_mask[:, 1:]  # (N, L-1)

        # --- Rama loss (skip for pre-rama checkpoints: m_rama == 0) ---
        if rama is not None and self.m_rama > 0:
            phi_ram = rama[:, :, 0:1]
            psi_ram = rama[:, :, 1:2]
            phi_rep_r = phi_ram.repeat([1, 1, self.m_rama])
            psi_rep_r = psi_ram.repeat([1, 1, self.m_rama])

            phi_wr = _wrap_to_mu(phi_rep_r, rama_mu_phi)
            psi_wr = _wrap_to_mu(psi_rep_r, rama_mu_psi)

            rama_pdf = normal_2d(phi_wr, psi_wr, rama_mu_phi, rama_mu_psi,
                                 rama_sigma_phi, rama_sigma_psi, rama_corr)
            rama_nll = -torch.log(torch.sum(rama_pdf * rama_pi, dim=-1) + 1e-9)  # (N, L-1)

            if rama_mask is not None:
                rama_nll = rama_nll * rama_mask
                denom = torch.clamp(rama_mask.sum(dim=-1), min=1.0)
                rama_nll = rama_nll.sum(dim=-1) / denom
                loss_rama = rama_nll.mean()
            else:
                loss_rama = rama_nll.mean()
        else:
            loss_rama = torch.tensor(0.0, device=out.device)

        return loss_r, loss_angle, loss_rama

