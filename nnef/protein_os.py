import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:  # pragma: no cover - optional dev dependency
    class SummaryWriter:  # type: ignore
        def __init__(self, *a, **kw):
            pass

        def add_scalar(self, *a, **kw):
            pass

        def close(self):
            pass

from model.local_ss import LocalEnergyCE
import pandas as pd

from paths import data_path


def _topk_neighbor_g_local_mps_safe(
    distance: torch.Tensor,
    g_others: torch.Tensor,
    g_seg: torch.Tensor,
    k: int,
    num: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute ordered neighbor indices for get_local_struct on CPU, then move to device.

    PyTorch MPS can raise SIGBUS on int64 ``sort`` / ``argsort`` (see warnings around
    top-k neighbor reordering). CPU indexing matches CUDA/CPU numerically for these ops.
    """
    d_cpu = distance.detach().cpu()
    go_cpu = g_others.detach().cpu()
    gs_cpu = g_seg.detach().cpu()
    topk_dist, topk_arg = torch.topk(d_cpu, k, dim=1, largest=False, sorted=True)
    g_topk = torch.gather(go_cpu, dim=1, index=topk_arg)
    g_topk_sorted, g_topk_sort_arg = torch.sort(g_topk, dim=-1)
    dist_topk_sorted = torch.gather(topk_dist, dim=1, index=g_topk_sort_arg)
    g_topk_diff = g_topk_sorted[:, None, :] - g_topk_sorted[:, :, None]
    g_mask_ref = torch.arange(k, dtype=torch.long)
    g_mask_ref = g_mask_ref - g_mask_ref[:, None]
    g_mask = torch.zeros(g_topk_sorted.size(0), k, k)
    g_mask[g_topk_diff == g_mask_ref] = 1
    dist_masked = dist_topk_sorted[:, None, :] * g_mask
    dist_topk_mean = torch.sum(dist_masked, dim=-1) / torch.sum(g_mask, dim=-1)
    dist_gnum_topk = (dist_topk_mean * 10 ** 6).long() * num + g_topk_sorted
    dist_gnum_topk_arg = torch.argsort(dist_gnum_topk, dim=-1)
    gnum_dist_topk_sorted = torch.gather(g_topk_sorted, dim=1, index=dist_gnum_topk_arg)
    return torch.cat((gs_cpu, gnum_dist_topk_sorted), dim=1).to(device)


class EnergyFun(nn.Module):
    def __init__(self, model, args, residue_sum=False, return_loss_terms=False):
        super().__init__()
        self.energy_fn = LocalEnergyCE(model, args)

        # energy_fn returns the mean loss of batch. if residue_sum is True, use the sum of loss in the batch.
        self.residue_sum = residue_sum
        self.return_loss_terms = return_loss_terms
        self.debug = args.debug
        if self.debug:
            self.writer = SummaryWriter('runs/fold/')
            self.counter = 0

    def forward(self, seq, coords, start_id, res_counts,
                rama=None, rama_mask=None,
                esm=None, coords_cart=None, seq_offset=None,
                dihedral=None):
        # LocalEnergyCE.forward returns 6 values after the Rama addition
        # (loss_r, loss_angle, loss_seq, loss_start_id, loss_res_counts, loss_rama);
        # older checkpoints / variants may still return 5. Handle both.
        out = self.energy_fn.forward(
            seq, coords, start_id, res_counts,
            rama=rama, rama_mask=rama_mask,
            esm=esm, coords_cart=coords_cart, seq_offset=seq_offset,
            dihedral=dihedral,
        )
        if len(out) == 6:
            loss_r, loss_angle, loss_profile, loss_start_id, loss_res_counts, loss_rama = out
        elif len(out) == 5:
            loss_r, loss_angle, loss_profile, loss_start_id, loss_res_counts = out
            loss_rama = torch.tensor(0.0, device=seq.device)
        else:
            raise ValueError(f"Unexpected number of loss terms from energy_fn: {len(out)}")

        energy = loss_r + loss_angle + loss_profile + loss_start_id + loss_res_counts + loss_rama

        if self.debug:
            self.writer.add_scalar('profile_loss', loss_profile.item(), self.counter)
            self.writer.add_scalar('coords_radius_loss', loss_r.item(), self.counter)
            self.writer.add_scalar('coords_angle_loss', loss_angle.item(), self.counter)
            self.writer.add_scalar('coords_rama_loss', loss_rama.item(), self.counter)
            self.writer.add_scalar('start_id_loss', loss_start_id.item(), self.counter)
            self.writer.add_scalar('res_counts_loss', loss_res_counts.item(), self.counter)
            self.writer.add_scalar('total_loss', energy.item(), self.counter)
            self.counter += 1

        if self.residue_sum:
            energy = energy * coords.size(0)

        if self.return_loss_terms:
            return energy, loss_r, loss_angle, loss_profile, loss_start_id, loss_res_counts, loss_rama
        else:
            return energy


class ProteinBase:
    k = 10  # besides the 5 residues in the central segment, use another k nearest neighbors
    use_graph_net = False
    ref_scale = 400.0
    use_ref = False
    energy_model_type = None
    # When True and N/CA/C are supplied on Protein, get_local_struct matches
    # ``data_prep_scripts.local_extractor_v2`` (same frame, reorder, start_id).
    # Set False via ``--legacy_local_frame`` for old Yang-style rotation only.
    use_local_frame_v2 = True
    struct_v2_dist_cutoff = 20.0
    # Inference-side feature-extraction switches. Mirror the training-side
    # flags on options.py; set by utils.test_setup. Default to False so any
    # code path that predates these additions stays bit-identical.
    use_cart_coords = False
    use_seq_offset = False
    seq_offset_max = 64
    use_esm = False
    use_dihedral = False

    def __init__(self):
        df = pd.read_csv(data_path('aa_freq_alpha-beta-train.csv'))
        self.energy_ref = torch.tensor(df['e_ref'].values, dtype=torch.float)


class Protein(ProteinBase):
    """
    A protein and its energies.
    This class compute the energy of a protein chain.
    group_num, coords, seq_feature should have the same first dimension.
    The protein chain should NOT have missing residues.
    In this case, group_num is from 1 to coords.shape[0].
    The nearest 4 residues in sequence for g are simply g-2, g-1, g+1, g+2.
    Internal coordinate system: C is at origin; A, B are the previous two residues.
    C-B along negative x. C-B-A in x-y plane.
    """

    def __init__(self, seq, coords, profile,
                 esm_full=None, dihedral_full=None, protein_id=None,
                 n_coords=None, ca_coords=None, c_coords=None,
                 chain_group_num=None):
        super().__init__()
        self.seq = seq  # (N,)
        self.coords = coords  # (N, 3) — Cβ (or CA) bead used as ``CB`` in v2
        self.coords_int = None
        self.profile = profile  # (N, E) for evolution profile, (N,) for residues
        # Optional full backbone for v2 local frames (N, CA, C). When all three
        # are set and use_local_frame_v2 is True, get_local_struct matches
        # ``local_extractor_v2`` training blocks.
        self.n_coords = n_coords
        self.ca_coords = ca_coords
        self.c_coords = c_coords
        # Bead ``group_num`` column (PDB residue ids), shape (N,). If None,
        # chain indices 0..N-1 are assumed for terminal/interior rules.
        self.chain_group_num = chain_group_num
        # Optional per-residue ESM embedding for the full chain, shape
        # (N, d_esm_in). Stored as-is (float16 or float32); `_gather_esm_local`
        # casts to coords.dtype at gather time. Left as None when --use_esm
        # is off so the energy path stays bit-identical.
        if esm_full is not None:
            assert esm_full.shape[0] == coords.shape[0], (
                f'esm_full length {esm_full.shape[0]} does not match coords '
                f'length {coords.shape[0]} for chain {protein_id!r}'
            )
            esm_full = esm_full.to(device=coords.device)
        self.esm_full = esm_full
        # Optional per-residue backbone dihedrals for the full chain, shape
        # (N, 2) in radians: column 0 = phi, column 1 = psi. NaNs at chain
        # termini (phi[0], psi[N-1]) are tolerated and zero-masked at gather
        # time; callers typically precompute this from N/CA/C atoms in decoy
        # bead CSVs (see utils._compute_phi_psi_from_backbone).
        if dihedral_full is not None:
            assert dihedral_full.shape == (coords.shape[0], 2), (
                f'dihedral_full shape {tuple(dihedral_full.shape)} does not '
                f'match (coords.shape[0]={coords.shape[0]}, 2) for '
                f'chain {protein_id!r}'
            )
            dihedral_full = dihedral_full.to(device=coords.device)
        self.dihedral_full = dihedral_full
        # reference energy of unfold state
        self.energy_ref = self.energy_ref.to(self.coords.device)
        self.energy_seq = torch.mean(self.energy_ref[profile]) * self.ref_scale
        self.protein_id = protein_id
        if n_coords is not None and ca_coords is not None and c_coords is not None:
            assert n_coords.shape == coords.shape, 'n_coords shape mismatch'
            assert ca_coords.shape == coords.shape, 'ca_coords shape mismatch'
            assert c_coords.shape == coords.shape, 'c_coords shape mismatch'
        if chain_group_num is not None:
            assert chain_group_num.shape[0] == coords.shape[0], (
                'chain_group_num length mismatch')

    def _has_v2_backbone(self):
        return (
            self.n_coords is not None
            and self.ca_coords is not None
            and self.c_coords is not None
        )

    def update_coords(self, coords):
        self.coords = coords

    def update_profile(self, profile):
        self.profile = profile
        self.energy_seq = torch.mean(self.energy_ref[profile]) * self.ref_scale

    def update_coords_internal(self, coords_int):
        self.coords_int = coords_int

    def update_cartesian_from_internal(self):
        coords_ref = self.coords[:3].detach()
        self.coords = self.internal_to_cartesian(coords_ref, self.coords_int)

    def update_internal_from_cartesian(self):
        self.coords_int = self.cartesian_to_internal(self.coords)

    @staticmethod
    def _get_internal_unit_vectors(c1, c2):
        # c1 is minus X-axis, c1 c2 are X-Y plane.
        z = torch.cross(c1, c2, dim=-1)
        x = -c1
        y = torch.cross(z, x, dim=-1)
        x = x / torch.norm(x, dim=-1, keepdim=True)
        y = y / torch.norm(y, dim=-1, keepdim=True)
        z = z / torch.norm(z, dim=-1, keepdim=True)
        return x, y, z

    def cartesian_to_internal(self, coords):
        # convert cartesian to internal,
        # coords -- (N, 3), internal (N-3, 3)
        c0 = coords[2:-1, :]  # c residue is the (0, 0, 0) of internal coordinate
        c1 = coords[1:-2, :] - c0  # c-1 residue
        c2 = coords[0:-3, :] - c0  # c-2 residue
        c_next = coords[3:, :] - c0  # c+1 residue
        x, y, z = self._get_internal_unit_vectors(c1, c2)
        R = torch.cat([x, y, z], dim=-1).view(-1, 3, 3)  # (N, 3, 3)
        c_next = torch.matmul(R, c_next[:, :, None]).squeeze(dim=-1)
        # theta is the angle to X-axis, phi is the angle in Y-Z plane
        r = torch.norm(c_next, dim=-1)
        theta = torch.acos(c_next[:, 0] / r)
        phi = torch.atan2(c_next[:, 2], c_next[:, 1])  # atan2 considers the quadrant
        coords_int = torch.stack((r, theta, phi), dim=1)
        return coords_int

    def cartesian_to_c_next(self, coords):
        # convert cartesian to c_next in internal coordinate system,
        # coords -- (N, 3), c_next (N-3, 3)
        c0 = coords[2:-1, :]  # c residue is the (0, 0, 0) of internal coordinate
        c1 = coords[1:-2, :] - c0  # c-1 residue
        c2 = coords[0:-3, :] - c0  # c-2 residue
        c_next = coords[3:, :] - c0  # c+1 residue
        x, y, z = self._get_internal_unit_vectors(c1, c2)
        R = torch.cat([x, y, z], dim=-1).view(-1, 3, 3)  # (N, 3, 3)
        c_next = torch.matmul(R, c_next[:, :, None]).squeeze(dim=-1)
        return c_next

    def internal_to_cartesian(self, coords_ref, coords_int):
        # convert internal to cartesian,
        # coords_ref: (3, 3), using coordinates of the first 3 residues as reference.
        N = coords_int.shape[0] + 3
        coords = torch.zeros((N, 3), dtype=coords_int.dtype, device=coords_int.device)
        coords[0:3] = coords_ref
        for i in range(2, N - 1):
            # calculate the unit vectors (x, y, z) of internal coordinate system i
            c1 = coords[i - 1] - coords[i]
            c2 = coords[i - 2] - coords[i]
            x, y, z = self._get_internal_unit_vectors(c1, c2)
            # rotate the internal coordinates of the next point to the reference coordinate system
            r, theta, phi = coords_int[i - 2]
            cos_angle, sin_angle = torch.cos(theta), torch.sin(theta)
            cos_torsion, sin_torsion = torch.cos(phi), torch.sin(phi)
            c_next = r * (cos_angle * x +
                          cos_torsion * sin_angle * y +
                          sin_torsion * sin_angle * z)
            coords[i + 1] = c_next + coords[i]
        return coords

    @staticmethod
    def get_jacobian_dxdz(coords):
        """
        dx/dr: Increasing or decreasing the bond length r extends or retracts all downstream coordinates
        along the bonds axis ur;
        dx/d_theta: moving a bond angle theta drives circular motion of all downstream coordinates
        around the normal vector n_theta;
        dx/d_phi: moving a torsion angle phi drives circular motion of all downstream coordinates around vector ux.
        """
        N = coords.size(0)
        ur = coords[3:] - coords[2:-1]  # (N-3, 3)
        ur = ur / torch.norm(ur, dim=-1, keepdim=True)

        ux = coords[2:-1] - coords[1:-2]  # (N-3, 3)
        ux = ux / torch.norm(ux, dim=-1, keepdim=True)

        n_theta = torch.cross(ux, ur, dim=-1)  # (N-3, 3)
        n_theta = n_theta / torch.norm(n_theta, dim=-1, keepdim=True)

        # each xi (dim0) is subtracted by all xj (dim1).
        xij = coords[None, 3:, :] - coords[2:-1, None, :]   # (N-3, N-3, 3)
        # each xi only move its downstream xj, so only keep the upper right triangle of the matrix.
        mask = torch.triu(torch.ones((N - 3, N - 3), device=coords.device))[:, :, None]

        dxdr = mask * ur[:, None, :].expand((-1, N - 3, -1))  # (N-3, N-3, 3)
        dxdtheta = mask * torch.cross(n_theta[:, None, :].expand((-1, N-3, -1)), xij, dim=-1)  # (N-3, N-3, 3)
        dxdphi = mask * torch.cross(ux[:, None, :].expand((-1, N-3, -1)), xij, dim=-1)  # (N-3, N-3, 3)

        return dxdr, dxdtheta, dxdphi

    @staticmethod
    def multiply_jacobian_dz(dz, dxdr, dxdtheta, dxdphi):
        """
        dx = Jacobian dxdz * dz,  (N-3, 3)
        dz (N-3, 3),
        dxdr, dxdtheta, dxdphi, (N-3, N-3, 3)
        """
        dr = dz[:, None, 0:1]  # (N-3, 1, 1)
        dtheta = dz[:, None, 1:2]  # (N-3, 1, 1)
        dphi = dz[:, None, 2:3]  # (N-3, 1, 1)

        # for each bead j, sum the moves caused by the dr / dtheta / dphi of all previous bead i.
        dx_r = torch.sum(dr * dxdr, dim=0)  # (N-3, 3)
        dx_theta = torch.sum(dtheta * dxdtheta, dim=0)  # (N-3, 3)
        dx_phi = torch.sum(dphi * dxdphi, dim=0)  # (N-3, 3)

        dx = dx_r + dx_theta + dx_phi
        return dx

    def get_dx_from_dz(self, coords, dz):
        """
        coords: (N, 3); dz: (N-3, 3)
        use the improved Euler method to calculate dx.
        x_p = x_t + dx_t/dz * dz
        dx = 0.5 * (dx_t/dz + dx_p/dz) * dz
        """
        dxdr, dxdtheta, dxdphi = self.get_jacobian_dxdz(coords)
        dx_1 = self.multiply_jacobian_dz(dz, dxdr, dxdtheta, dxdphi)
        coords_p = coords.clone()
        coords_p[3:] = coords_p[3:] + dx_1
        dxdr_p, dxdtheta_p, dxdphi_p = self.get_jacobian_dxdz(coords_p)
        dx_2 = self.multiply_jacobian_dz(dz, dxdr_p, dxdtheta_p, dxdphi_p)
        dx = 0.5 * (dx_1 + dx_2)  # (N-3, 3)
        # add 3 rows of zeros, (N-3, 3) -> (N, 3)
        head_zeros = torch.zeros((3, 3), device=dx.device)
        dx = torch.cat([head_zeros, dx], dim=0)
        return dx

    def get_gradz_from_gradx(self, coords, gradx):
        """
        coords: (N, 3); gradx: (N, 3)
        gradz: (N-3, 3)
        Each r / theta / phi of residue i will change the x, y, z of all following residues,
        so in back-propagation, grad_i should sum gradients from the x-y-z of grad_j (j>i).
        """
        dxdr, dxdtheta, dxdphi = self.get_jacobian_dxdz(coords)
        gradx = gradx[3:]
        grad_r = torch.sum(gradx[None, :, :] * dxdr, dim=(1, 2))  # (1, N-3, 3) * (N-3, N-3, 3) -> (N-3)
        grad_theta = torch.sum(gradx[None, :, :] * dxdtheta, dim=(1, 2))  # (1, N-3, 3) * (N-3, N-3, 3) -> (N-3)
        grad_phi = torch.sum(gradx[None, :, :] * dxdphi, dim=(1, 2))  # (1, N-3, 3) * (N-3, N-3, 3) -> (N-3)
        gradz = torch.stack([grad_r, grad_theta, grad_phi], dim=1)
        return gradz

    @staticmethod
    def get_distmap(coords):
        # coords: (N, 3); distmap: (N, N)
        x_map = coords[:, None, :] - coords[None, :, :]
        distmap = torch.norm(x_map, dim=-1)
        # distmap = torch.sqrt(torch.sum((coords[:, None, :] - coords[None, :, :])**2, dim=-1))
        return distmap

    @staticmethod
    def get_rad_gyration(coords):
        x_map = coords[:, None, :] - coords[None, :, :]
        distmap = torch.norm(x_map, dim=-1)
        rg2 = torch.mean(distmap**2)
        diag_idx = torch.arange(coords.shape[0])
        distmap[diag_idx, diag_idx] = 1.0
        idx = (distmap < 1.0)
        n = torch.sum(idx).item()
        collision = False
        if n > 0:
            collision = True
        return rg2, collision

    @staticmethod
    def check_coords(coords):
        # check if any two beads are too close
        x_map = coords[:, None, :] - coords[None, :, :]
        distmap = torch.norm(x_map, dim=-1)
        diag_idx = torch.arange(coords.shape[0])
        distmap[diag_idx, diag_idx] = 1.0
        idx = (distmap < 1e-1)
        n = torch.sum(idx).item()
        if n > 0:
            print(f'{n} beads pairs are too close.')
            print(distmap[idx])
        # check if any 3 adjacent beads are in a line
        c1 = coords[1:-1, :] - coords[:-2, :]
        c2 = coords[2:, :] - coords[1:-1, :]
        z = torch.cross(c1, c2, dim=-1)
        # check if c1 and c2 are along the same line
        z_norm = torch.norm(z, dim=1)
        n2 = z_norm[z_norm < 1e-6].shape[0]
        if n2 > 0:
            print(f'{n} beads triples are co-line.')

    @staticmethod
    def _local_rotation_matrix(c1, c2):
        # c1, c2 size (N, 3)
        z = torch.cross(c1, c2, dim=-1)
        # check if c1 and c2 are along the same line
        z_norm = torch.norm(z, dim=1)
        idx = (z_norm < 1e-6)
        if c2[idx].shape[0] > 0:
            c2[idx] = c2[idx] + 1e-4
        x = c1
        y = torch.cross(z, x, dim=-1)
        x = x / torch.norm(x, dim=1, keepdim=True)
        y = y / torch.norm(y, dim=1, keepdim=True)
        z = z / torch.norm(z, dim=1, keepdim=True)
        R = torch.cat([x, y, z], dim=-1).view(-1, 3, 3)  # (N, 3, 3)
        return R

    def get_local_struct(self):
        """Local blocks for energy evaluation.

        When ``ProteinBase.use_local_frame_v2`` and N/CA/C tensors are present,
        blocks match ``local_extractor_v2`` / ``hhsuite_CB_v2.h5`` training
        (intra-residue N–CA–C frame, Cβ coordinates, ``_re_order_block``,
        ``start_id`` from ``seg``). Otherwise falls back to the historical
        neighbor-bead rotation (or if the chain is shorter than one
        block = ``5 + k`` residues).
        """
        if (ProteinBase.use_local_frame_v2 and self._has_v2_backbone()
                and self.coords.size(0) >= (5 + self.k)):
            return self._get_local_struct_v2()
        return self._get_local_struct_legacy()

    def _get_local_struct_v2(self):
        """Mirror ``extract_blocks_v2`` + ``_re_order_block`` on the inference chain."""
        try:
            from data_prep_scripts.local_extractor_v2 import (
                compute_local_frame,
                _re_order_block,
            )
        except ImportError:  # pragma: no cover
            from nnef.data_prep_scripts.local_extractor_v2 import (  # type: ignore
                compute_local_frame,
                _re_order_block,
            )

        device = self.coords.device
        dtype = self.coords.dtype
        N = self.coords.size(0)
        k = self.k
        block_size = 5 + k
        CB = self.coords
        CA = self.ca_coords
        N_xyz = self.n_coords
        C_xyz = self.c_coords
        cutoff = float(ProteinBase.struct_v2_dist_cutoff)

        if self.chain_group_num is not None:
            ids_list = [int(x) for x in self.chain_group_num.detach().cpu().tolist()]
        else:
            ids_list = list(range(N))
        id_set = set(ids_list)
        id_to_row = {gid: i for i, gid in enumerate(ids_list)}
        sorted_ids = sorted(id_set)
        first_two = set(sorted_ids[:2])
        last_two = set(sorted_ids[-2:])

        def row_to_gid(row: int) -> int:
            return ids_list[row]

        prof_rows, coord_rows, sid_rows, rc_rows, gl_rows = [], [], [], [], []

        for cen_idx in range(N):
            gc = ids_list[cen_idx]
            n_np = N_xyz[cen_idx].detach().cpu().numpy().astype(np.float64)
            ca_np = CA[cen_idx].detach().cpu().numpy().astype(np.float64)
            c_np = C_xyz[cen_idx].detach().cpu().numpy().astype(np.float64)
            R_np = compute_local_frame(n_np, ca_np, c_np)
            if R_np is None:
                continue
            R = torch.tensor(R_np, device=device, dtype=dtype)

            CA_cen = CA[cen_idx]
            CB_cen = CB[cen_idx]
            dist_glob = torch.norm(CB - CB_cen, dim=-1)

            is_terminal = (gc in first_two) or (gc in last_two)

            if is_terminal:
                idx_combined = torch.argsort(dist_glob)[:block_size]
                if idx_combined.numel() < block_size:
                    continue
                mask_others = torch.ones(N, dtype=torch.bool, device=device)
                mask_others[idx_combined] = False
                segment_info = np.full(block_size, 5, dtype=np.int32)
            else:
                surr = [gc - 2, gc - 1, gc, gc + 1, gc + 2]
                if not all(s in id_to_row for s in surr):
                    continue
                idx_window = torch.tensor(
                    [id_to_row[s] for s in surr],
                    device=device, dtype=torch.long,
                )
                window_mask = torch.zeros(N, dtype=torch.bool, device=device)
                window_mask[idx_window] = True
                if int(window_mask.sum().item()) != 5:
                    continue
                # Match extract_blocks_v2: forbid re-picking peptide-window rows in top-k
                # (``np.where(mask_others, dist_glob, np.inf)`` with mask_others = ~window).
                dist_sort = dist_glob.clone()
                dist_sort[window_mask] = float('inf')
                topk_idx = torch.argsort(dist_sort)[:k]
                idx_combined = torch.cat([idx_window, topk_idx])
                mask_others = ~window_mask
                segment_info = np.full(block_size, 5, dtype=np.int32)
                for j in range(block_size):
                    gv = row_to_gid(int(idx_combined[j].item()))
                    if gv == gc:
                        segment_info[j] = 0
                    elif gv == gc - 1:
                        segment_info[j] = 1
                    elif gv == gc + 1:
                        segment_info[j] = 2
                    elif gv == gc - 2:
                        segment_info[j] = 3
                    elif gv == gc + 2:
                        segment_info[j] = 4

            diff = CB[idx_combined] - CA_cen
            cb_local = (R @ diff.T).T
            cb_center_local = R @ (CB_cen - CA_cen)
            dist_local = torch.norm(cb_local - cb_center_local.unsqueeze(0), dim=-1)

            dist_others = dist_glob[mask_others]
            c8 = int((dist_others < 8.0).sum().item())
            c10 = int((dist_others < 10.0).sum().item())
            c12 = int((dist_others < 12.0).sum().item())

            if float(dist_local.max().item()) >= cutoff:
                continue

            gnums_df = np.array(
                [row_to_gid(int(idx_combined[j].item())) for j in range(block_size)],
                dtype=np.int64,
            )
            df_blk = pd.DataFrame({
                'segment': segment_info,
                'group_num': gnums_df,
                'local_x': cb_local[:, 0].detach().cpu().numpy(),
                'local_y': cb_local[:, 1].detach().cpu().numpy(),
                'local_z': cb_local[:, 2].detach().cpu().numpy(),
                'distance': dist_local.detach().cpu().numpy(),
                'count8a': c8,
                'count10a': c10,
                'count12a': c12,
            })
            df_o = _re_order_block(df_blk)
            seg_o = df_o['seg'].values.astype(np.int8)
            start_id_np = np.zeros_like(seg_o, dtype=np.int8)
            start_id_np[1:] = (seg_o[1:] == seg_o[:-1]).astype(np.int8)

            g_row = torch.tensor(
                [id_to_row[int(gv)] for gv in df_o['group_num'].values],
                device=device, dtype=torch.long,
            )
            cb_o = torch.tensor(
                df_o[['local_x', 'local_y', 'local_z']].values,
                device=device, dtype=dtype,
            )
            rc = torch.tensor(
                df_o[['count8a', 'count10a', 'count12a']].values[0],
                device=device, dtype=dtype,
            )

            prof_rows.append(self.profile[g_row])
            coord_rows.append(cb_o)
            sid_rows.append(torch.tensor(start_id_np, device=device, dtype=torch.long))
            rc_rows.append(rc)
            gl_rows.append(g_row)

        if not prof_rows:
            return self._get_local_struct_legacy()

        profile_local = torch.stack(prof_rows, dim=0)
        coords_local = torch.stack(coord_rows, dim=0)
        start_id = torch.stack(sid_rows, dim=0)
        res_counts = torch.stack(rc_rows, dim=0)
        g_local = torch.stack(gl_rows, dim=0)
        return profile_local, coords_local, start_id, res_counts, g_local

    def _get_local_struct_legacy(self):
        group_coords = self.coords
        group_profile = self.profile
        # done in parallel for all residues. group_coords: (N, 3), group_profile: (N, E)
        device = group_coords.device
        num = group_coords.size(0)
        group_num = torch.arange(num, device=device)
        gc = group_num[2:-2]   # (num-4)
        g1 = group_num[1:-3]
        g2 = group_num[3:-1]
        g3 = group_num[:-4]
        g4 = group_num[4:]

        # index of the central segment
        g_seg = torch.stack((gc, g1, g2, g3, g4)).transpose(0, 1)  # (N-4, 5)
        # get group numbers EXCLUDE the central segment
        g_others = torch.arange(num, device=device).repeat(1, num-4)
        g_others = F.pad(g_others, (0, num-4)).view(num-4, num+1)[:, 5:].flatten()
        g_others = g_others[:-(num-4)].view(num-4, num-5)   # (N-4, N-5)

        coords_others = group_coords[g_others] - group_coords[gc][:, None, :]  # (N-4, N-5, 3)
        distance = torch.norm(coords_others, dim=-1)  # (N-4, N-5)

        # get number of residues within 8A, 10A, 12A
        count_8a = (distance < 8).sum(dim=-1)  # (N-4,)
        count_10a = (distance < 10).sum(dim=-1)
        count_12a = (distance < 12).sum(dim=-1)
        res_counts = torch.cat([count_8a[:, None], count_10a[:, None], count_12a[:, None]], dim=-1)  # (N-4, 3)

        if self.k > 0:
            if device.type == 'mps':
                g_local = _topk_neighbor_g_local_mps_safe(
                    distance, g_others, g_seg, self.k, num, device)
            else:
                # calculate the index of the nearest k neighbors
                topk_dist, topk_arg = torch.topk(distance, self.k, dim=1, largest=False, sorted=True)  # topk_arg size (N-4, k)
                g_topk = torch.gather(g_others, dim=1, index=topk_arg)  # (N-4, k)

                # re-order the topk index, g_topk and topk_dist is the group num and dist of the topk residues.
                g_topk_sorted, g_topk_sort_arg = torch.sort(g_topk, dim=-1)  # sort by group number
                # reorder dist_topk, so dist_topk_sorted is the distance sorted by group number
                dist_topk_sorted = torch.gather(topk_dist, dim=1, index=g_topk_sort_arg)
                # make a mask for each segment g_mask (N-4, k, k)
                g_topk_diff = g_topk_sorted[:, None, :] - g_topk_sorted[:, :, None]
                g_mask_ref = torch.arange(self.k, device=device)
                g_mask_ref = g_mask_ref - g_mask_ref[:, None]
                g_mask = torch.zeros(g_topk_sorted.size(0), self.k, self.k, device=device)
                g_mask[g_topk_diff == g_mask_ref] = 1
                # calculate the mean distance of each segment
                dist_masked = dist_topk_sorted[:, None, :] * g_mask
                dist_topk_mean = torch.sum(dist_masked, dim=-1) / torch.sum(g_mask, dim=-1)
                # reorder the groups by the segment distance and group number.
                dist_gnum_topk = (dist_topk_mean * 10 ** 6).long() * num + g_topk_sorted
                dist_gnum_topk_arg = torch.argsort(dist_gnum_topk, dim=-1)
                gnum_dist_topk_sorted = torch.gather(g_topk_sorted, dim=1, index=dist_gnum_topk_arg)

                # concat the index of the central segment and the k neighbors
                # g_local = torch.cat((g_seg, g_topk), dim=1)  # (N-4, 5+k)
                g_local = torch.cat((g_seg, gnum_dist_topk_sorted), dim=1)  # (N-4, 5+k)
        else:
            g_local = g_seg
        # extract the profile and coordinates using the index
        profile_local = group_profile[g_local]  # (N-4, 5+k, E)
        coords_local = group_coords[g_local] - group_coords[gc][:, None, :]  # (N-4, 5+k, 3)

        # rotate the local coordinates
        rotate_mat = self._local_rotation_matrix(coords_local[:, 1, :], coords_local[:, 2, :])
        # (N-4, 1, 3, 3) * (N-4, 5+k, 3, 1) -> (N-4, 5+k, 3, 1)
        coords_local = torch.matmul(rotate_mat[:, None, :, :], coords_local[:, :, :, None]).squeeze()

        start_id = torch.zeros_like(g_local)
        start_id[:, 1:5] = 1
        if self.k > 1:
            start_id[:, 6:][g_local[:, 6:] - g_local[:, 5:-1] == 1] = 1

        res_counts = res_counts.to(torch.float)

        # NOTE: returns g_local as a 5th element so get_energy / get_residue_energy
        # can derive the signed chain-distance feature without recomputing.
        # Only internal callers (same file) use this, so the extra element is
        # an additive, non-breaking change for this class.
        return profile_local, coords_local, start_id, res_counts, g_local

    @staticmethod
    def _local_cartesian_to_radian(coords_local):
        # r = torch.norm(coords_local, dim=-1)  # (N-4, 5+k)
        r = torch.norm(coords_local[:, 1:], dim=-1)  # (N-4, 5+k-1)  # exclude the origin
        zr = coords_local[:, 1:, 2] / r
        zr = torch.clamp(zr, min=-1.0 + 1e-6, max=1.0 - 1e-6)  # otherwise, acos(1.00001) results NaN
        theta = torch.acos(zr)  # exclude the origin
        # theta = torch.acos(coords_local[:, 1:, 2] / r[:, 1:])  # exclude the origin
        phi = torch.atan2(coords_local[:, 1:, 1], coords_local[:, 1:, 0])  # atan2 considers the quadrant,
        r = F.pad(r, (1, 0), value=0)
        theta = F.pad(theta, (1, 0), value=0)
        phi = F.pad(phi, (1, 0), value=0)
        coords = torch.stack((r, theta, phi), dim=2)  # (N-4, 5+k, 3)
        return coords

    def _build_extras(self, coords_local_cart, g_local):
        """Derive the optional inference-side extras that mirror the
        training-side dataset outputs in data_chimeric. Returns
        (coords_cart, seq_offset, esm_local, dihedral_local); any of the four
        may be None when its flag is off or the corresponding payload is not
        available.
        """
        coords_cart = coords_local_cart.clone() if self.use_cart_coords else None
        seq_offset = None
        if self.use_seq_offset:
            M = int(self.seq_offset_max)
            # Central residue sits at index 0 of every block (see get_local_struct).
            offset = g_local - g_local[:, 0:1]
            offset = torch.clamp(offset, min=-M, max=M) + M  # shift to [0, 2M]
            seq_offset = offset.to(torch.long)
        esm_local = self._gather_esm_local(g_local)
        dihedral_local = self._gather_dihedral_local(g_local)
        return coords_cart, seq_offset, esm_local, dihedral_local

    def _gather_esm_local(self, g_local):
        """Gather per-block ESM slices from the full-chain cache.

        Returns a tensor of shape ``(num_blocks, 5+k, d_esm_in)`` cast to the
        protein's coord dtype (float32 in normal use), or ``None`` if
        ``self.use_esm`` is off / no cache was supplied.

        Out-of-bounds indices in ``g_local`` (should not happen in practice
        because g_local is built from 0..N-1 and esm_full is asserted to be
        length N at __init__, but we keep the defensive check to mirror the
        training-side dataset contract) are clamped and the corresponding
        slices zeroed.
        """
        if not self.use_esm or self.esm_full is None:
            return None
        L_chain = self.esm_full.shape[0]
        invalid = (g_local < 0) | (g_local >= L_chain)
        clamped = g_local.clamp(min=0, max=L_chain - 1)
        esm_local = self.esm_full[clamped]
        if invalid.any():
            esm_local = esm_local.clone()
            esm_local[invalid] = 0.0
        return esm_local.to(self.coords.dtype)

    def _gather_dihedral_local(self, g_local):
        """Gather per-block backbone dihedrals, encoded as sin/cos 4-d.

        Returns a tensor shaped ``(num_blocks, 5+k, 4)`` = (sin phi, cos phi, sin psi,
        cos psi) cast to coord dtype, or ``None`` when ``use_dihedral`` is off
        or no chain-level dihedral cache was supplied.

        Chain-terminal NaNs (phi[0], psi[L-1]) and out-of-range g_local indices
        are zero-masked in all 4 channels, matching the training-side
        convention where `rama_mask==0` positions are zeroed before the
        additive `linear_x_dihedral` layer sees them.
        """
        if not self.use_dihedral or self.dihedral_full is None:
            return None
        L_chain = self.dihedral_full.shape[0]
        invalid_idx = (g_local < 0) | (g_local >= L_chain)
        clamped = g_local.clamp(min=0, max=L_chain - 1)
        phi_psi = self.dihedral_full[clamped]  # (num_blocks, 5+k, 2)
        nan_mask = torch.isnan(phi_psi).any(dim=-1) | invalid_idx
        phi_psi = torch.nan_to_num(phi_psi, nan=0.0, posinf=0.0, neginf=0.0)
        sin_pp = torch.sin(phi_psi)
        cos_pp = torch.cos(phi_psi)
        dihedral_local = torch.stack(
            [sin_pp[..., 0], cos_pp[..., 0], sin_pp[..., 1], cos_pp[..., 1]],
            dim=-1,
        )
        dihedral_local = dihedral_local * (~nan_mask).unsqueeze(-1).to(dihedral_local.dtype)
        return dihedral_local.to(self.coords.dtype)

    def get_energy(self, energy_fun):
        profile_local, coords_local, start_id, res_counts, g_local = self.get_local_struct()
        coords_cart, seq_offset, esm_local, dihedral_local = self._build_extras(
            coords_local, g_local)
        coords_local = self._local_cartesian_to_radian(coords_local)
        energy = energy_fun.forward(
            profile_local, coords_local, start_id, res_counts,
            coords_cart=coords_cart, seq_offset=seq_offset, esm=esm_local,
            dihedral=dihedral_local,
        )
        if self.use_ref:
            energy = energy - self.energy_seq
        return energy

    def get_residue_energy(self, energy_fun):
        # seq & profile have add up to seq_feature
        profile_local, coords_local, start_id, res_counts, g_local = self.get_local_struct()
        coords_cart, seq_offset, esm_local, dihedral_local = self._build_extras(
            coords_local, g_local)
        coords_local = self._local_cartesian_to_radian(coords_local)
        num = coords_local.size(0)
        residue_energy = np.zeros(num)
        for i in range(num):
            cc_i = coords_cart[i:i+1] if coords_cart is not None else None
            so_i = seq_offset[i:i+1] if seq_offset is not None else None
            em_i = esm_local[i:i+1] if esm_local is not None else None
            dh_i = dihedral_local[i:i+1] if dihedral_local is not None else None
            residue_energy[i] = energy_fun.forward(
                profile_local[i:i+1], coords_local[i:i+1],
                start_id[i:i+1], res_counts[i:i+1],
                coords_cart=cc_i, seq_offset=so_i, esm=em_i, dihedral=dh_i,
            )
        return residue_energy

    def get_local_struct_phy(self):
        # local structures for graph net input
        group_coords = self.coords
        group_profile = self.profile
        # done in parallel for all residues. group_coords: (N, 3), group_profile: (N, E)
        device = group_coords.device
        num = group_coords.size(0)
        group_num = torch.arange(num, device=device)
        gc = group_num[2:-2]   # (num-4)
        g1 = group_num[1:-3]
        g2 = group_num[3:-1]
        g3 = group_num[:-4]
        g4 = group_num[4:]

        # index of the central segment
        g_seg = torch.stack((gc, g1, g2, g3, g4)).transpose(0, 1)  # (N-4, 5)
        # get group numbers EXCLUDE the central segment
        g_others = torch.arange(num, device=device).repeat(1, num-4)
        g_others = F.pad(g_others, (0, num-4)).view(num-4, num+1)[:, 5:].flatten()
        g_others = g_others[:-(num-4)].view(num-4, num-5)   # (N-4, N-5)

        coords_others = group_coords[g_others] - group_coords[gc][:, None, :]  # (N-4, N-5, 3)
        distance = torch.norm(coords_others, dim=-1)  # (N-4, N-5)

        # calculate the index of the nearest k neighbors
        if device.type == 'mps':
            topk_dist, topk_arg = torch.topk(
                distance.detach().cpu(), self.k, dim=1, largest=False, sorted=True)
            g_topk = torch.gather(g_others.detach().cpu(), dim=1, index=topk_arg)
            g_topk_sorted, _ = torch.sort(g_topk, dim=-1)
            g_local = torch.cat((g_seg.detach().cpu(), g_topk_sorted), dim=1).to(device)
        else:
            topk_dist, topk_arg = torch.topk(distance, self.k, dim=1, largest=False, sorted=True)  # topk_arg size (N-4, k)
            g_topk = torch.gather(g_others, dim=1, index=topk_arg)  # (N-4, k)
            g_topk_sorted, _ = torch.sort(g_topk, dim=-1)  # sort by group number
            g_local = torch.cat((g_seg, g_topk_sorted), dim=1)  # (N-4, 5+k)

        profile_local = group_profile[g_local]  # (N-4, 5+k, E)
        coords_local = group_coords[g_local] - group_coords[gc][:, None, :]  # (N-4, 5+k, 3)

        # rotate the local coordinates
        rotate_mat = self._local_rotation_matrix(coords_local[:, 1, :], coords_local[:, 2, :])
        # (N-4, 1, 3, 3) * (N-4, 5+k, 3, 1) -> (N-4, 5+k, 3, 1)
        coords_local = torch.matmul(rotate_mat[:, None, :, :], coords_local[:, :, :, None]).squeeze()

        edge_type = g_local[:, None, :] - g_local[:, :, None]
        idx = (edge_type == 1) | (edge_type == -1) | (edge_type == 0)
        edge_type[~idx] = 2
        idx = (edge_type == 1) | (edge_type == -1)
        edge_type[idx] = 1
        return profile_local, coords_local, edge_type


class ProteinComplex(ProteinBase):
    """
    A protein complex and its energies.
    This class compute the energy of a few protein chains.
    group_num, coords, seq_feature should have the same first dimension.
    Each protein chain should NOT have missing residues.
    For the first chain group_num is from 1 to coords.shape[0].
    The the next chain group_num starts with 10 + group_num of the last residue in previous chain.
    The nearest 4 residues in sequence for g are simply g-2, g-1, g+1, g+2.
    Internal representation is not required for protein complex.
    """
    def __init__(self, chain, seq, coords, profile, protein_id=None):
        super().__init__()
        self.chain = chain  # (N,)  [0, 0, 0, 0, ... 1, 1, 1...]
        self.num_chain = torch.unique(chain).size(0)  # number of chain, eg. 2
        self.seq = seq  # (N,)
        self.coords = coords  # (N, 3)
        self.profile = profile  # (N, E)

        self.protein_id = protein_id
        # self.k = 10

    def update_coords(self, coords):
        self.coords = coords

    def update_profile(self, profile):
        self.profile = profile

    @staticmethod
    def _local_rotation_matrix(c1, c2):
        # c1, c2 size (N, 3)
        z = torch.cross(c1, c2, dim=-1)
        x = c1
        y = torch.cross(z, x, dim=-1)
        x = x / torch.norm(x, dim=1, keepdim=True)
        y = y / torch.norm(y, dim=1, keepdim=True)
        z = z / torch.norm(z, dim=1, keepdim=True)
        R = torch.cat([x, y, z], dim=-1).view(-1, 3, 3)  # (N, 3, 3)
        return R

    def setup_group_num(self, device):
        # num_chain = self.num_chain
        chain = self.chain

        group_num = torch.arange(chain.shape[0], device=device)
        # there is a bug.  The end of one chain and the start of next chain are 'connected' in group_num.
        # It will put them into the same segment if they are all neighbors of one residue.
        # for i in range(num_chain):
        #     idx = (chain == i)
        #     group_num[idx] += i * 10  # add a gap of 10 between chains

        gc_list = []
        g_seg_list = []
        g_others_list = []

        # for i in range(num_chain):
        for i in torch.unique(chain):
            idx = (chain == i)
            group_num_i = group_num[idx]
            # if the chain is shorter than 6 residues, ignore it.
            if len(group_num_i) < 6:
                continue
            gc = group_num_i[2:-2]  # (N_i-4)
            g1 = group_num_i[1:-3]
            g2 = group_num_i[3:-1]
            g3 = group_num_i[:-4]
            g4 = group_num_i[4:]

            gc_list.append(gc)
            # index of the central segment
            g_seg = torch.stack((gc, g1, g2, g3, g4)).transpose(0, 1)  # (N_i-4, 5)
            g_seg_list.append(g_seg)

            # get group numbers EXCLUDE the central segment
            num = group_num[idx].shape[0]
            g_others = torch.arange(num, device=device).repeat(1, num - 4)
            g_others = F.pad(g_others, (0, num - 4)).view(num - 4, num + 1)[:, 5:].flatten()
            g_others = g_others[:-(num - 4)].view(num - 4, num - 5)  # (N_i-4, N_i-5)
            g_others = g_others + group_num_i[0]

            g_other_chain = group_num[~idx].repeat(num - 4, 1)
            g_others = torch.cat((g_others, g_other_chain), dim=1)  # (N_i-4, N-5)
            g_others_list.append(g_others)

        gc = torch.cat(gc_list)
        g_seg = torch.cat(g_seg_list)
        g_others = torch.cat(g_others_list)

        return gc, g_seg, g_others

    def get_local_struct(self):
        group_coords = self.coords
        group_profile = self.profile
        # done in parallel for all residues. group_coords: (N, 3), group_profile: (N, E)
        device = group_coords.device
        num = self.coords.size(0)

        gc, g_seg, g_others = self.setup_group_num(device)

        coords_others = group_coords[g_others] - group_coords[gc][:, None, :]  # (N-4, N-5, 3)
        distance = torch.norm(coords_others, dim=-1)  # (N-4, N-5)

        # get number of residues within 8A, 10A, 12A
        count_8a = (distance < 8).sum(dim=-1)  # (N-4,)
        count_10a = (distance < 10).sum(dim=-1)
        count_12a = (distance < 12).sum(dim=-1)
        res_counts = torch.cat([count_8a[:, None], count_10a[:, None], count_12a[:, None]], dim=-1)  # (N-4, 3)

        # calculate the index of the nearest k neighbors
        if device.type == 'mps':
            g_local = _topk_neighbor_g_local_mps_safe(
                distance, g_others, g_seg, self.k, num, device)
        else:
            topk_dist, topk_arg = torch.topk(distance, self.k, dim=1, largest=False, sorted=True)  # topk_arg size (N-4, k)
            g_topk = torch.gather(g_others, dim=1, index=topk_arg)  # (N-4, k)

            # re-order the topk index, g_topk and topk_dist is the group num and dist of the topk residues.
            g_topk_sorted, g_topk_sort_arg = torch.sort(g_topk, dim=-1)  # sort by group number
            dist_topk_sorted = torch.gather(topk_dist, dim=1, index=g_topk_sort_arg)  # reorder dist_topk

            g_topk_diff = g_topk_sorted[:, None, :] - g_topk_sorted[:, :, None]
            g_mask_ref = torch.arange(self.k, device=device)
            g_mask_ref = g_mask_ref - g_mask_ref[:, None]
            g_mask = torch.zeros(g_topk_sorted.size(0), self.k, self.k, device=device)
            g_mask[g_topk_diff == g_mask_ref] = 1

            dist_masked = dist_topk_sorted[:, None, :] * g_mask
            dist_topk_mean = torch.sum(dist_masked, dim=-1) / torch.sum(g_mask, dim=-1)

            dist_gnum_topk = (dist_topk_mean * 10 ** 6).long() * num + g_topk_sorted
            dist_gnum_topk_arg = torch.argsort(dist_gnum_topk, dim=-1)
            gnum_dist_topk_sorted = torch.gather(g_topk_sorted, dim=1, index=dist_gnum_topk_arg)

            # g_local = torch.cat((g_seg, g_topk), dim=1)  # (N-4, 5+k)
            g_local = torch.cat((g_seg, gnum_dist_topk_sorted), dim=1)  # (N-4, 5+k)

        # extract the profile and coordinates using the index
        profile_local = group_profile[g_local]  # (N-4, 5+k, E)
        coords_local = group_coords[g_local] - group_coords[gc][:, None, :]  # (N-4, 5+k, 3)

        # rotate the local coordinates
        rotate_mat = self._local_rotation_matrix(coords_local[:, 1, :], coords_local[:, 2, :])
        # (N-4, 1, 3, 3) * (N-4, 5+k, 3, 1) -> (N-4, 5+k, 3, 1)
        coords_local = torch.matmul(rotate_mat[:, None, :, :], coords_local[:, :, :, None]).squeeze()

        start_id = torch.zeros_like(g_local)
        start_id[:, 1:5] = 1
        start_id[:, 6:][g_local[:, 6:] - g_local[:, 5:-1] == 1] = 1

        res_counts = res_counts.to(torch.float)

        return profile_local, coords_local, start_id, res_counts

    @staticmethod
    def _local_cartesian_to_radian(coords_local):
        r = torch.norm(coords_local, dim=-1)  # (N-4, 5+k)
        theta = torch.acos(coords_local[:, 1:, 2] / r[:, 1:])  # exclude the origin
        phi = torch.atan2(coords_local[:, 1:, 1], coords_local[:, 1:, 0])  # atan2 considers the quadrant,
        theta = F.pad(theta, (1, 0), value=0)
        phi = F.pad(phi, (1, 0), value=0)
        coords = torch.stack((r, theta, phi), dim=2)  # (N-4, 5+k, 3)
        return coords

    def get_energy(self, energy_fun):
        # seq & profile have add up to seq_feature
        profile_local, coords_local, start_id, res_counts = self.get_local_struct()
        coords_local = self._local_cartesian_to_radian(coords_local)
        energy = energy_fun.forward(profile_local, coords_local, start_id, res_counts)
        return energy

    def get_residue_energy(self, energy_fun):
        # seq & profile have add up to seq_feature
        profile_local, coords_local, start_id, res_counts = self.get_local_struct()
        coords_local = self._local_cartesian_to_radian(coords_local)
        num = coords_local.size(0)
        residue_energy = np.zeros(num)
        for i in range(num):
            residue_energy[i] = energy_fun.forward(profile_local[i:i+1], coords_local[i:i+1],
                                                   start_id[i:i+1], res_counts[i:i+1])
        return residue_energy




