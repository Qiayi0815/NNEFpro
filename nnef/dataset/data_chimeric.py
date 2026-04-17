from torch.utils.data import Dataset  # 从存储在.h5文件中的蛋白质结构数据中读取序列和坐标信息
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm

from paths import data_path


# 处理结构生物学中蛋白质序列和结构的相关数据（并加入 Ramachandran 矩阵）
class DatasetLocalGenCM(Dataset):
    def __init__(self, data, args,
                 pdb_h5_path=None,
                 seq_h5_path=None,
                 rama_h5_path=None,                              # NEW: Ramachandran 文件
                 rama_dataset_name='rama',                       # NEW: 每个PDB组内的键名
                 esm_h5_path=None,                               # NEW: optional per-residue ESM cache
                 esm_dataset_name='esm'
                 ):
        # Resolve default h5 locations lazily so callers can pass absolute paths
        # (e.g. from CLI flags) without this module reaching into the package.
        if pdb_h5_path is None:
            pdb_h5_path = data_path('hhsuite_CB_v2.h5')
        if seq_h5_path is None:
            seq_h5_path = data_path('hhsuite_pdb_seq_v2.h5')
        if rama_h5_path is None:
            rama_h5_path = data_path('hhsuite_rama_v2.h5')
        self.seq_factor = args.seq_factor          # 0.5 与序列相关的参数
        self.seq_len = args.seq_len + 1            # 序列长度（包含起始符号位），模型将考虑的蛋白质片段的长度
        self.noise_factor = args.noise_factor      # 0.001 数据中的噪声因子
        self.seq_type = args.seq_type              # 指定使用的序列类型
        self.residue_type_num = args.residue_type_num  # 残基种类数量（例如20代表标准氨基酸数量）
        self.no_homology = args.no_homology        # 决定是否使用同源序列

        # ---- Optional feature-extraction enhancements (see options.py) ----
        self.use_esm = bool(getattr(args, 'use_esm', False))
        self.use_cart_coords = bool(getattr(args, 'use_cart_coords', False))
        self.use_seq_offset = bool(getattr(args, 'use_seq_offset', False))
        self.use_dihedral = bool(getattr(args, 'use_dihedral', False))
        self.seq_offset_max = int(getattr(args, 'seq_offset_max', 64))
        self.esm_dim_in = int(getattr(args, 'esm_dim_in', 1280))
        # Any extra feature => we return the 10-tuple extended form. When none
        # are enabled we keep returning the original 6-tuple so downstream
        # collate and trainer behavior are bit-identical to baseline.
        self._has_extras = (
            self.use_esm or self.use_cart_coords
            or self.use_seq_offset or self.use_dihedral
        )

        if self.residue_type_num != 20:
            # 残基类型转换
            aa_types = pd.read_csv(data_path('aa_types.csv'))
            assert (self.residue_type_num in [2, 3, 5, 7, 9])
            res_type = aa_types[f'type{self.residue_type_num}']
            self.vocab = {x - 1: y for x, y in zip(aa_types.idx, res_type)}

        # 读取列表
        self.pdb_list = pd.read_csv(data)['pdb'].values
        self.num = self.pdb_list.shape[0]

        # ---- 打开 HDF5 并缓存到字典（尽量与原代码一致）----
        hh_data_pdb = h5py.File(pdb_h5_path, 'r', libver='latest', swmr=True)
        hh_data_seq = h5py.File(seq_h5_path, 'r', libver='latest', swmr=True)

        # NEW: 尝试打开 rama 文件；若不存在则标记
        self.has_rama = True
        try:
            hh_data_rama = h5py.File(rama_h5_path, 'r', libver='latest', swmr=True)
        except (OSError, FileNotFoundError):
            hh_data_rama = None
            self.has_rama = False

        # NEW: optional per-residue ESM cache.
        # Gracefully fall back to all-zero embeddings if the file is missing;
        # this keeps the wiring exercisable before the precompute script has
        # been run. A one-line warning is printed so the user notices.
        self.has_esm = False
        hh_data_esm = None
        if self.use_esm and esm_h5_path is not None:
            try:
                hh_data_esm = h5py.File(esm_h5_path, 'r', libver='latest', swmr=True)
                self.has_esm = True
            except (OSError, FileNotFoundError):
                print(f"[DatasetLocalGenCM] --use_esm is ON but ESM h5 not found at "
                      f"{esm_h5_path}; falling back to zero embeddings.")
                hh_data_esm = None
                self.has_esm = False

        self.group_num_dict = {}
        self.coords_dict = {}
        self.start_id_dict = {}
        self.res_counts_dict = {}
        self.seq_dict = {}
        self.rama_dict = {}  # NEW
        self.esm_dict = {}   # NEW

        for pdb in tqdm(self.pdb_list):
            data_pdb = hh_data_pdb[pdb]
            self.group_num_dict[pdb] = data_pdb['group_num'][()]
            self.coords_dict[pdb] = data_pdb['coords'][()]
            self.start_id_dict[pdb] = data_pdb['start_id'][()]
            self.res_counts_dict[pdb] = data_pdb['res_counts'][()]
            self.seq_dict[pdb] = hh_data_seq[pdb][()]

            # NEW: 读取 Rama 矩阵 (num_blocks, 15, 2)，若没有则存 None
            if self.has_rama and (pdb in hh_data_rama):
                grp = hh_data_rama[pdb]
                if rama_dataset_name in grp:
                    self.rama_dict[pdb] = grp[rama_dataset_name][()]
                else:
                    self.rama_dict[pdb] = None
            else:
                self.rama_dict[pdb] = None

            # NEW: per-residue ESM matrix (L_chain, d_esm). Missing -> None.
            if self.has_esm and (pdb in hh_data_esm):
                grp = hh_data_esm[pdb]
                if esm_dataset_name in grp:
                    self.esm_dict[pdb] = grp[esm_dataset_name][()]
                elif isinstance(grp, h5py.Dataset):
                    self.esm_dict[pdb] = grp[()]
                else:
                    self.esm_dict[pdb] = None
            else:
                self.esm_dict[pdb] = None

        hh_data_pdb.close()
        hh_data_seq.close()
        if hh_data_rama is not None:
            hh_data_rama.close()
        if hh_data_esm is not None:
            hh_data_esm.close()

        n_rama_loaded = sum(1 for v in self.rama_dict.values() if v is not None)
        if n_rama_loaded == 0:
            print(
                '[DatasetLocalGenCM] WARNING: no per-PDB Ramachandran blocks loaded '
                f'({n_rama_loaded}/{len(self.pdb_list)}). '
                f'Check --rama_h5_path ({rama_h5_path!s}) contains dataset '
                f'{rama_dataset_name!r} under each PDB key. '
                'When masks are all zero, LocalEnergyCE yields coords_rama_loss ~ 0. '
                'Use hhsuite_rama_v2.h5 from build_rama_h5_v2.py; not wo_rama stubs.'
            )

    def __len__(self):
        return self.num

    # 每次调用会返回一个样本。
    def __getitem__(self, item):
        # select a pdb id
        pdb = self.pdb_list[item]

        # load the chimeric sequences
        seq_chim_n = self.seq_dict[pdb]

        # random select one of the N chimeric sequence
        if self.no_homology:
            k = 0
        else:
            k = torch.randint(0, seq_chim_n.shape[0], (1,))[0]
        seq_chim = seq_chim_n[k]

        # random select one of the L local structures
        group_num_n = self.group_num_dict[pdb]
        i = torch.randint(0, group_num_n.shape[0], (1,))[0]
        group_num = group_num_n[i]
        group_num = group_num - 1   # Shift from 1-based to 0-based
        start_id = self.start_id_dict[pdb][i]
        coords = self.coords_dict[pdb][i]
        res_counts = self.res_counts_dict[pdb][i]

        # ---- NEW: 取出 Ramachandran 15x2 (若存在) ----
        rama15 = None
        if self.rama_dict[pdb] is not None:
            # 期望形状: (num_blocks, 15, 2)
            # 选择与本地结构同一个 i
            if i < self.rama_dict[pdb].shape[0]:
                rama15 = self.rama_dict[pdb][i]  # (15, 2)
            else:
                # 若索引越界（理论上不应发生），则视作缺失
                rama15 = None

        # Ensure seq_len does not exceed the length of the structure
        if self.seq_len > group_num.shape[0]:
            raise ValueError(f'args.seq_len ({self.seq_len}) > residue numbers ({group_num.shape[0]}) for {pdb}.')
        if self.seq_len < group_num.shape[0]:
            coords = coords[:self.seq_len]
            group_num = group_num[:self.seq_len]
            start_id = start_id[:self.seq_len]
            res_counts = res_counts[:self.seq_len]

        # Out-of-bounds guard. `group_num` is already shifted to 0-based above,
        # so valid indices are [0, L_chim - 1]. A handful of v2 blocks produced
        # by local_extractor_v2 sit right at the chain edge and end up with one
        # index == L_chim after the signed-offset alignment; the old check used
        # `>` (off-by-one) and let them through, which then crashed at
        # `seq_chim[L_chim]`. We also defensively reject negative indices in
        # case a PDB's raw group_num started below 1. A None return is handled
        # upstream by `collate_drop_none`.
        seq_len_chim = seq_chim.shape[0]
        if group_num.size == 0 or group_num.max() >= seq_len_chim or group_num.min() < 0:
            return None

        # Map the sequence indices to actual residue symbols/IDs
        seq = np.array([seq_chim[x] for x in group_num])

        if self.residue_type_num != 20:
            seq = np.array([self.vocab[x] for x in seq])

        seq = torch.tensor(seq, dtype=torch.long)

        # Convert coords to torch tensors and compute (r, theta, phi)
        coords = torch.tensor(coords, dtype=torch.float)
        if torch.any(torch.isnan(coords)) or torch.any(torch.isinf(coords)):
            raise ValueError(f"Invalid coordinates detected in PDB: {pdb}")

        # Keep the block-local Cartesian frame before the spherical conversion.
        # Only returned when --use_cart_coords is enabled; the spherical 'coords'
        # tensor below is unchanged so the baseline path is bit-identical.
        coords_cart = coords.clone()  # (L, 3), block-local Cartesian

        # 计算球坐标 (r, theta, phi)
        r = torch.norm(coords, dim=-1)
        eps = 1e-6
        r_clamped = r[1:] + eps
        cos_theta = coords[1:, 2] / r_clamped
        cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)  # Ensure valid range for arccos
        theta = torch.acos(cos_theta)
        phi = torch.atan2(coords[1:, 1], coords[1:, 0])
        theta = F.pad(theta, (1, 0), value=0)
        phi = F.pad(phi, (1, 0), value=0)
        coords = torch.stack((r, theta, phi), dim=1)  # (L, 3)

        start_id = torch.tensor(start_id, dtype=torch.long)
        res_counts = torch.tensor(res_counts, dtype=torch.float)

        # ---------------- NEW: 组织 Ramachandran 到 (L, 2) + mask (L,) ----------------
        # 期望输入：rama15 (15, 2)，角度单位：弧度，区间(-pi, pi]（如需可在此处 wrap）
        L = self.seq_len
        rama_full = torch.zeros((L, 2), dtype=torch.float)   # (L, 2)
        rama_mask = torch.zeros((L,), dtype=torch.float)     # (L, )

        if rama15 is not None:
            # numpy -> torch
            rama15 = torch.tensor(rama15, dtype=torch.float)  # (15, 2)
            valid = torch.isfinite(rama15).all(dim=-1).float()  # (15,)

            # Wrap to (-pi, pi] only where defined; NaN -> 0 before wrap for stability
            pi = float(np.pi)
            rama15_safe = torch.nan_to_num(rama15, nan=0.0, posinf=0.0, neginf=0.0)
            rama15_safe = ((rama15_safe + pi) % (2 * pi)) - pi

            if L == 15:
                rama_full[:] = rama15_safe
                rama_mask[:] = valid
            elif L > 15:
                start = (L - 15) // 2
                end = start + 15
                rama_full[start:end] = rama15_safe
                rama_mask[start:end] = valid
            else:
                rama_full[:] = rama15_safe[:L]
                rama_mask[:] = valid[:L]
        else:
            # 没有 rama 数据：保持零与零 mask（下游可忽略这些位置）
            pass

        # -----------------------------------------------------------------

        # Fast path: no optional feature-extraction enhancements requested.
        # Return the original 6-tuple unchanged so collate / trainer stay
        # bit-identical to the pre-existing baseline.
        if not self._has_extras:
            return seq, coords, start_id, res_counts, rama_full, rama_mask

        # -------- Optional extras (see options.py for the flags) ----------
        L = self.seq_len

        # (1) Per-residue ESM embedding for the 15-residue block.
        #     group_num (already shifted to 0-based above) gives the chain
        #     indices. If the ESM h5 doesn't have this PDB, or indices fall
        #     outside the cached sequence, the corresponding rows are zero.
        if self.use_esm:
            esm_block = torch.zeros((L, self.esm_dim_in), dtype=torch.float)
            esm_full = self.esm_dict.get(pdb, None)
            if esm_full is not None:
                esm_full_np = np.asarray(esm_full)
                if esm_full_np.ndim == 2 and esm_full_np.shape[1] >= 1:
                    L_chain, d_cache = esm_full_np.shape
                    if d_cache != self.esm_dim_in:
                        raise ValueError(
                            f"ESM cache dim ({d_cache}) for {pdb} does not "
                            f"match --esm_dim_in ({self.esm_dim_in}).")
                    gn = group_num.astype(np.int64)
                    in_range = (gn >= 0) & (gn < L_chain)
                    valid_idx = np.where(in_range)[0]
                    if valid_idx.size > 0:
                        gathered = esm_full_np[gn[in_range]].astype(np.float32)
                        esm_block[valid_idx] = torch.from_numpy(gathered)
        else:
            esm_block = torch.zeros((L, self.esm_dim_in), dtype=torch.float)

        # (2) Block-local Cartesian (x, y, z) complementing spherical (r, θ, φ).
        if not self.use_cart_coords:
            coords_cart = torch.zeros_like(coords)

        # (3) Signed chain offset relative to the central residue, clamped and
        #     shifted to a non-negative embedding index.
        if self.use_seq_offset:
            gn = group_num.astype(np.int64)
            offset = gn - gn[0]
            M = self.seq_offset_max
            offset_clamped = np.clip(offset, -M, M) + M  # [0, 2M]
            seq_offset = torch.from_numpy(offset_clamped.astype(np.int64))
        else:
            seq_offset = torch.zeros((L,), dtype=torch.long)

        # (4) Per-block backbone dihedrals as sin/cos 4-d. Reuses the exact
        #     (L, 2) `rama_full` / `rama_mask` that the loss head already
        #     consumes, so we don't re-hit the HDF5. Masked positions are zero
        #     in all 4 channels (distinguishable from a genuine phi=psi=0).
        if self.use_dihedral:
            sin_pp = torch.sin(rama_full)
            cos_pp = torch.cos(rama_full)
            dihedral_local = torch.stack(
                [sin_pp[:, 0], cos_pp[:, 0], sin_pp[:, 1], cos_pp[:, 1]],
                dim=-1,
            )
            dihedral_local = dihedral_local * rama_mask.unsqueeze(-1)
        else:
            dihedral_local = torch.zeros((L, 4), dtype=torch.float)

        return (seq, coords, start_id, res_counts, rama_full, rama_mask,
                esm_block, coords_cart, seq_offset, dihedral_local)


if __name__ == '__main__':
    import argparse
    from torch.utils.data import DataLoader, WeightedRandomSampler

    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_factor", type=float, default=0.5)
    parser.add_argument("--noise_factor", type=float, default=0.001)
    parser.add_argument("--seq_len", type=int, default=14)
    parser.add_argument("--dist_mask", action='store_true', default=False)
    parser.add_argument("--dist_cutoff", type=float, default=10)
    parser.add_argument("--seq_type", type=str, default='residue')
    parser.add_argument("--data_flag", type=str, default='training_30_CA_v2')
    parser.add_argument("--residue_type_num", type=int, default=20,
                        help='number of residue types used in the sequence vocabulary')
    parser.add_argument("--no_homology", action='store_true', default=False)

    # 可根据实际文件名修改
    parser.add_argument("--pdb_h5_path", type=str, default=data_path('hhsuite_CB_v2.h5'))
    parser.add_argument("--seq_h5_path", type=str, default=data_path('hhsuite_pdb_seq_v2.h5'))
    parser.add_argument("--rama_h5_path", type=str, default=data_path('hhsuite_rama_v2.h5'))
    parser.add_argument("--rama_dataset_name", type=str, default='rama')

    args = parser.parse_args()

    train_dataset = DatasetLocalGenCM(
        data_path('hhsuite_CB_pdb_list.csv'),
        args,
        pdb_h5_path=args.pdb_h5_path,
        seq_h5_path=args.seq_h5_path,
        rama_h5_path=args.rama_h5_path,
        rama_dataset_name=args.rama_dataset_name
    )

    pdb_weights = pd.read_csv(data_path('hhsuite_CB_pdb_list.csv'))['weight'].values

    datasampler = WeightedRandomSampler(weights=pdb_weights, num_samples=10)

    # 注意：若 __getitem__ 返回 None（越界 debug 情况），默认 collate_fn 会报错。
    # 生产中建议写一个 collate_fn 过滤 None，或改为 raise。
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=10,
        sampler=datasampler,
        pin_memory=True
    )
