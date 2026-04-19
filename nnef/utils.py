import os
import numpy as np
import pandas as pd
import torch
from Bio.PDB import Selection, PDBParser

from paths import data_path


def resolve_model_checkpoint_path(args):
    """Return path to ``state_dict`` .pt: ``--load_checkpoint`` wins over
    ``load_exp/models/model.pt``.
    """
    direct = getattr(args, 'load_checkpoint', None)
    if direct:
        path = os.path.expanduser(direct)
        if not os.path.isfile(path):
            raise FileNotFoundError(f'--load_checkpoint not found: {path}')
        return path
    exp = getattr(args, 'load_exp', None)
    if exp:
        path = os.path.join(os.path.expanduser(exp), 'models', 'model.pt')
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f'Expected checkpoint at {path} (from --load_exp); missing.')
        return path
    raise ValueError(
        'Provide --load_checkpoint PATH.pt or --load_exp DIR '
        '(with DIR/models/model.pt).',
    )


def test_setup(args):
    from protein_os import EnergyFun, ProteinBase
    from model import LocalTransformer

    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(
            'CUDA was requested (--device cuda) but no GPU is visible on this '
            'machine (common on Slurm login nodes, e.g. holylogin*). Allocate a '
            'GPU (salloc/sbatch on the gpu partition), or pass --device cpu '
            'for a slow CPU-only run.'
        )

    model = LocalTransformer(args)
    energy_fn = EnergyFun(model, args)

    # strict=False so a baseline checkpoint (no side-layer keys) can be
    # loaded into an extended model whose extras are gated off or pre-
    # initialized to zero. Report what is missing / unexpected so the user
    # notices any silent mismatch.
    ckpt_path = resolve_model_checkpoint_path(args)
    print(f'[test_setup] loading weights from {ckpt_path}')
    state = torch.load(ckpt_path, map_location=torch.device('cpu'))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[test_setup] load_state_dict(strict=False): "
              f"missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    model.to(device)
    model.eval()
    # Move loss head buffers (e.g. position_weights) that are not on ``model`` itself.
    energy_fn.to(device)

    ProteinBase.k = args.seq_len - 4
    ProteinBase.use_graph_net = args.use_graph_net
    ProteinBase.use_cart_coords = bool(getattr(args, 'use_cart_coords', False))
    ProteinBase.use_seq_offset = bool(getattr(args, 'use_seq_offset', False))
    ProteinBase.seq_offset_max = int(getattr(args, 'seq_offset_max', 64))
    ProteinBase.use_esm = bool(getattr(args, 'use_esm', False))
    ProteinBase.use_dihedral = bool(getattr(args, 'use_dihedral', False))
    ProteinBase.use_local_frame_v2 = not bool(
        getattr(args, 'legacy_local_frame', False))
    ProteinBase.struct_v2_dist_cutoff = float(
        getattr(args, 'struct_dist_cutoff', 20.0))

    return device, model, energy_fn, ProteinBase


def _compute_phi_psi_from_backbone(n_xyz, ca_xyz, c_xyz):
    """Compute per-residue (phi, psi) dihedrals from backbone N/CA/C atoms.

    Parameters
    ----------
    n_xyz, ca_xyz, c_xyz : np.ndarray, shape (L, 3)
        N, CA, C atom coordinates for a single chain in residue order.

    Returns
    -------
    np.ndarray of shape (L, 2), radians, column 0 = phi, column 1 = psi.
    Chain-terminal residues get NaN (phi[0] and psi[L-1]) because the
    required neighboring atoms are not available; downstream gather/encode
    treats NaN as "undefined" and zero-masks the sin/cos channels, matching
    the training-side `rama_mask==0` convention.

    Implementation note: dihedral computed via the standard cross-product
    formula, vectorised over all residues at once. Chain breaks are NOT
    detected here -- caller should ensure the input is a single continuous
    chain in residue order (which is what load_protein_decoy already does).
    """
    L = n_xyz.shape[0]
    assert ca_xyz.shape == (L, 3) and c_xyz.shape == (L, 3)

    def _dihedral(p1, p2, p3, p4):
        b1 = p2 - p1
        b2 = p3 - p2
        b3 = p4 - p3
        b2_unit = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-12)
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        m1 = np.cross(n1, b2_unit)
        x = (n1 * n2).sum(axis=-1)
        y = (m1 * n2).sum(axis=-1)
        return np.arctan2(y, x)

    phi = np.full(L, np.nan, dtype=np.float64)
    psi = np.full(L, np.nan, dtype=np.float64)
    if L >= 2:
        phi[1:] = _dihedral(c_xyz[:-1], n_xyz[1:], ca_xyz[1:], c_xyz[1:])
        psi[:-1] = _dihedral(n_xyz[:-1], ca_xyz[:-1], c_xyz[:-1], n_xyz[1:])
    return np.stack([phi, psi], axis=-1).astype(np.float32)


def _load_dihedral_from_beads(df_beads):
    """Try to build (L, 2) phi/psi radians from a decoy bead CSV.

    Returns a float32 numpy array of shape (L, 2), or ``None`` when the CSV
    doesn't carry N/C backbone columns (legacy decoy sets that were extracted
    before --use_dihedral existed only store CA/CB). In that case the caller
    leaves ``dihedral_full=None`` and the energy path stays bit-identical to
    the baseline (additive layer is zero-initialised).
    """
    required = ('xn', 'yn', 'zn', 'xc', 'yc', 'zc')
    if not all(col in df_beads.columns for col in required):
        return None
    # CA column names differ between bead extractors: older ones use plain
    # x/y/z for CA, newer ones use xca/yca/zca.
    if {'xca', 'yca', 'zca'}.issubset(df_beads.columns):
        ca = df_beads[['xca', 'yca', 'zca']].values
    else:
        ca = df_beads[['x', 'y', 'z']].values
    n_xyz = df_beads[['xn', 'yn', 'zn']].values
    c_xyz = df_beads[['xc', 'yc', 'zc']].values
    return _compute_phi_psi_from_backbone(
        np.asarray(n_xyz, dtype=np.float64),
        np.asarray(ca, dtype=np.float64),
        np.asarray(c_xyz, dtype=np.float64),
    )


def write_pdb(seq, coords, pdb_id, flag, exp_id):
    amino_acids = pd.read_csv(data_path('amino_acids.csv'))
    vocab = {x: y for x, y in zip(amino_acids.AA, amino_acids.AA3C)}
    seq = [vocab[x] for x in seq]

    num = np.arange(coords.shape[0])
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    with open(data_path('fold', exp_id, f'{pdb_id}_{flag}.pdb'), 'wt') as mf:
        for i in range(len(num)):
            mf.write(f'ATOM  {num[i]:5d}   CA {seq[i]} A{num[i]:4d}    {x[i]:8.3f}{y[i]:8.3f}{z[i]:8.3f}\n')


def write_pdb_sample(seq, coords_sample, pdb_id, flag, exp_id):
    amino_acids = pd.read_csv(data_path('amino_acids.csv'))
    # vocab = {x: y for x, y in zip(amino_acids.AA, amino_acids.AA3C)}
    # seq = [vocab[x] for x in seq]

    with open(data_path('fold', exp_id, f'{pdb_id}_{flag}.pdb'), 'wt') as mf:
        for j, coords in enumerate(coords_sample):
            num_steps = (j + 1)
            mf.write('MODEL        '+str(num_steps)+'\n')

            num = np.arange(coords.shape[0])
            x = coords[:, 0]
            y = coords[:, 1]
            z = coords[:, 2]
            for i in range(len(num)):
                mf.write(f'ATOM  {num[i]:5d}   CA {seq[i]} A{num[i]:4d}    {x[i]:8.3f}{y[i]:8.3f}{z[i]:8.3f}\n')
            mf.write('ENDMDL\n')


def write_pdb_sample2(seq, coords_sample, pdb_id, flag, save_dir):
    amino_acids = pd.read_csv(data_path('amino_acids.csv'))
    vocab = {x: y.upper() for x, y in zip(amino_acids.AA, amino_acids.AA3C)}
    seq = [vocab[x] for x in seq]

    with open(f'{save_dir}/{pdb_id}_{flag}.pdb', 'wt') as mf:
        for j, coords in enumerate(coords_sample):
            num_steps = (j + 1)
            mf.write('MODEL        '+str(num_steps)+'\n')

            num = np.arange(coords.shape[0])
            x = coords[:, 0]
            y = coords[:, 1]
            z = coords[:, 2]
            for i in range(len(num)):
                mf.write(f'ATOM  {num[i]:5d}   CA {seq[i]} A{num[i]:4d}    {x[i]:8.3f}{y[i]:8.3f}{z[i]:8.3f}\n')
            mf.write('ENDMDL\n')


def transform_profile(seq, profile, noise_factor, seq_factor):
    seq_len = len(seq)
    # add noise to profile
    noise = np.random.rand(seq_len, 20) * noise_factor
    profile += noise

    # profile[range(seq_len), seq] += seq_factor
    # # normalize the profile
    # profile /= profile.sum(axis=1)[:, None]

    df = pd.read_csv(data_path('aa_freq.csv'))
    aa_freq = df['freq'].values / df['freq'].sum()

    profile = profile / (profile + aa_freq)

    return profile


def load_protein_v0(data_dir, pdb_id, mode, device, args):
    amino_acids = pd.read_csv(data_path('amino_acids.csv'))
    vocab = {x: y - 1 for x, y in zip(amino_acids.AA, amino_acids.idx)}

    print(pdb_id)
    # pdb_id_bead = pdb_id.split('_')[0] + '_' + pdb_id.split('_')[2]

    df_beads = pd.read_csv(f'{data_dir}/{pdb_id}_bead.csv')
    df_profile = pd.read_csv(f'{data_dir}/{pdb_id}_profile.csv')

    seq = df_profile['group_name'].values
    seq_id = df_profile['group_name'].apply(lambda x: vocab[x]).values

    if mode == 'CA':
        coords = df_beads[['xca', 'yca', 'zca']].values
    elif mode == 'CB':
        coords = df_beads[['xcb', 'ycb', 'zcb']].values
    elif mode == 'CAS':
        coords = (df_beads[['xca', 'yca', 'zca']].values + df_beads[['xs', 'ys', 'zs']].values) / 2
    else:
        raise ValueError('mode should be CA / CB / CAS.')

    coords = torch.tensor(coords, dtype=torch.float, device=device)

    seq_type = args.seq_type
    if seq_type == 'residue':
        profile = torch.tensor(seq_id, dtype=torch.long, device=device)
    else:
        profile = df_profile[[f'aa{i}' for i in range(20)]].values
        profile = transform_profile(seq_id, profile, args.noise_factor, args.seq_factor)
        profile = torch.tensor(profile, dtype=torch.float, device=device)
    return seq, coords, profile


def load_protein(data_dir, pdb_id, mode, device, args):
    amino_acids = pd.read_csv(data_path('amino_acids.csv'))
    vocab = {x.upper(): y - 1 for x, y in zip(amino_acids.AA3C, amino_acids.idx)}

    print(pdb_id)

    df_beads = pd.read_csv(f'{data_dir}/{pdb_id}_bead.csv')

    seq = df_beads['group_name'].values
    seq_id = df_beads['group_name'].apply(lambda x: vocab[x]).values

    if mode == 'CA':
        coords = df_beads[['xca', 'yca', 'zca']].values
    elif mode == 'CB':
        coords = df_beads[['xcb', 'ycb', 'zcb']].values
    elif mode == 'CAS':
        coords = (df_beads[['xca', 'yca', 'zca']].values + df_beads[['xs', 'ys', 'zs']].values) / 2
    else:
        raise ValueError('mode should be CA / CB / CAS.')

    coords = torch.tensor(coords, dtype=torch.float, device=device)

    profile = torch.tensor(seq_id, dtype=torch.long, device=device)
    return seq, coords, profile


def load_protein_bead(bead_csv, mode, device):
    amino_acids = pd.read_csv(data_path('amino_acids.csv'))
    vocab = {x: y - 1 for x, y in zip(amino_acids.AA, amino_acids.idx)}
    vocab2 = {x.upper(): y - 1 for x, y in zip(amino_acids.AA3C, amino_acids.idx)}

    df_beads = pd.read_csv(bead_csv)

    seq = df_beads['group_name'].values
    if len(seq[0]) == 1:
        seq_id = df_beads['group_name'].apply(lambda x: vocab[x]).values
    else:
        seq_id = df_beads['group_name'].apply(lambda x: vocab2[x]).values

    if mode == 'CA':
        coords = df_beads[['xca', 'yca', 'zca']].values
    elif mode == 'CB':
        coords = df_beads[['xcb', 'ycb', 'zcb']].values
    else:
        raise ValueError('mode should be CA / CB.')

    coords = torch.tensor(coords, dtype=torch.float, device=device)
    profile = torch.tensor(seq_id, dtype=torch.long, device=device)

    return seq, coords, profile


def load_protein_decoy(pdb_id, decoy_id, mode, device, args):
    amino_acids = pd.read_csv(data_path('amino_acids.csv'))
    vocab = {x: y - 1 for x, y in zip(amino_acids.AA, amino_acids.idx)}

    decoy_set = args.decoy_set
    profile_set = 'pdb_profile_training_100'

    df_beads = pd.read_csv(data_path('decoys', decoy_set, pdb_id, f'{decoy_id}_bead.csv'))
    # Match ``local_extractor_v2`` / training: blocks assume residue rows sorted by group_num.
    if 'group_num' in df_beads.columns:
        df_beads = df_beads.sort_values('group_num').reset_index(drop=True)

    seq_type = args.seq_type
    if seq_type != 'residue':
        df_profile = pd.read_csv(data_path('decoys', decoy_set, profile_set, f'{pdb_id}_profile.csv'))

    seq = df_beads['group_name'].values
    seq_id = df_beads['group_name'].apply(lambda x: vocab[x]).values

    if mode == 'CA':
        # v2 / thesis bead CSVs name CA as xca,yca,zca; legacy extract_beads
        # used plain x,y,z for CA. Support both so decoy scoring works on the
        # shipped CASP / 3DRobot sets.
        if {'xca', 'yca', 'zca'}.issubset(df_beads.columns):
            coords = df_beads[['xca', 'yca', 'zca']].values
        else:
            coords = df_beads[['x', 'y', 'z']].values
    elif mode == 'CB':
        coords = df_beads[['xcb', 'ycb', 'zcb']].values
    elif mode == 'CAS':
        coords = (df_beads[['xca', 'yca', 'zca']].values + df_beads[['xs', 'ys', 'zs']].values) / 2
    else:
        raise ValueError('mode should be CA / CB / CAS.')

    coords = torch.tensor(coords, dtype=torch.float, device=device)

    if seq_type == 'residue':
        profile = torch.tensor(seq_id, dtype=torch.long, device=device)
    else:
        profile = df_profile[[f'aa{i}' for i in range(20)]].values
        profile = transform_profile(seq_id, profile, args.noise_factor, args.seq_factor)
        profile = torch.tensor(profile, dtype=torch.float, device=device)

    # Dihedral is computed lazily -- the np->torch cast only happens when the
    # caller actually asks for it (use_dihedral on). For legacy decoy sets
    # without N/C columns, `dihedral_full` is None and Protein falls back to
    # baseline behaviour (zero additive contribution).
    dihedral_np = _load_dihedral_from_beads(df_beads)
    if dihedral_np is not None:
        dihedral_full = torch.from_numpy(dihedral_np).to(device=device, dtype=torch.float32)
    else:
        dihedral_full = None

    if 'group_num' in df_beads.columns:
        chain_group_num = torch.from_numpy(
            df_beads['group_num'].values.astype(np.int64)
        ).to(device=device)
    else:
        chain_group_num = torch.arange(
            len(df_beads), device=device, dtype=torch.long,
        )

    req_bb = ('xn', 'yn', 'zn', 'xca', 'yca', 'zca', 'xc', 'yc', 'zc')
    if all(col in df_beads.columns for col in req_bb):
        n_xyz = torch.tensor(
            df_beads[['xn', 'yn', 'zn']].values,
            dtype=torch.float, device=device,
        )
        ca_xyz = torch.tensor(
            df_beads[['xca', 'yca', 'zca']].values,
            dtype=torch.float, device=device,
        )
        c_xyz = torch.tensor(
            df_beads[['xc', 'yc', 'zc']].values,
            dtype=torch.float, device=device,
        )
    else:
        n_xyz = ca_xyz = c_xyz = None

    return seq, coords, profile, dihedral_full, n_xyz, ca_xyz, c_xyz, chain_group_num


def extract_beads(pdb_path):
    amino_acids = pd.read_csv(data_path('amino_acids.csv'))
    vocab_aa = [x.upper() for x in amino_acids.AA3C]
    vocab_dict = {x.upper(): y for x, y in zip(amino_acids.AA3C, amino_acids.AA)}

    p = PDBParser()
    structure = p.get_structure('X', pdb_path)
    residue_list = Selection.unfold_entities(structure, 'R')

    ca_center_list = []
    cb_center_list = []
    n_center_list = []
    c_center_list = []
    res_name_list = []
    res_num_list = []
    chain_list = []

    for res in residue_list:
        if res.get_resname() not in vocab_aa:
            # raise ValueError('protein has non natural amino acids')
            continue

        try:
            res['CA'].get_coord()
            if res.get_resname() != 'GLY':
                res['CB'].get_coord()
        except KeyError:
            print(f'{pdb_path}, {res} missing CA / CB atoms')
            continue

        chain_list.append(res.parent.id)
        res_name_list.append(vocab_dict[res.get_resname()])
        res_num_list.append(res.id[1])

        ca_center_list.append(res['CA'].get_coord())
        if res.get_resname() != 'GLY':
            cb_center_list.append(res['CB'].get_coord())
        else:
            cb_center_list.append(res['CA'].get_coord())

        # Backbone N and C are needed to derive phi/psi dihedrals at inference
        # time (see utils._compute_phi_psi_from_backbone). Missing atoms -> NaN
        # so the downstream sin/cos gather treats the residue as "undefined".
        try:
            n_center_list.append(res['N'].get_coord())
        except KeyError:
            n_center_list.append(np.array([np.nan, np.nan, np.nan], dtype=np.float32))
        try:
            c_center_list.append(res['C'].get_coord())
        except KeyError:
            c_center_list.append(np.array([np.nan, np.nan, np.nan], dtype=np.float32))

    ca_center = np.vstack(ca_center_list)
    cb_center = np.vstack(cb_center_list)
    n_center = np.vstack(n_center_list)
    c_center = np.vstack(c_center_list)

    df = pd.DataFrame({'chain_id': chain_list,
                       'group_num': res_num_list,
                       'group_name': res_name_list,
                       'x': ca_center[:, 0],
                       'y': ca_center[:, 1],
                       'z': ca_center[:, 2],
                       'xcb': cb_center[:, 0],
                       'ycb': cb_center[:, 1],
                       'zcb': cb_center[:, 2],
                       'xn': n_center[:, 0],
                       'yn': n_center[:, 1],
                       'zn': n_center[:, 2],
                       'xc': c_center[:, 0],
                       'yc': c_center[:, 1],
                       'zc': c_center[:, 2]})

    df.to_csv(f'{pdb_path}_bead.csv', index=False)
    return df


def load_protein_pdb(pdb_path, mode, device):
    amino_acids = pd.read_csv(data_path('amino_acids.csv'))
    vocab = {x: y - 1 for x, y in zip(amino_acids.AA, amino_acids.idx)}
    vocab2 = {x.upper(): y - 1 for x, y in zip(amino_acids.AA3C, amino_acids.idx)}

    df_beads = extract_beads(pdb_path)

    seq = df_beads['group_name'].values
    if len(seq[0]) == 1:
        seq_id = df_beads['group_name'].apply(lambda x: vocab[x]).values
    else:
        seq_id = df_beads['group_name'].apply(lambda x: vocab2[x]).values

    if mode == 'CA':
        coords = df_beads[['xca', 'yca', 'zca']].values
    elif mode == 'CB':
        coords = df_beads[['xcb', 'ycb', 'zcb']].values
    else:
        raise ValueError('mode should be CA / CB.')

    coords = torch.tensor(coords, dtype=torch.float, device=device)
    profile = torch.tensor(seq_id, dtype=torch.long, device=device)

    return seq, coords, profile

