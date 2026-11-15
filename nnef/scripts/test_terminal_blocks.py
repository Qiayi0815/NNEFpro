"""Unit test: terminal-block layout and train/infer consistency of the v2 frame (WP0a/b).

For full and truncated chains (k=10 and k=4) checks that
  1. training extractor (local_extractor_v2.extract_blocks_v2), the reference
     inference path (Protein._get_local_struct_v2) and the vectorised path
     (_get_local_struct_v2_fast) give identical block members, order, start_id
     and local coordinates;
  2. slot 0 of every block is its own centre residue (terminal blocks included);
  3. an N-residue chain yields N blocks;
  4. filler slots (missing i±1/i±2 at the termini) are marked non-contiguous.

    python nnef/scripts/test_terminal_blocks.py
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

_NNEF = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _NNEF)
from data_prep_scripts.local_extractor_v2 import FILLER_SEG_BASE as EXT_FSB, extract_blocks_v2  # noqa: E402
from paths import data_path  # noqa: E402
from protein_os import FILLER_SEG_BASE, Protein, ProteinBase  # noqa: E402

AA = pd.read_csv(data_path('amino_acids.csv'))
VOCAB = {x: y - 1 for x, y in zip(AA.AA, AA.idx)}


def protein_from(df):
    t = lambda c: torch.tensor(df[c].values, dtype=torch.float32)  # noqa: E731
    xyz = lambda a: torch.stack([t(f'x{a}'), t(f'y{a}'), t(f'z{a}')], dim=1)  # noqa: E731
    prof = torch.tensor([VOCAB[s] for s in df.group_name.values])
    return Protein(df.group_name.values, xyz('cb'), prof, n_coords=xyz('n'), ca_coords=xyz('ca'),
                   c_coords=xyz('c'), chain_group_num=torch.tensor(df.group_num.values))


def check(df, k, label):
    ProteinBase.k = k
    ProteinBase.use_local_frame_v2 = True
    # extract_blocks_v2 itself applies no distance cutoff (_process_chain does), so
    # compare layouts with the cutoff disabled on the inference side too.
    ProteinBase.struct_v2_dist_cutoff = 1e9
    bs = 5 + k
    N = len(df)
    ext = extract_blocks_v2(df, k=k)
    assert ext is not None, f'{label}: extractor returned None'
    e_g = ext.group_num.values.reshape(-1, bs)
    e_seg = ext.seg.values.reshape(-1, bs)
    e_sid = np.zeros_like(e_seg)
    e_sid[:, 1:] = (e_seg[:, 1:] == e_seg[:, :-1])
    e_xyz = ext[['local_x', 'local_y', 'local_z']].values.reshape(-1, bs, 3)
    prot = protein_from(df)
    gid = df.group_num.values
    out = {}
    for fast in (False, True):
        ProteinBase.v2_fast = fast
        with torch.no_grad():
            _, coords, sid, _, g_local = prot.get_local_struct()
        out[fast] = (gid[g_local.numpy()], sid.numpy(), coords.numpy())
    ProteinBase.v2_fast = False
    ok = True
    for name, (g, sid, xyz) in (('reference', out[False]), ('fast', out[True])):
        same = (g.shape == e_g.shape and np.array_equal(g, e_g) and np.array_equal(sid, e_sid)
                and np.allclose(xyz, e_xyz, atol=1e-4))
        ok &= same
        print(f'  {label} k={k} {name:9s} vs extractor: blocks {g.shape[0]} vs {e_g.shape[0]} identical={same}')
    centre_ok = bool(np.array_equal(e_g[:, 0], np.sort(gid)[:e_g.shape[0]]) or set(e_g[:, 0]) == set(gid))
    n_ok = e_g.shape[0] == N
    # terminal fillers: block centred on the first residue has i-1/i-2 missing -> slots 1 and 3 fillers
    first = e_g[np.where(e_g[:, 0] == gid.min())[0][0]]
    fill_ok = e_sid[np.where(e_g[:, 0] == gid.min())[0][0], 1] == 0
    print(f'  {label} k={k}: slot0==centre for all blocks: {centre_ok}; N blocks == N residues ({N}): {n_ok}; '
          f'first-residue block {first[:5].tolist()} filler slot1 non-contiguous: {fill_ok}')
    return ok and centre_ok and n_ok and fill_ok


def check_off(df, k, label):
    """--terminal_blocks off: exactly the N-4 interior blocks, identical to their 'on' rows."""
    ProteinBase.k = k
    ProteinBase.struct_v2_dist_cutoff = 1e9
    prot = protein_from(df)
    gid = df.group_num.values
    ok = True
    for fast in (False, True):
        ProteinBase.v2_fast = fast
        res = {}
        for on in (True, False):
            ProteinBase.terminal_blocks = on
            with torch.no_grad():
                _, coords, _, _, g_local = prot.get_local_struct()
            res[on] = (coords, gid[g_local[:, 0].cpu().numpy()] if fast else g_local[:, 0].cpu().numpy())
        ProteinBase.terminal_blocks = True
        c_on, cen_on = res[True]
        c_off, cen_off = res[False]
        cen_on, cen_off = np.asarray(cen_on), np.asarray(cen_off)
        interior = ~np.isin(cen_on, np.concatenate([np.sort(cen_on)[:2], np.sort(cen_on)[-2:]]))
        same = (len(cen_off) == len(gid) - 4 and np.array_equal(cen_on[interior], cen_off)
                and torch.allclose(c_on[torch.as_tensor(interior)], c_off))
        print(f'{label} k={k} fast={fast}: off-blocks {len(cen_off)} (N-4={len(gid) - 4}), '
              f'identical to interior on-blocks: {same}')
        ok &= bool(same)
    return ok


def main():
    assert FILLER_SEG_BASE == EXT_FSB, 'FILLER_SEG_BASE differs between protein_os and extractor'
    bead = data_path('yang_small', '1ZZK', '1ZZK_bead.csv')
    df = pd.read_csv(bead).sort_values('group_num').reset_index(drop=True)
    results = [check(df, 10, '1ZZK full'),
               check(df.iloc[:20].reset_index(drop=True), 10, '1ZZK 1-20'),
               check(df.iloc[:12].reset_index(drop=True), 4, '1ZZK 1-12'),
               check(df.iloc[:9].reset_index(drop=True), 4, '1ZZK 1-9 (minimum for k=4)'),
               check_off(df, 10, '1ZZK full'),
               check_off(df.iloc[:12].reset_index(drop=True), 4, '1ZZK 1-12')]
    print('ALL PASS' if all(results) else 'FAIL')
    sys.exit(0 if all(results) else 1)


if __name__ == '__main__':
    main()
