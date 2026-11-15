"""Is the profile-driven drift a "neighbour rewiring" effect? (meeting diagnostic D1)

Hypothesis (3/4 experts): 86% of the profile weight scores NEIGHBOUR identities,
and which residues are neighbours is chosen by the geometry (hard top-k), so MD
can lower E_profile by pulling easy-to-predict (e.g. hydrophobic) residues into
many blocks' top-10. For native vs harvested far frames (RMSD >= rmsd_min) we
report, per chain:
  * frac_blocks_changed: blocks whose top-10 neighbour SET differs,
  * share of the profile drop carried by changed vs unchanged blocks,
  * mean change in in-degree (times a residue is picked as a neighbour) for
    hydrophobic vs polar residues.
If unchanged blocks carry most of the drop, rewiring is NOT the mechanism.

    python nnef/scripts/rewiring_diagnostic.py <yang model flags> --neg_dir runs/negatives/yang_r0 \
        --out eval/decomp/rewiring_yang.csv
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch

_NNEF = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _NNEF)
import options  # noqa: E402
from protein_os import Protein  # noqa: E402
from utils import load_protein_bead, test_setup  # noqa: E402

HYDROPHOBIC = set('AILMFVWC')


def per_block(prot, x, energy_fn):
    """Per-block profile loss (N-4,) and neighbour index sets (N-4, k)."""
    prot.coords = x
    profile_local, coords_local, start_id, res_counts, g_local = prot.get_local_struct()
    coords_cart, seq_offset, esm_local, dihedral_local = prot._build_extras(coords_local, g_local)
    coords_local = prot._local_cartesian_to_radian(coords_local)
    out = energy_fn.energy_fn.forward(profile_local, coords_local, start_id, res_counts,
                                      coords_cart=coords_cart, seq_offset=seq_offset,
                                      esm=esm_local, dihedral=dihedral_local)
    return out[2].detach().cpu().numpy(), g_local[:, 5:].detach().cpu().numpy()


def main():
    parser = options.get_local_gen_parser()
    parser.add_argument('--neg_dir', required=True)
    parser.add_argument('--rmsd_min', type=float, default=2.0)
    parser.add_argument('--max_chains', type=int, default=200)
    parser.add_argument('--out', required=True)
    args = options.parse_args_and_arch(parser)
    device, model, energy_fn, _pb = test_setup(args)
    energy_fn.energy_fn.reduction = 'keep_batch_dim'  # per-block losses

    beads = {os.path.basename(p)[:-len('_bead.csv')]: p
             for p in glob.glob(os.path.join(args.neg_dir, '_pool_shard*', 'pool_beads', '*_bead.csv'))}
    rows = []
    with torch.no_grad():
        for f in sorted(glob.glob(os.path.join(args.neg_dir, '*.npz')))[:args.max_chains]:
            cid = os.path.basename(f)[:-4]
            if cid not in beads:
                continue
            z = np.load(f)
            far = np.where(z['rmsd'] >= args.rmsd_min)[0]
            if len(far) == 0:
                continue
            seq, native, profile = load_protein_bead(beads[cid], args.mode, device)
            prot = Protein(seq, native.clone(), profile)
            p_nat, g_nat = per_block(prot, native, energy_fn)
            i = far[-1]
            x = torch.tensor(z['frames'][i], dtype=native.dtype, device=device)
            p_fr, g_fr = per_block(prot, x, energy_fn)
            changed = np.array([set(a) != set(b) for a, b in zip(g_nat, g_fr)])
            d = p_fr - p_nat
            n = native.shape[0]
            indeg_nat = np.bincount(g_nat.ravel(), minlength=n)
            indeg_fr = np.bincount(g_fr.ravel(), minlength=n)
            hyd = np.array([s in HYDROPHOBIC for s in seq])
            rows.append(dict(
                chain=cid, rmsd=float(z['rmsd'][i]), n_blocks=len(d),
                frac_blocks_changed=float(changed.mean()),
                d_profile_total=float(d.sum()),
                d_profile_changed=float(d[changed].sum()),
                d_profile_unchanged=float(d[~changed].sum()),
                d_indeg_hydrophobic=float((indeg_fr - indeg_nat)[hyd].mean()) if hyd.any() else np.nan,
                d_indeg_polar=float((indeg_fr - indeg_nat)[~hyd].mean()) if (~hyd).any() else np.nan,
            ))
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    tot = df.d_profile_total
    print(f'[rewire] {len(df)} chains; mean RMSD {df.rmsd.mean():.2f}')
    print(f'[rewire] frac blocks with changed neighbour set: mean {df.frac_blocks_changed.mean():.2f}')
    print(f'[rewire] profile drop (sum over blocks): mean {tot.mean():+.1f}  '
          f'changed-blocks {df.d_profile_changed.mean():+.1f}  unchanged-blocks {df.d_profile_unchanged.mean():+.1f}')
    print(f'[rewire] median share of drop from changed blocks: {(df.d_profile_changed / tot).median():.2f}')
    print(f'[rewire] per-block drop: changed {(df.d_profile_changed / (df.frac_blocks_changed * df.n_blocks)).median():+.2f}  '
          f'unchanged {(df.d_profile_unchanged / ((1 - df.frac_blocks_changed) * df.n_blocks)).median():+.2f}')
    print(f'[rewire] mean d in-degree: hydrophobic {df.d_indeg_hydrophobic.mean():+.3f}  polar {df.d_indeg_polar.mean():+.3f}')


if __name__ == '__main__':
    main()
