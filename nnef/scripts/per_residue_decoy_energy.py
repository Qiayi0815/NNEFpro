"""Per-residue decoy energies vs per-residue error: does a model score the TERMINI?

Whole-chain decoy ranking cannot show whether terminal residues are modelled
(ribbon scores blocks centred on every residue; Yang drops the first/last two).
For every decoy this writes, per residue,
  E_cen      energy of the block centred on that residue (NaN if no such block)
  E_contain  mean energy of all blocks the residue belongs to (window or neighbour)
  lddt       Cα-lDDT of that residue vs the native (15 Å inclusion, 0.5/1/2/4 Å)
and `--analyze` correlates energy with -lDDT across decoys, per residue, binned
by distance from the nearest chain end (0, 1, 2, 3-4, >=5 = interior).

Score (GPU, one batched pass per decoy):
    python nnef/scripts/per_residue_decoy_energy.py <model flags> --decoy_set 3DRobot_set \
        --targets 1BYIA,1C5EA --tag rib_k10_hard --out_dir eval/per_residue
Analyse (CPU):
    python nnef/scripts/per_residue_decoy_energy.py --analyze --out_dir eval/per_residue
The native is <target>/native_bead.csv when present (3DRobot), else the list
entry named *_native (peptide sets); decoys whose residues do not match the
native's group_num are scored on the shared residues only.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..')))
from paths import data_path  # noqa: E402

BINS = [(0, 0, '0'), (1, 1, '1'), (2, 2, '2'), (3, 4, '3-4'), (5, 10 ** 6, 'interior')]


def ca_lddt_per_residue(nat, dec, r0=15.0):
    """Cα lDDT per residue (rows of nat/dec correspond)."""
    dn = np.linalg.norm(nat[:, None] - nat[None], axis=-1)
    dd = np.linalg.norm(dec[:, None] - dec[None], axis=-1)
    inc = (dn < r0) & ~np.eye(len(nat), dtype=bool)
    err = np.abs(dn - dd)
    score = np.mean([(err < t) for t in (0.5, 1, 2, 4)], axis=0)
    return np.where(inc.sum(1) > 0, (score * inc).sum(1) / np.maximum(inc.sum(1), 1), np.nan)


def bead_df(decoy_set, target, name):
    df = pd.read_csv(data_path('decoys', decoy_set, target, f'{name}_bead.csv'))
    return df.sort_values('group_num').reset_index(drop=True)


def score(args):
    from decoy_score import _decoy_id_for, _load_decoy_list_df
    from protein_os import Protein
    from utils import load_protein_decoy, test_setup
    device, _model, energy_fn, _ = test_setup(args)
    out = os.path.join(args.out_dir, args.tag, args.decoy_set_name)
    os.makedirs(out, exist_ok=True)
    for target in args.targets.split(','):
        dst = os.path.join(out, f'{target}.npz')
        if os.path.exists(dst):
            continue
        names = [_decoy_id_for(n, args.decoy_set_name) for n in _load_decoy_list_df(target, args.decoy_set_name)['NAME']]
        nat_name = 'native' if os.path.exists(data_path('decoys', args.decoy_set_name, target, 'native_bead.csv')) \
            else next((n for n in names if n.endswith('_native')), None)
        if nat_name is not None:
            nat = bead_df(args.decoy_set_name, target, nat_name)
            gn = nat.group_num.values
            nat_ca = nat[['xca', 'yca', 'zca']].values
        else:
            # no native (e.g. casp14_full): energies only, residue columns = union
            # of all decoys' group_num, lDDT left NaN
            gn = np.unique(np.concatenate([pd.read_csv(
                data_path('decoys', args.decoy_set_name, target, f'{n}_bead.csv'),
                usecols=['group_num']).group_num.values for n in names]))
            nat_ca = None
        R = len(gn)
        E_cen = np.full((len(names), R), np.nan, np.float32)
        E_con = np.full((len(names), R), np.nan, np.float32)
        lddt = np.full((len(names), R), np.nan, np.float32)
        for j, name in enumerate(names):
            try:
                (seq, cb, prof, dih, n_xyz, ca_xyz, c_xyz, cgn) = load_protein_decoy(target, name, args.mode, device, args)
                prot = Protein(seq, cb, prof, dihedral_full=dih, n_coords=n_xyz, ca_coords=ca_xyz,
                               c_coords=c_xyz, chain_group_num=cgn)
                with torch.no_grad():
                    eb, g_local = prot.get_block_energies(energy_fn)
            except Exception as exc:  # noqa: BLE001  (bad decoy: leave NaN, keep going)
                print(f'[per_res] {target}/{name}: {exc}', flush=True)
                continue
            eb = eb.float().cpu().numpy()
            g_local = g_local.cpu().numpy()
            d_gn = cgn.cpu().numpy()
            col = {g: i for i, g in enumerate(gn)}
            rows = np.array([col.get(g, -1) for g in d_gn])               # decoy row -> native column
            cen = rows[g_local[:, 0]]
            ok = cen >= 0
            E_cen[j, cen[ok]] = eb[ok]
            acc, cnt = np.zeros(R), np.zeros(R)
            mem = rows[g_local]                                             # (B, 5+k)
            for b in range(mem.shape[0]):
                m = mem[b][mem[b] >= 0]
                acc[m] += eb[b]
                cnt[m] += 1
            E_con[j] = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
            if nat_ca is None:
                continue
            dec = bead_df(args.decoy_set_name, target, name)
            shared = np.isin(gn, dec.group_num.values)
            dec = dec.set_index('group_num').loc[gn[shared]]
            lddt[j, shared] = ca_lddt_per_residue(nat_ca[shared], dec[['xca', 'yca', 'zca']].values)
        np.savez_compressed(dst, names=np.array(names), group_num=gn, E_cen=E_cen, E_contain=E_con, lddt=lddt)
        print(f'[per_res] {args.tag} {target}: {len(names)} decoys x {R} residues -> {dst}', flush=True)


def analyze(out_dir):
    from scipy.stats import spearmanr
    rows = []
    for f in sorted(glob.glob(os.path.join(out_dir, '*', '*', '*.npz'))):
        tag, dset = f.split(os.sep)[-3], f.split(os.sep)[-2]
        z = np.load(f)
        R = len(z['group_num'])
        dist_end = np.minimum(np.arange(R), R - 1 - np.arange(R))
        for kind in ('E_cen', 'E_contain'):
            E = z[kind]
            for i in range(R):
                ok = np.isfinite(E[:, i]) & np.isfinite(z['lddt'][:, i])
                if ok.sum() < 20 or np.std(z['lddt'][ok, i]) < 1e-6 or np.std(E[ok, i]) < 1e-9:
                    continue
                r = -spearmanr(E[ok, i], z['lddt'][ok, i])[0]               # higher = better
                b = next(lbl for lo, hi, lbl in BINS if lo <= dist_end[i] <= hi)
                rows.append(dict(model=tag, set=dset, target=os.path.basename(f)[:-4], kind=kind,
                                 residue=i, dist_end=int(dist_end[i]), bin=b, rho=r))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, 'per_residue_rho.csv'), index=False)
    # per target: median rho within each bin, then median over targets
    t = df.groupby(['set', 'kind', 'model', 'target', 'bin']).rho.median().reset_index()
    s = t.groupby(['set', 'kind', 'model', 'bin']).rho.agg(['median', 'size']).reset_index()
    s.to_csv(os.path.join(out_dir, 'per_residue_summary.csv'), index=False)
    pd.set_option('display.width', 220)
    for (dset, kind), g in s.groupby(['set', 'kind']):
        print(f'\n=== {dset} / {kind}: median over targets of per-residue Spearman(-E, lDDT) by distance from chain end ===')
        print(g.pivot(index='model', columns='bin', values='median')[[b[2] for b in BINS]].round(3).to_string())


def main():
    if '--analyze' in sys.argv:
        out = sys.argv[sys.argv.index('--out_dir') + 1] if '--out_dir' in sys.argv else 'eval/per_residue'
        analyze(out)
        return
    import options
    parser = options.get_decoy_parser()      # provides --decoy_set
    parser.add_argument('--targets', required=True, help='comma-separated target ids')
    parser.add_argument('--tag', required=True, help='model tag for the output folder')
    parser.add_argument('--out_dir', default='eval/per_residue')
    args = parser.parse_args()
    args.decoy_set_name = args.decoy_set
    score(args)


if __name__ == '__main__':
    main()
