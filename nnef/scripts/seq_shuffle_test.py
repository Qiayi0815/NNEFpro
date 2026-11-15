"""Is the energy sequence-specific? Composition-preserving shuffle test on pep_nmr.

For each peptide, score three structures -- the native (medoid NMR model) and
the ideal-helix / ideal-hairpin controls -- with the TRUE sequence and with
`--n_shuffle` composition-preserving shuffles threaded onto the same
coordinates. A sequence-blind energy gives the same ΔE = E(helix) - E(native)
for every sequence; a sequence-specific one prefers the native fold more for
the true sequence than for shuffles:
    spec_z = (mean ΔE_shuffled - ΔE_true) / sd ΔE_shuffled   (> 0 = specific)
(analogously for the hairpin control). Energies are the mean block energy
(EnergyFun without residue_sum), as in decoy scoring.

    python nnef/scripts/seq_shuffle_test.py <model flags> --decoy_set pep_nmr \
        --targets 1A11,1L2Y --tag rib_k4_smooth_w10 --out_dir eval/seq_shuffle
    python nnef/scripts/seq_shuffle_test.py --analyze --out_dir eval/seq_shuffle
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..')))

STRUCTS = ('native', 'ctrl_helix', 'ctrl_hairpin')


def score(args):
    from protein_os import Protein
    from paths import data_path
    from utils import ideal_cb, ideal_cb_on, load_protein_decoy, test_setup
    device, _model, energy_fn, _ = test_setup(args)
    energy_fn.residue_sum = False
    aa = pd.read_csv(data_path('amino_acids.csv'))
    ALA = int(aa.idx[aa.AA == 'A'].iloc[0]) - 1
    rng = np.random.default_rng(args.seed)
    out = os.path.join(args.out_dir, args.tag)
    os.makedirs(out, exist_ok=True)
    for target in args.targets.split(','):
        dst = os.path.join(out, f'{target}.csv')
        if os.path.exists(dst):
            continue
        loaded = {}
        for st in STRUCTS:
            try:
                loaded[st] = load_protein_decoy(target, f'{target}_{st}', args.mode, device, args)
            except FileNotFoundError:
                pass
        if 'native' not in loaded:
            continue
        L = loaded['native'][2].shape[0]
        perms = [np.arange(L)] + [rng.permutation(L) for _ in range(args.n_shuffle)]
        rows = []
        for si, perm in enumerate(perms + [None]):
            # shuffle: -1 = true sequence, -2 = all alanine, >= 0 = composition-preserving shuffle
            row = dict(target=target, shuffle=-2 if perm is None else si - 1)
            for st, (seq, cb, prof, dih, n_xyz, ca_xyz, c_xyz, cgn) in loaded.items():
                if prof.shape[0] != L:
                    continue
                if perm is None:
                    seq_s, prof_s = np.full(L, 'A'), torch.full_like(prof, ALA)
                else:
                    seq_s, prof_s = np.asarray(seq)[perm], prof[torch.as_tensor(perm, device=prof.device)]
                if ideal_cb_on():                              # CB follows the threaded sequence (Gly: CB = CA)
                    cb = ideal_cb(torch.stack([n_xyz, ca_xyz, c_xyz, ca_xyz]), seq_s)
                prot = Protein(seq_s, cb, prof_s, dihedral_full=dih, n_coords=n_xyz,
                               ca_coords=ca_xyz, c_coords=c_xyz, chain_group_num=cgn)
                try:
                    with torch.no_grad():
                        row[f'E_{st}'] = float(prot.get_energy(energy_fn))
                except RuntimeError:
                    row[f'E_{st}'] = np.nan
            rows.append(row)
        pd.DataFrame(rows).to_csv(dst, index=False)
        print(f'[shuffle] {args.tag} {target}: {len(perms)} sequences', flush=True)


def analyze(out_dir):
    meta = pd.read_csv('nnef/data/decoys/pep_nmr/targets_summary.csv').set_index('pdb')
    res = []
    for d in sorted(glob.glob(os.path.join(out_dir, '*'))):
        if not os.path.isdir(d):
            continue
        for f in glob.glob(os.path.join(d, '*.csv')):
            x = pd.read_csv(f)
            t = x.target.iloc[0]
            r = dict(model=os.path.basename(d), target=t, ss=meta.loc[t, 'ss_class'] if t in meta.index else '?')
            e = x.set_index('shuffle')['E_native']
            sh_e = e[e.index >= 0].dropna()
            if -1 in e.index and len(sh_e) >= 5:                       # same structure, different sequence
                r['native_true_minus_shuf'] = e[-1] - sh_e.mean()      # < 0: true sequence scores lower
                r['native_true_beats_shuf'] = float((e[-1] < sh_e).mean())
                r['native_shuf_sd'] = sh_e.std()
                if -2 in e.index:
                    r['native_true_minus_ala'] = e[-1] - e[-2]
            for ctrl in ('ctrl_helix', 'ctrl_hairpin'):
                if f'E_{ctrl}' not in x:
                    continue
                dE = x[f'E_{ctrl}'] - x['E_native']                    # > 0: native preferred
                true, sh = dE[x.shuffle == -1].iloc[0], dE[x.shuffle >= 0].dropna()
                if (x.shuffle == -2).any():
                    r[f'{ctrl}_dE_ala'] = dE[x.shuffle == -2].iloc[0]
                if np.isfinite(true) and len(sh) >= 5:
                    r[f'{ctrl}_dE_true'] = true
                    r[f'{ctrl}_dE_shuf'] = sh.mean()
                    r[f'{ctrl}_spec_z'] = (true - sh.mean()) / (sh.std() + 1e-9)
                    r[f'{ctrl}_true_beats_all'] = float((true > sh).mean())
            res.append(r)
    df = pd.DataFrame(res)
    df.to_csv(os.path.join(out_dir, 'shuffle_per_target.csv'), index=False)
    pd.set_option('display.width', 220)
    cols = [c for c in df.columns if c.startswith(('ctrl_helix', 'ctrl_hairpin', 'native_'))]
    for grp, g in [('all', df)] + [(s, df[df.ss == s]) for s in ('helix', 'hairpin', 'coil', 'mixed')]:
        print(f'\n=== {grp}: medians (dE>0 = native preferred; spec_z>0 = true sequence more native-specific than shuffles) ===')
        print(g.groupby('model')[cols].median().round(3).to_string())


def main():
    if '--analyze' in sys.argv:
        out = sys.argv[sys.argv.index('--out_dir') + 1] if '--out_dir' in sys.argv else 'eval/seq_shuffle'
        analyze(out)
        return
    import options
    parser = options.get_decoy_parser()
    parser.add_argument('--targets', required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--out_dir', default='eval/seq_shuffle')
    parser.add_argument('--n_shuffle', type=int, default=20)
    score(parser.parse_args())


if __name__ == '__main__':
    main()
