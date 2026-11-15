"""How discontinuous is the learned energy? (validation point 4)

Hard top-k neighbour selection and distance-ordered slots make E(x) piecewise
smooth: it jumps whenever a block's neighbour SET or ORDER changes. Along the
straight line (all four beads N, CA, C, Cβ) from the native structure to the
endpoint of a T=0 Adam descent run by md_eval_bb.py (final_bb.npy), evaluate E
at n_points and split the total variation sum_i |E_{i+1} - E_i| into steps
where any block's neighbour list changed ('switch steps') and steps where it
did not. With fine sampling the smooth part per switch step is tiny, so the
switch share estimates how much of the energy change is discontinuous.

Same energy as the descent (summed NNEF energy, no bond prior), same flags as
the model's MD run:

    python nnef/scripts/energy_jump_diagnostic.py <model flags> \
        --root eval/md_eval/t0adam_heldout/yang_w1 --out eval/md_eval/jump/yang_w1.csv
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_NNEF = os.path.abspath(os.path.join(_HERE, '..'))
sys.path.insert(0, _NNEF)
sys.path.insert(0, _HERE)
import options  # noqa: E402
from md_eval_bb import load_backbone_bead  # noqa: E402
from protein_os import Protein, ProteinBase  # noqa: E402
from utils import test_setup  # noqa: E402


def main():
    parser = options.get_local_gen_parser()
    parser.add_argument('--root', required=True, help='md_eval_bb output root, searched for */meta.json')
    parser.add_argument('--n_points', type=int, default=1000)
    parser.add_argument('--out', required=True)
    args = options.parse_args_and_arch(parser)
    device, model, energy_fn, _ = test_setup(args)
    energy_fn.residue_sum = True
    ProteinBase.dynamic_rama = True
    ProteinBase.v2_fast = True
    rows = []
    for meta_path in sorted(glob.glob(f'{args.root}/**/meta.json', recursive=True)):
        d = os.path.dirname(meta_path)
        meta = json.load(open(meta_path))
        end_path = os.path.join(d, 'final_bb.npy')
        if 'native_bead' not in meta or not os.path.isfile(end_path) or meta.get('diverged'):
            continue
        seq, profile, X0, gnum = load_backbone_bead(meta['native_bead'], device)
        X1 = torch.tensor(np.load(end_path), dtype=X0.dtype, device=device)
        prot = Protein(seq, X0[3].clone(), profile, n_coords=X0[0].clone(), ca_coords=X0[1].clone(),
                       c_coords=X0[2].clone(), chain_group_num=gnum)
        E, sig = [], []
        with torch.no_grad():
            for t in np.linspace(0.0, 1.0, args.n_points):
                X = (1 - t) * X0 + t * X1
                prot.coords, prot.n_coords, prot.ca_coords, prot.c_coords = X[3], X[0], X[1], X[2]
                g_local = prot.get_local_struct()[-1]
                E.append(float(prot.get_energy(energy_fn)))
                sig.append(g_local.cpu().numpy().tobytes())
        E = np.array(E)
        dE = np.abs(np.diff(E))
        sw = np.array([sig[i] != sig[i + 1] for i in range(len(sig) - 1)])
        tv = float(dE.sum())
        rows.append(dict(cond=os.path.relpath(d, args.root).split(os.sep)[0], target=meta['target'],
                         n_res=meta['n_residues'], rmsd_end=meta['final_rmsd_to_native'],
                         n_switch_steps=int(sw.sum()), frac_switch_steps=float(sw.mean()),
                         tv_total=tv, tv_switch=float(dE[sw].sum()),
                         share_switch=float(dE[sw].sum() / max(tv, 1e-12)),
                         max_jump=float(dE.max()), median_smooth_step=float(np.median(dE[~sw])) if (~sw).any() else np.nan,
                         dE_end=float(E[-1] - E[0]),
                         monotone_frac=float(np.mean(np.diff(E) <= 0))))
        r = rows[-1]
        print(f"[jump] {r['cond']}/{r['target']}: switch steps {r['n_switch_steps']}/{len(sw)}, "
              f"|dE| share at switches {r['share_switch']:.2f}, max jump {r['max_jump']:.2f}, "
              f"E end-start {r['dE_end']:+.1f}", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    if len(df):
        print(df.groupby('cond')[['share_switch', 'frac_switch_steps', 'max_jump', 'monotone_frac']]
              .median().to_string())


if __name__ == '__main__':
    main()
