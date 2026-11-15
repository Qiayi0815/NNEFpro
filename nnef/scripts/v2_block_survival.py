"""Does the v2 20-A block cutoff create survivor bias? (ribbon diagnostic)

For 3DRobot targets, count kept v2 blocks per decoy (blocks whose local span
>= struct_dist_cutoff are DROPPED before the per-block mean energy). If worse
decoys (higher RMSD) lose more blocks, the mean is taken over a favourable
subset -> bias toward low energy for bad decoys.
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

_NNEF = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _NNEF)
import options  # noqa: E402
from protein_os import Protein, ProteinBase  # noqa: E402
from utils import load_protein_decoy, test_setup  # noqa: E402

parser = options.get_decoy_parser()
parser.add_argument('--n_targets', type=int, default=25)
args = options.parse_args_and_arch(parser)
device, model, energy_fn, _ = test_setup(args)
ProteinBase.v2_fast = True
targets = [t.strip() for t in open('fasrc/targets_3drobot.txt')][:args.n_targets]
r_frac, r_nan = [], []
for t in targets:
    lst = pd.read_csv(f'nnef/data/decoys/3DRobot_set/{t}/list.csv')
    fr, rm, nn = [], [], 0
    for name, rmsd in zip(lst.NAME, lst.RMSD):
        did = name[:-4] if name.endswith('.pdb') else name
        try:
            seq, coords, profile, dih, n, ca, c, cgn = load_protein_decoy(t, did, args.mode, device, args)
        except Exception:
            continue
        if n is None or not torch.isfinite(torch.stack([n, ca, c, coords])).all():
            nn += 1
            continue
        prot = Protein(seq, coords, profile, dihedral_full=dih, n_coords=n, ca_coords=ca, c_coords=c,
                       chain_group_num=cgn)
        with torch.no_grad():
            kept = prot.get_local_struct()[4].shape[0]
        fr.append(kept / coords.shape[0]); rm.append(rmsd)
    fr, rm = np.array(fr), np.array(rm)
    rr = pearsonr(fr, rm)[0] if fr.std() > 0 else float('nan')
    r_frac.append(rr); r_nan.append(nn)
    print(f'{t}: decoys {len(fr)} (non-finite backbone {nn})  kept-block fraction mean {fr.mean():.3f} '
          f'min {fr.min():.3f}  corr(kept_frac, RMSD) {rr:+.2f}')
print(f'[survival] median corr(kept_frac, RMSD) over {len(r_frac)} targets: {np.nanmedian(r_frac):+.2f}; '
      f'targets with any non-finite backbone decoys: {sum(x > 0 for x in r_nan)}')
