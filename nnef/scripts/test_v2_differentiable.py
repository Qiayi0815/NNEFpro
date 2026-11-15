"""Equivalence + gradient check for the differentiable v2 (N-CA-C) frame path.

1. Re-score a few ribbon decoys with the new torch path; energies must match the
   stored E1 decomposition (numpy path) to ~1e-3.
2. Gradients must be non-zero for CB, N, CA and C.
3. dynamic_rama (live phi/psi) must reproduce the cached-dihedral energy.
"""
import os
import sys

import pandas as pd
import torch

_NNEF = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _NNEF)
import options  # noqa: E402
from protein_os import Protein, ProteinBase  # noqa: E402
from utils import load_protein_decoy, test_setup  # noqa: E402

parser = options.get_decoy_parser()
args = options.parse_args_and_arch(parser)
device, model, energy_fn, _ = test_setup(args)
target = 'T1026' if args.decoy_set == 'casp14' else '1BYIA'
csv = f'nnef/data/decoys/{args.decoy_set}/decoy_loss_decomp_ribbon_full1000_27329320/{target}_decoy_loss.csv'
ref = pd.read_csv(csv).set_index('NAME')
names = list(ref.index[:5])
for name in names:
    decoy_id = name[:-4] if name.endswith('.pdb') else name
    seq, coords, profile, dih, n_xyz, ca_xyz, c_xyz, cgn = load_protein_decoy(target, decoy_id, args.mode, device, args)
    x = [t.clone().requires_grad_(True) for t in (coords, n_xyz, ca_xyz, c_xyz)]
    prot = Protein(seq, x[0], profile, dihedral_full=dih, n_coords=x[1], ca_coords=x[2], c_coords=x[3],
                   chain_group_num=cgn)
    ProteinBase.dynamic_rama = False
    e = prot.get_energy(energy_fn)
    e.backward()
    g = [float(t.grad.norm()) for t in x]
    ProteinBase.dynamic_rama = True
    with torch.no_grad():
        e_dyn = float(prot.get_energy(energy_fn))
    ProteinBase.dynamic_rama = False
    print(f'{name}: E_new {float(e):.4f}  E_stored {ref.loc[name, "loss"]:.4f}  '
          f'diff {float(e) - ref.loc[name, "loss"]:+.2e} | E_dynamic_rama {e_dyn:.4f} | '
          f'|grad| CB {g[0]:.3g} N {g[1]:.3g} CA {g[2]:.3g} C {g[3]:.3g}')

# ---- 4. vectorised v2_fast must reproduce the reference blocks exactly ----
import time  # noqa: E402
for name in list(ref.index[:8]):
    decoy_id = name[:-4] if name.endswith('.pdb') else name
    seq, coords, profile, dih, n_xyz, ca_xyz, c_xyz, cgn = load_protein_decoy(target, decoy_id, args.mode, device, args)
    prot = Protein(seq, coords, profile, dihedral_full=dih, n_coords=n_xyz, ca_coords=ca_xyz, c_coords=c_xyz,
                   chain_group_num=cgn)
    with torch.no_grad():
        ProteinBase.v2_fast = False
        t0 = time.time(); a = prot.get_local_struct(); ea = float(prot.get_energy(energy_fn)); ta = time.time() - t0
        ProteinBase.v2_fast = True
        t0 = time.time(); b = prot.get_local_struct(); eb = float(prot.get_energy(energy_fn)); tb = time.time() - t0
        ProteinBase.v2_fast = False
    same_shape = a[4].shape == b[4].shape
    ok = same_shape and torch.equal(a[4], b[4]) and torch.equal(a[2].long(), b[2].long()) \
        and torch.allclose(a[3].float(), b[3].float()) and torch.allclose(a[1], b[1], atol=1e-4)
    print(f'[fast] {name}: blocks {tuple(a[4].shape)} vs {tuple(b[4].shape)} identical={ok} '
          f'E_ref {ea:.4f} E_fast {eb:.4f} | time ref {ta*1e3:.0f} ms fast {tb*1e3:.0f} ms')
