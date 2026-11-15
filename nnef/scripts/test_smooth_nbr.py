"""Unit test: --smooth_nbr makes the block energy continuous (model/smooth_nbr.py).

On a real backbone (1ZZK, v2 frame, k=10 and k=4) with randomly initialised
models, move along a straight path from native to a perturbed structure,
locate every step where some block's neighbour list changes, bisect to the
switch point and measure the energy jump across an interval of 1e-9:
  * smooth model: jump ~ 0 (only the O(1e-9) smooth change)  -> continuous
  * hard model:   jump O(0.01-1) at interior-block switches   -> discontinuous
Terminal blocks (fixed filler slots) and the 20 A block-span cutoff are known,
separately reported sources; the cutoff is disabled here.
Also checks a training-style forward/backward (finite loss and gradients) and
that with all s = 1 and no reordering needed the layout is a no-op on slot 0-4.

    python nnef/scripts/test_smooth_nbr.py
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

_NNEF = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _NNEF)
import options  # noqa: E402
from model import LocalTransformer  # noqa: E402
from model.smooth_nbr import offset_bucket, smooth_layout  # noqa: E402
from paths import data_path  # noqa: E402
from protein_os import EnergyFun, Protein, ProteinBase  # noqa: E402

AA = pd.read_csv(data_path('amino_acids.csv'))
VOCAB = {x: y - 1 for x, y in zip(AA.AA, AA.idx)}


def make_energy(k, smooth):
    argv = ['--seq_len', str(k + 4), '--mixture_rama', '10', '--seq_head', 'masked_center',
            '--seq_type', 'residue', '--residue_type_num', '20', '--mixture_seq', '1',
            '--embed_size', '32', '--dim', '128', '--n_layers', '4', '--attn_heads', '4',
            '--mixture_r', '2', '--mixture_angle', '3',
            '--use_position_weights', '--smooth_gaussian']
    if smooth:
        argv += ['--smooth_nbr', '--smooth_tau', '1.0']
    sys.argv = ['x'] + argv
    args = options.parse_args_and_arch(options.get_local_gen_parser())
    torch.manual_seed(0)
    model = LocalTransformer(args).double().eval()
    efn = EnergyFun(model, args).double()
    efn.residue_sum = True
    ProteinBase.k = k
    ProteinBase.smooth_nbr = smooth
    return efn, args


def load():
    df = pd.read_csv(data_path('yang_small', '1ZZK', '1ZZK_bead.csv')).sort_values('group_num')
    t = lambda c: torch.tensor(df[c].values, dtype=torch.float64)  # noqa: E731
    xyz = lambda a: torch.stack([t(f'x{a}'), t(f'y{a}'), t(f'z{a}')], dim=1)  # noqa: E731
    X = torch.stack([xyz('n'), xyz('ca'), xyz('c'), xyz('cb')])
    prof = torch.tensor([VOCAB[s] for s in df.group_name.values])
    return df.group_name.values, prof, X, torch.tensor(df.group_num.values)


def path_energy(prot, efn, X0, X1, t):
    X = (1 - t) * X0 + t * X1
    prot.coords, prot.n_coords, prot.ca_coords, prot.c_coords = X[3], X[0], X[1], X[2]
    with torch.no_grad():
        g = prot.get_local_struct()[-1]
        return float(prot.get_energy(efn)), g


def switch_jumps(efn, k, n=120, seed=1, max_switches=12):
    seq, prof, X0, gnum = load()
    prot = Protein(seq, X0[3].clone(), prof, n_coords=X0[0].clone(), ca_coords=X0[1].clone(),
                   c_coords=X0[2].clone(), chain_group_num=gnum)
    gen = torch.Generator().manual_seed(seed)
    X1 = X0 + 1.5 * torch.randn(X0.shape, generator=gen, dtype=X0.dtype)
    ts = np.linspace(0, 1, n)
    E, G = zip(*[path_energy(prot, efn, X0, X1, t) for t in ts])
    L = X0.shape[1]
    terminal_rows = {0, 1, L - 2, L - 1}
    out = []
    for i in range(n - 1):
        if torch.equal(G[i], G[i + 1]) or len(out) >= max_switches:
            continue
        rows = set(torch.nonzero((G[i] != G[i + 1]).any(dim=1)).flatten().tolist())
        lo, hi = ts[i], ts[i + 1]
        for _ in range(30):                       # bisect to the switch point
            mid = 0.5 * (lo + hi)
            if torch.equal(path_energy(prot, efn, X0, X1, mid)[1], G[i]):
                lo = mid
            else:
                hi = mid
        jump = abs(path_energy(prot, efn, X0, X1, hi)[0] - path_energy(prot, efn, X0, X1, lo)[0])
        out.append(('terminal' if rows <= terminal_rows else 'interior', jump))
    return out


def train_step_ok(k):
    efn, args = make_energy(k, smooth=True)
    ce = efn.energy_fn
    ce.reduction = 'mean'
    ce.train()
    N, L = 8, k + 5
    torch.manual_seed(3)
    cart = torch.randn(N, L, 3, dtype=torch.float64) * 5
    r = cart.norm(dim=-1)
    th = torch.acos((cart[..., 2] / (r + 1e-6)).clamp(-1 + 1e-6, 1 - 1e-6))
    ph = torch.atan2(cart[..., 1], cart[..., 0])
    sph = torch.stack([r, th, ph], -1)
    off = torch.cat([torch.tensor([0, -1, 1, -2, 2]), torch.randint(-60, 60, (L - 5,))]).expand(N, L).clone()
    seq = torch.randint(0, 20, (N, L))
    out = ce(seq, sph, torch.zeros(N, L, dtype=torch.long), torch.zeros(N, L, dtype=torch.float64),
             rama=torch.rand(N, L, 2, dtype=torch.float64), rama_mask=torch.ones(N, L, dtype=torch.float64),
             coords_cart=cart, seq_offset=off)
    loss = sum(o.sum() for o in out)
    loss.backward()
    grads = [p.grad for p in ce.parameters() if p.grad is not None]
    ok = torch.isfinite(loss) and all(torch.isfinite(g).all() for g in grads)
    perm, s = smooth_layout(cart, off, 1.0)
    fixed_ok = bool((perm[:, :5] == torch.arange(5)).all()) and bool((s[:, :5] == 1).all())
    anchor_zero = bool(((s[:, 5:] == 0).sum(dim=1) >= 1).all())
    b = offset_bucket(torch.tensor([-1000, -9, -8, -1, 0, 1, 8, 9, 1000]))
    return ok, fixed_ok, anchor_zero, b.tolist()


def main():
    ProteinBase.use_local_frame_v2 = True
    ProteinBase.v2_fast = True
    ProteinBase.dynamic_rama = True
    ProteinBase.struct_v2_dist_cutoff = 1e9
    torch.set_default_dtype(torch.float64)
    all_pass = True
    for k in (10, 4):
        res = {}
        for smooth in (False, True):
            efn, _ = make_energy(k, smooth)
            res[smooth] = switch_jumps(efn, k)
        for smooth in (False, True):
            j = res[smooth]
            inter = [x for c, x in j if c == 'interior']
            term = [x for c, x in j if c == 'terminal']
            tag = 'smooth' if smooth else 'hard  '
            print(f'k={k:2d} {tag}: interior switches {len(inter):3d}  max jump {max(inter, default=0):.2e}  '
                  f'median {np.median(inter) if inter else 0:.2e} | terminal-only {len(term)} '
                  f'max {max(term, default=0):.2e}')
        inter_s = [x for c, x in res[True] if c == 'interior']
        inter_h = [x for c, x in res[False] if c == 'interior']
        ok = len(inter_s) > 0 and max(inter_s) < 1e-4 and np.median(inter_h) > 1e-3
        print(f'  -> {"PASS" if ok else "FAIL"}: smooth interior jumps vanish, hard ones do not')
        all_pass &= ok
        tok, fixed_ok, anchor_zero, b = train_step_ok(k)
        print(f'  train step finite: {tok}; fixed slots untouched: {fixed_ok}; '
              f'anchor s=0 per block: {anchor_zero}; buckets {b}')
        all_pass &= bool(tok) and fixed_ok and anchor_zero
    print('ALL PASS' if all_pass else 'SOME FAILED')


if __name__ == '__main__':
    main()
