"""
Exercise LocalEnergyCE.forward along two paths with a synthetic batch:

  (a) Flags OFF -> expects 6-tuple return, no extras threaded.
  (b) Flags ON  -> forward accepts esm / coords_cart / seq_offset without
                   shape errors and still returns a finite 6-tuple.

This is a CPU-only test; it does not touch h5 data or the optimizer.
"""
import os
import sys
import types
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
NNEF_DIR = os.path.dirname(THIS_DIR)
if NNEF_DIR not in sys.path:
    sys.path.insert(0, NNEF_DIR)

from model.local_ss import LocalTransformer, LocalEnergyCE  # noqa: E402


def make_args(**overrides):
    args = types.SimpleNamespace(
        seq_type='residue',
        residue_type_num=20,
        seq_len=14,
        embed_size=32,
        dim=128,
        n_layers=4,
        attn_heads=4,
        dropout=0.0,
        mixture_r=2,
        mixture_angle=3,
        mixture_rama=2,
        mixture_seq=1,
        mixture_res_counts=1,
        device='cpu',
        random_ref=False,
        smooth_gaussian=True,
        smooth_r=0.3,
        smooth_angle=15.0,
        reduction='sum_all',
        profile_prob=False,
        profile_loss_lamda=1.0,
        coords_angle_loss_lamda=1.0,
        coords_rama_loss_lamda=1.0,
        use_position_weights=False,
        cen_seg_loss_lamda=1.0,
        oth_seg_loss_lamda=1.0,
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def make_batch(args, N=4):
    L = args.seq_len + 1
    seq = torch.randint(0, args.residue_type_num, (N, L), dtype=torch.long)
    coords = torch.randn(N, L, 3).abs() + 1.0  # keep r > 0
    # The spherical (theta, phi) convention expects theta in [0, pi], phi in (-pi, pi].
    # But LocalEnergyCE computes NLL on (r, theta, phi); we just need finite.
    start_id = torch.randint(0, 2, (N, L), dtype=torch.long)
    res_counts = torch.zeros(N, 3)
    rama = torch.rand(N, L, 2) * 2 * torch.pi - torch.pi
    rama_mask = torch.ones(N, L)
    return seq, coords, start_id, res_counts, rama, rama_mask


def run(args, label, with_extras=False):
    torch.manual_seed(42)
    model = LocalTransformer(args).eval()
    energy = LocalEnergyCE(model, args)

    seq, coords, start_id, res_counts, rama, rama_mask = make_batch(args)
    kwargs = dict(rama=rama, rama_mask=rama_mask)
    if with_extras:
        N, L, _ = coords.shape
        kwargs.update(
            esm=torch.zeros(N, L, args.esm_dim_in),
            coords_cart=torch.zeros(N, L, 3),
            seq_offset=torch.zeros((N, L), dtype=torch.long),
        )
        if getattr(args, 'use_dihedral', False):
            kwargs['dihedral'] = torch.zeros(N, L, 4)
    out = energy.forward(seq, coords, start_id, res_counts, **kwargs)
    assert len(out) == 6, f"{label}: expected 6 losses, got {len(out)}"
    for i, t in enumerate(out):
        assert torch.isfinite(t).all(), f"{label}: loss[{i}] not finite: {t}"
    total = sum(t for t in out).item()
    print(f"[{label}] losses = " + ", ".join(f"{t.item():.4f}" for t in out) + f"; total={total:.4f}")


def main():
    run(make_args(), label="baseline (flags OFF)", with_extras=False)
    args_ext = make_args(
        use_esm=True, esm_dim_in=64, esm_dim_out=16,
        use_cart_coords=True,
        use_seq_offset=True, seq_offset_max=16,
    )
    run(args_ext, label="extended (flags ON, zero extras)", with_extras=True)

    # Full-stack run including dihedral: the 4-d sin/cos input is additive and
    # zero-init, so a zero-tensor dihedral gives the same total as the run
    # above up to whatever noise initialization chose for linear_x_dihedral
    # (none -- it's zero-init). We only check that the path runs finite.
    args_full = make_args(
        use_esm=True, esm_dim_in=64, esm_dim_out=16,
        use_cart_coords=True,
        use_seq_offset=True, seq_offset_max=16,
        use_dihedral=True,
    )
    run(args_full, label="extended + dihedral (flags ON, zero extras)",
        with_extras=True)

    print("ALL TRAINER-PATH CHECKS PASSED")


if __name__ == '__main__':
    main()
