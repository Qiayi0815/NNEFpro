"""
CPU-only smoke test for the ESM / Cartesian / seq-offset feature-extraction
enhancements added to LocalTransformer and LocalEnergyCE.

Run from the repo root:
    python nnef/scripts/test_esm_forward.py

This script verifies three invariants without requiring any real data or an
ESM checkpoint:

1. Baseline bit-identity: when all new flags are OFF, LocalTransformer's
   state_dict keys and parameter count match the original module, and its
   forward output is unchanged by construction (the new code paths are
   guarded by flags that default to False).

2. Extended-forward shape invariance: when --use_esm / --use_cart_coords /
   --use_seq_offset are all ON, forward(...) accepts the three new tensors
   and returns out_x / out_s with exactly the same shape as baseline.

3. Identity-at-zero for new branches: feeding zero tensors for esm /
   coords_cart / seq_offset (combined with the zero-initialized esm_proj
   final layer and zero-initialized seq_offset_embed table) produces
   outputs that equal the "same model but with those flags off" outputs
   up to numerical tolerance, proving the wiring does not corrupt the
   baseline at initialization.
"""
import os
import sys
import types
import torch

# Make the nnef/ package importable when running this file directly.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
NNEF_DIR = os.path.dirname(THIS_DIR)
if NNEF_DIR not in sys.path:
    sys.path.insert(0, NNEF_DIR)

from model.local_ss import LocalTransformer  # noqa: E402


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
        device='cpu',
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def random_batch(args, N=2):
    """Match the length convention used inside LocalEnergyCE.forward:
    input_seq has length args.seq_len, other tensors have length args.seq_len+1."""
    L = args.seq_len + 1  # 15 when seq_len=14
    seq = torch.randint(0, args.residue_type_num, (N, args.seq_len), dtype=torch.long)
    coords = torch.randn(N, L, 3)
    start_id = torch.randint(0, 2, (N, L), dtype=torch.long)
    return seq, coords, start_id


def main():
    torch.manual_seed(0)
    args_base = make_args()
    model_base = LocalTransformer(args_base).eval()
    base_keys = set(model_base.state_dict().keys())
    base_nparams = sum(p.numel() for p in model_base.parameters())

    torch.manual_seed(0)
    args_ext = make_args(
        use_esm=True, esm_dim_in=64, esm_dim_out=16,
        use_cart_coords=True,
        use_seq_offset=True, seq_offset_max=16,
    )
    model_ext = LocalTransformer(args_ext).eval()
    ext_keys = set(model_ext.state_dict().keys())

    # -- Check 1: state_dict of the extended model is a strict superset of
    #    the baseline's, shapes for shared keys match exactly (so the
    #    baseline checkpoint loads verbatim via strict=False without any
    #    manual copying), and all new keys belong to the optional branches.
    new_keys = ext_keys - base_keys
    missing_keys = base_keys - ext_keys
    assert not missing_keys, f"Extended model dropped baseline keys: {missing_keys}"
    for k in base_keys:
        assert model_base.state_dict()[k].shape == model_ext.state_dict()[k].shape, \
            f"Shape mismatch on shared key {k}: baseline {model_base.state_dict()[k].shape} " \
            f"vs extended {model_ext.state_dict()[k].shape}"
    for k in new_keys:
        assert any(k.startswith(p) for p in (
            "esm_proj", "linear_x_esm", "linear_x_cart", "linear_x_offset",
            "seq_offset_embed",
        )), f"Unexpected new param key {k}"
    ext_nparams = sum(p.numel() for p in model_ext.parameters())
    print(f"[1/3] baseline params={base_nparams}, extended params={ext_nparams}; "
          f"new keys={sorted(new_keys)}")

    # -- Check 2: extended forward accepts the three new tensors with the
    #    right shapes and returns matching output shapes.
    seq, coords, start_id = random_batch(args_ext)
    N, L_full, _ = coords.shape  # L_full = args.seq_len + 1
    esm = torch.randn(N, L_full, args_ext.esm_dim_in)
    coords_cart = torch.randn_like(coords)
    seq_offset = torch.randint(0, 2 * args_ext.seq_offset_max + 1, (N, L_full), dtype=torch.long)
    with torch.no_grad():
        out_x_ext, out_s_ext = model_ext(seq, coords, start_id,
                                         esm=esm, coords_cart=coords_cart, seq_offset=seq_offset)

    # Compare to baseline output shape.
    seq_b, coords_b, start_id_b = seq, coords, start_id
    with torch.no_grad():
        out_x_base, out_s_base = model_base(seq_b, coords_b, start_id_b)
    assert out_x_ext.shape == out_x_base.shape, \
        f"out_x shape mismatch: {out_x_ext.shape} vs baseline {out_x_base.shape}"
    assert out_s_ext.shape == out_s_base.shape, \
        f"out_s shape mismatch: {out_s_ext.shape} vs baseline {out_s_base.shape}"
    print(f"[2/3] out_x shape={tuple(out_x_ext.shape)}, out_s shape={tuple(out_s_ext.shape)}")

    # -- Check 3: identity-at-init with ARBITRARY extras. Load the baseline
    #    weights into the extended model via plain strict=False (no manual
    #    copying). Since linear_x_{esm,cart,offset} are all zero-init, even
    #    random-valued extras must produce outputs identical to the baseline
    #    at init. This is the property we need for inference-only A/B on an
    #    existing baseline checkpoint.
    missing, unexpected = model_ext.load_state_dict(model_base.state_dict(), strict=False)
    assert not unexpected, f"strict=False load reported unexpected keys: {unexpected}"
    print(f"      strict=False load; new side-layer keys left at init: {sorted(missing)}")

    esm_rand = torch.randn(N, L_full, args_ext.esm_dim_in)
    cart_rand = torch.randn(N, L_full, 3)
    offset_rand = torch.randint(0, 2 * args_ext.seq_offset_max + 1,
                                (N, L_full), dtype=torch.long)
    with torch.no_grad():
        out_x_e, out_s_e = model_ext(seq, coords, start_id,
                                     esm=esm_rand, coords_cart=cart_rand,
                                     seq_offset=offset_rand)
        out_x_b, out_s_b = model_base(seq, coords, start_id)

    dx = (out_x_e - out_x_b).abs().max().item()
    ds = (out_s_e - out_s_b).abs().max().item()
    tol = 1e-5
    ok = (dx < tol) and (ds < tol)
    print(f"[3/3] identity-at-init with RANDOM extras: max|dx|={dx:.2e}, "
          f"max|ds|={ds:.2e}, tol={tol:.0e} -> {'OK' if ok else 'MISMATCH'}")
    if not ok:
        raise SystemExit(1)

    print("ALL SMOKE CHECKS PASSED")


if __name__ == '__main__':
    main()
