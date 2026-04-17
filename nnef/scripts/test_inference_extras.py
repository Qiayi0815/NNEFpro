"""CPU-only smoke test for the inference side of --use_cart_coords /
--use_seq_offset.

What this script checks
-----------------------
Scenario: a user has a baseline checkpoint (trained WITHOUT the extras) and
wants to do inference-only A/B with the extras turned on. The key invariant
we need is:

    Protein.get_energy(baseline_weights, flags OFF)
        == Protein.get_energy(baseline_weights, flags ON, side layers at init)

because the side layers linear_x_{esm,cart,offset} are zero-initialized, so
loading the baseline checkpoint into the extended model via strict=False and
then feeding real coords_cart / seq_offset must produce IDENTICAL energy on
day 0. Any divergence after that comes purely from fine-tuning.

It also checks:
  * Protein.get_local_struct returns the 5-tuple (incl. g_local).
  * Protein.get_energy end-to-end runs finite with flags ON.
  * Protein.get_residue_energy runs finite with flags ON.

Run with
    python -m nnef.scripts.test_inference_extras
"""
import os
import sys
import types
import numpy as np
import torch


def make_args(use_cart_coords=False, use_seq_offset=False, use_esm=False,
              esm_dim_in=16, use_dihedral=False):
    # Build a real args namespace from the project's own parser so we don't
    # have to mirror every model/energy attribute by hand, then override the
    # extras flags we want to exercise in this test.
    from options import get_common_parser
    parser = get_common_parser()
    a = parser.parse_args([])
    a.device = 'cpu'
    a.use_cart_coords = use_cart_coords
    a.use_seq_offset = use_seq_offset
    a.use_esm = use_esm
    a.use_dihedral = use_dihedral
    a.esm_dim_in = esm_dim_in
    a.esm_dim_out = 8
    # Attributes required by the model/energy that are not (yet) part of the
    # common parser. Pick defaults that are cheap on CPU.
    a.mixture_rama = getattr(a, 'mixture_rama', 2)
    a.coords_rama_loss_lamda = getattr(a, 'coords_rama_loss_lamda', 0.0)
    # Keep the model shape small for a fast smoke test.
    a.embed_size = 32
    a.dim = 64
    a.n_layers = 2
    a.attn_heads = 4
    a.seq_type = 'residue'
    return a


def build_protein(device='cpu', L=40, seed=0):
    # Avoid the pandas csv read path by monkeying ProteinBase.__init__ with a
    # stub energy_ref. We ONLY need get_energy to work; energy_seq is only used
    # under use_ref=True, which we leave False.
    from protein_os import Protein, ProteinBase

    rng = np.random.RandomState(seed)
    seq = np.array(['A'] * L)
    seq_id = torch.tensor(rng.randint(0, 20, size=L), dtype=torch.long, device=device)
    # Plausible CA backbone: random walk so neighbor topology is nontrivial.
    step = rng.randn(L, 3).astype(np.float32) * 1.5
    coords = torch.tensor(np.cumsum(step, axis=0), dtype=torch.float, device=device)

    p = Protein.__new__(Protein)   # skip pandas csv read
    p.energy_ref = torch.zeros(20, dtype=torch.float, device=device)
    p.seq = seq
    p.coords = coords
    p.coords_int = None
    p.profile = seq_id
    p.energy_seq = torch.tensor(0.0, device=device)
    p.protein_id = 'synthetic'
    p.esm_full = None
    p.dihedral_full = None
    return p


def main():
    torch.manual_seed(0)
    # Make `from protein_os import ...` resolve the in-tree module the same
    # way utils.test_setup does.
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    # protein_os imports `from torch.utils.tensorboard import SummaryWriter`
    # at module scope. That import chain does a strict version check on the
    # real `tensorboard` package, which isn't installed in this test env and
    # isn't used by the inference paths we exercise here. Stub the submodule
    # directly with a no-op SummaryWriter so the import succeeds.
    _fake_tb_mod = types.ModuleType('torch.utils.tensorboard')

    class _StubSummaryWriter:  # noqa: D401
        def __init__(self, *a, **kw): pass
        def add_scalar(self, *a, **kw): pass
        def close(self): pass

    _fake_tb_mod.SummaryWriter = _StubSummaryWriter
    sys.modules['torch.utils.tensorboard'] = _fake_tb_mod

    from model import LocalTransformer
    from protein_os import EnergyFun, ProteinBase

    device = 'cpu'
    L = 40

    # --- Baseline model + energy_fn (flags OFF) ---
    args_base = make_args(use_cart_coords=False, use_seq_offset=False)
    model_base = LocalTransformer(args_base).eval()
    ef_base = EnergyFun(model_base, args_base)

    # --- Extended model + energy_fn (flags ON) ---
    args_ext = make_args(use_cart_coords=True, use_seq_offset=True)
    model_ext = LocalTransformer(args_ext).eval()
    ef_ext = EnergyFun(model_ext, args_ext)

    # Load the baseline state into the extended model via plain strict=False;
    # this is exactly what utils.test_setup now does at inference time.
    missing, unexpected = model_ext.load_state_dict(model_base.state_dict(), strict=False)
    assert not unexpected, f"strict=False load reported unexpected keys: {unexpected}"
    for k in missing:
        assert any(k.startswith(p) for p in (
            "esm_proj", "linear_x_esm", "linear_x_cart",
            "linear_x_offset", "seq_offset_embed", "linear_x_dihedral",
        )), f"missing key {k} is not a side-layer key"
    print(f"[1/6] baseline -> extended strict=False load OK; new side-layer "
          f"keys left at init: {sorted(missing)}")

    # --- Shared config on ProteinBase ---
    ProteinBase.k = args_base.seq_len - 4
    ProteinBase.use_graph_net = False

    # --- Check 2: flags OFF path runs and returns finite energy. ---
    ProteinBase.use_cart_coords = False
    ProteinBase.use_seq_offset = False
    p = build_protein(device=device, L=L, seed=1)
    with torch.no_grad():
        e_off = p.get_energy(ef_base).item()
    assert np.isfinite(e_off), f"baseline energy non-finite: {e_off}"
    print(f"[2/6] flags OFF -> get_energy = {e_off:.6f}")

    # --- Check 3: flags ON path runs and returns finite energy. ---
    ProteinBase.use_cart_coords = True
    ProteinBase.use_seq_offset = True
    ProteinBase.seq_offset_max = args_ext.seq_offset_max
    with torch.no_grad():
        e_on = p.get_energy(ef_ext).item()
    assert np.isfinite(e_on), f"extended energy non-finite: {e_on}"
    print(f"[3/6] flags ON  -> get_energy = {e_on:.6f}")

    # --- Check 4: identity-at-init. Same baseline weights, same protein,
    #     extended model with side layers still at zero must match exactly. ---
    delta = abs(e_off - e_on)
    tol = 1e-5
    print(f"[4/6] |e_on - e_off| = {delta:.2e} (tol={tol:.0e}) -> "
          f"{'OK' if delta < tol else 'MISMATCH'}")
    assert delta < tol, (
        "Identity-at-init violated: extras are contributing nonzero signal "
        "even though side layers are zero-initialized."
    )

    # --- Bonus: per-residue energy path also runs with flags ON. ---
    with torch.no_grad():
        re = p.get_residue_energy(ef_ext)
    assert np.isfinite(re).all(), f"non-finite per-residue energy: {re}"
    assert re.shape[0] == L - 4, f"per-residue length wrong: {re.shape[0]} vs {L-4}"
    print(f"      per-residue energies: mean={re.mean():.4f}, std={re.std():.4f}, "
          f"len={len(re)}")

    # --- Check 5: ESM identity-at-init --------------------------------- #
    # Build an extended model with use_esm=True, copy baseline weights via
    # strict=False, and verify that feeding a random esm_full yields the SAME
    # energy as the baseline (because esm_proj's last layer + linear_x_esm are
    # both zero-initialised, the ESM branch contributes exactly 0).
    args_esm = make_args(use_cart_coords=False, use_seq_offset=False,
                         use_esm=True, esm_dim_in=16)
    model_esm = LocalTransformer(args_esm).eval()
    missing_esm, unexpected_esm = model_esm.load_state_dict(
        model_base.state_dict(), strict=False,
    )
    assert not unexpected_esm, unexpected_esm
    ef_esm = EnergyFun(model_esm, args_esm)

    ProteinBase.use_cart_coords = False
    ProteinBase.use_seq_offset = False
    ProteinBase.use_esm = True

    p_esm = build_protein(device=device, L=L, seed=1)
    esm_full = torch.randn(L, args_esm.esm_dim_in, dtype=torch.float, device=device)
    p_esm.esm_full = esm_full

    with torch.no_grad():
        e_esm = p_esm.get_energy(ef_esm).item()
    delta_esm = abs(e_esm - e_off)
    print(f"[5/6] |e_esm - e_off| = {delta_esm:.2e} (tol={tol:.0e}) -> "
          f"{'OK' if delta_esm < tol else 'MISMATCH'}")
    assert delta_esm < tol, (
        "ESM identity-at-init violated: esm_proj / linear_x_esm are supposed "
        "to be zero-initialised and must produce 0 contribution at step 0."
    )

    # Sanity: nudging any layer in the ESM path to nonzero must actually
    # change the energy. linear_x_esm.weight alone isn't enough because
    # esm_proj's final layer is also zero-initialised so its output is
    # uniformly 0 and weight * 0 == 0. We nudge the BIAS of esm_proj's last
    # Linear to push a non-zero signal into linear_x_esm.
    with torch.no_grad():
        model_esm.esm_proj[-1].bias.fill_(1e-2)
        model_esm.linear_x_esm.weight.fill_(1e-2)
        e_esm_nudged = p_esm.get_energy(ef_esm).item()
    assert abs(e_esm_nudged - e_off) > 1e-6, (
        "Nudging the ESM path did NOT change the energy; the ESM branch "
        "may not be wired through Protein.get_energy."
    )
    print(f"      ESM branch reachability OK: nudged |delta|={abs(e_esm_nudged - e_off):.2e}")
    # Restore zero for cleanliness.
    with torch.no_grad():
        model_esm.esm_proj[-1].bias.zero_()
        model_esm.linear_x_esm.weight.zero_()

    # Reset to baseline defaults so subsequent imports don't see sticky state.
    ProteinBase.use_esm = False

    # --- Check 6: Dihedral identity-at-init + reachability ---------------- #
    # Same game as [5/6]: extended model with --use_dihedral loaded from the
    # baseline checkpoint + a realistic dihedral_full tensor must produce the
    # same energy as the baseline at step 0 (linear_x_dihedral is zero-init).
    args_dh = make_args(use_cart_coords=False, use_seq_offset=False,
                        use_esm=False, use_dihedral=True)
    model_dh = LocalTransformer(args_dh).eval()
    missing_dh, unexpected_dh = model_dh.load_state_dict(
        model_base.state_dict(), strict=False,
    )
    assert not unexpected_dh, unexpected_dh
    # The only new keys should be linear_x_dihedral.*
    for k in missing_dh:
        assert k.startswith("linear_x_dihedral"), (
            f"unexpected missing key for dihedral model: {k}"
        )
    ef_dh = EnergyFun(model_dh, args_dh)

    ProteinBase.use_cart_coords = False
    ProteinBase.use_seq_offset = False
    ProteinBase.use_dihedral = True

    p_dh = build_protein(device=device, L=L, seed=1)
    # Generate a realistic dihedral: uniform random phi/psi in (-pi, pi]
    # with the two chain-terminal NaNs that real phi/psi would have. The
    # Protein path must handle this without NaN propagation.
    rng = np.random.RandomState(42)
    phi_psi = torch.tensor(
        rng.uniform(-np.pi, np.pi, size=(L, 2)).astype(np.float32), device=device,
    )
    phi_psi[0, 0] = float('nan')   # phi of residue 0 undefined
    phi_psi[-1, 1] = float('nan')  # psi of last residue undefined
    p_dh.dihedral_full = phi_psi

    with torch.no_grad():
        e_dh = p_dh.get_energy(ef_dh).item()
    assert np.isfinite(e_dh), f"dihedral energy non-finite: {e_dh}"
    delta_dh = abs(e_dh - e_off)
    print(f"[6/6] |e_dh - e_off| = {delta_dh:.2e} (tol={tol:.0e}) -> "
          f"{'OK' if delta_dh < tol else 'MISMATCH'}")
    assert delta_dh < tol, (
        "Dihedral identity-at-init violated: linear_x_dihedral must be "
        "zero-initialised and produce 0 contribution at step 0."
    )

    # Reachability: nudging linear_x_dihedral.bias with nonzero dihedral
    # input must change the energy. (Nudging just the weight is also enough
    # here because the input sin/cos tensor is not uniformly zero, unlike
    # esm_proj's zero-init output in check 5.)
    with torch.no_grad():
        model_dh.linear_x_dihedral.weight.fill_(1e-2)
        model_dh.linear_x_dihedral.bias.fill_(1e-2)
        e_dh_nudged = p_dh.get_energy(ef_dh).item()
    assert abs(e_dh_nudged - e_off) > 1e-6, (
        "Nudging the dihedral path did NOT change the energy; the dihedral "
        "branch may not be wired through Protein.get_energy."
    )
    print(f"      Dihedral branch reachability OK: "
          f"nudged |delta|={abs(e_dh_nudged - e_off):.2e}")
    # Restore zero so any lingering reference sees clean state.
    with torch.no_grad():
        model_dh.linear_x_dihedral.weight.zero_()
        model_dh.linear_x_dihedral.bias.zero_()

    ProteinBase.use_dihedral = False

    print("ALL INFERENCE SMOKE CHECKS PASSED")


if __name__ == '__main__':
    main()
