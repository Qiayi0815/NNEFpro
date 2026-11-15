"""Smooth neighbour blocks (--smooth_nbr): make the block energy continuous in x.

The hard local block (5-residue window + k nearest neighbours, distance-ordered
slots, slot-index position embeddings, start_id connectivity) makes E(x)
piecewise: it jumps when two neighbours swap distance order or when the k-th
and (k+1)-th neighbours trade places (eval/md_eval/jump: ~60% of |dE| along
descent paths sits on these steps). Three changes remove every jump:

1. Order. The first ``n_fixed`` slots (centre + window, already fixed by the
   block builder) stay put; the neighbour slots are sorted by SEQUENCE index,
   not distance, and get a sequence-offset embedding instead of a slot-index
   position embedding. Distance swaps no longer reorder anything.
2. Set. Neighbour j gets a switching weight
       s_j = smoothstep((d_(k) - d_j) / tau),  d = Cb-Cb distance to the centre,
   anchored at d_(k), the distance of the k-th (farthest kept) neighbour. The
   farthest kept neighbour always has s = 0, so when it trades places with the
   (k+1)-th residue both sit at the anchor with s = 0: the set change is
   invisible. d_(k) is an order statistic, hence continuous in x.
3. Use. Attention to slot j gets the additive key bias log(s_j) and its
   coordinate terms are weighted by s_j (sums, not means over a varying count),
   so a slot with s -> 0 drops out continuously. start_id (which depends on the
   slot before it) is dropped; the offset embedding carries connectivity.

The transform runs inside LocalEnergyCE.forward, i.e. identically in training
and inference. Works on any block whose slot 0 is the centre and whose first
``n_fixed`` slots are the window (v2 extractor, legacy builder).
"""
import torch

N_FIXED = 5
# Offset buckets: exact for |o| <= 8, then two buckets per octave, capped.
_EXACT = 8
_N_LOG = 8
N_BUCKETS = 2 * (_EXACT + _N_LOG) + 1


def offset_bucket(offset):
    """Signed sequence offset (long, any size) -> bucket index in [0, N_BUCKETS)."""
    a = offset.abs()
    far = _EXACT + torch.clamp(torch.ceil(2 * torch.log2(a.clamp(min=1).float() / _EXACT)).long(), 1, _N_LOG)
    b = torch.where(a <= _EXACT, a, far)
    return torch.sign(offset) * b + _EXACT + _N_LOG


def smoothstep(u):
    u = u.clamp(0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def smooth_layout(coords_cart, offset, tau, n_fixed=N_FIXED):
    """coords_cart (N, L, 3) block-local Cartesian Cb; offset (N, L) long, raw
    sequence offset to the centre (slot 0). Returns
      perm (N, L) long  -- new slot order (fixed slots, then neighbours by offset)
      s    (N, L)       -- switching weight in the NEW order (1 for fixed slots)
    """
    N, L, _ = coords_cart.shape
    d = (coords_cart - coords_cart[:, :1]).norm(dim=-1)              # (N, L)
    d_nb = d[:, n_fixed:]
    anchor = d_nb.max(dim=1, keepdim=True).values
    s_nb = smoothstep((anchor - d_nb) / tau)
    s = torch.cat([torch.ones_like(d[:, :n_fixed]), s_nb], dim=1)
    order = torch.argsort(offset[:, n_fixed:], dim=1, stable=True) + n_fixed
    fixed = torch.arange(n_fixed, device=offset.device).expand(N, n_fixed)
    perm = torch.cat([fixed, order], dim=1)
    return perm, torch.gather(s, 1, perm)


def permute(x, perm):
    """Gather slot dimension (dim 1) of x (N, L, ...) by perm (N, L)."""
    if x is None:
        return None
    idx = perm
    for _ in range(x.dim() - 2):
        idx = idx.unsqueeze(-1)
    return torch.gather(x, 1, idx.expand(*perm.shape, *x.shape[2:]))


def key_bias_mask(s, n_heads, causal=None):
    """(N*h, L, L) additive attention mask: log(s_j) on key j (+ optional causal).

    The softmax weight of key j is then proportional to s_j, so it reaches 0
    exactly as s_j does. The floor (1e-30) only guards log(0); a soft floor such
    as log(s + 1e-6) leaves a weight of 1e-6 * exp(score gap), which is not small
    for large attention logits. Slot 0 (s = 1) is visible to every query, so no
    row is fully masked."""
    bias = torch.log(s.clamp(min=1e-30))[:, None, :]                # (N, 1, L)
    m = bias.expand(-1, s.size(1), -1)
    if causal is not None:
        m = m + causal[None]
    return m.repeat_interleave(n_heads, dim=0)
