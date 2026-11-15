#!/usr/bin/env python3
"""Smoke test for the von Mises angle distribution (--angle_dist vonmises).

Isolated test of LocalEnergyCE.get_mixture_loss for BOTH modes (gaussian legacy
+ vonmises new): builds a random `out` tensor of the correct width, runs the
loss, and checks (1) all structure losses finite, (2) gradient flows back to the
network output (so the head is trainable), (3) the two modes give DIFFERENT
angle/rama losses (proving the branch is actually taken).

Run on the cluster (has torch):
    ~/envs/nnef/bin/python nnef/scripts/test_vonmises_loss.py
"""
import os, sys, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))  # nnef/
import numpy as np
import torch
from model.local_ss import LocalEnergyCE

M_R, M_ANGLE, M_RAMA = 2, 10, 10
N, L = 4, 15  # batch, block length


def make_energy(angle_dist, r_dist='gaussian'):
    e = object.__new__(LocalEnergyCE)  # skip __init__ (needs a full model)
    e.m_r, e.m_angle, e.m_rama = M_R, M_ANGLE, M_RAMA
    e.angle_dist = angle_dist
    e.r_dist = r_dist
    e.smooth_gaussian = True
    e.smooth_r, e.smooth_angle = 0.3, 15.0
    return e


def run(angle_dist, out, target_coords, rama, rama_mask, r_dist='gaussian'):
    e = make_energy(angle_dist, r_dist)
    lr, la, lrama = e.get_mixture_loss(out, target_coords, rama=rama, rama_mask=rama_mask)
    return lr, la, lrama


def main():
    torch.manual_seed(0)
    width = 3 * M_R + 6 * M_ANGLE + 6 * M_RAMA
    # get_mixture_loss is called on out_x[:, :-1, :-2]; here we pass that slice directly.
    out = torch.randn(N, L - 1, width, requires_grad=True)

    # synthetic targets: r>0, theta in [0,pi], phi in [-pi,pi], rama in [-pi,pi]
    r = torch.rand(N, L - 1, 1) * 3 + 1
    theta = torch.rand(N, L - 1, 1) * np.pi
    phi = (torch.rand(N, L - 1, 1) * 2 - 1) * np.pi
    target_coords = torch.cat([r, theta, phi], dim=-1)
    rama = (torch.rand(N, L, 2) * 2 - 1) * np.pi        # forward drops first pos -> L-1
    rama_mask = torch.ones(N, L)

    # (label, angle_dist, r_dist)
    configs = [
        ('gaussian',      'gaussian', 'gaussian'),
        ('vonmises',      'vonmises', 'gaussian'),
        ('vmf',           'vmf',      'gaussian'),
        ('lognormal_r',   'gaussian', 'lognormal'),
        ('vmf+lognormal', 'vmf',      'lognormal'),
    ]
    results = {}
    for label, ad, rd in configs:
        out.grad = None
        lr, la, lrama = run(ad, out, target_coords, rama, rama_mask, r_dist=rd)
        total = lr + la + lrama
        total.backward()
        gmax = out.grad.abs().max().item()
        finite = all(torch.isfinite(x).all().item() for x in (lr, la, lrama))
        results[label] = (lr.item(), la.item(), lrama.item(), gmax, finite)
        print(f"[{label:14}] loss_r={lr.item():.4f}  loss_angle={la.item():.4f}  "
              f"loss_rama={lrama.item():.4f}  grad_max={gmax:.3e}  finite={finite}")

    print()
    ok = True
    for label, (lr, la, lrama, gmax, finite) in results.items():
        if not finite: print(f"FAIL: {label} has non-finite loss"); ok = False
        if gmax <= 0:  print(f"FAIL: {label} grad did not flow"); ok = False
    G = results['gaussian']
    # each family must actually change the term it targets
    if abs(G[1] - results['vonmises'][1]) < 1e-6:
        print("FAIL: vonmises angle == gaussian -> branch not taken"); ok = False
    if abs(G[1] - results['vmf'][1]) < 1e-6:
        print("FAIL: vmf angle == gaussian -> branch not taken"); ok = False
    if abs(results['vonmises'][1] - results['vmf'][1]) < 1e-6:
        print("FAIL: vmf angle == vonmises -> vmf branch not distinct"); ok = False
    if abs(G[0] - results['lognormal_r'][0]) < 1e-6:
        print("FAIL: lognormal r == gaussian -> r branch not taken"); ok = False
    # angle families must NOT touch the r term, and r_dist must NOT touch angle
    if abs(G[0] - results['vmf'][0]) > 1e-6:
        print("FAIL: vmf changed r loss -> leaked into radial term"); ok = False
    if abs(G[1] - results['lognormal_r'][1]) > 1e-6:
        print("FAIL: lognormal changed angle loss -> leaked into angle term"); ok = False

    print("\nSMOKE", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
