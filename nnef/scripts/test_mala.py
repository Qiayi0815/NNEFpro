"""Unit test for physics/mala.py (WP0f): MALA must sample exp(-U/kT) exactly.

Toy system: N independent 3-D harmonic oscillators U = k * sum |x|^2 (exact:
<x_i^2> = kT/(2k) per coordinate) plus one stiff bond pair U = k_b (|a-b|-d0)^2
(<(d-d0)^2> ~= kT/(2 k_b) for k_b d0^2 >> kT). Checks variance within 5%,
acceptance in range after warm-up, and configurational temperature within 10%.
Also shows that the plain update at a large step blows up while MALA stays finite.

    python nnef/scripts/test_mala.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from physics.mala import MALA, configurational_temperature, energy_and_grad  # noqa: E402

torch.manual_seed(0)
kT, k, kb, d0 = 0.05, 2.0, 100.0, 3.8


def U(x):
    osc = k * (x[:-2] ** 2).sum()
    bond = kb * ((x[-2] - x[-1]).norm() - d0) ** 2
    return osc + bond


x0 = torch.zeros(52, 3)
x0[-1, 0] = d0
chain = MALA(x0, energy_and_grad(U), kT=kT, eps=1e-3, warmup=3000)
xs, ds, g2, lap = [], [], 0.0, 0.0
for t in range(60000):
    chain.step()
    if t > 3000 and t % 10 == 0:
        xs.append(chain.x[:-2].clone())
        ds.append(float((chain.x[-2] - chain.x[-1]).norm() - d0))
    if t > 3000 and t % 500 == 0:
        a, b = configurational_temperature(U, chain.x)
        g2 += a
        lap += b
X = torch.stack(xs)
var_osc = float(X.var(dim=0).mean())
var_bond = float(torch.tensor(ds).var())
kT_conf = g2 / lap
ok_osc = abs(var_osc / (kT / (2 * k)) - 1) < 0.05
ok_bond = abs(var_bond / (kT / (2 * kb)) - 1) < 0.10
ok_acc = 0.45 < chain.acceptance < 0.70
ok_conf = abs(kT_conf / kT - 1) < 0.10
print(f'oscillator <x^2> {var_osc:.5f} vs exact {kT / (2 * k):.5f} -> {ok_osc}')
print(f'bond <(d-d0)^2> {var_bond:.6f} vs {kT / (2 * kb):.6f} -> {ok_bond}')
print(f'acceptance {chain.acceptance:.3f} (eps after warm-up {chain.eps:.2e}) -> {ok_acc}')
print(f'configurational kT {kT_conf:.4f} vs set kT {kT} -> {ok_conf}')

# plain update at a step above 2/lambda_max (lambda_max = 2*kb*2 = 400 -> a > 5e-3)
x = x0.clone()
a_big = 1e-2
for _ in range(200):
    _, g = energy_and_grad(U)(x)
    x = x - a_big * g + (2 * a_big * kT) ** 0.5 * torch.randn_like(x)
plain_finite = bool(torch.isfinite(x).all() and x.abs().max() < 1e3)
big = MALA(x0, energy_and_grad(U), kT=kT, eps=a_big, warmup=0)
for _ in range(200):
    big.step()
mala_finite = bool(torch.isfinite(big.x).all())
print(f'step {a_big}: plain Langevin stays finite: {plain_finite}; MALA stays finite: {mala_finite}')
ok = ok_osc and ok_bond and ok_acc and ok_conf and mala_finite
print('ALL PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
