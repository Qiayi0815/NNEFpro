"""Metropolis-adjusted Langevin (MALA) for NNEF energies (WP0d).

The plain update used so far, x <- x - a*grad + b*xi, samples exp(-E/kT) only in
the a -> 0 limit with kT = b^2 / (2a), and blows up when a exceeds ~2/lambda_max
of the Hessian. MALA keeps the same proposal but adds an accept/reject step, so
the chain samples exp(-E/kT) exactly at any step size and runaway proposals are
rejected instead of propagated.

    proposal  x' = x - eps * grad U(x) + sqrt(2 eps kT) * xi
    accept    min(1, exp(-(U(x') - U(x)) / kT + log q(x | x') - log q(x' | x)))
    log q(y | x) = -|y - x + eps * grad U(x)|^2 / (4 eps kT)

U must be the SUMMED (extensive) energy so that one kT means the same thing for
every chain length (WP0e).
"""
from __future__ import annotations

import math
from typing import Callable, Tuple

import torch

EnergyGrad = Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]


def energy_and_grad(energy_fn: Callable[[torch.Tensor], torch.Tensor]) -> EnergyGrad:
    """Wrap a scalar energy U(x) into x -> (U(x), dU/dx) via autograd."""
    def f(x):
        x = x.detach().requires_grad_(True)
        u = energy_fn(x)
        (g,) = torch.autograd.grad(u, x)
        return u.detach(), g.detach()
    return f


class MALA:
    """One MALA chain. ``step()`` returns True if the proposal was accepted.

    During ``warmup`` steps the step size adapts multiplicatively toward
    ``target_accept`` every ``adapt_every`` steps; afterwards it is frozen so the
    production chain is a valid Markov chain.
    """

    def __init__(self, x0: torch.Tensor, ug: EnergyGrad, kT: float, eps: float,
                 warmup: int = 2000, target_accept: float = 0.57, adapt_every: int = 50):
        self.ug, self.kT, self.eps = ug, float(kT), float(eps)
        self.x = x0.detach().clone()
        self.u, self.g = ug(self.x)
        self.warmup, self.target, self.adapt_every = warmup, target_accept, adapt_every
        self.n_step = 0
        self._win_acc = 0
        self.n_acc = 0
        self.n_prod = 0

    def _log_q(self, y, x, gx):
        return -((y - x + self.eps * gx) ** 2).sum() / (4.0 * self.eps * self.kT)

    def step(self) -> bool:
        noise = torch.randn_like(self.x)
        xp = self.x - self.eps * self.g + math.sqrt(2.0 * self.eps * self.kT) * noise
        up, gp = self.ug(xp)
        accept = False
        if torch.isfinite(up) and torch.isfinite(gp).all():
            log_a = (-(up - self.u) / self.kT
                     + self._log_q(self.x, xp, gp) - self._log_q(xp, self.x, self.g))
            accept = bool(torch.log(torch.rand((), device=self.x.device)) < log_a)
        if accept:
            self.x, self.u, self.g = xp.detach(), up, gp
        self.n_step += 1
        self._win_acc += int(accept)
        if self.n_step <= self.warmup:
            if self.n_step % self.adapt_every == 0:
                rate = self._win_acc / self.adapt_every
                self.eps *= math.exp(1.0 * (rate - self.target))
                self._win_acc = 0
        else:
            self.n_prod += 1
            self.n_acc += int(accept)
        return accept

    @property
    def acceptance(self) -> float:
        return self.n_acc / max(self.n_prod, 1)


def configurational_temperature(energy_fn: Callable[[torch.Tensor], torch.Tensor],
                                x: torch.Tensor, n_probe: int = 4,
                                h: float = 1e-3) -> Tuple[float, float]:
    """(|grad U|^2, Laplacian U) at x; average both over frames, then
    kT_conf = <|grad U|^2> / <lap U> (Butler et al., JCP 1998).

    The Laplacian is a Hutchinson estimate with Rademacher probes v, using a
    central difference of FIRST derivatives, v . (g(x+hv) - g(x-hv)) / 2h,
    because the transformer's fused attention kernel has no double backward."""
    ug = energy_and_grad(energy_fn)
    _, g = ug(x)
    lap = 0.0
    for _ in range(n_probe):
        v = torch.randint_like(x, 2) * 2 - 1
        _, gp = ug(x + h * v)
        _, gm = ug(x - h * v)
        lap += float(((gp - gm) * v).sum()) / (2.0 * h)
    return float((g ** 2).sum()), lap / n_probe
