"""Underdamped Langevin dynamics with holonomic distance constraints (BAOAB + SHAKE/RATTLE).

Why: with a stiff bonded prior the Cartesian step is set by the bond springs
(~0.02 A/step), so plain Langevin/MALA barely moves. Constraining the rigid
backbone distances instead (the same pairs as md_eval_bb.bond_pairs: each
residue's N/CA/C/CB rigid + the peptide unit) leaves only the phi/psi motions,
exactly the degrees of freedom of the rigid-geometry model, and the time step is
then limited by the learned energy only. This is the standard constrained MD
scheme (Leimkuhler & Matthews 2016, BAOAB with RATTLE), with unit masses and
reduced time units.

    sigma_k(x) = |x_a - x_b|^2 - d0_k^2 = 0          for every constrained pair k

Step (h = dt):
    B  v += h/2 F;          project v onto the constraint tangent space (RATTLE)
    A  x += h/2 v;          SHAKE x back onto the manifold, v = (x_new - x_old) / (h/2)
    O  v = c v + sqrt((1 - c^2) kT) xi,  c = exp(-gamma h);  project v
    A  as above
    B  as above (with the new force)
"""
from __future__ import annotations

import math
from typing import Callable, Tuple

import torch


class ConstrainedBAOAB:
    def __init__(self, x0: torch.Tensor, ug: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
                 pairs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                 kT: float, dt: float, gamma: float = 1.0, shake_tol: float = 1e-6, shake_iter: int = 100,
                 generator: torch.Generator = None):
        """x0: (4, L, 3) N/CA/C/CB; pairs: (atom_a, res_a, atom_b, res_b) index tensors."""
        self.shape = x0.shape
        L = x0.shape[1]
        ia, ra, ib, rb = pairs
        self.a = (ia * L + ra).to(x0.device)          # flat atom indices into (4L, 3)
        self.b = (ib * L + rb).to(x0.device)
        x = x0.detach().reshape(-1, 3).double().clone()
        # Glycine: CB is stored on top of CA. Treat it as a virtual site (always at CA, its force goes to CA)
        # and drop every constraint that involves it; a zero-length constraint has no gradient.
        cb, ca = torch.arange(3 * L, 4 * L, device=x0.device), torch.arange(L, 2 * L, device=x0.device)
        virt = (x[cb] - x[ca]).norm(dim=-1) < 1e-3
        self.v_cb, self.v_ca = cb[virt], ca[virt]
        keep = ~(torch.isin(self.a, self.v_cb) | torch.isin(self.b, self.v_cb))
        self.a, self.b = self.a[keep], self.b[keep]
        self.d0sq = ((x[self.a] - x[self.b]) ** 2).sum(-1)
        self.ug, self.kT, self.dt, self.gamma = ug, float(kT), float(dt), float(gamma)
        self.tol, self.max_iter = shake_tol, shake_iter
        self.gen = generator
        self.x = x
        self.v = torch.zeros_like(x)
        self.u, f = self._force(x)
        self.f = f
        self.v = self._project_v(self.x, self._noise() * math.sqrt(self.kT))   # thermal initial velocities
        self.n_step = 0
        self.n_accept = 0          # step_ghmc only
        self.last_dH = 0.0
        self.max_shake_iter = 0

    def _noise(self):
        n = torch.randn(self.x.shape, dtype=self.x.dtype, device=self.x.device, generator=self.gen)
        n[self.v_cb] = 0.0   # virtual sites carry no momentum
        return n

    def _force(self, x):
        x = x.clone()
        x[self.v_cb] = x[self.v_ca]
        u, g = self.ug(x.reshape(self.shape).float())
        f = -g.reshape(-1, 3).double()
        f[self.v_ca] += f[self.v_cb]
        f[self.v_cb] = 0.0
        return u, f

    def _G(self, x):
        """Constraint Jacobian (nc, 3N) as a dense matrix: d sigma_k / dx = 2 (x_a - x_b) at a, minus at b."""
        nc, N = len(self.a), x.shape[0]
        d = 2.0 * (x[self.a] - x[self.b])                       # (nc, 3)
        G = torch.zeros(nc, N, 3, dtype=x.dtype, device=x.device)
        k = torch.arange(nc, device=x.device)
        G[k, self.a] = d
        G[k, self.b] = -d
        return G.reshape(nc, -1)

    def _project_v(self, x, v):
        G = self._G(x)
        lam = torch.linalg.solve(G @ G.T, G @ v.reshape(-1))
        return v - (G.T @ lam).reshape(-1, 3)

    def _shake(self, x_new, x_ref):
        """Move x_new back onto the constraint manifold along the constraint gradients at x_ref."""
        Gr = self._G(x_ref)
        for it in range(self.max_iter):
            sig = ((x_new[self.a] - x_new[self.b]) ** 2).sum(-1) - self.d0sq
            if sig.abs().max() < self.tol:
                break
            Gn = self._G(x_new)
            lam = torch.linalg.solve(Gn @ Gr.T, sig)
            x_new = x_new - (Gr.T @ lam).reshape(-1, 3)
        self.max_shake_iter = max(self.max_shake_iter, it)
        return x_new

    def _A(self, h):
        x_new = self._shake(self.x + h * self.v, self.x)
        self.v = self._project_v(x_new, (x_new - self.x) / h)
        self.x = x_new

    def step(self):
        h = self.dt
        self.v = self._project_v(self.x, self.v + 0.5 * h * self.f)
        self._A(0.5 * h)
        c = math.exp(-self.gamma * h)
        self.v = self._project_v(self.x, c * self.v + math.sqrt((1 - c * c) * self.kT) * self._noise())
        self._A(0.5 * h)
        self.u, self.f = self._force(self.x)
        self.v = self._project_v(self.x, self.v + 0.5 * h * self.f)
        self.n_step += 1
        return self.u

    def step_ghmc(self, n_leap: int = 1):
        """Metropolized step (generalized HMC): O, then n_leap RATTLE velocity-Verlet steps (B A B),
        then accept with min(1, exp(-dH/kT)), H = U + |v|^2/2; on reject restore x and flip v.

        RATTLE is reversible and volume preserving on the constraint manifold, so the chain samples
        exp(-U/kT) exactly, whatever the integration error -- including energy jumps the force does not
        see and isolated force spikes (those steps are simply rejected)."""
        h = self.dt
        c = math.exp(-self.gamma * h * n_leap)
        self.v = self._project_v(self.x, c * self.v + math.sqrt((1 - c * c) * self.kT) * self._noise())
        x0, v0, u0, f0 = self.x.clone(), self.v.clone(), self.u, self.f.clone()
        H0 = float(u0) + 0.5 * float((v0 ** 2).sum())
        ok = True
        for _ in range(n_leap):
            self.v = self._project_v(self.x, self.v + 0.5 * h * self.f)
            self._A(h)
            self.u, self.f = self._force(self.x)
            self.v = self._project_v(self.x, self.v + 0.5 * h * self.f)
            if not (math.isfinite(float(self.u)) and torch.isfinite(self.x).all()):
                ok = False
                break
        dH = float(self.u) + 0.5 * float((self.v ** 2).sum()) - H0 if ok else math.inf
        self.last_dH = dH
        accept = dH <= 0 or float(torch.rand((), dtype=torch.float64, generator=self.gen)) < math.exp(-dH / self.kT)
        if accept:
            self.n_accept += 1
        else:
            self.x, self.u, self.f, self.v = x0, u0, f0, -v0
        self.n_step += 1
        return self.u

    @property
    def X(self):
        x = self.x.clone()
        x[self.v_cb] = x[self.v_ca]
        return x.reshape(self.shape).float()

    def kinetic_temperature(self):
        """2 KE / n_dof with n_dof = 3N - n_constraints (unit masses)."""
        ndof = self.x.numel() - 3 * len(self.v_cb) - len(self.a)
        return float((self.v ** 2).sum()) / ndof

    def max_violation(self):
        return float((((self.x[self.a] - self.x[self.b]) ** 2).sum(-1) - self.d0sq).abs().sqrt().max())
