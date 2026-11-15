"""Torsion-space replica-exchange Monte Carlo (REMC) for 4-bead NNEF backbones.

Why not Cartesian MALA: with the bonded prior stiff enough to hold the peptide
geometry, the MALA step is set by the bond modes (~0.02 A/step), so in 1e5
steps a chain only diffuses ~1 A and RMSD/RMSF never become stationary
(scratchpad/slides/new/pep_sim/fig_calib_convergence.png).

Here bond lengths, bond angles and omega are frozen at the starting structure's
values (rigid-geometry model) and only backbone (phi, psi) move, so every move
is a real conformational change. Moves (both symmetric -> plain Metropolis):

    small  one torsion (phi_i or psi_i) += N(0, sigma_small)          (Rosetta "small")
    shear  psi_{i-1} += d, phi_i -= d,  d ~ N(0, sigma_shear)           (Rosetta "shear")

A torsion change is a rigid rotation of one side of the chain about the bond;
the energy is invariant to rigid-body motion, so the smaller side is rotated.
Overlapping chains are excluded by a hard-sphere test on non-local CA-CA pairs
(no energy scale; identical for every model). R replicas on a geometric kT
ladder each make one move per step, their energies are evaluated in ONE batched
network pass, and neighbouring temperatures attempt configuration swaps.

Caveat: sampling is uniform in torsion space (no Fixman/metric correction), the
standard choice for rigid-geometry MC.
"""
from __future__ import annotations

import math
from typing import Callable, List, Sequence

import numpy as np

N, CA, C, CB = 0, 1, 2, 3


def dihedral(p0, p1, p2, p3):
    """Dihedral angle (rad) of 4 points (..., 3)."""
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - (b0 * b1n).sum(-1, keepdims=True) * b1n
    w = b2 - (b2 * b1n).sum(-1, keepdims=True) * b1n
    x = (v * w).sum(-1)
    y = (np.cross(b1n, v) * w).sum(-1)
    return np.arctan2(y, x)


def rotate(X, mask, a, b, ang):
    """Rotate atoms X[mask] (X: (4, L, 3)) by ang about the axis a -> b (Rodrigues)."""
    k = (b - a) / np.linalg.norm(b - a)
    P = X[mask] - a
    c, s = math.cos(ang), math.sin(ang)
    X[mask] = P * c + np.cross(k, P) * s + np.outer(P @ k, k) * (1 - c) + a


class Torsions:
    """Moving-atom masks for every rotatable backbone torsion of an L-residue chain.

    phi_i (axis N_i -> CA_i, i >= 1) moves C_i, CB_i and residues > i.
    psi_i (axis CA_i -> C_i, i <= L-2) moves residues > i.
    phi_0 / psi_{L-1} only move atoms that are not modelled and are skipped.
    """

    def __init__(self, L: int):
        self.L = L
        self.spec = []  # (kind, residue, axis_a (atom, res), axis_b, mask_after)
        for i in range(1, L):
            m = np.zeros((4, L), bool)
            m[[C, CB], i] = True
            m[:, i + 1:] = True
            self.spec.append(('phi', i, (N, i), (CA, i), m))
        for i in range(L - 1):
            m = np.zeros((4, L), bool)
            m[:, i + 1:] = True
            self.spec.append(('psi', i, (CA, i), (C, i), m))
        self.index = {(k, i): t for t, (k, i, *_rest) in enumerate(self.spec)}

    def apply(self, X, t, ang):
        kind, i, (aa, ra), (ab, rb), m = self.spec[t]
        a, b = X[aa, ra].copy(), X[ab, rb].copy()
        if m.sum() * 2 <= m.size:
            rotate(X, m, a, b, ang)
        else:  # rotate the other (smaller) side the opposite way: same internal geometry
            other = ~m
            other[aa, ra] = other[ab, rb] = False  # axis atoms stay put either way
            rotate(X, other, a, b, -ang)

    def value(self, X, t):
        kind, i = self.spec[t][:2]
        if kind == 'phi':
            return dihedral(X[C, i - 1], X[N, i], X[CA, i], X[C, i])
        return dihedral(X[N, i], X[CA, i], X[C, i], X[N, i + 1])

    def set_all(self, X, phi=None, psi=None):
        """Set every rotatable phi/psi to the given value (rad), e.g. an extended chain."""
        for t, (kind, i, *_r) in enumerate(self.spec):
            target = phi if kind == 'phi' else psi
            if target is not None:
                self.apply(X, t, float(target - self.value(X, t)))
        return X


class HardSphere:
    """Reject configurations with non-local CA-CA (|i-j| >= 3) closer than d_min."""

    def __init__(self, L: int, d_min: float):
        iu = np.triu_indices(L, k=3)
        self.i, self.j, self.d_min = iu[0], iu[1], d_min

    def min_dist(self, X):
        ca = X[CA]
        return float(np.linalg.norm(ca[self.i] - ca[self.j], axis=-1).min()) if len(self.i) else np.inf

    def ok(self, X):
        return self.min_dist(X) >= self.d_min


class REMC:
    """R replicas on a kT ladder; batch_energy(list of (4, L, 3) arrays) -> np.ndarray (R,)."""

    def __init__(self, X0: np.ndarray, kTs: Sequence[float], batch_energy: Callable,
                 hard_sphere: HardSphere, rng: np.random.Generator,
                 sigma_small: float = math.radians(15), sigma_shear: float = math.radians(15),
                 p_shear: float = 0.5, target_accept: float = 0.35):
        self.kT = np.asarray(kTs, float)
        self.R = len(self.kT)
        self.L = X0.shape[1]
        self.tor = Torsions(self.L)
        self.hs = hard_sphere
        self.rng = rng
        self.batch_energy = batch_energy
        self.X: List[np.ndarray] = [X0.astype(np.float64).copy() for _ in range(self.R)]
        self.E = np.asarray(batch_energy(self.X), float)
        self.sigma = np.array([[sigma_small, sigma_shear]] * self.R, float)  # per replica, per move type
        self.p_shear = p_shear
        self.target = target_accept
        self.walker = np.arange(self.R)  # walker id currently at each temperature
        self._win = np.zeros((self.R, 2, 2))  # accepted / tried per move type (adaptation window)
        self.acc = np.zeros((self.R, 2, 2))  # production accepted / tried
        self.swap = np.zeros((self.R - 1, 2))  # accepted / tried per neighbour pair
        self.n_clash = 0

    def _propose(self, r):
        X = self.X[r].copy()
        tor, L = self.tor, self.L
        if self.rng.random() < self.p_shear and L >= 3:
            mt = 1
            i = int(self.rng.integers(1, L))  # psi_{i-1}, phi_i
            d = self.rng.normal(0.0, self.sigma[r, 1])
            tor.apply(X, tor.index[('psi', i - 1)], d)
            tor.apply(X, tor.index[('phi', i)], -d)
        else:
            mt = 0
            t = int(self.rng.integers(len(tor.spec)))
            tor.apply(X, t, self.rng.normal(0.0, self.sigma[r, 0]))
        return X, mt

    def step(self, adapt: bool):
        props, mts, valid = [], [], []
        for r in range(self.R):
            Xp, mt = self._propose(r)
            props.append(Xp)
            mts.append(mt)
            valid.append(self.hs.ok(Xp))
        Ep = np.asarray(self.batch_energy(props), float)
        for r in range(self.R):
            mt = mts[r]
            ok = valid[r] and np.isfinite(Ep[r])
            if not valid[r]:
                self.n_clash += 1
            accept = ok and (math.log(self.rng.random() + 1e-300) < -(Ep[r] - self.E[r]) / self.kT[r])
            if accept:
                self.X[r], self.E[r] = props[r], Ep[r]
            book = self._win if adapt else self.acc
            book[r, mt, 0] += accept
            book[r, mt, 1] += 1
        return Ep

    def adapt(self):
        """Multiplicative step-size update toward target acceptance (warm-up only)."""
        rate = self._win[..., 0] / np.maximum(self._win[..., 1], 1)
        tried = self._win[..., 1] > 0
        self.sigma[tried] *= np.exp(rate[tried] - self.target)
        self.sigma = np.clip(self.sigma, math.radians(0.5), math.radians(90))
        self._win[:] = 0

    def exchange(self, parity: int):
        for r in range(parity, self.R - 1, 2):
            s = r + 1
            log_a = (1.0 / self.kT[r] - 1.0 / self.kT[s]) * (self.E[r] - self.E[s])
            self.swap[r, 1] += 1
            if math.log(self.rng.random() + 1e-300) < log_a:
                self.X[r], self.X[s] = self.X[s], self.X[r]
                self.E[r], self.E[s] = self.E[s], self.E[r]
                self.walker[r], self.walker[s] = self.walker[s], self.walker[r]
                self.swap[r, 0] += 1

    @property
    def acceptance(self):
        return self.acc[..., 0] / np.maximum(self.acc[..., 1], 1)  # (R, 2): small, shear

    @property
    def swap_rate(self):
        return self.swap[:, 0] / np.maximum(self.swap[:, 1], 1)
