
The “key flags” section is directly grounded in the argparse block (daemon + qmem).:contentReference[oaicite:1]{index=1}

---

## 3) “Core truncated code” (publishable skeleton)

Save this as `src/core_truncated.py`. It’s intentionally compact and **does not** reproduce your full dashboards/plots, but it preserves the conceptual spine: 4-qubit global state, periodic bath, optional SPSA knobs, optional q-datagram register.

```python
#!/usr/bin/env python3
"""
core_truncated.py

A *publishable skeleton* of the Transmon-Inspired Thermodynamic Neuron Dashboard.

This intentionally omits:
- giant Matplotlib dashboard rendering
- Wigner surface helpers
- hundreds of diagnostics

But preserves the core architecture:
- 4-qubit global density matrix evolution (16x16)
- periodic bath schedule
- optional SPSA (GRAPE-ish) pulse shaping
- optional q-datagram memory register (partial-SWAP)

Use this as a “public repo core” while keeping the full visual dashboard private.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import math
import numpy as np

# ----------------------------
# Basic helpers
# ----------------------------

def herm(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.conj().T)

def normalize_dm(rho: np.ndarray) -> np.ndarray:
    tr = float(np.trace(rho).real)
    return rho / max(tr, 1e-15)

def kron_all(ops):
    out = np.array([[1.0 + 0j]])
    for op in ops:
        out = np.kron(out, op)
    return out

# Pauli
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def op_on_qubit(op2: np.ndarray, q: int, n: int) -> np.ndarray:
    ops = [I2] * n
    ops[q] = op2
    return kron_all(ops)

# ----------------------------
# Configs
# ----------------------------

@dataclass
class GKSLParams:
    dt: float = 0.02
    tmax: float = 12.0

    # simple local Hamiltonian knobs (arb units)
    omega0: float = 4.0
    omega_drive: float = 1.3

    # simple local noise knobs (arb probabilities per step)
    p_dephase: float = 0.01
    p_amp_down: float = 0.02
    p_amp_up: float = 0.002

@dataclass
class BathSchedule:
    enable: bool = True
    period: float = 4.0
    duty: float = 0.35
    waveform: str = "square"

    def wave(self, t: float) -> float:
        if (not self.enable) or self.period <= 0:
            return 0.0
        x = (t % self.period) / self.period
        wf = self.waveform.lower()
        if wf == "square":
            return 1.0 if x < self.duty else 0.0
        if wf in ("sine", "sin"):
            return 0.5 * (1.0 + math.sin(2.0 * math.pi * x))
        if wf in ("triangle", "tri"):
            return 1.0 - abs(2.0 * x - 1.0)
        if wf in ("saw", "sawtooth"):
            return x
        if wf in ("gauss", "gaussian"):
            sigma = max(1e-6, self.duty / 6.0)
            d = min(x, 1.0 - x)
            return math.exp(-0.5 * (d / sigma) ** 2)
        return 0.0

@dataclass
class Couplings:
    J_cap: float = 0.10  # XX+YY
    J_ind:  float = 0.06 # ZZ

@dataclass
class DaemonConfig:
    enable: bool = False
    iterations: int = 10
    a: float = 0.2
    c: float = 0.08
    alpha: float = 0.602
    gamma: float = 0.101

@dataclass
class QMemConfig:
    enable: bool = False
    n_mem: int = 4
    theta_in: float = math.pi / 10
    theta_out: float = math.pi / 12
    inject_every: int = 1
    update_every: int = 4

# ----------------------------
# Minimal channels (Kraus)
# ----------------------------

def kraus_dephase(p: float):
    p = float(np.clip(p, 0, 1))
    if p <= 0:
        return [I2]
    return [math.sqrt(1 - p) * I2, math.sqrt(p) * Z]

def kraus_amp_down(g: float):
    g = float(np.clip(g, 0, 1))
    if g <= 0:
        return [I2]
    K0 = np.array([[1.0, 0.0], [0.0, math.sqrt(1 - g)]], dtype=complex)
    K1 = np.array([[0.0, math.sqrt(g)], [0.0, 0.0]], dtype=complex)
    return [K0, K1]

def kraus_amp_up(g: float):
    g = float(np.clip(g, 0, 1))
    if g <= 0:
        return [I2]
    K0 = np.array([[math.sqrt(1 - g), 0.0], [0.0, 1.0]], dtype=complex)
    K1 = np.array([[0.0, 0.0], [math.sqrt(g), 0.0]], dtype=complex)
    return [K0, K1]

def apply_kraus_local(rho: np.ndarray, Ks, q: int, n: int) -> np.ndarray:
    """Apply single-qubit Kraus set on qubit q of an n-qubit state."""
    out = np.zeros_like(rho)
    for K in Ks:
        K_full = op_on_qubit(K, q, n)
        out += K_full @ rho @ K_full.conj().T
    return out

# ----------------------------
# Core evolution
# ----------------------------

def unitary_step(H: np.ndarray, dt: float) -> np.ndarray:
    w, v = np.linalg.eigh(herm(H))
    U = (v * np.exp(-1j * w * dt)) @ v.conj().T
    return U

def build_H(n: int, p: GKSLParams, cpl: Couplings, drive: float) -> np.ndarray:
    H = np.zeros((2**n, 2**n), dtype=complex)

    # local terms
    for i in range(n):
        H += 0.5 * p.omega0 * op_on_qubit(Z, i, n)
        H += 0.5 * drive * op_on_qubit(X, i, n)

    # simple 2x2 lattice edges (0-1, 0-2, 1-3, 2-3)
    edges = [(0,1),(0,2),(1,3),(2,3)]
    for (i,j) in edges:
        H += 0.5 * cpl.J_cap * (op_on_qubit(X,i,n) @ op_on_qubit(X,j,n) + op_on_qubit(Y,i,n) @ op_on_qubit(Y,j,n))
        H += 0.5 * cpl.J_ind * (op_on_qubit(Z,i,n) @ op_on_qubit(Z,j,n))
    return H

# ----------------------------
# Optional QMem: partial SWAP
# ----------------------------

def swap_gate():
    return np.array([[1,0,0,0],
                     [0,0,1,0],
                     [0,1,0,0],
                     [0,0,0,1]], dtype=complex)

def partial_swap(theta: float):
    I4 = np.eye(4, dtype=complex)
    S  = swap_gate()
    return math.cos(theta) * I4 - 1j * math.sin(theta) * S

# NOTE: In the full code, memory is implemented as system+memory joint evolution.
# Here we only expose the “API shape”, not the full tensor plumbing.

# ----------------------------
# SPSA (daemon) skeleton
# ----------------------------

def objective_stub(rho: np.ndarray) -> float:
    """Placeholder: replace with coherence/QFI/entanglement/memory objectives."""
    return float(np.trace(rho @ rho).real)  # purity as a toy objective

def run_spsa_stub(p: GKSLParams, bath: BathSchedule, cpl: Couplings, cfg: DaemonConfig) -> float:
    rng = np.random.default_rng(0)
    theta = 0.0  # toy: one knob (drive scale)
    best = -1e9

    for k in range(cfg.iterations):
        ak = cfg.a / ((k + 1) ** cfg.alpha)
        ck = cfg.c / ((k + 1) ** cfg.gamma)
        Delta = rng.choice([-1.0, 1.0])

        Jp = simulate(p, bath, cpl, drive_scale=theta + ck * Delta)
        Jm = simulate(p, bath, cpl, drive_scale=theta - ck * Delta)
        ghat = (Jp - Jm) / (2 * ck) * Delta
        theta = theta + ak * ghat
        best = max(best, simulate(p, bath, cpl, drive_scale=theta))

    return best

def simulate(p: GKSLParams, bath: BathSchedule, cpl: Couplings, drive_scale: float = 1.0) -> float:
    n = 4
    nsteps = int(p.tmax / p.dt) + 1
    t = np.linspace(0, p.tmax, nsteps)

    # init |++++><++++|
    plus = np.array([1,1], dtype=complex) / math.sqrt(2)
    psi = plus
    for _ in range(n-1):
        psi = np.kron(psi, plus)
    rho = np.outer(psi, psi.conj())

    for tt in t:
        w = bath.wave(float(tt))
        drive = p.omega_drive * drive_scale * (1.0 + 0.30 * w)

        H = build_H(n, p, cpl, drive)
        U = unitary_step(H, p.dt)
        rho = U @ rho @ U.conj().T

        for q in range(n):
            rho = apply_kraus_local(rho, kraus_dephase(p.p_dephase), q, n)
            rho = apply_kraus_local(rho, kraus_amp_down(p.p_amp_down), q, n)
            rho = apply_kraus_local(rho, kraus_amp_up(p.p_amp_up), q, n)

        rho = normalize_dm(herm(rho))

    return objective_stub(rho)

if __name__ == "__main__":
    p = GKSLParams()
    bath = BathSchedule(enable=True, period=4.0, duty=0.35, waveform="square")
    cpl = Couplings()
    cfg = DaemonConfig(enable=True)

    score = run_spsa_stub(p, bath, cpl, cfg)
    print("DONE. Best (toy) score:", score)
