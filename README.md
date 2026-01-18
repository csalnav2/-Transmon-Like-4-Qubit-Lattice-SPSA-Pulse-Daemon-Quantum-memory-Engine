# -Transmon-Like-4-Qubit-Lattice-SPSA-Pulse-Daemon-
Floquet-driven 4‑qubit transmon‑inspired lattice simulator that uses SPSA pulse shaping to target ≥0.9996 for the mean transmon-subspace fidelity (proxy) while tracking q‑datagram memory retention and thermodynamic/quantum diagnostics

# Transmon-Inspired Thermodynamic Neuron Dashboard (4-Qubit Lattice)

A research/prototype simulator and visualization dashboard for a **4-qubit (2×2) transmon-inspired lattice**
driven by a **periodic Floquet bath schedule**, with optional **SPSA (GRAPE-ish) pulse shaping** and an optional
**persistent “q-datagram” memory register** (non-Markovian extension).

> ⚠️ This is a **physics-motivated prototype**, not a calibrated transmon device model. Several diagnostics
> (Berry/QGT, “thermodynamic length”, “subspace fidelity”) are implemented as **proxies** for visualization
> and control/optimization.

---

## What it does

- Evolves the **global 4-qubit density matrix (16×16)** under:
  - a coherent Hamiltonian with local terms + couplings (XX+YY, ZZ)
  - local noise channels (dephasing + amplitude down/up) applied as Kraus maps (CPTP by construction)
- Drives the system with a **periodic bath waveform** (square/sine/triangle/sawtooth/gaussian).
- Produces a large “dashboard” animation + metrics:
  - Bloch trajectories, coherence, purity, QFI proxies, Loschmidt echo proxy
  - Bures geometry proxies (Bures angle/length, Bures speed)
  - Lagged-Bures “memory currents”: separation (loss) and return/backflow (gain)
  - Pairwise entanglement: log-negativity + concurrence for all 6 pairs
  - OSEE (operator-space entanglement entropy), MI(01:23)
  - Choi diagnostics for local channels (λ_min, purity)
  - Transmon-ish circuit proxies: ω01(t), EJ/EC, leakage-risk proxy, “subspace fidelity” proxy

---

## Install

### Minimal
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
1.) Lipka-Bartosik, P., Perarnau-Llobet, M., & Brunner, N. (2024). Thermodynamic computing via autonomous quantum thermal machines. Sci. Adv. 10, eadm8792.
