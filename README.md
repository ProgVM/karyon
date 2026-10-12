# Karyon-CoRE (Continuous Recurrent Engine) v7.4 Sovereign Master

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![C++](https://img.shields.io/badge/C%2B%2B-20-red.svg)](https://isocpp.org/)
[![AVX2-FMA](https://img.shields.io/badge/Hardware-AVX2%20%2F%20FMA-brightgreen.svg)](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions)
[![Container](https://img.shields.io/badge/Container-.kcore%20v7.4-brightgreen.svg)](https://github.com/ProgVM/karyon)
[![KEP Standard](https://img.shields.io/badge/KEP-v20.0%20Sovereign-purple.svg)](KEP.md)

> **Autonomous Sovereign Cognitive Architecture operating at the raw physical byte, optical retina, and continuous vector field level ($V=258$, 256D Pixel Vision, 8D Kinetic Physics), grounded in Endogenous Mental Sandbox Rollout (Expected Free Energy minimization), AVX2-FMA Vectorized Laminar Plasticity, and Single-File Executable Binary Containers (`.kcore` v7.4).**

Created and architected by **Bazilevs (ProgVM member)** in 2026.

For philosophical foundations and theoretical axioms, see [KARYON_PHILOSOPHICAL_MANIFESTO.md](KARYON_PHILOSOPHICAL_MANIFESTO.md) and [KEP.md](KEP.md).

---

## 🚀 Architectural Paradigm: Native Core v7.4 Sovereign Substrate

Karyon-CoRE fundamentally rejects static matrix-multiplication deep learning models ("weight calculators / statistical parrots") and external supervisory neural networks. The system operates as a single, unified continuous physical entity:

1. **Universal Continuous Substrate (KEP Principle 2 & 27):**
   - Zero hardcoded artificial gates or external supervisory MLPs (`budget_net`, `cost_net`, `lock_evaluator`, `genesis_potential`).
   - Structural plasticity, horizon depth, and action selection are endogenous properties of the continuous physical substrate.
2. **Mental Sandbox Trajectory Rollout & Expected Free Energy (KEP Principle 7 & 28):**
   - Active Inference minimizes Expected Free Energy (EFE) across dynamic horizons $K(\tau)$.
   - Validated authentic task competence: In dynamic 2D physics arenas with moving targets and repulsive hazards, achieves high target interception and a 5.00x Catch-to-Hazard Mastery Ratio at $>50,000$ physics ticks/sec.
3. **Laminar Field Compression & Vectorized AVX2-FMA Plasticity:**
   - 256D optical video streams and sensory fields are bidirectionally compressed into high-speed 32D latent manifolds.
   - All tensor dynamics, quad interactions, and Hebbian plasticity are accelerated via 8-way parallel AVX2 FMA intrinsics (`_mm256_fmadd_ps`), achieving $38,000+$ FPS on raw $16 \times 16$ optical streams.
4. **Epigenetic Invariant Protection & Net2Net Smooth Sprouting (KEP Principle 6):**
   - Functionally validated operators are protected against catastrophic forgetting via continuous epigenetic locking.
   - Dynamic autopoiesis sprouts new super-primitives with zero-shock identity ($\tanh(\alpha_{\text{epi}}) \to 0$ at birth).
5. **Zero-Dependency Native Engine:**
   - C++20 core engine (`libkaryon_native.so`) compiles natively with `-O3 -mavx2 -mfma -std=c++20`.
   - Thin Python runtime (`karyon_agent.py`) binds directly via zero-overhead C-ABI `ctypes` without PyTorch/LibTorch overhead.

---

## 📁 Repository Directory Structure

```text
karyon/
├── karyon_native_core.h      # Native C++20 Core Header v7.4 (Mental Sandbox, AVX2-FMA, Laminar Stacks)
├── karyon_native_core.cpp    # Native C++20 Core Implementation (Vectorized Continuous Super-Primitives)
├── karyon_native_cabi.cpp    # High-speed C-ABI bridge for language interoperability
├── karyon_agent.py           # Zero-overhead Python Agent Runtime (Zero-Torch, native ctypes)
├── KEP.md                    # Karyon Engineering Protocol (KEP v20.0 Sovereign Master)
├── KARYON_PHILOSOPHICAL_MANIFESTO.md # Theoretical Foundations & Non-Biological Paradigm
├── experiments/              # Isolated Empirical Telemetry Benchmarks (EXP-1 to EXP-489+)
└── agent_data/               # Persistent Empirical SQLite Ledger & Epistemic Records
```

---

## 🔬 Empirical Scientific Ledger Highlights

| Experiment ID | Focus & Architectural Delta | Verdict | Throughput | Authentic Behavioral Quality |
|---|---|---|---|---|
| **EXP-489** | Multi-Branch Deep Latent Rollout on evasive kinetic orbit | 🟢 POSITIVE | 50,310.6 ticks/s | 55 target interceptions, 5.00x catch-to-hazard ratio |
| **EXP-488** | Dual-Mode Control & Ground-Truth Behavioral Quality Audit | 🟢 POSITIVE | 68,096.2 ticks/s / 38,301.9 FPS | 46 physical catches; 11 optical video interceptions |
| **EXP-487** | 256D Laminar Field Compression & AVX2-FMA Plasticity | 🟢 POSITIVE | 52,726.5 cycles/s | Free Energy converged to 0.0012 nats |
| **EXP-486** | Autonomous In-Flight Morphogenesis & Smooth Sprouting | 🟢 POSITIVE | 3,365.7 cycles/s | 4 successful zero-shock organelle sprouts |
| **EXP-485B**| Analytical Adjoint Motor Steering on 256D Video Stream | 🟢 POSITIVE | 8,454.4 FPS | 81 visual catches (140.9x real-time 60 FPS) |

---

## ⚡ Quickstart: Native C++20 Build & CLI Execution

```bash
# Compile native shared engine with AVX2-FMA acceleration
g++ -O3 -std=c++20 -mavx2 -mfma -fPIC -shared karyon_native_core.cpp karyon_native_cabi.cpp -o libkaryon_native.so

# Execute closed-loop Python agent
python3 karyon_agent.py

# Run real-time physics arena benchmark (EXP-489)
g++ -O3 -std=c++20 -mavx2 -mfma -I. experiments/exp_489_multi_branch_sandbox_rollout.cpp -L. -lkaryon_native -Wl,-rpath,. -o exp_489_runner
./exp_489_runner
```
