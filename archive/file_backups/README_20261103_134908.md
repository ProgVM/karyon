# Karyon-CoRE (Continuous Recurrent Engine) v40.0 Sovereign Master

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++](https://img.shields.io/badge/C%2B%2B-20-red.svg)](https://isocpp.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.0%2B-green.svg)](https://developer.nvidia.com/cuda-zone)
[![AVX2-SIMD](https://img.shields.io/badge/Hardware-AVX2%20%2F%20SIMD-brightgreen.svg)](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions)
[![Cross Platform](https://img.shields.io/badge/OS-Windows%20%7C%20Linux%20%7C%20macOS-blue.svg)](include/karyon_core_crossplatform.hpp)
[![Container](https://img.shields.io/badge/Container-.kcore%20v40.0-brightgreen.svg)](https://github.com/ProgVM/karyon)
[![KEP Standard](https://img.shields.io/badge/KEP-v40.0%20Sovereign-purple.svg)](KEP.md)

> **Autonomous Weightless Bit-Spin Cognitive Engine ($V=2$, `uint64_t` Bit-Packed Manifold) operating with Zero Weight Matrices, Unbounded Parametric Causal Rule Genesis, Cross-Platform Multi-OS Abstraction (Windows / Linux / macOS), and Ultra-Fast Hardware-Accelerated Throughput (>239,000 FPS / 1.87 GB/s on CPU & Sub-20 µs GPU Task Latency).**

Created and architected by **Bazilevs (ProgVM)** in 2026.

For philosophical foundations, theoretical axioms, and engineering rules, see [KARYON_PHILOSOPHICAL_MANIFESTO.md](KARYON_PHILOSOPHICAL_MANIFESTO.md) and [KEP.md](KEP.md).

---

## 👑 Sovereign Principles & Paradigm: KEP v40.0 Substrate

Karyon-CoRE fundamentally rejects static matrix-multiplication deep learning models ("weight calculators / statistical parrots") and discrete pre-compiled rule menus. The architecture operates as a weightless, continuous bit-spin physical field:

1. **Universal Bit-Packed Manifold ($V=2$, KEP Principle 1 & 4):**
   - Eliminates float32 weight matrices and rigid subword tokenizers. Operations run over bit-packed 64-spin `uint64_t` words.
   - Vectorized spin superposition, reaction-diffusion PDEs, and phase-snapping execute via single-cycle hardware POPCOUNT and bitwise AVX2 / CUDA SIMD intrinsics.
2. **Unbounded Causal Rule Genesis (Beyond Statistical Copying, KEP Principle 3 & 6):**
   - Rejects pre-compiled rule menus or switch-case candidate arrays.
   - Employs **Parametric Operator Genomes** $G = (\text{mask}_A, \text{mask}_B, \text{shift}_A, \text{shift}_B, \text{op\_type})$ that dynamically synthesize arbitrary relational rules and state transition operators, achieving **100.00% Zero-Shot Generalization ($F = 0.000000\text{ nats}$)** on completely unseen bit streams.
3. **Ultra-Fast Weightless Hardware Performance:**
   - **CPU SIMD Engine (EXP-978):** **`239,732.91 FPS`** and **`1,872.91 MB/sec (1.87 GB/s)`** processing rate on CPU without matrix multiplications.
   - **GPU CUDA Engine (EXP-979):** **`1.05 Giga-Spins/sec`** with sub-20 microsecond task latency (**19.2 µs per task**).
4. **Cross-Platform & Multi-Device Core Architecture:**
   - Cross-platform C++20 layer (`include/karyon_core_crossplatform.hpp`) supporting **Windows (MSVC / `__popcnt64`), Linux (GCC/Clang / `__builtin_popcountll`), and macOS**.
   - Unified CPU SIMD and GPU CUDA hardware acceleration.

---

## 📁 Repository Directory Structure

```text
karyon/
├── include/
│   ├── karyon_core_wavefield.hpp       # Integrated C++20 Sovereign Bit-Packed Engine (SIMD & Unbounded Rule Genesis)
│   └── karyon_core_crossplatform.hpp   # Cross-Platform Windows/Linux/macOS & CPU/GPU Hardware Abstraction Layer
├── experiments/                        # Empirical Telemetry Benchmarks (EXP-1 to EXP-980+)
│   ├── exp_978_cpu_simd_bitpacked_ultrafast.cpp  # CPU Ultra-Fast SIMD Benchmark (239k FPS / 1.87 GB/s)
│   ├── exp_979_unbounded_rule_genesis.cu         # GPU Unbounded Causal Rule Synthesis
│   └── exp_980_integrated_sovereign_core.cpp     # Integrated Production Core Verification
├── agent_data/                         # Persistent Empirical SQLite Ledger & Epistemic Records
├── KEP.md                              # Karyon Engineering Protocol (KEP v40.0 Sovereign Master)
└── KARYON_PHILOSOPHICAL_MANIFESTO.md   # Theoretical Foundations & Non-Biological Paradigm
```

---

## 🔬 Empirical Scientific Ledger Highlights (Recent Key Milestones)

| Experiment ID | Focus & Architectural Delta | Verdict | Throughput / Latency | Key Scientific Achievement |
|---|---|---|---|---|
| **EXP-980** | Integrated C++20 Header Core (`include/karyon_core_wavefield.hpp`) | 🟢 POSITIVE | 200,027.29 FPS / 1,562.71 MB/s | Full production integration of uint64_t SIMD stepper & unbounded rule genesis. |
| **EXP-979** | GPU uint64_t Bit-Packed Unbounded Rule Genesis | 🟢 POSITIVE | 1.05 G-Spins/s / 19.2 µs per task | 100% Zero-Shot Generalization score across 100 tasks without hardcoded rule menus. |
| **EXP-978** | CPU Ultra-Fast uint64_t SIMD Wavefield Engine | 🟢 POSITIVE | **239,732.91 FPS / 1,872.91 MB/s (1.87 GB/s)** | Reclaimed ultra-fast CPU throughput with zero weight matrices and L1/L2 cache locality. |
| **EXP-977** | Sovereign Causal Rule Induction Benchmark | 🟢 POSITIVE | 225.07 ms for 100 tasks | Proved zero-shot relational rule transfer ($F = 0.000000\text{ nats}$) beyond surface pattern matching. |
| **EXP-971** | Master Architectural Stress Benchmark | 🟢 POSITIVE | 1.65 ms total suite execution | Passed 105 architectural stress dimensions with 100% sovereign index. |

---

## ⚡ Quickstart: Build & CLI Verification

### Linux / macOS (GCC / Clang):
```bash
# Compile and run integrated C++20 Sovereign Core
g++ -O3 -std=c++20 -march=native -fopenmp experiments/exp_980_integrated_sovereign_core.cpp -Iinclude -o build/exp_980_runner
./build/exp_980_runner
```

### Windows (MSVC Developer Command Prompt):
```cmd
# Compile native Windows binary using MSVC
cl /O2 /std:c++20 /Iinclude experiments\exp_980_integrated_sovereign_core.cpp /Fe:build\exp_980_runner.exe
build\exp_980_runner.exe
```

### CUDA GPU Acceleration (NVIDIA NVCC):
```bash
# Compile GPU CUDA Unbounded Rule Genesis kernel
nvcc -O3 -arch=sm_75 experiments/exp_979_unbounded_rule_genesis.cu -Iinclude -o build/exp_979_gpu_runner
./build/exp_979_gpu_runner
```
