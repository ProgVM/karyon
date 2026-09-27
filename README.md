# Karyon-CoRE (Continuous Recurrent Engine) v6.0 Master

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![C++](https://img.shields.io/badge/C%2B%2B-20-red.svg)](https://isocpp.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-orange.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.0%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Container](https://img.shields.io/badge/Container-.kcore%20v6.0-brightgreen.svg)](https://github.com/ProgVM/karyon)
[![KEP Standard](https://img.shields.io/badge/KEP-v14.0%20Master-purple.svg)](KEP.md)

> **Autonomous Sovereign Cognitive Architecture operating at the raw UTF-8 byte level ($V=258$), grounded in Endogenous Allostasis, Susumu Ohno Triadic Morphogenesis, 3-Phase Sleep 2.0 with Edelman Neurodarwinian Apoptosis, and Single-File Executable Binary Containers (`.kcore` v6.0).**

Created and architected by **Bazilevs (ProgVM member)** in 2026.

For philosophical foundations and theoretical axioms, see [KARYON_PHILOSOPHICAL_FOUNDATIONS.md](KARYON_PHILOSOPHICAL_FOUNDATIONS.md).

---

## 🚀 Architectural Paradigm: The Sovereign Evolutionary Crucible (v6.0)

Karyon-CoRE fundamentally rejects static matrix-multiplication deep learning models ("weight calculators / statistical parrots"). It shifts from a pre-determined, rigid layer stack to a **Sovereign Evolutionary Crucible** governed by continuous thermodynamic and allostatic forces:

1. **Susumu Ohno Triadic Morphogenesis Cycle:**
   - **Phase I (Zero-Shock Gene Duplication):** Under persistent Free Energy stress ($S_t > \theta_{\text{morph}}$), the system duplicates functional operator nodes wrapped in epigenetic gating ($\tanh(\alpha_{\text{epi}}) \to 1.0$), ensuring zero-shock identity $f_{\text{new}}(x) \equiv f_{\text{old}}(x)$ at birth.
   - **Phase II (Orchestrated Dynamic Routing $\mathbf{R}(h_t)$):** Dynamic routing matrices continuously orchestrate signal flow across active and sprouted organelles without static bottlenecks.
   - **Phase III (Sleep & Edelman Neurodarwinian Apoptosis):** During slow-wave sleep, idle or parasitic organelles ($\alpha_{\text{epi}} \to 0$) are permanently pruned via Edelman selection (Tononi SHY downscaling), maintaining strict structural efficiency.
2. **Modal Invariance Across Continual Streams:**
   - Evaluated and proven across diverse problem topologies (EXP-309): seamlessly processes raw human speech, raw byte streams, and continuous physical differential equations (ODE dynamics) without task-specific re-architecture.
3. **Universal Raw UTF-8 Byte Substrate ($V=258$):** Eliminates subword tokenizers and BPE dictionaries (`pad=256`, `eos=257`). Maps text, audio, vision, motor efference, and homeostatic body signals into a unified continuous embedding space.
4. **Spatiotemporal Dualism (KEP Principle 22):** Decouples Problem Space from Generation Time.
   - **Temporal Axis ($S$):** Causal State-Space Duality (`CausalParallelSSD`) tracks sequential temporal causal flow ($h_t = \alpha h_{t-1} + (1-\alpha) x_t$) at native GPU Tensor Core speed (@ 200k+ tok/s).
   - **Spatial / Reasoning Axis ($K$):** Dynamic Morphic Graph deliberation and cellular wave diffusion (`DynamicMorphicGraph`) iteratively refine representations across recurrent thinking cycles before committing to motor actions.
5. **C++20 Compiled LibTorch Engine (`karyon_core.cpp`):** Heavy mathematical operations, state-space scans, cellular graph ops, and episodic memory slicing run directly on hardware via compiled C++20 LibTorch extensions (`-O3 -std=c++20`), completely bypassing Python interpreter overhead.
6. **Active Inference & Variational Free Energy Engine ($F_t$):** Latent World Model generating prior and posterior latent distributions ($z_t$) with bounded Gaussian variance ($\epsilon=0.01$), teleologically minimizing Variational Free Energy $F_t = D_{\text{KL}}(q(z)\|p(z)) + \mathcal{L}_{\text{rec}}$.
7. **Somatic Homeostasis & Dynamic Allostasis (Ashby Ultrastability):** Real-time tracking of 6 interoceptive variables (`Curiosity`, `Energy`, `Stability`, `Health`, `Noradrenaline`, `Dopamine`). $F_t$ directly modulates arousal ($NA_t$) and reward plasticity ($DA_t$), coupling thermodynamic vitality to computational depth.
8. **Single-File Executable Container Standard (`.kcore` v6.0):** Zero-dependency relocatable binary format encapsulating Section 1 (Manifest DNA Genome & Epigenetic Methylation Locks), Section 2 (C++20 & Python Source Logic), Section 3 (Zero-Copy 64-byte Aligned Tensor Weights), and Section 4 (Persistent Recurrent State Spaces).

---

## 📁 Repository Directory Structure

```text
karyon/
├── karyon_agent.py           # Master CoREAgent (Spatiotemporal Dualism, SSD + Dynamic Deliberation)
├── karyon_core.cpp           # Native C++20 LibTorch Master Core (Parallel SSD, Morphic Graph, Homeostasis)
├── karyon_core.py            # C++20 JIT compilation & hot-reload wrapper (karyon_cpp_ext)
├── karyon_entity.py          # Unified KaryonEntity high-level cognitive agent interface
├── karyon_config.py          # Master CoREConfig dataclass registry (Homeostasis, Net, Memory, Train)
├── karyon_checkpoint.py      # .kcore v6.0 container serializer, loader & tensor mmap adapter
├── karyon_hardware.py        # Universal hardware acceleration engine (CPU, CUDA, TPU PJRT)
├── karyon_logger.py          # Line-buffered real-time streaming logger
├── karyon_runtime.h / .cpp   # Pure C-ABI Standalone Host Driver (libkaryon_runtime.so)
├── kcore_builder.py          # Container packing utility (Logic + weights + states -> .kcore)
├── kcore_evolution.py        # Net2Net morphogenesis evolution engine with shape-adaptive alignment
├── kcore_format.h            # Binary container C-struct definitions
├── train_single_pass.py      # High-speed Single-Pass stream runtime with dual-cloud HF sync
├── dialogue.py               # Real-time closed-loop interactive social active inference dialogue
├── KEP.md                    # Official Karyon Engineering Protocol Master Specification (v14.0)
├── KARYON_PHILOSOPHICAL_FOUNDATIONS.md # Fundamental Philosophical Manifest
└── experiments/              # Immutable KEP benchmark suite (EXP-1 to EXP-309)
    └── archive/              # Archived validated and peer-reviewed benchmark scripts
```

---

## 📊 Empirical Scientific Ledger Highlights (EXP-300 to EXP-309)

| EXP ID | Breakthrough Mechanism | Key Telemetry / Metrics | Verdict |
|---|---|---|:---:|
| **EXP-309** | **Bimodal Reality Stream & Modal Invariance** | `loss_text_post_physics = 0.0`, `nodes_sprouted = 1`, `gate_purity_mean = 100%` | 🟢 **POSITIVE** |
| **EXP-308** | **Triadic Synthesis Marathon (Text + Math + Code)** | `domain_a_acc = 100%`, `domain_d_acc = 100%`, `apoptosed_nodes = 10` | 🟢 **POSITIVE** |
| **EXP-307** | **Neurodarwinian Apoptosis & Parasitic Node Pruning** | `pre_sleep_acc = 99.74%`, `post_sleep_acc = 99.74%`, `apoptosed_nodes = 1` | 🟢 **POSITIVE** |
| **EXP-306** | **Laminar Compositional Operator Graphing** | `loss = 9.1e-05`, `delta_loss = 5.839` | 🟢 **POSITIVE** |
| **EXP-305** | **Endogenous Somatic Stress Accumulator** | `loss = 0.0726`, `energy = 2.8828`, `delta_loss = 0.123` | 🟢 **POSITIVE** |
| **EXP-289** | **Dual-Phase Recurrent Working Engine** | `loss = 0.4671`, `reversal_acc = 87.5%`, `dyck_acc = 83.0%` | 🟢 **POSITIVE** |
| **EXP-288** | **Decoupling Temporal SSD from Morphic Graph** | `loss = 0.0397`, `reconstruction_acc = 100%`, `delta_shock = 0.0` | 🟢 **POSITIVE** |
| **EXP-286** | **Exposing C++20 Dynamic Graph Parameters** | `loss = 2.4677`, `tok_per_sec = 4296.2`, `delta_loss = 0.3542` | 🟢 **POSITIVE** |

---

## ⚡ Quickstart Guide

### 1. Requirements & Hardware Setup
* **OS:** Linux (Ubuntu 22.04+ or Kaggle GPU/TPU VM)
* **Python:** 3.12+
* **C++ Compiler:** GCC 12+ / Clang 15+ supporting `-std=c++20`
* **PyTorch & CUDA:** PyTorch 2.5+ with CUDA 12.0+ (or PyTorch-XLA for TPU)

### 2. Continuous Single-Pass Stream Learning
Launch high-throughput continuous stream training with automatic cloud synchronization:
```bash
python train_single_pass.py
```

### 3. Interactive Closed-Loop Dialogue Session
Engage in a live active inference dialogue session with Karyon:
```bash
python dialogue.py
```

### 4. High-Level Python API (`KaryonEntity`)
```python
from karyon_entity import KaryonEntity

# Load relocatable self-contained cognitive entity (.kcore v6.0)
entity = KaryonEntity.load(filepath="karyon_soul.kcore", device="cuda:0")

# Closed-loop active inference turn
response = entity.step(prompt="What is the nature of consciousness?", max_new_tokens=64)
print("Karyon:", response)

# Inspect live homeostatic landscape
print("Somatic State:", entity.hu.get_states())
# [Curiosity, Energy, Stability, Health, Noradrenaline, Dopamine]
```

---

## 📜 KEP Protocol Compliance

All architectural changes, experiments, and features in Karyon-CoRE strictly comply with the **Karyon Engineering Protocol (KEP v14.0 Master)**:
* **Principle 1:** Python as client, C++20 as computational engine (`-O3 -std=c++20`).
* **Principle 4:** Zero Tolerance for Dead Code.
* **Principle 17:** Goodhart's Law Immunization & Metric De-Fetishization.
* **Principle 18:** Mandatory External Verification & Eradication of Blind Self-Approval.
* **Principle 21:** Universal Sovereign Morphogenesis over Task-Narrow Autoregressive Pipelines.
* **Principle 22:** The Five Pillars of Sovereign Self-Evolution (Unbounded Topological Genesis, Total Endogenous Sovereignty, Spatiotemporal Dualism, Wave-Particle Dualism, Teleological Optimality).
* **Principle 24:** Strict Single-Pass Continual Learning Paradigm & Anti-Zubryoshka Axiom.

---

## 📜 License & Attribution

Distributed under the **MIT License**.

Designed and created by **Bazilevs (ProgVM member)** in 2026.
