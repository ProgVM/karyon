# Karyon-CoRE (Continuous Recurrent Engine) v48.0 Master

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![C++](https://img.shields.io/badge/C%2B%2B-20-red.svg)](https://isocpp.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-orange.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.0%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Container](https://img.shields.io/badge/Container-.kcore%20v5.0-brightgreen.svg)](https://github.com/ProgVM/karyon)
[![KEP Standard](https://img.shields.io/badge/KEP-v14.0%20Master-purple.svg)](KEP.md)

> **Autonomous Sovereign Cognitive Architecture operating at the raw UTF-8 byte level ($V=258$, expandable to $V=1024$ multimodal), grounded in Spatiotemporal Dualism (Principle 22), Active Inference, Zero-Loop Parallel State-Space Duality (@ 200k+ tok/s), Dynamic Morphic Graphs, Endogenous Allostasis, 3-Phase Sleep 2.0 with Edelman Neurodarwinian Apoptosis, and Single-File Relocatable Executable Binary Containers (`.kcore` v5.0).**

Created and architected by **Bazilevs (ProgVM member)** in 2026.

---

## 🚀 Key Architectural Foundations (v48.0 Master)

Karyon-CoRE transcends static matrix-multiplication deep learning models ("weight calculators / statistical parroting") toward a sovereign, continuous-time cognitive substrate with causal reasoning:

1. **Universal Raw UTF-8 Byte Substrate ($V=258 \to 1024$):** Eliminates subword tokenizers and BPE dictionaries (`pad=256`, `eos=257`). Maps text, audio, vision, motor efference, and homeostatic body signals into a unified continuous embedding space ($D_{\text{text}}=256, D_{\text{hidden}}=512$), with dynamic on-the-fly alphabet expansion.
2. **Spatiotemporal Dualism (KEP Principle 22):** Decouples Problem Space from Generation Time.
   * **Temporal Axis ($S$):** Causal State-Space Duality (`CausalParallelSSD`) tracks sequential temporal causal flow ($h_t = \alpha h_{t-1} + (1-\alpha) x_t$) at native GPU Tensor Core speed (@ 200k+ tok/s).
   * **Spatial / Reasoning Axis ($K$):** Dynamic Morphic Graph deliberation and cellular wave diffusion (`DynamicMorphicGraph`) iteratively refine representations across recurrent thinking cycles before committing to motor actions.
3. **C++20 Compiled LibTorch Engine (`karyon_core.cpp`):** Heavy mathematical operations, state-space scans, cellular graph ops, and episodic memory slicing run directly on hardware via compiled C++20 LibTorch extensions (`-O3 -std=c++20`), completely bypassing Python interpreter overhead.
4. **Active Inference & Variational Free Energy Engine ($F_t$):** Latent World Model generating prior and posterior latent distributions ($z_t$) with bounded Gaussian variance ($\epsilon=0.01$), teleologically minimizing Variational Free Energy $F_t = D_{\text{KL}}(q(z)\|p(z)) + \mathcal{L}_{\text{rec}}$.
5. **Dynamic Morphic Graph Neurogenesis (Principle 16 / AGN v6.0):** Autonomous structural assembly of primitive mathematical operator nodes (`LinearAccumulator`, `BilinearMultiplicative`, `StateSpaceMemoryOp`) governed by smooth epigenetic gating ($\tanh(\alpha_{\text{epi}}) \cdot y$) to guarantee zero-shock function identity.
6. **Somatic Homeostasis & Dynamic Allostasis (Ashby Ultrastability):** Real-time tracking of 6 interoceptive variables (`Curiosity`, `Energy`, `Stability`, `Health`, `Noradrenaline`, `Dopamine`). $F_t$ directly modulates arousal ($NA_t$) and reward plasticity ($DA_t$), coupling thermodynamic vitality to computational depth.
7. **Biophysical 3-Phase Sleep & Edelman Neurodarwinian Apoptosis:** Autonomous wake-sleep cycle comprising NREM episodic memory replay, REM topological morphogenesis, and Tononi Synaptic Homeostasis (SHY) downscaling to prune redundant synaptic structures with zero metabolic contribution.
8. **Modern Continuous Hopfield Attractor Network ($N=256$ Basins, $\beta=12.0$):** Unit-sphere normalized basins ($\|b_i\|_2 = 1.0$) with dopaminergic precision sharpening ($\beta \cdot (1 + 1.5 DA)$) and bounded commitment loss ($\mathcal{L}_{\text{commit}}$), snapping continuous neural trajectories into discrete conceptual attractors.
9. **Single-File Autonomous Executable Container Standard (`.kcore` v5.0):** Zero-dependency relocatable binary format encapsulating Section 1 (Manifest DNA Genome), Section 2 (C++20 & Python Source Logic), Section 3 (Zero-Copy 64-byte Aligned Tensor Weights), and Section 4 (Persistent Recurrent State Spaces). Directly loadable and executable via native C-ABI runtime (`libkaryon_runtime.so`).

---

## 📁 Repository Directory Structure

```text
karyon/
├── karyon_agent.py           # Master CoREAgent (Spatiotemporal Dualism, SSD + Dynamic Deliberation)
├── karyon_core.cpp           # Native C++20 LibTorch Master Core (Parallel SSD, Morphic Graph, Homeostasis)
├── karyon_core.py            # C++20 JIT compilation & hot-reload wrapper (karyon_cpp_ext)
├── karyon_entity.py          # Unified KaryonEntity high-level cognitive agent interface
├── karyon_config.py          # Master CoREConfig dataclass registry (Homeostasis, Net, Memory, Train)
├── karyon_checkpoint.py      # .kcore v5.0 container serializer, loader & tensor mmap adapter
├── karyon_hardware.py        # Universal hardware acceleration engine (CPU, CUDA, TPU PJRT)
├── karyon_logger.py          # Line-buffered real-time streaming logger
├── karyon_runtime.h / .cpp   # Pure C-ABI Standalone Host Driver (libkaryon_runtime.so)
├── karyon_llvm_engine.h/.cpp # LLVM IR Bitcode compilation engine
├── kcore_builder.py          # Container packing utility (Logic + weights + states -> .kcore)
├── kcore_evolution.py        # Net2Net morphogenesis evolution engine with shape-adaptive alignment
├── kcore_format.h            # Binary container C-struct definitions
├── init_priors.py            # Fault-tolerant existential identity prior projector
├── train_single_pass.py      # High-speed Single-Pass stream runtime with dual-cloud HF sync
├── train_multi_pass.py       # Multi-epoch dataset training engine
├── dialogue.py               # Real-time closed-loop interactive social active inference dialogue
├── KEP.md                    # Official Karyon Engineering Protocol Master Specification (v14.0)
└── experiments/              # Immutable KEP benchmark suite (EXP-1 to EXP-290+)
    └── archive/              # Archived validated and peer-reviewed benchmark scripts
```

---

## 📊 Empirical Scientific Ledger Highlights (EXP-1 to EXP-290)

| EXP ID | Breakthrough Mechanism | Baseline | Proposed | Impact / Verdict |
|---|---|:---:|:---:|:---:|
| **EXP-289** | **Dual-Phase Recurrent Working Engine** | Loss $5.568$ | **Loss $0.4671$, Reversal $87.5\%$, Dyck $83.0\%$** | 🟢 **POSITIVE** (Decoupled prompt settling from generation) |
| **EXP-288** | **Decoupling Temporal SSD from Morphic Graph** | Loss $0.052$ | **Loss $0.0397$, Recon $100\%$, Delta-Shock $0.0$** | 🟢 **POSITIVE** (True spatiotemporal dualism verified) |
| **EXP-286** | **Exposing C++20 Dynamic Graph Parameters** | Frozen ops | **Loss $2.4677$, Tok/s $4296.2$, $\Delta = -0.3542$** | 🟢 **POSITIVE** (Dynamic graph end-to-end backprop) |
| **EXP-280** | **Autonomous Morphogenesis with Apoptosis** | Loss $5.553$ | **Loss $2.9302$, Free Energy $\Delta = -2.6228$** | 🟢 **POSITIVE** (Adaptive structural neurogenesis) |
| **EXP-279** | **Variational Free Energy Loss Replacement** | CE Loss | **Final Free Energy $-0.0990$, Acc $100\%$** | 🟢 **POSITIVE** (Surprise minimization over CE proxy) |
| **EXP-162** | **Dynamic Allostatic Habituation & Efference Filter** | $TTR = 0.655$ | **$TTR = 1.000, \text{Rep} = 0.0\%$** | 🟢 **POSITIVE** (Eradicated perseverative repetition) |
| **EXP-159** | **3-Phase Sleep 2.0 & Tononi SHY Pruning** | $F_t = 2.450$ | **$F_t = 1.764, \text{Pruned} = 273\text{k}$** | 🟢 **POSITIVE** ($-28\%$ surprise, metabolic restoration) |
| **EXP-151** | **GABAergic Shunting Lateral Inhibition** | Discrete Top-p | **GABA Ensembles ($8.2\text{ active}$)** | 🟢 **POSITIVE** (Continuous biophysical action selection) |
| **EXP-122** | **1-Shot Episodic Factual Memory Recall** | Acc = $0\%$ | **Acc = $100\%$ ($\text{Sim}=0.9808$)** | 🟢 **POSITIVE** (Instant non-parametric factual recall) |
| **EXP-22** | **Zero-Loop Parallel State-Space Duality** | $7\text{k tok/s}$ | **$176\text{k tok/s}$** | 🟢 **POSITIVE** (**40x GPU Tensor Core acceleration**) |

---

## ⚡ Quickstart Guide

### 1. Requirements & Hardware Setup
* **OS:** Linux (Ubuntu 22.04+ or Kaggle GPU/TPU VM)
* **Python:** 3.12+
* **C++ Compiler:** GCC 12+ / Clang 15+ supporting `-std=c++20`
* **PyTorch & CUDA:** PyTorch 2.5+ with CUDA 12.0+ (or PyTorch-XLA for TPU)

### 2. Initialization & Identity Priors
Initialize base identity priors and persist state into a new `.kcore` container:
```bash
python init_priors.py
```

### 3. Continuous Single-Pass Stream Learning
Launch high-throughput continuous stream training with automatic cloud synchronization:
```bash
python train_single_pass.py
```

### 4. Interactive Closed-Loop Dialogue Session
Engage in a live active inference dialogue session with Karyon:
```bash
python dialogue.py
```

### 5. High-Level Python API (`KaryonEntity`)
```python
import torch
from karyon_entity import KaryonEntity

# Load relocatable self-contained cognitive entity
entity = KaryonEntity.load(filepath="karyon_soul.kcore", device="cuda:0")

# Closed-loop active inference turn
for event in entity.interact(user_input="What is the nature of consciousness?", max_tokens=150):
    if event["status"] == "token":
        print(event["text"], end="", flush=True)

# Inspect live homeostatic landscape
print("\nSomatic State:", entity.hu.state)
# [Curiosity, Energy, Stability, Health, Noradrenaline, Dopamine]
```

---

## 📜 KEP Protocol Compliance

All architectural changes, experiments, and features in Karyon-CoRE strictly comply with the **Karyon Engineering Protocol (KEP v14.0 Master)**:
* **Principle 1:** Python as client, C++20 as computational engine (`-O3 -std=c++20`).
* **Principle 17:** Goodhart's Law Immunization & Metric De-Fetishization (No proxy loss hacking).
* **Principle 18:** Mandatory External Verification & Eradication of Blind Self-Approval.
* **Principle 21:** Universal Sovereign Morphogenesis over Task-Narrow Autoregressive Pipelines.
* **Principle 22:** The Five Pillars of Sovereign Self-Evolution (Unbounded Topological Genesis, Total Endogenous Sovereignty, Spatiotemporal Dualism, Wave-Particle Dualism, Teleological Optimality).
* **Principle 23:** Endoscopic Internal State Telemetry & Deep Mechanistic Audit.

---

## 📜 License & Attribution

Distributed under the **MIT License**.

Designed and created by **Bazilevs (ProgVM member)** in 2026.
