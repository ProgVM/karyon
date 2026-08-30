# Karyon-CoRE (Continuous Recurrent Engine) v30.0 Master

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![C++](https://img.shields.io/badge/C%2B%2B-20-red.svg)](https://isocpp.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-orange.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.0%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Container](https://img.shields.io/badge/Container-.kcore%20v4.2-brightgreen.svg)](https://github.com/ProgVM/karyon)

> **Autonomous, Non-Deterministic Cognitive Architecture driven by Active Inference, Zero-Loop Parallel State-Space Duality, 2-Stage PW-LPER Cascaded Cortical Stacks, True Will Volitional Override Engine, System 2 Active Inference Mental Sandbox Search, Morphemic Boundary Macro-Resets, Somatic Homeostasis, and Single-File Autonomous Executable Binary Containers (`.kcore` v4.2).**

Created by **Bazilevs (ProgVM member)** in 2026.

---

## 🚀 Key Architectural Innovations (v30.0)

Karyon-CoRE transcends static matrix-multiplication deep learning models ("weight calculators") toward a living, continuous-time cognitive entity:

1. **Raw UTF-8 Byte Universal Representation ($V=258$):** Eliminates static tokenizers and BPE vocabularies (`pad=256`, `eos=257`). Maps text, vision, audio, motor efference, and homeostatic body signals into a unified embedding space ($D_{\text{text}}=256, D_{\text{hidden}}=768$).
2. **System 2 Active Inference Mental Sandbox Search (EXP-100 Validated 🟢):** Pauses motor output on high-entropy boundaries ($H > 0.70$) or word transitions to execute $K=6$ candidate counterfactual rollouts in `LatentPredictor` over $T=3$ steps, selecting trajectories that minimize Expected Free Energy $G(a)$.
3. **Entropy-Peak Morphemic Boundary Macro-Reset (EXP-100 Validated 🟢):** Snaps combined cortical representations to unit-sphere normalized Hopfield attractor basins on word transitions, eliminating cumulative byte-level error propagation and pseudo-morphemic drift.
4. **Hierarchical Volitional Override Module (True Will Engine - EXP-99 Validated 🟢):** Top-down cognitive goals (Stage 2) dynamically override bottom-up somatic fatigue ($\text{Energy} = 0.05$) and pain via Volitional Override Gate ($\Gamma_{\text{override}}$), logging Allostatic Strain (Health Debt).
5. **Continuous Volitional Active Inference Motor Module (EXP-98 Validated 🟢):** Evaluates Expected Free Energy $G(a)$ for motor trajectories under homeostatic prior preferences $P(u)$ and modulates motor readout logits directly.
6. **Native C++20 2-Stage Cascaded Cortical Stack:**
   * **Stage 1 (Fast Morpho-Syntactic Cortical Sheet):** Fast SSD ($\beta \in [0.005, 0.15]$) + Causal ConvSwiGLU $K=3$ (3072D expand) + Pre-LayerNorm Residual Highway.
   * **Precision-Weighted Laminar Error Routing (PW-LPER):** Computes top-down prediction errors and precision weights ($\pi_t$) to route ascending surprise signals to higher cortical sheets.
   * **Stage 2 (Slow Semantic-Discourse Cortical Sheet):** Slow SSD ($\beta \in [0.0001, 0.05]$) + Causal ConvSwiGLU $K=7$ (3072D expand) + Pre-LayerNorm Residual Highway.
7. **Multi-Scale Morphological Byte Pyramid Receptive Field (EXP-70):** Multi-kernel parallel 1D depthwise convolutions ($K=2, 4, 8$) with dynamic softmax scale gating.
8. **Modern Hopfield Attractor Network ($N=256$ Basins):** Continuous attractor landscape with dopamine ($DA$) precision sharpening and bounded commitment loss ($\mathcal{L}_{\text{commit}}$).
9. **Active Inference Latent World Model ($F_t$):** Generates internal prior and posterior latent distributions ($z_t$) and minimizes Variational Free Energy $F_t = D_{\text{KL}}(q(z)\|p(z)) + \mathcal{L}_{\text{rec}}$.
10. **Somatic Homeostasis & Dynamic Allostasis (Ashby Ultrastability):** Tracks 6 interoceptive variables (`Curiosity`, `Energy`, `Stability`, `Health`, `Noradrenaline`, `Dopamine`). Perceptive listening actively recovers metabolic energy (Magistretti 2015), while speech expends energy.
11. **Single-File Executable Binary Container (`.kcore` v4.2):** Zero-dependency portable binary packaging encapsulating Section 1 (Manifest DNA), Section 2 (Full Python & C++ Source Bundle), Section 3 (Zero-Copy Aligned Tensor Weights), and Section 4 (Persistent State Spaces). Executable without Python via native C-ABI (`libkaryon_runtime.so`).

---

## 📁 Repository Directory Structure

```text
karyon/
├── karyon_config.py          # Master CoREConfig dataclass registry
├── karyon_agent.py           # Production Master CoREAgent v30.0 (True Will & System 2 Sandbox)
├── karyon_core.cpp           # Native C++20 LibTorch Master Core (16 cognitive systems)
├── karyon_core.py            # C++ JIT compilation wrapper (karyon_cpp_ext_v24)
├── karyon_checkpoint.py      # .kcore binary container serializer, loader & extractor
├── karyon_logger.py          # Unified logger with line-buffered stdout streaming
├── karyon_runtime.h / .cpp   # Pure C-ABI Standalone Host Driver (libkaryon_runtime.so)
├── karyon_llvm_engine.h/.cpp # LLVM IR Bitcode compilation engine
├── kcore_builder.py          # Container packing utility
├── kcore_evolution.py        # Net2Net morphogenesis evolution engine
├── kcore_format.h            # Binary container C-struct definitions
├── init_priors.py            # Fault-tolerant existential identity prior projector
├── train_single_pass.py      # High-speed Multi-Pass streaming runtime (52k dataset, N=5)
├── dialogue.py               # Closed-loop interactive social active inference dialogue
├── diag_profile_pipeline.py  # Deep CUDA event pipeline profiler
└── experiments/              # KEP Isolated benchmark suite (exp_*.py)
```

---

## ⚡ Quickstart Guide

### 1. Requirements
* Linux (Ubuntu 22.04+ / Kaggle CUDA GPU)
* Python 3.12+ (for development/training)
* PyTorch 2.5+ with CUDA 12.0+
* GCC / Clang C++20 compiler with Ninja build tool

### 2. Initialization & Identity Priors
Initialize base identity priors and persist state into a new `.kcore` container:
```bash
python init_priors.py
```

### 3. Massive High-Speed Training (Alpaca-GPT4, N=5 Passes)
Run multi-pass continuous packed streaming on 52,002 Alpaca-GPT4 dialogues:
```bash
python train_single_pass.py
```

### 4. Interactive Closed-Loop Dialogue Session
Engage in real-time active inference dialogue with Karyon:
```bash
python dialogue.py
```

---

## 📜 License & Attribution

Distributed under the **MIT License**.

Designed and created by **Bazilevs (ProgVM member)** in 2026.
