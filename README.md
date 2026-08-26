# Karyon-CoRE (Continuous Recurrent Engine) v10.0 Master

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![C++](https://img.shields.io/badge/C%2B%2B-20-red.svg)](https://isocpp.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-orange.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.0%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Throughput](https://img.shields.io/badge/Throughput-176%2C000%2B%20tok%2Fs-brightgreen.svg)](https://github.com/ProgVM/karyon)

> **Autonomous, Non-Deterministic Cognitive Architecture driven by Active Inference, Zero-Loop State-Space Duality (176,000+ tok/s), SwiGLU Channel-Mixing, Somatic Homeostasis, and Single-File Executable Binary Containers (`.kcore` v4.0).**

Created by **Bazilevs (ProgVM member)** in 2026.

---

## 🚀 Key Architectural Innovations (v10.0)

Karyon-CoRE transcends static matrix-multiplication deep learning models ("weight calculators") toward a living, continuous-time cognitive entity:

1. **Raw UTF-8 Byte Universal Representation ($V=258$):** Eliminates static tokenizers and BPE vocabularies. Maps text, vision, audio, motor efference, and homeostatic body signals into a unified embedding space ($D=128/256$).
2. **Zero-Loop Parallel State-Space Duality Scan (Time-Mixing @ 176k tok/s):** Replaces serial recurrence loops with closed-form parallel matrix scanning ($\mathbf{Y}\sb{\text{chunk}} = \mathbf{Y}\sb{\text{intra}} + \mathbf{Y}\sb{\text{inter}}$), collapsing 512-step batch latency from $3700\text{ ms} \to \mathbf{92\text{ ms}}$ (**40x speedup**!).
3. **Parallel SwiGLU Knowledge Synthesis Block (Channel-Mixing):** Combines continuous-time temporal recurrence with dense non-linear cross-channel associative reasoning (`expand_dim = 1536`), eliminating byte-level entropy plateaus.
4. **Non-Linear Causal Byte Receptive Field ($K=4$, SiLU):** Applies causal 1D depthwise convolutions over byte streams, converting raw bytes into stable phonemes and morphemes with rolling-buffer test-time consistency.
5. **Afferent-Efferent Lexical Weight Tying:** Directly couples the motor readout projection with transposed byte embeddings ($W\sb{\text{emb}}^T / \sqrt{D\sb{\text{text}}}$), ensuring non-vanishing gradient highways ($\|\nabla W\sb{\text{emb}}\| > 0.04$).
6. **Multi-Modal Global Workspace Theory (GWT):** Unifies text, vision, motor efference, body signals, and mental queries via competitive attention selection into a single conscious frame.
7. **Somatic Homeostasis (Ashby Ultrastability):** Tracks 6 interoceptive variables (`Curiosity`, `Energy`, `Stability`, `Health`, `Noradrenaline`, `Dopamine`) to modulate plasticity, reactive temporal step $\Delta t$, and memory recall.
8. **Event-Boundary Theta Phase Reset:** Clears inter-chunk matrix carryover on $\langle\text{eos}\rangle$ (`id=257`), eliminating cross-prompt semantic interference across short dialogues.
9. **Active Capacity C++ Episodic Memory:** Dynamic slicing `keys.slice(1, 0, max_active)` and 2D-aligned similarity gating for $O(1)$ associative recall.
10. **Single-File Executable Container (`.kcore` v4.0):** Encapsulates C++ source, LLVM IR Bitcode, aligned weights, DNA genome, and dynamic persistent state spaces into a single portable binary file (`karyon_soul.kcore`).

---

## 📁 Repository Directory Structure

```text
karyon/
├── karyon_config.py          # Master CoREConfig registry (JSON serializable)
├── karyon_agent.py           # Production Master CoREAgent v15.2 (SSD + SwiGLU)
├── karyon_core.cpp           # Native C++20 LibTorch Master Core (10 cognitive systems)
├── karyon_core.py            # C++ JIT compilation wrapper (karyon_cpp_ext_v20)
├── karyon_checkpoint.py      # .kcore binary container serializer and loader
├── karyon_logger.py          # Unified logger with line-buffered stdout streaming
├── karyon_runtime.h / .cpp   # C-ABI Standalone Host Driver (libkaryon_runtime.so)
├── karyon_llvm_engine.h/.cpp # LLVM IR Bitcode compilation engine
├── kcore_builder.py          # Container packing utility
├── kcore_evolution.py        # Net2Net morphogenesis evolution engine
├── kcore_format.h            # Binary container struct definitions
├── init_priors.py            # Fault-tolerant existential identity prior projector
├── train_single_pass.py      # High-speed Multi-Pass streaming runtime (52k dataset)
├── dialogue.py               # Closed-loop interactive social dialogue session
└── experiments/              # KEP Isolated benchmark suite (exp_*.py)
```

---

## ⚡ Quickstart Guide

### 1. Requirements
* Linux (Ubuntu 22.04+ / Kaggle CUDA GPU)
* Python 3.12+
* PyTorch 2.5+ with CUDA 12.0+
* GCC / Clang C++20 compiler with Ninja build tool

### 2. Initialization & Identity Priors
Initialize base identity priors and persist state into a new `.kcore` container:
```bash
python init_priors.py
```

### 3. Massive High-Speed Training (176k tok/s)
Run multi-pass continuous streaming on 52,002 Alpaca-GPT4 dialogues:
```bash
python train_single_pass.py
```

### 4. Interactive Closed-Loop Dialogue Session
Engage in real-time active inference dialogue with Karyon:
```bash
python dialogue.py
```

---

## 🔬 Karyon Engineering Protocol (KEP) Rules

Karyon-CoRE development strictly follows the **KEP Protocol**:
1. **Hypothesis & Telemetry First:** No architectural modification is merged without passing an isolated benchmark (`exp_*.py`) logging before/after telemetry.
2. **Data-Driven Decisions:** Objective telemetry (Loss, PPL, Latency, Free Energy) dictates verdict (🟢 `POSITIVE`, ⚪ `NEUTRAL`, 🔴 `REJECTED`).
3. **Strict Code Preservation:** 100% complete, uncompressed code without placeholders.
4. **Diagnostic Speech Sampling:** Mandatory periodic generation auditing under Top-p nucleus sampling.
5. **Cumulative Experimental Continuity:** Validated positive features are carried forward across all modules.
6. **Universal Deep Diagnostics:** Real-time logging of sub-millisecond timings, gradient norms, VRAM, and somatic state.

---

## 📜 License & Attribution

Distributed under the **MIT License**.

Designed and created by **Bazilevs (ProgVM member)** in 2026.
 
