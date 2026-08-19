# Karyon-CoRE (Continuous Recurrent Engine) v5.5

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![C++](https://img.shields.io/badge/C%2B%2B-20-red.svg)](https://isocpp.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-orange.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.0%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)

> **Autonomous, Non-Deterministic Cognitive Architecture driven by Active Inference, 2nd-Order Stochastic Heun SDE Recurrent Core, Somatic Homeostasis, and Executable Binary Containers (`.kcore`).**

Created by **Bazilevs (ProgVM member)** in 2026.

---

## 🚀 Key Architectural Innovations

Unlike static feedforward deep learning models ("weight calculators"), **Karyon-CoRE** operates as a continuous-time, living cognitive entity:

1. **Raw UTF-8 Byte Universal Representation ($V=258$):** Eliminates static tokenizers and BPE vocabularies. Operates directly on raw UTF-8 byte streams ($V=258$), preserving 99.8% parameter efficiency.
2. **2nd-Order Stochastic Heun SDE Recurrent Core:** Models continuous non-linear hidden dynamics using Stratonovich Predictor-Corrector integration with hardware-derived Wiener noise ($\sigma \cdot dW_t$).
3. **Active Inference & Variational Free Energy ($F_t$):** Minimizes prediction surprise ($F_t = D_{\text{KL}} + \mathcal{L}_{\text{rec}}$) normalized across latent dimensions.
4. **Mastery-Gated DFET v3 Plasticity Gating:** Reduces online training Backprop FLOPs by **>60–90%** via statistical variance thresholding ($\mu + k_{\sigma} \sigma$), resting during predictable streams and conserving somatic energy.
5. **Bi-Directional Local Predictive Coding (KEP #11.1):** Enables real-time local layer error updates ($\mathbf{e}_{\text{combined}} = \mathbf{e}_{\text{bottom}} + 0.5 \cdot \mathbf{e}_{\text{topdown}}$) with zero global `.backward()` autograd graph locks, accelerating perception throughput to **100,000+ tokens/sec**.
6. **System 2 Active Inference Search Engine:** Pauses motor output during high surprise ($F_t > \text{Threshold}$) to execute counterfactual latent rollouts minimizing Expected Free Energy ($G$).
7. **Volitional Memory Read Gating (KEP #8):** Restricts episodic memory retrieval (`episodic_mem.read()`) to high-arousal ($NA > 0.12$) or uncertainty steps, saving 97.6% memory search FLOPs.
8. **Autonomous Net2Net Morphogenesis:** Performs identity-preserving, zero-loss structural neural topology growth ($hidden\_dim: 256 \to 320 \to 512$).
9. **Single-File Executable Container (`.kcore` v4.0):** Encapsulates C++ source/LLVM IR, aligned weights, DNA manifest, and dynamic persistent state spaces into a single portable binary file.

---

## 📁 Repository Directory Structure

```text
karyon/
├── karyon_config.py          # Unified CoREConfig registry (JSON serializable)
├── karyon_agent.py           # Master CoREAgent PyTorch architecture
├── karyon_core.cpp           # Native C++20 LibTorch JIT extension (SDE Heun, Memory)
├── karyon_core.py            # C++ JIT compilation and loader wrapper
├── karyon_checkpoint.py      # .kcore binary container serializer and loader
├── karyon_logger.py          # Unified logger with line-buffered stdout
├── karyon_runtime.h / .cpp   # C-ABI Standalone Host Driver (libkaryon_runtime.so)
├── karyon_llvm_engine.h/.cpp # LLVM IR Bitcode compilation engine
├── kcore_builder.py          # Container packing utility
├── kcore_evolution.py        # Net2Net morphogenesis evolution engine
├── kcore_format.h            # Binary container struct definitions
├── init_priors.py            # Fault-tolerant existential identity prior projector
├── train_single_pass.py      # High-speed Single-Pass streaming runtime (N=1)
├── dialogue.py               # Closed-loop interactive social dialogue session
├── diag_profile_pipeline.py  # Microsecond CUDA event profiler & diagnostic tool
└── experiments/              # KEP Isolated benchmark scripts (exp_*.py)
```

---

## ⚡ Quickstart Guide

### 1. Requirements & Dependencies
* Linux / Kaggle / Colab (Ubuntu 22.04+)
* Python 3.12+
* PyTorch 2.5+ with CUDA 12.0+
* GCC / Clang C++20 compiler

### 2. Initialization & Priors Projection
Initialize base identity priors into a new `.kcore` container:
```bash
python init_priors.py
```

### 3. Continuous Single-Pass Streaming (N=1 Learning)
Run high-throughput active inference streaming on Parquet conversational data:
```bash
python train_single_pass.py
```

### 4. Interactive Closed-Loop Dialogue Session
Engage in a live active inference dialogue session with Karyon:
```bash
python dialogue.py
```

### 5. Run CUDA Pipeline Diagnostics
Run low-level CUDA event timing and gradient flow inspection:
```bash
python diag_profile_pipeline.py
```

---

## 🔬 Karyon Engineering Protocol (KEP)

Karyon-CoRE development strictly follows the **KEP Protocol**:
1. **Hypothesis & Telemetry First:** No architectural modification is merged into main production files (`karyon_agent.py`, `karyon_core.cpp`) without passing an isolated benchmark script (`exp_*.py`) with side-by-side telemetry comparison.
2. **Data-Driven Decisions:** Modifications degrading Free Energy or latency are rejected and logged in the KEP Lessons Learned Registry.
3. **Cumulative Experimental Continuity (Rule #5):** All validated positive features (🟢 POSITIVE VERDICT) are preserved and carried forward across all future experiments.
4. **Universal Deep Diagnostics (Rule #6):** Every core script features in-line process diagnostics tracking timings, gradient norms, VRAM, and live speech sampling.

---

## 📜 License & Attribution

Distributed under the **MIT License**.

Designed and created by **Bazilevs (ProgVM member)** in 2026.

