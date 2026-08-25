# `KEP.md` — Karyon Engineering Protocol (KEP) Master Specification

> **Official Cybernetic & Biophysical Protocol for Karyon-CoRE Architecture Development**  
> **Author & Repository Owner:** Bazilevs (ProgVM)  
> **Standard:** KEP v6.1 Master (Mandatory for all human architects and AI collaborators).

---

## 1. Executive Philosophy & Cybernetic Manifesto

Karyon-CoRE is an open-source, non-deterministic cognitive architecture created by **Bazilevs (ProgVM member)** in 2026. It rejects the paradigm of static, dead matrix-multiplication "weight calculators" in favor of a continuous-time, self-regulating biological organism.

Development within this repository is governed strictly by empirical data, biophysical validity, and hardware-conscious engineering.

---

## 2. The 9 Core Operating Principles

1. **Principle 1 (Python as Client, C++20 as Engine):**
   Heavy mathematical operations, State-Space Duality (SSD) scans, Modern Hopfield energy relaxation, and episodic memory slicing run directly on GPU Tensor Cores via compiled C++20 LibTorch (`-O3 -std=c++20`). Python acts strictly as a thin orchestration client. Python-side per-token loops are prohibited.
2. **Principle 2 (Living AGI & Biological Realism — NON-NEGOTIABLE):**
   Every structural feature must be grounded in neuroscience and biophysics:
   * *Active Inference & Free Energy Minimization:* Continuous minimization of true Variational Free Energy ($F_t = D_{\text{KL}} + \mathcal{L}_{\text{rec}}$).
   * *Ashby Somatic Homeostasis:* Real-time tracking of 6 interoceptive variables (`Curiosity`, `Energy`, `Stability`, `Health`, `Noradrenaline`, `Dopamine`).
   * *Canonical Cortical Microcircuit:* 3-Tier Laminar Hierarchy (Layer IV Granular $\to$ Layers II/III Supragranular $\to$ Layer V/VI Infragranular Motor Readout).
   * *Continuous Multi-Timescale Dynamics:* Log-spaced oscillatory spectra spanning $\gamma \to \theta \to \alpha \to \delta$ rhythms.
   * *Rejection of Discrete Hacks:* Artificial transformer artifacts (such as discrete Mixture-of-Experts routing, hard un-smoothed token drops) are strictly forbidden in continuous dynamics.
3. **Principle 3 (Research & Cybernetics First):**
   Before proposing any architectural modification or hypothesis, search and analyze recent literature in neuroscience, cognitive science, and machine learning (Friston, Buzsáki, Hopfield, Mamba, S4).
4. **Principle 4 (Zero Tolerance for Dead Code & Stale Interfaces):**
   All repository files must be 100% synchronized with the latest genome DNA. Obsolete 128D down-projections, unused variables, and orphaned legacy scripts must be refactored or deleted immediately.
5. **Principle 5 (Mandatory Audit Table, Commit Message, and Ready Git Commands):**
   Every code-producing or archiving response MUST conclude strictly at the very end with a Self-Refactoring Audit Table followed by a standardized Conventional Git Commit Message (`type(scope): message`), and immediately inside a ```bash``` block provide the ready-made commands (`git add [./...]`, `git commit -m "..."`, `git push`).
6. **Principle 6 (Direct Module Imports in Experiments):**
   Experimental benchmarks (`experiments/exp_*.py`) must import production modules directly (`karyon_core`, `karyon_agent`) or construct strict self-contained prototypes maintaining full interface fidelity.
7. **Principle 7 (Axiom of Unshackled Flow):**
   Prohibits artificial dimensional bottlenecks (such as legacy $512 \to 128$ down-projections or low-rank $N=64$ basin constraints) unless mathematically mandated by proven cybernetic loss.
8. **Principle 8 (Compositional Depth Over Flat Width):**
   A single-layer network cannot substitute compositional reasoning with exponential width. Multi-stage hierarchical laminar depth is mandatory for complex multi-step reasoning.
9. **Principle 9 (Continuous Spontaneous Dual-Refactoring Mandate):**
   Refactoring is not a deferred or decorative task; it is an active, spontaneous, and continuous obligation executed on two mandatory axes:
   * **Axis A (Engineering & Hardware Safety):** Immediate elimination of runtime bugs, vulnerabilities, PCIe `.item()` sync stalls, unvectorized bottlenecks, and numerical instabilities.
   * **Axis B (Biophysical & KEP Compliance):** Continuous auditing for 100% fidelity to Principle 2, ensuring that no artificial crutches, fake variables, or non-biological shortcuts pollute the architecture.

---

## 3. The 8 Fundamental KEP Rules

### KEP Rule #1 (Hypothesis & Telemetry First)
No feature, layer, or theoretical modification may be merged into production codebase files (`karyon_agent.py`, `karyon_core.cpp`, `train_single_pass.py`) without passing an isolated benchmark script (`experiments/exp_*.py`) logging empirical telemetry.

### KEP Rule #2 (Data-Driven 3-Tier Decision Engine)
Objective metrics dictate verdicts. Every benchmark MUST conclude with an explicit evaluation:
* 🟢 **`POSITIVE`:** Significant improvement in Loss ($\ge 0.08$ on real dataset) or Perplexity with preserved throughput ($\ge 80\%$ baseline) and stable VRAM. Ready for production merge.
* ⚪ **`NEUTRAL / INCONCLUSIVE`:** Delta within statistical noise margin ($\pm 0.05$). No regression, but insufficient gain. Must NOT be merged into production.
* 🔴 **`REJECTED`:** Metric degradation ($\text{Loss}_{\text{prop}} > \text{Loss}_{\text{base}}$), throughput drop ($< 80\%$), or numerical instability ($\text{NaN}$). Discarded and logged in Lessons Learned.

### KEP Rule #3 (Strict Code Preservation — No Placeholders)
All code modifications must be 100% complete, uncompressed, production-grade, and free of placeholders (`...` or `// TODO`). Comments and docstrings must be strictly in English.

### KEP Rule #4 (Diagnostic Speech Sampling)
Every training runtime and benchmark script MUST periodically sample live text generation using Top-p nucleus sampling ($T=0.45, p=0.90$) to visually audit syntax, vocabulary, and semantic coherence across both short casual prompts (`"Hello!"`) and formal technical queries (`"Energy for Earth"`).

### KEP Rule #5 (Cumulative Experimental Continuity)
All validated positive features (🟢 POSITIVE) are permanently preserved and carried forward across all future experiments and production code.

### KEP Rule #6 (Universal Deep Diagnostics Protocol)
Every core script must log real-time sub-millisecond timings, parameter gradient norms across all layers, peak VRAM, and token throughput.

### KEP Rule #7 (Dataset Parity & Reference Baseline Protocol)
Benchmarks MUST evaluate on real datasets (`vicgalle/alpaca-gpt4`). A benchmark may be executed as a standalone evaluation when the canonical reference baseline metrics (Loss, PPL, Speed, VRAM) are already precisely known and documented in the ledger.

### KEP Rule #8 (Mandatory Experiment Archival Protocol)
Upon completion and verdict assignment, all experimental benchmark scripts (`exp_*.py`) are moved to `experiments/archive/` and committed to maintain an immutable scientific ledger.

---

## 4. Hardware & Numerical Safety Axioms (Failure-Prevention Checklist)

Every human architect and AI collaborator must strictly adhere to the following failure-prevention rules to prevent past critical bugs:

1. **Anti-Quadratic Attention:** All continuous time-mixing scans MUST chunk sequences into $Q=64$. Unchunked full attention matrices ($S \times S$) are strictly prohibited to prevent VRAM explosion.
2. **Anti-Empty Masking NaN:** Never divide unweighted losses by zero. Always use `.clamp_min(1.0)` on mask denominators and avoid empty target masking in chunked SFT.
3. **Anti-Amnesia Inter-Chunk Continuity:** Memory state $M_c$ must carry over across chunks and reset *only* at true event boundaries ($\langle\text{eos}\rangle = 257$).
4. **Anti-Sync PCIe Stall:** Never call `.item()` on GPU tensors inside high-frequency token generation loops. Track dynamic capacity using host-side primitives or vectorized tensor masks.
5. **Anti-Brownian Blur on Inference:** Stochastic Wiener noise $dW$ must be modulated by Somatic State ($u_t$) during training/exploration, and clamped/stabilized ($dW=0$ or deterministic) during evaluation/inference.
6. **Anti-Singularity Hopfield Energy:** Never use un-epsiloned Euclidean $p=2$ `torch.cdist` in AMP FP16. Always use Dot-Product Associative Energy (Modern Hopfield Network / Ramsauer 2020) to prevent $\sqrt{0} \to \text{NaN}$.
7. **Anti-Truncation Stream Packing:** Raw byte training must use Continuous Stream Packing ($S=2048$, 0% padding) with $\theta$-phase EOS resets, rather than chopping sentences with static padding.
8. **Anti-Variance Explosion (FP16 LayerNorm Boundary):** Every cross-module projection boundary in AMP FP16 MUST end with a `LayerNorm` to prevent activation variance from exceeding $65,504 \to +\infty \to \text{NaN}$.
