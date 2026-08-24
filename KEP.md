# `KEP.md` — Karyon Engineering Protocol (KEP) Master Specification
# (some of these rules describe the work for AI assistents)

> **Official Cybernetic Protocol for Karyon-CoRE Architecture Development**  
> **Author & Repository Owner:** Bazilevs (ProgVM)  
> **Standard:** KEP v5.1 (Mandatory for all human developers and AI collaborators).

---

## 1. Core Operating Principles

1. **Principle 1 (Python as Client, C++20 as Engine):** Heavy mathematical operations, State-Space Duality scans, and memory slicing run on GPU/Tensor Cores via compiled C++20 LibTorch (`-O3 -std=c++20`). Python acts strictly as a thin orchestration client.
2. **Principle 2 (Living AGI & Biological Realism):** Every feature must be grounded in biophysics (Active Inference, Ashby Somatic Homeostasis, Cortical Laminar Hierarchy, Neural Oscillations).
3. **Principle 3 (Research & Cybernetics First):** Search neuroscience and machine learning literature (Friston, Buzsáki, Mamba, MegaByte, Hopfield) before formulating hypotheses.
4. **Principle 4 (Zero Tolerance for Dead Code):** All modules must be synchronized. Obsolete, unreferenced, or failing legacy files must be immediately updated or deleted.
5. **Principle 5 (Conventional Commit & Ready Git Commands at the Very End):** Every AI response that creates or modifies code must conclude strictly at the very end with a Self-Refactoring Audit Table followed by the Conventional Git Commit Message (`type(scope): message`), and immediately inside a ```bash``` code block provide the three ready-made commands (`git add [./...]`, `git commit -m "..."`, `git push`).
6. **Principle 6 (Direct Module Imports in Experiments):** Benchmark scripts must import production modules directly or construct strict self-contained experimental prototypes.
7. **Principle 7 (Axiom of Unshackled Flow):** Prohibits artificial bottlenecking (such as legacy $512 \to 128$ down-projections or low-rank $N=64$ basin constraints) unless mathematically mandated by proven cybernetic loss.

---

## 2. The 8 Fundamental KEP Rules

### KEP Rule #1 (Hypothesis & Telemetry First)
No feature, layer, or theory is merged into production codebase files (`karyon_agent.py`, `karyon_core.cpp`, `train_single_pass.py`) without passing an isolated benchmark script (`experiments/exp_*.py`) logging before/after empirical telemetry.

### KEP Rule #2 (Data-Driven 3-Tier Decisions)
Objective metrics dictate verdicts. Every benchmark MUST conclude with an explicit evaluation:
* 🟢 **`POSITIVE`:** Significant improvement in Loss ($\ge 0.08$ on real dataset) or Perplexity with preserved throughput ($\ge 80\%$ baseline) and stable VRAM. Ready for production merge.
* ⚪ **`NEUTRAL / INCONCLUSIVE`:** Delta within statistical noise margin ($\pm 0.05$). No regression, but insufficient gain. Must NOT be merged into production.
* 🔴 **`REJECTED`:** Metric degradation ($\text{Loss}_{\text{prop}} > \text{Loss}_{\text{base}}$), throughput drop ($< 80\%$), or numerical instability ($\text{NaN}$). Discarded and logged in Lessons Learned.

### KEP Rule #3 (Strict Code Preservation)
All code modifications must be 100% complete, uncompressed, production-grade, and free of placeholders (`...` or `// TODO`). Comments and docstrings must be strictly in English.

### KEP Rule #4 (Diagnostic Speech Sampling)
Every training runtime and benchmark script MUST periodically sample live text generation using Top-p nucleus sampling ($T=0.7, p=0.90$) to visually audit syntax, vocabulary, and semantic coherence.

### KEP Rule #5 (Cumulative Experimental Continuity)
All validated positive features (🟢 POSITIVE) are preserved and carried forward across all future experiments and production code.

### KEP Rule #6 (Universal Deep Diagnostics Protocol)
Every core script must log real-time sub-millisecond timings, parameter gradient norms across all layers, peak VRAM, and token throughput.

### KEP Rule #7 (Production Dataset & Environment Parity Protocol)
No hypothesis aimed at generalization or plateau-breaking may be evaluated on trivial single-sentence synthetic loops. Benchmarks MUST evaluate across a representative slice of the production dataset (`vicgalle/alpaca-gpt4`, 50–100 real multi-topic batches, $B=32, S=512$) or on checkpointed models (`karyon_soul.kcore`) in the convergence regime.

### KEP Rule #8 (Mandatory Experiment Archival Protocol)
Upon completion and verdict assignment, all experimental benchmark scripts (`exp_*.py`) are moved to `experiments/archive/` to keep the root repository clean and maintain an immutable historical scientific ledger.
