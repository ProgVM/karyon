# `KEP.md` — Karyon Engineering Protocol (KEP) Master Specification
# (some of these rules describe the work for AI assistants)

> **Official Cybernetic & Biophysical Protocol for Karyon-CoRE Architecture Development**  
> **Author & Repository Owner:** Bazilevs (ProgVM)  
> **Standard:** KEP v6.2 Master (Mandatory for all human architects and AI collaborators).

---

## 1. Core Operating Principles

1. **Principle 1 (Python as Client, C++20 as Engine):** Heavy mathematical operations, State-Space Duality scans, and memory slicing run on GPU/Tensor Cores via compiled C++20 LibTorch (`-O3 -std=c++20`). Python acts strictly as a thin orchestration client.
2. **Principle 2 (Living AGI & Biological Realism — NON-NEGOTIABLE):** Every feature must be grounded in biophysics (Active Inference, Ashby Somatic Homeostasis, Cortical Laminar Hierarchy, Continuous Neural Oscillations). Rejection of discrete hacks (No discrete MoE).
3. **Principle 3 (Research & Cybernetics First):** Search neuroscience and machine learning literature (Friston, Buzsáki, Hopfield, Mamba) before formulating hypotheses.
4. **Principle 4 (Zero Tolerance for Dead Code):** All modules must be synchronized. Obsolete, unreferenced, or failing legacy files must be immediately updated or deleted.
5. **Principle 5 (Mandatory Audit Table, Commit Message, and Ready Git Commands):** Every AI response that creates, modifies, or archives code MUST conclude strictly at the very end with a Self-Refactoring Audit Table followed by a standardized Conventional Git Commit Message (`type(scope): message`), and immediately inside a ```bash``` block provide the ready-made commands (`git add [./...]`, `git commit -m "..."`, `git push`).
6. **Principle 6 (Direct Module Imports in Experiments):** Benchmark scripts must import production modules directly or construct strict self-contained prototypes maintaining full interface fidelity.
7. **Principle 7 (Axiom of Unshackled Flow):** Prohibits artificial dimensional bottlenecks (such as legacy $512 \to 128$ down-projections or low-rank $N=64$ basin constraints) unless mathematically mandated by proven cybernetic loss.
8. **Principle 8 (Compositional Depth Over Flat Width):** A single-layer network cannot substitute compositional reasoning with exponential width. Multi-stage hierarchical laminar depth is mandatory for complex multi-step reasoning.
9. **Principle 9 (Continuous Spontaneous Dual-Refactoring Mandate):** Refactoring is an active, spontaneous, and continuous obligation executed on Axis A (Technical & Hardware Safety) and Axis B (Biophysical & KEP Compliance).

---

## 2. The 8 Fundamental KEP Rules

### KEP Rule #1 (Hypothesis & Telemetry First)
No feature, layer, or theoretical modification may be merged into production codebase files without passing an isolated benchmark script (`experiments/exp_*.py`) logging empirical telemetry.

### KEP Rule #1.1 (Mandatory Debugging to Completion Principle)
If an experiment benchmark terminates abnormally, crashes, or produces inconclusive/distorted results due to internal implementation bugs, tensor broadcasting mismatches, or un-epsiloned mathematical singularities in the benchmark code itself, the researcher/AI MUST persist and iteratively debug the code to completion until a clean, bug-free, and unambiguous empirical benchmark run is achieved and documented.

### KEP Rule #2 (Data-Driven 3-Tier Decision Engine)
Objective metrics dictate verdicts:
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
Benchmarks MUST evaluate on real datasets (`vicgalle/alpaca-gpt4`). A benchmark may be executed as a standalone evaluation when canonical reference baseline metrics are already precisely known and documented in the ledger.

### KEP Rule #8 (Mandatory Experiment Archival Protocol)
Upon completion and verdict assignment, all experimental benchmark scripts (`exp_*.py`) are moved to `experiments/archive/` and committed to maintain an immutable scientific ledger.
