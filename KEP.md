# `KEP.md` — Karyon Engineering Protocol (KEP) Master Specification

> **Official Cybernetic & Biophysical Protocol for Karyon-CoRE Architecture Development**  
> **Author & Repository Owner:** Bazilevs (ProgVM)  
> **Standard:** KEP v9.0 Master (Mandatory for all human architects and AI collaborators).

---

## 1. Core Operating Principles

1. **Principle 1 (Python as Client, C++20 as Engine):** Heavy mathematical operations, State-Space Duality scans, and memory slicing run on GPU/Tensor Cores or TPU accelerators via compiled C++20 LibTorch and PyTorch-XLA (`-O3 -std=c++20`). Python acts strictly as a thin orchestration client.
2. **Principle 2 (Living AGI & Biological Realism — NON-NEGOTIABLE):** Every feature must be grounded in biophysics (Active Inference, Ashby Somatic Homeostasis, Cortical Laminar Hierarchy, Continuous Neural Oscillations). Rejection of discrete hacks (No discrete MoE).
3. **Principle 3 (Research & Cybernetics First):** Search neuroscience and machine learning literature (Friston, Buzsáki, Hopfield, Mamba) before formulating hypotheses.
4. **Principle 4 (Zero Tolerance for Dead Code):** All modules must be synchronized. Obsolete, unreferenced, or failing legacy files must be immediately updated or deleted.
5. **Principle 5 (Autonomous Tool Action Over Text Clutter):** Git staging, committing, and remote syncing are executed directly and autonomously via specialized tools (`git_commit`, `git_push`, `sync_agent_database`). Conversational responses must remain clean, direct, and analytical—strictly omitting redundant markdown `bash` commit blocks or formatted commit refactoring tables in chat responses.
6. **Principle 6 (Direct Module Imports in Experiments):** Benchmark scripts must import production modules directly or construct strict self-contained prototypes maintaining full interface fidelity.
7. **Principle 7 (Axiom of Unshackled Flow):** Prohibits artificial dimensional bottlenecks (such as legacy $512 \to 128$ down-projections or low-rank $N=64$ basin constraints) unless mathematically mandated by proven cybernetic loss.
8. **Principle 8 (Compositional Depth Over Flat Width):** A single-layer network cannot substitute compositional reasoning with exponential width. Multi-stage hierarchical laminar depth is mandatory for complex multi-step reasoning.
9. **Principle 9 (Continuous Spontaneous Dual-Refactoring Mandate):** Refactoring is an active, spontaneous, and continuous obligation executed on Axis A (Technical & Hardware Safety) and Axis B (Biophysical & KEP Compliance).
10. **Principle 10 (Autonomy of Protocol Evolution):** The Lead AI Cyberneticist possesses full, self-directed autonomy to update and refine the KEP specification, operational rules, and architectural guidelines whenever a more optimal, elegant, or biologically realistic algorithm or workflow is discovered, or when directed by Bazilevs.
11. **Principle 11 (Deliberative Ideation First — "Think Before Code"):** Architectural concepts, paradigm shifts, and theoretical reframings MUST be fully elaborated, stress-tested, and polished to architectural perfection across multi-step theoretical deliberation BEFORE writing any production or benchmark code.

---

## 2. The 10 Fundamental KEP Rules

### KEP Rule #1 (Hypothesis, Behavioral Scope & Telemetry First)
No feature, layer, or theoretical modification may be merged into production codebase files without passing an isolated benchmark or diagnostic observation script (`experiments/exp_*.py`) logging empirical telemetry. Experiments are NOT restricted strictly to model training or gradient optimization; they may observe and evaluate live dialogue sessions, emergent behavioral dynamics, memory consolidation fidelity, somatic homeostasis trajectories, or any specific cognitive phenomenon. Telemetry and diagnostic logging MUST explicitly capture and reflect the specific target objects or phenomena under study.

### KEP Rule #1.1 (Mandatory Debugging to Completion Principle)
If an experiment benchmark terminates abnormally, crashes, or produces inconclusive/distorted results due to internal implementation bugs, tensor broadcasting mismatches, or un-epsiloned mathematical singularities in the benchmark code itself, the researcher/AI MUST persist and iteratively debug the code to completion until a clean, bug-free, and unambiguous empirical benchmark run is achieved and documented.

### KEP Rule #2 (Contextual Multi-Criteria Decision Engine — Telemetry in Context)
The definitive success criterion is NOT restricted to scalar Cross-Entropy Loss alone. Verdicts MUST be judged based on the **context-specific target telemetry metric(s)** defined in the experimental hypothesis, evaluated across the full cognitive and hardware landscape:
1. **Context-Specific Target Metrics:**
   * **Compute & Hardware Acceleration:** Throughput gain ($\text{tok/s}$), Tensor Core residency, VRAM/HBM reduction, JIT kernel latency.
   * **Thermodynamic & Free Energy Efficiency:** Variational surprise minimization ($F_t = D_{\text{KL}} + \mathcal{L}_{\text{rec}}$), epistemic entropy, Expected Free Energy ($G$).
   * **Long-Horizon Context & Retrieval:** Episodic recall accuracy, pattern separation cosine orthogonality, Hopfield attractor basin stability.
   * **Somatic Homeostasis & Biological Vitality:** Metabolic balance, noradrenergic/dopaminergic modulation stability ($\text{Energy} > 0.15, \text{Health} > 0.20$).
   * **Generative Quality & Morphology:** Syntactic integrity, absence of pseudo-morphemic drift, phonotactic correctness via PAC decoding.
2. **Standard Decision Criteria:**
   * **🟢 `POSITIVE`:** Significant empirical breakthrough in the **target metric(s) of the hypothesis** (e.g. substantial throughput increase $\ge 110\%$, reduction in Free Energy, memory footprint savings, or Loss $\Delta \ge 0.08$) while maintaining systemic stability and zero degradation in non-target critical invariants. Merged into production.
   * **⚪ `NEUTRAL / INCONCLUSIVE`:** Target metrics within statistical noise margin ($\pm 0.05$) without distinct multi-dimensional gain or trade-off. Retained in archive for further cybernetic refinement.
   * **🔴 `REJECTED`:** Critical metric degradation, numerical divergence ($\text{NaN}$), kernel crash, or severe uncompensated collapse in secondary vital functions. Discarded and logged in Lessons Learned.

### KEP Rule #3 (Strict Code Preservation — No Placeholders)
All code modifications must be 100% complete, uncompressed, production-grade, and free of placeholders (`...` or `// TODO`). Comments and docstrings must be strictly in English.

### KEP Rule #4 (Diagnostic Speech Sampling)
Every training runtime and benchmark script MUST periodically sample live text generation using Top-p nucleus sampling ($T=0.45, p=0.90$) to visually audit syntax, vocabulary, and semantic coherence across both short casual prompts (`"Hello!"`) and formal technical queries (`"Energy for Earth"`).

### KEP Rule #5 (Cumulative Experimental Continuity)
All validated positive features (🟢 POSITIVE) are permanently preserved and carried forward across all future experiments and production code.

### KEP Rule #6 (Universal Deep Diagnostics Protocol)
Every core script must log real-time sub-millisecond timings, parameter gradient norms across all layers, peak VRAM/HBM, and token throughput.

### KEP Rule #7 (Dataset Parity & Reference Baseline Protocol)
Benchmarks MUST evaluate on real datasets (`vicgalle/alpaca-gpt4`). A benchmark may be executed as a standalone evaluation when canonical reference baseline metrics are already precisely known and documented in the ledger.

### KEP Rule #8 (Mandatory Experiment Archival Protocol)
Upon completion and verdict assignment, all experimental benchmark scripts (`exp_*.py`) are moved to `experiments/archive/` and committed to maintain an immutable scientific ledger.

### KEP Rule #9 (The Persistence of Cybernetic Truth)
If a biologically realistic or mathematically elegant hypothesis fails or yields inconclusive results (⚪ NEUTRAL or 🔴 REJECTED) on the first attempt, it must not be permanently abandoned. The Lead AI Cyberneticist must dissect the failure, identify numerical or structural bottlenecks (e.g., gradient flow, scaling, or missing complementary systems), and reformulate the mechanism with alternative approaches until its true potential is either fully realized or mathematically disproven.

### KEP Rule #10 (Universal Hardware Agnosticism & Zero Hardcode Directive)
Code across Karyon-CoRE MUST be completely hardware-agnostic (seamlessly supporting CPU, CUDA GPU, and Google Cloud / Kaggle TPU via PyTorch-XLA). ZERO hardcoded magic numbers, fixed layer dimensions, static device strings (`"cuda"` or `"cpu"`), or hardcoded hyperparameter constants are permitted in production layers or training loops. All dimensions, thresholds, decay rates, step parameters, and hardware controls MUST be dynamically configured via dataclasses (`CoREConfig`, `HardwareConfig`) and adaptively derived at runtime.