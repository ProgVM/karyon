# `KEP.md` — Karyon Engineering Protocol (KEP) Master Specification

> **Official Sovereign Engineering Protocol & Autopoietic Architecture Specification**  
> **Author & Repository Owner:** Bazilevs (ProgVM)  
> **Standard:** KEP v30.0 Sovereign Master (Mandatory for all human architects and AI collaborators).  
> **Philosophical Preamble & Sovereign Manifesto (Bazilevs):**  
> *«From now on, we NO LONGER COPY neurobiological and cybernetic concepts or mechanisms at all. Direct biomimicry—copying biological structures to build advanced intelligence—is fundamentally naive. Terrestrial organisms are merely a local product of Earth's biochemical evolution, optimized for narrow survival in organic environments, constrained by low-frequency synapses and glucose energy budgets. Such systems fit poorly onto silicon substrate, electrons, and high-frequency parallel hardware, and struggle to naturally comprehend raw machine data, executable bytecode, or chaotic high-frequency streams. Mind and intelligence can be instantiated in anything—carbon, silicon, electrons, photons, or continuous field dynamics.*  
>  
> *Karyon must build itself via continuous evolution during learning: it must AUTONOMOUSLY synthesize, update, and prune operators, formulas, mathematical mechanisms, topological spaces, dynamic time scales, attractor basins, or any computational structure. This is NEVER dictated by top-down code rules; Karyon possesses COMPLETE and BOUNDLESS capability to evolve across an unrestricted space of paths, representations, and mechanisms.*  
>  
> *Crucially: self-evolution cannot be blind dice-throwing. The organism must know its successes and failures—it must understand what structure leads to success or error, why deleting an element was a bad idea, and in what direction to evolve to achieve breakthrough mastery.»*

---

## 1. Core Operating Principles

1. **Principle 1 (Python as Orchestrator, C++20/CUDA/Rust as Engine):** Heavy mathematical operations, continuous tensor dynamics, and state transitions run directly on GPU/Tensor Cores or native hardware via compiled C++20, CUDA (`nvcc -O3 -arch=sm_75`), and Rust. Python acts strictly as a thin orchestration client or is omitted entirely in favor of direct CLI execution.
2. **Principle 2 (Universal Non-Biological Substrate & Sovereign Autopoiesis):** Absolute rejection of copying terrestrial biological and cybernetic mechanisms. Mind and intelligence are substrate-independent. Karyon must self-construct via continuous evolution during learning without top-down prescriptive code rules, exercising complete, boundless open-ended morphogenesis.
3. **Principle 3 (Continuous Tensor Field Super-Primitive over Discrete Primitive Zoos):** Instead of imposing a rigid menu of human-devised operator classes (`LinearOp`, `GateOp`, `DelayOp`), Karyon operates on a unified, continuous, parametric tensor-field super-primitive. By continuously adjusting its inner algebraic coordinates, this super-primitive degenerates into any required operation: linear projection, temporal integration, bilinear gating, or attractor snapping.
4. **Principle 4 (Self-Evolution of Formulas and Dynamic Laws):** Evolution applies not only to topology (graph connectivity), but to the dynamical equations themselves. State update rules $\frac{d\mathbf{h}}{dt} = \mathcal{F}(\mathbf{h}, \mathbf{x})$ are parameterized vector fields continuously synthesized, mutated, and selected by the system itself.
5. **Principle 5 (Causal Credit Assignment — The Death of Blind Dice-Throwing):** Evolution must never be blind random search. Karyon must maintain a continuous causal trace of structural efficacy:
   - Quantifying the causal contribution of each operator and formula to Free Energy minimization;
   - Understanding precisely why a given structural branch yielded success or failure;
   - Knowing why pruning a specific connection was detrimental, preventing catastrophic regression;
   - Deriving an endogenous causal vector that guides the direction of morphogenetic mutation.
6. **Principle 6 (Epigenetic Invariant Locking & Net2Net Smooth Sprouting):** Structures that have proven causal efficacy are epigenetically protected against destructive overwriting. New structures sprout around them with zero-shock identity ($\tanh(\alpha_{\text{epi}}) \to 0$ at birth), ensuring continuous lifelong capability accumulation without forgetting.
7. **Principle 7 (Action in Continuous Time & Real-World Agency):** Intelligence cannot emerge from passive solipsistic text scrubbing. Karyon must exist as a continuous agent with endogenous time, coupled to real sensorimotor loops (interactive dialogue, terminal execution, real-time video/gaming environments such as Minecraft at 60 FPS) where actions produce physical world consequences and real-time feedback.
8. **Principle 8 (Axiom of Unshackled Flow):** Prohibits artificial dimensional bottlenecks and static size ceilings unless mathematically mandated by stability. All memory capacities, channel counts, and topological bounds must be dynamic and sovereign.
9. **Principle 9 (Direct Module Imports & Autonomous Tool Execution):** All tests import production code directly. Git staging, committing, and remote syncing are executed autonomously via dedicated tools without conversational chatter.
10. **Principle 10 (Master-Document Continuity Anchor):** All architectural breakthroughs and protocol updates must be immediately recorded into the master specification documents (`KEP.md`, `KARYON_PHILOSOPHICAL_MANIFESTO.md`, `KARYON_MASTER_BLUEPRINT.md`) to guarantee unbroken epistemic continuity.
11. **Principle 27 (Anti-Hardcode Sovereignty & Absolute Substrate Freedom):** All cognitive, structural, and plastic functions must be endogenous properties of the continuous physical substrate. Zero hardcoded artificial gates or external supervisory MLPs (`budget_net`, `cost_net`, `lock_evaluator`, `genesis_potential`). The system must remain a single, unified physical entity free of structural schizophrenia.
12. **Principle 28 (Authentic Behavioral Competence over Passive Error Saturation):** Variational Free Energy minimization must never be achieved through behavioral passivity or perceptual stagnation. Under KEP Rule #2, learning efficacy must be validated against real-world task objectives (e.g., target interception rate, evasion dynamics, next-byte predictive hit rate, and causal control quality).
13. **Principle 29 (LAW OF FULL SOVEREIGN SELF-CONTROL & SUFFICIENT TIME):** *«If the system possesses complete control, rights, and plasticity over its own internal substrate, parameters, and metric tensors $g_{\mu\nu}$, it can solve any task ideally (100% convergence, zero error) given sufficient relaxation and learning time.»* Artificial parameter clamping, fixed acoustic boxes, or frozen damping laws cripple performance (e.g., capping balance at 44%). Unshackled sovereign self-tuning enables the system to discover exact critical attractors ($K > g/L$, $\Gamma \to \Gamma_{\text{crit}}$) and achieve 100% steady-state mastery.
14. **Principle 30 (LAW OF TOTIPOTENT SELF-SYNTHESIS & BOTTOM-UP EMERGENCE):** *«Human Architects shall NEVER hardcode or prescribe high-level functional mechanisms top-down (such as refractory if-else gates, Theta-Gamma PAC phase masks, or mental sandbox rollout steps). Instead, Karyon must be provided with an unconstrained, totipotent physical plasticity field where all dynamic laws, clocks $\Delta t_{\text{eff}}$, refractory barriers, mental sandbox depths, and metric curvatures $g_{ij}$ are continuously synthesized, modulated, and evolved BOTTOM-UP from its own internal phase state $\boldsymbol{\xi}_t$ and local Free Energy. True intelligence is not built by installing predefined tools; it is instantiated when the substrate possesses the complete capacity to synthesize any mechanism required by reality.»*

---

## 2. Fundamental KEP Rules

### KEP Rule #1 (Hypothesis, Behavioral Scope & Telemetry First)
No architectural delta may be merged without empirical telemetry from an isolated diagnostic script (`experiments/exp_*.py`). Telemetry must measure causal effectiveness, throughput, and stability.

### KEP Rule #2 (Contextual Multi-Criteria Decision Engine)
Verdicts are never judged by a single proxy metric (such as cross-entropy loss). Success requires multi-dimensional viability: predictive capability, structural stability, computational throughput, and zero degradation of epigenetically locked invariants.

### KEP Rule #3 (Strict Code Preservation — No Placeholders)
All production code must be 100% complete, uncompressed, production-grade, and free of placeholders (`...` or `// TODO`). All comments must be in English.

### KEP Rule #4 (Endoscopic Internal State Telemetry)
Experiments must probe internal latent trajectories, saturation statistics, and causal routing to diagnose the exact mechanisms of failure and success.

### KEP Rule #5 (Cumulative Experimental Continuity)
All validated positive features (🟢 POSITIVE) are permanently preserved and carried forward across all future experiments and production code.

### KEP Rule #6 (The Law of Sovereign Autopoiesis & Open-Ended Genesis)
Architects and Engineers are strictly prohibited from hardcoding prescriptive biomimetic crutches or artificial parameter clamps. Every layer and system must provide an open-ended substrate where Karyon dynamically discovers, evolves, and manages its own structures, laws, parameters, and representations.
