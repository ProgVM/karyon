# Karyon-CoRE (Continuous Recurrent Engine) v5.0
> **Master Architectural Blueprint & Context Registry for AI Assistant / Collaborators**

## 1. Executive Summary & Core Philosophy

**Karyon-CoRE** is an autonomous, non-deterministic cognitive architecture created by **Bazilevs (ProgVM member)** in 2026. It operates at the raw UTF-8 byte level ($V=258$), implementing Active Inference, Homeostatic State-Space Modeling, and 2nd-Order Stochastic Differential Equation (SDE) Recurrent Neural Integration accelerated via C++20 LibTorch extensions.

### Fundamental Principles:
1. **Byte-Level Universal Representation ($V=258$):** Eliminates tokenizers and BPE dictionaries. Maps all inputs into a single unified embedding space ($D=128/256$).
2. **2nd-Order Stochastic Heun SDE Recurrent Core:** Models continuous non-linear hidden state dynamics with hardware-derived Wiener noise ($\sigma \cdot dW_t$).
3. **Active Inference & Variational Free Energy Engine:** Generates internal prior and posterior latent distributions ($z_t$) and minimizes Free Energy $F_t$.
4. **Somatic Homeostasis & Ashby Ultrastability:** Tracks 6 interoceptive variables (`Curiosity`, `Energy`, `Stability`, `Health`, `Noradrenaline`, `Dopamine`) to modulate attention, memory recall, and time-delta integration ($dt$).
5. **Self-Contained Container Standard (`.kcore` v4.0):** Entire model logic (C++ / LLVM Bitcode), weights, genome DNA, and live states are encapsulated into a single portable binary file (`karyon_soul.kcore`).
6. **Self-Directed Neuroevolution (Morphogenesis):** Karyon autonomously evaluates, mutates, and expands its own neural topology ($hidden\_dim$, $latent\_dim$) using lossless Net2Net weight transformations driven by Free Energy minimization.

---

## 2. Theoretical & Mathematical Foundations

### A. 2nd-Order Stochastic Heun SDE Solver
Recurrent state transition follows a Stochastic Differential Equation solved via Stratonovich Predictor-Corrector integration:

$$\text{Predictor: } \tilde{h}_{t+1} = h_t + f(h_t, w_t, u_t) \Delta t_{\text{eff}} + \Delta W_t$$
$$\text{Corrector: } h_{t+1} = \tanh\left( h_t + \frac{1}{2} \left[ f(h_t, \dots) + f(\tilde{h}_{t+1}, \dots) \right] \Delta t_{\text{eff}} + \Delta W_t \right)$$

where $\Delta W_t \sim \mathcal{N}(0, \Delta t_{\text{eff}} \cdot \sigma^2 I)$ and $\Delta t_{\text{eff}} = \text{clamp}(dt \cdot (1.0 + 1.2 \cdot NA - 0.4 \cdot DA), 0.20, 2.50)$.

### B. Variational Free Energy Engine ($F_t$)
Active Inference minimizes the upper bound on surprise:

$$F_t = D_{\text{KL}}\left( Q(z_t \mid h_{t-1}, w_t) \parallel P(z_t \mid h_{t-1}) \right) + \mathcal{L}_{\text{rec}}(\hat{w}_t, w_t)$$

* **Dimension-Normalized KL Divergence:**
  $$D_{\text{KL}} = \frac{1}{d_z} \sum_{i=1}^{d_z} \frac{1}{2} \left[ \log \frac{\sigma_{\text{prior}, i}^2}{\sigma_{\text{post}, i}^2} + \frac{\sigma_{\text{post}, i}^2 + (\mu_{\text{post}, i} - \mu_{\text{prior}, i})^2}{\sigma_{\text{prior}, i}^2} - 1 \right]$$

### C. Continuous Differentiable Somatic Memory Gate
Volitional memory recall is self-governed by Karyon's internal neurotransmitter landscape:

$$G_{\text{recall}} = \sigma\left( 2.0 \cdot NA_t + 1.5 \cdot \text{Curiosity}_t - 0.5 \cdot (1.0 - \text{Energy}_t) \right)$$
$$W_{\text{integrated}} = W_{\text{current}} + G_{\text{recall}} \odot \text{Memory.Read}(W_{\text{current}})$$

---

## 3. The `.kcore` Binary Container Format Spec

The `.kcore` file is an autonomous, relocatable binary package:

```text
+-----------------------------------------------------------------------+
|  HEADER (32 Bytes)                                                    |
|  Magic: "KCORE\x02\x00\x00" | Sections: 4 | TotalSize: N | Flags: 0    |
+-----------------------------------------------------------------------+
|  SECTION 1: MANIFEST (JSON DNA Genome, layer shapes, tensor map)     |
+-----------------------------------------------------------------------+
|  SECTION 2: LOGIC (C++ Source / Compiled LLVM IR Bitcode .bc)         |
+-----------------------------------------------------------------------+
|  SECTION 3: WEIGHTS (64-byte aligned zero-copy tensor parameters)     |
+-----------------------------------------------------------------------+
|  SECTION 4: PERSISTENT STATE (h_fast, h_slow, u_t, Episodic Memory)   |
+-----------------------------------------------------------------------+
```

---

## 4. Codebase Directory Structure & File Roles

* **`karyon_core.cpp`**: C++17/20 LibTorch extension containing `ByteTokenizer`, `HomeostaticUnit`, `SensoryGatewayImpl`, `MotorGatewayImpl`, `DynamicRecurrentCore` (SDE Heun), `LatentPredictorImpl` (Active Inference), and `BatchedEpisodicMemoryImpl` (100% CUDA BMM memory).
* **`karyon_core.py`**: JIT compiler wrapper using `torch.utils.cpp_extension`.
* **`karyon_config.py`**: Master Dataclass configurations (`HomeostasisConfig`, `NetworkConfig`, `MemoryConfig`, `TrainConfig`).
* **`karyon_agent.py`**: `CoREAgent` PyTorch master module. Implements `forward()`, `forward_sequence()`, `generate_thought_and_speech()`, `get_complete_state_dict()`, and `load_complete_state_dict()`.
* **`karyon_checkpoint.py`**: Container serializer (`save_karyon`, `load_karyon`). Manages binary parsing, tensor mmap, and batch size/capacity adaptation (`adapt_and_copy_batch_buffer`).
* **`kcore_builder.py`**: Container packer utility that converts source logic, parameters, and states into `.kcore`.
* **`kcore_format.h`**: C++ struct definitions for binary header and section headers.
* **`kcore_evolution.py`**: `KaryonEvolver` neuroevolution engine. Implements Net2Net identity-preserving matrix expansion with strict layer-submatrix column offset alignment.
* **`karyon_runtime.h` / `karyon_runtime.cpp`**: Standalone C-ABI Runtime (`libkaryon_runtime.so`) exposing `karyon_load`, `karyon_perceive_stream`, `karyon_step`, `karyon_express_stream`, `karyon_adapt`, and `karyon_save`.
* **`init_priors.py`**: Fault-tolerant script/module `initialize_priors(recreate=False)` that projects identity priors into latent space.
* **`train_real_world.py`**: High-throughput GPU training script with sequence unrolling, telemetry dashboard logging, and Plateau-Driven Neuroevolution.
* **`train_single_pass.py`**: Single-Pass Stream Learning ($N=1$, zero epochs) with Error-Gated Neuromodulated Plasticity and moving-average thresholding.
* **`dialogue.py`**: Interactive Closed-Loop Social Active Inference session tracking human text reactions, calculating human surprise ($F_{\text{human}}$), and updating somatic states.

---

## 5. Karyon Engineering Protocol (KEP) & Lessons Learned

### KEP Rules:
1. **Hypothesis & Telemetry First (`exp_*.py`):** No feature or theory is pushed to main codebase files without an isolated experimental benchmark script that logs before/after telemetry.
2. **Data-Driven Decisions:** If an experiment degrades Free Energy or PPL, the mutation/theory is rejected and documented as a counter-example.
3. **Strict Parameter Persistence:** 100% of C++ and Python parameters must be registered and preserved during container saves.

### Proven Positive Pathways (What WORKS):
* **Pure Byte-Level $V=258$:** Achieved $PPL = 1.57$ (Speech Loss = $0.4504$) on raw UTF-8 text with 99.8% parameter savings compared to BPE tokenizers.
* **100% Vectorized CUDA Memory:** Using `torch::bmm`, `masked_fill`, and `index_put_` eliminated CPU-GPU sync stalls.
* **Concat-Aware Net2Net Submatrix Alignment:** Zero-padded layer expansion ($hidden\_dim: 256 \to 320 \to 480$, $latent\_dim: 64 \to 160$) preserves pre-trained representations with zero loss jump.
* **Shape-Adaptive State Restoring (`_safe_copy_param`):** Enables loading checkpoints across different dimension configurations without crashing.
* **GPU Sequence Unrolling (`forward_sequence`):** Eliminates Python `for`-loop overhead per token, pushing GPU utilization to 95%+.

### Proven Negative Pathways (What FAILS / Counter-Examples):
* **Direct 150k BPE LLM Vocab forcing onto SDE core:** Caused token soup, memory bloat (830MB), and broken generation loops.
* **Unvectorized C++ `.item()` loops:** Caused GPU pipeline stalls ($5.6\text{s}$ per batch).
* **Missing `.named_parameters()` in PyBind11:** Caused 95% of C++ weights to freeze during Adam optimization (stuck at Loss = $6.05$).
* **Unnormalized $D_{\text{KL}}$ Summation:** Summing KL over $d_z$ penalised larger models purely for having more dimensions.
* **Adam Step on Zero-Weights during candidate warm-up:** Adam's $\frac{g}{\sqrt{v}+\epsilon}$ division by $\epsilon$ caused $\pm lr$ jumps on zero-initialized weights, destroying Net2Net identity.

---

## 6. Next Architectural Milestones (v5.0+)

1. **System 2 Mental Sandbox / Latent Rollout:**
   Executing $K$ internal forward simulation steps in `LatentPredictor` ($h_t \to z_{t+1} \to \hat{w}_{t+1} \to h_{t+1}$) before generating motor output, selecting actions that minimize Expected Free Energy ($G$).
2. **Spontaneous Autonomous Intent:**
   When no human input is received, high `Curiosity` and `Energy` trigger spontaneous internal thoughts, episodic memory queries, and self-initiated dialogue turns.
3. **Sleep & Replay Memory Consolidation Phase:**
   When `Energy < 0.30`, Karyon enters `Sleep State`, replaying high-surprise memories from `BatchedEpisodicMemory` to consolidate synaptic weights and prune noise.
4. **Multimodal Byte Streaming Integration:**
   Connecting `SensoryStream` and `MotorStream` C-ABI structs to real-time raw PCM audio and vision pixel buffers.
