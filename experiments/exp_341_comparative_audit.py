"""
=====================================================================================
EXP-341: CONTROLLED COMPARATIVE STREAM & DEEP ENDOSCOPIC DIAGNOSIS
=====================================================================================
Direct Head-to-Head Comparative Streaming Audit on identical continuous stream slices:
  - Branch A: Complex Cognitive Tetrad (TriScalePAC + MorphicGraph + Anokhin + Sandbox + HDC)
  - Branch B: Elementary Atomic Basis (4 Math Primitives + C-SSD Soliton + ContinuousHopfield)

Telemetry Scope:
  1. Loss & Perplexity Trajectories every 50 steps (50, 100, 200, 300, 400)
  2. Diagnostic Speech Samples (KEP Rule #4) at step 200 and step 400
  3. Repetitive n-gram perseveration rate (%)
  4. Computational Profile: tok/s, step latency (ms), peak VRAM (MB)
=====================================================================================
"""
import os
import sys
import time
from typing import Dict, List, Tuple
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath("."))
import karyon_core as kc
import karyon_agent


# =====================================================================================
# BRANCH B: ELEMENTARY ATOMIC BASIS MODEL (EXP-336/337 FOUNDATIONAL ATOMS)
# =====================================================================================
class BranchBAtomicCore(nn.Module):
    """
    Pure lightweight atomic core:
      - 4 elementary mathematical primitives:
          1. Continuous Leaky Integrator (dx/dt = f(x, u))
          2. Bilinear Multiplicative Conjunction (m1 * m2)
          3. Linear Projector (W_proj * x)
          4. Stochastic Langevin Perturbation (xi ~ N(0, sigma^2))
      - C-SSD Soliton Scan (CausalParallelSSD)
      - Modern Continuous Hopfield Attractor Memory
    Zero heavy sandbox rollouts, zero hardcoded symbolic wrappers.
    """
    def __init__(self, vocab_size: int = 258, dim: int = 256, device_str: str = "cuda"):
        super().__init__()
        self.device = torch.device(device_str)
        self.dim = dim
        self.vocab_size = vocab_size

        # Universal Byte Embedding
        self.emb = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.emb.weight, 0.0, 0.02)

        # 1. C-SSD Soliton Temporal Scan Engine
        self.ssd = kc.CausalParallelSSD(dim, device_str)

        # 2. Elementary Mathematical Atoms
        self.atom_proj = nn.Linear(dim, dim)
        self.atom_m1 = nn.Linear(dim, dim, bias=False)
        self.atom_m2 = nn.Linear(dim, dim, bias=False)
        self.atom_w_dt = nn.Linear(dim, dim)
        self.atom_w_decay = nn.Linear(dim, dim)
        self.atom_w_noise = nn.Linear(dim, dim)

        # 3. Continuous Hopfield Memory Engine
        self.hopfield = kc.ContinuousHopfieldMemory(dim, 32, device_str, 256)

        # Motor Output Readout Head (Weight-tied)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.emb.weight
        self.to(self.device)

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        # input_seq: [B, S]
        x = self.emb(input_seq)  # [B, S, D]

        # Step 1: Causal SSD Soliton Scan
        h_ssd = self.ssd.forward(x)  # [B, S, D]

        # Step 2: Elementary Atomic Circuit Processing
        u_lin = self.atom_proj(h_ssd)
        m1 = self.atom_m1(h_ssd)
        m2 = self.atom_m2(h_ssd)
        conjunction_flux = m1 * m2

        decay = F.softplus(self.atom_w_decay(h_ssd)) + 1e-4
        dt = torch.sigmoid(self.atom_w_dt(h_ssd)) * 0.4 + 0.05
        sigma = F.softplus(self.atom_w_noise(h_ssd)) * 0.01
        noise = torch.randn_like(h_ssd) * sigma if self.training else 0.0

        dx_dt = u_lin + conjunction_flux - (decay * h_ssd) + noise
        h_atomic = h_ssd + dt * dx_dt

        # Step 3: Continuous Hopfield Attractor Snapping
        B, S, D = h_atomic.shape
        h_flat = h_atomic.view(B * S, D)
        h_snapped = self.hopfield.forward(h_flat).view(B, S, D)
        h_out = h_atomic + 0.1 * h_snapped

        return self.head(h_out)


# =====================================================================================
# DIAGNOSTIC HELPER FUNCTIONS
# =====================================================================================
def compute_n_gram_perseveration(tokens: List[int], n: int = 3) -> float:
    """
    Computes percentage of repetitive n-grams in the token sequence.
    Repetition rate = 1.0 - (unique_ngrams / total_ngrams)
    """
    if len(tokens) <= n:
        return 0.0
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return (repeated / len(ngrams)) * 100.0


def sample_speech(model: nn.Module, prompt_text: str = "The nature of mind is", max_tokens: int = 70, temp: float = 0.65, is_branch_a: bool = True) -> Tuple[str, float]:
    """
    Samples autoregressive diagnostic speech and measures n-gram perseveration rate.
    """
    model.eval()
    device = next(model.parameters()).device
    seed_bytes = list(prompt_text.encode("utf-8"))
    curr_tokens = torch.tensor([seed_bytes], dtype=torch.long, device=device)
    generated = list(seed_bytes)

    with torch.no_grad():
        for _ in range(max_tokens):
            if is_branch_a:
                logits = model(curr_tokens, thinking_steps=1)
            else:
                logits = model(curr_tokens)

            next_logits = logits[0, -1, :256] / max(temp, 1e-4)
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)
            curr_tokens = torch.tensor([generated[-128:]], dtype=torch.long, device=device)

    # Compute perseveration rate on the generated tail
    gen_tail = generated[len(seed_bytes):]
    perseveration_rate = compute_n_gram_perseveration(gen_tail, n=3)
    text = bytes([b for b in generated if b < 256]).decode("utf-8", errors="replace")
    return text, perseveration_rate


# =====================================================================================
# MASTER COMPARATIVE AUDIT RUNNER
# =====================================================================================
def run_exp_341_comparative_audit():
    print("=" * 85)
    print("EXP-341: CONTROLLED A/B STREAMING COMPARATIVE AUDIT (400 STEPS)")
    print("Branch A (Complex Tetrad) vs Branch B (Elementary Atomic Basis)")
    print("=" * 85)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Device: {device_str.upper()} | AMP: bfloat16")

    # Load stream
    stream_path = "data/karyon_multidomain_single_pass_stream.npy"
    if not os.path.exists(stream_path):
        raise FileNotFoundError(f"Stream dataset not found at {stream_path}")
    
    stream_mmap = np.load(stream_path, mmap_mode="r")
    total_tokens = len(stream_mmap)
    print(f"Stream loaded from {stream_path}: {total_tokens:,} tokens.")

    # Fixed hyperparameters for 100% fair parity
    batch_size = 16
    seq_len = 512
    max_steps = 400
    stride = batch_size * seq_len
    checkpoint_steps = [50, 100, 200, 300, 400]

    # Pre-extract exact batch indices to guarantee 100% identical data feed
    print(f"Pre-indexing {max_steps} batches of size ({batch_size}, {seq_len})...")
    batch_data_cache = []
    for step in range(max_steps):
        s_idx = (step * stride) % (total_tokens - stride - 2)
        raw_slice = stream_mmap[s_idx : s_idx + stride].astype(np.int64)
        tensor_x = torch.from_numpy(raw_slice.reshape(batch_size, seq_len))
        target_y = torch.roll(tensor_x, shifts=-1, dims=1)
        batch_data_cache.append((tensor_x, target_y))
    print(f"Dataset indexed: {len(batch_data_cache)} identical batches prepared in memory.")

    # ---------------------------------------------------------------------------------
    # Instantiate Models
    # ---------------------------------------------------------------------------------
    print("\n--- Instantiating Branch A: Complex Cognitive Tetrad ---")
    branch_a_agent = karyon_agent.CoREAgent(vocab_size=258, embed_dim=256, device=device_str).to(device)
    # Ensure foundational operators are bound in DynamicMorphicGraph
    if branch_a_agent.graph.k_nodes == 0:
        branch_a_agent.graph.add_node("LinearAccumulator", 256)
        branch_a_agent.graph.add_node("BilinearMultiplicative", 256)

    opt_a = optim.AdamW(branch_a_agent.parameters(), lr=1e-3, weight_decay=1e-4)

    print("\n--- Instantiating Branch B: Elementary Atomic Basis Core ---")
    branch_b_model = BranchBAtomicCore(vocab_size=258, dim=256, device_str=device_str).to(device)
    opt_b = optim.AdamW(branch_b_model.parameters(), lr=1e-3, weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss()

    # Telemetry logging containers
    telemetry_a: Dict[str, List] = {"steps": [], "loss": [], "ppl": [], "latency_ms": [], "tok_s": [], "vram_mb": [], "persev": [], "samples": {}}
    telemetry_b: Dict[str, List] = {"steps": [], "loss": [], "ppl": [], "latency_ms": [], "tok_s": [], "vram_mb": [], "persev": [], "samples": {}}

    print("\n" + "=" * 85)
    print("EXECUTING HEAD-TO-HEAD STREAMING SESSIONS (400 STEPS EACH)")
    print("=" * 85)

    # ---------------------------------------------------------------------------------
    # RUN BRANCH A
    # ---------------------------------------------------------------------------------
    print("\n>>> STARTING RUN: BRANCH A (COMPLEX TETRADA) <<<")
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    branch_a_agent.train()
    start_time_a = time.time()

    for step in range(1, max_steps + 1):
        x, y = batch_data_cache[step - 1]
        x, y = x.to(device), y.to(device)

        t0 = time.perf_counter()
        opt_a.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda" if "cuda" in device_str else "cpu", dtype=torch.bfloat16):
            logits = branch_a_agent(x, thinking_steps=2)
            loss = criterion(logits.view(-1, 258), y.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(branch_a_agent.parameters(), 1.0)
        opt_a.step()

        # Step latency & throughput
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        tok_s = (batch_size * seq_len) / max(dt, 1e-5)
        loss_val = float(loss.item())
        ppl_val = float(np.exp(min(loss_val, 15.0)))
        vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

        if step in checkpoint_steps:
            telemetry_a["steps"].append(step)
            telemetry_a["loss"].append(loss_val)
            telemetry_a["ppl"].append(ppl_val)
            telemetry_a["latency_ms"].append(dt * 1000.0)
            telemetry_a["tok_s"].append(tok_s)
            telemetry_a["vram_mb"].append(vram_mb)

            # Sample speech at steps 200 and 400
            sample_txt = ""
            persev = 0.0
            if step in [200, 400]:
                sample_txt, persev = sample_speech(branch_a_agent, "The nature of mind is", max_tokens=65, temp=0.65, is_branch_a=True)
                telemetry_a["samples"][step] = sample_txt
                telemetry_a["persev"].append(persev)
                branch_a_agent.train()

            print(f"[Branch A | Step {step:03d}/400] Loss: {loss_val:.4f} | PPL: {ppl_val:6.2f} | Latency: {dt*1000:6.1f}ms | {tok_s:6.1f} tok/s | VRAM: {vram_mb:.1f}MB")
            if sample_txt:
                print(f"  └─ Speech Sample [Step {step}]: {repr(sample_txt[:75])}... (Perseveration: {persev:.1f}%)")

    total_time_a = time.time() - start_time_a
    avg_tok_s_a = (max_steps * batch_size * seq_len) / total_time_a
    print(f"\nBranch A Completed in {total_time_a:.1f}s | Average Throughput: {avg_tok_s_a:.1f} tok/s")

    # ---------------------------------------------------------------------------------
    # RUN BRANCH B
    # ---------------------------------------------------------------------------------
    print("\n>>> STARTING RUN: BRANCH B (ELEMENTARY ATOMIC BASIS) <<<")
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    branch_b_model.train()
    start_time_b = time.time()

    for step in range(1, max_steps + 1):
        x, y = batch_data_cache[step - 1]
        x, y = x.to(device), y.to(device)

        t0 = time.perf_counter()
        opt_b.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda" if "cuda" in device_str else "cpu", dtype=torch.bfloat16):
            logits = branch_b_model(x)
            loss = criterion(logits.view(-1, 258), y.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(branch_b_model.parameters(), 1.0)
        opt_b.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        tok_s = (batch_size * seq_len) / max(dt, 1e-5)
        loss_val = float(loss.item())
        ppl_val = float(np.exp(min(loss_val, 15.0)))
        vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

        if step in checkpoint_steps:
            telemetry_b["steps"].append(step)
            telemetry_b["loss"].append(loss_val)
            telemetry_b["ppl"].append(ppl_val)
            telemetry_b["latency_ms"].append(dt * 1000.0)
            telemetry_b["tok_s"].append(tok_s)
            telemetry_b["vram_mb"].append(vram_mb)

            # Sample speech at steps 200 and 400
            sample_txt = ""
            persev = 0.0
            if step in [200, 400]:
                sample_txt, persev = sample_speech(branch_b_model, "The nature of mind is", max_tokens=65, temp=0.65, is_branch_a=False)
                telemetry_b["samples"][step] = sample_txt
                telemetry_b["persev"].append(persev)
                branch_b_model.train()

            print(f"[Branch B | Step {step:03d}/400] Loss: {loss_val:.4f} | PPL: {ppl_val:6.2f} | Latency: {dt*1000:6.1f}ms | {tok_s:6.1f} tok/s | VRAM: {vram_mb:.1f}MB")
            if sample_txt:
                print(f"  └─ Speech Sample [Step {step}]: {repr(sample_txt[:75])}... (Perseveration: {persev:.1f}%)")

    total_time_b = time.time() - start_time_b
    avg_tok_s_b = (max_steps * batch_size * seq_len) / total_time_b
    print(f"\nBranch B Completed in {total_time_b:.1f}s | Average Throughput: {avg_tok_s_b:.1f} tok/s")

    # ---------------------------------------------------------------------------------
    # PLOTTING DIAGNOSTIC COMPARISON CHARTS
    # ---------------------------------------------------------------------------------
    plot_path = "experiments/exp_341_comparative_audit.png"
    os.makedirs("experiments", exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("#101010")

    for ax in axes:
        ax.set_facecolor("#181818")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("#444444")
        ax.grid(True, color="#2e2e2e", linestyle=":")

    # Panel 1: Loss Trajectory
    axes[0].plot(telemetry_a["steps"], telemetry_a["loss"], marker="o", color="#ff0055", linewidth=2.0, label="Branch A: Complex Tetrad")
    axes[0].plot(telemetry_b["steps"], telemetry_b["loss"], marker="s", color="#00ffcc", linewidth=2.0, label="Branch B: Elementary Atomic Basis")
    axes[0].set_title(f"Loss Convergence (400 Steps)\nFinal A: {telemetry_a['loss'][-1]:.4f} vs Final B: {telemetry_b['loss'][-1]:.4f}")
    axes[0].set_xlabel("Continuous Stream Steps")
    axes[0].set_ylabel("Cross-Entropy Loss (nats/byte)")
    axes[0].legend(facecolor="#222222", labelcolor="white")

    # Panel 2: Computational Throughput
    axes[1].plot(telemetry_a["steps"], telemetry_a["tok_s"], marker="o", color="#ff0055", linewidth=2.0, label="Branch A Throughput")
    axes[1].plot(telemetry_b["steps"], telemetry_b["tok_s"], marker="s", color="#00ffcc", linewidth=2.0, label="Branch B Throughput")
    axes[1].set_title(f"Throughput Comparison\nAvg A: {avg_tok_s_a:.0f} tok/s vs Avg B: {avg_tok_s_b:.0f} tok/s")
    axes[1].set_xlabel("Continuous Stream Steps")
    axes[1].set_ylabel("Tokens / Second")
    axes[1].legend(facecolor="#222222", labelcolor="white")

    # Panel 3: N-Gram Perseveration Rate (%)
    steps_eval = [200, 400]
    x_indices = np.arange(len(steps_eval))
    width = 0.35
    axes[2].bar(x_indices - width/2, telemetry_a["persev"], width, color="#ff0055", alpha=0.8, label="Branch A (Tetrad)")
    axes[2].bar(x_indices + width/2, telemetry_b["persev"], width, color="#00ffcc", alpha=0.8, label="Branch B (Atomic)")
    axes[2].set_title(f"N-Gram Perseveration Rate (%) at Steps 200 & 400\nLower is better")
    axes[2].set_xticks(x_indices)
    axes[2].set_xticklabels([f"Step {s}" for s in steps_eval])
    axes[2].set_ylabel("Repetitive 3-Gram Rate (%)")
    axes[2].legend(facecolor="#222222", labelcolor="white")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  ✓ Comparative Diagnostic Plot Saved: {plot_path}")

    # ---------------------------------------------------------------------------------
    # CONCISE COMPARATIVE SUMMARY TABLE
    # ---------------------------------------------------------------------------------
    print("\n" + "=" * 95)
    print("EXP-341: CONCISE COMPARATIVE AUDIT TELEMETRY SUMMARY")
    print("=" * 95)
    header = f"{'Step':<6} | {'Loss (A)':<10} | {'Loss (B)':<10} | {'PPL (A)':<10} | {'PPL (B)':<10} | {'tok/s (A)':<10} | {'tok/s (B)':<10} | {'VRAM A/B (MB)'}"
    print(header)
    print("-" * 95)
    for i, st in enumerate(checkpoint_steps):
        row = (
            f"{st:<6} | "
            f"{telemetry_a['loss'][i]:<10.4f} | "
            f"{telemetry_b['loss'][i]:<10.4f} | "
            f"{telemetry_a['ppl'][i]:<10.2f} | "
            f"{telemetry_b['ppl'][i]:<10.2f} | "
            f"{telemetry_a['tok_s'][i]:<10.1f} | "
            f"{telemetry_b['tok_s'][i]:<10.1f} | "
            f"{telemetry_a['vram_mb'][i]:.1f} / {telemetry_b['vram_mb'][i]:.1f}"
        )
        print(row)
    print("=" * 95)

    print("\n--- DIAGNOSTIC SPEECH AUDIT (KEP RULE #4) ---")
    print(f"Prompt: 'The nature of mind is'")
    print(f"Step 200:")
    print(f"  • Branch A (Tetrad)  : {repr(telemetry_a['samples'][200])}")
    print(f"  • Branch B (Atomic)  : {repr(telemetry_b['samples'][200])}")
    print(f"Step 400:")
    print(f"  • Branch A (Tetrad)  : {repr(telemetry_a['samples'][400])}")
    print(f"  • Branch B (Atomic)  : {repr(telemetry_b['samples'][400])}")

    print("\n--- PERSEVERATION / CYCLING AUDIT ---")
    print(f"  • Branch A Perseveration: Step 200 = {telemetry_a['persev'][0]:.1f}%, Step 400 = {telemetry_a['persev'][1]:.1f}%")
    print(f"  • Branch B Perseveration: Step 200 = {telemetry_b['persev'][0]:.1f}%, Step 400 = {telemetry_b['persev'][1]:.1f}%")
    print("=" * 95)


if __name__ == "__main__":
    run_exp_341_comparative_audit()
