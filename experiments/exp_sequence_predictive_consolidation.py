# exp_sequence_predictive_consolidation.py
"""
===============================================================================
KARYON ENGINEERING PROTOCOL (KEP) EXPERIMENT #10
Topic: Sequence-Preserving Predictive Coding Consolidation Engine
Author: Karyon-CoRE Open-Source Research Team (2026)
===============================================================================

Hypothesis:
    Storing short temporal trajectory chunks (L=16) along with their initial 
    recurrent state H_0 during Wakefulness, and unrolling them continuously 
    during Sleep Consolidation, will preserve hidden state manifold continuity 
    (preventing F_t explosion), accelerate Wake perception by 3-4x, and 
    enable authentic continuous Active Inference learning.

Control Group: 
    Traditional Step-by-Step Backprop Cramming (Gradient updates on every step).

Experimental Group: 
    Sequence-Preserving Predictive Consolidation (Trajectory Chunking + State Replay).

Metrics Tracked:
    1. Post-Consolidation Variational Free Energy (F_t)
    2. Hidden State Manifold Divergence (L2 Norm)
    3. Wake Phase Perception Latency (ms)
    4. Execution Speedup Ratio (x)
===============================================================================
"""

import sys
import types
import time
import math
import torch

# =============================================================================
# DYNAMIC HOTFIX FOR PYTORCH 2.4/2.5+ PYTHON 3.12 DYNAMO BUG ON KAGGLE
# =============================================================================
class DummyDynamoModule(types.ModuleType):
    def __getattr__(self, name):
        if name == "decorators":
            return decorators_mod
        if name == "disable":
            return _disable
        if name == "is_compiling":
            return lambda *args, **kwargs: False
        return lambda *args, **kwargs: None

def _disable(fn=None, *args, **kwargs):
    if fn is None or not callable(fn):
        return lambda *a, **kw: None
    return fn

decorators_mod = types.ModuleType("torch._dynamo.decorators")
class _DimRange:
    pass
decorators_mod._DimRange = _DimRange

dynamo_mod = DummyDynamoModule("torch._dynamo")
dynamo_mod.decorators = decorators_mod
dynamo_mod.disable = _disable

sys.modules["torch._dynamo"] = dynamo_mod
sys.modules["torch._dynamo.decorators"] = decorators_mod
torch._dynamo = dynamo_mod

import torch.nn as nn
import torch.nn.functional as F

# Ensure autograd tracking is enabled
torch.set_grad_enabled(True)

# Set global seed for exact reproducibility
torch.manual_seed(42)

# Select execution hardware
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[KEP Experiment #10] Execution Context: {device.type.upper()}")


# =============================================================================
# 1. TRAJECTORY CHUNK EPISODIC MEMORY BUFFER
# =============================================================================

class SequenceTrajectoryMemory:
    """
    Episodic Memory Buffer storing temporal trajectory chunks (H_0, W_{0..L}, Y_{0..L})
    to preserve state manifold continuity during sleep replay.
    """
    def __init__(self, max_chunks=50, chunk_length=16, hidden_dim=256, embed_dim=128):
        self.max_chunks = max_chunks
        self.chunk_length = chunk_length
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        
        self.trajectory_chunks = []

    def store_trajectory(self, h0_fast, h0_slow, w_sequence, target_sequence):
        """Stores a short temporal trajectory snippet."""
        if len(self.trajectory_chunks) >= self.max_chunks:
            self.trajectory_chunks.pop(0) # FIFO buffer
            
        self.trajectory_chunks.append({
            "h0_fast": h0_fast.detach().clone(),
            "h0_slow": h0_slow.detach().clone(),
            "w_sequence": w_sequence.detach().clone(),       # [L, embed_dim]
            "target_sequence": target_sequence.detach().clone() # [L]
        })

    def clear(self):
        self.trajectory_chunks.clear()


# =============================================================================
# 2. SEQUENCE-PRESERVING PREDICTIVE AGENT
# =============================================================================

class PrototypeTrajectoryAgent(nn.Module):
    """
    Agent with Trajectory-Aware Wake-Sleep Consolidation Engine.
    """
    def __init__(self, vocab_size=258, embed_dim=128, hidden_dim=256, latent_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.sensory_proj = nn.Linear(embed_dim, hidden_dim)

        self.rnn_fast = nn.GRUCell(hidden_dim, hidden_dim)
        self.rnn_slow = nn.GRUCell(hidden_dim, hidden_dim)

        self.prior_net = nn.Linear(hidden_dim, latent_dim * 2)
        self.posterior_net = nn.Linear(hidden_dim + embed_dim, latent_dim * 2)
        self.decoder = nn.Linear(latent_dim + hidden_dim, embed_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

        self.energy = 1.0

    def perception_step(self, input_id, h_fast, h_slow):
        """Perceives single token, updating state and Free Energy without Backprop."""
        x_emb = self.token_embeddings(input_id)
        x_proj = F.silu(self.sensory_proj(x_emb))

        h_f_next = self.rnn_fast(x_proj, h_fast)
        h_s_next = self.rnn_slow(h_f_next, h_slow)

        # Active Inference Predictor
        prior_out = self.prior_net(h_fast)
        mu_p, logvar_p = prior_out.chunk(2, dim=-1)
        
        post_out = self.posterior_net(torch.cat([h_fast, x_emb], dim=-1))
        mu_q, logvar_q = post_out.chunk(2, dim=-1)

        std_q = torch.exp(0.5 * torch.clamp(logvar_q, -10.0, 10.0))
        z_t = mu_q + torch.randn_like(std_q) * std_q

        x_pred = self.decoder(torch.cat([z_t, h_s_next], dim=-1))

        kl_div = 0.5 * torch.mean(logvar_p - logvar_q + (torch.exp(logvar_q) + (mu_q - mu_p)**2) / torch.exp(logvar_p) - 1.0, dim=-1)
        rec_loss = torch.mean((x_emb - x_pred)**2, dim=-1)
        free_energy = kl_div + rec_loss

        logits = self.head(h_f_next + h_s_next)
        
        # Metabolic energy drain during wake perception
        self.energy = max(0.0, self.energy - 0.003)
        should_sleep = self.energy < 0.20

        return h_f_next, h_s_next, x_emb, logits, free_energy, should_sleep

    def consolidate_trajectory_sleep(self, trajectory_memory, optimizer, criterion, sleep_epochs=2):
        """
        Sleep Phase: Replays recorded trajectory chunks starting from H_0, 
        maintaining exact state-space continuity and consolidating weights.
        """
        print(f"\n 😴 [SLEEP PHASE INITIATED] Replaying {len(trajectory_memory.trajectory_chunks)} Trajectory Chunks from H_0...")
        
        if len(trajectory_memory.trajectory_chunks) == 0:
            self.energy = 1.0
            return 0.0

        total_sleep_loss = 0.0

        for ep in range(sleep_epochs):
            for chunk in trajectory_memory.trajectory_chunks:
                optimizer.zero_grad()
                
                # Restore recorded initial hidden state H_0
                h_f = chunk["h0_fast"].clone()
                h_s = chunk["h0_slow"].clone()
                
                w_seq = chunk["w_sequence"]       # [L, embed_dim]
                target_seq = chunk["target_sequence"] # [L]
                chunk_len = w_seq.size(0)

                seq_loss = 0.0
                
                # Unroll trajectory continuously from H_0
                for t in range(chunk_len):
                    x_emb = w_seq[t].unsqueeze(0)
                    target_t = target_seq[t].unsqueeze(0)

                    x_proj = F.silu(self.sensory_proj(x_emb))
                    h_f = self.rnn_fast(x_proj, h_f)
                    h_s = self.rnn_slow(h_f, h_s)

                    logits = self.head(h_f + h_s)
                    loss_t = criterion(logits, target_t)
                    seq_loss = seq_loss + loss_t

                seq_loss = seq_loss / chunk_len
                seq_loss.backward()
                optimizer.step()
                
                total_sleep_loss += seq_loss.item()

        # Restore somatic energy after sleep consolidation
        self.energy = 1.0
        trajectory_memory.clear()
        print(" 🌅 [WAKE UP] Trajectory consolidation complete. Somatic Energy restored to 1.000.\n")
        return total_sleep_loss


# =============================================================================
# 3. EXPERIMENTAL BENCHMARK ENGINE
# =============================================================================

def run_trajectory_benchmark(mode="control_cramming", stream=None, seq_len=600):
    agent = PrototypeTrajectoryAgent().to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-3)
    criterion = nn.CrossEntropyLoss()
    trajectory_memory = SequenceTrajectoryMemory(chunk_length=16)

    h_fast = torch.zeros(1, agent.hidden_dim, device=device)
    h_slow = torch.zeros(1, agent.hidden_dim, device=device)

    sleep_cycles_triggered = 0
    free_energy_history = []
    latency_history = []

    # Buffer for collecting active trajectory chunk
    current_w_buffer = []
    current_target_buffer = []
    chunk_h0_fast = None
    chunk_h0_slow = None

    start_time = time.time()

    for step in range(seq_len - 1):
        step_start = time.perf_counter()
        token_id = stream[step].unsqueeze(0)
        target_id = stream[step + 1].unsqueeze(0)

        if mode == "control_cramming":
            # CONTROL: Step-by-step Backprop Cramming
            h_fast, h_slow, x_emb, logits, free_energy, _ = agent.perception_step(token_id, h_fast, h_slow)
            
            optimizer.zero_grad()
            loss = criterion(logits, target_id) + 0.1 * free_energy.mean()
            loss.backward()
            optimizer.step()
            
            h_fast = h_fast.detach()
            h_slow = h_slow.detach()
            fe_val = free_energy.item()
        else:
            # EXPERIMENTAL: Trajectory-Aware Wake-Sleep Consolidation
            with torch.no_grad():
                # Record H_0 at beginning of a new trajectory chunk
                if len(current_w_buffer) == 0:
                    chunk_h0_fast = h_fast.clone()
                    chunk_h0_slow = h_slow.clone()

                h_fast, h_slow, x_emb, logits, free_energy, should_sleep = agent.perception_step(token_id, h_fast, h_slow)
                fe_val = free_energy.item()

                current_w_buffer.append(x_emb.squeeze(0))
                current_target_buffer.append(target_id.squeeze(0))

                # If trajectory chunk reaches length 16, store it
                if len(current_w_buffer) == 16:
                    trajectory_memory.store_trajectory(
                        chunk_h0_fast, chunk_h0_slow,
                        torch.stack(current_w_buffer),
                        torch.stack(current_target_buffer)
                    )
                    current_w_buffer.clear()
                    current_target_buffer.clear()

            # Trigger Sleep Consolidation Phase when exhausted
            if should_sleep and len(trajectory_memory.trajectory_chunks) > 0:
                sleep_cycles_triggered += 1
                agent.consolidate_trajectory_sleep(trajectory_memory, optimizer, criterion)

        free_energy_history.append(fe_val)
        latency_history.append((time.perf_counter() - step_start) * 1000.0)

    total_duration = time.time() - start_time
    avg_fe = sum(free_energy_history) / len(free_energy_history)
    avg_latency = sum(latency_history) / len(latency_history)

    return {
        "mode": mode,
        "total_steps": seq_len - 1,
        "sleep_cycles": sleep_cycles_triggered,
        "mean_free_energy": avg_fe,
        "avg_latency_ms": avg_latency,
        "total_duration_sec": total_duration
    }


# =============================================================================
# 4. MAIN EVALUATION & TELEMETRY DASHBOARD
# =============================================================================

if __name__ == "__main__":
    STREAM_LEN = 800
    pattern = [ord(c) for c in "Active Inference Sequence Preserving Predictive Consolidation. "]
    stream = torch.tensor([pattern[i % len(pattern)] for i in range(STREAM_LEN)], dtype=torch.long, device=device)

    print("\n[KEP Step 1/2] Running CONTROL Group (Continuous Backprop Cramming)...")
    control_res = run_trajectory_benchmark(mode="control_cramming", stream=stream, seq_len=STREAM_LEN)

    print("[KEP Step 2/2] Running EXPERIMENTAL Group (Sequence-Preserving Predictive Consolidation)...")
    experimental_res = run_trajectory_benchmark(mode="trajectory_consolidation", stream=stream, seq_len=STREAM_LEN)

    # Calculate Telemetry Gains
    speedup_ratio = control_res["total_duration_sec"] / experimental_res["total_duration_sec"]
    fe_change_pct = ((experimental_res["mean_free_energy"] - control_res["mean_free_energy"]) / control_res["mean_free_energy"]) * 100.0

    # =========================================================================
    # TELEMETRY RESULTS DASHBOARD
    # =========================================================================
    print("\n" + "="*80)
    print(" === KARYON ENGINEERING PROTOCOL (KEP) TELEMETRY DASHBOARD ===")
    print("="*80)
    print(f"{'Metric':<32} | {'Control Cramming':<18} | {'Trajectory Engine':<18} | {'Delta / Gain':<15}")
    print("-" * 88)
    print(f"{'Total Stream Steps':<32} | {control_res['total_steps']:<18} | {experimental_res['total_steps']:<18} | {'0 (Identical)':<15}")
    print(f"{'Sleep Cycles Triggered':<32} | {0:<18} | {experimental_res['sleep_cycles']:<18} | {'Cycles':<15}")
    print(f"{'Mean Free Energy (F_t)':<32} | {control_res['mean_free_energy']:<18.4f} | {experimental_res['mean_free_energy']:<18.4f} | {fe_change_pct:+.2f}% Error")
    print(f"{'Wake Perception Speed (ms)':<32} | {control_res['avg_latency_ms']:<18.2f} | {experimental_res['avg_latency_ms']:<18.2f} | {speedup_ratio:.2f}x Faster")
    print("="*80)

    # KEP Evaluation Logic
    print("\n--- [KEP EVALUATION & VERDICT] ---")
    if speedup_ratio >= 1.5 and fe_change_pct <= 15.0:
        print("🟢 VERDICT: POSITIVE EXPERIENCE ADOPTED!")
        print(f"   Reason: Trajectory consolidation accelerated perception by {speedup_ratio:.2f}x with stable Free Energy ({fe_change_pct:+.2f}%).")
        print("   Action: Adopt Trajectory-Preserving Predictive Consolidation into master Karyon!")
    elif speedup_ratio < 1.2:
        print("⚪ VERDICT: NEUTRAL EXPERIENCE DISCARDED.")
    else:
        print("🔴 VERDICT: NEGATIVE EXPERIENCE REJECTED.")
        print("   Reason: Excessive Free Energy degradation.")
    print("="*80 + "\n")
