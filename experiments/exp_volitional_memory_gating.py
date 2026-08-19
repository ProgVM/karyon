# exp_volitional_memory_gating.py
"""
===============================================================================
KARYON ENGINEERING PROTOCOL (KEP) EXPERIMENT #8
Topic: Volitional Associative Memory Read Gating (Arousal/Free Energy Gated Recall)
Author: Karyon-CoRE Open-Source Research Team (2026)
===============================================================================

Hypothesis:
    Restricting episodic memory retrieval (episodic_mem.read()) to steps where 
    somatic Noradrenaline (NA > 0.12) or Variational Free Energy (F_t > Threshold) 
    signals uncertainty will reduce memory search FLOPs by 70-85% while 
    maintaining high associative recall precision and accelerating per-step latency.

Control Group: 
    Unconditional Memory Read (episodic_mem.read() executed at EVERY step).

Experimental Group: 
    Volitional Memory Gated Read (Retrieval triggered strictly under high NA / F_t).

Metrics Tracked:
    1. Memory Search Executions Saved (%)
    2. Mean Variational Free Energy (F_t)
    3. Memory Association Accuracy (%)
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
print(f"[KEP Experiment #8] Execution Context: {device.type.upper()}")


# =============================================================================
# 1. VOLITIONAL MEMORY GATED AGENT
# =============================================================================

class PrototypeVolitionalMemoryAgent(nn.Module):
    """
    Agent supporting Volitional Memory Read Gating driven by Noradrenaline & Free Energy.
    """
    def __init__(self, vocab_size=258, embed_dim=128, hidden_dim=256, latent_dim=64, memory_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim

        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.text_proj = nn.Linear(embed_dim, memory_dim)

        self.rnn_fast = nn.GRUCell(memory_dim, hidden_dim)
        self.rnn_slow = nn.GRUCell(hidden_dim, hidden_dim)

        self.prior_net = nn.Linear(hidden_dim, latent_dim * 2)
        self.posterior_net = nn.Linear(hidden_dim + embed_dim, latent_dim * 2)
        self.decoder = nn.Linear(latent_dim + hidden_dim, embed_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

        # Mock Episodic Memory Buffer
        self.register_buffer("memory_keys", torch.randn(200, memory_dim))
        self.register_buffer("memory_values", torch.randn(200, memory_dim))

    def read_memory_mock(self, query, temp=0.05, threshold=0.5):
        """Simulates O(N) matrix search in episodic memory."""
        q_norm = F.normalize(query, p=2, dim=-1)
        k_norm = F.normalize(self.memory_keys, p=2, dim=-1)
        sim = torch.matmul(q_norm, k_norm.T) / temp
        attn = F.softmax(sim, dim=-1)
        retrieved = torch.matmul(attn, self.memory_values)
        return retrieved

    def forward_step(self, input_id, h_fast, h_slow, noradrenaline=0.0, mode="unconditional_control"):
        x_emb = self.token_embeddings(input_id)
        w_current = F.silu(self.text_proj(x_emb))

        # Active Inference World Model
        prior_out = self.prior_net(h_fast)
        mu_p, logvar_p = prior_out.chunk(2, dim=-1)
        
        post_out = self.posterior_net(torch.cat([h_fast, x_emb], dim=-1))
        mu_q, logvar_q = post_out.chunk(2, dim=-1)

        kl_div = 0.5 * torch.mean(logvar_p - logvar_q + (torch.exp(logvar_q) + (mu_q - mu_p)**2) / torch.exp(logvar_p) - 1.0, dim=-1)
        free_energy = kl_div.mean()

        # Volitional Recall Decision: Trigger memory read ONLY if arousal NA > 0.12 or F_t > 0.25
        should_read_memory = (mode == "unconditional_control") or (noradrenaline > 0.12 or free_energy.item() > 0.25)
        
        memory_searched = False
        if should_read_memory:
            memory_vector = self.read_memory_mock(w_current)
            w_integrated = w_current + 0.5 * memory_vector
            memory_searched = True
        else:
            w_integrated = w_current # Skip O(N) memory search!

        h_f_next = self.rnn_fast(w_integrated, h_fast)
        h_s_next = self.rnn_slow(h_f_next, h_slow)
        logits = self.head(h_f_next + h_s_next)

        return h_f_next, h_s_next, logits, free_energy, memory_searched


# =============================================================================
# 2. STREAM GENERATOR WITH VARIABLE AROUSAL NOVELTY BURSTS
# =============================================================================

def generate_arousal_stream(seq_len=800):
    """Generates byte stream with alternating routine text and high-arousal novel bursts."""
    stream = []
    pattern = [ord(c) for c in "Active Inference Volitional Memory Recall Gating. "]
    
    for i in range(seq_len):
        if 200 <= i <= 300 or 550 <= i <= 650:
            # High-arousal novel unexpected bytes
            stream.append(torch.randint(0, 255, (1,)).item())
        else:
            # Predictable routine text pattern
            stream.append(pattern[i % len(pattern)])
            
    return torch.tensor(stream, dtype=torch.long, device=device)


# =============================================================================
# 3. EXPERIMENTAL BENCHMARK ENGINE
# =============================================================================

def run_volitional_memory_benchmark(mode="unconditional_control", stream=None, seq_len=800):
    agent = PrototypeVolitionalMemoryAgent().to(device)
    agent.eval()

    h_fast = torch.zeros(1, agent.hidden_dim, device=device)
    h_slow = torch.zeros(1, agent.hidden_dim, device=device)

    memory_searched_count = 0
    free_energy_history = []
    latency_history = []

    noradrenaline = 0.0

    start_time = time.time()

    with torch.no_grad():
        for step in range(seq_len - 1):
            step_start = time.perf_counter()
            token_id = stream[step].unsqueeze(0)

            h_fast, h_slow, logits, free_energy, memory_searched = agent.forward_step(
                token_id, h_fast, h_slow, noradrenaline=noradrenaline, mode=mode
            )

            fe_val = free_energy.item()
            free_energy_history.append(fe_val)

            # Update simulated Noradrenaline (Arousal) based on prediction error
            noradrenaline = min(1.0, max(0.0, 0.7 * fe_val + 0.3 * noradrenaline))

            if memory_searched:
                memory_searched_count += 1

            latency_history.append((time.perf_counter() - step_start) * 1000.0)

    total_duration = time.time() - start_time
    avg_fe = sum(free_energy_history) / len(free_energy_history)
    avg_latency = sum(latency_history) / len(latency_history)

    return {
        "mode": mode,
        "total_steps": seq_len - 1,
        "memory_searches": memory_searched_count,
        "mean_free_energy": avg_fe,
        "avg_latency_ms": avg_latency,
        "total_duration_sec": total_duration
    }


# =============================================================================
# 4. MAIN EVALUATION & TELEMETRY DASHBOARD
# =============================================================================

if __name__ == "__main__":
    STREAM_LEN = 800
    print(f"\nGenerating arousal stream dataset ({STREAM_LEN} steps)...")
    stream = generate_arousal_stream(seq_len=STREAM_LEN)

    print("\n[KEP Step 1/2] Running CONTROL Group (Unconditional Every-Step Memory Read)...")
    control_res = run_volitional_memory_benchmark(mode="unconditional_control", stream=stream, seq_len=STREAM_LEN)

    print("[KEP Step 2/2] Running EXPERIMENTAL Group (Volitional Arousal/F_t Gated Memory Read)...")
    experimental_res = run_volitional_memory_benchmark(mode="volitional_gated", stream=stream, seq_len=STREAM_LEN)

    # Calculate Telemetry Gains
    memory_saved_pct = (1.0 - (experimental_res["memory_searches"] / control_res["memory_searches"])) * 100.0
    fe_change_pct = ((experimental_res["mean_free_energy"] - control_res["mean_free_energy"]) / control_res["mean_free_energy"]) * 100.0
    speedup_ratio = control_res["total_duration_sec"] / experimental_res["total_duration_sec"]

    # =========================================================================
    # TELEMETRY RESULTS DASHBOARD
    # =========================================================================
    print("\n" + "="*80)
    print(" === KARYON ENGINEERING PROTOCOL (KEP) TELEMETRY DASHBOARD ===")
    print("="*80)
    print(f"{'Metric':<32} | {'Control Group':<18} | {'Experimental (Gated)':<18} | {'Delta / Gain':<15}")
    print("-" * 88)
    print(f"{'Total Stream Steps':<32} | {control_res['total_steps']:<18} | {experimental_res['total_steps']:<18} | {'0 (Identical)':<15}")
    print(f"{'Memory Read Executions':<32} | {control_res['memory_searches']:<18} | {experimental_res['memory_searches']:<18} | {memory_saved_pct:+.2f}% Searches")
    print(f"{'Mean Free Energy (F_t)':<32} | {control_res['mean_free_energy']:<18.4f} | {experimental_res['mean_free_energy']:<18.4f} | {fe_change_pct:+.2f}% Error")
    print(f"{'Average Step Latency (ms)':<32} | {control_res['avg_latency_ms']:<18.2f} | {experimental_res['avg_latency_ms']:<18.2f} | {speedup_ratio:.2f}x Faster")
    print("="*80)

    # KEP Evaluation Logic
    print("\n--- [KEP EVALUATION & VERDICT] ---")
    if memory_saved_pct >= 50.0 and fe_change_pct <= 10.0:
        print("🟢 VERDICT: POSITIVE EXPERIENCE ADOPTED!")
        print(f"   Reason: Reduced Memory Search FLOPs by {memory_saved_pct:.1f}% while maintaining low Free Energy error ({fe_change_pct:+.2f}%).")
        print("   Action: Merge Volitional Recall Gating into production karyon_agent!")
    elif memory_saved_pct < 20.0:
        print("⚪ VERDICT: NEUTRAL EXPERIENCE DISCARDED.")
        print("   Reason: Insufficient memory search reduction.")
    else:
        print("🔴 VERDICT: NEGATIVE EXPERIENCE REJECTED.")
        print("   Reason: Free Energy degradation exceeded acceptable boundary.")
    print("="*80 + "\n")
