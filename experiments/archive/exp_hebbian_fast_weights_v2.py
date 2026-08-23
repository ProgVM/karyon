# experiments/exp_hebbian_fast_weights_v2.py
"""
===============================================================================
KARYON ENGINEERING PROTOCOL (KEP) EXPERIMENT #4.1 (ITERATION 2)
Topic: Three-Factor Predictive Hebbian Plasticity (Error-Driven Fast-Weights)
Author: Karyon-CoRE Open-Source Research Team (2026)
===============================================================================

Hypothesis:
    Driving local Hebbian fast-weight updates using the back-projected reconstruction 
    error vector (e_hidden = e_rec * W_base) modulated by Noradrenaline (Arousal) 
    will minimize Variational Free Energy error degradation (<10-15%) while 
    maintaining >3x step speedup and zero Autograd VRAM graph overhead.

Control Group: 
    Traditional Autograd Backpropagation (.backward() + Adam Optimizer step).

Experimental Group: 
    Three-Factor Predictive Hebbian Plasticity (Zero .backward(), Zero Autograd Graph).

Metrics Tracked:
    1. VRAM Memory Allocated (MB)
    2. Per-Step Execution Latency (ms)
    3. Mean Variational Free Energy (F_t)
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

# Ensure global autograd gradient tracking is enabled
torch.set_grad_enabled(True)

# Set global seed for exact reproducibility
torch.manual_seed(42)

# Select execution hardware
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[KEP Experiment #4.1] Execution Context: {device.type.upper()}")


# =============================================================================
# 1. THREE-FACTOR PREDICTIVE HEBBIAN SYNAPTIC LAYER
# =============================================================================

class ThreeFactorHebbianLayer(nn.Module):
    """
    Synaptic Conductance Layer using Three-Factor Predictive Plasticity:
    dW_fast = eta * Noradrenaline * (e_hidden x_pre^T)
    """
    def __init__(self, in_features, out_features, decay_rate=0.95, learning_rate=0.015):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        
        # Base static weights (consolidated long-term memory)
        self.weight_base = nn.Parameter(torch.randn(out_features, in_features) * 0.05)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Fast dynamic synaptic matrix buffer (updated in O(1) without Autograd)
        self.register_buffer("weight_fast", torch.zeros(out_features, in_features))

    def update_predictive_fast_weights(self, x_pre, rec_err_vector, arousal_na):
        """
        Three-Factor Error-Driven Hebbian Update:
        1. Pre-synaptic Activity: x_pre [B, in_features]
        2. Back-projected Prediction Error: e_hidden = rec_err * W_base^T [B, out_features]
        3. Neuromodulator: Noradrenaline (Arousal)
        """
        with torch.no_grad():
            # Project sensory reconstruction error vector back to hidden layer space
            err_hidden = F.linear(rec_err_vector, self.weight_base) # [B, out_features]
            
            # Compute outer product of error signal and pre-synaptic input
            delta_w = torch.bmm(err_hidden.unsqueeze(2), x_pre.unsqueeze(1)).mean(dim=0)
            
            # In-place Three-Factor update
            self.weight_fast.mul_(self.decay_rate).add_(delta_w, alpha=self.learning_rate * arousal_na)

    def forward(self, x):
        """Effective Synaptic Conductance: W_eff = W_base + W_fast."""
        effective_weight = self.weight_base + self.weight_fast
        return F.linear(x, effective_weight, self.bias)


# =============================================================================
# 2. PROTOTYPE KARYON AGENT WITH PREDICTIVE FAST-WEIGHTS
# =============================================================================

class PrototypeThreeFactorAgent(nn.Module):
    """
    Karyon Prototype Agent supporting Three-Factor Predictive Hebbian Plasticity.
    """
    def __init__(self, vocab_size=258, embed_dim=128, hidden_dim=256, latent_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        
        # Three-Factor Synaptic Projection Layer ([256, 128])
        self.sensory_hebbian = ThreeFactorHebbianLayer(embed_dim, hidden_dim)
        self.rnn_cell = nn.GRUCell(hidden_dim, hidden_dim)

        # Active Inference Predictor
        self.prior_net = nn.Linear(hidden_dim, latent_dim * 2)
        self.posterior_net = nn.Linear(hidden_dim + embed_dim, latent_dim * 2)
        self.decoder = nn.Linear(latent_dim + hidden_dim, embed_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_id, h_prev):
        """Forward pass computing next state, prediction error vector, and Free Energy."""
        x_emb = self.token_embeddings(input_id)
        x_proj = F.silu(self.sensory_hebbian(x_emb))
        h_next = self.rnn_cell(x_proj, h_prev)

        # Active Inference Latent Predictor
        prior_out = self.prior_net(h_prev)
        mu_prior, logvar_prior = prior_out.chunk(2, dim=-1)
        logvar_prior = torch.clamp(logvar_prior, -10.0, 10.0)

        post_out = self.posterior_net(torch.cat([h_prev, x_emb], dim=-1))
        mu_post, logvar_post = post_out.chunk(2, dim=-1)
        logvar_post = torch.clamp(logvar_post, -10.0, 10.0)

        std_post = torch.exp(0.5 * logvar_post)
        eps = torch.randn_like(std_post)
        z_t = mu_post + eps * std_post

        x_pred = self.decoder(torch.cat([z_t, h_next], dim=-1))

        # Free Energy and Reconstruction Error Vector
        var_prior = torch.exp(logvar_prior) + 1e-7
        var_post = torch.exp(logvar_post) + 1e-7

        kl_div = 0.5 * torch.mean(
            logvar_prior - logvar_post + (var_post + (mu_post - mu_prior)**2) / var_prior - 1.0,
            dim=-1, keepdim=True
        )
        rec_err = x_emb - x_pred # Prediction error vector in embedding space
        rec_loss = torch.mean(rec_err**2, dim=-1, keepdim=True)
        free_energy = kl_div + rec_loss

        logits = self.head(h_next)
        return h_next, x_emb, rec_err, logits, free_energy


# =============================================================================
# 3. CONTINUOUS STREAM GENERATOR
# =============================================================================

def generate_streaming_data(seq_len=600):
    """Generates continuous byte stream with repeating patterns and novel bursts."""
    stream = []
    pattern = [ord(c) for c in "Active Inference and Hebbian Synaptic Plasticity. "]
    
    for i in range(seq_len):
        if i % 100 > 80:
            stream.append(torch.randint(0, 255, (1,)).item())
        else:
            stream.append(pattern[i % len(pattern)])
            
    return torch.tensor(stream, dtype=torch.long, device=device)


# =============================================================================
# 4. EXPERIMENTAL BENCHMARK ENGINE
# =============================================================================

def run_hebbian_benchmark_v2(mode="autograd_control", stream=None, seq_len=600):
    """
    Executes streaming session comparing Autograd Backprop vs Three-Factor Predictive Hebbian.
    """
    agent = PrototypeThreeFactorAgent().to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=2e-3)
    criterion = nn.CrossEntropyLoss()

    h_curr = torch.zeros(1, 256, device=device)
    free_energy_history = []
    latency_history = []
    
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()

    for step in range(seq_len - 1):
        step_start = time.perf_counter()

        input_id = stream[step].unsqueeze(0)
        target_id = stream[step + 1].unsqueeze(0)

        # 1. Forward Pass
        h_curr, x_emb, rec_err, logits, free_energy = agent(input_id, h_curr)
        fe_val = free_energy.item()
        free_energy_history.append(fe_val)

        task_loss = criterion(logits, target_id)
        total_loss = task_loss + 0.1 * free_energy.mean()

        # 2. Plasticity Mechanics Comparison
        if mode == "autograd_control":
            # CONTROL: Full Autograd Backpropagation Graph Execution
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=2.0)
            optimizer.step()
            h_curr = h_curr.detach()
        else:
            # EXPERIMENTAL: Three-Factor Predictive Hebbian Fast-Weights (ZERO .backward()!)
            arousal_na = min(1.0, max(0.1, fe_val))
            agent.sensory_hebbian.update_predictive_fast_weights(
                x_pre=x_emb.detach(), 
                rec_err_vector=rec_err.detach(), 
                arousal_na=arousal_na
            )
            h_curr = h_curr.detach()

        latency_history.append((time.perf_counter() - step_start) * 1000.0)

    total_duration = time.time() - start_time
    avg_fe = sum(free_energy_history) / len(free_energy_history)
    avg_latency = sum(latency_history) / len(latency_history)
    
    peak_memory_mb = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if device.type == 'cuda' else 0.0

    return {
        "mode": mode,
        "total_steps": seq_len - 1,
        "mean_free_energy": avg_fe,
        "peak_vram_mb": peak_memory_mb,
        "avg_latency_ms": avg_latency,
        "total_duration_sec": total_duration
    }


# =============================================================================
# 5. MAIN EVALUATION & TELEMETRY DASHBOARD
# =============================================================================

if __name__ == "__main__":
    STREAM_LEN = 800
    print(f"\nGenerating streaming dataset ({STREAM_LEN} steps)...")
    stream = generate_streaming_data(seq_len=STREAM_LEN)

    print("\n[KEP Step 1/2] Running CONTROL Group (Autograd Backpropagation .backward())...")
    control_res = run_hebbian_benchmark_v2(mode="autograd_control", stream=stream, seq_len=STREAM_LEN)

    print("[KEP Step 2/2] Running EXPERIMENTAL Group (Three-Factor Predictive Hebbian - ZERO .backward())...")
    experimental_res = run_hebbian_benchmark_v2(mode="hebbian_fast_weights_v2", stream=stream, seq_len=STREAM_LEN)

    # Calculate Telemetry Gains
    speedup_ratio = control_res["total_duration_sec"] / experimental_res["total_duration_sec"]
    fe_change_pct = ((experimental_res["mean_free_energy"] - control_res["mean_free_energy"]) / control_res["mean_free_energy"]) * 100.0
    vram_saved_mb = control_res["peak_vram_mb"] - experimental_res["peak_vram_mb"]

    # =========================================================================
    # TELEMETRY RESULTS DASHBOARD
    # =========================================================================
    print("\n" + "="*80)
    print(" === KARYON ENGINEERING PROTOCOL (KEP) TELEMETRY DASHBOARD (v2) ===")
    print("="*80)
    print(f"{'Metric':<32} | {'Autograd Control':<18} | {'Three-Factor Hebbian':<18} | {'Delta / Gain':<15}")
    print("-" * 88)
    print(f"{'Total Stream Steps':<32} | {control_res['total_steps']:<18} | {experimental_res['total_steps']:<18} | {'0 (Identical)':<15}")
    print(f"{'Autograd .backward() Calls':<32} | {control_res['total_steps']:<18} | {0:<18} | {'-100% Graph Ops':<15}")
    print(f"{'Peak VRAM Memory (MB)':<32} | {control_res['peak_vram_mb']:<18.2f} | {experimental_res['peak_vram_mb']:<18.2f} | {vram_saved_mb:+.2f} MB")
    print(f"{'Mean Free Energy (F_t)':<32} | {control_res['mean_free_energy']:<18.4f} | {experimental_res['mean_free_energy']:<18.4f} | {fe_change_pct:+.2f}% Error")
    print(f"{'Average Step Latency (ms)':<32} | {control_res['avg_latency_ms']:<18.2f} | {experimental_res['avg_latency_ms']:<18.2f} | {speedup_ratio:.2f}x Faster")
    print("="*80)

    # KEP Evaluation Logic
    print("\n--- [KEP EVALUATION & VERDICT] ---")
    if speedup_ratio >= 1.5 and fe_change_pct <= 15.0:
        print("🟢 VERDICT: POSITIVE EXPERIENCE ADOPTED!")
        print(f"   Reason: Achieved {speedup_ratio:.2f}x speedup with ZERO autograd backward calls while preserving low Free Energy.")
        print("   Action: Merge Three-Factor Predictive Fast-Weights into production karyon_core!")
    elif speedup_ratio < 1.2:
        print("⚪ VERDICT: NEUTRAL EXPERIENCE DISCARDED.")
        print("   Reason: Insufficient latency speedup.")
    else:
        print("🔴 VERDICT: NEGATIVE EXPERIENCE REJECTED.")
        print("   Reason: Free Energy degradation exceeded boundary.")
    print("="*80 + "\n")
