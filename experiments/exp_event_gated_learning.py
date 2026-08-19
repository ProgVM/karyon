# exp_event_gated_learning.py
"""
===============================================================================
KARYON ENGINEERING PROTOCOL (KEP) EXPERIMENT #2
Topic: Event-Driven Active Plasticity via Dynamic Free Energy Thresholding (DFET)
Author: Karyon-CoRE Open-Source Research Team (2026)
===============================================================================

Hypothesis:
    By gating gradient backpropagation updates based on Variational Free Energy 
    (F_t > Threshold_t) modulated by Somatic Noradrenaline (Arousal), we can 
    reduce total Backprop FLOPs by 60-80% during online streaming (N=1) while 
    maintaining or improving predictive accuracy and conserving somatic energy.

Control Group: 
    Traditional Continuous Online Learning (Backprop triggered at EVERY step).

Experimental Group: 
    Event-Gated Plasticity (Backprop triggered ONLY during high surprise / novelty).

Metrics Tracked:
    1. Mean Variational Free Energy (F_t)
    2. Total Gradient Backprop Steps (FLOPs Proxy)
    3. FLOPs Reduction Ratio (%)
    4. Final Somatic Energy Level (Homeostatic State)
    5. Execution Time / Latency (ms/step)
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
    """Dynamic interceptor providing safe no-op fallbacks for any torch._dynamo attribute."""
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

# Ensure global autograd gradient tracking is explicitly enabled
torch.set_grad_enabled(True)

# Quick sanity check for optimizer and step() execution
_test_param = nn.Parameter(torch.randn(2, 2))
_test_opt = torch.optim.Adam([_test_param], lr=1e-3)
_test_param.grad = torch.randn(2, 2)
_test_opt.step()
print("[KEP Optimizer Check] PyTorch Adam & optimizer.step() executed successfully!")

# Set global seed for reproducibility across Kaggle runs
torch.manual_seed(42)

# Select execution hardware
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[KEP Experiment #2] Execution Context: {device.type.upper()}")


# =============================================================================
# 1. STANDALONE PROTOTYPE MODULES (Matching Production Karyon Architecture)
# =============================================================================

class SyntheticSomaticUnit:
    """
    Simulates Karyon Homeostatic Somatic Variables:
    State: [Curiosity, Energy, Stability, Health, Noradrenaline (NA), Dopamine (DA)]
    """
    def __init__(self, device='cpu'):
        self.device = device
        # Initial physiological setpoints
        self.state = torch.tensor([[0.5, 1.0, 1.0, 1.0, 0.0, 0.0]], dtype=torch.float32, device=device)

    def update(self, free_energy: float, was_adapted: bool):
        """Updates internal physiology based on surprise and action costs."""
        curiosity, energy, stability, health, na, da = self.state[0].tolist()

        # Energy consumption: Backpropagation costs 5x more energy than state inference
        energy_cost = 0.008 if was_adapted else 0.0015
        energy = max(0.0, energy - energy_cost)

        # Noradrenaline (Arousal) rises with Free Energy (Surprise)
        na = min(1.0, max(0.0, 0.7 * free_energy + 0.3 * na))

        # Dopamine (Reward) signals successful adaptation
        da = min(1.0, max(0.0, 0.5 * da + (0.2 if (was_adapted and free_energy < 0.2) else 0.0)))

        # Update state tensor
        self.state = torch.tensor([[curiosity, energy, stability, health, na, da]], dtype=torch.float32, device=self.device)
        return self.state


class PrototypeKaryonAgent(nn.Module):
    """
    Lightweight, self-contained implementation of Karyon Active Inference Core
    combining Sensory Projection, Latent World Model, and Motor/Text Generation.
    """
    def __init__(self, embed_dim=128, hidden_dim=256, latent_dim=64, vocab_size=258):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size

        # Embeddings & Sensory Gate
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.sensory_proj = nn.Linear(embed_dim, hidden_dim)

        # Recurrent Core
        self.rnn_cell = nn.GRUCell(hidden_dim, hidden_dim)

        # Active Inference Latent Predictor (Prior & Posterior)
        self.prior_net = nn.Linear(hidden_dim, latent_dim * 2)
        self.posterior_net = nn.Linear(hidden_dim + embed_dim, latent_dim * 2)
        self.decoder = nn.Linear(latent_dim + hidden_dim, embed_dim)

        # Output Head
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_id, h_prev):
        """
        Single step forward pass computing next state, predicted next embedding,
        output logits, and Variational Free Energy (F_t).
        """
        # 1. Embed input token
        x_emb = self.token_embeddings(input_id) # [B, embed_dim]
        x_proj = F.silu(self.sensory_proj(x_emb))

        # 2. Update recurrent hidden state
        h_next = self.rnn_cell(x_proj, h_prev)

        # 3. Latent World Model (Prior & Posterior Distributions)
        prior_params = self.prior_net(h_prev)
        mu_prior, logvar_prior = prior_params.chunk(2, dim=-1)
        logvar_prior = torch.clamp(logvar_prior, -10.0, 10.0)

        post_params = self.posterior_net(torch.cat([h_prev, x_emb], dim=-1))
        mu_post, logvar_post = post_params.chunk(2, dim=-1)
        logvar_post = torch.clamp(logvar_post, -10.0, 10.0)

        # Sample latent variable z_t via reparameterization trick
        std_post = torch.exp(0.5 * logvar_post)
        eps = torch.randn_like(std_post)
        z_t = mu_post + eps * std_post

        # Predict input embedding representation
        x_pred = self.decoder(torch.cat([z_t, h_next], dim=-1))

        # 4. Calculate Variational Free Energy (F_t = KL Divergence + Reconstruction Error)
        var_prior = torch.exp(logvar_prior) + 1e-7
        var_post = torch.exp(logvar_post) + 1e-7

        kl_div = 0.5 * torch.mean(
            logvar_prior - logvar_post + (var_post + (mu_post - mu_prior)**2) / var_prior - 1.0,
            dim=-1, keepdim=True
        )
        rec_loss = torch.mean((x_emb - x_pred)**2, dim=-1, keepdim=True)
        free_energy = kl_div + rec_loss

        # 5. Compute output token logits
        logits = self.head(h_next)

        return h_next, logits, free_energy, rec_loss


# =============================================================================
# 2. DATASET GENERATOR (Continuous Stream Simulation)
# =============================================================================

def generate_streaming_data(seq_len=600, vocab_size=258):
    """
    Generates a synthetic continuous byte stream containing predictable patterns 
    mixed with periodic unexpected novel bursts (simulating real-world streams).
    """
    stream = []
    pattern = [ord(c) for c in "Active Inference minimizes Free Energy in real-time. "]
    
    for i in range(seq_len):
        if i % 120 > 100:
            # Introduce high-entropy unexpected novel byte burst
            stream.append(torch.randint(0, 255, (1,)).item())
        else:
            # Predictable repeating byte structure
            stream.append(pattern[i % len(pattern)])
            
    return torch.tensor(stream, dtype=torch.long, device=device)


# =============================================================================
# 3. EXPERIMENTAL BENCHMARK ENGINE
# =============================================================================

def run_experiment_session(mode="control", data_stream=None, seq_len=600):
    """
    Runs a complete single-pass (N=1) continuous streaming session.
    """
    # Explicitly enable gradient computation for training pass
    torch.set_grad_enabled(True)

    agent = PrototypeKaryonAgent().to(device)
    agent.train()
    
    optimizer = torch.optim.Adam(agent.parameters(), lr=2e-3)
    somatic_unit = SyntheticSomaticUnit(device=device)
    criterion = nn.CrossEntropyLoss()

    h_curr = torch.zeros(1, 256, device=device)
    
    # Telemetry tracking variables
    total_backprop_steps = 0
    free_energy_history = []
    latency_history = []
    
    moving_avg_fe = 0.15 # Baseline moving average initialization
    alpha_ma = 0.05       # Moving average smoothing factor

    start_time = time.time()

    for step in range(seq_len - 1):
        step_start = time.perf_counter()

        input_id = data_stream[step].unsqueeze(0)
        target_id = data_stream[step + 1].unsqueeze(0)

        # 1. Forward Pass (Inference)
        h_curr, logits, free_energy, rec_loss = agent(input_id, h_curr)
        fe_val = free_energy.item()
        free_energy_history.append(fe_val)

        # Compute cross-entropy task loss
        task_loss = criterion(logits, target_id)
        total_step_loss = task_loss + 0.1 * free_energy.mean()

        # Update moving average of Free Energy
        moving_avg_fe = (1.0 - alpha_ma) * moving_avg_fe + alpha_ma * fe_val

        # Get current Noradrenaline (Arousal) level
        na_level = somatic_unit.state[0, 4].item()

        # 2. Plasticity Gating Decision
        was_adapted = False

        if mode == "control":
            # CONTROL: Always trigger Backpropagation
            should_backprop = True
        else:
            # EXPERIMENTAL (DFET): Trigger Backprop ONLY when Free Energy exceeds dynamic threshold
            dynamic_threshold = max(0.02, moving_avg_fe * (1.20 - 0.40 * na_level))
            should_backprop = fe_val > dynamic_threshold

        # 3. Execution of Gradient Backpropagation
        if should_backprop:
            optimizer.zero_grad()
            total_step_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=2.0)
            optimizer.step()
            
            # Detach recurrent state to prevent gradient accumulation across infinite time
            h_curr = h_curr.detach()
            total_backprop_steps += 1
            was_adapted = True
        else:
            # Detach hidden state without backpropagation (0 Backprop FLOPs!)
            h_curr = h_curr.detach()

        # 4. Somatic Physiology Update
        somatic_unit.update(fe_val, was_adapted)

        step_elapsed_ms = (time.perf_counter() - step_start) * 1000.0
        latency_history.append(step_elapsed_ms)

    total_duration = time.time() - start_time
    avg_fe = sum(free_energy_history) / len(free_energy_history)
    avg_latency = sum(latency_history) / len(latency_history)
    final_energy = somatic_unit.state[0, 1].item()

    return {
        "mode": mode,
        "total_steps": seq_len - 1,
        "backprop_steps": total_backprop_steps,
        "mean_free_energy": avg_fe,
        "final_energy": final_energy,
        "avg_latency_ms": avg_latency,
        "total_duration_sec": total_duration
    }


# =============================================================================
# 4. MAIN EXPERIMENTAL COMPARISON & TELEMETRY DASHBOARD
# =============================================================================

if __name__ == "__main__":
    STREAM_LENGTH = 800
    print(f"\nGenerating continuous streaming dataset ({STREAM_LENGTH} steps)...")
    stream_data = generate_streaming_data(seq_len=STREAM_LENGTH)

    print("\n[KEP Step 1/2] Running CONTROL Group (Continuous Unconditional Backprop)...")
    control_results = run_experiment_session(mode="control", data_stream=stream_data, seq_len=STREAM_LENGTH)

    print("[KEP Step 2/2] Running EXPERIMENTAL Group (Dynamic Free Energy Thresholding)...")
    experimental_results = run_experiment_session(mode="event_gated", data_stream=stream_data, seq_len=STREAM_LENGTH)

    # Calculate Telemetry Gains
    flops_saved_pct = (1.0 - (experimental_results["backprop_steps"] / control_results["backprop_steps"])) * 100.0
    fe_change_pct = ((experimental_results["mean_free_energy"] - control_results["mean_free_energy"]) / control_results["mean_free_energy"]) * 100.0
    speedup_ratio = control_results["total_duration_sec"] / experimental_results["total_duration_sec"]

    # =========================================================================
    # TELEMETRY RESULTS DASHBOARD
    # =========================================================================
    print("\n" + "="*80)
    print(" === KARYON ENGINEERING PROTOCOL (KEP) TELEMETRY DASHBOARD ===")
    print("="*80)
    print(f"{'Metric':<32} | {'Control Group':<18} | {'Experimental (DFET)':<18} | {'Delta / Gain':<15}")
    print("-" * 88)
    print(f"{'Total Stream Steps':<32} | {control_results['total_steps']:<18} | {experimental_results['total_steps']:<18} | {'0 (Identical)':<15}")
    print(f"{'Backprop Steps (FLOPs Proxy)':<32} | {control_results['backprop_steps']:<18} | {experimental_results['backprop_steps']:<18} | {flops_saved_pct:+.2f}% FLOPs")
    print(f"{'Mean Free Energy (F_t)':<32} | {control_results['mean_free_energy']:<18.4f} | {experimental_results['mean_free_energy']:<18.4f} | {fe_change_pct:+.2f}% Error")
    print(f"{'Final Somatic Energy Level':<32} | {control_results['final_energy']:<18.3f} | {experimental_results['final_energy']:<18.3f} | {experimental_results['final_energy'] - control_results['final_energy']:+.3f} Energy")
    print(f"{'Average Step Latency (ms)':<32} | {control_results['avg_latency_ms']:<18.2f} | {experimental_results['avg_latency_ms']:<18.2f} | {speedup_ratio:.2f}x Faster")
    print("="*80)

    # KEP Evaluation Logic
    print("\n--- [KEP EVALUATION & VERDICT] ---")
    if flops_saved_pct >= 50.0 and fe_change_pct <= 15.0:
        print("🟢 VERDICT: POSITIVE EXPERIENCE ADOPTED!")
        print(f"   Reason: Reduced Backprop FLOPs by {flops_saved_pct:.1f}% while preserving prediction accuracy.")
        print("   Action: Merge Event-Gated Dynamic Thresholding (DFET) into production runtime!")
    elif flops_saved_pct < 20.0:
        print("⚪ VERDICT: NEUTRAL EXPERIENCE DISCARDED.")
        print("   Reason: Insufficient FLOPs reduction gain.")
    else:
        print("🔴 VERDICT: NEGATIVE EXPERIENCE REJECTED.")
        print("   Reason: Excessive degradation of Variational Free Energy.")
    print("="*80 + "\n")
