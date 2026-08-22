# experiments/exp_event_gated_learning_v3.py
"""
===============================================================================
KARYON ENGINEERING PROTOCOL (KEP) EXPERIMENT #2.2 (ITERATION 3)
Topic: Calibrated Statistical Thresholding (DFET v3: Goldilocks Sigma Gating)
Author: Karyon-CoRE Open-Source Research Team (2026)
===============================================================================

Hypothesis:
    Calibrating the statistical variance multiplier to k_sigma = 0.45 will achieve 
    52-60% FLOPs reduction while keeping Free Energy error growth strictly under 
    5-6%, securing a positive KEP verdict with >1.5x real-world speedup.

Control Group: 
    Traditional Continuous Online Learning (Unconditional Backprop at EVERY step).

Experimental Group: 
    Calibrated Statistical Event-Gated Plasticity (k_sigma = 0.45).

Metrics Tracked:
    1. Mean Variational Free Energy (F_t)
    2. Total Gradient Backprop Steps (FLOPs Proxy)
    3. FLOPs Reduction Ratio (%)
    4. Final Somatic Energy Reserve
    5. Execution Speedup Ratio (x)
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

# Quick sanity check for optimizer
_test_param = nn.Parameter(torch.randn(2, 2))
_test_opt = torch.optim.Adam([_test_param], lr=1e-3)
_test_param.grad = torch.randn(2, 2)
_test_opt.step()
print("[KEP Optimizer Check] PyTorch Adam & optimizer.step() executed successfully!")

# Set global seed for exact reproducibility across Kaggle runs
torch.manual_seed(42)

# Select execution hardware
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[KEP Experiment #2.2] Execution Context: {device.type.upper()}")


# =============================================================================
# 1. STANDALONE PROTOTYPE MODULES
# =============================================================================

class SyntheticSomaticUnitV3:
    """
    Homeostatic Somatic Controller with Metabolic Energy Recovery.
    State: [Curiosity, Energy, Stability, Health, Noradrenaline (NA), Dopamine (DA)]
    """
    def __init__(self, device='cpu'):
        self.device = device
        self.state = torch.tensor([[0.5, 1.0, 1.0, 1.0, 0.0, 0.0]], dtype=torch.float32, device=device)

    def update(self, free_energy: float, was_adapted: bool):
        """Updates internal physiology with metabolic energy recovery when resting."""
        curiosity, energy, stability, health, na, da = self.state[0].tolist()

        if was_adapted:
            energy = max(0.0, energy - 0.005)
        else:
            energy = min(1.0, energy + 0.0012)

        na = min(1.0, max(0.0, 0.6 * free_energy + 0.4 * na))
        da = min(1.0, max(0.0, 0.5 * da + (0.1 if (not was_adapted and energy > 0.5) else 0.0)))

        self.state = torch.tensor([[curiosity, energy, stability, health, na, da]], dtype=torch.float32, device=self.device)
        return self.state


class PrototypeKaryonAgent(nn.Module):
    """
    Self-contained Active Inference Recurrent Core.
    """
    def __init__(self, embed_dim=128, hidden_dim=256, latent_dim=64, vocab_size=258):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size

        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.sensory_proj = nn.Linear(embed_dim, hidden_dim)
        self.rnn_cell = nn.GRUCell(hidden_dim, hidden_dim)

        self.prior_net = nn.Linear(hidden_dim, latent_dim * 2)
        self.posterior_net = nn.Linear(hidden_dim + embed_dim, latent_dim * 2)
        self.decoder = nn.Linear(latent_dim + hidden_dim, embed_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_id, h_prev):
        """Single step forward pass computing state, prediction, and Free Energy."""
        x_emb = self.token_embeddings(input_id)
        x_proj = F.silu(self.sensory_proj(x_emb))
        h_next = self.rnn_cell(x_proj, h_prev)

        prior_params = self.prior_net(h_prev)
        mu_prior, logvar_prior = prior_params.chunk(2, dim=-1)
        logvar_prior = torch.clamp(logvar_prior, -10.0, 10.0)

        post_params = self.posterior_net(torch.cat([h_prev, x_emb], dim=-1))
        mu_post, logvar_post = post_params.chunk(2, dim=-1)
        logvar_post = torch.clamp(logvar_post, -10.0, 10.0)

        std_post = torch.exp(0.5 * logvar_post)
        eps = torch.randn_like(std_post)
        z_t = mu_post + eps * std_post

        x_pred = self.decoder(torch.cat([z_t, h_next], dim=-1))

        var_prior = torch.exp(logvar_prior) + 1e-7
        var_post = torch.exp(logvar_post) + 1e-7

        kl_div = 0.5 * torch.mean(
            logvar_prior - logvar_post + (var_post + (mu_post - mu_prior)**2) / var_prior - 1.0,
            dim=-1, keepdim=True
        )
        rec_loss = torch.mean((x_emb - x_pred)**2, dim=-1, keepdim=True)
        free_energy = kl_div + rec_loss

        logits = self.head(h_next)
        return h_next, logits, free_energy, rec_loss


# =============================================================================
# 2. DATASET GENERATOR
# =============================================================================

def generate_streaming_data(seq_len=800, vocab_size=258):
    """Generates continuous byte stream with structured patterns & novel bursts."""
    stream = []
    pattern = [ord(c) for c in "Active Inference minimizes Free Energy in real-time. "]
    
    for i in range(seq_len):
        if i % 100 > 85:
            stream.append(torch.randint(0, 255, (1,)).item())
        else:
            stream.append(pattern[i % len(pattern)])
            
    return torch.tensor(stream, dtype=torch.long, device=device)


# =============================================================================
# 3. EXPERIMENTAL BENCHMARK ENGINE (DFET v3 Calibrated Gating)
# =============================================================================

def run_experiment_session_v3(mode="control", data_stream=None, seq_len=800):
    """Runs single-pass continuous learning with Calibrated Sigma Gating."""
    torch.set_grad_enabled(True)

    agent = PrototypeKaryonAgent().to(device)
    agent.train()
    
    optimizer = torch.optim.Adam(agent.parameters(), lr=2e-3)
    somatic_unit = SyntheticSomaticUnitV3(device=device)
    criterion = nn.CrossEntropyLoss()

    h_curr = torch.zeros(1, 256, device=device)
    
    total_backprop_steps = 0
    free_energy_history = []
    latency_history = []
    
    moving_mean_fe = 0.15
    moving_var_fe = 0.01
    alpha_ma = 0.05

    start_time = time.time()

    for step in range(seq_len - 1):
        step_start = time.perf_counter()

        input_id = data_stream[step].unsqueeze(0)
        target_id = data_stream[step + 1].unsqueeze(0)

        # 1. Forward Pass
        h_curr, logits, free_energy, rec_loss = agent(input_id, h_curr)
        fe_val = free_energy.item()
        free_energy_history.append(fe_val)

        task_loss = criterion(logits, target_id)
        total_step_loss = task_loss + 0.1 * free_energy.mean()

        moving_mean_fe = (1.0 - alpha_ma) * moving_mean_fe + alpha_ma * fe_val
        moving_var_fe = (1.0 - alpha_ma) * moving_var_fe + alpha_ma * ((fe_val - moving_mean_fe)**2)
        moving_std_fe = math.sqrt(max(1e-6, moving_var_fe))

        na_level = somatic_unit.state[0, 4].item()

        # 2. Calibrated Plasticity Gating Decision
        was_adapted = False

        if mode == "control":
            should_backprop = True
        else:
            # DFET v3: Calibrated threshold multiplier (k_sigma = 0.45 base)
            k_sigma = max(0.15, 0.45 - 0.25 * na_level)
            dynamic_threshold = moving_mean_fe + k_sigma * moving_std_fe
            should_backprop = fe_val > dynamic_threshold

        # 3. Backpropagation Step
        if should_backprop:
            optimizer.zero_grad()
            total_step_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=2.0)
            optimizer.step()
            
            h_curr = h_curr.detach()
            total_backprop_steps += 1
            was_adapted = True
        else:
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
# 4. MAIN EXPERIMENTAL COMPARISON & DASHBOARD
# =============================================================================

if __name__ == "__main__":
    STREAM_LENGTH = 800
    print(f"\nGenerating continuous streaming dataset ({STREAM_LENGTH} steps)...")
    stream_data = generate_streaming_data(seq_len=STREAM_LENGTH)

    print("\n[KEP Step 1/2] Running CONTROL Group (Continuous Unconditional Backprop)...")
    control_results = run_experiment_session_v3(mode="control", data_stream=stream_data, seq_len=STREAM_LENGTH)

    print("[KEP Step 2/2] Running EXPERIMENTAL Group (DFET v3: Calibrated Sigma Gating)...")
    experimental_results = run_experiment_session_v3(mode="event_gated", data_stream=stream_data, seq_len=STREAM_LENGTH)

    # Calculate Telemetry Gains
    flops_saved_pct = (1.0 - (experimental_results["backprop_steps"] / control_results["backprop_steps"])) * 100.0
    fe_change_pct = ((experimental_results["mean_free_energy"] - control_results["mean_free_energy"]) / control_results["mean_free_energy"]) * 100.0
    speedup_ratio = control_results["total_duration_sec"] / experimental_results["total_duration_sec"]

    # =========================================================================
    # TELEMETRY RESULTS DASHBOARD
    # =========================================================================
    print("\n" + "="*80)
    print(" === KARYON ENGINEERING PROTOCOL (KEP) TELEMETRY DASHBOARD (v3) ===")
    print("="*80)
    print(f"{'Metric':<32} | {'Control Group':<18} | {'Experimental (v3)':<18} | {'Delta / Gain':<15}")
    print("-" * 88)
    print(f"{'Total Stream Steps':<32} | {control_results['total_steps']:<18} | {experimental_results['total_steps']:<18} | {'0 (Identical)':<15}")
    print(f"{'Backprop Steps (FLOPs Proxy)':<32} | {control_results['backprop_steps']:<18} | {experimental_results['backprop_steps']:<18} | {flops_saved_pct:+.2f}% FLOPs")
    print(f"{'Mean Free Energy (F_t)':<32} | {control_results['mean_free_energy']:<18.4f} | {experimental_results['mean_free_energy']:<18.4f} | {fe_change_pct:+.2f}% Error")
    print(f"{'Final Somatic Energy Level':<32} | {control_results['final_energy']:<18.3f} | {experimental_results['final_energy']:<18.3f} | {experimental_results['final_energy'] - control_results['final_energy']:+.3f} Energy")
    print(f"{'Average Step Latency (ms)':<32} | {control_results['avg_latency_ms']:<18.2f} | {experimental_results['avg_latency_ms']:<18.2f} | {speedup_ratio:.2f}x Faster")
    print("="*80)

    # KEP Evaluation Logic
    print("\n--- [KEP EVALUATION & VERDICT] ---")
    if flops_saved_pct >= 50.0 and fe_change_pct <= 10.0:
        print("🟢 VERDICT: POSITIVE EXPERIENCE ADOPTED!")
        print(f"   Reason: Reduced Backprop FLOPs by {flops_saved_pct:.1f}% while keeping Free Energy error under 10%.")
        print("   Action: Merge Calibrated DFET v3 into production runtime!")
    elif flops_saved_pct < 25.0:
        print("⚪ VERDICT: NEUTRAL EXPERIENCE DISCARDED.")
        print("   Reason: Insufficient FLOPs reduction gain.")
    else:
        print("🔴 VERDICT: NEGATIVE EXPERIENCE REJECTED.")
        print("   Reason: Free Energy degradation exceeded acceptable boundary.")
    print("="*80 + "\n")
