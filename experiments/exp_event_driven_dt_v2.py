# exp_event_driven_dt_v2.py
"""
===============================================================================
KARYON ENGINEERING PROTOCOL (KEP) EXPERIMENT #3.1 (ITERATION 2)
Topic: Logarithmic Time Integration & Passive Decay State-Skipping
Author: Karyon-CoRE Open-Source Research Team (2026)
===============================================================================

Hypothesis:
    Applying logarithmic time-delta compression (dt_eff = 1 + log(1 + dt_acc)) 
    combined with lightweight exponential passive decay during skipped steps 
    will prevent non-linear SDE trajectory explosion (L2 divergence < 2.5) 
    while preserving >30% SDE FLOPs reduction and >1.3x speedup.

Control Group: 
    Fixed-step continuous execution (Full SDE integration at every single step).

Experimental Group: 
    Event-Driven Skipping with Logarithmic dt Compression & Passive Decay.

Metrics Tracked:
    1. Recurrent Core Executions (FLOPs Saved %)
    2. Hidden State Divergence Error (L2 Norm)
    3. Execution Speedup Ratio (x)
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

# Set global seed for reproducibility
torch.manual_seed(42)

# Select execution hardware
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[KEP Experiment #3.1] Execution Context: {device.type.upper()}")


# =============================================================================
# 1. 2ND-ORDER STOCHASTIC HEUN SDE CORE WITH LOGARITHMIC DT
# =============================================================================

class StochasticHeunSDECoreV2(nn.Module):
    """
    2nd-Order Predictor-Corrector Heun SDE Recurrent Engine.
    """
    def __init__(self, hidden_dim=256, unified_dim=128, gamma=0.1, sigma=0.001):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.sigma = sigma

        self.slow_f = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim)
        )

        self.fast_f = nn.Sequential(
            nn.Linear(hidden_dim * 2 + unified_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, h_fast_prev, h_slow_prev, w_t, dt=1.0):
        """2nd-Order Predictor-Corrector Heun SDE Step with bounded dt."""
        # Bounded dt preventing activation numerical explosion
        bounded_dt = min(2.5, max(0.1, dt))
        sqrt_dt = math.sqrt(bounded_dt)
        
        dW_slow = torch.randn_like(h_slow_prev) * sqrt_dt * self.sigma
        dW_fast = torch.randn_like(h_fast_prev) * sqrt_dt * self.sigma

        # 1. Slow drift predictor-corrector
        k1_slow = -self.gamma * h_slow_prev + self.slow_f(h_slow_prev)
        h_slow_pred = h_slow_prev + bounded_dt * k1_slow + dW_slow

        k2_slow = -self.gamma * h_slow_pred + self.slow_f(h_slow_pred)
        h_slow_next = torch.tanh(h_slow_prev + 0.5 * bounded_dt * (k1_slow + k2_slow) + dW_slow)

        # 2. Fast drift predictor-corrector
        fast_in_1 = torch.cat([h_fast_prev, w_t, h_slow_next], dim=-1)
        k1_fast = -self.gamma * h_fast_prev + self.fast_f(fast_in_1)
        h_fast_pred = h_fast_prev + bounded_dt * k1_fast + dW_fast

        fast_in_2 = torch.cat([h_fast_pred, w_t, h_slow_next], dim=-1)
        k2_fast = -self.gamma * h_fast_pred + self.fast_f(fast_in_2)
        h_fast_next = torch.tanh(h_fast_prev + 0.5 * bounded_dt * (k1_fast + k2_fast) + dW_fast)

        return h_fast_next, h_slow_next


class LogarithmicEventAgent(nn.Module):
    """
    Karyon Agent with Logarithmic dt Compression and Passive Decay Skipping.
    """
    def __init__(self, vocab_size=258, text_dim=128, hidden_dim=256, unified_dim=128, gamma=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.unified_dim = unified_dim
        self.gamma = gamma

        self.token_embeddings = nn.Embedding(vocab_size, text_dim)
        self.text_proj = nn.Linear(text_dim, unified_dim)

        self.sde_core = StochasticHeunSDECoreV2(hidden_dim, unified_dim, gamma=gamma)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward_step(self, input_id, h_fast, h_slow, prev_sensory, accumulated_dt=0.0, event_threshold=0.08, mode="control"):
        """
        Processes step. On sensory shift < threshold:
        Applies fast O(1) passive decay without full SDE matrix evaluation.
        """
        x_emb = self.token_embeddings(input_id)
        w_t = F.silu(self.text_proj(x_emb))

        if prev_sensory is None:
            sensory_delta = 999.0
        else:
            sensory_delta = torch.norm(w_t - prev_sensory, p=2).item()

        sde_executed = False

        if mode == "control" or sensory_delta >= event_threshold:
            # Compress accumulated time deltas logarithmically: dt_eff = 1.0 + log(1 + dt_acc)
            effective_dt = 1.0 + math.log(1.0 + accumulated_dt) if accumulated_dt > 0 else 1.0
            
            # Execute full 2nd-order SDE integration step
            h_fast_next, h_slow_next = self.sde_core(h_fast, h_slow, w_t, dt=effective_dt)
            reset_accumulated_dt = 0.0 # Reset accumulated time
            sde_executed = True
        else:
            # LIGHTWEIGHT PASSIVE DECAY: O(1) continuous state drift without matrix multiplications
            decay_factor = math.exp(-self.gamma * 0.05)
            h_fast_next = h_fast * decay_factor
            h_slow_next = h_slow * decay_factor
            reset_accumulated_dt = accumulated_dt + 1.0

        logits = self.head(h_fast_next + h_slow_next)
        return h_fast_next, h_slow_next, w_t, logits, reset_accumulated_dt, sde_executed


# =============================================================================
# 2. CONTINUOUS STREAM DATA GENERATOR WITH STATIC PAUSES
# =============================================================================

def generate_paused_stream(seq_len=1000):
    """
    Generates byte stream with repeating static byte sequences (pauses/repeats) 
    interspersed with dynamic text events.
    """
    stream = []
    text_pattern = [ord(c) for c in "Continuous Active Inference with Dynamic Time Delta Integration. "]
    
    for i in range(seq_len):
        if 200 <= i <= 350 or 500 <= i <= 650:
            stream.append(32) # Space byte ' '
        else:
            stream.append(text_pattern[i % len(text_pattern)])

    return torch.tensor(stream, dtype=torch.long, device=device)


# =============================================================================
# 3. EXPERIMENTAL BENCHMARK ENGINE
# =============================================================================

def run_event_driven_benchmark_v2(mode="control", stream=None, event_threshold=0.08):
    agent = LogarithmicEventAgent().to(device)
    agent.eval()

    h_fast = torch.zeros(1, 256, device=device)
    h_slow = torch.zeros(1, 256, device=device)
    prev_sensory = None

    accumulated_dt = 0.0
    sde_execution_count = 0
    total_steps = stream.size(0) - 1

    latency_history = []
    start_time = time.time()

    with torch.no_grad():
        for step in range(total_steps):
            step_start = time.perf_counter()
            token_id = stream[step].unsqueeze(0)

            h_fast, h_slow, prev_sensory, logits, accumulated_dt, was_executed = agent.forward_step(
                token_id, h_fast, h_slow, prev_sensory,
                accumulated_dt=accumulated_dt,
                event_threshold=event_threshold,
                mode=mode
            )

            if was_executed:
                sde_execution_count += 1

            latency_history.append((time.perf_counter() - step_start) * 1000.0)

    total_duration = time.time() - start_time
    avg_latency = sum(latency_history) / len(latency_history)

    return {
        "mode": mode,
        "total_steps": total_steps,
        "sde_executions": sde_execution_count,
        "avg_latency_ms": avg_latency,
        "total_duration_sec": total_duration,
        "final_state": h_fast.clone()
    }


# =============================================================================
# 4. MAIN EVALUATION & TELEMETRY DASHBOARD
# =============================================================================

if __name__ == "__main__":
    STREAM_LEN = 1000
    print(f"\nGenerating event-driven stream dataset ({STREAM_LEN} steps with static pauses)...")
    stream = generate_paused_stream(seq_len=STREAM_LEN)

    print("\n[KEP Step 1/2] Running CONTROL Group (Continuous Fixed-Step SDE Execution)...")
    control_res = run_event_driven_benchmark_v2(mode="control", stream=stream)

    print("[KEP Step 2/2] Running EXPERIMENTAL Group (Logarithmic dt Compression & Passive Decay)...")
    experimental_res = run_event_driven_benchmark_v2(mode="event_driven", stream=stream, event_threshold=0.08)

    # Calculate Telemetry Gains
    sde_saved_pct = (1.0 - (experimental_res["sde_executions"] / control_res["sde_executions"])) * 100.0
    speedup_ratio = control_res["total_duration_sec"] / experimental_res["total_duration_sec"]
    state_divergence = torch.norm(control_res["final_state"] - experimental_res["final_state"]).item()

    # =========================================================================
    # TELEMETRY RESULTS DASHBOARD (v2)
    # =========================================================================
    print("\n" + "="*80)
    print(" === KARYON ENGINEERING PROTOCOL (KEP) TELEMETRY DASHBOARD (v2) ===")
    print("="*80)
    print(f"{'Metric':<32} | {'Control Group':<18} | {'Experimental (v2)':<18} | {'Delta / Gain':<15}")
    print("-" * 88)
    print(f"{'Total Stream Steps':<32} | {control_res['total_steps']:<18} | {experimental_res['total_steps']:<18} | {'0 (Identical)':<15}")
    print(f"{'SDE Core Executions (FLOPs)':<32} | {control_res['sde_executions']:<18} | {experimental_res['sde_executions']:<18} | {sde_saved_pct:+.2f}% FLOPs")
    print(f"{'State Divergence (L2 Norm)':<32} | {'0.0000 (Base)':<18} | {state_divergence:<18.4f} | {'Diff':<15}")
    print(f"{'Average Step Latency (ms)':<32} | {control_res['avg_latency_ms']:<18.2f} | {experimental_res['avg_latency_ms']:<18.2f} | {speedup_ratio:.2f}x Faster")
    print("="*80)

    # KEP Evaluation Logic
    print("\n--- [KEP EVALUATION & VERDICT] ---")
    if sde_saved_pct >= 30.0 and state_divergence <= 2.5:
        print("🟢 VERDICT: POSITIVE EXPERIENCE ADOPTED!")
        print(f"   Reason: Saved {sde_saved_pct:.1f}% SDE Core Executions with minimal state drift ({state_divergence:.3f}).")
        print("   Action: Merge Logarithmic dt Compression & Passive Decay into production karyon_core!")
    elif sde_saved_pct < 15.0:
        print("⚪ VERDICT: NEUTRAL EXPERIENCE DISCARDED.")
        print("   Reason: Insufficient FLOPs reduction gain.")
    else:
        print("🔴 VERDICT: NEGATIVE EXPERIENCE REJECTED.")
        print("   Reason: Excessive hidden state divergence.")
    print("="*80 + "\n")
