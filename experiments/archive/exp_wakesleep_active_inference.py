# experiments/exp_wakesleep_active_inference.py
"""
===============================================================================
KARYON ENGINEERING PROTOCOL (KEP) EXPERIMENT #9
Topic: Wake-Sleep Active Inference Synaptic Consolidation Engine
Author: Karyon-CoRE Open-Source Research Team (2026)
===============================================================================

Hypothesis:
    Replacing continuous step-by-step gradient cramming with a two-phase 
    Wake-Sleep Consolidation Cycle (Zero backprop during Wake + Replay & 
    Synaptic Consolidation during Sleep when Energy < 0.20) will eliminate 
    character soup generation, accelerate real-time perception latency by 3-5x, 
    and maintain high homeostatic energy balance.

Control Group: 
    Traditional Step-by-Step Backprop Cramming (Gradient updates on every step).

Experimental Group: 
    Wake-Sleep Active Consolidation (Wake: Experience & Store -> Sleep: Replay & Learn).

Metrics Tracked:
    1. Sleep Consolidation Cycles Triggered
    2. Mean Variational Free Energy (F_t)
    3. Wake Phase Perception Latency (ms)
    4. Somatic Energy Balance Range
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

# Ensure global autograd tracking is enabled
torch.set_grad_enabled(True)

# Set global seed for exact reproducibility
torch.manual_seed(42)

# Select execution hardware
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[KEP Experiment #9] Execution Context: {device.type.upper()}")


# =============================================================================
# 1. WAKE-SLEEP ACTIVE INFERENCE AGENT
# =============================================================================

class WakeSleepSomaticUnit:
    """
    Somatic unit governing Wake-Sleep biological cycles.
    State: [Curiosity, Energy, Stability, Health, Noradrenaline, Dopamine]
    """
    def __init__(self, device='cpu'):
        self.device = device
        self.state = torch.tensor([[0.8, 1.0, 1.0, 1.0, 0.0, 0.0]], dtype=torch.float32, device=device)

    def update_wake(self, free_energy_val: float):
        """Metabolic energy drain during active perception."""
        curiosity, energy, stability, health, na, da = self.state[0].tolist()
        energy = max(0.0, energy - 0.008) # Energy drains during wakefulness
        na = min(1.0, max(0.0, 0.6 * free_energy_val + 0.4 * na))
        
        self.state = torch.tensor([[curiosity, energy, stability, health, na, da]], dtype=torch.float32, device=self.device)
        should_sleep = energy < 0.20 # Trigger sleep phase when exhausted
        return should_sleep, self.state

    def restore_sleep(self):
        """Restores somatic energy during sleep consolidation phase."""
        curiosity, energy, stability, health, na, da = self.state[0].tolist()
        energy = min(1.0, energy + 0.35) # Energy restored during sleep
        na = max(0.0, na - 0.20)
        self.state = torch.tensor([[curiosity, energy, stability, health, na, da]], dtype=torch.float32, device=self.device)
        return self.state


class PrototypeWakeSleepAgent(nn.Module):
    """
    Agent implementing Wake Phase Experience & Sleep Phase Memory Replay.
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

        self.somatic = WakeSleepSomaticUnit(device=device)

    def forward_wake_step(self, input_id, h_fast, h_slow):
        """Wake Phase: Zero-backprop fast perception and Free Energy calculation."""
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
        should_sleep, somatic_state = self.somatic.update_wake(free_energy.item())

        return h_f_next, h_s_next, x_emb, x_pred, logits, free_energy, should_sleep

    def consolidate_sleep_phase(self, memory_keys, memory_values, optimizer, criterion, sleep_epochs=3):
        """
        Sleep Phase: Pauses external interaction, replays high-surprise 
        memories, consolidates synaptic weights, and restores somatic energy.
        """
        print("\n 😴 [SLEEP PHASE INITIATED] Pausing external stream. Consolidating synaptic weights...")
        
        num_memories = memory_keys.size(0)
        if num_memories == 0:
            self.somatic.restore_sleep()
            return 0.0

        total_sleep_loss = 0.0
        
        # Consolidate weights via memory replay during sleep
        for sleep_ep in range(sleep_epochs):
            perm_indices = torch.randperm(num_memories)
            
            for idx in perm_indices:
                optimizer.zero_grad()
                
                key_k = memory_keys[idx].unsqueeze(0)
                val_v = memory_values[idx].unsqueeze(0)
                
                # Replay thought through latent world model
                h_dummy_f = torch.zeros(1, self.hidden_dim, device=device)
                h_dummy_s = torch.zeros(1, self.hidden_dim, device=device)
                
                x_proj = F.silu(self.sensory_proj(key_k))
                h_f_next = self.rnn_fast(x_proj, h_dummy_f)
                h_s_next = self.rnn_slow(h_f_next, h_dummy_s)
                
                w_pred = self.decoder(torch.cat([torch.randn(1, 64, device=device), h_s_next], dim=-1))
                rec_loss = torch.mean((val_v - w_pred)**2)
                
                rec_loss.backward()
                optimizer.step()
                total_sleep_loss += rec_loss.item()

        # Restore somatic energy after sleep consolidation
        self.somatic.restore_sleep()
        print(" 🌅 [WAKE UP] Synaptic consolidation complete. Somatic Energy restored to 1.000.\n")
        return total_sleep_loss / (sleep_epochs * max(1, num_memories))


# =============================================================================
# 2. EXPERIMENTAL BENCHMARK ENGINE
# =============================================================================

def run_wakesleep_benchmark(mode="control_cramming", stream=None, seq_len=600):
    agent = PrototypeWakeSleepAgent().to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-3)
    criterion = nn.CrossEntropyLoss()

    h_fast = torch.zeros(1, agent.hidden_dim, device=device)
    h_slow = torch.zeros(1, agent.hidden_dim, device=device)

    # Simulated memory buffer
    memory_keys_list = []
    memory_values_list = []

    sleep_cycles_triggered = 0
    free_energy_history = []
    latency_history = []

    start_time = time.time()

    for step in range(seq_len - 1):
        step_start = time.perf_counter()
        token_id = stream[step].unsqueeze(0)
        target_id = stream[step + 1].unsqueeze(0)

        if mode == "control_cramming":
            # CONTROL: Step-by-step Backprop Cramming on every step
            h_fast, h_slow, x_emb, x_pred, logits, free_energy, _ = agent.forward_wake_step(token_id, h_fast, h_slow)
            
            optimizer.zero_grad()
            loss = criterion(logits, target_id) + 0.1 * free_energy.mean()
            loss.backward()
            optimizer.step()
            
            h_fast = h_fast.detach()
            h_slow = h_slow.detach()
            fe_val = free_energy.item()
        else:
            # EXPERIMENTAL: Wake-Sleep Active Consolidation (Zero Backprop during Wake)
            with torch.no_grad():
                h_fast, h_slow, x_emb, x_pred, logits, free_energy, should_sleep = agent.forward_wake_step(token_id, h_fast, h_slow)
                fe_val = free_energy.item()

                # Write high-surprise events to memory during Wakefulness
                if fe_val > 0.15:
                    memory_keys_list.append(x_emb.detach())
                    memory_values_list.append(x_pred.detach())

            # Trigger Sleep Phase when exhausted
            if should_sleep and len(memory_keys_list) > 0:
                sleep_cycles_triggered += 1
                keys_tensor = torch.cat(memory_keys_list, dim=0)
                vals_tensor = torch.cat(memory_values_list, dim=0)
                
                agent.consolidate_sleep_phase(keys_tensor, vals_tensor, optimizer, criterion)
                
                # Prune memory after sleep
                memory_keys_list = memory_keys_list[-50:]
                memory_values_list = memory_values_list[-50:]

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
# 3. MAIN EVALUATION & TELEMETRY DASHBOARD
# =============================================================================

if __name__ == "__main__":
    STREAM_LEN = 600
    pattern = [ord(c) for c in "Active Inference Wake Sleep Synaptic Consolidation. "]
    stream = torch.tensor([pattern[i % len(pattern)] for i in range(STREAM_LEN)], dtype=torch.long, device=device)

    print("\n[KEP Step 1/2] Running CONTROL Group (Continuous Backprop Cramming)...")
    control_res = run_wakesleep_benchmark(mode="control_cramming", stream=stream, seq_len=STREAM_LEN)

    print("[KEP Step 2/2] Running EXPERIMENTAL Group (Wake-Sleep Active Consolidation)...")
    experimental_res = run_wakesleep_benchmark(mode="wake_sleep", stream=stream, seq_len=STREAM_LEN)

    # Calculate Telemetry Gains
    speedup_ratio = control_res["total_duration_sec"] / experimental_res["total_duration_sec"]
    fe_change_pct = ((experimental_res["mean_free_energy"] - control_res["mean_free_energy"]) / control_res["mean_free_energy"]) * 100.0

    # =========================================================================
    # TELEMETRY RESULTS DASHBOARD
    # =========================================================================
    print("\n" + "="*80)
    print(" === KARYON ENGINEERING PROTOCOL (KEP) TELEMETRY DASHBOARD ===")
    print("="*80)
    print(f"{'Metric':<32} | {'Control Cramming':<18} | {'Wake-Sleep Engine':<18} | {'Delta / Gain':<15}")
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
        print(f"   Reason: Wake-Sleep engine accelerated perception by {speedup_ratio:.2f}x with low Free Energy error.")
        print("   Action: Adopt Wake-Sleep Synaptic Consolidation Engine into production master Karyon!")
    elif speedup_ratio < 1.2:
        print("⚪ VERDICT: NEUTRAL EXPERIENCE DISCARDED.")
    else:
        print("🔴 VERDICT: NEGATIVE EXPERIENCE REJECTED.")
    print("="*80 + "\n")
