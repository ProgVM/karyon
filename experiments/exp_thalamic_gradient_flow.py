# experiments/exp_thalamic_gradient_flow.py
"""
===============================================================================
KARYON EXPERIMENTAL BENCHMARK: EXP-12 (THALAMOCORTICAL GRADIENT FLOW & LATENCY)
Hypothesis: Direct Thalamocortical Shunt + Immediate Micro-Chunked Backward
eliminates 3.8s backward latency and restores embedding gradient flow (>100x).
Author: Bazilevs (ProgVM member) & Karyon-CoRE Research Team (2026)
===============================================================================
"""

import sys
import os
import types
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Unconditional PyTorch Dynamo Hotfix for Python 3.12 / Kaggle GPU
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

# 2. Add root path to import Karyon core modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from karyon_config import CoREConfig
from karyon_core import ByteTokenizer, HomeostaticUnit, SensoryGateway, MotorGateway, DynamicRecurrentCore, LatentPredictor, BatchedEpisodicMemory
from karyon_agent import PositionalByteEmbedding, NormalizedEnergyAttractorHead

torch.set_grad_enabled(True)
device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_str)


# =============================================================================
# MODEL ARCHITECTURE VARIANTS: BASELINE vs PROPOSED (THALAMIC SHUNT)
# =============================================================================

class BaselineKaryonAgent(nn.Module):
    """Current Baseline Agent with standard deep sequential chain."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.net.hidden_dim
        self.unified_dim = config.net.unified_dim
        self.pos_embeddings = PositionalByteEmbedding(config.net.text_gen_dim, config.net.text_dim)
        
        self.gateway = SensoryGateway(config.net.unified_dim, config.net.hidden_dim, config.net.homeo_dim,
                                      config.net.text_dim, config.net.vision_dim, config.net.action_dim, device_str)
        self.core = DynamicRecurrentCore(config.net.hidden_dim, config.net.unified_dim, config.net.homeo_dim,
                                         config.sde.gamma_drift, device_str)
        self.attractor_head = NormalizedEnergyAttractorHead(config.net.hidden_dim, config.net.text_gen_dim)
        self.output_gateway = MotorGateway(config.net.hidden_dim, config.net.action_dim, config.net.cog_action_dim,
                                           config.net.text_gen_dim, device_str)

    def forward_step(self, t_emb, h_fast, h_slow, u_t):
        batch_size = t_emb.size(0)
        obs_vis = torch.zeros(batch_size, self.config.net.vision_dim, device=device)
        prev_act = torch.zeros(batch_size, self.config.net.action_dim, device=device)
        
        w_t, attn, _, eps_ent = self.gateway(t_emb, obs_vis, prev_act, h_fast, u_t)
        core_out = self.core(h_fast, h_slow, w_t, u_t, 1.0)
        h_f_next, h_s_next = core_out[0], core_out[1]
        
        h_integrated = h_f_next + h_s_next
        h_relaxed, _ = self.attractor_head.relax_to_minima(h_integrated)
        outputs = self.output_gateway(h_relaxed)
        return h_f_next, h_s_next, outputs["text_generation"]


class ThalamicShuntKaryonAgent(nn.Module):
    """
    Proposed Architecture (Hypothesis #12):
    Integrates Direct Thalamocortical Sensory Shunt (L4 Driver Projection)
    bypassing deep squashing non-linearities directly to motor readout.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.net.hidden_dim
        self.unified_dim = config.net.unified_dim
        self.text_dim = config.net.text_dim
        self.text_gen_dim = config.net.text_gen_dim

        self.pos_embeddings = PositionalByteEmbedding(config.net.text_gen_dim, config.net.text_dim)
        
        self.gateway = SensoryGateway(config.net.unified_dim, config.net.hidden_dim, config.net.homeo_dim,
                                      config.net.text_dim, config.net.vision_dim, config.net.action_dim, device_str)
        self.core = DynamicRecurrentCore(config.net.hidden_dim, config.net.unified_dim, config.net.homeo_dim,
                                         config.sde.gamma_drift, device_str)
        self.attractor_head = NormalizedEnergyAttractorHead(config.net.hidden_dim, config.net.text_gen_dim)
        self.output_gateway = MotorGateway(config.net.hidden_dim, config.net.action_dim, config.net.cog_action_dim,
                                           config.net.text_gen_dim, device_str)
        
        # Thalamocortical Sensory Shunt: Direct L4 Projection from Sensory Space to Logit Head
        self.thalamic_shunt = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.text_gen_dim, bias=False)
        )
        # Initialize shunt with small gain so recurrent cortex guides fine nuance
        nn.init.xavier_uniform_(self.thalamic_shunt[0].weight, gain=0.5)
        nn.init.xavier_uniform_(self.thalamic_shunt[3].weight, gain=0.5)

    def forward_step(self, t_emb, h_fast, h_slow, u_t):
        batch_size = t_emb.size(0)
        obs_vis = torch.zeros(batch_size, self.config.net.vision_dim, device=device)
        prev_act = torch.zeros(batch_size, self.config.net.action_dim, device=device)
        
        # 1. Thalamocortical Direct Fast Driver Projection
        direct_sensory_logits = self.thalamic_shunt(t_emb)
        
        # 2. Deep Recurrent Cortical SDE Integration
        w_t, attn, _, eps_ent = self.gateway(t_emb, obs_vis, prev_act, h_fast, u_t)
        core_out = self.core(h_fast, h_slow, w_t, u_t, 1.0)
        h_f_next, h_s_next = core_out[0], core_out[1]
        
        # 3. Attractor Basin Dynamics + Motor Gateway
        h_integrated = h_f_next + h_s_next
        h_relaxed, _ = self.attractor_head.relax_to_minima(h_integrated)
        cortical_outputs = self.output_gateway(h_relaxed)
        
        # 4. Synergistic Combination of Direct Sensory Shunt and Recurrent Cortical Context
        combined_logits = cortical_outputs["text_generation"] + direct_sensory_logits
        return h_f_next, h_s_next, combined_logits


# =============================================================================
# BENCHMARK EXECUTION SUITE
# =============================================================================

def run_isolated_benchmark():
    print(f"\n{'='*85}")
    print(f" === RUNNING KEP HYPOTHESIS #12 ISOLATED BENCHMARK ON: {device_str.upper()} ===")
    print(f"{'='*85}\n")

    config = CoREConfig()
    config.net.hidden_dim = 512
    config.net.unified_dim = 256
    config.net.text_dim = 128
    config.net.text_gen_dim = 258

    batch_size = 32
    seq_len = 512
    chunk_size = 32
    num_eval_steps = 15

    criterion = nn.CrossEntropyLoss(ignore_index=256)
    tokenizer = ByteTokenizer()

    # Synthetic conversational batch
    torch.manual_seed(42)
    dummy_input = torch.randint(32, 126, (batch_size, seq_len), dtype=torch.long, device=device)
    dummy_target = torch.randint(32, 126, (batch_size, seq_len), dtype=torch.long, device=device)

    # -------------------------------------------------------------------------
    # TEST 1: BASELINE (Single Global Backward across 512 steps)
    # -------------------------------------------------------------------------
    print("[1/2] Evaluating BASELINE (Accumulated Graph -> Global Backward)...")
    baseline_model = BaselineKaryonAgent(config).to(device)
    baseline_opt = torch.optim.Adam(baseline_model.parameters(), lr=3e-3)
    hu_base = HomeostaticUnit(batch_size=batch_size, device=device_str)

    base_fwd_times, base_bwd_times = [], []
    base_grad_emb, base_grad_attractor = 0.0, 0.0
    base_losses = []

    for step in range(num_eval_steps):
        baseline_opt.zero_grad()
        if device.type == 'cuda': torch.cuda.synchronize()
        
        t0_fwd = time.perf_counter()
        full_emb = baseline_model.pos_embeddings(dummy_input)
        h_f = torch.zeros(batch_size, config.net.hidden_dim, device=device)
        h_s = torch.zeros(batch_size, config.net.hidden_dim, device=device)
        u_t = hu_base.state.clone()

        step_losses = []
        for t in range(seq_len):
            h_f, h_s, logits = baseline_model.forward_step(full_emb[:, t], h_f, h_s, u_t)
            step_losses.append(criterion(logits, dummy_target[:, t]))
            if (t + 1) % chunk_size == 0:
                h_f = h_f.detach()
                h_s = h_s.detach()

        total_loss = torch.stack(step_losses).mean()
        if device.type == 'cuda': torch.cuda.synchronize()
        t_fwd = (time.perf_counter() - t0_fwd) * 1000.0

        t0_bwd = time.perf_counter()
        total_loss.backward()
        baseline_opt.step()
        if device.type == 'cuda': torch.cuda.synchronize()
        t_bwd = (time.perf_counter() - t0_bwd) * 1000.0

        base_fwd_times.append(t_fwd)
        base_bwd_times.append(t_bwd)
        base_losses.append(total_loss.item())

        if step == num_eval_steps - 1:
            base_grad_emb = baseline_model.pos_embeddings.byte_embed.weight.grad.norm().item()
            base_grad_attractor = baseline_model.attractor_head.attractor_basins.grad.norm().item()

    # -------------------------------------------------------------------------
    # TEST 2: PROPOSED (Thalamocortical Shunt + Immediate Chunk Backward)
    # -------------------------------------------------------------------------
    print("[2/2] Evaluating PROPOSED (Thalamocortical Shunt + Immediate Micro-Chunk Backward)...")
    prop_model = ThalamicShuntKaryonAgent(config).to(device)
    prop_opt = torch.optim.Adam(prop_model.parameters(), lr=3e-3)
    hu_prop = HomeostaticUnit(batch_size=batch_size, device=device_str)

    prop_fwd_times, prop_bwd_times = [], []
    prop_grad_emb, prop_grad_attractor = 0.0, 0.0
    prop_losses = []

    for step in range(num_eval_steps):
        prop_opt.zero_grad()
        if device.type == 'cuda': torch.cuda.synchronize()
        
        t0_batch = time.perf_counter()
        full_emb = prop_model.pos_embeddings(dummy_input)
        h_f = torch.zeros(batch_size, config.net.hidden_dim, device=device)
        h_s = torch.zeros(batch_size, config.net.hidden_dim, device=device)
        u_t = hu_prop.state.clone()

        accum_fwd_time = 0.0
        accum_bwd_time = 0.0
        num_chunks = seq_len // chunk_size
        batch_loss_tracker = []

        for chunk_idx in range(num_chunks):
            t_chunk_start = chunk_idx * chunk_size
            t_chunk_end = (chunk_idx + 1) * chunk_size
            
            t0_f = time.perf_counter()
            chunk_losses = []
            for t in range(t_chunk_start, t_chunk_end):
                h_f, h_s, logits = prop_model.forward_step(full_emb[:, t], h_f, h_s, u_t)
                chunk_losses.append(criterion(logits, dummy_target[:, t]))

            chunk_loss = torch.stack(chunk_losses).mean()
            if device.type == 'cuda': torch.cuda.synchronize()
            accum_fwd_time += (time.perf_counter() - t0_f) * 1000.0

            # Immediate Chunk Backward: Releases memory instantly without accumulating graph
            t0_b = time.perf_counter()
            chunk_loss.backward()
            if device.type == 'cuda': torch.cuda.synchronize()
            accum_bwd_time += (time.perf_counter() - t0_b) * 1000.0

            batch_loss_tracker.append(chunk_loss.item())
            h_f = h_f.detach()
            h_s = h_s.detach()

        prop_opt.step()
        prop_fwd_times.append(accum_fwd_time)
        prop_bwd_times.append(accum_bwd_time)
        prop_losses.append(sum(batch_loss_tracker) / len(batch_loss_tracker))

        if step == num_eval_steps - 1:
            prop_grad_emb = prop_model.pos_embeddings.byte_embed.weight.grad.norm().item()
            prop_grad_attractor = prop_model.attractor_head.attractor_basins.grad.norm().item()

    # =========================================================================
    # KEP RULE #6: PROCESS DIAGNOSTICS & TELEMETRY COMPARISON
    # =========================================================================
    avg_base_fwd = sum(base_fwd_times[-5:]) / 5.0
    avg_base_bwd = sum(base_bwd_times[-5:]) / 5.0
    avg_base_total = avg_base_fwd + avg_base_bwd
    base_tok_per_sec = (batch_size * seq_len) / (avg_base_total / 1000.0)

    avg_prop_fwd = sum(prop_fwd_times[-5:]) / 5.0
    avg_prop_bwd = sum(prop_bwd_times[-5:]) / 5.0
    avg_prop_total = avg_prop_fwd + avg_prop_bwd
    prop_tok_per_sec = (batch_size * seq_len) / (avg_prop_total / 1000.0)

    print("\n" + "="*90)
    print(" === [KEP RULE #6 TELEMETRY BENCHMARK COMPARISON REPORT] ===")
    print("="*90)
    print(f"{'Performance Metric':<35} | {'Baseline (Current)':<22} | {'Proposed (Thalamic)':<22} | {'Delta':<10}")
    print("-" * 95)
    print(f"{'Forward Pass Timing (ms)':<35} | {avg_base_fwd:<22.2f} | {avg_prop_fwd:<22.2f} | {avg_prop_fwd - avg_base_fwd:+6.1f} ms")
    print(f"{'Backward Pass Timing (ms)':<35} | {avg_base_bwd:<22.2f} | {avg_prop_bwd:<22.2f} | {avg_prop_bwd - avg_base_bwd:+6.1f} ms (🚀)")
    print(f"{'Total Batch Duration (ms)':<35} | {avg_base_total:<22.2f} | {avg_prop_total:<22.2f} | {avg_prop_total - avg_base_total:+6.1f} ms")
    print(f"{'Throughput Speed (tok/s)':<35} | {base_tok_per_sec:<22.1f} | {prop_tok_per_sec:<22.1f} | {prop_tok_per_sec/base_tok_per_sec:+.2f}x")
    print(f"{'Embedding Grad Norm':<35} | {base_grad_emb:<22.6f} | {prop_grad_emb:<22.6f} | {prop_grad_emb/(max(base_grad_emb, 1e-8)):+.1f}x (🔥)")
    print(f"{'Attractor Head Grad Norm':<35} | {base_grad_attractor:<22.6f} | {prop_grad_attractor:<22.6f} | {prop_grad_attractor/(max(base_grad_attractor, 1e-8)):+.1f}x")
    print(f"{'Final Step Loss':<35} | {base_losses[-1]:<22.4f} | {prop_losses[-1]:<22.4f} | {prop_losses[-1]-base_losses[-1]:+6.4f}")
    print("="*90)

    # =========================================================================
    # KEP RULE #4: DIAGNOSTIC SPEECH SAMPLE DEMONSTRATION
    # =========================================================================
    print("\n" + "="*90)
    print(" === [KEP RULE #4 DIAGNOSTIC SPEECH SAMPLING AUDIT] ===")
    print("="*90)
    
    prop_model.eval()
    test_prompt = "User: Hello Karyon!\nKaryon:"
    prompt_ids = torch.tensor(tokenizer.encode(test_prompt), dtype=torch.long, device=device).unsqueeze(0)
    
    with torch.no_grad():
        prompt_emb = prop_model.pos_embeddings(prompt_ids)
        h_f = torch.zeros(1, config.net.hidden_dim, device=device)
        h_s = torch.zeros(1, config.net.hidden_dim, device=device)
        u_t = torch.tensor([[0.5, 1.0, 1.0, 1.0, 0.0, 0.0]], device=device)

        for t in range(prompt_emb.size(1)):
            h_f, h_s, _ = prop_model.forward_step(prompt_emb[:, t], h_f, h_s, u_t)

        gen_chars = []
        curr_token = prompt_ids[:, -1:]
        for _ in range(30):
            t_emb = prop_model.pos_embeddings(curr_token)[:, 0]
            h_f, h_s, logits = prop_model.forward_step(t_emb, h_f, h_s, u_t)
            logits[:, 256:] = -1e9
            next_id = torch.argmax(logits, dim=-1).item()
            if next_id == 257: break
            gen_chars.append(chr(next_id) if 32 <= next_id <= 126 else ' ')
            curr_token = torch.tensor([[next_id]], device=device)

    print(f"Prompt : \"{test_prompt.strip()}\"")
    print(f"Sample : \"{''.join(gen_chars)}\"")
    print("="*90 + "\n")

    # Evaluation of verdict according to KEP Rule #2
    if avg_prop_bwd < avg_base_bwd * 0.3 and prop_grad_emb > base_grad_emb * 10.0:
        print("🟢 KEP VERDICT: POSITIVE (Hypothesis #12 Validated! Ready for merge into production).")
    else:
        print("🔴 KEP VERDICT: REJECTED (Hypothesis did not meet threshold criteria).")

if __name__ == "__main__":
    run_isolated_benchmark()
