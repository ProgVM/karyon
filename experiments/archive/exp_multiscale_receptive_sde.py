# experiments/exp_multiscale_receptive_sde.py
"""
feat(exp): implement multiscale temporal sde and causal byte receptive field

===============================================================================
KARYON EXPERIMENTAL BENCHMARK: EXP-15 (MULTISCALE TEMPORAL SDE & CAUSAL SHUNT)
Hypothesis: Multi-Scale SDE (slow semantic theta-rhythm + fast gamma-rhythm) +
Causal N-gram Byte Receptive Field (K=4) + Low-Pass EMA Somatic Surprisal
eliminates loss plateaus and produces structured, multi-word linguistic coherence.
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

torch.set_grad_enabled(True)
device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_str)


# =============================================================================
# MODULE 1: CAUSAL N-GRAM BYTE RECEPTIVE FIELD (K=4)
# =============================================================================

class CausalByteReceptiveField(nn.Module):
    """
    Biological Sensory Receptive Field (K=4).
    Applies strict causal 1D depthwise convolution over previous bytes,
    converting raw byte streams into phonemic and morphological representations.
    """
    def __init__(self, text_dim=128, kernel_size=4):
        super().__init__()
        self.kernel_size = kernel_size
        self.text_dim = text_dim
        self.conv = nn.Conv1d(
            in_channels=text_dim,
            out_channels=text_dim,
            kernel_size=kernel_size,
            groups=text_dim, # Depthwise causal filter
            bias=False
        )
        self.norm = nn.LayerNorm(text_dim)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq shape: [Batch, SeqLen, TextDim]
        b, s, d = x_seq.size()
        x_trans = x_seq.transpose(1, 2) # [B, TextDim, SeqLen]
        # Strict left causal padding
        x_padded = F.pad(x_trans, (self.kernel_size - 1, 0), mode='constant', value=0.0)
        conv_out = self.conv(x_padded) # [B, TextDim, SeqLen]
        out = self.norm(conv_out.transpose(1, 2) + x_seq) # Residual connection
        return out


# =============================================================================
# MODULE 2: POSITIONAL BYTE EMBEDDING WITH RECEPTIVE FIELD INTEGRATION
# =============================================================================

class OffsetPositionalByteEmbedding(nn.Module):
    def __init__(self, vocab_size=258, text_dim=128, max_len=4096):
        super().__init__()
        self.vocab_size = vocab_size
        self.text_dim = text_dim
        self.byte_embed = nn.Embedding(vocab_size, text_dim)
        self.receptive_field = CausalByteReceptiveField(text_dim=text_dim, kernel_size=4)
        
        pe = torch.zeros(max_len, text_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, text_dim, 2).float() * (-math.log(10000.0) / text_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, input_ids: torch.Tensor, start_pos: int = 0, apply_receptive_field: bool = True) -> torch.Tensor:
        seq_len = input_ids.size(1)
        tok_emb = self.byte_embed(input_ids)
        pos_emb = self.pe[:, start_pos : start_pos + seq_len, :]
        embedded = tok_emb + pos_emb
        if apply_receptive_field:
            embedded = self.receptive_field(embedded)
        return embedded


# =============================================================================
# MODULE 3: DESATURATED HOPFIELD ATTRACTOR MEMORY
# =============================================================================

class DesaturatedHopfieldAttractorHead(nn.Module):
    def __init__(self, hidden_dim=512, vocab_size=258, num_attractors=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scale = 1.0 / math.sqrt(hidden_dim)
        self.attractor_basins = nn.Parameter(torch.randn(num_attractors, hidden_dim) * 0.05)

    def relax_to_minima(self, h_state: torch.Tensor):
        norm_dist_sq = (torch.cdist(h_state, self.attractor_basins, p=2)**2) * self.scale
        attn_weights = F.softmax(-norm_dist_sq, dim=-1)
        attractor_shift = torch.matmul(attn_weights, self.attractor_basins)
        h_relaxed = h_state + 0.25 * attractor_shift
        energy = -torch.logsumexp(-norm_dist_sq, dim=-1, keepdim=True)
        return h_relaxed, energy


# =============================================================================
# MODULE 4: MULTI-SCALE TEMPORAL SDE CORE (SLOW THETA + FAST GAMMA CLOCKS)
# =============================================================================

class MultiScaleTemporalSDECore(nn.Module):
    """
    Biological Multi-Scale Temporal SDE.
    - h_slow: Operates at theta-scale (dt_slow = 0.15 * dt_fast), acting as a steady semantic anchor.
    - h_fast: Operates at gamma-scale (per-byte), dynamically conditioned on h_slow and incoming tokens.
    """
    def __init__(self, hidden_dim=512, unified_dim=256, homeo_dim=6, gamma=0.10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gamma = gamma

        # Slow Semantic Core (Theta Rhythm)
        self.slow_drift_net = nn.Sequential(
            nn.Linear(hidden_dim + homeo_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim)
        )

        # Fast Phonemic Core (Gamma Rhythm)
        self.fast_drift_net = nn.Sequential(
            nn.Linear(hidden_dim + unified_dim + hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, h_f_prev, h_s_prev, w_t, u_t, dt=1.0):
        na = u_t[:, 4:5]
        da = u_t[:, 5:6]

        # Fast gamma clock with reactive arousal modulation
        dt_fast = torch.clamp(dt * (1.0 - 0.5 * na + 0.5 * da), 0.30, 2.00)
        # Slow theta clock (6x-8x slower integration to maintain multi-word context)
        dt_slow = dt_fast * 0.15

        sigma = 1e-3
        sqrt_dt_slow = torch.sqrt(dt_slow)
        sqrt_dt_fast = torch.sqrt(dt_fast)

        dW_slow = torch.randn_like(h_s_prev) * sqrt_dt_slow * sigma
        dW_fast = torch.randn_like(h_f_prev) * sqrt_dt_fast * sigma

        # 1. Slow Semantic State Integration (Stratonovich-Heun)
        k1_s = -self.gamma * h_s_prev + self.slow_drift_net(torch.cat([h_s_prev, u_t], dim=-1))
        h_s_pred = h_s_prev + dt_slow * k1_s + dW_slow
        k2_s = -self.gamma * h_s_pred + self.slow_drift_net(torch.cat([h_s_pred, u_t], dim=-1))
        h_s_next = torch.tanh(h_s_prev + 0.5 * dt_slow * (k1_s + k2_s) + dW_slow)

        # 2. Fast Phonemic State Integration (Conditioned on Slow Semantic Anchor)
        w_t_mod = w_t * (1.0 + 0.3 * na)
        k1_f = -self.gamma * h_f_prev + self.fast_drift_net(torch.cat([h_f_prev, w_t_mod, h_s_next], dim=-1))
        h_f_pred = h_f_prev + dt_fast * k1_f + dW_fast
        k2_f = -self.gamma * h_f_pred + self.fast_drift_net(torch.cat([h_f_pred, w_t_mod, h_s_next], dim=-1))
        h_f_next = torch.tanh(h_f_prev + 0.5 * dt_fast * (k1_f + k2_f) + dW_fast)

        return h_f_next, h_s_next, dt_fast


# =============================================================================
# BENCHMARK AGENTS
# =============================================================================

class BaselineAgent(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.net.hidden_dim
        self.unified_dim = config.net.unified_dim
        self.text_dim = config.net.text_dim
        self.text_gen_dim = config.net.text_gen_dim
        self.inv_sqrt_text_dim = 1.0 / math.sqrt(self.text_dim)

        self.pos_embeddings = OffsetPositionalByteEmbedding(self.text_gen_dim, self.text_dim)
        self.gateway = SensoryGateway(self.unified_dim, self.hidden_dim, config.net.homeo_dim,
                                      self.text_dim, config.net.vision_dim, config.net.action_dim, device_str)
        self.core = DynamicRecurrentCore(self.hidden_dim, self.unified_dim, config.net.homeo_dim,
                                         config.sde.gamma_drift, device_str)
        self.attractor_head = DesaturatedHopfieldAttractorHead(self.hidden_dim, self.text_gen_dim)
        self.motor_text_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.text_dim),
            nn.SiLU(),
            nn.LayerNorm(self.text_dim)
        )

    def forward_step(self, t_emb, h_f, h_s, u_t):
        batch_size = t_emb.size(0)
        obs_vis = torch.zeros(batch_size, self.config.net.vision_dim, device=device)
        prev_act = torch.zeros(batch_size, self.config.net.action_dim, device=device)

        w_t, _, _, _ = self.gateway(t_emb, obs_vis, prev_act, h_f, u_t)
        core_out = self.core(h_f, h_s, w_t, u_t, 1.0)
        h_f_next, h_s_next = core_out[0], core_out[1]

        h_integrated = h_f_next + h_s_next
        h_relaxed, _ = self.attractor_head.relax_to_minima(h_integrated)

        h_proj = self.motor_text_proj(h_relaxed)
        logits = F.linear(h_proj, self.pos_embeddings.byte_embed.weight) * self.inv_sqrt_text_dim
        return h_f_next, h_s_next, logits


class ProposedMultiScaleAgent(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.net.hidden_dim
        self.unified_dim = config.net.unified_dim
        self.text_dim = config.net.text_dim
        self.text_gen_dim = config.net.text_gen_dim
        self.inv_sqrt_text_dim = 1.0 / math.sqrt(self.text_dim)

        self.pos_embeddings = OffsetPositionalByteEmbedding(self.text_gen_dim, self.text_dim)
        self.gateway = SensoryGateway(self.unified_dim, self.hidden_dim, config.net.homeo_dim,
                                      self.text_dim, config.net.vision_dim, config.net.action_dim, device_str)
        self.core = MultiScaleTemporalSDECore(self.hidden_dim, self.unified_dim, config.net.homeo_dim,
                                              config.sde.gamma_drift)
        self.attractor_head = DesaturatedHopfieldAttractorHead(self.hidden_dim, self.text_gen_dim)
        self.motor_text_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.text_dim),
            nn.SiLU(),
            nn.LayerNorm(self.text_dim)
        )

    def forward_step(self, t_emb, h_f, h_s, u_t):
        batch_size = t_emb.size(0)
        obs_vis = torch.zeros(batch_size, self.config.net.vision_dim, device=device)
        prev_act = torch.zeros(batch_size, self.config.net.action_dim, device=device)

        w_t, _, _, _ = self.gateway(t_emb, obs_vis, prev_act, h_f, u_t)
        h_f_next, h_s_next, eff_dt = self.core(h_f, h_s, w_t, u_t, 1.0)

        h_integrated = h_f_next + h_s_next
        h_relaxed, _ = self.attractor_head.relax_to_minima(h_integrated)

        h_proj = self.motor_text_proj(h_relaxed)
        logits = F.linear(h_proj, self.pos_embeddings.byte_embed.weight) * self.inv_sqrt_text_dim
        return h_f_next, h_s_next, logits


# =============================================================================
# BENCHMARK EXECUTION SUITE
# =============================================================================

def run_isolated_benchmark():
    print(f"\n{'='*85}")
    print(f" === RUNNING KEP HYPOTHESIS #15 BENCHMARK ON: {device_str.upper()} ===")
    print(f"{'='*85}\n")

    config = CoREConfig()
    config.net.hidden_dim = 512
    config.net.unified_dim = 256
    config.net.text_dim = 128
    config.net.text_gen_dim = 258

    batch_size = 32
    seq_len = 512
    chunk_size = 32
    num_eval_steps = 25

    criterion = nn.CrossEntropyLoss(ignore_index=256)
    tokenizer = ByteTokenizer()

    # Structured multi-word natural text
    sample_text = (
        "User: What is the primary source of energy for Earth?\n"
        "Karyon: The primary source of energy for Earth is the Sun, which radiates light and heat "
        "driving biological photosynthesis and planetary climate systems across continuous time."
    )
    tokens_raw = tokenizer.encode(sample_text)
    repeats = ((seq_len + 1) // len(tokens_raw)) + 2
    full_tokens = (tokens_raw * repeats)[:seq_len + 1]
    
    input_tokens = torch.tensor([full_tokens[:seq_len]], dtype=torch.long, device=device).repeat(batch_size, 1)
    target_tokens = torch.tensor([full_tokens[1:seq_len + 1]], dtype=torch.long, device=device).repeat(batch_size, 1)

    # -------------------------------------------------------------------------
    # TEST 1: BASELINE (Uniform Rate SDE + No Causal Receptive Field)
    # -------------------------------------------------------------------------
    print("[1/2] Evaluating BASELINE (Single-Scale SDE without Receptive Shunt)...")
    baseline_model = BaselineAgent(config).to(device)
    base_opt = torch.optim.Adam(baseline_model.parameters(), lr=3e-3)
    hu_base = HomeostaticUnit(batch_size=batch_size, device=device_str)

    base_losses = []
    t0_base = time.perf_counter()

    for step in range(num_eval_steps):
        base_opt.zero_grad()
        h_f = torch.zeros(batch_size, config.net.hidden_dim, device=device)
        h_s = torch.zeros(batch_size, config.net.hidden_dim, device=device)
        u_t = hu_base.state.clone().detach()

        num_chunks = seq_len // chunk_size
        batch_loss_tracker = []

        for chunk_idx in range(num_chunks):
            c_start = chunk_idx * chunk_size
            c_end = min((chunk_idx + 1) * chunk_size, seq_len)

            # Baseline embeds without causal receptive field
            chunk_emb = baseline_model.pos_embeddings(input_tokens[:, c_start:c_end], start_pos=c_start, apply_receptive_field=False)
            chunk_targets = target_tokens[:, c_start:c_end]
            curr_chunk_len = chunk_emb.size(1)

            chunk_losses = []
            for t in range(curr_chunk_len):
                h_f, h_s, logits = baseline_model.forward_step(chunk_emb[:, t], h_f, h_s, u_t)
                chunk_losses.append(criterion(logits, chunk_targets[:, t]))

            chunk_loss = torch.stack(chunk_losses).mean()
            (chunk_loss / float(num_chunks)).backward()
            batch_loss_tracker.append(chunk_loss.item())

            h_f = h_f.detach()
            h_s = h_s.detach()

        base_opt.step()
        base_losses.append(sum(batch_loss_tracker) / len(batch_loss_tracker))

    total_base_time = (time.perf_counter() - t0_base) * 1000.0 / num_eval_steps

    # -------------------------------------------------------------------------
    # TEST 2: PROPOSED (MultiScale Temporal SDE + Causal Receptive Shunt + EMA)
    # -------------------------------------------------------------------------
    print("[2/2] Evaluating PROPOSED (MultiScale SDE + Causal Receptive Field + EMA Surprisal)...")
    prop_model = ProposedMultiScaleAgent(config).to(device)
    prop_opt = torch.optim.Adam(prop_model.parameters(), lr=3e-3)
    hu_prop = HomeostaticUnit(batch_size=batch_size, device=device_str)

    prop_losses = []
    t0_prop = time.perf_counter()
    action_cost_tensor = torch.full((batch_size, 1), 0.001, device=device)
    cog_action_tensor = torch.zeros((batch_size, 1), dtype=torch.int64, device=device)
    ema_surprise = 0.0
    alpha_ema = 0.05

    for step in range(num_eval_steps):
        prop_opt.zero_grad()
        h_f = torch.zeros(batch_size, config.net.hidden_dim, device=device)
        h_s = torch.zeros(batch_size, config.net.hidden_dim, device=device)
        u_t = hu_prop.state.clone().detach()

        num_chunks = seq_len // chunk_size
        batch_loss_tracker = []

        for chunk_idx in range(num_chunks):
            c_start = chunk_idx * chunk_size
            c_end = min((chunk_idx + 1) * chunk_size, seq_len)

            # Proposed embeds WITH Causal N-Gram Receptive Field (K=4)
            chunk_emb = prop_model.pos_embeddings(input_tokens[:, c_start:c_end], start_pos=c_start, apply_receptive_field=True)
            chunk_targets = target_tokens[:, c_start:c_end]
            curr_chunk_len = chunk_emb.size(1)

            chunk_losses = []
            for t in range(curr_chunk_len):
                h_f, h_s, logits = prop_model.forward_step(chunk_emb[:, t], h_f, h_s, u_t)
                loss_t = criterion(logits, chunk_targets[:, t])
                chunk_losses.append(loss_t)

                # Low-Pass EMA Filtered Somatic Surprisal
                with torch.no_grad():
                    curr_loss_val = loss_t.detach().item()
                    ema_surprise = (1.0 - alpha_ema) * ema_surprise + alpha_ema * (curr_loss_val / 4.0)
                    somatic_surprise = torch.clamp(torch.tensor([[ema_surprise]], device=device), 0.0, 0.40).repeat(batch_size, 1)
                    zero_entropy = torch.zeros((batch_size, 1), device=device)
                    u_t = hu_prop.update(action_cost_tensor, somatic_surprise, zero_entropy, cog_action_tensor).detach()

            chunk_loss = torch.stack(chunk_losses).mean()
            (chunk_loss / float(num_chunks)).backward()
            batch_loss_tracker.append(chunk_loss.item())

            h_f = h_f.detach()
            h_s = h_s.detach()

        prop_opt.step()
        prop_losses.append(sum(batch_loss_tracker) / len(batch_loss_tracker))

    total_prop_time = (time.perf_counter() - t0_prop) * 1000.0 / num_eval_steps

    # =========================================================================
    # KEP RULE #6: PROCESS DIAGNOSTICS & TELEMETRY REPORT
    # =========================================================================
    print("\n" + "="*90)
    print(" === [KEP RULE #6 TELEMETRY BENCHMARK COMPARISON REPORT] ===")
    print("="*90)
    print(f"{'Performance Metric':<35} | {'Baseline (Single Scale)':<22} | {'Proposed (MultiScale+Conv)':<22} | {'Delta':<10}")
    print("-" * 95)
    print(f"{'Step Duration (ms)':<35} | {total_base_time:<22.2f} | {total_prop_time:<22.2f} | {total_prop_time - total_base_time:+6.1f} ms")
    print(f"{'Initial Loss (Step 1)':<35} | {base_losses[0]:<22.4f} | {prop_losses[0]:<22.4f} | {prop_losses[0] - base_losses[0]:+6.4f}")
    print(f"{'Final Loss (Step 25)':<35} | {base_losses[-1]:<22.4f} | {prop_losses[-1]:<22.4f} | {prop_losses[-1] - base_losses[-1]:+6.4f} (🔥)")
    print(f"{'Perplexity (PPL Step 25)':<35} | {math.exp(base_losses[-1]):<22.2f} | {math.exp(prop_losses[-1]):<22.2f} | {math.exp(prop_losses[-1]) - math.exp(base_losses[-1]):+6.2f} (🚀)")
    print(f"{'Somatic Noradrenaline (NA)':<35} | {hu_base.state[0, 4].item():<22.4f} | {hu_prop.state[0, 4].item():<22.4f} | {hu_prop.state[0, 4].item() - hu_base.state[0, 4].item():+6.4f} (🎯)")
    print("="*90)

    # =========================================================================
    # KEP RULE #4: DIAGNOSTIC SPEECH SAMPLING AUDIT
    # =========================================================================
    print("\n" + "="*90)
    print(" === [KEP RULE #4 DIAGNOSTIC SPEECH SAMPLING AUDIT] ===")
    print("="*90)

    def generate_sample(model, name, use_rf):
        model.eval()
        prompt = "User: What is the primary source of energy for Earth?\nKaryon:"
        p_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
        
        with torch.no_grad():
            p_emb = model.pos_embeddings(p_ids, start_pos=0, apply_receptive_field=use_rf)
            h_f = torch.zeros(1, config.net.hidden_dim, device=device)
            h_s = torch.zeros(1, config.net.hidden_dim, device=device)
            u_t = torch.tensor([[0.5, 1.0, 1.0, 1.0, 0.2, 0.0]], device=device)

            for t in range(p_emb.size(1)):
                h_f, h_s, _ = model.forward_step(p_emb[:, t], h_f, h_s, u_t)

            chars = []
            curr_tok = p_ids[:, -1:]
            for s in range(60):
                t_emb = model.pos_embeddings(curr_tok, start_pos=p_emb.size(1) + s, apply_receptive_field=False)[:, 0]
                h_f, h_s, logits = model.forward_step(t_emb, h_f, h_s, u_t)
                logits[:, 256:] = -1e9
                next_id = torch.argmax(logits, dim=-1).item()
                if next_id == 257: break
                chars.append(chr(next_id) if 32 <= next_id <= 126 else ' ')
                curr_tok = torch.tensor([[next_id]], device=device)

        print(f"[{name}] Sample -> \"{''.join(chars)}\"")

    generate_sample(baseline_model, "Baseline (Single-Scale SDE)", use_rf=False)
    generate_sample(prop_model, "Proposed (MultiScale SDE + Receptive Field)", use_rf=True)
    print("="*90 + "\n")

    if prop_losses[-1] < base_losses[-1] - 0.3 and 0.05 <= hu_prop.state[0, 4].item() <= 0.50:
        print("🟢 KEP VERDICT: POSITIVE (Hypothesis #15 Validated! Ready for merge into production).")
    elif abs(prop_losses[-1] - base_losses[-1]) <= 0.3:
        print("⚪ KEP VERDICT: NEUTRAL (Inconclusive - Requires further architectural investigation).")
    else:
        print("🔴 KEP VERDICT: REJECTED (Hypothesis degraded performance).")

if __name__ == "__main__":
    run_isolated_benchmark()
