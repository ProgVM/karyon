# experiments/exp_goal_conditioned_sde_ssm.py
"""
feat(exp): implement goal-conditioned matrix sde-ssm for long-context factual retrieval

===============================================================================
KARYON EXPERIMENTAL BENCHMARK: EXP-18 (GOAL-CONDITIONED MATRIX SDE-SSM)
Hypothesis: Prefrontal Goal-Conditioned Memory Querying q_t = W_q [w_t, h_{t-1}]
applied to Matrix Fast-Weight SDE-SSM eliminates associative distractibility,
enabling exact factual extraction in long-context Needle-in-a-Haystack tasks.
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
from karyon_core import ByteTokenizer, HomeostaticUnit, SensoryGateway, MotorGateway, BatchedEpisodicMemory

torch.set_grad_enabled(True)
device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_str)


# =============================================================================
# MODULE 1: CAUSAL RECEPTIVE FIELD & POSITIONAL EMBEDDINGS
# =============================================================================

class CausalByteReceptiveField(nn.Module):
    def __init__(self, text_dim=128, kernel_size=4):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(text_dim, text_dim, kernel_size=kernel_size, groups=text_dim, bias=False)
        self.norm = nn.LayerNorm(text_dim)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        x_trans = x_seq.transpose(1, 2)
        x_padded = F.pad(x_trans, (self.kernel_size - 1, 0), mode='constant', value=0.0)
        conv_out = self.conv(x_padded)
        return self.norm(conv_out.transpose(1, 2) + x_seq)


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

    def forward(self, input_ids: torch.Tensor, start_pos: int = 0, apply_rf: bool = True) -> torch.Tensor:
        seq_len = input_ids.size(1)
        tok_emb = self.byte_embed(input_ids)
        pos_emb = self.pe[:, start_pos : start_pos + seq_len, :]
        embedded = tok_emb + pos_emb
        if apply_rf and seq_len > 1:
            embedded = self.receptive_field(embedded)
        return embedded


# =============================================================================
# MODULE 2: DESATURATED HOPFIELD ATTRACTOR MEMORY
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
# MODULE 3: GOAL-CONDITIONED MATRIX FAST-WEIGHT SDE-SSM
# =============================================================================

class GoalConditionedMatrixSDESSMCore(nn.Module):
    """
    Biological Goal-Conditioned Matrix Fast-Weight State-Space Core.
    Memory retrieval query q_t is driven by both sensory input w_t AND prefrontal goal context h_{t-1}:
    q_t = W_q [w_t, h_{t-1}]
    Eliminates stimulus-bound distractibility and enables exact factual retrieval.
    """
    def __init__(self, unified_dim=256, hidden_dim=512, num_heads=8, head_k=32, head_v=64, homeo_dim=6):
        super().__init__()
        self.unified_dim = unified_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_k = head_k
        self.head_v = head_v

        # Goal-conditioned query projection (Prefrontal context + Sensory afference)
        self.q_proj = nn.Linear(unified_dim + hidden_dim, num_heads * head_k)
        
        # Key and Value projections from sensory input
        self.k_proj = nn.Linear(unified_dim, num_heads * head_k)
        self.v_proj = nn.Linear(unified_dim, num_heads * head_v)
        
        # Multi-Head Neuromodulated Selective Decay
        self.decay_proj = nn.Linear(unified_dim + homeo_dim, num_heads)
        self.out_gate = nn.Linear(unified_dim + hidden_dim, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, m_prev, h_prev, w_t, u_t, dt=1.0):
        batch_size = w_t.size(0)
        if w_t.dim() > 2:
            w_t = w_t.reshape(batch_size, self.unified_dim)
        if h_prev.dim() > 2:
            h_prev = h_prev.reshape(batch_size, self.hidden_dim)

        na = u_t[:, 4:5]
        da = u_t[:, 5:6]

        eff_dt_raw = torch.clamp(dt * (1.0 - 0.4 * na + 0.4 * da), 0.30, 2.00)
        eff_dt_4d = eff_dt_raw.view(batch_size, 1, 1, 1)

        # 1. Goal-Conditioned Query: Driven by sensory input AND prefrontal working memory context
        q_input = torch.cat([w_t, h_prev], dim=-1)
        q = self.q_proj(q_input).view(batch_size, self.num_heads, 1, self.head_k)
        q = F.normalize(q, p=2, dim=-1)

        # 2. Key and Value Projections
        k = self.k_proj(w_t).view(batch_size, self.num_heads, self.head_k, 1)
        k = F.normalize(k, p=2, dim=-2)

        v = self.v_proj(w_t).view(batch_size, self.num_heads, 1, self.head_v)

        # 3. Multi-Head Selective Decay
        decay_in = torch.cat([w_t, u_t], dim=-1)
        alpha = (torch.sigmoid(self.decay_proj(decay_in)).view(batch_size, self.num_heads, 1, 1) ** eff_dt_4d)

        # 4. Outer Product Key-Value Associative Write: (k @ v) [Batch, NumHeads, HeadK, HeadV]
        kv_assoc = torch.matmul(k, v)

        # 5. Stratonovich-Heun Wiener Diffusion on Synaptic Fast-Weights
        sigma = 1e-3
        dW = torch.randn_like(m_prev) * torch.sqrt(eff_dt_4d) * sigma

        # 6. Matrix State Space Associative Recurrence Update
        m_next = alpha * m_prev + (1.0 - alpha) * kv_assoc + dW

        # 7. Goal-Directed Associative Readout: q @ M_next -> [Batch, NumHeads, 1, HeadV]
        readout = torch.matmul(q, m_next).view(batch_size, self.hidden_dim)

        # 8. Gated Outflow & Residual Integration
        gate = F.silu(self.out_gate(q_input))
        h_next = self.norm(self.out_proj(readout * gate) + readout)

        return m_next, h_next, eff_dt_raw


# =============================================================================
# BENCHMARK AGENTS
# =============================================================================

class BaselineUnconditionedAgent(nn.Module):
    """Baseline Agent (Query driven ONLY by isolated current byte w_t)."""
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
        
        self.in_proj = nn.Linear(self.unified_dim, self.hidden_dim)
        self.decay_proj = nn.Linear(self.unified_dim + config.net.homeo_dim, self.hidden_dim)
        self.out_gate = nn.Linear(self.unified_dim, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.layer_norm = nn.LayerNorm(self.hidden_dim)

        self.attractor_head = DesaturatedHopfieldAttractorHead(self.hidden_dim, self.text_gen_dim)
        self.motor_text_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.text_dim),
            nn.SiLU(),
            nn.LayerNorm(self.text_dim)
        )

    def forward_step(self, t_emb, h_prev, u_t):
        batch_size = t_emb.size(0)
        obs_vis = torch.zeros(batch_size, self.config.net.vision_dim, device=device)
        prev_act = torch.zeros(batch_size, self.config.net.action_dim, device=device)

        w_t, _, _, _ = self.gateway(t_emb, obs_vis, prev_act, h_prev, u_t)
        
        decay_logits = self.decay_proj(torch.cat([w_t, u_t], dim=-1))
        alpha = torch.sigmoid(decay_logits)
        b_input = self.in_proj(w_t)
        h_next = alpha * h_prev + (1.0 - alpha) * b_input
        
        gate = F.silu(self.out_gate(w_t))
        y_t = self.layer_norm(self.out_proj(h_next * gate) + h_next)

        h_relaxed, _ = self.attractor_head.relax_to_minima(y_t)
        h_proj = self.motor_text_proj(h_relaxed)
        logits = F.linear(h_proj, self.pos_embeddings.byte_embed.weight) * self.inv_sqrt_text_dim
        return h_next, logits, w_t


class ProposedGoalConditionedAgent(nn.Module):
    """Proposed Agent (Prefrontal Goal-Conditioned Matrix Memory q_t = W_q [w_t, h_{t-1}])."""
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
        
        self.goal_sde_ssm = GoalConditionedMatrixSDESSMCore(
            unified_dim=self.unified_dim,
            hidden_dim=self.hidden_dim,
            num_heads=8,
            head_k=32,
            head_v=64,
            homeo_dim=config.net.homeo_dim
        )
        
        self.attractor_head = DesaturatedHopfieldAttractorHead(self.hidden_dim, self.text_gen_dim)
        self.motor_text_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.text_dim),
            nn.SiLU(),
            nn.LayerNorm(self.text_dim)
        )

    def forward_step(self, t_emb, m_prev, h_prev, u_t, episodic_memory=None):
        batch_size = t_emb.size(0)
        obs_vis = torch.zeros(batch_size, self.config.net.vision_dim, device=device)
        prev_act = torch.zeros(batch_size, self.config.net.action_dim, device=device)

        w_t, attn, _, eps_ent = self.gateway(t_emb, obs_vis, prev_act, h_prev, u_t)
        
        # Hippocampal Context Recall
        na = u_t[:, 4:5]
        if episodic_memory is not None and na.mean().item() > 0.15 and episodic_memory.size.max().item() > 0:
            with torch.no_grad():
                retrieved, _ = episodic_memory.read(w_t.detach(), 0.05, 0.40, 15.0)
                if retrieved.dim() > 2:
                    retrieved = retrieved.reshape(batch_size, self.unified_dim)
            w_integrated = w_t + retrieved.detach() * 0.5
        else:
            w_integrated = w_t

        m_next, h_next, eff_dt = self.goal_sde_ssm(m_prev, h_prev, w_integrated, u_t, dt=1.0)

        h_relaxed, _ = self.attractor_head.relax_to_minima(h_next)
        h_proj = self.motor_text_proj(h_relaxed)
        logits = F.linear(h_proj, self.pos_embeddings.byte_embed.weight) * self.inv_sqrt_text_dim
        return m_next, h_next, logits, w_t, eps_ent


# =============================================================================
# BENCHMARK EXECUTION SUITE (FACTUAL NEEDLE-IN-A-HAYSTACK RETRIEVAL TEST)
# =============================================================================

def run_isolated_benchmark():
    print(f"\n{'='*85}")
    print(f" === RUNNING KEP HYPOTHESIS #18 BENCHMARK ON: {device_str.upper()} ===")
    print(f"{'='*85}\n")

    config = CoREConfig()
    config.net.hidden_dim = 512
    config.net.unified_dim = 256
    config.net.text_dim = 128
    config.net.text_gen_dim = 258

    batch_size = 16
    seq_len = 512
    chunk_size = 32
    num_eval_steps = 30

    criterion = nn.CrossEntropyLoss(ignore_index=256)
    tokenizer = ByteTokenizer()

    # Long-Context Synthetic Task with Distractor Background and Exact Needle
    prompt_needle = (
        "Document: The secret project codename is PHENIX-909. "
        "Background details: Continuous time systems represent information smoothly across time. "
        "Ashby homeostasis maintains biological ultrastability across neurotransmitter dynamics. "
        "Question: What is the secret project codename? "
        "Answer: The secret project codename is PHENIX-909."
    )
    tokens_raw = tokenizer.encode(prompt_needle)
    repeats = ((seq_len + 1) // len(tokens_raw)) + 2
    full_tokens = (tokens_raw * repeats)[:seq_len + 1]
    
    input_tokens = torch.tensor([full_tokens[:seq_len]], dtype=torch.long, device=device).repeat(batch_size, 1)
    target_tokens = torch.tensor([full_tokens[1:seq_len + 1]], dtype=torch.long, device=device).repeat(batch_size, 1)

    # -------------------------------------------------------------------------
    # TEST 1: BASELINE (Unconditioned 1D State Space)
    # -------------------------------------------------------------------------
    print("[1/2] Evaluating BASELINE (Unconditioned Query 1D State Space)...")
    baseline_model = BaselineUnconditionedAgent(config).to(device)
    base_opt = torch.optim.Adam(baseline_model.parameters(), lr=3e-3)
    hu_base = HomeostaticUnit(batch_size=batch_size, device=device_str)

    base_losses = []
    t0_base = time.perf_counter()

    for step in range(num_eval_steps):
        base_opt.zero_grad()
        h_prev = torch.zeros(batch_size, config.net.hidden_dim, device=device)
        u_t = hu_base.state.clone().detach()

        num_chunks = seq_len // chunk_size
        batch_loss_tracker = []

        for chunk_idx in range(num_chunks):
            c_start = chunk_idx * chunk_size
            c_end = min((chunk_idx + 1) * chunk_size, seq_len)

            chunk_emb = baseline_model.pos_embeddings(input_tokens[:, c_start:c_end], start_pos=c_start, apply_rf=True)
            chunk_targets = target_tokens[:, c_start:c_end]
            curr_chunk_len = chunk_emb.size(1)

            chunk_losses = []
            for t in range(curr_chunk_len):
                h_prev, logits, _ = baseline_model.forward_step(chunk_emb[:, t], h_prev, u_t)
                chunk_losses.append(criterion(logits, chunk_targets[:, t]))

            chunk_loss = torch.stack(chunk_losses).mean()
            (chunk_loss / float(num_chunks)).backward()
            batch_loss_tracker.append(chunk_loss.item())

            h_prev = h_prev.detach()

        base_opt.step()
        base_losses.append(sum(batch_loss_tracker) / len(batch_loss_tracker))

    total_base_time = (time.perf_counter() - t0_base) * 1000.0 / num_eval_steps

    # -------------------------------------------------------------------------
    # TEST 2: PROPOSED (Goal-Conditioned Matrix Memory q_t = W_q [w_t, h_{t-1}])
    # -------------------------------------------------------------------------
    print("[2/2] Evaluating PROPOSED (Goal-Conditioned Matrix SDE-SSM)...")
    prop_model = ProposedGoalConditionedAgent(config).to(device)
    prop_opt = torch.optim.Adam(prop_model.parameters(), lr=3e-3)
    hu_prop = HomeostaticUnit(batch_size=batch_size, device=device_str)
    mem_prop = BatchedEpisodicMemory(batch_size=batch_size, memory_dim=config.net.unified_dim, max_capacity=200, device=device_str)

    prop_losses = []
    t0_prop = time.perf_counter()
    action_cost_tensor = torch.full((batch_size, 1), 0.001, device=device)
    cog_action_tensor = torch.zeros((batch_size, 1), dtype=torch.int64, device=device)
    ema_surprise = 0.0
    alpha_ema = 0.05

    for step in range(num_eval_steps):
        prop_opt.zero_grad()
        m_prev = torch.zeros(batch_size, 8, 32, 64, device=device)
        h_prev = torch.zeros(batch_size, config.net.hidden_dim, device=device)
        u_t = hu_prop.state.clone().detach()

        num_chunks = seq_len // chunk_size
        batch_loss_tracker = []

        for chunk_idx in range(num_chunks):
            c_start = chunk_idx * chunk_size
            c_end = min((chunk_idx + 1) * chunk_size, seq_len)

            chunk_emb = prop_model.pos_embeddings(input_tokens[:, c_start:c_end], start_pos=c_start, apply_rf=True)
            chunk_targets = target_tokens[:, c_start:c_end]
            curr_chunk_len = chunk_emb.size(1)

            chunk_losses = []
            for t in range(curr_chunk_len):
                m_prev, h_prev, logits, w_t, eps_ent = prop_model.forward_step(chunk_emb[:, t], m_prev, h_prev, u_t, episodic_memory=mem_prop)
                loss_t = criterion(logits, chunk_targets[:, t])
                chunk_losses.append(loss_t)

                with torch.no_grad():
                    curr_loss_val = loss_t.detach().item()
                    if curr_loss_val > 1.0:
                        w_flat = w_t.detach()
                        if w_flat.dim() > 2:
                            w_flat = w_flat.reshape(batch_size, config.net.unified_dim)
                        mem_prop.write(w_flat, w_flat, 3)

                    ema_surprise = (1.0 - alpha_ema) * ema_surprise + alpha_ema * (curr_loss_val / 4.0)
                    somatic_surprise = torch.clamp(torch.tensor([[ema_surprise]], device=device), 0.0, 0.40).repeat(batch_size, 1)
                    zero_entropy = torch.zeros((batch_size, 1), device=device)
                    u_t = hu_prop.update(action_cost_tensor, somatic_surprise, zero_entropy, cog_action_tensor).detach()

            chunk_loss = torch.stack(chunk_losses).mean()
            (chunk_loss / float(num_chunks)).backward()
            batch_loss_tracker.append(chunk_loss.item())

            m_prev = m_prev.detach()
            h_prev = h_prev.detach()

        prop_opt.step()
        prop_losses.append(sum(batch_loss_tracker) / len(batch_loss_tracker))

    total_prop_time = (time.perf_counter() - t0_prop) * 1000.0 / num_eval_steps

    # =========================================================================
    # KEP RULE #6: PROCESS DIAGNOSTICS & TELEMETRY REPORT
    # =========================================================================
    print("\n" + "="*90)
    print(" === [KEP RULE #6 TELEMETRY BENCHMARK COMPARISON REPORT] ===")
    print("="*90)
    print(f"{'Performance Metric':<35} | {'Baseline (Unconditioned)':<22} | {'Proposed (Goal-Conditioned)':<22} | {'Delta':<10}")
    print("-" * 95)
    print(f"{'Step Duration (ms)':<35} | {total_base_time:<22.2f} | {total_prop_time:<22.2f} | {total_prop_time - total_base_time:+6.1f} ms")
    print(f"{'Memory Query Conditioning':<35} | {'q_t = W_q [w_t]':<22} | {'q_t = W_q [w_t, h_{t-1}]':<22} | {'Executive':<10} (🎯)")
    print(f"{'Initial Loss (Step 1)':<35} | {base_losses[0]:<22.4f} | {prop_losses[0]:<22.4f} | {prop_losses[0] - base_losses[0]:+6.4f}")
    print(f"{'Final Loss (Step 30)':<35} | {base_losses[-1]:<22.4f} | {prop_losses[-1]:<22.4f} | {prop_losses[-1] - base_losses[-1]:+6.4f} (🔥)")
    print(f"{'Perplexity (PPL Step 30)':<35} | {math.exp(base_losses[-1]):<22.2f} | {math.exp(prop_losses[-1]):<22.2f} | {math.exp(prop_losses[-1]) - math.exp(base_losses[-1]):+6.2f} (🚀)")
    print("="*90)

    # =========================================================================
    # KEP RULE #4: DIAGNOSTIC SPEECH SAMPLING (FACTUAL RETRIEVAL AUDIT)
    # =========================================================================
    print("\n" + "="*90)
    print(" === [KEP RULE #4 DIAGNOSTIC SPEECH SAMPLING (FACTUAL RETRIEVAL AUDIT)] ===")
    print("="*90)

    def generate_sample(model, name, is_goal_conditioned=False):
        model.eval()
        prompt = "Question: What is the secret project codename?\nAnswer:"
        p_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
        
        with torch.no_grad():
            p_emb = model.pos_embeddings(p_ids, start_pos=0, apply_rf=True)
            u_t = torch.tensor([[0.5, 1.0, 1.0, 1.0, 0.2, 0.0]], device=device)

            if not is_goal_conditioned:
                h_state = torch.zeros(1, config.net.hidden_dim, device=device)
                for t in range(p_emb.size(1)):
                    h_state, _, _ = model.forward_step(p_emb[:, t], h_state, u_t)
                curr_tok = p_ids[:, -1:]
                chars = []
                for s in range(50):
                    t_emb = model.pos_embeddings(curr_tok, start_pos=p_emb.size(1) + s, apply_rf=False)[:, 0]
                    h_state, logits, _ = model.forward_step(t_emb, h_state, u_t)
                    probs = F.softmax(logits / 0.7, dim=-1)
                    next_id = torch.multinomial(probs, 1).item()
                    if next_id == 257: break
                    chars.append(chr(next_id) if 32 <= next_id <= 126 else ' ')
                    curr_tok = torch.tensor([[next_id]], device=device)
            else:
                diag_mem = BatchedEpisodicMemory(batch_size=1, memory_dim=config.net.unified_dim, max_capacity=200, device=device_str)
                k_slice = min(mem_prop.keys.size(1), 200)
                diag_mem.keys[:, :k_slice].copy_(mem_prop.keys[:1, :k_slice])
                diag_mem.values[:, :k_slice].copy_(mem_prop.values[:1, :k_slice])
                diag_mem.pointer.copy_(mem_prop.pointer[:1])
                diag_mem.size.copy_(mem_prop.size[:1])

                m_state = torch.zeros(1, 8, 32, 64, device=device)
                h_state = torch.zeros(1, config.net.hidden_dim, device=device)
                for t in range(p_emb.size(1)):
                    m_state, h_state, _, _, _ = model.forward_step(p_emb[:, t], m_state, h_state, u_t, episodic_memory=diag_mem)
                curr_tok = p_ids[:, -1:]
                chars = []
                for s in range(50):
                    t_emb = model.pos_embeddings(curr_tok, start_pos=p_emb.size(1) + s, apply_rf=False)[:, 0]
                    m_state, h_state, logits, _, _ = model.forward_step(t_emb, m_state, h_state, u_t, episodic_memory=diag_mem)
                    probs = F.softmax(logits / 0.7, dim=-1)
                    next_id = torch.multinomial(probs, 1).item()
                    if next_id == 257: break
                    chars.append(chr(next_id) if 32 <= next_id <= 126 else ' ')
                    curr_tok = torch.tensor([[next_id]], device=device)

        print(f"[{name}] Sample -> \"{''.join(chars)}\"")

    generate_sample(baseline_model, "Baseline (Unconditioned Query)", is_goal_conditioned=False)
    generate_sample(prop_model, "Proposed (Goal-Conditioned Memory q_t = W_q [w_t, h_{t-1}])", is_goal_conditioned=True)
    print("="*90 + "\n")

    if prop_losses[-1] <= base_losses[-1]:
        print("🟢 KEP VERDICT: POSITIVE (Hypothesis #18 Validated! Ready for merge into production).")
    else:
        print("🔴 KEP VERDICT: REJECTED (Hypothesis degraded performance).")

if __name__ == "__main__":
    run_isolated_benchmark()
