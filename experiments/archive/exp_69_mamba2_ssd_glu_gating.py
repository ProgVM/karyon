# experiments/exp_69_mamba2_ssd_glu_gating.py
"""
===============================================================================
KARYON EXPERIMENTAL BENCHMARK: EXP-69 (MAMBA-2 SSD SWIGLU OUTPUT GATING)
Hypothesis: Adding an elementwise multiplicative gate z = SiLU(W_z X) to the 
State-Space Duality readout (y_normed * z) inside ParallelLogDecaySSDLayer
supplies content-dependent non-linear channel gating directly in time-mixing,
further accelerating convergence and breaking below 1.80 nats/byte.
Author: Bazilevs (ProgVM member) & Karyon-CoRE Research Team (2026)
===============================================================================
"""

import sys
import os
import time
import math
import types
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset

# Dynamo Hotfix for Python 3.12 / Kaggle GPU
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

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import karyon_config, karyon_core, karyon_logger
from karyon_config import CoREConfig
from karyon_agent import OffsetPositionalByteEmbedding
from karyon_core import ByteTokenizer, HomeostaticUnit, BatchedEpisodicMemory, ParallelSwiGLUBlock

logger = karyon_logger.get_logger()
torch.set_grad_enabled(True)

device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_str)
use_amp = (device_str == 'cuda')


# =============================================================================
# 0. ROTARY POSITION EMBEDDING (RoPE) UTILITIES
# =============================================================================
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# =============================================================================
# 1. MAMBA-2 SSD LAYER WITH ELEMENTWISE SILU OUTPUT GATING & ROPE
# =============================================================================
class Mamba2GatedSSDLayerWithRoPE(nn.Module):
    def __init__(self, in_dim: int = 768, out_dim: int = 768, num_heads: int = 12,
                 head_k: int = 64, head_v: int = 128, min_beta: float = 0.0005, max_beta: float = 0.08):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_k = head_k
        self.head_v = head_v
        self.inv_sqrt_k = 1.0 / math.sqrt(head_k)

        self.q_proj = nn.Linear(in_dim, num_heads * head_k)
        self.k_proj = nn.Linear(in_dim, num_heads * head_k)
        self.v_proj = nn.Linear(in_dim, num_heads * head_v)
        self.z_proj = nn.Linear(in_dim, num_heads * head_v) # Mamba-2 GLU Gating Branch
        self.delta_proj = nn.Linear(in_dim, num_heads)

        betas = torch.exp(torch.linspace(math.log(max_beta), math.log(min_beta), num_heads))
        alphas = 1.0 - betas
        logit_init = torch.log(alphas / (1.0 - alphas)).view(1, num_heads, 1)
        self.decay_logits = nn.Parameter(logit_init)

        self.head_norm = nn.GroupNorm(num_groups=num_heads, num_channels=num_heads * head_v)
        self.out_proj = nn.Linear(num_heads * head_v, out_dim)
        self.norm = nn.LayerNorm(out_dim)

        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_k, 2, dtype=torch.float32) / head_k))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _get_rope_cos_sin(self, chunk_len: int, device: torch.device, dtype: torch.dtype):
        t = torch.arange(chunk_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().view(1, 1, chunk_len, self.head_k).to(dtype)
        sin = emb.sin().view(1, 1, chunk_len, self.head_k).to(dtype)
        return cos, sin

    def forward(self, x_seq: torch.Tensor, m_prev: torch.Tensor, u_t: torch.Tensor, 
                saliency_gate: torch.Tensor = None, dt: float = 1.0):
        batch_size, chunk_len, _ = x_seq.size()

        if u_t.size(0) != batch_size:
            u_t = u_t[:batch_size] if u_t.size(0) > batch_size else u_t.expand(batch_size, -1)

        curiosity = u_t.select(1, 0).view(batch_size, 1, 1, 1)
        na = u_t.select(1, 4).view(batch_size, 1, 1, 1)
        da = u_t.select(1, 5).view(batch_size, 1, 1, 1)
        eff_dt = torch.clamp(dt * (1.0 - 0.4 * na + 0.4 * da), 0.30, 2.00)

        q = (self.q_proj(x_seq).view(batch_size, chunk_len, self.num_heads, self.head_k).transpose(1, 2)) * self.inv_sqrt_k
        k = self.k_proj(x_seq).view(batch_size, chunk_len, self.num_heads, self.head_k).transpose(1, 2)
        v = self.v_proj(x_seq).view(batch_size, chunk_len, self.num_heads, self.head_v).transpose(1, 2)
        z = F.silu(self.z_proj(x_seq)).view(batch_size * chunk_len, self.num_heads * self.head_v) # GLU Gate

        cos, sin = self._get_rope_cos_sin(chunk_len, x_seq.device, x_seq.dtype)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        selective_delta = F.softplus(self.delta_proj(x_seq)).view(batch_size, chunk_len, self.num_heads).transpose(1, 2)
        base_alpha = torch.sigmoid(self.decay_logits)
        alpha = torch.pow(base_alpha, (selective_delta * eff_dt.squeeze(-1)).clamp(0.1, 10.0))

        if saliency_gate is not None:
            alpha = alpha * (1.0 - 0.80 * saliency_gate)

        alpha = torch.clamp(alpha, 1e-4, 0.9999)
        log_alpha = torch.log(alpha)
        beta = 1.0 - alpha

        lambda_t = torch.cumsum(log_alpha, dim=-1)

        log_decay_matrix = lambda_t.unsqueeze(-1) - lambda_t.unsqueeze(-2)
        decay_matrix = torch.exp(torch.clamp(log_decay_matrix, -20.0, 0.0))

        pos = torch.arange(chunk_len, device=x_seq.device)
        causal_mask = (pos.unsqueeze(1) >= pos.unsqueeze(0)).float().view(1, 1, chunk_len, chunk_len)
        decay_weights = decay_matrix * causal_mask

        s_matrix = torch.matmul(q, k.transpose(-1, -2)) * decay_weights
        y_intra = torch.matmul(s_matrix, v)

        decay_to_start = torch.exp(torch.clamp(lambda_t, -20.0, 0.0)).unsqueeze(-1)
        y_inter = torch.matmul(q.float() * decay_to_start, m_prev.float()).to(q.dtype)

        y_total = (y_intra + y_inter).transpose(1, 2).reshape(batch_size * chunk_len, self.num_heads * self.head_v)
        y_normed = self.head_norm(y_total)
        
        # Apply Mamba-2 Elementwise GLU Gating before output projection
        y_gated = y_normed * z
        h_chunk = self.norm(self.out_proj(y_gated))

        lambda_end = lambda_t[:, :, -1:].unsqueeze(-1)
        decay_to_end = torch.exp(torch.clamp(lambda_end - lambda_t.unsqueeze(-1), -20.0, 0.0))
        k_decayed = k.float() * decay_to_end * beta.unsqueeze(-1).float()
        kv_chunk_update = torch.matmul(k_decayed.transpose(-1, -2), v.float())

        alpha_chunk = torch.exp(torch.clamp(lambda_t[:, :, -1:], -20.0, 0.0)).unsqueeze(-1)
        sigma_somatic = 1e-3 * (0.8 * curiosity.float() + 0.4 * na.float() + 0.1)
        dW = torch.randn_like(m_prev) * torch.sqrt(eff_dt.float()) * sigma_somatic
        m_next = alpha_chunk * m_prev.float() + kv_chunk_update + dW

        return h_chunk, m_next, eff_dt.mean().item()


# =============================================================================
# 2. CORTICAL STAGE WITH MAMBA-2 GATED SSD
# =============================================================================
class CorticalStage(nn.Module):
    def __init__(self, hidden_dim: int = 768, expand_dim: int = 3072, num_heads: int = 12,
                 head_k: int = 64, head_v: int = 128, min_beta: float = 0.0005, max_beta: float = 0.08, device_str: str = 'cpu'):
        super().__init__()
        self.pre_norm_ssd = nn.LayerNorm(hidden_dim)
        self.ssd = Mamba2GatedSSDLayerWithRoPE(
            in_dim=hidden_dim, out_dim=hidden_dim, num_heads=num_heads,
            head_k=head_k, head_v=head_v, min_beta=min_beta, max_beta=max_beta
        )
        self.pre_norm_swiglu = nn.LayerNorm(hidden_dim)
        self.swiglu = karyon_core.ParallelSwiGLUBlock(
            hidden_dim=hidden_dim, expand_dim=expand_dim, device=device_str
        )

    def forward(self, x: torch.Tensor, m_prev: torch.Tensor, u_t: torch.Tensor, 
                saliency_gate: torch.Tensor = None, dt: float = 1.0):
        norm_x = self.pre_norm_ssd(x)
        h_ssd, m_next, eff_dt = self.ssd(norm_x, m_prev, u_t, saliency_gate=saliency_gate, dt=dt)
        x_res1 = x + h_ssd.view_as(x)

        norm_res1 = self.pre_norm_swiglu(x_res1)
        h_swiglu = self.swiglu(norm_res1.contiguous().view(-1, x.size(-1)))
        x_out = x_res1 + h_swiglu.view_as(x_res1)

        return x_out, m_next, eff_dt


# =============================================================================
# 3. ENTROPY-ADAPTIVE BOUNDARY DETECTOR
# =============================================================================
class EntropyAdaptiveBoundaryDetector(nn.Module):
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        self.boundary_gate_net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.register_buffer('boundary_bytes', torch.tensor([32, 10, 44, 46, 58, 59, 63, 33, 34, 39], dtype=torch.long))

    def forward(self, h_stage1: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        pred_boundary = self.boundary_gate_net(h_stage1)
        is_token_boundary = torch.isin(input_ids, self.boundary_bytes).float().unsqueeze(-1)
        saliency = torch.clamp(0.05 + 0.60 * pred_boundary + 0.35 * is_token_boundary, 0.0, 1.0)
        return saliency.squeeze(-1).unsqueeze(1)


# =============================================================================
# 4. EXP-69 AGENT WITH MAMBA-2 GATED SSD STACK
# =============================================================================
class Exp69Mamba2GatedAgent(nn.Module):
    def __init__(self, config: CoREConfig, device_str: str = 'cpu'):
        super().__init__()
        self.device_str = device_str
        self.device = torch.device(device_str)
        self.config = config
        self.text_dim = config.net.text_dim
        self.hidden_dim = 768
        self.expand_dim = 3072
        self.num_heads = 12
        self.head_k = 64
        self.head_v = 128
        self.text_gen_dim = 258

        self.pos_embeddings = OffsetPositionalByteEmbedding(
            vocab_size=self.text_gen_dim, text_dim=self.text_dim, max_len=8192, device_str=device_str
        ).to(self.device)
        nn.init.normal_(self.pos_embeddings.byte_embed.weight, mean=0.0, std=0.08)

        self.in_proj = nn.Linear(self.text_dim, self.hidden_dim).to(self.device)

        # Stage 1: Fast Morpho-Syntactic Sheet
        self.stage1 = CorticalStage(
            hidden_dim=self.hidden_dim, expand_dim=self.expand_dim, num_heads=self.num_heads,
            head_k=self.head_k, head_v=self.head_v, min_beta=0.005, max_beta=0.15, device_str=device_str
        ).to(self.device)

        # Boundary Detector
        self.boundary_detector = EntropyAdaptiveBoundaryDetector(hidden_dim=self.hidden_dim).to(self.device)

        # LPER Top-Down Predictor
        self.topdown_pred_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        ).to(self.device)

        # Stage 2: Slow Semantic Sheet
        self.stage2 = CorticalStage(
            hidden_dim=self.hidden_dim, expand_dim=self.expand_dim, num_heads=self.num_heads,
            head_k=self.head_k, head_v=self.head_v, min_beta=0.0001, max_beta=0.05, device_str=device_str
        ).to(self.device)

        self.attractor_head = karyon_core.DesaturatedHopfieldAttractorHead(
            hidden_dim=self.hidden_dim, vocab_size=self.text_gen_dim,
            num_attractors=256, device=device_str
        )

        self.motor_text_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.text_dim),
            nn.SiLU(),
            nn.LayerNorm(self.text_dim)
        ).to(self.device)

    def forward_sequence(self, input_seq: torch.Tensor, target_seq: torch.Tensor, hu_batch, criterion, chunk_size: int = 64):
        batch_size, seq_len = input_seq.size()
        m_s1 = torch.zeros(batch_size, self.num_heads, self.head_k, self.head_v, dtype=torch.float32, device=self.device)
        m_s2 = torch.zeros(batch_size, self.num_heads, self.head_k, self.head_v, dtype=torch.float32, device=self.device)
        curr_u_t = hu_batch.state.clone().detach()

        h1_prev_last = torch.zeros(batch_size, 1, self.hidden_dim, device=self.device)

        num_chunks = max(1, seq_len // chunk_size)
        chunk_losses = []
        commit_losses = []
        eff_dts = []

        da_level = curr_u_t[:, 5:6]
        motor_gain = (1.0 + 1.0 * da_level).unsqueeze(1)

        for chunk_idx in range(num_chunks):
            c_start = chunk_idx * chunk_size
            c_end = min((chunk_idx + 1) * chunk_size, seq_len)
            c_len = c_end - c_start
            chunk_in = input_seq[:, c_start:c_end]
            chunk_tgt = target_seq[:, c_start:c_end]

            chunk_emb = self.pos_embeddings(chunk_in, start_pos=c_start, apply_rf=True)
            h_in = self.in_proj(chunk_emb)

            # Stage 1: Fast Morpho-Syntactic Pass
            h_s1, m_s1, dt1 = self.stage1(h_in, m_s1, curr_u_t, dt=1.0)

            # Detect Word/Morpheme Boundary Saliency
            saliency_gate = self.boundary_detector(h_s1, chunk_in)

            # --- LPER Error Routing ---
            h1_shifted = torch.cat([h1_prev_last, h_s1[:, :-1, :]], dim=1)
            e1 = h_s1 - self.topdown_pred_net(h1_shifted)
            h1_prev_last = h_s1[:, -1:, :].detach()

            # Stage 2: Slow Semantic Pass on Error e1
            h_s2, m_s2, dt2 = self.stage2(e1, m_s2, curr_u_t, saliency_gate=saliency_gate, dt=1.0)
            eff_dts.append((dt1 + dt2) / 2.0)

            # Combined Laminar Representation
            h_combined = h_s1 + h_s2

            h_flat = h_combined.contiguous().view(-1, self.hidden_dim)
            h_relaxed, commit_loss = self.attractor_head.relax_to_minima(h_flat, curr_u_t)

            h_proj = self.motor_text_proj(h_relaxed).view(batch_size, c_len, self.text_dim)
            h_proj_gain = (h_proj * motor_gain).contiguous().view(-1, self.text_dim)

            logits_flat = F.linear(h_proj_gain, self.pos_embeddings.byte_embed.weight)
            targets_flat = chunk_tgt.contiguous().view(-1)

            loss = criterion(logits_flat, targets_flat)
            chunk_losses.append(loss)
            commit_losses.append(commit_loss)

            with torch.no_grad():
                has_eos = (chunk_in == 257).any(dim=-1).view(batch_size, 1, 1, 1).float()
                m_s1 = (m_s1 * (1.0 - has_eos)).detach()
                m_s2 = (m_s2 * (1.0 - has_eos)).detach()

        avg_loss = torch.stack(chunk_losses).mean()
        avg_commit = torch.stack(commit_losses).mean()
        avg_eff_dt = sum(eff_dts) / len(eff_dts)

        total_loss = avg_loss + 0.05 * avg_commit
        return total_loss, avg_loss.item(), m_s2, h_s2[:, -1, :], curr_u_t, avg_eff_dt

    def generate_natural_speech(self, prompt: str, hu_state, max_tokens: int = 75) -> str:
        self.eval()
        tokenizer = ByteTokenizer()
        prompt_ids = tokenizer.encode(prompt)
        if prompt_ids and prompt_ids[-1] == 257:
            prompt_ids = prompt_ids[:-1]

        prompt_t = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        m_s1 = torch.zeros(1, self.num_heads, self.head_k, self.head_v, dtype=torch.float32, device=self.device)
        m_s2 = torch.zeros(1, self.num_heads, self.head_k, self.head_v, dtype=torch.float32, device=self.device)

        prompt_emb = self.pos_embeddings(prompt_t, start_pos=0, apply_rf=True)
        prompt_len = prompt_t.size(1)

        h1_prev_last = torch.zeros(1, 1, self.hidden_dim, device=self.device)

        for c_idx in range(0, prompt_len, 64):
            c_emb = prompt_emb[:, c_idx : min(c_idx + 64, prompt_len), :]
            c_in = prompt_t[:, c_idx : min(c_idx + 64, prompt_len)]
            h_in = self.in_proj(c_emb)

            h_s1, m_s1, _ = self.stage1(h_in, m_s1, hu_state, dt=1.0)
            sal_gate = self.boundary_detector(h_s1, c_in)

            h1_shifted = torch.cat([h1_prev_last, h_s1[:, :-1, :]], dim=1)
            e1 = h_s1 - self.topdown_pred_net(h1_shifted)
            h1_prev_last = h_s1[:, -1:, :].detach()

            h_s2, m_s2, _ = self.stage2(e1, m_s2, hu_state, saliency_gate=sal_gate, dt=1.0)

        rolling_ids = prompt_ids.copy()
        generated_chars = []

        with torch.no_grad():
            for step in range(max_tokens):
                context = rolling_ids[-4:]
                win_t = torch.tensor([context], dtype=torch.long, device=self.device)
                win_emb = self.pos_embeddings(win_t, start_pos=len(rolling_ids) - len(context), apply_rf=True)
                t_emb = win_emb[:, -1:, :]

                h_in = self.in_proj(t_emb)

                h_s1, m_s1, _ = self.stage1(h_in, m_s1, hu_state, dt=1.0)
                sal_gate = self.boundary_detector(h_s1, win_t[:, -1:])

                h1_shifted = torch.cat([h1_prev_last, h_s1[:, :-1, :]], dim=1)
                e1 = h_s1 - self.topdown_pred_net(h1_shifted)
                h1_prev_last = h_s1[:, -1:, :].detach()

                h_s2, m_s2, _ = self.stage2(e1, m_s2, hu_state, saliency_gate=sal_gate, dt=1.0)
                h_combined = h_s1 + h_s2

                h_flat = h_combined.contiguous().view(-1, self.hidden_dim)
                h_rel, _ = self.attractor_head.relax_to_minima(h_flat, hu_state)
                h_proj = self.motor_text_proj(h_rel)

                raw_logits = F.linear(h_proj, self.pos_embeddings.byte_embed.weight)

                # Mask PAD & invalid control codes
                raw_logits[:, 256] = -1e9
                raw_logits[:, :9] = -1e9
                raw_logits[:, 11:13] = -1e9
                raw_logits[:, 14:32] = -1e9
                raw_logits[:, 127:256] = -1e9
                if step < 8:
                    raw_logits[:, 257] = -1e9

                p_dist = F.softmax(raw_logits, dim=-1)
                entropy = -(p_dist * torch.log(p_dist + 1e-9)).sum(dim=-1).item()
                is_boundary = (len(rolling_ids) > 0 and rolling_ids[-1] in [32, 10, 44, 46])

                if is_boundary or entropy > 0.70:
                    temp = 0.45
                    top_p = 0.88
                else:
                    temp = 0.08
                    top_p = 0.99

                logits = raw_logits / max(temp, 1e-4)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                to_remove = cumulative_probs > top_p
                to_remove[..., 1:] = to_remove[..., :-1].clone()
                to_remove[..., 0] = False
                indices_to_remove = to_remove.scatter(1, sorted_indices, to_remove)
                logits[indices_to_remove] = -1e9

                probs = F.softmax(logits, dim=-1)
                probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
                prob_sum = probs.sum(dim=-1, keepdim=True)
                if (prob_sum <= 0).any():
                    probs = torch.full_like(probs, 1.0 / 258)
                else:
                    probs = probs / prob_sum

                next_token_id = torch.multinomial(probs, num_samples=1).item()

                if next_token_id == 257:
                    break
                rolling_ids.append(next_token_id)
                char_c = chr(next_token_id) if 32 <= next_token_id <= 126 or next_token_id in [9, 10, 13] else ' '
                generated_chars.append(char_c)

        self.train()
        return "".join(generated_chars).strip()


# =============================================================================
# 5. BENCHMARK RUNNER
# =============================================================================
def prepare_packed_stream(num_batches: int = 300, batch_size: int = 32, seq_len: int = 1024):
    logger.info(f"Loading Real Dataset (vicgalle/alpaca-gpt4) for EXP-69 (S={seq_len}, Steps={num_batches})...")
    ds = load_dataset("vicgalle/alpaca-gpt4", split="train")
    tokenizer = ByteTokenizer()
    full_stream = []

    for item in ds:
        inst = item.get("instruction", "").strip()
        out = item.get("output", "").strip()
        if inst and out:
            dialog = f"User: {inst}\nKaryon: {out}"
            full_stream.extend(tokenizer.encode(dialog))
        if len(full_stream) >= num_batches * batch_size * (seq_len + 1):
            break

    batches = []
    block_size = seq_len + 1
    for b in range(num_batches):
        batch_tensors = []
        for s in range(batch_size):
            start = (b * batch_size + s) * block_size
            end = start + block_size
            chunk = full_stream[start:end]
            if len(chunk) < block_size:
                chunk = chunk + [256] * (block_size - len(chunk))
            batch_tensors.append(torch.tensor(chunk, dtype=torch.long))
        batches.append(torch.stack(batch_tensors, dim=0).to(device))

    logger.info(f"Prepared {len(batches)} Real Packed Batches (B={batch_size}, S={seq_len}, Total Tokens: {len(batches)*batch_size*seq_len/1e6:.2f}M).")
    return batches


def run_exp_69_benchmark():
    print("\n" + "="*85)
    print(" === [KEP EXPERIMENTAL BENCHMARK: EXP-69 (MAMBA-2 SSD SWIGLU OUTPUT GATING)] ===")
    print("="*85)
    print(f"Hardware Device : {device_str.upper()} | AMP FP16 Enabled: {use_amp}")

    config = CoREConfig()
    config.net.text_dim = 256
    config.net.hidden_dim = 768
    config.net.expand_dim = 3072
    config.net.num_heads = 12

    b_size, seq_len = 32, 1024
    num_eval_steps = 300
    chunk_size = 64

    batches = prepare_packed_stream(num_batches=num_eval_steps, batch_size=b_size, seq_len=seq_len)
    criterion = nn.CrossEntropyLoss(ignore_index=256)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    diag_prompt = "User: What is the primary source of energy for Earth?\nKaryon:"

    print("\n[1/1] Initializing EXP-69 Mamba-2 Gated Agent (300 Steps)...")
    torch.manual_seed(42)
    agent = Exp69Mamba2GatedAgent(config, device_str=device_str).to(device)
    hu = HomeostaticUnit(batch_size=b_size, device=device_str)

    optimizer = torch.optim.AdamW(agent.parameters(), lr=3.0e-3, weight_decay=0.01)

    warmup_steps = 30
    stable_steps = 200
    decay_steps = num_eval_steps - (warmup_steps + stable_steps)

    def wsd_schedule(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        elif step < warmup_steps + stable_steps:
            return 1.0
        else:
            p = float(step - (warmup_steps + stable_steps)) / float(max(1, decay_steps))
            return 0.33 + 0.67 * 0.5 * (1.0 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=wsd_schedule)

    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()
    losses = []

    for step in range(num_eval_steps):
        t_batch_start = time.perf_counter()
        batch = batches[step]
        input_s = batch[:, :-1]
        target_s = batch[:, 1:]

        optimizer.zero_grad()
        t_fwd_0 = time.perf_counter()
        with torch.amp.autocast(device_type=device_str, dtype=torch.float16, enabled=use_amp):
            tot_loss, speech_loss, m_curr, h_proxy, curr_u_t, eff_dt = agent.forward_sequence(
                input_s, target_s, hu, criterion, chunk_size=chunk_size
            )
        t_fwd = (time.perf_counter() - t_fwd_0) * 1000.0

        t_bwd_0 = time.perf_counter()
        scaler.scale(tot_loss).backward()
        scaler.unscale_(optimizer)
        grad_norm_total = torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=3.0).item()

        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        scale_after = scaler.get_scale()

        if scale_before <= scale_after:
            scheduler.step()
        t_bwd = (time.perf_counter() - t_bwd_0) * 1000.0

        # Somatic Coupling
        with torch.no_grad():
            cost_t = torch.full((b_size, 1), 0.001, device=device)
            err_t = torch.full((b_size, 1), float(speech_loss * 0.1), device=device)
            ent_t = torch.full((b_size, 1), 0.05, device=device)
            cog_t = torch.zeros((b_size, 1), dtype=torch.int64, device=device)
            hu.update(cost_t, err_t, ent_t, cog_t)

        losses.append(speech_loss)

        t_step_total = (time.perf_counter() - t_batch_start) * 1000.0
        tok_sec = (b_size * seq_len) / (t_step_total / 1000.0)

        # KEP PROCESS DIAGNOSTICS DASHBOARD (EVERY 15 STEPS)
        if (step + 1) % 15 == 0 or step == num_eval_steps - 1:
            cur_loss = sum(losses[-15:]) / min(len(losses), 15)
            cur_lr = optimizer.param_groups[0]['lr']
            cur_ppl = math.exp(min(cur_loss, 20.0))

            curiosity, energy, stability, health, na, da = hu.state[0].tolist()
            peak_vram = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if device.type == 'cuda' else 0.0

            g_emb = agent.pos_embeddings.byte_embed.weight.grad.norm().item() if agent.pos_embeddings.byte_embed.weight.grad is not None else 0.0
            g_s1 = 0.0
            for p in agent.stage1.parameters():
                if p.grad is not None:
                    g_s1 = max(g_s1, p.grad.norm().item())
            g_s2 = 0.0
            for p in agent.stage2.parameters():
                if p.grad is not None:
                    g_s2 = max(g_s2, p.grad.norm().item())

            print("\n" + "="*95)
            print(f" === [KEP EXP-69 PROCESS DIAGNOSTICS DASHBOARD | STEP {step+1:03d}/{num_eval_steps}] ===")
            print("="*95)
            print(f"Metrics Progress          : Speech Loss = {speech_loss:.4f} (Avg: {cur_loss:.4f}, PPL: {cur_ppl:.2f}) | LR = {cur_lr:.6f}")
            print(f"Timing & Throughput       : Forward: {t_fwd:.1f}ms | Backward: {t_bwd:.1f}ms | Total Step: {t_step_total:.1f}ms | {tok_sec:.1f} tok/s")
            print(f"Somatic State (Ashby)     : Curiosity: {curiosity:.3f} | Energy: {energy:.3f} | NA: {na:.3f} | DA: {da:.3f} | dt_eff: {eff_dt:.3f}")
            print(f"Gradient Flow Inspection  : Total: {grad_norm_total:.4f} | Emb: {g_emb:.4f} | S1: {g_s1:.4f} | S2: {g_s2:.4f}")
            print(f"Hardware Resources        : Peak VRAM: {peak_vram:.1f} MB")
            print("="*95)

        if (step + 1) % 75 == 0:
            sample_text = agent.generate_natural_speech(diag_prompt, hu.state[0:1], max_tokens=65)
            logger.info(f"💬 [Live Diagnostic Speech Sample @ Step {step+1}] -> \"{sample_text}\"")

    if device.type == 'cuda': torch.cuda.synchronize()
    total_time_sec = time.perf_counter() - t_start
    final_loss = sum(losses[-30:]) / 30.0
    final_sample = agent.generate_natural_speech(diag_prompt, hu.state[0:1], max_tokens=75)

    print("\n" + "="*95)
    print(" === [KEP EXP-69 FINAL TELEMETRY DASHBOARD] ===")
    print("="*95)
    print(f"{'Performance Metric':<36} | {'EXP-69 Mamba-2 Gated Value':<40}")
    print("-" * 95)
    print(f"{'Initial Loss (Step 1)':<36} | {losses[0]:<40.4f}")
    print(f"{'Step 100 Loss':<36} | {losses[99]:<40.4f}")
    print(f"{'Step 200 Loss':<36} | {losses[199]:<40.4f}")
    print(f"{'Final Steady-State Speech Loss':<36} | {final_loss:<40.4f} (PPL: {math.exp(final_loss):.2f})")
    print(f"{'Total Loss Drop (Delta)':<36} | {final_loss - losses[0]:<40.4f}")
    print(f"{'Throughput Speed':<36} | {(num_eval_steps * b_size * seq_len) / total_time_sec:<40.1f} tok/s")
    print(f"{'Total Training Time':<36} | {total_time_sec:<40.2f} sec ({total_time_sec/60.0:.1f} min)")
    print("="*95)

    print("\n" + "="*95)
    print(" === [KEP RULE #4 FINAL DIAGNOSTIC SPEECH SAMPLE AUDIT] ===")
    print("="*95)
    print(f"Prompt : \"{diag_prompt}\"")
    print(f"Output : \"{final_sample}\"")
    print("="*95 + "\n")


if __name__ == "__main__":
    run_exp_69_benchmark()
