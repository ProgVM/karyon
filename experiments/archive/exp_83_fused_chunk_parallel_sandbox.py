# experiments/exp_83_fused_chunk_parallel_sandbox.py
"""
===============================================================================
KARYON EXPERIMENTAL BENCHMARK: EXP-83 (FUSED CHUNK-PARALLEL + SYSTEM 2 SANDBOX)
Hypothesis: 
  1. Scaling PW-LPER parameter gradients by 0.005 eliminates the gradient norm explosion
     (176,882 -> ~10.0), restoring full gradient vitality to byte embeddings, Stage 1,
     and Stage 2 cortical sheets under global gradient clipping.
  2. Fused Chunk-Parallel SSD with full 1024-token unrolling preserves 100% untruncated
     BPTT across all chunks, dropping speech loss below 1.20 nats/byte.
  3. System 2 Active Inference Mental Sandbox / Latent Rollout counterfactually 
     simulates future latent states during high surprise (F_t > 0.20), eliminating
     pseudo-morphemic drift and producing grounded, coherent thought and speech.
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
from karyon_core import (
    ByteTokenizer, HomeostaticUnit, BatchedEpisodicMemory,
    MultiScaleBytePyramidReceptiveField, CausalConvSwiGLUBlock,
    EntropyAdaptiveBoundaryDetector, PrecisionWeightedLPER,
    DesaturatedHopfieldAttractorHead, LatentPredictor
)

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
# 1. FUSED CHUNK-PARALLEL SSD LAYER (Q=64 CHUNKS, FULL SEQUENCE BATCHED)
# =============================================================================
class FusedChunkParallelSSDLayer(nn.Module):
    def __init__(self, in_dim: int = 768, out_dim: int = 768, num_heads: int = 12,
                 head_k: int = 64, head_v: int = 128, min_beta: float = 0.0005, max_beta: float = 0.08,
                 chunk_size: int = 64):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_k = head_k
        self.head_v = head_v
        self.chunk_size = chunk_size
        self.inv_sqrt_k = 1.0 / math.sqrt(head_k)

        self.q_proj = nn.Linear(in_dim, num_heads * head_k)
        self.k_proj = nn.Linear(in_dim, num_heads * head_k)
        self.v_proj = nn.Linear(in_dim, num_heads * head_v)
        self.z_proj = nn.Linear(in_dim, num_heads * head_v)
        self.delta_proj = nn.Linear(in_dim, num_heads)

        betas = torch.exp(torch.linspace(math.log(max_beta), math.log(min_beta), num_heads))
        alphas = 1.0 - betas
        logit_init = torch.log(alphas / (1.0 - alphas)).view(1, 1, num_heads, 1)
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
        cos = emb.cos().view(1, 1, 1, chunk_len, self.head_k).to(dtype)
        sin = emb.sin().view(1, 1, 1, chunk_len, self.head_k).to(dtype)
        return cos, sin

    def forward(self, x_seq: torch.Tensor, m_prev: torch.Tensor, u_t: torch.Tensor,
                saliency_gate: torch.Tensor = None, dt: float = 1.0) -> tuple:
        batch_size, seq_len, _ = x_seq.size()
        Q = self.chunk_size
        pad_len = 0

        # Dynamically adjust chunk size for short sequences
        if seq_len < Q:
            Q = seq_len
            num_chunks = 1
        else:
            pad_len = (Q - (seq_len % Q)) % Q
            if pad_len > 0:
                x_seq = F.pad(x_seq.transpose(1, 2), (pad_len, 0), mode='constant', value=0.0).transpose(1, 2)
                if saliency_gate is not None:
                    saliency_gate = F.pad(saliency_gate, (pad_len, 0), mode='constant', value=0.0)
                seq_len = x_seq.size(1)
            num_chunks = seq_len // Q

        curiosity = u_t.select(1, 0).view(batch_size, 1, 1, 1, 1)
        na = u_t.select(1, 4).view(batch_size, 1, 1, 1, 1)
        da = u_t.select(1, 5).view(batch_size, 1, 1, 1, 1)
        eff_dt = torch.clamp(dt * (1.0 - 0.4 * na + 0.4 * da), 0.30, 2.00)

        # 1. Full-sequence batched projections
        q_full = self.q_proj(x_seq).view(batch_size, num_chunks, Q, self.num_heads, self.head_k).transpose(2, 3) * self.inv_sqrt_k
        k_full = self.k_proj(x_seq).view(batch_size, num_chunks, Q, self.num_heads, self.head_k).transpose(2, 3)
        v_full = self.v_proj(x_seq).view(batch_size, num_chunks, Q, self.num_heads, self.head_v).transpose(2, 3)
        z_full = F.silu(self.z_proj(x_seq)).view(batch_size * seq_len, self.num_heads * self.head_v)

        cos, sin = self._get_rope_cos_sin(Q, x_seq.device, x_seq.dtype)
        q_full, k_full = apply_rotary_pos_emb(q_full, k_full, cos, sin)

        delta_full = F.softplus(self.delta_proj(x_seq)).view(batch_size, num_chunks, Q, self.num_heads).permute(0, 1, 3, 2)
        base_alpha = torch.sigmoid(self.decay_logits)
        alpha = torch.pow(base_alpha, (delta_full * eff_dt.squeeze(-1)).clamp(0.1, 10.0))

        if saliency_gate is not None:
            sal_chunk = saliency_gate.view(batch_size, num_chunks, 1, Q)
            alpha = alpha * (1.0 - 0.80 * sal_chunk)

        alpha = torch.clamp(alpha, 1e-4, 0.9999)
        log_alpha = torch.log(alpha)
        beta = 1.0 - alpha

        # 2. Intra-chunk exact log-space decay scan
        lambda_t = torch.cumsum(log_alpha, dim=-1)
        log_decay_matrix = lambda_t.unsqueeze(-1) - lambda_t.unsqueeze(-2)
        decay_matrix = torch.exp(torch.clamp(log_decay_matrix, -20.0, 0.0))

        pos = torch.arange(Q, device=x_seq.device)
        causal_mask = (pos.unsqueeze(1) >= pos.unsqueeze(0)).float().view(1, 1, 1, Q, Q)
        decay_weights = decay_matrix * causal_mask

        s_matrix = torch.matmul(q_full, k_full.transpose(-1, -2)) * decay_weights
        y_intra = torch.matmul(s_matrix, v_full)

        # 3. Inter-chunk state propagation (Sequential recurrence over C chunks with full autograd)
        decay_to_start = torch.exp(torch.clamp(lambda_t, -20.0, 0.0)).unsqueeze(-1)
        lambda_end = lambda_t[:, :, :, -1:].unsqueeze(-1)
        decay_to_end = torch.exp(torch.clamp(lambda_end - lambda_t.unsqueeze(-1), -20.0, 0.0))
        
        k_decayed = k_full.float() * decay_to_end * beta.unsqueeze(-1).float()
        kv_chunk_updates = torch.matmul(k_decayed.transpose(-1, -2), v_full.float())
        alpha_chunks = torch.exp(torch.clamp(lambda_t[:, :, :, -1:], -20.0, 0.0)).unsqueeze(-1)

        m_curr = m_prev.float()
        y_inter_list = []

        sigma_somatic = 1e-3 * (0.8 * curiosity.squeeze(1).float() + 0.4 * na.squeeze(1).float() + 0.1)

        for c in range(num_chunks):
            q_c = q_full[:, c]
            dec_start_c = decay_to_start[:, c]
            y_inter_c = torch.matmul(q_c.float() * dec_start_c, m_curr).to(q_full.dtype)
            y_inter_list.append(y_inter_c)

            alpha_c = alpha_chunks[:, c]
            kv_c = kv_chunk_updates[:, c]
            dW_c = torch.randn_like(m_curr) * torch.sqrt(eff_dt.squeeze(1).float()) * sigma_somatic
            m_curr = alpha_c * m_curr + kv_c + dW_c

        y_inter = torch.stack(y_inter_list, dim=1)

        # 4. Total output recombination
        y_total = (y_intra + y_inter).permute(0, 1, 3, 2, 4).reshape(batch_size * seq_len, self.num_heads * self.head_v)
        y_normed = self.head_norm(y_total)
        y_gated = y_normed * z_full
        h_seq = self.norm(self.out_proj(y_gated)).view(batch_size, seq_len, self.out_dim)

        if pad_len > 0:
            h_seq = h_seq[:, pad_len:, :]

        return h_seq, m_curr, eff_dt.mean().item()


# =============================================================================
# 2. CORTICAL STAGE (FUSED CHUNK-PARALLEL SSD + CONVSWIGLU)
# =============================================================================
class FusedCorticalStage(nn.Module):
    def __init__(self, hidden_dim: int = 768, expand_dim: int = 3072, num_heads: int = 12,
                 head_k: int = 64, head_v: int = 128, min_beta: float = 0.0005, max_beta: float = 0.08,
                 swiglu_kernel_size: int = 3, chunk_size: int = 64, device_str: str = 'cpu'):
        super().__init__()
        self.pre_norm_ssd = nn.LayerNorm(hidden_dim)
        self.ssd = FusedChunkParallelSSDLayer(
            in_dim=hidden_dim, out_dim=hidden_dim, num_heads=num_heads,
            head_k=head_k, head_v=head_v, min_beta=min_beta, max_beta=max_beta,
            chunk_size=chunk_size
        )
        self.pre_norm_swiglu = nn.LayerNorm(hidden_dim)
        self.swiglu = CausalConvSwiGLUBlock(
            hidden_dim=hidden_dim, expand_dim=expand_dim, kernel_size=swiglu_kernel_size,
            device=device_str
        )

    def forward(self, x: torch.Tensor, m_prev: torch.Tensor, u_t: torch.Tensor,
                saliency_gate: torch.Tensor = None, dt: float = 1.0) -> tuple:
        norm_x = self.pre_norm_ssd(x)
        h_ssd, m_next, eff_dt = self.ssd(norm_x, m_prev, u_t, saliency_gate=saliency_gate, dt=dt)
        x_res1 = x + h_ssd

        norm_res1 = self.pre_norm_swiglu(x_res1)
        x_out = self.swiglu(norm_res1)

        return x_out, m_next, eff_dt


# =============================================================================
# 3. EXP-83 FUSED CHUNK-PARALLEL + SYSTEM 2 SANDBOX AGENT
# =============================================================================
class Exp83FusedSandboxAgent(nn.Module):
    def __init__(self, config: CoREConfig, device_str: str = 'cpu'):
        super().__init__()
        self.device_str = device_str
        self.device = torch.device(device_str)
        self.config = config
        self.text_dim = config.net.text_dim
        self.hidden_dim = config.net.hidden_dim
        self.expand_dim = config.net.expand_dim
        self.latent_dim = getattr(config.net, 'latent_dim', 128)
        self.num_heads = config.net.num_heads
        self.head_k = 64
        self.head_v = 128
        self.text_gen_dim = 258

        self.byte_embed = nn.Embedding(self.text_gen_dim, self.text_dim).to(self.device)
        nn.init.normal_(self.byte_embed.weight, mean=0.0, std=0.08)
        self.pyramid_rf = MultiScaleBytePyramidReceptiveField(text_dim=self.text_dim, device=device_str)

        pe = torch.zeros(8192, self.text_dim)
        position = torch.arange(0, 8192, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.text_dim, 2).float() * (-math.log(10000.0) / self.text_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

        self.in_proj = nn.Linear(self.text_dim, self.hidden_dim).to(self.device)

        # Stage 1: Fast Morpho-Syntactic Sheet
        self.stage1 = FusedCorticalStage(
            hidden_dim=self.hidden_dim, expand_dim=self.expand_dim, num_heads=self.num_heads,
            head_k=self.head_k, head_v=self.head_v, min_beta=0.005, max_beta=0.15,
            swiglu_kernel_size=3, chunk_size=64, device_str=device_str
        ).to(self.device)

        # Boundary Detector
        self.boundary_detector = EntropyAdaptiveBoundaryDetector(hidden_dim=self.hidden_dim, device=device_str)

        # PW-LPER Module with 0.005 Gradient Scaling Hook
        self.pw_lper = PrecisionWeightedLPER(hidden_dim=self.hidden_dim, device=device_str)
        for p in self.pw_lper.parameters():
            if p.requires_grad:
                p.register_hook(lambda grad: grad * 0.005)

        # Stage 2: Slow Semantic Sheet
        self.stage2 = FusedCorticalStage(
            hidden_dim=self.hidden_dim, expand_dim=self.expand_dim, num_heads=self.num_heads,
            head_k=self.head_k, head_v=self.head_v, min_beta=0.0001, max_beta=0.05,
            swiglu_kernel_size=7, chunk_size=64, device_str=device_str
        ).to(self.device)

        self.attractor_head = DesaturatedHopfieldAttractorHead(
            hidden_dim=self.hidden_dim, vocab_size=self.text_gen_dim,
            num_attractors=256, device=device_str
        )

        self.motor_text_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.text_dim),
            nn.SiLU(),
            nn.LayerNorm(self.text_dim)
        ).to(self.device)

        self.episodic_sensory_proj = nn.Linear(self.text_dim, self.text_dim).to(self.device)

        self.world_model = LatentPredictor(
            hidden_dim=self.hidden_dim, unified_dim=self.text_dim,
            latent_dim=self.latent_dim, device=device_str
        )

    def embed_sequence(self, input_ids: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        seq_len = input_ids.size(1)
        tok_emb = self.byte_embed(input_ids)
        pos_emb = self.pe[:, start_pos : start_pos + seq_len, :]
        embedded = tok_emb + pos_emb
        if seq_len > 1:
            embedded = self.pyramid_rf(embedded)
        return embedded

    def forward_sequence(self, input_seq: torch.Tensor, target_seq: torch.Tensor, hu_batch, 
                         criterion, episodic_memory=None) -> tuple:
        batch_size, seq_len = input_seq.size()
        curr_u_t = hu_batch.state.clone().detach()

        m_s1_init = torch.zeros(batch_size, self.num_heads, self.head_k, self.head_v, dtype=torch.float32, device=self.device)
        m_s2_init = torch.zeros(batch_size, self.num_heads, self.head_k, self.head_v, dtype=torch.float32, device=self.device)
        h1_prev_last = torch.zeros(batch_size, 1, self.hidden_dim, device=self.device)

        # 1. Full-Sequence Multi-Scale Pyramid Embeddings
        full_emb = self.embed_sequence(input_seq, start_pos=0)
        h_in = self.in_proj(full_emb)

        # 2. Stage 1 Fast Morpho-Syntactic Cortical Pass (Fused Chunk-Parallel Scan)
        h_s1, m_s1_next, dt1 = self.stage1(h_in, m_s1_init, curr_u_t, None, 1.0)

        # 3. Dynamic Word/Morpheme Boundary Saliency Detector (Full Sequence)
        saliency_gate = self.boundary_detector(h_s1, input_seq)

        # 4. Precision-Weighted Laminar Error Routing (Full Sequence)
        e1_weighted, _, mean_pi = self.pw_lper(h_s1, h1_prev_last, curr_u_t)

        # 5. Stage 2 Slow Semantic-Discourse Cortical Pass (Fused Chunk-Parallel Scan)
        h_s2, m_s2_next, dt2 = self.stage2(e1_weighted, m_s2_init, curr_u_t, saliency_gate, 1.0)
        eff_dt = (dt1 + dt2) / 2.0

        # 6. Combined Laminar Representation
        h_combined = h_s1 + h_s2

        # 7. Modern Hopfield Attractor Relaxation (Full Sequence)
        h_flat = h_combined.contiguous().view(-1, self.hidden_dim)
        h_relaxed, commit_loss = self.attractor_head.relax_to_minima(h_flat, curr_u_t)

        # 8. Dopaminergic Afferent-Efferent Motor Readout & Logits
        da_level = curr_u_t[:, 5:6]
        motor_gain = (1.0 + 1.0 * da_level).unsqueeze(1)
        
        h_proj = self.motor_text_proj(h_relaxed).view(batch_size, seq_len, self.text_dim)
        h_proj_gain = (h_proj * motor_gain).contiguous().view(-1, self.text_dim)
        logits_flat = F.linear(h_proj_gain, self.byte_embed.weight)

        targets_flat = target_seq.contiguous().view(-1)
        speech_loss = criterion(logits_flat, targets_flat)

        # 9. Active Inference World Model Predictor
        w_current_last = self.episodic_sensory_proj(full_emb[:, -1, :])
        h_prev_proxy = h_combined[:, 0, :]
        h_curr_last = h_combined[:, -1, :]
        w_pred, kl_div, fe, _ = self.world_model(h_prev_proxy, h_curr_last, w_current_last)

        rec_loss = (1.0 - F.cosine_similarity(w_current_last, w_pred, dim=-1, eps=1e-8)).mean()
        free_energy_loss = kl_div.mean() + rec_loss
        ortho_loss = self.attractor_head.compute_pattern_separation_loss()

        # High-Surprise Episodic Memory Write
        with torch.no_grad():
            if free_energy_loss.item() > 0.20 and episodic_memory is not None:
                episodic_memory.write(w_current_last.detach().float(), w_pred.detach().float(), protected_slots=3)

        total_loss = speech_loss + 0.05 * free_energy_loss + 0.05 * commit_loss + 0.01 * ortho_loss

        h_proxy = m_s2_next.view(batch_size, -1)[:, :self.hidden_dim]
        return (total_loss, speech_loss.item(), free_energy_loss.item(), 
                m_s2_next, h_proxy, curr_u_t, eff_dt, mean_pi)

    def generate_natural_speech(self, prompt: str, hu_state, max_tokens: int = 75) -> str:
        """System 2 Active Inference Mental Sandbox / Latent Rollout Enabled."""
        self.eval()
        tokenizer = ByteTokenizer()
        prompt_ids = tokenizer.encode(prompt)
        if prompt_ids and prompt_ids[-1] == 257:
            prompt_ids = prompt_ids[:-1]

        prompt_t = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        m_s1 = torch.zeros(1, self.num_heads, self.head_k, self.head_v, dtype=torch.float32, device=self.device)
        m_s2 = torch.zeros(1, self.num_heads, self.head_k, self.head_v, dtype=torch.float32, device=self.device)

        prompt_emb = self.embed_sequence(prompt_t, start_pos=0)
        h1_prev_last = torch.zeros(1, 1, self.hidden_dim, device=self.device)

        # Process prompt in one single parallel pass
        h_in = self.in_proj(prompt_emb)
        h_s1, m_s1, _ = self.stage1(h_in, m_s1, hu_state, None, 1.0)
        sal_gate = self.boundary_detector(h_s1, prompt_t)

        e1_weighted, h1_prev_last, _ = self.pw_lper(h_s1, h1_prev_last, hu_state)
        h_s2, m_s2, _ = self.stage2(e1_weighted, m_s2, hu_state, sal_gate, 1.0)

        rolling_ids = prompt_ids.copy()
        generated_chars = []
        last_fe = 0.0

        with torch.no_grad():
            for step in range(max_tokens):
                context = rolling_ids[-8:]
                win_t = torch.tensor([context], dtype=torch.long, device=self.device)
                win_emb = self.embed_sequence(win_t, start_pos=len(rolling_ids) - len(context))
                t_emb = win_emb[:, -1:, :]

                h_in = self.in_proj(t_emb)

                h_s1, m_s1, _ = self.stage1(h_in, m_s1, hu_state, None, 1.0)
                sal_gate = self.boundary_detector(h_s1, win_t[:, -1:])

                e1_weighted, h1_prev_last, _ = self.pw_lper(h_s1, h1_prev_last, hu_state)
                h_s2, m_s2, _ = self.stage2(e1_weighted, m_s2, hu_state, sal_gate, 1.0)
                h_combined = h_s1 + h_s2

                # System 2 Active Inference Mental Sandbox (Counterfactual simulation during high surprise)
                if step > 0 and last_fe > 0.25:
                    h_sandbox = h_combined.clone()
                    w_curr = self.episodic_sensory_proj(t_emb.squeeze(1))
                    for _ in range(2):
                        w_pred, _, _, _ = self.world_model(h_sandbox.squeeze(1), h_sandbox.squeeze(1), w_curr)
                        h_sb_in = self.in_proj(w_pred).unsqueeze(1)
                        h_s1_sb, _, _ = self.stage1(h_sb_in, m_s1, hu_state, None, 1.0)
                        h_sandbox = h_s1_sb
                    h_combined = h_sandbox

                h_flat = h_combined.contiguous().view(-1, self.hidden_dim)
                h_relaxed, _ = self.attractor_head.relax_to_minima(h_flat, hu_state)
                h_proj = self.motor_text_proj(h_relaxed)

                raw_logits = F.linear(h_proj, self.byte_embed.weight)

                # Active Inference World Model update
                w_current = self.episodic_sensory_proj(t_emb.squeeze(1))
                w_pred, kl_div, fe, _ = self.world_model(h_combined.squeeze(1), h_combined.squeeze(1), w_current)
                last_fe = fe.mean().item()

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


def prepare_packed_stream(num_batches: int = 300, batch_size: int = 32, seq_len: int = 1024):
    logger.info(f"Loading Real Dataset (vicgalle/alpaca-gpt4) for EXP-83 (S={seq_len}, Steps={num_batches})...")
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


def run_exp_83_benchmark():
    print("\n" + "="*85)
    print(" === [KEP EXPERIMENTAL BENCHMARK: EXP-83 (FUSED CHUNK-PARALLEL + SYSTEM 2 SANDBOX)] ===")
    print("="*85)
    print(f"Hardware Device : {device_str.upper()} | AMP FP16 Enabled: {use_amp}")

    config = CoREConfig()
    config.net.text_dim = 256
    config.net.hidden_dim = 768
    config.net.expand_dim = 3072
    config.net.num_heads = 12

    b_size, seq_len = 32, 1024
    num_eval_steps = 300

    batches = prepare_packed_stream(num_batches=num_eval_steps, batch_size=b_size, seq_len=seq_len)
    criterion = nn.CrossEntropyLoss(ignore_index=256)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    diag_prompt = "User: What is the primary source of energy for Earth?\nKaryon:"

    print("\n[1/1] Initializing EXP-83 Fused Sandbox CoREAgent (300 Steps)...")
    torch.manual_seed(42)
    agent = Exp83FusedSandboxAgent(config, device_str=device_str).to(device)
    hu = HomeostaticUnit(batch_size=b_size, device=device_str)
    episodic_mem = BatchedEpisodicMemory(batch_size=b_size, memory_dim=config.net.text_dim, max_capacity=500, device=device_str)

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
    fe_list = []

    for step in range(num_eval_steps):
        t_batch_start = time.perf_counter()
        batch = batches[step]
        input_s = batch[:, :-1]
        target_s = batch[:, 1:]

        optimizer.zero_grad()
        t_fwd_0 = time.perf_counter()
        with torch.amp.autocast(device_type=device_str, dtype=torch.float16, enabled=use_amp):
            tot_loss, speech_loss, fe_val, m_curr, h_proxy, curr_u_t, eff_dt, avg_pi = agent.forward_sequence(
                input_s, target_s, hu, criterion, episodic_memory=episodic_mem
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
            ent_t = torch.full((b_size, 1), float(fe_val), device=device)
            cog_t = torch.zeros((b_size, 1), dtype=torch.int64, device=device)
            hu.update(cost_t, err_t, ent_t, cog_t)

        losses.append(speech_loss)
        fe_list.append(fe_val)

        t_step_total = (time.perf_counter() - t_batch_start) * 1000.0
        tok_sec = (b_size * seq_len) / (t_step_total / 1000.0)

        # KEP PROCESS DIAGNOSTICS DASHBOARD (EVERY 15 STEPS)
        if (step + 1) % 15 == 0 or step == num_eval_steps - 1:
            cur_loss = sum(losses[-15:]) / min(len(losses), 15)
            cur_fe = sum(fe_list[-15:]) / min(len(fe_list), 15)
            cur_lr = optimizer.param_groups[0]['lr']
            cur_ppl = math.exp(min(cur_loss, 20.0))

            curiosity, energy, stability, health, na, da = hu.state[0].tolist()
            peak_vram = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if device.type == 'cuda' else 0.0

            g_emb = agent.byte_embed.weight.grad.norm().item() if agent.byte_embed.weight.grad is not None else 0.0
            g_pw = 0.0
            for p in agent.pw_lper.parameters():
                if p.grad is not None:
                    g_pw = max(g_pw, p.grad.norm().item())
            g_s1 = 0.0
            for p in agent.stage1.parameters():
                if p.grad is not None:
                    g_s1 = max(g_s1, p.grad.norm().item())
            g_s2 = 0.0
            for p in agent.stage2.parameters():
                if p.grad is not None:
                    g_s2 = max(g_s2, p.grad.norm().item())

            print("\n" + "="*95)
            print(f" === [KEP EXP-83 PROCESS DIAGNOSTICS DASHBOARD | STEP {step+1:03d}/{num_eval_steps}] ===")
            print("="*95)
            print(f"Metrics Progress          : Speech Loss = {speech_loss:.4f} (Avg: {cur_loss:.4f}, PPL: {cur_ppl:.2f}) | LR = {cur_lr:.6f}")
            print(f"Active Inference Dynamics : Free Energy = {fe_val:.4f} (Avg: {cur_fe:.4f}) | Ascending Precision = {avg_pi:.4f}")
            print(f"Timing & Throughput       : Forward: {t_fwd:.1f}ms | Backward: {t_bwd:.1f}ms | Total Step: {t_step_total:.1f}ms | {tok_sec:.1f} tok/s")
            print(f"Somatic State (Ashby)     : Curiosity: {curiosity:.3f} | Energy: {energy:.3f} | NA: {na:.3f} | DA: {da:.3f} | dt_eff: {eff_dt:.3f}")
            print(f"Gradient Flow Inspection  : Total: {grad_norm_total:.4f} | Emb: {g_emb:.4f} | PW-LPER: {g_pw:.4f} | S1: {g_s1:.4f} | S2: {g_s2:.4f}")
            print(f"Hardware Resources        : Peak VRAM: {peak_vram:.1f} MB")
            print("="*95)

        if (step + 1) % 75 == 0:
            sample_text = agent.generate_natural_speech(diag_prompt, hu.state[0:1], max_tokens=65)
            logger.info(f"💬 [Live Diagnostic Speech Sample @ Step {step+1}] -> \"{sample_text}\"")

    if device.type == 'cuda': torch.cuda.synchronize()
    total_time_sec = time.perf_counter() - t_start
    final_loss = sum(losses[-30:]) / 30.0
    final_fe = sum(fe_list[-30:]) / 30.0
    final_sample = agent.generate_natural_speech(diag_prompt, hu.state[0:1], max_tokens=75)

    print("\n" + "="*95)
    print(" === [KEP EXP-83 FINAL TELEMETRY DASHBOARD] ===")
    print("="*95)
    print(f"{'Performance Metric':<36} | {'EXP-83 Fused Sandbox Value':<40}")
    print("-" * 95)
    print(f"{'Initial Loss (Step 1)':<36} | {losses[0]:<40.4f}")
    print(f"{'Step 100 Loss':<36} | {losses[99]:<40.4f}")
    print(f"{'Step 200 Loss':<36} | {losses[199]:<40.4f}")
    print(f"{'Final Steady-State Speech Loss':<36} | {final_loss:<40.4f} (PPL: {math.exp(final_loss):.2f})")
    print(f"{'Variational Free Energy (F_t)':<36} | {final_fe:<40.4f}")
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

    return final_loss, (num_eval_steps * b_size * seq_len) / total_time_sec, final_sample


if __name__ == "__main__":
    run_exp_83_benchmark()
