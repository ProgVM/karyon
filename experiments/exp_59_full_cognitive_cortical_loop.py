# experiments/exp_59_full_cognitive_cortical_loop.py
"""
===============================================================================
KARYON EXPERIMENTAL BENCHMARK: EXP-59 (FULL COGNITIVE CORTICAL LOOP)
100% Natural Biophysical Dynamics (Zero Logit Masking / Zero Crutches) +
Complete 10-System Active Inference Loop + Float32 Episodic Memory Safety +
Ultra-Detailed Real-Time Process Diagnostics Dashboard (Every 15 Steps).
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

import karyon_config, karyon_core, karyon_agent, karyon_logger
from karyon_config import CoREConfig
from karyon_agent import OffsetPositionalByteEmbedding
from karyon_core import ByteTokenizer, HomeostaticUnit, BatchedEpisodicMemory, ParallelSwiGLUBlock

logger = karyon_logger.get_logger()
torch.set_grad_enabled(True)

device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_str)
use_amp = (device_str == 'cuda')


# =============================================================================
# 1. BIOPHYSICAL MULTI-MODAL SENSORY GATEWAY (GLOBAL WORKSPACE THEORY)
# =============================================================================
class NaturalGlobalWorkspaceGateway(nn.Module):
    def __init__(self, unified_dim: int = 256, hidden_dim: int = 512, homeo_dim: int = 6,
                 text_dim: int = 256, vision_dim: int = 256, action_dim: int = 3, device_str: str = 'cpu'):
        super().__init__()
        self.unified_dim = unified_dim
        self.inv_sqrt_dim = 1.0 / math.sqrt(unified_dim)

        self.text_proj = nn.Linear(text_dim, unified_dim)
        self.vision_proj = nn.Linear(vision_dim, unified_dim)
        self.motor_proj = nn.Linear(action_dim, unified_dim)
        self.homeo_proj = nn.Linear(homeo_dim, unified_dim)
        self.mind_proj = nn.Linear(hidden_dim, unified_dim)

        self.attention_query_layer = nn.Linear(hidden_dim, unified_dim)
        self.channel_norm = nn.LayerNorm(unified_dim)
        self.query_norm = nn.LayerNorm(unified_dim)

    def forward(self, text_input, vision_input, motor_input, h_prev, u_t):
        batch_size = h_prev.size(0)

        text_ch = self.text_proj(text_input)
        vis_ch = self.vision_proj(vision_input)
        mot_ch = self.motor_proj(motor_input)
        body_ch = self.homeo_proj(u_t)
        mind_ch = self.mind_proj(h_prev)

        stacked = torch.stack([text_ch, vis_ch, mot_ch, body_ch, mind_ch], dim=1)
        norm_stacked = self.channel_norm(stacked)

        query = self.query_norm(self.attention_query_layer(h_prev)).unsqueeze(1)
        sim = torch.sum(query * norm_stacked, dim=-1) * self.inv_sqrt_dim

        # Natural Noradrenaline (Arousal) Gain
        na_gain = u_t[:, 4:5] * 0.5
        sim[:, 0:1] = sim[:, 0:1] + na_gain

        attn_weights = F.softmax(sim, dim=-1)
        epistemic_entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-9), dim=-1, keepdim=True)
        w_t = torch.sum(attn_weights.unsqueeze(-1) * stacked, dim=1)

        return w_t, attn_weights, epistemic_entropy


# =============================================================================
# 2. SPECIALIZED BALANCED SSD LAYER (MAMBA-2 GROUPNORM)
# =============================================================================
class BalancedSSDLayer(nn.Module):
    def __init__(self, in_dim: int = 512, out_dim: int = 512, num_heads: int = 8,
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
        self.delta_proj = nn.Linear(in_dim, num_heads)

        betas = torch.exp(torch.linspace(math.log(max_beta), math.log(min_beta), num_heads))
        alphas = 1.0 - betas
        logit_init = torch.log(alphas / (1.0 - alphas)).view(1, num_heads, 1, 1)
        self.decay_logits = nn.Parameter(logit_init)

        self.head_norm = nn.GroupNorm(num_groups=num_heads, num_channels=num_heads * head_v)
        self.out_proj = nn.Linear(num_heads * head_v, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x_seq: torch.Tensor, m_prev: torch.Tensor, u_t: torch.Tensor, dt: float = 1.0):
        batch_size, chunk_len, _ = x_seq.size()

        curiosity = u_t.select(1, 0).view(batch_size, 1, 1, 1)
        na = u_t.select(1, 4).view(batch_size, 1, 1, 1)
        da = u_t.select(1, 5).view(batch_size, 1, 1, 1)
        eff_dt = torch.clamp(dt * (1.0 - 0.4 * na + 0.4 * da), 0.30, 2.00)

        q = (self.q_proj(x_seq).view(batch_size, chunk_len, self.num_heads, self.head_k).transpose(1, 2)) * self.inv_sqrt_k
        k = self.k_proj(x_seq).view(batch_size, chunk_len, self.num_heads, self.head_k).transpose(1, 2)
        v = self.v_proj(x_seq).view(batch_size, chunk_len, self.num_heads, self.head_v).transpose(1, 2)

        selective_delta = F.softplus(self.delta_proj(x_seq)).view(batch_size, chunk_len, self.num_heads, 1).transpose(1, 2)
        base_alpha = torch.sigmoid(self.decay_logits)
        alpha = torch.pow(base_alpha, (selective_delta * eff_dt).clamp(0.1, 10.0))
        beta = 1.0 - alpha

        pos = torch.arange(chunk_len, device=x_seq.device, dtype=torch.float32)
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)
        causal_mask = (diff >= 0).float().view(1, 1, chunk_len, chunk_len)

        mean_alpha = alpha.mean(dim=2, keepdim=True)
        decay_weights = torch.pow(mean_alpha, diff.clamp_min(0).view(1, 1, chunk_len, chunk_len)) * causal_mask

        s_matrix = torch.matmul(q, k.transpose(-1, -2)) * decay_weights
        y_intra = torch.matmul(s_matrix, v)

        decay_to_start = torch.pow(mean_alpha.float(), (pos + 1.0).view(1, 1, chunk_len, 1))
        y_inter = torch.matmul(q.float() * decay_to_start, m_prev.float()).to(q.dtype)

        y_total = (y_intra + y_inter).transpose(1, 2).reshape(batch_size * chunk_len, self.num_heads * self.head_v)
        y_normed = self.head_norm(y_total)
        h_chunk = self.norm(self.out_proj(y_normed))

        decay_to_end = torch.pow(mean_alpha.float(), (float(chunk_len) - 1.0 - pos).view(1, 1, chunk_len, 1))
        k_decayed = k.float() * decay_to_end * beta.mean(dim=2, keepdim=True).float()
        kv_chunk_update = torch.matmul(k_decayed.transpose(-1, -2), v.float())

        alpha_chunk = torch.pow(mean_alpha.float(), float(chunk_len))
        sigma_somatic = 1e-3 * (0.8 * curiosity.float() + 0.4 * na.float() + 0.1)
        dW = torch.randn_like(m_prev) * torch.sqrt(eff_dt.float()) * sigma_somatic
        m_next = alpha_chunk * m_prev.float() + kv_chunk_update + dW

        return h_chunk, m_next, eff_dt.mean().item()


# =============================================================================
# 3. CORTICAL STAGE (SSD + SWIGLU + PRE-LAYERNORM)
# =============================================================================
class CorticalStage(nn.Module):
    def __init__(self, hidden_dim: int = 512, expand_dim: int = 2048, num_heads: int = 8,
                 head_k: int = 64, head_v: int = 128, min_beta: float = 0.0005, max_beta: float = 0.08, device_str: str = 'cpu'):
        super().__init__()
        self.pre_norm_ssd = nn.LayerNorm(hidden_dim)
        self.ssd = BalancedSSDLayer(
            in_dim=hidden_dim, out_dim=hidden_dim, num_heads=num_heads,
            head_k=head_k, head_v=head_v, min_beta=min_beta, max_beta=max_beta
        )
        self.pre_norm_swiglu = nn.LayerNorm(hidden_dim)
        self.swiglu = karyon_core.ParallelSwiGLUBlock(
            hidden_dim=hidden_dim, expand_dim=expand_dim, device=device_str
        )

    def forward(self, x: torch.Tensor, m_prev: torch.Tensor, u_t: torch.Tensor, dt: float = 1.0):
        norm_x = self.pre_norm_ssd(x)
        h_ssd, m_next, eff_dt = self.ssd(norm_x, m_prev, u_t, dt)
        x_res1 = x + h_ssd.view_as(x)

        norm_res1 = self.pre_norm_swiglu(x_res1)
        h_swiglu = self.swiglu(norm_res1.contiguous().view(-1, x.size(-1)))
        x_out = x_res1 + h_swiglu.view_as(x_res1)

        return x_out, m_next, eff_dt


# =============================================================================
# 4. STABLE ACTIVE INFERENCE LATENT WORLD MODEL
# =============================================================================
class StableLatentPredictor(nn.Module):
    def __init__(self, hidden_dim: int = 512, unified_dim: int = 256, latent_dim: int = 128):
        super().__init__()
        self.prior_net = nn.Linear(hidden_dim, latent_dim * 2)
        self.posterior_net = nn.Linear(hidden_dim + unified_dim, latent_dim * 2)
        self.decoder_net = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, unified_dim * 2),
            nn.SiLU(),
            nn.Linear(unified_dim * 2, unified_dim)
        )

    def forward(self, h_fast_prev, h_slow_curr, w_t):
        prior_out = self.prior_net(h_fast_prev)
        mu_prior, logvar_prior = prior_out.chunk(2, dim=-1)
        logvar_prior = torch.clamp(logvar_prior, -4.0, 4.0)

        post_out = self.posterior_net(torch.cat([h_fast_prev, w_t], dim=-1))
        mu_post, logvar_post = post_out.chunk(2, dim=-1)
        logvar_post = torch.clamp(logvar_post, -4.0, 4.0)

        std_post = torch.exp(0.5 * logvar_post)
        z_t = mu_post + torch.randn_like(std_post) * std_post

        w_pred = self.decoder_net(torch.cat([z_t, h_slow_curr], dim=-1))

        var_prior = torch.exp(logvar_prior) + 0.01
        var_post = torch.exp(logvar_post) + 0.01

        kl_div = 0.5 * torch.mean(
            logvar_prior - logvar_post + (var_post + (mu_post - mu_prior)**2) / var_prior - 1.0,
            dim=-1, keepdim=True
        )

        rec_loss = (1.0 - F.cosine_similarity(w_t, w_pred, dim=-1, eps=1e-8)).unsqueeze(-1)
        free_energy = kl_div + rec_loss

        return w_pred, kl_div, free_energy, rec_loss


# =============================================================================
# 5. CRISP CONTINUOUS HOPFIELD ATTRACTOR HEAD
# =============================================================================
class CrispContinuousHopfieldHead(nn.Module):
    def __init__(self, hidden_dim: int = 512, vocab_size: int = 258, num_attractors: int = 256, device_str: str = 'cpu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_attractors = num_attractors
        self.base_beta = 12.0

        self.attractor_query = nn.Linear(hidden_dim, hidden_dim)
        self.attractor_basins = nn.Parameter(torch.randn(num_attractors, hidden_dim) / math.sqrt(hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)

    def relax_to_minima(self, h_state: torch.Tensor, u_t: torch.Tensor):
        da_val = 0.0
        if u_t is not None and u_t.numel() >= 6:
            da_val = u_t[0, 5].item()
        eff_beta = self.base_beta * (1.0 + 1.5 * da_val)

        norm_basins = F.normalize(self.attractor_basins, p=2, dim=-1)
        q_h = F.normalize(self.attractor_query(h_state), p=2, dim=-1)

        sim = torch.matmul(q_h, norm_basins.transpose(0, 1)) * eff_beta
        attn_weights = F.softmax(sim, dim=-1)
        attractor_shift = torch.matmul(attn_weights, norm_basins)

        h_relaxed = self.norm(h_state + 0.35 * attractor_shift)
        commit_loss = F.mse_loss(h_state, h_relaxed.detach()) + 0.25 * F.mse_loss(h_state.detach(), h_relaxed)
        return h_relaxed, commit_loss


# =============================================================================
# 6. EXP-59 MASTER FULL-COGNITIVE-LOOP AGENT
# =============================================================================
class FullCognitiveCoREAgent(nn.Module):
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

        self.pos_embeddings = OffsetPositionalByteEmbedding(
            vocab_size=self.text_gen_dim, text_dim=self.text_dim, max_len=8192, device_str=device_str
        ).to(self.device)
        nn.init.normal_(self.pos_embeddings.byte_embed.weight, mean=0.0, std=0.08)

        self.gateway = NaturalGlobalWorkspaceGateway(
            unified_dim=self.text_dim, hidden_dim=self.hidden_dim, homeo_dim=6,
            text_dim=self.text_dim, vision_dim=256, action_dim=3, device_str=device_str
        ).to(self.device)

        self.in_proj = nn.Linear(self.text_dim, self.hidden_dim).to(self.device)

        self.stage1 = CorticalStage(
            hidden_dim=self.hidden_dim, expand_dim=self.expand_dim, num_heads=self.num_heads,
            head_k=self.head_k, head_v=self.head_v, min_beta=0.005, max_beta=0.15, device_str=device_str
        ).to(self.device)

        self.stage2 = CorticalStage(
            hidden_dim=self.hidden_dim, expand_dim=self.expand_dim, num_heads=self.num_heads,
            head_k=self.head_k, head_v=self.head_v, min_beta=0.0001, max_beta=0.05, device_str=device_str
        ).to(self.device)

        self.world_model = StableLatentPredictor(
            hidden_dim=self.hidden_dim, unified_dim=self.text_dim, latent_dim=self.latent_dim
        ).to(self.device)

        self.attractor_head = CrispContinuousHopfieldHead(
            hidden_dim=self.hidden_dim, vocab_size=self.text_gen_dim,
            num_attractors=256, device_str=device_str
        ).to(self.device)

        self.motor_text_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.text_dim),
            nn.SiLU(),
            nn.LayerNorm(self.text_dim)
        ).to(self.device)

    def forward_sequence(self, input_seq: torch.Tensor, target_seq: torch.Tensor, hu_batch, episodic_memory, criterion, chunk_size: int = 64):
        batch_size, seq_len = input_seq.size()
        m_s1 = torch.zeros(batch_size, self.num_heads, self.head_k, self.head_v, dtype=torch.float32, device=self.device)
        m_s2 = torch.zeros(batch_size, self.num_heads, self.head_k, self.head_v, dtype=torch.float32, device=self.device)
        curr_u_t = hu_batch.state.clone().detach()
        h_prev = torch.zeros(batch_size, self.hidden_dim, device=self.device)

        num_chunks = max(1, seq_len // chunk_size)
        chunk_losses = []
        commit_losses = []
        fe_losses = []
        kl_losses = []
        rec_losses = []
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

            w_slice = chunk_emb[:, -1, :]
            dummy_vis = torch.zeros(batch_size, 256, device=self.device)
            dummy_mot = torch.zeros(batch_size, 3, device=self.device)
            w_current, _, epistemic_ent = self.gateway(w_slice, dummy_vis, dummy_mot, h_prev, curr_u_t)

            # Volitional Memory Recall (Dtype Safety: cast float to prevent c10::Half overflow)
            na_val = curr_u_t[:, 4:5].mean().item()
            if episodic_memory is not None and na_val > 0.12 and getattr(episodic_memory, 'max_active_cpu', 0) > 0:
                with torch.no_grad():
                    ret_mem, max_sim = episodic_memory.read(w_current.detach().float(), temperature=0.05, threshold=0.70, sigmoid_beta=15.0)
                    if (max_sim > 0.70).any():
                        w_current = w_current + ret_mem.to(w_current.dtype) * 0.20

            h_in = self.in_proj(chunk_emb)

            h_s1, m_s1, dt1 = self.stage1(h_in, m_s1, curr_u_t, dt=1.0)
            h_s2, m_s2, dt2 = self.stage2(h_s1, m_s2, curr_u_t, dt=1.0)
            eff_dts.append((dt1 + dt2) / 2.0)

            h_flat = h_s2.contiguous().view(-1, self.hidden_dim)
            h_relaxed, commit_loss = self.attractor_head.relax_to_minima(h_flat, curr_u_t)

            h_proj = self.motor_text_proj(h_relaxed).view(batch_size, c_len, self.text_dim)
            h_proj_gain = (h_proj * motor_gain).contiguous().view(-1, self.text_dim)

            logits_flat = F.linear(h_proj_gain, self.pos_embeddings.byte_embed.weight)
            targets_flat = chunk_tgt.contiguous().view(-1)

            loss = criterion(logits_flat, targets_flat)
            chunk_losses.append(loss)
            commit_losses.append(commit_loss)

            h_last = h_s2[:, -1, :]
            w_pred, kl_div, fe, rec_l = self.world_model(h_prev, h_last, w_current)
            h_prev = h_last.detach()

            fe_losses.append(fe.mean())
            kl_losses.append(kl_div.mean())
            rec_losses.append(rec_l.mean())

            # Episodic Storage on High Surprise (Float32 buffer safety)
            with torch.no_grad():
                if fe.mean().item() > 0.20 and episodic_memory is not None:
                    episodic_memory.write(w_current.detach().float(), w_pred.detach().float(), protected_slots=3)

                has_eos = (chunk_in == 257).any(dim=-1).view(batch_size, 1, 1, 1).float()
                m_s1 = (m_s1 * (1.0 - has_eos)).detach()
                m_s2 = (m_s2 * (1.0 - has_eos)).detach()

        avg_loss = torch.stack(chunk_losses).mean()
        avg_commit = torch.stack(commit_losses).mean()
        avg_fe = torch.stack(fe_losses).mean()
        avg_kl = torch.stack(kl_losses).mean()
        avg_rec = torch.stack(rec_losses).mean()
        avg_eff_dt = sum(eff_dts) / len(eff_dts)

        total_loss = avg_loss + 0.01 * avg_fe + 0.05 * avg_commit
        return (total_loss, avg_loss.item(), avg_fe.item(), avg_kl.item(),
                avg_rec.item(), avg_commit.item(), avg_eff_dt)

    def generate_natural_speech(self, prompt: str, episodic_memory, hu_state, max_tokens: int = 75) -> str:
        """100% Pure Natural Autoregressive Speech Synthesis (Zero Artificial Masking)."""
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

        for c_idx in range(0, prompt_len, 64):
            c_emb = prompt_emb[:, c_idx : min(c_idx + 64, prompt_len), :]
            h_in = self.in_proj(c_emb)
            h_s1, m_s1, _ = self.stage1(h_in, m_s1, hu_state, 1.0)
            h_s2, m_s2, _ = self.stage2(h_s1, m_s2, hu_state, 1.0)

        rolling_ids = prompt_ids.copy()
        generated_chars = []

        with torch.no_grad():
            for step in range(max_tokens):
                context = rolling_ids[-4:]
                win_t = torch.tensor([context], dtype=torch.long, device=self.device)
                win_emb = self.pos_embeddings(win_t, start_pos=len(rolling_ids) - len(context), apply_rf=True)
                t_emb = win_emb[:, -1:, :]

                h_in = self.in_proj(t_emb)
                h_s1, m_s1, _ = self.stage1(h_in, m_s1, hu_state, 1.0)
                h_s2, m_s2, _ = self.stage2(h_s1, m_s2, hu_state, 1.0)

                h_flat = h_s2.contiguous().view(-1, self.hidden_dim)
                h_rel, _ = self.attractor_head.relax_to_minima(h_flat, hu_state)
                h_proj = self.motor_text_proj(h_rel)

                raw_logits = F.linear(h_proj, self.pos_embeddings.byte_embed.weight)

                # Pure Top-p nucleus sampling (No artificial masks!)
                p_dist = F.softmax(raw_logits, dim=-1)
                entropy = -(p_dist * torch.log(p_dist + 1e-9)).sum(dim=-1).item()
                is_boundary = (len(rolling_ids) > 0 and rolling_ids[-1] in [32, 10, 44, 46])

                if is_boundary or entropy > 0.70:
                    temp = 0.50
                    top_p = 0.90
                else:
                    temp = 0.10
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
                next_token_id = torch.multinomial(probs, num_samples=1).item()

                if next_token_id == 257:
                    break
                rolling_ids.append(next_token_id)
                char_c = chr(next_token_id) if 32 <= next_token_id <= 126 or next_token_id in [9, 10, 13] else ' '
                generated_chars.append(char_c)

        self.train()
        return "".join(generated_chars).strip()


# =============================================================================
# 7. REAL DATASET & RUNNER WITH MAXIMUM DEEP DIAGNOSTICS (EVERY 15 STEPS)
# =============================================================================
def prepare_packed_stream(num_batches: int = 300, batch_size: int = 32, seq_len: int = 1024):
    logger.info(f"Loading Real Dataset (vicgalle/alpaca-gpt4) for EXP-59 (S={seq_len}, Steps={num_batches})...")
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

def run_exp_59_benchmark():
    print("\n" + "="*85)
    print(" === [KEP EXPERIMENTAL BENCHMARK: EXP-59 (FULL COGNITIVE CORTICAL LOOP)] ===")
    print("="*85)
    print(f"Hardware Device : {device_str.upper()} | AMP FP16 Enabled: {use_amp}")

    config = CoREConfig()
    config.net.text_dim = 256
    config.net.hidden_dim = 512
    config.net.expand_dim = 2048
    config.net.num_heads = 8

    b_size, seq_len = 32, 1024
    num_eval_steps = 300
    chunk_size = 64

    batches = prepare_packed_stream(num_batches=num_eval_steps, batch_size=b_size, seq_len=seq_len)
    criterion = nn.CrossEntropyLoss(ignore_index=256)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    diag_prompt = "User: What is the primary source of energy for Earth?\nKaryon:"

    print("\n[1/1] Initializing Full 10-System Cognitive Architecture (300 Steps on S=1024)...")
    torch.manual_seed(42)
    agent = FullCognitiveCoREAgent(config, device_str=device_str).to(device)
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
    losses, fe_list, kl_list, rec_list = [], [], [], []

    for step in range(num_eval_steps):
        t_batch_start = time.perf_counter()
        batch = batches[step]
        input_s = batch[:, :-1]
        target_s = batch[:, 1:]

        optimizer.zero_grad()
        t_fwd_0 = time.perf_counter()
        with torch.amp.autocast(device_type=device_str, dtype=torch.float16, enabled=use_amp):
            tot_loss, speech_loss, fe_val, kl_val, rec_val, commit_val, eff_dt = agent.forward_sequence(
                input_s, target_s, hu, episodic_mem, criterion, chunk_size=chunk_size
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

        # Real Homeostatic Somatic Coupling
        with torch.no_grad():
            cost_t = torch.full((b_size, 1), 0.001, device=device)
            err_t = torch.full((b_size, 1), float(speech_loss * 0.1), device=device)
            ent_t = torch.full((b_size, 1), float(fe_val), device=device)
            cog_t = torch.zeros((b_size, 1), dtype=torch.int64, device=device)
            hu.update(cost_t, err_t, ent_t, cog_t)

        losses.append(speech_loss)
        fe_list.append(fe_val)
        kl_list.append(kl_val)
        rec_list.append(rec_val)

        t_step_total = (time.perf_counter() - t_batch_start) * 1000.0
        tok_sec = (b_size * seq_len) / (t_step_total / 1000.0)

        # =====================================================================
        # KEP DEEP PROCESS DIAGNOSTICS DASHBOARD (EVERY 15 STEPS)
        # =====================================================================
        if (step + 1) % 15 == 0 or step == num_eval_steps - 1:
            cur_loss = sum(losses[-15:]) / min(len(losses), 15)
            cur_fe = sum(fe_list[-15:]) / min(len(fe_list), 15)
            cur_kl = sum(kl_list[-15:]) / min(len(kl_list), 15)
            cur_rec = sum(rec_list[-15:]) / min(len(rec_list), 15)
            cur_lr = optimizer.param_groups[0]['lr']
            cur_ppl = math.exp(min(cur_loss, 20.0))

            curiosity, energy, stability, health, na, da = hu.state[0].tolist()
            peak_vram = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if device.type == 'cuda' else 0.0

            g_emb = agent.pos_embeddings.byte_embed.weight.grad.norm().item() if agent.pos_embeddings.byte_embed.weight.grad is not None else 0.0
            g_gwt = agent.gateway.text_proj.weight.grad.norm().item() if agent.gateway.text_proj.weight.grad is not None else 0.0
            g_s1 = agent.stage1.ssd.q_proj.weight.grad.norm().item() if agent.stage1.ssd.q_proj.weight.grad is not None else 0.0
            g_s2 = agent.stage2.ssd.q_proj.weight.grad.norm().item() if agent.stage2.ssd.q_proj.weight.grad is not None else 0.0
            g_sw1 = agent.stage1.swiglu.w_gate.weight.grad.norm().item() if agent.stage1.swiglu.w_gate.weight.grad is not None else 0.0
            g_wm = agent.world_model.decoder_net[0].weight.grad.norm().item() if agent.world_model.decoder_net[0].weight.grad is not None else 0.0
            g_hop = agent.attractor_head.attractor_basins.grad.norm().item() if agent.attractor_head.attractor_basins.grad is not None else 0.0

            print("\n" + "="*95)
            print(f" === [KEP PROCESS DIAGNOSTICS DASHBOARD | STEP {step+1:03d}/{num_eval_steps}] ===")
            print("="*95)
            print(f"Metrics Progress          : Speech Loss = {speech_loss:.4f} (Avg: {cur_loss:.4f}, PPL: {cur_ppl:.2f}) | LR = {cur_lr:.6f}")
            print(f"Active Inference (F_t)    : Total F_t = {cur_fe:.4f} | KL-Div = {cur_kl:.4f} | Rec-Loss = {cur_rec:.4f} | Commit = {commit_val:.4f}")
            print(f"Timing & Throughput       : Forward: {t_fwd:.1f}ms | Backward: {t_bwd:.1f}ms | Total Step: {t_step_total:.1f}ms | {tok_sec:.1f} tok/s")
            print(f"Somatic State (Ashby)     : Curiosity: {curiosity:.3f} | Energy: {energy:.3f} | Stability: {stability:.3f} | NA: {na:.3f} | DA: {da:.3f} | dt_eff: {eff_dt:.3f}")
            print(f"Gradient Flow Inspection  : Total: {grad_norm_total:.4f} | Emb: {g_emb:.4f} | GWT: {g_gwt:.4f} | S1: {g_s1:.4f} | S2: {g_s2:.4f} | SwiGLU: {g_sw1:.4f} | WM: {g_wm:.4f} | Hopfield: {g_hop:.4f}")
            print(f"Hardware Resources        : Peak VRAM: {peak_vram:.1f} MB | Episodic Active Slots: {episodic_mem.max_active_cpu}")
            print("="*95)

        # KEP Rule #4 Speech Auditing every 75 steps
        if (step + 1) % 75 == 0:
            sample_text = agent.generate_natural_speech(diag_prompt, episodic_mem, hu.state, max_tokens=65)
            logger.info(f"💬 [Live Diagnostic Speech Sample @ Step {step+1}] -> \"{sample_text}\"")

    if device.type == 'cuda': torch.cuda.synchronize()
    total_time_sec = time.perf_counter() - t_start
    final_loss = sum(losses[-30:]) / 30.0
    final_fe = sum(fe_list[-30:]) / 30.0
    final_sample = agent.generate_natural_speech(diag_prompt, episodic_mem, hu.state, max_tokens=75)

    # 2. FINAL TELEMETRY DASHBOARD
    print("\n" + "="*95)
    print(" === [KEP EXP-59 FULL COGNITIVE LOOP FINAL TELEMETRY DASHBOARD] ===")
    print("="*95)
    print(f"{'Performance Metric':<36} | {'EXP-59 Full Architecture Value':<40}")
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


if __name__ == "__main__":
    run_exp_59_benchmark()
