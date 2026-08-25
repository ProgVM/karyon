# experiments/exp_54_unified_dual_refactored_core.py
"""
===============================================================================
KARYON EXPERIMENTAL BENCHMARK: EXP-54 (UNIFIED DUAL-REFACTORED CORE)
Full Spontaneous Dual-Refactoring on Axis A (Mamba-2 GroupNorm SSD + Crisp Hopfield)
and Axis B (Natural GWT Attention + Real Free Energy Somatic Coupling + Theta PAC).
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
from karyon_core import ByteTokenizer, HomeostaticUnit, ParallelSwiGLUBlock, LatentPredictor

logger = karyon_logger.get_logger()
torch.set_grad_enabled(True)

device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_str)
use_amp = (device_str == 'cuda')


# =============================================================================
# 1. REFACTORED AXIS B: NATURAL SENSORY GATEWAY (ZERO ARTIFICIAL BIAS)
# =============================================================================
class NaturalSensoryGateway(nn.Module):
    """
    Global Workspace Gateway with pure biophysical competitive attention.
    Eradicates artificial hardcoded '+1.5f' biases. Modalities compete purely
    based on sensory salience, somatic arousal (NA), and top-down mental volition.
    """
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

        # 1. Project all sensory channels
        text_ch = self.text_proj(text_input)
        vis_ch = self.vision_proj(vision_input)
        mot_ch = self.motor_proj(motor_input)
        body_ch = self.homeo_proj(u_t)
        mind_ch = self.mind_proj(h_prev)

        stacked_channels = torch.stack([text_ch, vis_ch, mot_ch, body_ch, mind_ch], dim=1) # [B, 5, D]
        norm_channels = self.channel_norm(stacked_channels)

        # 2. Volitional query from top-down state
        query = self.query_norm(self.attention_query_layer(h_prev)).unsqueeze(1) # [B, 1, D]

        # 3. Pure dot-product competitive attention without artificial bias
        sim = torch.sum(query * norm_channels, dim=-1) * self.inv_sqrt_dim # [B, 5]
        
        # Modulate text salience purely via Noradrenaline (Arousal)
        na_salience = u_t[:, 4:5] * 0.5
        sim[:, 0:1] = sim[:, 0:1] + na_salience

        attn_weights = F.softmax(sim, dim=-1)
        epistemic_entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-9), dim=-1, keepdim=True)
        w_t = torch.sum(attn_weights.unsqueeze(-1) * stacked_channels, dim=1)

        return w_t, attn_weights, epistemic_entropy


# =============================================================================
# 2. REFACTORED AXIS A: BALANCED MAMBA-2 SSD CORE + CRISP HOPFIELD ATTRACTOR
# =============================================================================
class RefactoredSSDCore(nn.Module):
    """
    Balanced Mamba-2 Parallel SSD Core with GroupNorm Head Equalization,
    FP32 State Scan, and Linguistic Timescales [0.92, 0.9995].
    """
    def __init__(self, text_dim: int = 256, unified_dim: int = 256, hidden_dim: int = 512,
                 num_heads: int = 8, head_k: int = 64, head_v: int = 128, device_str: str = 'cpu'):
        super().__init__()
        self.text_dim = text_dim
        self.unified_dim = unified_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_k = head_k
        self.head_v = head_v
        self.inv_sqrt_k = 1.0 / math.sqrt(head_k)

        self.sensory_proj = nn.Linear(text_dim, unified_dim)
        self.q_proj = nn.Linear(unified_dim, num_heads * head_k)
        self.k_proj = nn.Linear(unified_dim, num_heads * head_k)
        self.v_proj = nn.Linear(unified_dim, num_heads * head_v)
        self.delta_proj = nn.Linear(unified_dim, num_heads)

        # Multi-timescale decay spectrum: alpha in [0.92, 0.9995]
        betas = torch.exp(torch.linspace(math.log(0.08), math.log(0.0005), num_heads))
        alphas = 1.0 - betas
        logit_init = torch.log(alphas / (1.0 - alphas)).view(1, num_heads, 1, 1)
        self.decay_logits = nn.Parameter(logit_init)

        self.head_norm = nn.GroupNorm(num_groups=num_heads, num_channels=num_heads * head_v)
        self.out_proj = nn.Linear(num_heads * head_v, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward_chunk_parallel_ssd(self, chunk_emb: torch.Tensor, m_prev: torch.Tensor, u_t: torch.Tensor, dt: float = 1.0):
        batch_size, chunk_len, _ = chunk_emb.size()

        curiosity = u_t.select(1, 0).view(batch_size, 1, 1, 1)
        na = u_t.select(1, 4).view(batch_size, 1, 1, 1)
        da = u_t.select(1, 5).view(batch_size, 1, 1, 1)
        eff_dt = torch.clamp(dt * (1.0 - 0.4 * na + 0.4 * da), 0.30, 2.00)

        w_chunk = self.sensory_proj(chunk_emb)

        q = (self.q_proj(w_chunk).view(batch_size, chunk_len, self.num_heads, self.head_k).transpose(1, 2)) * self.inv_sqrt_k
        k = self.k_proj(w_chunk).view(batch_size, chunk_len, self.num_heads, self.head_k).transpose(1, 2)
        v = self.v_proj(w_chunk).view(batch_size, chunk_len, self.num_heads, self.head_v).transpose(1, 2)

        selective_delta = F.softplus(self.delta_proj(w_chunk)).view(batch_size, chunk_len, self.num_heads, 1).transpose(1, 2)
        base_alpha = torch.sigmoid(self.decay_logits)
        alpha = torch.pow(base_alpha, (selective_delta * eff_dt).clamp(0.1, 10.0))
        beta = 1.0 - alpha

        pos = torch.arange(chunk_len, device=chunk_emb.device, dtype=torch.float32)
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)
        causal_mask = (diff >= 0).float().view(1, 1, chunk_len, chunk_len)

        mean_alpha = alpha.mean(dim=2, keepdim=True)
        decay_weights = torch.pow(mean_alpha, diff.clamp_min(0).view(1, 1, chunk_len, chunk_len)) * causal_mask

        # 1. Intra-chunk attention
        s_matrix = torch.matmul(q, k.transpose(-1, -2)) * decay_weights
        y_intra = torch.matmul(s_matrix, v)

        # 2. Inter-chunk state retrieval in FP32
        decay_to_start = torch.pow(mean_alpha.float(), (pos + 1.0).view(1, 1, chunk_len, 1))
        y_inter = torch.matmul(q.float() * decay_to_start, m_prev.float()).to(q.dtype)

        y_total = (y_intra + y_inter).transpose(1, 2).reshape(batch_size * chunk_len, self.num_heads * self.head_v)

        # 3. Head equalization
        y_normed = self.head_norm(y_total)
        h_chunk = self.norm(self.out_proj(y_normed))

        # 4. State update with EMA beta bounds
        decay_to_end = torch.pow(mean_alpha.float(), (float(chunk_len) - 1.0 - pos).view(1, 1, chunk_len, 1))
        k_decayed = k.float() * decay_to_end * beta.mean(dim=2, keepdim=True).float()
        kv_chunk_update = torch.matmul(k_decayed.transpose(-1, -2), v.float())

        alpha_chunk = torch.pow(mean_alpha.float(), float(chunk_len))
        sigma_somatic = 1e-3 * (0.8 * curiosity.float() + 0.4 * na.float() + 0.1)
        dW = torch.randn_like(m_prev) * torch.sqrt(eff_dt.float()) * sigma_somatic
        m_next = alpha_chunk * m_prev.float() + kv_chunk_update + dW

        return h_chunk, m_next


class CrispContinuousHopfieldHead(nn.Module):
    """
    Modern Continuous Hopfield Head with unit-sphere normalization (||b_i||=1)
    and dopamine-modulated contrast sharpening (beta=12.0).
    """
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

    def compute_pattern_separation_loss(self):
        norm_basins = F.normalize(self.attractor_basins, p=2, dim=-1)
        cosine_matrix = torch.matmul(norm_basins, norm_basins.transpose(0, 1))
        eye = torch.eye(self.num_attractors, device=self.attractor_basins.device)
        return F.mse_loss(cosine_matrix, eye)


# =============================================================================
# 3. COMPLETE DUAL-REFACTORED KARYON AGENT
# =============================================================================
class DualRefactoredCoREAgent(nn.Module):
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
        self.inv_sqrt_text_dim = 1.0 / math.sqrt(self.text_dim)

        self.pos_embeddings = OffsetPositionalByteEmbedding(
            vocab_size=self.text_gen_dim, text_dim=self.text_dim, max_len=8192, device_str=device_str
        ).to(self.device)

        self.gateway = NaturalSensoryGateway(
            unified_dim=self.text_dim, hidden_dim=self.hidden_dim, homeo_dim=6,
            text_dim=self.text_dim, vision_dim=256, action_dim=3, device_str=device_str
        ).to(self.device)

        self.ssd_core = RefactoredSSDCore(
            text_dim=self.text_dim, unified_dim=self.text_dim, hidden_dim=self.hidden_dim,
            num_heads=self.num_heads, head_k=self.head_k, head_v=self.head_v, device_str=device_str
        ).to(self.device)

        self.channel_mixer = karyon_core.ParallelSwiGLUBlock(
            hidden_dim=self.hidden_dim, expand_dim=self.expand_dim, device=device_str
        )

        self.attractor_head = CrispContinuousHopfieldHead(
            hidden_dim=self.hidden_dim, vocab_size=self.text_gen_dim,
            num_attractors=256, device_str=device_str
        ).to(self.device)

        self.world_model = LatentPredictor(
            hidden_dim=self.hidden_dim, unified_dim=self.text_dim, latent_dim=self.latent_dim, device=device_str
        )

        self.motor_text_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.text_dim),
            nn.SiLU(),
            nn.LayerNorm(self.text_dim)
        ).to(self.device)

    def forward_sequence(self, input_seq: torch.Tensor, target_seq: torch.Tensor, hu_batch, criterion, chunk_size: int = 64):
        batch_size, seq_len = input_seq.size()
        m_curr = torch.zeros(batch_size, self.num_heads, self.head_k, self.head_v, dtype=torch.float32, device=self.device)
        curr_u_t = hu_batch.state.clone().detach()
        h_prev = torch.zeros(batch_size, self.hidden_dim, device=self.device)

        num_chunks = max(1, seq_len // chunk_size)
        chunk_losses = []
        commit_losses = []
        fe_losses = []

        for chunk_idx in range(num_chunks):
            c_start = chunk_idx * chunk_size
            c_end = min((chunk_idx + 1) * chunk_size, seq_len)
            chunk_in = input_seq[:, c_start:c_end]
            chunk_tgt = target_seq[:, c_start:c_end]

            chunk_emb = self.pos_embeddings(chunk_in, start_pos=c_start, apply_rf=True)
            h_chunk, m_curr = self.ssd_core.forward_chunk_parallel_ssd(chunk_emb, m_curr, curr_u_t, 1.0)

            h_reasoned = self.channel_mixer(h_chunk)
            h_relaxed, commit_loss = self.attractor_head.relax_to_minima(h_reasoned, curr_u_t)

            h_proj = self.motor_text_proj(h_relaxed)
            logits_flat = F.linear(h_proj, self.pos_embeddings.byte_embed.weight) * self.inv_sqrt_text_dim
            targets_flat = chunk_tgt.contiguous().view(-1)

            loss = criterion(logits_flat, targets_flat)
            chunk_losses.append(loss)
            commit_losses.append(commit_loss)

            # Active Inference World Model Loss
            w_slice = chunk_emb[:, -1, :]
            h_last = h_reasoned.view(batch_size, -1, self.hidden_dim)[:, -1, :]
            w_pred, kl_div, _, _ = self.world_model(h_prev, h_last, w_slice)
            h_prev = h_last.detach()

            rec_loss = (1.0 - F.cosine_similarity(w_slice, w_pred, dim=-1, eps=1e-8)).mean()
            chunk_fe = kl_div.mean() + rec_loss
            fe_losses.append(chunk_fe)

            has_eos = (chunk_in == 257).any(dim=-1).view(batch_size, 1, 1, 1).float()
            m_curr = (m_curr * (1.0 - has_eos)).detach()

        avg_loss = torch.stack(chunk_losses).mean()
        avg_commit = torch.stack(commit_losses).mean()
        avg_fe = torch.stack(fe_losses).mean()
        ortho_loss = self.attractor_head.compute_pattern_separation_loss()
        
        total_loss = avg_loss + 0.05 * avg_fe + 0.05 * avg_commit + 0.01 * ortho_loss
        return total_loss, avg_loss.item(), avg_fe.item()

    def generate_theta_pac_speech(self, prompt: str, max_tokens: int = 70) -> str:
        """Theta-Gamma Phase-Amplitude Coupled (PAC) Autoregressive Speech Synthesis."""
        self.eval()
        tokenizer = ByteTokenizer()
        prompt_ids = tokenizer.encode(prompt)
        if prompt_ids and prompt_ids[-1] == 257:
            prompt_ids = prompt_ids[:-1]

        prompt_t = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        m_curr = torch.zeros(1, self.num_heads, self.head_k, self.head_v, dtype=torch.float32, device=self.device)
        hu_state = torch.tensor([[0.5, 1.0, 1.0, 1.0, 0.0, 0.0]], device=self.device)

        prompt_emb = self.pos_embeddings(prompt_t, start_pos=0, apply_rf=True)
        prompt_len = prompt_t.size(1)
        h_step = None

        for c_idx in range(0, prompt_len, 64):
            c_emb = prompt_emb[:, c_idx : min(c_idx + 64, prompt_len), :]
            h_step, m_curr = self.ssd_core.forward_chunk_parallel_ssd(c_emb, m_curr, hu_state, 1.0)

        rolling_ids = prompt_ids.copy()
        generated_chars = []

        with torch.no_grad():
            for step in range(max_tokens):
                context = rolling_ids[-4:]
                win_t = torch.tensor([context], dtype=torch.long, device=self.device)
                win_emb = self.pos_embeddings(win_t, start_pos=len(rolling_ids) - len(context), apply_rf=True)
                t_emb = win_emb[:, -1:, :]

                h_step, m_curr = self.ssd_core.forward_chunk_parallel_ssd(t_emb, m_curr, hu_state, 1.0)
                h_out = self.channel_mixer(h_step)
                h_rel, _ = self.attractor_head.relax_to_minima(h_out, hu_state)
                h_proj = self.motor_text_proj(h_rel)

                raw_logits = F.linear(h_proj, self.pos_embeddings.byte_embed.weight) * self.inv_sqrt_text_dim
                raw_logits[:, 256:] = -1e9

                # Theta-Gamma Phase Transition Detection
                p_dist = F.softmax(raw_logits, dim=-1)
                entropy = -(p_dist * torch.log(p_dist + 1e-9)).sum(dim=-1).item()
                is_boundary = (len(rolling_ids) > 0 and rolling_ids[-1] in [32, 10, 44, 46])

                if is_boundary or entropy > 0.65:
                    temp = 0.40
                    top_p = 0.90
                else:
                    temp = 0.05  # Pristine MAP spelling inside morphemes
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
# 4. PARITY BENCHMARK EXECUTION
# =============================================================================
def prepare_packed_batches(num_batches: int = 150, batch_size: int = 32, seq_len: int = 512):
    logger.info("Loading Real Dataset (vicgalle/alpaca-gpt4) for EXP-54...")
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

    logger.info(f"Prepared {len(batches)} Real Packed Batches (B={batch_size}, S={seq_len}).")
    return batches

def run_exp_54_benchmark():
    print("\n" + "="*85)
    print(" === [KEP EXPERIMENTAL BENCHMARK: EXP-54 (UNIFIED DUAL-REFACTORED CORE)] ===")
    print("="*85)
    print(f"Hardware Device : {device_str.upper()} | AMP FP16 Enabled: {use_amp}")

    config = CoREConfig()
    config.net.text_dim = 256
    config.net.unified_dim = 256
    config.net.hidden_dim = 512
    config.net.expand_dim = 2048
    config.net.num_heads = 8
    config.net.head_k = 64
    config.net.head_v = 128
    config.train.chunk_size = 64

    b_size, seq_len = 32, 512
    num_eval_steps = 150
    chunk_size = 64

    batches = prepare_packed_batches(num_batches=num_eval_steps, batch_size=b_size, seq_len=seq_len)
    criterion = nn.CrossEntropyLoss(ignore_index=256)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    diag_prompt = "User: What is the primary source of energy for Earth?\nKaryon:"

    # 1. EVALUATE DUAL-REFACTORED CORE
    print("\n[1/1] Benchmarking EXP-54 Dual-Refactored Unified Agent...")
    torch.manual_seed(42)
    agent = DualRefactoredCoREAgent(config, device_str=device_str).to(device)
    optimizer = torch.optim.AdamW(agent.parameters(), lr=3e-3, weight_decay=0.01)
    hu = HomeostaticUnit(batch_size=b_size, device=device_str)

    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()
    losses = []
    fe_list = []

    for step in range(num_eval_steps):
        batch = batches[step]
        input_s = batch[:, :-1]
        target_s = batch[:, 1:]

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device_str, dtype=torch.float16, enabled=use_amp):
            tot_loss, speech_loss, fe_loss = agent.forward_sequence(input_s, target_s, hu, criterion, chunk_size=chunk_size)

        scaler.scale(tot_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=3.0)
        scaler.step(optimizer)
        scaler.update()

        # Real Free Energy Somatic Update (No Fake Variables!)
        with torch.no_grad():
            cost_t = torch.full((b_size, 1), 0.001, device=device)
            err_t = torch.full((b_size, 1), float(speech_loss * 0.1), device=device)
            ent_t = torch.full((b_size, 1), float(fe_loss), device=device)
            cog_t = torch.zeros((b_size, 1), dtype=torch.int64, device=device)
            hu.update(cost_t, err_t, ent_t, cog_t)

        losses.append(speech_loss)
        fe_list.append(fe_loss)

    if device.type == 'cuda': torch.cuda.synchronize()
    total_time_sec = time.perf_counter() - t_start
    peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if device.type == 'cuda' else 0.0
    throughput = (num_eval_steps * b_size * seq_len) / total_time_sec
    final_loss = sum(losses[-25:]) / 25.0
    final_fe = sum(fe_list[-25:]) / 25.0
    sample_text = agent.generate_theta_pac_speech(diag_prompt, max_tokens=70)

    # 2. TELEMETRY DASHBOARD
    print("\n" + "="*85)
    print(" === [KEP EXP-54 DUAL-REFACTORING TELEMETRY DASHBOARD] ===")
    print("="*85)
    print(f"{'Performance Metric':<36} | {'EXP-54 Unified Architecture Value':<40}")
    print("-" * 85)
    print(f"{'Final Steady-State Speech Loss':<36} | {final_loss:<40.4f} (PPL: {math.exp(final_loss):.2f})")
    print(f"{'Variational Free Energy (F_t)':<36} | {final_fe:<40.4f}")
    print(f"{'Throughput Speed':<36} | {throughput:<40.1f} tok/s")
    print(f"{'Peak VRAM Memory':<36} | {peak_vram_mb:<40.1f} MB")
    print("="*85)

    print("\n" + "="*85)
    print(" === [KEP RULE #4 LIVE DIAGNOSTIC THETA-PAC SPEECH AUDIT] ===")
    print("="*85)
    print(f"Prompt : \"{diag_prompt}\"")
    print(f"Output : \"{sample_text}\"")
    print("="*85 + "\n")


if __name__ == "__main__":
    run_exp_54_benchmark()
