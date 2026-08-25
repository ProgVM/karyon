# experiments/exp_55_deep_compositional_cortical_stack.py
"""
===============================================================================
KARYON EXPERIMENTAL BENCHMARK: EXP-55 (2-STAGE DEEP COMPOSITIONAL CORTICAL STACK)
Evaluating 2-Stage Cascaded Cortical Stack (Stage 1: Fast Morpho-Syntactic SSM ->
Stage 2: Slow Semantic-Discourse SSM -> Crisp Hopfield) on 10M Token Scaled Stream.
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
# 1. SPECIALIZED BALANCED SSD LAYER (MAMBA-2 GROUPNORM)
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

        return h_chunk, m_next


# =============================================================================
# 2. CORTICAL STAGE (SSD TIME-MIXING + SWIGLU CHANNEL-MIXING)
# =============================================================================
class CorticalStage(nn.Module):
    """
    Cascaded Cortical Stage:
    1. Pre-LayerNorm Parallel SSD Time-Mixing
    2. Pre-LayerNorm Parallel SwiGLU Channel-Mixing (2048D)
    3. Dual Residual Highway
    """
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
        # 1. Time-mixing with residual
        norm_x = self.pre_norm_ssd(x)
        h_ssd, m_next = self.ssd(norm_x, m_prev, u_t, dt)
        x_res1 = x + h_ssd.view_as(x)

        # 2. Channel-mixing with residual
        norm_res1 = self.pre_norm_swiglu(x_res1)
        h_swiglu = self.swiglu(norm_res1.contiguous().view(-1, x.size(-1)))
        x_out = x_res1 + h_swiglu.view_as(x_res1)

        return x_out, m_next


# =============================================================================
# 3. CRISP HOPFIELD ATTRACTOR HEAD
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
# 4. EXP-55 2-STAGE DEEP COMPOSITIONAL CORTICAL AGENT
# =============================================================================
class DeepCorticalStackAgent(nn.Module):
    def __init__(self, config: CoREConfig, device_str: str = 'cpu'):
        super().__init__()
        self.device_str = device_str
        self.device = torch.device(device_str)
        self.config = config
        self.text_dim = config.net.text_dim
        self.hidden_dim = config.net.hidden_dim
        self.expand_dim = config.net.expand_dim
        self.num_heads = config.net.num_heads
        self.head_k = 64
        self.head_v = 128
        self.text_gen_dim = 258
        self.inv_sqrt_text_dim = 1.0 / math.sqrt(self.text_dim)

        # 1. Byte Embedding + Receptive Field
        self.pos_embeddings = OffsetPositionalByteEmbedding(
            vocab_size=self.text_gen_dim, text_dim=self.text_dim, max_len=8192, device_str=device_str
        ).to(self.device)
        self.in_proj = nn.Linear(self.text_dim, self.hidden_dim).to(self.device)

        # 2. Stage 1: Fast Morpho-Syntactic Cortical Stage (alpha in [0.85, 0.995])
        self.stage1 = CorticalStage(
            hidden_dim=self.hidden_dim, expand_dim=self.expand_dim, num_heads=self.num_heads,
            head_k=self.head_k, head_v=self.head_v, min_beta=0.005, max_beta=0.15, device_str=device_str
        ).to(self.device)

        # 3. Stage 2: Slow Semantic-Discourse Cortical Stage (alpha in [0.95, 0.9999])
        self.stage2 = CorticalStage(
            hidden_dim=self.hidden_dim, expand_dim=self.expand_dim, num_heads=self.num_heads,
            head_k=self.head_k, head_v=self.head_v, min_beta=0.0001, max_beta=0.05, device_str=device_str
        ).to(self.device)

        # 4. Attractor & Readout
        self.attractor_head = CrispContinuousHopfieldHead(
            hidden_dim=self.hidden_dim, vocab_size=self.text_gen_dim,
            num_attractors=256, device_str=device_str
        ).to(self.device)

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

        num_chunks = max(1, seq_len // chunk_size)
        chunk_losses = []
        commit_losses = []

        for chunk_idx in range(num_chunks):
            c_start = chunk_idx * chunk_size
            c_end = min((chunk_idx + 1) * chunk_size, seq_len)
            chunk_in = input_seq[:, c_start:c_end]
            chunk_tgt = target_seq[:, c_start:c_end]

            chunk_emb = self.pos_embeddings(chunk_in, start_pos=c_start, apply_rf=True)
            h_in = self.in_proj(chunk_emb)

            # Deep Compositional Cascading: Stage 1 -> Stage 2
            h_s1, m_s1 = self.stage1(h_in, m_s1, curr_u_t, dt=1.0)
            h_s2, m_s2 = self.stage2(h_s1, m_s2, curr_u_t, dt=1.0)

            # Continuous Attractor Minima Relaxation
            h_flat = h_s2.contiguous().view(-1, self.hidden_dim)
            h_relaxed, commit_loss = self.attractor_head.relax_to_minima(h_flat, curr_u_t)

            h_proj = self.motor_text_proj(h_relaxed)
            logits_flat = F.linear(h_proj, self.pos_embeddings.byte_embed.weight) * self.inv_sqrt_text_dim
            targets_flat = chunk_tgt.contiguous().view(-1)

            loss = criterion(logits_flat, targets_flat)
            chunk_losses.append(loss)
            commit_losses.append(commit_loss)

            has_eos = (chunk_in == 257).any(dim=-1).view(batch_size, 1, 1, 1).float()
            m_s1 = (m_s1 * (1.0 - has_eos)).detach()
            m_s2 = (m_s2 * (1.0 - has_eos)).detach()

        avg_loss = torch.stack(chunk_losses).mean()
        avg_commit = torch.stack(commit_losses).mean()
        total_loss = avg_loss + 0.05 * avg_commit
        return total_loss, avg_loss.item()

    def generate_speech(self, prompt: str, max_tokens: int = 70) -> str:
        self.eval()
        tokenizer = ByteTokenizer()
        prompt_ids = tokenizer.encode(prompt)
        if prompt_ids and prompt_ids[-1] == 257:
            prompt_ids = prompt_ids[:-1]

        prompt_t = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        m_s1 = torch.zeros(1, self.num_heads, self.head_k, self.head_v, dtype=torch.float32, device=self.device)
        m_s2 = torch.zeros(1, self.num_heads, self.head_k, self.head_v, dtype=torch.float32, device=self.device)
        hu_state = torch.tensor([[0.5, 1.0, 1.0, 1.0, 0.0, 0.0]], device=self.device)

        prompt_emb = self.pos_embeddings(prompt_t, start_pos=0, apply_rf=True)
        prompt_len = prompt_t.size(1)

        # Process prompt in 64-byte chunks
        for c_idx in range(0, prompt_len, 64):
            c_emb = prompt_emb[:, c_idx : min(c_idx + 64, prompt_len), :]
            h_in = self.in_proj(c_emb)
            h_s1, m_s1 = self.stage1(h_in, m_s1, hu_state, 1.0)
            h_s2, m_s2 = self.stage2(h_s1, m_s2, hu_state, 1.0)

        rolling_ids = prompt_ids.copy()
        generated_chars = []

        with torch.no_grad():
            for step in range(max_tokens):
                context = rolling_ids[-4:]
                win_t = torch.tensor([context], dtype=torch.long, device=self.device)
                win_emb = self.pos_embeddings(win_t, start_pos=len(rolling_ids) - len(context), apply_rf=True)
                t_emb = win_emb[:, -1:, :]

                h_in = self.in_proj(t_emb)
                h_s1, m_s1 = self.stage1(h_in, m_s1, hu_state, 1.0)
                h_s2, m_s2 = self.stage2(h_s1, m_s2, hu_state, 1.0)

                h_flat = h_s2.contiguous().view(-1, self.hidden_dim)
                h_rel, _ = self.attractor_head.relax_to_minima(h_flat, hu_state)
                h_proj = self.motor_text_proj(h_rel)

                raw_logits = F.linear(h_proj, self.pos_embeddings.byte_embed.weight) * self.inv_sqrt_text_dim
                raw_logits[:, 256:] = -1e9

                p_dist = F.softmax(raw_logits, dim=-1)
                entropy = -(p_dist * torch.log(p_dist + 1e-9)).sum(dim=-1).item()
                is_boundary = (len(rolling_ids) > 0 and rolling_ids[-1] in [32, 10, 44, 46])

                if is_boundary or entropy > 0.60:
                    temp = 0.40
                    top_p = 0.90
                else:
                    temp = 0.05
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
# 5. REAL DATASET (S=1024, 300 STEPS = 9.8M TOKENS)
# =============================================================================
def prepare_packed_stream(num_batches: int = 300, batch_size: int = 32, seq_len: int = 1024):
    logger.info(f"Loading Real Dataset (vicgalle/alpaca-gpt4) for EXP-55 (S={seq_len}, Steps={num_batches})...")
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

def run_exp_55_benchmark():
    print("\n" + "="*85)
    print(" === [KEP EXPERIMENTAL BENCHMARK: EXP-55 (2-STAGE DEEP CORTICAL STACK)] ===")
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

    # 1. EVALUATE 2-STAGE DEEP CORTICAL STACK
    print("\n[1/1] Training 2-Stage Deep Compositional Cortical Stack (300 Steps on S=1024)...")
    torch.manual_seed(42)
    agent = DeepCorticalStackAgent(config, device_str=device_str).to(device)
    optimizer = torch.optim.AdamW(agent.parameters(), lr=3e-3, weight_decay=0.01)
    hu = HomeostaticUnit(batch_size=b_size, device=device_str)

    # Cosine LR Scheduler with Warmup
    warmup_steps = 30
    def lr_schedule(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, num_eval_steps - warmup_steps))
        return max(0.033, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)

    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()
    losses = []

    for step in range(num_eval_steps):
        batch = batches[step]
        input_s = batch[:, :-1]
        target_s = batch[:, 1:]

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device_str, dtype=torch.float16, enabled=use_amp):
            tot_loss, speech_loss = agent.forward_sequence(input_s, target_s, hu, criterion, chunk_size=chunk_size)

        scaler.scale(tot_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=3.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        losses.append(speech_loss)

        if (step + 1) % 50 == 0 or step == num_eval_steps - 1:
            cur_loss = sum(losses[-20:]) / min(len(losses), 20)
            cur_lr = optimizer.param_groups[0]['lr']
            print(f"  Step [{step+1:03d}/{num_eval_steps}] | Speech Loss: {speech_loss:.4f} (Avg: {cur_loss:.4f}, PPL: {math.exp(cur_loss):.2f}) | LR: {cur_lr:.6f}")

    if device.type == 'cuda': torch.cuda.synchronize()
    total_time_sec = time.perf_counter() - t_start
    peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if device.type == 'cuda' else 0.0
    throughput = (num_eval_steps * b_size * seq_len) / total_time_sec
    final_loss = sum(losses[-30:]) / 30.0
    sample_text = agent.generate_speech(diag_prompt, max_tokens=70)

    # 2. TELEMETRY DASHBOARD
    print("\n" + "="*85)
    print(" === [KEP EXP-55 DEEP CORTICAL STACK TELEMETRY DASHBOARD] ===")
    print("="*85)
    print(f"{'Performance Metric':<36} | {'EXP-55 2-Stage Deep Stack Value':<40}")
    print("-" * 85)
    print(f"{'Initial Loss (Step 1)':<36} | {losses[0]:<40.4f}")
    print(f"{'Midpoint Loss (Step 150)':<36} | {losses[149]:<40.4f}")
    print(f"{'Final Steady-State Loss (Step 300)':<36} | {final_loss:<40.4f} (PPL: {math.exp(final_loss):.2f})")
    print(f"{'Total Loss Drop (Delta)':<36} | {final_loss - losses[0]:<40.4f}")
    print(f"{'Throughput Speed':<36} | {throughput:<40.1f} tok/s")
    print(f"{'Peak VRAM Memory':<36} | {peak_vram_mb:<40.1f} MB")
    print(f"{'Total Training Time':<36} | {total_time_sec:<40.2f} sec ({total_time_sec/60.0:.1f} min)")
    print("="*85)

    print("\n" + "="*85)
    print(" === [KEP RULE #4 LIVE DIAGNOSTIC SPEECH SAMPLE AUDIT] ===")
    print("="*85)
    print(f"Prompt : \"{diag_prompt}\"")
    print(f"Output : \"{sample_text}\"")
    print("="*85 + "\n")


if __name__ == "__main__":
    run_exp_55_benchmark()
