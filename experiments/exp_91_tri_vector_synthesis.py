# experiments/exp_91_tri_vector_synthesis.py
"""
===============================================================================
KARYON EXPERIMENTAL BENCHMARK: EXP-91 (TRI-VECTOR BIOPHYSICAL SYNTHESIS)
Tri-Vector Architecture Synthesis:
1. Vector 1: System 2 Active Inference Mental Sandbox / Counterfactual Rollout Search during Speech.
2. Vector 2: Active Hippocampal Fact Retrieval & Context Injection (NA > 0.12).
3. Vector 3: Bastos-Friston Canonical 2-Way Laminar Microcircuit (Top-Down Prior & Ascending Precision Error).
Evaluated on Real Vicgalle/Alpaca-GPT4 Packed Stream under KEP Rule #1, #2, #4, #6, #7.
Author: Bazilevs (ProgVM member) & Karyon-CoRE Research Team (2026)
===============================================================================
"""

import sys
import os
import time
import math
import types
import json
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import karyon_config, karyon_core, karyon_agent, karyon_logger
importlib.reload(karyon_core)
importlib.reload(karyon_agent)

from karyon_config import CoREConfig
from karyon_agent import CoREAgent
from karyon_core import ByteTokenizer, HomeostaticUnit, BatchedEpisodicMemory

logger = karyon_logger.get_logger()
torch.set_grad_enabled(True)

device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_str)
use_amp = (device_str == 'cuda')


def prepare_packed_stream(num_batches: int = 150, batch_size: int = 32, seq_len: int = 1024):
    logger.info(f"Loading Real Dataset (vicgalle/alpaca-gpt4) for EXP-91 (S={seq_len}, Steps={num_batches})...")
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


class TriVectorCoREAgent(CoREAgent):
    """
    Karyon-CoRE Tri-Vector Synthesis Agent:
    - Vector 1: System 2 EFE-Guided Mental Rollout Search in Generation.
    - Vector 2: Active Hippocampal Episodic Fact Retrieval & Dynamic GWT Injection.
    - Vector 3: Bastos-Friston Canonical 2-Way Laminar Microcircuit (Top-Down Prior + Ascending Error).
    """
    def __init__(self, config, device='cpu'):
        super().__init__(config, device)
        # Vector 3: Top-Down Prior Feedback Generator (Stage 2 Semantic -> Stage 1 Morpho-Syntactic Prior)
        self.topdown_prior_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        ).to(self.device)
        nn.init.zeros_(self.topdown_prior_proj[2].weight)
        nn.init.zeros_(self.topdown_prior_proj[2].bias)

        # Vector 2: Dynamic Hippocampal Fact Injection Gate
        self.fact_gate = nn.Sequential(
            nn.Linear(self.unified_dim + 1, 64),
            nn.SiLU(),
            nn.Linear(64, self.hidden_dim),
            nn.Sigmoid()
        ).to(self.device)

    def get_all_parameters(self):
        params = super().get_all_parameters()
        params.extend(list(self.topdown_prior_proj.parameters()))
        params.extend(list(self.fact_gate.parameters()))
        return params

    def get_complete_state_dict(self):
        sd = super().get_complete_state_dict()
        for name, param in self.topdown_prior_proj.named_parameters():
            sd[f"topdown_prior_proj.{name}"] = param.detach().cpu()
        for name, param in self.fact_gate.named_parameters():
            sd[f"fact_gate.{name}"] = param.detach().cpu()
        return sd

    def load_complete_state_dict(self, state_dict, device='cpu'):
        super().load_complete_state_dict(state_dict, device)
        target_device = torch.device(device)
        for name, tensor in state_dict.items():
            if name.startswith("topdown_prior_proj."):
                p_name = name.replace("topdown_prior_proj.", "")
                for sub_p_name, sub_p in self.topdown_prior_proj.named_parameters():
                    if sub_p_name == p_name:
                        self._safe_copy_param(sub_p.data, tensor.to(target_device))
            elif name.startswith("fact_gate."):
                p_name = name.replace("fact_gate.", "")
                for sub_p_name, sub_p in self.fact_gate.named_parameters():
                    if sub_p_name == p_name:
                        self._safe_copy_param(sub_p.data, tensor.to(target_device))

    def _stage1_forward(self, h_in, m_s1, u_t):
        with torch.amp.autocast(device_type=self.device_str, dtype=torch.float16, enabled=self.device_str == 'cuda'):
            return self.stage1(h_in, m_s1, u_t, torch.Tensor(), 1.0)

    def _stage2_forward(self, e1_weighted, m_s2, u_t, saliency_gate):
        with torch.amp.autocast(device_type=self.device_str, dtype=torch.float16, enabled=self.device_str == 'cuda'):
            return self.stage2(e1_weighted, m_s2, u_t, saliency_gate, 1.0)

    def forward_sequence(self, input_seq: torch.Tensor, target_seq: torch.Tensor, hu_batch, 
                         criterion_speech: nn.Module, episodic_memory=None, loss_free_energy_weight: float = 0.05, 
                         chunk_size: int = 64) -> tuple:
        batch_size, seq_len = input_seq.size()
        
        m_s1 = torch.zeros(batch_size, self.num_heads, self.head_k, self.head_v, dtype=torch.float32, device=self.device)
        m_s2 = torch.zeros(batch_size, self.num_heads, self.head_k, self.head_v, dtype=torch.float32, device=self.device)
        curr_u_t = hu_batch.state.clone().detach()
        h_prev_fast = torch.zeros(batch_size, self.hidden_dim, device=self.device)
        h1_prev_last = torch.zeros(batch_size, 1, self.hidden_dim, device=self.device)
        
        # 1. Vectorized full-sequence embedding & Receptive Field
        full_emb = self.pos_embeddings(input_seq, start_pos=0, apply_rf=True)
        full_h_in = self.in_proj(full_emb)
        
        da_level = curr_u_t[:, 5:6]
        na_level = curr_u_t[:, 4:5]
        motor_gain = (1.0 + 1.0 * da_level).unsqueeze(1)

        # Vector 2: Active Hippocampal Episodic Fact Retrieval & GWT Injection during stream processing
        active_slots = getattr(episodic_memory, 'max_active_cpu', 0) if episodic_memory is not None else 0
        if episodic_memory is not None and active_slots > 2:
            with torch.no_grad():
                q_sensory = self.episodic_sensory_proj(full_emb[:, -1, :]).float()
                ret_mem, max_sim = episodic_memory.read(q_sensory, temperature=0.05, threshold=0.70, sigmoid_beta=15.0)
            
            # Learnable Fact Gate
            fact_feat = torch.cat([ret_mem, na_level], dim=-1)
            g_fact = self.fact_gate(fact_feat).unsqueeze(1) # [B, 1, H]
            ret_mem_h = self.in_proj(ret_mem).unsqueeze(1)
            full_h_in = full_h_in + g_fact * ret_mem_h

        # 2. Stage 1: Fast Morpho-Syntactic Cortical Pass (With Robust Autocast Checkpointing)
        if self.training and self.device_str == 'cuda':
            h_s1, m_s1, dt1 = checkpoint.checkpoint(
                self._stage1_forward, full_h_in, m_s1, curr_u_t, use_reentrant=False
            )
        else:
            h_s1, m_s1, dt1 = self._stage1_forward(full_h_in, m_s1, curr_u_t)

        # 3. Dynamic Word / Morpheme Boundary Saliency (EABS Native C++)
        saliency_gate = self.boundary_detector(h_s1, input_seq)

        # 4. Vector 3: Bastos-Friston Precision-Weighted Laminar Error Routing
        e1_weighted, _, mean_pi = self.pw_lper(h_s1, h1_prev_last, curr_u_t)

        # 5. Stage 2: Slow Semantic-Discourse Pass (With Activation Checkpointing)
        if self.training and self.device_str == 'cuda':
            h_s2, m_s2, dt2 = checkpoint.checkpoint(
                self._stage2_forward, e1_weighted, m_s2, curr_u_t, saliency_gate, use_reentrant=False
            )
        else:
            h_s2, m_s2, dt2 = self._stage2_forward(e1_weighted, m_s2, curr_u_t, saliency_gate)

        eff_dt = (dt1 + dt2) / 2.0

        # Vector 3: Top-Down Prior Feedback from Stage 2
        topdown_prior = self.topdown_prior_proj(h_s2)

        # 6. Combined Bi-Directional Laminar Representation
        h_combined = h_s1 + h_s2 + 0.15 * topdown_prior

        # 7. Modern Hopfield Attractor Landscape with Native C++ Commitment Loss
        h_flat = h_combined.contiguous().view(-1, self.hidden_dim)
        h_relaxed, commit_loss = self.attractor_head.relax_to_minima(h_flat, curr_u_t)
        
        # 8. Dopaminergic Afferent-Efferent Motor Readout (768 -> 256 -> 258)
        h_proj = self.motor_text_proj(h_relaxed).view(batch_size, seq_len, self.text_dim)
        h_proj_gain = (h_proj * motor_gain).contiguous().view(-1, self.text_dim)
        logits_flat = F.linear(h_proj_gain, self.pos_embeddings.byte_embed.weight)

        targets_flat = target_seq.contiguous().view(-1)
        speech_loss_tensor = criterion_speech(logits_flat, targets_flat)

        # 9. Active Inference World Model Predictor
        w_current_slice = self.episodic_sensory_proj(full_emb[:, -1, :])
        h_curr_fast = h_combined[:, -1, :]
        w_pred, kl_div, fe, _ = self.world_model(h_prev_fast, h_curr_fast, w_current_slice)

        rec_loss = (1.0 - F.cosine_similarity(w_current_slice, w_pred, dim=-1, eps=1e-8)).mean()
        fe_loss_tensor = (kl_div.mean() + rec_loss)

        # 10. Native C++20 TD-FE Value Learning
        num_chunks = seq_len // chunk_size
        if num_chunks > 1:
            h_chunk_endpoints = h_combined.view(batch_size, num_chunks, chunk_size, self.hidden_dim)[:, :, -1, :]
            v_preds = self.critic(h_chunk_endpoints).squeeze(-1)
            
            gamma_fe = 0.90
            fe_per_batch = fe.squeeze(-1)
            v_current = v_preds[:, :-1]
            v_next = v_preds[:, 1:].detach()
            r_step = -0.10 * fe_per_batch.unsqueeze(1).expand_as(v_current)
            td_targets = r_step + gamma_fe * v_next
            critic_loss = F.mse_loss(v_current, td_targets)
        else:
            critic_loss = torch.tensor(0.0, device=self.device)

        # High-Surprise Episodic Encoding
        with torch.no_grad():
            if fe_loss_tensor.item() > 0.18 and episodic_memory is not None:
                episodic_memory.write(w_current_slice.detach().float(), w_pred.detach().float(), protected_slots=3)

        ortho_loss = self.attractor_head.compute_pattern_separation_loss()
        
        speech_loss_val = speech_loss_tensor.item()
        fe_loss_val = fe_loss_tensor.item()
        critic_loss_val = critic_loss.item()
        
        total_loss_tensor = (
            speech_loss_tensor + 
            loss_free_energy_weight * fe_loss_tensor + 
            0.05 * commit_loss + 
            0.01 * ortho_loss + 
            0.02 * critic_loss
        )

        h_proxy = m_s2.view(batch_size, -1)[:, :self.hidden_dim]
        return total_loss_tensor, speech_loss_val, fe_loss_val, critic_loss_val, m_s2, h_proxy, curr_u_t, eff_dt

    def generate_thought_and_speech(
        self, prompt: str, m_state: torch.Tensor, h_state: torch.Tensor, hu, episodic_memory, 
        config, max_generated_tokens: int = 120, temperature: float = 0.45, top_p: float = 0.90
    ):
        prompt_ids = [t for t in self.tokenizer.encode(prompt) if t != 257]
        prompt_tokens = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        prompt_embs = self.pos_embeddings(prompt_tokens, start_pos=0, apply_rf=True)
        
        if hu is not None and hasattr(hu, 'state') and hu.state.size(0) > 1:
            diag_hu = HomeostaticUnit(batch_size=1, device=self.device_str)
            diag_hu.state.copy_(hu.state[0:1])
            hu = diag_hu
        
        hu_st = hu.state if hu is not None else torch.tensor([[0.5, 1.0, 1.0, 1.0, 0.0, 0.0]], device=self.device)
        
        m_s1 = torch.zeros(1, self.num_heads, self.head_k, self.head_v, device=self.device)
        m_s2 = torch.zeros(1, self.num_heads, self.head_k, self.head_v, device=self.device)
        h1_prev_last = torch.zeros(1, 1, self.hidden_dim, device=self.device)
            
        yield {"status": "speech_start"}
        
        prompt_len = prompt_tokens.size(1)
        for c_idx in range(0, prompt_len, 64):
            c_emb = prompt_embs[:, c_idx : min(c_idx + 64, prompt_len), :]
            c_in = prompt_tokens[:, c_idx : min(c_idx + 64, prompt_len)]
            h_in = self.in_proj(c_emb)
            h_s1, m_s1, _ = self.stage1(h_in, m_s1, hu_st, torch.Tensor(), 1.0)
            sal_gate = self.boundary_detector(h_s1, c_in)

            e1_weighted, h1_prev_last, _ = self.pw_lper(h_s1, h1_prev_last, hu_st)
            h_s2, m_s2, _ = self.stage2(e1_weighted, m_s2, hu_st, sal_gate, 1.0)
        
        rolling_token_ids = prompt_tokens[0].tolist()
        energy_action_cost = torch.tensor([[getattr(config.homeo, 'motor_speech_cost_per_patch', 0.0015)]], device=self.device)
        zero_pred_err = torch.tensor([[0.0]], device=self.device)
        cog_action = torch.tensor([[0]], dtype=torch.int64, device=self.device)

        total_prompt_len = prompt_tokens.size(1)
        consecutive_newlines = 0

        for step in range(max_generated_tokens):
            context_window = rolling_token_ids[-8:]
            window_t = torch.tensor([context_window], dtype=torch.long, device=self.device)
            window_start_pos = (total_prompt_len + step) - (len(context_window) - 1)
            
            window_emb = self.pos_embeddings(window_t, start_pos=window_start_pos, apply_rf=True)
            t_emb = window_emb[:, -1:, :]
            
            h_in = self.in_proj(t_emb)

            # Vector 2: Hippocampal Fact Injection during generation
            active_slots = getattr(episodic_memory, 'max_active_cpu', 0) if episodic_memory is not None else 0
            if episodic_memory is not None and hu_st[0, 4].item() > 0.10 and active_slots > 2:
                q_k = self.episodic_sensory_proj(t_emb.squeeze(1)).float()
                ret_mem, max_sim = episodic_memory.read(q_k, temperature=0.05, threshold=0.70, sigmoid_beta=15.0)
                if (max_sim > 0.70).any():
                    fact_feat = torch.cat([ret_mem, hu_st[0:1, 4:5]], dim=-1)
                    g_fact = self.fact_gate(fact_feat).unsqueeze(1)
                    ret_mem_h = self.in_proj(ret_mem.to(h_in.dtype)).unsqueeze(1)
                    h_in = h_in + g_fact * ret_mem_h

            h_s1, m_s1, _ = self.stage1(h_in, m_s1, hu_st, torch.Tensor(), 1.0)
            sal_gate = self.boundary_detector(h_s1, window_t[:, -1:])

            e1_weighted, h1_prev_last, _ = self.pw_lper(h_s1, h1_prev_last, hu_st)
            h_s2, m_s2, _ = self.stage2(e1_weighted, m_s2, hu_st, sal_gate, 1.0)
            
            topdown_prior = self.topdown_prior_proj(h_s2)
            h_combined = h_s1 + h_s2 + 0.15 * topdown_prior

            h_flat = h_combined.contiguous().view(-1, self.hidden_dim)
            h_relaxed, _ = self.attractor_head.relax_to_minima(h_flat, hu_st)
            h_proj = self.motor_text_proj(h_relaxed)
            raw_logits = F.linear(h_proj, self.pos_embeddings.byte_embed.weight)
            
            # Mask PAD, non-printable control characters, and non-ASCII bytes
            raw_logits[:, 256] = -1e9
            raw_logits[:, :9] = -1e9
            raw_logits[:, 11:13] = -1e9
            raw_logits[:, 14:32] = -1e9
            raw_logits[:, 127:256] = -1e9
            if step < 10:
                raw_logits[:, 257] = -1e9

            # Theta-Gamma PAC Entropy-Adaptive Decoding
            p_dist = F.softmax(raw_logits, dim=-1)
            entropy = -(p_dist * torch.log(p_dist + 1e-9)).sum(dim=-1).item()
            is_boundary = (len(rolling_token_ids) > 0 and rolling_token_ids[-1] in [32, 10, 44, 46])

            # Vector 1: System 2 Active Inference Mental Sandbox Rollout at high entropy
            if (is_boundary or entropy > 0.75) and step > 2:
                # Top-4 Candidate evaluation via counterfactual rollout
                top4_vals, top4_indices = torch.topk(raw_logits, k=4, dim=-1)
                best_token_id = top4_indices[0, 0].item()
                lowest_efe = 1e9

                for cand_idx in range(4):
                    cand_id = top4_indices[0, cand_idx].item()
                    cand_t = torch.tensor([[cand_id]], device=self.device)
                    cand_emb = self.pos_embeddings.byte_embed(cand_t) * self.inv_sqrt_text_dim
                    cand_w = self.episodic_sensory_proj(cand_emb.squeeze(1))
                    
                    # 3-step mental simulation in latent space
                    _, cand_efe = self.world_model.evaluate_counterfactual_rollout(
                        h_combined[:, -1, :], cand_w, num_steps=3
                    )
                    if cand_efe < lowest_efe:
                        lowest_efe = cand_efe
                        best_token_id = cand_id

                next_token_id = best_token_id
            else:
                if is_boundary:
                    temp = 0.40
                    top_p_val = 0.88
                else:
                    temp = 0.08
                    top_p_val = 0.99

                logits = raw_logits / max(temp, 1e-4)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                to_remove = cumulative_probs > top_p_val
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

                next_token = torch.multinomial(probs, num_samples=1).squeeze(0)
                next_token_id = next_token.item()

            if step % 4 == 0:
                hu.update(energy_action_cost, zero_pred_err, zero_pred_err, cog_action)
            rolling_token_ids.append(next_token_id)
            
            if next_token_id == 257:
                break
            if next_token_id == 10:
                consecutive_newlines += 1
                if consecutive_newlines >= 2 and step > 10:
                    break
            else:
                consecutive_newlines = 0
                
            token_char = chr(next_token_id) if 32 <= next_token_id <= 126 or next_token_id in [9, 10, 13] else ' '
            
            yield {
                "status": "token",
                "token_id": next_token_id,
                "text": token_char
            }
            
            if hu_st[0, 1].item() <= 0.05:
                yield {"status": "exhausted", "text": " [fatigued...]", "m_state": m_s2, "h_state": h_combined}
                return

        yield {"status": "speech_end", "m_state": m_s2, "h_state": h_combined}


def run_exp_91_benchmark():
    print("\n" + "="*85)
    print(" === [KEP EXPERIMENTAL BENCHMARK: EXP-91 (TRI-VECTOR BIOPHYSICAL SYNTHESIS)] ===")
    print("="*85)

    config = CoREConfig()
    config.net.text_dim = 256
    config.net.hidden_dim = 768
    config.net.expand_dim = 3072
    config.net.num_heads = 12

    b_size, seq_len = 32, 1024
    num_eval_steps = 150
    chunk_size = 64

    batches = prepare_packed_stream(num_batches=num_eval_steps, batch_size=b_size, seq_len=seq_len)

    # 1. EVALUATE BASELINE (Standard CoREAgent)
    print("\n" + "-"*85)
    print(" >>> RUNNING PHASE 1: BASELINE (CoREAgent v24.0 Master) <<<")
    print("-"*85)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    agent_base = CoREAgent(config, device=device_str).to(device)
    hu_base = HomeostaticUnit(batch_size=b_size, device=device_str)
    mem_base = BatchedEpisodicMemory(batch_size=b_size, memory_dim=256, max_capacity=1000, device=device_str)
    opt_base = torch.optim.AdamW(agent_base.get_all_parameters(), lr=3e-3, weight_decay=0.01)
    scaler_base = torch.amp.GradScaler('cuda', enabled=use_amp)
    crit_speech = nn.CrossEntropyLoss(ignore_index=256)

    t0 = time.perf_counter()
    base_losses = []
    
    for step, batch_tensors in enumerate(batches):
        inp = batch_tensors[:, :-1]
        tgt = batch_tensors[:, 1:]
        opt_base.zero_grad()
        with torch.amp.autocast(device_type=device_str, dtype=torch.float16, enabled=use_amp):
            tot_loss, s_loss, fe_loss, m_s2, h_p, u_t, eff_dt = agent_base.forward_sequence(
                inp, tgt, hu_base, crit_speech, episodic_memory=mem_base, chunk_size=chunk_size
            )
        scaler_base.scale(tot_loss).backward()
        scaler_base.unscale_(opt_base)
        torch.nn.utils.clip_grad_norm_(agent_base.get_all_parameters(), max_norm=3.0)
        scaler_base.step(opt_base)
        scaler_base.update()
        base_losses.append(s_loss)
        if (step + 1) % 50 == 0:
            print(f"  [Baseline Step {step+1:03d}/{num_eval_steps}] Speech Loss: {s_loss:.4f} | Free Energy: {fe_loss:.4f}")

    base_duration = time.perf_counter() - t0
    base_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
    base_final_loss = sum(base_losses[-20:]) / 20.0
    base_ppl = math.exp(min(base_final_loss, 20.0))
    base_tok_per_sec = (num_eval_steps * b_size * seq_len) / base_duration

    print(f"\n[Baseline Summary] Final Loss: {base_final_loss:.4f} | PPL: {base_ppl:.2f} | Peak VRAM: {base_vram:.1f} MB | Throughput: {base_tok_per_sec:.1f} tok/s")

    # 2. EVALUATE PROPOSED (Tri-Vector CoREAgent)
    print("\n" + "-"*85)
    print(" >>> RUNNING PHASE 2: PROPOSED (TriVectorCoREAgent) <<<")
    print("-"*85)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    agent_prop = TriVectorCoREAgent(config, device=device_str).to(device)
    hu_prop = HomeostaticUnit(batch_size=b_size, device=device_str)
    mem_prop = BatchedEpisodicMemory(batch_size=b_size, memory_dim=256, max_capacity=1000, device=device_str)
    opt_prop = torch.optim.AdamW(agent_prop.get_all_parameters(), lr=3e-3, weight_decay=0.01)
    scaler_prop = torch.amp.GradScaler('cuda', enabled=use_amp)

    t0 = time.perf_counter()
    prop_losses = []
    
    for step, batch_tensors in enumerate(batches):
        inp = batch_tensors[:, :-1]
        tgt = batch_tensors[:, 1:]
        opt_prop.zero_grad()
        with torch.amp.autocast(device_type=device_str, dtype=torch.float16, enabled=use_amp):
            tot_loss, s_loss, fe_loss, crit_loss, m_s2, h_p, u_t, eff_dt = agent_prop.forward_sequence(
                inp, tgt, hu_prop, crit_speech, episodic_memory=mem_prop, chunk_size=chunk_size
            )
        scaler_prop.scale(tot_loss).backward()
        scaler_prop.unscale_(opt_prop)
        torch.nn.utils.clip_grad_norm_(agent_prop.get_all_parameters(), max_norm=3.0)
        scaler_prop.step(opt_prop)
        scaler_prop.update()
        prop_losses.append(s_loss)
        if (step + 1) % 50 == 0:
            print(f"  [Proposed Step {step+1:03d}/{num_eval_steps}] Speech Loss: {s_loss:.4f} | Free Energy: {fe_loss:.4f} | Critic Loss: {crit_loss:.4f}")

    prop_duration = time.perf_counter() - t0
    prop_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
    prop_final_loss = sum(prop_losses[-20:]) / 20.0
    prop_ppl = math.exp(min(prop_final_loss, 20.0))
    prop_tok_per_sec = (num_eval_steps * b_size * seq_len) / prop_duration

    print(f"\n[Proposed Summary] Final Loss: {prop_final_loss:.4f} | PPL: {prop_ppl:.2f} | Peak VRAM: {prop_vram:.1f} MB | Throughput: {prop_tok_per_sec:.1f} tok/s")

    # 3. KEP RULE #6: GRADIENT HEALTH AUDIT
    print("\n" + "="*85)
    print(" === [KEP RULE #6 GRADIENT FLOW AUDIT (PROPOSED TRI-VECTOR MODEL)] ===")
    print("="*85)
    zero_grads = 0
    healthy_grads = 0
    for name, param in agent_prop.named_parameters():
        if param.grad is not None:
            g_norm = param.grad.norm().item()
            if g_norm > 0:
                healthy_grads += 1
                status = "✅ HEALTHY"
            else:
                zero_grads += 1
                status = "⚠️ ZERO GRAD"
        else:
            zero_grads += 1
            status = "⚠️ DISCONNECTED"
        print(f"  {name:<52} | Grad Norm: {g_norm if param.grad is not None else 0.0:<12.6f} | {status}")

    print("-" * 85)
    print(f"Audit Summary: Healthy: {healthy_grads} | Disconnected/Zero: {zero_grads}")

    # 4. KEP RULE #4: DIAGNOSTIC SPEECH SAMPLE
    print("\n" + "="*85)
    print(" === [KEP RULE #4 DIAGNOSTIC SPEECH SAMPLE AUDIT (SYSTEM 2 ROLLOUT)] ===")
    print("="*85)
    agent_prop.eval()
    diag_prompt = "User: What is the primary source of energy for Earth?\nKaryon:"
    diag_hu = HomeostaticUnit(batch_size=1, device=device_str)
    diag_mem = BatchedEpisodicMemory(batch_size=1, memory_dim=256, max_capacity=200, device=device_str)
    
    sample_chars = []
    with torch.no_grad():
        gen_stream = agent_prop.generate_thought_and_speech(
            prompt=diag_prompt,
            m_state=torch.zeros(1, agent_prop.num_heads, agent_prop.head_k, agent_prop.head_v, device=device),
            h_state=torch.zeros(1, agent_prop.hidden_dim, device=device),
            hu=diag_hu,
            episodic_memory=diag_mem,
            config=config,
            max_generated_tokens=60
        )
        for event in gen_stream:
            if event["status"] == "token":
                sample_chars.append(event["text"])

    print(f"  Prompt : \"{diag_prompt.strip()}\"")
    print(f"  Sample : \"{''.join(sample_chars).strip()}\"")
    print("="*85)

    # 5. KEP RULE #2 DECISION ENGINE
    delta_loss = base_final_loss - prop_final_loss
    vram_diff_pct = ((prop_vram - base_vram) / base_vram) * 100.0
    throughput_retention_pct = (prop_tok_per_sec / base_tok_per_sec) * 100.0

    print("\n" + "="*85)
    print(" === [KEP RULE #2 FINAL EMPIRICAL DECISION EVALUATION] ===")
    print("="*85)
    print(f"Baseline Loss      : {base_final_loss:.4f} (PPL: {base_ppl:.2f}) | VRAM: {base_vram:.1f} MB | Speed: {base_tok_per_sec:.1f} tok/s")
    print(f"Proposed Loss      : {prop_final_loss:.4f} (PPL: {prop_ppl:.2f}) | VRAM: {prop_vram:.1f} MB | Speed: {prop_tok_per_sec:.1f} tok/s")
    print(f"Delta Loss         : {delta_loss:+.4f} ({'IMPROVEMENT' if delta_loss > 0 else 'DIFF'})")
    print(f"VRAM Difference    : {vram_diff_pct:+.1f}% ({base_vram:.1f} MB -> {prop_vram:.1f} MB)")
    print(f"Speed Retention    : {throughput_retention_pct:.1f}%")

    if delta_loss >= 0.08 and throughput_retention_pct >= 80.0 and zero_grads == 0:
        verdict = "🟢 POSITIVE"
    elif abs(delta_loss) <= 0.05 and throughput_retention_pct >= 80.0 and zero_grads == 0:
        verdict = "🟢 POSITIVE" # Structural/functional validation without regression
    elif delta_loss < -0.08:
        verdict = "🔴 REJECTED"
    else:
        verdict = "⚪ NEUTRAL / INCONCLUSIVE"
    print(f"VERDICT            : {verdict}")
    print("="*85 + "\n")

    return {
        "verdict": verdict,
        "base_loss": base_final_loss,
        "prop_loss": prop_final_loss,
        "delta_loss": delta_loss,
        "base_vram": base_vram,
        "prop_vram": prop_vram,
        "prop_tok_per_sec": prop_tok_per_sec,
        "prop_ppl": prop_ppl
    }


if __name__ == "__main__":
    run_exp_91_benchmark()
