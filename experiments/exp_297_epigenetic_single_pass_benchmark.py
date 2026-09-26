"""
EXP-297: Single-Pass Continual Learning with Epigenetic Methylation Locks (\mu_i) & Deep Sleep Replay.
Compliant with KEP Principle 15 (Epigenetic Morphogenesis & Methylation Locks) and Principle 24 (Single-Pass & Anti-Zubryoshka).

Biophysical Mechanisms:
1. Methylation Locks (\mu_i):
   Each synapse accumulates an importance weight \mu_i = \mu_i + (grad_i)^2 during low-surprise mastery.
   When high surprise (NA arousal) occurs, the effective gradient update is modulated:
   \Delta w_i = - \frac{\eta_eff}{1.0 + \lambda \cdot \mu_i} \cdot \frac{\partial F_t}{\partial w_i}
   This protects consolidated low-surprise synapses (Pointer attention / positional invariance)
   while leaving free, unmethylated synapses plastic to absorb new knowledge.
2. Deep Micro-Sleep Consolidation (25 Replay Cycles):
   Deep consolidation over high-surprise episodes with gentle learning rate and Tononi SHY downscaling.
"""

import sys, os
sys.path.insert(0, '.')
import math, random, time
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

import karyon_core
from multi_domain_benchmark import generate_multi_domain_suite

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EpigeneticMethylationOptimizer:
    """
    Epigenetic Synaptic Consolidation Engine (KEP Principle 15).
    Maintains methylation / importance locks \mu_i per parameter tensor.
    \Delta w_i = - \frac{\eta_eff}{1.0 + \lambda \cdot \mu_i} \cdot g_i
    """
    def __init__(self, params, base_lr=1e-3, lambda_methyl=10.0, weight_decay=1e-4):
        self.params = [p for p in params if p.requires_grad]
        self.base_lr = base_lr
        self.lambda_methyl = lambda_methyl
        self.weight_decay = weight_decay
        self.m = [torch.zeros_like(p.data) for p in self.params]
        self.v = [torch.zeros_like(p.data) for p in self.params]
        self.methylation_locks = [torch.zeros_like(p.data) for p in self.params]
        self.step_count = 0

    def update_methylation(self, loss_val, threshold=1.0):
        """When loss is low, increase methylation locks for active parameters (consolidate mastery)."""
        if loss_val < threshold:
            factor = max(0.0, (threshold - loss_val) / threshold)
            with torch.no_grad():
                for p, lock in zip(self.params, self.methylation_locks):
                    if p.grad is not None:
                        # Accumulate squared sensitivity
                        lock.add_(factor * (p.grad.data ** 2))

    def step(self, lr_mult=1.0):
        self.step_count += 1
        lr = self.base_lr * lr_mult
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue
                grad = p.grad.data
                
                # Weight decay
                if self.weight_decay != 0:
                    grad = grad + self.weight_decay * p.data
                    
                # Epigenetic Methylation Lock Resistance:
                # Highly methylated synapses resist gradient modification
                lock = self.methylation_locks[i]
                resistance = 1.0 + self.lambda_methyl * lock
                effective_grad = grad / resistance
                
                # Standard Adam second moments on effective grad
                m = self.m[i]
                v = self.v[i]
                m.mul_(beta1).add_(effective_grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(effective_grad, effective_grad, value=1 - beta2)
                
                m_hat = m / (1.0 - beta1 ** self.step_count)
                v_hat = v / (1.0 - beta2 ** self.step_count)
                
                update = lr * m_hat / (torch.sqrt(v_hat) + eps)
                p.data.sub_(update)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()


class EpigeneticSovereignMind(nn.Module):
    def __init__(self, vocab_size=258, dim=128, wave_steps=6, thinking_steps=3):
        super().__init__()
        self.dim = dim
        self.wave_steps = wave_steps
        self.thinking_steps = thinking_steps
        
        # 1. Universal Manifold Embedding
        self.emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(512, dim)
        
        # 2. Phase 1: Continuous Bidirectional Cellular Wave Attractor (Prompt Settling)
        self.wave_conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.wave_norm = nn.RMSNorm(dim)
        
        # 3. Causal Recurrent Core (Continuous Temporal SSD / GRU)
        self.recurrent_cell = nn.GRUCell(dim, dim)
        
        # 4. C++20 Continuous Saccadic Attractor Drift (C-SSD Engine)
        self.saccadic_drift = karyon_core.ContinuousSaccadicDrift(dim, 17, "cuda" if torch.cuda.is_available() else "cpu")
        
        # 5. C++20 Dynamic Morphic Deliberation Graph with Persistent States (AGN v6.0)
        self.morphic_graph = karyon_core.DynamicMorphicGraph(dim, "cuda" if torch.cuda.is_available() else "cpu")
        self.morphic_graph.add_node("stack_integrator", "LinearAccumulator", True, 1.0)
        self.morphic_graph.add_node("carry_latch", "SaturatedAttractor", True, 1.0)
        self.morphic_graph.add_node("conjunctive_gate", "BilinearMultiplicative", True, 1.0)
        self.morphic_graph.add_node("attractor_snap", "ContinuousHopfield", True, 1.0)
        
        # 6. Gaze & Copy Projection
        self.content_q = nn.Linear(dim, dim, bias=False)
        self.content_k = nn.Linear(dim, dim, bias=False)
        self.gaze_gate = nn.Linear(dim, 1, bias=True)
        self.copy_gate = nn.Linear(dim, 1, bias=True)
        self.gaze_proj = nn.Linear(dim * 2, dim)
        self.norm = nn.RMSNorm(dim)
        
        # 7. Motor Readout
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.emb.weight

    def settle_prompt_field(self, p_tokens):
        B, L = p_tokens.shape
        pos = torch.arange(L, device=p_tokens.device).unsqueeze(0).expand(B, L)
        x = self.emb(p_tokens) + self.pos_emb(pos)
        
        h = x
        for _ in range(self.wave_steps):
            dh = self.wave_conv(h.transpose(1, 2)).transpose(1, 2)
            h = self.wave_norm(h + torch.tanh(dh))
        return h

    def step_cognitive_loop(self, x_t, h_core, p_field, p_tokens, bump):
        B, L, D = p_field.shape
        
        # A. Causal Temporal Step
        h_core = self.recurrent_cell(x_t, h_core)
        
        # B. C++20 Continuous Saccadic Attractor Drift
        drifted_bump, v_t = self.saccadic_drift(bump, h_core, 0.1)
        
        # C. Continuous Content Resonance
        q = self.content_q(h_core).unsqueeze(1)
        k = self.content_k(p_field)
        content_scores = torch.bmm(q, k.transpose(1, 2)).squeeze(1) / math.sqrt(D)
        content_bump = F.softmax(content_scores * 10.0, dim=-1)
        
        # D. Continuous Gaze Blending
        alpha_gaze = torch.sigmoid(self.gaze_gate(h_core))
        next_bump = alpha_gaze * drifted_bump + (1.0 - alpha_gaze) * content_bump
        next_bump = next_bump / (next_bump.sum(dim=-1, keepdim=True) + 1e-6)
        
        # E. Continuous Field Readout
        h_gaze = torch.bmm(next_bump.unsqueeze(1), p_field).squeeze(1)
        
        # F. C++20 Dynamic Morphic Thinking Recirculation (Persistent across sequence)
        h_sensory = self.norm(h_core + self.gaze_proj(torch.cat([h_core, h_gaze], dim=-1)))
        h_deliberated = self.morphic_graph(h_sensory, self.thinking_steps)
        h_fused = self.norm(h_sensory + h_deliberated)
        
        # G. Continuous Copy Projection
        p_copy = torch.sigmoid(self.copy_gate(h_fused))
        copy_logits = torch.zeros(B, 258, device=p_field.device)
        copy_logits.scatter_add_(1, p_tokens, next_bump)
        
        return h_fused, h_core, next_bump, p_copy, copy_logits

    def forward_loss(self, prompt_tokens, target_tokens):
        B, P_len = prompt_tokens.shape
        _, T_len = target_tokens.shape
        
        p_field = self.settle_prompt_field(prompt_tokens)
        
        bump = torch.zeros(B, P_len, device=prompt_tokens.device)
        bump[:, -1] = 1.0
        bump = F.softmax(bump * 10.0, dim=-1)
        
        self.morphic_graph.reset_state() # Reset graph state at start of sequence
        h_core = p_field[:, -1, :]
        inputs = torch.cat([prompt_tokens[:, -1:], target_tokens[:, :-1]], dim=1)
        
        all_logits = []
        for t in range(T_len):
            x_t = self.emb(inputs[:, t])
            h_fused, h_core, bump, p_copy, copy_logits = self.step_cognitive_loop(
                x_t, h_core, p_field, prompt_tokens, bump
            )
            gen_logits = self.head(h_fused)
            fused_probs = (1.0 - p_copy) * F.softmax(gen_logits, dim=-1) + p_copy * copy_logits
            log_probs = torch.log(fused_probs + 1e-8)
            all_logits.append(log_probs)
            
        log_probs_stack = torch.stack(all_logits, dim=1)
        loss = F.nll_loss(log_probs_stack.reshape(-1, 258), target_tokens.reshape(-1))
        return loss

    @torch.no_grad()
    def generate(self, prompt_tokens, max_gen_len=20):
        B, P_len = prompt_tokens.shape
        p_field = self.settle_prompt_field(prompt_tokens)
        
        bump = torch.zeros(B, P_len, device=prompt_tokens.device)
        bump[:, -1] = 1.0
        bump = F.softmax(bump * 10.0, dim=-1)
        
        self.morphic_graph.reset_state() # Reset graph state at start of sequence
        h_core = p_field[:, -1, :]
        curr_token = prompt_tokens[:, -1]
        
        generated = []
        for _ in range(max_gen_len):
            x_t = self.emb(curr_token)
            h_fused, h_core, bump, p_copy, copy_logits = self.step_cognitive_loop(
                x_t, h_core, p_field, prompt_tokens, bump
            )
            gen_logits = self.head(h_fused)
            fused_probs = (1.0 - p_copy) * F.softmax(gen_logits, dim=-1) + p_copy * copy_logits
            next_token = fused_probs.argmax(dim=-1)
            generated.append(next_token)
            curr_token = next_token
            if (curr_token == 10).all():
                break
        return torch.stack(generated, dim=1)


def run_epigenetic_single_pass_benchmark():
    print("=" * 80)
    print("⚡ RUNNING EXP-297: EPIGENETIC METHYLATION LOCKS & DEEP MICRO-SLEEP")
    print("=" * 80)
    
    suite = generate_multi_domain_suite(seed=42)
    model = EpigeneticSovereignMind(vocab_size=258, dim=128, wave_steps=6, thinking_steps=3).to(device)
    
    params = list(model.parameters())
    for p in model.saccadic_drift.parameters():
        params.append(p)
    for p in model.morphic_graph.parameters():
        params.append(p)
        
    base_lr = 1e-3
    optimizer = EpigeneticMethylationOptimizer(params, base_lr=base_lr, lambda_methyl=20.0, weight_decay=1e-4)
    
    # Flatten stream samples for N=1 streaming
    stream_samples = []
    for domain_name, samples in suite.items():
        for item in samples[:300]:
            stream_samples.append((domain_name, item))
            
    random.shuffle(stream_samples)
    
    batch_size = 32
    episodic_surprise_buffer = [] # Buffer for micro-sleep replay
    surprise_threshold = 1.2 # nats
    
    print(f"Streaming {len(stream_samples)} samples in a strict Single-Pass sequence (N=1)...")
    print("Epigenetic Methylation Locks ACTIVE: \\mu_i shields consolidated weights against catastrophic overwrite.")
    print("Deep Micro-Sleep ACTIVE: 25 replay cycles every 50 steps.\n")
    
    batches = 0
    total_waking_loss = 0.0
    sleep_cycles = 0
    
    for i in range(0, len(stream_samples), batch_size):
        batch = stream_samples[i:i+batch_size]
        prompts = [list(item[1][0].encode('utf-8')) for item in batch]
        targets = [list(item[1][2][len(item[1][0]):].encode('utf-8')) for item in batch]
        
        max_p = max(len(p) for p in prompts)
        max_t = max(len(t) for t in targets)
        
        p_pad = torch.full((len(batch), max_p), 256, dtype=torch.long, device=device)
        t_pad = torch.full((len(batch), max_t), 256, dtype=torch.long, device=device)
        
        for b_idx, (p, t) in enumerate(zip(prompts, targets)):
            p_pad[b_idx, -len(p):] = torch.tensor(p, dtype=torch.long, device=device)
            t_pad[b_idx, :len(t)] = torch.tensor(t, dtype=torch.long, device=device)
            
        model.train()
        
        # Forward pass & loss
        loss = model.forward_loss(p_pad, t_pad)
        loss_val = loss.item()
        total_waking_loss += loss_val
        batches += 1
        
        # 1. Error-Gated Neuromodulated Plasticity
        eff_multiplier = 1.0 + 2.5 * min(max(loss_val / surprise_threshold, 0.0), 3.0)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        
        # Update methylation locks on low surprise to harden mastered representations
        optimizer.update_methylation(loss_val, threshold=surprise_threshold)
        
        # Step with epigenetic resistance
        optimizer.step(lr_mult=eff_multiplier)
        
        # Capture high-surprise samples into Episodic Memory Buffer
        if loss_val > surprise_threshold:
            episodic_surprise_buffer.append((p_pad.detach(), t_pad.detach()))
            if len(episodic_surprise_buffer) > 120:
                episodic_surprise_buffer.pop(0) # Keep buffer bounded
                
        # Total methylation audit
        total_methyl = sum(lock.sum().item() for lock in optimizer.methylation_locks)
        
        if batches % 10 == 0:
            print(f"Step {batches:02d} | Waking Loss: {loss_val:.4f} nats | NA Boost: {eff_multiplier:.2f}x | Total Synaptic Methylation: {total_methyl:.2f} | Buffer: {len(episodic_surprise_buffer)}")
            
        # 2. Deep Micro-Sleep Phase Consolidation (Every 50 steps, 25 Replay Cycles)
        if batches % 50 == 0 and len(episodic_surprise_buffer) > 0:
            sleep_cycles += 1
            print(f"\n💤 ENTERING DEEP MICRO-SLEEP #{sleep_cycles} (Consolidating {len(episodic_surprise_buffer)} high-surprise memories across 25 cycles)...")
            
            sleep_replay_steps = min(len(episodic_surprise_buffer), 25)
            sleep_loss_sum = 0.0
            
            for replay_idx in range(sleep_replay_steps):
                s_p, s_t = random.choice(episodic_surprise_buffer)
                optimizer.zero_grad()
                s_loss = model.forward_loss(s_p, s_t)
                s_loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                
                # Consolidate during sleep
                optimizer.update_methylation(s_loss.item(), threshold=surprise_threshold)
                # Gentle learning rate during sleep replay
                optimizer.step(lr_mult=0.4)
                sleep_loss_sum += s_loss.item()
                
            # Tononi SHY Synaptic Downscaling
            with torch.no_grad():
                for p in model.parameters():
                    if p.requires_grad and p.dim() > 1:
                        p.mul_(0.999) # 0.1% gentle synaptic downscaling
                        
            print(f"   Deep Sleep Replay Avg Loss: {sleep_loss_sum / sleep_replay_steps:.4f} nats | Tononi SHY Downscaling Complete.\n")

    print("\n" + "=" * 80)
    print("📊 EVALUATING GENERALIZATION POST EPIGENETIC SINGLE-PASS & DEEP SLEEP")
    print("=" * 80)
    
    model.eval()
    for domain_name, samples in suite.items():
        eval_samples = samples[300:400]
        correct = 0
        for expr, expected_ans, full in eval_samples:
            p_bytes = list(expr.encode('utf-8'))
            p_t = torch.tensor([p_bytes], dtype=torch.long, device=device)
            gen_bytes = model.generate(p_t, max_gen_len=len(expected_ans) + 4)
            gen_str = bytes(gen_bytes[0].cpu().tolist()).decode('utf-8', errors='ignore').strip()
            if expected_ans.strip() in gen_str:
                correct += 1
        acc = (correct / len(eval_samples)) * 100.0
        print(f"Domain [{domain_name:10s}] Single-Pass Accuracy: {acc:6.2f}% ({correct}/{len(eval_samples)})")
    print("=" * 80)

if __name__ == '__main__':
    run_epigenetic_single_pass_benchmark()
