"""
Full Cognitive Benchmark: C++20 C-SSD (Continuous Saccadic Dynamics)
Coupled with Dynamic Morphic Deliberation:
1. Continuous Saccadic Attractor Drift (C-SSD in C++20 LibTorch)
2. LinearAccumulatorOp (Continuous Stack Depth Integrator for Dyck-1)
3. SaturatedAttractorOp (Bi-stable Carry Latch for Arithmetic Addition)
4. BilinearMultiplicativeOp (Conjunctive Gating)
5. ContinuousHopfieldOp (Discrete Attractor Snapping)

KEP Principle 2, 21, 22 compliant (100% Continuous Recurrent Dynamic Substrate).
"""

import sys, os
sys.path.insert(0, '.')
import math, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F

import karyon_core
from multi_domain_benchmark import generate_multi_domain_suite

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# INTEGRATED SOVEREIGN C-SSD COGNITIVE ENGINE
# ==============================================================================

class IntegratedSovereignCognitiveMind(nn.Module):
    def __init__(self, vocab_size=258, dim=128, wave_steps=6, thinking_steps=4):
        super().__init__()
        self.dim = dim
        self.wave_steps = wave_steps
        self.thinking_steps = thinking_steps
        
        # 1. Manifold Embedding
        self.emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(512, dim)
        
        # 2. Phase 1: Continuous Bidirectional Cellular Waves (Prompt Settling)
        self.wave_conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.wave_norm = nn.RMSNorm(dim)
        
        # 3. Causal Recurrent Core (Continuous Temporal SSD / GRU)
        self.recurrent_cell = nn.GRUCell(dim, dim)
        
        # 4. C++20 Continuous Saccadic Attractor Drift (C-SSD Engine)
        self.saccadic_drift = karyon_core.ContinuousSaccadicDrift(dim, 17, "cuda" if torch.cuda.is_available() else "cpu")
        
        # 5. C++20 Dynamic Morphic Deliberation Graph (AGN v6.0)
        # Equipped with LinearAccumulator (Stack depth) + SaturatedAttractor (Carry latch)
        self.morphic_graph = karyon_core.DynamicMorphicGraph(dim, "cuda" if torch.cuda.is_available() else "cpu")
        self.morphic_graph.add_node("stack_integrator", "LinearAccumulator", True, 1.0)
        self.morphic_graph.add_node("carry_latch", "SaturatedAttractor", True, 1.0)
        self.morphic_graph.add_node("conjunctive_gate", "BilinearMultiplicative", True, 1.0)
        self.morphic_graph.add_node("attractor_snap", "ContinuousHopfield", True, 1.0)
        
        # 6. Saccadic Content & Copy Manifold
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

    def step_cognitive_loop(self, x_t, h_core, p_field, p_tokens, bump, u_t=None):
        B, L, D = p_field.shape
        
        # A. Causal Temporal Step
        h_core = self.recurrent_cell(x_t, h_core)
        
        # B. C++20 Continuous Saccadic Attractor Drift (Wave Gaze)
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
        
        # F. C++20 Dynamic Morphic Thinking Recirculation (Stack + Carry + Conjunctive Logic)
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


# ==============================================================================
# EMPIRICAL VALIDATION HARNESS
# ==============================================================================

def run_integrated_benchmark():
    print("=" * 80)
    print("⚡ RUNNING FULL C++20 C-SSD & DYNAMIC MORPHIC COGNITIVE BENCHMARK")
    print("=" * 80)
    
    suite = generate_multi_domain_suite(seed=42)
    model = IntegratedSovereignCognitiveMind(vocab_size=258, dim=128, wave_steps=6, thinking_steps=3).to(device)
    
    # Collect all parameters including C++20 graph & drift parameters
    params = list(model.parameters())
    for p in model.saccadic_drift.parameters():
        params.append(p)
    for p in model.morphic_graph.parameters():
        params.append(p)
    
    optimizer = torch.optim.AdamW(params, lr=1.5e-3, weight_decay=1e-4)
    
    num_epochs = 150
    batch_size = 32
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        
        for domain_name, samples in suite.items():
            random.shuffle(samples)
            for i in range(0, min(128, len(samples)), batch_size):
                batch = samples[i:i+batch_size]
                prompts = [list(item[0].encode('utf-8')) for item in batch]
                targets = [list(item[2][len(item[0]):].encode('utf-8')) for item in batch]
                
                max_p = max(len(p) for p in prompts)
                max_t = max(len(t) for t in targets)
                
                p_pad = torch.full((len(batch), max_p), 256, dtype=torch.long, device=device)
                t_pad = torch.full((len(batch), max_t), 256, dtype=torch.long, device=device)
                
                for b_idx, (p, t) in enumerate(zip(prompts, targets)):
                    p_pad[b_idx, -len(p):] = torch.tensor(p, dtype=torch.long, device=device)
                    t_pad[b_idx, :len(t)] = torch.tensor(t, dtype=torch.long, device=device)
                    
                optimizer.zero_grad()
                loss = model.forward_loss(p_pad, t_pad)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                batches += 1
                
        avg_loss = total_loss / max(1, batches)
        if epoch % 30 == 0 or epoch == 1 or epoch == num_epochs:
            print(f"Epoch {epoch:03d}/{num_epochs:03d} | Continuous Cross-Entropy Loss: {avg_loss:.4f} nats")
            
    print("\n" + "=" * 80)
    print("📊 EVALUATING FULL COGNITIVE MULTI-DOMAIN TASK ACCURACY")
    print("=" * 80)
    
    model.eval()
    results = {}
    for domain_name, samples in suite.items():
        eval_samples = samples[:100]
        correct = 0
        for expr, expected_ans, full in eval_samples:
            p_bytes = list(expr.encode('utf-8'))
            p_t = torch.tensor([p_bytes], dtype=torch.long, device=device)
            gen_bytes = model.generate(p_t, max_gen_len=len(expected_ans) + 4)
            gen_str = bytes(gen_bytes[0].cpu().tolist()).decode('utf-8', errors='ignore').strip()
            if expected_ans.strip() in gen_str:
                correct += 1
        acc = (correct / len(eval_samples)) * 100.0
        results[domain_name] = acc
        print(f"Domain [{domain_name:10s}] Integrated Cognitive Accuracy: {acc:6.2f}% ({correct}/{len(eval_samples)})")
        
    print("=" * 80)
    return results

if __name__ == '__main__':
    run_integrated_benchmark()
