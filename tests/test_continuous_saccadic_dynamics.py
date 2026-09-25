"""
Continuous Saccadic Dynamics (C-SSD) & Continuous Attractor Drift Engine (v3.0)
Biophysically Continuous Attractor Neural Network (CANN) with Phase-Harmonic Spatial Basis
and Dual Continuous Readout (Direct Token Copying & Attractor Readout).
Strictly Continuous Recurrent Substrate (KEP Principle 2, 21, 22). Zero Discrete Hacks.
"""

import sys, os
sys.path.insert(0, '.')
import math, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F

from multi_domain_benchmark import generate_multi_domain_suite

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 1. CONTINUOUS ATTRACTOR DRIFT (CANN / CONTINUOUS PHASE-SHIFT DYNAMICS)
# ==============================================================================

class ContinuousAttractorDriftV3(nn.Module):
    """
    Continuous Attractor Neural Network (CANN) with Phase-Harmonic Spatial Basis, Asymmetric Drift,
    and Continuous Copy Gate.
    du/dt = -u + W_sym * f(u) + v(t) * (W_asym * f(u)) + I_ext
    """
    def __init__(self, dim=128, max_len=128, num_filters=17):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.num_filters = num_filters
        
        # Continuous velocity & shift dynamics (continuous force generator)
        self.velocity_net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.SiLU(),
            nn.Linear(dim // 2, 1, bias=True) # continuous scalar drift velocity v_t in [-1, 1]
        )
        
        # Symmetric (recurrent tuning) & Asymmetric (drift propagation) continuous kernels
        self.w_sym = nn.Parameter(torch.randn(1, 1, num_filters) * 0.1)
        self.w_asym = nn.Parameter(torch.randn(1, 1, num_filters) * 0.1)
        
        # Precision & Gating
        self.beta_scale = nn.Parameter(torch.tensor(20.0)) # Crisp continuous attractor snapping
        self.gaze_gate = nn.Linear(dim, 1, bias=True)
        self.copy_gate = nn.Linear(dim, 1, bias=True)
        self.content_q = nn.Linear(dim, dim, bias=False)
        self.content_k = nn.Linear(dim, dim, bias=False)
        self.gaze_proj = nn.Linear(dim * 2, dim)
        self.norm = nn.RMSNorm(dim)

    def continuous_bump_drift(self, bump_state, velocity, beta=20.0):
        """
        Updates continuous attractor bump via differential drift convolution:
        delta_bump = W_sym * bump + v_t * (W_asym * bump)
        """
        B, L = bump_state.shape
        pad = self.num_filters // 2
        bump_pad = F.pad(bump_state.unsqueeze(1), (pad, pad), mode='replicate')
        
        sym_force = F.conv1d(bump_pad, self.w_sym)
        asym_force = F.conv1d(bump_pad, self.w_asym)
        
        drift_force = sym_force + velocity.view(B, 1, 1) * asym_force
        new_potential = bump_state + drift_force.squeeze(1)
        
        new_bump = F.softmax(new_potential * beta, dim=-1)
        return new_bump

    def forward(self, h_core, p_field, p_tokens, prev_bump, u_t=None):
        """
        h_core: [B, D] (recurrent state)
        p_field: [B, L, D] (continuous prompt memory field)
        p_tokens: [B, L] (raw prompt bytes)
        prev_bump: [B, L] (continuous attractor distribution)
        """
        B, L, D = p_field.shape
        
        # 1. Continuous drift velocity estimation: v_t in [-1, +1]
        v_t = torch.tanh(self.velocity_net(h_core)).squeeze(-1)
        
        # 2. Dynamic biophysical precision
        da_gain = 0.0
        if u_t is not None and u_t.numel() > 5:
            da_gain = u_t[:, 5].mean()
        beta = torch.clamp(self.beta_scale * (1.0 + 1.5 * da_gain), min=6.0, max=50.0)
        
        # 3. Continuous Attractor Drift Step
        drifted_bump = self.continuous_bump_drift(prev_bump, v_t, beta=beta)
        
        # 4. Smooth Content Attention (Continuous Memory Correlation)
        q = self.content_q(h_core).unsqueeze(1)
        k = self.content_k(p_field)
        content_scores = torch.bmm(q, k.transpose(1, 2)).squeeze(1) / math.sqrt(D)
        content_bump = F.softmax(content_scores * (beta / 2.0), dim=-1)
        
        # 5. Continuous Allostatic Gaze Blending (Drift + Content Resonance)
        alpha_gaze = torch.sigmoid(self.gaze_gate(h_core))
        next_bump = alpha_gaze * drifted_bump + (1.0 - alpha_gaze) * content_bump
        next_bump = next_bump / (next_bump.sum(dim=-1, keepdim=True) + 1e-6)
        
        # 6. Continuous Field Readout via smooth manifold integration
        h_gaze = torch.bmm(next_bump.unsqueeze(1), p_field).squeeze(1)
        
        # 7. Continuous Direct Token Projection (Copy Mechanism)
        p_copy = torch.sigmoid(self.copy_gate(h_core))
        copy_logits = torch.zeros(B, 258, device=p_field.device)
        copy_logits.scatter_add_(1, p_tokens, next_bump)
        
        # 8. Continuous Residual Synthesis
        h_fused = self.norm(h_core + self.gaze_proj(torch.cat([h_core, h_gaze], dim=-1)))
        
        return h_fused, next_bump, p_copy, copy_logits, v_t


# ==============================================================================
# 2. CONTINUOUS RECUPERATIVE DUAL-PHASE C-SSD ARCHITECTURE
# ==============================================================================

class ContinuousSovereignAgentV3(nn.Module):
    def __init__(self, vocab_size=258, dim=128, wave_steps=6):
        super().__init__()
        self.dim = dim
        self.emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(512, dim)
        self.wave_steps = wave_steps
        
        # Continuous Local Cellular Wave Layers (Bidirectional settling)
        self.wave_conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.wave_norm = nn.RMSNorm(dim)
        
        # Causal Recurrent Core
        self.recurrent_cell = nn.GRUCell(dim, dim)
        
        # Continuous Saccadic Attractor Drift Engine
        self.saccadic_drift = ContinuousAttractorDriftV3(dim=dim, max_len=256)
        
        # Continuous Readout Manifold
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
            h_core = self.recurrent_cell(x_t, h_core)
            h_fused, bump, p_copy, copy_logits, v_t = self.saccadic_drift(h_core, p_field, prompt_tokens, bump)
            
            gen_logits = self.head(h_fused)
            # Continuous Logit Fusion: (1 - p_copy) * Gen + p_copy * Copy
            fused_probs = (1.0 - p_copy) * F.softmax(gen_logits, dim=-1) + p_copy * copy_logits
            log_probs = torch.log(fused_probs + 1e-8)
            all_logits.append(log_probs)
            
        log_probs_stack = torch.stack(all_logits, dim=1) # [B, T, V]
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
            h_core = self.recurrent_cell(x_t, h_core)
            h_fused, bump, p_copy, copy_logits, v_t = self.saccadic_drift(h_core, p_field, prompt_tokens, bump)
            
            gen_logits = self.head(h_fused)
            fused_probs = (1.0 - p_copy) * F.softmax(gen_logits, dim=-1) + p_copy * copy_logits
            next_token = fused_probs.argmax(dim=-1)
            generated.append(next_token)
            curr_token = next_token
            if (curr_token == 10).all(): # '\n'
                break
        return torch.stack(generated, dim=1)


# ==============================================================================
# 3. EMPIRICAL VALIDATION HARNESS
# ==============================================================================

def run_continuous_saccadic_benchmark():
    print("=" * 80)
    print("⚡ STARTING CONTINUOUS SACCADIC DYNAMICS V3 (C-SSD) BENCHMARK")
    print("=" * 80)
    
    suite = generate_multi_domain_suite(seed=42)
    model = ContinuousSovereignAgentV3(vocab_size=258, dim=128, wave_steps=6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                batches += 1
                
        avg_loss = total_loss / max(1, batches)
        if epoch % 30 == 0 or epoch == 1 or epoch == num_epochs:
            print(f"Epoch {epoch:03d}/{num_epochs:03d} | Continuous Cross-Entropy Loss: {avg_loss:.4f} nats")
            
    print("\n" + "=" * 80)
    print("📊 EVALUATING MULTI-DOMAIN CONTINUOUS TASK ACCURACY")
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
        print(f"Domain [{domain_name:10s}] Continuous Saccadic Accuracy: {acc:6.2f}% ({correct}/{len(eval_samples)})")
        
    print("=" * 80)
    return results

if __name__ == '__main__':
    run_continuous_saccadic_benchmark()
