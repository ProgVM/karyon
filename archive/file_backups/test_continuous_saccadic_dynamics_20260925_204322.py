"""
Continuous Saccadic Dynamics (C-SSD) & Continuous Attractor Drift Engine (v1.0)
Biophysically Continuous Attractor Neural Network (CANN) & Continuous Phase-Shift Memory.
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
# 1. CONTINUOUS ATTRACTOR DRIFT (CANN / CONTINUOUS SACCADIC DYNAMICS)
# ==============================================================================

class ContinuousAttractorDrift(nn.Module):
    """
    Continuous Attractor Neural Network (CANN) with Smooth Velocity-Driven Drift.
    Models continuous bump propagation across 1D neural coordinate space:
    du/dt = -u + W * f(u) + v(t) * (dW_asym * f(u)) + I_ext
    Dopamine (DA) / Noradrenaline (NA) modulates the continuous attractor sharpness (beta).
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
        # Continuous phase modulation frequency bank (Fourier / Harmonic spatial basis)
        self.register_buffer("spatial_freqs", torch.linspace(0.1, math.pi, dim // 2))
        
        # Symmetric (recurrent tuning) & Asymmetric (drift propagation) continuous kernels
        self.w_sym = nn.Parameter(torch.randn(1, 1, num_filters) * 0.1)
        self.w_asym = nn.Parameter(torch.randn(1, 1, num_filters) * 0.1)
        
        # Precision & Gating
        self.beta_scale = nn.Parameter(torch.tensor(8.0))
        self.gaze_proj = nn.Linear(dim * 2, dim)
        self.norm = nn.RMSNorm(dim)

    def continuous_bump_drift(self, bump_state, velocity, beta=12.0):
        """
        Updates continuous attractor bump via differential drift convolution:
        delta_bump = W_sym * bump + v_t * (W_asym * bump)
        """
        B, L = bump_state.shape
        # Pad circularly/reflectively for smooth continuous convolution
        pad = self.num_filters // 2
        bump_pad = F.pad(bump_state.unsqueeze(1), (pad, pad), mode='replicate')
        
        # Continuous symmetric maintenance force
        sym_force = F.conv1d(bump_pad, self.w_sym)
        # Continuous asymmetric directional drift force
        asym_force = F.conv1d(bump_pad, self.w_asym)
        
        # Combined continuous velocity-modulated drift
        drift_force = sym_force + velocity.view(B, 1, 1) * asym_force
        new_potential = bump_state + drift_force.squeeze(1)
        
        # Continuous Hopfield / CANN non-linear energy normalization
        new_bump = F.softmax(new_potential * beta, dim=-1)
        return new_bump

    def forward(self, h_core, p_field, prev_bump, u_t=None):
        """
        h_core: [B, D] (recurrent state)
        p_field: [B, L, D] (continuous prompt memory field)
        prev_bump: [B, L] (continuous attractor distribution)
        """
        B, L, D = p_field.shape
        
        # 1. Continuous drift velocity estimation: v_t in [-1, +1]
        v_t = torch.tanh(self.velocity_net(h_core)).squeeze(-1)
        
        # 2. Dynamic biophysical precision (sharpened by Dopamine / Noradrenaline)
        da_gain = 0.0
        if u_t is not None and u_t.numel() > 5:
            da_gain = u_t[:, 5].mean() # Dopamine channel
        beta = torch.clamp(self.beta_scale * (1.0 + 1.5 * da_gain), min=2.0, max=30.0)
        
        # 3. Continuous Attractor Drift Step
        next_bump = self.continuous_bump_drift(prev_bump, v_t, beta=beta)
        
        # 4. Continuous Field Readout via smooth manifold integration
        # h_gaze = integral(bump(x) * Field(x) dx)
        h_gaze = torch.bmm(next_bump.unsqueeze(1), p_field).squeeze(1)
        
        # 5. Continuous Residual Synthesis
        h_fused = self.norm(h_core + self.gaze_proj(torch.cat([h_core, h_gaze], dim=-1)))
        
        return h_fused, next_bump, v_t


# ==============================================================================
# 2. CONTINUOUS RECUPERATIVE DUAL-PHASE C-SSD ARCHITECTURE
# ==============================================================================

class ContinuousSovereignAgent(nn.Module):
    """
    100% Continuous Recurrent Agent:
    - Phase 1: Continuous Bidirectional Cellular Waves (Prompt Settling into Continuous Memory Field)
    - Phase 2: Continuous Saccadic Dynamics (C-SSD) with Continuous Attractor Drift (CANN)
    - Causal Continuous Recurrence with Zero Discrete Registers.
    """
    def __init__(self, vocab_size=258, dim=128, wave_steps=6):
        super().__init__()
        self.dim = dim
        self.emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(512, dim)
        self.wave_steps = wave_steps
        
        # Continuous Local Cellular Wave Layers (Bidirectional settling)
        self.wave_conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.wave_norm = nn.RMSNorm(dim)
        
        # Causal Recurrent Core (Continuous SSD / GRU)
        self.recurrent_cell = nn.GRUCell(dim, dim)
        
        # Continuous Saccadic Attractor Drift Engine
        self.saccadic_drift = ContinuousAttractorDrift(dim=dim, max_len=256)
        
        # Continuous Readout Manifold
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.emb.weight # Direct synaptic tie

    def settle_prompt_field(self, p_tokens):
        """
        Settles prompt tokens into continuous manifold field via continuous cellular diffusion.
        """
        B, L = p_tokens.shape
        pos = torch.arange(L, device=p_tokens.device).unsqueeze(0).expand(B, L)
        x = self.emb(p_tokens) + self.pos_emb(pos)
        
        # Continuous Cellular Wave Recirculation
        h = x
        for _ in range(self.wave_steps):
            # 1D continuous local diffusion
            dh = self.wave_conv(h.transpose(1, 2)).transpose(1, 2)
            h = self.wave_norm(h + torch.tanh(dh))
        return h

    def forward_loss(self, prompt_tokens, target_tokens):
        B, P_len = prompt_tokens.shape
        _, T_len = target_tokens.shape
        
        # Phase 1: Continuous Field Settling
        p_field = self.settle_prompt_field(prompt_tokens)
        
        # Initial Continuous Attractor Bump (Centered or Initial Boundary)
        bump = torch.zeros(B, P_len, device=prompt_tokens.device)
        bump[:, -1] = 1.0 # Initialize gaze at prompt-target boundary
        bump = F.softmax(bump * 10.0, dim=-1)
        
        h_core = p_field[:, -1, :] # Initial continuous thought state
        
        # Phase 2: Continuous Causal Saccadic Recurrence
        # Prepend boundary token (prompt last token)
        inputs = torch.cat([prompt_tokens[:, -1:], target_tokens[:, :-1]], dim=1)
        
        all_logits = []
        for t in range(T_len):
            x_t = self.emb(inputs[:, t])
            h_core = self.recurrent_cell(x_t, h_core)
            h_fused, bump, v_t = self.saccadic_drift(h_core, p_field, bump)
            logits_t = self.head(h_fused)
            all_logits.append(logits_t)
            
        logits = torch.stack(all_logits, dim=1) # [B, T, V]
        loss = F.cross_entropy(logits.reshape(-1, 258), target_tokens.reshape(-1))
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
            h_fused, bump, v_t = self.saccadic_drift(h_core, p_field, bump)
            logits = self.head(h_fused)
            next_token = logits.argmax(dim=-1)
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
    print("⚡ STARTING CONTINUOUS SACCADIC DYNAMICS (C-SSD) BENCHMARK")
    print("=" * 80)
    
    suite = generate_multi_domain_suite(seed=42)
    model = ContinuousSovereignAgent(vocab_size=258, dim=128, wave_steps=6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    
    # Train Loop on Multi-Domain Continuous Stream
    num_epochs = 120
    batch_size = 32
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        
        # Sample across all 5 domains
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
        if epoch % 20 == 0 or epoch == 1 or epoch == num_epochs:
            print(f"Epoch {epoch:03d}/{num_epochs:03d} | Continuous Cross-Entropy Loss: {avg_loss:.4f} nats")
            
    # Evaluation
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
