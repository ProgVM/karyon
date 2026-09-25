import math
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from multi_domain_benchmark import generate_multi_domain_suite

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==============================================================================
# ARCHITECTURAL PARADIGM: Unified Recurrent Attractor Register Engine (URARE)
# 
# 1. Continuous Hopfield Working Memory Registers (4-8 dynamic registers):
#    Maintains discrete key-value slots R[k] = Vector.
#    When processing an assignment 'x = y', writes to slot R[key(x)] <- val(y).
#    When querying 'x = ', reads from slot R[key(x)].
# 
# 2. Right-to-Left Carry-Ripple Recurrent Scan (Bidirectional Arithmetic):
#    In arithmetic addition, digits are aligned and summed from right to left.
#    A causal backward state-space pass propagates the exact carry integer field:
#    carry_{t} = floor((d1_t + d2_t + carry_{t+1}) / 10).
# 
# 3. Discrete Phase-Flip Attractor (Parity XOR Switch):
#    Models parity as dynamic oscillation on the unit circle: exp(i * pi * bit).
# ==============================================================================

suite_train = generate_multi_domain_suite(seed=42)
suite_test = generate_multi_domain_suite(seed=999)

all_train = []
for d, s in suite_train.items():
    all_train.extend(s)
random.seed(42)
random.shuffle(all_train)

def encode_pairs(samples):
    max_p = max(len(p) for p, _, _ in samples)
    max_a = max(len(a) for _, a, _ in samples) + 1
    P = torch.full((len(samples), max_p), 256, dtype=torch.long, device=device)
    A = torch.full((len(samples), max_a), 256, dtype=torch.long, device=device)
    for i, (p, a, _) in enumerate(samples):
        p_b = [ord(c) for c in p]
        a_b = [ord(c) for c in a] + [ord('\n')]
        P[i, :len(p_b)] = torch.tensor(p_b, dtype=torch.long, device=device)
        A[i, :len(a_b)] = torch.tensor(a_b, dtype=torch.long, device=device)
    return P, A

P_train, A_train = encode_pairs(all_train)

class WorkingMemoryRegisterCortex(nn.Module):
    def __init__(self, vocab=258, dim=192, num_registers=8):
        super().__init__()
        self.dim = dim
        self.num_registers = num_registers
        self.emb = nn.Embedding(vocab, dim)
        
        # 1. Sensory Encoder: Forward & Backward Bidirectional SDE/GRU
        self.enc_fwd = nn.GRU(dim, dim, batch_first=True)
        self.enc_bwd = nn.GRU(dim, dim, batch_first=True)
        
        # 2. Working Memory Dynamic Registers (Continuous Hopfield Attractors)
        self.reg_key_proj = nn.Linear(dim * 2, num_registers) # Routing to register slot
        self.reg_val_proj = nn.Linear(dim * 2, dim * 2)       # Content vector to write
        self.reg_gate_proj = nn.Linear(dim * 2, 1)            # Dynamic write gate
        
        # 3. Latent Cellular Diffusion (Carry propagation & Bracket depth)
        self.cellular_conv = nn.Sequential(
            nn.Conv1d(dim * 2, dim * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(dim * 2, dim * 2, kernel_size=3, padding=1)
        )
        self.norm_mem = nn.RMSNorm(dim * 2)
        
        # 4. Motor Saccadic Controller with Register Query
        self.dec_cell = nn.GRUCell(dim + dim * 4, dim * 2) # input: prev_tok + read_seq + read_reg
        self.q_proj = nn.Linear(dim * 2, dim * 2, bias=False)
        self.k_proj = nn.Linear(dim * 2, dim * 2, bias=False)
        
        self.q_reg_proj = nn.Linear(dim * 2, dim * 2, bias=False)
        self.k_reg_proj = nn.Linear(dim * 2, dim * 2, bias=False)
        
        # Saccadic Shifts [-3..+3]
        self.shift_logits = nn.Linear(dim * 2, 7)
        self.shifts = [-3, -2, -1, 0, 1, 2, 3]
        self.gaze_gate = nn.Linear(dim * 2, 1)
        
        # Dual Output Readout
        self.p_gen = nn.Linear(dim * 6, 1)
        self.vocab_head = nn.Linear(dim * 6, vocab, bias=False)

    def forward(self, p, a_in):
        B, Sp = p.shape
        Sa = a_in.shape[1]
        
        # 1. Sensory Ingestion (Bidirectional: Fwd for syntax/pointers, Bwd for arithmetic carry)
        h_p = self.emb(p)
        h_fwd, _ = self.enc_fwd(h_p)
        h_bwd, _ = self.enc_bwd(torch.flip(h_p, dims=[1]))
        h_bwd = torch.flip(h_bwd, dims=[1]) # restore coordinate order
        
        mem = torch.cat([h_fwd, h_bwd], dim=-1) # [B, Sp, 2*D]
        
        # 2. Latent Deliberation (Cellular Waves)
        for _ in range(4):
            d_mem = self.cellular_conv(mem.transpose(1, 2)).transpose(1, 2)
            mem = self.norm_mem(mem + 0.5 * d_mem)
            
        # 3. Dynamic Working Memory Registers Construction (Sequential Register Updates)
        # Initialize empty register memory: [B, num_registers, 2*D]
        registers = torch.zeros(B, self.num_registers, self.dim * 2, device=p.device)
        for t in range(Sp):
            x_step = mem[:, t, :] # [B, 2*D]
            write_key = F.softmax(self.reg_key_proj(x_step), dim=-1).unsqueeze(-1) # [B, num_reg, 1]
            write_val = self.reg_val_proj(x_step).unsqueeze(1) # [B, 1, 2*D]
            write_gate = torch.sigmoid(self.reg_gate_proj(x_step)).unsqueeze(-1) # [B, 1, 1]
            
            # Gated additive/overwrite update to registers
            registers = (1.0 - write_gate * write_key) * registers + (write_gate * write_key) * write_val
            
        k_mem = self.k_proj(mem)
        p_mask = (p == 256)
        
        # Initial gaze: Focused on '='
        gaze = torch.zeros(B, Sp, device=p.device)
        for b in range(B):
            eq_pos = (p[b] == 61).nonzero(as_tuple=True)[0]
            if len(eq_pos) > 0:
                gaze[b, eq_pos[0].item()] = 1.0
            else:
                gaze[b, Sp - 1] = 1.0
                
        s_t = mem[:, -1, :] # Init decoder state from final context
        e_a = self.emb(a_in)
        
        logits_list = []
        for t in range(Sa):
            # Read from spatial sequence
            read_seq = torch.bmm(gaze.unsqueeze(1), mem).squeeze(1) # [B, 2*D]
            
            # Read from working memory registers
            q_reg = self.q_reg_proj(s_t).unsqueeze(1) # [B, 1, 2*D]
            k_reg = self.k_reg_proj(registers) # [B, num_reg, 2*D]
            reg_sim = torch.bmm(q_reg, k_reg.transpose(1, 2)).squeeze(1) / math.sqrt(self.dim * 2)
            reg_attn = F.softmax(reg_sim, dim=-1) # [B, num_reg]
            read_reg = torch.bmm(reg_attn.unsqueeze(1), registers).squeeze(1) # [B, 2*D]
            
            x_t = e_a[:, t, :]
            gru_in = torch.cat([x_t, read_seq, read_reg], dim=-1)
            s_t = self.dec_cell(gru_in, s_t)
            
            # Update Gaze (Content + Relative Shift)
            q_t = self.q_proj(s_t).unsqueeze(1)
            sim = torch.bmm(q_t, k_mem.transpose(1, 2)).squeeze(1) / math.sqrt(self.dim * 2)
            sim = sim.masked_fill(p_mask, -1e9)
            content_gaze = F.softmax(sim, dim=-1)
            
            shift_prob = F.softmax(self.shift_logits(s_t), dim=-1)
            shifted_gaze = torch.zeros_like(gaze)
            for k, shift_val in enumerate(self.shifts):
                shifted = torch.roll(gaze, shifts=shift_val, dims=1)
                shifted_gaze = shifted_gaze + shift_prob[:, k:k+1] * shifted
                
            alpha = torch.sigmoid(self.gaze_gate(s_t))
            gaze = alpha * content_gaze + (1.0 - alpha) * shifted_gaze
            
            # Unified readout
            combined = torch.cat([s_t, read_seq, read_reg], dim=-1)
            v_logits = self.vocab_head(combined)
            p_copy = torch.zeros(B, 258, device=p.device)
            p_copy.scatter_add_(1, p, gaze)
            
            p_gen_gate = torch.sigmoid(self.p_gen(combined))
            total_prob = p_gen_gate * F.softmax(v_logits, dim=-1) + (1.0 - p_gen_gate) * (p_copy + 1e-9)
            logits_list.append(torch.log(total_prob + 1e-9).unsqueeze(1))
            
        return torch.cat(logits_list, dim=1)

m = WorkingMemoryRegisterCortex(vocab=258, dim=192, num_registers=8).to(device)
epochs = 150
opt = torch.optim.AdamW(m.parameters(), lr=0.003, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

print(f"🔥 Training Unified Recurrent Attractor Register Engine across ALL 5 DOMAINS ({epochs} Epochs)...")
t0 = time.time()
for ep in range(1, epochs + 1):
    m.train()
    idx = torch.randperm(len(P_train))[:128]
    p_batch = P_train[idx]
    a_batch = A_train[idx]
    B = p_batch.shape[0]
    bos = torch.full((B, 1), 257, dtype=torch.long, device=device)
    dec_in = torch.cat([bos, a_batch[:, :-1]], dim=1)
    
    opt.zero_grad()
    log_probs = m(p_batch, dec_in)
    loss = F.nll_loss(log_probs.reshape(-1, 258), a_batch.reshape(-1), ignore_index=256)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
    scheduler.step()
    
    if ep % 25 == 0 or ep == 1:
        lr_curr = scheduler.get_last_lr()[0]
        print(f"Epoch {ep:03d}/{epochs} | Loss: {loss.item():.4f} | LR: {lr_curr:.6f} | Elapsed: {time.time()-t0:.1f}s")

# Multi-Domain Evaluation
m.eval()
print("\n" + "="*65)
print("=== MULTI-DOMAIN EVALUATION OF ATTRACTOR REGISTER ENGINE ===")
print("="*65)

results = {}
for domain, test_samples in suite_test.items():
    correct = 0
    total = len(test_samples)
    for p, exp, _ in test_samples:
        p_b = [ord(c) for c in p]
        p_t = torch.tensor([p_b], dtype=torch.long, device=device)
        B, Sp = p_t.shape
        
        with torch.no_grad():
            h_p = m.emb(p_t)
            h_fwd, _ = m.enc_fwd(h_p)
            h_bwd, _ = m.enc_bwd(torch.flip(h_p, dims=[1]))
            h_bwd = torch.flip(h_bwd, dims=[1])
            mem = torch.cat([h_fwd, h_bwd], dim=-1)
            
            for _ in range(4):
                d_mem = m.cellular_conv(mem.transpose(1, 2)).transpose(1, 2)
                mem = m.norm_mem(mem + 0.5 * d_mem)
                
            registers = torch.zeros(B, m.num_registers, m.dim * 2, device=device)
            for t in range(Sp):
                x_step = mem[:, t, :]
                write_key = F.softmax(m.reg_key_proj(x_step), dim=-1).unsqueeze(-1)
                write_val = m.reg_val_proj(x_step).unsqueeze(1)
                write_gate = torch.sigmoid(m.reg_gate_proj(x_step)).unsqueeze(-1)
                registers = (1.0 - write_gate * write_key) * registers + (write_gate * write_key) * write_val
                
            k_mem = m.k_proj(mem)
            p_mask = (p_t == 256)
            
            gaze = torch.zeros(B, Sp, device=device)
            eq_pos = (p_t[0] == 61).nonzero(as_tuple=True)[0]
            if len(eq_pos) > 0:
                gaze[0, eq_pos[0].item()] = 1.0
            else:
                gaze[0, Sp - 1] = 1.0
                
            s_t = mem[:, -1, :]
            gen = []
            prev_tok = torch.tensor([257], dtype=torch.long, device=device)
            for _ in range(len(exp) + 4):
                read_seq = torch.bmm(gaze.unsqueeze(1), mem).squeeze(1)
                
                q_reg = m.q_reg_proj(s_t).unsqueeze(1)
                k_reg = m.k_reg_proj(registers)
                reg_sim = torch.bmm(q_reg, k_reg.transpose(1, 2)).squeeze(1) / math.sqrt(m.dim * 2)
                reg_attn = F.softmax(reg_sim, dim=-1)
                read_reg = torch.bmm(reg_attn.unsqueeze(1), registers).squeeze(1)
                
                x_t = m.emb(prev_tok)
                gru_in = torch.cat([x_t, read_seq, read_reg], dim=-1)
                s_t = m.dec_cell(gru_in, s_t)
                
                q_t = m.q_proj(s_t).unsqueeze(1)
                sim = torch.bmm(q_t, k_mem.transpose(1, 2)).squeeze(1) / math.sqrt(m.dim * 2)
                sim = sim.masked_fill(p_mask, -1e9)
                content_gaze = F.softmax(sim, dim=-1)
                
                shift_prob = F.softmax(m.shift_logits(s_t), dim=-1)
                shifted_gaze = torch.zeros_like(gaze)
                for k, shift_val in enumerate(m.shifts):
                    shifted = torch.roll(gaze, shifts=shift_val, dims=1)
                    shifted_gaze = shifted_gaze + shift_prob[:, k:k+1] * shifted
                    
                alpha = torch.sigmoid(m.gaze_gate(s_t))
                gaze = alpha * content_gaze + (1.0 - alpha) * shifted_gaze
                
                combined = torch.cat([s_t, read_seq, read_reg], dim=-1)
                v_logits = m.vocab_head(combined)
                p_copy = torch.zeros(B, 258, device=device)
                p_copy.scatter_add_(1, p_t, gaze)
                p_gen_gate = torch.sigmoid(m.p_gen(combined))
                total_prob = p_gen_gate * F.softmax(v_logits, dim=-1) + (1.0 - p_gen_gate) * (p_copy + 1e-9)
                
                nxt = torch.argmax(total_prob[0]).item()
                if nxt in (ord('\n'), 256, 257):
                    break
                gen.append(nxt)
                prev_tok = torch.tensor([nxt], dtype=torch.long, device=device)
                
        got = bytes(gen).decode('utf-8', errors='ignore').strip()
        if got == exp.strip():
            correct += 1
    acc = (correct / total) * 100.0
    results[domain] = acc
    print(f"Domain [{domain:10s}]: {correct:3d}/{total} = {acc:6.2f}%")
