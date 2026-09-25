import math
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from multi_domain_benchmark import generate_multi_domain_suite

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Deep Extended Evolution & Training Experiment (EXP-290 Candidate)
# Objectives:
# 1. Scale Epochs: 100 -> 350 Epochs with Cosine Annealing Learning Rate Schedule.
# 2. Increase Thinking Depth: K=4 -> K=6 cellular wave propagation steps.
# 3. Increase Dimension: dim=128 -> dim=192 to give more capacity for parallel domain storage.
# 4. Multi-Domain Loss Balancing: Temperature-weighted softmax on hard tasks.

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

class DeepDualPhaseRecurrentEngine(nn.Module):
    def __init__(self, vocab=258, dim=192, thinking_cycles=6):
        super().__init__()
        self.dim = dim
        self.thinking_cycles = thinking_cycles
        self.emb = nn.Embedding(vocab, dim)
        
        # Sensory Encoder (Bidirectional RNN)
        self.enc = nn.GRU(dim, dim, batch_first=True, bidirectional=True)
        
        # Latent Thinking Recirculation (Local Cellular Conv + Dynamic Feedback)
        self.cellular_conv = nn.Sequential(
            nn.Conv1d(dim * 2, dim * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(dim * 2, dim * 2, kernel_size=3, padding=1)
        )
        self.norm_mem = nn.RMSNorm(dim * 2)
        
        # Motor Saccadic Controller
        self.dec_cell = nn.GRUCell(dim + dim * 2, dim * 2)
        self.q_proj = nn.Linear(dim * 2, dim * 2, bias=False)
        self.k_proj = nn.Linear(dim * 2, dim * 2, bias=False)
        
        # Shifts: [-3, -2, -1, 0, 1, 2, 3]
        self.shift_logits = nn.Linear(dim * 2, 7)
        self.shifts = [-3, -2, -1, 0, 1, 2, 3]
        self.gaze_gate = nn.Linear(dim * 2, 1)
        
        self.p_gen = nn.Linear(dim * 4, 1)
        self.vocab_head = nn.Linear(dim * 4, vocab, bias=False)

    def forward(self, p, a_in):
        B, Sp = p.shape
        Sa = a_in.shape[1]
        
        h_p = self.emb(p)
        mem, hn = self.enc(h_p) # [B, Sp, 2*D]
        
        # Phase 1: Latent Problem Settling (K Thinking Cycles)
        for _ in range(self.thinking_cycles):
            d_mem = self.cellular_conv(mem.transpose(1, 2)).transpose(1, 2)
            mem = self.norm_mem(mem + 0.5 * d_mem)
            
        k_mem = self.k_proj(mem)
        p_mask = (p == 256)
        
        gaze = torch.zeros(B, Sp, device=p.device)
        for b in range(B):
            eq_pos = (p[b] == 61).nonzero(as_tuple=True)[0]
            if len(eq_pos) > 0:
                gaze[b, eq_pos[0].item()] = 1.0
            else:
                gaze[b, Sp - 1] = 1.0
                
        s_t = hn.transpose(0, 1).reshape(B, -1)
        e_a = self.emb(a_in)
        
        logits_list = []
        for t in range(Sa):
            read_vec = torch.bmm(gaze.unsqueeze(1), mem).squeeze(1)
            x_t = e_a[:, t, :]
            gru_in = torch.cat([x_t, read_vec], dim=-1)
            s_t = self.dec_cell(gru_in, s_t)
            
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
            
            combined = torch.cat([s_t, read_vec], dim=-1)
            v_logits = self.vocab_head(combined)
            p_copy = torch.zeros(B, 258, device=p.device)
            p_copy.scatter_add_(1, p, gaze)
            
            p_gen_gate = torch.sigmoid(self.p_gen(combined))
            total_prob = p_gen_gate * F.softmax(v_logits, dim=-1) + (1.0 - p_gen_gate) * (p_copy + 1e-9)
            logits_list.append(torch.log(total_prob + 1e-9).unsqueeze(1))
            
        return torch.cat(logits_list, dim=1)

m = DeepDualPhaseRecurrentEngine(vocab=258, dim=192, thinking_cycles=6).to(device)
epochs = 350
opt = torch.optim.AdamW(m.parameters(), lr=0.003, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

print(f"🔥 Launching Deep Extended Evolution & Training (Epochs: {epochs}, Dim: 192, Thinking Cycles: 6)...")
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
    
    if ep % 50 == 0 or ep == 1:
        lr_curr = scheduler.get_last_lr()[0]
        print(f"Epoch {ep:03d}/{epochs} | Loss: {loss.item():.4f} | LR: {lr_curr:.6f} | Elapsed: {time.time()-t0:.1f}s")

# Extended Multi-Domain Evaluation
m.eval()
print("\n" + "="*65)
print("=== DEEP EXTENDED EVOLUTION MULTI-DOMAIN EVALUATION RESULTS ===")
print("="*65)

results = {}
for domain, test_samples in suite_test.items():
    correct = 0
    total = len(test_samples)
    for p, exp, _ in test_samples:
        p_b = [ord(c) for c in p]
        p_t = torch.tensor([p_b], dtype=torch.long, device=device)
        B, Sp = p_t.shape
        h_p = m.emb(p_t)
        mem, hn = m.enc(h_p)
        for _ in range(m.thinking_cycles):
            d_mem = m.cellular_conv(mem.transpose(1, 2)).transpose(1, 2)
            mem = m.norm_mem(mem + 0.5 * d_mem)
            
        k_mem = m.k_proj(mem)
        p_mask = (p_t == 256)
        s_t = hn.transpose(0, 1).reshape(B, -1)
        
        gaze = torch.zeros(B, Sp, device=device)
        eq_pos = (p_t[0] == 61).nonzero(as_tuple=True)[0]
        if len(eq_pos) > 0:
            gaze[0, eq_pos[0].item()] = 1.0
        else:
            gaze[0, Sp - 1] = 1.0
            
        gen = []
        prev_tok = torch.tensor([257], dtype=torch.long, device=device)
        with torch.no_grad():
            for _ in range(len(exp) + 4):
                read_vec = torch.bmm(gaze.unsqueeze(1), mem).squeeze(1)
                x_t = m.emb(prev_tok)
                gru_in = torch.cat([x_t, read_vec], dim=-1)
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
                
                combined = torch.cat([s_t, read_vec], dim=-1)
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

print(f"\nFinal Training Loss: {loss.item():.4f}")
