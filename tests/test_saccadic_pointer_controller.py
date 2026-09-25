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
# Saccadic Attractor Gaze Controller:
# 1. Prompt Ingestion + 4 Cycles of Cellular Deliberation
# 2. Stateful Gaze Pointer (Saccadic Shift heads [-2, -1, 0, +1, +2])
# 3. Dynamic Attention over Prompt Positions + SSD Motor Core
# ==============================================================================

class SaccadicSovereignMind(nn.Module):
    def __init__(self, vocab_size=258, dim=128, wave_steps=4):
        super().__init__()
        self.dim = dim
        self.emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(512, dim)
        nn.init.normal_(self.emb.weight, 0.0, 1.0 / math.sqrt(dim))
        nn.init.normal_(self.pos_emb.weight, 0.0, 1.0 / math.sqrt(dim))
        
        # Cellular Wave Deliberation
        self.wave_conv = nn.Conv1d(dim, dim * 2, kernel_size=3, padding=1, bias=False)
        self.wave_swi_g = nn.Linear(dim, dim, bias=False)
        self.wave_swi_v = nn.Linear(dim, dim, bias=False)
        self.wave_swi_o = nn.Linear(dim, dim, bias=False)
        self.wave_norm = nn.RMSNorm(dim)
        self.wave_steps = wave_steps
        
        # Motor SSD Core
        self.ssd = karyon_core.CausalParallelSSD(dim, str(device))
        
        # Saccadic Attention Head: Multi-Head Content + Position Alignment
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.norm_motor = nn.RMSNorm(dim)
        
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.emb.weight

    def deliberate_prompt(self, p_tokens):
        B, P = p_tokens.shape
        pos = torch.arange(P, device=p_tokens.device).unsqueeze(0)
        h = self.emb(p_tokens) + self.pos_emb(pos)
        
        for _ in range(self.wave_steps):
            wave_feat = self.wave_conv(h.transpose(1, 2)).transpose(1, 2)
            g, v = wave_feat.chunk(2, dim=-1)
            wave = F.silu(g) * v
            swi = self.wave_swi_o(F.silu(self.wave_swi_g(h)) * self.wave_swi_v(h))
            h = self.wave_norm(h + 0.5 * (wave + swi))
        return h

    def decode_motor(self, a_tokens, p_field):
        B, A = a_tokens.shape
        pos = torch.arange(A, device=a_tokens.device).unsqueeze(0)
        x_a = self.emb(a_tokens) + self.pos_emb(pos)
        
        h_motor = self.ssd.forward(x_a)
        
        q = self.q_proj(h_motor) # [B, A, D]
        k = self.k_proj(p_field) # [B, P, D]
        v = self.v_proj(p_field) # [B, P, D]
        
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.dim)
        attn = F.softmax(scores, dim=-1)
        gaze_ctx = torch.bmm(attn, v)
        
        out = self.norm_motor(h_motor + self.out_proj(gaze_ctx))
        return self.head(out)

# Training test on Reversal & Dyck specifically
suite_train = generate_multi_domain_suite(seed=42)
suite_test = generate_multi_domain_suite(seed=999)

all_train = []
for d, s in suite_train.items():
    all_train.extend(s)
random.seed(42)
random.shuffle(all_train)

def encode_data(samples):
    max_p = max(len(p) for p, _, _ in samples)
    max_a = max(len(a) for _, a, _ in samples) + 2
    P = torch.full((len(samples), max_p), 256, dtype=torch.long, device=device)
    A_in = torch.full((len(samples), max_a), 256, dtype=torch.long, device=device)
    A_tgt = torch.full((len(samples), max_a), 256, dtype=torch.long, device=device)
    for i, (p, a, _) in enumerate(samples):
        p_b = [ord(c) for c in p]
        a_b = [ord(c) for c in a] + [ord('\n')]
        P[i, :len(p_b)] = torch.tensor(p_b, dtype=torch.long, device=device)
        a_in_seq = [257] + a_b[:-1]
        A_in[i, :len(a_in_seq)] = torch.tensor(a_in_seq, dtype=torch.long, device=device)
        A_tgt[i, :len(a_b)] = torch.tensor(a_b, dtype=torch.long, device=device)
    return P, A_in, A_tgt

P_tr, Ain_tr, Atgt_tr = encode_data(all_train)

agent = SaccadicSovereignMind(vocab_size=258, dim=128, wave_steps=4).to(device)
opt = torch.optim.AdamW(agent.parameters(), lr=0.003, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-4)

print("\n--- Training SaccadicSovereignMind (100 Epochs) ---")
for ep in range(1, 101):
    agent.train()
    idx = torch.randperm(len(P_tr))[:128]
    p_b = P_tr[idx]
    ain_b = Ain_tr[idx]
    atgt_b = Atgt_tr[idx]
    
    opt.zero_grad()
    p_field = agent.deliberate_prompt(p_b)
    logits = agent.decode_motor(ain_b, p_field)
    
    loss = F.cross_entropy(logits.reshape(-1, 258), atgt_b.reshape(-1), ignore_index=256)
    loss.backward()
    opt.step()
    scheduler.step()
    
    if ep % 20 == 0 or ep == 1:
        print(f"Epoch {ep:03d} | Loss: {loss.item():.4f}")

agent.eval()
print("\n" + "=" * 80)
print("=== SACCADIC SOVEREIGN MIND MULTI-DOMAIN EVALUATION ===")
print("=" * 80)

for domain, samples in suite_test.items():
    correct = 0
    total = len(samples)
    sample_preds = []
    for p, exp, _ in samples:
        p_b = [ord(c) for c in p]
        curr_p = torch.tensor([p_b], dtype=torch.long, device=device)
        with torch.no_grad():
            p_field = agent.deliberate_prompt(curr_p)
            gen = []
            curr_a = torch.tensor([[257]], dtype=torch.long, device=device)
            for _ in range(len(exp) + 3):
                logits = agent.decode_motor(curr_a, p_field)
                nxt = torch.argmax(logits[0, -1, :]).item()
                if nxt in (ord('\n'), 256, 257):
                    break
                gen.append(nxt)
                curr_a = torch.cat([curr_a, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
        got = bytes(gen).decode('utf-8', errors='ignore')
        is_match = (got.strip() == exp.strip())
        if is_match:
            correct += 1
        if len(sample_preds) < 3:
            sample_preds.append((p, exp, got, is_match))
            
    acc = (correct / total) * 100.0
    print(f"\nDomain [{domain.upper():10s}]: Accuracy = {acc:.2f}% ({correct}/{total})")
    for p, exp, got, ok in sample_preds:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status} | Prompt: {p!r:30s} | Exp: {exp!r:10s} | Got: {got!r:10s}")
