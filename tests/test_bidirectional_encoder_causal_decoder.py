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
# Sovereign Dual-Phase Mind:
# Phase 1: Problem Perception & Cellular Latent Wave (Bidirectional on Prompt)
# Phase 2: Autoregressive Efference (Causal Motor Generation with Cross-Attention)
# ==============================================================================

class DualPhaseSovereignMind(nn.Module):
    def __init__(self, vocab_size=258, dim=128, wave_steps=4):
        super().__init__()
        self.dim = dim
        self.emb = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.emb.weight, 0.0, 1.0 / math.sqrt(dim))
        
        # Phase 1: Bidirectional Cellular Wave Cortex for Prompt Deliberation
        self.wave_conv = nn.Conv1d(dim, dim * 2, kernel_size=3, padding=1, bias=False)
        self.wave_swi_g = nn.Linear(dim, dim, bias=False)
        self.wave_swi_v = nn.Linear(dim, dim, bias=False)
        self.wave_swi_o = nn.Linear(dim, dim, bias=False)
        self.wave_norm = nn.RMSNorm(dim)
        self.wave_steps = wave_steps
        
        # Phase 2: Causal State-Space Engine (Motor Decoding)
        self.ssd = karyon_core.CausalParallelSSD(dim, str(device))
        
        # Cross-Gaze Mechanism: Motor state queries Prompt Wave Field
        self.gaze_q = nn.Linear(dim, dim, bias=False)
        self.gaze_k = nn.Linear(dim, dim, bias=False)
        self.gaze_v = nn.Linear(dim, dim, bias=False)
        self.gaze_norm = nn.RMSNorm(dim)
        
        # Motor Efference Readout
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.emb.weight

    def deliberate_prompt(self, p_tokens):
        # p_tokens: [B, P]
        h = self.emb(p_tokens) # [B, P, D]
        for _ in range(self.wave_steps):
            wave_feat = self.wave_conv(h.transpose(1, 2)).transpose(1, 2)
            g, v = wave_feat.chunk(2, dim=-1)
            wave = F.silu(g) * v
            swi = self.wave_swi_o(F.silu(self.wave_swi_g(h)) * self.wave_swi_v(h))
            h = self.wave_norm(h + 0.5 * (wave + swi))
        return h # Deliberated Prompt Field [B, P, D]

    def decode_motor(self, a_tokens, p_field):
        # a_tokens: [B, A]
        x_a = self.emb(a_tokens)
        h_motor = self.ssd.forward(x_a) # [B, A, D] (Causal SSD)
        
        # Cross-Gaze into Prompt Field
        q = self.gaze_q(h_motor) # [B, A, D]
        k = self.gaze_k(p_field) # [B, P, D]
        v = self.gaze_v(p_field) # [B, P, D]
        
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.dim)
        attn = F.softmax(scores, dim=-1) # [B, A, P]
        gaze_ctx = torch.bmm(attn, v) # [B, A, D]
        
        out = self.gaze_norm(h_motor + gaze_ctx)
        return self.head(out)

print("DualPhaseSovereignMind class compiled.")

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
        # Teacher forcing: A_in starts with EOS/BOS=257
        a_in_seq = [257] + a_b[:-1]
        A_in[i, :len(a_in_seq)] = torch.tensor(a_in_seq, dtype=torch.long, device=device)
        A_tgt[i, :len(a_b)] = torch.tensor(a_b, dtype=torch.long, device=device)
    return P, A_in, A_tgt

P_tr, Ain_tr, Atgt_tr = encode_data(all_train)

agent = DualPhaseSovereignMind(vocab_size=258, dim=128, wave_steps=4).to(device)
opt = torch.optim.AdamW(agent.parameters(), lr=0.003, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-4)

print("\n--- Training DualPhaseSovereignMind (100 Epochs) ---")
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
print("=== DUAL-PHASE SOVEREIGN MIND MULTI-DOMAIN EVALUATION ===")
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
