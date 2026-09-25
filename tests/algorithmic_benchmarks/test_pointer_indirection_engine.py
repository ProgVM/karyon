import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from multi_domain_benchmark import generate_multi_domain_suite

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Deep Analysis of Pointer Indirection Task:
# Prompt: "a=2; b=8; c=4; d=9; c=d; a=c; b=a; c=" -> Expected: "9"
# How does a biological/human brain solve pointer indirection?
# 1. Look at target variable at the end: 'c=' (target = 'c')
# 2. Trace backwards or resolve chain:
#    - 'c=d' -> c becomes 'd' (target becomes 'd')
#    - then find 'd=9' -> value is '9'!
# Or forward execution:
#    - Start state: {a:2, b:8, c:4, d:9}
#    - Step 1: c=d -> {a:2, b:8, c:9, d:9}
#    - Step 2: a=c -> {a:9, b:8, c:9, d:9}
#    - Step 3: b=a -> {a:9, b:9, c:9, d:9}
#    - Query c -> returns 9!
#
# Can a Relational Transition Matrix (Transition Graph over 4 variables + 10 digits)
# perfectly model this forward execution state?
# State: S in R^{4 x 14} (for vars a,b,c,d and values 0..9)
# An assignment "x=y;" is an operator: S[x] <- S[y] if y is a variable, or one_hot(y) if y is a digit!

suite_train = generate_multi_domain_suite(seed=42)
suite_test = generate_multi_domain_suite(seed=999)

def encode_task(samples):
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

P_ptr_tr, A_ptr_tr = encode_task(suite_train['pointer'])

# Let's test a Relational Dynamic Working Memory (Continuous Outer-Product Register Binding)
# R_t = R_{t-1} + (v_t - R_{t-1} k_t) (k_t)^T
# This is the EXACT optimal fast-weight outer-product update rule (Schmidhuber 1992 / Ba et al. 2016)!
class FastWeightRelationalCortex(nn.Module):
    def __init__(self, vocab=258, dim=128):
        super().__init__()
        self.dim = dim
        self.emb = nn.Embedding(vocab, dim)
        
        # SDE / Recurrent Encoder
        self.enc = nn.GRU(dim, dim, batch_first=True)
        
        # Fast Weight Linear Projections
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.beta_gate = nn.Linear(dim, 1) # Learning rate / write strength
        
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.head = nn.Linear(dim * 2, vocab, bias=False)

    def forward(self, p, a_in):
        B, Sp = p.shape
        Sa = a_in.shape[1]
        
        h_p = self.emb(p)
        mem, hn = self.enc(h_p) # [B, Sp, D]
        
        # Initialize Fast Weight Matrix M: [B, D, D]
        M = torch.zeros(B, self.dim, self.dim, device=p.device)
        
        # Sequential Fast-Weight Updates:
        # M_t = M_{t-1} + beta_t * (v_t - M_{t-1} k_t) @ k_t^T
        for t in range(Sp):
            x_t = mem[:, t, :] # [B, D]
            k_t = F.normalize(self.k_proj(x_t), dim=-1) # [B, D]
            v_t = self.v_proj(x_t) # [B, D]
            beta = torch.sigmoid(self.beta_gate(x_t)).unsqueeze(-1) # [B, 1, 1]
            
            # Current memory readout
            curr_read = torch.bmm(M, k_t.unsqueeze(-1)).squeeze(-1) # [B, D]
            delta_v = (v_t - curr_read).unsqueeze(2) # [B, D, 1]
            
            # Outer product update
            outer = torch.bmm(delta_v, k_t.unsqueeze(1)) # [B, D, D]
            M = M + beta * outer
            
        # Motor Readout
        q = F.normalize(self.q_proj(hn.squeeze(0)), dim=-1) # [B, D]
        retrieved_val = torch.bmm(M, q.unsqueeze(-1)).squeeze(-1) # [B, D]
        
        combined = torch.cat([hn.squeeze(0), retrieved_val], dim=-1)
        logits = self.head(combined).unsqueeze(1) # [B, 1, vocab]
        
        # For sequence matching, repeat logits across output steps
        return logits.repeat(1, Sa, 1)

m_fw = FastWeightRelationalCortex().to(device)
opt_fw = torch.optim.AdamW(m_fw.parameters(), lr=0.003)

print("Testing Fast-Weight Relational Cortex on Pointers (80 Epochs)...")
for ep in range(1, 81):
    m_fw.train()
    idx = torch.randperm(len(P_ptr_tr))[:128]
    p_b = P_ptr_tr[idx]
    a_b = A_ptr_tr[idx]
    B = p_b.shape[0]
    bos = torch.full((B, 1), 257, dtype=torch.long, device=device)
    dec_in = torch.cat([bos, a_b[:, :-1]], dim=1)
    
    opt_fw.zero_grad()
    logits = m_fw(p_b, dec_in)
    loss = F.cross_entropy(logits[:, 0, :], a_b[:, 0], ignore_index=256)
    loss.backward()
    opt_fw.step()
    if ep % 20 == 0 or ep == 1:
        print(f"Epoch {ep:02d} | Loss: {loss.item():.4f}")

m_fw.eval()
correct = 0
for p, exp, _ in suite_test['pointer']:
    p_b = [ord(c) for c in p]
    p_t = torch.tensor([p_b], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = m_fw(p_t, torch.zeros(1, 1, dtype=torch.long, device=device))
        nxt = torch.argmax(logits[0, 0, :]).item()
        got = chr(nxt)
        if got == exp.strip():
            correct += 1

print(f"\n🎯 Fast-Weight Relational Cortex Pointer Accuracy: {correct}/{len(suite_test['pointer'])} = {correct/len(suite_test['pointer'])*100:.2f}%")
for p, exp, _ in suite_test['pointer'][:8]:
    p_b = [ord(c) for c in p]
    p_t = torch.tensor([p_b], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = m_fw(p_t, torch.zeros(1, 1, dtype=torch.long, device=device))
        got = chr(torch.argmax(logits[0, 0, :]).item())
        ok = '✅' if got == exp.strip() else '❌'
        print(f"  {ok} '{p}' -> Target: '{exp.strip()}' | Got: '{got}'")
