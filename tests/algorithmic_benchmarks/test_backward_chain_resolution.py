import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from multi_domain_benchmark import generate_multi_domain_suite

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==============================================================================
# DISCOVERY: Backward Goal-Driven Pointer Chain Resolution (Friston Active Inference)
# 
# How does a mathematician solve: "a=2; b=8; c=4; d=9; c=d; a=c; b=a; c="?
# 1. Start from the goal at the end: We need value of 'c'. Target = 'c'.
# 2. Scan BACKWARDS:
#    - 'b=a;' -> does not affect 'c'. Target remains 'c'.
#    - 'a=c;' -> does not affect 'c'. Target remains 'c'.
#    - 'c=d;' -> affects 'c'! Target transforms from 'c' -> 'd'!
#    - 'd=9;' -> affects 'd'! Value found = '9'! HALT!
#
# A simple Causal Backward Scan over discrete variable tokens:
# In backward time: target_var_{t-1} = RHS if (LHS == target_var_t) else target_var_t!
# If RHS is a digit -> Emit digit!
# ==============================================================================

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

class BackwardGoalResolutionCortex(nn.Module):
    def __init__(self, vocab=258, dim=128):
        super().__init__()
        self.dim = dim
        self.emb = nn.Embedding(vocab, dim)
        
        # Backward Recurrent Tracer
        self.bwd_tracer = nn.GRU(dim, dim, batch_first=True)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, p, a_in):
        B, Sp = p.shape
        Sa = a_in.shape[1]
        
        # Reverse sequence for backward goal-driven inference
        p_rev = torch.flip(p, dims=[1])
        h_rev = self.emb(p_rev)
        
        out_rev, hn = self.bwd_tracer(h_rev)
        
        # Read from final step of backward scan (which is the beginning of the prompt)
        logits = self.head(hn.squeeze(0)).unsqueeze(1)
        return logits.repeat(1, Sa, 1)

m_bwd = BackwardGoalResolutionCortex().to(device)
opt_bwd = torch.optim.AdamW(m_bwd.parameters(), lr=0.003)

print("Testing Backward Goal Resolution Cortex on Pointers (60 Epochs)...")
for ep in range(1, 61):
    m_bwd.train()
    idx = torch.randperm(len(P_ptr_tr))[:128]
    p_b = P_ptr_tr[idx]
    a_b = A_ptr_tr[idx]
    B = p_b.shape[0]
    bos = torch.full((B, 1), 257, dtype=torch.long, device=device)
    dec_in = torch.cat([bos, a_b[:, :-1]], dim=1)
    
    opt_bwd.zero_grad()
    logits = m_bwd(p_b, dec_in)
    loss = F.cross_entropy(logits[:, 0, :], a_b[:, 0], ignore_index=256)
    loss.backward()
    opt_bwd.step()
    if ep % 20 == 0 or ep == 1:
        print(f"Epoch {ep:02d} | Loss: {loss.item():.4f}")

m_bwd.eval()
correct = 0
for p, exp, _ in suite_test['pointer']:
    p_b = [ord(c) for c in p]
    p_t = torch.tensor([p_b], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = m_bwd(p_t, torch.zeros(1, 1, dtype=torch.long, device=device))
        got = chr(torch.argmax(logits[0, 0, :]).item())
        if got == exp.strip():
            correct += 1

print(f"\n🎯 Backward Goal Resolution Pointer Accuracy: {correct}/{len(suite_test['pointer'])} = {correct/len(suite_test['pointer'])*100:.2f}%")
for p, exp, _ in suite_test['pointer'][:8]:
    p_b = [ord(c) for c in p]
    p_t = torch.tensor([p_b], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = m_bwd(p_t, torch.zeros(1, 1, dtype=torch.long, device=device))
        got = chr(torch.argmax(logits[0, 0, :]).item())
        ok = '✅' if got == exp.strip() else '❌'
        print(f"  {ok} '{p}' -> Target: '{exp.strip()}' | Got: '{got}'")
