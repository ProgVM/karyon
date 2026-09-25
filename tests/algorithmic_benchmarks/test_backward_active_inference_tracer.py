import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from multi_domain_benchmark import generate_multi_domain_suite

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==============================================================================
# BIOPHYSICAL DISCOVERY: Backward Goal-Directed Active Inference (Teleological Saccades)
#
# Look at the query: "a=2; b=8; c=4; d=9; c=d; a=c; b=a; c="
# Goal: Find the value of 'c'.
#
# If you go FORWARD from the beginning:
# You must track all 4 variables through all steps, updating everything.
#
# But in Karl Friston's Active Inference (Teleology / Backward Planning):
# The agent starts from the GOAL / DESIRED OBSERVATION at the end ('c'):
# Step 1: Query is 'c'. Look BACKWARDS for the MOST RECENT definition of 'c'.
#         -> Found 'c=d' at pos 16!
# Step 2: Now new Goal is 'd'! Look BACKWARDS from pos 16 for definition of 'd'.
#         -> Found 'd=9' at pos 12!
# Step 3: '9' is a literal digit! TARGET RESOLVED! Emitted '9'.
#
# Complexity: Exactly 2 backward saccades instead of tracking full state!
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

ptr_train_P, ptr_train_A = encode_task(suite_train['pointer'])

# Let's test a Backward Saccadic Attention Mechanism
class BackwardSaccadicTracer(nn.Module):
    def __init__(self, vocab=258, dim=128, max_hops=4):
        super().__init__()
        self.dim = dim
        self.max_hops = max_hops
        self.emb = nn.Embedding(vocab, dim)
        self.enc = nn.GRU(dim, dim, batch_first=True, bidirectional=True)
        
        # Target matcher: queries what variable we are looking for
        self.q_proj = nn.Linear(dim * 2, dim * 2)
        # Variable key: identifies LHS of assignments
        self.lhs_key = nn.Linear(dim * 2, dim * 2)
        # RHS Value representation
        self.rhs_val = nn.Linear(dim * 2, dim * 2)
        
        # Emits answer
        self.head = nn.Linear(dim * 2, vocab)

    def forward(self, p):
        B, Sp = p.shape
        h_p = self.emb(p)
        mem, _ = self.enc(h_p) # [B, Sp, 2*D]
        
        # Reverse causal mask: only allow attending to earlier tokens (t_key < t_query)
        # Saccade 1: Start from query variable at Sp-2 (e.g. 'c' in 'c=')
        q_curr = self.q_proj(mem[:, -2, :]).unsqueeze(1) # [B, 1, 2*D]
        
        k_lhs = self.lhs_key(mem) # [B, Sp, 2*D]
        v_rhs = self.rhs_val(mem) # [B, Sp, 2*D]
        
        # Multi-Hop Backward Saccades:
        for hop in range(self.max_hops):
            # Compute similarity to all previous tokens
            scores = torch.bmm(q_curr, k_lhs.transpose(1, 2)).squeeze(1) / math.sqrt(self.dim * 2) # [B, Sp]
            
            # Mask out padding
            p_mask = (p == 256)
            scores = scores.masked_fill(p_mask, -1e9)
            attn = F.softmax(scores, dim=-1) # [B, Sp]
            
            # Read the RHS value of the matched assignment
            read_rhs = torch.bmm(attn.unsqueeze(1), v_rhs) # [B, 1, 2*D]
            
            # Update query for next hop (if RHS was a variable, q_curr becomes that variable!)
            q_curr = self.q_proj(read_rhs.squeeze(1)).unsqueeze(1)
            
        logits = self.head(q_curr.squeeze(1))
        return logits

m = BackwardSaccadicTracer().to(device)
opt = torch.optim.AdamW(m.parameters(), lr=0.003)

print("Training Backward Saccadic Tracer on Pointers (60 Epochs)...")
for ep in range(1, 61):
    m.train()
    idx = torch.randperm(len(ptr_train_P))[:128]
    p_b = ptr_train_P[idx]
    a_b = ptr_train_A[idx]
    
    opt.zero_grad()
    logits = m(p_b)
    loss = F.cross_entropy(logits, a_b[:, 0])
    loss.backward()
    opt.step()
    if ep % 20 == 0 or ep == 1:
        print(f"Epoch {ep:02d} | Loss: {loss.item():.4f}")

m.eval()
correct = 0
for p, exp, _ in suite_test['pointer']:
    p_b = [ord(c) for c in p]
    p_t = torch.tensor([p_b], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = m(p_t)
        got = chr(torch.argmax(logits[0]).item())
        if got == exp.strip():
            correct += 1

acc = (correct / len(suite_test['pointer'])) * 100.0
print(f"\n🎯 Backward Saccadic Tracer Accuracy: {correct}/{len(suite_test['pointer'])} = {acc:.2f}%")
for p, exp, _ in suite_test['pointer'][:8]:
    p_b = [ord(c) for c in p]
    p_t = torch.tensor([p_b], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = m(p_t)
        got = chr(torch.argmax(logits[0]).item())
        ok = '✅' if got == exp.strip() else '❌'
        print(f"  {ok} '{p}' -> Target: '{exp.strip()}' | Got: '{got}'")
