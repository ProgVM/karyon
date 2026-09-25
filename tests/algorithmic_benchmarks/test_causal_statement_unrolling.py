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
# DISCOVERY: Segment-Level / Statement-Level Deliberation
# In code / execution trace:
# Input: "a=2; b=8; c=4; d=9; c=d; a=c; b=a; c="
# Statements are delimited by ';'
# Stmt 0: "a=2"
# Stmt 1: "b=8"
# Stmt 2: "c=4"
# Stmt 3: "d=9"
# Stmt 4: "c=d"
# Stmt 5: "a=c"
# Stmt 6: "b=a"
# Query : "c="
#
# If the state transition operates at the STATEMENT BOUNDARY (Chunked Event Memory):
# Variable State S in R^{4 x 10} (one-hot distributions over digits for each variable a, b, c, d)
# When statement "X = Y" arrives:
# if Y is digit d -> S[X] = one_hot(d)
# if Y is variable V -> S[X] = S[V]
#
# Let's test if a Statement-Level Neural Recurrent Cell solves Pointers with 100.00% accuracy!
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

class StatementLevelProgramCortex(nn.Module):
    def __init__(self, vocab=258, dim=128):
        super().__init__()
        self.dim = dim
        self.emb = nn.Embedding(vocab, dim)
        self.enc = nn.GRU(dim, dim, batch_first=True, bidirectional=True)
        
        # Soft variable selector: identifies LHS variable (a, b, c, d -> 0..3)
        self.lhs_head = nn.Linear(dim * 2, 4)
        
        # Soft RHS selector: identifies if RHS is variable (0..3) or digit (4..13)
        self.rhs_head = nn.Linear(dim * 2, 14)
        
        # Gate: is this segment a valid assignment statement?
        self.stmt_gate = nn.Linear(dim * 2, 1)
        
        # Query head: identifies target variable from "c="
        self.query_head = nn.Linear(dim * 2, 4)
        self.digit_out  = nn.Linear(10, vocab)

    def forward(self, p):
        B, Sp = p.shape
        h_p = self.emb(p)
        mem, _ = self.enc(h_p) # [B, Sp, 2*D]
        
        # Register State: 4 variables, each having a distribution over 10 digits (0..9)
        # S: [B, 4, 10]
        # Initialized to uniform or zeros
        S = torch.zeros(B, 4, 10, device=p.device)
        
        # Digits 0..9 one-hot matrix: [10, 10]
        digit_eye = torch.eye(10, device=p.device).unsqueeze(0).repeat(B, 1, 1) # [B, 10, 10]
        
        # Process every token; assignment takes effect upon parsing
        for t in range(Sp):
            x_t = mem[:, t, :]
            gate = torch.sigmoid(self.stmt_gate(x_t)).unsqueeze(-1) # [B, 1, 1]
            lhs = F.softmax(self.lhs_head(x_t), dim=-1).unsqueeze(-1) # [B, 4, 1]
            rhs = F.softmax(self.rhs_head(x_t), dim=-1) # [B, 14]
            
            rhs_var   = rhs[:, :4].unsqueeze(1) # [B, 1, 4]
            rhs_digit = rhs[:, 4:].unsqueeze(1) # [B, 1, 10]
            
            # Value to assign to LHS:
            # If RHS is a variable, value = rhs_var @ S -> [B, 1, 10]
            # If RHS is a digit, value = rhs_digit @ digit_eye -> [B, 1, 10]
            val_from_var = torch.bmm(rhs_var, S) # [B, 1, 10]
            val_from_dig = torch.bmm(rhs_digit, digit_eye) # [B, 1, 10]
            assigned_val = val_from_var + val_from_dig # [B, 1, 10]
            
            # Overwrite LHS in state:
            # S = (1 - gate * lhs) * S + (gate * lhs) * assigned_val
            S = (1.0 - gate * lhs) * S + (gate * lhs) * assigned_val
            
        # Read final queried variable
        q_var = F.softmax(self.query_head(mem[:, -1, :]), dim=-1).unsqueeze(1) # [B, 1, 4]
        final_digit_dist = torch.bmm(q_var, S).squeeze(1) # [B, 10]
        
        logits = self.digit_out(final_digit_dist)
        return logits

m = StatementLevelProgramCortex().to(device)
opt = torch.optim.AdamW(m.parameters(), lr=0.005)

print("Training Statement-Level Program Cortex on Pointers (60 Epochs)...")
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
print(f"\n🎯 Statement-Level Program Cortex Accuracy: {correct}/{len(suite_test['pointer'])} = {acc:.2f}%")
for p, exp, _ in suite_test['pointer'][:8]:
    p_b = [ord(c) for c in p]
    p_t = torch.tensor([p_b], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = m(p_t)
        got = chr(torch.argmax(logits[0]).item())
        ok = '✅' if got == exp.strip() else '❌'
        print(f"  {ok} '{p}' -> Target: '{exp.strip()}' | Got: '{got}'")
