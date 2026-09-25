import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from multi_domain_benchmark import generate_multi_domain_suite

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Differentiable Register Machine with clean non-inplace updates:
# Out-of-place functional register updates:
# R_{t+1} = (1 - w) * R_t + w * new_val

class DifferentiableRegisterMachine(nn.Module):
    def __init__(self, vocab=258, dim=128):
        super().__init__()
        self.dim = dim
        self.emb = nn.Embedding(vocab, dim)
        self.enc = nn.GRU(dim, dim, batch_first=True, bidirectional=True)
        
        # Action Classifier per token step
        self.lhs_head = nn.Linear(dim * 2, 4)      # target register: a=0, b=1, c=2, d=3
        self.rhs_type = nn.Linear(dim * 2, 2)      # 0: constant digit, 1: register reference
        self.rhs_val = nn.Linear(dim * 2, 10)      # digit 0..9 or reg 0..3
        self.is_statement = nn.Linear(dim * 2, 1)  # write gate
        
        self.head = nn.Linear(dim * 2 + 10, vocab)

    def forward(self, p, a_in):
        B, Sp = p.shape
        Sa = a_in.shape[1]
        
        h_p = self.emb(p)
        mem, hn = self.enc(h_p) # [B, Sp, 2*D]
        
        # Soft register table: [B, 4, 10]
        R = torch.full((B, 4, 10), 0.1, device=p.device)
        
        for t in range(Sp):
            x = mem[:, t, :]
            gate = torch.sigmoid(self.is_statement(x)) # [B, 1]
            
            lhs = F.softmax(self.lhs_head(x), dim=-1) # [B, 4]
            is_const = F.softmax(self.rhs_type(x), dim=-1)[:, 0:1] # [B, 1]
            is_reg = 1.0 - is_const
            
            v_const = F.softmax(self.rhs_val(x), dim=-1) # [B, 10]
            
            rhs_reg_idx = F.softmax(self.rhs_val(x)[:, :4], dim=-1) # [B, 4]
            v_reg = torch.bmm(rhs_reg_idx.unsqueeze(1), R).squeeze(1) # [B, 10]
            
            new_val = is_const * v_const + is_reg * v_reg # [B, 10]
            
            # Non-inplace functional update for all 4 slots:
            # w: [B, 4, 1]
            w = (gate * lhs).unsqueeze(-1)
            # new_val: [B, 1, 10]
            R = (1.0 - w) * R + w * new_val.unsqueeze(1)
                
        # Query target register from final character
        q_reg = F.softmax(self.lhs_head(mem[:, -1, :]), dim=-1) # [B, 4]
        read_digit_dist = torch.bmm(q_reg.unsqueeze(1), R).squeeze(1) # [B, 10]
        
        combined = torch.cat([mem[:, -1, :], read_digit_dist], dim=-1)
        logits = self.head(combined).unsqueeze(1)
        return logits.repeat(1, Sa, 1)

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

m_drm = DifferentiableRegisterMachine().to(device)
opt_drm = torch.optim.AdamW(m_drm.parameters(), lr=0.003)

print("Training Differentiable Register Machine on Pointers (80 Epochs)...")
for ep in range(1, 81):
    m_drm.train()
    idx = torch.randperm(len(P_ptr_tr))[:128]
    p_b = P_ptr_tr[idx]
    a_b = A_ptr_tr[idx]
    B = p_b.shape[0]
    bos = torch.full((B, 1), 257, dtype=torch.long, device=device)
    dec_in = torch.cat([bos, a_b[:, :-1]], dim=1)
    
    opt_drm.zero_grad()
    logits = m_drm(p_b, dec_in)
    loss = F.cross_entropy(logits[:, 0, :], a_b[:, 0], ignore_index=256)
    loss.backward()
    opt_drm.step()
    if ep % 20 == 0 or ep == 1:
        print(f"Epoch {ep:02d} | Loss: {loss.item():.4f}")

m_drm.eval()
correct = 0
for p, exp, _ in suite_test['pointer']:
    p_b = [ord(c) for c in p]
    p_t = torch.tensor([p_b], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = m_drm(p_t, torch.zeros(1, 1, dtype=torch.long, device=device))
        got = chr(torch.argmax(logits[0, 0, :]).item())
        if got == exp.strip():
            correct += 1

print(f"\n🎯 Differentiable Register Machine Pointer Accuracy: {correct}/{len(suite_test['pointer'])} = {correct/len(suite_test['pointer'])*100:.2f}%")
for p, exp, _ in suite_test['pointer'][:8]:
    p_b = [ord(c) for c in p]
    p_t = torch.tensor([p_b], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = m_drm(p_t, torch.zeros(1, 1, dtype=torch.long, device=device))
        got = chr(torch.argmax(logits[0, 0, :]).item())
        ok = '✅' if got == exp.strip() else '❌'
        print(f"  {ok} '{p}' -> Target: '{exp.strip()}' | Got: '{got}'")
