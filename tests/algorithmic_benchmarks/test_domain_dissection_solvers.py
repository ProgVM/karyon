import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from multi_domain_benchmark import generate_multi_domain_suite

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==============================================================================
# SCIENTIFIC ISOLATION STUDY: Finding the Exact Biological Circuit for Each Task
# 1. Study Domain: Pointers ('a=2; b=8; c=4; d=9; c=d; a=c; b=a; c=')
#    Why does soft register writing fail?
#    Because register addressing must be SHARP (Hard Softmax / Temperature T=0.1 or Gumbel)
#    and key representation must bind strictly to the variable identifier ('a', 'b', 'c', 'd')!
# 
# 2. Study Domain: Addition ('873 + 573 = 1446')
#    How does arithmetic computation truly happen?
#    Via Recurrent Carry-Propagating Turing / Cellular Automation!
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

# Let's test a Sharp Key-Value Dynamic Binding Memory for Pointers
class SharpKeyValueBindingCortex(nn.Module):
    def __init__(self, vocab=258, dim=128, num_slots=16):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.emb = nn.Embedding(vocab, dim)
        self.enc = nn.GRU(dim, dim, batch_first=True, bidirectional=True)
        
        # Key & Value Projection for Dynamic Binding
        self.k_proj = nn.Linear(dim * 2, dim)
        self.v_proj = nn.Linear(dim * 2, dim)
        self.write_gate = nn.Linear(dim * 2, 1)
        
        # Decoder
        self.dec_cell = nn.GRUCell(dim + dim, dim)
        self.query_proj = nn.Linear(dim, dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, p, a_in):
        B, Sp = p.shape
        Sa = a_in.shape[1]
        
        h_p = self.emb(p)
        mem, hn = self.enc(h_p) # [B, Sp, 2*D]
        
        # Sharp Key-Value Binding Table
        # Memory slots: [B, num_slots, D]
        mem_keys = torch.zeros(B, self.num_slots, self.dim, device=p.device)
        mem_vals = torch.zeros(B, self.num_slots, self.dim, device=p.device)
        mem_usage = torch.zeros(B, self.num_slots, 1, device=p.device)
        
        for t in range(Sp):
            x = mem[:, t, :]
            gate = torch.sigmoid(self.write_gate(x)) # Is this an assignment step?
            k_t = F.normalize(self.k_proj(x), dim=-1)
            v_t = self.v_proj(x)
            
            # Find matching slot or empty slot
            sim = torch.bmm(k_t.unsqueeze(1), mem_keys.transpose(1, 2)).squeeze(1) # [B, num_slots]
            slot_attn = F.softmax(sim * 10.0, dim=-1).unsqueeze(-1) # Sharp routing
            
            # Write to matching slot
            mem_keys = mem_keys + gate.unsqueeze(-1) * slot_attn * k_t.unsqueeze(1)
            mem_vals = (1.0 - gate.unsqueeze(-1) * slot_attn) * mem_vals + (gate.unsqueeze(-1) * slot_attn) * v_t.unsqueeze(1)
            
        s_t = hn.mean(dim=0)
        e_a = self.emb(a_in)
        logits = []
        
        for t in range(Sa):
            x_t = e_a[:, t, :]
            # Query memory table
            q = F.normalize(self.query_proj(s_t), dim=-1).unsqueeze(1)
            sim = torch.bmm(q, mem_keys.transpose(1, 2)).squeeze(1)
            slot_attn = F.softmax(sim * 10.0, dim=-1)
            read_v = torch.bmm(slot_attn.unsqueeze(1), mem_vals).squeeze(1)
            
            s_t = self.dec_cell(torch.cat([x_t, read_v], dim=-1), s_t)
            logits.append(self.head(s_t).unsqueeze(1))
            
        return torch.cat(logits, dim=1)

m_ptr = SharpKeyValueBindingCortex().to(device)
opt_ptr = torch.optim.AdamW(m_ptr.parameters(), lr=0.003)

print("Testing Sharp Key-Value Binding Cortex on Pointers (60 Epochs)...")
for ep in range(1, 61):
    m_ptr.train()
    idx = torch.randperm(len(P_ptr_tr))[:128]
    p_b = P_ptr_tr[idx]
    a_b = A_ptr_tr[idx]
    B = p_b.shape[0]
    bos = torch.full((B, 1), 257, dtype=torch.long, device=device)
    dec_in = torch.cat([bos, a_b[:, :-1]], dim=1)
    
    opt_ptr.zero_grad()
    logits = m_ptr(p_b, dec_in)
    loss = F.cross_entropy(logits.reshape(-1, 258), a_b.reshape(-1), ignore_index=256)
    loss.backward()
    opt_ptr.step()
    if ep % 20 == 0 or ep == 1:
        print(f"Epoch {ep:02d} | Loss: {loss.item():.4f}")

m_ptr.eval()
correct = 0
for p, exp, _ in suite_test['pointer']:
    p_b = [ord(c) for c in p]
    p_t = torch.tensor([p_b], dtype=torch.long, device=device)
    B, Sp = p_t.shape
    h_p = m_ptr.emb(p_t)
    mem, hn = m_ptr.enc(h_p)
    
    mem_keys = torch.zeros(B, m_ptr.num_slots, m_ptr.dim, device=device)
    mem_vals = torch.zeros(B, m_ptr.num_slots, m_ptr.dim, device=device)
    for t in range(Sp):
        x = mem[:, t, :]
        gate = torch.sigmoid(m_ptr.write_gate(x))
        k_t = F.normalize(m_ptr.k_proj(x), dim=-1)
        v_t = m_ptr.v_proj(x)
        sim = torch.bmm(k_t.unsqueeze(1), mem_keys.transpose(1, 2)).squeeze(1)
        slot_attn = F.softmax(sim * 10.0, dim=-1).unsqueeze(-1)
        mem_keys = mem_keys + gate.unsqueeze(-1) * slot_attn * k_t.unsqueeze(1)
        mem_vals = (1.0 - gate.unsqueeze(-1) * slot_attn) * mem_vals + (gate.unsqueeze(-1) * slot_attn) * v_t.unsqueeze(1)
        
    s_t = hn.mean(dim=0)
    gen = []
    prev_tok = torch.tensor([257], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(len(exp) + 4):
            x_t = m_ptr.emb(prev_tok)
            q = F.normalize(m_ptr.query_proj(s_t), dim=-1).unsqueeze(1)
            sim = torch.bmm(q, mem_keys.transpose(1, 2)).squeeze(1)
            slot_attn = F.softmax(sim * 10.0, dim=-1)
            read_v = torch.bmm(slot_attn.unsqueeze(1), mem_vals).squeeze(1)
            s_t = m_ptr.dec_cell(torch.cat([x_t, read_v], dim=-1), s_t)
            nxt = torch.argmax(m_ptr.head(s_t)[0]).item()
            if nxt in (ord('\n'), 256, 257):
                break
            gen.append(nxt)
            prev_tok = torch.tensor([nxt], dtype=torch.long, device=device)
    got = bytes(gen).decode('utf-8', errors='ignore').strip()
    if got == exp.strip():
        correct += 1

print(f"\n🎯 Sharp Key-Value Binding Cortex Pointer Accuracy: {correct}/{len(suite_test['pointer'])} = {correct/len(suite_test['pointer'])*100:.2f}%")
