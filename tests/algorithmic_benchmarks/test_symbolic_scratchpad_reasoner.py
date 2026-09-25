import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from multi_domain_benchmark import generate_multi_domain_suite

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==============================================================================
# FUNDAMENTAL DISCOVERY: Why End-to-End Single-Pass Fails on Multi-Step Reasoning
#
# Consider: "a=2; b=8; c=4; d=9; c=d; a=c; b=a; c="
# A human does NOT output '9' in one subcortical reflex.
# A human executes an INTERNAL SCRATCHPAD (Mental Sandbox):
# State_0: a=2, b=8, c=4, d=9
# State_1 (c=d): c becomes 9
# State_2 (a=c): a becomes 9
# State_3 (b=a): b becomes 9
# Query c -> Read 9.
#
# What if the model generates its own LATENT THOUGHT TRACE (Recurrent Sandbox Rollout)
# without requiring human supervision for each intermediate step?
#
# Architecture: Recurrent Graph Memory Hop (RGMH):
# 1. An Edge Matrix A_t in R^{14 x 14} (representing nodes: 'a', 'b', 'c', 'd', '0'..'9')
# 2. Each token transition updates the graph adjacency matrix A_t.
# 3. Transitive Closure of A_t gives the exact reachable digit!
#    Reachable_Matrix = (A_t)^K
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

class LatentGraphTransitionCortex(nn.Module):
    def __init__(self, vocab=258, dim=128, num_nodes=14):
        super().__init__()
        self.dim = dim
        self.num_nodes = num_nodes
        self.emb = nn.Embedding(vocab, dim)
        self.enc = nn.GRU(dim, dim, batch_first=True, bidirectional=True)
        
        # Source & Target Node Classifiers
        self.src_head = nn.Linear(dim * 2, num_nodes)
        self.dst_head = nn.Linear(dim * 2, num_nodes)
        self.edge_gate = nn.Linear(dim * 2, 1)
        
        self.head = nn.Linear(dim * 2 + num_nodes, vocab)

    def forward(self, p, a_in):
        B, Sp = p.shape
        Sa = a_in.shape[1]
        
        h_p = self.emb(p)
        mem, hn = self.enc(h_p) # [B, Sp, 2*D]
        
        # Continuous Adjacency Matrix A: [B, num_nodes, num_nodes]
        # Nodes 0..3: a, b, c, d
        # Nodes 4..13: digits 0..9
        A = torch.eye(self.num_nodes, device=p.device).unsqueeze(0).repeat(B, 1, 1)
        
        for t in range(Sp):
            x = mem[:, t, :]
            gate = torch.sigmoid(self.edge_gate(x)).unsqueeze(-1) # [B, 1, 1]
            src = F.softmax(self.src_head(x), dim=-1).unsqueeze(-1) # [B, num_nodes, 1] (LHS)
            dst = F.softmax(self.dst_head(x), dim=-1).unsqueeze(1)  # [B, 1, num_nodes] (RHS)
            
            # An assignment LHS <- RHS creates a directed edge: src -> dst
            # First clear old outgoing edges from src:
            # A[src, :] = 0
            # A[src, dst] = 1
            A = (1.0 - gate * src) * A + (gate * src) * dst
            
        # Transitive Closure via matrix power: A_closure = A^6
        # Propagates multi-hop pointers (c -> d -> 9 becomes c -> 9 directly!)
        A_clos = A
        for _ in range(5):
            A_clos = torch.bmm(A_clos, A)
            A_clos = A_clos / (A_clos.sum(dim=-1, keepdim=True) + 1e-8)
            
        # Query target register from final character (e.g., 'c=')
        q_src = F.softmax(self.src_head(mem[:, -1, :]), dim=-1).unsqueeze(1) # [B, 1, num_nodes]
        reachable_dist = torch.bmm(q_src, A_clos).squeeze(1) # [B, num_nodes]
        
        combined = torch.cat([mem[:, -1, :], reachable_dist], dim=-1)
        logits = self.head(combined).unsqueeze(1)
        return logits.repeat(1, Sa, 1)

m_lg = LatentGraphTransitionCortex().to(device)
opt_lg = torch.optim.AdamW(m_lg.parameters(), lr=0.003)

print("Training Latent Graph Transition Cortex on Pointers (80 Epochs)...")
for ep in range(1, 81):
    m_lg.train()
    idx = torch.randperm(len(P_ptr_tr))[:128]
    p_b = P_ptr_tr[idx]
    a_b = A_ptr_tr[idx]
    B = p_b.shape[0]
    bos = torch.full((B, 1), 257, dtype=torch.long, device=device)
    dec_in = torch.cat([bos, a_b[:, :-1]], dim=1)
    
    opt_lg.zero_grad()
    logits = m_lg(p_b, dec_in)
    loss = F.cross_entropy(logits[:, 0, :], a_b[:, 0], ignore_index=256)
    loss.backward()
    opt_lg.step()
    if ep % 20 == 0 or ep == 1:
        print(f"Epoch {ep:02d} | Loss: {loss.item():.4f}")

m_lg.eval()
correct = 0
for p, exp, _ in suite_test['pointer']:
    p_b = [ord(c) for c in p]
    p_t = torch.tensor([p_b], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = m_lg(p_t, torch.zeros(1, 1, dtype=torch.long, device=device))
        got = chr(torch.argmax(logits[0, 0, :]).item())
        if got == exp.strip():
            correct += 1

print(f"\n🎯 Latent Graph Transition Cortex Pointer Accuracy: {correct}/{len(suite_test['pointer'])} = {correct/len(suite_test['pointer'])*100:.2f}%")
for p, exp, _ in suite_test['pointer'][:8]:
    p_b = [ord(c) for c in p]
    p_t = torch.tensor([p_b], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = m_lg(p_t, torch.zeros(1, 1, dtype=torch.long, device=device))
        got = chr(torch.argmax(logits[0, 0, :]).item())
        ok = '✅' if got == exp.strip() else '❌'
        print(f"  {ok} '{p}' -> Target: '{exp.strip()}' | Got: '{got}'")
