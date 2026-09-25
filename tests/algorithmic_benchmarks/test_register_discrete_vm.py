import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from multi_domain_benchmark import generate_multi_domain_suite

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Let's inspect why backward GRU only gets 23%:
# The sequence is raw bytes: 'a', '=', '2', ';', ' ', 'b', '=', '8', ...
# A character-level GRU sees 40+ raw byte tokens. The transition logic is:
# If byte == '=' -> store LHS / RHS tokens.
# If we use a Gated Attentive Deliberation Transformer (System 2 Latent Reasoning):
# Self-Attention across the 40 bytes allows 'c=' at the end to directly attend to 'c=d',
# which in turn attends to 'd=9', which attends to '9'!
# That is a 3-layer Transformer Attention! (Each layer hops 1 pointer step: c -> d -> 9)!

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

class DeliberativeHoppingTransformer(nn.Module):
    def __init__(self, vocab=258, dim=128, num_layers=4):
        super().__init__()
        self.dim = dim
        self.emb = nn.Embedding(vocab, dim)
        self.pos = nn.Parameter(torch.randn(1, 64, dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=4, dim_feedforward=256, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(dim, vocab)

    def forward(self, p, a_in):
        B, Sp = p.shape
        Sa = a_in.shape[1]
        
        h = self.emb(p) + self.pos[:, :Sp, :]
        p_mask = (p == 256)
        
        # Self-Attention across sequence (allows multi-step pointer hopping!)
        mem = self.transformer(h, src_key_padding_mask=p_mask)
        
        # Readout from the last token ('=')
        logits = self.head(mem[:, -1, :]).unsqueeze(1)
        return logits.repeat(1, Sa, 1)

m_tr = DeliberativeHoppingTransformer().to(device)
opt_tr = torch.optim.AdamW(m_tr.parameters(), lr=0.003)

print("Testing Deliberative Hopping Transformer (4 Layers) on Pointers (60 Epochs)...")
for ep in range(1, 61):
    m_tr.train()
    idx = torch.randperm(len(P_ptr_tr))[:128]
    p_b = P_ptr_tr[idx]
    a_b = A_ptr_tr[idx]
    B = p_b.shape[0]
    bos = torch.full((B, 1), 257, dtype=torch.long, device=device)
    dec_in = torch.cat([bos, a_b[:, :-1]], dim=1)
    
    opt_tr.zero_grad()
    logits = m_tr(p_b, dec_in)
    loss = F.cross_entropy(logits[:, 0, :], a_b[:, 0], ignore_index=256)
    loss.backward()
    opt_tr.step()
    if ep % 20 == 0 or ep == 1:
        print(f"Epoch {ep:02d} | Loss: {loss.item():.4f}")

m_tr.eval()
correct = 0
for p, exp, _ in suite_test['pointer']:
    p_b = [ord(c) for c in p]
    p_t = torch.tensor([p_b], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = m_tr(p_t, torch.zeros(1, 1, dtype=torch.long, device=device))
        got = chr(torch.argmax(logits[0, 0, :]).item())
        if got == exp.strip():
            correct += 1

print(f"\n🎯 Deliberative Hopping Transformer Pointer Accuracy: {correct}/{len(suite_test['pointer'])} = {correct/len(suite_test['pointer'])*100:.2f}%")
for p, exp, _ in suite_test['pointer'][:8]:
    p_b = [ord(c) for c in p]
    p_t = torch.tensor([p_b], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = m_tr(p_t, torch.zeros(1, 1, dtype=torch.long, device=device))
        got = chr(torch.argmax(logits[0, 0, :]).item())
        ok = '✅' if got == exp.strip() else '❌'
        print(f"  {ok} '{p}' -> Target: '{exp.strip()}' | Got: '{got}'")
