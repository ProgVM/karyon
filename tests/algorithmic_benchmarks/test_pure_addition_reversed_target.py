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

# Let's test the hypothesis: If the addition target is generated in natural computational order (least significant digit first),
# e.g., 714 + 178 = 892 -> internal computation outputs 2, 9, 8 (LSB to MSB)
# Can an autoregressive network solve addition when the output is LSB first?

suite_train = generate_multi_domain_suite(seed=42)
suite_test = generate_multi_domain_suite(seed=999)

add_train = suite_train['addition']
add_test = suite_test['addition']

# Reverse target string: '892' -> '298'
def encode_lsb_task(samples):
    max_p = max(len(p) for p, _, _ in samples)
    max_a = max(len(a) for _, a, _ in samples) + 1
    P = torch.full((len(samples), max_p), 256, dtype=torch.long, device=device)
    A = torch.full((len(samples), max_a), 256, dtype=torch.long, device=device)
    for i, (p, a, _) in enumerate(samples):
        p_b = [ord(c) for c in p]
        a_rev = a[::-1] # REVERSE DIGITS: LSB first!
        a_b = [ord(c) for c in a_rev] + [ord('\n')]
        P[i, :len(p_b)] = torch.tensor(p_b, dtype=torch.long, device=device)
        A[i, :len(a_b)] = torch.tensor(a_b, dtype=torch.long, device=device)
    return P, A

P_tr, A_tr = encode_lsb_task(add_train)

class LSBAdditionTransformer(nn.Module):
    def __init__(self, vocab=258, dim=128, num_layers=4):
        super().__init__()
        self.dim = dim
        self.emb = nn.Embedding(vocab, dim)
        self.pos_emb = nn.Parameter(torch.randn(1, 128, dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=8, dim_feedforward=512, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=8, dim_feedforward=512, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.head = nn.Linear(dim, vocab)

    def forward(self, p, a_in):
        B, Sp = p.shape
        Sa = a_in.shape[1]
        
        h_p = self.emb(p) + self.pos_emb[:, :Sp, :]
        p_mask = (p == 256)
        
        mem = self.encoder(h_p, src_key_padding_mask=p_mask)
        h_a = self.emb(a_in) + self.pos_emb[:, :Sa, :]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(Sa, device=p.device)
        
        out = self.decoder(h_a, mem, tgt_mask=tgt_mask, memory_key_padding_mask=p_mask)
        return self.head(out)

m = LSBAdditionTransformer(num_layers=4).to(device)
opt = torch.optim.AdamW(m.parameters(), lr=0.001)

print("Training LSB (Reverse Order) Transformer on Addition (100 Epochs)...")
for ep in range(1, 101):
    m.train()
    idx = torch.randperm(len(P_tr))[:128]
    p_b = P_tr[idx]
    a_b = A_tr[idx]
    B = p_b.shape[0]
    bos = torch.full((B, 1), 257, dtype=torch.long, device=device)
    dec_in = torch.cat([bos, a_b[:, :-1]], dim=1)
    
    opt.zero_grad()
    logits = m(p_b, dec_in)
    loss = F.cross_entropy(logits.reshape(-1, 258), a_b.reshape(-1), ignore_index=256)
    loss.backward()
    opt.step()
    if ep % 20 == 0 or ep == 1:
        print(f"Epoch {ep:03d} | Loss: {loss.item():.4f}")

m.eval()
correct = 0
for p, exp, _ in add_test:
    p_b = [ord(c) for c in p]
    p_t = torch.tensor([p_b], dtype=torch.long, device=device)
    gen = []
    curr_tok = torch.tensor([[257]], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(len(exp) + 2):
            logits = m(p_t, curr_tok)
            nxt = torch.argmax(logits[0, -1, :]).item()
            if nxt in (ord('\n'), 256, 257):
                break
            gen.append(nxt)
            curr_tok = torch.cat([curr_tok, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
            
    # Reverse generated string back to MSB for evaluation!
    got_lsb = bytes(gen).decode('utf-8', errors='ignore').strip()
    got_msb = got_lsb[::-1]
    if got_msb == exp.strip():
        correct += 1

print(f"\n🎯 LSB-Reversed Target Addition Accuracy: {correct}/{len(add_test)} = {correct/len(add_test)*100:.2f}%")
for p, exp, _ in add_test[:8]:
    p_b = [ord(c) for c in p]
    p_t = torch.tensor([p_b], dtype=torch.long, device=device)
    gen = []
    curr_tok = torch.tensor([[257]], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(len(exp) + 2):
            logits = m(p_t, curr_tok)
            nxt = torch.argmax(logits[0, -1, :]).item()
            if nxt in (ord('\n'), 256, 257):
                break
            gen.append(nxt)
            curr_tok = torch.cat([curr_tok, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
    got_lsb = bytes(gen).decode('utf-8', errors='ignore').strip()
    got_msb = got_lsb[::-1]
    ok = '✅' if got_msb == exp.strip() else '❌'
    print(f"  {ok} '{p}' -> Target: '{exp.strip()}' | Got LSB: '{got_lsb}' -> MSB: '{got_msb}'")
