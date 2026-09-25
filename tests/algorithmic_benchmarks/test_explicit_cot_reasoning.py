import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==============================================================================
# SCIENTIFIC PROOF: Chain-of-Thought Scratchpad vs Direct Reflex
#
# Question: Why do all models (Transformers, RNNs, SDEs) fail to exceed 30% on Pointers?
#
# Because: "a=2; b=8; c=4; d=9; c=d; a=c; b=a; c="
# If the model is forced to output the answer DIRECTLY:
# Output: "9" (1 token!)
# The model has ONLY 1 token of compute to resolve 7 assignments and 3 hops of indirection!
#
# BUT if the model generates a CHAIN-OF-THOUGHT (CoT) Scratchpad:
# Output: "[c=d->9] [a=c->9] [b=a->9] c=9"
# The model has 25 tokens of sequential autoregressive compute!
# Each token resolves exactly ONE step of logic!
#
# Let's test this directly: Can an ordinary Transformer solve Pointers with 100% accuracy
# if trained with an explicit step-by-step reasoning trace?
# ==============================================================================

def generate_pointer_with_cot(seed=42, num_samples=500):
    random.seed(seed)
    samples = []
    variables = ['a', 'b', 'c', 'd']
    
    for _ in range(num_samples):
        state = {}
        history = []
        # Initial assignments
        for v in variables:
            val = random.randint(0, 9)
            state[v] = val
            history.append(f"{v}={val}")
            
        # 3 to 5 re-assignments
        trace_steps = []
        for _ in range(random.randint(3, 5)):
            lhs = random.choice(variables)
            rhs = random.choice(variables)
            if lhs != rhs:
                history.append(f"{lhs}={rhs}")
                state[lhs] = state[rhs]
                trace_steps.append(f"{lhs}->{state[lhs]}")
                
        target_var = random.choice(variables)
        prompt = "; ".join(history) + f"; {target_var}="
        
        # Direct answer:
        direct_ans = str(state[target_var])
        
        # CoT Answer:
        cot_ans = " " + " ".join(trace_steps) + f" => {state[target_var]}"
        samples.append((prompt, direct_ans, cot_ans))
        
    return samples

data_train = generate_pointer_with_cot(seed=42, num_samples=500)
data_test = generate_pointer_with_cot(seed=999, num_samples=200)

print(f"Sample Prompt: '{data_train[0][0]}'")
print(f"Direct Answer: '{data_train[0][1]}'")
print(f"CoT Answer   : '{data_train[0][2]}'")

def encode_pairs(samples, use_cot=False):
    max_p = max(len(p) for p, _, _ in samples)
    max_a = max(len(cot if use_cot else ans) for _, ans, cot in samples) + 1
    P = torch.full((len(samples), max_p), 256, dtype=torch.long, device=device)
    A = torch.full((len(samples), max_a), 256, dtype=torch.long, device=device)
    for i, (p, ans, cot) in enumerate(samples):
        target = cot if use_cot else ans
        p_b = [ord(c) for c in p]
        a_b = [ord(c) for c in target] + [ord('\n')]
        P[i, :len(p_b)] = torch.tensor(p_b, dtype=torch.long, device=device)
        A[i, :len(a_b)] = torch.tensor(a_b, dtype=torch.long, device=device)
    return P, A

P_tr_cot, A_tr_cot = encode_pairs(data_train, use_cot=True)
P_te_cot, A_te_cot = encode_pairs(data_test, use_cot=True)

class StandardTransformer(nn.Module):
    def __init__(self, vocab=258, dim=128, layers=4):
        super().__init__()
        self.dim = dim
        self.emb = nn.Embedding(vocab, dim)
        self.pos_p = nn.Parameter(torch.randn(1, 128, dim) * 0.02)
        self.pos_a = nn.Parameter(torch.randn(1, 128, dim) * 0.02)
        
        enc_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=8, dim_feedforward=512, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        
        dec_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=8, dim_feedforward=512, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=layers)
        self.head = nn.Linear(dim, vocab)

    def forward(self, p, a_in):
        B, Sp = p.shape
        Sa = a_in.shape[1]
        
        h_p = self.emb(p) + self.pos_p[:, :Sp, :]
        p_mask = (p == 256)
        mem = self.encoder(h_p, src_key_padding_mask=p_mask)
        
        h_a = self.emb(a_in) + self.pos_a[:, :Sa, :]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(Sa, device=p.device)
        out = self.decoder(h_a, mem, tgt_mask=tgt_mask, memory_key_padding_mask=p_mask)
        return self.head(out)

m_cot = StandardTransformer(layers=4).to(device)
opt_cot = torch.optim.AdamW(m_cot.parameters(), lr=0.001)

print("\n--- Training Standard Transformer with Chain-of-Thought (80 Epochs) ---")
for ep in range(1, 81):
    m_cot.train()
    idx = torch.randperm(len(P_tr_cot))[:128]
    p_b = P_tr_cot[idx]
    a_b = A_tr_cot[idx]
    B = p_b.shape[0]
    bos = torch.full((B, 1), 257, dtype=torch.long, device=device)
    dec_in = torch.cat([bos, a_b[:, :-1]], dim=1)
    
    opt_cot.zero_grad()
    logits = m_cot(p_b, dec_in)
    loss = F.cross_entropy(logits.reshape(-1, 258), a_b.reshape(-1), ignore_index=256)
    loss.backward()
    opt_cot.step()
    if ep % 20 == 0 or ep == 1:
        print(f"Epoch {ep:02d} | CoT Loss: {loss.item():.4f}")

m_cot.eval()
correct = 0
for p, direct_ans, cot_ans in data_test:
    p_b = [ord(c) for c in p]
    p_t = torch.tensor([p_b], dtype=torch.long, device=device)
    gen = []
    curr_tok = torch.tensor([[257]], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(len(cot_ans) + 5):
            logits = m_cot(p_t, curr_tok)
            nxt = torch.argmax(logits[0, -1, :]).item()
            if nxt in (ord('\n'), 256, 257):
                break
            gen.append(nxt)
            curr_tok = torch.cat([curr_tok, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
            
    got = bytes(gen).decode('utf-8', errors='ignore')
    # Check if final answer after '=> ' matches!
    if '=>' in got:
        final_extracted = got.split('=>')[-1].strip()
        if final_extracted == direct_ans.strip():
            correct += 1

acc = (correct / len(data_test)) * 100.0
print(f"\n🎯 Chain-of-Thought Pointer Accuracy: {correct}/{len(data_test)} = {acc:.2f}%")
for p, direct_ans, cot_ans in data_test[:6]:
    p_b = [ord(c) for c in p]
    p_t = torch.tensor([p_b], dtype=torch.long, device=device)
    gen = []
    curr_tok = torch.tensor([[257]], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(len(cot_ans) + 5):
            logits = m_cot(p_t, curr_tok)
            nxt = torch.argmax(logits[0, -1, :]).item()
            if nxt in (ord('\n'), 256, 257):
                break
            gen.append(nxt)
            curr_tok = torch.cat([curr_tok, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
    got = bytes(gen).decode('utf-8', errors='ignore')
    final_extracted = got.split('=>')[-1].strip() if '=>' in got else 'N/A'
    ok = '✅' if final_extracted == direct_ans.strip() else '❌'
    print(f"  {ok} Prompt: '{p}'")
    print(f"     Generated CoT: '{got}' | Final: '{final_extracted}' | Expected: '{direct_ans}'")
