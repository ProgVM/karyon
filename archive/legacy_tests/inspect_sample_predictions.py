import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import karyon_core
from karyon_agent import CoREAgent
from multi_domain_benchmark import generate_multi_domain_suite

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = CoREAgent(vocab_size=258, embed_dim=128, device=str(device))
agent.add_node("bilinear", "BilinearMultiplicative", False, 1.0)
agent.add_node("hopfield", "ContinuousHopfield", False, 1.0)

suite_train = generate_multi_domain_suite(seed=42)
suite_test = generate_multi_domain_suite(seed=999)

all_train = []
for d, s in suite_train.items():
    all_train.extend(s)

def encode_pairs(samples):
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

P_train, A_train = encode_pairs(all_train)
opt = torch.optim.AdamW(agent.parameters(), lr=0.003)

for ep in range(1, 81):
    agent.train()
    idx = torch.randperm(len(P_train))[:128]
    p_b = P_train[idx]
    a_b = A_train[idx]
    full_seq = torch.cat([p_b, a_b], dim=1)
    targets = full_seq[:, 1:].clone()
    targets[:, :p_b.shape[1]-1] = 256
    opt.zero_grad()
    logits = agent(full_seq[:, :-1], thinking_steps=3)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 258), targets.reshape(-1), ignore_index=256)
    loss.backward()
    opt.step()

agent.eval()
print("\n=== SAMPLE GENERATIONS PER DOMAIN ===")
for domain in ['pointer', 'reversal', 'addition', 'parity', 'dyck']:
    samples = suite_test[domain][:3]
    print(f"\n--- Domain: {domain} ---")
    for p, exp, _ in samples:
        p_b = [ord(c) for c in p]
        curr_seq = torch.tensor([p_b], dtype=torch.long, device=device)
        gen = []
        with torch.no_grad():
            for _ in range(len(exp) + 5):
                logits = agent(curr_seq, thinking_steps=3)
                nxt = torch.argmax(logits[0, -1, :]).item()
                if nxt in (ord('\n'), 256, 257):
                    break
                gen.append(nxt)
                curr_seq = torch.cat([curr_seq, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
        got = bytes(gen).decode('utf-8', errors='ignore')
        print(f"  Prompt: {p!r:30s} | Exp: {exp!r:10s} | Got: {got!r:10s}")
