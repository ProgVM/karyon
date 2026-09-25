import sys, os
sys.path.insert(0, '.')
import math, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from multi_domain_benchmark import generate_multi_domain_suite
from tests.test_dpr_clean_gaze import NaturalSaccadicMind

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

suite_train = generate_multi_domain_suite(seed=42)
suite_test = generate_multi_domain_suite(seed=999)

for domain in ['reversal', 'dyck', 'pointer']:
    print(f"\n" + "=" * 60)
    print(f"=== TRAINING ISOLATED ON DOMAIN: {domain.upper()} ===")
    print("=" * 60)
    
    train_samples = suite_train[domain]
    test_samples = suite_test[domain]
    
    def encode_data(samples):
        max_p = max(len(p) for p, _, _ in samples)
        max_a = max(len(a) for _, a, _ in samples) + 1
        P = torch.full((len(samples), max_p), 256, dtype=torch.long, device=device)
        A_in = torch.full((len(samples), max_a), 256, dtype=torch.long, device=device)
        A_tgt = torch.full((len(samples), max_a), 256, dtype=torch.long, device=device)
        for i, (p, a, _) in enumerate(samples):
            p_b = [ord(c) for c in p]
            a_b = [ord(c) for c in a] + [ord('\n')]
            P[i, :len(p_b)] = torch.tensor(p_b, dtype=torch.long, device=device)
            a_in_seq = [257] + a_b[:-1]
            A_in[i, :len(a_in_seq)] = torch.tensor(a_in_seq, dtype=torch.long, device=device)
            A_tgt[i, :len(a_b)] = torch.tensor(a_b, dtype=torch.long, device=device)
        return P, A_in, A_tgt

    P_tr, Ain_tr, Atgt_tr = encode_data(train_samples)
    
    agent = NaturalSaccadicMind(vocab_size=258, dim=128, wave_steps=4).to(device)
    opt = torch.optim.AdamW(agent.parameters(), lr=0.003, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80, eta_min=1e-4)
    
    for ep in range(1, 81):
        agent.train()
        opt.zero_grad()
        logits = agent.forward_train(P_tr, Ain_tr)
        loss = F.cross_entropy(logits.reshape(-1, 258), Atgt_tr.reshape(-1), ignore_index=256)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        opt.step()
        scheduler.step()
        if ep % 20 == 0:
            print(f"  Epoch {ep:02d} | Loss: {loss.item():.4f}")
            
    agent.eval()
    correct = 0
    sample_preds = []
    for p, exp, _ in test_samples:
        p_b = [ord(c) for c in p]
        curr_p = torch.tensor([p_b], dtype=torch.long, device=device)
        with torch.no_grad():
            gen_tokens = agent.generate(curr_p, max_len=len(exp)+2)[0].tolist()
            out_b = []
            for t in gen_tokens:
                if t in (ord('\n'), 256, 257):
                    break
                out_b.append(t)
            got = bytes(out_b).decode('utf-8', errors='ignore')
            is_match = (got.strip() == exp.strip())
            if is_match:
                correct += 1
            if len(sample_preds) < 3:
                sample_preds.append((p, exp, got, is_match))
    acc = (correct / len(test_samples)) * 100.0
    print(f"\n>>> ISOLATED ACCURACY [{domain.upper()}]: {acc:.2f}% ({correct}/{len(test_samples)})")
    for p, exp, got, ok in sample_preds:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status} | Prompt: {p!r:30s} | Exp: {exp!r:10s} | Got: {got!r:10s}")
