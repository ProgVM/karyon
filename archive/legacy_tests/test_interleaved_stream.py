import sys, random
sys.path.insert(0, '.')
import torch
import torch.nn.functional as F
import karyon_agent
from multi_domain_benchmark import generate_multi_domain_suite

device = 'cuda' if torch.cuda.is_available() else 'cpu'
suite_train = generate_multi_domain_suite(seed=101)
suite_eval = generate_multi_domain_suite(seed=202)

# Interleave streams so network sees diverse domains continuously
train_items = []
for d in ['pointer', 'reversal', 'addition', 'parity', 'dyck']:
    for p, ans, full in suite_train[d]:
        train_items.append((d, p, ans, full))

random.seed(42)
random.shuffle(train_items) # Interleaved stream across life!

agent = karyon_agent.CoREAgent(embed_dim=256, device=device)
optimizer = torch.optim.AdamW(agent.parameters(), lr=1e-3, weight_decay=1e-4)

print("Training on interleaved continuous stream (N=1)...")
agent.train()
for step, (d, p, ans, full) in enumerate(train_items):
    b_seq = torch.tensor(list(full.encode('utf-8')), dtype=torch.long, device=device).unsqueeze(0)
    inp, tgt = b_seq[:, :-1], b_seq[:, 1:]
    logits = agent(inp, thinking_steps=4)
    loss = F.cross_entropy(logits.reshape(-1, 258), tgt.reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
    optimizer.step()

    if (step + 1) % 25 == 0:
        agent.execute_deep_allostatic_sleep(downscaling_factor=0.01, sprout_probability=0.75, prune_threshold=0.01)
        optimizer = torch.optim.AdamW(agent.parameters(), lr=1e-3, weight_decay=1e-4)

    if (step + 1) % 400 == 0:
        print(f"Step {step+1}/2000 | Loss: {loss.item():.4f}")

# Quick zero-shot eval on 50 samples of each
agent.eval()
print("\n--- Zero-Shot Eval on Interleaved Stream ---")
with torch.no_grad():
    for d in ['pointer', 'reversal', 'addition', 'parity', 'dyck']:
        correct = 0
        samples = suite_eval[d][:50]
        for p, ans, full in samples:
            p_bytes = torch.tensor(list(p.encode('utf-8')), dtype=torch.long, device=device).unsqueeze(0)
            curr = p_bytes
            gen_bytes = []
            for _ in range(len(ans) + 2):
                logits = agent(curr, thinking_steps=4)
                nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                val = nxt.item()
                if val == 10 or val == 257: # newline or EOS
                    break
                gen_bytes.append(val)
                curr = torch.cat([curr, nxt], dim=1)
            pred = bytes(gen_bytes).decode('utf-8', errors='ignore').strip()
            if pred == ans.strip():
                correct += 1
        print(f"Domain {d:10s} Acc: {(correct/len(samples))*100:.2f}% ({correct}/{len(samples)}) | Last Pred: {pred!r} vs Expected: {ans!r}")
