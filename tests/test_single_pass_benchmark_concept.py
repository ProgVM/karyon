import sys, os
sys.path.insert(0, '.')
import torch
import torch.nn.functional as F
import karyon_agent
from multi_domain_benchmark import generate_multi_domain_suite

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running Single-Pass benchmark verification on device: {device}")

suite = generate_multi_domain_suite(seed=42)
agent = karyon_agent.CoREAgent(device=device)
optimizer = torch.optim.AdamW(agent.parameters(), lr=1e-3, weight_decay=1e-4)

# We evaluate strictly in a continuous STREAM of examples (N=1 single pass)
# No repeated epochs over shuffled data.
all_tasks = []
for domain, items in suite.items():
    for p, ans, full in items[:40]: # Stream of items
        all_tasks.append((domain, p, ans, full))

print(f"Total stream items: {len(all_tasks)}")
rolling_f_t = []
sleep_count = 0

for step, (domain, p, ans, full) in enumerate(all_tasks):
    agent.train()
    # UTF-8 byte stream input
    b_seq = torch.tensor(list(full.encode('utf-8')), dtype=torch.long, device=device).unsqueeze(0)
    if b_seq.shape[1] < 2:
        continue
    
    inp = b_seq[:, :-1]
    tgt = b_seq[:, 1:]
    
    logits = agent(inp, thinking_steps=4)
    loss = F.cross_entropy(logits.reshape(-1, 258), tgt.reshape(-1))
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
    optimizer.step()
    
    f_t = float(loss.item())
    rolling_f_t.append(f_t)
    
    # Biophysical allostasis: if surprise is high or periodically during stream, enter brief sleep consolidation
    if (step + 1) % 25 == 0:
        sleep_res = agent.execute_deep_allostatic_sleep(downscaling_factor=0.01, sprout_probability=0.8, prune_threshold=0.01)
        sleep_count += 1
        optimizer = torch.optim.AdamW(agent.parameters(), lr=1e-3, weight_decay=1e-4) # Re-bind new dynamic parameters

    if (step + 1) % 50 == 0:
        avg_f = sum(rolling_f_t[-20:]) / 20.0
        print(f"Stream Step {step+1:3d} | Domain: {domain:10s} | Instant F_t: {f_t:.4f} | Avg F_t: {avg_f:.4f} | Topology Nodes: {agent.graph.k_nodes}")

print(f"\nFinal Stream Finished! Total Sleep Cycles: {sleep_count}")
print(f"Final Graph Topology: {agent.get_topology_manifest()}")
