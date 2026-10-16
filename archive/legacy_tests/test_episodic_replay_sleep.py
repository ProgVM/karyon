import sys, random
sys.path.insert(0, '.')
import torch
import torch.nn.functional as F
import karyon_agent
from multi_domain_benchmark import generate_multi_domain_suite

device = 'cuda' if torch.cuda.is_available() else 'cpu'
suite_train = generate_multi_domain_suite(seed=101)
suite_eval = generate_multi_domain_suite(seed=202)

# Interleaved single-pass stream
train_items = []
for d in ['pointer', 'reversal', 'addition', 'parity', 'dyck']:
    for p, ans, full in suite_train[d]:
        train_items.append((d, p, ans, full))

random.seed(42)
random.shuffle(train_items)

agent = karyon_agent.CoREAgent(embed_dim=256, device=device)
optimizer = torch.optim.AdamW(agent.parameters(), lr=1e-3, weight_decay=1e-4)

# Episodic Memory Buffer (Stores high-surprise / high-salience experiences)
class ContinuousEpisodicBuffer:
    def __init__(self, capacity=200):
        self.capacity = capacity
        self.buffer = [] # list of (b_seq, surprise)

    def record(self, b_seq, surprise):
        if len(self.buffer) < self.capacity:
            self.buffer.append((b_seq.detach(), surprise))
        else:
            # Replace lowest surprise item if current surprise is higher
            min_idx = min(range(len(self.buffer)), key=lambda i: self.buffer[i][1])
            if surprise > self.buffer[min_idx][1]:
                self.buffer[min_idx] = (b_seq.detach(), surprise)

    def sample_replay_batch(self, batch_size=8):
        if not self.buffer:
            return []
        k = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, k)

episodic_memory = ContinuousEpisodicBuffer(capacity=300)

print("Starting Single-Pass Stream Learning with Biophysical Episodic Replay Sleep...")
agent.train()
sleep_count = 0

for step, (d, p, ans, full) in enumerate(train_items):
    b_seq = torch.tensor(list(full.encode('utf-8')), dtype=torch.long, device=device).unsqueeze(0)
    inp, tgt = b_seq[:, :-1], b_seq[:, 1:]
    
    # 1. Waking Step: Perception & Active Inference
    logits = agent(inp, thinking_steps=4)
    loss = F.cross_entropy(logits.reshape(-1, 258), tgt.reshape(-1))
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
    optimizer.step()
    
    f_t = float(loss.item())
    
    # Record to Episodic Memory if surprise/Free Energy is non-trivial
    if f_t > 0.5:
        episodic_memory.record(b_seq, f_t)
        
    # 2. Phase-Sleep Trigger (Every 25 steps or when Free Energy spikes)
    if (step + 1) % 25 == 0:
        sleep_count += 1
        # Phase A: Replay high-surprise episodic memories (NREM Replay Phase)
        replay_samples = episodic_memory.sample_replay_batch(batch_size=4)
        for rep_seq, _ in replay_samples:
            r_inp, r_tgt = rep_seq[:, :-1], rep_seq[:, 1:]
            r_logits = agent(r_inp, thinking_steps=4)
            r_loss = F.cross_entropy(r_logits.reshape(-1, 258), r_tgt.reshape(-1))
            optimizer.zero_grad()
            r_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            optimizer.step()
            
        # Phase B: Tononi SHY Synaptic Scaling + Epigenetic Morphogenesis (REM Sleep Phase)
        agent.execute_deep_allostatic_sleep(
            downscaling_factor=0.01,
            sprout_probability=0.5,
            prune_threshold=0.01
        )
        optimizer = torch.optim.AdamW(agent.parameters(), lr=1e-3, weight_decay=1e-4)

    if (step + 1) % 400 == 0:
        print(f"Step {step+1:4d}/2000 | Instant F_t: {f_t:.4f} | Memory Size: {len(episodic_memory.buffer)} | Sleep Cycles: {sleep_count}")

# 3. Comprehensive Evaluation Across All 5 Domains
print("\n--- Zero-Shot Streaming Evaluation (50 items/domain) ---")
agent.eval()
domain_results = {}
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
                if val == 10 or val == 257:
                    break
                gen_bytes.append(val)
                curr = torch.cat([curr, nxt], dim=1)
            pred = bytes(gen_bytes).decode('utf-8', errors='ignore').strip()
            if pred == ans.strip():
                correct += 1
        acc = (correct / len(samples)) * 100.0
        domain_results[d] = acc
        print(f"Domain: {d:10s} | Exact Match: {acc:5.2f}% ({correct}/{len(samples)}) | Ex: {pred!r} vs {ans!r}")

print(f"\nMean Multi-Domain Accuracy: {sum(domain_results.values())/len(domain_results):.2f}%")
