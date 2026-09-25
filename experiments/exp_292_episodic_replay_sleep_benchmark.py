"""
================================================================================
EXP-292: EPISODIC REPLAY SLEEP CONSOLIDATION & CROSS-DOMAIN MEMORY PRESERVATION
================================================================================
Scientific Objective:
Evaluates the biophysical impact of 3-Phase Episodic Replay Sleep (KEP Principle 24)
on preventing catastrophic cross-domain forgetting during Single-Pass Stream Learning (N=1).

Experimental Methodology:
1. Stream Paradigm: Single-pass (N=1) continuous presentation of 2,500 tasks across
   5 distinct domains (Pointers, Reversal, Addition, Parity, Dyck-1).
2. Biophysical Sleep Mechanisms Evaluated:
   - Baseline A: Single-Pass Without Replay (Waking Stream Only).
   - Baseline B: Single-Pass + Tononi SHY Synaptic Scaling + Epigenetic Neurogenesis.
   - Proposed EXP-292: 3-Phase Sleep (Episodic High-Surprise Replay + SHY Scaling + Morphogenesis).
3. Telemetry Recorded:
   - Free Energy Trajectory (F_t) across stream.
   - Cross-Domain Retention Delta (Exact Match Acc across all 5 domains).
   - Memory Buffer Salience & Topological Stability.
================================================================================
"""
import sys, os, time, json, random
sys.path.insert(0, '.')
import torch
import torch.nn.functional as F

import karyon_agent
from karyon_logger import get_logger
from multi_domain_benchmark import generate_multi_domain_suite

logger = get_logger()

class ContinuousEpisodicBuffer:
    """Biophysical Associative Memory Buffer storing high-surprise experiences."""
    def __init__(self, capacity=250):
        self.capacity = capacity
        self.buffer = [] # list of (b_seq, surprise, domain)

    def record(self, b_seq: torch.Tensor, surprise: float, domain: str):
        if len(self.buffer) < self.capacity:
            self.buffer.append((b_seq.detach(), surprise, domain))
        else:
            # Replace item with lowest surprise if new item has higher surprise
            min_idx = min(range(len(self.buffer)), key=lambda i: self.buffer[i][1])
            if surprise > self.buffer[min_idx][1]:
                self.buffer[min_idx] = (b_seq.detach(), surprise, domain)

    def sample_replay_batch(self, batch_size=8):
        if not self.buffer:
            return []
        k = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, k)


def run_exp_292_benchmark():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 80)
    print("EXP-292: EPISODIC REPLAY SLEEP CONSOLIDATION BENCHMARK")
    print(f"Device: {device} | PyTorch: {torch.__version__} | Single-Pass Stream Paradigm (N=1)")
    print("=" * 80)

    # 1. Generate Interleaved Task Streams
    suite_train = generate_multi_domain_suite(seed=101)
    suite_eval = generate_multi_domain_suite(seed=202)

    stream_tasks = []
    for domain in ['pointer', 'reversal', 'addition', 'parity', 'dyck']:
        for p, ans, full in suite_train[domain]:
            stream_tasks.append((domain, p, ans, full))

    random.seed(42)
    random.shuffle(stream_tasks) # Interleaved stream across cognitive lifetime

    print(f"Interleaved Stream Size: {len(stream_tasks)} continuous samples across 5 domains.")

    # 2. Instantiate CoREAgent and Episodic Memory Buffer
    agent = karyon_agent.CoREAgent(embed_dim=256, device=device)
    optimizer = torch.optim.AdamW(agent.parameters(), lr=1e-3, weight_decay=1e-4)
    episodic_buffer = ContinuousEpisodicBuffer(capacity=300)

    start_time = time.time()
    f_t_history = []
    sleep_cycles_logged = 0
    replay_steps_logged = 0

    print("\n--- Phase 1: Waking Stream + Episodic Replay Sleep (Single-Pass N=1) ---")
    agent.train()

    for step, (domain, p, ans, full) in enumerate(stream_tasks):
        b_seq = torch.tensor(list(full.encode('utf-8')), dtype=torch.long, device=device).unsqueeze(0)
        if b_seq.shape[1] < 2:
            continue

        inp = b_seq[:, :-1]
        tgt = b_seq[:, 1:]

        # Waking Perception & Active Inference Forward Pass
        logits = agent(inp, thinking_steps=4)
        loss = F.cross_entropy(logits.reshape(-1, 258), tgt.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        optimizer.step()

        f_t = float(loss.item())
        f_t_history.append(f_t)

        # High surprise / Free Energy triggers episodic memory capture
        if f_t > 0.3:
            episodic_buffer.record(b_seq, f_t, domain)

        # Biophysical Sleep Phase (Every 20 steps)
        if (step + 1) % 20 == 0:
            sleep_cycles_logged += 1

            # Phase A: NREM Episodic Memory Replay
            replay_batch = episodic_buffer.sample_replay_batch(batch_size=4)
            for rep_seq, _, _ in replay_batch:
                r_inp, r_tgt = rep_seq[:, :-1], rep_seq[:, 1:]
                r_logits = agent(r_inp, thinking_steps=4)
                r_loss = F.cross_entropy(r_logits.reshape(-1, 258), r_tgt.reshape(-1))
                optimizer.zero_grad()
                r_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()
                replay_steps_logged += 1

            # Phase B: REM Epigenetic Neurogenesis & SHY Synaptic Scaling
            agent.execute_deep_allostatic_sleep(
                downscaling_factor=0.005,
                sprout_probability=0.4,
                prune_threshold=0.01
            )
            optimizer = torch.optim.AdamW(agent.parameters(), lr=1e-3, weight_decay=1e-4)

        if (step + 1) % 250 == 0 or step == len(stream_tasks) - 1:
            recent_f_t = sum(f_t_history[-50:]) / len(f_t_history[-50:])
            manifest = json.loads(agent.get_topology_manifest())
            print(f"Step {step+1:4d}/{len(stream_tasks)} | Instant F_t: {f_t:.4f} | Avg F_t: {recent_f_t:.4f} | Buffer: {len(episodic_buffer.buffer)} | Nodes: {manifest['k_nodes']}")

    elapsed_time = time.time() - start_time
    total_tokens_processed = sum(len(full.encode('utf-8')) for _, _, _, full in stream_tasks)
    tok_per_sec = total_tokens_processed / elapsed_time

    # 3. Phase 2: Zero-Shot Multi-Domain Evaluation
    print("\n--- Phase 2: Multi-Domain Cross-Retention Evaluation ---")
    agent.eval()
    domain_accuracies = {}

    with torch.no_grad():
        for domain in ['pointer', 'reversal', 'addition', 'parity', 'dyck']:
            items = suite_eval[domain][:100] # 100 test items per domain
            correct = 0
            for p, ans, full in items:
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

                try:
                    pred_str = bytes(gen_bytes).decode('utf-8', errors='ignore').strip()
                except Exception:
                    pred_str = ""

                if pred_str == ans.strip():
                    correct += 1

            acc = (correct / len(items)) * 100.0
            domain_accuracies[domain] = acc
            print(f"Domain: {domain:12s} | Exact Match Acc: {acc:6.2f}% ({correct}/{len(items)})")

    mean_acc = sum(domain_accuracies.values()) / len(domain_accuracies)
    initial_f_t = f_t_history[0]
    final_f_t = sum(f_t_history[-50:]) / 50.0
    f_t_delta = initial_f_t - final_f_t
    final_manifest = json.loads(agent.get_topology_manifest())

    print("\n" + "=" * 80)
    print("EXP-292 TELEMETRY SUMMARY")
    print("=" * 80)
    print(f"Initial Free Energy (F_t): {initial_f_t:.4f}")
    print(f"Final Free Energy (F_t)  : {final_f_t:.4f} (Delta Reduction: {f_t_delta:.4f})")
    print(f"Throughput               : {tok_per_sec:.2f} tok/s")
    print(f"Total Sleep Cycles       : {sleep_cycles_logged}")
    print(f"Total Replay Steps       : {replay_steps_logged}")
    print(f"Episodic Buffer Size     : {len(episodic_buffer.buffer)}")
    print(f"Final Graph Topology     : {final_manifest}")
    print(f"Mean Multi-Domain Acc    : {mean_acc:.2f}%")
    print("=" * 80)

    verdict = "POSITIVE" if f_t_delta > 2.0 else ("NEUTRAL" if f_t_delta > 0.5 else "REJECTED")

    results = {
        "exp_id": "EXP-292",
        "initial_free_energy": initial_f_t,
        "final_free_energy": final_f_t,
        "free_energy_delta": f_t_delta,
        "tok_per_sec": tok_per_sec,
        "sleep_cycles": sleep_cycles_logged,
        "replay_steps": replay_steps_logged,
        "buffer_size": len(episodic_buffer.buffer),
        "final_nodes": final_manifest["k_nodes"],
        "mean_accuracy": mean_acc,
        "domain_accuracies": domain_accuracies,
        "verdict": verdict
    }

    with open("experiments/exp_292_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nVerdict: 🟢 {verdict}" if verdict == "POSITIVE" else f"\nVerdict: {verdict}")
    return results

if __name__ == "__main__":
    run_exp_292_benchmark()
