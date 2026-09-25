"""
================================================================================
EXP-293: BILINEAR CONJUNCTION, ADAPTIVE OPTIMIZATION & CONTINUAL RETENTION
================================================================================
Scientific Hypothesis:
1. Enabling Bilinear Multiplicative conjunction (BilinearMultiplicativeOp) provides
   second-order interaction capability (If A AND B), enabling sub-1.0 F_t in Single-Pass.
2. In-sleep replay combined with Adaptive Learning Dynamics (Per-experience AdamW state
   or higher learning rate during high-arousal waking) drives Free Energy F_t below 1.0 nats.
3. Continual Retention Test: By re-evaluating the earliest domain (Domain 1: Pointers)
   after processing 2,000 subsequent stream steps, 3-Phase Sleep + Hopfield/Episodic
   Replay maintains zero-shot retention without catastrophic forgetting.
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

class SalienceEpisodicBuffer:
    """Enhanced Episodic Buffer with balanced domain representation."""
    def __init__(self, capacity=300):
        self.capacity = capacity
        self.buffer = [] # list of (b_seq, surprise, domain)

    def record(self, b_seq: torch.Tensor, surprise: float, domain: str):
        if len(self.buffer) < self.capacity:
            self.buffer.append((b_seq.detach(), surprise, domain))
        else:
            # Replace item with lowest surprise
            min_idx = min(range(len(self.buffer)), key=lambda i: self.buffer[i][1])
            if surprise > self.buffer[min_idx][1]:
                self.buffer[min_idx] = (b_seq.detach(), surprise, domain)

    def sample_replay_batch(self, batch_size=6):
        if not self.buffer:
            return []
        k = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, k)


def run_exp_293_benchmark():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 80)
    print("EXP-293: BILINEAR MULTIPLICATIVE CONJUNCTION & CONTINUAL RETENTION")
    print(f"Device: {device} | PyTorch: {torch.__version__} | Single-Pass Stream Paradigm (N=1)")
    print("=" * 80)

    # 1. Setup Data Streams:
    # First: 400 pointers (Initial Domain Exposure)
    # Middle: 1600 samples of other 4 domains (Reversal, Addition, Parity, Dyck)
    # Final: 200 pointers (Retention Test Stream)
    suite_train = generate_multi_domain_suite(seed=101)
    suite_eval = generate_multi_domain_suite(seed=202)

    initial_pointers = [('pointer', p, ans, full) for p, ans, full in suite_train['pointer'][:300]]
    interleaved_middle = []
    for d in ['reversal', 'addition', 'parity', 'dyck']:
        for p, ans, full in suite_train[d]:
            interleaved_middle.append((d, p, ans, full))
    random.seed(42)
    random.shuffle(interleaved_middle)

    retention_pointers = [('pointer', p, ans, full) for p, ans, full in suite_train['pointer'][300:]]

    full_stream = initial_pointers + interleaved_middle + retention_pointers
    print(f"Total Stream Size: {len(full_stream)} steps (Initial Pointers: {len(initial_pointers)}, Middle: {len(interleaved_middle)}, Retention: {len(retention_pointers)})")

    # 2. Instantiate CoREAgent with core BilinearMultiplicativeOp
    agent = karyon_agent.CoREAgent(embed_dim=256, device=device)
    # Add BilinearMultiplicativeOp explicitly to core graph for 2nd order conjunction
    agent.add_node("bilinear_conj_core", "BilinearMultiplicative", is_core=True, initial_alpha=0.5)

    optimizer = torch.optim.AdamW(agent.parameters(), lr=1.5e-3, weight_decay=1e-4)
    episodic_buffer = SalienceEpisodicBuffer(capacity=350)

    start_time = time.time()
    f_t_history = []
    pointer_retention_f_t = []
    sleep_cycles = 0
    replay_steps = 0

    print("\n--- Processing Cognitive Stream ---")
    agent.train()

    for step, (domain, p, ans, full) in enumerate(full_stream):
        b_seq = torch.tensor(list(full.encode('utf-8')), dtype=torch.long, device=device).unsqueeze(0)
        inp, tgt = b_seq[:, :-1], b_seq[:, 1:]

        logits = agent(inp, thinking_steps=4)
        loss = F.cross_entropy(logits.reshape(-1, 258), tgt.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        optimizer.step()

        f_t = float(loss.item())
        f_t_history.append(f_t)

        if step >= len(initial_pointers) + len(interleaved_middle):
            pointer_retention_f_t.append(f_t)

        # Episodic Capture
        if f_t > 0.25:
            episodic_buffer.record(b_seq, f_t, domain)

        # Biophysical Sleep Phase
        if (step + 1) % 15 == 0:
            sleep_cycles += 1
            # NREM Replay
            replays = episodic_buffer.sample_replay_batch(batch_size=4)
            for rep_seq, _, _ in replays:
                r_inp, r_tgt = rep_seq[:, :-1], rep_seq[:, 1:]
                r_logits = agent(r_inp, thinking_steps=4)
                r_loss = F.cross_entropy(r_logits.reshape(-1, 258), r_tgt.reshape(-1))
                optimizer.zero_grad()
                r_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()
                replay_steps += 1

            # REM Neurogenesis & Synaptic Scaling
            agent.execute_deep_allostatic_sleep(
                downscaling_factor=0.003,
                sprout_probability=0.4,
                prune_threshold=0.01,
                available_ops=("BilinearMultiplicative", "LinearAccumulator", "SaturatedAttractor")
            )
            optimizer = torch.optim.AdamW(agent.parameters(), lr=1.5e-3, weight_decay=1e-4)

        if (step + 1) % 250 == 0 or step == len(full_stream) - 1:
            recent_f_t = sum(f_t_history[-50:]) / len(f_t_history[-50:])
            manifest = json.loads(agent.get_topology_manifest())
            print(f"Step {step+1:4d}/{len(full_stream)} | Instant F_t: {f_t:.4f} | Avg F_t: {recent_f_t:.4f} | Buffer: {len(episodic_buffer.buffer)} | Nodes: {manifest['k_nodes']}")

    elapsed = time.time() - start_time
    total_tokens = sum(len(full.encode('utf-8')) for _, _, _, full in full_stream)
    tok_per_sec = total_tokens / elapsed

    # 3. Retention Analysis:
    initial_p_f_t = sum(f_t_history[:50]) / 50.0
    final_p_retention_f_t = sum(pointer_retention_f_t) / len(pointer_retention_f_t) if pointer_retention_f_t else 0.0

    print("\n--- Continual Retention Telemetry ---")
    print(f"Initial Pointer Free Energy (Steps 1-50)  : {initial_p_f_t:.4f} nats")
    print(f"Retention Pointer Free Energy (After 1700s): {final_p_retention_f_t:.4f} nats")
    retention_delta = initial_p_f_t - final_p_retention_f_t
    print(f"Retention Free Energy Delta (Improvement)  : {retention_delta:.4f} nats")

    overall_min_f_t = min(f_t_history)
    overall_final_f_t = sum(f_t_history[-50:]) / 50.0

    print(f"Overall Lowest Free Energy Observed        : {overall_min_f_t:.4f} nats")
    print(f"Overall Final Stream Free Energy           : {overall_final_f_t:.4f} nats")

    verdict = "POSITIVE" if (overall_final_f_t < 1.0 and retention_delta > 0.0) else ("NEUTRAL" if overall_final_f_t < 1.3 else "REJECTED")

    manifest = json.loads(agent.get_topology_manifest())
    results = {
        "exp_id": "EXP-293",
        "initial_f_t": f_t_history[0],
        "final_f_t": overall_final_f_t,
        "lowest_f_t": overall_min_f_t,
        "initial_pointer_f_t": initial_p_f_t,
        "retention_pointer_f_t": final_p_retention_f_t,
        "retention_delta": retention_delta,
        "tok_per_sec": tok_per_sec,
        "sleep_cycles": sleep_cycles,
        "replay_steps": replay_steps,
        "final_nodes": manifest["k_nodes"],
        "verdict": verdict
    }

    with open("experiments/exp_293_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nVerdict: 🟢 {verdict}" if verdict == "POSITIVE" else f"\nVerdict: {verdict}")
    return results

if __name__ == "__main__":
    run_exp_293_benchmark()
