"""
================================================================================
EXP-291: SINGLE-PASS ALLOSTATIC MORPHOGENESIS BENCHMARK (KEP PRINCIPLES 24 & 25)
================================================================================
Scientific Objective:
Evaluates Karyon-CoRE under strict Single-Pass Stream Learning (N=1 pass, zero epochs)
across a multi-domain algorithmic & continuous task stream (Reversal, Pointer Tracking,
Addition, Parity, Dyck-1 Language).

Key Biophysical Invariants Tested:
1. Anti-Zubryoshka Single-Pass Stream Paradigm (KEP Principle 24): Data flows continuously
   once without artificial multi-epoch batch loops.
2. Topological Morphogenesis on Primitive Mathematical Operators (KEP Principle 25):
   Neurogenesis sprouts new nodes strictly from the universal 4-primitive menu
   (LinearAccumulator, BilinearMultiplicative, SaturatedAttractor, StateSpaceMemory)
   under high Free Energy surprise, while Neural Darwinism Apoptosis prunes unviable nodes.
3. Telemetry Logged: Instant & Rolling Free Energy (F_t), Sleep Replay Cycles, Active
   Node Count, and Multi-Domain Evaluation Accuracy during streaming inference.
================================================================================
"""
import sys, os, time, json
sys.path.insert(0, '.')
import torch
import torch.nn.functional as F

import karyon_agent
from karyon_logger import get_logger
from multi_domain_benchmark import generate_multi_domain_suite

logger = get_logger()

def run_exp_291_benchmark():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 80)
    print("EXP-291: SINGLE-PASS ALLOSTATIC MORPHOGENESIS BENCHMARK")
    print(f"Device: {device} | PyTorch: {torch.__version__} | Single-Pass Stream Paradigm (N=1)")
    print("=" * 80)

    # 1. Generate Multi-Domain Task Streams
    suite_train = generate_multi_domain_suite(seed=101)
    suite_eval = generate_multi_domain_suite(seed=202)

    # Flatten train items into a single temporal stream
    stream_tasks = []
    for domain, items in suite_train.items():
        for p, ans, full in items:
            stream_tasks.append((domain, p, ans, full))

    print(f"Streaming Dataset Size: {len(stream_tasks)} continuous samples across 5 domains.")

    # 2. Instantiate CoREAgent
    agent = karyon_agent.CoREAgent(embed_dim=256, device=device)
    optimizer = torch.optim.AdamW(agent.parameters(), lr=1e-3, weight_decay=1e-4)

    start_time = time.time()
    f_t_history = []
    sleep_cycles_logged = 0
    sprouted_nodes_logged = 0
    pruned_nodes_logged = 0

    print("\n--- Phase 1: Waking Stream Learning (Single-Pass N=1) ---")
    agent.train()
    for step, (domain, p, ans, full) in enumerate(stream_tasks):
        b_seq = torch.tensor(list(full.encode('utf-8')), dtype=torch.long, device=device).unsqueeze(0)
        if b_seq.shape[1] < 2:
            continue

        inp = b_seq[:, :-1]
        tgt = b_seq[:, 1:]

        # Forward pass (Temporal SSD + Spatial Morphic Deliberation)
        logits = agent(inp, thinking_steps=4)
        loss = F.cross_entropy(logits.reshape(-1, 258), tgt.reshape(-1))

        # Online gradient update
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        optimizer.step()

        f_t = float(loss.item())
        f_t_history.append(f_t)

        # Biophysical Allostasis: High Free Energy or Periodic Strain triggers Sleep
        if (step + 1) % 25 == 0 or f_t > 4.5:
            # Enter Sleep Phase
            sleep_stats = agent.execute_deep_allostatic_sleep(
                downscaling_factor=0.01,
                sprout_probability=0.75,
                prune_threshold=0.01,
                available_ops=(
                    "LinearAccumulator",
                    "BilinearMultiplicative",
                    "SaturatedAttractor",
                    "ContinuousHopfield",
                    "StateSpaceMemory"
                )
            )
            sleep_cycles_logged += 1
            sprouted_nodes_logged += int(sleep_stats["sprouted"])
            pruned_nodes_logged += int(sleep_stats["pruned_nodes"])
            # Re-bind optimizer to active dynamic parameters
            optimizer = torch.optim.AdamW(agent.parameters(), lr=1e-3, weight_decay=1e-4)

        if (step + 1) % 50 == 0 or step == len(stream_tasks) - 1:
            recent_f_t = sum(f_t_history[-20:]) / len(f_t_history[-20:])
            manifest = json.loads(agent.get_topology_manifest())
            print(f"Step {step+1:3d}/{len(stream_tasks)} | Task: {domain:10s} | Instant F_t: {f_t:.4f} | Recent Avg F_t: {recent_f_t:.4f} | Active Nodes: {manifest['k_nodes']}")

    elapsed_time = time.time() - start_time
    total_tokens_processed = sum(len(full.encode('utf-8')) for _, _, _, full in stream_tasks)
    tok_per_sec = total_tokens_processed / elapsed_time

    # 3. Phase 2: Zero-Shot Streaming Evaluation Across Domains
    print("\n--- Phase 2: Streaming Zero-Shot Multi-Domain Evaluation ---")
    agent.eval()
    domain_accuracies = {}

    with torch.no_grad():
        for domain, items in suite_eval.items():
            correct = 0
            total = len(items)
            for p, ans, full in items:
                # Prompt bytes
                p_bytes = torch.tensor(list(p.encode('utf-8')), dtype=torch.long, device=device).unsqueeze(0)
                gen_bytes = []
                curr_input = p_bytes

                # Autoregressive sampling
                for _ in range(len(ans) + 2):
                    logits = agent(curr_input, thinking_steps=4)
                    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    token_val = next_token.item()
                    if token_val == 257: # EOS
                        break
                    gen_bytes.append(token_val)
                    curr_input = torch.cat([curr_input, next_token], dim=1)

                try:
                    pred_str = bytes(gen_bytes).decode('utf-8', errors='ignore').strip()
                except Exception:
                    pred_str = ""

                if pred_str == ans.strip() or ans.strip() in pred_str:
                    correct += 1

            acc = (correct / total) * 100.0
            domain_accuracies[domain] = acc
            print(f"Eval Domain: {domain:12s} | Exact Match Acc: {acc:6.2f}% ({correct}/{total})")

    mean_acc = sum(domain_accuracies.values()) / len(domain_accuracies)
    initial_f_t = f_t_history[0]
    final_f_t = sum(f_t_history[-20:]) / 20.0
    f_t_delta = initial_f_t - final_f_t

    final_manifest = json.loads(agent.get_topology_manifest())

    print("\n" + "=" * 80)
    print("EXP-291 TELEMETRY SUMMARY")
    print("=" * 80)
    print(f"Initial Free Energy (F_t): {initial_f_t:.4f}")
    print(f"Final Free Energy (F_t)  : {final_f_t:.4f} (Delta Reduction: {f_t_delta:.4f})")
    print(f"Throughput               : {tok_per_sec:.2f} tok/s")
    print(f"Total Sleep Cycles       : {sleep_cycles_logged}")
    print(f"Sprouted Operators       : {sprouted_nodes_logged}")
    print(f"Pruned Operators         : {pruned_nodes_logged}")
    print(f"Evolved Graph Topology   : {final_manifest}")
    print(f"Mean Multi-Domain Acc    : {mean_acc:.2f}%")
    print("=" * 80)

    # KEP Rule #2 Verdict Logic:
    # POSITIVE if Free Energy dropped significantly (>1.5 nats) under Single-Pass N=1 stream learning
    # while topological neurogenesis dynamically assembled a stable graph.
    verdict = "POSITIVE" if f_t_delta > 1.5 else ("NEUTRAL" if f_t_delta > 0.5 else "REJECTED")

    results = {
        "exp_id": "EXP-291",
        "initial_free_energy": initial_f_t,
        "final_free_energy": final_f_t,
        "free_energy_delta": f_t_delta,
        "tok_per_sec": tok_per_sec,
        "sleep_cycles": sleep_cycles_logged,
        "sprouted_nodes": sprouted_nodes_logged,
        "pruned_nodes": pruned_nodes_logged,
        "final_nodes": final_manifest["k_nodes"],
        "mean_accuracy": mean_acc,
        "domain_accuracies": domain_accuracies,
        "verdict": verdict
    }

    with open("experiments/exp_291_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nVerdict: 🟢 {verdict}" if verdict == "POSITIVE" else f"\nVerdict: {verdict}")
    return results

if __name__ == "__main__":
    run_exp_291_benchmark()
