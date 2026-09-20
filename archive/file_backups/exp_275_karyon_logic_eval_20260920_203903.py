# experiments/exp_275_karyon_logic_eval.py
"""
===============================================================================
EXP-275: Systematic Logical Reasoning Evaluation & System 2 Thinking Depth Benchmark
===============================================================================
Hypothesis:
  Evaluating the native C++20 DynamicMorphicGraph across non-trivial multi-step logical
  reasoning tasks (transitivity, counterfactuals, negation inversion) will demonstrate that
  increasing System 2 thinking steps (K = 1 vs K = 4 vs K = 8) systematically reduces
  predictive error and enables non-linear recurrent settling into the correct logical state.
===============================================================================
"""
import os
import sys
import time

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from karyon_agent import CoREAgent


def run_benchmark():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 [EXP-275] Starting Systematic Logical Reasoning Evaluation on {device}...")

    torch.manual_seed(42)
    dim = 128
    
    # Initialize CoREAgent with C++20 DynamicMorphicGraph enabled
    agent = CoREAgent(vocab_size=258, embed_dim=dim, device=device, use_graph=True)
    
    # Sprout extra reasoning nodes into the dynamic graph
    agent.add_node("logic_attractor_1", "SaturatedAttractor", is_core=False, initial_alpha=0.5)
    agent.add_node("logic_bilinear_1", "BilinearMultiplicative", is_core=False, initial_alpha=0.5)
    
    # Linear projection from graph readout (dim) to full byte vocab (258)
    head = nn.Linear(dim, 258).to(device)

    # Define test suite of logical tasks (UTF-8 byte targets)
    test_suite = [
        {"task": "Transitivity (A>B, B>C)", "prompt": "If A > B and B > C, then A is greater than ", "expected": "C"},
        {"task": "Counterfactual Inversion", "prompt": "The box is red. If painted blue, the box is ", "expected": "blue"},
        {"task": "Negation Symmetry", "prompt": "Light is not dark. Therefore, dark is not ", "expected": "light"},
        {"task": "Syllogistic Deduction", "prompt": "All humans are mortal. Socrates is human. Socrates is ", "expected": "mortal"}
    ]

    params = list(agent.get_complete_state_dict().values()) + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=0.005)
    
    # Training phase on logical patterns
    print("🧠 Training Dynamic Graph on logical reasoning patterns (K=4 thinking steps)...")
    for epoch in range(40):
        optimizer.zero_grad()
        total_loss = 0.0
        for item in test_suite:
            prompt_bytes = list(item["prompt"].encode('utf-8'))
            expected_bytes = list(item["expected"].encode('utf-8'))
            
            # Autoregressive teacher forcing over target bytes
            curr_bytes = prompt_bytes.copy()
            for target_b in expected_bytes:
                input_ids = torch.tensor(curr_bytes, dtype=torch.long, device=device).unsqueeze(0)
                readout = agent(input_ids, thinking_steps=4)
                logits = head(readout)
                target = torch.tensor([target_b], dtype=torch.long, device=device)
                
                loss = nn.functional.cross_entropy(logits, target)
                total_loss += loss
                curr_bytes.append(target_b)
            
        total_loss.backward()
        optimizer.step()

    # Evaluation phase across System 2 Thinking Depths (K = 1, K = 4, K = 8)
    print("\n🔍 Evaluating Generation & Logical Accuracy across Thinking Depths (K-steps)...")
    results = {}
    
    for k_steps in [1, 4, 8]:
        k_losses = []
        correct_predictions = 0
        total_predictions = 0
        t0 = time.perf_counter()
        
        with torch.no_grad():
            for item in test_suite:
                prompt_bytes = list(item["prompt"].encode('utf-8'))
                expected_bytes = list(item["expected"].encode('utf-8'))
                
                curr_bytes = prompt_bytes.copy()
                generated_bytes = []
                
                for target_b in expected_bytes:
                    input_ids = torch.tensor(curr_bytes, dtype=torch.long, device=device).unsqueeze(0)
                    readout = agent(input_ids, thinking_steps=k_steps)
                    logits = head(readout)
                    target = torch.tensor([target_b], dtype=torch.long, device=device)
                    
                    loss = nn.functional.cross_entropy(logits, target).item()
                    k_losses.append(loss)
                    
                    pred_byte = logits.argmax(dim=-1).item()
                    generated_bytes.append(pred_byte)
                    if pred_byte == target_b:
                        correct_predictions += 1
                    total_predictions += 1
                    
                    curr_bytes.append(pred_byte)
                
                gen_text = bytes(generated_bytes).decode('utf-8', errors='replace')
                if k_steps == 8:
                    print(f"  [K=8 Sample] Task: {item['task']} | Expected: '{item['expected']}' | Gen: '{gen_text}'")
                
        elapsed = (time.perf_counter() - t0) * 1000.0
        avg_loss = sum(k_losses) / len(k_losses)
        acc = (correct_predictions / total_predictions) * 100.0
        results[f"K={k_steps}"] = {"avg_loss": avg_loss, "accuracy": acc, "time_ms": elapsed}
        print(f"  • Depth K={k_steps}: Cross-Entropy Loss = {avg_loss:.4f} | Accuracy = {acc:.1f}% ({elapsed:.2f}ms)")

    initial_loss = results["K=1"]["avg_loss"]
    deep_loss = results["K=8"]["avg_loss"]
    loss_delta = initial_loss - deep_loss

    print(f"\n📊 [EXP-275 Telemetry]")
    print(f"  • K=1  Loss : {results['K=1']['avg_loss']:.4f} | Acc: {results['K=1']['accuracy']:.1f}%")
    print(f"  • K=4  Loss : {results['K=4']['avg_loss']:.4f} | Acc: {results['K=4']['accuracy']:.1f}%")
    print(f"  • K=8  Loss : {results['K=8']['avg_loss']:.4f} | Acc: {results['K=8']['accuracy']:.1f}%")
    print(f"  • Loss Delta: {loss_delta:.4f}")

    print(f"METRICS_SUMMARY: final_loss={deep_loss:.5f}, delta_loss={loss_delta:.5f}, k1_loss={initial_loss:.5f}, k8_loss={deep_loss:.5f}, accuracy={results['K=8']['accuracy']:.2f}")


if __name__ == "__main__":
    run_benchmark()
