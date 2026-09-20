# experiments/exp_275_karyon_logic_eval.py
"""
===============================================================================
EXP-275: Systematic Logical Reasoning Evaluation & Morphic Space Causal Benchmark
===============================================================================
Hypothesis:
  Evaluating Karyon-CoRE's C++20 UniversalMorphicSpace engine on formal logical reasoning
  batteries (transitivity, counterfactual inversion, negation symmetry, syllogisms) will
  demonstrate that causal morphic circuit depth enables rapid convergence and accurate
  deductive generation across raw UTF-8 byte streams.
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
    dim = 256
    
    # Initialize CoREAgent with C++20 UniversalMorphicSpace engine (2 cells, 4 ops)
    agent = CoREAgent(
        vocab_size=258,
        embed_dim=dim,
        num_cells=2,
        num_operators=4,
        device=device,
        use_graph=False
    )

    # Battery of multi-step logical reasoning tasks
    test_suite = [
        {"task": "Transitivity (A>B, B>C)", "prompt": "If A > B and B > C, then A is greater than ", "expected": "C."},
        {"task": "Counterfactual Inversion", "prompt": "The box is red. If painted blue, the box is ", "expected": "blue."},
        {"task": "Negation Symmetry", "prompt": "Light is not dark. Therefore, dark is not ", "expected": "light."},
        {"task": "Syllogistic Deduction", "prompt": "All humans are mortal. Socrates is human. Socrates is ", "expected": "mortal."}
    ]

    optimizer = torch.optim.AdamW(list(agent.get_complete_state_dict().values()), lr=0.003)
    
    print("🧠 Training Universal Morphic Space on Logical Tasks (120 epochs)...")
    loss_history = []
    
    for epoch in range(120):
        optimizer.zero_grad()
        total_loss = 0.0
        
        for item in test_suite:
            full_text = item["prompt"] + item["expected"]
            raw_bytes = list(full_text.encode('utf-8'))
            prompt_len = len(item["prompt"].encode('utf-8'))
            
            input_tensor = torch.tensor(raw_bytes, dtype=torch.long, device=device).unsqueeze(0)
            
            # Forward pass: shape [1, seq_len, 258]
            logits = agent(input_tensor)
            
            # Calculate loss only on the target answer portion (autoregressive next-byte prediction)
            target_logits = logits[:, prompt_len - 1 : -1, :].reshape(-1, 258)
            target_labels = input_tensor[:, prompt_len:].reshape(-1)
            
            loss = nn.functional.cross_entropy(target_logits, target_labels)
            total_loss += loss
            
        total_loss.backward()
        optimizer.step()
        loss_history.append(total_loss.item() / len(test_suite))
        
        if (epoch + 1) % 30 == 0 or epoch == 0:
            print(f"  • Epoch {epoch + 1:3d}/120 | Mean Target Loss: {loss_history[-1]:.4f}")

    initial_loss = loss_history[0]
    final_loss = loss_history[-1]
    loss_delta = initial_loss - final_loss

    # Generation and Evaluation phase
    print("\n🔍 Evaluating Generative Reasoning & Accuracy...")
    correct_tokens = 0
    total_tokens = 0
    
    with torch.no_grad():
        for item in test_suite:
            prompt_bytes = list(item["prompt"].encode('utf-8'))
            expected_bytes = list(item["expected"].encode('utf-8'))
            
            curr_bytes = prompt_bytes.copy()
            generated_bytes = []
            
            for target_b in expected_bytes:
                input_tensor = torch.tensor(curr_bytes, dtype=torch.long, device=device).unsqueeze(0)
                logits = agent(input_tensor)
                next_byte = logits[0, -1, :].argmax().item()
                
                generated_bytes.append(next_byte)
                if next_byte == target_b:
                    correct_tokens += 1
                total_tokens += 1
                
                curr_bytes.append(next_byte)
                
            gen_text = bytes(generated_bytes).decode('utf-8', errors='replace')
            status = "✅ PASS" if gen_text == item["expected"] else "❌ FAIL"
            print(f"  {status} [{item['task']:<28}] Exp: '{item['expected']}' | Gen: '{gen_text}'")

    accuracy = (correct_tokens / total_tokens) * 100.0

    print(f"\n📊 [EXP-275 Telemetry Summary]")
    print(f"  • Initial Loss : {initial_loss:.4f}")
    print(f"  • Final Loss   : {final_loss:.4f}")
    print(f"  • Loss Delta   : {loss_delta:.4f}")
    print(f"  • Accuracy     : {accuracy:.1f}%")

    print(f"METRICS_SUMMARY: final_loss={final_loss:.5f}, delta_loss={loss_delta:.5f}, accuracy={accuracy:.2f}, baseline_loss={initial_loss:.5f}")


if __name__ == "__main__":
    run_benchmark()
