# experiments/exp_266_symbolic_math_recirculation.py
"""
===============================================================================
EXP-266: SYMBOLIC MATHEMATICAL RECIRCULATION BENCHMARK
Evaluating C++20 Dynamic Recurrent Latent Recirculation (System 2 Mental Sandbox)
on exact mathematical calculation and out-of-distribution (OOD) generalization.
===============================================================================
"""
import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
import os
import random

import karyon_core as kcore
from karyon_logger import get_logger

logger = get_logger()

# Define the 3 mathematical operations
def star(a, b):
    return 2 * a + b

def hash_op(a, b):
    return a * b - a

def delta(a, b):
    return a**2 + b

def generate_dataset(a_range, b_range):
    """
    Generates a dataset of mathematical operations formatted as raw byte streams.
    E.g., "star(7,9)=23\n"
    """
    examples = []
    for a in a_range:
        for b in b_range:
            # Operation 1: star
            ans_star = star(a, b)
            examples.append(f"star({a},{b})={ans_star}\n")
            # Operation 2: hash
            ans_hash = hash_op(a, b)
            examples.append(f"hash({a},{b})={ans_hash}\n")
            # Operation 3: delta
            ans_delta = delta(a, b)
            examples.append(f"delta({a},{b})={ans_delta}\n")
    return examples

def tokenize_examples(examples, max_len=32):
    """
    Converts text examples to padded byte tensors.
    """
    tensors = []
    for ex in examples:
        # Convert string to list of ASCII/UTF-8 bytes
        bytes_list = list(ex.encode("utf-8"))
        # Pad or truncate to max_len
        if len(bytes_list) < max_len:
            bytes_list += [256] * (max_len - len(bytes_list))
        else:
            bytes_list = bytes_list[:max_len]
        tensors.append(torch.tensor(bytes_list, dtype=torch.long))
    return torch.stack(tensors)

def evaluate_accuracy(agent, examples, device, recirc_mode="auto"):
    """
    Evaluates the mathematical calculation accuracy of the agent.
    Checks if the generated answer after '=' matches the expected answer.
    """
    agent.eval()
    correct = 0
    total = len(examples)
    
    with torch.no_grad():
        for ex in examples:
            # Extract prompt up to '='
            parts = ex.split("=")
            prompt = parts[0] + "="
            expected_ans = parts[1].strip()
            
            # Tokenize prompt
            prompt_bytes = list(prompt.encode("utf-8"))
            prompt_tensor = torch.tensor([prompt_bytes], dtype=torch.long, device=device)
            
            # Set neurotransmitters based on recirc_mode
            if recirc_mode == "high":
                # High Noradrenaline and Dopamine forces recirculation
                u_t = torch.tensor([[0.5, 1.0, 0.8, 1.0, 0.8, 0.8]], device=device)
            elif recirc_mode == "low":
                # Low Noradrenaline and Dopamine disables recirculation
                u_t = torch.tensor([[0.5, 1.0, 0.8, 1.0, 0.0, 0.0]], device=device)
            else:
                # Auto
                u_t = torch.tensor([[0.5, 1.0, 0.8, 1.0, 0.4, 0.4]], device=device)
            
            # Autoregressive generation
            generated_bytes = prompt_bytes.copy()
            for _ in range(8):  # Max answer length
                input_tensor = torch.tensor([generated_bytes], dtype=torch.long, device=device)
                logits = agent.forward(input_tensor, u_t)
                next_token_logits = logits[0, -1, :]
                next_token = next_token_logits.argmax(-1).item()
                if next_token == ord('\n') or next_token >= 256:
                    break
                generated_bytes.append(next_token)
            
            # Decode generated answer
            generated_str = bytes(generated_bytes[len(prompt_bytes):]).decode("utf-8", errors="ignore").strip()
            if generated_str == expected_ans:
                correct += 1
            else:
                print(f"Mismatch: '{prompt}' -> Got '{generated_str}', Expected '{expected_ans}'")
                
    return correct / total if total > 0 else 0.0

def run_experiment():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🏁 Starting EXP-266 Symbolic Mathematical Recirculation Benchmark on: {device.upper()}")
    
    # 1. Generate Datasets
    # In-Domain: a, b in [1..15]
    indomain_range = list(range(1, 16))
    indomain_raw = generate_dataset(indomain_range, indomain_range)
    random.seed(42)
    random.shuffle(indomain_raw)
    
    # Split into train (80%) and validation (20%)
    split_idx = int(len(indomain_raw) * 0.8)
    train_raw = indomain_raw[:split_idx]
    val_raw = indomain_raw[split_idx:]
    
    # Out-of-Distribution (OOD) Extrapolation: a, b in [16..20]
    ood_range = list(range(16, 21))
    ood_raw = generate_dataset(ood_range, ood_range)
    
    print(f"Dataset summary:")
    print(f"  • Train examples: {len(train_raw)}")
    print(f"  • In-Domain Val examples: {len(val_raw)}")
    print(f"  • OOD Val examples: {len(ood_raw)}")
    
    train_tokens = tokenize_examples(train_raw, max_len=32).to(device)
    
    # 2. Initialize CognitiveEvolvableAgent
    vocab_size = 258
    dim = 256
    agent = kcore.CognitiveEvolvableAgent(vocab_size, dim, 16, device)
    # Sprout initial organelles
    agent.sprout_organelle("sensory_gateway", state_dim=128, num_operators=8)
    agent.sprout_organelle("somatic_integrator", state_dim=128, num_operators=8)
    agent.sprout_organelle("math_coprocessor", state_dim=128, num_operators=8)
    
    # 3. Train the agent
    print("\n--- Training CognitiveEvolvableAgent ---")
    optimizer = optim.AdamW(agent.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=256)
    
    epochs = 100
    batch_size = 32
    
    for epoch in range(1, epochs + 1):
        agent.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # Shuffle train tokens
        indices = torch.randperm(train_tokens.size(0))
        shuffled_tokens = train_tokens[indices]
        
        for i in range(0, shuffled_tokens.size(0), batch_size):
            batch = shuffled_tokens[i:i+batch_size]
            if batch.size(0) < 2:
                continue
            
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            
            optimizer.zero_grad()
            
            # Step method runs the complete Active Inference loop inside C++
            logits, free_energy, u_t = agent.step(inputs, targets, torch.Tensor())
            
            loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
            total_loss = loss + 0.1 * free_energy.mean()
            
            total_loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        if epoch % 10 == 0 or epoch == 1:
            print(f"  [Epoch {epoch:03d}/{epochs}] Loss: {avg_loss:.4f} | Homeo: {u_t.cpu().numpy().round(3)}")
            
    # 4. Comparative Evaluation across Recirculation Modes
    print("\n--- Phase 2: Comparative Evaluation across Recirculation Modes ---")
    
    # Sample subset of validation for quick clean accuracy check
    val_subset = val_raw[:20]
    ood_subset = ood_raw[:20]
    
    print("\nChecking In-Domain Interpolation:")
    print(" >>> Mode: Recirculation DISABLED (Low Neurotransmitters) <<<")
    acc_low_recirc = evaluate_accuracy(agent, val_subset, device, recirc_mode="low")
    print(f"  • In-Domain Accuracy (Low Recirc): {acc_low_recirc * 100:.2f}%")
    
    print("\n >>> Mode: Recirculation ENABLED (High Neurotransmitters) <<<")
    acc_high_recirc = evaluate_accuracy(agent, val_subset, device, recirc_mode="high")
    print(f"  • In-Domain Accuracy (High Recirc): {acc_high_recirc * 100:.2f}%")
    
    print("\nChecking Out-of-Distribution (OOD) Extrapolation:")
    print(" >>> Mode: Recirculation DISABLED (Low Neurotransmitters) <<<")
    acc_low_ood = evaluate_accuracy(agent, ood_subset, device, recirc_mode="low")
    print(f"  • OOD Accuracy (Low Recirc): {acc_low_ood * 100:.2f}%")
    
    print("\n >>> Mode: Recirculation ENABLED (High Neurotransmitters) <<<")
    acc_high_ood = evaluate_accuracy(agent, ood_subset, device, recirc_mode="high")
    print(f"  • OOD Accuracy (High Recirc): {acc_high_ood * 100:.2f}%")
    
    # Save results
    results = {
        "epoch_loss": avg_loss,
        "in_domain_low_recirc_acc": acc_low_recirc,
        "in_domain_high_recirc_acc": acc_high_recirc,
        "ood_low_recirc_acc": acc_low_ood,
        "ood_high_recirc_acc": acc_high_ood,
    }
    with open("experiments/exp_266_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\n✅ EXP-266 Completed Successfully!")

if __name__ == "__main__":
    run_experiment()
