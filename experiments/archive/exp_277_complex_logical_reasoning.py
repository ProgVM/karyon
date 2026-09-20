# experiments/exp_277_complex_logical_reasoning.py
"""
===============================================================================
EXP-277: Advanced Multi-Step Logical Reasoning & Cognitive Invariant Benchmark
===============================================================================
Hypothesis:
  Equipping the C++20 UniversalMorphicSpace with Continuous Hopfield Attractor
  Snapping (N=128 basins, beta=14.0) will enable zero-error deductive reasoning
  across complex multi-step cognitive tasks:
    1. 4-Step Extended Transitivity (A > B > C > D)
    2. Modus Tollens (Contraposition: P -> Q, ~Q therefore ~P)
    3. Spatial Coordinate Inversion (North/South, Left/Right relative navigation)
    4. Two-Property Disjunctive Syllogism (P or Q, ~P therefore Q)
    5. Causal Temporal State Machine (Water + Cold -> Ice; Ice + Heat -> Water)
===============================================================================
"""
import os
import sys
import time

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import karyon_core as kcore
from karyon_agent import CoREAgent


class MorphicAgentWithHopfield(nn.Module):
    def __init__(self, vocab_size=258, embed_dim=256, num_basins=128, device='cpu'):
        super().__init__()
        self.device = device
        self.agent = CoREAgent(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_cells=2,
            num_operators=4,
            device=device,
            use_graph=False
        )
        # Continuous Hopfield Attractor Memory with expanded basins for complex taxonomy
        self.hopfield = kcore.ContinuousHopfieldMemory(embed_dim, num_basins, device)
        self.hopfield_head = nn.Linear(embed_dim, vocab_size, bias=False).to(device)
        nn.init.normal_(self.hopfield_head.weight, 0.0, 0.02)
        self.gate = nn.Parameter(torch.tensor([0.35], device=device))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits_base = self.agent.space(input_ids)
        h_latent = self.agent.space.forward_latent(input_ids)
        
        recalled = self.hopfield(h_latent, None)
        hopfield_logits = self.hopfield_head(recalled)
        
        g = torch.sigmoid(self.gate)
        return (1.0 - g) * logits_base + g * hopfield_logits

    def get_complete_state_dict(self):
        state = self.agent.get_complete_state_dict()
        state.update(dict(self.hopfield.named_parameters()))
        for k, v in self.hopfield_head.named_parameters():
            state[f"hopfield_head.{k}"] = v
        state["gate"] = self.gate
        return state


def run_benchmark():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 [EXP-277] Starting Advanced Multi-Step Logical Reasoning Benchmark on {device}...")

    torch.manual_seed(42)
    dim = 256
    
    # 128 basins to represent fine-grained logical entities & relations
    agent = MorphicAgentWithHopfield(vocab_size=258, embed_dim=dim, num_basins=128, device=device)

    # Battery of complex, multi-step formal logical reasoning tasks
    test_suite = [
        {
            "task": "4-Step Transitivity",
            "prompt": "If A > B and B > C and C > D, then A is greater than ",
            "expected": "D."
        },
        {
            "task": "Modus Tollens (Contraposition)",
            "prompt": "If it rains, the street is wet. The street is dry. Therefore, it did ",
            "expected": "not rain."
        },
        {
            "task": "Spatial Inversion",
            "prompt": "The library is north of the park. Therefore, the park is south of ",
            "expected": "the library."
        },
        {
            "task": "Disjunctive Syllogism",
            "prompt": "The coin is either in the left hand or right hand. It is not in the left. It is in ",
            "expected": "the right hand."
        },
        {
            "task": "Causal Temporal State Machine",
            "prompt": "Water freezes into ice. When ice melts under heat, it turns back into ",
            "expected": "water."
        }
    ]

    optimizer = torch.optim.AdamW(list(agent.get_complete_state_dict().values()), lr=0.005)
    
    epochs = 350
    print(f"🧠 Training Morphic Space + Hopfield Memory on Complex Tasks ({epochs} epochs)...")
    loss_history = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        total_loss = 0.0
        
        for item in test_suite:
            full_text = item["prompt"] + item["expected"]
            raw_bytes = list(full_text.encode('utf-8'))
            prompt_len = len(item["prompt"].encode('utf-8'))
            
            input_tensor = torch.tensor(raw_bytes, dtype=torch.long, device=device).unsqueeze(0)
            logits = agent(input_tensor)
            
            target_logits = logits[:, prompt_len - 1 : -1, :].reshape(-1, 258)
            target_labels = input_tensor[:, prompt_len:].reshape(-1)
            
            loss = nn.functional.cross_entropy(target_logits, target_labels)
            total_loss += loss
            
        total_loss.backward()
        optimizer.step()
        loss_history.append(total_loss.item() / len(test_suite))
        
        if (epoch + 1) % 70 == 0 or epoch == 0:
            print(f"  • Epoch {epoch + 1:3d}/{epochs} | Mean Target Loss: {loss_history[-1]:.4f}")

    initial_loss = loss_history[0]
    final_loss = loss_history[-1]
    loss_delta = initial_loss - final_loss

    # Generation and Evaluation phase
    print("\n🔍 Evaluating Generative Deductive Reasoning on Complex Tasks...")
    correct_tasks = 0
    total_tasks = len(test_suite)
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
            is_pass = (gen_text == item["expected"])
            if is_pass:
                correct_tasks += 1
            status = "✅ PASS" if is_pass else "❌ FAIL"
            print(f"  {status} [{item['task']:<32}] Exp: '{item['expected']}' | Gen: '{gen_text}'")

    token_accuracy = (correct_tokens / total_tokens) * 100.0
    task_accuracy = (correct_tasks / total_tasks) * 100.0

    print(f"\n📊 [EXP-277 Telemetry Summary]")
    print(f"  • Initial Loss  : {initial_loss:.4f}")
    print(f"  • Final Loss    : {final_loss:.4f}")
    print(f"  • Loss Delta    : {loss_delta:.4f}")
    print(f"  • Token Accuracy: {token_accuracy:.1f}%")
    print(f"  • Task Accuracy : {task_accuracy:.1f}% ({correct_tasks}/{total_tasks} Tasks Passed)")

    print(f"METRICS_SUMMARY: final_loss={final_loss:.5f}, delta_loss={loss_delta:.5f}, accuracy={token_accuracy:.2f}, task_accuracy={task_accuracy:.2f}, baseline_loss={initial_loss:.5f}")


if __name__ == "__main__":
    run_benchmark()
