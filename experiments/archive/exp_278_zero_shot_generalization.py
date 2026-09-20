# experiments/exp_278_zero_shot_generalization.py
"""
===============================================================================
EXP-278: Zero-Shot Logical Generalization over Learned Premise Substrates
===============================================================================
Hypothesis:
  If Karyon-CoRE is trained strictly on raw factual premises (unstructured world data)
  without ever seeing question-answer templates or deduction queries, the C++20 Morphic
  Space combined with Continuous Hopfield Attractors (N=128 basins) will perform
  zero-shot logical composition in its latent space. At test time, frozen forward pass
  dynamics will snap combined premise representations into correct unseen logical conclusions.
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
    print(f"🚀 [EXP-278] Starting Zero-Shot Logical Generalization Benchmark on {device}...")

    torch.manual_seed(42)
    dim = 256
    agent = MorphicAgentWithHopfield(vocab_size=258, embed_dim=dim, num_basins=128, device=device)

    # 1. TRAINING DATA: Raw premises (World facts only, no questions, no deduction templates)
    premises = [
        "A is greater than B. B is greater than C. C is greater than D.",
        "All humans are mortal. Socrates is human.",
        "The box is red. If you paint a red box blue, it becomes blue.",
        "The library is north of the park. The park is south of the library.",
        "Water freezes into ice. Ice melts into water."
    ]

    # 2. TEST DATA: Completely unseen logical queries requiring zero-shot deduction over premises
    test_queries = [
        {
            "task": "Transitivity (A > C)",
            "prompt": "Since A is greater than B and B is greater than C, then A is greater than ",
            "expected": "C."
        },
        {
            "task": "Zero-Shot Syllogism",
            "prompt": "Socrates is human. Therefore, Socrates is ",
            "expected": "mortal."
        },
        {
            "task": "Zero-Shot Color Inversion",
            "prompt": "We have a red box. If we paint it blue, the box is ",
            "expected": "blue."
        },
        {
            "task": "Zero-Shot Spatial Relation",
            "prompt": "The park is south of the library. Therefore, the library is north of ",
            "expected": "the park."
        },
        {
            "task": "Zero-Shot Phase Change",
            "prompt": "When ice melts, it turns back into ",
            "expected": "water."
        }
    ]

    optimizer = torch.optim.AdamW(list(agent.get_complete_state_dict().values()), lr=0.004)
    
    epochs = 400
    print(f"🧠 Training Morphic Space strictly on raw premises ({epochs} epochs)...")
    print("⚠️  (Deduction queries are completely hidden from training!)")
    
    loss_history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        total_loss = 0.0
        
        for text in premises:
            raw_bytes = list(text.encode('utf-8'))
            input_tensor = torch.tensor(raw_bytes, dtype=torch.long, device=device).unsqueeze(0)
            logits = agent(input_tensor)
            
            # Predict next byte across the entire premise string (standard language modeling)
            target_logits = logits[:, :-1, :].reshape(-1, 258)
            target_labels = input_tensor[:, 1:].reshape(-1)
            
            loss = nn.functional.cross_entropy(target_logits, target_labels)
            total_loss += loss
            
        total_loss.backward()
        optimizer.step()
        loss_history.append(total_loss.item() / len(premises))
        
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"  • Epoch {epoch + 1:3d}/{epochs} | Premise Reconstruction Loss: {loss_history[-1]:.4f}")

    initial_loss = loss_history[0]
    final_loss = loss_history[-1]
    loss_delta = initial_loss - final_loss

    # 3. ZERO-SHOT EVALUATION PHASE (Frozen weights)
    print("\n❄️  Freezing weights. Initiating Zero-Shot Logical Generalization Evaluation...")
    agent.eval()
    
    correct_tasks = 0
    total_tasks = len(test_queries)
    correct_tokens = 0
    total_tokens = 0
    
    with torch.no_grad():
        for item in test_queries:
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
            print(f"  {status} [{item['task']:<28}] Exp: '{item['expected']}' | Gen: '{gen_text}'")

    token_accuracy = (correct_tokens / total_tokens) * 100.0
    task_accuracy = (correct_tasks / total_tasks) * 100.0

    print(f"\n📊 [EXP-278 Telemetry Summary]")
    print(f"  • Base Facts Loss: {final_loss:.4f}")
    print(f"  • Token Accuracy : {token_accuracy:.1f}%")
    print(f"  • Task Accuracy  : {task_accuracy:.1f}% ({correct_tasks}/{total_tasks} Tasks Passed)")

    print(f"METRICS_SUMMARY: final_loss={final_loss:.5f}, delta_loss={loss_delta:.5f}, accuracy={token_accuracy:.2f}, task_accuracy={task_accuracy:.2f}, baseline_loss={initial_loss:.5f}")


if __name__ == "__main__":
    run_benchmark()
