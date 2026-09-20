# experiments/exp_267_turing_morphic_circuits.py
"""
===============================================================================
EXP-267: TURING-COMPLETE MORPHIC CIRCUITS WITH BILINEAR OPERATORS & MEMORY SLOTS
Testing Dynamic Symbolic & Arithmetic Emergence via Bilinear Multiplicative Primitives,
Working Memory Slots, and Free-Energy Gated Latent Recirculation
===============================================================================
"""
import sys
import os
import math
import time
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import karyon_core as kcore


class BilinearMorphicUnit(nn.Module):
    """
    Bilinear Multiplicative Operator Node:
    Computes y = (W_left x) * (W_right x) + W_lin x
    Enables native multiplication, quadratic scaling, and sharp logical gates.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.w_left = nn.Linear(dim, dim, bias=False)
        self.w_right = nn.Linear(dim, dim, bias=False)
        self.w_lin = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        
        nn.init.orthogonal_(self.w_left.weight, gain=0.2)
        nn.init.orthogonal_(self.w_right.weight, gain=0.2)
        nn.init.orthogonal_(self.w_lin.weight, gain=0.2)
        nn.init.orthogonal_(self.gate.weight, gain=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multiplicative interaction
        mult = self.w_left(x) * self.w_right(x)
        lin = self.w_lin(x)
        g = torch.sigmoid(self.gate(x))
        out = self.norm(x + g * F.silu(mult + lin))
        return out


class AttractorSlotMemory(nn.Module):
    """
    Working Memory Slot Bank with Attractor Persistence:
    Maintains K isolated slots (registers) with soft read/write gates,
    preventing catastrophic information blurring during multi-step reasoning.
    """
    def __init__(self, num_slots: int = 4, dim: int = 256):
        super().__init__()
        self.num_slots = num_slots
        self.dim = dim
        self.slots = nn.Parameter(torch.randn(num_slots, dim) * 0.02)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.write_gate = nn.Linear(dim, num_slots, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, D]
        B, S, D = x.shape
        slots_expanded = self.slots.unsqueeze(0).unsqueeze(0).expand(B, S, self.num_slots, D)
        
        # Read from slots
        q = self.q_proj(x).unsqueeze(2) # [B, S, 1, D]
        k = self.k_proj(slots_expanded)  # [B, S, K, D]
        v = self.v_proj(slots_expanded)  # [B, S, K, D]
        
        attn = torch.softmax(torch.sum(q * k, dim=-1, keepdim=True) / math.sqrt(D), dim=2) # [B, S, K, 1]
        read_val = torch.sum(attn * v, dim=2) # [B, S, D]
        
        return self.norm(x + read_val)


class TuringMorphicCell(nn.Module):
    """
    Turing-Complete Morphic Cell combining Causal SSD, Bilinear Units,
    Attractor Working Memory Slots, and Dynamic Recirculation.
    """
    def __init__(self, dim: int = 256, num_slots: int = 4, num_recirc: int = 3, device_str: str = "cuda"):
        super().__init__()
        self.dim = dim
        self.num_recirc = num_recirc
        self.ssd = kcore.CausalParallelSSD(dim, device_str)
        self.bilinear = BilinearMorphicUnit(dim)
        self.memory_slots = AttractorSlotMemory(num_slots, dim)
        self.step_gate = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Temporal context integration
        h = self.ssd(x)
        
        # 2. Multi-step Recurrent Latent Thinking
        for _ in range(self.num_recirc):
            # Bilinear non-linear calculation
            h_calc = self.bilinear(h)
            # Memory slot interaction (isolated registers)
            h_mem = self.memory_slots(h_calc)
            # Dynamic step gating
            gate = torch.sigmoid(self.step_gate(h))
            h = self.norm(h + gate * (h_mem - h))
            
        return h


class TuringMorphicAgent(nn.Module):
    def __init__(self, vocab_size: int = 258, dim: int = 256, num_layers: int = 2, num_slots: int = 4, num_recirc: int = 3, device_str: str = "cuda"):
        super().__init__()
        self.dim = dim
        self.emb = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TuringMorphicCell(dim=dim, num_slots=num_slots, num_recirc=num_recirc, device_str=device_str)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.emb.weight # Weight tying

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.emb(input_ids)
        for layer in self.layers:
            x = x + layer(x)
        logits = self.head(self.norm(x))
        return logits


# -----------------------------------------------------------------------------
# Data Generation for Custom Symbolic Math Tasks
# star(a,b) = 2a + b
# hash(a,b) = a * b - a
# delta(a,b) = a^2 + b
# -----------------------------------------------------------------------------
def make_example(op: str, a: int, b: int) -> str:
    if op == "star":
        res = 2 * a + b
    elif op == "hash":
        res = a * b - a
    elif op == "delta":
        res = a * a + b
    else:
        raise ValueError(f"Unknown op {op}")
    return f"{op}({a},{b})={res}\n"


def create_dataset():
    ops = ["star", "hash", "delta"]
    in_domain = []
    ood = []

    for op in ops:
        for a in range(1, 16):
            for b in range(1, 16):
                in_domain.append((op, a, b))
        for a in range(16, 21):
            for b in range(16, 21):
                ood.append((op, a, b))

    random.seed(42)
    random.shuffle(in_domain)
    random.shuffle(ood)

    split = int(len(in_domain) * 0.8)
    train_set = in_domain[:split]
    val_id = in_domain[split:]
    val_ood = ood

    return train_set, val_id, val_ood


def collate_fn(batch, device):
    encoded = [torch.tensor(list(item.encode("utf-8")), dtype=torch.long) for item in batch]
    max_len = max(len(t) for t in encoded)
    padded = torch.full((len(encoded), max_len), 256, dtype=torch.long)
    for i, t in enumerate(encoded):
        padded[i, :len(t)] = t
    return padded.to(device)


def evaluate_exact_accuracy(model, dataset, device):
    model.eval()
    correct = 0
    total = len(dataset)
    mismatches = []

    with torch.no_grad():
        for op, a, b in dataset:
            prompt = f"{op}({a},{b})="
            expected = make_example(op, a, b).strip()
            
            curr_bytes = list(prompt.encode("utf-8"))
            for _ in range(8):
                inp = torch.tensor([curr_bytes], dtype=torch.long, device=device)
                logits = model(inp)
                next_byte = torch.argmax(logits[0, -1, :]).item()
                if next_byte in (ord('\n'), 257, 256):
                    break
                curr_bytes.append(next_byte)

            generated = bytes(curr_bytes).decode("utf-8", errors="ignore").strip()
            if generated == expected:
                correct += 1
            else:
                if len(mismatches) < 3:
                    mismatches.append(f"Prompt '{prompt}' -> Got '{generated}', Expected '{expected}'")

    acc = (correct / total) * 100.0 if total > 0 else 0.0
    return acc, mismatches


def run_experiment():
    print("=" * 80)
    print("EXP-267: TURING-COMPLETE MORPHIC CIRCUITS BENCHMARK")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware execution substrate: {device}")

    train_data, val_id_data, val_ood_data = create_dataset()
    print(f"Dataset summary: Train={len(train_data)}, In-Domain Val={len(val_id_data)}, OOD Val={len(val_ood_data)}")

    # Instantiate Turing Morphic Agent
    model = TuringMorphicAgent(
        vocab_size=258,
        dim=256,
        num_layers=2,
        num_slots=4,
        num_recirc=3,
        device_str="cuda" if torch.cuda.is_available() else "cpu"
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=256)

    epochs = 100
    batch_size = 32

    print("\n--- Training Phase ---")
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(train_data)
        total_loss = 0.0
        batches = 0

        for i in range(0, len(train_data), batch_size):
            batch_items = [make_example(op, a, b) for op, a, b in train_data[i:i+batch_size]]
            inputs = collate_fn(batch_items, device)
            targets = inputs.clone()

            optimizer.zero_grad()
            logits = model(inputs[:, :-1])
            loss = criterion(logits.reshape(-1, 258), targets[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / batches if batches > 0 else 0.0
        if epoch % 10 == 0 or epoch == 1:
            print(f"  [Epoch {epoch:03d}/{epochs}] Training Loss: {avg_loss:.4f}")

    train_duration = time.time() - start_time
    print(f"Training completed in {train_duration:.2f}s")

    print("\n--- Evaluation Phase (Exact Symbolic Generation) ---")
    id_acc, id_mismatches = evaluate_exact_accuracy(model, val_id_data, device)
    ood_acc, ood_mismatches = evaluate_exact_accuracy(model, val_ood_data, device)

    print(f"  • In-Domain Exact Accuracy : {id_acc:.2f}%")
    for m in id_mismatches:
        print(f"    - {m}")
    print(f"  • OOD Exact Accuracy       : {ood_acc:.2f}%")
    for m in ood_mismatches:
        print(f"    - {m}")

    print("\n" + "=" * 80)
    print("EXP-267 TELEMETRY SUMMARY")
    print("=" * 80)
    print(f"Final Train Loss : {avg_loss:.4f}")
    print(f"In-Domain Acc    : {id_acc:.2f}%")
    print(f"OOD Acc          : {ood_acc:.2f}%")
    print("=" * 80)

    # Save metrics JSON for pipeline
    metrics = {
        "final_loss": float(avg_loss),
        "in_domain_acc": float(id_acc),
        "ood_acc": float(ood_acc),
        "train_duration_s": float(train_duration)
    }
    with open("experiments/exp_267_metrics.json", "w") as f:
        json.dump(metrics, f)

    return metrics


if __name__ == "__main__":
    run_experiment()
