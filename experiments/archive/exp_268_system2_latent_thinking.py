# experiments/exp_268_system2_latent_thinking.py
"""
===============================================================================
EXP-268: TRUE SYSTEM 2 LATENT THINKING WITH COGNITIVE RECIRCULATION
Testing exact symbolic calculation by separating "thinking time" from "output time".
Forces the hidden state to undergo K pure latent recurrent steps at the "=" boundary
before generating any characters, trained end-to-end via BPTT.
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
from typing import Optional, Tuple, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class BilinearMorphicUnit(nn.Module):
    """
    Bilinear Multiplicative Operator Node:
    Computes y = (W_left x) * (W_right x) + W_lin x
    Enables native multiplication and sharp logical gates.
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
        mult = self.w_left(x) * self.w_right(x)
        lin = self.w_lin(x)
        g = torch.sigmoid(self.gate(x))
        return self.norm(x + g * F.silu(mult + lin))


class AttractorSlotMemory(nn.Module):
    """
    Working Memory Slot Bank with Attractor Persistence:
    Maintains K isolated slots (registers) with soft read/write gates.
    """
    def __init__(self, num_slots: int = 4, dim: int = 256):
        super().__init__()
        self.num_slots = num_slots
        self.dim = dim
        self.slots = nn.Parameter(torch.randn(num_slots, dim) * 0.02)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        slots_expanded = self.slots.unsqueeze(0).unsqueeze(0).expand(B, S, self.num_slots, D)
        
        q = self.q_proj(x).unsqueeze(2) # [B, S, 1, D]
        k = self.k_proj(slots_expanded)  # [B, S, K, D]
        v = self.v_proj(slots_expanded)  # [B, S, K, D]
        
        attn = torch.softmax(torch.sum(q * k, dim=-1, keepdim=True) / math.sqrt(D), dim=2) # [B, S, K, 1]
        read_val = torch.sum(attn * v, dim=2) # [B, S, D]
        
        return self.norm(x + read_val)


class MorphicRecurrentCell(nn.Module):
    """
    Morphic Recurrent Cell combining Bilinear units and slot memory.
    """
    def __init__(self, dim: int = 256, num_slots: int = 4):
        super().__init__()
        self.dim = dim
        self.bilinear = BilinearMorphicUnit(dim)
        self.memory_slots = AttractorSlotMemory(num_slots, dim)
        self.gate = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h: torch.Tensor, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        # h: [B, D]
        # x: [B, D] (optional input)
        h_combined = h + x if x is not None else h
        
        h_3d = h_combined.unsqueeze(1) # [B, 1, D]
        h_calc = self.bilinear(h_3d)
        h_mem = self.memory_slots(h_calc)
        h_out = h_mem.squeeze(1)
        
        g = torch.sigmoid(self.gate(h_combined))
        return self.norm(h_combined + g * (h_out - h_combined))


class System2MorphicAgent(nn.Module):
    """
    True System 2 Agent:
    1. Encodes the query sequence.
    2. Runs K steps of pure latent recirculation on the final query state.
    3. Decodes the target sequence autoregressively, initialized with the evolved state.
    """
    def __init__(self, vocab_size: int = 258, dim: int = 256, num_slots: int = 4, num_recirc: int = 8):
        super().__init__()
        self.dim = dim
        self.num_recirc = num_recirc
        self.emb = nn.Embedding(vocab_size, dim)
        self.cell = MorphicRecurrentCell(dim, num_slots)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.emb.weight # Weight tying

    def encode_query(self, query_ids: torch.Tensor) -> torch.Tensor:
        # query_ids: [B, L_q]
        B, L_q = query_ids.shape
        h = torch.zeros(B, self.dim, device=query_ids.device)
        
        # Process query sequentially to build recurrent state
        for t in range(L_q):
            x = self.emb(query_ids[:, t])
            h = self.cell(h, x)
        return h

    def run_latent_thinking(self, h: torch.Tensor) -> torch.Tensor:
        # Run K steps of pure latent evolution (no inputs, no outputs)
        for _ in range(self.num_recirc):
            h = self.cell(h, x=None)
        return h

    def forward(self, query_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        # query_ids: [B, L_q] (padded left)
        # target_ids: [B, L_t] (target sequence)
        B, L_t = target_ids.shape
        
        # 1. Encode query
        h = self.encode_query(query_ids)
        
        # 2. Latent thinking phase (System 2)
        h = self.run_latent_thinking(h)
        
        # 3. Decode target step-by-step
        logits_list = []
        for t in range(L_t):
            # Predict next token from current state
            logits = self.head(self.norm(h))
            logits_list.append(logits.unsqueeze(1))
            
            # Feed current target token as input to transition to next state
            x = self.emb(target_ids[:, t])
            h = self.cell(h, x)
            
        return torch.cat(logits_list, dim=1) # [B, L_t, Vocab]


# -----------------------------------------------------------------------------
# Symbolic Mathematics Tasks Data Generation
# -----------------------------------------------------------------------------
def make_example(op: str, a: int, b: int) -> Tuple[str, str]:
    if op == "star":
        res = 2 * a + b
    elif op == "hash":
        res = a * b - a
    elif op == "delta":
        res = a * a + b
    else:
        raise ValueError(f"Unknown op {op}")
    return f"{op}({a},{b})=", f"{res}\n"


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


def collate_fn(batch_pairs: List[Tuple[str, str]], device) -> Tuple[torch.Tensor, torch.Tensor]:
    # Left-pad queries to align the "=" character at the final index
    encoded_queries = [torch.tensor(list(q.encode("utf-8")), dtype=torch.long) for q, _ in batch_pairs]
    max_q_len = max(len(q) for q in encoded_queries)
    padded_queries = torch.full((len(batch_pairs), max_q_len), 256, dtype=torch.long) # 256 is pad
    for i, q in enumerate(encoded_queries):
        padded_queries[i, max_q_len - len(q):] = q

    # Right-pad targets
    encoded_targets = [torch.tensor(list(t.encode("utf-8")), dtype=torch.long) for _, t in batch_pairs]
    max_t_len = max(len(t) for t in encoded_targets)
    padded_targets = torch.full((len(batch_pairs), max_t_len), 256, dtype=torch.long)
    for i, t in enumerate(encoded_targets):
        padded_targets[i, :len(t)] = t

    return padded_queries.to(device), padded_targets.to(device)


def evaluate_exact_accuracy(model, dataset, device) -> Tuple[float, List[str]]:
    model.eval()
    correct = 0
    total = len(dataset)
    mismatches = []

    with torch.no_grad():
        for op, a, b in dataset:
            q_str, t_str = make_example(op, a, b)
            expected = t_str.strip()
            
            # Encode query
            q_tensor = torch.tensor([list(q_str.encode("utf-8"))], dtype=torch.long, device=device)
            h = model.encode_query(q_tensor)
            
            # Latent thinking
            h = model.run_latent_thinking(h)
            
            # Decode step-by-step
            decoded_bytes = []
            for _ in range(8):
                logits = model.head(model.norm(h))
                next_byte = torch.argmax(logits[0]).item()
                if next_byte in (ord('\n'), 257, 256):
                    break
                decoded_bytes.append(next_byte)
                
                # Feed generated byte back into cell
                x = model.emb(torch.tensor([next_byte], dtype=torch.long, device=device))
                h = model.cell(h, x)

            generated = bytes(decoded_bytes).decode("utf-8", errors="ignore").strip()
            if generated == expected:
                correct += 1
            else:
                if len(mismatches) < 3:
                    mismatches.append(f"Query '{q_str}' -> Got '{generated}', Expected '{expected}'")

    acc = (correct / total) * 100.0 if total > 0 else 0.0
    return acc, mismatches


def run_experiment():
    print("=" * 80)
    print("EXP-268: TRUE SYSTEM 2 LATENT THINKING BENCHMARK")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware execution substrate: {device}")

    train_data, val_id_data, val_ood_data = create_dataset()
    print(f"Dataset summary: Train={len(train_data)}, In-Domain Val={len(val_id_data)}, OOD Val={len(val_ood_data)}")

    # Instantiate System 2 Agent with 8 Latent Thinking Steps
    model = System2MorphicAgent(
        vocab_size=258,
        dim=256,
        num_slots=4,
        num_recirc=8
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=256)

    epochs = 150
    batch_size = 32

    print("\n--- Training Phase (End-to-End BPTT over Latent Recirculation) ---")
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(train_data)
        total_loss = 0.0
        batches = 0

        for i in range(0, len(train_data), batch_size):
            batch_pairs = [make_example(op, a, b) for op, a, b in train_data[i:i+batch_size]]
            queries, targets = collate_fn(batch_pairs, device)

            optimizer.zero_grad()
            logits = model(queries, targets)
            
            # Compute cross-entropy loss over target sequence
            loss = criterion(logits.reshape(-1, 258), targets.reshape(-1))
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
    print("EXP-268 TELEMETRY SUMMARY")
    print("=" * 80)
    print(f"Final Train Loss : {avg_loss:.4f}")
    print(f"In-Domain Acc    : {id_acc:.2f}%")
    print(f"OOD Acc          : {ood_acc:.2f}%")
    print("=" * 80)

    metrics = {
        "final_loss": float(avg_loss),
        "in_domain_acc": float(id_acc),
        "ood_acc": float(ood_acc),
        "train_duration_s": float(train_duration)
    }
    with open("experiments/exp_268_metrics.json", "w") as f:
        json.dump(metrics, f)

    return metrics


if __name__ == "__main__":
    run_experiment()
