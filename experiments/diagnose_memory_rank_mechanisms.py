"""
Deep Mechanistic Diagnosis: What Creates a Full-Rank Holographic Memory Matrix?
Investigating why Memory Matrix Effective Rank stays at ~2.5 despite state rank expansion.
"""

import math
import torch
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_memory_mechanisms():
    dim = 258
    n_steps = 1000

    # 1. Outer Product on single state: M = sum outer(psi, psi*)
    # 2. Outer Product on multi-scale / multi-head phase components: M = sum_h outer(psi_h, psi_h*)
    # 3. Holographic Associative Matrix: M = sum outer(psi_t, psi_{t-1}*) (Associative Transition Matrix!)

    # Let's generate a trajectory of complex states with high rank
    phases = torch.randn(n_steps, dim, device=DEVICE)
    psi_seq = torch.exp(1j * phases) / math.sqrt(dim) # rank is full!

    # A: Auto-Associative Outer Product: outer(psi_t, psi_t*)
    M_auto = torch.zeros(dim, dim, dtype=torch.cfloat, device=DEVICE)
    for t in range(n_steps):
        M_auto = 0.99 * M_auto + 0.05 * torch.outer(psi_seq[t], psi_seq[t].conj())

    # B: Hetero-Associative Transition Matrix: outer(psi_t, psi_{t-1}*)
    M_trans = torch.zeros(dim, dim, dtype=torch.cfloat, device=DEVICE)
    for t in range(1, n_steps):
        M_trans = 0.99 * M_trans + 0.05 * torch.outer(psi_seq[t], psi_seq[t-1].conj())

    # SVD of both
    _, S_auto, _ = torch.svd(M_auto)
    pr_auto = (torch.sum(S_auto ** 2) ** 2) / torch.sum(S_auto ** 4)

    _, S_trans, _ = torch.svd(M_trans)
    pr_trans = (torch.sum(S_trans ** 2) ** 2) / torch.sum(S_trans ** 4)

    print(f"Random Full-Rank State Sequence:")
    print(f"Auto-associative M_auto Effective Rank:   {pr_auto.item():.2f} / {dim}")
    print(f"Hetero-associative M_trans Effective Rank: {pr_trans.item():.2f} / {dim}")

if __name__ == "__main__":
    test_memory_mechanisms()
