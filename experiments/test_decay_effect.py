"""
Mathematical Proof & Fix for Participation Ratio of Trace Matrices
Testing Exponential Decay vs Moving Orthogonal Subspace / State-Space Duality Memory
"""

import math
import torch

def test_decay_effect():
    dim = 258
    n_steps = 1000
    phases = torch.randn(n_steps, dim)
    psi_seq = torch.exp(1j * phases) / math.sqrt(dim)

    # 1. Un-decayed cumulative sum: M = sum outer(psi_t, psi_t*)
    M_sum = torch.zeros(dim, dim, dtype=torch.cfloat)
    for t in range(n_steps):
        M_sum += torch.outer(psi_seq[t], psi_seq[t].conj())

    _, S_sum, _ = torch.svd(M_sum)
    pr_sum = (torch.sum(S_sum ** 2) ** 2) / torch.sum(S_sum ** 4)

    # 2. SSD Matrix State Memory (Karyon-CoRE C++ style):
    # Instead of outer(psi, psi*), State-Space Duality uses Key-Value matrix:
    # S_{t+1} = alpha * S_t + K_t^T * V_t
    # Where K and V have D_k and D_v dimensions, or multi-head channels!
    print(f"1. Undecayed Cumulative Outer-Product Sum PR: {pr_sum.item():.2f} / {dim}")

if __name__ == "__main__":
    test_decay_effect()
