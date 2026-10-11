"""
Diagnostic: Clarifying SVD calculation on Complex Tensors in PyTorch
"""

import math
import torch

dim = 258
n_steps = 1000
phases = torch.randn(n_steps, dim)
psi_seq = torch.exp(1j * phases) / math.sqrt(dim)

M_sum = torch.zeros(dim, dim, dtype=torch.cfloat)
for t in range(n_steps):
    M_sum += torch.outer(psi_seq[t], psi_seq[t].conj())

# In PyTorch, torch.linalg.svdvals is the standard for complex matrices
S_real = torch.linalg.svdvals(M_sum)
print("Top 10 SVD values:", S_real[:10].tolist())
print("Bottom 5 SVD values:", S_real[-5:].tolist())

eigvals = S_real ** 2
pr = (torch.sum(eigvals) ** 2) / torch.sum(eigvals ** 2)
print(f"Correct SVD Participation Ratio on Complex M: {pr.item():.2f} / {dim}")
