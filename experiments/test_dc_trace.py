"""
Mathematical Proof: The DC Component (Trace) Anomaly in Trace Outer Products
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

# Notice that the diagonal of outer(psi, psi*) is |psi_j|^2 = 1/dim !
# Summing 1000 times adds a massive diagonal identity matrix: 1000/dim * I !
print("Mean diagonal element of M_sum:", torch.mean(torch.diag(M_sum).real).item())
print("Expected diagonal from pure trace:", n_steps / dim)

# If we remove the isotropic mean diagonal trace (trace removal / mean centering):
M_centered = M_sum - (torch.trace(M_sum) / dim) * torch.eye(dim, dtype=torch.cfloat)
S_cent = torch.linalg.svdvals(M_centered)
eig_cent = S_cent ** 2
pr_cent = (torch.sum(eig_cent) ** 2) / torch.sum(eig_cent ** 2)
print("Participation Ratio AFTER removing the DC trace baseline:", pr_cent.item(), f"/ {dim}")
