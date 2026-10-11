"""
Why SVD of M_sum has one gigantic eigenvalue:
Because each term outer(psi, psi*) is rank 1!
Wait, why does S_real have 371.1 as the first value?
Let's see what vector corresponds to 371.1!
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

U, S, Vh = torch.linalg.svd(M_sum)
print("Top 5 singular values:", S[:5].tolist())
# Check how much trace is captured by the first singular value:
print(f"Energy in mode 0: {(S[0]**2 / torch.sum(S**2)).item() * 100:.2f}%")
