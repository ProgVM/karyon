import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from multi_domain_benchmark import generate_multi_domain_suite

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==============================================================================
# DEEP ENDOSCOPIC GRADIENT & ATTENTION PROBE (Principle 23)
# Why did Sharp Hopfield fail to reach 100%?
# Let's inspect:
# 1. Gradient norms of query, key, value projections
# 2. Attention map: where does q actually attend at hop 0, hop 1, hop 2?
# ==============================================================================

suite_train = generate_multi_domain_suite(seed=42)
suite_test = generate_multi_domain_suite(seed=999)

def encode_task(samples):
    max_p = max(len(p) for p, _, _ in samples)
    max_a = max(len(a) for _, a, _ in samples) + 1
    P = torch.full((len(samples), max_p), 256, dtype=torch.long, device=device)
    A = torch.full((len(samples), max_a), 256, dtype=torch.long, device=device)
    for i, (p, a, _) in enumerate(samples):
        p_b = [ord(c) for c in p]
        a_b = [ord(c) for c in a] + [ord('\n')]
        P[i, :len(p_b)] = torch.tensor(p_b, dtype=torch.long, device=device)
        A[i, :len(a_b)] = torch.tensor(a_b, dtype=torch.long, device=device)
    return P, A

ptr_train_P, ptr_train_A = encode_task(suite_train['pointer'])

# Let's inspect the exact string:
p_sample = suite_test['pointer'][0][0]
exp_sample = suite_test['pointer'][0][1]
print(f"Sample: '{p_sample}' -> Target: '{exp_sample}'")
for i, ch in enumerate(p_sample):
    print(f"Pos {i:02d}: '{ch}'")

