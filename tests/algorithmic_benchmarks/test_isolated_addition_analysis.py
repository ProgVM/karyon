import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from multi_domain_benchmark import generate_multi_domain_suite

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Deep Dive: Addition Domain
suite_train = generate_multi_domain_suite(seed=42)
suite_test = generate_multi_domain_suite(seed=999)

add_train = suite_train['addition']
add_test = suite_test['addition']

print(f"Addition Train Samples: {len(add_train)} | Test Samples: {len(add_test)}")
for p, a, _ in add_train[:8]:
    print(f"Prompt: '{p}' -> Answer: '{a}'")
