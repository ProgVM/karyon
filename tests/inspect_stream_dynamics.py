import sys
sys.path.insert(0, '.')
import torch
from multi_domain_benchmark import generate_multi_domain_suite

suite = generate_multi_domain_suite(seed=101)
for domain in ['pointer', 'reversal', 'addition', 'parity', 'dyck']:
    p, ans, full = suite[domain][0]
    print(f"Domain: {domain:10s} | Prompt: {p!r:40s} | Full: {full!r}")
