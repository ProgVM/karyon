import sys, os
sys.path.insert(0, '.')
import torch
from experiments.exp_290_dual_phase_recurrent_working_engine import MultiHeadSaccadicMind
from multi_domain_benchmark import generate_multi_domain_suite

# Let's inspect test predictions for Reversal and Pointer closely
suite_test = generate_multi_domain_suite(seed=999)
print("Samples from Reversal test suite:")
for p, exp, _ in suite_test['reversal'][:5]:
    print(f"P: {p!r:25s} -> Exp: {exp!r}")

print("\nSamples from Pointer test suite:")
for p, exp, _ in suite_test['pointer'][:5]:
    print(f"P: {p!r:45s} -> Exp: {exp!r}")
