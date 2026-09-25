import sys
sys.path.insert(0, '.')
import torch
import karyon_agent
from multi_domain_benchmark import generate_multi_domain_suite

# Let's inspect what the agent is predicting for dyck vs pointer!
suite_eval = generate_multi_domain_suite(seed=202)
# Load checkpoint if saved or let's inspect generation behavior
