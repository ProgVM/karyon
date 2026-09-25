import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import karyon_core

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ssd = karyon_core.CausalParallelSSD(64, str(device))
x = torch.randn(2, 10, 64, device=device, requires_grad=True)
out = ssd(x)
loss = out.sum()
loss.backward()
print("Grad on x:", x.grad is not None, x.grad.abs().sum().item())
params = list(ssd.parameters())
print("Grad on param 0:", params[0].grad is not None, params[0].grad.abs().sum().item())
