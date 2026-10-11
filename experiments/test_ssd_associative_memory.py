"""
Deep Mechanistic Investigation: How does an Autoregressive System reach 100% accuracy and Loss < 0.1?
Testing the exact mathematical limits of sequence prediction:
1. Pure Multi-Head State Space Duality (SSD) / Mamba-2 / Linear Recurrence
2. Associative Memory with Key-Value Outer Product S_t = S_{t-1} + K_t^T V_t
3. Modern Hopfield Energy Snapping
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_associative_key_value_memory():
    # If memory uses Keys and Values:
    # State S in R^{D_k x D_v}
    # Read: q = Q(x_t), pred = q @ S_t
    # Write: S_{t+1} = alpha * S_t + k_t^T * v_t
    # This is EXACTLY the State Space Duality / Fast Weight Programmer / Modern Hopfield formulation!
    
    text = (
        "def karyon_sovereign_autopoiesis(stream):\n"
        "    super_operator = Liouvillian(phase_space)\n"
        "    return super_operator.synthesize()\n"
    )
    raw = [ord(c) for c in text] * 5
    
    dim = 258
    dim_k = 64
    dim_v = 64
    
    class SSDAssociativeCore(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(dim, 128)
            self.W_q = nn.Linear(128, dim_k, bias=False)
            self.W_k = nn.Linear(128, dim_k, bias=False)
            self.W_v = nn.Linear(128, dim_v, bias=False)
            self.W_out = nn.Linear(dim_v, dim, bias=False)
            self.decay = nn.Parameter(torch.tensor(0.95))
            
        def forward(self, x_seq):
            # x_seq: [T]
            T = len(x_seq)
            embeds = self.emb(x_seq) # [T, 128]
            Q = self.W_q(embeds) # [T, dim_k]
            K = self.W_k(embeds) # [T, dim_k]
            V = self.W_v(embeds) # [T, dim_v]
            
            S = torch.zeros(dim_k, dim_v, device=DEVICE)
            outputs = []
            d = torch.sigmoid(self.decay)
            
            for t in range(T):
                q_t = Q[t] # [dim_k]
                # Read from associative memory matrix
                read_t = torch.matmul(q_t, S) # [dim_v]
                out_t = self.W_out(read_t) # [dim]
                outputs.append(out_t)
                
                # Write to associative memory matrix
                k_t = K[t] # [dim_k]
                v_t = V[t] # [dim_v]
                S = d * S + torch.outer(k_t, v_t)
                
            return torch.stack(outputs)
            
    model = SSDAssociativeCore().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    
    x_tensor = torch.tensor(raw[:-1], device=DEVICE)
    y_tensor = torch.tensor(raw[1:], device=DEVICE)
    
    print("\n=== Testing Key-Value Associative Memory State Space ===")
    for epoch in range(1, 31):
        optimizer.zero_grad()
        logits = model(x_tensor) # [T, dim]
        loss = F.cross_entropy(logits, y_tensor)
        loss.backward()
        optimizer.step()
        
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == y_tensor).float().mean().item() * 100.0
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:2d} | Loss: {loss.item():.4f} nats | Accuracy: {acc:.2f}% | Decay: {torch.sigmoid(model.decay).item():.3f}")

if __name__ == "__main__":
    test_associative_key_value_memory()
