"""
Diagnostic 4: Pure Online Single-Pass Stream Learning with Non-Zero Context.
Can an associative state space reach >90% accuracy and Loss < 0.2 in STRICT SINGLE-PASS (Epoch 1)?
Let's test single-pass online learning with continuous recurrence!
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_single_pass_associative():
    dim = 258
    dim_k = 64
    dim_v = 64
    
    text = (
        "def karyon_sovereign_autopoiesis(stream):\n"
        "    super_operator = Liouvillian(phase_space)\n"
        "    return super_operator.synthesize()\n"
    )
    raw = [ord(c) for c in text] * 10 # 880 bytes
    
    class SinglePassAssociativeField(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(dim, 128)
            self.W_q = nn.Linear(128, dim_k, bias=False)
            self.W_k = nn.Linear(128, dim_k, bias=False)
            self.W_v = nn.Linear(128, dim_v, bias=False)
            self.W_out = nn.Linear(dim_v, dim, bias=False)
            self.decay = nn.Parameter(torch.tensor(0.92))
            
        def step(self, x_byte, S):
            emb = self.emb(x_byte)
            q = self.W_q(emb)
            k = self.W_k(emb)
            v = self.W_v(emb)
            
            read = torch.matmul(q, S)
            logits = self.W_out(read)
            
            d = torch.sigmoid(self.decay)
            new_S = d * S + torch.outer(k, v)
            return logits, new_S
            
    model = SinglePassAssociativeField().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    
    S = torch.zeros(dim_k, dim_v, device=DEVICE)
    
    total_loss = 0.0
    correct = 0
    total = len(raw) - 1
    
    print("\n=== Single-Pass Online Stream Evolution ===")
    
    for t in range(total):
        x = torch.tensor(raw[t], device=DEVICE)
        target = torch.tensor(raw[t+1], device=DEVICE)
        
        optimizer.zero_grad()
        logits, next_S = model.step(x, S.detach())
        loss = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
        loss.backward()
        optimizer.step()
        
        pred = torch.argmax(logits).item()
        if pred == raw[t+1]:
            correct += 1
        total_loss += loss.item()
        S = next_S.detach()
        
        if (t + 1) % 100 == 0:
            window_acc = (correct / (t + 1)) * 100.0
            window_loss = total_loss / (t + 1)
            print(f"Byte {t+1:3d}/{total} | Avg Loss: {window_loss:.4f} nats | Acc: {window_acc:.2f}% | Current Loss: {loss.item():.4f}")

if __name__ == "__main__":
    test_single_pass_associative()
