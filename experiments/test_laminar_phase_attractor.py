"""
Diagnostic 3 (Fixed): Testing 2-Stage Laminar Phase-Attractor Field
"""

import math
import torch
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_non_commutative_gauge_binding():
    dim_fast = 258
    dim_slow = 258
    
    text = (
        "def karyon_sovereign_autopoiesis(stream):\n"
        "    super_operator = Liouvillian(phase_space)\n"
        "    return super_operator.synthesize()\n"
    )
    raw = [ord(c) for c in text] * 5 # 440 bytes

    Gen_x = torch.nn.Parameter(torch.randn(dim_fast, dim_fast, device=DEVICE) / math.sqrt(dim_fast))
    W_slow = torch.nn.Parameter(torch.randn(dim_slow, dim_fast, device=DEVICE) / math.sqrt(dim_fast))
    J_fast = torch.nn.Parameter(torch.randn(dim_fast, dim_fast, device=DEVICE) / math.sqrt(dim_fast))
    J_slow = torch.nn.Parameter(torch.randn(dim_slow, dim_slow, device=DEVICE) / math.sqrt(dim_slow))
    W_out = torch.nn.Parameter(torch.randn(dim_fast, dim_fast + dim_slow, device=DEVICE) / math.sqrt(dim_fast + dim_slow))
    beta = torch.nn.Parameter(torch.tensor(15.0, device=DEVICE))
    
    optimizer = torch.optim.AdamW([Gen_x, W_slow, J_fast, J_slow, W_out, beta], lr=0.03)
    
    print("\n=== Testing 2-Stage Laminar Phase-Attractor Field ===")
    
    for epoch in range(15):
        h_fast = torch.zeros(dim_fast, device=DEVICE)
        h_slow = torch.zeros(dim_slow, device=DEVICE)
        
        ep_loss = 0.0
        ep_correct = 0
        
        for t in range(len(raw) - 1):
            x = raw[t]
            y = raw[t + 1]
            
            optimizer.zero_grad()
            
            x_onehot = F.one_hot(torch.tensor(x, device=DEVICE), num_classes=dim_fast).float()
            
            # Fast phase rotation
            phase_fast = torch.mv(Gen_x, x_onehot)
            h_fast_in = torch.tanh(torch.mv(J_fast, h_fast.detach()) + phase_fast)
            
            # Slow state transition (integrates fast state over time)
            slow_drive = torch.mv(W_slow, h_fast_in)
            h_slow_in = torch.tanh(0.95 * torch.mv(J_slow, h_slow.detach()) + 0.2 * slow_drive)
            
            # Combined Laminar State
            h_total = torch.cat([h_fast_in, h_slow_in], dim=-1)
            
            # Readout with Inverse Temperature beta
            logits = F.softplus(beta) * torch.mv(W_out, h_total)
            
            loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([y], device=DEVICE))
            loss.backward()
            optimizer.step()
            
            pred = torch.argmax(logits).item()
            if pred == y:
                ep_correct += 1
            ep_loss += loss.item()
            
            h_fast = h_fast_in.detach()
            h_slow = h_slow_in.detach()
            
        acc = (ep_correct / (len(raw) - 1)) * 100.0
        avg_l = ep_loss / (len(raw) - 1)
        curr_b = F.softplus(beta).item()
        print(f"Epoch {epoch+1:2d} | Loss: {avg_l:.4f} nats | Accuracy: {acc:.2f}% | Beta: {curr_b:.2f}")

if __name__ == "__main__":
    test_non_commutative_gauge_binding()
