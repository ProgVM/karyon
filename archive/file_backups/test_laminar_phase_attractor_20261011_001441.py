"""
Diagnostic 3: Breaking the 1.4 Nat Barrier towards 0.01 - 0.1 Nats.
Why can't a single-stage linear phase mixture separate overlapping contexts?
Because in text/code, the letter 'e' occurs in 'def', 'operator', 'phase_space', 'return'!
If context is a simple linear sum: alpha * Psi_{t-1} + exp(i*phase),
then after 'e' in 'def' and 'e' in 'operator', the states partially overlap.

To get Loss < 0.1 and Accuracy > 95%, we need:
1. Multi-Order Phase Binding (Non-Commutative Product or Tensor Product in C^D):
   Psi_t = Rot(x_t) @ Psi_{t-1} (Non-commutative Lie group action: G_1 * G_2 != G_2 * G_1).
   This makes "d" -> "e" completely orthogonal to "t" -> "e"!
2. Deep Attractor Resonance (Hopfield Energy Relaxation in Phase Space):
   Instead of just linear projection W_read @ Psi,
   the state relaxes in the energy landscape:
   E(Psi) = - 1/2 <Psi, M Psi> - <h, Psi>
   snapping into the exact deterministic memory basin!
"""

import math
import torch
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_non_commutative_gauge_binding():
    dim = 258
    text = (
        "def karyon_sovereign_autopoiesis(stream):\n"
        "    super_operator = Liouvillian(phase_space)\n"
        "    return super_operator.synthesize()\n"
    )
    raw = [ord(c) for c in text] * 5 # 440 bytes
    
    # Non-commutative Unitary Matrix Rotations for each byte:
    # Instead of diagonal phase shift U = diag(exp(i*theta)),
    # we use full Unitary Lie Group action: U(x) = expm(i * H_x) or Cayley transform!
    # Cayley transform: U(x) = (I - i A_x) (I + i A_x)^{-1} guarantees EXACT unitarity!
    # Or in 2-layer state: Fast Phase + Slow Carrier (Laminar PAC Phase-Space Duality)
    
    # Let's test a 2-stage Continuous Harmonic Wavefield:
    # Stage 1: Fast Phase (Phoneme/Byte)
    # Stage 2: Slow Attractor Basin (Word/Context)
    
    # Parameter definitions:
    dim_fast = 258
    dim_slow = 258
    
    # Embeddings / Generators
    Gen_x = torch.randn(dim_fast, dim_fast, device=DEVICE, requires_grad=True)
    W_slow = torch.randn(dim_slow, dim_fast, device=DEVICE, requires_grad=True)
    
    # Recurrent transition matrices
    J_fast = torch.randn(dim_fast, dim_fast, device=DEVICE, requires_grad=True) / math.sqrt(dim_fast)
    J_slow = torch.randn(dim_slow, dim_slow, device=DEVICE, requires_grad=True) / math.sqrt(dim_slow)
    
    # Resonant Readout
    W_out = torch.randn(dim_fast, dim_fast + dim_slow, device=DEVICE, requires_grad=True) / math.sqrt(dim_fast + dim_slow)
    beta = torch.tensor(15.0, device=DEVICE, requires_grad=True)
    
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
            # Nonlinear leaky state transition
            h_fast_in = torch.tanh(torch.mv(J_fast, h_fast.detach()) + phase_fast)
            
            # Slow state transition (integrates fast state over time)
            # Word-level / context attractor
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
