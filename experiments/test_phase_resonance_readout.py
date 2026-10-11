"""
Diagnostic 2: Continuous Phase Coherence vs Deterministic Sequence Memory.
Let's find out how a continuous physical wavefield can reach >95% accuracy and loss < 0.1
on an incoming stream without artificial Transformers or static tables.

Mechanisms to test:
1. Holographic Bound Context:
   Instead of just the current byte rotating Psi, the state vector Psi integrates a continuous
   traveling wave / delay-line phase convolution:
   Psi_t = alpha * Psi_{t-1} + (1 - alpha) * exp(i * theta(x_t))
   Or fractional power phase binding: Psi_t = (Psi_{t-1} * exp(i * omega))^gamma * exp(i * theta(x_t))
2. Phase Resonance Readout (Continuous Hopfield Projection / Associative Binding):
   Instead of just |Psi_j|^2 (which is uncoupled 1-to-1 index matching),
   the readout is an Associative Phase Resonance:
   logits = Sharpness * Re( Psi_t^dagger * Basis_j ) or | <Basis_j | Psi_t> |^2p
"""

import math
import torch
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_phase_resonance_readout():
    dim = 258
    text = (
        "def karyon_sovereign_autopoiesis(stream):\n"
        "    super_operator = Liouvillian(phase_space)\n"
        "    return super_operator.synthesize()\n"
    )
    raw = [ord(c) for c in text] * 5 # 440 bytes
    
    # Let's test a simple, elegant continuous model:
    # 1. Byte phase generator: Theta in R^{dim x dim}
    # 2. Resonant Readout Matrix: W_read in C^{dim x dim}
    # 3. Dynamic Sharpness / Temperature beta
    
    Theta = torch.randn(dim, dim, device=DEVICE, requires_grad=True)
    W_read_r = torch.randn(dim, dim, device=DEVICE, requires_grad=True)
    W_read_i = torch.randn(dim, dim, device=DEVICE, requires_grad=True)
    beta = torch.tensor(10.0, device=DEVICE, requires_grad=True) # Sharpness parameter!
    alpha = torch.tensor(0.85, device=DEVICE, requires_grad=True) # Context retention
    
    optimizer = torch.optim.AdamW([Theta, W_read_r, W_read_i, beta, alpha], lr=0.02)
    
    psi_r = torch.zeros(dim, device=DEVICE)
    psi_i = torch.zeros(dim, device=DEVICE)
    
    total_loss = 0.0
    correct = 0
    
    print("Testing Continuous Phase Resonance Field...")
    losses = []
    accs = []
    
    for epoch in range(10): # Let's see if the continuous physical field CAN learn to reach loss < 0.1
        psi_r = torch.zeros(dim, device=DEVICE)
        psi_i = torch.zeros(dim, device=DEVICE)
        ep_loss = 0.0
        ep_correct = 0
        
        for t in range(len(raw) - 1):
            x = raw[t]
            y = raw[t + 1]
            
            optimizer.zero_grad()
            
            # Input phase
            x_onehot = F.one_hot(torch.tensor(x, device=DEVICE), num_classes=dim).float()
            phase = torch.mv(Theta, x_onehot)
            cos_p = torch.cos(phase)
            sin_p = torch.sin(phase)
            
            # Context integration (Continuous Bound Wavepacket)
            # Psi_t = alpha * Psi_{t-1} + (1 - alpha) * exp(i * phase)
            curr_a = torch.sigmoid(alpha)
            new_r = curr_a * psi_r.detach() + (1.0 - curr_a) * cos_p
            new_i = curr_a * psi_i.detach() + (1.0 - curr_a) * sin_p
            norm = torch.sqrt(torch.sum(new_r**2 + new_i**2) + 1e-7)
            new_r = new_r / norm
            new_i = new_i / norm
            
            # Resonant Readout: Projected amplitude along each byte's basis in phase space!
            # z_j = < W_j | Psi >
            z_r = torch.mv(W_read_r, new_r) + torch.mv(W_read_i, new_i)
            z_i = torch.mv(W_read_r, new_i) - torch.mv(W_read_i, new_r)
            
            # Amplitude squared (Born intensity)
            intensity = z_r**2 + z_i**2
            
            # Phase Sharpness / Inverse Temperature
            curr_beta = F.softplus(beta)
            logits = curr_beta * torch.log(intensity + 1e-9)
            
            loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([y], device=DEVICE))
            loss.backward()
            optimizer.step()
            
            pred = torch.argmax(logits).item()
            if pred == y:
                ep_correct += 1
            ep_loss += loss.item()
            
            psi_r = new_r.detach()
            psi_i = new_i.detach()
            
        acc = (ep_correct / (len(raw) - 1)) * 100.0
        avg_l = ep_loss / (len(raw) - 1)
        print(f"Epoch {epoch+1:2d} | Loss: {avg_l:.4f} nats | Accuracy: {acc:.2f}% | Beta: {curr_beta.item():.2f}")

if __name__ == "__main__":
    test_phase_resonance_readout()
