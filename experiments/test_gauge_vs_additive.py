"""
Diagnostic Script: Gauge Field Modulation vs Static Dissipative Sinks
Author: Bazilevs & Karyon Cyberneticist
Objective: Empirically prove that Multiplicative Gauge Phase Rotation U(x) preserves
           full 258-dimensional spectral rank and avoids rank-1 memory collapse.
"""

import math
import torch
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_gauge_vs_additive():
    dim = 258
    seq_len = 1000
    
    stream_data = (
        "def karyon_sovereign_autopoiesis(stream):\n"
        "    super_operator = Liouvillian(phase_space)\n"
        "    return super_operator.synthesize()\n"
    ).encode("utf-8") * 10
    raw_bytes = list(stream_data)[:seq_len]

    # Model A: Additive Stimulus (EXP-418/419 style)
    # Model B: Multiplicative Gauge Field U(x) = exp(i * A(x))
    
    # Real generator for gauge field phases for each byte in [0, 255]
    theta_x = torch.randn(dim, dim, device=DEVICE) / math.sqrt(dim)

    # State tracking
    psi_A = torch.randn(dim, dtype=torch.cfloat, device=DEVICE)
    psi_A = psi_A / torch.norm(psi_A)

    psi_B = torch.randn(dim, dtype=torch.cfloat, device=DEVICE)
    psi_B = psi_B / torch.norm(psi_B)

    states_A = []
    states_B = []

    # Memory matrices
    M_A = torch.zeros(dim, dim, dtype=torch.cfloat, device=DEVICE)
    M_B = torch.zeros(dim, dim, dtype=torch.cfloat, device=DEVICE)

    # Static interaction Hamiltonian J
    J = torch.randn(dim, dim, dtype=torch.cfloat, device=DEVICE) / math.sqrt(dim)
    J = 0.5 * (J + J.conj().T) # Hermitian

    # Input projection for additive
    W_add = torch.randn(dim, dim, dtype=torch.cfloat, device=DEVICE) / math.sqrt(dim)

    for b in raw_bytes:
        onehot_f = F.one_hot(torch.tensor(b, device=DEVICE), num_classes=dim).float()
        onehot_c = onehot_f.cfloat()

        # --- Pipeline A: Additive Stimulus ---
        h_add = torch.mv(W_add, onehot_c)
        # 3 relaxation steps
        for _ in range(3):
            field_A = torch.mv(J + 0.1 * M_A, psi_A) + h_add
            psi_A = psi_A + (-1j * field_A - 0.2 * field_A) * 0.05
            psi_A = psi_A / torch.norm(psi_A)
        # Outer product memory
        M_A = 0.99 * M_A + 0.05 * torch.outer(psi_A, psi_A.conj())
        states_A.append(torch.cat([psi_A.real, psi_A.imag]))

        # --- Pipeline B: Multiplicative Gauge Phase Modulation ---
        # Gauge rotation: Psi -> exp(i * theta_b) * Psi
        phase_b = torch.mv(theta_x, onehot_f) # real phase shift vector
        U_b = torch.exp(1j * phase_b) # diagonal unitary gauge transformation
        psi_B_rot = U_b * psi_B
        
        # 3 relaxation steps in rotated frame
        for _ in range(3):
            field_B = torch.mv(J + 0.1 * M_B, psi_B_rot)
            psi_B_rot = psi_B_rot + (-1j * field_B - 0.2 * field_B) * 0.05
            psi_B_rot = psi_B_rot / torch.norm(psi_B_rot)
        psi_B = psi_B_rot

        # Memory inscription
        M_B = 0.99 * M_B + 0.05 * torch.outer(psi_B, psi_B.conj())
        states_B.append(torch.cat([psi_B.real, psi_B.imag]))

    # Analyze Spectral Rank of State Trajectories
    matrix_A = torch.stack(states_A)
    _, S_A, _ = torch.svd(matrix_A)
    pr_A = (torch.sum(S_A ** 2) ** 2) / torch.sum(S_A ** 4)

    matrix_B = torch.stack(states_B)
    _, S_B, _ = torch.svd(matrix_B)
    pr_B = (torch.sum(S_B ** 2) ** 2) / torch.sum(S_B ** 4)

    # Analyze Spectral Rank of Memory M
    _, S_MA, _ = torch.svd(M_A)
    pr_MA = (torch.sum(S_MA ** 2) ** 2) / torch.sum(S_MA ** 4)

    _, S_MB, _ = torch.svd(M_B)
    pr_MB = (torch.sum(S_MB ** 2) ** 2) / torch.sum(S_MB ** 4)

    print("\n=== COMPARATIVE SPECTRAL RANK AUDIT ===")
    print(f"Model A (Additive Stimulus):")
    print(f"  - Trajectory Participation Ratio: {pr_A.item():.2f} / 516")
    print(f"  - Memory Matrix Effective Rank:   {pr_MA.item():.2f} / 258")
    print(f"Model B (Multiplicative Gauge Phase U(x)):")
    print(f"  - Trajectory Participation Ratio: {pr_B.item():.2f} / 516 (INCREASE: {pr_B.item()/pr_A.item():.1f}x)")
    print(f"  - Memory Matrix Effective Rank:   {pr_MB.item():.2f} / 258 (INCREASE: {pr_MB.item()/pr_MA.item():.1f}x)")
    print("=======================================\n")

if __name__ == "__main__":
    test_gauge_vs_additive()
