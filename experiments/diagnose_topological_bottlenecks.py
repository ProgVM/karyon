"""
Topological & Mathematical Diagnostic Suite for Continuous Phase Field & Holographic Memory
Author: Bazilevs & Lead Cyberneticist
Objective: Rigorous endoscopic audit of representation capacity, spectral rank collapse,
           latent-observation coupling, and memory interference in EXP-417/418.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_topological_diagnostics():
    print("=" * 80)
    print("=== RUNNING DEEP TOPOLOGICAL & SPECTRAL DIAGNOSTIC SUITE ===")
    print("=" * 80)

    dim = 258
    stream_data = (
        "def karyon_sovereign_autopoiesis(stream):\n"
        "    super_operator = Liouvillian(phase_space)\n"
        "    return super_operator.synthesize()\n"
    ).encode("utf-8") * 10
    raw_bytes = list(stream_data)[:1000]

    # Initialize model weights similar to EXP-417/418
    j_real = torch.randn(dim, dim, device=DEVICE) / math.sqrt(dim)
    j_imag = torch.randn(dim, dim, device=DEVICE) / math.sqrt(dim)
    j_real = 0.5 * (j_real + j_real.T)
    j_imag = 0.5 * (j_imag - j_imag.T)

    w_stim_r = torch.randn(dim, dim, device=DEVICE) / math.sqrt(dim)
    w_stim_i = torch.randn(dim, dim, device=DEVICE) / math.sqrt(dim)

    # State
    p_r = torch.zeros(dim, device=DEVICE)
    p_i = torch.zeros(dim, device=DEVICE)
    m_r = torch.zeros(dim, dim, device=DEVICE)
    m_i = torch.zeros(dim, dim, device=DEVICE)

    state_history = []
    stimulus_history = []
    memory_norms = []
    entropy_history = []
    cos_sim_stim_state = []

    for t in range(len(raw_bytes) - 1):
        x_byte = raw_bytes[t]
        target_byte = raw_bytes[t + 1]

        x_onehot = F.one_hot(torch.tensor(x_byte, device=DEVICE), num_classes=dim).float()
        h_r = torch.mv(w_stim_r, x_onehot)
        h_i = torch.mv(w_stim_i, x_onehot)

        # Relaxation
        gamma = 0.2
        dt = 0.05
        alpha_mem = 0.25

        eff_j_r = j_real + alpha_mem * m_r
        eff_j_i = j_imag + alpha_mem * m_i

        for step in range(6):
            field_r = torch.mv(eff_j_r, p_r) - torch.mv(eff_j_i, p_i) + h_r
            field_i = torch.mv(eff_j_r, p_i) + torch.mv(eff_j_i, p_r) + h_i

            norm_sq = p_r ** 2 + p_i ** 2
            v_r = 0.05 * norm_sq * p_r
            v_i = 0.05 * norm_sq * p_i

            dH_r = field_r - v_r
            dH_i = field_i - v_i

            p_r = p_r + (dH_i - gamma * dH_r) * dt
            p_i = p_i + (-dH_r - gamma * dH_i) * dt

        psi_norm = torch.sqrt(torch.sum(p_r ** 2 + p_i ** 2) + 1e-7)
        p_r_norm = p_r / psi_norm
        p_i_norm = p_i / psi_norm

        p_r = p_r_norm.detach()
        p_i = p_i_norm.detach()

        # Update Holographic Memory
        delta_m_r = torch.outer(p_r_norm, p_r_norm) + torch.outer(p_i_norm, p_i_norm)
        delta_m_i = torch.outer(p_i_norm, p_r_norm) - torch.outer(p_r_norm, p_i_norm)
        m_r = 0.99 * m_r + 0.05 * delta_m_r
        m_i = 0.99 * m_i + 0.05 * delta_m_i

        # Diagnostics
        probs = p_r_norm ** 2 + p_i_norm ** 2
        probs = probs / torch.sum(probs)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()

        stim_norm = torch.sqrt(torch.sum(h_r ** 2 + h_i ** 2) + 1e-7)
        h_r_norm = h_r / stim_norm
        h_i_norm = h_i / stim_norm
        cos_sim = torch.sum(p_r_norm * h_r_norm + p_i_norm * h_i_norm).item()

        state_history.append(torch.cat([p_r_norm, p_i_norm]))
        stimulus_history.append(torch.cat([h_r_norm, h_i_norm]))
        entropy_history.append(entropy)
        cos_sim_stim_state.append(cos_sim)
        memory_norms.append(torch.norm(m_r).item())

    state_matrix = torch.stack(state_history) # [1000, 516]
    
    # 1. Singular Value Decomposition of State History (Effective Capacity)
    U, S, V = torch.svd(state_matrix)
    eigvals = S ** 2
    participation_ratio = (torch.sum(eigvals) ** 2) / torch.sum(eigvals ** 2)

    # 2. Singular Value Decomposition of Holographic Memory M
    m_complex = torch.complex(m_r, m_i)
    _, S_m, _ = torch.svd(m_complex)
    m_participation_ratio = (torch.sum(S_m ** 2) ** 2) / torch.sum(S_m ** 4)

    # 3. Target Byte Alignment Audit
    print("\n--- [TOPOLOGICAL AUDIT RESULTS] ---")
    print(f"1. State Trajectory Participation Ratio (Effective State Dim): {participation_ratio.item():.2f} / {dim*2}")
    print(f"2. Holographic Memory Spectral Effective Rank (M Capacity):   {m_participation_ratio.item():.2f} / {dim}")
    print(f"3. Mean Cosine Similarity (Current Stimulus vs State Vector): {sum(cos_sim_stim_state)/len(cos_sim_stim_state):.4f}")
    print(f"4. Born Probability Entropy (Mean Output Entropy):          {sum(entropy_history)/len(entropy_history):.4f} nats (Max possible: {math.log(dim):.4f})")
    print(f"5. Top Singular Values of Memory Matrix M:                   {S_m[:5].tolist()}")
    print("----------------------------------------------------\n")

if __name__ == "__main__":
    run_topological_diagnostics()
