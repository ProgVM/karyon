"""
EXP-522: Massively Parallel Batched CUDA/LibTorch Continuum Simulator
Karyon-CoRE v28.0 Sovereign Architecture

Simulating B=1024 parallel 2D spherical pendulum agents simultaneously
controlled by a 2D continuous metric tensor field on GPU (Tesla T4).
Pure PyTorch/CUDA tensor algebra without CPU sync stalls (.item() in loop).
"""

import time
import torch

def benchmark_batched_cuda_continuum(batch_size: int = 1024, total_steps: int = 10000, device: str = "cuda"):
    print("========================================================================")
    print(f"=== EXP-522: MASSIVELY PARALLEL BATCHED CUDA CONTINUUM (B={batch_size}) ===")
    print(f"=== GPU Accelerator: {torch.cuda.get_device_name(0)} | Total Steps: {total_steps} ===")
    print("========================================================================\n")

    dev = torch.device(device)

    # 16 Fourier modes per agent
    modes_k = torch.arange(1, 17, device=dev, dtype=torch.float32).unsqueeze(0) # (1, 16)
    k_sq = modes_k ** 2
    inv_k = 1.0 / modes_k

    # Agent states (Batch, 16)
    q = torch.zeros(batch_size, 16, device=dev, dtype=torch.float32)
    p = torch.zeros(batch_size, 16, device=dev, dtype=torch.float32)
    stiffness = torch.full((batch_size, 16), 2.0, device=dev, dtype=torch.float32)
    damping = torch.full((batch_size, 16), 1.0, device=dev, dtype=torch.float32)

    # Physical environment: B parallel inverted pendulums
    # Random initial perturbations in [-0.5, +0.5] rad
    theta = (torch.rand(batch_size, device=dev, dtype=torch.float32) - 0.5) * 0.90
    omega = torch.zeros(batch_size, device=dev, dtype=torch.float32)

    dt = 0.02
    g = 9.81
    L = 1.0

    upright_count = 0
    steady_state_upright_count = 0

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for step in range(total_steps):
        # Free Energy: F = 1/2 e^2 + 0.02 de^2
        fe = 0.5 * theta ** 2 + 0.02 * omega ** 2
        is_upright = (torch.abs(theta) < 0.20)
        upright_count += is_upright.sum()
        if step >= (total_steps // 2):
            steady_state_upright_count += is_upright.sum()

        # Spatial plucking per mode: (B, 16)
        plucking_base = ((theta + 0.15 * omega) * 10.0).unsqueeze(1)
        F_k = plucking_base * inv_k

        # Modal dynamics parameters
        omega_sq = stiffness * k_sq
        gamma = 0.5 * damping

        # Particular equilibrium
        q_star = F_k / (omega_sq + 1e-7)
        q_rel = q - q_star

        disc = omega_sq - gamma ** 2
        omega_d = torch.sqrt(torch.clamp(disc, min=1e-5))
        decay = torch.exp(-gamma * dt)
        cos_wt = torch.cos(omega_d * dt)
        sin_wt = torch.sin(omega_d * dt)
        inv_wd = 1.0 / omega_d

        q_next = decay * (q_rel * cos_wt + (p + gamma * q_rel) * inv_wd * sin_wt) + q_star
        p_next = decay * (p * cos_wt - (omega_sq * q_rel + gamma * p) * inv_wd * sin_wt)

        q = torch.clamp(q_next, -10.0, 10.0)
        p = torch.clamp(p_next, -30.0, 30.0)

        # Autopoietic sovereign self-tuning (Principle 29)
        active_fe = (fe > 1e-12).unsqueeze(1)
        d_k = 0.08 * torch.abs(theta).unsqueeze(1) * dt * inv_k
        d_gamma = 0.08 * torch.abs(omega).unsqueeze(1) * dt

        stiffness = torch.clamp(stiffness + torch.where(active_fe, d_k, 0.0), 0.1, 50.0)
        damping = torch.clamp(damping + torch.where(active_fe, d_gamma, 0.0), 0.1, 20.0)

        # Boundary restorative moment: \tau = 12 * \sum_k k * K * q
        torque = 12.0 * torch.sum(modes_k * stiffness * q, dim=1)

        # Physical pendulum update
        alpha = (g / L) * torch.sin(theta) - torque - 0.1 * omega
        omega = omega + dt * alpha
        theta = theta + dt * omega

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    elapsed = t1 - t0
    total_agent_steps = batch_size * total_steps
    throughput = total_agent_steps / elapsed

    overall_pct = (upright_count.item() / total_agent_steps) * 100.0
    steady_state_pct = (steady_state_upright_count.item() / (batch_size * (total_steps // 2))) * 100.0
    mean_fe = fe.mean().item()
    mean_theta = torch.abs(theta).mean().item()

    print(">>> CUDA BATCH SIMULATION COMPLETE:")
    print(f"  * Batch Size (Parallel Organisms) : {batch_size}")
    print(f"  * Steps per Organism              : {total_steps}")
    print(f"  * Total Agent Steps Evaluated     : {total_agent_steps:,}")
    print(f"  * Total Elapsed Time              : {elapsed:.3f} s")
    print(f"  * CUDA Throughput                 : {throughput:,.1f} agent-steps/sec")
    print(f"  * Overall Upright %               : {overall_pct:.2f}%")
    print(f"  * Steady-State % (Last 50% steps) : {steady_state_pct:.2f}%")
    print(f"  * Mean Equilibrium Angle          : {mean_theta:.3e} rad")
    print(f"  * Final Mean Free Energy          : {mean_fe:.3e}")
    print("========================================================================")

if __name__ == "__main__":
    benchmark_batched_cuda_continuum(batch_size=1024, total_steps=10000, device="cuda")
