import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import os
import time

# ==============================================================================
# EXP-732: DIRICHLET ACTION WAVEFIELD SUPERPOSITION (DAWS)
# Direct physical resolution of the Multiverse Wavefield Interference:
# Completely eliminates top-down SSD layers/hardcoded architectures.
# Replaces destructive phase cancellation (cos/sin in EXP-731) and Winner-Take-All
# collapse (EXP-730) with a Pure Normalized Feynman Path Integral Weighting (Dirichlet/Gibbs)
# across all U = 128 continuous parallel universes in imaginary time \tau = i*t.
#
# 1. Physical Action Functional for Universe u:
#    S_u = \frac{1}{2} ||P_u||^2 + V(\Phi_u) - G(x_t)
#
# 2. Normalized Feynman Path Integral Weighting (Dirichlet-Gibbs Measure):
#    W_u = \frac{\exp(-(S_u - \min S) / \hbar_{eff})}{\sum_k \exp(-(S_k - \min S) / \hbar_{eff})}
#    Ensures 100% constructive superposition without signal cancellation!
#
# 3. Dynamic Planck Scale \hbar_{eff} driven by Endogenous Free Energy F_t:
#    When F_t is high (surprise), \hbar_{eff} expands (broad quantum exploration).
#    When F_t -> 0 (mastery), \hbar_{eff} sharpens (focused path integral).
#
# 4. Ring Laplacian Inter-Universe Tunneling & Causal Epigenetic Locking:
#    Information diffuses across neighboring realities via Ring Laplacian:
#    \Delta_U \Phi_u = \Phi_{u+1} - 2\Phi_u + \Phi_{u-1}
#    Causal credit C_{u,t} = \gamma C_{u,t-1} + \Delta F_u \cdot (\Phi_u \otimes d\Phi_u)
#
# 5. Pure Continuous One-Pass Streaming (Zero Epochs).
# ==============================================================================

class DirichletMultiverseWavefieldSubstrate(nn.Module):
    """
    U = 128 Parallel Continuous Universes operating on a unified
    Dirichlet-Gibbs Feynman Path Integral Action Wavefield.
    """
    def __init__(self, num_universes=128, dim_field=128, base_hbar=1.5, trace_decay=0.90, causal_decay=0.95):
        super().__init__()
        self.num_universes = num_universes
        self.dim_field = dim_field
        self.base_hbar = base_hbar
        self.trace_decay = trace_decay
        self.causal_decay = causal_decay
        
        # Raw Byte Embedding into Spatial Wavefield Coordinates
        self.emb = nn.Embedding(258, dim_field)
        
        # Fundamental Multiverse Field Operators
        self.W_transverse = nn.Linear(dim_field, dim_field, bias=False)
        self.W_gauge = nn.Linear(dim_field, dim_field, bias=False)
        
        # Endocrine Synthesizers
        self.endocrine_gen = nn.Linear(dim_field + 2, 8)
        self.metric_gen = nn.Linear(dim_field + 9, dim_field)
        
        # Readout Motor Head
        self.wave_readout = nn.Linear(dim_field, 258, bias=False)
        
        # Per-Universe Causal Buffers [U, D, D]
        self.register_buffer("eligibility_traces", torch.zeros(num_universes, dim_field, dim_field))
        self.register_buffer("causal_credits", torch.zeros(num_universes, dim_field, dim_field))
        self.register_buffer("epigenetic_locks", torch.zeros(num_universes, dim_field, dim_field))
        self.register_buffer("prev_free_energy_u", torch.zeros(num_universes))
        
    def init_states(self, device):
        phi = torch.randn(self.num_universes, self.dim_field, device=device) * 0.1
        momenta = torch.zeros(self.num_universes, self.dim_field, device=device)
        self.eligibility_traces.zero_()
        self.causal_credits.zero_()
        self.epigenetic_locks.zero_()
        self.prev_free_energy_u.zero_()
        return phi, momenta
        
    def forward_step(self, byte_in, prev_error, phi_curr, p_curr):
        """
        Evolves all U = 128 universes with constructive Dirichlet Feynman Path Integral Weighting.
        """
        device = byte_in.device
        U, D = self.num_universes, self.dim_field
        
        # Broadcast input excitation across all 128 universes [U, D]
        x_wave = self.emb(byte_in).expand(U, -1)
        
        # 1. Introspective Self-Model & Endocrine Synthesis per Universe
        s_hat = torch.tanh(self.W_transverse(phi_curr)) # [U, D]
        F_self_u = torch.mean((s_hat - phi_curr) ** 2, dim=-1, keepdim=True) # [U, 1]
        
        endo_in = torch.cat([phi_curr, F_self_u, prev_error.expand(U, 1)], dim=-1) # [U, D + 2]
        H_endo_u = torch.sigmoid(self.endocrine_gen(endo_in)) # [U, 8]
        xi_u = torch.cat([phi_curr, F_self_u, H_endo_u], dim=-1) # [U, D + 9]
        
        # 2. Metric Conductance g_u per Universe
        g_u = torch.sigmoid(self.metric_gen(xi_u)) * 0.9 + 0.1 # [U, D]
        
        # 3. Inter-Universe Ring Laplacian Quantum Tunneling
        phi_prev = torch.roll(phi_curr, shifts=1, dims=0)
        phi_next = torch.roll(phi_curr, shifts=-1, dims=0)
        tunneling_flux = 0.05 * (phi_prev + phi_next - 2.0 * phi_curr) # [U, D]
        
        # 4. Soliton Wave Mechanics & Field Evolution
        kinetic_energy = 0.5 * torch.sum(p_curr ** 2, dim=-1) # [U]
        potential_energy = 0.5 * torch.sum(phi_curr ** 2, dim=-1) # [U]
        coupling_energy = -torch.sum(phi_curr * x_wave, dim=-1) # [U]
        
        # Physical Action S_u for each Universe
        S_u = kinetic_energy + potential_energy + coupling_energy # [U]
        
        # 5. Dynamic Planck Scale hbar_eff driven by Mean Free Energy
        mean_F = torch.mean(F_self_u) + prev_error
        hbar_eff = self.base_hbar * (1.0 + 2.0 * torch.clamp(mean_F, 0.0, 5.0))
        
        # 6. Dirichlet-Gibbs Feynman Path Integral Weighting (100% Constructive Superposition)
        S_shifted = (S_u - torch.min(S_u)) / (hbar_eff + 1e-6)
        W_dirichlet = F.softmax(-S_shifted, dim=0).unsqueeze(-1) # [U, 1]
        
        # 7. Constructive Wavefield Assembly across Multiverse
        constructed_wavefield = torch.sum(W_dirichlet * phi_curr, dim=0, keepdim=True) # [1, D]
        
        # 8. Field Differential & Equation Update (Hamilton-Jacobi / Langevin)
        d_Phi_u = torch.tanh(self.W_transverse(phi_curr) + x_wave) - phi_curr + tunneling_flux
        
        # 9. Endogenous Causal Epigenetics
        delta_F_u = F_self_u.squeeze(-1) - self.prev_free_energy_u # [U]
        self.prev_free_energy_u.copy_(F_self_u.squeeze(-1).detach())
        
        outer_prod_u = torch.bmm(phi_curr.unsqueeze(2), d_Phi_u.unsqueeze(1)) # [U, D, D]
        self.eligibility_traces.copy_((self.trace_decay * self.eligibility_traces + outer_prod_u).detach())
        
        causal_update = delta_F_u.view(U, 1, 1) * self.eligibility_traces
        self.causal_credits.copy_((self.causal_decay * self.causal_credits + causal_update).detach())
        
        # Epigenetic Lock Hysteresis
        lock_mask = torch.sigmoid((self.causal_credits - 0.01) * 100.0)
        self.epigenetic_locks.copy_(torch.maximum(self.epigenetic_locks, lock_mask).detach())
        
        # 10. Update Field States with Epigenetic Protection
        effective_d_Phi = d_Phi_u * (1.0 - 0.8 * torch.mean(self.epigenetic_locks, dim=-1))
        new_Momenta = p_curr * 0.8 + 0.2 * effective_d_Phi
        new_Field = phi_curr + g_u * new_Momenta
        
        # 11. Motor Logit Readout
        logits = self.wave_readout(constructed_wavefield) # [1, 258]
        
        # Calculate Entropy of Dirichlet Weights
        entropy_dirichlet = -torch.sum(W_dirichlet * torch.log(W_dirichlet + 1e-12))
        active_universes = torch.sum(W_dirichlet > (1.0 / (2.0 * U))).item()
        
        return logits, new_Field, new_Momenta, torch.mean(F_self_u), active_universes, entropy_dirichlet.item(), hbar_eff.item()

def run_experiment():
    print("================================================================================")
    print("STARTING EXP-732: DIRICHLET ACTION WAVEFIELD SUPERPOSITION (DAWS)")
    print("Testing Dirichlet-Gibbs Feynman Path Integral Weighting across U = 128 Universes.")
    print("================================================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Hardware: {device}")
    
    # Continuous text stream (Single continuous pass, 4096 bytes)
    stream_text = (
        "Karyon-CoRE v30.0 Sovereign Multiverse Specification. "
        "Mind is instantiated in parallel continuous wavefield universes without static synaptic matrices. "
        "U = 128 parallel universes evolve physical field dynamics in continuous time. "
        "Constructive Feynman path integral interference selects actions and text representations. "
        "Autopoietic metric fields adapt locally within each universe based on Free Energy minimization. "
    ) * 16 # ~4096 bytes

    raw_bytes = list(stream_text.encode('utf-8'))[:4096]
    seq_tensor = torch.tensor(raw_bytes, dtype=torch.long, device=device)

    Total_S = seq_tensor.shape[0]
    print(f"[*] Sequence Stream Length S = {Total_S} bytes | Multiverse Capacity U = 128 Universes")
    
    # Initialize Dirichlet Multiverse Substrate
    model = DirichletMultiverseWavefieldSubstrate(num_universes=128, dim_field=128, base_hbar=1.5).to(device)
    phi_curr, p_curr = model.init_states(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Telemetry Tracking
    start_time = time.time()
    total_loss = 0.0
    loss_history = []
    prev_error = torch.tensor(0.0, device=device)
    
    print("\n--- Initiating Pure Continuous One-Pass Streaming (Zero Epochs) ---")
    
    chunk_size = 128
    num_chunks = Total_S // chunk_size
    step_count = 0
    
    for c in range(num_chunks):
        c_start = c * chunk_size
        c_end = c_start + chunk_size
        chunk_bytes = seq_tensor[c_start:c_end]
        
        optimizer.zero_grad()
        chunk_loss = 0.0
        
        # Detach states at beginning of chunk for BPTT truncation
        phi_curr = phi_curr.detach()
        p_curr = p_curr.detach()
        
        for i in range(chunk_size - 1):
            byte_in = chunk_bytes[i:i+1]
            byte_target = chunk_bytes[i+1:i+2]
            
            logits, phi_curr, p_curr, F_self, active_u, entropy_w, hbar_val = model.forward_step(byte_in, prev_error, phi_curr, p_curr)
            step_loss = criterion(logits, byte_target)
            chunk_loss += step_loss
            prev_error = step_loss.detach()
            
        chunk_loss_avg = chunk_loss / (chunk_size - 1)
        chunk_loss_avg.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        loss_val = chunk_loss_avg.item()
        total_loss += loss_val
        step_count += 1
        loss_history.append(loss_val)
        
        lock_pct = (torch.sum(model.epigenetic_locks > 0.5).item() / model.epigenetic_locks.numel()) * 100.0
        ppl = math.exp(min(loss_val, 20.0))
        
        if (c + 1) % 4 == 0 or c == num_chunks - 1:
            print(f"Chunk {c+1:2d}/{num_chunks:2d} (t={c_end:4d}) | Loss: {loss_val:.4f} nats | PPL: {ppl:.2f} | Active U: {active_u:3d}/128 | S_Ent: {entropy_w:.3f} | hbar: {hbar_val:.2f} | Locks: {lock_pct:.2f}%")
            
    elapsed_time = time.time() - start_time
    final_loss = total_loss / step_count
    final_ppl = math.exp(min(final_loss, 20.0))
    tok_per_sec = Total_S / elapsed_time
    
    print("\n================================================================================")
    print("EXP-732 FINAL TELEMETRY RESULTS")
    print("================================================================================")
    print(f"Final Chunk Loss   : {loss_val:.4f} nats/byte")
    print(f"Mean Stream Loss   : {final_loss:.4f} nats/byte")
    print(f"Final Perplexity   : {final_ppl:.2f}")
    print(f"Processing Speed   : {tok_per_sec:.2f} bytes/sec")
    print(f"Elapsed Time       : {elapsed_time:.2f} seconds")
    print("================================================================================")
    
    # Save results summary
    metrics = {
        "exp_id": "EXP-732",
        "final_loss": round(loss_val, 4),
        "mean_loss": round(final_loss, 4),
        "ppl": round(math.exp(min(loss_val, 20.0)), 2),
        "tok_per_sec": round(tok_per_sec, 2),
        "elapsed_time": round(elapsed_time, 2),
        "num_universes": 128,
        "mechanism": "Dirichlet Action Wavefield Superposition (DAWS)"
    }
    
    with open("experiments/exp_732_results.json", "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    run_experiment()
