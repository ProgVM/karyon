import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import os
import time

# ==============================================================================
# EXP-733.2: SOVEREIGN EPIGENETIC METRIC WAVEFIELD - CAUSAL LOCK FIX (SEM-WAVE-2)
# KEP Protocol v30.0 Master Compliance
#
# Fixes Epigenetic Lock Direction:
# In EXP-733 v1, lock_mask locked channels with C_u > +0.005 (channels that INCREASED Free Energy).
# In EXP-733.2, C_u < -0.005 (channels that REDUCE Free Energy, \Delta F < 0) are epigenetically shielded!
# ==============================================================================

class SovereignEpigeneticMetricWavefieldSubstrate(nn.Module):
    def __init__(self, num_universes=64, dim_field=256, num_attractors=64, beta_hopfield=12.0):
        super().__init__()
        self.num_universes = num_universes
        self.dim_field = dim_field
        self.num_attractors = num_attractors
        self.beta_hopfield = beta_hopfield
        
        # 1. Raw Byte Continuous Wave Injection (V=258 -> R^256)
        self.wave_inj = nn.Embedding(258, dim_field)
        
        # 2. Continuous Field Super-Primitive Operator (Principle 3)
        self.W_transverse = nn.Linear(dim_field, dim_field, bias=False)
        self.W_potential = nn.Linear(dim_field, dim_field, bias=False)
        
        # 3. Unit-Sphere Normalized Soliton Attractor Basins (Continuous Hopfield)
        raw_basins = torch.randn(num_attractors, dim_field)
        self.basins = nn.Parameter(F.normalize(raw_basins, p=2, dim=-1))
        
        # 4. Endogenous Introspection & Endocrine Synthesizers
        self.endocrine_gen = nn.Linear(dim_field + 2, 8)
        self.metric_gen = nn.Linear(dim_field + 9, dim_field) # Endogenous metric g_u \in R^256
        self.clock_gen = nn.Linear(dim_field + 9, 1)          # Relativistic clock \Delta t_{eff}
        
        # 5. Direct Motor Readout Head
        self.head = nn.Linear(dim_field, 258, bias=False)
        
        # 6. Persistent Causal Credit & Epigenetic Invariant Buffers
        self.register_buffer("prev_free_energy_u", torch.zeros(num_universes))
        self.register_buffer("eligibility_traces", torch.zeros(num_universes, dim_field))
        self.register_buffer("causal_credits", torch.zeros(num_universes, dim_field))
        self.register_buffer("epigenetic_locks", torch.zeros(num_universes, dim_field))

    def init_multiverse_state(self, device):
        phi = torch.randn(self.num_universes, self.dim_field, device=device) * 0.05
        momenta = torch.zeros(self.num_universes, self.dim_field, device=device)
        self.prev_free_energy_u.zero_()
        self.eligibility_traces.zero_()
        self.causal_credits.zero_()
        self.epigenetic_locks.zero_()
        return phi, momenta

    def forward_step(self, byte_in, prev_error, phi_curr, p_curr):
        U, D = self.num_universes, self.dim_field
        device = byte_in.device
        
        # 1. Continuous Wave Injection
        x_wave = self.wave_inj(byte_in).expand(U, -1) # [U, 256]
        
        # 2. Introspective Free Energy F_{self,u}
        s_hat = torch.tanh(self.W_transverse(phi_curr)) # [U, 256]
        F_self_u = torch.mean((s_hat - phi_curr) ** 2, dim=-1, keepdim=True) # [U, 1]
        
        # 3. Dynamic Endocrine & Phase State \xi_{u,t}
        endo_in = torch.cat([phi_curr, F_self_u, prev_error.expand(U, 1)], dim=-1)
        H_endo_u = torch.sigmoid(self.endocrine_gen(endo_in))
        xi_u = torch.cat([phi_curr, F_self_u, H_endo_u], dim=-1)
        
        # 4. Endogenous Metric Tensor g_u and Relativistic Clock \Delta t_{eff}
        g_u = torch.sigmoid(self.metric_gen(xi_u)) * 0.9 + 0.1 # [U, 256]
        dt_eff_u = torch.sigmoid(self.clock_gen(xi_u)) * 0.4 + 0.1 # [U, 1]
        
        # 5. Continuous Hopfield Soliton Phase Attractor Snapping
        phi_norm = F.normalize(phi_curr, p=2, dim=-1)
        basin_affinity = torch.matmul(phi_norm, self.basins.t()) * self.beta_hopfield
        p_attractor = F.softmax(basin_affinity, dim=-1)
        snap_target = torch.matmul(p_attractor, self.basins)
        F_attractor = snap_target - phi_curr
        
        # 6. Inter-Universe Ring Laplacian Tunneling
        phi_prev = torch.roll(phi_curr, shifts=1, dims=0)
        phi_next = torch.roll(phi_curr, shifts=-1, dims=0)
        tunneling_flux = 0.05 * (phi_prev + phi_next - 2.0 * phi_curr)
        
        # 7. Field Acceleration (Langevin-Hamiltonian Dynamics)
        V_force = -self.W_potential(phi_curr)
        accel = s_hat + x_wave + V_force + 0.35 * F_attractor + tunneling_flux - phi_curr
        
        # 8. Corrected Endogenous Causal Credit & Epigenetic Invariant Locking
        delta_F_u = F_self_u.squeeze(-1) - self.prev_free_energy_u # \Delta F = F_curr - F_prev
        self.prev_free_energy_u.copy_(F_self_u.squeeze(-1).detach())
        
        flux_diag = (phi_curr * p_curr).detach()
        self.eligibility_traces.mul_(0.90).add_(flux_diag, alpha=0.10)
        
        # Causal credit: C_{u,t} = \gamma C_{u,t-1} + \Delta F_u * e_t
        self.causal_credits.mul_(0.95).add_(self.eligibility_traces * delta_F_u.unsqueeze(-1), alpha=0.05)
        
        # CORRECTED LOCK CONDITION:
        # \Delta F < 0 (Free Energy DROPPED) => C_u < 0 (Channel caused Free Energy Reduction)
        # Lock channels where C_u < -0.002
        proven_beneficial = (-self.causal_credits - 0.002) * 50.0
        lock_mask = torch.sigmoid(proven_beneficial) # [U, 256]
        self.epigenetic_locks.copy_(torch.maximum(self.epigenetic_locks, lock_mask).detach())
        
        # 9. Symplectic Evolution with Epigenetic Protection
        plasticity_shield = 1.0 - 0.50 * self.epigenetic_locks # Max 50% shielding to preserve plasticity
        effective_accel = accel * plasticity_shield
        
        new_Momenta = p_curr * 0.85 + dt_eff_u * effective_accel
        new_Field = phi_curr + dt_eff_u * (g_u * new_Momenta)
        
        # 10. Action Functional S_u
        kinetic_energy = 0.5 * torch.sum(new_Momenta ** 2, dim=-1)
        potential_energy = 0.5 * torch.sum(new_Field ** 2, dim=-1)
        coupling_energy = -torch.sum(new_Field * x_wave, dim=-1)
        S_u = kinetic_energy + potential_energy + coupling_energy
        
        mean_F = torch.mean(F_self_u) + prev_error
        hbar_eff = 1.2 * (1.0 + 1.5 * torch.clamp(mean_F, 0.0, 4.0))
        
        # 11. Dirichlet-Gibbs Feynman Path Integral Weighting
        S_shifted = (S_u - torch.min(S_u)) / (hbar_eff + 1e-6)
        W_feynman = F.softmax(-S_shifted, dim=0).unsqueeze(-1)
        
        psi_observable = torch.sum(W_feynman * new_Field, dim=0, keepdim=True)
        
        # 12. Motor Readout
        logits = self.head(psi_observable)
        
        active_u = torch.sum(W_feynman > (1.0 / (2.0 * U))).item()
        lock_pct = (torch.sum(self.epigenetic_locks > 0.5).item() / self.epigenetic_locks.numel()) * 100.0
        
        return {
            "logits": logits,
            "new_Field": new_Field,
            "new_Momenta": new_Momenta,
            "F_self": torch.mean(F_self_u),
            "active_universes": active_u,
            "lock_pct": lock_pct,
            "hbar_eff": hbar_eff.item()
        }

def run_experiment():
    print("================================================================================")
    print("=== STARTING EXP-733.2: SEM-WAVE (CAUSAL EPIGENETIC LOCK DIRECTION FIX) ===")
    print("================================================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Hardware Substrate: {device}")
    
    stream_text = (
        "Karyon-CoRE v30.0 Sovereign Epigenetic Metric Wavefield Specification. "
        "Mind and intelligence are instantiated in continuous physical wavefield universes. "
        "Each universe u evolves a full 256-dimensional spatial-temporal field state. "
        "Endogenous metric tensor g_u and relativistic clock dt_eff adapt locally from Free Energy. "
        "Unit-sphere normalized soliton attractor basins provide crisp conceptual snapping. "
        "Constructive Feynman path integral interference selects actions in imaginary time. "
    ) * 16 # ~4096 raw bytes stream

    raw_bytes = list(stream_text.encode('utf-8'))[:4096]
    seq_tensor = torch.tensor(raw_bytes, dtype=torch.long, device=device)
    Total_S = seq_tensor.shape[0]

    model = SovereignEpigeneticMetricWavefieldSubstrate(num_universes=64, dim_field=256, num_attractors=64).to(device)
    phi_curr, p_curr = model.init_multiverse_state(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    criterion = nn.CrossEntropyLoss()
    
    chunk_size = 128
    num_chunks = Total_S // chunk_size
    
    start_time = time.time()
    total_loss = 0.0
    loss_history = []
    prev_error = torch.tensor(0.0, device=device)
    
    for c in range(num_chunks):
        c_start = c * chunk_size
        c_end = c_start + chunk_size
        chunk_bytes = seq_tensor[c_start:c_end]
        
        optimizer.zero_grad()
        chunk_loss = 0.0
        
        phi_curr = phi_curr.detach()
        p_curr = p_curr.detach()
        
        for i in range(chunk_size - 1):
            byte_in = chunk_bytes[i:i+1]
            byte_target = chunk_bytes[i+1:i+2]
            
            step_out = model.forward_step(byte_in, prev_error, phi_curr, p_curr)
            logits = step_out["logits"]
            phi_curr = step_out["new_Field"]
            p_curr = step_out["new_Momenta"]
            
            step_loss = criterion(logits, byte_target)
            chunk_loss += step_loss
            prev_error = step_loss.detach()
            
        chunk_loss_avg = chunk_loss / (chunk_size - 1)
        chunk_loss_avg.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        loss_val = chunk_loss_avg.item()
        total_loss += loss_val
        loss_history.append(loss_val)
        
        ppl = math.exp(min(loss_val, 20.0))
        active_u = step_out["active_universes"]
        lock_pct = step_out["lock_pct"]
        hbar_val = step_out["hbar_eff"]
        
        if (c + 1) % 4 == 0 or c == num_chunks - 1:
            print(f"Chunk {c+1:2d}/{num_chunks:2d} (t={c_end:4d}) | Loss: {loss_val:.4f} nats | PPL: {ppl:.2f} | Active U: {active_u:2d}/64 | hbar: {hbar_val:.2f} | Locks: {lock_pct:.2f}%")

    elapsed_time = time.time() - start_time
    final_loss = loss_history[-1]
    mean_loss = total_loss / num_chunks
    final_ppl = math.exp(min(final_loss, 20.0))
    tok_per_sec = Total_S / elapsed_time
    
    print("\n================================================================================")
    print("EXP-733.2 FINAL TELEMETRY RESULTS")
    print("================================================================================")
    print(f"Initial Chunk Loss : {loss_history[0]:.4f} nats/byte")
    print(f"Final Chunk Loss   : {final_loss:.4f} nats/byte")
    print(f"Mean Stream Loss   : {mean_loss:.4f} nats/byte")
    print(f"Final Perplexity   : {final_ppl:.2f}")
    print(f"Processing Speed   : {tok_per_sec:.2f} bytes/sec")
    print(f"Elapsed Time       : {elapsed_time:.2f} seconds")
    print(f"Final Epigenetic Locks : {lock_pct:.2f}%")
    
    baseline_loss = 3.5434
    delta_loss = baseline_loss - final_loss
    print(f"Delta Loss vs EXP-732 ({baseline_loss:.4f}): {delta_loss:+.4f} nats")
    
    verdict = "POSITIVE" if delta_loss >= 0.080 else ("NEUTRAL" if delta_loss >= 0 else "REJECTED")
    print(f"Automated KEP Rule #2 Verdict: {verdict}")
    print("================================================================================")
    
    metrics = {
        "loss": round(final_loss, 4),
        "mean_loss": round(mean_loss, 4),
        "ppl": round(final_ppl, 2),
        "delta_loss": round(delta_loss, 4),
        "baseline_loss": baseline_loss,
        "tok_per_sec": round(tok_per_sec, 2),
        "elapsed_time": round(elapsed_time, 2),
        "active_universes": active_u,
        "epigenetic_locks_pct": round(lock_pct, 2),
        "verdict": verdict
    }
    
    with open("experiments/exp_733_results.json", "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    run_experiment()
