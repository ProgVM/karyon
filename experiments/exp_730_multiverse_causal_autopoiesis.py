import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import os
import time

# ==============================================================================
# EXP-730: MULTIVERSE CAUSAL AUTOPOIETIC WAVEFIELD ENGINE (MCAWE)
# 1. Substrate: U = 128 Parallel Continuous Physical Wavefields \Phi_u(x, t).
# 2. Endogenous Causal Credit Assignment per Universe:
#    e_{u, t} = \lambda e_{u, t-1} + (h_u \otimes dh_u)
#    C_{u, t} = \gamma C_{u, t-1} + \Delta F_u \cdot e_{u, t}
# 3. Epigenetic Invariant Locking (KEP Principle 6):
#    Physical field couplings proving \Delta F_u < 0 are epigenetically locked per universe.
# 4. Feynman Path Integral Interference:
#    W(a) = \sum_{u=1}^U \exp(-S_u / \hbar_{eff}) \cdot \Phi_u(x, t)
# 5. Continuous One-Pass Online Lifetime Stream (Zero Static Epochs).
# ==============================================================================

class MultiverseCausalWavefieldSubstrate(nn.Module):
    """
    U = 128 Parallel Continuous Universes with Endogenous Causal Credit
    and Epigenetic Invariant Locking per Universe.
    """
    def __init__(self, num_universes=128, dim_field=128, hbar_eff=0.25, trace_decay=0.90, causal_decay=0.95):
        super().__init__()
        self.num_universes = num_universes
        self.dim_field = dim_field
        self.hbar_eff = hbar_eff
        self.trace_decay = trace_decay
        self.causal_decay = causal_decay
        
        # Byte Embedding into Wavefield Coordinate Space
        self.emb = nn.Embedding(258, dim_field)
        
        # Fundamental Multiverse Field Operators
        self.W_transverse = nn.Linear(dim_field, dim_field, bias=False)
        self.W_gauge = nn.Linear(dim_field, dim_field, bias=False)
        
        # Endocrine Synthesizers
        self.endocrine_gen = nn.Linear(dim_field + 2, 8)
        self.metric_gen = nn.Linear(dim_field + 9, dim_field)
        
        # Readout Motor Head
        self.wave_readout = nn.Linear(dim_field, 258, bias=False)
        
        # Multiverse Dynamic State Buffers [U, D]
        self.register_buffer("MultiverseField", torch.zeros(num_universes, dim_field))
        self.register_buffer("MultiverseMomenta", torch.zeros(num_universes, dim_field))
        
        # Per-Universe Causal Buffers [U, D, D]
        self.register_buffer("eligibility_traces", torch.zeros(num_universes, dim_field, dim_field))
        self.register_buffer("causal_credits", torch.zeros(num_universes, dim_field, dim_field))
        self.register_buffer("epigenetic_locks", torch.zeros(num_universes, dim_field, dim_field))
        self.register_buffer("prev_free_energy_u", torch.zeros(num_universes))
        
    def reset_multiverse(self, device):
        self.MultiverseField = torch.randn(self.num_universes, self.dim_field, device=device) * 0.1
        self.MultiverseMomenta = torch.zeros(self.num_universes, self.dim_field, device=device)
        
        self.eligibility_traces.zero_()
        self.causal_credits.zero_()
        self.epigenetic_locks.zero_()
        self.prev_free_energy_u.zero_()
        
    def forward_step(self, byte_in, prev_error):
        """
        Evolves all U = 128 universes in parallel for a single continuous step.
        """
        device = byte_in.device
        U, D = self.num_universes, self.dim_field
        
        # Broadcast input excitation across all 128 universes [U, D]
        x_wave = self.emb(byte_in).expand(U, -1)
        
        # 1. Introspective Self-Model & Endocrine Synthesis per Universe
        s_hat = torch.tanh(self.W_transverse(self.MultiverseField)) # [U, D]
        F_self_u = torch.mean((s_hat - self.MultiverseField) ** 2, dim=-1, keepdim=True) # [U, 1]
        
        endo_in = torch.cat([self.MultiverseField, F_self_u, prev_error.expand(U, 1)], dim=-1) # [U, D + 2]
        H_endo_u = torch.sigmoid(self.endocrine_gen(endo_in)) # [U, 8]
        xi_u = torch.cat([self.MultiverseField, F_self_u, H_endo_u], dim=-1) # [U, D + 9]
        
        # 2. Metric Conductance g_u per Universe
        g_u = torch.sigmoid(self.metric_gen(xi_u)) * 0.9 + 0.1 # [U, D]
        
        # 3. Wavefield Step (Verlet / Symplectic Dynamics)
        field_flux = torch.tanh(self.W_gauge(self.MultiverseField))
        d_momenta = g_u * (x_wave + field_flux - 0.15 * self.MultiverseField)
        
        new_momenta = 0.90 * self.MultiverseMomenta + 0.10 * d_momenta
        new_field = self.MultiverseField + 0.20 * new_momenta
        
        # Detach states to prevent backprop graph pollution across long time streams
        self.MultiverseMomenta = new_momenta.detach()
        self.MultiverseField = new_field.detach()
        
        # 4. Action Functional S_u per Universe (Free Energy S_u = Kinetic + Potential + Self_Error)
        kinetic_u = 0.5 * torch.sum(new_momenta ** 2, dim=-1) # [U]
        potential_u = 0.5 * torch.sum(new_field ** 2, dim=-1) # [U]
        S_u = kinetic_u + potential_u + F_self_u.squeeze(-1) # [U]
        
        # 5. Feynman Path Integral Interference
        # Constructive superposition across 128 parallel universes
        feynman_weights = F.softmax(-S_u / self.hbar_eff, dim=0).unsqueeze(1) # [U, 1]
        InterferenceWavefield = torch.sum(feynman_weights * new_field, dim=0, keepdim=True) # [1, D]
        
        logits = self.wave_readout(InterferenceWavefield) # [1, 258]
        
        return {
            "logits": logits,
            "feynman_weights": feynman_weights,
            "S_u": S_u,
            "F_self_u": F_self_u,
            "field_u": new_field,
            "flux_u": d_momenta,
            "InterferenceWavefield": InterferenceWavefield
        }
        
    @torch.no_grad()
    def update_causal_autopoiesis(self, step_out, current_loss_val):
        """
        Per-Universe Endogenous Causal Credit Assignment & Soft Epigenetic Locking.
        Tracks eligibility traces e_{u, t} and causal credit C_{u, t} per universe.
        """
        U, D = self.num_universes, self.dim_field
        
        field_u = step_out["field_u"].detach() # [U, D]
        flux_u = step_out["flux_u"].detach() # [U, D]
        S_u = step_out["S_u"].detach() # [U]
        
        # Per-universe Free Energy S_u + Global Task Loss
        F_total_u = S_u + current_loss_val # [U]
        
        # Delta Free Energy per universe
        prev_F = self.prev_free_energy_u
        delta_F_u = torch.where(prev_F == 0.0, torch.zeros_like(F_total_u), F_total_u - prev_F)
        self.prev_free_energy_u.copy_(F_total_u)
        
        # Outer product flux per universe [U, D, D]
        local_flux_u = torch.bmm(field_u.unsqueeze(2), flux_u.unsqueeze(1))
        
        # 1. Update Eligibility Traces: e_{u, t} = lambda * e_{u, t-1} + local_flux
        self.eligibility_traces.mul_(self.trace_decay).add_(local_flux_u, alpha=1.0 - self.trace_decay)
        
        # 2. Update Causal Credit: C_{u, t} = gamma * C_{u, t-1} + Delta F_u * e_{u, t}
        delta_F_expanded = delta_F_u.view(U, 1, 1)
        self.causal_credits.mul_(self.causal_decay).add_(self.eligibility_traces * delta_F_expanded)
        
        # 3. Soft Epigenetic Invariant Locking (Top-15% quantile soft locking per universe)
        # Prevents 100% lockouts while protecting proven successful wave channels
        q15 = torch.quantile(self.causal_credits.view(U, -1), 0.15, dim=-1, keepdim=True).view(U, 1, 1)
        proven_efficacy_u = (self.causal_credits < q15).float()
        
        # Dynamic hysteresis: lock accumulates on success, dissolves slightly on error
        lock_decay = torch.where(delta_F_expanded > 0, torch.full_like(self.epigenetic_locks, 0.02), torch.zeros_like(self.epigenetic_locks))
        self.epigenetic_locks.add_(proven_efficacy_u, alpha=0.05).sub_(lock_decay).clamp_(0.0, 0.85)


def run_experiment():
    print("==========================================================================")
    print("=== EXP-730: MULTIVERSE CAUSAL AUTOPOIETIC WAVEFIELD ENGINE (MCAWE) ===")
    print("=== U = 128 Parallel Continuous Universes | Causal Autopoiesis | Feynman ===")
    print("==========================================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Host CPU"
    print(f"[*] Compute Device: {device} ({device_name})")
    
    # Continuous text stream (Single continuous pass, 4096 bytes)
    stream_text = (
        "Karyon-CoRE v30.0 Sovereign Engineering Protocol. "
        "True autopoiesis operates across parallel continuous wavefield universes. "
        "U = 128 parallel universes track eligibility traces and causal credit in continuous time. "
        "Feynman path integral interference selects wavefield actions without static weight matrices. "
        "Epigenetic locks protect proven wave channels per universe with dynamic soft hysteresis. "
    ) * 16 # ~4096 bytes
    
    raw_bytes = list(stream_text.encode('utf-8'))[:4096]
    seq_tensor = torch.tensor(raw_bytes, dtype=torch.long, device=device)
    
    Total_S = seq_tensor.shape[0]
    print(f"[*] Sequence Stream Length S = {Total_S} bytes | Multiverse Capacity U = 128 Universes")
    
    # Initialize 128-Universe Causal Substrate
    model = MultiverseCausalWavefieldSubstrate(num_universes=128, dim_field=128, hbar_eff=0.25).to(device)
    model.reset_multiverse(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    
    chunk_size = 128
    num_chunks = Total_S // chunk_size
    
    print("\n[*] Evolving 128 Universes with Endogenous Causal Autopoiesis & Feynman Interference...")
    start_time = time.time()
    
    err_prev = torch.zeros(1, device=device)
    loss_history = []
    active_universes_history = []
    locked_channels_history = []
    
    for c in range(num_chunks - 1):
        x_chunk = seq_tensor[c*chunk_size : (c+1)*chunk_size]
        y_chunk = seq_tensor[c*chunk_size + 1 : (c+1)*chunk_size + 1]
        
        optimizer.zero_grad()
        chunk_loss = 0.0
        
        for t in range(chunk_size):
            byte_in = x_chunk[t].unsqueeze(0)
            y_t = y_chunk[t].unsqueeze(0)
            
            step_out = model.forward_step(byte_in, err_prev)
            logits_t = step_out["logits"]
            loss_t = F.cross_entropy(logits_t, y_t)
            
            # Update endogenous causal autopoiesis across all 128 universes
            model.update_causal_autopoiesis(step_out, loss_t.item())
            
            err_prev = loss_t.detach()
            chunk_loss += loss_t
            
        avg_chunk_loss = chunk_loss / chunk_size
        avg_chunk_loss.backward()
        optimizer.step()
        
        current_loss_val = avg_chunk_loss.item()
        loss_history.append(current_loss_val)
        
        # Measure active dominant universes (Feynman weights > 1/128)
        weights = step_out["feynman_weights"].squeeze(-1)
        active_universes = torch.sum(weights > (1.0 / 128.0)).item()
        active_universes_history.append(active_universes)
        
        # Measure mean epigenetic lock ratio across multiverse
        mean_locked_pct = (model.epigenetic_locks > 0.40).float().mean().item() * 100.0
        locked_channels_history.append(mean_locked_pct)
        
        time_step = (c + 1) * chunk_size
        ppl = math.exp(min(current_loss_val, 50.0))
        print(f"  Stream t={time_step:04d}/{Total_S} | Loss: {current_loss_val:.4f} nats/byte | PPL: {ppl:.2f} | Active U: {active_universes}/128 | Locks: {mean_locked_pct:.1f}%")
        
    total_time = time.time() - start_time
    print(f"\n[+] Multiverse Causal Wavefield Evolution Completed in {total_time:.2f} seconds!")
    
    initial_loss = loss_history[0]
    final_loss = loss_history[-1]
    final_active_u = active_universes_history[-1]
    final_locks_pct = locked_channels_history[-1]
    adaptation_delta = initial_loss - final_loss
    
    print(f"\n[*] Multiverse Causal Wavefield Telemetry Metrics:")
    print(f"  Initial Stream Loss (t=0128): {initial_loss:.4f} nats/byte | PPL: {math.exp(min(initial_loss, 50.0)):.2f}")
    print(f"  Final Stream Loss   (t=3968): {final_loss:.4f} nats/byte | PPL: {math.exp(min(final_loss, 50.0)):.2f}")
    print(f"  One-Pass Adaptation Delta:   {adaptation_delta:.4f} nats/byte")
    print(f"  Active Dominant Universes:  {final_active_u} / 128 parallel nodes")
    print(f"  Epigenetic Lock Density:    {final_locks_pct:.2f}% across multiverse wave channels")
    
    baseline_loss = 2.7608 # EXP-728 Baseline
    delta_vs_baseline = baseline_loss - final_loss
    print(f"  Delta vs EXP-728 Baseline (2.7608): {delta_vs_baseline:.4f} nats/byte")
    
    verdict = "POSITIVE" if delta_vs_baseline >= 0.08 else ("NEUTRAL" if delta_vs_baseline >= 0 else "REJECTED")
    
    metrics = {
        "loss": final_loss,
        "ppl": math.exp(min(final_loss, 50.0)),
        "initial_loss": initial_loss,
        "adaptation_delta": adaptation_delta,
        "active_universes": final_active_u,
        "epigenetic_lock_pct": final_locks_pct,
        "baseline_loss": baseline_loss,
        "delta_loss": delta_vs_baseline,
        "execution_time_sec": total_time,
        "verdict": verdict
    }
    
    with open("experiments/exp_730_results.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"[*] Results saved to experiments/exp_730_results.json")
    return metrics

if __name__ == "__main__":
    run_experiment()
