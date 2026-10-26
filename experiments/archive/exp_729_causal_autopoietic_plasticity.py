import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import os
import time

# ==============================================================================
# EXP-729: CAUSAL AUTOPOIETIC PLASTICITY (CAP)
# Full, Endogenous, Causal Autopoiesis based on:
# 1. Eligibility Traces e_t = lambda * e_{t-1} + grad_local
# 2. Continuous Causal Credit Assignment C_t = gamma * C_{t-1} + Delta F_t * e_t
#    Directly quantifying the causal contribution of each channel to Free Energy minimization.
# 3. Epigenetic Invariant Locking (KEP Principle 6):
#    Channels proving negative cumulative causal cost (C_t << 0 -> Free Energy drop)
#    are epigenetically locked against destructive overwriting.
# 4. Endogenous Causal Plasticity: Field metric conductance g_t and recurrent coupling
#    adapt online based on endogenous causal credit, not blind backprop.
# ==============================================================================

class CausalAutopoieticPlasticityEngine(nn.Module):
    """
    Substrate with Endogenous Causal Credit Assignment and Epigenetic Locking.
    """
    def __init__(self, d_model=256, trace_decay=0.90, causal_decay=0.95):
        super().__init__()
        self.d_model = d_model
        self.trace_decay = trace_decay
        self.causal_decay = causal_decay
        
        self.emb = nn.Embedding(258, d_model)
        self.W_in = nn.Linear(d_model, d_model)
        
        # Recurrent physical field coupling matrix
        self.W_rec = nn.Parameter(torch.randn(d_model, d_model) * (1.0 / math.sqrt(d_model)))
        self.norm = nn.LayerNorm(d_model)
        
        # Introspective Self-Model (F_self = ||s_hat - s||^2)
        self.self_predictor = nn.Linear(d_model, d_model)
        
        # Endocrine Synthesizer (H_0 Success, H_1 Error, H_2..H_7 Dynamic)
        self.endocrine_gen = nn.Linear(d_model + 2, 8)
        
        # Autopoietic Field Metric Conductance Synthesizer
        self.W_g = nn.Linear(d_model + 9, d_model)
        self.W_alpha = nn.Linear(d_model + 9, d_model)
        
        # Readout Motor Head
        self.head = nn.Linear(d_model, 258, bias=False)
        
        # State Buffers for Endogenous Causal Tracking (Online Lifetime)
        self.register_buffer("eligibility_trace", torch.zeros(d_model, d_model))
        self.register_buffer("causal_credit", torch.zeros(d_model, d_model))
        self.register_buffer("epigenetic_locks", torch.zeros(d_model, d_model))
        self.register_buffer("prev_free_energy", torch.zeros(1))
        self.register_buffer("pending_delta_W", torch.zeros(d_model, d_model))
        
    def reset_causal_state(self, device):
        self.eligibility_trace.zero_()
        self.causal_credit.zero_()
        self.epigenetic_locks.zero_()
        self.prev_free_energy.zero_()
        self.pending_delta_W.zero_()
        
    def forward_step(self, x_t, h_prev, prev_error):
        """
        Single step execution in continuous time with internal causal evaluation.
        """
        B, D = x_t.shape
        
        # 1. Introspective Self-Model (s_self and F_self)
        s_hat = torch.tanh(self.self_predictor(h_prev))
        F_self = torch.mean((s_hat - h_prev) ** 2, dim=-1, keepdim=True) # [B, 1]
        
        # 2. Dynamic Endocrine State (H_endo)
        endo_in = torch.cat([h_prev, F_self, prev_error.unsqueeze(-1)], dim=-1)
        H_endo = torch.sigmoid(self.endocrine_gen(endo_in)) # [B, 8]
        xi_t = torch.cat([h_prev, F_self, H_endo], dim=-1) # [B, D + 9]
        
        # 3. Epigenetic Metric Conductance g_t and Decay alpha_t
        g_t = torch.sigmoid(self.W_g(xi_t)) * 0.9 + 0.1 # [B, D]
        alpha_t = torch.sigmoid(self.W_alpha(xi_t)) * 0.199 + 0.80 # [B, D]
        
        # 4. Continuous Wavefield Step
        h_norm = self.norm(h_prev)
        rec_flux = torch.matmul(h_norm, self.W_rec)
        dh = g_t * torch.tanh(self.W_in(x_t) + rec_flux)
        
        h_next = alpha_t * h_prev + (1.0 - alpha_t) * dh
        logits = self.head(h_next)
        
        return {
            "logits": logits,
            "h_next": h_next,
            "h_norm": h_norm,
            "dh": dh,
            "F_self": F_self,
            "xi_t": xi_t,
            "g_t": g_t
        }
        
    @torch.no_grad()
    def accumulate_causal_plasticity(self, h_norm, dh, current_F, lr_causal=1e-3):
        """
        Endogenous Causal Plasticity Accumulation (KEP Principles 5 & 6):
        Accumulates eligibility traces and causal delta_W in pending buffer
        without modifying in-place parameters during forward graph execution.
        """
        h_norm_d = h_norm.detach().squeeze(0)
        dh_d = dh.detach().squeeze(0)
        
        # Delta Free Energy (Delta F = F_curr - F_prev)
        if self.prev_free_energy.item() == 0.0:
            delta_F = 0.0
        else:
            delta_F = current_F - self.prev_free_energy.item()
        self.prev_free_energy.fill_(current_F)
        
        # Outer product of pre-synaptic h_norm and post-synaptic dh
        local_flux = torch.matmul(h_norm_d.unsqueeze(1), dh_d.unsqueeze(0)) # [D, D]
        
        # 1. Update Eligibility Trace: e_t = lambda * e_{t-1} + flux
        self.eligibility_trace.mul_(self.trace_decay).add_(local_flux, alpha=1.0 - self.trace_decay)
        
        # 2. Update Causal Credit: C_t = gamma * C_{t-1} + Delta F * e_t
        self.causal_credit.mul_(self.causal_decay).add_(self.eligibility_trace, alpha=delta_F)
        
        # 3. Epigenetic Invariant Locking (Principle 6)
        proven_efficacy = (self.causal_credit < -0.005).float()
        self.epigenetic_locks.add_(proven_efficacy, alpha=0.1).clamp_(0.0, 1.0)
        
        # 4. Accumulate Endogenous Plasticity Step:
        plasticity_mask = 1.0 - self.epigenetic_locks
        endogenous_delta_W = -lr_causal * self.causal_credit * plasticity_mask
        self.pending_delta_W.add_(endogenous_delta_W.clamp(-0.02, 0.02))

    @torch.no_grad()
    def apply_pending_causal_updates(self):
        """
        Applies accumulated causal plasticity directly to W_rec after backward().
        """
        self.W_rec.add_(self.pending_delta_W)
        self.pending_delta_W.zero_()


def run_experiment():
    print("==========================================================================")
    print("=== EXP-729: CAUSAL AUTOPOIETIC PLASTICITY (CAP) BENCHMARK ===")
    print("=== Endogenous Causal Traces | Epigenetic Invariant Locking | One-Pass ===")
    print("==========================================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Host CPU"
    print(f"[*] Compute Device: {device} ({device_name})")
    
    # Continuous text stream (Single continuous pass, 4096 bytes)
    stream_text = (
        "Karyon-CoRE v30.0 Sovereign Engineering Protocol. "
        "True autopoiesis requires endogenous causal credit assignment. "
        "Each physical operator tracks its eligibility trace and measures delta Free Energy. "
        "Synapses and channels that causally reduce Free Energy are epigenetically locked against overwriting. "
        "Evolution is never blind random search: the substrate knows why a structural connection succeeded. "
    ) * 16 # ~4096 bytes
    
    raw_bytes = list(stream_text.encode('utf-8'))[:4096]
    seq_tensor = torch.tensor(raw_bytes, dtype=torch.long, device=device).unsqueeze(0)
    
    B, Total_S = seq_tensor.shape
    print(f"[*] Sequence Stream Length S = {Total_S} bytes | Batch B = {B}")
    
    # Initialize CAP Substrate
    model = CausalAutopoieticPlasticityEngine(d_model=256).to(device)
    model.reset_causal_state(device)
    
    # Outer optimizer handles embeddings and readouts; W_rec is updated by endogenous causal autopoiesis
    optimizer = torch.optim.AdamW([
        {"params": model.emb.parameters(), "lr": 2e-3},
        {"params": model.W_in.parameters(), "lr": 2e-3},
        {"params": model.self_predictor.parameters(), "lr": 2e-3},
        {"params": model.endocrine_gen.parameters(), "lr": 2e-3},
        {"params": model.W_g.parameters(), "lr": 2e-3},
        {"params": model.W_alpha.parameters(), "lr": 2e-3},
        {"params": model.head.parameters(), "lr": 2e-3},
    ])
    
    chunk_size = 128
    num_chunks = Total_S // chunk_size
    
    print("\n[*] Processing Continuous Stream with Online Causal Credit & Epigenetic Locking...")
    start_time = time.time()
    
    h_curr = torch.zeros(B, model.d_model, device=device)
    err_prev = torch.zeros(B, device=device)
    
    loss_history = []
    locked_synapses_history = []
    
    for c in range(num_chunks - 1):
        x_chunk = seq_tensor[:, c*chunk_size : (c+1)*chunk_size]
        y_chunk = seq_tensor[:, c*chunk_size + 1 : (c+1)*chunk_size + 1]
        
        optimizer.zero_grad()
        
        chunk_loss = 0.0
        x_emb = model.emb(x_chunk)
        
        for t in range(chunk_size):
            x_t = x_emb[:, t, :]
            y_t = y_chunk[:, t]
            
            step_out = model.forward_step(x_t, h_curr, err_prev)
            logits_t = step_out["logits"]
            loss_t = F.cross_entropy(logits_t, y_t)
            
            # Variational Free Energy F = prediction_error + F_self
            total_F = loss_t.item() + step_out["F_self"].item()
            
            # Accumulate Endogenous Causal Update in pending buffer
            model.accumulate_causal_plasticity(step_out["h_norm"], step_out["dh"], total_F, lr_causal=2e-3)
            
            h_curr = step_out["h_next"]
            err_prev = loss_t.detach().repeat(B)
            chunk_loss += loss_t
            
        avg_chunk_loss = chunk_loss / chunk_size
        avg_chunk_loss.backward()
        optimizer.step()
        
        # Apply accumulated endogenous causal updates to W_rec safely after backward()
        model.apply_pending_causal_updates()
        
        # Detach state for continuous recurrent time flow
        h_curr = h_curr.detach()
        err_prev = err_prev.detach()
        
        current_loss_val = avg_chunk_loss.item()
        loss_history.append(current_loss_val)
        
        locked_ratio = (model.epigenetic_locks > 0.5).float().mean().item() * 100.0
        locked_synapses_history.append(locked_ratio)
        
        time_step = (c + 1) * chunk_size
        ppl = math.exp(min(current_loss_val, 50.0))
        print(f"  Stream t={time_step:04d}/{Total_S} | Loss: {current_loss_val:.4f} nats/byte | PPL: {ppl:.2f} | Epigenetic Locks: {locked_ratio:.2f}%")
        
    total_time = time.time() - start_time
    print(f"\n[+] Stream Run Completed in {total_time:.2f} seconds!")
    
    initial_loss = loss_history[0]
    final_loss = loss_history[-1]
    final_locked_ratio = locked_synapses_history[-1]
    adaptation_delta = initial_loss - final_loss
    
    print(f"\n[*] Causal Autopoiesis Empirical Metrics:")
    print(f"  Initial Stream Loss (t=0128): {initial_loss:.4f} nats/byte | PPL: {math.exp(min(initial_loss, 50.0)):.2f}")
    print(f"  Final Stream Loss   (t=3968): {final_loss:.4f} nats/byte | PPL: {math.exp(min(final_loss, 50.0)):.2f}")
    print(f"  Online Adaptation Delta:     {adaptation_delta:.4f} nats/byte")
    print(f"  Epigenetically Locked Channels: {final_locked_ratio:.2f}% of physical coupling matrix")
    
    baseline_loss = 2.7608 # EXP-728 One-Pass Baseline
    delta_vs_baseline = baseline_loss - final_loss
    print(f"  Delta vs EXP-728 Baseline (2.7608): {delta_vs_baseline:.4f} nats/byte")
    
    verdict = "POSITIVE" if delta_vs_baseline >= 0.08 else ("NEUTRAL" if delta_vs_baseline >= 0 else "REJECTED")
    
    metrics = {
        "loss": final_loss,
        "ppl": math.exp(min(final_loss, 50.0)),
        "initial_loss": initial_loss,
        "adaptation_delta": adaptation_delta,
        "locked_channels_pct": final_locked_ratio,
        "baseline_loss": baseline_loss,
        "delta_loss": delta_vs_baseline,
        "execution_time_sec": total_time,
        "verdict": verdict
    }
    
    with open("experiments/exp_729_results.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"[*] Results saved to experiments/exp_729_results.json")
    return metrics

if __name__ == "__main__":
    run_experiment()
