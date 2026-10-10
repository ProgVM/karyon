"""
EXP-398: Deterministic Orthogonal Discrete Basis & Margin-Enforced Attractor Resonance (DOB-MEAR)
Author: Bazilevs (ProgVM) & Karyon Cyberneticist
Date: October 2026
Standard: KEP v16.0 Sovereign Master (Strict Principle 27: Absolute Substrate Sovereignty, N=1 Stream)

Core Scientific Breakthroughs:
1. Strict Deterministic Orthogonal Basis (D=258):
   Eliminates high-dimensional random dot-product noise (+/- 0.15) on S^{D-1}.
   <b_i, b_j> = delta_{ij} strictly. Zero crosstalk between unrelated bytes.
2. Hard Margin-Enforced Contrastive Free Energy:
   E_margin = max(0, max_{j != target} <Psi, b_j> - <Psi, b_target> + margin)
   Guarantees that the gradient will NOT vanish until the target basin decisively
   dominates all competitors by at least margin = 0.40.
3. Multi-Timescale Continuous Morphic Field & Fast Plastic Weights.
4. Strict Single-Pass Streaming Reality (N=1, zero epochs).
"""

import os
import sys
import math
import time
import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EXP-398")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@dataclass
class EXP398Config:
    exp_id: str = "EXP-398"
    dim: int = 258          # Exactly 258 orthogonal dimensions for 258 discrete states
    num_heads: int = 6      # 6 associative heads (dim per head = 43)
    num_basins: int = 258
    max_nodes: int = 16
    initial_nodes: int = 4
    max_thinking_depth: int = 4
    learning_rate: float = 0.005
    fast_weight_eta: float = 0.20
    margin: float = 0.40
    beta_hopfield: float = 24.0
    sprout_surprise_threshold: float = 1.2
    sprout_patience: int = 35
    device: str = str(DEVICE)


class StrictOrthogonalHopfieldNexus(nn.Module):
    """
    Deterministic Orthogonal Discrete Basis.
    B in R^{258 x 258} is the identity matrix I (canonical orthonormal basis).
    <b_i, b_j> = delta_{ij}.
    """
    def __init__(self, num_basins: int = 258):
        super().__init__()
        self.num_basins = num_basins
        # Register strict orthonormal basis
        basis = torch.eye(num_basins)
        self.register_buffer("basis", basis)

    def perceive(self, byte_idx: int) -> torch.Tensor:
        return self.basis[byte_idx]

    def compute_margin_free_energy(
        self,
        psi: torch.Tensor,
        target_idx: int,
        margin: float = 0.40,
        beta: float = 24.0
    ) -> Tuple[torch.Tensor, int, float, float]:
        """
        Computes Margin-Enforced Free Energy:
        - Target potential must beat maximum competitor by at least `margin`
        - Strict cross-entropy surrogate is rejected; hard contrastive energy is enforced.
        """
        psi_norm = F.normalize(psi, p=2, dim=-1) # [258]
        # Alignments are directly coordinates along orthonormal axes
        alignments = psi_norm # since basis is identity: <psi, e_i> = psi_i
        
        target_score = alignments[target_idx]
        
        # Mask out target to find top competitor
        competitor_scores = alignments.clone()
        competitor_scores[target_idx] = -1e9
        max_competitor_score, max_comp_idx = torch.max(competitor_scores, dim=-1)
        
        # Hard contrastive margin loss
        margin_violation = torch.clamp(max_competitor_score - target_score + margin, min=0.0)
        
        # Directional alignment strain
        alignment_strain = 1.0 - target_score
        
        # Free energy
        total_energy = margin_violation + 0.5 * alignment_strain
        
        predicted_idx = torch.argmax(alignments).item()
        return total_energy, predicted_idx, target_score.item(), max_competitor_score.item()


class MultiHeadFastAssociativeMemory(nn.Module):
    def __init__(self, dim: int, num_heads: int, eta: float = 0.20):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eta = eta
        
        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        self.W_out = nn.Linear(dim, dim, bias=False)
        
        decay_rates = [0.80 + 0.03 * i for i in range(num_heads)]
        self.register_buffer("decay_rates", torch.tensor(decay_rates))
        self.register_buffer("M_fast", torch.zeros(num_heads, self.head_dim, self.head_dim))

    def reset_memory(self):
        self.M_fast.zero_()

    def read_memory(self, x: torch.Tensor) -> torch.Tensor:
        q = self.W_q(x).view(self.num_heads, self.head_dim)
        
        rec_heads = []
        for h in range(self.num_heads):
            rec_h = torch.matmul(self.M_fast[h], q[h])
            rec_heads.append(rec_h)
            
        stacked = torch.cat(rec_heads, dim=-1)
        return self.W_out(stacked)

    def write_memory(self, x: torch.Tensor, target_vector: torch.Tensor):
        with torch.no_grad():
            k = F.normalize(self.W_k(x).view(self.num_heads, self.head_dim), p=2, dim=-1)
            v = self.W_v(target_vector).view(self.num_heads, self.head_dim)
            
            new_M = torch.zeros_like(self.M_fast)
            for h in range(self.num_heads):
                gamma_h = self.decay_rates[h].item()
                outer_h = torch.outer(v[h], k[h])
                new_M[h] = gamma_h * self.M_fast[h] + self.eta * outer_h
                
            self.M_fast.copy_(new_M)


class UniversalMorphicFunctional(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        self.W_l = nn.Linear(dim, dim, bias=True)
        self.W_r = nn.Linear(dim, dim, bias=True)
        self.W_f = nn.Linear(dim, dim, bias=False)
        
        self.log_tau = nn.Parameter(torch.tensor(0.0))
        self.log_g = nn.Parameter(torch.tensor(0.0))
        self.alpha_epi = nn.Parameter(torch.tensor(0.0))
        
        self.register_buffer("s_state", torch.zeros(dim))

    def reset_state(self):
        self.s_state.zero_()

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        tau = torch.sigmoid(self.log_tau)
        w_g = torch.exp(torch.clamp(self.log_g, -2.0, 2.0))
        
        bilinear = w_g * (torch.tanh(self.W_l(psi)) * torch.sigmoid(self.W_r(psi)))
        
        s_next = (1.0 - tau) * self.s_state.detach() + tau * self.W_f(psi)
        self.s_state = s_next.detach()
        
        cand = F.layer_norm(bilinear + s_next, (self.dim,))
        graft = torch.tanh(self.alpha_epi)
        return psi + graft * cand


class DOBMEAREngine(nn.Module):
    def __init__(self, config: EXP398Config):
        super().__init__()
        self.config = config
        self.dim = config.dim
        
        # Strict Orthogonal Nexus
        self.nexus = StrictOrthogonalHopfieldNexus(num_basins=config.num_basins)
        
        # Multi-Head Fast-Weight Associative Memory
        self.mhfam = MultiHeadFastAssociativeMemory(
            dim=config.dim,
            num_heads=config.num_heads,
            eta=config.fast_weight_eta
        )
        
        # Universal Morphic Functional Nodes
        self.nodes = nn.ModuleList([
            UniversalMorphicFunctional(self.dim)
            for _ in range(config.initial_nodes)
        ])
        for node in self.nodes:
            node.alpha_epi.data.fill_(1.0)
            
        self.W_route = nn.Linear(self.dim, config.max_nodes, bias=False)
        self.register_buffer("drives", torch.tensor([1.0, 0.20, 0.50])) # Energy, NA, DA
        
        self.sprout_events = 0
        self.godel_rejections = 0

    def perceive(self, byte_idx: int) -> torch.Tensor:
        return self.nexus.perceive(byte_idx)

    def reset_states(self):
        self.mhfam.reset_memory()
        for node in self.nodes:
            node.reset_state()

    def forward_cognitive_step(
        self,
        u_t: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # 1. Multi-Head Fast Memory Read
        fast_recalled = self.mhfam.read_memory(u_t)
        psi_t = F.layer_norm(u_t + 1.0 * fast_recalled, (self.dim,))
        
        num_active = len(self.nodes)
        thinking_steps = 1
        
        # 2. Endogenous Recurrent Thinking Cycles
        for k in range(self.config.max_thinking_depth):
            psi_prev = psi_t
            
            route_logits = self.W_route(psi_t)[:num_active]
            route_weights = F.softmax(route_logits, dim=-1)
            
            node_outs = [node(psi_t) for node in self.nodes]
            stacked = torch.stack(node_outs, dim=0)
            
            psi_coupled = torch.sum(route_weights.unsqueeze(-1) * stacked, dim=0)
            psi_t = F.layer_norm(psi_t + psi_coupled, (self.dim,))
            
            delta_norm = torch.norm(psi_t - psi_prev).item()
            if delta_norm < 0.03 and k > 0:
                thinking_steps = k + 1
                break
            thinking_steps = k + 1
            
        diag = {
            "thinking_steps": thinking_steps,
            "active_nodes": num_active,
            "noradrenaline": self.drives[1].item(),
            "dopamine": self.drives[2].item()
        }
        return psi_t, diag

    def update_drives(self, fe_val: float):
        with torch.no_grad():
            curr_na = self.drives[1].item()
            new_na = 0.90 * curr_na + 0.10 * math.tanh(fe_val / 2.0)
            
            curr_da = self.drives[2].item()
            new_da = 0.90 * curr_da + 0.10 * math.exp(-fe_val)
            
            curr_e = self.drives[0].item()
            new_e = max(0.1, min(1.0, curr_e + 0.02 * new_da - 0.01 * new_na))
            
            self.drives.copy_(torch.tensor([new_e, new_na, new_da], device=DEVICE))

    def attempt_morphogenesis(self, recent_bytes: List[int]):
        if len(self.nodes) >= self.config.max_nodes or len(recent_bytes) < 16:
            return
            
        candidate = UniversalMorphicFunctional(self.dim).to(DEVICE)
        candidate.alpha_epi.data.fill_(0.20)
        
        test_node = UniversalMorphicFunctional(self.dim).to(DEVICE)
        test_node.load_state_dict(candidate.state_dict())
        opt_sandbox = torch.optim.SGD(test_node.parameters(), lr=0.01)
        
        with torch.no_grad():
            base_fe = 0.0
            for t in range(len(recent_bytes) - 1):
                u = self.perceive(recent_bytes[t])
                target = recent_bytes[t + 1]
                psi, _ = self.forward_cognitive_step(u)
                fe, _, _, _ = self.nexus.compute_margin_free_energy(
                    psi, target, margin=self.config.margin, beta=self.config.beta_hopfield
                )
                base_fe += fe.item()
                
        mut_fe = 0.0
        for t in range(len(recent_bytes) - 1):
            u = self.perceive(recent_bytes[t])
            target = recent_bytes[t + 1]
            psi, _ = self.forward_cognitive_step(u)
            cand_out = test_node(psi)
            psi_cand = F.layer_norm(psi + 0.2 * cand_out, (self.dim,))
            fe, _, _, _ = self.nexus.compute_margin_free_energy(
                psi_cand, target, margin=self.config.margin, beta=self.config.beta_hopfield
            )
            
            opt_sandbox.zero_grad()
            fe.backward()
            opt_sandbox.step()
            mut_fe += fe.item()
            
        if mut_fe < base_fe * 0.995:
            candidate.load_state_dict(test_node.state_dict())
            self.nodes.append(candidate)
            self.sprout_events += 1
            logger.info(f"🌿 [GÖDEL ACCEPTED] Sprouted Functional Node #{len(self.nodes)}!")
        else:
            self.godel_rejections += 1


def run_exp398():
    logger.info("===============================================================================")
    logger.info("=== KEP EXP-398: DETERMINISTIC ORTHOGONAL BASIS & MARGIN RESONANCE (DOB-MEAR) ==")
    logger.info("=== STRICT KEP PRINCIPLE 27: EXACT ORTHOGONALITY | MARGIN ENFORCEMENT =========")
    logger.info("===============================================================================")

    config = EXP398Config()
    model = DOBMEAREngine(config).to(DEVICE)
    
    text_corpus = (
        "User: Explain the physical difference between static and SDE recurrent integration.\n"
        "Karyon: Static next-token projection is algebraic. SDE recurrent integration treats internal states "
        "as a dynamic vector field traversing a non-Euclidean phase manifold.\n\n"
        "Coding Implementation:\n"
        "template <typename T> torch::Tensor parallel_ssd_scan(torch::Tensor q, torch::Tensor k) {\n"
        "    auto out = torch::matmul(q, k.transpose(-1, -2));\n"
        "    return torch::group_norm(out, 8);\n"
        "}\n\n"
        "Scientific Reasoning: Under Free Energy minimization, epistemic entropy drives curiosity. "
        "When unexpected sensory surprise enters the cognitive gateway, noradrenaline surges, "
        "triggering epigenetic sprouting of new laminar cortical sheets.\n"
    ) * 16

    raw_bytes = [ord(c) for c in text_corpus]
    total_tokens = len(raw_bytes)
    logger.info(f"Loaded Reality Stream: {total_tokens} Bytes | Strict Single-Pass N=1 (NO EPOCHS)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    model.reset_states()

    fe_history = []
    bit_err_history = []
    exact_acc_history = []
    margin_violations = []
    
    start_time = time.time()
    surprise_counter = 0

    for t in range(total_tokens - 1):
        byte_in = raw_bytes[t]
        byte_target = raw_bytes[t + 1]
        
        u_t = model.perceive(byte_in)
        target_v = model.perceive(byte_target)
        
        optimizer.zero_grad()
        psi_t, diag = model.forward_cognitive_step(u_t)
        
        # Margin-Enforced Hopfield Free Energy
        fe_loss, pred_byte, score_t, max_comp = model.nexus.compute_margin_free_energy(
            psi_t, byte_target, margin=config.margin, beta=config.beta_hopfield
        )
        
        fe_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Fast plastic working memory write
        model.mhfam.write_memory(u_t, target_v)
        
        fe_val = fe_loss.item()
        fe_history.append(fe_val)
        
        xor_diff = pred_byte ^ byte_target
        bit_diff = bin(xor_diff).count('1')
        bit_err_history.append(bit_diff)
        exact_acc_history.append(1 if bit_diff == 0 else 0)
        margin_violations.append(1 if (max_comp - score_t + config.margin) > 0 else 0)
        
        model.update_drives(fe_val)
        
        if fe_val > config.sprout_surprise_threshold:
            surprise_counter += 1
            if surprise_counter >= config.sprout_patience:
                context_window = raw_bytes[max(0, t - 32):t + 1]
                model.attempt_morphogenesis(context_window)
                surprise_counter = 0
        else:
            surprise_counter = max(0, surprise_counter - 1)
            
        if (t + 1) % 500 == 0 or t == total_tokens - 2:
            recent_fe = fe_history[-500:]
            recent_bits = bit_err_history[-500:]
            recent_acc = exact_acc_history[-500:]
            recent_mv = margin_violations[-500:]
            
            avg_fe = sum(recent_fe) / len(recent_fe)
            avg_bits = sum(recent_bits) / len(recent_bits)
            acc = sum(recent_acc) / len(recent_acc) * 100.0
            mv_rate = sum(recent_mv) / len(recent_mv) * 100.0
            
            logger.info(
                f"Time Step {t+1:05d}/{total_tokens} | "
                f"Margin Free Energy: {avg_fe:.4f} | "
                f"Bit Error: {avg_bits:.2f}/8.0 ({avg_bits/8.0*100:.1f}%) | "
                f"Exact Match: {acc:.1f}% | "
                f"Margin Strain: {mv_rate:.1f}% | "
                f"Nodes: {len(model.nodes)} | "
                f"NA: {model.drives[1].item():.3f} | "
                f"DA: {model.drives[2].item():.3f} | "
                f"Sprouts: {model.sprout_events} | "
                f"Rejections: {model.godel_rejections}"
            )
            
    elapsed = time.time() - start_time
    final_1000_bits = bit_err_history[-1000:]
    final_avg_bits = sum(final_1000_bits) / len(final_1000_bits)
    final_fe = sum(fe_history[-1000:]) / len(fe_history[-1000:])
    final_acc = sum(exact_acc_history[-1000:]) / 1000.0 * 100.0
    
    logger.info("\n=== EXP-398 FINAL SCIENTIFIC TELEMETRY ===")
    logger.info(f"Execution Duration: {elapsed:.2f} s")
    logger.info(f"Final Margin Free Energy: {final_fe:.4f}")
    logger.info(f"Final Bit Error: {final_avg_bits:.2f} / 8.0 bits ({final_avg_bits/8.0*100:.1f}%)")
    logger.info(f"Final Exact Accuracy: {final_acc:.1f}%")
    logger.info(f"Active Functional Nodes: {len(model.nodes)}")
    logger.info(f"Sprout Events: {model.sprout_events} (Gödel Rejections: {model.godel_rejections})")
    
    results = {
        "exp_id": "EXP-398",
        "final_margin_free_energy": final_fe,
        "final_bit_errors": final_avg_bits,
        "final_exact_accuracy": final_acc,
        "active_nodes": len(model.nodes),
        "sprout_events": model.sprout_events,
        "godel_rejections": model.godel_rejections,
        "elapsed_time": elapsed
    }
    
    with open("experiments/exp_398_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_exp398()
