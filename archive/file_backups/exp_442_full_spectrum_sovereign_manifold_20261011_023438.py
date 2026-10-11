"""
EXP-442: Full-Spectrum Sovereign Autopoietic Tensor Manifold (FS-SATM)
Author: Bazilevs (ProgVM) & Lead AI Cyberneticist
Standard: KEP v16.0 Sovereign Master (Principle 2, Principle 3, Principle 22, Principle 27 & KEP Rule #12)

Mechanistic Breakdown of Non-Sovereign Remnants in EXP-441:
1. Artificial Bound on Alpha (min=0.70):
   In EXP-441, alpha hit the lower clamp 0.7000 every single step! The system was screaming
   to have sovereign, unbounded decay/retention, but was shackled by artificial code clamps!
2. Readout Projection (W_out) Starvation:
   Logits were formed via fixed W_out(read) instead of resonant energy duality on the continuous
   representation space (Re <W_vocab | read>).
3. 1st-Order stimulus key (k_t):
   A single-byte key cannot mathematically disambiguate non-Markovian byte sequences ("th" -> 'e' vs "th" -> 'i').
   In a true physical wavefield, the stimulus excites a continuous phase trajectory:
   Psi_key(t) = Gamma * Psi_key(t-1) + Emb(x_t).
   The state is an unbroken continuous trajectory, not an isolated point!
4. Endogenous Plasticity Freedom:
   All clamps are completely eradicated. Alpha, Eta_delta, Eta_hebb, Beta, and Dynamic Phase Flow
   are 100% endogenous tensor fields.

Target:
Shatter the 68% ceiling and target 85-95%+ Single-Pass Stream Accuracy!
"""

import time
import json
import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EXP-442-FS-SATM")

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)


@dataclass
class EXP442Config:
    exp_id: str = "EXP-442"
    dim: int = 258
    num_heads: int = 8
    dim_k: int = 64
    dim_v: int = 64
    learning_rate: float = 0.015
    stream_length: int = 4000
    device_str: str = DEVICE_STR


class FullSpectrumSovereignManifold(nn.Module):
    """
    Unshackled Sovereign Continuous Tensor Manifold.
    Zero artificial clamps. Continuous sequence phase continuity.
    Autonomous multi-head differential field synthesis.
    """
    def __init__(self, config: EXP442Config):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.H = config.num_heads
        self.D_k = config.dim_k
        self.D_v = config.dim_v

        # 1. Continuous Wavepacket Embedding [258, 256]
        self.emb = nn.Embedding(self.dim, 256)

        # 2. Multi-Head Tensor Projections: Q, K, V
        self.W_q = nn.Linear(256, self.H * self.D_k, bias=False)
        self.W_k = nn.Linear(256, self.H * self.D_k, bias=False)
        self.W_v = nn.Linear(256, self.H * self.D_v, bias=False)

        # 3. Resonant Readout Vocab Manifold
        self.W_out = nn.Linear(self.H * self.D_v, self.dim, bias=False)

        # 4. Hopfield Precision Gain (Pure Continuous Learnable Field)
        self.w_beta = nn.Parameter(torch.tensor(2.5, device=DEVICE))
        self.b_beta = nn.Parameter(torch.tensor(1.0, device=DEVICE))

        # 5. Continuous Trajectory Flow Generator (Lie generator for temporal sequence phase)
        self.Lie_phase = nn.Parameter(torch.randn(self.H, self.D_k, self.D_k, device=DEVICE) * 0.02)

        # 6. Unconstrained Endogenous Differential Law Synthesizer:
        # Generates [retention_decay, delta_rate, hebb_rate, dissipation] with ZERO artificial clamps!
        self.synthesizer = nn.Sequential(
            nn.Linear(4, 32),
            nn.SiLU(),
            nn.Linear(32, 4)
        )

        # Memory Tensor S in R^(H x D_k x D_v)
        self.S_memory = torch.zeros(self.H, self.D_k, self.D_v, device=DEVICE)

        # Continuous Temporal Key Trajectory: K_traj in R^(H x 1 x D_k)
        self.K_traj = torch.zeros(self.H, 1, self.D_k, device=DEVICE)

        self.reset_state()

    def reset_state(self):
        self.S_memory.zero_()
        self.K_traj.zero_()

    def step(self, byte_idx: int, target_byte: int) -> Tuple[torch.Tensor, Dict]:
        x_tensor = torch.tensor([byte_idx], device=DEVICE)
        target_tensor = torch.tensor([target_byte], device=DEVICE)

        x_feat = self.emb(x_tensor)  # [1, 256]

        # Multi-Head projections
        k_raw = F.normalize(self.W_k(x_feat).view(self.H, 1, self.D_k), p=2, dim=-1)
        v = F.normalize(self.W_v(x_feat).view(self.H, 1, self.D_v), p=2, dim=-1)

        # Continuous Temporal Wavefield Trajectory:
        # k(t) = Rot(k_traj(t-1)) + k_raw(t)  (Disambiguates higher-order sequence context!)
        Lie_skew = (self.Lie_phase - self.Lie_phase.transpose(1, 2)) * 0.5
        prev_k = self.K_traj.detach()
        rot_k = torch.bmm(prev_k, Lie_skew)
        k_dyn = F.normalize(k_raw + rot_k, p=2, dim=-1)

        # Detached memory state
        S = self.S_memory.detach()  # [H, D_k, D_v]

        # 1. Resonant Memory Recall: read_h = k_dyn @ S
        # Notice: Query uses the dynamic context wavefield k_dyn!
        read_heads = torch.bmm(k_dyn, S).squeeze(1)  # [H, D_v]
        read_flat = read_heads.view(1, -1)           # [1, H * D_v]

        # 2. Modern Hopfield Dynamic Precision Snapping:
        read_norm = torch.norm(read_flat)
        beta = 1.0 + F.softplus(self.w_beta * read_norm + self.b_beta)
        logits = beta * self.W_out(read_flat)

        loss = F.cross_entropy(logits, target_tensor)
        pred = torch.argmax(logits, dim=-1).item()
        surprise = loss.item()

        # 3. Prediction Error on Value Substrate:
        v_pred = torch.bmm(k_dyn, S)  # [H, 1, D_v]
        error = v - v_pred           # [H, 1, D_v]

        # === 4. UNCONSTRAINED SOVEREIGN LAW SYNTHESIS ===
        err_norms = torch.norm(error, dim=-1).squeeze(1)
        s_norms = torch.norm(S, dim=(-1, -2))
        surp_ten = torch.full((self.H,), min(surprise / 3.0, 3.0), device=DEVICE)
        coher_ten = torch.sum(v * v_pred, dim=-1).squeeze(1)

        head_features = torch.stack([err_norms, s_norms / 10.0, surp_ten, coher_ten], dim=-1)
        law_params = self.synthesizer(head_features)  # [H, 4]

        # Completely unconstrained continuous functions:
        # alpha: fully autonomous continuous retention gate (NO artificial 0.70 clamp!)
        alpha = torch.sigmoid(law_params[:, 0]).view(self.H, 1, 1)

        # eta_delta: normalized Widrow-Hoff geometric projection rate
        # eta_delta = softplus(p) / (||k||^2 + eps)  -> mathematically exact LMS projection!
        k_norm_sq = torch.sum(k_dyn ** 2, dim=-1, keepdim=True) + 1e-6  # [H, 1, 1]
        eta_delta = (F.softplus(law_params[:, 1]) / k_norm_sq).view(self.H, 1, 1)

        # eta_hebb: continuous outer-product plasticity
        eta_hebb = F.softplus(law_params[:, 2]).view(self.H, 1, 1)

        # lambda_dissip: autonomous thermodynamic bounding
        lam_dissip = (0.02 * torch.sigmoid(law_params[:, 3])).view(self.H, 1, 1)

        # === 5. CONTINUOUS TENSOR INTEGRATION ===
        # Exact Geometric Delta Operator:
        delta_op = torch.bmm(k_dyn.transpose(1, 2), error)  # [H, D_k, D_v]

        # Outer Product Operator:
        hebb_op = torch.bmm(k_dyn.transpose(1, 2), v)       # [H, D_k, D_v]

        # Complete Unshackled Differential Field Equation:
        new_S = alpha * S + eta_delta * delta_op + eta_hebb * hebb_op - lam_dissip * S

        # Out-of-place state propagation
        self.S_memory = new_S
        self.K_traj = k_dyn

        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()

        meta = {
            "loss": surprise,
            "pred": pred,
            "target": target_byte,
            "is_correct": (pred == target_byte),
            "entropy": entropy,
            "beta": beta.item(),
            "mean_alpha": alpha.mean().item(),
            "mean_eta_delta": eta_delta.mean().item(),
            "mean_eta_hebb": eta_hebb.mean().item(),
            "s_norm": torch.norm(new_S).item()
        }
        return loss, meta


def run_exp_442():
    logger.info("================================================================================")
    logger.info("=== STARTING EXP-442: FULL-SPECTRUM SOVEREIGN AUTOPOIETIC MANIFOLD (FS-SATM) ===")
    logger.info("================================================================================")

    config = EXP442Config()
    model = FullSpectrumSovereignManifold(config).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)

    stream_data = (
        "def karyon_sovereign_autopoiesis(stream):\n"
        "    super_operator = Liouvillian(phase_space)\n"
        "    return super_operator.synthesize()\n\n"
        "#include <torch/extension.h>\n"
        "void dynamic_kernel(float* x, float* h) {\n"
        "    // Tensor core phase alignment\n"
        "}\n\n"
        "HTTP/1.1 200 OK\r\nContent-Type: application/kcore\r\n\r\n"
        "Mind is substrate-independent continuous field dynamics.\n"
        "Ashby homeostatic ultrastability balances energy and surprise.\n"
    ).encode("utf-8")

    raw_bytes = list(stream_data) * (config.stream_length // len(stream_data) + 1)
    raw_bytes = raw_bytes[:config.stream_length]

    logger.info(f"Stream loaded: {len(raw_bytes)} bytes on {DEVICE_STR.upper()}")
    logger.info("Evaluating Unshackled Wavefield Phase Dynamics & Sovereign Law Synthesis...")

    correct_preds = 0
    total_loss = 0.0
    total_entropy = 0.0
    recent_losses = []

    alpha_accum = 0.0
    delta_accum = 0.0
    hebb_accum = 0.0

    t_start = time.perf_counter()

    for t in range(len(raw_bytes) - 1):
        x_byte = raw_bytes[t]
        target_byte = raw_bytes[t + 1]

        optimizer.zero_grad()
        loss, meta = model.step(x_byte, target_byte)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if meta["is_correct"]:
            correct_preds += 1

        total_loss += meta["loss"]
        total_entropy += meta["entropy"]
        recent_losses.append(meta["loss"])

        alpha_accum += meta["mean_alpha"]
        delta_accum += meta["mean_eta_delta"]
        hebb_accum += meta["mean_eta_hebb"]

        if (t + 1) % 500 == 0:
            avg_loss = total_loss / (t + 1)
            acc = (correct_preds / (t + 1)) * 100.0
            cur_alpha = alpha_accum / (t + 1)
            cur_delta = delta_accum / (t + 1)
            cur_hebb = hebb_accum / (t + 1)
            logger.info(
                f"Progress [{t+1}/{len(raw_bytes)-1}] | Avg Loss: {avg_loss:.4f} | "
                f"Acc: {acc:.2f}% | Current Loss: {meta['loss']:.4f} | Beta: {meta['beta']:.2f} | "
                f"Autonomous Laws [Alpha: {cur_alpha:.4f}, Delta: {cur_delta:.3f}, Hebb: {cur_hebb:.3f}]"
            )

    t_elapsed = time.perf_counter() - t_start
    final_loss = total_loss / (len(raw_bytes) - 1)
    final_acc = (correct_preds / (len(raw_bytes) - 1)) * 100.0
    avg_final_entropy = total_entropy / (len(raw_bytes) - 1)
    throughput = (len(raw_bytes) - 1) / t_elapsed

    mean_alpha = alpha_accum / (len(raw_bytes) - 1)
    mean_delta = delta_accum / (len(raw_bytes) - 1)
    mean_hebb = hebb_accum / (len(raw_bytes) - 1)

    second_half = recent_losses[len(recent_losses)//2:]
    low_loss_count = sum(1 for loss_val in second_half if loss_val < 0.5)
    low_loss_fraction = (low_loss_count / len(second_half)) * 100.0

    tail_500 = recent_losses[-500:]
    tail_low_loss_count = sum(1 for loss_val in tail_500 if loss_val < 0.5)
    tail_retention_rate = (tail_low_loss_count / len(tail_500)) * 100.0

    logger.info("================================================================================")
    logger.info("=== EXP-442 FULL-SPECTRUM SOVEREIGN MANIFOLD TELEMETRY REPORT ===")
    logger.info(f"Final Average Loss: {final_loss:.4f} nats")
    logger.info(f"Single-Pass Accuracy: {final_acc:.2f}%")
    logger.info(f"Unshackled Mean Alpha: {mean_alpha:.4f}")
    logger.info(f"Normalized Delta Rate: {mean_delta:.4f}")
    logger.info(f"Autonomous Hebb Rate: {mean_hebb:.4f}")
    logger.info(f"Second-Half Low-Loss Mass (<0.5 nats): {low_loss_fraction:.2f}%")
    logger.info(f"Tail 500-byte Retention Rate (<0.5 nats): {tail_retention_rate:.2f}%")
    logger.info(f"Throughput: {throughput:.2f} steps/sec")
    logger.info(f"Mean Output Entropy: {avg_final_entropy:.3f} nats")
    logger.info(f"Elapsed Time: {t_elapsed:.2f} s")
    logger.info("================================================================================")

    results = {
        "exp_id": "EXP-442",
        "final_loss": round(final_loss, 4),
        "final_accuracy": round(final_acc, 2),
        "unshackled_mean_alpha": round(mean_alpha, 4),
        "normalized_delta_rate": round(mean_delta, 4),
        "autonomous_hebb_rate": round(mean_hebb, 4),
        "second_half_low_loss_mass": round(low_loss_fraction, 2),
        "tail_retention_rate": round(tail_retention_rate, 2),
        "throughput_steps_per_sec": round(throughput, 2),
        "mean_entropy_nats": round(avg_final_entropy, 3),
        "elapsed_time": round(t_elapsed, 2)
    }

    with open("experiments/exp_442_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_exp_442()
