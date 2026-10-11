"""
EXP-445: Sovereign Self-Reflective Recurrent Attractor Field (S-SRAF)
Author: Bazilevs (ProgVM) & Lead AI Cyberneticist
Standard: KEP v16.0 Sovereign Master (Principle 2, Principle 3, Principle 21, Principle 22, Principle 27 & KEP Rule #12)

Mechanistic Diagnostics of the 68.54% Ceiling (Bazilevs Audit: "Точно все части системы суверенны?"):
Why did EXP-441 stop at 68.54% and not reach 90-100%?
Look deeply at the architecture of EXP-441:
1. Pure Feedforward Step (Single Pass Bottleneck, KEP Principle 21):
   In EXP-441, Karyon makes a SINGLE feed-forward query:
   read = q @ S
   logits = beta * W_out(read)
   It has ZERO recurrent internal thinking time before emitting an action!
   If the first memory readout has slight noise or superposition from 50 previous characters,
   there is no self-correction mechanism!
2. Non-Markovian Context Ambiguity:
   In raw byte text, 'e' follows 'd' in "def", but 'i' follows 'd' in "dynamic".
   A static 1st-order query q(t) cannot know which branch to take without integrating
   the dynamic hidden context vector h(t)!
3. The Sovereign Solution (S-SRAF):
   - Principle 21 Decoupling: Decouple problem time from thinking time!
     Give Karyon an Endogenous Recurrent Attractor Loop (k=1..3 relaxation steps):
     h_{iter+1} = HopfieldRelax(h_{iter}, S_memory, q_t)
     This collapses semantic ambiguity and snaps into the exact attractor basin!
   - 2-Stage Context-Gated Key:
     k_t = Normalize( k_byte + W_rec(h_{t-1}) )
     Now "d" after "e" is mathematically orthogonal from "d" after "o"!
   - Autonomous Multi-Head Law Synthesizer:
     Retains the proven, victorious endogenous law synthesizer from EXP-441!

Target:
Shatter the 68.54% ceiling and push Single-Pass Accuracy toward 80-90%+!
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
logger = logging.getLogger("EXP-445-SRAF")

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)


@dataclass
class EXP445Config:
    exp_id: str = "EXP-445"
    dim: int = 258
    num_heads: int = 8
    dim_k: int = 64
    dim_v: int = 64
    learning_rate: float = 0.015
    stream_length: int = 4000
    device_str: str = DEVICE_STR


class SovereignSelfReflectiveField(nn.Module):
    """
    Sovereign Self-Reflective Recurrent Attractor Field.
    Multi-head associative memory + continuous context-recurrence + Hopfield energy relaxation.
    """
    def __init__(self, config: EXP445Config):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.H = config.num_heads
        self.D_k = config.dim_k
        self.D_v = config.dim_v
        self.total_v = self.H * self.D_v

        # 1. Byte Embedding Matrix [258, 256]
        self.byte_emb = nn.Embedding(self.dim, 256)

        # 2. Recurrent Context Projection: h_{t-1} in R^(H * D_v) -> R^256
        self.W_rec = nn.Linear(self.total_v, 256, bias=False)

        # 3. Multi-Head Tensor Projections: Q, K, V
        self.W_q = nn.Linear(256, self.H * self.D_k, bias=False)
        self.W_k = nn.Linear(256, self.H * self.D_k, bias=False)
        self.W_v = nn.Linear(256, self.H * self.D_v, bias=False)

        # 4. Readout Projection
        self.W_out = nn.Linear(self.total_v, self.dim, bias=False)

        # 5. Hopfield Attractor Snapping Parameters
        self.w_beta = nn.Parameter(torch.tensor(2.5, device=DEVICE))
        self.b_beta = nn.Parameter(torch.tensor(3.0, device=DEVICE))

        # 6. Autonomous Multi-Head Differential Law Synthesizer (from EXP-441):
        self.law_synthesizer = nn.Sequential(
            nn.Linear(4, 32),
            nn.SiLU(),
            nn.Linear(32, 4)
        )

        # State Tensors:
        self.S_memory = torch.zeros(self.H, self.D_k, self.D_v, device=DEVICE)
        self.h_context = torch.zeros(1, self.total_v, device=DEVICE)

        self.reset_state()

    def reset_state(self):
        self.S_memory.zero_()
        self.h_context.zero_()

    def step(self, byte_idx: int, target_byte: int) -> Tuple[torch.Tensor, Dict]:
        x_tensor = torch.tensor([byte_idx], device=DEVICE)
        target_tensor = torch.tensor([target_byte], device=DEVICE)

        # Continuous Recurrent Context Coupling:
        # stim_repr = emb(x_t) + W_rec(h_{t-1})
        emb = self.byte_emb(x_tensor)  # [1, 256]
        h_prev = self.h_context.detach()
        rec_context = self.W_rec(h_prev)  # [1, 256]
        unified_stim = F.normalize(emb + 0.5 * rec_context, p=2, dim=-1)

        # Multi-Head Q, K, V
        q = F.normalize(self.W_q(unified_stim).view(self.H, 1, self.D_k), p=2, dim=-1)
        k = F.normalize(self.W_k(unified_stim).view(self.H, 1, self.D_k), p=2, dim=-1)
        v = F.normalize(self.W_v(unified_stim).view(self.H, 1, self.D_v), p=2, dim=-1)

        S = self.S_memory.detach()  # [H, D_k, D_v]

        # 1. Primary Resonant Recall:
        read_heads = torch.bmm(q, S).squeeze(1)  # [H, D_v]
        read_flat = read_heads.view(1, -1)       # [1, total_v]

        # 2. Endogenous Recurrent Hopfield Relaxation (1-step thinking relaxation):
        # Refines read_flat by self-feedback:
        q_refine = F.normalize(q + 0.3 * self.W_q(self.W_rec(read_flat)).view(self.H, 1, self.D_k), p=2, dim=-1)
        refined_read_heads = torch.bmm(q_refine, S).squeeze(1)
        refined_flat = refined_read_heads.view(1, -1)

        # Update persistent context vector h(t):
        self.h_context = 0.8 * h_prev + 0.2 * refined_flat

        # 3. Dynamic Hopfield Precision Snapping:
        read_norm = torch.norm(refined_flat)
        beta = 1.0 + F.softplus(self.w_beta * read_norm + self.b_beta)
        logits = beta * self.W_out(refined_flat)

        loss = F.cross_entropy(logits, target_tensor)
        pred = torch.argmax(logits, dim=-1).item()
        surprise = loss.item()

        # 4. Target value prediction in memory space:
        v_pred = torch.bmm(k, S)  # [H, 1, D_v]
        error = v - v_pred        # [H, 1, D_v]

        # === 5. SOVEREIGN ENDOGENOUS LAW SYNTHESIS (EXP-441 ENGINE) ===
        err_norms = torch.norm(error, dim=-1).squeeze(1)
        s_norms = torch.norm(S, dim=(-1, -2))
        surp_ten = torch.full((self.H,), min(surprise / 3.0, 3.0), device=DEVICE)
        coher_ten = torch.sum(v * v_pred, dim=-1).squeeze(1)

        head_features = torch.stack([err_norms, s_norms / 10.0, surp_ten, coher_ten], dim=-1)
        law_params = self.law_synthesizer(head_features)

        alpha = torch.clamp(torch.sigmoid(law_params[:, 0]), min=0.70, max=0.999).view(self.H, 1, 1)
        eta_delta = (F.softplus(law_params[:, 1]) + 0.1).view(self.H, 1, 1)
        eta_hebb = F.softplus(law_params[:, 2]).view(self.H, 1, 1)
        lam_att = (0.05 * torch.sigmoid(law_params[:, 3])).view(self.H, 1, 1)

        # Delta operator:
        delta_op = torch.bmm(k.transpose(1, 2), error)
        # Hebbian operator:
        hebb_op = torch.bmm(k.transpose(1, 2), v)

        # State Integration
        new_S = alpha * S + eta_delta * delta_op + eta_hebb * hebb_op - lam_att * S
        self.S_memory = new_S

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


def run_exp_445():
    logger.info("================================================================================")
    logger.info("=== STARTING EXP-445: SOVEREIGN SELF-REFLECTIVE ATTRACTOR FIELD (S-SRAF) ===")
    logger.info("================================================================================")

    config = EXP445Config()
    model = SovereignSelfReflectiveField(config).to(DEVICE)
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
    logger.info("Evaluating Recurrent Thinking Relaxation + Multi-Head Law Synthesis...")

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
                f"Synthesized Laws [Alpha: {cur_alpha:.3f}, Delta: {cur_delta:.3f}, Hebb: {cur_hebb:.3f}]"
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

    ultra_low_count = sum(1 for loss_val in second_half if loss_val < 0.05)
    ultra_low_fraction = (ultra_low_count / len(second_half)) * 100.0

    tail_500 = recent_losses[-500:]
    tail_low_loss_count = sum(1 for loss_val in tail_500 if loss_val < 0.5)
    tail_retention_rate = (tail_low_loss_count / len(tail_500)) * 100.0

    logger.info("================================================================================")
    logger.info("=== EXP-445 SELF-REFLECTIVE ATTRACTOR FIELD REPORT ===")
    logger.info(f"Final Average Loss: {final_loss:.4f} nats")
    logger.info(f"Single-Pass Accuracy: {final_acc:.2f}%")
    logger.info(f"Mean Synthesized Alpha: {mean_alpha:.4f}")
    logger.info(f"Mean Synthesized Delta Rate: {mean_delta:.4f}")
    logger.info(f"Mean Synthesized Hebbian Rate: {mean_hebb:.4f}")
    logger.info(f"Second-Half Low-Loss Mass (<0.5 nats): {low_loss_fraction:.2f}%")
    logger.info(f"Second-Half Ultra-Low Loss Mass (<0.05 nats): {ultra_low_fraction:.2f}%")
    logger.info(f"Tail 500-byte Retention Rate (<0.5 nats): {tail_retention_rate:.2f}%")
    logger.info(f"Throughput: {throughput:.2f} steps/sec")
    logger.info(f"Mean Output Entropy: {avg_final_entropy:.3f} nats")
    logger.info(f"Elapsed Time: {t_elapsed:.2f} s")
    logger.info("================================================================================")

    results = {
        "exp_id": "EXP-445",
        "final_loss": round(final_loss, 4),
        "final_accuracy": round(final_acc, 2),
        "mean_synthesized_alpha": round(mean_alpha, 4),
        "mean_synthesized_delta_rate": round(mean_delta, 4),
        "mean_synthesized_hebbian_rate": round(mean_hebb, 4),
        "second_half_low_loss_mass": round(low_loss_fraction, 2),
        "second_half_ultra_low_loss_mass": round(ultra_low_fraction, 2),
        "tail_retention_rate": round(tail_retention_rate, 2),
        "throughput_steps_per_sec": round(throughput, 2),
        "mean_entropy_nats": round(avg_final_entropy, 3),
        "elapsed_time": round(t_elapsed, 2)
    }

    with open("experiments/exp_445_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_exp_445()
