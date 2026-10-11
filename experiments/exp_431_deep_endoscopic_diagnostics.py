"""
EXP-431: Deep Mechanistic Endoscopic Diagnostic Engine
Author: Bazilevs (ProgVM) & Lead AI Cyberneticist
Standard: KEP Principle 23 (Endoscopic Internal State Telemetry & Deep Mechanistic Audit)

Objective:
Perform an exhaustive, multi-scale, micro-probing diagnostic on the entire continuous
single-pass architecture in real-time across 4,000 stream bytes.

Audited Subsystems & Diagnostics:
1. Spectral Health & Rank of Associative Matrices S_fast and S_slow:
   - SVD spectrum, condition number, Effective Rank (Participation Ratio PR = (sum s_i)^2 / sum s_i^2).
   - Trace saturation, norm drift, eigenvalue decay.
2. Geometry of Attractor-Repeller Field:
   - Cosine overlap between W_att readout and W_rep fields.
   - Force cancellation check: is repulsion colliding with valid concept retrieval?
   - Beta and Gamma scaling trajectories over time.
3. Gradient Flow & Layer Sensitivity Audit:
   - Norm of gradients across byte_emb, W_q, W_k, W_v, W_att, W_rep, and beta/gamma params.
   - Vanishing or exploding gradient detection.
4. Error Anatomy & Micro-Token Typology:
   - Are errors occurring at Word Boundaries, Code Punctuation, Whitespace, or Repeated Tokens?
   - Loss distribution by byte frequency and byte category (alphanumeric vs control vs symbols).
   - "Cold start" vs "Interference" vs "Capacity limit" breakdown.
"""

import time
import json
import logging
from dataclasses import dataclass
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EXP-431-ENDOSCOPY")

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)


@dataclass
class EXP431Config:
    exp_id: str = "EXP-431"
    dim: int = 258
    num_heads: int = 8
    dim_k: int = 64
    dim_v: int = 64
    num_repellers: int = 32
    learning_rate: float = 0.015
    sleep_interval: int = 400
    sleep_steps: int = 6
    sleep_batch_size: int = 32
    stream_length: int = 4000
    eta_fast: float = 0.95
    eta_slow: float = 0.50
    gamma_rep: float = 0.40
    rep_decay: float = 0.90
    device_str: str = DEVICE_STR


class EndoscopicDiagnosticCore(nn.Module):
    def __init__(self, config: EXP431Config):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.num_heads = config.num_heads
        self.dim_k = config.dim_k
        self.dim_v = config.dim_v

        self.byte_emb = nn.Embedding(self.dim, 256)
        self.W_q = nn.Linear(256, self.num_heads * self.dim_k, bias=False)
        self.W_k = nn.Linear(256, self.num_heads * self.dim_k, bias=False)
        self.W_v = nn.Linear(256, self.num_heads * self.dim_v, bias=False)

        self.W_att = nn.Linear(2 * self.num_heads * self.dim_v, self.dim, bias=False)
        self.W_rep = nn.Linear(2 * self.num_heads * self.dim_v, self.dim, bias=False)

        self.w_beta = nn.Parameter(torch.tensor(1.5, device=DEVICE))
        self.b_beta = nn.Parameter(torch.tensor(2.5, device=DEVICE))
        self.w_gamma = nn.Parameter(torch.tensor(0.8, device=DEVICE))

        self.fast_decay_params = nn.Parameter(torch.linspace(0.85, 2.2, self.num_heads, device=DEVICE))
        self.slow_decay_params = nn.Parameter(torch.linspace(4.6, 7.6, self.num_heads, device=DEVICE))

        self.S_fast = torch.zeros(self.num_heads, self.dim_k, self.dim_v, device=DEVICE)
        self.S_slow = torch.zeros(self.num_heads, self.dim_k, self.dim_v, device=DEVICE)

        self.repellers: List[torch.Tensor] = []
        self.episodic_buffer: List[Tuple[int, int]] = []
        self.max_buffer_size = 256

    def reset_state(self):
        self.S_fast = torch.zeros(self.num_heads, self.dim_k, self.dim_v, device=DEVICE)
        self.S_slow = torch.zeros(self.num_heads, self.dim_k, self.dim_v, device=DEVICE)
        self.repellers.clear()
        self.episodic_buffer.clear()

    def forward_stream_byte(self, x_byte: int, target_byte: int) -> Tuple[torch.Tensor, Dict, torch.Tensor]:
        x_tensor = torch.tensor([x_byte], device=DEVICE)
        target_tensor = torch.tensor([target_byte], device=DEVICE)

        emb = self.byte_emb(x_tensor)

        q = F.normalize(self.W_q(emb).view(self.num_heads, 1, self.dim_k), p=2, dim=-1)
        k = F.normalize(self.W_k(emb).view(self.num_heads, 1, self.dim_k), p=2, dim=-1)
        v = F.normalize(self.W_v(emb).view(self.num_heads, 1, self.dim_v), p=2, dim=-1)

        read_fast = torch.bmm(q, self.S_fast.detach()).squeeze(1).view(1, -1)
        read_slow = torch.bmm(q, self.S_slow.detach()).squeeze(1).view(1, -1)
        read_combined = torch.cat([read_fast, read_slow], dim=-1)

        beta = 1.0 + F.softplus(self.w_beta * torch.norm(read_combined) + self.b_beta)
        logits_att = beta * self.W_att(read_combined)

        repulsion = torch.zeros_like(logits_att)
        rep_sim_max = 0.0
        if len(self.repellers) > 0:
            valid_repellers = torch.stack(self.repellers, dim=0)
            sim_rep = torch.matmul(F.normalize(read_combined, p=2, dim=-1), F.normalize(valid_repellers, p=2, dim=-1).t())
            rep_sim_max = sim_rep.max().item()
            rep_weights = F.softmax(sim_rep * 4.0, dim=-1)
            rep_field = torch.matmul(rep_weights, valid_repellers)
            gamma = F.softplus(self.w_gamma)
            repulsion = gamma * self.W_rep(rep_field)

        logits = logits_att - repulsion
        loss = F.cross_entropy(logits, target_tensor)
        pred = torch.argmax(logits, dim=-1).item()
        surprise_val = loss.item()

        if surprise_val > 1.5:
            if len(self.episodic_buffer) >= self.max_buffer_size:
                self.episodic_buffer.pop(0)
            self.episodic_buffer.append((x_byte, target_byte))

        # Memory update
        with torch.no_grad():
            alpha_fast = torch.clamp(torch.sigmoid(self.fast_decay_params), min=0.70, max=0.92).view(self.num_heads, 1, 1)
            alpha_slow = torch.clamp(torch.sigmoid(self.slow_decay_params), min=0.990, max=0.9995).view(self.num_heads, 1, 1)

            v_pred_fast = torch.bmm(k, self.S_fast)
            error_fast = v - v_pred_fast
            delta_fast = torch.bmm(k.transpose(1, 2), error_fast)
            self.S_fast = alpha_fast * self.S_fast + self.config.eta_fast * delta_fast

            v_pred_slow = torch.bmm(k, self.S_slow)
            error_slow = v - v_pred_slow
            delta_slow = torch.bmm(k.transpose(1, 2), error_slow)

            gate_slow = torch.clamp(torch.tensor(surprise_val / 3.0, device=DEVICE), min=0.1, max=1.0)
            self.S_slow = alpha_slow * self.S_slow + (self.config.eta_slow * gate_slow) * delta_slow

            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()

        # Telemetry package
        meta = {
            "loss": surprise_val,
            "pred": pred,
            "target": target_byte,
            "is_correct": (pred == target_byte),
            "entropy": entropy,
            "beta": beta.item(),
            "rep_sim_max": rep_sim_max,
            "norm_read_fast": torch.norm(read_fast).item(),
            "norm_read_slow": torch.norm(read_slow).item(),
            "norm_att": torch.norm(logits_att).item(),
            "norm_rep": torch.norm(repulsion).item(),
            "repellers_count": len(self.repellers),
        }
        return loss, meta, read_combined.detach().squeeze(0)

    def update_repellers_post_backward(self, state_vec: torch.Tensor, pred: int, target: int, surprise: float):
        if pred != target and surprise > 1.2:
            if len(self.repellers) >= self.config.num_repellers:
                self.repellers.pop(0)
            self.repellers.append(state_vec.clone())
        else:
            for i in range(len(self.repellers)):
                self.repellers[i] = self.repellers[i] * self.config.rep_decay

    def compute_spectral_health(self) -> Dict[str, float]:
        with torch.no_grad():
            # Spectral SVD on slow memory (averaged across heads)
            pr_slow_list = []
            cond_slow_list = []
            for h in range(self.num_heads):
                mat = self.S_slow[h]
                s = torch.linalg.svdvals(mat)
                s_sum = torch.sum(s)
                s_sq_sum = torch.sum(s ** 2)
                pr = ((s_sum ** 2) / (s_sq_sum + 1e-8)).item()
                pr_slow_list.append(pr)
                cond = (s[0] / (s[-1] + 1e-8)).item()
                cond_slow_list.append(cond)

            pr_fast_list = []
            for h in range(self.num_heads):
                mat = self.S_fast[h]
                s = torch.linalg.svdvals(mat)
                pr_fast_list.append(((torch.sum(s) ** 2) / (torch.sum(s ** 2) + 1e-8)).item())

            return {
                "slow_pr_mean": sum(pr_slow_list) / len(pr_slow_list),
                "slow_cond_mean": sum(cond_slow_list) / len(cond_slow_list),
                "fast_pr_mean": sum(pr_fast_list) / len(pr_fast_list),
                "norm_s_slow": torch.norm(self.S_slow).item(),
                "norm_s_fast": torch.norm(self.S_fast).item()
            }


def run_deep_diagnostics():
    logger.info("================================================================================")
    logger.info("=== STARTING EXP-431: DEEP MECHANISTIC ENDOSCOPIC DIAGNOSTIC AUDIT ===")
    logger.info("================================================================================")

    config = EXP431Config()
    model = EndoscopicDiagnosticCore(config).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)

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

    # Tracking Structures
    error_positions = []
    error_chars = []
    high_loss_reasons = {
        "word_start": 0,       # immediately after whitespace
        "punctuation": 0,      # symbols like :, ;, (, ), {, }
        "indentation": 0,      # whitespace / newlines
        "alphanumeric": 0      # regular letters inside words
    }

    grad_norms = {
        "byte_emb": [],
        "W_q": [],
        "W_k": [],
        "W_v": [],
        "W_att": [],
        "W_rep": [],
        "w_beta": [],
        "w_gamma": []
    }

    step_losses = []
    step_rep_sims = []
    step_norm_att = []
    step_norm_rep = []

    t_start = time.perf_counter()

    for t in range(len(raw_bytes) - 1):
        x_byte = raw_bytes[t]
        target_byte = raw_bytes[t + 1]

        optimizer.zero_grad()
        loss, meta, state_vec = model.forward_stream_byte(x_byte, target_byte)
        loss.backward()

        # Audit Gradient Norms periodically
        if t % 100 == 0:
            if model.byte_emb.weight.grad is not None:
                grad_norms["byte_emb"].append(torch.norm(model.byte_emb.weight.grad).item())
            if model.W_q.weight.grad is not None:
                grad_norms["W_q"].append(torch.norm(model.W_q.weight.grad).item())
            if model.W_k.weight.grad is not None:
                grad_norms["W_k"].append(torch.norm(model.W_k.weight.grad).item())
            if model.W_v.weight.grad is not None:
                grad_norms["W_v"].append(torch.norm(model.W_v.weight.grad).item())
            if model.W_att.weight.grad is not None:
                grad_norms["W_att"].append(torch.norm(model.W_att.weight.grad).item())
            if model.W_rep.weight.grad is not None:
                grad_norms["W_rep"].append(torch.norm(model.W_rep.weight.grad).item())
            if model.w_beta.grad is not None:
                grad_norms["w_beta"].append(torch.norm(model.w_beta.grad).item())
            if model.w_gamma.grad is not None:
                grad_norms["w_gamma"].append(torch.norm(model.w_gamma.grad).item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.update_repellers_post_backward(state_vec, meta["pred"], target_byte, meta["loss"])

        step_losses.append(meta["loss"])
        step_rep_sims.append(meta["rep_sim_max"])
        step_norm_att.append(meta["norm_att"])
        step_norm_rep.append(meta["norm_rep"])

        # Error Typology Analysis
        if not meta["is_correct"]:
            error_positions.append(t)
            error_chars.append(chr(target_byte) if 32 <= target_byte <= 126 else f"\\x{target_byte:02x}")
            # Categorize error
            prev_char = chr(x_byte) if 32 <= x_byte <= 126 else ""
            targ_char = chr(target_byte) if 32 <= target_byte <= 126 else ""
            if prev_char in (" ", "\n", "\t"):
                high_loss_reasons["word_start"] += 1
            elif targ_char in (":", ";", "(", ")", "{", "}", "[", "]", "=", "<", ">", "/", "*", "+", "-"):
                high_loss_reasons["punctuation"] += 1
            elif targ_char in (" ", "\n", "\t"):
                high_loss_reasons["indentation"] += 1
            else:
                high_loss_reasons["alphanumeric"] += 1

        if (t + 1) % 1000 == 0:
            spectral = model.compute_spectral_health()
            logger.info(
                f"[T={t+1}] | Avg Loss: {sum(step_losses)/(t+1):.4f} | "
                f"S_slow PR: {spectral['slow_pr_mean']:.2f}/64 | "
                f"S_slow Cond: {spectral['slow_cond_mean']:.2f} | "
                f"S_fast PR: {spectral['fast_pr_mean']:.2f} | "
                f"Norm Att: {meta['norm_att']:.2f} | Norm Rep: {meta['norm_rep']:.2f}"
            )

    t_elapsed = time.perf_counter() - t_start
    final_spectral = model.compute_spectral_health()

    # Detailed Audit Computations
    total_steps = len(step_losses)
    second_half_losses = step_losses[total_steps // 2:]
    second_half_errors = [1 for l_val in second_half_losses if l_val >= 0.5]
    logger.info(f"Second Half High-Loss Steps (>=0.5): {len(second_half_errors)} / {len(second_half_losses)}")

    # Correlation between Repulsion strength and Loss
    # Did high repulsion help or hurt?
    rep_active_losses = [step_losses[i] for i in range(total_steps) if step_norm_rep[i] > 0.1]
    rep_inactive_losses = [step_losses[i] for i in range(total_steps) if step_norm_rep[i] <= 0.1]

    avg_loss_rep_active = sum(rep_active_losses) / len(rep_active_losses) if rep_active_losses else 0.0
    avg_loss_rep_inactive = sum(rep_inactive_losses) / len(rep_inactive_losses) if rep_inactive_losses else 0.0

    # Error distribution
    total_errors = len(error_positions)
    error_breakdown = {k: round((v / (total_errors + 1e-8)) * 100.0, 2) for k, v in high_loss_reasons.items()}

    logger.info("================================================================================")
    logger.info("=== EXP-431 DEEP ENDOSCOPIC DIAGNOSTIC AUDIT RESULTS ===")
    logger.info(f"Total Stream Steps: {total_steps} | Elapsed: {t_elapsed:.2f}s | Throughput: {total_steps/t_elapsed:.2f} steps/s")
    logger.info(f"Final Average Loss: {sum(step_losses)/total_steps:.4f} nats")
    logger.info(f"Total Prediction Errors: {total_errors} ({total_errors/total_steps*100:.2f}%)")
    logger.info("--- 1. ERROR TYPOLOGY & MECHANISTIC ORIGIN ---")
    logger.info(f"Errors at Word Boundaries (First letter after space/newline): {error_breakdown['word_start']}%")
    logger.info(f"Errors at Alphanumeric Characters (Inside words): {error_breakdown['alphanumeric']}%")
    logger.info(f"Errors at Punctuation / Code Syntax Symbols: {error_breakdown['punctuation']}%")
    logger.info(f"Errors at Whitespace / Newlines / Indents: {error_breakdown['indentation']}%")
    logger.info("--- 2. SPECTRAL & MEMORY INTEGRITY ---")
    logger.info(f"S_slow Participation Ratio (PR Rank): {final_spectral['slow_pr_mean']:.2f} / 64")
    logger.info(f"S_slow Condition Number: {final_spectral['slow_cond_mean']:.2f}")
    logger.info(f"S_fast Participation Ratio: {final_spectral['fast_pr_mean']:.2f} / 64")
    logger.info(f"S_slow Frobenius Norm: {final_spectral['norm_s_slow']:.2f}")
    logger.info("--- 3. ATTRACTOR vs REPELLER INTERFERENCE AUDIT ---")
    logger.info(f"Avg Loss when Repellers Active: {avg_loss_rep_active:.4f} nats")
    logger.info(f"Avg Loss when Repellers Inactive: {avg_loss_rep_inactive:.4f} nats")
    logger.info(f"Mean Attractor Norm: {sum(step_norm_att)/len(step_norm_att):.2f}")
    logger.info(f"Mean Repeller Norm: {sum(step_norm_rep)/len(step_norm_rep):.2f}")
    logger.info("--- 4. GRADIENT FLOW AUDIT ---")
    for k, v in grad_norms.items():
        if v:
            logger.info(f"Gradient Norm [{k}]: mean={sum(v)/len(v):.4f}, min={min(v):.4f}, max={max(v):.4f}")
    logger.info("================================================================================")

    report = {
        "total_steps": total_steps,
        "elapsed_seconds": round(t_elapsed, 2),
        "final_loss": round(sum(step_losses) / total_steps, 4),
        "total_errors": total_errors,
        "error_breakdown_percent": error_breakdown,
        "spectral_health": final_spectral,
        "avg_loss_rep_active": round(avg_loss_rep_active, 4),
        "avg_loss_rep_inactive": round(avg_loss_rep_inactive, 4),
        "grad_norms_mean": {k: round(sum(v)/len(v), 4) if v else 0.0 for k, v in grad_norms.items()}
    }

    with open("experiments/exp_431_diagnostic_report.json", "w") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    run_deep_diagnostics()
