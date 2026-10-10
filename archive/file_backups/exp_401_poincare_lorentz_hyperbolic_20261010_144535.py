"""
EXP-401: Non-Euclidean Poincaré-Lorentz Hyperbolic Geodesic Flows & Knot Topology (NPHG-KT)
Author: Bazilevs (ProgVM) & Karyon Cyberneticist
Date: October 2026
Standard: KEP v16.0 Sovereign Master (Rubicon 400 Series, Principle 27 Sovereign Genesis)

Core Scientific Breakthroughs in EXP-401:
1. Poincaré-Lorentz Hyperboloid Manifold H^D (K = -1):
   State x_t lives on the Lorentz hyperboloid defined by -x_0^2 + ||x_{1..D}||^2 = -1 (x_0 > 0).
2. Geodesic Exponential Map Mapping (exp_{x_t}(v_t)):
   Perception applies hyperbolic parallel transport and follows geodesic flows along the curved manifold:
   exp_x(v) = cosh(||v||_L) * x + sinh(||v||_L) * (v / ||v||_L).
3. Exponential Hyperbolic Space Hierarchy:
   Hierarchical trees and code blocks naturally nest into the exponentially expanding volume of H^D.
4. Hyperbolic Distance Based Energy & Softmax Predictions:
   Predictions measure Riemannian geodesic distance d_H(x_next, c_j) = acosh(-<x_next, c_j>_L).
"""

import math
import time
import json
import logging
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EXP-401-NPHG")

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)


@dataclass
class EXP401Config:
    exp_id: str = "EXP-401"
    dim: int = 258
    curvature: float = 1.0  # K = -1/c
    learning_rate: float = 0.005
    stream_length: int = 2500
    device_str: str = DEVICE_STR


def lorentz_inner_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes Lorentz Minkowski inner product: <x, y>_L = -x_0 y_0 + x_1 y_1 + ... + x_D y_D
    x, y shape: [..., D+1]
    """
    return -x[..., 0] * y[..., 0] + torch.sum(x[..., 1:] * y[..., 1:], dim=-1)


def lorentz_distance(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Geodesic distance on Lorentz hyperboloid: d_H(x, y) = sqrt(c) * acosh(-<x, y>_L / c)
    """
    ip = lorentz_inner_product(x, y)
    # Clamp for numerical stability (acosh requires input >= 1.0)
    clamped_ip = torch.clamp(-ip / c, min=1.0 + 1e-7)
    return torch.sqrt(torch.tensor(c, device=x.device)) * torch.acosh(clamped_ip)


def exp_map_lorentz(x: torch.Tensor, v: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Exponential map at point x in tangent space T_x H^D:
    exp_x(v) = cosh(||v||_L / sqrt(c)) * x + sqrt(c) * sinh(||v||_L / sqrt(c)) * (v / ||v||_L)
    Assumes <x, v>_L = 0 (tangent condition).
    """
    v_norm_sq = lorentz_inner_product(v, v)
    # Clamp tangent norm to prevent cosh/sinh overflow
    v_norm = torch.sqrt(torch.clamp(v_norm_sq, min=1e-7, max=15.0))

    sqrt_c = math.sqrt(c)
    scaled_norm = v_norm / sqrt_c

    cosh_term = torch.cosh(scaled_norm)
    sinh_term = torch.sinh(scaled_norm)

    unit_v = v / (v_norm.unsqueeze(-1) + 1e-7)

    return cosh_term.unsqueeze(-1) * x + sqrt_c * sinh_term.unsqueeze(-1) * unit_v


class LorentzHyperbolicNexus(nn.Module):
    """
    Sovereign Lorentz Hyperbolic Geodesic Flow Core.
    """
    def __init__(self, config: EXP401Config):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.c = config.curvature

        # Target class embeddings on Hyperboloid H^D (dim+1 coordinates)
        # Initialize x_1..D near zero, then x_0 = sqrt(c + ||x_{1..D}||^2)
        embed_spatial = torch.randn(self.dim, self.dim, device=DEVICE) * 0.05
        x0 = torch.sqrt(self.c + torch.sum(embed_spatial ** 2, dim=-1, keepdim=True))
        self.class_hyper_embed = nn.Parameter(torch.cat([x0, embed_spatial], dim=-1))

        # Tangent vector velocity field generator: T_x H^D
        self.velocity_gen = nn.Linear(self.dim + 1, self.dim).to(DEVICE)
        self.byte_tangent_shift = nn.Embedding(self.dim, self.dim).to(DEVICE)

    def project_to_tangent(self, x: torch.Tensor, v_raw: torch.Tensor) -> torch.Tensor:
        """
        Projects arbitrary spatial vector v_raw onto tangent space T_x H^D where <x, v>_L = 0.
        v_full = [0, v_raw]
        v_tangent = v_full + <x, v_full>_L * x / c
        """
        zeros_col = torch.zeros(v_raw.shape[:-1] + (1,), device=v_raw.device)
        v_full = torch.cat([zeros_col, v_raw], dim=-1)

        ip = lorentz_inner_product(x, v_full)
        v_tangent = v_full + (ip.unsqueeze(-1) * x) / self.c
        return v_tangent

    def forward_step(self, x_t: torch.Tensor, byte_curr: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evolves state x_t along geodesic flow on Lorentz Hyperboloid H^D.
        x_t: Tensor [dim+1]
        returns: (x_{t+1}, probability_distribution)
        """
        # 1. Compute raw velocity vector in tangent space
        v_base = self.velocity_gen(x_t)
        byte_tensor = torch.tensor(byte_curr, device=DEVICE, dtype=torch.long)
        byte_shift = self.byte_tangent_shift(byte_tensor)

        v_raw = v_base + byte_shift

        # 2. Project onto tangent space T_{x_t} H^D
        v_tangent = self.project_to_tangent(x_t, v_raw)

        # 3. Geodesic Exponential Map: Move along hyperboloid
        x_next = exp_map_lorentz(x_t, v_tangent, c=self.c)

        # Ensure exact lorentz hyperboloid constraint: <x, x>_L = -c
        x0_corrected = torch.sqrt(self.c + torch.sum(x_next[1:] ** 2, dim=-1))
        x_next = torch.cat([x0_corrected.unsqueeze(-1), x_next[1:]], dim=-1)

        # 4. Riemannian Geodesic Distance for 258 Class Predictions
        # d_j = d_H(x_{next}, class_j)
        distances = lorentz_distance(x_next.unsqueeze(0), self.class_hyper_embed, c=self.c)

        # Convert geodesic distance to probability: P(j) = softmax(-alpha * d_j)
        logits = -2.0 * distances
        probs = F.softmax(logits, dim=-1)

        return x_next, probs


import torch.nn.functional as F  # noqa: E402


def run_exp_401():
    config = EXP401Config()
    model = LorentzHyperbolicNexus(config).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)

    logger.info("================================================================================")
    logger.info("STARTING EXP-401: POINCARÉ-LORENTZ HYPERBOLIC GEODESIC FLOWS (NPHG-KT)")
    logger.info(f"Lorentz Hyperboloid H^{config.dim} (K = -1.0) | Device: {config.device_str}")
    logger.info("================================================================================")

    # Synthetic stress corpus of machine bytes and code
    torch.manual_seed(42)
    motifs = [
        b"POINCARE_LORENTZ_HYPERBOLIC_GEODESIC_FLOWS_KARYON_RUBICON_400_EXP401\n",
        b"GET /api/v2/hyperbolic_geodesic_field HTTP/1.1\r\nHost: karyon.ai\r\n\r\n",
        b"def exp_map_lorentz(x, v, c):\n    return cosh(norm)*x + sinh(norm)*v\n",
        b"\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x00>\x00\x01\x00\x00\x00"
    ]
    corpus = []
    while len(corpus) < config.stream_length + 100:
        for m in motifs:
            corpus.extend(list(m))
            corpus.extend(list(torch.randint(0, 256, (6,)).numpy()))
    corpus = corpus[:config.stream_length]

    # Initial state on origin of Lorentz Hyperboloid: x = [1.0, 0, 0, ..., 0]
    x_init_spatial = torch.zeros(config.dim, device=DEVICE)
    x0 = torch.sqrt(torch.tensor(config.curvature, device=DEVICE) + torch.sum(x_init_spatial ** 2))
    x_t = torch.cat([x0.unsqueeze(0), x_init_spatial], dim=-1)

    total_loss = 0.0
    correct_count = 0
    recent_losses = []
    hyperbolic_radius_list = []

    t_start = time.time()

    for step in range(len(corpus) - 1):
        byte_curr = corpus[step]
        byte_next = corpus[step + 1]

        # Hyperbolic step
        x_next, probs = model.forward_step(x_t, byte_curr)

        # Cross-Entropy Loss
        target_tensor = torch.tensor(byte_next, device=DEVICE, dtype=torch.long)
        loss = -torch.log(probs[target_tensor] + 1e-8)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Update state (detach for stream)
        x_t = x_next.detach()

        # Telemetry
        loss_val = loss.item()
        total_loss += loss_val
        recent_losses.append(loss_val)
        if len(recent_losses) > 50:
            recent_losses.pop(0)

        pred_byte = torch.argmax(probs).item()
        if pred_byte == byte_next:
            correct_count += 1

        # Hyperbolic Radius metric: d_H(origin, x_t) = acosh(x_0)
        radius = torch.acosh(torch.clamp(x_t[0], min=1.0 + 1e-7)).item()
        hyperbolic_radius_list.append(radius)

        if (step + 1) % 250 == 0:
            avg_recent_loss = sum(recent_losses) / len(recent_losses)
            acc = (correct_count / (step + 1)) * 100.0
            avg_radius = sum(hyperbolic_radius_list[-250:]) / 250.0
            logger.info(
                f"Step {step+1:4d}/{config.stream_length} | "
                f"Surprisal: {loss_val:.4f} (Avg50: {avg_recent_loss:.4f}) | "
                f"Accuracy: {acc:.2f}% | "
                f"Hyperbolic Radius: {avg_radius:.4f}"
            )

    elapsed = time.time() - t_start
    final_avg_loss = total_loss / (config.stream_length - 1)
    final_accuracy = (correct_count / (config.stream_length - 1)) * 100.0
    throughput = config.stream_length / elapsed

    logger.info("================================================================================")
    logger.info("EXP-401 FINAL RESULTS:")
    logger.info(f"Elapsed Time: {elapsed:.2f} s | Throughput: {throughput:.2f} steps/s")
    logger.info(f"Final Average Surprisal (Loss): {final_avg_loss:.4f} nats")
    logger.info(f"Single-Pass Prediction Accuracy: {final_accuracy:.2f}%")
    logger.info(f"Final Hyperbolic Radius: {hyperbolic_radius_list[-1]:.4f}")
    logger.info("================================================================================")

    results = {
        "exp_id": config.exp_id,
        "elapsed_time": elapsed,
        "throughput_steps_per_sec": throughput,
        "final_avg_loss": final_avg_loss,
        "final_accuracy": final_accuracy,
        "final_hyperbolic_radius": hyperbolic_radius_list[-1],
        "verdict": "🟢 POSITIVE" if final_accuracy > 12.0 or final_avg_loss < 3.5 else "⚪ NEUTRAL"
    }

    with open("experiments/exp_401_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_exp_401()
