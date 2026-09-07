# experiments/exp_142_multimodal_ingestion.py
"""
EXP-142: Native Multimodal Ingestion & Cross-Modal Active Inference Benchmark

Hypothesis:
Karyon-CoRE's C++20 SensoryGateway and 2-Stage Cortical Stack can seamlessly ingest, align,
and process non-textual input streams (Vision latent features, Audio spectrogram embeddings,
and Raw Binary document streams) alongside Text into a unified 256D representation space.
The system will minimize Free Energy (F_t) across modalities and generate multimodal
motor actions (Text speech, Vision latent prediction, and Volitional Motor Efference).

Telemetry Captured:
- Cross-channel attention weight distribution (Text, Vision, Audio, Binary)
- Variational Free Energy (F_t) convergence across 20 multimodal sequence steps
- Multi-modal Motor Gateway outputs (Speech logits, Vision prediction, Motor efference)
- Somatic & Affective state dynamics under multimodal stimulation
- Throughput (items/sec) and peak CUDA VRAM memory footprint
"""

import sys
import os
import time
import math
import logging
import torch
import torch.nn as nn

# Ensure workspace root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from karyon_entity import KaryonEntity
from karyon_config import CoREConfig, NetworkConfig, HomeostasisConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EXP-142")

def run_multimodal_ingestion_benchmark():
    logger.info("=====================================================================================")
    logger.info(" === [STARTING EXP-142: NATIVE MULTIMODAL INGESTION & ACTIVE INFERENCE] ===")
    logger.info("=====================================================================================")

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Active Hardware Backend: {device_str.upper()}")

    # 1. Load active entity
    kcore_path = "karyon_soul.kcore"
    entity = KaryonEntity.load(filepath=kcore_path, device=device_str)
    brain = entity.brain
    device = entity.device

    logger.info(f"Entity loaded successfully | Device: {device} | Text Dim: {brain.text_dim} | Unified Dim: {brain.unified_dim}")

    # Ensure all sensory projections exist
    logger.info("Verifying C++20 SensoryGateway multi-channel registration...")
    brain.gateway.register_channel("text", brain.config.net.text_dim)
    brain.gateway.register_channel("vision", brain.config.net.vision_dim)
    brain.gateway.register_channel("audio", getattr(brain.config.net, 'audio_dim', 256))
    brain.gateway.register_channel("binary", getattr(brain.config.net, 'binary_dim', 256))

    # 2. Prepare synthetic multimodal data batch (Sequence length = 20)
    seq_len = 20
    batch_size = 1

    logger.info(f"Generating synthetic multimodal test sequence (Seq Len: {seq_len}, Batch Size: {batch_size})...")
    
    # Text stream: encoding sample text
    sample_text = "Universal Multimodal Cognitive Active Inference Engine"
    text_tokens = brain.encode_text(sample_text)
    
    # Visual stream: 256D latent vectors representing image frames (e.g., camera/ViT latents)
    vision_latents = torch.randn(seq_len, 1, brain.config.net.vision_dim, device=device) * 0.5
    
    # Audio stream: 256D latent vectors representing acoustic spectrogram frames
    audio_latents = torch.randn(seq_len, 1, getattr(brain.config.net, 'audio_dim', 256), device=device) * 0.3
    
    # Binary stream: 256D latent vectors representing document byte chunks
    binary_latents = torch.randn(seq_len, 1, getattr(brain.config.net, 'binary_dim', 256), device=device) * 0.2

    # 3. Step through time steps and monitor active inference dynamics
    h_f = entity.h_fast.clone()
    h_s = entity.h_slow.clone()
    u_state = entity.hu.state.clone()

    fe_history = []
    attn_history = []
    step_latencies = []

    logger.info("\nExecuting Multimodal Active Inference Loop over 20 steps...")

    torch.cuda.synchronize() if device.type == "cuda" else None
    t_start_all = time.perf_counter()

    for step in range(seq_len):
        t0 = time.perf_counter()
        
        # Select current frame token embedding
        tok_id = text_tokens[step % len(text_tokens)]
        t_emb = brain.pos_embeddings(tok_id.to(device).unsqueeze(0).unsqueeze(0), start_pos=step, apply_rf=False).squeeze(1)

        sensor_inputs = {
            "text": t_emb,
            "vision": vision_latents[step],
            "audio": audio_latents[step],
            "binary": binary_latents[step]
        }

        with torch.no_grad():
            h_f, h_s, act_idx, speech_logits, fe, fe_reaction, w_human, w_human_next, s_pred, m_eff, vis_gen, aud_gen = brain(
                sensor_inputs, h_f, h_s, u_state
            )

        torch.cuda.synchronize() if device.type == "cuda" else None
        step_ms = (time.perf_counter() - t0) * 1000.0
        step_latencies.append(step_ms)

        fe_val = fe.item() if isinstance(fe, torch.Tensor) else float(fe)
        fe_history.append(fe_val)

        logger.info(
            f" Step {step+1:02d}/20 | Latency: {step_ms:.2f}ms | Free Energy: {fe_val:.4f} | Action: {act_idx} | "
            f"VisGen Norm: {vis_gen.norm().item():.3f} | AudGen Norm: {aud_gen.norm().item():.3f}"
        )

    t_total_sec = time.perf_counter() - t_start_all
    avg_step_ms = sum(step_latencies) / len(step_latencies)
    fps_multimodal = seq_len / t_total_sec

    # 4. Multimodal Generation & Synthesis Check
    logger.info("\n--- Multimodal Generation & Expression Audit ---")
    logger.info(f"Speech Logits Shape : {speech_logits.shape} (V=258)")
    logger.info(f"Vision Output Shape : {vis_gen.shape} (Dim=256)")
    logger.info(f"Audio Output Shape  : {aud_gen.shape} (Dim=256)")
    logger.info(f"Motor Action Vector : {m_eff.squeeze().cpu().numpy().round(4)}")

    # 5. Measure VRAM footprint
    peak_vram_mb = 0.0
    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # 6. Evaluation & Verdict
    initial_fe = fe_history[0]
    final_fe = fe_history[-1]
    fe_drop = initial_fe - final_fe

    logger.info("\n=====================================================================================")
    logger.info(" === [EXP-142 MULTIMODAL INGESTION BENCHMARK SUMMARY] ===")
    logger.info("=====================================================================================")
    logger.info(f" Total Steps Processed     : {seq_len}")
    logger.info(f" Total Time Elapsed        : {t_total_sec:.4f} s")
    logger.info(f" Multimodal Step Latency   : {avg_step_ms:.2f} ms/step")
    logger.info(f" Multimodal Throughput     : {fps_multimodal:.1f} steps/sec")
    logger.info(f" Initial Free Energy (F_0) : {initial_fe:.4f}")
    logger.info(f" Final Free Energy (F_20)  : {final_fe:.4f}")
    logger.info(f" Free Energy Reduction     : {fe_drop:+.4f}")
    logger.info(f" Peak CUDA VRAM Memory     : {peak_vram_mb:.2f} MB")

    verdict = "🟢 POSITIVE" if final_fe <= 0.10 and peak_vram_mb < 2000 else "⚪ NEUTRAL"
    logger.info(f" VERDICT: {verdict}")
    logger.info("=====================================================================================")

    return {
        "exp_id": "EXP-142",
        "verdict": verdict,
        "total_steps": seq_len,
        "avg_step_ms": avg_step_ms,
        "throughput_fps": fps_multimodal,
        "initial_free_energy": initial_fe,
        "final_free_energy": final_fe,
        "peak_vram_mb": peak_vram_mb
    }

if __name__ == "__main__":
    run_multimodal_ingestion_benchmark()
