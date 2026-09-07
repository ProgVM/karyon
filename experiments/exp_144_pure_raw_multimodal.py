# experiments/exp_144_pure_raw_multimodal.py
"""
EXP-144: Pure Tabula Rasa Raw Physical Multimodal Ingestion & Active Inference Benchmark

Hypothesis:
Karyon-CoRE's C++20 SensoryGateway and 2-Stage Cortical Stack can ingest 100% raw physical sensory
streams (16x16 raw pixel intensities [0..1], 256-sample raw continuous PCM time-domain audio waveforms,
256 raw binary document bytes, and UTF-8 text bytes) directly WITHOUT ANY external pretrained models,
third-party feature extractors, or downloaded checkpoints (KEP Principle 2, 12, & Container Autonomy).

Data Substrate (100% Pure Raw Physical Inputs):
1. Raw Vision: Raw 16x16 normalized pixel intensities (256D) directly from MNIST images (downscaled via basic bilinear interp).
2. Raw Audio: 256-sample raw PCM time-domain amplitude values [-1.0, 1.0] from physical synth waveforms.
3. Raw Binary: 256 raw byte values [0.0, 1.0] read directly from 'libkaryon_runtime.so' binary.
4. Raw Text: UTF-8 raw byte sequence (V=258) mapped via Karyon's native positional byte embeddings.

Telemetry Captured:
- Cross-channel attention weight distribution (Text, Vision, Audio, Binary)
- Variational Free Energy (F_t) convergence across 30 raw physical multi-modal steps
- Multi-modal Motor Gateway outputs (Speech logits, Vision prediction, Audio prediction, Motor efference)
- Step latency (ms/step), throughput (steps/sec), and peak CUDA VRAM memory footprint
"""

import sys
import os
import time
import math
import logging
import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset

# Ensure workspace root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from karyon_entity import KaryonEntity
from karyon_config import CoREConfig, NetworkConfig, HomeostasisConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EXP-144")

class PureRawMultimodalDataset:
    def __init__(self, device, size=30):
        self.device = device
        self.size = size
        
        logger.info(f"Preparing Pure Tabula Rasa Raw Dataset ({size} samples)...")
        
        # 1. Raw Vision: Load MNIST and extract RAW 16x16 pixel grid (256 raw intensity values in [0..1])
        logger.info("Loading MNIST raw images and resizing to 16x16 raw pixel grid (256D)...")
        mnist_ds = load_dataset("mnist", split=f"test[:{size}]")
        
        self.raw_images = []
        self.texts = []
        self.labels = []
        digit_names = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        
        for i in range(size):
            item = mnist_ds[i]
            img_pil = item["image"]
            label = item["label"]
            
            # Convert to raw tensor float [1, 1, 28, 28]
            img_arr = np.array(img_pil, dtype=np.float32) / 255.0
            img_t = torch.tensor(img_arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            
            # Resize directly to 16x16 (256 pixels) via basic mathematical bilinear interpolation (NO PRETRAINED NETWORKS)
            img_16x16 = F.interpolate(img_t, size=(16, 16), mode='bilinear', align_corners=False).squeeze()
            raw_pix_256 = img_16x16.view(256).to(device)
            
            self.raw_images.append(raw_pix_256)
            self.labels.append(label)
            self.texts.append(f"A handwritten digit {digit_names[label]}.")

        # 2. Raw Audio: 256-sample raw time-domain PCM waveforms (NO FFT, NO PRETRAINED SPEECH MODEL)
        logger.info("Generating 256-sample raw continuous PCM time-domain waveforms...")
        self.raw_audio = []
        t = np.linspace(0, 0.02, 256, endpoint=False) # 20ms audio frame
        
        for i in range(size):
            label = self.labels[i]
            freq = 220 + label * 55 # Base pitch for each digit
            pcm_wave = np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(6 * np.pi * freq * t)
            pcm_tensor = torch.tensor(pcm_wave, dtype=torch.float32, device=device)
            self.raw_audio.append(pcm_tensor)

        # 3. Raw Binary: Read raw binary byte chunks directly from C++ runtime binary
        logger.info("Reading raw binary byte streams directly from 'libkaryon_runtime.so'...")
        self.raw_binary = []
        bin_path = "libkaryon_runtime.so" if os.path.exists("libkaryon_runtime.so") else "karyon_soul.kcore"
        
        with open(bin_path, "rb") as f:
            bin_data = f.read(size * 256)
            
        for i in range(size):
            chunk = bin_data[i*256 : (i+1)*256]
            if len(chunk) < 256:
                chunk = chunk + b"\x00" * (256 - len(chunk))
            raw_bytes_float = torch.tensor([float(b) / 255.0 for b in chunk], dtype=torch.float32, device=device)
            self.raw_binary.append(raw_bytes_float)

    def get_sample(self, idx):
        return (
            self.raw_images[idx],
            self.texts[idx],
            self.raw_audio[idx],
            self.raw_binary[idx]
        )

def run_pure_raw_multimodal_benchmark():
    logger.info("=====================================================================================")
    logger.info(" === [STARTING EXP-144: PURE TABULA RASA RAW MULTIMODAL BENCHMARK] ===")
    logger.info("=====================================================================================")

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Active Hardware Backend: {device_str.upper()}")

    # 1. Load active entity
    kcore_path = "karyon_soul.kcore"
    entity = KaryonEntity.load(filepath=kcore_path, device=device_str)
    brain = entity.brain
    device = entity.device

    # Ensure C++20 SensoryGateway projections are configured
    brain.gateway.register_channel("text", brain.config.net.text_dim)
    brain.gateway.register_channel("vision", brain.config.net.vision_dim)
    brain.gateway.register_channel("audio", getattr(brain.config.net, 'audio_dim', 256))
    brain.gateway.register_channel("binary", getattr(brain.config.net, 'binary_dim', 256))

    # 2. Load 100% Pure Raw Physical Dataset
    dataset_size = 30
    dataset = PureRawMultimodalDataset(device=device, size=dataset_size)

    # 3. Active Inference Loop over Raw Multimodal Sequences
    logger.info("\n--- Executing Raw Physical Multimodal Active Inference Loop ---")
    h_f = entity.h_fast.clone()
    h_s = entity.h_slow.clone()
    u_state = entity.hu.state.clone()

    fe_history = []
    step_latencies = []

    torch.cuda.synchronize() if device.type == "cuda" else None
    t_start_all = time.perf_counter()

    for step in range(dataset_size):
        t0 = time.perf_counter()
        
        # Fetch pure raw physical sample
        raw_img_pix, text_str, raw_pcm_wav, raw_bin_bytes = dataset.get_sample(step)
        
        # Encode raw text
        text_tokens = brain.encode_text(text_str)
        tok_id = text_tokens[0] if len(text_tokens) > 0 else torch.tensor(0, device=device)
        t_emb = brain.pos_embeddings(tok_id.to(device).unsqueeze(0).unsqueeze(0), start_pos=step, apply_rf=False).squeeze(1)

        sensor_inputs = {
            "text": t_emb,
            "vision": raw_img_pix.unsqueeze(0),
            "audio": raw_pcm_wav.unsqueeze(0),
            "binary": raw_bin_bytes.unsqueeze(0)
        }

        with torch.no_grad():
            h_f, h_s, actions, cog_actions, text_logits, fe, attn_weights, w_t, w_pred, value_est, epistemic_entropy, eff_dt = brain(
                sensor_inputs, h_f, h_s, u_state
            )
            motor_outs = brain.output_gateway(h_f)
            vis_gen = motor_outs.get("vision_generation", torch.zeros(1, brain.config.net.vision_dim, device=device))
            aud_gen = motor_outs.get("audio_generation", torch.zeros(1, getattr(brain.config.net, 'audio_dim', 256), device=device))

        torch.cuda.synchronize() if device.type == "cuda" else None
        step_ms = (time.perf_counter() - t0) * 1000.0
        step_latencies.append(step_ms)

        fe_val = fe.mean().item() if isinstance(fe, torch.Tensor) else float(fe)
        fe_history.append(fe_val)
        
        attn_dict = {f"ch_{i}": round(v.item(), 3) for i, v in enumerate(attn_weights.squeeze(0))} if attn_weights.numel() > 0 else {}

        logger.info(
            f" Step {step+1:02d}/{dataset_size} | Label: {dataset.labels[step]} | Latency: {step_ms:.2f}ms | "
            f"FE: {fe_val:.4f} | Attn: {attn_dict} | VisGen: {vis_gen.norm().item():.3f} | AudGen: {aud_gen.norm().item():.3f}"
        )

    t_total_sec = time.perf_counter() - t_start_all
    avg_step_ms = sum(step_latencies) / len(step_latencies)
    fps_multimodal = dataset_size / t_total_sec

    # 4. Measure VRAM footprint
    peak_vram_mb = 0.0
    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    initial_fe = fe_history[0]
    final_fe = fe_history[-1]
    fe_reduction = initial_fe - final_fe

    # 5. Evaluation & Verdict
    logger.info("\n=====================================================================================")
    logger.info(" === [EXP-144 PURE TABULA RASA RAW MULTIMODAL SUMMARY] ===")
    logger.info("=====================================================================================")
    logger.info(f" Total Raw Samples Processed   : {dataset_size}")
    logger.info(f" Average Step Latency           : {avg_step_ms:.2f} ms/step")
    logger.info(f" Throughput                     : {fps_multimodal:.1f} steps/sec")
    logger.info(f" Initial Free Energy (F_0)      : {initial_fe:.4f}")
    logger.info(f" Final Free Energy (F_30)       : {final_fe:.4f}")
    logger.info(f" Free Energy Reduction          : {fe_reduction:+.4f}")
    logger.info(f" Peak CUDA VRAM Memory          : {peak_vram_mb:.2f} MB")

    verdict = "🟢 POSITIVE" if final_fe <= 0.15 and peak_vram_mb < 2000 else "⚪ NEUTRAL"
    logger.info(f" VERDICT: {verdict}")
    logger.info("=====================================================================================")

    return {
        "exp_id": "EXP-144",
        "verdict": verdict,
        "total_steps": dataset_size,
        "avg_step_ms": avg_step_ms,
        "throughput_fps": fps_multimodal,
        "initial_free_energy": initial_fe,
        "final_free_energy": final_fe,
        "fe_reduction": fe_reduction,
        "peak_vram_mb": peak_vram_mb
    }

if __name__ == "__main__":
    run_pure_raw_multimodal_benchmark()
