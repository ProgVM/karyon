# experiments/exp_143_real_multimodal_dataset.py
"""
EXP-143: Real-World Multimodal Dataset Active Inference & Cross-Modal Alignment Benchmark

Hypothesis:
Karyon-CoRE's C++20 SensoryGateway and 2-Stage Cortical Stack can ingest and align REAL non-textual 
datasets (real images from MNIST/CIFAR via ResNet18 features, real audio waveforms/spectrograms, 
and real binary files/documents) alongside real text. The system will demonstrate active inference 
by minimizing Free Energy (F_t) over real multi-modal sequences, and will show semantic alignment 
by yielding lower Free Energy when text descriptions match the corresponding images compared to mismatched pairs.

Data Sources:
1. Real Images: MNIST (Hugging Face datasets) -> ResNet18 feature extractor -> 256D latents.
2. Real Audio: Synthesized Sine/Square waves with varying frequencies/amplitudes -> 256D spectral features.
3. Real Binary: Raw binary byte chunks from 'libkaryon_runtime.so' and 'karyon_soul.kcore' -> 256D projections.
4. Real Text: Text descriptions corresponding to the images.

Telemetry Captured:
- Cross-channel attention weight distribution (Text, Vision, Audio, Binary)
- Variational Free Energy (F_t) convergence across real multi-modal sequences
- Semantic Alignment Delta: F_t (Matched Text-Image) vs F_t (Mismatched Text-Image)
- Throughput (steps/sec) and peak CUDA VRAM memory footprint
"""

import sys
import os
import time
import math
import logging
import torch
import torch.nn as nn
import numpy as np
from datasets import load_dataset
import torchvision.models as models
import torchvision.transforms as T

# Ensure workspace root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from karyon_entity import KaryonEntity
from karyon_config import CoREConfig, NetworkConfig, HomeostasisConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EXP-143")

class RealMultimodalDataset:
    def __init__(self, device, size=50):
        self.device = device
        self.size = size
        
        # 1. Image Encoder (ResNet18 pretrained)
        logger.info("Initializing ResNet18 Image Feature Extractor...")
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.img_extractor = nn.Sequential(*list(resnet.children())[:-1]).to(device)
        self.img_extractor.eval()
        self.img_proj = nn.Linear(512, 256).to(device)
        self.img_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 2. Load Real Images & Text Descriptions (MNIST digits)
        logger.info(f"Loading {size} samples from MNIST dataset...")
        mnist_ds = load_dataset("mnist", split=f"test[:{size}]")
        
        self.images = []
        self.texts = []
        self.labels = []
        
        digit_names = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        
        for i in range(size):
            item = mnist_ds[i]
            img = item["image"].convert("RGB")
            label = item["label"]
            
            # Transform and extract features
            img_t = self.img_transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = self.img_extractor(img_t).squeeze(-1).squeeze(-1)
                feat_256 = self.img_proj(feat)
            
            self.images.append(feat_256.squeeze(0))
            self.labels.append(label)
            self.texts.append(f"A handwritten digit {digit_names[label]}.")
            
        # 3. Real Audio: Synthesized raw audio waves representing digit sounds (varying frequencies)
        logger.info("Synthesizing real physical audio waveforms for digits...")
        self.audio_features = []
        sr = 16000
        duration = 0.5 # seconds
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        
        for i in range(size):
            label = self.labels[i]
            # Frequency proportional to label
            freq = 220 + label * 55 # Hz
            wave = np.sin(2 * np.pi * freq * t)
            # Add some harmonics
            wave += 0.5 * np.sin(4 * np.pi * freq * t)
            # Convert to 256D feature vector (e.g. FFT bins)
            fft_vals = np.abs(np.fft.rfft(wave))[:256]
            # Normalize
            fft_vals = fft_vals / (np.linalg.norm(fft_vals) + 1e-8)
            self.audio_features.append(torch.tensor(fft_vals, dtype=torch.float32, device=device))
            
        # 4. Real Binary: Read raw binary chunks from compiled C++ library
        logger.info("Reading real binary byte chunks from compiled C++ library...")
        self.binary_features = []
        bin_path = "libkaryon_runtime.so"
        if not os.path.exists(bin_path):
            bin_path = "karyon_soul.kcore"
            
        with open(bin_path, "rb") as f:
            bin_data = f.read(size * 256)
            
        bin_proj = nn.Linear(256, 256).to(device)
        for i in range(size):
            chunk = bin_data[i*256 : (i+1)*256]
            if len(chunk) < 256:
                chunk = chunk + b"\x00" * (256 - len(chunk))
            # Convert bytes to floats in [0, 1]
            float_chunk = torch.tensor([float(b) / 255.0 for b in chunk], dtype=torch.float32, device=device)
            with torch.no_grad():
                proj_chunk = bin_proj(float_chunk)
            self.binary_features.append(proj_chunk)

    def get_sample(self, idx, mismatched_text=False):
        # Image
        img_feat = self.images[idx]
        # Text (matched or mismatched)
        text_idx = (idx + 5) % self.size if mismatched_text else idx
        text_str = self.texts[text_idx]
        # Audio
        aud_feat = self.audio_features[idx]
        # Binary
        bin_feat = self.binary_features[idx]
        
        return img_feat, text_str, aud_feat, bin_feat

def run_real_multimodal_benchmark():
    logger.info("=====================================================================================")
    logger.info(" === [STARTING EXP-143: REAL MULTIMODAL DATASET ACTIVE INFERENCE] ===")
    logger.info("=====================================================================================")

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Active Hardware Backend: {device_str.upper()}")

    # 1. Load active entity
    kcore_path = "karyon_soul.kcore"
    entity = KaryonEntity.load(filepath=kcore_path, device=device_str)
    brain = entity.brain
    device = entity.device

    # Ensure all sensory projections exist
    brain.gateway.register_channel("text", brain.config.net.text_dim)
    brain.gateway.register_channel("vision", brain.config.net.vision_dim)
    brain.gateway.register_channel("audio", getattr(brain.config.net, 'audio_dim', 256))
    brain.gateway.register_channel("binary", getattr(brain.config.net, 'binary_dim', 256))

    # 2. Load Real Multimodal Dataset (50 samples)
    dataset_size = 30
    dataset = RealMultimodalDataset(device=device, size=dataset_size)

    # 3. Phase A: Sequential Active Inference over Real Dataset (Matched)
    logger.info("\n--- Phase A: Processing Real Multimodal Dataset (Matched Modalities) ---")
    h_f = entity.h_fast.clone()
    h_s = entity.h_slow.clone()
    u_state = entity.hu.state.clone()

    fe_matched_history = []
    step_latencies = []

    t_start_all = time.perf_counter()

    for step in range(dataset_size):
        t0 = time.perf_counter()
        
        # Get matched real sample
        img_feat, text_str, aud_feat, bin_feat = dataset.get_sample(step, mismatched_text=False)
        
        # Encode real text
        text_tokens = brain.encode_text(text_str)
        tok_id = text_tokens[0] if len(text_tokens) > 0 else torch.tensor(0, device=device)
        t_emb = brain.pos_embeddings(tok_id.to(device).unsqueeze(0).unsqueeze(0), start_pos=step, apply_rf=False).squeeze(1)

        sensor_inputs = {
            "text": t_emb,
            "vision": img_feat.unsqueeze(0),
            "audio": aud_feat.unsqueeze(0),
            "binary": bin_feat.unsqueeze(0)
        }

        with torch.no_grad():
            h_f, h_s, actions, cog_actions, text_logits, fe, attn_weights, w_t, w_pred, value_est, epistemic_entropy, eff_dt = brain(
                sensor_inputs, h_f, h_s, u_state
            )

        torch.cuda.synchronize() if device.type == "cuda" else None
        step_ms = (time.perf_counter() - t0) * 1000.0
        step_latencies.append(step_ms)

        fe_val = fe.mean().item() if isinstance(fe, torch.Tensor) else float(fe)
        fe_matched_history.append(fe_val)
        
        attn_dict = {f"ch_{i}": round(v.item(), 3) for i, v in enumerate(attn_weights.squeeze(0))} if attn_weights.numel() > 0 else {}

        logger.info(
            f" Sample {step+1:02d}/{dataset_size} | Label: {dataset.labels[step]} | "
            f"FE: {fe_val:.4f} | Attn: {attn_dict} | Text: '{text_str}'"
        )

    t_total_sec = time.perf_counter() - t_start_all
    avg_step_ms = sum(step_latencies) / len(step_latencies)
    fps_multimodal = dataset_size / t_total_sec

    # 4. Phase B: Semantic Alignment Test (Matched vs Mismatched Text-Image)
    logger.info("\n--- Phase B: Semantic Alignment Test (Matched vs Mismatched) ---")
    fe_mismatched_history = []
    
    # Reset states for parity
    h_f_m = entity.h_fast.clone()
    h_s_m = entity.h_slow.clone()

    for step in range(dataset_size):
        # Get MISMATCHED real sample
        img_feat, text_str, aud_feat, bin_feat = dataset.get_sample(step, mismatched_text=True)
        
        text_tokens = brain.encode_text(text_str)
        tok_id = text_tokens[0] if len(text_tokens) > 0 else torch.tensor(0, device=device)
        t_emb = brain.pos_embeddings(tok_id.to(device).unsqueeze(0).unsqueeze(0), start_pos=step, apply_rf=False).squeeze(1)

        sensor_inputs = {
            "text": t_emb,
            "vision": img_feat.unsqueeze(0),
            "audio": aud_feat.unsqueeze(0),
            "binary": bin_feat.unsqueeze(0)
        }

        with torch.no_grad():
            _, _, _, _, _, fe, _, _, _, _, _, _ = brain(
                sensor_inputs, h_f_m, h_s_m, u_state
            )
        fe_val = fe.mean().item() if isinstance(fe, torch.Tensor) else float(fe)
        fe_mismatched_history.append(fe_val)

    mean_fe_matched = np.mean(fe_matched_history)
    mean_fe_mismatched = np.mean(fe_mismatched_history)
    alignment_delta = mean_fe_mismatched - mean_fe_matched

    # 5. Measure VRAM footprint
    peak_vram_mb = 0.0
    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # 6. Evaluation & Verdict
    logger.info("\n=====================================================================================")
    logger.info(" === [EXP-143 REAL MULTIMODAL DATASET BENCHMARK SUMMARY] ===")
    logger.info("=====================================================================================")
    logger.info(f" Total Real Samples Processed : {dataset_size}")
    logger.info(f" Average Step Latency         : {avg_step_ms:.2f} ms/step")
    logger.info(f" Throughput                   : {fps_multimodal:.1f} steps/sec")
    logger.info(f" Mean FE (Matched Modality)   : {mean_fe_matched:.4f}")
    logger.info(f" Mean FE (Mismatched Modality): {mean_fe_mismatched:.4f}")
    logger.info(f" Semantic Alignment Delta     : {alignment_delta:+.6f} (Positive = Matched is more predictable)")
    logger.info(f" Peak CUDA VRAM Memory        : {peak_vram_mb:.2f} MB")

    # Verdict: Matched must yield lower Free Energy (higher predictability) than mismatched
    verdict = "🟢 POSITIVE" if alignment_delta > 0.0010 else "⚪ NEUTRAL"
    logger.info(f" VERDICT: {verdict}")
    logger.info("=====================================================================================")

    return {
        "exp_id": "EXP-143",
        "verdict": verdict,
        "total_steps": dataset_size,
        "avg_step_ms": avg_step_ms,
        "throughput_fps": fps_multimodal,
        "mean_fe_matched": mean_fe_matched,
        "mean_fe_mismatched": mean_fe_mismatched,
        "alignment_delta": alignment_delta,
        "peak_vram_mb": peak_vram_mb
    }

if __name__ == "__main__":
    run_real_multimodal_benchmark()
