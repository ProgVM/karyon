"""
EXP-206: Integrated Anchored Hopfield & Volitional Motor Head (Production Refactoring & Validation)
Hypothesis: Merging the validated EXP-205 Anchored Allostatically-Gated Hopfield Motor Relaxation
and EXP-200 Allostatically-Gated Volitional Motor Head directly into production karyon_agent.py
will sustain or improve loss convergence (Loss Delta >= 0.08) while ensuring zero-delta function identity at birth.
"""

import sys
import os
import time
import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from karyon_config import CoREConfig
from karyon_agent import CoREAgent
from karyon_core import HomeostaticUnit, BatchedEpisodicMemory
from karyon_checkpoint import load_karyon

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("exp_206_integrated_anchored_hopfield_motor")

def run_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    soul_path = "karyon_soul.kcore"
    if not os.path.exists(soul_path):
        logger.error(f"Checkpoint file {soul_path} not found!")
        sys.exit(1)

    logger.info("Loading baseline KaryonEntity from karyon_soul.kcore...")
    
    core_config = CoREConfig()
    core_config.net.text_dim = 256
    core_config.net.unified_dim = 256
    core_config.net.hidden_dim = 768
    core_config.net.latent_dim = 128
    core_config.net.expand_dim = 3072
    core_config.net.num_heads = 12
    core_config.net.num_attractors = 256
    core_config.net.text_gen_dim = 258
    core_config.train.batch_size = 4

    model = CoREAgent(config=core_config, device=str(device)).to(device)
    hu = HomeostaticUnit(batch_size=4, device=str(device))
    episodic_mem = BatchedEpisodicMemory(batch_size=4, memory_dim=core_config.net.unified_dim, max_capacity=1000, device=str(device))

    load_karyon(model, episodic_mem, hu, filepath=soul_path, device=str(device))
    model.eval()

    # Create synthetic batch mimicking real UTF-8 stream
    batch_size = 4
    seq_len = 128
    dummy_input = torch.randint(0, 256, (batch_size, seq_len), dtype=torch.long, device=device)

    # Baseline evaluation
    with torch.no_grad():
        out_base = model.forward_sequence(dummy_input)
        loss_base = out_base.get('loss', None)
        if loss_base is None:
            logits = out_base['logits']
            loss_base = F.cross_entropy(logits.view(-1, logits.size(-1)), dummy_input.view(-1)).item()
        else:
            loss_base = loss_base.item()

    logger.info(f"Baseline Loss: {loss_base:.4f}")

    # Inject / Verify Anchored Basins in Hopfield Attractor Head
    if hasattr(model, 'attractor_head') and hasattr(model.attractor_head, 'basins'):
        with torch.no_grad():
            centroids = model.pos_embeddings.byte_embed.weight.data.clone() # [258, 256]
            if centroids.size(1) != model.attractor_head.basins.size(1):
                # Expand or project if dimensions differ
                proj = nn.Linear(centroids.size(1), model.attractor_head.basins.size(1), bias=False).to(device)
                nn.init.orthogonal_(proj.weight)
                centroids = proj(centroids)
            model.attractor_head.basins.data.copy_(centroids)
            logger.info("Anchored Hopfield basins successfully updated with token centroids.")

    # Benchmark proposed integrated model
    start_time = time.time()
    iterations = 20
    losses = []

    with torch.no_grad():
        for _ in range(iterations):
            out_prop = model.forward_sequence(dummy_input)
            l = out_prop.get('loss', None)
            if l is None:
                logits = out_prop['logits']
                l = F.cross_entropy(logits.view(-1, logits.size(-1)), dummy_input.view(-1)).item()
            else:
                l = l.item()
            losses.append(l)

    elapsed = time.time() - start_time
    avg_loss = sum(losses) / len(losses)
    total_tokens = iterations * batch_size * seq_len
    tok_per_sec = total_tokens / elapsed

    delta_loss = loss_base - avg_loss
    logger.info(f"Integrated Model Loss: {avg_loss:.4f} | Delta Loss: {delta_loss:.4f} | Tok/s: {tok_per_sec:.1f}")

    metrics = {
        "loss": avg_loss,
        "baseline_loss": loss_base,
        "delta_loss": delta_loss,
        "tok_per_sec": tok_per_sec
    }

    with open("experiments/exp_206_results.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Results successfully saved to experiments/exp_206_results.json")

if __name__ == '__main__':
    run_experiment()
