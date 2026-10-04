# experiments/exp_345_pac_predictive_gating.py
"""
===============================================================================
EXP-345: THETA-GAMMA PHASE-AMPLITUDE COUPLING (PAC) GATED LOCAL PREDICTIVE UPDATE
===============================================================================
Implements KEP Principle 24 (Strict Single-Pass Stream Learning & Anti-Zubryoshka),
Principle 26 (The Triad of Grounded Meaning & Pragmatic Agency), and KEP Rule #2
(Loss Delta >= 0.08 empirical validation threshold).

Architectural Delta:
1. Slow-Theta Rhythmic Auto-Oscillator:
   theta_{t+1} = (theta_t + omega_theta * dt) mod 2pi
2. Theta-Gamma Phase-Amplitude Coupling (PAC) Modulation:
   A_gamma(theta_t) = 1 / (1 + exp(-kappa * (cos(theta_t - phi_opt) - theta_thresh)))
3. Somatic Homeostasis-Gated Weight Modulation (Ashby Homeostasis H_somatic):
   Delta W_{ij} = -eta * A_gamma(theta_t) * (dF/dW_{ij}) * exp(-beta * |H_somatic - H_target|)
   
This phase-locked gating prevents local predictive gradient drift during Socratic
stream learning by anchoring plasticity updates strictly to peak gamma alignment
within slow theta cycles, stabilized by somatic neurotransmitter homeostasis.
"""

import os
import sys
import time
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from karyon_agent import KaryonAgent, KaryonConfig
from karyon_entity import SocraticTutor

def run_exp_345_pac_predictive_gating():
    print("=" * 80)
    print("EXP-345: THETA-GAMMA PAC GATED LOCAL PREDICTIVE UPDATE BENCHMARK")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Target Device: {device}")

    # 1. Initialize Karyon Agent Configuration & Model
    config = KaryonConfig(
        vocab_size=1024,
        d_model=128,
        num_layers=4,
        num_organelles=4,
        max_seq_len=256
    )
    agent = KaryonAgent(config).to(device)
    agent.eval()

    # 2. Initialize Socratic Tutor Stream & Benchmark Lessons
    tutor = SocraticTutor()
    lessons = tutor.generate_curriculum_stream(num_lessons=20)
    print(f"[*] Generated Socratic Curriculum Stream: {len(lessons)} lessons across arithmetic, identity, causality.")

    # 3. PAC Auto-Oscillator & Somatic State Initialization
    theta_phase = 0.0
    omega_theta = 0.5236 # ~30 deg per step (slow theta rhythm)
    phi_opt = math.pi / 2.0 # Optimal phase alignment for gamma bursting
    kappa = 8.0 # PAC coupling sharpness
    theta_thresh = 0.2 # Threshold for gamma phase gating
    
    h_somatic = 1.0 # Current Ashby Somatic Homeostasis energy
    h_target = 1.0 # Target homeostasis baseline
    beta_homeostasis = 0.8 # Decay penalty for homeostasis deviation

    start_time = time.time()
    initial_loss = None
    final_loss = None
    loss_history = []
    pac_modulation_history = []
    teaching_accuracies = []

    print("\n[*] Starting Socratic Single-Pass Stream Learning with PAC Gated Plasticity...")
    
    for step, lesson in enumerate(lessons):
        # Update Theta Oscillator Phase
        theta_phase = (theta_phase + omega_theta) % (2.0 * math.pi)
        
        # Calculate Gamma Amplitude via Phase-Amplitude Coupling (PAC)
        cos_diff = math.cos(theta_phase - phi_opt)
        a_gamma = 1.0 / (1.0 + math.exp(-kappa * (cos_diff - theta_thresh)))
        pac_modulation_history.append(a_gamma)

        # Update Somatic Homeostasis Factor
        somatic_factor = math.exp(-beta_homeostasis * abs(h_somatic - h_target))
        
        # Effective Gated Learning Rate for Local Predictive Step
        base_lr = 1e-2
        effective_lr = base_lr * a_gamma * somatic_factor

        # Prepare Tensors
        prompt = lesson["prompt"]
        target = lesson["target"]
        
        prompt_ids = torch.tensor([[ord(c) % config.vocab_size for c in prompt]], dtype=torch.long, device=device)
        target_ids = torch.tensor([[ord(c) % config.vocab_size for c in target]], dtype=torch.long, device=device)

        # Measure baseline prediction error before step
        with torch.no_grad():
            logits_before = agent(prompt_ids, thinking_steps=2)
            # Alignment loss measure
            if logits_before.shape[1] >= target_ids.shape[1]:
                sub_logits = logits_before[:, :target_ids.shape[1], :]
                loss_before = F.cross_entropy(sub_logits.reshape(-1, config.vocab_size), target_ids.reshape(-1)).item()
            else:
                loss_before = 3.5

        if initial_loss is None:
            initial_loss = loss_before

        # Execute PAC-Gated Local Predictive Coding Step
        if effective_lr > 1e-4:
            step_loss = agent.teach_predictive_step(
                input_ids=prompt_ids,
                target_ids=target_ids,
                learning_rate=effective_lr,
                weight_decay=1e-4,
                thinking_steps=2,
                target_organelles_only=True
            )
        else:
            step_loss = loss_before

        loss_history.append(step_loss)
        final_loss = step_loss

        # Dynamic Somatic State Update based on Learning Signal
        h_somatic = 0.95 * h_somatic + 0.05 * (1.0 if step_loss < 2.0 else 0.5)

        # Evaluate Socratic Acquisition Response
        with torch.no_grad():
            logits_after = agent(prompt_ids, thinking_steps=2)
            pred_tokens = logits_after.argmax(dim=-1)[0]
            pred_text = "".join([chr(tok.item() % 128) if 32 <= (tok.item() % 128) <= 126 else "?" for tok in pred_tokens])
            is_correct = target in pred_text
            teaching_accuracies.append(1.0 if is_correct else 0.0)

        if (step + 1) % 5 == 0 or step == len(lessons) - 1:
            print(f"  Step {step+1:02d}/{len(lessons):02d} | Theta Phase: {theta_phase:.2f} rad | A_gamma (PAC): {a_gamma:.4f} | Eff LR: {effective_lr:.5f} | Loss: {step_loss:.4f}")

    duration = time.time() - start_time
    loss_delta = initial_loss - final_loss if (initial_loss is not None and final_loss is not None) else 0.0
    taught_acc = (sum(teaching_accuracies) / len(teaching_accuracies)) * 100.0 if teaching_accuracies else 0.0
    mean_pac_modulation = sum(pac_modulation_history) / len(pac_modulation_history) if pac_modulation_history else 0.0

    print("\n" + "=" * 80)
    print("EXP-345 EMPIRICAL TELEMETRY RESULTS")
    print("=" * 80)
    print(f"Initial Loss (Pre-PAC Stream): {initial_loss:.4f}")
    print(f"Final Loss (Post-PAC Stream): {final_loss:.4f}")
    print(f"Loss Delta (Baseline Delta) : {loss_delta:.4f}")
    print(f"Taught Response Accuracy    : {taught_acc:.2f}%")
    print(f"Mean Gamma PAC Modulation   : {mean_pac_modulation:.4f}")
    print(f"Execution Duration          : {duration:.2f}s")

    # KEP Rule #2 Verdict Evaluation (Loss Delta >= 0.08)
    if loss_delta >= 0.08 and final_loss < 2.5:
        verdict = "POSITIVE"
        print("\n[VERDICT: POSITIVE] KEP Rule #2 Satisfied! Loss Delta >= 0.08 and phase-amplitude coupling established stable local convergence.")
    else:
        verdict = "REJECTED"
        print(f"\n[VERDICT: REJECTED] Loss Delta {loss_delta:.4f} did not meet KEP Rule #2 threshold (>= 0.08).")

    # Print JSON Metrics block for automated telemetry parser
    metrics = {
        "initial_loss": round(initial_loss, 4),
        "final_loss": round(final_loss, 4),
        "loss_delta": round(loss_delta, 4),
        "taught_accuracy_pct": round(taught_acc, 2),
        "mean_pac_modulation": round(mean_pac_modulation, 4),
        "train_duration_sec": round(duration, 2),
        "verdict": verdict
    }
    print(f"\nFINAL_METRICS_JSON: {json.dumps(metrics)}")
    return metrics

if __name__ == "__main__":
    run_exp_345_pac_predictive_gating()
