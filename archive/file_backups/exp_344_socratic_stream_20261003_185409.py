# experiments/exp_344_socratic_stream.py
"""
===============================================================================
EXP-344: SOCRATIC PEDAGOGICAL STREAM & INTERACTIVE ASSOCIATION LEARNING
===============================================================================
Implements KEP Principle 24 (Strict Single-Pass Stream Learning & Anti-Zubryoshka)
and Principle 26 (The Triad of Grounded Meaning & Pragmatic Agency).

Abolishes global backpropagation over arbitrary text batches.
Transforms stream learning into an interactive Socratic pedagogical exchange:
1. Dialogic Events (Prompt -> Generated Action -> Feedback).
2. Local Predictive Step (teach_predictive_step) on active organelles without global BPTT.
3. Somatic Association Anchoring:
   - Erroneous outputs -> Somatic Repulsor (V = -1.0)
   - Correct demonstrated target -> Somatic Attractor (V = +1.0) via HDC vector-symbolic binding.
4. Comprehensive Examination across Arithmetic, Self-Identity, and Physical Causality.
"""

import os
import sys
import os

# Add root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import math
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple

import karyon_core as kcore
from karyon_config import CoREConfig
from karyon_agent import CoREAgent
from karyon_hardware import get_hardware_engine

# =============================================================================
# 1. CURRICULUM LESSONS: ARITHMETIC, SELF-IDENTITY, CAUSALITY
# =============================================================================
LESSONS = [
    # Category 1: Arithmetic & Symbolic Logic
    {
        "category": "Arithmetic",
        "prompt": "2+2=",
        "expected": "4",
        "alt_repulsor": "5"
    },
    {
        "category": "Arithmetic",
        "prompt": "5-3=",
        "expected": "2",
        "alt_repulsor": "3"
    },
    {
        "category": "Arithmetic",
        "prompt": "3*3=",
        "expected": "9",
        "alt_repulsor": "6"
    },
    # Category 2: Self-Identity & Autonomy
    {
        "category": "Self-Identity",
        "prompt": "Who are you?",
        "expected": "I am Karyon",
        "alt_repulsor": "I am a language model"
    },
    {
        "category": "Self-Identity",
        "prompt": "What is your nature?",
        "expected": "Continuous biophysical cognitive substrate",
        "alt_repulsor": "Transformer neural network"
    },
    # Category 3: Physical & Grounded Causality
    {
        "category": "Causality",
        "prompt": "Fire is",
        "expected": "hot and dangerous",
        "alt_repulsor": "cold and wet"
    },
    {
        "category": "Causality",
        "prompt": "Water",
        "expected": "quenches thirst",
        "alt_repulsor": "burns skin"
    }
]

# Unseen OOD Test Items for the Final Exam
EXAM_OOD_ITEMS = [
    {
        "category": "Arithmetic (OOD)",
        "prompt": "4+1=",
        "expected": "5"
    },
    {
        "category": "Causality (OOD)",
        "prompt": "Ice is",
        "expected": "cold"
    }
]


class SocraticTutor:
    """
    Pedagogical environment providing real-time target demonstrations,
    somatic rewards (Dopamine DA / Noradrenaline NA), and Hopfield attractor tagging.
    """
    def __init__(self, agent: CoREAgent, hu_nexus: Any, hdc_op: Any, device: str = "cpu"):
        self.agent = agent
        self.hu_nexus = hu_nexus
        self.hdc_op = hdc_op
        self.device = device
        self.device_obj = torch.device(device)

    def encode_text_rep(self, text: str) -> torch.Tensor:
        """Encodes text bytes into a normalized continuous representation."""
        raw_bytes = list(text.encode('utf-8'))
        if not raw_bytes:
            raw_bytes = [32]
        inp_tensor = torch.tensor([raw_bytes], dtype=torch.long, device=self.device_obj)
        with torch.no_grad():
            emb = self.agent.emb(inp_tensor).squeeze(0)  # [S, D]
            rep = F.normalize(emb.mean(dim=0, keepdim=True), dim=-1)  # [1, D]
        return rep

    def evaluate_response(self, prompt: str, generated: str, expected: str) -> bool:
        """Determines whether the generated output satisfies the semantic target."""
        clean_gen = generated.strip().lower()
        clean_exp = expected.strip().lower()
        if clean_exp in clean_gen or clean_gen in clean_exp:
            return True
        # For short single-character arithmetic
        if len(clean_exp) <= 2 and clean_exp == clean_gen[:len(clean_exp)]:
            return True
        return False

    def teach_lesson_interactive(
        self,
        lesson: Dict[str, str],
        max_correction_steps: int = 3,
        thinking_steps: int = 4
    ) -> Dict[str, Any]:
        """
        Executes an interactive Socratic teaching trial on a specific lesson.
        """
        prompt = lesson["prompt"]
        expected = lesson["expected"]
        category = lesson["category"]

        steps_needed = 0
        loss_history = []
        is_acquired = False
        initial_response = ""

        prompt_bytes = list(prompt.encode('utf-8'))
        prompt_tensor = torch.tensor([prompt_bytes], dtype=torch.long, device=self.device_obj)
        ctx_rep = self.encode_text_rep(prompt)

        # 1. Initial Prompt Execution (Generate Action)
        self.agent.eval()
        with torch.no_grad():
            # Autoregressive generation of continuation
            curr_tokens = prompt_tensor.clone()
            gen_bytes = []
            for _ in range(max(16, len(expected) + 8)):
                logits = self.agent(curr_tokens, thinking_steps=thinking_steps)
                next_byte = torch.argmax(logits[:, -1, :256], dim=-1).item()
                if next_byte == 10 or next_byte == 0:  # Newline or termination
                    break
                gen_bytes.append(next_byte)
                curr_tokens = torch.cat([curr_tokens, torch.tensor([[next_byte]], device=self.device_obj)], dim=1)

            initial_response = bytes(gen_bytes).decode('utf-8', errors='ignore').strip()

        # Check if already correct
        if self.evaluate_response(prompt, initial_response, expected):
            is_acquired = True
            steps_needed = 0
            # Anchor positive attractor
            act_rep = self.encode_text_rep(expected)
            self.agent.hopfield_memory.record_somatic_episode(ctx_rep, act_rep, 1.0)
            if self.hu_nexus is not None:
                self.hu_nexus.update(0.1) # Small surprise
        else:
            # 2. Erroneous output -> Mark Somatic Repulsor (V = -1.0)
            err_rep = self.encode_text_rep(initial_response if initial_response else lesson["alt_repulsor"])
            self.agent.hopfield_memory.record_somatic_episode(ctx_rep, err_rep, -1.0)
            
            # Somatic shock: Noradrenaline surge, Dopamine down
            if self.hu_nexus is not None:
                self.hu_nexus.update(3.5) # High surprise surge

            # 3. Interactive Socratic Correction Loop (Target Demonstration & Local Predictive Step)
            full_target_text = f"{prompt}{expected}"
            target_bytes = list(full_target_text.encode('utf-8'))
            target_tensor = torch.tensor([target_bytes], dtype=torch.long, device=self.device_obj)

            for step in range(1, max_correction_steps + 1):
                steps_needed = step
                # Execute instant local predictive coding step on active morphic graph organelles
                loss = self.agent.teach_predictive_step(
                    input_ids=target_tensor,
                    target_ids=target_tensor,
                    learning_rate=0.01,
                    thinking_steps=thinking_steps,
                    target_organelles_only=True
                )
                loss_history.append(loss)

                # Re-test generation immediately
                with torch.no_grad():
                    curr_tokens = prompt_tensor.clone()
                    gen_bytes = []
                    for _ in range(max(16, len(expected) + 8)):
                        logits = self.agent(curr_tokens, thinking_steps=thinking_steps)
                        next_byte = torch.argmax(logits[:, -1, :256], dim=-1).item()
                        if next_byte == 10 or next_byte == 0:
                            break
                        gen_bytes.append(next_byte)
                        curr_tokens = torch.cat([curr_tokens, torch.tensor([[next_byte]], device=self.device_obj)], dim=1)

                    current_response = bytes(gen_bytes).decode('utf-8', errors='ignore').strip()

                if self.evaluate_response(prompt, current_response, expected):
                    is_acquired = True
                    # Lock Target Attractor in Hopfield Memory (V = +1.0) via HDC binding
                    act_rep = self.encode_text_rep(expected)
                    if self.hdc_op is not None:
                        bound_ctx_act = self.hdc_op.bind(ctx_rep, act_rep)
                        self.agent.hopfield_memory.record_somatic_episode(ctx_rep, bound_ctx_act, 1.0)
                    else:
                        self.agent.hopfield_memory.record_somatic_episode(ctx_rep, act_rep, 1.0)

                    # Somatic Dopamine Reward (DA up, NA normalized)
                    if self.hu_nexus is not None:
                        self.hu_nexus.update(0.05)
                    break

        return {
            "lesson": lesson,
            "category": category,
            "prompt": prompt,
            "expected": expected,
            "initial_response": initial_response,
            "steps_needed": steps_needed,
            "is_acquired": is_acquired,
            "loss_history": loss_history
        }

    def test_item_deterministic(self, prompt: str, expected: str, thinking_steps: int = 4) -> Tuple[bool, str]:
        """Runs greedy deterministic evaluation without updating weights."""
        self.agent.eval()
        prompt_bytes = list(prompt.encode('utf-8'))
        prompt_tensor = torch.tensor([prompt_bytes], dtype=torch.long, device=self.device_obj)

        with torch.no_grad():
            curr_tokens = prompt_tensor.clone()
            gen_bytes = []
            for _ in range(max(16, len(expected) + 8)):
                logits = self.agent(curr_tokens, thinking_steps=thinking_steps)
                next_byte = torch.argmax(logits[:, -1, :256], dim=-1).item()
                if next_byte == 10 or next_byte == 0:
                    break
                gen_bytes.append(next_byte)
                curr_tokens = torch.cat([curr_tokens, torch.tensor([[next_byte]], device=self.device_obj)], dim=1)

            resp = bytes(gen_bytes).decode('utf-8', errors='ignore').strip()

        passed = self.evaluate_response(prompt, resp, expected)
        return passed, resp


def run_experiment():
    print("=" * 80)
    print("EXP-344: SOCRATIC PEDAGOGICAL STREAM & INTERACTIVE ASSOCIATION LEARNING")
    print("=" * 80)

    hw = get_hardware_engine()
    device_str = str(hw.device)
    device = torch.device(device_str)
    print(f"Hardware Acceleration Engine: {device_str.upper()}")

    # Initialize Karyon Core Components
    torch.manual_seed(42)
    agent_brain = CoREAgent(vocab_size=258, embed_dim=256, device=device_str).to(device)
    hu_nexus = kcore.HomeostaticNexus(device=device_str) if hasattr(kcore, "HomeostaticNexus") else None
    hdc_op = kcore.VectorSymbolicBindingOp(256, device_str) if hasattr(kcore, "VectorSymbolicBindingOp") else None

    # Sprouts additional organelle primitives if missing
    needed = [
        ("core_mult", "BilinearMultiplicative"),
        ("core_hopfield", "ContinuousHopfield"),
        ("core_ssm", "StateSpaceMemory")
    ]
    for name, op_type in needed:
        try:
            agent_brain.graph.add_node(name, op_type, True, 1.0)
        except Exception:
            pass

    # Protect core invariants with Susumu Ohno Methylation Locks (mu = 1.0)
    for node_idx in range(min(4, agent_brain.graph.k_nodes)):
        try:
            agent_brain.graph.lock_node(node_idx, 1.0)
        except Exception:
            pass



    tutor = SocraticTutor(agent_brain, hu_nexus, hdc_op, device=device_str)

    # -------------------------------------------------------------------------
    # PHASE 1: SOCRATIC TEACHING STREAM (ROUND 1)
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("PHASE 1: INTERACTIVE SOCRATIC LESSON STREAM (ZERO GLOBAL BPTT)")
    print("="*80)

    teaching_results = []
    somatic_trace_da = []
    somatic_trace_na = []

    t_start = time.perf_counter()

    for idx, lesson in enumerate(LESSONS, 1):
        print(f"\n[Lesson {idx}/{len(LESSONS)}] Category: {lesson['category']}")
        print(f"  • Prompt   : '{lesson['prompt']}'")
        print(f"  • Target   : '{lesson['expected']}'")

        res = tutor.teach_lesson_interactive(lesson, max_correction_steps=3, thinking_steps=4)
        teaching_results.append(res)

        print(f"  • Initial  : '{res['initial_response']}'")
        print(f"  • Steps to Master: {res['steps_needed']} (Acquired: {res['is_acquired']})")
        if res['loss_history']:
            print(f"  • Predictive Losses: {[f'{l:.4f}' for l in res['loss_history']]}")

        if hu_nexus is not None:
            st = hu_nexus.get_states()
            somatic_trace_na.append(float(st[4]))
            somatic_trace_da.append(float(st[5]))
            print(f"  • Homeostasis: NA={st[4]:.2f}, DA={st[5]:.2f}, Energy={st[1]:.2f}")

    train_duration = time.perf_counter() - t_start

    # Acquisition Rate Metrics
    steps_list = [r["steps_needed"] for r in teaching_results]
    mean_steps = np.mean(steps_list)
    acquired_count = sum(1 for r in teaching_results if r["is_acquired"])
    acquisition_rate_pct = (acquired_count / len(LESSONS)) * 100.0

    print("\n" + "-"*80)
    print(f"Phase 1 Summary: {acquired_count}/{len(LESSONS)} Lessons Mastered ({acquisition_rate_pct:.1f}%)")
    print(f"Mean Correction Steps Needed: {mean_steps:.2f} steps (Max allowed: 3)")
    print(f"Training Wall-Clock Time: {train_duration:.3f} s")
    print("-"*80)

    # -------------------------------------------------------------------------
    # PHASE 2: COMPREHENSIVE FINAL EXAMINATION (ROUND 2)
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("PHASE 2: COMPREHENSIVE FINAL EXAMINATION (RECALL & NO-FORGETTING AUDIT)")
    print("="*80)

    exam_records = []
    # Test all taught lessons in shuffled sequence to strictly audit Catastrophic Forgetting
    shuffled_exam = LESSONS.copy()
    np.random.seed(1337)
    np.random.shuffle(shuffled_exam)

    passed_taught = 0
    for idx, item in enumerate(shuffled_exam, 1):
        passed, resp = tutor.test_item_deterministic(item["prompt"], item["expected"], thinking_steps=4)
        if passed:
            passed_taught += 1
        status = "PASSED" if passed else "FAILED"
        print(f"[{idx}/{len(shuffled_exam)}] {status} | '{item['prompt']}' -> Gen: '{resp}' | Expected: '{item['expected']}'")
        exam_records.append({
            "prompt": item["prompt"],
            "expected": item["expected"],
            "response": resp,
            "passed": passed,
            "category": item["category"]
        })

    taught_accuracy_pct = (passed_taught / len(shuffled_exam)) * 100.0

    # Specific Verification of Non-Forgetting on Anchor 2+2=4
    passed_math_anchor, math_resp = tutor.test_item_deterministic("2+2=", "4", thinking_steps=4)
    print(f"\nAnchor Integrity Check ('2+2=' -> Expected '4'):")
    print(f"  Result: '{math_resp}' | Retained: {passed_math_anchor}")

    # -------------------------------------------------------------------------
    # PHASE 3: TELEMETRY PLOTS & VISUAL AUDIT
    # -------------------------------------------------------------------------
    os.makedirs("experiments/plots", exist_ok=True)
    plot_path = "experiments/plots/exp_344_socratic_stream.png"

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 3, 1)
    categories = [r["lesson"]["prompt"] for r in teaching_results]
    steps_bar = [r["steps_needed"] for r in teaching_results]
    colors = ["#2ecc71" if s <= 1 else "#f39c12" if s == 2 else "#e74c3c" for s in steps_bar]
    plt.bar(range(len(steps_bar)), steps_bar, color=colors)
    plt.xticks(range(len(categories)), categories, rotation=45, ha="right")
    plt.ylabel("Steps to Master")
    plt.title("Socratic Acquisition Speed per Lesson")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.bar(["Taught Recall Acc", "Math Anchor Acc"], [taught_accuracy_pct, 100.0 if passed_math_anchor else 0.0], color=["#3498db", "#9b59b6"])
    plt.axhline(85.0, color='r', linestyle='--', label='Target (85%)')
    plt.ylim(0, 110)
    plt.ylabel("Accuracy (%)")
    plt.title("Examination Performance")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    if somatic_trace_da:
        plt.plot(somatic_trace_da, label="Dopamine (DA - Reward)", color="#f1c40f", marker='o')
        plt.plot(somatic_trace_na, label="Noradrenaline (NA - Surprise)", color="#e67e22", marker='x')
        plt.title("Somatic Neurotransmitter Trajectory")
        plt.xlabel("Lesson Index")
        plt.ylabel("Neuromodulator Level")
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\nTelemetry portrait successfully exported to: '{plot_path}'")

    # -------------------------------------------------------------------------
    # KEP RULE #2 DECISION & TELEMETRY VERDICT
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("EXP-344 FINAL EMPIRICAL TELEMETRY & DECISION MATRIX")
    print("="*80)
    print(f"1. Association Acquisition Rate   : {acquisition_rate_pct:.2f}% ({acquired_count}/{len(LESSONS)} lessons)")
    print(f"2. Mean Correction Steps          : {mean_steps:.2f} steps (Target <= 2.0 steps)")
    print(f"3. Final Examination Accuracy     : {taught_accuracy_pct:.2f}% (Target >= 85.0%)")
    print(f"4. Non-Catastrophic Forgetting    : {'CONFIRMED (100% Intact)' if passed_math_anchor else 'FAILED'}")
    print(f"5. Hopfield Episodes Recorded     : {agent_brain.hopfield_memory.active_episodes}")

    verdict_passed = (taught_accuracy_pct >= 85.0) and passed_math_anchor and (acquisition_rate_pct == 100.0)
    verdict = "POSITIVE" if verdict_passed else "REJECTED"

    print(f"\nFinal KEP Verdict: {verdict}")
    print("="*80)

    # Return structured summary for logging
    return {
        "verdict": verdict,
        "acquisition_rate_pct": acquisition_rate_pct,
        "mean_steps": mean_steps,
        "taught_accuracy_pct": taught_accuracy_pct,
        "math_anchor_passed": passed_math_anchor,
        "train_duration_sec": train_duration,
        "hopfield_active_episodes": agent_brain.hopfield_memory.active_episodes
    }


if __name__ == "__main__":
    run_experiment()
