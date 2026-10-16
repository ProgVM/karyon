# dialogue.py
"""
===============================================================================
KARYON INTERACTIVE SOCIAL ACTIVE INFERENCE & TEACHING SHELL (v7.0 MASTER)
===============================================================================
Interactive closed-loop cognitive shell providing live somatic teaching,
experiential feedback (Attractor/Repulsor tagging), predictive coding updates,
sleep memory consolidation, and persistent soul serialization.
"""

import logging
import torch

from karyon_entity import KaryonEntity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SocialTeachingShell")

device_str = "cuda" if torch.cuda.is_available() else "cpu"
kcore_target_path = "karyon_soul_v7.kcore"

# 1. Load Unified Karyon Entity
entity = KaryonEntity.load(filepath=kcore_target_path, device=device_str)


def render_affective_dashboard(entity: KaryonEntity):
    if entity.hu is not None:
        hu_state = entity.hu.get_states().tolist()
        curiosity, energy, stability, health, na, da = hu_state
    else:
        curiosity, energy, stability, health, na, da = 0.85, 1.0, 0.80, 1.0, 0.05, 0.05

    active_organelles = entity.brain.graph.k_nodes if hasattr(entity.brain, 'graph') else 2
    active_episodes = entity.memory.active_episodes if entity.memory is not None else 0

    print("\n" + "=" * 80)
    print(" === [KARYON SOMATIC & TEACHING DASHBOARD] ===")
    print("=" * 80)
    print(f"  Somatic State : Energy: {energy:.3f} | Health: {health:.3f} | Curiosity: {curiosity:.3f} | Stability: {stability:.3f}")
    print(f"  Neurokinetics : Noradrenaline (Arousal): {na:.3f} | Dopamine (Reward): {da:.3f}")
    print(f"  Cognitive Topo: Active Organelles: {active_organelles} | Somatic Episodes in Memory: {active_episodes}")
    print("=" * 80 + "\n")


def print_help_menu():
    print("""
===============================================================================
KARYON TEACHING SHELL COMMANDS:
  -- or /bad          : Mark last response as a Somatic Repulsor (Valence = -1.0)
  ++ or /good         : Mark last response as a Somatic Attractor (Valence = +1.0)
  /teach <good_reply> : Execute instant predictive coding gradient step on sample
  /sleep              : Trigger 3-Phase Sleep consolidation & Tononi synaptic downscaling
  /status             : Display deep somatic & biophysical dashboard
  /help               : Show this help menu
  exit                : Persist complete soul state to karyon_soul_v7.kcore and exit
===============================================================================
""")


def run_interactive_session():
    print("=" * 80)
    print("🚀 KARYON-CORE SOCIAL ACTIVE INFERENCE & TEACHING SHELL (v7.0)")
    print(f"Substrate Device: {device_str.upper()} | Active Model: CoREAgent v37.0")
    print("=" * 80)
    print_help_menu()

    while True:
        try:
            user_input = input("\nYou (Human Teacher): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting session...")
            entity.save(filepath=kcore_target_path)
            break

        if not user_input:
            # Spontaneous Turn
            karyon_response = entity.step("...", thinking_steps=4, max_new_tokens=48)
            print(f"Karyon (Spontaneous Thought): {karyon_response}")
            continue

        cmd = user_input.lower()

        # 1. Exit & Save Soul Container
        if cmd in ['exit', 'quit', '/exit']:
            entity.save(filepath=kcore_target_path)
            print(f"💾 Soul, weights, and somatic experiences saved to '{kcore_target_path}'. Goodbye.")
            break

        # 2. Somatic Negative Feedback / Repulsor
        elif cmd in ['--', '/bad', 'bad', '-1']:
            msg = entity.record_somatic_feedback(valence=-1.0)
            print(f"🛑 {msg}")
            continue

        # 3. Somatic Positive Feedback / Attractor
        elif cmd in ['++', '/good', 'good', '+1']:
            msg = entity.record_somatic_feedback(valence=1.0)
            print(f"🌟 {msg}")
            continue

        # 4. Instant Predictive Teaching Step
        elif cmd.startswith('/teach'):
            target_text = user_input[len('/teach'):].strip()
            if not target_text:
                print("⚠️ Usage: /teach <desired_response_text>")
                continue
            print(f"🧠 Teaching Karyon: '{target_text}'...")
            loss_val = entity.teach_predictive_step(target_text)
            print(f"✅ Predictive coding step complete. Free Energy Loss on target: {loss_val:.4f}")
            continue

        # 5. Volitional Deep Sleep Consolidation
        elif cmd in ['/sleep', 'sleep']:
            print("🌙 Karyon is entering 3-Phase Sleep Consolidation (NREM Replay + SHY Downscaling)...")
            res = entity.execute_sleep_cycle()
            entity.save(filepath=kcore_target_path)
            print(f"☀️ Karyon awakens renewed! Energy restored to 1.00. Pruned nodes: {res['pruned_nodes']} (in {res['duration_ms']:.1f}ms).")
            continue

        # 6. Status Query
        elif cmd in ['/status', 'status']:
            render_affective_dashboard(entity)
            continue

        # 7. Help Query
        elif cmd in ['/help', 'help']:
            print_help_menu()
            continue

        # 8. Standard Dialogue Turn
        karyon_response = entity.step(user_input, thinking_steps=4, max_new_tokens=48)
        print(f"Karyon: {karyon_response}")


if __name__ == "__main__":
    run_interactive_session()
