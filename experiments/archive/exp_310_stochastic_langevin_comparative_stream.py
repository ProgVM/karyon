import os
import random
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from karyon_agent import CoREAgent  # noqa: E402


def compute_repetition_rate(tokens: list, n: int = 2) -> float:
    """
    Computes n-gram repetition rate: 1.0 - (unique_ngrams / total_ngrams).
    Higher values indicate severe limit cycle perseveration / looping.
    """
    if len(tokens) <= n:
        return 0.0
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    unique_ngrams = len(set(ngrams))
    total_ngrams = len(ngrams)
    return float(1.0 - (unique_ngrams / total_ngrams))


def sample_speech(agent, prompt: str, length: int = 60, temperature: float = 0.5) -> str:
    """
    Diagnostic Speech Sampling under Top-p / temperature decoding (KEP Rule #4).
    """
    prompt_bytes = list(prompt.encode('utf-8'))
    agent.eval()
    with torch.no_grad():
        generated_bytes = list(prompt_bytes)
        for _ in range(length):
            inp = torch.tensor([generated_bytes], dtype=torch.long, device=agent.device)
            # Forward through agent
            logits = agent(inp)[:, -1, :256]  # vocab bytes
            probs = torch.softmax(logits / max(temperature, 1e-4), dim=-1)
            # Sample next byte
            next_byte = torch.multinomial(probs, num_samples=1).item()
            generated_bytes.append(next_byte)
    agent.train()
    # Decode to UTF-8 with error replacement
    return bytes(generated_bytes).decode('utf-8', errors='replace')


def run_exp_310_stochastic_langevin_comparative_stream():
    print("=" * 85)
    print("EXP-310: COMPARATIVE STREAMING BENCHMARK — DETERMINISTIC VS. HARDWARE-LANGEVIN STOCHASTIC")
    print("=" * 85)

    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    print(f"Hardware Compute Device: {device} | Engine: C++20 DynamicMorphicGraph + CoREAgent")

    # 1. Load Real Text Stream Data (Alpaca / Foundations)
    corpus_files = [
        os.path.join(os.path.dirname(__file__), '..', 'KARYON_PHILOSOPHICAL_FOUNDATIONS.md'),
        os.path.join(os.path.dirname(__file__), '..', 'KEP.md')
    ]
    raw_text = ""
    for cp in corpus_files:
        if os.path.exists(cp):
            with open(cp, 'r', encoding='utf-8') as f:
                raw_text += f.read() + "\n\n"
    if len(raw_text) < 1000:
        raw_text = raw_text * 10

    raw_bytes = list(raw_text.encode('utf-8'))
    print(f"Loaded Real Continuous Byte Stream: {len(raw_bytes)} UTF-8 bytes")

    # Stream batch configuration: Single-Pass N=1 streaming chunks
    chunk_len = 32
    num_stream_steps = 750
    embed_dim = 128

    # Fixed seed for data streaming generation
    random.seed(42)
    stream_chunks = []
    for _ in range(num_stream_steps):
        max_idx = max(1, len(raw_bytes) - chunk_len - 1)
        idx = random.randint(0, max_idx)
        chunk = raw_bytes[idx:idx + chunk_len + 1]
        stream_chunks.append(chunk)

    print(f"Prepared {num_stream_steps} Sequential Single-Pass Chunks (Length: {chunk_len} bytes each)")
    print("-" * 85)

    results = {}

    # Run Both Arms: Branch A (Deterministic) vs Branch B (Hardware-Langevin)
    for arm in ["A_Deterministic", "B_HardwareLangevin"]:
        print(f"\n>>> LAUNCHING ARM: {arm} <<<")

        # Reset seeds per arm for fair architectural initialization
        torch.manual_seed(1337)
        random.seed(1337)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(1337)

        agent = CoREAgent(vocab_size=258, embed_dim=embed_dim, device=device_str)
        agent.to(device)
        agent.train()

        if arm == "A_Deterministic":
            # Branch A: Deterministic - strictly 4 standard primitives (Linear, Bilinear, SatAttractor, Hopfield)
            agent.add_node("bilinear_1", "BilinearMultiplicative", is_core=False, initial_alpha=0.5)
            agent.add_node("hopfield_1", "ContinuousHopfield", is_core=False, initial_alpha=0.5)
            langevin_node_idx = None
        else:
            # Branch B: Hardware-Langevin Stochastic Stream
            agent.add_node("bilinear_1", "BilinearMultiplicative", is_core=False, initial_alpha=0.5)
            agent.add_node("hopfield_1", "ContinuousHopfield", is_core=False, initial_alpha=0.5)
            langevin_node_idx = agent.add_node("stochastic_langevin_1", "StochasticLangevin", is_core=False, initial_alpha=0.8)
            print(f"  • Injected StochasticLangevinOp at node index {langevin_node_idx} (gamma=1.0, hardware entropy)")

        print(f"  • Active Topology: {agent.get_topology_manifest()}")

        # Optimizer
        optimizer = optim.AdamW(agent.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        losses = []
        free_energies = []
        perseverations = []
        speech_samples = {}
        sigma_eff_telemetry = []

        start_arm_time = time.time()
        total_tokens_processed = 0

        f_t_running = 1.0  # Initial surprise

        for step in range(1, num_stream_steps + 1):
            chunk = stream_chunks[step - 1]
            inp_t = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
            tgt_t = torch.tensor([chunk[1:]], dtype=torch.long, device=device)

            optimizer.zero_grad()

            # Forward pass
            logits = agent(inp_t, thinking_steps=2)  # [1, S, 258]

            loss = criterion(logits.view(-1, 258), tgt_t.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            optimizer.step()

            loss_val = loss.item()
            losses.append(loss_val)

            # Approximate Variational Free Energy F_t = CrossEntropy + KL divergence proxy
            f_t_running = 0.9 * f_t_running + 0.1 * loss_val
            free_energies.append(f_t_running)
            total_tokens_processed += (len(chunk) - 1)

            # If Stochastic Arm, monitor endogenous sigma_eff dynamics
            if arm == "B_HardwareLangevin":
                param_map = agent.graph.named_parameters_map()
                # Find w_sigma for stochastic node
                w_sigma_key = None
                for k in param_map:
                    if "stochastic_langevin" in k and "w_sigma" in k:
                        w_sigma_key = k
                        break
                if w_sigma_key is not None:
                    with torch.no_grad():
                        w_sig_norm = param_map[w_sigma_key].norm().item()
                        stress_factor = 1.0 + 1.0 * float(torch.tanh(torch.tensor(f_t_running)).item())
                        sigma_eff_val = 0.05 * w_sig_norm * stress_factor
                        sigma_eff_telemetry.append(sigma_eff_val)

            # Periodic Diagnostic Speech Sampling (KEP Rule #4) & Perseveration Check
            if step in [250, 500, 750]:
                sample = sample_speech(agent, prompt="Karyon ", length=64, temperature=0.5)
                speech_samples[step] = sample
                sample_bytes = list(sample.encode('utf-8'))
                rep_rate = compute_repetition_rate(sample_bytes, n=3)
                perseverations.append(rep_rate)
                print(f"  [Step {step:3d}] Loss: {loss_val:.4f} | F_t: {f_t_running:.4f} | 3-Gram Repetition Rate: {rep_rate*100:.1f}%")
                print(f"   Speech Diagnostic: {repr(sample[:60])}...")

        duration = time.time() - start_arm_time
        tok_per_sec = total_tokens_processed / max(duration, 1e-4)
        avg_rep_rate = sum(perseverations) / max(len(perseverations), 1)
        final_loss = sum(losses[-50:]) / 50.0

        results[arm] = {
            "losses": losses,
            "free_energies": free_energies,
            "final_loss": final_loss,
            "speech_samples": speech_samples,
            "perseverations": perseverations,
            "avg_rep_rate": avg_rep_rate,
            "tok_per_sec": tok_per_sec,
            "duration": duration,
            "sigma_eff": sigma_eff_telemetry
        }
        print(f"Arm {arm} Finished in {duration:.2f}s | Throughput: {tok_per_sec:.1f} tok/s | Avg Rep Rate: {avg_rep_rate*100:.2f}%")

    print("\n" + "=" * 85)
    print("=== EXP-310 COMPARATIVE EMPIRICAL SUMMARY SCOREBOARD ===")
    print("=" * 85)

    det = results["A_Deterministic"]
    sto = results["B_HardwareLangevin"]

    print(f"{'Metric':<35} | {'Arm A (Deterministic)':<22} | {'Arm B (Stochastic Langevin)':<25}")
    print("-" * 88)
    print(f"{'Final Convergence Loss':<35} | {det['final_loss']:<22.4f} | {sto['final_loss']:<25.4f}")
    print(f"{'Final Free Energy F_t':<35} | {det['free_energies'][-1]:<22.4f} | {sto['free_energies'][-1]:<25.4f}")
    print(f"{'Avg 3-Gram Repetition Rate':<35} | {det['avg_rep_rate']*100:<21.2f}% | {sto['avg_rep_rate']*100:<24.2f}%")
    print(f"{'Perseveration Collapse Drop':<35} | {'Reference':<22} | {(det['avg_rep_rate'] - sto['avg_rep_rate'])*100:<+24.2f}%")
    print(f"{'Throughput (tok/sec)':<35} | {det['tok_per_sec']:<22.1f} | {sto['tok_per_sec']:<25.1f}")
    print(f"{'Hardware Box-Muller Overhead':<35} | {'Baseline':<22} | {((det['tok_per_sec'] - sto['tok_per_sec'])/det['tok_per_sec'])*100:<+24.2f}%")

    print("\n[DIAGNOSTIC SPEECH COMPARISON ACROSS HORIZONS]")
    for st in [250, 500, 750]:
        print(f"\n--- Checkpoint Step {st} ---")
        print(f"  • Det (A) : {repr(det['speech_samples'][st])}")
        print(f"  • Sto (B) : {repr(sto['speech_samples'][st])}")

    if sto['sigma_eff']:
        min_sig = min(sto['sigma_eff'])
        max_sig = max(sto['sigma_eff'])
        mean_sig = sum(sto['sigma_eff']) / len(sto['sigma_eff'])
        print("\n[ENDOGENOUS LANGEVIN NOISE MODULATION TELEMETRY]")
        print(f"  • Min sigma_eff  : {min_sig:.5f} (Quiet low-surprise state)")
        print(f"  • Max sigma_eff  : {max_sig:.5f} (High-surprise arousal burst)")
        print(f"  • Mean sigma_eff : {mean_sig:.5f}")

    return results


if __name__ == '__main__':
    run_exp_310_stochastic_langevin_comparative_stream()
