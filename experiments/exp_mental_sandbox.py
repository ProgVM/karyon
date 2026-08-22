# experiments/exp_mental_sandbox.py
"""
===============================================================================
KARYON ENGINEERING PROTOCOL (KEP) EXPERIMENT #1
Topic: Latent Mental Sandbox / Counterfactual Rollouts via LatentPredictor
Author: Karyon-CoRE Open-Source Research Team (2026)
===============================================================================

Hypothesis:
    Executing K internal mental rollout steps within the Active Inference Latent 
    Predictor (world model) prior to motor/speech generation reduces Variational 
    Free Energy (F_t) and doubles target byte confidence when prior context exists.

Control Group:
    Direct Generation (K=0 Mental Rollout Steps).

Experimental Group:
    Latent Sandbox Simulation (K=1, K=3, K=5 Internal Rollout Steps).

Metrics Tracked:
    1. Variational Free Energy (F_t)
    2. Target Byte Probability (%)
    3. Top-1 Predicted Byte ID / ASCII Symbol
    4. Execution Latency (ms)
===============================================================================
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# Set global seed for exact reproducibility on Kaggle GPU instances
torch.manual_seed(42)

# Select execution hardware
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[KEP Experiment #1] Execution Context: {device.type.upper()}")


# =============================================================================
# 1. STANDALONE PROTOTYPE MODULES (Matching Production Karyon Architecture)
# =============================================================================

class PrototypeLatentPredictor(nn.Module):
    """
    Active Inference World Model predicting latent representations (w_pred) 
    and computing Variational Free Energy (F_t = KL + Reconstruction Loss).
    """
    def __init__(self, hidden_dim=256, unified_dim=128, latent_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.unified_dim = unified_dim
        self.latent_dim = latent_dim

        self.prior_net = nn.Linear(hidden_dim, latent_dim * 2)
        self.posterior_net = nn.Linear(hidden_dim + unified_dim, latent_dim * 2)
        
        self.decoder_net = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, unified_dim * 2),
            nn.SiLU(),
            nn.Linear(unified_dim * 2, unified_dim)
        )

    def forward(self, h_fast_prev, h_slow_curr, w_t):
        """Calculates predicted sensory representation and Free Energy."""
        # Prior distribution parameters
        prior_out = self.prior_net(h_fast_prev)
        mu_prior, logvar_prior = prior_out.chunk(2, dim=-1)
        logvar_prior = torch.clamp(logvar_prior, -10.0, 10.0)

        # Posterior distribution parameters
        post_input = torch.cat([h_fast_prev, w_t], dim=-1)
        post_out = self.posterior_net(post_input)
        mu_post, logvar_post = post_out.chunk(2, dim=-1)
        logvar_post = torch.clamp(logvar_post, -10.0, 10.0)

        # Reparameterization trick
        std_post = torch.exp(0.5 * logvar_post)
        eps = torch.randn_like(std_post)
        z_t = mu_post + eps * std_post

        # Decode predicted sensory vector w_pred
        dec_input = torch.cat([z_t, h_slow_curr], dim=-1)
        w_pred = self.decoder_net(dec_input)

        # Variational Free Energy calculation
        var_prior = torch.exp(logvar_prior) + 1e-7
        var_post = torch.exp(logvar_post) + 1e-7

        kl_div = 0.5 * torch.mean(
            logvar_prior - logvar_post + (var_post + (mu_post - mu_prior)**2) / var_prior - 1.0,
            dim=-1, keepdim=True
        )
        rec_loss = torch.mean((w_t - w_pred)**2, dim=-1, keepdim=True)
        free_energy = kl_div + rec_loss

        return w_pred, kl_div, free_energy


class PrototypeMentalSandboxAgent(nn.Module):
    """
    Self-contained prototype agent simulating Karyon's Latent Sandbox Rollouts.
    """
    def __init__(self, vocab_size=258, text_dim=128, hidden_dim=256, unified_dim=128, latent_dim=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.unified_dim = unified_dim

        self.token_embeddings = nn.Embedding(vocab_size, text_dim)
        self.text_proj = nn.Linear(text_dim, unified_dim)

        self.rnn_fast = nn.GRUCell(unified_dim, hidden_dim)
        self.rnn_slow = nn.GRUCell(hidden_dim, hidden_dim)

        self.world_model = PrototypeLatentPredictor(hidden_dim, unified_dim, latent_dim)
        self.text_head = nn.Linear(hidden_dim, vocab_size)

    def encode_text(self, text: str):
        """Encodes UTF-8 string into byte IDs terminated by EOS (257)."""
        bytes_list = [ord(c) if ord(c) < 256 else 63 for c in text]
        bytes_list.append(257) # EOS token
        return torch.tensor(bytes_list, dtype=torch.long, device=device)

    def step_perception(self, token_id, h_fast, h_slow):
        """Single perception step through sensory projection and recurrent core."""
        t_emb = self.token_embeddings(token_id)
        w_current = F.silu(self.text_proj(t_emb))

        h_f_next = self.rnn_fast(w_current, h_fast)
        h_s_next = self.rnn_slow(h_f_next, h_slow)

        w_pred, kl_div, free_energy = self.world_model(h_fast, h_s_next, w_current)
        logits = self.text_head(h_f_next + h_s_next)

        return h_f_next, h_s_next, w_current, logits, free_energy

    def run_latent_rollout(self, h_fast, h_slow, w_current, k_steps: int):
        """
        Executes K internal mental rollout steps in LatentPredictor without motor output.
        """
        h_f, h_s = h_fast.clone(), h_slow.clone()
        w_curr = w_current.clone()

        for _ in range(k_steps):
            # Predict next latent state using World Model
            w_pred, _, _ = self.world_model(h_f, h_s, w_curr)
            # Update internal fast and slow recurrent states
            h_f = self.rnn_fast(w_pred, h_f)
            h_s = self.rnn_slow(h_f, h_s)
            w_curr = w_pred

        return h_f, h_s


# =============================================================================
# 2. EXPERIMENTAL BENCHMARK EVALUATION
# =============================================================================

def run_mental_sandbox_experiment():
    """Evaluates K=0, K=1, K=3, K=5 rollout steps on diagnostic prompt benchmarks."""
    agent = PrototypeMentalSandboxAgent().to(device)
    agent.eval()

    test_samples = [
        {"prompt": "User: What is the primary source of energy for Earth?\nKaryon:", "target_byte": ord('T')},
        {"prompt": "User: If you mix red and blue, you get\nKaryon:", "target_byte": ord('p')},
        {"prompt": "User: 2 + 2 =\nKaryon:", "target_byte": ord('4')}
    ]

    rollout_k_values = [0, 1, 3, 5]
    summary_telemetry = []

    print("\n" + "="*80)
    print(" === KARYON ENGINEERING EXPERIMENT #1: MENTAL SANDBOX BENCHMARK ===")
    print("="*80)

    for idx, sample in enumerate(test_samples):
        prompt_text = sample["prompt"]
        target_id = sample["target_byte"]
        tokens = agent.encode_text(prompt_text)

        print(f"\nSample #{idx+1}: '{prompt_text.strip()}' (Target Byte: '{chr(target_id)}' / ID: {target_id})")
        print("-" * 80)
        print(f"{'Rollout Steps':<15} | {'Free Energy (F_t)':<20} | {'Target Prob (%)':<18} | {'Top-1 Token':<12} | {'Latency (ms)':<12}")
        print("-" * 80)

        for k_steps in rollout_k_values:
            h_f = torch.zeros(1, agent.hidden_dim, device=device)
            h_s = torch.zeros(1, agent.hidden_dim, device=device)

            start_time = time.perf_counter()

            with torch.no_grad():
                # Process prompt context through perception
                for t_id in tokens[:-1]:
                    h_f, h_s, w_curr, _, _ = agent.step_perception(t_id.unsqueeze(0), h_f, h_s)

                # Execute Mental Sandbox Rollouts (If K > 0)
                if k_steps > 0:
                    h_f, h_s = agent.run_latent_rollout(h_f, h_s, w_curr, k_steps)

                # Final motor speech generation step
                last_token = tokens[-1]
                h_f, h_s, _, logits, free_energy = agent.step_perception(last_token.unsqueeze(0), h_f, h_s)

                # Calculate confidence metrics
                probs = F.softmax(logits, dim=-1)
                target_prob = probs[0, target_id].item() * 100.0
                top1_id = torch.argmax(probs, dim=-1).item()
                top1_char = chr(top1_id) if 32 <= top1_id <= 126 else '.'

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            summary_telemetry.append({
                "sample_idx": idx,
                "k_steps": k_steps,
                "free_energy": free_energy.mean().item(),
                "target_prob": target_prob,
                "latency_ms": elapsed_ms
            })

            print(f"{'K=' + str(k_steps):<15} | {free_energy.mean().item():<20.4f} | {target_prob:<18.2f}% | {top1_char + ' (' + str(top1_id) + ')':<12} | {elapsed_ms:<12.2f}")

    # =========================================================================
    # KEP EVALUATION & VERDICT LOGIC
    # =========================================================================
    print("\n" + "="*80)
    print(" === KEP EVALUATION & VERDICT ===")
    print("="*80)

    # Analyze average Free Energy reduction for K=1 vs K=0 across samples with context
    k0_fe = [item["free_energy"] for item in summary_telemetry if item["k_steps"] == 0]
    k1_fe = [item["free_energy"] for item in summary_telemetry if item["k_steps"] == 1]
    
    avg_k0_fe = sum(k0_fe) / len(k0_fe)
    avg_k1_fe = sum(k1_fe) / len(k1_fe)
    fe_reduction = ((avg_k0_fe - avg_k1_fe) / avg_k0_fe) * 100.0

    if avg_k1_fe < avg_k0_fe:
        print("🟢 VERDICT: POSITIVE EXPERIENCE ADOPTED (WITH EXPECTED FREE ENERGY GATING)!")
        print(f"   Reason: K=1 Mental Rollout steps reduce average Free Energy F_t by {fe_reduction:.2f}%.")
        print("   Action: Integrate Latent Sandbox into production, gated by Expected Free Energy G!")
    else:
        print("🔴 VERDICT: NEGATIVE EXPERIENCE REJECTED.")
        print("   Reason: Rollout steps added internal noise without reducing Free Energy.")
    print("="*80 + "\n")


# =============================================================================
# 3. MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    run_mental_sandbox_experiment()
