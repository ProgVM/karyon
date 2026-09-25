"""
===============================================================================
EXP-290: Dual-Phase Recurrent Working Engine (DPR-WE) with Holographic Saccadic Gaze
Hypothesis:
Decoupling problem comprehension from motor generation via:
1. Phase 1: Bidirectional Cellular Wave Deliberation (6 weight-tied local cellular diffusion cycles)
2. Phase 2: Recurrent Saccadic Multi-Head Gaze Motor Decoder
solves algorithmic reasoning (Dyck-1, Reversal, Pointers) without quadratic attention overhead.

Benchmark: Multi-Domain Algorithmic Suite (400 samples/domain x 5 domains = 2000 evaluation instances)
===============================================================================
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math, random, time, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from multi_domain_benchmark import generate_multi_domain_suite

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 80)
print("EXP-290: DUAL-PHASE RECURRENT WORKING ENGINE (DPR-WE) SCIENTIFIC BENCHMARK")
print(f"Hardware Compute Device: {device} | CUDA Available: {torch.cuda.is_available()}")
print("=" * 80)

# ==============================================================================
# Model Architecture: Dual-Phase Recurrent Working Engine (DPR-WE)
# ==============================================================================

class MultiHeadSaccadicMind(nn.Module):
    def __init__(self, vocab_size=258, dim=128, num_heads=4, wave_steps=6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(512, dim)
        nn.init.normal_(self.emb.weight, 0.0, 1.0 / math.sqrt(dim))
        nn.init.normal_(self.pos_emb.weight, 0.0, 1.0 / math.sqrt(dim))
        
        # Phase 1: Deep Cellular Wave Deliberation (Local Spatial Diffusion)
        self.wave_conv = nn.Conv1d(dim, dim * 2, kernel_size=3, padding=1, bias=False)
        self.wave_swi_g = nn.Linear(dim, dim, bias=False)
        self.wave_swi_v = nn.Linear(dim, dim, bias=False)
        self.wave_swi_o = nn.Linear(dim, dim, bias=False)
        self.wave_norm = nn.RMSNorm(dim)
        self.wave_steps = wave_steps
        
        # Phase 2: Multi-Head Saccadic Gaze Motor Decoder
        self.gru = nn.GRUCell(dim, dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_gate = nn.Linear(dim * 2, dim, bias=False)
        self.norm_motor = nn.RMSNorm(dim)
        
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.emb.weight # Synaptic weight-tying

    def deliberate_prompt(self, p_tokens):
        B, P = p_tokens.shape
        pos = torch.arange(P, device=p_tokens.device).unsqueeze(0)
        h = self.emb(p_tokens) + self.pos_emb(pos)
        for _ in range(self.wave_steps):
            wave_feat = self.wave_conv(h.transpose(1, 2)).transpose(1, 2)
            g, v = wave_feat.chunk(2, dim=-1)
            wave = F.silu(g) * v
            swi = self.wave_swi_o(F.silu(self.wave_swi_g(h)) * self.wave_swi_v(h))
            h = self.wave_norm(h + 0.5 * (wave + swi))
        return h

    def step(self, x_t, h_prev, p_field):
        B, P, D = p_field.shape
        H = self.num_heads
        d_k = self.head_dim
        
        h_next = self.gru(x_t, h_prev) # [B, D]
        
        q = self.q_proj(h_next).view(B, H, 1, d_k)
        k = self.k_proj(p_field).view(B, P, H, d_k).transpose(1, 2)
        v = self.v_proj(p_field).view(B, P, H, d_k).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
        attn = F.softmax(scores, dim=-1)
        gaze_ctx = torch.matmul(attn, v).view(B, D)
        
        fused = self.out_gate(torch.cat([h_next, gaze_ctx], dim=-1))
        out = self.norm_motor(h_next + fused)
        logits = self.head(out)
        return logits, h_next

    def forward_train(self, p_tokens, a_tokens):
        B, A = a_tokens.shape
        p_field = self.deliberate_prompt(p_tokens)
        h_prev = p_field.mean(dim=1)
        
        logits_list = []
        for t in range(A):
            x_t = self.emb(a_tokens[:, t])
            logits_t, h_prev = self.step(x_t, h_prev, p_field)
            logits_list.append(logits_t)
            
        return torch.stack(logits_list, dim=1)

    def generate(self, p_tokens, max_len=15):
        B = p_tokens.shape[0]
        p_field = self.deliberate_prompt(p_tokens)
        h_prev = p_field.mean(dim=1)
        
        curr_token = torch.tensor([257] * B, dtype=torch.long, device=p_tokens.device)
        generated = []
        
        for _ in range(max_len):
            x_t = self.emb(curr_token)
            logits_t, h_prev = self.step(x_t, h_prev, p_field)
            nxt = torch.argmax(logits_t, dim=-1)
            generated.append(nxt)
            curr_token = nxt
            
        return torch.stack(generated, dim=1)

def run():
    suite_train = generate_multi_domain_suite(seed=42)
    suite_test = generate_multi_domain_suite(seed=999)
    
    all_train = []
    for d, s in suite_train.items():
        all_train.extend(s)
    random.seed(42)
    random.shuffle(all_train)
    
    def encode_data(samples):
        max_p = max(len(p) for p, _, _ in samples)
        max_a = max(len(a) for _, a, _ in samples) + 1
        P = torch.full((len(samples), max_p), 256, dtype=torch.long, device=device)
        A_in = torch.full((len(samples), max_a), 256, dtype=torch.long, device=device)
        A_tgt = torch.full((len(samples), max_a), 256, dtype=torch.long, device=device)
        for i, (p, a, _) in enumerate(samples):
            p_b = [ord(c) for c in p]
            a_b = [ord(c) for c in a] + [ord('\n')]
            P[i, :len(p_b)] = torch.tensor(p_b, dtype=torch.long, device=device)
            a_in_seq = [257] + a_b[:-1]
            A_in[i, :len(a_in_seq)] = torch.tensor(a_in_seq, dtype=torch.long, device=device)
            A_tgt[i, :len(a_b)] = torch.tensor(a_b, dtype=torch.long, device=device)
        return P, A_in, A_tgt

    P_tr, Ain_tr, Atgt_tr = encode_data(all_train)
    
    agent = MultiHeadSaccadicMind(vocab_size=258, dim=128, num_heads=4, wave_steps=6).to(device)
    num_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    print(f"Instantiated MultiHeadSaccadicMind (DPR-WE). Total Parameters: {num_params:,}")
    
    opt = torch.optim.AdamW(agent.parameters(), lr=0.003, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=150, eta_min=1e-4)
    
    print("\n--- Training MultiHeadSaccadicMind Jointly (150 Epochs) ---")
    t0 = time.time()
    for ep in range(1, 151):
        agent.train()
        idx = torch.randperm(len(P_tr))[:128]
        p_b = P_tr[idx]
        ain_b = Ain_tr[idx]
        atgt_b = Atgt_tr[idx]
        
        opt.zero_grad()
        logits = agent.forward_train(p_b, ain_b)
        loss = F.cross_entropy(logits.reshape(-1, 258), atgt_b.reshape(-1), ignore_index=256)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        opt.step()
        scheduler.step()
        
        if ep % 30 == 0 or ep == 1:
            print(f"Epoch {ep:03d} | CrossEntropy: {loss.item():.4f} nats | LR: {scheduler.get_last_lr()[0]:.6f}")

    train_duration = time.time() - t0
    print(f"\nTraining completed in {train_duration:.2f}s.")
    
    agent.eval()
    print("\n" + "=" * 80)
    print("=== EXP-290 MULTI-DOMAIN EVALUATION RESULTS ===")
    print("=" * 80)
    
    accs = {}
    for domain, samples in suite_test.items():
        correct = 0
        total = len(samples)
        sample_preds = []
        for p, exp, _ in samples:
            p_b = [ord(c) for c in p]
            curr_p = torch.tensor([p_b], dtype=torch.long, device=device)
            with torch.no_grad():
                gen_tokens = agent.generate(curr_p, max_len=len(exp)+2)[0].tolist()
                out_b = []
                for t in gen_tokens:
                    if t in (ord('\n'), 256, 257):
                        break
                    out_b.append(t)
                got = bytes(out_b).decode('utf-8', errors='ignore')
                is_match = (got.strip() == exp.strip())
                if is_match:
                    correct += 1
                if len(sample_preds) < 3:
                    sample_preds.append((p, exp, got, is_match))
                    
        acc = (correct / total) * 100.0
        accs[domain] = acc
        print(f"\nDomain [{domain.upper():10s}]: Accuracy = {acc:.2f}% ({correct}/{total})")
        for p, exp, got, ok in sample_preds:
            status = "✅ PASS" if ok else "❌ FAIL"
            print(f"  {status} | Prompt: {p!r:30s} | Exp: {exp!r:10s} | Got: {got!r:10s}")

    mean_acc = sum(accs.values()) / len(accs)
    print("\n" + "=" * 80)
    print("=== FINAL EXPERIMENTAL SUMMARY ===")
    print(f"Final CrossEntropy Loss : {loss.item():.4f} nats")
    print(f"Dyck-1 Accuracy         : {accs.get('dyck', 0.0):.2f}%")
    print(f"Pointer Accuracy        : {accs.get('pointer', 0.0):.2f}%")
    print(f"Reversal Accuracy       : {accs.get('reversal', 0.0):.2f}%")
    print(f"Parity Accuracy         : {accs.get('parity', 0.0):.2f}%")
    print(f"Addition Accuracy       : {accs.get('addition', 0.0):.2f}%")
    print(f"Mean Benchmark Accuracy : {mean_acc:.2f}%")
    print("=" * 80)
    
    # KEP Rule #2 Contextual Verdict
    verdict = "POSITIVE" if accs.get('dyck', 0.0) >= 80.0 and mean_acc >= 30.0 else "NEUTRAL"
    status_emoji = "🟢" if verdict == "POSITIVE" else "⚪"
    print(f"\nFinal Verdict: {status_emoji} {verdict}")
    
    results = {
        "exp_id": "EXP-290",
        "final_loss": loss.item(),
        "dyck_accuracy": accs.get('dyck', 0.0),
        "pointer_accuracy": accs.get('pointer', 0.0),
        "reversal_accuracy": accs.get('reversal', 0.0),
        "parity_accuracy": accs.get('parity', 0.0),
        "addition_accuracy": accs.get('addition', 0.0),
        "mean_accuracy": mean_acc,
        "verdict": verdict
    }
    with open("experiments/exp_290_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run()
