import sys, os
sys.path.insert(0, '.')
import math, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import karyon_core
from multi_domain_benchmark import generate_multi_domain_suite

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# Dual-Phase Recurrent Working Engine (DPR-WE) with Saccadic Pointer Gaze
# 1. Phase 1: Problem Ingestion + Weight-Tied 1D Cellular Wave (4 cycles)
# 2. Phase 2: Saccadic Attractor Controller with Relative Shifts [-3..+3] + Copy Gate
# ==============================================================================

class SaccadicPointerController(nn.Module):
    def __init__(self, dim=128, num_shifts=7):
        super().__init__()
        self.dim = dim
        self.num_shifts = num_shifts # e.g. -3, -2, -1, 0, +1, +2, +3
        self.shift_offsets = list(range(-(num_shifts // 2), num_shifts // 2 + 1))
        
        self.gru = nn.GRUCell(dim, dim)
        self.shift_head = nn.Linear(dim, num_shifts, bias=False)
        self.content_q = nn.Linear(dim, dim, bias=False)
        self.content_k = nn.Linear(dim, dim, bias=False)
        
        self.gaze_gate = nn.Linear(dim, 1, bias=True) # Balance between content and saccadic pointer
        self.copy_gate = nn.Linear(dim, 1, bias=True) # Pointing / Copying vs Vocabulary generation
        self.out_proj = nn.Linear(dim * 2, dim, bias=False)
        self.norm = nn.RMSNorm(dim)

    def forward(self, x_t, h_prev, ptr_dist_prev, p_field, p_tokens):
        # x_t: [B, D]
        # h_prev: [B, D]
        # ptr_dist_prev: [B, P] (pointer probability distribution over prompt positions)
        # p_field: [B, P, D]
        # p_tokens: [B, P]
        B, P, D = p_field.shape
        
        h_next = self.gru(x_t, h_prev) # [B, D]
        
        # 1. Saccadic Relative Shift: P_new(i) = sum_k shift(k) * P_prev(i - k)
        shift_logits = self.shift_head(h_next) # [B, num_shifts]
        shift_probs = F.softmax(shift_logits, dim=-1) # [B, num_shifts]
        
        # Convolve ptr_dist_prev with shift_probs
        # Circular / bounded shift
        shifted_ptr = torch.zeros_like(ptr_dist_prev)
        for idx, offset in enumerate(self.shift_offsets):
            prob = shift_probs[:, idx].unsqueeze(1) # [B, 1]
            if offset > 0:
                shifted = torch.cat([torch.zeros(B, offset, device=p_field.device), ptr_dist_prev[:, :-offset]], dim=1)
            elif offset < 0:
                shifted = torch.cat([ptr_dist_prev[:, -offset:], torch.zeros(B, -offset, device=p_field.device)], dim=1)
            else:
                shifted = ptr_dist_prev
            shifted_ptr = shifted_ptr + prob * shifted
            
        # Renormalize shifted pointer
        shifted_ptr = shifted_ptr / (shifted_ptr.sum(dim=-1, keepdim=True) + 1e-6)
        
        # 2. Content-Addressable Attention
        q = self.content_q(h_next).unsqueeze(1) # [B, 1, D]
        k = self.content_k(p_field) # [B, P, D]
        content_scores = torch.bmm(q, k.transpose(1, 2)).squeeze(1) / math.sqrt(D) # [B, P]
        content_probs = F.softmax(content_scores, dim=-1)
        
        # 3. Dynamic Fusion of Gaze Pointer (Saccadic + Content)
        alpha_gaze = torch.sigmoid(self.gaze_gate(h_next)) # [B, 1]
        final_ptr = alpha_gaze * shifted_ptr + (1.0 - alpha_gaze) * content_probs # [B, P]
        
        # Read attended sensory state
        gaze_feature = torch.bmm(final_ptr.unsqueeze(1), p_field).squeeze(1) # [B, D]
        
        # Direct Copy Distribution over Vocab
        p_copy = torch.sigmoid(self.copy_gate(h_next)) # [B, 1]
        copy_logits = torch.zeros(B, 258, device=p_field.device)
        copy_logits.scatter_add_(1, p_tokens, final_ptr)
        
        out_state = self.norm(h_next + self.out_proj(torch.cat([h_next, gaze_feature], dim=-1)))
        return out_state, h_next, final_ptr, p_copy, copy_logits

class DPRWEAgent(nn.Module):
    def __init__(self, vocab_size=258, dim=128, wave_steps=4):
        super().__init__()
        self.dim = dim
        self.emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(512, dim)
        nn.init.normal_(self.emb.weight, 0.0, 1.0 / math.sqrt(dim))
        nn.init.normal_(self.pos_emb.weight, 0.0, 1.0 / math.sqrt(dim))
        
        # Phase 1: Cellular Wave Deliberation
        self.wave_conv = nn.Conv1d(dim, dim * 2, kernel_size=3, padding=1, bias=False)
        self.wave_swi_g = nn.Linear(dim, dim, bias=False)
        self.wave_swi_v = nn.Linear(dim, dim, bias=False)
        self.wave_swi_o = nn.Linear(dim, dim, bias=False)
        self.wave_norm = nn.RMSNorm(dim)
        self.wave_steps = wave_steps
        
        # Phase 2: Saccadic Controller
        self.controller = SaccadicPointerController(dim, num_shifts=7)
        self.gen_head = nn.Linear(dim, vocab_size, bias=False)
        self.gen_head.weight = self.emb.weight

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

    def forward_train(self, p_tokens, a_tokens):
        B, A = a_tokens.shape
        B, P = p_tokens.shape
        p_field = self.deliberate_prompt(p_tokens)
        h_prev = p_field.mean(dim=1)
        
        # Initial pointer focused on the last token of prompt (the '=')
        ptr_dist = torch.zeros(B, P, device=p_tokens.device)
        ptr_dist[:, -1] = 1.0
        
        logits_list = []
        for t in range(A):
            x_t = self.emb(a_tokens[:, t])
            out_t, h_prev, ptr_dist, p_copy, copy_dist = self.controller(x_t, h_prev, ptr_dist, p_field, p_tokens)
            gen_dist = F.softmax(self.gen_head(out_t), dim=-1)
            
            # Mixture of Generation and Pointer-Copy
            mixture_dist = (1.0 - p_copy) * gen_dist + p_copy * copy_dist
            log_probs = torch.log(mixture_dist.clamp(min=1e-9))
            logits_list.append(log_probs)
            
        return torch.stack(logits_list, dim=1) # [B, A, V] (Log probabilities)

    def generate(self, p_tokens, max_len=15):
        B, P = p_tokens.shape
        p_field = self.deliberate_prompt(p_tokens)
        h_prev = p_field.mean(dim=1)
        
        ptr_dist = torch.zeros(B, P, device=p_tokens.device)
        ptr_dist[:, -1] = 1.0
        
        curr_token = torch.tensor([257] * B, dtype=torch.long, device=p_tokens.device)
        generated = []
        
        for _ in range(max_len):
            x_t = self.emb(curr_token)
            out_t, h_prev, ptr_dist, p_copy, copy_dist = self.controller(x_t, h_prev, ptr_dist, p_field, p_tokens)
            gen_dist = F.softmax(self.gen_head(out_t), dim=-1)
            mixture_dist = (1.0 - p_copy) * gen_dist + p_copy * copy_dist
            nxt = torch.argmax(mixture_dist, dim=-1)
            generated.append(nxt)
            curr_token = nxt
            
        return torch.stack(generated, dim=1)

# Training test on all 5 domains
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

agent = DPRWEAgent(vocab_size=258, dim=128, wave_steps=4).to(device)
opt = torch.optim.AdamW(agent.parameters(), lr=0.003, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-4)

print("\n--- Training DPRWEAgent (100 Epochs) ---")
for ep in range(1, 101):
    agent.train()
    idx = torch.randperm(len(P_tr))[:128]
    p_b = P_tr[idx]
    ain_b = Ain_tr[idx]
    atgt_b = Atgt_tr[idx]
    
    opt.zero_grad()
    log_probs = agent.forward_train(p_b, ain_b)
    loss = F.nll_loss(log_probs.reshape(-1, 258), atgt_b.reshape(-1), ignore_index=256)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
    opt.step()
    scheduler.step()
    
    if ep % 20 == 0 or ep == 1:
        print(f"Epoch {ep:03d} | Loss: {loss.item():.4f}")

agent.eval()
print("\n" + "=" * 80)
print("=== DPR-WE SOVEREIGN MULTI-DOMAIN EVALUATION ===")
print("=" * 80)

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
    print(f"\nDomain [{domain.upper():10s}]: Accuracy = {acc:.2f}% ({correct}/{total})")
    for p, exp, got, ok in sample_preds:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status} | Prompt: {p!r:30s} | Exp: {exp!r:10s} | Got: {got!r:10s}")
