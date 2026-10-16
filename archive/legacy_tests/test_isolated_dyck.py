"""
Minimal Isolated Diagnostic Unit-Test: Dyck-1 Language Recognition
Direct Stack-Potential Closed Loop vs Dyck-1 Grammar.

Target: Isolate the stack integrator, close the feedback loop directly,
and verify if Dyck-1 can be learned to >90% accuracy without blurring.
"""

import sys, os
sys.path.insert(0, '.')
import random, time
import torch
import torch.nn as nn
import torch.nn.functional as F

from multi_domain_benchmark import generate_multi_domain_suite

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class IsolatedDyckEngine(nn.Module):
    def __init__(self, vocab_size=258, dim=64):
        super().__init__()
        self.dim = dim
        self.emb = nn.Embedding(vocab_size, dim)
        
        # 1. Continuous Stack Accumulator (Leaky / Linear Integrator)
        # Token '(' -> +1, Token ')' -> -1, Violation Latch -> < 0
        self.w_stack_step = nn.Linear(dim, 1, bias=False)
        self.decay_param = nn.Parameter(torch.tensor(0.0)) # sigmoid -> leak
        
        # 2. Sequential Hidden State
        self.gru = nn.GRUCell(dim + 1, dim)
        
        # 3. Direct Readout Coupling Stack + Hidden State
        self.head = nn.Linear(dim + 1, vocab_size)

    def forward_loss(self, prompt_tokens, target_tokens):
        B, P_len = prompt_tokens.shape
        _, T_len = target_tokens.shape
        
        # Phase 1: Encode Prompt and track stack potential
        h = torch.zeros(B, self.dim, device=prompt_tokens.device)
        stack = torch.zeros(B, 1, device=prompt_tokens.device)
        min_stack = torch.zeros(B, 1, device=prompt_tokens.device)
        
        for t in range(P_len):
            tok = prompt_tokens[:, t]
            x = self.emb(tok)
            # Stack delta
            d_stack = self.w_stack_step(x)
            stack = stack + d_stack
            min_stack = torch.minimum(min_stack, stack)
            # Recurrent step with stack awareness
            h = self.gru(torch.cat([x, stack], dim=-1), h)
            
        # Target Generation
        inputs = torch.cat([prompt_tokens[:, -1:], target_tokens[:, :-1]], dim=1)
        all_logits = []
        for t in range(T_len):
            tok = inputs[:, t]
            x = self.emb(tok)
            h = self.gru(torch.cat([x, stack], dim=-1), h)
            # Direct projection using stack state!
            logits = self.head(torch.cat([h, stack], dim=-1))
            all_logits.append(logits)
            
        logits_stack = torch.stack(all_logits, dim=1)
        loss = F.cross_entropy(logits_stack.reshape(-1, 258), target_tokens.reshape(-1))
        return loss

    @torch.no_grad()
    def generate(self, prompt_tokens, max_len=10):
        B, P_len = prompt_tokens.shape
        h = torch.zeros(B, self.dim, device=prompt_tokens.device)
        stack = torch.zeros(B, 1, device=prompt_tokens.device)
        
        for t in range(P_len):
            tok = prompt_tokens[:, t]
            x = self.emb(tok)
            d_stack = self.w_stack_step(x)
            stack = stack + d_stack
            h = self.gru(torch.cat([x, stack], dim=-1), h)
            
        curr_token = prompt_tokens[:, -1]
        generated = []
        for _ in range(max_len):
            x = self.emb(curr_token)
            h = self.gru(torch.cat([x, stack], dim=-1), h)
            logits = self.head(torch.cat([h, stack], dim=-1))
            next_token = logits.argmax(dim=-1)
            generated.append(next_token)
            curr_token = next_token
            if (curr_token == 10).all():
                break
        return torch.stack(generated, dim=1)

def test_isolated_dyck():
    suite = generate_multi_domain_suite(seed=42)
    dyck_data = suite['dyck']
    print(f"Total Dyck samples: {len(dyck_data)}")
    
    model = IsolatedDyckEngine(vocab_size=258, dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    
    train_data = dyck_data[:300]
    test_data = dyck_data[300:400]
    
    batch_size = 32
    print("Training Isolated Dyck Engine with closed stack loop...")
    for epoch in range(1, 61):
        model.train()
        random.shuffle(train_data)
        total_loss = 0.0
        batches = 0
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            prompts = [list(item[0].encode('utf-8')) for item in batch]
            targets = [list(item[2][len(item[0]):].encode('utf-8')) for item in batch]
            
            max_p = max(len(p) for p in prompts)
            max_t = max(len(t) for t in targets)
            
            p_pad = torch.full((len(batch), max_p), 256, dtype=torch.long, device=device)
            t_pad = torch.full((len(batch), max_t), 256, dtype=torch.long, device=device)
            
            for b_idx, (p, t) in enumerate(zip(prompts, targets)):
                p_pad[b_idx, -len(p):] = torch.tensor(p, dtype=torch.long, device=device)
                t_pad[b_idx, :len(t)] = torch.tensor(t, dtype=torch.long, device=device)
                
            optimizer.zero_grad()
            loss = model.forward_loss(p_pad, t_pad)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch:02d} | Dyck Loss: {total_loss/batches:.4f} nats")
            
    # Evaluation
    model.eval()
    correct = 0
    for expr, ans, full in test_data:
        p_bytes = list(expr.encode('utf-8'))
        p_t = torch.tensor([p_bytes], dtype=torch.long, device=device)
        gen = model.generate(p_t, max_len=len(ans) + 4)
        gen_str = bytes(gen[0].cpu().tolist()).decode('utf-8', errors='ignore').strip()
        if ans.strip() in gen_str:
            correct += 1
            
    acc = (correct / len(test_data)) * 100.0
    print(f"\n🔥 ISOLATED DYCK-1 ACCURACY: {acc:.2f}% ({correct}/{len(test_data)})")

if __name__ == '__main__':
    test_isolated_dyck()
