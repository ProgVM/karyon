import random
import torch
from test_deep_evolution_extended import m, suite_test, device

m.eval()
print("=== DEEP DIAGNOSTIC OF BOTTLENECK TASKS: POINTER, PARITY, ADDITION ===")

for domain in ['pointer', 'parity', 'addition']:
    print(f"\n==================== DOMAIN: {domain.upper()} ====================")
    samples = suite_test[domain][:6]
    for p, exp, _ in samples:
        p_b = [ord(c) for c in p]
        p_t = torch.tensor([p_b], dtype=torch.long, device=device)
        B, Sp = p_t.shape
        h_p = m.emb(p_t)
        mem, hn = m.enc(h_p)
        for _ in range(m.thinking_cycles):
            d_mem = m.cellular_conv(mem.transpose(1, 2)).transpose(1, 2)
            mem = m.norm_mem(mem + 0.5 * d_mem)
            
        k_mem = m.k_proj(mem)
        p_mask = (p_t == 256)
        s_t = hn.transpose(0, 1).reshape(B, -1)
        
        gaze = torch.zeros(B, Sp, device=device)
        eq_pos = (p_t[0] == 61).nonzero(as_tuple=True)[0]
        if len(eq_pos) > 0:
            gaze[0, eq_pos[0].item()] = 1.0
        else:
            gaze[0, Sp - 1] = 1.0
            
        gen = []
        prev_tok = torch.tensor([257], dtype=torch.long, device=device)
        with torch.no_grad():
            for _ in range(len(exp) + 4):
                read_vec = torch.bmm(gaze.unsqueeze(1), mem).squeeze(1)
                x_t = m.emb(prev_tok)
                gru_in = torch.cat([x_t, read_vec], dim=-1)
                s_t = m.dec_cell(gru_in, s_t)
                
                q_t = m.q_proj(s_t).unsqueeze(1)
                sim = torch.bmm(q_t, k_mem.transpose(1, 2)).squeeze(1) / (m.dim * 2)**0.5
                sim = sim.masked_fill(p_mask, -1e9)
                content_gaze = torch.softmax(sim, dim=-1)
                
                shift_prob = torch.softmax(m.shift_logits(s_t), dim=-1)
                shifted_gaze = torch.zeros_like(gaze)
                for k, shift_val in enumerate(m.shifts):
                    shifted = torch.roll(gaze, shifts=shift_val, dims=1)
                    shifted_gaze = shifted_gaze + shift_prob[:, k:k+1] * shifted
                    
                alpha = torch.sigmoid(m.gaze_gate(s_t))
                gaze = alpha * content_gaze + (1.0 - alpha) * shifted_gaze
                
                combined = torch.cat([s_t, read_vec], dim=-1)
                v_logits = m.vocab_head(combined)
                p_copy = torch.zeros(B, 258, device=device)
                p_copy.scatter_add_(1, p_t, gaze)
                p_gen_gate = torch.sigmoid(m.p_gen(combined))
                total_prob = p_gen_gate * torch.softmax(v_logits, dim=-1) + (1.0 - p_gen_gate) * (p_copy + 1e-9)
                
                nxt = torch.argmax(total_prob[0]).item()
                if nxt in (ord('\n'), 256, 257):
                    break
                gen.append(nxt)
                prev_tok = torch.tensor([nxt], dtype=torch.long, device=device)
                
        got = bytes(gen).decode('utf-8', errors='ignore').strip()
        ok = '✅' if got == exp.strip() else '❌'
        print(f"  {ok} Prompt: {p:<45} | Target: '{exp.strip()}' | Got: '{got}' (p_gen={p_gen_gate.item():.2f})")
