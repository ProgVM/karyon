import sys
sys.path.insert(0, '.')
import torch
import karyon_agent
from multi_domain_benchmark import generate_multi_domain_suite

# Let's inspect what the agent is actually outputting during autoregression in EXP-291!
suite = generate_multi_domain_suite(seed=202)
agent = karyon_agent.CoREAgent(embed_dim=256, device='cpu')

for domain in ['pointer', 'reversal', 'dyck']:
    p, ans, full = suite[domain][0]
    p_bytes = torch.tensor(list(p.encode('utf-8')), dtype=torch.long).unsqueeze(0)
    print(f"\nDomain: {domain} | Prompt: {p!r} | Expected Ans: {ans!r}")
    
    # Let's see logits on the last token of prompt
    logits = agent(p_bytes, thinking_steps=4)
    next_tok = torch.argmax(logits[:, -1, :], dim=-1).item()
    print(f"First predicted token: {next_tok} ({chr(next_tok) if 32 <= next_tok <= 126 else 'non-ascii'})")
