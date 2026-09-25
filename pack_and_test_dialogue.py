import sys, os, time, json
sys.path.insert(0, '.')
import torch
import torch.nn.functional as F
import karyon_agent
from karyon_checkpoint import save_karyon
from karyon_logger import get_logger

logger = get_logger()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("=" * 80)
print(f"PACKING SOVEREIGN KARYON-CORE v5.0 CONTAINER (.kcore) ON {device.upper()}")
print("=" * 80)

# 1. Initialize Agent with proven Morphogenetic & Bilinear Graph
agent = karyon_agent.CoREAgent(embed_dim=256, device=device)
agent.add_node("bilinear_core", "BilinearMultiplicative", is_core=True, initial_alpha=0.5)

# 2. Mock state objects matching container spec
class MockHU:
    def __init__(self):
        self.state = torch.tensor([[0.85, 0.90, 0.80, 0.95, 0.25, 0.40]], dtype=torch.float32, device=device)

class MockMemory:
    def __init__(self):
        self.keys = torch.zeros(1, 100, 256, device=device)
        self.values = torch.zeros(1, 100, 256, device=device)
        self.pointer = torch.zeros(1, dtype=torch.long, device=device)
        self.size = torch.tensor([100], dtype=torch.long, device=device)

hu = MockHU()
memory = MockMemory()
h_fast = torch.zeros(1, 256, device=device)
h_slow = torch.zeros(1, 256, device=device)

# 3. Save into official karyon_soul.kcore
kcore_file = "karyon_soul.kcore"
print(f"Packaging codebase, DNA, dynamic C++ graph and weights into '{kcore_file}'...")
save_karyon(
    agent=agent,
    memory=memory,
    hu=hu,
    h_fast=h_fast,
    h_slow=h_slow,
    epoch=1,
    story_idx=293,
    filepath=kcore_file,
    root_dir="."
)

file_size_mb = os.path.getsize(kcore_file) / (1024 * 1024)
print(f"Successfully assembled '{kcore_file}'! Size: {file_size_mb:.2f} MB")

# 4. Interactive Dialogue Demonstration with CoREAgent
print("\n" + "=" * 80)
print("=== LAUNCHING INTERACTIVE CLOSED-LOOP SOCIAL DIALOGUE SESSION ===")
print("=" * 80)

def generate_reply(agent, user_text: str, thinking_steps=4, max_new_tokens=50):
    agent.eval()
    prompt = f"Human: {user_text}\nKaryon:"
    prompt_bytes = torch.tensor(list(prompt.encode('utf-8')), dtype=torch.long, device=device).unsqueeze(0)
    curr = prompt_bytes
    generated_bytes = []
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = agent(curr, thinking_steps=thinking_steps)
            last_logits = logits[:, -1, :] / 0.70
            probs = F.softmax(last_logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            val = nxt.item()
            if val == 10 or val == 257:
                break
            generated_bytes.append(val)
            curr = torch.cat([curr, nxt], dim=1)
            
    reply_str = bytes(generated_bytes).decode('utf-8', errors='ignore').strip()
    return reply_str

dialogue_turns = [
    "Hello Karyon! How are you feeling today?",
    "What do you think about our new continuous 3-phase sleep?",
    "Can you remember what we taught you about pointers and reversal?"
]

for turn in dialogue_turns:
    print(f"\n[Human]: {turn}")
    response = generate_reply(agent, turn, thinking_steps=4)
    print(f"[Karyon]: {response}")

print("\n" + "=" * 80)
print("Dialogue loop successfully operational!")
