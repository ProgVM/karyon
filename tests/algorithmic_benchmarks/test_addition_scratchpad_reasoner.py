import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==============================================================================
# SCIENTIFIC DISCOVERY: Arithmetic Scratchpad / Mental Calculation
# How does a human or reasoning model calculate "714 + 178"?
# Direct answer: "892" (Requires multi-step mental compute).
# Scratchpad calculation:
# "714 + 178 = [4+8=12, c=1] [1+7+1=9, c=0] [7+1+0=8] -> 892"
#
# Let's test if Karyon with a Latent Recurrent Tape (Turing Machine head)
# that iteratively adds column by column can achieve 100.00% exact accuracy!
# ==============================================================================

# Explicit Column-by-Column Recurrent Substrate:
# Input: "714 + 178 = "
# The model has 2 spatial pointers: p1 at end of num1, p2 at end of num2.
# Step 1: read d1=4, d2=8, carry=0 -> sum=12 -> out='2', carry=1, move p1 left, p2 left.
# Step 2: read d1=1, d2=7, carry=1 -> sum=9  -> out='9', carry=0, move p1 left, p2 left.
# Step 3: read d1=7, d2=1, carry=0 -> sum=8  -> out='8', carry=0, move p1 left, p2 left.
# Reverse emitted digits ['2', '9', '8'] -> '892'!

class ArithmeticTuringCortex(nn.Module):
    def __init__(self, vocab=258, dim=128):
        super().__init__()
        self.dim = dim
        self.emb = nn.Embedding(vocab, dim)
        self.enc = nn.GRU(dim, dim, batch_first=True, bidirectional=True)
        
        # Digit transition MLP: takes (embed_d1, embed_d2, carry_state) -> (next_digit_logits, next_carry_state)
        self.carry_emb = nn.Embedding(2, 32)
        self.adder_mlp = nn.Sequential(
            nn.Linear(dim * 4 + 32, 256),
            nn.GELU(),
            nn.Linear(256, 10 + 2) # 10 digit logits + 2 carry logits
        )
        
        # Pointers for number 1 and number 2
        self.q_p1 = nn.Linear(dim * 2, dim * 2)
        self.q_p2 = nn.Linear(dim * 2, dim * 2)

    def forward(self, p):
        B, Sp = p.shape
        h_p = self.emb(p)
        mem, _ = self.enc(h_p) # [B, Sp, 2*D]
        
        # In a fully neural learned adder, let's verify if column addition converges to 100%
        # Let's test end-to-end differentiable neural column stepping!
        return mem

print("Testing Arithmetic Turing Model...")
