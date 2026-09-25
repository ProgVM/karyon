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
# DISCOVERY: Arithmetic Substrate with Scratchpad Thinking Time
# Why does direct "A + B = C" fail at 0%?
# Because predicting 'C' in forward order requires non-local carry calculation across all digits.
# If Karyon is trained with scratchpad format:
# Input: "714 + 178 = "
# Thought trace: "4+8=12, 1+7+1=9, 7+1=8 -> 892"
#
# Can an end-to-end recurrent network generate the sum if given an internal latent memory tape?
# Let's test Latent Tape Memory on addition!
# ==============================================================================

print("Cleaning up test scripts...")
