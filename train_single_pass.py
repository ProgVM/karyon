# train_single_pass.py
"""
===============================================================================
KARYON TRAINING RUNTIME ALIAS & BACKWARD COMPATIBILITY PROXY
Forwarding invocation to train_multi_pass.py (v31.0 Master Multi-Pass Streaming Runtime).
Author: Bazilevs (ProgVM member) & Karyon-CoRE Research Team (2026)
===============================================================================
"""

import sys
import train_multi_pass

if __name__ == "__main__":
    train_multi_pass.run_multi_pass_training()
