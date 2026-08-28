# experiments/exp_88_multimodal_training.py
"""
===============================================================================
KARYON MULTIMODAL ASSOCIATIVE TRAINING BENCHMARK (EXP-88)
Tests cross-modal translation and associative learning in Karyon-CoRE v23.0.
Implements Recurrent Multimodal Associative Learning (RMAL) where the text
command is first unrolled through the 2-Stage Cortical Stack, and then the
multimodal step is executed based on the resulting recurrent state.
===============================================================================
"""

import sys
import os
import time
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())

from karyon_config import CoREConfig
from karyon_agent import CoREAgent
from karyon_core import HomeostaticUnit, BatchedEpisodicMemory

def run_multimodal_experiment():
    print("[EXP-88] Initializing Karyon Multimodal Associative Training Benchmark...")
    
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    print(f"[EXP-88] Using device: {device_str.upper()}")

    # 1. Setup Configuration
    config = CoREConfig()
    config.net.text_dim = 256
    config.net.unified_dim = 256
    config.net.hidden_dim = 768
    config.net.expand_dim = 3072
    config.net.num_heads = 12
    config.net.latent_dim = 128
    config.net.num_attractors = 256
    config.net.text_gen_dim = 258
    config.train.batch_size = 16
    
    # 2. Initialize Agent and Somatic Units
    agent = CoREAgent(config=config, device=device_str).to(device)
    hu = HomeostaticUnit(batch_size=config.train.batch_size, device=device_str)
    episodic_mem = BatchedEpisodicMemory(batch_size=config.train.batch_size, memory_dim=config.net.unified_dim, max_capacity=100, device=device_str)
    
    optimizer = torch.optim.AdamW(agent.get_all_parameters(), lr=1e-3, weight_decay=0.01)
    
    # 3. Generate Correlated Multimodal Dataset
    # Class 1: "blue circle"     -> Vision = +1.0, Audio = +2.0
    # Class 2: "red square"      -> Vision = -1.0, Audio = -2.0
    # Class 3: "green triangle"  -> Vision = +0.5, Audio = -0.5
    # Class 4: "yellow star"     -> Vision = -0.5, Audio = +1.5
    classes = [
        {"name": "blue circle",     "vis": 1.0,  "aud": 2.0},
        {"name": "red square",      "vis": -1.0, "aud": -2.0},
        {"name": "green triangle",  "vis": 0.5,  "aud": -0.5},
        {"name": "yellow star",     "vis": -0.5, "aud": 1.5}
    ]
    
    class_tensors = []
    for c in classes:
        ids = agent.encode_text(c["name"]).to(device)
        v_target = torch.ones(config.net.vision_dim, device=device) * c["vis"]
        a_target = torch.ones(config.net.audio_dim, device=device) * c["aud"]
        class_tensors.append({
            "ids": ids,
            "vis_target": v_target,
            "aud_target": a_target
        })

    print(f"[EXP-88] Starting Recurrent Multimodal Associative Learning (200 steps across {len(classes)} classes)...")
    agent.train()
    
    start_time = time.perf_counter()
    
    for step in range(200):
        optimizer.zero_grad()
        
        batch_text_ids = []
        batch_vision_targets = []
        batch_audio_targets = []
        
        for b in range(config.train.batch_size):
            c_idx = (b + step) % len(classes)
            item = class_tensors[c_idx]
            
            batch_text_ids.append(item["ids"])
            batch_vision_targets.append(item["vis_target"])
            batch_audio_targets.append(item["aud_target"])
            
        max_len = max(len(ids) for ids in batch_text_ids)
        padded_text_ids = torch.stack([F.pad(ids, (0, max_len - len(ids)), value=256) for ids in batch_text_ids])
        
        vision_targets = torch.stack(batch_vision_targets)
        audio_targets = torch.stack(batch_audio_targets)
        
        # Phase 1: Unroll text command through the 2-Stage Cortical Stack
        m_s1 = torch.zeros(config.train.batch_size, agent.num_heads, agent.head_k, agent.head_v, device=device)
        m_s2 = torch.zeros(config.train.batch_size, agent.num_heads, agent.head_k, agent.head_v, device=device)
        
        text_emb = agent.pos_embeddings(padded_text_ids, start_pos=0, apply_rf=True)
        h_in = agent.in_proj(text_emb)
        
        h_s1, m_s1, _ = agent.stage1(h_in, m_s1, hu.state.detach(), torch.Tensor(), 1.0)
        sal_gate = agent.boundary_detector(h_s1, padded_text_ids)
        e1_weighted, _, _ = agent.pw_lper(h_s1, torch.zeros(config.train.batch_size, 1, agent.hidden_dim, device=device), hu.state.detach())
        h_s2, m_s2, _ = agent.stage2(e1_weighted, m_s2, hu.state.detach(), sal_gate, 1.0)
        
        # Phase 2: Execute the Multimodal Step using the resulting recurrent state
        # Masking Strategy:
        # 50% of the time: Provide text only (mask vision and audio to 0) to force cross-modal generation from recurrent state!
        # 50% of the time: Provide noisy multimodal input
        batch_vision = []
        batch_audio = []
        for b in range(config.train.batch_size):
            c_idx = (b + step) % len(classes)
            item = class_tensors[c_idx]
            mask_mode = torch.rand(1).item()
            if mask_mode < 0.50:
                batch_vision.append(torch.zeros_like(item["vis_target"]))
                batch_audio.append(torch.zeros_like(item["aud_target"]))
            else:
                batch_vision.append(item["vis_target"] + torch.randn_like(item["vis_target"]) * 0.05)
                batch_audio.append(item["aud_target"] + torch.randn_like(item["aud_target"]) * 0.05)
                
        vision_in = torch.stack(batch_vision)
        audio_in = torch.stack(batch_audio)
        
        # The text input for the multimodal step is the last token's embedding
        text_in = text_emb[:, -1, :]
        
        sensor_dict = {
            'text': text_in,
            'vision': vision_in,
            'audio': audio_in,
            'binary': torch.zeros(config.train.batch_size, config.net.binary_dim, device=device),
            'telepathic': torch.zeros(config.train.batch_size, config.net.telepathic_dim, device=device),
            'motor_efference': torch.zeros(config.train.batch_size, config.net.action_dim, device=device)
        }
        
        outs, fe, commit_loss, attn_weights, channel_names, m_s1, m_s2, z_t = agent.forward_multimodal_step(
            sensor_dict, m_s1, m_s2, hu.state.detach()
        )
        
        pred_vision = outs["vision_generation"]
        pred_audio = outs["audio_generation"]
        
        loss_vision = F.mse_loss(pred_vision, vision_targets)
        loss_audio = F.mse_loss(pred_audio, audio_targets)
        loss_fe = fe.mean()
        
        total_loss = loss_vision + loss_audio + 0.1 * loss_fe + 0.05 * commit_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.get_all_parameters(), max_norm=2.0)
        optimizer.step()
        
        action_cost = torch.full((config.train.batch_size, 1), 0.002, device=device)
        pred_err = torch.full((config.train.batch_size, 1), float(total_loss.item() * 0.1), device=device)
        hu.update(action_cost, pred_err, fe.detach(), torch.zeros((config.train.batch_size, 1), dtype=torch.int64, device=device))
        hu.state = hu.state.detach()
        
        if (step + 1) % 25 == 0:
            print(f"  Step {step+1:03d}/200 | Total Loss: {total_loss.item():.4f} | Vision MSE: {loss_vision.item():.4f} | Audio MSE: {loss_audio.item():.4f} | Free Energy: {loss_fe.item():.4f}")

    duration = time.perf_counter() - start_time
    print(f"[EXP-88] Multimodal Association Training complete in {duration:.2f}s.")

    # =========================================================================
    # 4. EVALUATION: ZERO-SHOT CROSS-MODAL TRANSLATION (TEXT ONLY QUERY)
    # =========================================================================
    print("\n[EXP-88] Evaluating Zero-Shot Cross-Modal Translation (Text-Only Query -> Generate Vision & Audio)...")
    agent.eval()
    
    cos_sim_list = []
    
    print("\n" + "="*85)
    print(" === ZERO-SHOT CROSS-MODAL TRANSLATION TEST RESULTS ===")
    print("="*85)
    
    with torch.no_grad():
        for idx, item in enumerate(class_tensors):
            c_name = classes[idx]["name"]
            test_text = torch.stack([item["ids"]])
            
            # Phase 1: Unroll text command to build recurrent state
            m_s1_eval = torch.zeros(1, agent.num_heads, agent.head_k, agent.head_v, device=device)
            m_s2_eval = torch.zeros(1, agent.num_heads, agent.head_k, agent.head_v, device=device)
            hu_eval = HomeostaticUnit(batch_size=1, device=device_str)
            
            test_emb_seq = agent.pos_embeddings(test_text, start_pos=0, apply_rf=True)
            h_in_eval = agent.in_proj(test_emb_seq)
            
            h_s1_eval, m_s1_eval, _ = agent.stage1(h_in_eval, m_s1_eval, hu_eval.state, torch.Tensor(), 1.0)
            sal_gate_eval = agent.boundary_detector(h_s1_eval, test_text)
            e1_weighted_eval, _, _ = agent.pw_lper(h_s1_eval, torch.zeros(1, 1, agent.hidden_dim, device=device), hu_eval.state)
            h_s2_eval, m_s2_eval, _ = agent.stage2(e1_weighted_eval, m_s2_eval, hu_eval.state, sal_gate_eval, 1.0)
            
            # Phase 2: Multimodal step with ZERO vision and audio inputs
            sensor_dict_test = {
                'text': test_emb_seq[:, -1, :],
                'vision': torch.zeros(1, config.net.vision_dim, device=device),
                'audio': torch.zeros(1, config.net.audio_dim, device=device),
                'binary': torch.zeros(1, config.net.binary_dim, device=device),
                'telepathic': torch.zeros(1, config.net.telepathic_dim, device=device),
                'motor_efference': torch.zeros(1, config.net.action_dim, device=device)
            }
            
            outs_test, fe_test, _, _, _, _, _, _ = agent.forward_multimodal_step(sensor_dict_test, m_s1_eval, m_s2_eval, hu_eval.state)
            
            gen_vis = outs_test["vision_generation"]
            gen_aud = outs_test["audio_generation"]
            
            cos_vis = F.cosine_similarity(gen_vis, item["vis_target"].unsqueeze(0), dim=-1).item()
            cos_aud = F.cosine_similarity(gen_aud, item["aud_target"].unsqueeze(0), dim=-1).item()
            
            cos_sim_list.extend([cos_vis, cos_aud])
            
            print(f"  Query: \"{c_name}\"")
            print(f"    - Generated Vision Mean: {gen_vis.mean().item():+.3f} (Target: {classes[idx]['vis']:+.1f}) | Cosine Sim: {cos_vis:+.4f}")
            print(f"    - Generated Audio  Mean: {gen_aud.mean().item():+.3f} (Target: {classes[idx]['aud']:+.1f}) | Cosine Sim: {cos_aud:+.4f}")

    print("="*85 + "\n")

    avg_cos_sim = sum(cos_sim_list) / len(cos_sim_list)
    final_loss = float(total_loss.item())
    peak_vram = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if device_str == 'cuda' else 0.0

    if avg_cos_sim >= 0.85:
        verdict = "🟢 POSITIVE"
        print(f"[EXP-88] VERDICT: 🟢 POSITIVE! Average Cosine Similarity = {avg_cos_sim:.4f} >= 0.85. True cross-modal associative synthesis achieved!")
    elif avg_cos_sim >= 0.50:
        verdict = "⚪ NEUTRAL / INCONCLUSIVE"
        print(f"[EXP-88] VERDICT: ⚪ NEUTRAL. Average Cosine Similarity = {avg_cos_sim:.4f}.")
    else:
        verdict = "🔴 REJECTED"
        print(f"[EXP-88] VERDICT: 🔴 REJECTED. Average Cosine Similarity = {avg_cos_sim:.4f}.")

    results = {
        "verdict": verdict,
        "final_loss": final_loss,
        "avg_cosine_similarity": avg_cos_sim,
        "vram_mb": peak_vram,
        "duration_sec": duration
    }
    print(f"EXP-88 JSON RESULT: {json.dumps(results)}")

if __name__ == "__main__":
    run_multimodal_experiment()
