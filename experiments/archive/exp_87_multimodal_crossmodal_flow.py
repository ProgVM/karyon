# experiments/exp_87_multimodal_crossmodal_flow.py
"""
===============================================================================
KARYON EXPERIMENTAL BENCHMARK: EXP-87 (UNIVERSAL MULTIMODAL & CROSS-MODAL FLOW)
Tests universal cross-modal ingestion and generation across 5 modalities:
- Text (raw UTF-8 bytes)
- Vision (pixel frame vectors)
- Audio (raw PCM audio waveform vectors)
- Binary (arbitrary document/executable bytes)
- Telepathic Thought Vectors (inter-agent latent mind state vectors)
Author: Bazilevs (ProgVM member) & Karyon-CoRE Research Team (2026)
===============================================================================
"""

import sys
import os
import time
import math
import types
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch Dynamo Hotfix for Python 3.12 / Kaggle GPU
class DummyDynamoModule(types.ModuleType):
    def __getattr__(self, name):
        if name == "decorators":
            return decorators_mod
        if name == "disable":
            return _disable
        if name == "is_compiling":
            return lambda *args, **kwargs: False
        return lambda *args, **kwargs: None

def _disable(fn=None, *args, **kwargs):
    if fn is None or not callable(fn):
        return lambda *a, **kw: None
    return fn

decorators_mod = types.ModuleType("torch._dynamo.decorators")
class _DimRange:
    pass
decorators_mod._DimRange = _DimRange

dynamo_mod = DummyDynamoModule("torch._dynamo")
dynamo_mod.decorators = decorators_mod
dynamo_mod.disable = _disable

sys.modules["torch._dynamo"] = dynamo_mod
sys.modules["torch._dynamo.decorators"] = decorators_mod
torch._dynamo = dynamo_mod

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import karyon_config, karyon_core, karyon_logger
from karyon_config import CoREConfig
from karyon_core import ByteTokenizer, HomeostaticUnit, BatchedEpisodicMemory, CorticalStage, PrecisionWeightedLPER, EntropyAdaptiveBoundaryDetector, DesaturatedHopfieldAttractorHead, LatentPredictor

logger = karyon_logger.get_logger()
torch.set_grad_enabled(True)

device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_str)
use_amp = (device_str == 'cuda')


# =============================================================================
# 1. UNIVERSAL MULTIMODAL SENSORY GATEWAY (8-CHANNEL GLOBAL WORKSPACE)
# =============================================================================

class UniversalMultimodalSensoryGateway(nn.Module):
    """
    Ingests 8 parallel sensory and cognitive channels into a unified Global Workspace:
    1. Text (raw UTF-8 byte embeddings)
    2. Vision (pixel frame features)
    3. Audio (raw PCM audio features)
    4. Binary (arbitrary binary stream features)
    5. Telepathic (inter-agent latent mind states)
    6. Motor (efference feedback)
    7. Body (Ashby homeostatic state u_t)
    8. Mind (internal recurrent state h_prev)
    """
    def __init__(self, unified_dim=256, hidden_dim=768, homeo_dim=6, text_dim=256, 
                 vision_dim=256, audio_dim=256, binary_dim=256, telepathic_dim=256, action_dim=3, device='cpu'):
        super().__init__()
        self.unified_dim = unified_dim
        self.hidden_dim = hidden_dim

        self.text_proj = nn.Linear(text_dim, unified_dim)
        self.vision_proj = nn.Linear(vision_dim, unified_dim)
        self.audio_proj = nn.Linear(audio_dim, unified_dim)
        self.binary_proj = nn.Linear(binary_dim, unified_dim)
        self.telepathic_proj = nn.Linear(telepathic_dim, unified_dim)
        self.motor_proj = nn.Linear(action_dim, unified_dim)
        
        self.homeo_proj = nn.Linear(homeo_dim, unified_dim)
        self.mind_proj = nn.Linear(hidden_dim, unified_dim)
        self.attention_query_layer = nn.Linear(hidden_dim, unified_dim)

        self.channel_norm = nn.LayerNorm(unified_dim)
        self.query_norm = nn.LayerNorm(unified_dim)

    def forward(self, text_in, vision_in, audio_in, binary_in, telepathic_in, motor_in, h_prev, u_t):
        batch_size = h_prev.size(0)
        projected_channels = []
        channel_names = []
        channel_masks = []

        # 1. Text Channel
        t_max = text_in.abs().max(dim=-1, keepdim=True).values
        t_act = (t_max > 1e-5).float()
        projected_channels.append(self.text_proj(text_in))
        channel_names.append("text")
        channel_masks.append((1.0 - t_act) * -1e9)

        # 2. Vision Channel
        v_max = vision_in.abs().max(dim=-1, keepdim=True).values
        v_act = (v_max > 1e-5).float()
        projected_channels.append(self.vision_proj(vision_in))
        channel_names.append("vision")
        channel_masks.append((1.0 - v_act) * -1e9)

        # 3. Audio Channel
        a_max = audio_in.abs().max(dim=-1, keepdim=True).values
        a_act = (a_max > 1e-5).float()
        projected_channels.append(self.audio_proj(audio_in))
        channel_names.append("audio")
        channel_masks.append((1.0 - a_act) * -1e9)

        # 4. Binary Stream Channel
        b_max = binary_in.abs().max(dim=-1, keepdim=True).values
        b_act = (b_max > 1e-5).float()
        projected_channels.append(self.binary_proj(binary_in))
        channel_names.append("binary")
        channel_masks.append((1.0 - b_act) * -1e9)

        # 5. Telepathic Inter-Agent Thought Channel
        tp_max = telepathic_in.abs().max(dim=-1, keepdim=True).values
        tp_act = (tp_max > 1e-5).float()
        projected_channels.append(self.telepathic_proj(telepathic_in))
        channel_names.append("telepathic")
        channel_masks.append((1.0 - tp_act) * -1e9)

        # 6. Motor Channel
        m_max = motor_in.abs().max(dim=-1, keepdim=True).values
        m_act = (m_max > 1e-5).float()
        projected_channels.append(self.motor_proj(motor_in))
        channel_names.append("motor")
        channel_masks.append((1.0 - m_act) * -1e9)

        # 7. Body Channel
        projected_channels.append(self.homeo_proj(u_t))
        channel_names.append("body")
        channel_masks.append(torch.zeros(batch_size, 1, device=h_prev.device))

        # 8. Mind Channel
        projected_channels.append(self.mind_proj(h_prev))
        channel_names.append("mind")
        channel_masks.append(torch.zeros(batch_size, 1, device=h_prev.device))

        # Stack & Compute GWT Attention
        stacked = torch.stack(projected_channels, dim=1) # [B, 8, unified_dim]
        norm_stacked = self.channel_norm(stacked)

        query = self.attention_query_layer(h_prev).unsqueeze(1)
        norm_query = self.query_norm(query)

        sim = (norm_query * norm_stacked).sum(dim=-1) / math.sqrt(self.unified_dim)
        masks = torch.cat(channel_masks, dim=1)
        sim = sim + masks

        attn_weights = torch.softmax(sim, dim=-1)
        epistemic_entropy = -(attn_weights * torch.log(attn_weights + 1e-9)).sum(dim=-1, keepdim=True)

        w_t = (attn_weights.unsqueeze(-1) * stacked).sum(dim=1)
        return w_t, attn_weights, channel_names, epistemic_entropy


# =============================================================================
# 2. UNIVERSAL MULTIMODAL MOTOR GATEWAY (CROSS-MODAL EFFERENCE READOUT)
# =============================================================================

class UniversalMultimodalMotorGateway(nn.Module):
    """
    Generates multi-modal & cross-modal outputs from cortical state h_relaxed:
    1. Text (byte logits [B, 258])
    2. Vision (pixel frame vector [B, 256])
    3. Audio (PCM audio vector [B, 256])
    4. Binary (binary byte vector [B, 256])
    5. Telepathic (inter-agent thought vector [B, 256])
    6. Motor action [B, 3]
    7. Cognitive gating [B, 3]
    """
    def __init__(self, hidden_dim=768, action_dim=3, cog_action_dim=3, text_gen_dim=258,
                 vision_dim=256, audio_dim=256, binary_dim=256, telepathic_dim=256, device='cpu'):
        super().__init__()
        self.motor_action = nn.Linear(hidden_dim, action_dim)
        self.cognitive_gating = nn.Linear(hidden_dim, cog_action_dim)
        self.text_generation = nn.Linear(hidden_dim, text_gen_dim)

        self.vision_generation = nn.Sequential(
            nn.Linear(hidden_dim, vision_dim),
            nn.SiLU(),
            nn.Linear(vision_dim, vision_dim)
        )
        self.audio_generation = nn.Sequential(
            nn.Linear(hidden_dim, audio_dim),
            nn.SiLU(),
            nn.Linear(audio_dim, audio_dim)
        )
        self.binary_generation = nn.Sequential(
            nn.Linear(hidden_dim, binary_dim),
            nn.SiLU(),
            nn.Linear(binary_dim, binary_dim)
        )
        self.telepathic_generation = nn.Sequential(
            nn.Linear(hidden_dim, telepathic_dim),
            nn.SiLU(),
            nn.LayerNorm(telepathic_dim)
        )

    def forward(self, h_relaxed):
        return {
            "motor_action": self.motor_action(h_relaxed),
            "cognitive_gating": self.cognitive_gating(h_relaxed),
            "text_generation": self.text_generation(h_relaxed),
            "vision_generation": self.vision_generation(h_relaxed),
            "audio_generation": self.audio_generation(h_relaxed),
            "binary_generation": self.binary_generation(h_relaxed),
            "telepathic_generation": self.telepathic_generation(h_relaxed),
        }


# =============================================================================
# 3. EXPERIMENTAL MULTIMODAL CORE AGENT PROTOTYPE
# =============================================================================

class MultimodalCoREAgent(nn.Module):
    def __init__(self, config, device='cpu'):
        super().__init__()
        self.device_str = 'cuda' if (str(device).find('cuda') != -1 and torch.cuda.is_available()) else 'cpu'
        self.device = torch.device(self.device_str)
        self.config = config
        self.hidden_dim = config.net.hidden_dim
        self.unified_dim = config.net.unified_dim
        self.text_dim = config.net.text_dim
        self.expand_dim = getattr(config.net, 'expand_dim', 3072)
        self.latent_dim = getattr(config.net, 'latent_dim', 128)
        self.num_heads = getattr(config.net, 'num_heads', 12)
        self.head_k = getattr(config.net, 'head_k', 64)
        self.head_v = getattr(config.net, 'head_v', 128)

        self.gateway = UniversalMultimodalSensoryGateway(
            unified_dim=self.unified_dim, hidden_dim=self.hidden_dim, homeo_dim=6,
            text_dim=256, vision_dim=256, audio_dim=256, binary_dim=256, telepathic_dim=256,
            action_dim=3, device=self.device_str
        ).to(self.device)

        self.in_proj = nn.Linear(self.unified_dim, self.hidden_dim).to(self.device)

        self.stage1 = CorticalStage(
            hidden_dim=self.hidden_dim, expand_dim=self.expand_dim, num_heads=self.num_heads,
            head_k=self.head_k, head_v=self.head_v, min_beta=0.005, max_beta=0.15,
            swiglu_kernel_size=3, device=self.device_str
        )
        self.boundary_detector = EntropyAdaptiveBoundaryDetector(hidden_dim=self.hidden_dim, device=self.device_str)
        self.pw_lper = PrecisionWeightedLPER(hidden_dim=self.hidden_dim, device=self.device_str)

        self.stage2 = CorticalStage(
            hidden_dim=self.hidden_dim, expand_dim=self.expand_dim, num_heads=self.num_heads,
            head_k=self.head_k, head_v=self.head_v, min_beta=0.0001, max_beta=0.05,
            swiglu_kernel_size=7, device=self.device_str
        )

        self.world_model = LatentPredictor(
            hidden_dim=self.hidden_dim, unified_dim=self.unified_dim, latent_dim=self.latent_dim, device=self.device_str
        )

        self.attractor_head = DesaturatedHopfieldAttractorHead(
            hidden_dim=self.hidden_dim, vocab_size=258, num_attractors=256, device=self.device_str
        )

        self.output_gateway = UniversalMultimodalMotorGateway(
            hidden_dim=self.hidden_dim, action_dim=3, cog_action_dim=3, text_gen_dim=258,
            vision_dim=256, audio_dim=256, binary_dim=256, telepathic_dim=256, device=self.device_str
        ).to(self.device)

        self.motor_text_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.text_dim),
            nn.SiLU(),
            nn.LayerNorm(self.text_dim)
        ).to(self.device)

        self.text_embed = nn.Embedding(258, 256).to(self.device)

    def forward_multimodal_step(self, sensor_dict, m_s1, m_s2, u_t):
        text_in = sensor_dict.get('text', torch.zeros(m_s1.size(0), 256, device=self.device))
        vision_in = sensor_dict.get('vision', torch.zeros(m_s1.size(0), 256, device=self.device))
        audio_in = sensor_dict.get('audio', torch.zeros(m_s1.size(0), 256, device=self.device))
        binary_in = sensor_dict.get('binary', torch.zeros(m_s1.size(0), 256, device=self.device))
        telepathic_in = sensor_dict.get('telepathic', torch.zeros(m_s1.size(0), 256, device=self.device))
        motor_in = sensor_dict.get('motor', torch.zeros(m_s1.size(0), 3, device=self.device))

        h_prev_proxy = m_s1.view(m_s1.size(0), -1)[:, :self.hidden_dim]

        w_t, attn_weights, channel_names, epistemic_entropy = self.gateway(
            text_in, vision_in, audio_in, binary_in, telepathic_in, motor_in, h_prev_proxy, u_t
        )

        x_in = self.in_proj(w_t).unsqueeze(1) # [B, 1, 768]

        h_s1_out, m_s1_next, dt1 = self.stage1(x_in, m_s1, u_t, torch.Tensor(), 1.0)
        dummy_ids = torch.zeros(x_in.size(0), 1, dtype=torch.long, device=self.device)
        sal_gate = self.boundary_detector(h_s1_out, dummy_ids)

        h1_prev_proxy = m_s1.view(m_s1.size(0), -1)[:, :self.hidden_dim].unsqueeze(1)
        e1_weighted, _, _ = self.pw_lper(h_s1_out, h1_prev_proxy, u_t)

        h_s2_out, m_s2_next, dt2 = self.stage2(e1_weighted, m_s2, u_t, sal_gate, 1.0)

        h_combined = h_s1_out + h_s2_out
        h_flat = h_combined.view(-1, self.hidden_dim)
        h_relaxed, commit_loss = self.attractor_head.relax_to_minima(h_flat, u_t)

        outs = self.output_gateway(h_relaxed)

        w_pred, kl_div, fe, z_t = self.world_model(h_prev_proxy, h_relaxed, w_t)

        return outs, fe, commit_loss, attn_weights, channel_names, m_s1_next.detach(), m_s2_next.detach(), z_t


# =============================================================================
# 4. EXP-87 EXPERIMENTAL BENCHMARK RUNNER
# =============================================================================

def run_exp_87_benchmark():
    print("\n" + "="*85)
    print(" === [KEP EXPERIMENTAL BENCHMARK: EXP-87 (UNIVERSAL MULTIMODAL CROSS-MODAL FLOW)] ===")
    print("="*85)
    print(f"Hardware Device : {device_str.upper()} | AMP FP16 Enabled: {use_amp}")

    config = CoREConfig()
    config.net.text_dim = 256
    config.net.hidden_dim = 768
    config.net.expand_dim = 3072
    config.net.num_heads = 12

    b_size = 32
    num_eval_steps = 200

    print("\n[1/1] Initializing MultimodalCoREAgent Prototype (200 Steps Cross-Modal Training)...")
    torch.manual_seed(42)
    agent = MultimodalCoREAgent(config=config, device=device_str).to(device)
    hu = HomeostaticUnit(batch_size=b_size, device=device_str)

    optimizer = torch.optim.AdamW(agent.parameters(), lr=3.0e-3, weight_decay=0.01)

    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()

    losses = []
    fe_list = []

    m_s1 = torch.zeros(b_size, agent.num_heads, agent.head_k, agent.head_v, device=device)
    m_s2 = torch.zeros(b_size, agent.num_heads, agent.head_k, agent.head_v, device=device)

    for step in range(num_eval_steps):
        t_step_0 = time.perf_counter()
        optimizer.zero_grad()

        # Generate synthetic multi-modal inputs:
        # Cross-Modal Scenario A: High Vision & Audio input, target Text & Telepathic output!
        # Cross-Modal Scenario B: High Text & Binary input, target Vision & Audio output!
        if step % 2 == 0:
            sensor_dict = {
                'text': torch.zeros(b_size, 256, device=device),
                'vision': torch.randn(b_size, 256, device=device),
                'audio': torch.randn(b_size, 256, device=device),
                'binary': torch.zeros(b_size, 256, device=device),
                'telepathic': torch.zeros(b_size, 256, device=device),
                'motor': torch.zeros(b_size, 3, device=device)
            }
            target_vision = sensor_dict['vision'] # Self-reconstruction target
            target_text_ids = torch.randint(32, 126, (b_size,), device=device)
            target_audio = torch.randn(b_size, 256, device=device)
            target_telepathic = torch.randn(b_size, 256, device=device)
            target_binary = torch.randn(b_size, 256, device=device)
        else:
            sensor_dict = {
                'text': torch.randn(b_size, 256, device=device),
                'vision': torch.zeros(b_size, 256, device=device),
                'audio': torch.zeros(b_size, 256, device=device),
                'binary': torch.randn(b_size, 256, device=device),
                'telepathic': torch.randn(b_size, 256, device=device),
                'motor': torch.zeros(b_size, 3, device=device)
            }
            target_vision = torch.randn(b_size, 256, device=device)
            target_text_ids = torch.randint(32, 126, (b_size,), device=device)
            target_audio = torch.randn(b_size, 256, device=device)
            target_telepathic = sensor_dict['telepathic'] # Inter-agent sync
            target_binary = sensor_dict['binary']

        with torch.amp.autocast(device_type=device_str, dtype=torch.float16, enabled=use_amp):
            outs, fe, commit_loss, attn_weights, ch_names, m_s1, m_s2, z_t = agent.forward_multimodal_step(
                sensor_dict, m_s1, m_s2, hu.state
            )

            # Compute Multi-Modal & Cross-Modal Loss
            l_text = F.cross_entropy(outs["text_generation"], target_text_ids)
            l_vision = F.mse_loss(outs["vision_generation"], target_vision)
            l_audio = F.mse_loss(outs["audio_generation"], target_audio)
            l_binary = F.mse_loss(outs["binary_generation"], target_binary)
            l_telepathic = (1.0 - F.cosine_similarity(outs["telepathic_generation"], target_telepathic, dim=-1)).mean()

            total_loss = l_text + l_vision + l_audio + l_binary + l_telepathic + 0.05 * fe.mean() + 0.05 * commit_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=3.0)
        optimizer.step()

        losses.append(total_loss.item())
        fe_list.append(fe.mean().item())

        if (step + 1) % 25 == 0 or step == num_eval_steps - 1:
            mean_loss = sum(losses[-25:]) / len(losses[-25:])
            mean_fe = sum(fe_list[-25:]) / len(fe_list[-25:])
            step_dt = (time.perf_counter() - t_step_0) * 1000.0
            throughput_items = b_size / (step_dt / 1000.0)
            vram_mb = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if device.type == 'cuda' else 0.0

            # Print attention distribution across the 8 channels
            last_attn = attn_weights.mean(dim=0).tolist()
            attn_str = " | ".join([f"{ch_names[i]}: {last_attn[i]:.3f}" for i in range(len(ch_names))])

            print(f"  Step {step+1:03d}/{num_eval_steps} | Total Loss = {mean_loss:.4f} | Free Energy = {mean_fe:.4f} | Step Time: {step_dt:.1f} ms | Throughput: {throughput_items:.1f} items/s | VRAM: {vram_mb:.1f} MB")
            print(f"    GWT Attention Dist -> {attn_str}")

    t_total = time.perf_counter() - t_start
    final_loss = sum(losses[-30:]) / len(losses[-30:])
    final_fe = sum(fe_list[-30:]) / len(fe_list[-30:])
    throughput_items = (num_eval_steps * b_size) / t_total
    vram_peak = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if device.type == 'cuda' else 0.0

    print("\n" + "="*85)
    print(" === [EXP-87 EXPERIMENTAL BENCHMARK RESULTS] ===")
    print("="*85)
    print(f"Final Multi-Modal Loss   : {final_loss:.4f} nats")
    print(f"Final Free Energy (F_t)  : {final_fe:.4f}")
    print(f"Total Step Execution Time: {t_total:.2f} s")
    print(f"Multimodal Throughput   : {throughput_items:.1f} items/s")
    print(f"Peak VRAM Memory         : {vram_peak:.1f} MB")

    # KEP Rule #2 Verdict Assessment
    if final_loss < 5.0 and not math.isnan(final_loss) and final_fe < 0.05:
        verdict = "🟢 POSITIVE"
        print(f"Verdict: {verdict} (Universal Cross-Modal Ingestion & Generation Validated!)")
    else:
        verdict = "⚪ NEUTRAL / INCONCLUSIVE"
        print(f"Verdict: {verdict}")

    return {
        "exp_id": "EXP-87",
        "final_loss": final_loss,
        "free_energy": final_fe,
        "vram_mb": vram_peak,
        "items_per_sec": throughput_items,
        "verdict": verdict
    }

if __name__ == "__main__":
    run_exp_87_benchmark()
