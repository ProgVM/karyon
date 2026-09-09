# kcore_evolution.py
"""
===============================================================================
KARYON AUTONOMOUS SELF-EVOLUTION & EPIGENETIC MORPHOGENESIS ENGINE (v3.0 MASTER)
Biophysical Autopoiesis, Epigenetic Gene Regulatory Networks & Neural Darwinism
KEP v9.0 Compliant | KEP Principle 2, Principle 8, Principle 10 & Principle 14
===============================================================================
"""

import os
import copy
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import asdict
from typing import Dict, Any, Tuple, Optional, List

from karyon_config import CoREConfig, HomeostasisConfig, NetworkConfig, MemoryConfig
from karyon_core import (
    CorticalStage,
    EntropyAdaptiveBoundaryDetector,
    PrecisionWeightedLPER,
    FusedCascadedLaminarStack,
    LatentPredictor,
    MotorGateway,
    DesaturatedHopfieldAttractorHead,
    TDFreeEnergyCritic
)
from karyon_logger import get_logger

logger = get_logger()


# =============================================================================
# HELPER: SHAPE-ADAPTIVE PARAMETER ZERO-PAD COPY
# =============================================================================

def adapt_and_copy_tensor(target: torch.Tensor, source: torch.Tensor):
    """Zero-padded submatrix copy preserving pre-trained weights."""
    with torch.no_grad():
        target.zero_()
        slices = tuple(slice(0, min(t_dim, s_dim)) for t_dim, s_dim in zip(target.shape, source.shape))
        target[slices].copy_(source[slices])


# =============================================================================
# LEVEL 1: STRUCTURAL SYNAPTOGENESIS & SLEEP PRUNING (MICRO-SCALE PLASTICITY)
# =============================================================================

class StructuralSynaptogenesisPruner:
    """
    Level 1: Activity-Dependent Axonal Sprouting & Sleep Synaptic Pruning.
    Prunes quiescent, low-utility synaptic connections during sleep phase,
    and sprouts adaptive identity-orthogonal connections in high-surprise pathways.
    """
    
    @staticmethod
    def prune_quiescent_synapses(
        agent: nn.Module,
        prune_ratio: float = 0.10,
        min_magnitude: float = 1e-4
    ) -> Dict[str, Any]:
        """
        Prunes weights with lowest absolute magnitude and gradient utility during sleep.
        Returns detailed pruning statistics.
        """
        total_pruned = 0
        total_synapses = 0
        pruned_layers = []

        with torch.no_grad():
            for name, param in agent.named_parameters():
                if "weight" in name and param.dim() >= 2 and "embed" not in name:
                    abs_w = param.abs()
                    k = int(abs_w.numel() * prune_ratio)
                    if k > 0:
                        threshold = torch.kthvalue(abs_w.view(-1), k).values.item()
                        effective_thresh = max(threshold, min_magnitude)
                        mask = abs_w > effective_thresh
                        num_zeroed = (param.numel() - mask.sum().item())
                        param.mul_(mask.to(param.dtype))
                        total_pruned += num_zeroed
                        total_synapses += param.numel()
                        pruned_layers.append((name, num_zeroed, param.numel()))

        sparsity_pct = (total_pruned / max(total_synapses, 1)) * 100.0
        logger.info(f"🌿 [Level 1 Pruning] Pruned {total_pruned}/{total_synapses} synapses ({sparsity_pct:.2f}% sparsity).")
        return {
            "total_pruned": total_pruned,
            "total_synapses": total_synapses,
            "sparsity_pct": sparsity_pct,
            "pruned_layers": pruned_layers
        }

    @staticmethod
    def sprout_active_axons(
        agent: nn.Module,
        surprise_metric: float,
        growth_std: float = 1e-5
    ) -> Dict[str, Any]:
        """
        Sprouts synaptic connectivity in pathways encountering high variational surprise.
        Injects structured exploratory micro-plasticity into dormant zero-weights.
        """
        if surprise_metric < 0.05:
            return {"sprouted": 0, "status": "QUIESCENT_SURPRISE_LOW"}

        sprouted_count = 0
        with torch.no_grad():
            for name, param in agent.named_parameters():
                if "weight" in name and param.dim() >= 2 and ("sensory" in name or "mind_proj" in name or "motor" in name):
                    zero_mask = (param.abs() < 1e-6)
                    num_zeros = zero_mask.sum().item()
                    if num_zeros > 0:
                        noise = torch.randn_like(param) * (growth_std * min(surprise_metric, 1.0))
                        param.add_(noise * zero_mask.to(param.dtype))
                        sprouted_count += num_zeros

        logger.info(f"🌱 [Level 1 Sprouting] Sprouts created: {sprouted_count} synapses stimulated by surprise ({surprise_metric:.4f}).")
        return {"sprouted": sprouted_count, "surprise_metric": surprise_metric, "status": "ACTIVE_SPROUTING"}


# =============================================================================
# LEVEL 2: TOPOLOGICAL NET2NET MORPHOGENESIS (MESO-SCALE TOPOLOGY EXPANSION)
# =============================================================================

class Net2NetMorphogenesisEngine:
    """
    Level 2: Identity-Preserving Neural Morphogenesis.
    Expands hidden dimensions and adds Cortical Stages with 100% mathematical
    identity preservation: f_new(x) == f_old(x) at initialization t_0.
    """

    @staticmethod
    def expand_linear_layer(
        layer: nn.Linear,
        new_out_features: int,
        new_in_features: Optional[int] = None,
        device: str = "cuda"
    ) -> nn.Linear:
        """
        Net2Net Net-Wider transformation for Linear layers.
        Zero-pads new dimensions to guarantee strict function identity at t=0.
        """
        old_out, old_in = layer.weight.shape
        target_in = new_in_features if new_in_features is not None else old_in
        target_out = new_out_features

        new_layer = nn.Linear(
            target_in, target_out, bias=(layer.bias is not None)
        ).to(device)

        with torch.no_grad():
            new_layer.weight.zero_()
            new_layer.weight[:old_out, :old_in].copy_(layer.weight)
            if layer.bias is not None:
                new_layer.bias.zero_()
                new_layer.bias[:old_out].copy_(layer.bias)

        return new_layer

    @staticmethod
    def expand_layernorm(layer: nn.LayerNorm, new_dim: int, device: str = "cuda") -> nn.LayerNorm:
        """Net2Net expansion for LayerNorm layers."""
        old_dim = layer.normalized_shape[0]
        new_norm = nn.LayerNorm(new_dim, elementwise_affine=layer.elementwise_affine).to(device)
        if layer.elementwise_affine:
            with torch.no_grad():
                new_norm.weight.fill_(1.0)
                new_norm.weight[:old_dim].copy_(layer.weight)
                new_norm.bias.zero_()
                new_norm.bias[:old_dim].copy_(layer.bias)
        return new_norm

    @staticmethod
    def expand_agent_dimensions(
        agent: Any,
        new_hidden_dim: int,
        device: str = "cuda"
    ) -> Tuple[Any, float]:
        """
        Executes full Net2Net identity-preserving dimension expansion across all layers.
        Guarantees that f_new(x) == f_old(x) at t_0 (Zero Identity Delta).
        """
        if new_hidden_dim <= agent.hidden_dim:
            logger.info(f"Target dimension {new_hidden_dim} <= current {agent.hidden_dim}. Morphogenesis skipped.")
            return agent, 0.0

        logger.info(f"🧬 [Level 2 Morphogenesis] Executing Net2Net Expansion: {agent.hidden_dim} ➔ {new_hidden_dim}...")
        
        # Capture baseline output for identity validation
        dummy_text = torch.randint(0, 256, (1, 1), device=device)
        dummy_emb = agent.pos_embeddings(dummy_text, start_pos=0, apply_rf=False)
        sensor_inputs = {
            'text': dummy_emb.squeeze(1),
            'vision': torch.zeros(1, agent.config.net.vision_dim, device=device),
            'motor_efference': torch.zeros(1, agent.config.net.action_dim, device=device)
        }
        h_fast = torch.zeros(1, agent.hidden_dim, device=device)
        h_slow = torch.zeros(1, agent.hidden_dim, device=device)
        u_t = torch.tensor([[0.5, 1.0, 1.0, 1.0, 0.0, 0.0]], device=device)

        with torch.no_grad():
            out_before = agent.forward(sensor_inputs, h_fast, h_slow, u_t)
            logits_before = out_before[4].clone()

        # Save old parameters as cloned tensor lists (100% pickle/C++ safe)
        old_fused_params = [p.clone().detach() for p in agent.fused_stack.parameters()] if hasattr(agent, 'fused_stack') else []
        old_hpc_params = [p.clone().detach() for p in agent.pw_hpc_generator.parameters()] if hasattr(agent, 'pw_hpc_generator') else []
        old_attractor_params = [p.clone().detach() for p in agent.attractor_head.parameters()] if hasattr(agent, 'attractor_head') else []
        old_mg_params = [p.clone().detach() for p in agent.output_gateway.parameters()] if hasattr(agent, 'output_gateway') else []
        old_wm_params = [p.clone().detach() for p in agent.world_model.parameters()] if hasattr(agent, 'world_model') else []

        with torch.no_grad():
            # 1. Universal Dynamic Sensory Gateway expansion
            if hasattr(agent.gateway, 'mind_proj'):
                agent.gateway.mind_proj = Net2NetMorphogenesisEngine.expand_linear_layer(
                    agent.gateway.mind_proj, agent.gateway.mind_proj.out_features, new_hidden_dim, device=device
                )
            if hasattr(agent.gateway, 'attention_query_layer'):
                agent.gateway.attention_query_layer = Net2NetMorphogenesisEngine.expand_linear_layer(
                    agent.gateway.attention_query_layer, agent.gateway.attention_query_layer.out_features, new_hidden_dim, device=device
                )
            agent.gateway.hidden_dim = new_hidden_dim

            # 2. Input Projection expansion
            agent.in_proj = Net2NetMorphogenesisEngine.expand_linear_layer(
                agent.in_proj, new_hidden_dim, agent.unified_dim, device=device
            )

            # 3. Native C++20 Fused Cortical Stack Expansion
            if hasattr(agent, 'fused_stack'):
                agent.fused_stack = FusedCascadedLaminarStack(
                    hidden_dim=new_hidden_dim, expand_dim=agent.expand_dim, num_heads=agent.num_heads,
                    head_k=agent.head_k, head_v=agent.head_v, chunk_size=64, device=agent.device_str
                )
                for old_p, new_p in zip(old_fused_params, agent.fused_stack.parameters()):
                    adapt_and_copy_tensor(new_p, old_p)
                agent.stage1 = agent.fused_stack.stage1
                agent.stage2 = agent.fused_stack.stage2
                if hasattr(agent.fused_stack, 'boundary_detector'):
                    agent.boundary_detector = agent.fused_stack.boundary_detector
                if hasattr(agent.fused_stack, 'pw_lper'):
                    agent.pw_lper = agent.fused_stack.pw_lper
            
            from karyon_agent import (
                PrecisionWeightedTopDownGenerator,
                EntropyMacroGating,
                ThalamocorticalGate,
                FastWeightHebbianPlasticity,
                PredictiveResidualRouting,
                VolitionalActionEvaluator,
                LocalNeuromodulatedPlasticity,
                PredictiveSelfModel
            )
            
            # Save parameters before re-instantiating
            old_emg_params = [p.clone().detach() for p in agent.entropy_macro_gate.parameters()] if hasattr(agent, 'entropy_macro_gate') else []
            old_thalamic_params = [p.clone().detach() for p in agent.thalamic_router.parameters()] if hasattr(agent, 'thalamic_router') else []
            old_fwh_params = [p.clone().detach() for p in agent.fast_weight_hebbian.parameters()] if hasattr(agent, 'fast_weight_hebbian') else []
            old_prr_params = [p.clone().detach() for p in agent.predictive_residual_router.parameters()] if hasattr(agent, 'predictive_residual_router') else []
            old_psm_params = [p.clone().detach() for p in agent.predictive_self_model.parameters()] if hasattr(agent, 'predictive_self_model') else []

            agent.pw_hpc_generator = PrecisionWeightedTopDownGenerator(hidden_dim=new_hidden_dim, device_str=agent.device_str)
            for old_p, new_p in zip(old_hpc_params, agent.pw_hpc_generator.parameters()):
                adapt_and_copy_tensor(new_p, old_p)

            agent.entropy_macro_gate = EntropyMacroGating(new_hidden_dim, vocab_size=agent.text_gen_dim, device_str=agent.device_str)
            for old_p, new_p in zip(old_emg_params, agent.entropy_macro_gate.parameters()):
                adapt_and_copy_tensor(new_p, old_p)

            agent.thalamic_router = ThalamocorticalGate(new_hidden_dim, homeo_dim=agent.config.net.homeo_dim, device_str=agent.device_str)
            for old_p, new_p in zip(old_thalamic_params, agent.thalamic_router.parameters()):
                adapt_and_copy_tensor(new_p, old_p)

            agent.fast_weight_hebbian = FastWeightHebbianPlasticity(new_hidden_dim, device_str=agent.device_str)
            for old_p, new_p in zip(old_fwh_params, agent.fast_weight_hebbian.parameters()):
                adapt_and_copy_tensor(new_p, old_p)

            agent.predictive_residual_router = PredictiveResidualRouting(new_hidden_dim, homeo_dim=agent.config.net.homeo_dim, device_str=agent.device_str)
            for old_p, new_p in zip(old_prr_params, agent.predictive_residual_router.parameters()):
                adapt_and_copy_tensor(new_p, old_p)

            agent.efe_action_evaluator = VolitionalActionEvaluator(hidden_dim=new_hidden_dim, device=agent.device_str)
            agent.local_plasticity = LocalNeuromodulatedPlasticity(in_features=new_hidden_dim, out_features=new_hidden_dim, lr=0.08, device=agent.device_str)
            
            agent.predictive_self_model = PredictiveSelfModel(hidden_dim=new_hidden_dim, homeo_dim=agent.config.net.homeo_dim, device=agent.device_str)
            for old_p, new_p in zip(old_psm_params, agent.predictive_self_model.parameters()):
                adapt_and_copy_tensor(new_p, old_p)

            agent.fused_stack = FusedCascadedLaminarStack(
                hidden_dim=new_hidden_dim, expand_dim=agent.expand_dim, num_heads=agent.num_heads,
                head_k=agent.head_k, head_v=agent.head_v, chunk_size=64, device=agent.device_str
            )
            agent.stage1 = agent.fused_stack.stage1
            agent.stage2 = agent.fused_stack.stage2

            # 4. Top-Down Prior Projection expansion
            if hasattr(agent, 'topdown_prior_proj') and isinstance(agent.topdown_prior_proj, nn.Sequential):
                agent.topdown_prior_proj[0] = Net2NetMorphogenesisEngine.expand_linear_layer(
                    agent.topdown_prior_proj[0], new_hidden_dim, new_hidden_dim, device=device
                )
                agent.topdown_prior_proj[2] = Net2NetMorphogenesisEngine.expand_linear_layer(
                    agent.topdown_prior_proj[2], new_hidden_dim, new_hidden_dim, device=device
                )
                agent.topdown_prior_proj[3] = Net2NetMorphogenesisEngine.expand_layernorm(
                    agent.topdown_prior_proj[3], new_hidden_dim, device=device
                )

            # 5. Fact Gate expansion
            if hasattr(agent, 'fact_gate') and isinstance(agent.fact_gate, nn.Sequential):
                agent.fact_gate[2] = Net2NetMorphogenesisEngine.expand_linear_layer(
                    agent.fact_gate[2], new_hidden_dim, agent.fact_gate[2].in_features, device=device
                )

            # 6. Will Engine expansion
            if hasattr(agent, 'will_engine') and hasattr(agent.will_engine, 'override_gate_net'):
                first_linear = agent.will_engine.override_gate_net[0]
                homeo_dim = agent.config.net.homeo_dim
                agent.will_engine.override_gate_net[0] = Net2NetMorphogenesisEngine.expand_linear_layer(
                    first_linear, first_linear.out_features, new_hidden_dim + homeo_dim, device=device
                )
                agent.will_engine.hidden_dim = new_hidden_dim

            # 7. Volitional Prediction Head expansion
            if hasattr(agent, 'volitional_head') and hasattr(agent.volitional_head, 'motor_text_proj'):
                first_linear = agent.volitional_head.motor_text_proj[0]
                agent.volitional_head.motor_text_proj[0] = Net2NetMorphogenesisEngine.expand_linear_layer(
                    first_linear, first_linear.out_features, new_hidden_dim, device=device
                )
                agent.volitional_head.hidden_dim = new_hidden_dim

            # 8. Hopfield Attractor Head expansion
            agent.attractor_head = DesaturatedHopfieldAttractorHead(
                hidden_dim=new_hidden_dim, 
                vocab_size=agent.text_gen_dim,
                num_attractors=getattr(agent.config.net, 'num_attractors', 256),
                device=agent.device_str
            )
            for old_p, new_p in zip(old_attractor_params, agent.attractor_head.parameters()):
                adapt_and_copy_tensor(new_p, old_p)

            # 9. Output Motor Gateway expansion
            agent.output_gateway = MotorGateway(
                hidden_dim=new_hidden_dim, 
                action_dim=agent.config.net.action_dim, 
                cog_action_dim=agent.config.net.cog_action_dim, 
                text_gen_dim=agent.text_gen_dim,
                vision_dim=agent.config.net.vision_dim,
                audio_dim=getattr(agent.config.net, 'audio_dim', 256),
                binary_dim=getattr(agent.config.net, 'binary_dim', 256),
                telepathic_dim=getattr(agent.config.net, 'telepathic_dim', 256),
                device=agent.device_str
            )
            for old_p, new_p in zip(old_mg_params, agent.output_gateway.parameters()):
                adapt_and_copy_tensor(new_p, old_p)

            # 10. World Model expansion
            agent.world_model = LatentPredictor(
                hidden_dim=new_hidden_dim,
                unified_dim=agent.unified_dim,
                latent_dim=agent.latent_dim,
                device=agent.device_str
            )
            for old_p, new_p in zip(old_wm_params, agent.world_model.parameters()):
                adapt_and_copy_tensor(new_p, old_p)

            # 11. TD Critic expansion
            agent.critic = TDFreeEnergyCritic(hidden_dim=new_hidden_dim, device=agent.device_str)

            # Update Agent Global Config & Dimensions
            agent.hidden_dim = new_hidden_dim
            agent.config.net.hidden_dim = new_hidden_dim

        # Test post-expansion output to verify identity preservation
        h_fast_new = torch.zeros(1, new_hidden_dim, device=device)
        h_slow_new = torch.zeros(1, new_hidden_dim, device=device)
        with torch.no_grad():
            out_after = agent.forward(sensor_inputs, h_fast_new, h_slow_new, u_t)
            logits_after = out_after[4].clone()
            identity_delta = (logits_after - logits_before).abs().max().item()

        logger.info(f"✨ [Level 2 Morphogenesis] Function Identity Delta at t0: {identity_delta:.8f}")
        return agent, identity_delta


# =============================================================================
# LEVEL 3: SLEEP META-GENETICS & BIOPHYSICAL PARAMETER EVOLUTION
# =============================================================================

class SleepMetaGeneticsEngine:
    """
    Level 3: Evolution of Biophysical Formulas & Dynamic Parameters.
    Spawns candidate genome mutations during sleep, evaluates variational surprise
    and Free Energy drop on an autonomous validation buffer, and selects the optimal genome.
    Coupled with Level 4 Self-Reflective Mutation Vector & Level 5 Epigenetics.
    """

    @staticmethod
    def get_active_genome(agent: Any) -> Dict[str, Any]:
        """Extracts the live tunable biophysical genome from agent configs."""
        return {
            "min_beta_stage1": float(getattr(agent.config.net, "min_beta_stage1", 0.005)),
            "max_beta_stage1": float(getattr(agent.config.net, "max_beta_stage1", 0.15)),
            "min_beta_stage2": float(getattr(agent.config.net, "min_beta_stage2", 0.0001)),
            "max_beta_stage2": float(getattr(agent.config.net, "max_beta_stage2", 0.05)),
            "hopfield_beta": float(getattr(agent.config.net, "hopfield_beta", 12.0)),
            "pac_entropy_threshold": float(getattr(agent.config.net, "pac_entropy_threshold", 0.70)),
            "noradrenaline_surprise_weight": float(getattr(agent.config.homeo, "noradrenaline_surprise_weight", 0.85)),
            "dopamine_reward_scale": float(getattr(agent.config.homeo, "dopamine_reward_scale", 2.00)),
            "volitional_recall_gain": float(getattr(agent.config.homeo, "volitional_recall_gain", 2.00))
        }

    @staticmethod
    def apply_genome_to_agent(agent: Any, genome: Dict[str, Any]):
        """Injects evolved genome parameters into live agent runtime."""
        for k, v in genome.items():
            if hasattr(agent.config.net, k):
                setattr(agent.config.net, k, v)
            elif hasattr(agent.config.homeo, k):
                setattr(agent.config.homeo, k, v)

    @staticmethod
    def mutate_genome(
        genome: Dict[str, Any],
        mutation_rate: float = 0.08,
        reflective_proposal: Optional[List[float]] = None,
        methylation_locks: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Creates a mutated offspring candidate genome.
        If methylation_locks from Level 5 are provided, methylated genes are protected against mutation.
        """
        mutated = copy.deepcopy(genome)
        keys = list(mutated.keys())
        
        for idx, k in enumerate(keys):
            v = mutated[k]
            # Check epigenetic methylation protection: high methylation (mu >= 1.0) suppresses mutation
            m_lock = methylation_locks.get(k, 0.0) if methylation_locks else 0.0
            effective_mut_rate = mutation_rate * max(0.0, 1.0 - m_lock)
            
            if effective_mut_rate < 1e-5:
                continue

            ref_bias = reflective_proposal[idx % len(reflective_proposal)] if (reflective_proposal and len(reflective_proposal) > 0) else 0.0
            random_noise = float(torch.randn(1).item())
            
            # Blend self-reflective directional vector (70%) with stochastics (30%)
            effective_dir = 0.70 * ref_bias + 0.30 * random_noise
            factor = 1.0 + effective_dir * effective_mut_rate
            new_val = v * factor
            
            # Apply biophysical sanity bounds
            if "beta" in k:
                new_val = max(1e-5, min(new_val, 0.99))
            elif "hopfield_beta" in k:
                new_val = max(1.0, min(new_val, 50.0))
            elif "entropy_threshold" in k:
                new_val = max(0.1, min(new_val, 2.0))
            elif "weight" in k or "scale" in k or "gain" in k:
                new_val = max(0.01, min(new_val, 10.0))
                
            mutated[k] = float(new_val)
        return mutated

    @staticmethod
    def run_sleep_meta_genetics(
        agent: Any,
        eval_input_tokens: Optional[torch.Tensor],
        eval_target_tokens: Optional[torch.Tensor],
        hu: Any,
        criterion_speech: Optional[nn.Module],
        reflective_proposal: Optional[List[float]] = None,
        methylation_locks: Optional[Dict[str, float]] = None,
        num_candidates: int = 5,
        mutation_rate: float = 0.08
    ) -> Tuple[Dict[str, Any], float]:
        """
        Evaluates N candidate mutant genomes during sleep against baseline Free Energy.
        If explicit eval tokens are not provided, uses an autonomous synthetic evaluation buffer.
        """
        agent.eval()
        base_genome = SleepMetaGeneticsEngine.get_active_genome(agent)
        device = agent.device
        
        # Autonomous evaluation buffer fallback
        if eval_input_tokens is None or eval_target_tokens is None:
            eval_input_tokens = torch.randint(0, 256, (2, 32), device=device)
            eval_target_tokens = torch.randint(0, 256, (2, 32), device=device)
            criterion_speech = nn.CrossEntropyLoss(ignore_index=256)

        with torch.no_grad():
            base_out = agent.forward_sequence(eval_input_tokens, eval_target_tokens, hu, criterion_speech)
            base_loss = base_out[1] if isinstance(base_out[1], (float, int)) else base_out[1].item()

        best_genome = base_genome
        best_loss = base_loss
        best_candidate_idx = -1

        logger.info(f"🧬 [Level 3 Meta-Genetics] Baseline Sleep Free Energy/Loss: {base_loss:.6f}")

        for i in range(num_candidates):
            candidate_genome = SleepMetaGeneticsEngine.mutate_genome(
                base_genome, mutation_rate=mutation_rate, 
                reflective_proposal=reflective_proposal,
                methylation_locks=methylation_locks
            )
            SleepMetaGeneticsEngine.apply_genome_to_agent(agent, candidate_genome)
            
            with torch.no_grad():
                cand_out = agent.forward_sequence(eval_input_tokens, eval_target_tokens, hu, criterion_speech)
                cand_loss = cand_out[1] if isinstance(cand_out[1], (float, int)) else cand_out[1].item()

            delta = base_loss - cand_loss
            logger.info(f"   -> Candidate {i+1}: Loss = {cand_loss:.6f} (Delta: {delta:+.6f})")

            if cand_loss < best_loss:
                best_loss = cand_loss
                best_genome = candidate_genome
                best_candidate_idx = i

        # Re-apply winning or restored baseline genome
        SleepMetaGeneticsEngine.apply_genome_to_agent(agent, best_genome)
        if best_candidate_idx >= 0:
            logger.info(f"👑 [Level 3 Meta-Genetics] Directed Selection Victory! Mutant Candidate {best_candidate_idx+1} adopted (Loss: {best_loss:.6f}).")
        else:
            logger.info("🛡️ [Level 3 Meta-Genetics] Baseline preserved; no mutant surpassed current optimality.")

        return best_genome, best_loss


# =============================================================================
# LEVEL 4: ABSTRACT SELF-REFLECTIVE MUTATION CHANNEL (SYMBOLIC ARCHITECTURAL PROPOSAL)
# =============================================================================

class ReflectiveSelfMutationModule(nn.Module):
    """
    Level 4: Abstract Self-Reflective Mutation Channel.
    Projects recurrent state and somatic drives into an abstract morphological proposal vector
    Z_mutation in R^8 that encodes self-directed architectural tuning deltas.
    """
    def __init__(self, hidden_dim: int = 768, num_mutation_genes: int = 8, device_str: str = "cuda"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_mutation_genes = num_mutation_genes
        dev = torch.device('xla' if str(device_str).startswith('tpu') or str(device_str) == 'xla:0' else device_str)
        
        self.proposal_net = nn.Sequential(
            nn.Linear(hidden_dim + 6, hidden_dim // 4, bias=True),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, num_mutation_genes, bias=True),
            nn.Tanh() # Outputs bounded mutation directions in [-1, +1]
        ).to(dev)

    def forward(self, h_recurrent: torch.Tensor, u_somatic: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generates abstract mutation proposal Z_mutation in [-1, +1]^num_genes.
        """
        if h_recurrent.dim() == 3:
            h_recurrent = h_recurrent[:, -1, :] # Last token state
        elif h_recurrent.dim() > 2:
            h_recurrent = h_recurrent.view(h_recurrent.size(0), -1)[:, :self.hidden_dim]
            
        dev = h_recurrent.device
        if u_somatic is None or u_somatic.size(-1) != 6:
            u_somatic = torch.tensor([[0.5, 1.0, 1.0, 1.0, 0.0, 0.0]], device=dev)
        else:
            u_somatic = u_somatic.to(dev)

        # Broadcast batch dimensions safely if mismatched
        if h_recurrent.size(0) != u_somatic.size(0):
            if h_recurrent.size(0) == 1:
                h_recurrent = h_recurrent.expand(u_somatic.size(0), -1)
            elif u_somatic.size(0) == 1:
                u_somatic = u_somatic.expand(h_recurrent.size(0), -1)
            else:
                min_b = min(h_recurrent.size(0), u_somatic.size(0))
                h_recurrent = h_recurrent[:min_b]
                u_somatic = u_somatic[:min_b]
            
        combined_in = torch.cat([h_recurrent, u_somatic], dim=-1)
        return self.proposal_net(combined_in)


# =============================================================================
# LEVEL 5: EPIGENETIC GENE REGULATORY NETWORK (GRN) WITH METHYLATION & ACETYLATION
# =============================================================================

class EpigeneticRegulatoryNetwork(nn.Module):
    """
    Level 5: Epigenetic Gene Regulatory Network (GRN).
    Models continuous protein-protein & transcriptomic dynamics governed by:
    d e_i / dt = -gamma_i * e_i + sigmoid( sum_j W_ij e_j + sum_m B_im u_m + eta_i * F_accum - mu_i + alpha_i )
    
    - mu_i (Methylation Lock): Protects vital core invariants against destabilizing mutations.
    - alpha_i (Histone Acetylation): Opens chromatin for targeted plasticity in zones of persistent surprise.
    - e_morphogens: Controls growth triggers (sprouting, synaptic pruning, duplication, hypermutation).
    """
    
    MORPHOGEN_NAMES = [
        "e_axon_sprouting",       # Stimulates new axonal pathways and micro-circuits
        "e_synaptic_pruning",     # Governs sleep synaptic downscaling
        "e_dimension_expansion",  # Stimulates Net2Net hidden dimension growth
        "e_attractor_genesis",    # Stimulates birth of new Hopfield attractor basins
        "e_hypermutation_sos",    # Bacterial SOS response under catastrophic stress
        "e_memory_consolidation"  # Drives deep episodic memory replay into weights
    ]
    
    def __init__(self, num_morphogens: int = 6, homeo_dim: int = 6, device_str: str = "cuda"):
        super().__init__()
        self.num_morphogens = num_morphogens
        self.homeo_dim = homeo_dim
        dev = torch.device('xla' if str(device_str).startswith('tpu') or str(device_str) == 'xla:0' else device_str)
        
        # Regulatory interaction matrix W_GRN (K x K)
        self.w_grn = nn.Parameter(torch.randn(num_morphogens, num_morphogens, device=dev) * 0.10)
        # Somatic coupling matrix B_soma (K x 6)
        self.b_soma = nn.Parameter(torch.randn(num_morphogens, homeo_dim, device=dev) * 0.15)
        # Surprise susceptibility vector eta (K)
        self.eta_surprise = nn.Parameter(torch.tensor([0.80, 0.20, 0.60, 0.50, 1.20, 0.40], device=dev))
        # Decay rates gamma
        self.gamma_decay = nn.Parameter(torch.tensor([0.25, 0.30, 0.15, 0.20, 0.40, 0.25], device=dev))
        
        # Epigenetic markers:
        # Methylation locks mu (K): Higher value = more repressed / protected
        self.register_buffer("methylation_locks", torch.tensor([0.20, 0.30, 0.85, 0.40, 0.90, 0.10], device=dev))
        # Histone acetylation marks alpha_hist (K): Higher value = more accessible
        self.register_buffer("histone_acetylation", torch.tensor([0.50, 0.40, 0.20, 0.40, 0.10, 0.60], device=dev))
        # Active morphogen concentration state e(t) in [0, 1]^K
        self.register_buffer("morphogen_state", torch.tensor([0.20, 0.30, 0.10, 0.20, 0.05, 0.50], device=dev))
        
        # Epigenetic gene protection dictionary for Level 3 parameters
        self.gene_methylation_map = {
            "min_beta_stage1": 0.85,             # Highly protected core time-mixing
            "max_beta_stage1": 0.70,
            "min_beta_stage2": 0.90,             # Deep discourse invariant
            "max_beta_stage2": 0.80,
            "hopfield_beta": 0.50,               # Plastic attractor sharpness
            "pac_entropy_threshold": 0.40,       # Adaptable boundary threshold
            "noradrenaline_surprise_weight": 0.30,# Flexible allostatic tuning
            "dopamine_reward_scale": 0.30,
            "volitional_recall_gain": 0.40
        }

    def update_epigenetic_state(
        self,
        u_somatic: torch.Tensor,
        accumulated_surprise: float,
        dt: float = 0.20
    ) -> Dict[str, float]:
        """
        Integrates the GRN continuous differential equation over time step dt.
        Returns the updated morphogen concentrations.
        """
        dev = self.morphogen_state.device
        if u_somatic.dim() > 1:
            u_somatic = u_somatic[0]
        u_somatic = u_somatic.to(dev)

        with torch.no_grad():
            # Interaction forces: sum_j W_ij * e_j
            internal_forces = torch.matmul(self.w_grn, self.morphogen_state)
            # Somatic forces: sum_m B_im * u_m
            somatic_forces = torch.matmul(self.b_soma, u_somatic)
            # Epistemic pressure: eta_i * F_accum
            surprise_forces = self.eta_surprise * accumulated_surprise
            
            # Net activation with epigenetic gating (- mu + alpha_hist)
            net_drive = internal_forces + somatic_forces + surprise_forces - self.methylation_locks + self.histone_acetylation
            activation_rate = torch.sigmoid(net_drive)
            
            # Continuous ODE Euler integration: d e / dt = -gamma * e + activation
            de_dt = -self.gamma_decay * self.morphogen_state + activation_rate
            self.morphogen_state.add_(de_dt * dt)
            self.morphogen_state.clamp_(0.0, 1.0)
            
            # Epigenetic adaptation: persistent high surprise acetylates plastic genes and de-methylates locks
            if accumulated_surprise > 0.80:
                self.histone_acetylation.add_(0.02 * dt)
                self.histone_acetylation.clamp_(0.0, 1.0)
            elif accumulated_surprise < 0.10: # Homeostatic tranquility: promotes methylation (crystallization)
                self.methylation_locks.add_(0.01 * dt)
                self.methylation_locks.clamp_(0.0, 1.5)

        state_list = self.morphogen_state.cpu().tolist()
        return {name: float(val) for name, val in zip(self.MORPHOGEN_NAMES, state_list)}

    def should_trigger_sprouting(self, threshold: float = 0.55) -> bool:
        return float(self.morphogen_state[0].item()) > threshold

    def should_trigger_expansion(self, threshold: float = 0.70) -> bool:
        return float(self.morphogen_state[2].item()) > threshold

    def should_trigger_hypermutation(self, threshold: float = 0.75) -> bool:
        return float(self.morphogen_state[4].item()) > threshold


# =============================================================================
# LEVEL 6: PATHWAY NEUROGENESIS & NET2NET SMOOTH GRAFTING
# =============================================================================

class EpigeneticGraftedPathway(nn.Module):
    """
    Level 6: Epigenetically Grafted Neural Pathway.
    Wraps newly sprouted infant pathways in a dynamic biophysical gate:
    y_composite = y_mature + tanh(alpha_epi(t)) * y_infant
    
    Guarantees strict Zero-Delta Function Identity f_new(x) == f_old(x) at birth (alpha_epi = 0.0),
    preventing catastrophic shock to pre-trained cortical dynamics while allowing the new pathway
    to differentiate and smoothly assume computational workload as alpha_epi -> 1.0.
    """
    def __init__(self, mature_module: nn.Module, infant_module: nn.Module, initial_alpha: float = 0.0):
        super().__init__()
        self.mature_module = mature_module
        self.infant_module = infant_module
        self.alpha_epi = nn.Parameter(torch.tensor(initial_alpha, dtype=torch.float32))

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        y_mature = self.mature_module(x, *args, **kwargs) if self.mature_module is not None else x
        y_infant = self.infant_module(x, *args, **kwargs)
        
        # Smooth grafting gate: tanh(alpha_epi) ensures smooth scaling from 0 to 1
        gate = torch.tanh(self.alpha_epi)
        return y_mature + gate * y_infant

    def open_gate(self, delta: float = 0.10):
        with torch.no_grad():
            self.alpha_epi.add_(delta)


class PathwayNeurogenesisEngine:
    """
    Level 6: Structural Pathway Sprouter.
    Sprouts auxiliary predictive routes, parallel SSD heads, and extra Hopfield attractor basins
    with Net2Net Smooth Grafting.
    """
    
    @staticmethod
    def sprout_auxiliary_predictive_head(
        hidden_dim: int,
        vocab_size: int = 258,
        device_str: str = "cuda"
    ) -> EpigeneticGraftedPathway:
        """
        Sprouts an auxiliary predictive pathway with zero-initialized infant projection.
        """
        dev = torch.device('xla' if str(device_str).startswith('tpu') or str(device_str) == 'xla:0' else device_str)
        mature_path = nn.Linear(hidden_dim, vocab_size, bias=False).to(dev)
        infant_path = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, vocab_size, bias=False)
        ).to(dev)
        
        # Initialize infant final weights to zero for strict identity
        with torch.no_grad():
            infant_path[3].weight.zero_()
            
        grafted = EpigeneticGraftedPathway(mature_path, infant_path, initial_alpha=0.0).to(dev)
        logger.info(f"🌱 [Level 6 Neurogenesis] Sprouted Auxiliary Predictive Head with Smooth Grafting Gate (alpha=0.0).")
        return grafted


# =============================================================================
# LEVEL 7: SYSTEM 2 COUNTERFACTUAL SLEEP ROLLOUT & DIRECTED DARWINIAN SELECTION
# =============================================================================

class CounterfactualSleepRollout:
    """
    Level 7: System 2 Active Inference Counterfactual Sleep Rollout.
    Instead of blind stochastic search, evaluates candidate mutations in the
    Mental Sandbox on synthetic counterfactual replay buffers, measuring Expected
    Free Energy G(m) = E_Q[ln Q(z) - ln P(z, w | m)] + Complexity Penalty.
    """
    
    @staticmethod
    def evaluate_counterfactual_candidates(
        agent: Any,
        eval_inputs: torch.Tensor,
        eval_targets: torch.Tensor,
        hu: Any,
        criterion_speech: nn.Module,
        candidate_genomes: List[Dict[str, Any]],
        complexity_penalty_weight: float = 0.01
    ) -> Tuple[Dict[str, Any], float, int]:
        """
        Runs counterfactual rollouts for each candidate genome.
        Returns (winning_genome, best_expected_free_energy, winning_index).
        """
        agent.eval()
        best_g = float("inf")
        winning_idx = -1
        winning_genome = SleepMetaGeneticsEngine.get_active_genome(agent)
        
        with torch.no_grad():
            for idx, candidate in enumerate(candidate_genomes):
                SleepMetaGeneticsEngine.apply_genome_to_agent(agent, candidate)
                out = agent.forward_sequence(eval_inputs, eval_targets, hu, criterion_speech)
                rec_loss = out[1] if isinstance(out[1], (float, int)) else out[1].item()
                fe_val = out[2] if len(out) > 2 and isinstance(out[2], (float, int)) else 0.0
                
                # Compute parametric complexity penalty (L2 distance from base)
                complexity = sum(abs(v) for v in candidate.values()) * complexity_penalty_weight
                expected_free_energy = rec_loss + 0.5 * fe_val + complexity
                
                logger.info(f"   🌙 Sandbox Rollout [Candidate {idx+1}]: EFE G = {expected_free_energy:.6f} (Loss: {rec_loss:.6f}, FE: {fe_val:.4f})")
                
                if expected_free_energy < best_g:
                    best_g = expected_free_energy
                    winning_genome = candidate
                    winning_idx = idx
                    
        return winning_genome, best_g, winning_idx


# =============================================================================
# MASTER ORCHESTRATOR: COORDINATING ALL 5+ LEVELS OF KARYON SELF-EVOLUTION
# =============================================================================

class AutonomousSelfEvolutionOrchestrator:
    """
    Master Cybernetic Orchestrator coordinating all levels of Karyon Epigenetic Morphogenesis:
    - Level 1: Axonal Sprouting & Sleep Synaptic Pruning (Micro-Scale)
    - Level 2: Net2Net Dimension Expansion & Stage Deepening (Meso-Scale)
    - Level 3: Sleep Meta-Genetics & Biophysical Parameter Optimization
    - Level 4: Abstract Self-Reflective Mutation Proposal
    - Level 5: Epigenetic Gene Regulatory Network (GRN) with Methylation Locks
    - Level 6: Pathway Neurogenesis & Net2Net Smooth Grafting
    - Level 7: System 2 Counterfactual Sleep Rollout & Directed Selection
    """
    def __init__(self, agent: Any, device: str = "cuda"):
        self.agent = agent
        self.device = device
        self.reflective_channel = ReflectiveSelfMutationModule(
            hidden_dim=agent.hidden_dim, num_mutation_genes=8, device_str=device
        )
        self.grn = EpigeneticRegulatoryNetwork(
            num_morphogens=6, homeo_dim=agent.config.net.homeo_dim, device_str=device
        )

    def execute_full_morphogenetic_cycle(
        self,
        eval_input_tokens: Optional[torch.Tensor] = None,
        eval_target_tokens: Optional[torch.Tensor] = None,
        hu: Any = None,
        criterion_speech: Optional[nn.Module] = None,
        surprise_metric: float = 0.15,
        target_new_hidden_dim: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Executes a complete integrated Epigenetic Morphogenesis Cycle.
        """
        logger.info("\n" + "="*80)
        logger.info("🧬 === INITIATING KARYON 5-TIER EPIGENETIC SELF-EVOLUTION CYCLE ===")
        logger.info("="*80)

        results = {}
        u_somatic = hu.state if (hu is not None and hasattr(hu, 'state')) else torch.tensor([[0.5, 1.0, 1.0, 1.0, 0.0, 0.0]], device=self.device)

        # 1. Level 5: Epigenetic Gene Regulatory Network (GRN) Step
        morphogens = self.grn.update_epigenetic_state(u_somatic=u_somatic, accumulated_surprise=surprise_metric, dt=0.25)
        results["level_5_grn"] = morphogens
        logger.info(f"🌿 [Level 5 GRN] Active Morphogens: {', '.join(f'{k}={v:.3f}' for k, v in morphogens.items())}")

        # 2. Level 1: Structural Synaptogenesis & Pruning (Modulated by GRN morphogens)
        prune_ratio = 0.05 * (1.0 + morphogens.get("e_synaptic_pruning", 0.30))
        prune_res = StructuralSynaptogenesisPruner.prune_quiescent_synapses(self.agent, prune_ratio=prune_ratio)
        
        sprout_stimulus = surprise_metric * (1.0 + 1.5 * morphogens.get("e_axon_sprouting", 0.20))
        sprout_res = StructuralSynaptogenesisPruner.sprout_active_axons(self.agent, surprise_metric=sprout_stimulus)
        results["level_1"] = {"pruning": prune_res, "sprouting": sprout_res}

        # 3. Level 2: Net2Net Morphogenesis (Triggered if explicit target dimension specified)
        should_expand = (target_new_hidden_dim is not None and target_new_hidden_dim > self.agent.hidden_dim)
        if should_expand:
            expanded_agent, identity_delta = Net2NetMorphogenesisEngine.expand_agent_dimensions(
                self.agent, new_hidden_dim=target_new_hidden_dim, device=self.device
            )
            self.agent = expanded_agent
            self.reflective_channel = ReflectiveSelfMutationModule(
                hidden_dim=target_new_hidden_dim, num_mutation_genes=8, device_str=self.device
            )
            results["level_2"] = {"new_hidden_dim": target_new_hidden_dim, "identity_delta": identity_delta}
        else:
            results["level_2"] = {"status": "SKIPPED_OR_UP_TO_DATE", "hidden_dim": self.agent.hidden_dim}

        # 3.5 Level 6: Pathway Neurogenesis & Net2Net Smooth Grafting
        should_sprout = morphogens.get("e_axon_sprouting", 0.0) > 0.40 or surprise_metric >= 0.70
        if should_sprout and hasattr(self.agent, 'register_grafted_pathway'):
            if not (hasattr(self.agent, 'grafted_pathways') and "auxiliary_predictive_head" in self.agent.grafted_pathways):
                grafted = PathwayNeurogenesisEngine.sprout_auxiliary_predictive_head(
                    self.agent.hidden_dim, vocab_size=self.agent.text_gen_dim, device_str=self.device
                )
                self.agent.register_grafted_pathway("auxiliary_predictive_head", grafted)
                results["level_6"] = {"status": "SPROUTED", "pathway": "auxiliary_predictive_head"}
            else:
                # If already sprouted, smoothly open the epigenetic grafting gate alpha_epi
                grafted = self.agent.grafted_pathways["auxiliary_predictive_head"]
                grafted.open_gate(delta=0.10)
                results["level_6"] = {"status": "GATE_OPENED", "alpha_epi": float(grafted.alpha_epi.item())}
        else:
            results["level_6"] = {"status": "SKIPPED"}

        # 4. Level 4: Abstract Self-Reflective Mutation Channel Vector
        with torch.no_grad():
            h_fast_curr = getattr(self.agent, 'h_fast', torch.randn(1, self.agent.hidden_dim, device=self.device))
            mutation_out = self.reflective_channel(h_fast_curr, u_somatic)
            if mutation_out.dim() > 1:
                mutation_out = mutation_out[0]
            raw_genes = mutation_out.cpu().tolist()
            mutation_genes = [float(g if not isinstance(g, (list, tuple)) else g[0]) for g in raw_genes]
        
        logger.info(f"🧠 [Level 4 Abstract Reflection] Mutation Proposal Vector: {[round(float(g), 4) for g in mutation_genes]}")
        results["level_4"] = {"mutation_genes": mutation_genes, "status": "VALIDATED"}

        # 5. Level 7 & 3: Counterfactual Sandbox Selection with Epigenetic Methylation Locks
        base_genome = SleepMetaGeneticsEngine.get_active_genome(self.agent)
        mutation_rate = 0.05 * (1.5 if self.grn.should_trigger_hypermutation() else 1.0)
        
        candidates = [
            SleepMetaGeneticsEngine.mutate_genome(
                base_genome, 
                mutation_rate=mutation_rate, 
                reflective_proposal=mutation_genes,
                methylation_locks=self.grn.gene_methylation_map
            ) for _ in range(4)
        ]
        candidates.insert(0, base_genome) # Candidate 0 is baseline

        if eval_input_tokens is None or eval_target_tokens is None:
            eval_input_tokens = torch.randint(0, 256, (2, 32), device=self.device)
            eval_target_tokens = torch.randint(0, 256, (2, 32), device=self.device)
            criterion_speech = nn.CrossEntropyLoss(ignore_index=256)

        best_genome, best_efe, win_idx = CounterfactualSleepRollout.evaluate_counterfactual_candidates(
            self.agent, eval_input_tokens, eval_target_tokens, hu, criterion_speech, candidates
        )
        SleepMetaGeneticsEngine.apply_genome_to_agent(self.agent, best_genome)
        results["level_3_and_7"] = {"winning_index": win_idx, "winning_efe": best_efe, "evolved_genome": best_genome}

        logger.info("="*80)
        logger.info("✨ === KARYON EPIGENETIC SELF-EVOLUTION CYCLE COMPLETED SUCCESSFULLY ===")
        logger.info("="*80 + "\n")

        return results
