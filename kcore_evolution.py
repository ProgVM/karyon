# kcore_evolution.py
import copy
import json
import os
import struct
import torch
import torch.nn as nn
import numpy as np

class KaryonEvolver:
    def __init__(self, kcore_file_path="karyon_soul.kcore", device="cpu"):
        self.kcore_file_path = kcore_file_path
        self.device = device

    def mutate_genome(self, manifest):
        """Applies random architectural mutations to Karyon's DNA manifest with key safety."""
        default_genome = {
            "hidden_dim": 256,
            "latent_dim": 64,
            "sde_gamma": 0.1,
            "activation_func": "silu"
        }
        
        existing_genome = manifest.get("genome", {})
        for k, v in default_genome.items():
            if k not in existing_genome:
                existing_genome[k] = v

        mutated_genome = copy.deepcopy(existing_genome)
        mutation_type = np.random.choice(["expand_hidden", "expand_latent", "tune_gamma"])
        
        if mutation_type == "expand_hidden":
            mutated_genome["hidden_dim"] += 32
            print(f"[Neuroevolution] DNA Mutation: Expanding hidden_dim -> {mutated_genome['hidden_dim']}")
        elif mutation_type == "expand_latent":
            mutated_genome["latent_dim"] += 16
            print(f"[Neuroevolution] DNA Mutation: Expanding latent_dim -> {mutated_genome['latent_dim']}")
        elif mutation_type == "tune_gamma":
            curr_gamma = mutated_genome.get("sde_gamma", 0.1)
            mutated_genome["sde_gamma"] = float(np.clip(curr_gamma + np.random.uniform(-0.02, 0.02), 0.01, 0.5))
            print(f"[Neuroevolution] DNA Mutation: Tuned SDE gamma -> {mutated_genome['sde_gamma']:.4f}")
            
        manifest["genome"] = mutated_genome
        return manifest

    def morph_agent_weights(self, old_state_dict, old_genome, new_genome):
        """Morphs weight tensors preserving concatenated submatrix locations with 100% mathematical precision."""
        new_state_dict = {}
        
        old_h = old_genome.get("hidden_dim", 256)
        new_h = new_genome.get("hidden_dim", 256)
        
        old_z = old_genome.get("latent_dim", 64)
        new_z = new_genome.get("latent_dim", 64)

        unified_dim = 128
        action_dim = 3
        cog_dim = 3
        text_gen_dim = 258
        homeo_dim = 6

        for name, old_w in old_state_dict.items():
            # 1. world_model.prior_net [new_z * 2, new_h]
            if name == "world_model.prior_net.weight":
                new_w = torch.zeros((new_z * 2, new_h), dtype=old_w.dtype, device=old_w.device)
                new_w[:old_z * 2, :old_h] = old_w[:old_z * 2, :old_h]
                new_state_dict[name] = new_w
            elif name == "world_model.prior_net.bias":
                new_b = torch.zeros((new_z * 2,), dtype=old_w.dtype, device=old_w.device)
                new_b[:old_z * 2] = old_w[:old_z * 2]
                new_state_dict[name] = new_b

            # 2. world_model.posterior_net [new_z * 2, new_h + unified_dim]
            # Concat order: [h_fast (hidden_dim), w_t (unified_dim)]
            elif name == "world_model.posterior_net.weight":
                new_w = torch.zeros((new_z * 2, new_h + unified_dim), dtype=old_w.dtype, device=old_w.device)
                new_w[:old_z * 2, :old_h] = old_w[:old_z * 2, :old_h]
                new_w[:old_z * 2, new_h : new_h + unified_dim] = old_w[:old_z * 2, old_h : old_h + unified_dim]
                new_state_dict[name] = new_w
            elif name == "world_model.posterior_net.bias":
                new_b = torch.zeros((new_z * 2,), dtype=old_w.dtype, device=old_w.device)
                new_b[:old_z * 2] = old_w[:old_z * 2]
                new_state_dict[name] = new_b

            # 3. world_model.decoder_net.0 [unified_dim * 2, new_z + new_h]
            # Concat order: [z_t (latent_dim), h_slow (hidden_dim)]
            elif name == "world_model.decoder_net.0.weight":
                new_w = torch.zeros((unified_dim * 2, new_z + new_h), dtype=old_w.dtype, device=old_w.device)
                new_w[:, :old_z] = old_w[:, :old_z]
                new_w[:, new_z : new_z + old_h] = old_w[:, old_z : old_z + old_h]
                new_state_dict[name] = new_w
            elif name == "world_model.decoder_net.0.bias":
                new_state_dict[name] = old_w.clone()

            # 4. core.slow_f.0 [new_h, new_h + homeo_dim]
            # Concat order: [h_slow (hidden_dim), u_t (homeo_dim)]
            elif name == "core.slow_f.0.weight":
                new_w = torch.zeros((new_h, new_h + homeo_dim), dtype=old_w.dtype, device=old_w.device)
                new_w[:old_h, :old_h] = old_w[:old_h, :old_h]
                new_w[:old_h, new_h : new_h + homeo_dim] = old_w[:old_h, old_h : old_h + homeo_dim]
                new_state_dict[name] = new_w
            elif name == "core.slow_f.0.bias":
                new_b = torch.zeros((new_h,), dtype=old_w.dtype, device=old_w.device)
                new_b[:old_h] = old_w[:old_h]
                new_state_dict[name] = new_b

            # 5. core.fast_f.0 [new_h * 2, new_h + unified_dim + new_h]
            # Concat order: [h_fast (hidden_dim), w_t (unified_dim), h_slow (hidden_dim)]
            elif name == "core.fast_f.0.weight":
                new_w = torch.zeros((new_h * 2, new_h + unified_dim + new_h), dtype=old_w.dtype, device=old_w.device)
                new_w[:old_h * 2, :old_h] = old_w[:old_h * 2, :old_h]
                new_w[:old_h * 2, new_h : new_h + unified_dim] = old_w[:old_h * 2, old_h : old_h + unified_dim]
                new_w[:old_h * 2, new_h + unified_dim : new_h + unified_dim + old_h] = old_w[:old_h * 2, old_h + unified_dim : old_h + unified_dim + old_h]
                new_state_dict[name] = new_w
            elif name == "core.fast_f.0.bias":
                new_b = torch.zeros((new_h * 2,), dtype=old_w.dtype, device=old_w.device)
                new_b[:old_h * 2] = old_w[:old_h * 2]
                new_state_dict[name] = new_b

            # 6. core.fast_f.2 [new_h, new_h * 2]
            elif name == "core.fast_f.2.weight":
                new_w = torch.zeros((new_h, new_h * 2), dtype=old_w.dtype, device=old_w.device)
                new_w[:old_h, :old_h * 2] = old_w[:old_h, :old_h * 2]
                new_state_dict[name] = new_w
            elif name == "core.fast_f.2.bias":
                new_b = torch.zeros((new_h,), dtype=old_w.dtype, device=old_w.device)
                new_b[:old_h] = old_w[:old_h]
                new_state_dict[name] = new_b

            # 7. Gateway and Output Projections
            elif name in ["gateway.mind_proj.weight", "gateway.attention_query_layer.weight",
                          "output_gateway.motor_action.weight", "output_gateway.cognitive_gating.weight",
                          "output_gateway.text_generation.weight", "critic.weight"]:
                out_d, in_d = old_w.shape
                target_out = out_d
                target_in = new_h if in_d == old_h else in_d
                new_w = torch.zeros((target_out, target_in), dtype=old_w.dtype, device=old_w.device)
                new_w[:out_d, :old_h] = old_w[:out_d, :old_h]
                new_state_dict[name] = new_w

            else:
                new_state_dict[name] = old_w.clone()

        return new_state_dict
