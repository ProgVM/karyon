#include <torch/extension.h>
#include <vector>
#include <string>
#include <cmath>
#include <tuple>
#include <memory>
#include <algorithm>
#include <iostream>

// ============================================================================
// KARYON COGNITIVE SUBSTRATE & ALLOSENSORY HOMEOSTASIS CORE
// v32.0 - Active Inference, Dynamic Homeostasis, and Epigenetic Neurogenesis
// ============================================================================

// 1. BYTE-LEVEL UNIVERSAL REPRESENTATION & EMBEDDING MANIFOLD
struct UniversalManifoldImpl : public torch::nn::Module {
    int64_t vocab_size;
    int64_t dim;
    torch::nn::Embedding embedding{nullptr};

    UniversalManifoldImpl(int64_t vocab_size = 258, int64_t dim = 256, std::string device_str = "cpu")
        : vocab_size(vocab_size), dim(dim) {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        embedding = register_module("embedding", torch::nn::Embedding(vocab_size, dim));
        torch::nn::init::normal_(embedding->weight, 0.0, 0.02);
        this->to(device);
    }

    torch::Tensor forward(torch::Tensor tokens) {
        return embedding->forward(tokens);
    }
};
TORCH_MODULE(UniversalManifold);

// 2. CAUSAL PARALLEL SSD SCAN OPERATOR
class CausalParallelSSDImpl : public torch::nn::Module {
public:
    int64_t dim;
    torch::Tensor log_decay;

    CausalParallelSSDImpl(int64_t dim, std::string device_str = "cpu") : dim(dim) {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        auto init_decay = torch::linspace(std::log(0.005f), std::log(0.2f), dim, torch::TensorOptions().device(device));
        log_decay = register_parameter("log_decay", init_decay);
        this->to(device);
    }

    torch::Tensor forward(torch::Tensor x) {
        // x: [batch, seq_len, dim]
        int64_t batch = x.size(0);
        int64_t seq_len = x.size(1);
        auto device = x.device();

        auto decay = torch::exp(log_decay).view({dim, 1, 1}); // [dim, 1, 1]
        auto indices = torch::arange(seq_len, torch::TensorOptions().device(device).dtype(torch::kFloat32));
        auto delta_pos = (indices.view({seq_len, 1}) - indices.view({1, seq_len})).clamp_min(0.0f); // [seq_len, seq_len]
        auto causal_mask = (indices.view({seq_len, 1}) >= indices.view({1, seq_len})).to(torch::kFloat32);

        // decay_kernel: [dim, seq_len, seq_len] where decay_kernel[d, t, s] = exp(-decay[d] * (t - s)) for s <= t
        auto decay_kernel = torch::exp(-delta_pos.unsqueeze(0) * decay) * causal_mask.unsqueeze(0);

        // Transpose x to [batch, dim, seq_len]
        auto x_perm = x.permute({0, 2, 1}); // [batch, dim, seq_len]

        // For each batch item b and dim d: y[b, d] = decay_kernel[d] @ x[b, d]
        // decay_kernel: [dim, seq_len, seq_len], x_perm: [batch, dim, seq_len]
        // Broadcasted bmm: [batch, dim, seq_len, 1] -> [batch, dim, seq_len]
        auto y_perm = torch::matmul(decay_kernel.unsqueeze(0), x_perm.unsqueeze(-1)).squeeze(-1);

        // Permute back to [batch, seq_len, dim]
        return y_perm.permute({0, 2, 1});
    }
};
TORCH_MODULE(CausalParallelSSD);

// 3. PARALLEL OPERATOR BANK
class ParallelOperatorBankImpl : public torch::nn::Module {
public:
    int64_t dim;
    int64_t state_dim;
    int64_t num_operators;

    ParallelOperatorBankImpl(int64_t dim, int64_t state_dim = 128, int64_t num_operators = 8)
        : dim(dim), state_dim(state_dim), num_operators(num_operators) {}

    torch::Tensor compute_operators(torch::Tensor h_state) {
        bool is_3d = (h_state.dim() == 3);
        auto x = is_3d ? h_state : h_state.unsqueeze(1);

        auto op_0 = x;
        auto op_1 = torch::gelu(x);
        auto op_2 = torch::silu(x);
        auto op_3 = torch::tanh(x);
        auto op_4 = torch::sin(x * 3.14159265f);
        auto op_5 = torch::abs(x);
        auto op_6 = x * torch::sigmoid(x);
        auto op_7 = torch::cos(x) * torch::sin(x);

        auto stacked = torch::stack({op_0, op_1, op_2, op_3, op_4, op_5, op_6, op_7}, -1);
        if (!is_3d) {
            stacked = stacked.squeeze(1);
        }
        return stacked;
    }
};
TORCH_MODULE(ParallelOperatorBank);

// 4. LATENT ACTIVE INFERENCE PREDICTOR (Free Energy Engine)
class LatentPredictorImpl : public torch::nn::Module {
public:
    int64_t dim;
    int64_t latent_dim;

    torch::nn::Linear prior_mu{nullptr};
    torch::nn::Linear prior_logvar{nullptr};
    torch::nn::Linear post_mu{nullptr};
    torch::nn::Linear post_logvar{nullptr};

    LatentPredictorImpl(int64_t dim, int64_t latent_dim = 64, std::string device_str = "cpu")
        : dim(dim), latent_dim(latent_dim) {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

        prior_mu = register_module("prior_mu", torch::nn::Linear(dim, latent_dim));
        prior_logvar = register_module("prior_logvar", torch::nn::Linear(dim, latent_dim));
        post_mu = register_module("post_mu", torch::nn::Linear(dim, latent_dim));
        post_logvar = register_module("post_logvar", torch::nn::Linear(dim, latent_dim));

        this->to(device);
    }

    // Computes dimension-normalized KL surprise
    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor h_state, torch::Tensor target_emb) {
        // h_state: [batch, seq_len, dim] or [batch, dim]
        auto prior_m = prior_mu->forward(h_state);
        auto prior_lv = prior_logvar->forward(h_state).clamp(-10.0f, 10.0f);

        auto post_m = post_mu->forward(target_emb);
        auto post_lv = post_logvar->forward(target_emb).clamp(-10.0f, 10.0f);

        auto var_prior = torch::exp(prior_lv);
        auto var_post = torch::exp(post_lv);

        // KL Divergence
        auto kl = 0.5f * (prior_lv - post_lv + (var_post + torch::pow(post_m - prior_m, 2)) / var_prior - 1.0f);
        auto mean_kl = kl.mean(-1); // Average over latent dimensions

        // Sample latent z via reparameterization trick
        auto std_dev = torch::exp(0.5f * post_lv);
        auto eps = torch::randn_like(std_dev);
        auto z = post_m + eps * std_dev;

        return std::make_tuple(mean_kl, z);
    }
};
TORCH_MODULE(LatentPredictor);

// 5. ALLOSTATIC HOMEOSTATIC NEXUS (Dynamic Multi-Dimensional Homeostasis)
class HomeostaticNexusImpl : public torch::nn::Module {
public:
    std::vector<std::string> variable_names;
    torch::Tensor current_values;
    torch::Tensor target_values;
    torch::Tensor decay_rates;
    torch::Tensor sensitivities;

    std::string device_str;

    HomeostaticNexusImpl(std::string device_str = "cpu") : device_str(device_str) {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

        // Initialize with 6 core biophysical variables
        variable_names = {"Curiosity", "Energy", "Stability", "Health", "Noradrenaline", "Dopamine"};
        
        current_values = register_buffer("current_values", torch::tensor({0.8f, 1.0f, 0.9f, 1.0f, 0.15f, 0.15f}, torch::TensorOptions().device(device)));
        target_values = register_buffer("target_values", torch::tensor({1.0f, 1.0f, 1.0f, 1.0f, 0.15f, 0.15f}, torch::TensorOptions().device(device)));
        decay_rates = register_buffer("decay_rates", torch::tensor({0.01f, 0.005f, 0.02f, 0.001f, 0.1f, 0.1f}, torch::TensorOptions().device(device)));
        sensitivities = register_buffer("sensitivities", torch::tensor({0.1f, 0.2f, 0.05f, 0.02f, 0.5f, 0.3f}, torch::TensorOptions().device(device)));

        this->to(device);
    }

    // Dynamic Homeostatic Sprouting: Spontaneous expansion of allostatic space
    void sprout_dimension(std::string name, float init_val, float target_val, float decay, float sensitivity) {
        if (std::find(variable_names.begin(), variable_names.end(), name) != variable_names.end()) {
            return; // Dimension already exists
        }

        auto device = current_values.device();
        variable_names.push_back(name);

        // Concatenate parameters
        auto new_current = torch::cat({current_values, torch::tensor({init_val}, torch::TensorOptions().device(device))});
        auto new_target = torch::cat({target_values, torch::tensor({target_val}, torch::TensorOptions().device(device))});
        auto new_decay = torch::cat({decay_rates, torch::tensor({decay}, torch::TensorOptions().device(device))});
        auto new_sensitivity = torch::cat({sensitivities, torch::tensor({sensitivity}, torch::TensorOptions().device(device))});

        // Re-assign buffers directly without re-registering to avoid PyTorch buffer collision
        current_values = new_current;
        target_values = new_target;
        decay_rates = new_decay;
        sensitivities = new_sensitivity;
    }

    // Update homeostasis based on active inference surprise (Free Energy)
    torch::Tensor update(torch::Tensor free_energy, torch::Tensor reward) {
        auto device = current_values.device();
        
        auto mean_fe = free_energy.mean().item<float>();
        auto mean_r = reward.mean().item<float>();

        auto old_vals = current_values.clone();
        auto target_diff = target_values - old_vals;
        
        // Base SDE allostatic update
        auto updated = old_vals + decay_rates * target_diff;

        // Influence of surprise
        updated[0] = (updated[0] + 0.01f * mean_fe).clamp(0.0f, 1.0f); // Reduced sensitivity from 0.1 to 0.01
        updated[1] = (updated[1] - 0.005f * mean_fe).clamp(0.0f, 1.0f); // Reduced energy drain
        updated[2] = (updated[2] - 0.01f * mean_fe).clamp(0.0f, 1.0f);
        updated[4] = (updated[4] + sensitivities[4] * 0.05f * mean_fe).clamp(0.0f, 1.0f); // Dampened NA spike
        updated[5] = (updated[5] + sensitivities[5] * mean_r - 0.01f * mean_fe).clamp(0.0f, 1.0f);

        for (size_t i = 6; i < variable_names.size(); ++i) {
            updated[i] = (updated[i] - sensitivities[i] * 0.01f * mean_fe).clamp(0.0f, 1.0f);
        }

        current_values.copy_(updated);
        return current_values;
    }

    torch::Tensor get_values() {
        return current_values;
    }

    std::vector<std::string> get_names() {
        return variable_names;
    }
};
TORCH_MODULE(HomeostaticNexus);

// 6. OMNI-MORPHIC CAUSAL COGNITIVE NODE
class OmniMorphicNodeImpl : public torch::nn::Module {
public:
    int64_t dim;
    int64_t state_dim;
    int64_t num_operators;

    torch::Tensor affinity_query;

    torch::nn::Linear hyper_w1{nullptr};
    torch::nn::Linear hyper_w2{nullptr};
    torch::nn::Linear base_in{nullptr};
    torch::nn::Linear base_out{nullptr};
    torch::nn::LayerNorm in_norm{nullptr};
    torch::nn::LayerNorm state_norm{nullptr};

    CausalParallelSSD causal_ssd{nullptr};
    std::shared_ptr<ParallelOperatorBankImpl> op_bank{nullptr};

    OmniMorphicNodeImpl(int64_t dim, int64_t state_dim = 128, int64_t num_operators = 8, std::string device_str = "cpu")
        : dim(dim), state_dim(state_dim), num_operators(num_operators) {

        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

        affinity_query = register_parameter("affinity_query", torch::randn({dim}, torch::TensorOptions().device(device)) * (1.0 / std::sqrt(dim)));

        hyper_w1 = register_module("hyper_w1", torch::nn::Linear(dim, dim / 2));
        hyper_w2 = register_module("hyper_w2", torch::nn::Linear(dim / 2, (dim * state_dim) + (state_dim * dim) + num_operators + 4));

        base_in = register_module("base_in", torch::nn::Linear(torch::nn::LinearOptions(dim, state_dim).bias(false)));
        base_out = register_module("base_out", torch::nn::Linear(torch::nn::LinearOptions(state_dim, dim).bias(false)));
        in_norm = register_module("in_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));
        state_norm = register_module("state_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({state_dim})));

        causal_ssd = register_module("causal_ssd", CausalParallelSSD(state_dim, device_str));
        op_bank = std::make_shared<ParallelOperatorBankImpl>(dim, state_dim, num_operators);
        register_module("op_bank", op_bank);

        torch::nn::init::normal_(hyper_w1->weight, 0.0, 0.02);
        torch::nn::init::zeros_(hyper_w1->bias);
        torch::nn::init::normal_(hyper_w2->weight, 0.0, 0.02);
        torch::nn::init::zeros_(hyper_w2->bias);

        this->to(device);
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor signal_manifold, torch::Tensor u_t) {
        bool is_4d = (signal_manifold.dim() == 4);
        int64_t batch, num_signals, seq_len;
        torch::Tensor manifold_ref = signal_manifold;

        if (is_4d) {
            batch = signal_manifold.size(0);
            num_signals = signal_manifold.size(1);
            seq_len = signal_manifold.size(2);
        } else {
            batch = signal_manifold.size(0);
            num_signals = signal_manifold.size(1);
            seq_len = 1;
            manifold_ref = signal_manifold.unsqueeze(2);
        }

        auto keys = manifold_ref.mean(2);
        auto q = affinity_query.view({1, 1, dim}).expand({batch, 1, dim});
        auto attn_logits = torch::matmul(q, keys.transpose(1, 2)).squeeze(1) * (1.0 / std::sqrt(dim));
        auto attn_weights = torch::softmax(attn_logits, -1);

        auto weights_expanded = attn_weights.unsqueeze(-1).unsqueeze(-1);
        auto x_attended = torch::sum(manifold_ref * weights_expanded, 1);
        x_attended = in_norm->forward(x_attended);

        auto context = x_attended.mean(1);
        if (u_t.defined() && u_t.numel() > 0) {
            auto u_flat = (u_t.dim() > 1) ? u_t.view({batch, -1}) : u_t.unsqueeze(0).expand({batch, -1});
            if (u_flat.size(0) == batch && u_flat.size(-1) <= dim) {
                context = context + torch::constant_pad_nd(u_flat, {0, dim - u_flat.size(-1)});
            }
        }

        auto hyper_h = torch::silu(hyper_w1->forward(context));
        auto morphic_params = hyper_w2->forward(hyper_h);

        double scale_in = 1.0 / std::sqrt(dim);
        double scale_out = 1.0 / std::sqrt(state_dim);

        int64_t ptr = 0;
        auto w_in = morphic_params.slice(1, ptr, ptr + (dim * state_dim)).view({batch, dim, state_dim}) * scale_in;
        ptr += dim * state_dim;
        auto w_out = morphic_params.slice(1, ptr, ptr + (state_dim * dim)).view({batch, state_dim, dim}) * scale_out;
        ptr += state_dim * dim;

        auto operator_coeffs = torch::softmax(morphic_params.slice(1, ptr, ptr + num_operators), -1).view({batch, 1, 1, num_operators});

        auto projected = base_in->forward(x_attended) + torch::matmul(x_attended, w_in);
        projected = state_norm->forward(projected);

        auto causal_states = causal_ssd->forward(projected);

        auto all_ops = op_bank->compute_operators(causal_states);
        auto blended = torch::sum(all_ops * operator_coeffs, -1);

        auto broadcast_signal = base_out->forward(blended) + torch::matmul(blended, w_out);
        if (!is_4d) {
            broadcast_signal = broadcast_signal.squeeze(1);
        }

        return std::make_tuple(broadcast_signal, attn_weights);
    }
};
TORCH_MODULE(OmniMorphicNode);

// 7. OMNI-CONTINUOUS GRAPH SUBSTRATE WITH EPIGENETIC SPONTANEOUS NEUROGENESIS
class OmniContinuousGraphSubstrateImpl : public torch::nn::Module {
public:
    int64_t dim;
    int64_t max_nodes;
    std::string device_str;

    std::vector<std::shared_ptr<OmniMorphicNodeImpl>> nodes;
    std::vector<torch::Tensor> alpha_nodes;
    std::vector<std::string> node_names;

    OmniContinuousGraphSubstrateImpl(int64_t dim, int64_t max_nodes = 16, std::string device_str = "cpu")
        : dim(dim), max_nodes(max_nodes), device_str(device_str) {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        this->to(device);
    }

    bool sprout_node(std::string name, int64_t state_dim = 128, int64_t num_operators = 8) {
        if (nodes.size() >= max_nodes || std::find(node_names.begin(), node_names.end(), name) != node_names.end()) {
            return false;
        }

        auto node = std::make_shared<OmniMorphicNodeImpl>(dim, state_dim, num_operators, device_str);
        std::string node_key = "node_" + std::to_string(nodes.size());
        register_module(node_key, node);
        nodes.push_back(node);

        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        auto alpha = register_parameter(node_key + "_alpha", torch::zeros({}, torch::TensorOptions().device(device)));
        alpha_nodes.push_back(alpha);
        node_names.push_back(name);

        return true;
    }

    // Spontaneous Neurogenesis: Sprout node on high Free Energy surprise
    bool trigger_spontaneous_neurogenesis(float free_energy_surprise, float threshold = 1.2f) {
        if (free_energy_surprise > threshold && nodes.size() < max_nodes) {
            std::string generated_name = "spontaneous_node_" + std::to_string(nodes.size());
            return sprout_node(generated_name, 128, 8);
        }
        return false;
    }

    // Neural Darwinism Pruning: Decay and prune inactive nodes (where alpha is close to zero)
    int64_t execute_neural_darwinism(float decay_rate = 0.001f, float prune_threshold = 0.005f) {
        int64_t pruned_count = 0;
        torch::NoGradGuard no_grad;
        for (size_t i = 0; i < alpha_nodes.size(); ++i) {
            // Apply slight L1 decay on alphas to encourage sparsification (Darwinian pressure)
            auto val = alpha_nodes[i].item<float>();
            if (std::abs(val) > 0.0f) {
                float sign = (val > 0.0f) ? 1.0f : -1.0f;
                alpha_nodes[i].add_(-decay_rate * sign);
            }
        }
        return pruned_count;
    }

    std::tuple<torch::Tensor, std::vector<torch::Tensor>> forward(std::vector<torch::Tensor> signal_list, torch::Tensor u_t) {
        if (signal_list.empty()) {
            throw std::invalid_argument("signal_list cannot be empty");
        }

        if (nodes.empty()) {
            return std::make_tuple(signal_list[0], signal_list);
        }

        auto sample = signal_list[0];
        bool is_3d = (sample.dim() == 3);
        int64_t seq_len = is_3d ? sample.size(1) : 1;

        std::vector<torch::Tensor> norm_signals;
        for (const auto& s : signal_list) {
            if (s.dim() == 2) {
                norm_signals.push_back(s.unsqueeze(1).expand({-1, seq_len, -1}));
            } else {
                norm_signals.push_back(s);
            }
        }

        std::vector<torch::Tensor> current_signals = norm_signals;

        for (size_t i = 0; i < nodes.size(); ++i) {
            auto manifold_stack = torch::stack(current_signals, 1);
            
            torch::Tensor node_out, attn_w;
            std::tie(node_out, attn_w) = nodes[i]->forward(manifold_stack, u_t);

            auto alpha = alpha_nodes[i];
            auto gate = torch::tanh(alpha);
            auto gated_signal = gate * node_out;
            current_signals.push_back(gated_signal);
        }

        auto final_signal = current_signals[0].clone();
        for (size_t i = norm_signals.size(); i < current_signals.size(); ++i) {
            final_signal = final_signal + current_signals[i];
        }

        if (!is_3d) {
            final_signal = final_signal.squeeze(1);
        }

        return std::make_tuple(final_signal, current_signals);
    }
};
TORCH_MODULE(OmniContinuousGraphSubstrate);

// 8. THE COMPLETE EVOLVABLE ACTIVE INFERENCE COGNITIVE AGENT
class CognitiveEvolvableAgentImpl : public torch::nn::Module {
public:
    int64_t vocab_size;
    int64_t dim;
    std::string device_str;

    UniversalManifold manifold{nullptr};
    OmniContinuousGraphSubstrate substrate{nullptr};
    LatentPredictor latent_predictor{nullptr};
    HomeostaticNexus homeostasis{nullptr};
    torch::nn::Linear motor_head{nullptr};

    CognitiveEvolvableAgentImpl(int64_t vocab_size = 258, int64_t dim = 256, int64_t max_nodes = 16, std::string device_str = "cpu")
        : vocab_size(vocab_size), dim(dim), device_str(device_str) {

        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

        manifold = register_module("manifold", UniversalManifold(vocab_size, dim, device_str));
        substrate = register_module("substrate", OmniContinuousGraphSubstrate(dim, max_nodes, device_str));
        latent_predictor = register_module("latent_predictor", LatentPredictor(dim, 64, device_str));
        homeostasis = register_module("homeostasis", HomeostaticNexus(device_str));
        motor_head = register_module("motor_head", torch::nn::Linear(dim, vocab_size));

        torch::nn::init::normal_(motor_head->weight, 0.0, 0.02);
        torch::nn::init::zeros_(motor_head->bias);

        this->to(device);
    }

    bool sprout_organelle(std::string name, int64_t state_dim = 128, int64_t num_operators = 8) {
        return substrate->sprout_node(name, state_dim, num_operators);
    }

    void sprout_homeostatic_dimension(std::string name, float init_val, float target_val, float decay, float sensitivity) {
        homeostasis->sprout_dimension(name, init_val, target_val, decay, sensitivity);
    }

    // High-speed parallel forward pass
    torch::Tensor forward(torch::Tensor tokens, torch::Tensor u_t) {
        auto emb = manifold->forward(tokens); // [batch, seq_len, dim]
        
        std::vector<torch::Tensor> signals = {emb};
        torch::Tensor final_emb;
        std::vector<torch::Tensor> updated_signals;
        std::tie(final_emb, updated_signals) = substrate->forward(signals, u_t);

        return motor_head->forward(final_emb); // [batch, seq_len, vocab_size]
    }

    torch::Tensor forward(torch::Tensor tokens) {
        auto u_t = homeostasis->get_values();
        return forward(tokens, u_t);
    }
    // High-speed parallel forward pass returning final latent embedding
    torch::Tensor forward_latent(torch::Tensor tokens, torch::Tensor u_t) {
        auto emb = manifold->forward(tokens); // [batch, seq_len, dim]
        
        std::vector<torch::Tensor> signals = {emb};
        torch::Tensor final_emb;
        std::vector<torch::Tensor> updated_signals;
        std::tie(final_emb, updated_signals) = substrate->forward(signals, u_t);

        return final_emb; // [batch, seq_len, dim]
    }

    torch::Tensor forward_latent(torch::Tensor tokens) {
        auto u_t = homeostasis->get_values();
        return forward_latent(tokens, u_t);
    }

    // Forward pass taking an injected external latent tensor and projecting to logits
    torch::Tensor forward_motor(torch::Tensor latent) {
        return motor_head->forward(latent);
    }

    // Active Inference forward pass returning logits, Free Energy, and updated homeostasis
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward_active_inference(torch::Tensor tokens, torch::Tensor reward) {
        auto u_t = homeostasis->get_values();
        auto emb = manifold->forward(tokens);
        
        std::vector<torch::Tensor> signals = {emb};
        torch::Tensor final_emb;
        std::vector<torch::Tensor> updated_signals;
        std::tie(final_emb, updated_signals) = substrate->forward(signals, u_t);

        // Compute Active Inference Free Energy
        torch::Tensor free_energy, z;
        std::tie(free_energy, z) = latent_predictor->forward(final_emb, emb);

        // Update homeostatic values
        auto updated_u_t = homeostasis->update(free_energy, reward);

        // Spontaneous neurogenesis and Darwinian decay
        float mean_fe = free_energy.mean().item<float>();
        substrate->trigger_spontaneous_neurogenesis(mean_fe, 1.2f);
        substrate->execute_neural_darwinism(0.001f, 0.005f);

        auto logits = motor_head->forward(final_emb);
        return std::make_tuple(logits, free_energy, updated_u_t);
    }

    // Autoregressive Top-P PAC Decoder (Vector 3)
    torch::Tensor generate_thought_and_speech(torch::Tensor seed_tokens, int64_t max_new_tokens, float temperature = 0.45f, float top_p = 0.90f) {
        auto device = seed_tokens.device();
        auto current_seq = seed_tokens.clone();

        for (int64_t i = 0; i < max_new_tokens; ++i) {
            auto logits = forward(current_seq);
            auto next_token_logits = logits.select(1, -1) / temperature; // [batch, vocab_size]

            // Apply Top-P Nucleus Sampling
            auto probs = torch::softmax(next_token_logits, -1);
            auto sorted_probs_tuple = torch::sort(probs, -1, true);
            auto sorted_probs = std::get<0>(sorted_probs_tuple);
            auto sorted_indices = std::get<1>(sorted_probs_tuple);

            auto cumulative_probs = torch::cumsum(sorted_probs, -1);
            auto mask = cumulative_probs - sorted_probs > top_p;
            sorted_probs.masked_fill_(mask, 0.0f);
            sorted_probs = sorted_probs / sorted_probs.sum(-1, true);

            auto next_token_idx = torch::multinomial(sorted_probs, 1);
            auto next_token = sorted_indices.gather(-1, next_token_idx);

            current_seq = torch::cat({current_seq, next_token}, 1);
        }

        return current_seq;
    }
};
TORCH_MODULE(CognitiveEvolvableAgent);

// ============================================================================
// PYBIND11 MODULE BINDINGS FOR THE NEW HOMEOSTATIC EVOLUTION ERA
// ============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<UniversalManifoldImpl, torch::nn::Module, std::shared_ptr<UniversalManifoldImpl>>(m, "UniversalManifold")
        .def(py::init<int64_t, int64_t, std::string>(), py::arg("vocab_size") = 258, py::arg("dim") = 256, py::arg("device") = "cpu")
        .def("forward", &UniversalManifoldImpl::forward)
        .def("__call__", &UniversalManifoldImpl::forward);

    py::class_<CausalParallelSSDImpl, torch::nn::Module, std::shared_ptr<CausalParallelSSDImpl>>(m, "CausalParallelSSD")
        .def(py::init<int64_t, std::string>(), py::arg("dim") = 256, py::arg("device") = "cpu")
        .def("forward", &CausalParallelSSDImpl::forward)
        .def("__call__", &CausalParallelSSDImpl::forward);

    py::class_<ParallelOperatorBankImpl, torch::nn::Module, std::shared_ptr<ParallelOperatorBankImpl>>(m, "ParallelOperatorBank")
        .def(py::init<int64_t, int64_t, int64_t>(), py::arg("dim") = 256, py::arg("state_dim") = 128, py::arg("num_operators") = 8)
        .def("compute_operators", &ParallelOperatorBankImpl::compute_operators);

    py::class_<LatentPredictorImpl, torch::nn::Module, std::shared_ptr<LatentPredictorImpl>>(m, "LatentPredictor")
        .def(py::init<int64_t, int64_t, std::string>(), py::arg("dim") = 256, py::arg("latent_dim") = 64, py::arg("device") = "cpu")
        .def("forward", &LatentPredictorImpl::forward)
        .def("__call__", &LatentPredictorImpl::forward);

    py::class_<HomeostaticNexusImpl, torch::nn::Module, std::shared_ptr<HomeostaticNexusImpl>>(m, "HomeostaticNexus")
        .def(py::init<std::string>(), py::arg("device") = "cpu")
        .def("sprout_dimension", &HomeostaticNexusImpl::sprout_dimension)
        .def("update", &HomeostaticNexusImpl::update)
        .def("get_values", &HomeostaticNexusImpl::get_values)
        .def("get_names", &HomeostaticNexusImpl::get_names);

    py::class_<OmniMorphicNodeImpl, torch::nn::Module, std::shared_ptr<OmniMorphicNodeImpl>>(m, "OmniMorphicNode")
        .def(py::init<int64_t, int64_t, int64_t, std::string>(), py::arg("dim") = 256, py::arg("state_dim") = 128, py::arg("num_operators") = 8, py::arg("device") = "cpu")
        .def("forward", &OmniMorphicNodeImpl::forward, py::arg("signal_manifold"), py::arg("u_t") = torch::Tensor())
        .def("__call__", &OmniMorphicNodeImpl::forward, py::arg("signal_manifold"), py::arg("u_t") = torch::Tensor());

    py::class_<OmniContinuousGraphSubstrateImpl, torch::nn::Module, std::shared_ptr<OmniContinuousGraphSubstrateImpl>>(m, "OmniContinuousGraphSubstrate")
        .def(py::init<int64_t, int64_t, std::string>(), py::arg("dim") = 256, py::arg("max_nodes") = 16, py::arg("device") = "cpu")
        .def_readonly("node_names", &OmniContinuousGraphSubstrateImpl::node_names)
        .def("sprout_node", &OmniContinuousGraphSubstrateImpl::sprout_node, py::arg("name"), py::arg("state_dim") = 128, py::arg("num_operators") = 8)
        .def("trigger_spontaneous_neurogenesis", &OmniContinuousGraphSubstrateImpl::trigger_spontaneous_neurogenesis)
        .def("execute_neural_darwinism", &OmniContinuousGraphSubstrateImpl::execute_neural_darwinism)
        .def("forward", &OmniContinuousGraphSubstrateImpl::forward, py::arg("signal_list"), py::arg("u_t") = torch::Tensor())
        .def("__call__", &OmniContinuousGraphSubstrateImpl::forward, py::arg("signal_list"), py::arg("u_t") = torch::Tensor());

    py::class_<CognitiveEvolvableAgentImpl, torch::nn::Module, std::shared_ptr<CognitiveEvolvableAgentImpl>>(m, "CognitiveEvolvableAgent")
        .def(py::init<int64_t, int64_t, int64_t, std::string>(), py::arg("vocab_size") = 258, py::arg("dim") = 256, py::arg("max_nodes") = 16, py::arg("device") = "cpu")
        .def_property_readonly("homeostasis", [](std::shared_ptr<CognitiveEvolvableAgentImpl> a) { return a->homeostasis.ptr(); })
        .def_property_readonly("substrate", [](std::shared_ptr<CognitiveEvolvableAgentImpl> a) { return a->substrate.ptr(); })
        .def("sprout_organelle", &CognitiveEvolvableAgentImpl::sprout_organelle, py::arg("name"), py::arg("state_dim") = 128, py::arg("num_operators") = 8)
        .def("sprout_homeostatic_dimension", &CognitiveEvolvableAgentImpl::sprout_homeostatic_dimension, py::arg("name"), py::arg("init_val"), py::arg("target_val"), py::arg("decay"), py::arg("sensitivity"))
        .def("forward", py::overload_cast<torch::Tensor, torch::Tensor>(&CognitiveEvolvableAgentImpl::forward), py::arg("tokens"), py::arg("u_t"))
        .def("forward", py::overload_cast<torch::Tensor>(&CognitiveEvolvableAgentImpl::forward), py::arg("tokens"))
        .def("__call__", py::overload_cast<torch::Tensor, torch::Tensor>(&CognitiveEvolvableAgentImpl::forward), py::arg("tokens"), py::arg("u_t"))
        .def("__call__", py::overload_cast<torch::Tensor>(&CognitiveEvolvableAgentImpl::forward), py::arg("tokens"))
        .def("forward_active_inference", &CognitiveEvolvableAgentImpl::forward_active_inference, py::arg("tokens"), py::arg("reward"))
        .def("generate_thought_and_speech", &CognitiveEvolvableAgentImpl::generate_thought_and_speech, py::arg("seed_tokens"), py::arg("max_new_tokens"), py::arg("temperature") = 0.45f, py::arg("top_p") = 0.90f)
        .def("parameters", [](std::shared_ptr<CognitiveEvolvableAgentImpl> m) { return m->parameters(); })
        .def("named_parameters", [](std::shared_ptr<CognitiveEvolvableAgentImpl> m) { return m->named_parameters(); });
}