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
// v33.0 - Active Inference, Laminar Error Routing, and Continuous Hopfield Memory
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

    CausalParallelSSDImpl(int64_t dim, std::string device_str = "cpu", float min_decay = 0.005f, float max_decay = 0.2f) : dim(dim) {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        auto init_decay = torch::linspace(std::log(min_decay), std::log(max_decay), dim, torch::TensorOptions().device(device));
        log_decay = register_parameter("log_decay", init_decay);
        this->to(device);
    }

    torch::Tensor forward(torch::Tensor x) {
        int64_t batch = x.size(0);
        int64_t seq_len = x.size(1);
        auto device = x.device();

        auto decay = torch::exp(log_decay).view({dim, 1, 1}); // [dim, 1, 1]
        auto indices = torch::arange(seq_len, torch::TensorOptions().device(device).dtype(torch::kFloat32));
        auto delta_pos = (indices.view({seq_len, 1}) - indices.view({1, seq_len})).clamp_min(0.0f);
        auto causal_mask = (indices.view({seq_len, 1}) >= indices.view({1, seq_len})).to(torch::kFloat32);

        auto decay_kernel = torch::exp(-delta_pos.unsqueeze(0) * decay) * causal_mask.unsqueeze(0);
        auto x_perm = x.permute({0, 2, 1}); // [batch, dim, seq_len]
        auto y_perm = torch::matmul(decay_kernel.unsqueeze(0), x_perm.unsqueeze(-1)).squeeze(-1);

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

// 4. ACTIVE INFERENCE LATENT WORLD MODEL
class LatentPredictorImpl : public torch::nn::Module {
public:
    int64_t dim;
    int64_t latent_dim;
    torch::nn::Linear prior_mu{nullptr};
    torch::nn::Linear prior_logvar{nullptr};
    torch::nn::Linear post_mu{nullptr};
    torch::nn::Linear post_logvar{nullptr};

    LatentPredictorImpl(int64_t dim = 256, int64_t latent_dim = 64, std::string device_str = "cpu")
        : dim(dim), latent_dim(latent_dim) {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

        prior_mu = register_module("prior_mu", torch::nn::Linear(dim, latent_dim));
        prior_logvar = register_module("prior_logvar", torch::nn::Linear(dim, latent_dim));
        post_mu = register_module("post_mu", torch::nn::Linear(dim * 2, latent_dim));
        post_logvar = register_module("post_logvar", torch::nn::Linear(dim * 2, latent_dim));

        torch::nn::init::normal_(prior_mu->weight, 0.0, 0.02);
        torch::nn::init::zeros_(prior_mu->bias);
        torch::nn::init::normal_(prior_logvar->weight, 0.0, 0.02);
        torch::nn::init::zeros_(prior_logvar->bias);
        torch::nn::init::normal_(post_mu->weight, 0.0, 0.02);
        torch::nn::init::zeros_(post_mu->bias);
        torch::nn::init::normal_(post_logvar->weight, 0.0, 0.02);
        torch::nn::init::zeros_(post_logvar->bias);

        this->to(device);
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor h_t, torch::Tensor w_t) {
        auto p_mu = prior_mu->forward(h_t);
        auto p_logvar = prior_logvar->forward(h_t).clamp(-10.0, 2.0);

        auto post_in = torch::cat({h_t, w_t}, -1);
        auto q_mu = post_mu->forward(post_in);
        auto q_logvar = post_logvar->forward(post_in).clamp(-10.0, 2.0);

        auto p_var = torch::exp(p_logvar) + 1e-4f;
        auto q_var = torch::exp(q_logvar) + 1e-4f;

        auto kl = 0.5f * (torch::log(p_var / q_var) + (q_var + torch::pow(q_mu - p_mu, 2)) / p_var - 1.0f);
        auto kl_div = kl.sum(-1);

        auto std = torch::sqrt(q_var);
        auto eps = torch::randn_like(std);
        auto z = q_mu + eps * std;

        return std::make_tuple(kl_div, z);
    }
};
TORCH_MODULE(LatentPredictor);

// 5. HOMEOSTATIC NEXUS & ASHBY ULTRASTABILITY
class HomeostaticNexusImpl : public torch::nn::Module {
public:
    std::vector<std::string> dimension_names;
    torch::Tensor values;
    torch::Tensor setpoints;
    torch::Tensor decay_rates;
    torch::Tensor sensitivity;
    std::string device_str;

    HomeostaticNexusImpl(std::string device_str = "cpu") : device_str(device_str) {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

        dimension_names = {"energy", "integrity", "curiosity", "stability", "noradrenaline", "dopamine"};
        int64_t n = dimension_names.size();

        values = register_parameter("values", torch::tensor({1.0f, 1.0f, 0.8f, 0.9f, 0.1f, 0.1f}, torch::TensorOptions().device(device)));
        setpoints = register_buffer("setpoints", torch::tensor({1.0f, 1.0f, 0.5f, 1.0f, 0.0f, 0.0f}, torch::TensorOptions().device(device)));
        decay_rates = register_buffer("decay_rates", torch::tensor({0.001f, 0.0005f, 0.005f, 0.002f, 0.05f, 0.05f}, torch::TensorOptions().device(device)));
        sensitivity = register_buffer("sensitivity", torch::tensor({0.01f, 0.05f, 0.02f, 0.01f, 0.1f, 0.1f}, torch::TensorOptions().device(device)));

        this->to(device);
    }

    void sprout_dimension(std::string name, float init_val, float target_val, float decay, float sens) {
        if (std::find(dimension_names.begin(), dimension_names.end(), name) != dimension_names.end()) {
            return;
        }
        auto device = values.device();
        dimension_names.push_back(name);

        torch::NoGradGuard no_grad;
        values = register_parameter("values", torch::cat({values, torch::tensor({init_val}, torch::TensorOptions().device(device))}));
        setpoints = register_buffer("setpoints", torch::cat({setpoints, torch::tensor({target_val}, torch::TensorOptions().device(device))}));
        decay_rates = register_buffer("decay_rates", torch::cat({decay_rates, torch::tensor({decay}, torch::TensorOptions().device(device))}));
        sensitivity = register_buffer("sensitivity", torch::cat({sensitivity, torch::tensor({sens}, torch::TensorOptions().device(device))}));
    }

    torch::Tensor update(torch::Tensor free_energy, torch::Tensor reward) {
        torch::NoGradGuard no_grad;
        auto fe_scalar = free_energy.mean().item<float>();
        auto rew_scalar = reward.mean().item<float>();

        auto val_acc = values.accessor<float, 1>();
        val_acc[4] = std::clamp(val_acc[4] + 0.1f * fe_scalar - 0.05f, 0.0f, 1.0f);
        val_acc[5] = std::clamp(val_acc[5] + 0.2f * rew_scalar - 0.05f, 0.0f, 1.0f);
        val_acc[0] = std::clamp(val_acc[0] - 0.001f - 0.002f * fe_scalar + 0.005f * rew_scalar, 0.0f, 1.0f);
        val_acc[1] = std::clamp(val_acc[1] - 0.005f * fe_scalar + 0.002f * rew_scalar, 0.0f, 1.0f);
        val_acc[2] = std::clamp(val_acc[2] + 0.01f * fe_scalar - 0.005f, 0.0f, 1.0f);
        val_acc[3] = std::clamp(val_acc[3] - 0.01f * fe_scalar + 0.01f * (1.0f - val_acc[4]), 0.0f, 1.0f);

        return values.clone();
    }

    torch::Tensor get_values() {
        return values.clone();
    }

    std::vector<std::string> get_names() {
        return dimension_names;
    }
};
TORCH_MODULE(HomeostaticNexus);

// 6. CONTINUOUS HOPFIELD ATTRACTOR EPISODIC MEMORY (VECTOR C)
class ContinuousHopfieldMemoryImpl : public torch::nn::Module {
public:
    int64_t dim;
    int64_t num_basins;
    torch::Tensor memory_keys;
    torch::Tensor memory_values;
    torch::nn::Linear mem_gate{nullptr};

    ContinuousHopfieldMemoryImpl(int64_t dim = 256, int64_t num_basins = 32, std::string device_str = "cpu")
        : dim(dim), num_basins(num_basins) {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

        auto k = torch::randn({num_basins, dim}, torch::TensorOptions().device(device)) * (1.0f / std::sqrt(dim));
        auto v = torch::randn({num_basins, dim}, torch::TensorOptions().device(device)) * 0.02f;

        memory_keys = register_parameter("memory_keys", k);
        memory_values = register_parameter("memory_values", v);
        mem_gate = register_module("mem_gate", torch::nn::Linear(dim, 1));
        torch::nn::init::zeros_(mem_gate->weight);
        torch::nn::init::zeros_(mem_gate->bias);

        this->to(device);
    }

    torch::Tensor forward(torch::Tensor x) {
        // x: [batch, seq_len, dim] or [batch, dim]
        int64_t batch = x.size(0);
        int64_t seq_len = (x.dim() == 3) ? x.size(1) : 1;
        auto x_2d = x.reshape({-1, dim}); // [batch * seq_len, dim]

        auto norm_x = x_2d / (torch::norm(x_2d, -1, true) + 1e-6f);
        auto norm_k = memory_keys / (torch::norm(memory_keys, -1, true) + 1e-6f);

        // sim: [batch * seq_len, num_basins]
        auto sim = torch::matmul(norm_x, norm_k.t()) * 8.0f;
        auto attn = torch::softmax(sim, -1);

        // retrieved: [batch * seq_len, dim]
        auto retrieved = torch::matmul(attn, memory_values);
        auto gate = torch::sigmoid(mem_gate->forward(x_2d));
        auto out = gate * retrieved;

        if (x.dim() == 3) {
            return out.view({batch, seq_len, dim});
        }
        return out.view({batch, dim});
    }
};
TORCH_MODULE(ContinuousHopfieldMemory);

// 7. OMNI-MORPHIC NODE WITH LAMINAR ERROR RESIDUAL & HOPFIELD MEMORY
class OmniMorphicNodeImpl : public torch::nn::Module {
public:
    int64_t dim;
    int64_t state_dim;
    int64_t num_operators;
    std::string device_str;

    torch::Tensor affinity_query;
    torch::nn::Linear hyper_w1{nullptr};
    torch::nn::Linear hyper_w2{nullptr};
    torch::nn::LayerNorm state_norm{nullptr};

    CausalParallelSSD causal_ssd{nullptr};
    ParallelOperatorBank operator_bank{nullptr};
    ContinuousHopfieldMemory episodic_memory{nullptr};

    // Vector B: Laminar Prediction Head & Residual Norm
    torch::nn::Linear pred_l1{nullptr};
    torch::nn::Linear pred_l2{nullptr};
    torch::nn::LayerNorm ln_residual{nullptr};

    OmniMorphicNodeImpl(int64_t dim = 256, int64_t state_dim = 128, int64_t num_operators = 8,
                       std::string device_str = "cpu", float min_decay = 0.005f, float max_decay = 0.2f)
        : dim(dim), state_dim(state_dim), num_operators(num_operators), device_str(device_str) {

        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

        affinity_query = register_parameter("affinity_query", torch::randn({dim}, torch::TensorOptions().device(device)) * 0.02f);

        hyper_w1 = register_module("hyper_w1", torch::nn::Linear(dim, num_operators * state_dim));
        hyper_w2 = register_module("hyper_w2", torch::nn::Linear(dim, state_dim * dim));
        state_norm = register_module("state_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));

        torch::nn::init::normal_(hyper_w1->weight, 0.0, 0.02);
        torch::nn::init::zeros_(hyper_w1->bias);
        torch::nn::init::normal_(hyper_w2->weight, 0.0, 0.02);
        torch::nn::init::zeros_(hyper_w2->bias);

        causal_ssd = register_module("causal_ssd", CausalParallelSSD(dim, device_str, min_decay, max_decay));
        operator_bank = register_module("operator_bank", ParallelOperatorBank(dim, state_dim, num_operators));
        episodic_memory = register_module("episodic_memory", ContinuousHopfieldMemory(dim, 32, device_str));

        pred_l1 = register_module("pred_l1", torch::nn::Linear(dim, dim));
        pred_l2 = register_module("pred_l2", torch::nn::Linear(dim, dim));
        ln_residual = register_module("ln_residual", torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));

        torch::nn::init::normal_(pred_l1->weight, 0.0, 0.02);
        torch::nn::init::zeros_(pred_l1->bias);
        torch::nn::init::normal_(pred_l2->weight, 0.0, 0.02);
        torch::nn::init::zeros_(pred_l2->bias);

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

        auto weights_expanded = attn_weights.view({batch, num_signals, 1, 1});
        auto x_attended = (manifold_ref * weights_expanded).sum(1);

        auto context = x_attended.mean(1);
        if (u_t.defined() && u_t.numel() > 0) {
            auto u_flat = (u_t.dim() > 1) ? u_t.view({batch, -1}) : u_t.unsqueeze(0).expand({batch, -1});
            if (u_flat.size(0) == batch && u_flat.size(-1) <= dim) {
                auto u_padded = torch::zeros({batch, dim}, x_attended.options());
                u_padded.slice(1, 0, u_flat.size(-1)).copy_(u_flat);
                context = context + u_padded * 0.1f;
            }
        }

        auto w1_raw = hyper_w1->forward(context).view({batch, num_operators, state_dim});
        auto w1 = torch::softmax(w1_raw, 1); // [batch, num_operators, state_dim]
        auto w2 = hyper_w2->forward(context).view({batch, state_dim, dim}); // [batch, state_dim, dim]

        // Fast temporal scan
        auto h_ssd = causal_ssd->forward(x_attended); // [batch, seq_len, dim]

        // Laminar Prediction & Error Residual Extraction (Vector B)
        auto pred = pred_l2->forward(torch::gelu(pred_l1->forward(h_ssd)));
        auto err_residual = ln_residual->forward(x_attended - pred);
        auto h_effective = h_ssd + err_residual; // [batch, seq_len, dim]

        // Operator bank mixing: ops is [batch, seq_len, dim, num_operators]
        auto ops = operator_bank->compute_operators(h_effective);
        // We project ops (dim, num_operators) with w1 (num_operators, state_dim) -> [batch, seq_len, state_dim]
        // ops: [batch, seq_len, dim, num_operators], w1: [batch, num_operators, state_dim]
        // einsum: b t d k , b k s -> b t s
        auto h_hidden = torch::einsum("btdk,bks->bts", {ops, w1}) * (1.0f / std::sqrt(dim));
        // node_output: [batch, seq_len, state_dim] @ w2: [batch, state_dim, dim] -> [batch, seq_len, dim]
        auto node_output = torch::einsum("bts,bsd->btd", {h_hidden, w2});

        // Episodic Hopfield Memory injection (Vector C)
        auto mem_injection = episodic_memory->forward(node_output);
        node_output = node_output + mem_injection;

        auto normalized_output = state_norm->forward(node_output + x_attended);

        if (!is_4d) {
            normalized_output = normalized_output.squeeze(1);
        }

        return std::make_tuple(normalized_output, attn_weights);
    }
};
TORCH_MODULE(OmniMorphicNode);

// 8. OMNI-CONTINUOUS GRAPH SUBSTRATE WITH EPIGENETIC SPONTANEOUS NEUROGENESIS
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

    bool sprout_node(std::string name, int64_t state_dim = 128, int64_t num_operators = 8, float min_decay = 0.005f, float max_decay = 0.2f) {
        if (nodes.size() >= max_nodes || std::find(node_names.begin(), node_names.end(), name) != node_names.end()) {
            return false;
        }

        auto node = std::make_shared<OmniMorphicNodeImpl>(dim, state_dim, num_operators, device_str, min_decay, max_decay);
        std::string node_key = "node_" + std::to_string(nodes.size());
        register_module(node_key, node);
        nodes.push_back(node);

        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        auto alpha = register_parameter(node_key + "_alpha", torch::zeros({}, torch::TensorOptions().device(device)));
        alpha_nodes.push_back(alpha);
        node_names.push_back(name);

        return true;
    }

    bool trigger_spontaneous_neurogenesis(float free_energy_surprise, float threshold = 1.2f) {
        if (free_energy_surprise > threshold && nodes.size() < max_nodes) {
            std::string generated_name = "spontaneous_node_" + std::to_string(nodes.size());
            return sprout_node(generated_name, 128, 8);
        }
        return false;
    }

    int64_t execute_neural_darwinism(float decay_rate = 0.001f, float prune_threshold = 0.005f) {
        int64_t pruned_count = 0;
        torch::NoGradGuard no_grad;
        for (size_t i = 0; i < alpha_nodes.size(); ++i) {
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

// 9. THE COMPLETE EVOLVABLE ACTIVE INFERENCE COGNITIVE AGENT
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

    bool sprout_organelle(std::string name, int64_t state_dim = 128, int64_t num_operators = 8, float min_decay = 0.005f, float max_decay = 0.2f) {
        return substrate->sprout_node(name, state_dim, num_operators, min_decay, max_decay);
    }

    void sprout_homeostatic_dimension(std::string name, float init_val, float target_val, float decay, float sensitivity) {
        homeostasis->sprout_dimension(name, init_val, target_val, decay, sensitivity);
    }

    torch::Tensor forward(torch::Tensor tokens, torch::Tensor u_t) {
        auto emb = manifold->forward(tokens);
        
        std::vector<torch::Tensor> signals = {emb};
        torch::Tensor final_emb;
        std::vector<torch::Tensor> updated_signals;
        std::tie(final_emb, updated_signals) = substrate->forward(signals, u_t);

        return motor_head->forward(final_emb);
    }

    torch::Tensor forward(torch::Tensor tokens) {
        auto u_t = homeostasis->get_values();
        return forward(tokens, u_t);
    }

    torch::Tensor forward_latent(torch::Tensor tokens, torch::Tensor u_t) {
        auto emb = manifold->forward(tokens);
        
        std::vector<torch::Tensor> signals = {emb};
        torch::Tensor final_emb;
        std::vector<torch::Tensor> updated_signals;
        std::tie(final_emb, updated_signals) = substrate->forward(signals, u_t);

        return final_emb;
    }

    torch::Tensor forward_latent(torch::Tensor tokens) {
        auto u_t = homeostasis->get_values();
        return forward_latent(tokens, u_t);
    }

    torch::Tensor forward_motor(torch::Tensor latent) {
        return motor_head->forward(latent);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward_active_inference(torch::Tensor tokens, torch::Tensor reward) {
        auto u_t = homeostasis->get_values();
        auto emb = manifold->forward(tokens);
        
        std::vector<torch::Tensor> signals = {emb};
        torch::Tensor final_emb;
        std::vector<torch::Tensor> updated_signals;
        std::tie(final_emb, updated_signals) = substrate->forward(signals, u_t);

        torch::Tensor free_energy, z;
        std::tie(free_energy, z) = latent_predictor->forward(final_emb, emb);

        auto updated_u_t = homeostasis->update(free_energy, reward);

        float mean_fe = free_energy.mean().item<float>();
        substrate->trigger_spontaneous_neurogenesis(mean_fe, 1.2f);
        substrate->execute_neural_darwinism(0.001f, 0.005f);

        auto logits = motor_head->forward(final_emb);
        return std::make_tuple(logits, free_energy, updated_u_t);
    }

    torch::Tensor generate_thought_and_speech(torch::Tensor seed_tokens, int64_t max_new_tokens = 32, float temperature = 0.45f, float top_p = 0.90f) {
        torch::NoGradGuard no_grad;
        auto current_seq = seed_tokens.clone();

        for (int64_t step = 0; step < max_new_tokens; ++step) {
            auto logits = forward(current_seq);
            auto next_token_logits = logits.select(1, -1);

            if (temperature > 0.0f) {
                next_token_logits = next_token_logits / temperature;
                auto probs = torch::softmax(next_token_logits, -1);

                auto sorted_probs_indices = torch::sort(probs, -1, true);
                auto sorted_probs = std::get<0>(sorted_probs_indices);
                auto sorted_indices = std::get<1>(sorted_probs_indices);

                auto cumulative_probs = torch::cumsum(sorted_probs, -1);
                auto sorted_indices_to_remove = cumulative_probs > top_p;
                sorted_indices_to_remove.slice(-1, 1).copy_(sorted_indices_to_remove.slice(-1, 0, -1).clone());
                sorted_indices_to_remove.slice(-1, 0, 1).fill_(false);

                sorted_probs.masked_fill_(sorted_indices_to_remove, 0.0f);
                sorted_probs = sorted_probs / sorted_probs.sum(-1, true);

                auto next_token_idx = torch::multinomial(sorted_probs, 1);
                auto next_token = sorted_indices.gather(-1, next_token_idx);

                current_seq = torch::cat({current_seq, next_token}, 1);
            } else {
                auto next_token = next_token_logits.argmax(-1, true);
                current_seq = torch::cat({current_seq, next_token}, 1);
            }
        }
        return current_seq;
    }
};
TORCH_MODULE(CognitiveEvolvableAgent);

// ============================================================================
// PYBIND11 MODULE BINDINGS
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

    py::class_<ContinuousHopfieldMemoryImpl, torch::nn::Module, std::shared_ptr<ContinuousHopfieldMemoryImpl>>(m, "ContinuousHopfieldMemory")
        .def(py::init<int64_t, int64_t, std::string>(), py::arg("dim") = 256, py::arg("num_basins") = 32, py::arg("device") = "cpu")
        .def("forward", &ContinuousHopfieldMemoryImpl::forward)
        .def("__call__", &ContinuousHopfieldMemoryImpl::forward);

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
        .def(py::init<int64_t, int64_t, int64_t, std::string, float, float>(),
             py::arg("dim") = 256, py::arg("state_dim") = 128, py::arg("num_operators") = 8, py::arg("device") = "cpu",
             py::arg("min_decay") = 0.005f, py::arg("max_decay") = 0.2f)
        .def("forward", &OmniMorphicNodeImpl::forward, py::arg("signal_manifold"), py::arg("u_t") = torch::Tensor())
        .def("__call__", &OmniMorphicNodeImpl::forward, py::arg("signal_manifold"), py::arg("u_t") = torch::Tensor());

    py::class_<OmniContinuousGraphSubstrateImpl, torch::nn::Module, std::shared_ptr<OmniContinuousGraphSubstrateImpl>>(m, "OmniContinuousGraphSubstrate")
        .def(py::init<int64_t, int64_t, std::string>(), py::arg("dim") = 256, py::arg("max_nodes") = 16, py::arg("device") = "cpu")
        .def_readonly("node_names", &OmniContinuousGraphSubstrateImpl::node_names)
        .def("sprout_node", &OmniContinuousGraphSubstrateImpl::sprout_node, py::arg("name"), py::arg("state_dim") = 128, py::arg("num_operators") = 8, py::arg("min_decay") = 0.005f, py::arg("max_decay") = 0.2f)
        .def("trigger_spontaneous_neurogenesis", &OmniContinuousGraphSubstrateImpl::trigger_spontaneous_neurogenesis)
        .def("execute_neural_darwinism", &OmniContinuousGraphSubstrateImpl::execute_neural_darwinism)
        .def("forward", &OmniContinuousGraphSubstrateImpl::forward, py::arg("signal_list"), py::arg("u_t") = torch::Tensor())
        .def("__call__", &OmniContinuousGraphSubstrateImpl::forward, py::arg("signal_list"), py::arg("u_t") = torch::Tensor());

    py::class_<CognitiveEvolvableAgentImpl, torch::nn::Module, std::shared_ptr<CognitiveEvolvableAgentImpl>>(m, "CognitiveEvolvableAgent")
        .def(py::init<int64_t, int64_t, int64_t, std::string>(), py::arg("vocab_size") = 258, py::arg("dim") = 256, py::arg("max_nodes") = 16, py::arg("device") = "cpu")
        .def_property_readonly("homeostasis", [](std::shared_ptr<CognitiveEvolvableAgentImpl> a) { return a->homeostasis.ptr(); })
        .def_property_readonly("substrate", [](std::shared_ptr<CognitiveEvolvableAgentImpl> a) { return a->substrate.ptr(); })
        .def("sprout_organelle", &CognitiveEvolvableAgentImpl::sprout_organelle, py::arg("name"), py::arg("state_dim") = 128, py::arg("num_operators") = 8, py::arg("min_decay") = 0.005f, py::arg("max_decay") = 0.2f)
        .def("sprout_homeostatic_dimension", &CognitiveEvolvableAgentImpl::sprout_homeostatic_dimension, py::arg("name"), py::arg("init_val"), py::arg("target_val"), py::arg("decay"), py::arg("sensitivity"))
        .def("forward", py::overload_cast<torch::Tensor, torch::Tensor>(&CognitiveEvolvableAgentImpl::forward), py::arg("tokens"), py::arg("u_t"))
        .def("forward", py::overload_cast<torch::Tensor>(&CognitiveEvolvableAgentImpl::forward), py::arg("tokens"))
        .def("__call__", py::overload_cast<torch::Tensor, torch::Tensor>(&CognitiveEvolvableAgentImpl::forward), py::arg("tokens"), py::arg("u_t"))
        .def("__call__", py::overload_cast<torch::Tensor>(&CognitiveEvolvableAgentImpl::forward), py::arg("tokens"))
        .def("forward_latent", py::overload_cast<torch::Tensor, torch::Tensor>(&CognitiveEvolvableAgentImpl::forward_latent), py::arg("tokens"), py::arg("u_t"))
        .def("forward_latent", py::overload_cast<torch::Tensor>(&CognitiveEvolvableAgentImpl::forward_latent), py::arg("tokens"))
        .def("forward_motor", &CognitiveEvolvableAgentImpl::forward_motor, py::arg("latent"))
        .def("forward_active_inference", &CognitiveEvolvableAgentImpl::forward_active_inference, py::arg("tokens"), py::arg("reward"))
        .def("generate_thought_and_speech", &CognitiveEvolvableAgentImpl::generate_thought_and_speech, py::arg("seed_tokens"), py::arg("max_new_tokens"), py::arg("temperature") = 0.45f, py::arg("top_p") = 0.90f)
        .def("parameters", [](std::shared_ptr<CognitiveEvolvableAgentImpl> m) { return m->parameters(); })
        .def("named_parameters", [](std::shared_ptr<CognitiveEvolvableAgentImpl> m) { return m->named_parameters(); });
}
