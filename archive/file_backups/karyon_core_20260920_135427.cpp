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

        auto alpha = torch::sigmoid(log_decay).view({1, 1, dim});
        auto alpha_cum = torch::cumprod(alpha.expand({batch, seq_len, dim}), 1);

        auto h = torch::zeros({batch, seq_len, dim}, x.options());
        auto current_state = torch::zeros({batch, dim}, x.options());

        for (int64_t t = 0; t < seq_len; ++t) {
            auto xt = x.select(1, t);
            auto decay = alpha.select(1, 0);
            current_state = decay * current_state + (1.0f - decay) * xt;
            h.select(1, t).copy_(current_state);
        }
        return h;
    }
};
TORCH_MODULE(CausalParallelSSD);

// 3. PARALLEL OPERATOR BANK
class ParallelOperatorBankImpl : public torch::nn::Module {
public:
    int64_t dim;
    int64_t state_dim;
    int64_t num_operators;

    torch::Tensor op_weights;

    ParallelOperatorBankImpl(int64_t dim = 256, int64_t state_dim = 128, int64_t num_operators = 8)
        : dim(dim), state_dim(state_dim), num_operators(num_operators) {
        op_weights = register_parameter("op_weights", torch::randn({num_operators, dim, dim}) * 0.02f);
    }

    torch::Tensor compute_operators(torch::Tensor h) {
        int64_t batch = h.size(0);
        int64_t seq_len = h.size(1);
        int64_t d = h.size(2);

        auto h_flat = h.reshape({batch * seq_len, d});
        std::vector<torch::Tensor> op_outputs;

        for (int64_t i = 0; i < num_operators; ++i) {
            auto w = op_weights[i];
            auto out = torch::matmul(h_flat, w);
            op_outputs.push_back(out.view({batch, seq_len, d, 1}));
        }
        return torch::cat(op_outputs, -1); // [batch, seq_len, dim, num_operators]
    }
};
TORCH_MODULE(ParallelOperatorBank);

// 4. ACTIVE INFERENCE LATENT WORLD MODEL
class LatentPredictorImpl : public torch::nn::Module {
public:
    int64_t dim;
    int64_t latent_dim;

    torch::nn::Linear mu_prior{nullptr};
    torch::nn::Linear logvar_prior{nullptr};
    torch::nn::Linear mu_posterior{nullptr};
    torch::nn::Linear logvar_posterior{nullptr};
    torch::nn::Linear decoder{nullptr};

    LatentPredictorImpl(int64_t dim = 256, int64_t latent_dim = 64, std::string device_str = "cpu")
        : dim(dim), latent_dim(latent_dim) {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

        mu_prior = register_module("mu_prior", torch::nn::Linear(dim, latent_dim));
        logvar_prior = register_module("logvar_prior", torch::nn::Linear(dim, latent_dim));
        mu_posterior = register_module("mu_posterior", torch::nn::Linear(dim, latent_dim));
        logvar_posterior = register_module("logvar_posterior", torch::nn::Linear(dim, latent_dim));
        decoder = register_module("decoder", torch::nn::Linear(latent_dim, dim));

        torch::nn::init::normal_(mu_prior->weight, 0.0, 0.02);
        torch::nn::init::zeros_(mu_prior->bias);
        torch::nn::init::normal_(logvar_prior->weight, 0.0, 0.02);
        torch::nn::init::zeros_(logvar_prior->bias);

        torch::nn::init::normal_(mu_posterior->weight, 0.0, 0.02);
        torch::nn::init::zeros_(mu_posterior->bias);
        torch::nn::init::normal_(logvar_posterior->weight, 0.0, 0.02);
        torch::nn::init::zeros_(logvar_posterior->bias);

        torch::nn::init::normal_(decoder->weight, 0.0, 0.02);
        torch::nn::init::zeros_(decoder->bias);

        this->to(device);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor h, torch::Tensor target) {
        auto mu_p = mu_prior->forward(h);
        auto logvar_p = logvar_prior->forward(h);

        auto mu_q = mu_posterior->forward(target);
        auto logvar_q = logvar_posterior->forward(target);

        auto std = torch::exp(0.5f * logvar_q);
        auto eps = torch::randn_like(std);
        auto z = mu_q + eps * std;

        auto recon = decoder->forward(z);
        return std::make_tuple(recon, mu_p, logvar_p, mu_q, logvar_q);
    }
};
TORCH_MODULE(LatentPredictor);

// 5. HOMEOSTATIC NEXUS & ASHBY ULTRASTABILITY
class HomeostaticNexusImpl : public torch::nn::Module {
public:
    std::vector<std::string> dimension_names;
    torch::Tensor current_states;
    torch::Tensor target_states;
    torch::Tensor decay_rates;
    torch::Tensor sensitivities;

    HomeostaticNexusImpl(std::string device_str = "cpu") {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

        dimension_names = {"Curiosity", "Energy", "Stability", "Health", "Noradrenaline", "Dopamine"};
        current_states = register_buffer("current_states", torch::tensor({0.5f, 1.0f, 0.8f, 1.0f, 0.2f, 0.1f}, torch::TensorOptions().device(device)));
        target_states = register_buffer("target_states", torch::tensor({0.5f, 1.0f, 1.0f, 1.0f, 0.1f, 0.1f}, torch::TensorOptions().device(device)));
        decay_rates = register_buffer("decay_rates", torch::tensor({0.01f, 0.005f, 0.002f, 0.001f, 0.05f, 0.05f}, torch::TensorOptions().device(device)));
        sensitivities = register_buffer("sensitivities", torch::tensor({0.1f, 0.2f, 0.15f, 0.1f, 0.3f, 0.4f}, torch::TensorOptions().device(device)));

        this->to(device);
    }

    void sprout_dimension(std::string name, float init_val, float target_val, float decay, float sensitivity) {
        dimension_names.push_back(name);
        auto device = current_states.device();

        current_states = torch::cat({current_states, torch::tensor({init_val}, torch::TensorOptions().device(device))});
        target_states = torch::cat({target_states, torch::tensor({target_val}, torch::TensorOptions().device(device))});
        decay_rates = torch::cat({decay_rates, torch::tensor({decay}, torch::TensorOptions().device(device))});
        sensitivities = torch::cat({sensitivities, torch::tensor({sensitivity}, torch::TensorOptions().device(device))});
    }

    torch::Tensor get_states() {
        return current_states;
    }

    std::vector<std::string> get_names() {
        return dimension_names;
    }

    torch::Tensor update(torch::Tensor prediction_error) {
        torch::NoGradGuard no_grad;
        auto err = prediction_error.mean().item<float>();

        // Ashby Ultrastability update loop
        auto diff = target_states - current_states;
        current_states.add_(diff * decay_rates);

        // Noradrenaline spike on high surprise / prediction error
        int64_t na_idx = -1;
        int64_t da_idx = -1;
        for (size_t i = 0; i < dimension_names.size(); ++i) {
            if (dimension_names[i] == "Noradrenaline") na_idx = i;
            if (dimension_names[i] == "Dopamine") da_idx = i;
        }

        if (na_idx != -1) {
            current_states[na_idx].add_(err * sensitivities[na_idx]);
            current_states[na_idx] = torch::clamp(current_states[na_idx], 0.0f, 1.0f);
        }

        if (da_idx != -1) {
            // Dopamine spikes on reward (inverse prediction error)
            float reward = std::exp(-err * 2.0f);
            current_states[da_idx].add_(reward * sensitivities[da_idx]);
            current_states[da_idx] = torch::clamp(current_states[da_idx], 0.0f, 1.0f);
        }

        return current_states;
    }
};
TORCH_MODULE(HomeostaticNexus);

// 6. CONTINUOUS HOPFIELD ATTRACTOR EPISODIC MEMORY WITH DOPAMINERGIC MODULATION (VECTOR C)
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

    torch::Tensor forward(torch::Tensor x, torch::Tensor u_t = torch::Tensor()) {
        // x: [batch, seq_len, dim] or [batch, dim]
        int64_t batch = x.size(0);
        int64_t seq_len = (x.dim() == 3) ? x.size(1) : 1;
        int64_t d = x.size(-1);
        auto x_2d = x.reshape({-1, d}); // [batch * seq_len, d]

        auto norm_x = x_2d / (torch::sqrt(torch::sum(x_2d * x_2d, -1, true)) + 1e-6f);
        auto norm_k = memory_keys / (torch::sqrt(torch::sum(memory_keys * memory_keys, -1, true)) + 1e-6f);

        // Dopaminergic precision scaling: higher dopamine -> sharper attractor basins (higher beta)
        float beta = 8.0f;
        if (u_t.defined() && u_t.numel() > 0) {
            // Dopamine is index 5 in homeostasis dimension states
            auto da_val = u_t.reshape({batch, -1}).select(-1, 5).mean().item<float>();
            beta = beta * (1.0f + 1.5f * da_val);
        }

        // sim: [batch * seq_len, num_basins]
        auto sim = torch::matmul(norm_x, norm_k.t()) * beta;
        auto attn = torch::softmax(sim, -1);

        // retrieved: [batch * seq_len, dim]
        auto retrieved = torch::matmul(attn, memory_values);
        auto gate = torch::sigmoid(mem_gate->forward(x_2d));
        auto out = gate * retrieved;

        if (x.dim() == 3) {
            return out.view({batch, seq_len, d});
        }
        return out.view({batch, d});
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

    torch::nn::Linear pred_l1{nullptr};
    torch::nn::Linear pred_l2{nullptr};
    torch::nn::LayerNorm ln_residual{nullptr};

    OmniMorphicNodeImpl(int64_t dim = 256, int64_t state_dim = 128, int64_t num_operators = 8, std::string device_str = "cpu",
                        float min_decay = 0.005f, float max_decay = 0.2f)
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
        auto h_hidden = torch::einsum("btdk,bks->bts", {ops, w1}) * (1.0f / std::sqrt(dim));
        auto node_output = torch::einsum("bts,bsd->btd", {h_hidden, w2});

        // Episodic Hopfield Memory injection (Vector C) with Dopaminergic Modulation
        auto mem_injection = episodic_memory->forward(node_output, u_t);
        node_output = node_output + mem_injection;

        auto normalized_output = state_norm->forward(node_output + x_attended);

        if (!is_4d) {
            normalized_output = normalized_output.squeeze(1);
        }

        return std::make_tuple(normalized_output, attn_weights);
    }
};
TORCH_MODULE(OmniMorphicNode);

// 8. OMNI-CONTINUOUS GRAPH SUBSTRATE WITH EPIGENETIC SPONTANEOUS NEUROGENESIS, RECIRCULATION & ROUTING
class OmniContinuousGraphSubstrateImpl : public torch::nn::Module {
public:
    int64_t dim;
    int64_t max_nodes;
    std::string device_str;

    std::vector<std::shared_ptr<OmniMorphicNodeImpl>> nodes;
    std::vector<torch::Tensor> alpha_nodes;
    std::vector<std::string> node_names;

    // Continuous Dynamic Routing & Recirculation Parameters (Principle 19)
    torch::Tensor routing_weights; // [max_nodes, max_nodes]
    torch::nn::Linear recirc_gate{nullptr};

    OmniContinuousGraphSubstrateImpl(int64_t dim, int64_t max_nodes = 16, std::string device_str = "cpu")
        : dim(dim), max_nodes(max_nodes), device_str(device_str) {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

        // Initialize fully connected dynamic routing matrix
        routing_weights = register_parameter("routing_weights", torch::randn({max_nodes, max_nodes}, torch::TensorOptions().device(device)) * 0.05f);

        // Recirculation gate coupled to interoceptive neurotransmitter states (u_t)
        recirc_gate = register_module("recirc_gate", torch::nn::Linear(6, 1));
        torch::nn::init::zeros_(recirc_gate->weight);
        torch::nn::init::zeros_(recirc_gate->bias);

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
        int64_t batch = sample.size(0);
        int64_t seq_len = is_3d ? sample.size(1) : 1;
        auto device = sample.device();

        std::vector<torch::Tensor> norm_signals;
        for (const auto& s : signal_list) {
            if (s.dim() == 2) {
                norm_signals.push_back(s.unsqueeze(1).expand({-1, seq_len, -1}));
            } else {
                norm_signals.push_back(s);
            }
        }

        std::vector<torch::Tensor> current_signals = norm_signals;

        // 1. Dynamic Recirculation Gate (System 2 Mental Sandbox)
        float recirc_gamma = 0.0f;
        if (u_t.defined() && u_t.numel() > 0) {
            auto u_flat = u_t.reshape({batch, -1}).slice(1, 0, 6);
            recirc_gamma = torch::sigmoid(recirc_gate->forward(u_flat)).mean().template item<float>();
        }

        // We run up to 3 recirculation steps if recirc_gamma > 0.15
        int64_t max_recirc_steps = (recirc_gamma > 0.15f) ? 3 : 1;
        auto prev_recirc_state = torch::zeros_like(current_signals[0]);

        for (int64_t step = 0; step < max_recirc_steps; ++step) {
            std::vector<torch::Tensor> step_signals = current_signals;
            
            // Mix previous step state if recirculating
            if (step > 0) {
                step_signals[0] = (1.0f - recirc_gamma) * step_signals[0] + recirc_gamma * prev_recirc_state;
            }

            for (size_t i = 0; i < nodes.size(); ++i) {
                // 2. Continuous Dynamic Routing: Node i attends to previous signals based on learned routing weights
                auto route_softmax = torch::softmax(routing_weights[i].slice(0, 0, step_signals.size()), -1);
                
                // Stack active signals to route
                auto stacked_signals = torch::stack(step_signals, 1); // [batch, num_signals, seq_len, dim]
                
                // Einsum routes signals: r: [num_signals], s: [batch, num_signals, seq, dim] -> [batch, seq, dim]
                auto routed_input = torch::einsum("r, brsd -> bsd", {route_softmax, stacked_signals});

                torch::Tensor node_out, attn_w;
                // Node expects a stack representation of shape [batch, num_signals, seq, dim]
                std::tie(node_out, attn_w) = nodes[i]->forward(routed_input.unsqueeze(1), u_t);

                auto alpha = alpha_nodes[i];
                auto gate = torch::tanh(alpha);
                auto gated_signal = gate * node_out;
                step_signals.push_back(gated_signal);
            }
            
            // Collect the sum of sprouted nodes as step state
            auto step_sum = torch::zeros_like(step_signals[0]);
            for (size_t i = norm_signals.size(); i < step_signals.size(); ++i) {
                step_sum = step_sum + step_signals[i];
            }
            prev_recirc_state = step_sum;
            current_signals = step_signals;
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

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> step(torch::Tensor tokens, torch::Tensor target_tokens, torch::Tensor u_t) {
        auto emb = manifold->forward(tokens);
        auto target_emb = manifold->forward(target_tokens);

        std::vector<torch::Tensor> signals = {emb};
        torch::Tensor final_emb;
        std::vector<torch::Tensor> updated_signals;

        std::tie(final_emb, updated_signals) = substrate->forward(signals, u_t);

        // Active Inference Predictor step
        torch::Tensor recon, mu_p, logvar_p, mu_q, logvar_q;
        std::tie(recon, mu_p, logvar_p, mu_q, logvar_q) = latent_predictor->forward(final_emb, target_emb);

        // Compute Variational Free Energy F_t
        auto recon_loss = torch::mse_loss(recon, target_emb);
        auto kl_loss = -0.5f * torch::sum(1.0f + logvar_q - logvar_p - (logvar_q.exp() + (mu_q - mu_p).pow(2)) / logvar_p.exp());
        auto free_energy = recon_loss + 0.01f * kl_loss;

        // Update somatic neurotransmitters based on Free Energy surprise
        auto updated_u_t = homeostasis->update(free_energy);

        auto logits = motor_head->forward(final_emb);

        float mean_fe = free_energy.mean().item<float>();
        substrate->trigger_spontaneous_neurogenesis(mean_fe, 1.2f);
        substrate->execute_neural_darwinism(0.001f, 0.005f);

        return std::make_tuple(logits, free_energy, updated_u_t);
    }

    torch::Tensor generate_thought_and_speech(torch::Tensor seed_tokens, int64_t max_new_tokens = 32, float temperature = 0.45f, float top_p = 0.90f) {
        torch::NoGradGuard no_grad;
        auto current_seq = seed_tokens.clone();

        for (int64_t step = 0; step < max_new_tokens; ++step) {
            auto logits = forward(current_seq, torch::Tensor());
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

// 10. UNIVERSAL MORPHIC CIRCUIT CELL (NATIVE C++20 - MCC v2.0)
class UniversalMorphicCellImpl : public torch::nn::Module {
public:
    int64_t dim;
    int64_t num_units;
    std::string device_str;

    CausalParallelSSD ssd{nullptr};

    torch::Tensor log_alpha;
    torch::nn::Linear w_atom{nullptr};
    torch::nn::Linear gate_atom{nullptr};

    torch::Tensor routing_matrix;
    torch::Tensor w_units;
    torch::Tensor formula_gate;
    torch::Tensor unit_alphas;

    torch::nn::Linear tunnel_proj{nullptr};
    torch::nn::LayerNorm norm{nullptr};
    torch::Tensor alpha_epi;

    UniversalMorphicCellImpl(int64_t dim = 256, int64_t num_units = 4, std::string device_str = "cpu",
                             float min_decay = 0.005f, float max_decay = 0.2f)
        : dim(dim), num_units(num_units), device_str(device_str) {

        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

        // C++20 Causal SSD for temporal context
        ssd = register_module("ssd", CausalParallelSSD(dim, device_str, min_decay, max_decay));

        // Fast Micro-Operator Core (Gamma Flow)
        log_alpha = register_parameter("log_alpha", torch::randn({dim}, torch::TensorOptions().device(device)) * 0.1f - 2.0f);

        auto lin_opts = torch::nn::LinearOptions(dim, dim).bias(false);
        w_atom = register_module("w_atom", torch::nn::Linear(lin_opts));
        gate_atom = register_module("gate_atom", torch::nn::Linear(lin_opts));

        torch::nn::init::orthogonal_(w_atom->weight, 0.2);
        torch::nn::init::orthogonal_(gate_atom->weight, 0.2);

        // Dynamic Circuit Builder & Signal Transporter (Vector D)
        routing_matrix = register_parameter("routing_matrix", torch::randn({num_units, num_units}, torch::TensorOptions().device(device)) * 0.05f);
        w_units = register_parameter("w_units", torch::randn({num_units, dim, dim}, torch::TensorOptions().device(device)) * (0.2f / std::sqrt(static_cast<float>(dim))));
        formula_gate = register_parameter("formula_gate", torch::randn({num_units, 1, 1, dim}, torch::TensorOptions().device(device)) * 0.01f);
        unit_alphas = register_parameter("unit_alphas", torch::ones({num_units, 1, 1, 1}, torch::TensorOptions().device(device)));

        tunnel_proj = register_module("tunnel_proj", torch::nn::Linear(lin_opts));
        torch::nn::init::orthogonal_(tunnel_proj->weight, 0.2);

        norm = register_module("norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));
        alpha_epi = register_parameter("alpha_epi", torch::ones({1}, torch::TensorOptions().device(device)));

        this->to(device);
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x, torch::Tensor tunnel_in = torch::Tensor()) {
        int64_t B = x.size(0);
        int64_t S = x.size(1);
        int64_t D = x.size(2);
        int64_t U = num_units;

        // 1. Temporal context extraction via causal parallel C++ SSD
        auto x_ssd = ssd->forward(x);

        torch::Tensor x_in;
        if (tunnel_in.defined() && tunnel_in.numel() > 0) {
            x_in = x_ssd + tunnel_in;
        } else {
            x_in = x_ssd;
        }

        // 2. Local Micro-Operator (Gamma Flow)
        auto x_proj = torch::silu(w_atom->forward(x_in));
        auto alpha = torch::sigmoid(log_alpha).view({1, 1, -1});
        auto integrated = x_proj * (1.0f - alpha);
        auto gated = integrated * torch::sigmoid(gate_atom->forward(x_in));
        auto x_local = x_in + gated;

        // 3. Dynamic Circuit Builder & Signal Transporter (Vector D)
        auto route_weights = torch::softmax(routing_matrix, -1);
        auto bus = x_local.unsqueeze(0).expand({U, -1, -1, -1});
        auto routed = torch::einsum("uv, vbsd -> ubsd", {route_weights, bus});

        auto routed_flat = routed.reshape({U, B * S, D});
        auto lin_flat = torch::bmm(routed_flat, w_units);
        auto lin_out = lin_flat.reshape({U, B, S, D});

        auto gate = torch::sigmoid(routed * formula_gate);
        auto formula_out = torch::silu(lin_out * gate);
        auto normed = torch::layer_norm(formula_out, {D});

        auto circuit_out = (routed + torch::tanh(unit_alphas) * normed).mean(0);

        // 4. Epigenetic zero-shock output
        auto cell_out = x + torch::tanh(alpha_epi) * norm->forward(circuit_out);
        auto tunnel_out = tunnel_proj->forward(cell_out);

        return std::make_tuple(cell_out, tunnel_out);
    }
};
TORCH_MODULE(UniversalMorphicCell);

// 11. UNIVERSAL MORPHIC CIRCUIT SPACE (NATIVE C++20 - MCS v2.0)
class UniversalMorphicSpaceImpl : public torch::nn::Module {
public:
    int64_t vocab_size;
    int64_t dim;
    int64_t num_cells;
    int64_t num_units;
    std::string device_str;

    torch::nn::Embedding emb{nullptr};
    std::vector<UniversalMorphicCell> cells;
    torch::Tensor cross_cell_routing;
    torch::nn::LayerNorm norm{nullptr};
    torch::nn::Linear head{nullptr};

    UniversalMorphicSpaceImpl(int64_t vocab_size = 258, int64_t dim = 256, int64_t num_cells = 2, int64_t num_units = 4,
                              std::string device_str = "cpu", float min_decay = 0.005f, float max_decay = 0.2f)
        : vocab_size(vocab_size), dim(dim), num_cells(num_cells), num_units(num_units), device_str(device_str) {

        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

        emb = register_module("emb", torch::nn::Embedding(vocab_size, dim));
        torch::nn::init::normal_(emb->weight, 0.0, 0.02);

        for (int64_t i = 0; i < num_cells; ++i) {
            auto cell = UniversalMorphicCell(dim, num_units, device_str, min_decay, max_decay);
            register_module("cell_" + std::to_string(i), cell);
            cells.push_back(cell);
        }

        cross_cell_routing = register_parameter("cross_cell_routing", torch::randn({num_cells, num_cells}, torch::TensorOptions().device(device)) * 0.05f);

        norm = register_module("norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));

        auto head_opts = torch::nn::LinearOptions(dim, vocab_size).bias(false);
        head = register_module("head", torch::nn::Linear(head_opts));
        torch::nn::init::normal_(head->weight, 0.0, 0.02);

        this->to(device);
    }

    torch::Tensor forward_latent(torch::Tensor tokens) {
        auto x = emb->forward(tokens);
        int64_t C = num_cells;

        auto cross_weights = torch::softmax(cross_cell_routing, -1);
        std::vector<torch::Tensor> cell_states(C, x);
        std::vector<torch::Tensor> cell_tunnels(C, torch::Tensor());

        for (int64_t i = 0; i < C; ++i) {
            bool has_tunnel = false;
            for (int64_t k = 0; k < C; ++k) {
                if (cell_tunnels[k].defined() && cell_tunnels[k].numel() > 0) {
                    has_tunnel = true;
                    break;
                }
            }

            torch::Tensor tunnel_sig;
            if (has_tunnel) {
                std::vector<torch::Tensor> tunnel_list;
                for (int64_t k = 0; k < C; ++k) {
                    if (cell_tunnels[k].defined() && cell_tunnels[k].numel() > 0) {
                        tunnel_list.push_back(cell_tunnels[k]);
                    } else {
                        tunnel_list.push_back(torch::zeros_like(x));
                    }
                }
                auto stacked = torch::stack(tunnel_list, 0); // [C, B, S, D]
                auto w_i = cross_weights[i]; // [C]
                tunnel_sig = torch::einsum("c, cbsd -> bsd", {w_i, stacked});
            }

            torch::Tensor c_out, t_out;
            std::tie(c_out, t_out) = cells[i]->forward(cell_states[i], tunnel_sig);
            cell_states[i] = c_out;
            cell_tunnels[i] = t_out;
        }

        auto stacked_states = torch::stack(cell_states, 0);
        auto final_state = norm->forward(stacked_states.mean(0));
        return final_state;
    }

    torch::Tensor forward(torch::Tensor tokens) {
        auto final_state = forward_latent(tokens);
        return head->forward(final_state);
    }
};
TORCH_MODULE(UniversalMorphicSpace);


// ============================================================================
// PYBIND11 MODULE BINDINGS
// ============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<UniversalManifoldImpl, torch::nn::Module, std::shared_ptr<UniversalManifoldImpl>>(m, "UniversalManifold")
        .def(py::init<int64_t, int64_t, std::string>(), py::arg("vocab_size") = 258, py::arg("dim") = 256, py::arg("device") = "cpu")
        .def("forward", &UniversalManifoldImpl::forward)
        .def("__call__", &UniversalManifoldImpl::forward);

    py::class_<CausalParallelSSDImpl, torch::nn::Module, std::shared_ptr<CausalParallelSSDImpl>>(m, "CausalParallelSSD")
        .def(py::init<int64_t, std::string, float, float>(), py::arg("dim") = 256, py::arg("device") = "cpu", py::arg("min_decay") = 0.005f, py::arg("max_decay") = 0.2f)
        .def("forward", &CausalParallelSSDImpl::forward)
        .def("__call__", &CausalParallelSSDImpl::forward);

    py::class_<ParallelOperatorBankImpl, torch::nn::Module, std::shared_ptr<ParallelOperatorBankImpl>>(m, "ParallelOperatorBank")
        .def(py::init<int64_t, int64_t, int64_t>(), py::arg("dim") = 256, py::arg("state_dim") = 128, py::arg("num_operators") = 8)
        .def("compute_operators", &ParallelOperatorBankImpl::compute_operators);

    py::class_<ContinuousHopfieldMemoryImpl, torch::nn::Module, std::shared_ptr<ContinuousHopfieldMemoryImpl>>(m, "ContinuousHopfieldMemory")
        .def(py::init<int64_t, int64_t, std::string>(), py::arg("dim") = 256, py::arg("num_basins") = 32, py::arg("device") = "cpu")
        .def("forward", [](ContinuousHopfieldMemoryImpl& self, torch::Tensor x, std::optional<torch::Tensor> u_t) {
            return self.forward(x, u_t.has_value() ? u_t.value() : torch::Tensor());
        }, py::arg("x"), py::arg("u_t") = py::none())
        .def("__call__", [](ContinuousHopfieldMemoryImpl& self, torch::Tensor x, std::optional<torch::Tensor> u_t) {
            return self.forward(x, u_t.has_value() ? u_t.value() : torch::Tensor());
        }, py::arg("x"), py::arg("u_t") = py::none());

    py::class_<LatentPredictorImpl, torch::nn::Module, std::shared_ptr<LatentPredictorImpl>>(m, "LatentPredictor")
        .def(py::init<int64_t, int64_t, std::string>(), py::arg("dim") = 256, py::arg("latent_dim") = 64, py::arg("device") = "cpu")
        .def("forward", &LatentPredictorImpl::forward)
        .def("__call__", &LatentPredictorImpl::forward);

    py::class_<HomeostaticNexusImpl, torch::nn::Module, std::shared_ptr<HomeostaticNexusImpl>>(m, "HomeostaticNexus")
        .def(py::init<std::string>(), py::arg("device") = "cpu")
        .def("sprout_dimension", &HomeostaticNexusImpl::sprout_dimension)
        .def("update", &HomeostaticNexusImpl::update)
        .def("get_states", &HomeostaticNexusImpl::get_states)
        .def("get_names", &HomeostaticNexusImpl::get_names);

    py::class_<OmniMorphicNodeImpl, torch::nn::Module, std::shared_ptr<OmniMorphicNodeImpl>>(m, "OmniMorphicNode")
        .def(py::init<int64_t, int64_t, int64_t, std::string, float, float>(),
             py::arg("dim") = 256, py::arg("state_dim") = 128, py::arg("num_operators") = 8, py::arg("device") = "cpu",
             py::arg("min_decay") = 0.005f, py::arg("max_decay") = 0.2f)
        .def("forward", &OmniMorphicNodeImpl::forward)
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
        .def("sprout_organelle", &CognitiveEvolvableAgentImpl::sprout_organelle)
        .def("sprout_homeostatic_dimension", &CognitiveEvolvableAgentImpl::sprout_homeostatic_dimension)
        .def("forward", &CognitiveEvolvableAgentImpl::forward, py::arg("tokens"), py::arg("u_t") = torch::Tensor())
        .def("__call__", &CognitiveEvolvableAgentImpl::forward, py::arg("tokens"), py::arg("u_t") = torch::Tensor())
        .def("step", &CognitiveEvolvableAgentImpl::step, py::arg("tokens"), py::arg("target_tokens"), py::arg("u_t"))
        .def("generate_thought_and_speech", &CognitiveEvolvableAgentImpl::generate_thought_and_speech, py::arg("seed_tokens"), py::arg("max_new_tokens") = 32, py::arg("temperature") = 0.45f, py::arg("top_p") = 0.90f)
        .def("parameters", [](std::shared_ptr<CognitiveEvolvableAgentImpl> a) { return a->parameters(); })
        .def("named_parameters", [](std::shared_ptr<CognitiveEvolvableAgentImpl> a) { return a->named_parameters(); });

    py::class_<UniversalMorphicCellImpl, torch::nn::Module, std::shared_ptr<UniversalMorphicCellImpl>>(m, "UniversalMorphicCell")
        .def(py::init<int64_t, int64_t, std::string, float, float>(),
             py::arg("dim") = 256, py::arg("num_units") = 4, py::arg("device") = "cpu",
             py::arg("min_decay") = 0.005f, py::arg("max_decay") = 0.2f)
        .def("forward", [](UniversalMorphicCellImpl& self, torch::Tensor x, std::optional<torch::Tensor> tunnel_in) {
            return self.forward(x, tunnel_in.has_value() ? tunnel_in.value() : torch::Tensor());
        }, py::arg("x"), py::arg("tunnel_in") = py::none())
        .def("__call__", [](UniversalMorphicCellImpl& self, torch::Tensor x, std::optional<torch::Tensor> tunnel_in) {
            return self.forward(x, tunnel_in.has_value() ? tunnel_in.value() : torch::Tensor());
        }, py::arg("x"), py::arg("tunnel_in") = py::none())
        .def("parameters", [](std::shared_ptr<UniversalMorphicCellImpl> m) { return m->parameters(); })
        .def("named_parameters", [](std::shared_ptr<UniversalMorphicCellImpl> m) { return m->named_parameters(); });

    py::class_<UniversalMorphicSpaceImpl, torch::nn::Module, std::shared_ptr<UniversalMorphicSpaceImpl>>(m, "UniversalMorphicSpace")
        .def(py::init<int64_t, int64_t, int64_t, int64_t, std::string, float, float>(),
             py::arg("vocab_size") = 258, py::arg("dim") = 256, py::arg("num_cells") = 2, py::arg("num_units") = 4,
             py::arg("device") = "cpu", py::arg("min_decay") = 0.005f, py::arg("max_decay") = 0.2f)
        .def("forward", &UniversalMorphicSpaceImpl::forward, py::arg("tokens"))
        .def("__call__", &UniversalMorphicSpaceImpl::forward, py::arg("tokens"))
        .def("forward_latent", &UniversalMorphicSpaceImpl::forward_latent, py::arg("tokens"))
        .def("parameters", [](std::shared_ptr<UniversalMorphicSpaceImpl> m) { return m->parameters(); })
        .def("named_parameters", [](std::shared_ptr<UniversalMorphicSpaceImpl> m) { return m->named_parameters(); })
        .def("named_parameters_map", [](std::shared_ptr<UniversalMorphicSpaceImpl> m) {
            std::map<std::string, torch::Tensor> params;
            for (const auto& pair : m->named_parameters()) {
                params[pair.key()] = pair.value();
            }
            return params;
        });
}
