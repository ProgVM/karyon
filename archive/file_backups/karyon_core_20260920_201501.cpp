#include <torch/extension.h>
#include <vector>
#include <string>
#include <cmath>
#include <tuple>
#include <memory>
#include <algorithm>
#include <iostream>
#include <map>

// ============================================================================
// KARYON COGNITIVE SUBSTRATE & ALLOSENSORY HOMEOSTASIS CORE
// v34.0 - Active Inference, Laminar Error Routing, and Continuous Hopfield Memory
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
        return torch::cat(op_outputs, -1); // [batch, seq, dim, num_operators]
    }
};
TORCH_MODULE(ParallelOperatorBank);

// 4. CONTINUOUS HOPFIELD ATTRACTOR NETWORK
class ContinuousHopfieldMemoryImpl : public torch::nn::Module {
public:
    int64_t dim;
    int64_t num_basins;
    std::string device_str;

    torch::Tensor basins; // [num_basins, dim]
    torch::Tensor scaling; // [num_basins]

    ContinuousHopfieldMemoryImpl(int64_t dim = 256, int64_t num_basins = 32, std::string device_str = "cpu")
        : dim(dim), num_basins(num_basins), device_str(device_str) {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

        auto init_basins = torch::randn({num_basins, dim}, torch::TensorOptions().device(device));
        init_basins = init_basins / init_basins.norm(2, -1, true); // Unit-sphere normalization
        basins = register_parameter("basins", init_basins);

        scaling = register_parameter("scaling", torch::ones({num_basins}, torch::TensorOptions().device(device)) * 12.0f); // beta=12.0
        this->to(device);
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor u_t) {
        auto device = x.device();
        auto norm_x = x / (x.norm(2, -1, true) + 1e-6f);

        // Compute cosine similarities: [batch, seq, dim] * [dim, num_basins] -> [batch, seq, num_basins]
        auto sim = torch::matmul(norm_x, basins.t());

        // Dynamic Dopaminergic Precision Sharpening
        float da_val = 0.0f;
        if (u_t.defined() && u_t.numel() > 0) {
            da_val = u_t.slice(-1, 5, 6).mean().template item<float>();
        }
        auto beta = scaling * (1.0f + 1.5f * da_val);

        auto energy_weights = torch::softmax(sim * beta.view({1, 1, -1}), -1);

        // Retrieve mapped memory representation: [batch, seq, num_basins] * [num_basins, dim] -> [batch, seq, dim]
        auto recalled = torch::matmul(energy_weights, basins);
        return recalled;
    }
};
TORCH_MODULE(ContinuousHopfieldMemory);

// 5. LATENT ACTIVE INFERENCE WORLD MODEL
class LatentPredictorImpl : public torch::nn::Module {
public:
    int64_t dim;
    int64_t latent_dim;
    std::string device_str;

    torch::nn::Linear prior_mean{nullptr};
    torch::nn::Linear prior_logvar{nullptr};
    torch::nn::Linear post_mean{nullptr};
    torch::nn::Linear post_logvar{nullptr};

    LatentPredictorImpl(int64_t dim = 256, int64_t latent_dim = 64, std::string device_str = "cpu")
        : dim(dim), latent_dim(latent_dim), device_str(device_str) {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

        prior_mean = register_module("prior_mean", torch::nn::Linear(dim, latent_dim));
        prior_logvar = register_module("prior_logvar", torch::nn::Linear(dim, latent_dim));
        post_mean = register_module("post_mean", torch::nn::Linear(dim * 2, latent_dim));
        post_logvar = register_module("post_logvar", torch::nn::Linear(dim * 2, latent_dim));

        this->to(device);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor h_prev, torch::Tensor x_curr) {
        auto p_mean = prior_mean->forward(h_prev);
        auto p_logvar = prior_logvar->forward(h_prev);

        auto concat_in = torch::cat({h_prev, x_curr}, -1);
        auto q_mean = post_mean->forward(concat_in);
        auto q_logvar = post_logvar->forward(concat_in);

        return std::make_tuple(p_mean, p_logvar, q_mean, q_logvar);
    }
};
TORCH_MODULE(LatentPredictor);

// 6. ALLOSESTATIC HOMEOSTATIC NEXUS
class HomeostaticNexusImpl : public torch::nn::Module {
public:
    std::vector<std::string> names;
    std::vector<float> setpoints;
    std::vector<float> states;
    std::string device_str;

    HomeostaticNexusImpl(std::string device_str = "cpu") : device_str(device_str) {
        names = {"Curiosity", "Energy", "Stability", "Health", "Noradrenaline", "Dopamine"};
        setpoints = {0.8f, 0.9f, 0.7f, 0.95f, 0.3f, 0.5f};
        states = {0.8f, 0.9f, 0.7f, 0.95f, 0.3f, 0.5f};
    }

    void sprout_homeostatic_dimension(std::string name, float setpoint) {
        if (std::find(names.begin(), names.end(), name) == names.end()) {
            names.push_back(name);
            setpoints.push_back(setpoint);
            states.push_back(setpoint);
        }
    }

    void update(torch::Tensor free_energy_surprise) {
        float f_t = free_energy_surprise.mean().template item<float>();

        // Allostatic update rules coupled to surprise (Principle 14)
        states[4] = std::clamp(states[4] * 0.9f + f_t * 0.2f, 0.05f, 0.95f); // Noradrenaline arousal
        states[5] = std::clamp(states[5] * 0.95f + (0.5f - f_t) * 0.1f, 0.05f, 0.95f); // Dopamine reward
        states[1] = std::clamp(states[1] - 0.002f + states[5] * 0.001f, 0.05f, 1.0f); // Metabolic energy consumption
        states[2] = std::clamp(states[2] * 0.98f + (states[1] > 0.3f ? 0.02f : -0.05f), 0.05f, 1.0f); // Stability
        states[3] = std::clamp(states[3] * 0.999f - (states[1] < 0.15f ? 0.01f : 0.0f), 0.05f, 1.0f); // Health
    }

    torch::Tensor get_states() {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        return torch::tensor(states, torch::TensorOptions().device(device));
    }

    std::vector<std::string> get_names() {
        return names;
    }
};
TORCH_MODULE(HomeostaticNexus);

// 7. OMNI-MORPHIC NODE COMPONENT
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

    OmniMorphicNodeImpl(int64_t dim = 256, int64_t state_dim = 128, int64_t num_operators = 8, std::string device_str = "cpu", float min_decay = 0.005f, float max_decay = 0.2f)
        : dim(dim), state_dim(state_dim), num_operators(num_operators), device_str(device_str) {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

        affinity_query = register_parameter("affinity_query", torch::randn({dim}, torch::TensorOptions().device(device)) * 0.05f);

        hyper_w1 = register_module("hyper_w1", torch::nn::Linear(dim, state_dim));
        hyper_w2 = register_module("hyper_w2", torch::nn::Linear(state_dim, num_operators));

        causal_ssd = register_module("causal_ssd", CausalParallelSSD(dim, device_str, min_decay, max_decay));
        operator_bank = register_module("operator_bank", ParallelOperatorBank(dim, state_dim, num_operators));

        state_norm = register_module("state_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));
        this->to(device);
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor signal_manifold, torch::Tensor u_t) {
        int64_t batch = signal_manifold.size(0);
        int64_t seq_len = signal_manifold.size(2);
        auto device = signal_manifold.device();

        // 1. Softmax Attention Routing
        auto signal_norm = signal_manifold / (signal_manifold.norm(2, -1, true) + 1e-6f);
        auto attn_scores = torch::einsum("bnsd, d -> bns", {signal_norm, affinity_query});

        // Noradrenergic gain scaling
        float na_val = 0.0f;
        if (u_t.defined() && u_t.numel() > 0) {
            na_val = u_t.slice(-1, 4, 5).mean().template item<float>();
        }
        auto routing_weights = torch::softmax(attn_scores * (1.0f + 2.0f * na_val), 1);

        auto routed_input = torch::einsum("bns, bnsd -> bsd", {routing_weights, signal_manifold});

        // 2. State-Space Temporal Scan (Continuous Memory)
        auto scan_state = causal_ssd->forward(routed_input);

        // 3. Parallel Operator Mixing
        auto op_bank_out = operator_bank->compute_operators(scan_state); // [batch, seq, dim, num_operators]

        // Dynamic operator selection via hypernetwork
        auto h_mean = scan_state.mean(1); // [batch, dim]
        auto op_logits = hyper_w2->forward(torch::relu(hyper_w1->forward(h_mean))); // [batch, num_operators]
        auto op_probs = torch::softmax(op_logits, -1); // [batch, num_operators]

        auto final_out = torch::einsum("bsdo, bo -> bsd", {op_bank_out, op_probs});
        final_out = state_norm->forward(final_out);

        return std::make_tuple(final_out, routing_weights);
    }
};
TORCH_MODULE(OmniMorphicNode);

// 8. OMNI-CONTINUOUS GRAPH SUBSTRATE
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
            auto u_flat = u_t.reshape({-1, u_t.size(-1)}).slice(1, 0, 6);
            if (u_flat.size(0) != batch) {
                u_flat = u_flat.expand({batch, -1});
            }
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
    int64_t max_nodes;
    std::string device_str;

    UniversalManifold manifold{nullptr};
    HomeostaticNexus homeostasis{nullptr};
    OmniContinuousGraphSubstrate substrate{nullptr};
    ContinuousHopfieldMemory memory{nullptr};
    LatentPredictor world_model{nullptr};

    torch::nn::LayerNorm norm{nullptr};
    torch::nn::Linear head{nullptr};

    CognitiveEvolvableAgentImpl(int64_t vocab_size = 258, int64_t dim = 256, int64_t max_nodes = 16, std::string device_str = "cpu")
        : vocab_size(vocab_size), dim(dim), max_nodes(max_nodes), device_str(device_str) {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

        manifold = register_module("manifold", UniversalManifold(vocab_size, dim, device_str));
        homeostasis = register_module("homeostasis", HomeostaticNexus(device_str));
        substrate = register_module("substrate", OmniContinuousGraphSubstrate(dim, max_nodes, device_str));
        memory = register_module("memory", ContinuousHopfieldMemory(dim, 32, device_str));
        world_model = register_module("world_model", LatentPredictor(dim, 64, device_str));

        norm = register_module("norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));

        auto head_opts = torch::nn::LinearOptions(dim, vocab_size).bias(false);
        head = register_module("head", torch::nn::Linear(head_opts));
        torch::nn::init::normal_(head->weight, 0.0, 0.02);

        this->to(device);
    }

    void sprout_organelle(std::string name, int64_t state_dim = 128, int64_t num_operators = 8, float min_decay = 0.005f, float max_decay = 0.2f) {
        substrate->sprout_node(name, state_dim, num_operators, min_decay, max_decay);
    }

    void sprout_homeostatic_dimension(std::string name, float setpoint) {
        homeostasis->sprout_homeostatic_dimension(name, setpoint);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor tokens, torch::Tensor u_t = torch::Tensor()) {
        auto device = tokens.device();
        auto embed = manifold->forward(tokens); // [batch, seq, dim]

        if (!u_t.defined() || u_t.numel() == 0) {
            u_t = homeostasis->get_states();
        }

        // 1. Attractor memory recall
        auto mem_recalled = memory->forward(embed, u_t);

        // 2. Dynamic graph routing & processing
        torch::Tensor graph_out;
        std::vector<torch::Tensor> signals;
        std::tie(graph_out, signals) = substrate->forward({embed, mem_recalled}, u_t);

        // 3. Active Inference Prediction
        torch::Tensor p_mu, p_logvar, q_mu, q_logvar;
        std::tie(p_mu, p_logvar, q_mu, q_logvar) = world_model->forward(embed, graph_out);

        // 4. Readout with Dopaminergic motor resonance gain (Principle 2)
        float da_val = u_t.slice(-1, 5, 6).mean().template item<float>();
        auto final_state = norm->forward(graph_out);
        auto logits = head->forward(final_state) * (1.0f + 1.5f * da_val);

        return std::make_tuple(logits, p_mu, p_logvar, q_mu, q_logvar);
    }

    torch::Tensor step(torch::Tensor tokens, torch::Tensor target_tokens, torch::Tensor u_t) {
        torch::Tensor logits, p_mu, p_logvar, q_mu, q_logvar;
        std::tie(logits, p_mu, p_logvar, q_mu, q_logvar) = forward(tokens, u_t);

        // Reconstruction cross entropy
        auto loss_rec = torch::nll_loss(torch::log_softmax(logits.view({-1, vocab_size}), -1), target_tokens.view(-1));

        // Dimension-Normalized KL Divergence (Principle 2)
        auto kl = 0.5f * torch::sum(p_logvar - q_logvar + (torch::exp(q_logvar) + torch::pow(q_mu - p_mu, 2)) / torch::exp(p_logvar) - 1.0f, -1).mean();
        
        // Active inference free energy
        auto free_energy = loss_rec + 0.01f * kl;

        // Update homeostatic Nexus with surprise
        homeostasis->update(free_energy);

        // Spontaneous Neurogenesis trigger
        substrate->trigger_spontaneous_neurogenesis(free_energy.template item<float>());

        return free_energy;
    }

    std::vector<int64_t> generate_thought_and_speech(torch::Tensor seed_tokens, int64_t max_new_tokens = 32, float temperature = 0.45f, float top_p = 0.90f) {
        torch::NoGradGuard no_grad;
        auto device = seed_tokens.device();
        std::vector<int64_t> generated;

        auto current_tokens = seed_tokens.clone();

        for (int64_t step = 0; step < max_new_tokens; ++step) {
            torch::Tensor logits, p_mu, p_logvar, q_mu, q_logvar;
            std::tie(logits, p_mu, p_logvar, q_mu, q_logvar) = forward(current_tokens);

            auto next_token_logits = logits.select(1, logits.size(1) - 1) / temperature;
            auto probs = torch::softmax(next_token_logits, -1);

            // Simple Top-p nucleus sampling
            auto sorted_probs_tuple = torch::sort(probs, -1, true);
            auto sorted_probs = std::get<0>(sorted_probs_tuple);
            auto sorted_indices = std::get<1>(sorted_probs_tuple);

            auto cumulative_probs = torch::cumsum(sorted_probs, -1);
            auto cutoff = cumulative_probs > top_p;
            // Keep first element even if it exceeds top_p
            cutoff.select(1, 0).copy_(torch::zeros({probs.size(0)}, torch::kBool));

            sorted_probs.masked_fill_(cutoff, 0.0f);
            sorted_probs = sorted_probs / sorted_probs.sum(-1, true);

            auto next_token_idx = torch::multinomial(sorted_probs, 1);
            auto next_token = sorted_indices.gather(-1, next_token_idx);

            generated.push_back(next_token.item<int64_t>());
            current_tokens = torch::cat({current_tokens, next_token}, -1);
        }
        return generated;
    }
};
TORCH_MODULE(CognitiveEvolvableAgent);

// 10. UNIVERSAL MORPHIC CELL
class UniversalMorphicCellImpl : public torch::nn::Module {
public:
    int64_t dim;
    int64_t num_units;
    std::string device_str;

    std::vector<CausalParallelSSD> ssd_units;
    torch::nn::Linear gate_in{nullptr};
    torch::nn::Linear gate_out{nullptr};
    torch::nn::LayerNorm norm{nullptr};

    UniversalMorphicCellImpl(int64_t dim = 256, int64_t num_units = 4, std::string device_str = "cpu", float min_decay = 0.005f, float max_decay = 0.2f)
        : dim(dim), num_units(num_units), device_str(device_str) {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

        for (int64_t i = 0; i < num_units; ++i) {
            auto ssd = CausalParallelSSD(dim, device_str, min_decay, max_decay);
            register_module("ssd_unit_" + std::to_string(i), ssd);
            ssd_units.push_back(ssd);
        }

        gate_in = register_module("gate_in", torch::nn::Linear(dim, dim));
        gate_out = register_module("gate_out", torch::nn::Linear(dim * num_units, dim));
        norm = register_module("norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));

        this->to(device);
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x, torch::Tensor tunnel_in) {
        auto device = x.device();
        auto gated_in = torch::silu(gate_in->forward(x));

        if (tunnel_in.defined() && tunnel_in.numel() > 0) {
            gated_in = gated_in + tunnel_in;
        }

        std::vector<torch::Tensor> unit_outputs;
        for (int64_t i = 0; i < num_units; ++i) {
            unit_outputs.push_back(ssd_units[i]->forward(gated_in));
        }

        auto concat_out = torch::cat(unit_outputs, -1); // [batch, seq, dim * num_units]
        auto mixed_out = gate_out->forward(concat_out);

        auto final_out = norm->forward(mixed_out + x);
        return std::make_tuple(final_out, mixed_out);
    }
};
TORCH_MODULE(UniversalMorphicCell);

// 11. UNIVERSAL MORPHIC SPACE
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
// 12. NATIVE C++ DYNAMIC MORPHIC GRAPH OPERATORS (EXP-273)
// ============================================================================

struct GraphOp : public torch::nn::Module {
    virtual torch::Tensor forward(torch::Tensor x) = 0;
};

struct LinearAccumulatorOpImpl : public GraphOp {
    torch::Tensor w, b;
    LinearAccumulatorOpImpl(int64_t dim, std::string device_str = "cpu") {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        w = register_parameter("w", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * (0.2f / std::sqrt(dim)));
        b = register_parameter("b", torch::zeros({dim}, torch::TensorOptions().device(device)));
    }
    torch::Tensor forward(torch::Tensor x) override {
        return torch::matmul(x, w.t()) + b;
    }
};
TORCH_MODULE(LinearAccumulatorOp);


struct BilinearMultiplicativeOpImpl : public GraphOp {
    torch::Tensor w_left, w_right, w_out;
    BilinearMultiplicativeOpImpl(int64_t dim, std::string device_str = "cpu") {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        w_left = register_parameter("w_left", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * (0.2f / std::sqrt(dim)));
        w_right = register_parameter("w_right", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * (0.2f / std::sqrt(dim)));
        w_out = register_parameter("w_out", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * (0.15f / std::sqrt(dim)));
    }
    torch::Tensor forward(torch::Tensor x) override {
        auto left = torch::matmul(x, w_left.t());
        auto right = torch::matmul(x, w_right.t());
        return torch::matmul(torch::silu(left * right), w_out.t());
    }
};
TORCH_MODULE(BilinearMultiplicativeOp);


struct SaturatedAttractorOpImpl : public GraphOp {
    torch::Tensor w_gate, w_val, w_out;
    SaturatedAttractorOpImpl(int64_t dim, std::string device_str = "cpu") {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        w_gate = register_parameter("w_gate", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * (0.2f / std::sqrt(dim)));
        w_val = register_parameter("w_val", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * (0.2f / std::sqrt(dim)));
        w_out = register_parameter("w_out", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * (0.15f / std::sqrt(dim)));
    }
    torch::Tensor forward(torch::Tensor x) override {
        auto gate = torch::sigmoid(torch::matmul(x, w_gate.t()));
        auto val = torch::tanh(torch::matmul(x, w_val.t()));
        return torch::matmul(gate * val, w_out.t());
    }
};
TORCH_MODULE(SaturatedAttractorOp);


class DynamicMorphicGraphImpl : public torch::nn::Module {
public:
    int64_t dim;
    std::string device_str;
    int64_t k_nodes = 0;

    std::vector<std::shared_ptr<GraphOp>> node_ops;
    std::vector<torch::Tensor> alpha_epi;
    std::vector<bool> is_core_node;
    std::vector<std::string> node_names;
    std::vector<std::string> node_types;

    torch::Tensor w_route;
    torch::Tensor w_sensory_in, w_motor_out;

    DynamicMorphicGraphImpl(int64_t dim = 128, std::string device_str = "cpu")
        : dim(dim), device_str(device_str) {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

        w_route = register_parameter("w_route", torch::zeros({0, 0}, torch::TensorOptions().device(device)));
        w_sensory_in = register_parameter("w_sensory_in", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * 0.2f);
        w_motor_out = register_parameter("w_motor_out", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * 0.2f);
        this->to(device);
    }

    void add_node(std::string name, std::string op_type, bool is_core = false, float initial_alpha = 0.0f) {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        
        std::shared_ptr<GraphOp> op;
        if (op_type == "LinearAccumulator") {
            op = std::make_shared<LinearAccumulatorOpImpl>(dim, device_str);
        } else if (op_type == "BilinearMultiplicative") {
            op = std::make_shared<BilinearMultiplicativeOpImpl>(dim, device_str);
        } else if (op_type == "SaturatedAttractor") {
            op = std::make_shared<SaturatedAttractorOpImpl>(dim, device_str);
        } else {
            throw std::invalid_argument("Unknown operator type: " + op_type);
        }

        std::string module_key = "op_" + std::to_string(k_nodes);
        register_module(module_key, op);
        node_ops.push_back(op);
        
        is_core_node.push_back(is_core);
        node_names.push_back(name);
        node_types.push_back(op_type);

        auto alpha_val = torch::tensor(initial_alpha, torch::TensorOptions().device(device).requires_grad(!is_core));
        auto alpha_param = register_parameter(module_key + "_alpha", alpha_val);
        alpha_epi.push_back(alpha_param);

        int64_t old_k = k_nodes;
        int64_t new_k = old_k + 1;
        k_nodes = new_k;

        // Expand dynamic routing matrix w_route
        auto new_w_route = torch::zeros({new_k, new_k}, torch::TensorOptions().device(device));
        if (old_k > 0) {
            torch::NoGradGuard no_grad;
            new_w_route.slice(0, 0, old_k).slice(1, 0, old_k).copy_(w_route.data());
            // Random connection weights between new node and existing nodes
            auto rand_col = torch::randn({old_k}, torch::TensorOptions().device(device)) * (0.1f / std::sqrt(old_k));
            auto rand_row = torch::randn({old_k}, torch::TensorOptions().device(device)) * (0.1f / std::sqrt(old_k));
            new_w_route.slice(0, 0, old_k).narrow(1, old_k, 1).copy_(rand_col.unsqueeze(1));
            new_w_route.narrow(0, old_k, 1).slice(1, 0, old_k).copy_(rand_row.unsqueeze(0));
            new_w_route.index_put_({old_k, old_k}, 0.05f);
        }
        new_w_route.set_requires_grad(true);
        w_route.set_data(new_w_route);
    }

    torch::Tensor forward(torch::Tensor x_sensory, int64_t thinking_steps = 4) {
        int64_t B = x_sensory.size(0);
        int64_t K = k_nodes;
        auto device = x_sensory.device();

        auto node_states = torch::zeros({K, B, dim}, torch::TensorOptions().device(device));
        auto sensory_in = torch::matmul(x_sensory, w_sensory_in.t());
        node_states[0] = sensory_in;

        for (int64_t step = 0; step < thinking_steps; ++step) {
            auto aggregated_inputs = torch::einsum("ij,ibd->jbd", {torch::tanh(w_route), node_states});
            aggregated_inputs[0] = aggregated_inputs[0] + sensory_in;

            std::vector<torch::Tensor> new_states;
            for (int64_t j = 0; j < K; ++j) {
                auto raw_out = node_ops[j]->forward(aggregated_inputs[j]);
                auto alpha = alpha_epi[j];
                auto graft_gate = torch::tanh(alpha);
                auto grafted_out = graft_gate * raw_out;
                new_states.push_back(grafted_out);
            }
            node_states = torch::stack(new_states, 0);
        }

        auto motor_latent = node_states[1];
        auto readout = torch::matmul(motor_latent, w_motor_out.t());
        return readout;
    }

    std::string get_topology_manifest() {
        std::string manifest = "{\"k_nodes\":" + std::to_string(k_nodes) + ",\"nodes\":[";
        for (int64_t i = 0; i < k_nodes; ++i) {
            manifest += "{\"name\":\"" + node_names[i] + "\",\"type\":\"" + node_types[i] + "\",\"is_core\":" + (is_core_node[i] ? "true" : "false") + "}";
            if (i < k_nodes - 1) manifest += ",";
        }
        manifest += "]}";
        return manifest;
    }
};
TORCH_MODULE(DynamicMorphicGraph);


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
        .def("sprout_homeostatic_dimension", &HomeostaticNexusImpl::sprout_homeostatic_dimension)
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
        .def("sprout_organelle", &CognitiveEvolvableAgentImpl::sprout_organelle,
             py::arg("name"), py::arg("state_dim") = 128, py::arg("num_operators") = 8,
             py::arg("min_decay") = 0.005f, py::arg("max_decay") = 0.2f)
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

    py::class_<LinearAccumulatorOpImpl, torch::nn::Module, std::shared_ptr<LinearAccumulatorOpImpl>>(m, "LinearAccumulatorOp")
        .def(py::init<int64_t, std::string>(), py::arg("dim"), py::arg("device_str") = "cpu")
        .def("forward", &LinearAccumulatorOpImpl::forward)
        .def("__call__", &LinearAccumulatorOpImpl::forward);

    py::class_<BilinearMultiplicativeOpImpl, torch::nn::Module, std::shared_ptr<BilinearMultiplicativeOpImpl>>(m, "BilinearMultiplicativeOp")
        .def(py::init<int64_t, std::string>(), py::arg("dim"), py::arg("device_str") = "cpu")
        .def("forward", &BilinearMultiplicativeOpImpl::forward)
        .def("__call__", &BilinearMultiplicativeOpImpl::forward);

    py::class_<SaturatedAttractorOpImpl, torch::nn::Module, std::shared_ptr<SaturatedAttractorOpImpl>>(m, "SaturatedAttractorOp")
        .def(py::init<int64_t, std::string>(), py::arg("dim"), py::arg("device_str") = "cpu")
        .def("forward", &SaturatedAttractorOpImpl::forward)
        .def("__call__", &SaturatedAttractorOpImpl::forward);

    py::class_<DynamicMorphicGraphImpl, torch::nn::Module, std::shared_ptr<DynamicMorphicGraphImpl>>(m, "DynamicMorphicGraph")
        .def(py::init<int64_t, std::string>(), py::arg("dim") = 128, py::arg("device_str") = "cpu")
        .def_readonly("k_nodes", &DynamicMorphicGraphImpl::k_nodes)
        .def("add_node", &DynamicMorphicGraphImpl::add_node, py::arg("name"), py::arg("op_type"), py::arg("is_core") = false, py::arg("initial_alpha") = 0.0f)
        .def("forward", &DynamicMorphicGraphImpl::forward, py::arg("x_sensory"), py::arg("thinking_steps") = 4)
        .def("__call__", &DynamicMorphicGraphImpl::forward, py::arg("x_sensory"), py::arg("thinking_steps") = 4)
        .def("get_topology_manifest", &DynamicMorphicGraphImpl::get_topology_manifest)
        .def("named_parameters_map", [](std::shared_ptr<DynamicMorphicGraphImpl> m) {
            std::map<std::string, torch::Tensor> params;
            for (const auto& pair : m->named_parameters()) {
                params[pair.key()] = pair.value();
            }
            return params;
        });
}
