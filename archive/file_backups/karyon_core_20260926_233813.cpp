// karyon_core.cpp
#include <torch/extension.h>
#include <torch/torch.h>
#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <map>
#include <optional>
#include <stdexcept>
#include <iostream>

namespace py = pybind11;

// ============================================================================
// 1. UNIVERSAL MANIFOLD EMBEDDING (Raw Byte V=258 Representation Space)
// ============================================================================
class UniversalManifoldImpl : public torch::nn::Module {
public:
    int64_t vocab_size;
    int64_t dim;
    torch::Tensor byte_embedding;

    UniversalManifoldImpl(int64_t vocab_size = 258, int64_t dim = 256, std::string device_str = "cpu")
        : vocab_size(vocab_size), dim(dim) {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        byte_embedding = register_parameter("byte_embedding", torch::randn({vocab_size, dim}, torch::TensorOptions().device(device)) * (1.0f / std::sqrt(dim)));
        this->to(device);
    }

    torch::Tensor forward(torch::Tensor tokens) {
        return torch::embedding(byte_embedding, tokens);
    }
};
TORCH_MODULE(UniversalManifold);

// ============================================================================
// 2. CAUSAL PARALLEL SSD (State-Space Duality Parallel Scan Engine)
// ============================================================================
class CausalParallelSSDImpl : public torch::nn::Module {
public:
    int64_t dim;
    std::string device_str;
    torch::Tensor log_decay;
    torch::Tensor w_in, w_out;

    CausalParallelSSDImpl(int64_t dim = 256, std::string device_str = "cpu", float min_decay = 0.005f, float max_decay = 0.2f)
        : dim(dim), device_str(device_str) {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        
        auto log_min = std::log(min_decay);
        auto log_max = std::log(max_decay);
        log_decay = register_parameter("log_decay", torch::linspace(log_min, log_max, dim, torch::TensorOptions().device(device)));
        
        w_in = register_parameter("w_in", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * (1.0f / std::sqrt(dim)));
        w_out = register_parameter("w_out", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * (1.0f / std::sqrt(dim)));
        this->to(device);
    }

    torch::Tensor forward(torch::Tensor x) {
        auto B = x.size(0);
        auto S = x.size(1);
        auto D = x.size(2);
        auto device = x.device();

        auto decay = torch::exp(log_decay).clamp(0.001f, 0.999f);
        auto u = torch::matmul(x, w_in.t());

        auto indices = torch::arange(S, torch::TensorOptions().device(device)).to(torch::kFloat32);
        auto dist = indices.unsqueeze(1) - indices.unsqueeze(0);
        auto causal_mask = dist >= 0;

        auto log_d = -decay.unsqueeze(0).unsqueeze(0) * dist.unsqueeze(-1);
        auto decay_weights = torch::exp(log_d) * causal_mask.unsqueeze(-1).to(torch::kFloat32);

        auto y = torch::einsum("skd,bkd->bsd", {decay_weights, u});
        return torch::matmul(y, w_out.t());
    }
};
TORCH_MODULE(CausalParallelSSD);

// ============================================================================
// 3. PARALLEL OPERATOR BANK (Multi-Timescale Memory Operators)
// ============================================================================
class ParallelOperatorBankImpl : public torch::nn::Module {
public:
    int64_t dim;
    int64_t state_dim;
    int64_t num_operators;
    torch::Tensor op_weights;

    ParallelOperatorBankImpl(int64_t dim = 256, int64_t state_dim = 128, int64_t num_operators = 8)
        : dim(dim), state_dim(state_dim), num_operators(num_operators) {
        op_weights = register_parameter("op_weights", torch::randn({num_operators, dim, state_dim}) * (1.0f / std::sqrt(dim)));
    }

    torch::Tensor compute_operators(torch::Tensor h) {
        return torch::einsum("bd,ndk->bnk", {h, op_weights});
    }
};
TORCH_MODULE(ParallelOperatorBank);

// ============================================================================
// 4. CONTINUOUS HOPFIELD MEMORY
// ============================================================================
class ContinuousHopfieldMemoryImpl : public torch::nn::Module {
public:
    int64_t dim;
    int64_t num_basins;
    torch::Tensor basins;

    ContinuousHopfieldMemoryImpl(int64_t dim = 256, int64_t num_basins = 32, std::string device_str = "cpu")
        : dim(dim), num_basins(num_basins) {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        basins = register_parameter("basins", torch::randn({num_basins, dim}, torch::TensorOptions().device(device)) * (1.0f / std::sqrt(dim)));
        this->to(device);
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor u_t = torch::Tensor()) {
        auto norm_b = torch::nn::functional::normalize(basins, torch::nn::functional::NormalizeFuncOptions().dim(-1));
        float beta = 8.0f;
        if (u_t.defined() && u_t.numel() >= 6) {
            float da = u_t[5].item<float>();
            beta *= (1.0f + 1.5f * da);
        }
        auto sim = torch::matmul(x, norm_b.t()) * beta;
        auto attn = torch::softmax(sim, -1);
        return torch::matmul(attn, norm_b);
    }
};
TORCH_MODULE(ContinuousHopfieldMemory);

// ============================================================================
// 5. CONTINUOUS SACCADIC DRIFT (C-SSD)
// ============================================================================
class ContinuousSaccadicDriftImpl : public torch::nn::Module {
public:
    int64_t dim;
    int64_t num_filters;
    std::string device_str;
    torch::Tensor drift_weight;
    torch::Tensor spread_weight;

    ContinuousSaccadicDriftImpl(int64_t dim = 128, int64_t num_filters = 17, std::string device_str = "cpu")
        : dim(dim), num_filters(num_filters), device_str(device_str) {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        drift_weight = register_parameter("drift_weight", torch::randn({dim, 1}, torch::TensorOptions().device(device)) * 0.05f);
        spread_weight = register_parameter("spread_weight", torch::randn({dim, 1}, torch::TensorOptions().device(device)) * 0.05f);
        this->to(device);
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor bump_state, torch::Tensor h_core, float da_gain = 0.0f, torch::Tensor salience_bias = torch::Tensor()) {
        auto B = bump_state.size(0);
        auto device = bump_state.device();

        auto drift_delta = torch::matmul(h_core, drift_weight).squeeze(-1);
        auto dynamic_step = (0.1f + 0.5f * torch::tanh(drift_delta)).clamp(0.01f, 2.0f);

        auto spread_delta = torch::matmul(h_core, spread_weight).squeeze(-1);
        auto sigma = (1.5f + 1.0f * torch::sigmoid(spread_delta)).clamp(0.5f, 5.0f);

        auto grid = torch::arange(num_filters, torch::TensorOptions().device(device)).to(torch::kFloat32);
        auto center = (num_filters - 1) / 2.0f;
        auto relative_offsets = grid - center;

        auto diff = relative_offsets.unsqueeze(0) - dynamic_step.unsqueeze(1);
        auto kernel = torch::exp(-0.5f * torch::pow(diff / sigma.unsqueeze(1), 2.0f));
        kernel = kernel / (kernel.sum(-1, true) + 1e-6f);

        auto bump_padded = torch::nn::functional::pad(
            bump_state.unsqueeze(1),
            torch::nn::functional::PadFuncOptions({(num_filters - 1) / 2, (num_filters - 1) / 2}).mode(torch::kReplicate)
        );

        auto drifted_bump = torch::zeros_like(bump_state);
        for (int64_t b = 0; b < B; ++b) {
            auto b_padded = bump_padded.slice(0, b, b + 1);
            auto k = kernel.slice(0, b, b + 1).unsqueeze(0);
            drifted_bump.slice(0, b, b + 1) = torch::conv1d(b_padded, k).squeeze(1);
        }

        if (salience_bias.defined() && salience_bias.numel() > 0) {
            drifted_bump = drifted_bump * torch::sigmoid(salience_bias);
            drifted_bump = drifted_bump / (drifted_bump.sum(-1, true) + 1e-6f);
        }

        return std::make_tuple(drifted_bump, dynamic_step);
    }
};
TORCH_MODULE(ContinuousSaccadicDrift);

// ============================================================================
// 6. LATENT PREDICTOR (Active Inference World Model)
// ============================================================================
class LatentPredictorImpl : public torch::nn::Module {
public:
    int64_t dim;
    int64_t latent_dim;
    torch::Tensor w_mu, w_logvar;

    LatentPredictorImpl(int64_t dim = 256, int64_t latent_dim = 64, std::string device_str = "cpu")
        : dim(dim), latent_dim(latent_dim) {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        w_mu = register_parameter("w_mu", torch::randn({dim, latent_dim}, torch::TensorOptions().device(device)) * (1.0f / std::sqrt(dim)));
        w_logvar = register_parameter("w_logvar", torch::randn({dim, latent_dim}, torch::TensorOptions().device(device)) * (1.0f / std::sqrt(dim)));
        this->to(device);
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor h) {
        auto mu = torch::matmul(h, w_mu);
        auto logvar = torch::matmul(h, w_logvar).clamp(-6.0f, 2.0f);
        return std::make_tuple(mu, logvar);
    }
};
TORCH_MODULE(LatentPredictor);

// ============================================================================
// 7. HOMEOSTATIC NEXUS
// ============================================================================
class HomeostaticNexusImpl : public torch::nn::Module {
public:
    std::vector<std::string> state_names;
    torch::Tensor states;

    HomeostaticNexusImpl(std::string device_str = "cpu") {
        state_names = {"Curiosity", "Energy", "Stability", "Health", "Noradrenaline", "Dopamine"};
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        states = register_buffer("states", torch::tensor({0.5f, 1.0f, 0.8f, 1.0f, 0.1f, 0.1f}, torch::TensorOptions().device(device)));
    }

    void sprout_homeostatic_dimension(std::string name, float initial_val = 0.5f) {
        state_names.push_back(name);
        auto new_val = torch::tensor({initial_val}, states.options());
        states = torch::cat({states, new_val}, 0);
    }

    void update(torch::Tensor deltas) {
        states = (states + deltas).clamp(0.0f, 1.0f);
    }

    torch::Tensor get_states() { return states; }
    std::vector<std::string> get_names() { return state_names; }
};
TORCH_MODULE(HomeostaticNexus);

// ============================================================================
// 8. GRAPH OPERATOR PRIMITIVES
// ============================================================================
class GraphOp : public torch::nn::Module {
public:
    virtual torch::Tensor forward(torch::Tensor x) = 0;
    virtual ~GraphOp() = default;
};

class LinearAccumulatorOpImpl : public GraphOp {
public:
    int64_t dim;
    torch::Tensor w;
    LinearAccumulatorOpImpl(int64_t dim, std::string device_str) : dim(dim) {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        w = register_parameter("w", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * (1.0f / std::sqrt((float)dim)));
        this->to(device);
    }
    torch::Tensor forward(torch::Tensor x) override {
        return torch::matmul(x, w.t());
    }
};

class BilinearMultiplicativeOpImpl : public GraphOp {
public:
    int64_t dim;
    torch::Tensor w_a, w_b;
    BilinearMultiplicativeOpImpl(int64_t dim, std::string device_str) : dim(dim) {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        w_a = register_parameter("w_a", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * (1.0f / std::sqrt((float)dim)));
        w_b = register_parameter("w_b", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * (1.0f / std::sqrt((float)dim)));
        this->to(device);
    }
    torch::Tensor forward(torch::Tensor x) override {
        auto a = torch::matmul(x, w_a.t());
        auto b = torch::matmul(x, w_b.t());
        return a * torch::sigmoid(b);
    }
};

class SaturatedAttractorOpImpl : public GraphOp {
public:
    int64_t dim;
    torch::Tensor w;
    SaturatedAttractorOpImpl(int64_t dim, std::string device_str) : dim(dim) {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        w = register_parameter("w", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * (1.0f / std::sqrt((float)dim)));
        this->to(device);
    }
    torch::Tensor forward(torch::Tensor x) override {
        return torch::tanh(torch::matmul(x, w.t()));
    }
};

class ContinuousHopfieldOpImpl : public GraphOp {
public:
    int64_t dim;
    int64_t num_basins;
    torch::Tensor patterns;
    ContinuousHopfieldOpImpl(int64_t dim, std::string device_str, int64_t num_basins = 16)
        : dim(dim), num_basins(num_basins) {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        patterns = register_parameter("patterns", torch::randn({num_basins, dim}, torch::TensorOptions().device(device)) * (1.0f / std::sqrt((float)dim)));
        this->to(device);
    }
    torch::Tensor forward(torch::Tensor x) override {
        auto norm_pat = torch::nn::functional::normalize(patterns, torch::nn::functional::NormalizeFuncOptions().dim(-1));
        auto scores = torch::matmul(x, norm_pat.t()) * 8.0f;
        auto attn = torch::softmax(scores, -1);
        return torch::matmul(attn, norm_pat);
    }
};

class StateSpaceMemoryOpImpl : public GraphOp {
public:
    int64_t dim;
    torch::Tensor decay;
    torch::Tensor w_proj;
    StateSpaceMemoryOpImpl(int64_t dim, std::string device_str) : dim(dim) {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        decay = register_parameter("decay", torch::full({dim}, -1.0f, torch::TensorOptions().device(device)));
        w_proj = register_parameter("w_proj", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * (1.0f / std::sqrt((float)dim)));
        this->to(device);
    }
    torch::Tensor forward(torch::Tensor x) override {
        auto d = torch::sigmoid(decay);
        auto proj = torch::matmul(x, w_proj.t());
        return d * proj;
    }
};

// ============================================================================
// 9. DYNAMIC MORPHIC GRAPH & COMMUTATION ORCHESTRATOR R(h_t)
// ============================================================================
class DynamicMorphicGraphImpl : public torch::nn::Module {
public:
    int64_t dim;
    std::string device_str;
    int64_t k_nodes = 0;
    int64_t total_sprouted_so_far = 0;

    std::vector<std::shared_ptr<GraphOp>> node_ops;
    std::vector<torch::Tensor> alpha_epi;
    std::vector<bool> is_core_node;
    std::vector<std::string> node_names;
    std::vector<std::string> node_types;
    std::vector<float> methylation_locks; // Susumu Ohno Gene Lock: 1.0 = Frozen, 0.0 = Plastic

    torch::Tensor w_route;
    torch::Tensor w_route_ctx; // Dynamic Commutation Orchestrator Matrix [64 * 64, dim]
    torch::Tensor w_sensory_in, w_motor_out;

    DynamicMorphicGraphImpl(int64_t dim = 128, std::string device_str = "cpu")
        : dim(dim), device_str(device_str) {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

        w_route = register_parameter("w_route", torch::zeros({64, 64}, torch::TensorOptions().device(device)));
        w_route_ctx = register_parameter("w_route_ctx", torch::randn({64 * 64, dim}, torch::TensorOptions().device(device)) * 0.01f);
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
        } else if (op_type == "ContinuousHopfield") {
            op = std::make_shared<ContinuousHopfieldOpImpl>(dim, device_str);
        } else if (op_type == "StateSpaceMemory") {
            op = std::make_shared<StateSpaceMemoryOpImpl>(dim, device_str);
        } else {
            op = std::make_shared<LinearAccumulatorOpImpl>(dim, device_str);
        }

        std::string prefix = "node_" + std::to_string(node_ops.size()) + "_" + name;
        for (auto& p : op->named_parameters()) {
            register_parameter(prefix + "." + p.key(), p.value());
        }

        node_ops.push_back(op);
        node_names.push_back(name);
        node_types.push_back(op_type);
        is_core_node.push_back(is_core);
        methylation_locks.push_back(0.0f);

        auto alpha_val = torch::tensor(initial_alpha, torch::TensorOptions().device(device).requires_grad(!is_core));
        alpha_epi.push_back(register_parameter("alpha_" + name, alpha_val));
        k_nodes = node_ops.size();
    }

    void lock_node(int64_t idx, float lock_value = 1.0f) {
        if (idx >= 0 && idx < (int64_t)node_ops.size()) {
            methylation_locks[idx] = lock_value;
            bool freeze = (lock_value >= 1.0f);
            alpha_epi[idx].set_requires_grad(!freeze);
            for (auto& p : node_ops[idx]->named_parameters()) {
                p.value().set_requires_grad(!freeze);
            }
        }
    }

    int64_t duplicate_node(int64_t src_idx, std::string new_name, float initial_alpha = 0.0f) {
        if (src_idx < 0 || src_idx >= (int64_t)node_ops.size()) {
            throw std::runtime_error("Invalid source node index for duplication.");
        }
        std::string op_type = node_types[src_idx];
        add_node(new_name, op_type, false, initial_alpha);
        int64_t dst_idx = node_ops.size() - 1;

        // 1. Exact parameter weight duplication
        auto src_params = node_ops[src_idx]->named_parameters();
        auto dst_params = node_ops[dst_idx]->named_parameters();
        torch::NoGradGuard no_grad;
        for (auto& sp : src_params) {
            for (auto& dp : dst_params) {
                if (sp.key() == dp.key()) {
                    dp.value().copy_(sp.value());
                }
            }
        }

        // 2. Lock parent source node
        lock_node(src_idx, 1.0f);

        // 3. Child node is unlocked (methylation_lock = 0.0f)
        lock_node(dst_idx, 0.0f);

        // 4. Duplicate routing connection profile in w_route
        if (dst_idx < 64 && src_idx < 64) {
            w_route[dst_idx].copy_(w_route[src_idx]);
            w_route.select(1, dst_idx).copy_(w_route.select(1, src_idx));
        }

        return dst_idx;
    }

    std::map<std::string, torch::Tensor> get_active_parameters_map() {
        std::map<std::string, torch::Tensor> params;
        params["w_route"] = w_route;
        params["w_route_ctx"] = w_route_ctx;
        params["w_sensory_in"] = w_sensory_in;
        params["w_motor_out"] = w_motor_out;
        for (size_t i = 0; i < node_ops.size(); ++i) {
            params["alpha_" + node_names[i]] = alpha_epi[i];
            std::string prefix = "node_" + std::to_string(i) + "_" + node_names[i];
            for (auto& p : node_ops[i]->named_parameters()) {
                params[prefix + "." + p.key()] = p.value();
            }
        }
        return params;
    }

    int64_t prune_inactive_nodes(float threshold = 0.02f) {
        int64_t pruned_count = 0;
        torch::NoGradGuard no_grad;
        for (size_t i = 0; i < node_ops.size(); ++i) {
            if (!is_core_node[i] && methylation_locks[i] < 0.5f) {
                float gate = std::abs(std::tanh(alpha_epi[i].item<float>()));
                if (gate < threshold) {
                    alpha_epi[i].zero_();
                    for (auto& p : node_ops[i]->named_parameters()) {
                        p.value().zero_();
                    }
                    pruned_count++;
                }
            }
        }
        return pruned_count;
    }

    torch::Tensor persistent_node_states;
    bool has_persistent_states = false;

    void reset_state() {
        has_persistent_states = false;
        persistent_node_states = torch::Tensor();
    }

    torch::Tensor forward(torch::Tensor x_sensory, int64_t thinking_steps = 4) {
        auto B = x_sensory.size(0);
        int64_t K = k_nodes;
        auto device = x_sensory.device();

        torch::Tensor node_states;
        if (!has_persistent_states || !persistent_node_states.defined() || 
            persistent_node_states.size(0) != K || persistent_node_states.size(1) != B || persistent_node_states.device() != device) {
            node_states = torch::zeros({K, B, dim}, torch::TensorOptions().device(device));
        } else {
            node_states = persistent_node_states;
        }

        auto sensory_in = torch::matmul(x_sensory, w_sensory_in.t());
        node_states[0] = node_states[0] + sensory_in;

        // Context-Dependent Commutation Routing: R(h_t) = Softmax(W_route + W_route_ctx * x_sensory)
        auto delta_route = torch::matmul(x_sensory, w_route_ctx.t()).view({B, 64, 64});
        auto active_w_route = w_route.slice(0, 0, K).slice(1, 0, K);
        auto active_delta = delta_route.slice(1, 0, K).slice(2, 0, K);

        auto dynamic_routing_logits = active_w_route.unsqueeze(0) + active_delta;
        auto routing_matrix = torch::softmax(dynamic_routing_logits, 2);

        for (int64_t step = 0; step < thinking_steps; ++step) {
            auto aggregated_inputs_b = torch::einsum("bik,ibd->bkd", {routing_matrix, node_states});
            auto aggregated_inputs = aggregated_inputs_b.permute({1, 0, 2});
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

        persistent_node_states = node_states.detach();
        has_persistent_states = true;

        auto final_aggregated = node_states.sum(0);
        return torch::matmul(final_aggregated, w_motor_out.t());
    }

    std::vector<std::string> get_topology_manifest() {
        std::vector<std::string> manifest;
        for (int64_t i = 0; i < k_nodes; ++i) {
            std::string status = is_core_node[i] ? "CORE" : (methylation_locks[i] >= 1.0f ? "LOCKED" : "PLASTIC");
            manifest.push_back(node_names[i] + ":" + node_types[i] + ":" + status);
        }
        return manifest;
    }
};
TORCH_MODULE(DynamicMorphicGraph);

// ============================================================================
// 10. PYBIND11 MODULE BINDINGS
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

    py::class_<ContinuousSaccadicDriftImpl, torch::nn::Module, std::shared_ptr<ContinuousSaccadicDriftImpl>>(m, "ContinuousSaccadicDrift")
        .def(py::init<int64_t, int64_t, std::string>(), py::arg("dim") = 128, py::arg("num_filters") = 17, py::arg("device_str") = "cpu")
        .def("forward", &ContinuousSaccadicDriftImpl::forward, py::arg("bump_state"), py::arg("h_core"), py::arg("da_gain") = 0.0f, py::arg("salience_bias") = torch::Tensor())
        .def("__call__", &ContinuousSaccadicDriftImpl::forward, py::arg("bump_state"), py::arg("h_core"), py::arg("da_gain") = 0.0f, py::arg("salience_bias") = torch::Tensor());

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

    py::class_<ContinuousHopfieldOpImpl, torch::nn::Module, std::shared_ptr<ContinuousHopfieldOpImpl>>(m, "ContinuousHopfieldOp")
        .def(py::init<int64_t, int64_t, std::string>(), py::arg("dim"), py::arg("num_basins") = 16, py::arg("device_str") = "cpu")
        .def("forward", &ContinuousHopfieldOpImpl::forward)
        .def("__call__", &ContinuousHopfieldOpImpl::forward);

    py::class_<StateSpaceMemoryOpImpl, torch::nn::Module, std::shared_ptr<StateSpaceMemoryOpImpl>>(m, "StateSpaceMemoryOp")
        .def(py::init<int64_t, std::string>(), py::arg("dim"), py::arg("device_str") = "cpu")
        .def("forward", &StateSpaceMemoryOpImpl::forward)
        .def("__call__", &StateSpaceMemoryOpImpl::forward);

    py::class_<DynamicMorphicGraphImpl, torch::nn::Module, std::shared_ptr<DynamicMorphicGraphImpl>>(m, "DynamicMorphicGraph")
        .def(py::init<int64_t, std::string>(), py::arg("dim") = 128, py::arg("device_str") = "cpu")
        .def_readonly("k_nodes", &DynamicMorphicGraphImpl::k_nodes)
        .def("add_node", &DynamicMorphicGraphImpl::add_node, py::arg("name"), py::arg("op_type"), py::arg("is_core") = false, py::arg("initial_alpha") = 0.0f)
        .def("lock_node", &DynamicMorphicGraphImpl::lock_node, py::arg("idx"), py::arg("lock_value") = 1.0f)
        .def("duplicate_node", &DynamicMorphicGraphImpl::duplicate_node, py::arg("src_idx"), py::arg("new_name"), py::arg("initial_alpha") = 0.0f)
        .def("prune_inactive_nodes", &DynamicMorphicGraphImpl::prune_inactive_nodes, py::arg("threshold") = 0.02f)
        .def("reset_state", &DynamicMorphicGraphImpl::reset_state)
        .def("forward", &DynamicMorphicGraphImpl::forward, py::arg("x_sensory"), py::arg("thinking_steps") = 4)
        .def("__call__", &DynamicMorphicGraphImpl::forward, py::arg("x_sensory"), py::arg("thinking_steps") = 4)
        .def("get_topology_manifest", &DynamicMorphicGraphImpl::get_topology_manifest)
        .def("named_parameters_map", [](std::shared_ptr<DynamicMorphicGraphImpl> m) {
            return m->get_active_parameters_map();
        });
}
