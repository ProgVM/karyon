#include <torch/torch.h>
#include <torch/extension.h>
#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <memory>
#include <map>
#include <algorithm>

/**
 * ============================================================================
 * KARYON DYNAMIC MORPHIC GRAPH ENGINE (C++20 LibTorch)
 * ============================================================================
 * Implements:
 * 1. CausalParallelSSD (Continuous linear state space time-mixing)
 * 2. ContinuousSaccadicDrift (C-SSD Soliton Gaze)
 * 3. DynamicMorphicGraph (Adaptive Recurrent Neurogenesis, Susumu Ohno Gene Lock,
 *    and Context-Dependent Softmax Commutation Orchestration R(h_t))
 * ============================================================================
 */

// --- 1. CAUSAL PARALLEL SSD ---
struct CausalParallelSSDImpl : public torch::nn::Module {
    int64_t dim;
    std::string device_str;
    torch::Tensor decay_log;
    torch::Tensor w_in, w_out;

    CausalParallelSSDImpl(int64_t dim, std::string device_str = "cpu", float min_decay = 0.005f, float max_decay = 0.2f)
        : dim(dim), device_str(device_str) {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        
        auto log_min = std::log(min_decay);
        auto log_max = std::log(max_decay);
        decay_log = register_parameter("decay_log", torch::linspace(log_min, log_max, dim, torch::TensorOptions().device(device)));
        
        w_in = register_parameter("w_in", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * (1.0f / std::sqrt((float)dim)));
        w_out = register_parameter("w_out", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * (1.0f / std::sqrt((float)dim)));
        this->to(device);
    }

    torch::Tensor forward(torch::Tensor x) {
        // x: [B, S, D]
        auto B = x.size(0);
        auto S = x.size(1);
        auto D = x.size(2);
        auto device = x.device();

        auto decay = torch::exp(decay_log).clamp(0.001f, 0.999f); // [D]
        auto u = torch::matmul(x, w_in.t()); // [B, S, D]

        // Compute causal SSD parallel decay matrix
        auto indices = torch::arange(S, torch::TensorOptions().device(device)).to(torch::kFloat32);
        auto dist = indices.unsqueeze(1) - indices.unsqueeze(0); // [S, S]
        auto causal_mask = dist >= 0;

        // Log-space decay formulation for numerical stability: exp(-dist * decay)
        auto log_decay = -decay.unsqueeze(0).unsqueeze(0) * dist.unsqueeze(-1); // [S, S, D]
        auto decay_weights = torch::exp(log_decay) * causal_mask.unsqueeze(-1).to(torch::kFloat32); // [S, S, D]

        // Einsum parallel matrix scan: Y[b, s, d] = sum_k (decay_weights[s, k, d] * u[b, k, d])
        auto y = torch::einsum("skd,bkd->bsd", {decay_weights, u});
        return torch::matmul(y, w_out.t());
    }
};
TORCH_MODULE(CausalParallelSSD);


// --- 2. CONTINUOUS SACCADIC DRIFT (C-SSD) ---
struct ContinuousSaccadicDriftImpl : public torch::nn::Module {
    int64_t dim;
    int64_t kernel_size;
    std::string device_str;
    torch::Tensor drift_weight;
    torch::Tensor spread_weight;

    ContinuousSaccadicDriftImpl(int64_t dim, int64_t kernel_size = 17, std::string device_str = "cpu")
        : dim(dim), kernel_size(kernel_size), device_str(device_str) {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        drift_weight = register_parameter("drift_weight", torch::randn({dim, 1}, torch::TensorOptions().device(device)) * 0.05f);
        spread_weight = register_parameter("spread_weight", torch::randn({dim, 1}, torch::TensorOptions().device(device)) * 0.05f);
        this->to(device);
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor bump, torch::Tensor h_core, float default_step = 0.1f, torch::Tensor salience_bias = torch::Tensor()) {
        auto B = bump.size(0);
        auto L = bump.size(1);
        auto device = bump.device();

        auto drift_delta = torch::matmul(h_core, drift_weight).squeeze(-1); // [B]
        auto dynamic_step = (default_step + 0.5f * torch::tanh(drift_delta)).clamp(0.01f, 2.0f); // [B]

        auto spread_delta = torch::matmul(h_core, spread_weight).squeeze(-1); // [B]
        auto sigma = (1.5f + 1.0f * torch::sigmoid(spread_delta)).clamp(0.5f, 5.0f); // [B]

        auto grid = torch::arange(kernel_size, torch::TensorOptions().device(device)).to(torch::kFloat32);
        auto center = (kernel_size - 1) / 2.0f;
        auto relative_offsets = grid - center; // [-K/2, ..., K/2]

        // Shifted gaussian kernel
        auto diff = relative_offsets.unsqueeze(0) - dynamic_step.unsqueeze(1); // [B, K]
        auto kernel = torch::exp(-0.5f * torch::pow(diff / sigma.unsqueeze(1), 2.0f));
        kernel = kernel / (kernel.sum(-1, true) + 1e-6f);

        auto bump_padded = torch::nn::functional::pad(
            bump.unsqueeze(1),
            torch::nn::functional::PadFuncOptions({(kernel_size - 1) / 2, (kernel_size - 1) / 2}).mode(torch::kReplicate)
        );

        auto drifted_bump = torch::zeros_like(bump);
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


// --- 3. DYNAMIC MORPHIC GRAPH OPERATOR PRIMITIVES ---

struct GraphOp : public torch::nn::Module {
    virtual torch::Tensor forward(torch::Tensor x) = 0;
    virtual ~GraphOp() = default;
};

struct LinearAccumulatorOpImpl : public GraphOp {
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

struct BilinearMultiplicativeOpImpl : public GraphOp {
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

struct SaturatedAttractorOpImpl : public GraphOp {
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

struct ContinuousHopfieldOpImpl : public GraphOp {
    int64_t dim;
    int64_t num_patterns;
    torch::Tensor patterns;
    ContinuousHopfieldOpImpl(int64_t dim, std::string device_str, int64_t num_patterns = 32)
        : dim(dim), num_patterns(num_patterns) {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        patterns = register_parameter("patterns", torch::randn({num_patterns, dim}, torch::TensorOptions().device(device)) * (1.0f / std::sqrt((float)dim)));
        this->to(device);
    }
    torch::Tensor forward(torch::Tensor x) override {
        // x: [B, D]
        auto norm_pat = torch::nn::functional::normalize(patterns, torch::nn::functional::NormalizeFuncOptions().dim(-1));
        auto scores = torch::matmul(x, norm_pat.t()) * 8.0f; // [B, P]
        auto attn = torch::softmax(scores, -1);
        return torch::matmul(attn, norm_pat);
    }
};

struct StateSpaceMemoryOpImpl : public GraphOp {
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


// --- 4. DYNAMIC MORPHIC GRAPH WITH SUSUMU OHNO GENE LOCK & COMMUTATION ORCHESTRATOR ---

struct DynamicMorphicGraphImpl : public torch::nn::Module {
    int64_t dim;
    std::string device_str;
    int64_t k_nodes = 0;

    std::vector<std::shared_ptr<GraphOp>> node_ops;
    std::vector<torch::Tensor> alpha_epi;
    std::vector<bool> is_core_node;
    std::vector<std::string> node_names;
    std::vector<std::string> node_types;
    std::vector<float> methylation_locks; // Susumu Ohno Gene Lock: 1.0 = Frozen, 0.0 = Plastic

    torch::Tensor w_route;
    torch::Tensor w_route_ctx; // Commutation Orchestrator Projection: [64 * 64, dim]
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

        // 2. Lock parent source node if not already locked
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

    void reset_states() {
        has_persistent_states = false;
        persistent_node_states = torch::Tensor();
    }

    torch::Tensor forward(torch::Tensor x_sensory, int64_t thinking_steps = 3) {
        // x_sensory: [B, dim]
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

        // Context-dependent dynamic commutation routing: R(h_t) = Softmax(W_route + W_route_ctx * x_sensory)
        // delta_route: [B, 64 * 64] -> reshaped to [B, 64, 64]
        auto delta_route = torch::matmul(x_sensory, w_route_ctx.t()).view({B, 64, 64});
        auto active_w_route = w_route.slice(0, 0, K).slice(1, 0, K); // [K, K]
        auto active_delta = delta_route.slice(1, 0, K).slice(2, 0, K); // [B, K, K]

        // Dynamic routing tensor per batch sample: [B, K, K]
        auto dynamic_routing_logits = active_w_route.unsqueeze(0) + active_delta;
        auto routing_matrix = torch::softmax(dynamic_routing_logits, 2); // Softmax over target nodes (columns)

        for (int64_t step = 0; step < thinking_steps; ++step) {
            // Aggregated inputs across dynamic routing: [B, j, d] -> permuted to [j, B, d]
            // einsum: "bik,ibd->bkd" where i is source node, k is destination node
            auto aggregated_inputs_b = torch::einsum("bik,ibd->bkd", {routing_matrix, node_states});
            auto aggregated_inputs = aggregated_inputs_b.permute({1, 0, 2}); // [K, B, d]
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


// --- PYBIND11 BINDINGS ---
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    torch::python::bind_module<CausalParallelSSDImpl>(m, "CausalParallelSSD")
        .def(py::init<int64_t, std::string, float, float>(),
             py::arg("dim"), py::arg("device_str") = "cpu", py::arg("min_decay") = 0.005f, py::arg("max_decay") = 0.2f)
        .def("forward", &CausalParallelSSDImpl::forward);

    torch::python::bind_module<ContinuousSaccadicDriftImpl>(m, "ContinuousSaccadicDrift")
        .def(py::init<int64_t, int64_t, std::string>(),
             py::arg("dim"), py::arg("kernel_size") = 17, py::arg("device_str") = "cpu")
        .def("forward", &ContinuousSaccadicDriftImpl::forward,
             py::arg("bump"), py::arg("h_core"), py::arg("default_step") = 0.1f, py::arg("salience_bias") = torch::Tensor());

    torch::python::bind_module<DynamicMorphicGraphImpl>(m, "DynamicMorphicGraph")
        .def(py::init<int64_t, std::string>(), py::arg("dim") = 128, py::arg("device_str") = "cpu")
        .def("add_node", &DynamicMorphicGraphImpl::add_node,
             py::arg("name"), py::arg("op_type"), py::arg("is_core") = false, py::arg("initial_alpha") = 0.0f)
        .def("lock_node", &DynamicMorphicGraphImpl::lock_node,
             py::arg("idx"), py::arg("lock_value") = 1.0f)
        .def("duplicate_node", &DynamicMorphicGraphImpl::duplicate_node,
             py::arg("src_idx"), py::arg("new_name"), py::arg("initial_alpha") = 0.0f)
        .def("prune_inactive_nodes", &DynamicMorphicGraphImpl::prune_inactive_nodes,
             py::arg("threshold") = 0.02f)
        .def("reset_states", &DynamicMorphicGraphImpl::reset_states)
        .def("forward", &DynamicMorphicGraphImpl::forward,
             py::arg("x_sensory"), py::arg("thinking_steps") = 3)
        .def_readonly("k_nodes", &DynamicMorphicGraphImpl::k_nodes)
        .def("get_topology_manifest", &DynamicMorphicGraphImpl::get_topology_manifest)
        .def("named_parameters_map", [](std::shared_ptr<DynamicMorphicGraphImpl> m) {
            return m->get_active_parameters_map();
        });
}
