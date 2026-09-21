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
// 2. CAUSAL PARALLEL SSD SCAN OPERATOR (Vectorized Zero-Loop 1D Causal SSM Engine)
// ============================================================================
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

        if (seq_len == 1) {
            auto alpha = torch::sigmoid(log_decay).view({1, 1, dim});
            return (1.0f - alpha) * x;
        }

        auto alpha = torch::sigmoid(log_decay).view({dim, 1, 1});
        auto k = torch::arange(seq_len, torch::TensorOptions().device(device).dtype(x.dtype()));
        auto kernel = torch::flip(torch::pow(alpha, k.view({1, 1, seq_len})), {-1});

        auto x_scaled = (x * (1.0f - alpha.view({1, 1, dim}))).permute({0, 2, 1});
        auto x_pad = torch::nn::functional::pad(x_scaled, torch::nn::functional::PadFuncOptions({seq_len - 1, 0}));
        std::vector<int64_t> stride = {1};
        std::vector<int64_t> padding = {0};
        std::vector<int64_t> dilation = {1};
        auto h = at::conv1d(x_pad, kernel, std::nullopt, stride, padding, dilation, dim);
        
        return h.permute({0, 2, 1});
    }
};
TORCH_MODULE(CausalParallelSSD);


// ============================================================================
// 3. PARALLEL OPERATOR BANK (SwiGLU & Non-Linear Functional Field)
// ============================================================================
class ParallelOperatorBankImpl : public torch::nn::Module {
public:
    int64_t dim;
    int64_t state_dim;
    int64_t num_operators;

    torch::Tensor w_gate;
    torch::Tensor w_up;
    torch::Tensor w_down;

    ParallelOperatorBankImpl(int64_t dim = 256, int64_t state_dim = 128, int64_t num_operators = 8)
        : dim(dim), state_dim(state_dim), num_operators(num_operators) {
        w_gate = register_parameter("w_gate", torch::randn({num_operators, dim, state_dim}) * (1.0f / std::sqrt(dim)));
        w_up = register_parameter("w_up", torch::randn({num_operators, dim, state_dim}) * (1.0f / std::sqrt(dim)));
        w_down = register_parameter("w_down", torch::randn({num_operators, state_dim, dim}) * (1.0f / std::sqrt(state_dim)));
    }

    torch::Tensor compute_operators(torch::Tensor x) {
        auto gate = torch::einsum("bsd,nde->bsne", {x, w_gate});
        auto up = torch::einsum("bsd,nde->bsne", {x, w_up});
        auto act = torch::silu(gate) * up;
        auto out = torch::einsum("bsne,ned->bsnd", {act, w_down});
        return out;
    }
};
TORCH_MODULE(ParallelOperatorBank);


// ============================================================================
// 4. CONTINUOUS HOPFIELD ATTRACTOR NETWORK
// ============================================================================
class ContinuousHopfieldMemoryImpl : public torch::nn::Module {
public:
    int64_t dim;
    int64_t num_basins;
    std::string device_str;

    torch::Tensor basins;
    torch::Tensor scaling;

    ContinuousHopfieldMemoryImpl(int64_t dim = 256, int64_t num_basins = 32, std::string device_str = "cpu")
        : dim(dim), num_basins(num_basins), device_str(device_str) {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

        auto init_basins = torch::randn({num_basins, dim}, torch::TensorOptions().device(device));
        init_basins = init_basins / (init_basins.norm(2, -1, true) + 1e-6f);
        basins = register_parameter("basins", init_basins);

        scaling = register_parameter("scaling", torch::ones({num_basins}, torch::TensorOptions().device(device)) * 12.0f);
        this->to(device);
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor u_t) {
        auto device = x.device();
        auto norm_x = x / (x.norm(2, -1, true) + 1e-6f);

        auto sim = torch::matmul(norm_x, basins.t());

        float da_val = 0.0f;
        if (u_t.defined() && u_t.numel() > 0) {
            da_val = u_t.slice(-1, 5, 6).mean().template item<float>();
        }
        auto beta = scaling * (1.0f + 1.5f * da_val);

        auto energy_weights = torch::softmax(sim * beta.view({1, 1, -1}), -1);
        auto recalled = torch::matmul(energy_weights, basins);
        return recalled;
    }
};
TORCH_MODULE(ContinuousHopfieldMemory);


// ============================================================================
// 5. LATENT ACTIVE INFERENCE PREDICTOR (Free Energy Engine F_t)
// ============================================================================
class LatentPredictorImpl : public torch::nn::Module {
public:
    int64_t dim;
    int64_t latent_dim;

    torch::Tensor w_prior_mu, w_prior_logvar;
    torch::Tensor w_post_mu, w_post_logvar;

    LatentPredictorImpl(int64_t dim = 256, int64_t latent_dim = 64, std::string device_str = "cpu")
        : dim(dim), latent_dim(latent_dim) {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        w_prior_mu = register_parameter("w_prior_mu", torch::randn({dim, latent_dim}, torch::TensorOptions().device(device)) * 0.02f);
        w_prior_logvar = register_parameter("w_prior_logvar", torch::zeros({dim, latent_dim}, torch::TensorOptions().device(device)));
        w_post_mu = register_parameter("w_post_mu", torch::randn({dim * 2, latent_dim}, torch::TensorOptions().device(device)) * 0.02f);
        w_post_logvar = register_parameter("w_post_logvar", torch::zeros({dim * 2, latent_dim}, torch::TensorOptions().device(device)));
        this->to(device);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor h_prev, torch::Tensor h_curr) {
        auto p_mu = torch::matmul(h_prev, w_prior_mu);
        auto p_logvar = torch::matmul(h_prev, w_prior_logvar);

        auto h_cat = torch::cat({h_prev, h_curr}, -1);
        auto q_mu = torch::matmul(h_cat, w_post_mu);
        auto q_logvar = torch::matmul(h_cat, w_post_logvar);

        return std::make_tuple(p_mu, p_logvar, q_mu, q_logvar);
    }
};
TORCH_MODULE(LatentPredictor);


// ============================================================================
// 6. HOMEOSTATIC NEXUS (Ashby Ultrastability Engine)
// ============================================================================
class HomeostaticNexusImpl : public torch::nn::Module {
public:
    std::string device_str;
    std::vector<std::string> dim_names;
    torch::Tensor u_t;
    torch::Tensor decay_rates;

    HomeostaticNexusImpl(std::string device_str = "cpu") : device_str(device_str) {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        dim_names = {"Curiosity", "Energy", "Stability", "Health", "Noradrenaline", "Dopamine"};
        u_t = register_buffer("u_t", torch::tensor({0.8f, 1.0f, 0.9f, 1.0f, 0.2f, 0.1f}, torch::TensorOptions().device(device)));
        decay_rates = register_buffer("decay_rates", torch::tensor({0.001f, 0.0005f, 0.0008f, 0.0002f, 0.01f, 0.01f}, torch::TensorOptions().device(device)));
        this->to(device);
    }

    void sprout_homeostatic_dimension(std::string name, float initial_val = 0.5f, float decay = 0.001f) {
        auto device = u_t.device();
        dim_names.push_back(name);
        u_t = register_buffer("u_t", torch::cat({u_t, torch::tensor({initial_val}, torch::TensorOptions().device(device))}));
        decay_rates = register_buffer("decay_rates", torch::cat({decay_rates, torch::tensor({decay}, torch::TensorOptions().device(device))}));
    }

    torch::Tensor update(float free_energy, float recon_loss) {
        torch::NoGradGuard no_grad;
        auto u = u_t.clone();
        u[0] = torch::clamp(u[0] + 0.1f * free_energy - 0.01f, 0.0f, 1.0f);
        u[1] = torch::clamp(u[1] - 0.005f * (1.0f + free_energy) + 0.002f, 0.05f, 1.0f);
        u[2] = torch::clamp(u[2] - 0.05f * recon_loss + 0.01f, 0.0f, 1.0f);
        u[3] = torch::clamp(u[3] - 0.01f * (1.0f - u[2]) + 0.005f, 0.1f, 1.0f);
        u[4] = torch::clamp(u[4] + 0.2f * free_energy - 0.05f, 0.0f, 1.0f);
        u[5] = torch::clamp(u[5] + 0.1f * (1.0f - recon_loss) - 0.02f, 0.0f, 1.0f);
        u_t.copy_(u);
        return u_t;
    }

    torch::Tensor get_states() { return u_t; }
    std::vector<std::string> get_names() { return dim_names; }
};
TORCH_MODULE(HomeostaticNexus);


// ============================================================================
// 7. EXPANDED PRIMITIVE GRAPH OPERATORS (Zero-Hardcode Mathematical Menu)
// ============================================================================

struct GraphOp : public torch::nn::Module {
    virtual torch::Tensor forward(torch::Tensor x) = 0;
};

// 7.1 Linear Accumulator Primitive
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

// 7.2 Bilinear Multiplicative Primitive
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

// 7.3 Saturated Non-Linear Attractor Primitive
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

// 7.4 Continuous Hopfield Energy Snapping Primitive
struct ContinuousHopfieldOpImpl : public GraphOp {
    int64_t dim;
    int64_t num_basins;
    torch::Tensor basins;
    torch::Tensor beta;

    ContinuousHopfieldOpImpl(int64_t dim, int64_t num_basins = 16, std::string device_str = "cpu")
        : dim(dim), num_basins(num_basins) {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        auto init_basins = torch::randn({num_basins, dim}, torch::TensorOptions().device(device));
        init_basins = init_basins / (init_basins.norm(2, -1, true) + 1e-6f);
        basins = register_parameter("basins", init_basins);
        beta = register_parameter("beta", torch::tensor(12.0f, torch::TensorOptions().device(device)));
    }

    torch::Tensor forward(torch::Tensor x) override {
        auto norm_x = x / (x.norm(2, -1, true) + 1e-6f);
        auto sim = torch::matmul(norm_x, basins.t());
        auto weights = torch::softmax(sim * beta, -1);
        return torch::matmul(weights, basins);
    }
};
TORCH_MODULE(ContinuousHopfieldOp);

// 7.5 State Space Continuous Memory Primitive
struct StateSpaceMemoryOpImpl : public GraphOp {
    int64_t dim;
    torch::Tensor log_decay;
    torch::Tensor w_in, w_out;

    StateSpaceMemoryOpImpl(int64_t dim, std::string device_str = "cpu") : dim(dim) {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        log_decay = register_parameter("log_decay", torch::randn({dim}, torch::TensorOptions().device(device)) * 0.1f - 1.0f);
        w_in = register_parameter("w_in", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * (0.2f / std::sqrt(dim)));
        w_out = register_parameter("w_out", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * (0.2f / std::sqrt(dim)));
    }

    torch::Tensor forward(torch::Tensor x) override {
        auto decay = torch::sigmoid(log_decay);
        auto u = torch::matmul(x, w_in.t());
        auto h = (1.0f - decay) * u;
        return torch::matmul(torch::silu(h), w_out.t());
    }
};
TORCH_MODULE(StateSpaceMemoryOp);


// ============================================================================
// 8. DYNAMIC MORPHIC GRAPH (AGN v6.0 / Sprouting + Apoptosis Engine)
// ============================================================================

class DynamicMorphicGraphImpl : public torch::nn::Module {
public:
    int64_t dim;
    std::string device_str;
    int64_t k_nodes = 0;
    int64_t total_sprouted_so_far = 0; // Persistent monotonic id generator to prevent name collisions

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

        w_route = register_parameter("w_route", torch::zeros({64, 64}, torch::TensorOptions().device(device)));
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
            op = std::make_shared<ContinuousHopfieldOpImpl>(dim, 16, device_str);
        } else if (op_type == "StateSpaceMemory") {
            op = std::make_shared<StateSpaceMemoryOpImpl>(dim, device_str);
        } else {
            throw std::invalid_argument("Unknown operator type: " + op_type);
        }

        // Use a globally unique identifier index to prevent register_module collisions post-apoptosis
        std::string module_key = "op_" + std::to_string(total_sprouted_so_far);
        total_sprouted_so_far++;

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

        if (old_k > 0) {
            torch::NoGradGuard no_grad;
            auto rand_col = torch::randn({old_k}, torch::TensorOptions().device(device)) * (0.1f / std::sqrt(old_k));
            auto rand_row = torch::randn({old_k}, torch::TensorOptions().device(device)) * (0.1f / std::sqrt(old_k));
            w_route.slice(0, 0, old_k).narrow(1, old_k, 1).copy_(rand_col.unsqueeze(1));
            w_route.narrow(0, old_k, 1).slice(1, 0, old_k).copy_(rand_row.unsqueeze(0));
            w_route.index_put_({old_k, old_k}, 0.05f);
        }
    }

    int64_t prune_inactive_nodes(float threshold = 0.02f) {
        std::vector<int64_t> keep_idx;
        for (int64_t i = 0; i < k_nodes; ++i) {
            float alpha_val = std::abs(torch::tanh(alpha_epi[i]).template item<float>());
            if (is_core_node[i] || alpha_val >= threshold) {
                keep_idx.push_back(i);
            }
        }

        int64_t pruned_count = k_nodes - keep_idx.size();
        if (pruned_count == 0) return 0;

        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        torch::NoGradGuard no_grad;

        int64_t new_k = keep_idx.size();
        auto keep_tensor = torch::tensor(keep_idx, torch::TensorOptions().device(device).dtype(torch::kLong));
        auto new_w = w_route.index({keep_tensor, keep_tensor}).clone();

        std::vector<std::shared_ptr<GraphOp>> new_ops;
        std::vector<torch::Tensor> new_alphas;
        std::vector<bool> new_core;
        std::vector<std::string> new_names;
        std::vector<std::string> new_types;

        for (size_t idx = 0; idx < keep_idx.size(); ++idx) {
            int64_t orig = keep_idx[idx];
            new_ops.push_back(node_ops[orig]);
            new_alphas.push_back(alpha_epi[orig]);
            new_core.push_back(is_core_node[orig]);
            new_names.push_back(node_names[orig]);
            new_types.push_back(node_types[orig]);
        }

        w_route.slice(0, 0, new_k).slice(1, 0, new_k).copy_(new_w);
        node_ops = new_ops;
        alpha_epi = new_alphas;
        is_core_node = new_core;
        node_names = new_names;
        node_types = new_types;
        k_nodes = new_k;

        return pruned_count;
    }

    torch::Tensor forward(torch::Tensor x_sensory, int64_t thinking_steps = 4) {
        int64_t B = x_sensory.size(0);
        int64_t K = k_nodes;
        auto device = x_sensory.device();

        auto node_states = torch::zeros({K, B, dim}, torch::TensorOptions().device(device));
        auto sensory_in = torch::matmul(x_sensory, w_sensory_in.t());
        node_states[0] = sensory_in;

        auto active_w_route = w_route.slice(0, 0, K).slice(1, 0, K);

        for (int64_t step = 0; step < thinking_steps; ++step) {
            auto aggregated_inputs = torch::einsum("ij,ibd->jbd", {torch::tanh(active_w_route), node_states});
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
// 9. PYBIND11 MODULE BINDINGS
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
        .def("prune_inactive_nodes", &DynamicMorphicGraphImpl::prune_inactive_nodes, py::arg("threshold") = 0.02f)
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
