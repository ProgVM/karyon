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
#include <random>

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
    torch::Tensor log_decay;
    torch::Tensor w_k, w_v, w_q, w_out;

    CausalParallelSSDImpl(int64_t dim = 256, std::string device_str = "cpu", float min_decay = 0.005f, float max_decay = 0.2f)
        : dim(dim) {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        
        auto lin = torch::linspace(std::log(min_decay), std::log(max_decay), dim, torch::TensorOptions().device(device));
        log_decay = register_parameter("log_decay", lin);

        w_k = register_parameter("w_k", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * (1.0f / std::sqrt(dim)));
        w_v = register_parameter("w_v", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * (1.0f / std::sqrt(dim)));
        w_q = register_parameter("w_q", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * (1.0f / std::sqrt(dim)));
        w_out = register_parameter("w_out", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * (1.0f / std::sqrt(dim)));
        this->to(device);
    }

    torch::Tensor forward(torch::Tensor x) {
        // x: [B, S, D]
        auto B = x.size(0);
        auto S = x.size(1);
        auto D = x.size(2);

        auto k = torch::matmul(x, w_k.t());
        auto v = torch::matmul(x, w_v.t());
        auto q = torch::matmul(x, w_q.t());

        auto decay = torch::exp(-torch::exp(log_decay)); // [D]

        // Intra-chunk associative scan approximation for continuous stream
        auto kv = k * v; // [B, S, D]
        auto out = torch::zeros_like(x);

        auto state = torch::zeros({B, D}, x.options());
        for (int64_t t = 0; t < S; ++t) {
            state = state * decay.unsqueeze(0) + kv.select(1, t);
            out.select(1, t).copy_(q.select(1, t) * state);
        }

        return torch::matmul(out, w_out.t());
    }
};
TORCH_MODULE(CausalParallelSSD);

// ============================================================================
// 2B. ENDOGENOUS THETA-GAMMA PAC COUPLER (Dual-Scale Chrono-Actuated Engine)
// KEP Principle 22 (Total Endogenous Sovereignty): Direct continuous-time delta_t
// integration and commit gating g_t generated endogenously by the network.
// ============================================================================
class EndogenousThetaGammaPACImpl : public torch::nn::Module {
public:
    int64_t dim;
    int64_t num_hopfield_basins;
    bool use_hopfield_snapping;
    float hopfield_beta;

    // Fast & Slow State-Space Duality Projections
    torch::Tensor w_k_fast, w_v_fast, w_q_fast, w_out_fast, log_decay_fast;
    torch::Tensor w_k_slow, w_v_slow, w_q_slow, w_out_slow, log_decay_slow;
    
    // Direct Chrono-Actuators (Endogenous Time-Warping and Event Gating)
    torch::Tensor w_delta_t;      // [1, dim] -> Softplus(W_dt * h_t)
    torch::Tensor b_delta_t;      // [1]
    torch::Tensor w_commit;       // [1, dim] -> Sigmoid(W_commit * h_t + beta * tanh(F_t))
    torch::Tensor b_commit;       // [1]
    torch::Tensor beta_commit;    // [1]
    
    // Bilinear PAC Coupling & Slow Macro-Compressor
    torch::Tensor w_compress;     // [dim, dim]
    torch::Tensor b_compress;     // [dim]
    torch::Tensor w_bilinear_a;   // [dim, dim]
    torch::Tensor w_bilinear_b;   // [dim, dim]
    torch::Tensor w_pac_out;      // [dim, dim]

    // Modern Continuous Hopfield Attractor Memory on Macro Commits (EXP-313)
    torch::Tensor hopfield_basins; // [num_basins, dim]

    EndogenousThetaGammaPACImpl(int64_t dim = 128, std::string device_str = "cpu",
                                float fast_min_decay = 0.05f, float fast_max_decay = 0.5f,
                                float slow_min_decay = 0.0005f, float slow_max_decay = 0.01f,
                                int64_t num_hopfield_basins = 256,
                                bool use_hopfield_snapping = true,
                                float hopfield_beta = 12.0f)
        : dim(dim), num_hopfield_basins(num_hopfield_basins),
          use_hopfield_snapping(use_hopfield_snapping), hopfield_beta(hopfield_beta) {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        float scale = 1.0f / std::sqrt((float)dim);

        // Fast SSD Parameters (Gamma / Micro-scale)
        auto lin_fast = torch::linspace(std::log(fast_min_decay), std::log(fast_max_decay), dim, torch::TensorOptions().device(device));
        log_decay_fast = register_parameter("log_decay_fast", lin_fast);
        w_k_fast = register_parameter("w_k_fast", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * scale);
        w_v_fast = register_parameter("w_v_fast", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * scale);
        w_q_fast = register_parameter("w_q_fast", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * scale);
        w_out_fast = register_parameter("w_out_fast", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * scale);

        // Slow SSD Parameters (Theta / Macro-scale)
        auto lin_slow = torch::linspace(std::log(slow_min_decay), std::log(slow_max_decay), dim, torch::TensorOptions().device(device));
        log_decay_slow = register_parameter("log_decay_slow", lin_slow);
        w_k_slow = register_parameter("w_k_slow", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * scale);
        w_v_slow = register_parameter("w_v_slow", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * scale);
        w_q_slow = register_parameter("w_q_slow", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * scale);
        w_out_slow = register_parameter("w_out_slow", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * scale);

        // Direct Chrono-Actuators
        w_delta_t = register_parameter("w_delta_t", torch::randn({1, dim}, torch::TensorOptions().device(device)) * scale);
        b_delta_t = register_parameter("b_delta_t", torch::tensor({0.5413f}, torch::TensorOptions().device(device))); // softplus(0.5413) ≈ 1.0
        w_commit = register_parameter("w_commit", torch::randn({1, dim}, torch::TensorOptions().device(device)) * scale);
        b_commit = register_parameter("b_commit", torch::tensor({-2.0f}, torch::TensorOptions().device(device))); // default closed gate
        beta_commit = register_parameter("beta_commit", torch::tensor({1.5f}, torch::TensorOptions().device(device)));

        // Bilinear Coupling and Macro Compression
        w_compress = register_parameter("w_compress", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * scale);
        b_compress = register_parameter("b_compress", torch::zeros({dim}, torch::TensorOptions().device(device)));
        w_bilinear_a = register_parameter("w_bilinear_a", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * scale);
        w_bilinear_b = register_parameter("w_bilinear_b", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * scale);
        w_pac_out = register_parameter("w_pac_out", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * scale);

        // Continuous Hopfield Memory Attractor Basins: [num_basins, dim] normalized
        auto raw_basins = torch::randn({num_hopfield_basins, dim}, torch::TensorOptions().device(device)) * scale;
        hopfield_basins = register_parameter("hopfield_basins", torch::nn::functional::normalize(raw_basins, torch::nn::functional::NormalizeFuncOptions().dim(-1)));

        this->to(device);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward(
        torch::Tensor x,
        torch::Tensor free_energy = torch::Tensor(),
        torch::Tensor init_h_fast = torch::Tensor(),
        torch::Tensor init_h_slow = torch::Tensor()) {
        // x: [B, S, D]
        auto B = x.size(0);
        auto S = x.size(1);
        auto D = x.size(2);
        auto dev = x.device();
        auto opts = x.options();

        torch::Tensor f_t_seq;
        if (free_energy.defined() && free_energy.numel() > 0) {
            if (free_energy.dim() == 1) {
                f_t_seq = free_energy.unsqueeze(0).unsqueeze(-1).expand({B, S, 1});
            } else if (free_energy.dim() == 2) {
                f_t_seq = free_energy.unsqueeze(-1);
            } else {
                f_t_seq = free_energy;
            }
        } else {
            f_t_seq = torch::zeros({B, S, 1}, opts);
        }

        // Fast & Slow Projections across Sequence
        auto k_fast = torch::matmul(x, w_k_fast.t());
        auto v_fast = torch::matmul(x, w_v_fast.t());
        auto q_fast = torch::matmul(x, w_q_fast.t());
        auto kv_fast = k_fast * v_fast; // [B, S, D]

        // Base continuous continuous matrix decay rates A
        auto base_a_fast = torch::exp(log_decay_fast); // [D]
        auto base_a_slow = torch::exp(log_decay_slow); // [D]

        // Normalize Hopfield Basins onto unit sphere: [num_basins, D]
        auto norm_basins = torch::nn::functional::normalize(hopfield_basins, torch::nn::functional::NormalizeFuncOptions().dim(-1));

        // Allocate state & telemetry trace buffers
        torch::Tensor state_fast = (init_h_fast.defined() && init_h_fast.sizes() == torch::IntArrayRef({B, D})) ? init_h_fast : torch::zeros({B, D}, opts);
        torch::Tensor state_slow = (init_h_slow.defined() && init_h_slow.sizes() == torch::IntArrayRef({B, D})) ? init_h_slow : torch::zeros({B, D}, opts);

        auto delta_t_trace = torch::zeros({B, S, 1}, opts);
        auto commit_gate_trace = torch::zeros({B, S, 1}, opts);
        auto hopfield_snapped_trace = torch::zeros({B, S, D}, opts);
        auto y_out = torch::zeros_like(x);

        for (int64_t t = 0; t < S; ++t) {
            auto x_t = x.select(1, t); // [B, D]
            auto fe_t = f_t_seq.select(1, t); // [B, 1]

            // 1. Direct Time-Speed Actuator: Delta t_t = Softplus(W_dt * h_t + b_dt)
            auto dt_logits = torch::matmul(x_t, w_delta_t.t()) + b_delta_t; // [B, 1]
            auto dt_t = torch::softplus(dt_logits) + 0.05f; // ensure minimal physical causality dt >= 0.05
            delta_t_trace.select(1, t).copy_(dt_t);

            // 2. Continuous time-warped state transition: decay = exp(-A * delta_t_t)
            auto decay_fast_t = torch::exp(-base_a_fast.unsqueeze(0) * dt_t); // [B, D]
            state_fast = state_fast * decay_fast_t + kv_fast.select(1, t) * dt_t;

            auto h_fast_t = torch::matmul(q_fast.select(1, t) * state_fast, w_out_fast.t()); // [B, D]

            // 3. Direct Macro Commit Valve: g_t = Sigmoid(W_commit * h_fast + b_commit + beta * tanh(F_t))
            auto commit_logits = torch::matmul(h_fast_t, w_commit.t()) + b_commit + beta_commit * torch::tanh(fe_t);
            auto g_t = torch::sigmoid(commit_logits); // [B, 1]
            commit_gate_trace.select(1, t).copy_(g_t);

            // 4. Macro-Compression & Slow State Transition
            auto fast_compressed = torch::tanh(torch::matmul(h_fast_t, w_compress.t()) + b_compress); // [B, D]

            // 4a. Continuous Hopfield Attractor Snapping on Macro-Commits (Wave-Particle Collapse)
            torch::Tensor slow_input = fast_compressed;
            if (use_hopfield_snapping) {
                // h_snapped = Softmax(beta_hop * (fast_compressed * K^T)) * V
                // [B, D] x [D, num_basins] -> [B, num_basins]
                auto norm_fast = torch::nn::functional::normalize(fast_compressed, torch::nn::functional::NormalizeFuncOptions().dim(-1));
                auto hopfield_scores = torch::matmul(norm_fast, norm_basins.t()) * hopfield_beta;
                auto hopfield_attn = torch::softmax(hopfield_scores, -1); // [B, num_basins]
                auto h_snapped = torch::matmul(hopfield_attn, norm_basins); // [B, D]
                slow_input = h_snapped;
                hopfield_snapped_trace.select(1, t).copy_(h_snapped);
            } else {
                hopfield_snapped_trace.select(1, t).copy_(fast_compressed);
            }

            auto k_slow_t = torch::matmul(slow_input, w_k_slow.t());
            auto v_slow_t = torch::matmul(slow_input, w_v_slow.t());
            auto q_slow_t = torch::matmul(x_t, w_q_slow.t());
            auto kv_slow_t = k_slow_t * v_slow_t;

            // Slow decay also integrated with endogenous time, but updated through commit gate g_t
            auto decay_slow_t = torch::exp(-base_a_slow.unsqueeze(0) * dt_t);
            auto state_slow_candidate = state_slow * decay_slow_t + kv_slow_t;
            // Gated commit: state_slow updates only when g_t opens!
            state_slow = (1.0f - g_t) * state_slow + g_t * state_slow_candidate;

            auto h_slow_t = torch::matmul(q_slow_t * state_slow, w_out_slow.t()); // [B, D]

            // 5. Bilinear PAC Modulation: Fast micro-dynamics modulated by Slow macro-phase
            auto mod_a = torch::matmul(h_slow_t, w_bilinear_a.t());
            auto mod_b = torch::sigmoid(torch::matmul(h_slow_t, w_bilinear_b.t()));
            auto pac_phase_mod = mod_a * mod_b; // [B, D]

            auto y_t = h_fast_t * (1.0f + pac_phase_mod);
            y_out.select(1, t).copy_(torch::matmul(y_t, w_pac_out.t()));
        }

        return std::make_tuple(y_out, delta_t_trace, commit_gate_trace, state_fast, state_slow, hopfield_snapped_trace);
    }
};
TORCH_MODULE(EndogenousThetaGammaPAC);

// ============================================================================
// 3. PARALLEL OPERATOR BANK
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
        auto norm_basins = torch::nn::functional::normalize(basins, torch::nn::functional::NormalizeFuncOptions().dim(-1));
        float beta = 8.0f;
        if (u_t.defined() && u_t.numel() >= 6) {
            float da = u_t.select(0, 5).item<float>();
            beta = beta * (1.0f + 1.5f * da);
        }

        auto scores = torch::matmul(x, norm_basins.t()) * beta;
        auto attn = torch::softmax(scores, -1);
        return torch::matmul(attn, norm_basins);
    }
};
TORCH_MODULE(ContinuousHopfieldMemory);

// ============================================================================
// 5. CONTINUOUS SACCADIC ATTRACTOR DRIFT (C-SSD Engine)
// ============================================================================
class ContinuousSaccadicDriftImpl : public torch::nn::Module {
public:
    int64_t dim;
    int64_t num_filters;
    torch::Tensor conv_weights;
    torch::Tensor w_drift;

    ContinuousSaccadicDriftImpl(int64_t dim = 128, int64_t num_filters = 17, std::string device_str = "cpu")
        : dim(dim), num_filters(num_filters) {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        conv_weights = register_parameter("conv_weights", torch::randn({num_filters}, torch::TensorOptions().device(device)) * 0.1f);
        w_drift = register_parameter("w_drift", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * (1.0f / std::sqrt(dim)));
        this->to(device);
    }

    torch::Tensor forward(torch::Tensor bump_state, torch::Tensor h_core, float da_gain = 0.0f, torch::Tensor salience_bias = torch::Tensor()) {
        auto B = bump_state.size(0);
        auto S = bump_state.size(1);

        auto padded = torch::nn::functional::pad(bump_state.unsqueeze(1), torch::nn::functional::PadFuncOptions({num_filters / 2, num_filters / 2}).mode(torch::kCircular));
        auto weights = conv_weights.view({1, 1, num_filters});
        auto filtered = torch::conv1d(padded, weights).squeeze(1);

        auto drift = torch::matmul(h_core, w_drift.t()).mean(-1, true); // [B, 1]
        if (salience_bias.defined() && salience_bias.numel() > 0) {
            drift = drift + salience_bias;
        }

        auto new_bump = filtered + (1.0f + 1.2f * da_gain) * drift;
        return torch::softmax(new_bump, -1);
    }
};
TORCH_MODULE(ContinuousSaccadicDrift);

// ============================================================================
// 6. ACTIVE INFERENCE LATENT PREDICTOR (Free Energy Engine F_t)
// ============================================================================
class LatentPredictorImpl : public torch::nn::Module {
public:
    int64_t dim;
    int64_t latent_dim;
    torch::Tensor w_mu_prior, w_logvar_prior;
    torch::Tensor w_mu_post, w_logvar_post;

    LatentPredictorImpl(int64_t dim = 256, int64_t latent_dim = 64, std::string device = "cpu")
        : dim(dim), latent_dim(latent_dim) {
        auto dev = device.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        w_mu_prior = register_parameter("w_mu_prior", torch::randn({latent_dim, dim}, torch::TensorOptions().device(dev)) * (1.0f / std::sqrt(dim)));
        w_logvar_prior = register_parameter("w_logvar_prior", torch::zeros({latent_dim, dim}, torch::TensorOptions().device(dev)));
        w_mu_post = register_parameter("w_mu_post", torch::randn({latent_dim, dim}, torch::TensorOptions().device(dev)) * (1.0f / std::sqrt(dim)));
        w_logvar_post = register_parameter("w_logvar_post", torch::zeros({latent_dim, dim}, torch::TensorOptions().device(dev)));
        this->to(dev);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor h) {
        auto mu_prior = torch::matmul(h, w_mu_prior.t());
        auto logvar_prior = torch::clamp(torch::matmul(h, w_logvar_prior.t()), -10.0f, 2.0f);
        auto mu_post = torch::matmul(h, w_mu_post.t());
        auto logvar_post = torch::clamp(torch::matmul(h, w_logvar_post.t()), -10.0f, 2.0f);
        return std::make_tuple(mu_prior, logvar_prior, mu_post, logvar_post);
    }
};
TORCH_MODULE(LatentPredictor);

// ============================================================================
// 7. HOMEOSTATIC NEXUS (Allostatic Dynamic Regulation)
// ============================================================================
class HomeostaticNexusImpl : public torch::nn::Module {
public:
    torch::Tensor states; // [6] or dynamically expanded
    std::vector<std::string> state_names;
    torch::Tensor setpoints;
    torch::Tensor recovery_rates;
    torch::Tensor sensitivity;

    HomeostaticNexusImpl(std::string device_str = "cpu") {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        state_names = {"Curiosity", "Energy", "Stability", "Health", "Noradrenaline", "Dopamine"};
        states = register_buffer("states", torch::tensor({0.8f, 1.0f, 0.9f, 1.0f, 0.1f, 0.1f}, torch::TensorOptions().device(device)));
        setpoints = register_buffer("setpoints", torch::tensor({0.5f, 1.0f, 0.8f, 1.0f, 0.05f, 0.05f}, torch::TensorOptions().device(device)));
        recovery_rates = register_buffer("recovery_rates", torch::tensor({0.01f, 0.005f, 0.02f, 0.001f, 0.05f, 0.05f}, torch::TensorOptions().device(device)));
        sensitivity = register_buffer("sensitivity", torch::tensor({0.1f, 0.05f, 0.2f, 0.01f, 0.3f, 0.2f}, torch::TensorOptions().device(device)));
        this->to(device);
    }

    void sprout_homeostatic_dimension(std::string name, float init_val, float setpoint, float rec_rate, float sens) {
        auto dev = states.device();
        state_names.push_back(name);
        states = torch::cat({states, torch::tensor({init_val}, torch::TensorOptions().device(dev))});
        setpoints = torch::cat({setpoints, torch::tensor({setpoint}, torch::TensorOptions().device(dev))});
        recovery_rates = torch::cat({recovery_rates, torch::tensor({rec_rate}, torch::TensorOptions().device(dev))});
        sensitivity = torch::cat({sensitivity, torch::tensor({sens}, torch::TensorOptions().device(dev))});
    }

    void update(float free_energy_surprise) {
        torch::NoGradGuard no_grad;
        auto error = free_energy_surprise;
        // Noradrenaline arousal rises with surprise
        states[4] = torch::clamp(states[4] + sensitivity[4] * error - recovery_rates[4] * (states[4] - setpoints[4]), 0.0f, 2.0f);
        // Energy depletes with surprise
        states[1] = torch::clamp(states[1] - sensitivity[1] * error + recovery_rates[1] * (setpoints[1] - states[1]), 0.0f, 1.0f);
        // Stability decreases with large surprise
        states[2] = torch::clamp(states[2] - sensitivity[2] * (error > 1.0f ? error : 0.0f) + recovery_rates[2] * (setpoints[2] - states[2]), 0.0f, 1.0f);
    }

    torch::Tensor get_states() { return states; }
    std::vector<std::string> get_names() { return state_names; }
};
TORCH_MODULE(HomeostaticNexus);

// ============================================================================
// HARDWARE ENTROPY SAMPLER UTILITY (Thread-Safe Chunk-Buffered Box-Muller)
// Refills in 4096-element batches from std::random_device to eradicate syscall overhead
// ============================================================================
class HardwareEntropyBuffer {
private:
    static constexpr size_t BUFFER_SIZE = 4096;
    std::vector<float> pool;
    size_t cursor = BUFFER_SIZE; // force initial refill

    void refill() {
        std::random_device rd;
        pool.resize(BUFFER_SIZE);
        for (size_t i = 0; i < BUFFER_SIZE; i += 2) {
            uint32_t r1 = rd();
            uint32_t r2 = rd();
            double u1 = (static_cast<double>(r1) + 1.0) / (static_cast<double>(UINT32_MAX) + 2.0);
            double u2 = (static_cast<double>(r2) + 1.0) / (static_cast<double>(UINT32_MAX) + 2.0);
            double mag = std::sqrt(-2.0 * std::log(u1));
            double angle = 2.0 * M_PI * u2;
            pool[i] = static_cast<float>(mag * std::cos(angle));
            if (i + 1 < BUFFER_SIZE) {
                pool[i + 1] = static_cast<float>(mag * std::sin(angle));
            }
        }
        cursor = 0;
    }

public:
    void sample(float* out, size_t n) {
        size_t written = 0;
        while (written < n) {
            if (cursor >= BUFFER_SIZE) {
                refill();
            }
            size_t available = BUFFER_SIZE - cursor;
            size_t to_copy = std::min(available, n - written);
            std::memcpy(out + written, pool.data() + cursor, to_copy * sizeof(float));
            cursor += to_copy;
            written += to_copy;
        }
    }
};

inline torch::Tensor sample_hardware_gaussian_entropy(c10::IntArrayRef shape, torch::Device device) {
    static thread_local HardwareEntropyBuffer entropy_buf;
    int64_t total_elements = 1;
    for (auto s : shape) total_elements *= s;

    auto cpu_tensor = torch::empty(shape, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    float* ptr = cpu_tensor.data_ptr<float>();
    entropy_buf.sample(ptr, static_cast<size_t>(total_elements));

    return cpu_tensor.to(device);
}

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
// STOCHASTIC LANGEVIN OP (5th Atomic Operator with Hardware System Entropy)
// y_t = W_drift * x_t + sigma_eff * xi_t^hardware
// sigma_eff = Softplus(W_sigma * h_t) * (1.0 + gamma * tanh(F_t))
// ============================================================================
class StochasticLangevinOpImpl : public GraphOp {
public:
    int64_t dim;
    float gamma;
    float current_free_energy;
    std::string device_str;
    torch::Tensor w_drift;
    torch::Tensor w_sigma;

    StochasticLangevinOpImpl(int64_t dim, std::string device_str, float gamma = 1.0f)
        : dim(dim), gamma(gamma), current_free_energy(0.0f), device_str(device_str) {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        w_drift = register_parameter("w_drift", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * (1.0f / std::sqrt((float)dim)));
        w_sigma = register_parameter("w_sigma", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * 0.05f);
        this->to(device);
    }

    void set_free_energy(float f_t) {
        current_free_energy = f_t;
    }

    torch::Tensor forward(torch::Tensor x) override {
        return forward_with_fe(x, current_free_energy);
    }

    torch::Tensor forward_with_fe(torch::Tensor x, float f_t) {
        // 1. Drift term: W_drift * x
        auto drift = torch::matmul(x, w_drift.t());

        // 2. Endogenous noise scale: sigma_eff = Softplus(W_sigma * x) * (1.0 + gamma * tanh(F_t))
        auto raw_sigma = torch::nn::functional::softplus(torch::matmul(x, w_sigma.t()));
        float somatic_stress_factor = 1.0f + gamma * std::tanh(f_t);
        auto sigma_eff = raw_sigma * somatic_stress_factor;

        // 3. Hardware Gaussian entropy tensor directly sampled from OS random_device
        auto xi_hardware = sample_hardware_gaussian_entropy(x.sizes(), x.device());

        // 4. Langevin Stochastic Integration
        return drift + sigma_eff * xi_hardware;
    }

    torch::Tensor forward_deterministic(torch::Tensor x) {
        // Forced deterministic projection for verification (sigma = 0)
        return torch::matmul(x, w_drift.t());
    }
};

// ============================================================================
// PROGRAMMABLE DELAY OP (6th Atomic Operator - Axonal Delay Line Buffer)
// Circular buffer holding historical states [tau_max, B, dim].
// Dynamic delay tau_t = 1 + (tau_max - 1) * sigmoid(w_tau * h_t).
// Differentiable continuous reading via linear interpolation between floor(tau) and ceil(tau).
// ============================================================================
class ProgrammableDelayOpImpl : public GraphOp {
public:
    int64_t dim;
    int64_t tau_max;
    std::string device_str;
    torch::Tensor w_tau;
    torch::Tensor ring_buffer;
    int64_t buffer_ptr = 0;
    bool buffer_initialized = false;

    ProgrammableDelayOpImpl(int64_t dim, std::string device_str, int64_t tau_max = 16)
        : dim(dim), tau_max(tau_max), device_str(device_str) {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        w_tau = register_parameter("w_tau", torch::randn({1, dim}, torch::TensorOptions().device(device)) * (1.0f / std::sqrt((float)dim)));
        this->to(device);
    }

    void reset_buffer() {
        buffer_initialized = false;
        buffer_ptr = 0;
        ring_buffer = torch::Tensor();
    }

    torch::Tensor forward(torch::Tensor x) override {
        auto B = x.size(0);
        auto device = x.device();

        // 1. Initialize ring buffer [tau_max, B, dim] if uninitialized or batch mismatch
        if (!buffer_initialized || !ring_buffer.defined() || ring_buffer.size(1) != B || ring_buffer.device() != device) {
            ring_buffer = torch::zeros({tau_max, B, dim}, torch::TensorOptions().device(device));
            buffer_ptr = 0;
            buffer_initialized = true;
        }

        // 2. Store current incoming state at buffer_ptr
        ring_buffer[buffer_ptr] = x.detach();

        // 3. Compute dynamic delay tau_t in range [1.0, tau_max]
        // tau_t = 1.0 + (tau_max - 1.0) * sigmoid(x * w_tau^T) -> [B, 1]
        auto sig_tau = torch::sigmoid(torch::matmul(x, w_tau.t())); // [B, 1]
        auto tau_t = 1.0f + (static_cast<float>(tau_max) - 1.0f) * sig_tau; // [B, 1]

        // 4. Circular buffer index lookup:
        // read_idx = buffer_ptr - tau_t (modulo tau_max)
        auto tau_flat = tau_t.squeeze(-1); // [B]
        auto tau_floor = tau_flat.floor(); // [B]
        auto tau_frac = (tau_flat - tau_floor).unsqueeze(-1); // [B, 1]

        auto curr_ptr_tensor = torch::full({B}, static_cast<float>(buffer_ptr), torch::TensorOptions().device(device));
        auto idx0 = torch::remainder(curr_ptr_tensor - tau_floor + static_cast<float>(tau_max * 2), static_cast<float>(tau_max)).to(torch::kLong);
        auto idx1 = torch::remainder(idx0 - 1 + static_cast<float>(tau_max), static_cast<float>(tau_max)).to(torch::kLong);

        // Gather states for each batch element
        // ring_buffer is [tau_max, B, dim], permute to [B, tau_max, dim]
        auto buf_perm = ring_buffer.permute({1, 0, 2}); // [B, tau_max, dim]
        auto b_indices = torch::arange(B, torch::TensorOptions().device(device));

        auto state0 = buf_perm.index({b_indices, idx0}); // [B, dim]
        auto state1 = buf_perm.index({b_indices, idx1}); // [B, dim]

        // Linear interpolation across continuous delay
        auto delayed_out = (1.0f - tau_frac) * state0 + tau_frac * state1;

        // Advance circular buffer pointer
        buffer_ptr = (buffer_ptr + 1) % tau_max;

        return delayed_out;
    }

    torch::Tensor forward_fixed_delay(torch::Tensor x, int64_t fixed_tau) {
        auto B = x.size(0);
        auto device = x.device();
        if (!buffer_initialized || !ring_buffer.defined() || ring_buffer.size(1) != B || ring_buffer.device() != device) {
            ring_buffer = torch::zeros({tau_max, B, dim}, torch::TensorOptions().device(device));
            buffer_ptr = 0;
            buffer_initialized = true;
        }

        ring_buffer[buffer_ptr] = x.detach();
        int64_t lookback = std::clamp(fixed_tau, (int64_t)0, tau_max - 1);
        int64_t read_idx = (buffer_ptr - lookback + tau_max) % tau_max;
        auto delayed_out = ring_buffer[read_idx].clone();

        buffer_ptr = (buffer_ptr + 1) % tau_max;
        return delayed_out;
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

    torch::Tensor w_halt;

    DynamicMorphicGraphImpl(int64_t dim = 128, std::string device_str = "cpu")
        : dim(dim), device_str(device_str) {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        w_route = register_parameter("w_route", torch::zeros({64, 64}, torch::TensorOptions().device(device)));
        w_route_ctx = register_parameter("w_route_ctx", torch::randn({64 * 64, dim}, torch::TensorOptions().device(device)) * (0.01f / std::sqrt((float)dim)));
        w_sensory_in = register_parameter("w_sensory_in", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * 0.2f);
        w_motor_out = register_parameter("w_motor_out", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * 0.2f);
        w_halt = register_parameter("w_halt", torch::randn({1, dim}, torch::TensorOptions().device(device)) * (1.0f / std::sqrt((float)dim)));
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
        } else if (op_type == "StochasticLangevin") {
            op = std::make_shared<StochasticLangevinOpImpl>(dim, device_str);
        } else if (op_type == "ProgrammableDelay") {
            op = std::make_shared<ProgrammableDelayOpImpl>(dim, device_str, 16);
        } else {
            op = std::make_shared<LinearAccumulatorOpImpl>(dim, device_str);
        }

        std::string prefix = "node_" + std::to_string(node_ops.size()) + "_" + name;
        for (auto& p : op->named_parameters()) {
            register_parameter(prefix + "_" + p.key(), p.value());
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
    std::vector<float> get_methylation_locks() const {
        return methylation_locks;
    }
    void set_methylation_locks(const std::vector<float>& locks) {
        for (size_t i = 0; i < locks.size() && i < methylation_locks.size(); ++i) {
            lock_node(i, locks[i]);
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

        // 3. Sprout with zero-shock identity (alpha_epi = initial_alpha)
        alpha_epi[dst_idx].copy_(torch::tensor(initial_alpha, alpha_epi[dst_idx].options()));

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
        params["w_halt"] = w_halt;
        for (size_t i = 0; i < node_ops.size(); ++i) {
            params["alpha_" + node_names[i]] = alpha_epi[i];
            std::string prefix = "node_" + std::to_string(i) + "_" + node_names[i];
            for (auto& p : node_ops[i]->named_parameters()) {
                params[prefix + "_" + p.key()] = p.value();
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

    std::tuple<torch::Tensor, float> forward_adaptive(torch::Tensor x_sensory, int64_t max_thinking_steps = 8, float halt_threshold = 0.8f, float epsilon_halt = 1e-3f) {
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

        auto delta_route = torch::matmul(x_sensory, w_route_ctx.t()).view({B, 64, 64});
        auto active_w_route = w_route.slice(0, 0, K).slice(1, 0, K);
        auto active_delta = delta_route.slice(1, 0, K).slice(2, 0, K);

        auto dynamic_routing_logits = active_w_route.unsqueeze(0) + active_delta;
        
        std::vector<torch::Tensor> gate_factors;
        for (int64_t j = 0; j < K; ++j) {
            gate_factors.push_back(torch::tanh(alpha_epi[j]).abs());
        }
        auto gates_tensor = torch::stack(gate_factors, 0);
        auto inactive_mask = (gates_tensor < 1e-4f).unsqueeze(0).unsqueeze(0);
        dynamic_routing_logits = dynamic_routing_logits.masked_fill(inactive_mask, -1e4f);

        auto routing_matrix = torch::softmax(dynamic_routing_logits, 2);

        int64_t actual_steps_taken = 0;
        torch::Tensor prev_aggregated;

        for (int64_t step = 0; step < max_thinking_steps; ++step) {
            actual_steps_taken++;
            auto node_states_b = node_states.permute({1, 0, 2});
            auto aggregated_inputs_b = torch::einsum("bik,bid->bkd", {routing_matrix, node_states_b});
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

            // Adaptive Pondering Check
            torch::Tensor curr_aggregated = torch::zeros({B, dim}, torch::TensorOptions().device(device));
            for (int64_t j = 0; j < K; ++j) {
                curr_aggregated = curr_aggregated + torch::tanh(alpha_epi[j]) * node_states[j];
            }

            // Halting probability p_halt = sigmoid(w_halt * curr_aggregated)
            auto p_halt = torch::sigmoid(torch::matmul(curr_aggregated, w_halt.t())).mean().item<float>();

            float delta_f = 1.0f;
            if (prev_aggregated.defined()) {
                delta_f = (curr_aggregated - prev_aggregated).norm().item<float>() / (static_cast<float>(B * dim) + 1e-6f);
            }
            prev_aggregated = curr_aggregated;

            // Early exit if halting probability exceeded or state trajectory stabilized
            if (step >= 1 && (p_halt > halt_threshold || delta_f < epsilon_halt)) {
                break;
            }
        }

        persistent_node_states = node_states.detach();
        has_persistent_states = true;

        torch::Tensor final_aggregated = torch::zeros({B, dim}, torch::TensorOptions().device(device));
        for (int64_t j = 0; j < K; ++j) {
            auto gate = torch::tanh(alpha_epi[j]);
            final_aggregated = final_aggregated + gate * node_states[j];
        }
        auto out = torch::matmul(final_aggregated, w_motor_out.t());
        return std::make_tuple(out, static_cast<float>(actual_steps_taken));
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
        // routing_matrix shape: [B, K_source, K_target] where Softmax is computed along K_target dimension
        auto delta_route = torch::matmul(x_sensory, w_route_ctx.t()).view({B, 64, 64});
        auto active_w_route = w_route.slice(0, 0, K).slice(1, 0, K);
        auto active_delta = delta_route.slice(1, 0, K).slice(2, 0, K);

        auto dynamic_routing_logits = active_w_route.unsqueeze(0) + active_delta; // [B, K_src, K_tgt]
        
        std::vector<torch::Tensor> gate_factors;
        for (int64_t j = 0; j < K; ++j) {
            gate_factors.push_back(torch::tanh(alpha_epi[j]).abs());
        }
        auto gates_tensor = torch::stack(gate_factors, 0); // [K]
        auto inactive_mask = (gates_tensor < 1e-4f).unsqueeze(0).unsqueeze(0); // [1, 1, K_tgt]
        dynamic_routing_logits = dynamic_routing_logits.masked_fill(inactive_mask, -1e4f);

        auto routing_matrix = torch::softmax(dynamic_routing_logits, 2); // Softmax across target nodes

        for (int64_t step = 0; step < thinking_steps; ++step) {
            // Correct Tensor Contraction for Routing:
            // routing_matrix: [B, i_src, k_tgt]
            // node_states: [i_src, B, d] -> permuted to [B, i_src, d]
            // Output aggregated_inputs_b: [B, k_tgt, d]
            auto node_states_b = node_states.permute({1, 0, 2}); // [B, i_src, d]
            auto aggregated_inputs_b = torch::einsum("bik,bid->bkd", {routing_matrix, node_states_b});
            auto aggregated_inputs = aggregated_inputs_b.permute({1, 0, 2}); // [k_tgt, B, d]
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

        // Final output is weighted by epigenetic gates to ensure exact Net2Net zero-shock when alpha=0.0
        torch::Tensor final_aggregated = torch::zeros({B, dim}, torch::TensorOptions().device(device));
        for (int64_t j = 0; j < K; ++j) {
            auto gate = torch::tanh(alpha_epi[j]);
            final_aggregated = final_aggregated + gate * node_states[j];
        }
        return torch::matmul(final_aggregated, w_motor_out.t());
    }

    std::vector<std::string> get_topology_manifest() {
        std::vector<std::string> manifest;
        for (int64_t i = 0; i < k_nodes; ++i) {
            manifest.push_back(node_names[i] + ":" + node_types[i] + ":" + (is_core_node[i] ? "core" : "grafted"));
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

    py::class_<EndogenousThetaGammaPACImpl, torch::nn::Module, std::shared_ptr<EndogenousThetaGammaPACImpl>>(m, "EndogenousThetaGammaPAC")
        .def(py::init<int64_t, std::string, float, float, float, float>(),
             py::arg("dim") = 128, py::arg("device") = "cpu",
             py::arg("fast_min_decay") = 0.05f, py::arg("fast_max_decay") = 0.5f,
             py::arg("slow_min_decay") = 0.0005f, py::arg("slow_max_decay") = 0.01f)
        .def("forward", &EndogenousThetaGammaPACImpl::forward,
             py::arg("x"),
             py::arg("free_energy") = torch::Tensor(),
             py::arg("init_h_fast") = torch::Tensor(),
             py::arg("init_h_slow") = torch::Tensor())
        .def("__call__", &EndogenousThetaGammaPACImpl::forward,
             py::arg("x"),
             py::arg("free_energy") = torch::Tensor(),
             py::arg("init_h_fast") = torch::Tensor(),
             py::arg("init_h_slow") = torch::Tensor());

    py::class_<ParallelOperatorBankImpl, torch::nn::Module, std::shared_ptr<ParallelOperatorBankImpl>>(m, "ParallelOperatorBank")
        .def(py::init<int64_t, int64_t, int64_t>(), py::arg("dim") = 256, py::arg("state_dim") = 128, py::arg("num_operators") = 8)
        .def("compute_operators", &ParallelOperatorBankImpl::compute_operators);

    py::class_<ContinuousHopfieldMemoryImpl, torch::nn::Module, std::shared_ptr<ContinuousHopfieldMemoryImpl>>(m, "ContinuousHopfieldMemory")
        .def(py::init<int64_t, int64_t, std::string>(), py::arg("dim") = 256, py::arg("num_basins") = 32, py::arg("device_str") = "cpu")
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
        .def(py::init<int64_t, std::string, int64_t>(), py::arg("dim"), py::arg("device_str") = "cpu", py::arg("num_basins") = 16)
        .def("forward", &ContinuousHopfieldOpImpl::forward)
        .def("__call__", &ContinuousHopfieldOpImpl::forward);

    py::class_<StateSpaceMemoryOpImpl, torch::nn::Module, std::shared_ptr<StateSpaceMemoryOpImpl>>(m, "StateSpaceMemoryOp")
        .def(py::init<int64_t, std::string>(), py::arg("dim"), py::arg("device_str") = "cpu")
        .def("forward", &StateSpaceMemoryOpImpl::forward)
        .def("__call__", &StateSpaceMemoryOpImpl::forward);

    py::class_<StochasticLangevinOpImpl, torch::nn::Module, std::shared_ptr<StochasticLangevinOpImpl>>(m, "StochasticLangevinOp")
        .def(py::init<int64_t, std::string, float>(), py::arg("dim"), py::arg("device_str") = "cpu", py::arg("gamma") = 1.0f)
        .def("set_free_energy", &StochasticLangevinOpImpl::set_free_energy, py::arg("f_t"))
        .def("forward", &StochasticLangevinOpImpl::forward)
        .def("__call__", &StochasticLangevinOpImpl::forward)
        .def("forward_with_fe", &StochasticLangevinOpImpl::forward_with_fe, py::arg("x"), py::arg("f_t") = 0.0f)
        .def("forward_deterministic", &StochasticLangevinOpImpl::forward_deterministic, py::arg("x"));

    py::class_<ProgrammableDelayOpImpl, torch::nn::Module, std::shared_ptr<ProgrammableDelayOpImpl>>(m, "ProgrammableDelayOp")
        .def(py::init<int64_t, std::string, int64_t>(), py::arg("dim"), py::arg("device_str") = "cpu", py::arg("tau_max") = 16)
        .def("reset_buffer", &ProgrammableDelayOpImpl::reset_buffer)
        .def("forward", &ProgrammableDelayOpImpl::forward)
        .def("__call__", &ProgrammableDelayOpImpl::forward)
        .def("forward_fixed_delay", &ProgrammableDelayOpImpl::forward_fixed_delay, py::arg("x"), py::arg("fixed_tau"));

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
        .def("forward_adaptive", &DynamicMorphicGraphImpl::forward_adaptive, py::arg("x_sensory"), py::arg("max_thinking_steps") = 8, py::arg("halt_threshold") = 0.8f, py::arg("epsilon_halt") = 1e-3f)
        .def("get_topology_manifest", &DynamicMorphicGraphImpl::get_topology_manifest)
        .def("get_methylation_locks", &DynamicMorphicGraphImpl::get_methylation_locks)
        .def("set_methylation_locks", &DynamicMorphicGraphImpl::set_methylation_locks, py::arg("locks"))
        .def("named_parameters_map", [](std::shared_ptr<DynamicMorphicGraphImpl> m) {
            return m->get_active_parameters_map();
        });
}
