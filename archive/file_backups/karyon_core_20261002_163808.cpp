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

    // EXP-314: Dynamic Somatic Neurotransmitter Chrono-Dilation Parameters
    torch::Tensor alpha_na;       // [1] -> scale of noradrenaline deceleration
    torch::Tensor alpha_da;       // [1] -> scale of dopamine acceleration
    torch::Tensor lambda_na;      // [1] -> decay rate of NA state
    torch::Tensor lambda_da;      // [1] -> decay rate of DA state
    torch::Tensor tau_f;          // [1] -> normalizer for Free Energy surprise

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
        w_pac_out = register_parameter("w_pac_out", torch::eye(dim, torch::TensorOptions().device(device)) + torch::randn({dim, dim}, torch::TensorOptions().device(device)) * (scale * 0.1f));

        // Continuous Hopfield Memory Attractor Basins: [num_basins, dim] normalized
        auto raw_basins = torch::randn({num_hopfield_basins, dim}, torch::TensorOptions().device(device)) * scale;
        hopfield_basins = register_parameter("hopfield_basins", torch::nn::functional::normalize(raw_basins, torch::nn::functional::NormalizeFuncOptions().dim(-1)));

        // EXP-314: Neurotransmitter calibration constants
        alpha_na = register_parameter("alpha_na", torch::tensor({1.5f}, torch::TensorOptions().device(device)));
        alpha_da = register_parameter("alpha_da", torch::tensor({1.2f}, torch::TensorOptions().device(device)));
        lambda_na = register_parameter("lambda_na", torch::tensor({0.85f}, torch::TensorOptions().device(device)));
        lambda_da = register_parameter("lambda_da", torch::tensor({0.85f}, torch::TensorOptions().device(device)));
        tau_f = register_parameter("tau_f", torch::tensor({1.0f}, torch::TensorOptions().device(device)));

        this->to(device);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward(
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
                if (free_energy.size(0) == B) {
                    f_t_seq = free_energy.unsqueeze(1).unsqueeze(-1).expand({B, S, 1});
                } else if (free_energy.size(0) == S) {
                    f_t_seq = free_energy.unsqueeze(0).unsqueeze(-1).expand({B, S, 1});
                } else {
                    f_t_seq = free_energy.mean().unsqueeze(0).unsqueeze(1).unsqueeze(-1).expand({B, S, 1});
                }
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

        // Somatic neurotransmitter states
        torch::Tensor na_state = torch::zeros({B, 1}, opts);
        torch::Tensor da_state = torch::zeros({B, 1}, opts);
        torch::Tensor fe_prev = torch::zeros({B, 1}, opts);

        auto delta_t_trace = torch::zeros({B, S, 1}, opts);
        auto commit_gate_trace = torch::zeros({B, S, 1}, opts);
        auto na_trace = torch::zeros({B, S, 1}, opts);
        auto da_trace = torch::zeros({B, S, 1}, opts);
        auto hopfield_snapped_trace = torch::zeros({B, S, D}, opts);
        auto y_out = torch::zeros_like(x);

        for (int64_t t = 0; t < S; ++t) {
            auto x_t = x.select(1, t); // [B, D]
            auto fe_t = f_t_seq.select(1, t); // [B, 1]

            // Dynamic Somatic Neurotransmitter updates
            // NA arousal: tracks current Free Energy surprise normalized by tau_f
            na_state = lambda_na * na_state + (1.0f - lambda_na) * torch::clamp(fe_t / tau_f, 0.0f, 5.0f);
            
            // DA reward: tracks the positive reduction/improvement of Free Energy scaled for somatic parity
            auto delta_fe = (t == 0) ? torch::zeros_like(fe_t) : torch::clamp(fe_prev - fe_t, 0.0f, 10.0f);
            da_state = lambda_da * da_state + (1.0f - lambda_da) * (delta_fe * 3.0f); // 3x scaling for somatic parity
            
            fe_prev = fe_t;

            na_trace.select(1, t).copy_(na_state);
            da_trace.select(1, t).copy_(da_state);

            // Reciprocal inhibition: Dopamine inhibits Noradrenaline arousal
            auto na_eff = torch::clamp(na_state - 1.2f * da_state, 0.0f, 10.0f);

            // 1. Direct Time-Speed Actuator modulated by Noradrenaline (dilation) and Dopamine (acceleration)
            auto dt_logits = torch::matmul(x_t, w_delta_t.t()) + b_delta_t; // [B, 1]
            // Dilation: -alpha_na * tanh(na_eff) | Acceleration: +alpha_da * tanh(da_state)
            auto dt_modulated = dt_logits - alpha_na * torch::tanh(na_eff) + alpha_da * torch::tanh(da_state);
            auto dt_t = torch::softplus(dt_modulated) + 0.05f; // ensure minimal physical causality dt >= 0.05
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

        return std::make_tuple(y_out, delta_t_trace, commit_gate_trace, na_trace, da_trace, state_fast, state_slow, hopfield_snapped_trace);
    }
};
TORCH_MODULE(EndogenousThetaGammaPAC);

// ============================================================================
// 2C. TRI-SCALE HIERARCHICAL PAC COUPLER (Gamma -> Theta -> Delta Cascade)
// KEP Principle 2 & Principle 8 (Compositional Depth Over Flat Width)
// EXP-315: Full 3-scale hierarchical chrono-coupling (Fast Gamma -> Meso Theta -> Macro Delta)
// ============================================================================
class TriScaleHierarchicalPACImpl : public torch::nn::Module {
public:
    int64_t dim;
    int64_t num_hopfield_basins;
    bool use_hopfield_snapping;
    float hopfield_beta;

    // 1. Fast Gamma SSD (Micro-scale / Bytes / Phonemes) tau in [0.05, 0.50]
    torch::Tensor w_k_gamma, w_v_gamma, w_q_gamma, w_out_gamma, log_decay_gamma;

    // 2. Meso Theta SSD (Meso-scale / Words / Morphemes) tau in [0.005, 0.05]
    torch::Tensor w_k_theta, w_v_theta, w_q_theta, w_out_theta, log_decay_theta;

    // 3. Macro Delta SSD (Macro-scale / Sentences / Discourse / Invariants) tau in [0.0001, 0.001]
    torch::Tensor w_k_delta, w_v_delta, w_q_delta, w_out_delta, log_decay_delta;

    // Endogenous Chrono-Actuator (Time-Warping via NA/DA)
    torch::Tensor w_delta_t, b_delta_t;
    torch::Tensor alpha_na, alpha_da, lambda_na, lambda_da, tau_f;

    // Gate 1 (Meso Valve / Theta Commit): Detects word/morpheme boundaries from Gamma
    torch::Tensor w_gate1, b_gate1, beta_gate1;
    torch::Tensor w_comp1, b_comp1;

    // Gate 2 (Macro Valve / Delta Commit): Detects sentence/topic shifts from Theta
    torch::Tensor w_gate2, b_gate2, beta_gate2;
    torch::Tensor w_comp2, b_comp2;

    // Modern Continuous Hopfield Basins on Macro Delta Commits
    torch::Tensor hopfield_basins; // [num_basins, dim]

    // Bilinear Cascaded Modulations
    torch::Tensor w_bilinear_a_theta, w_bilinear_b_theta;
    torch::Tensor w_bilinear_a_delta, w_bilinear_b_delta;
    torch::Tensor w_pac_out;

    TriScaleHierarchicalPACImpl(int64_t dim = 128, std::string device_str = "cpu",
                                float gamma_min_decay = 0.05f, float gamma_max_decay = 0.50f,
                                float theta_min_decay = 0.005f, float theta_max_decay = 0.05f,
                                float delta_min_decay = 0.0001f, float delta_max_decay = 0.001f,
                                int64_t num_hopfield_basins = 256,
                                bool use_hopfield_snapping = true,
                                float hopfield_beta = 12.0f)
        : dim(dim), num_hopfield_basins(num_hopfield_basins),
          use_hopfield_snapping(use_hopfield_snapping), hopfield_beta(hopfield_beta) {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        float scale = 1.0f / std::sqrt((float)dim);

        // 1. Fast Gamma SSD
        auto lin_gamma = torch::linspace(std::log(gamma_min_decay), std::log(gamma_max_decay), dim, torch::TensorOptions().device(device));
        log_decay_gamma = register_parameter("log_decay_gamma", lin_gamma);
        w_k_gamma = register_parameter("w_k_gamma", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * scale);
        w_v_gamma = register_parameter("w_v_gamma", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * scale);
        w_q_gamma = register_parameter("w_q_gamma", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * scale);
        w_out_gamma = register_parameter("w_out_gamma", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * scale);

        // 2. Meso Theta SSD
        auto lin_theta = torch::linspace(std::log(theta_min_decay), std::log(theta_max_decay), dim, torch::TensorOptions().device(device));
        log_decay_theta = register_parameter("log_decay_theta", lin_theta);
        w_k_theta = register_parameter("w_k_theta", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * scale);
        w_v_theta = register_parameter("w_v_theta", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * scale);
        w_q_theta = register_parameter("w_q_theta", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * scale);
        w_out_theta = register_parameter("w_out_theta", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * scale);

        // 3. Macro Delta SSD
        auto lin_delta = torch::linspace(std::log(delta_min_decay), std::log(delta_max_decay), dim, torch::TensorOptions().device(device));
        log_decay_delta = register_parameter("log_decay_delta", lin_delta);
        w_k_delta = register_parameter("w_k_delta", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * scale);
        w_v_delta = register_parameter("w_v_delta", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * scale);
        w_q_delta = register_parameter("w_q_delta", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * scale);
        w_out_delta = register_parameter("w_out_delta", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * scale);

        // Chrono-Actuator
        w_delta_t = register_parameter("w_delta_t", torch::randn({1, dim}, torch::TensorOptions().device(device)) * scale);
        b_delta_t = register_parameter("b_delta_t", torch::tensor({0.5413f}, torch::TensorOptions().device(device)));
        alpha_na = register_parameter("alpha_na", torch::tensor({1.5f}, torch::TensorOptions().device(device)));
        alpha_da = register_parameter("alpha_da", torch::tensor({1.2f}, torch::TensorOptions().device(device)));
        lambda_na = register_parameter("lambda_na", torch::tensor({0.85f}, torch::TensorOptions().device(device)));
        lambda_da = register_parameter("lambda_da", torch::tensor({0.85f}, torch::TensorOptions().device(device)));
        tau_f = register_parameter("tau_f", torch::tensor({1.0f}, torch::TensorOptions().device(device)));

        // Gate 1 (Meso Valve / Theta): Default bias set for ~1 in 5-6 steps
        w_gate1 = register_parameter("w_gate1", torch::randn({1, dim}, torch::TensorOptions().device(device)) * scale);
        b_gate1 = register_parameter("b_gate1", torch::tensor({-2.0f}, torch::TensorOptions().device(device)));
        beta_gate1 = register_parameter("beta_gate1", torch::tensor({2.0f}, torch::TensorOptions().device(device)));
        w_comp1 = register_parameter("w_comp1", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * scale);
        b_comp1 = register_parameter("b_comp1", torch::zeros({dim}, torch::TensorOptions().device(device)));

        // Gate 2 (Macro Valve / Delta): Default bias set for ~1 in 30-50 steps (more negative bias)
        w_gate2 = register_parameter("w_gate2", torch::randn({1, dim}, torch::TensorOptions().device(device)) * scale);
        b_gate2 = register_parameter("b_gate2", torch::tensor({-6.0f}, torch::TensorOptions().device(device)));
        beta_gate2 = register_parameter("beta_gate2", torch::tensor({1.75f}, torch::TensorOptions().device(device)));
        w_comp2 = register_parameter("w_comp2", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * scale);
        b_comp2 = register_parameter("b_comp2", torch::zeros({dim}, torch::TensorOptions().device(device)));

        // Hopfield Basins on Unit Sphere
        auto raw_basins = torch::randn({num_hopfield_basins, dim}, torch::TensorOptions().device(device)) * scale;
        hopfield_basins = register_parameter("hopfield_basins", torch::nn::functional::normalize(raw_basins, torch::nn::functional::NormalizeFuncOptions().dim(-1)));

        // Bilinear Cascaded Projections
        w_bilinear_a_theta = register_parameter("w_bilinear_a_theta", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * scale);
        w_bilinear_b_theta = register_parameter("w_bilinear_b_theta", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * scale);
        w_bilinear_a_delta = register_parameter("w_bilinear_a_delta", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * scale);
        w_bilinear_b_delta = register_parameter("w_bilinear_b_delta", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * scale);
        w_pac_out = register_parameter("w_pac_out", torch::eye(dim, torch::TensorOptions().device(device)) + torch::randn({dim, dim}, torch::TensorOptions().device(device)) * (scale * 0.1f));

        this->to(device);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward(
        torch::Tensor x,
        torch::Tensor free_energy = torch::Tensor(),
        torch::Tensor init_h_gamma = torch::Tensor(),
        torch::Tensor init_h_theta = torch::Tensor(),
        torch::Tensor init_h_delta = torch::Tensor()) {
        auto B = x.size(0);
        auto S = x.size(1);
        auto D = x.size(2);
        auto dev = x.device();
        auto opts = x.options();

        torch::Tensor f_t_seq;
        if (free_energy.defined() && free_energy.numel() > 0) {
            if (free_energy.dim() == 1) {
                if (free_energy.size(0) == B) {
                    f_t_seq = free_energy.unsqueeze(1).unsqueeze(-1).expand({B, S, 1});
                } else if (free_energy.size(0) == S) {
                    f_t_seq = free_energy.unsqueeze(0).unsqueeze(-1).expand({B, S, 1});
                } else {
                    f_t_seq = free_energy.mean().unsqueeze(0).unsqueeze(1).unsqueeze(-1).expand({B, S, 1});
                }
            } else if (free_energy.dim() == 2) {
                f_t_seq = free_energy.unsqueeze(-1);
            } else {
                f_t_seq = free_energy;
            }
        } else {
            f_t_seq = torch::zeros({B, S, 1}, opts);
        }

        // Projections for Gamma
        auto k_gamma = torch::matmul(x, w_k_gamma.t());
        auto v_gamma = torch::matmul(x, w_v_gamma.t());
        auto q_gamma = torch::matmul(x, w_q_gamma.t());
        auto kv_gamma = k_gamma * v_gamma; // [B, S, D]

        // Continuous decay rates
        auto base_a_gamma = torch::exp(log_decay_gamma);
        auto base_a_theta = torch::exp(log_decay_theta);
        auto base_a_delta = torch::exp(log_decay_delta);

        auto norm_basins = torch::nn::functional::normalize(hopfield_basins, torch::nn::functional::NormalizeFuncOptions().dim(-1));

        // States
        torch::Tensor state_gamma = (init_h_gamma.defined() && init_h_gamma.sizes() == torch::IntArrayRef({B, D})) ? init_h_gamma : torch::zeros({B, D}, opts);
        torch::Tensor state_theta = (init_h_theta.defined() && init_h_theta.sizes() == torch::IntArrayRef({B, D})) ? init_h_theta : torch::zeros({B, D}, opts);
        torch::Tensor state_delta = (init_h_delta.defined() && init_h_delta.sizes() == torch::IntArrayRef({B, D})) ? init_h_delta : torch::zeros({B, D}, opts);

        torch::Tensor na_state = torch::zeros({B, 1}, opts);
        torch::Tensor da_state = torch::zeros({B, 1}, opts);
        torch::Tensor fe_prev = torch::zeros({B, 1}, opts);

        auto delta_t_trace = torch::zeros({B, S, 1}, opts);
        auto gate1_trace = torch::zeros({B, S, 1}, opts);
        auto gate2_trace = torch::zeros({B, S, 1}, opts);
        auto na_trace = torch::zeros({B, S, 1}, opts);
        auto da_trace = torch::zeros({B, S, 1}, opts);
        auto h_gamma_trace = torch::zeros({B, S, D}, opts);
        auto h_theta_trace = torch::zeros({B, S, D}, opts);
        auto h_delta_trace = torch::zeros({B, S, D}, opts);
        auto hopfield_snapped_trace = torch::zeros({B, S, D}, opts);
        auto y_out = torch::zeros_like(x);

        for (int64_t t = 0; t < S; ++t) {
            auto x_t = x.select(1, t); // [B, D]
            auto fe_t = f_t_seq.select(1, t); // [B, 1]

            // Dynamic Somatic Neurotransmitters
            na_state = lambda_na * na_state + (1.0f - lambda_na) * torch::clamp(fe_t / tau_f, 0.0f, 5.0f);
            auto delta_fe = (t == 0) ? torch::zeros_like(fe_t) : torch::clamp(fe_prev - fe_t, 0.0f, 10.0f);
            da_state = lambda_da * da_state + (1.0f - lambda_da) * (delta_fe * 3.0f);
            fe_prev = fe_t;

            na_trace.select(1, t).copy_(na_state);
            da_trace.select(1, t).copy_(da_state);

            auto na_eff = torch::clamp(na_state - 1.2f * da_state, 0.0f, 10.0f);

            // Chrono-Actuator: delta_t
            auto dt_logits = torch::matmul(x_t, w_delta_t.t()) + b_delta_t;
            auto dt_modulated = dt_logits - alpha_na * torch::tanh(na_eff) + alpha_da * torch::tanh(da_state);
            auto dt_t = torch::softplus(dt_modulated) + 0.05f;
            delta_t_trace.select(1, t).copy_(dt_t);

            // 1. Fast Gamma SSD
            auto decay_gamma_t = torch::exp(-base_a_gamma.unsqueeze(0) * dt_t);
            state_gamma = state_gamma * decay_gamma_t + kv_gamma.select(1, t) * dt_t;
            auto h_gamma_t = torch::matmul(q_gamma.select(1, t) * state_gamma, w_out_gamma.t());
            h_gamma_trace.select(1, t).copy_(h_gamma_t);

            // 2. Gate 1 (Meso Valve / Theta Commit): g1_t = Sigmoid(W_g1 * h_gamma_t + b_g1 + beta_g1 * tanh(fe_t))
            auto gate1_logits = torch::matmul(h_gamma_t, w_gate1.t()) + b_gate1 + beta_gate1 * torch::tanh(fe_t);
            auto g1_t = torch::sigmoid(gate1_logits); // [B, 1]
            gate1_trace.select(1, t).copy_(g1_t);

            // Compress Gamma to Theta input: Compress_1(h_gamma(t))
            auto gamma_comp = torch::tanh(torch::matmul(h_gamma_t, w_comp1.t()) + b_comp1);

            auto k_theta_t = torch::matmul(gamma_comp, w_k_theta.t());
            auto v_theta_t = torch::matmul(gamma_comp, w_v_theta.t());
            auto q_theta_t = torch::matmul(gamma_comp, w_q_theta.t());
            auto kv_theta_t = k_theta_t * v_theta_t;

            // Meso Theta SSD updated through Gate 1: h_theta(t) = (1 - g1) * h_theta(t-1) + g1 * candidate
            auto decay_theta_t = torch::exp(-base_a_theta.unsqueeze(0) * dt_t);
            // Decay state continuously, but add new input ONLY when Gate 1 opens!
            state_theta = state_theta * decay_theta_t + g1_t * kv_theta_t;
            auto h_theta_t = torch::matmul(q_theta_t * state_theta, w_out_theta.t());
            h_theta_trace.select(1, t).copy_(h_theta_t);

            // 3. Gate 2 (Macro Valve / Delta Commit): g2_t = Sigmoid(W_g2 * h_theta_t + b_g2 + beta_g2 * tanh(fe_t))
            auto gate2_logits = torch::matmul(h_theta_t, w_gate2.t()) + b_gate2 + beta_gate2 * torch::tanh(fe_t);
            auto g2_t = torch::sigmoid(gate2_logits); // [B, 1]
            gate2_trace.select(1, t).copy_(g2_t);

            // Compress Theta to Delta input: Compress_2(h_theta(t))
            auto theta_comp = torch::tanh(torch::matmul(h_theta_t, w_comp2.t()) + b_comp2);

            // Continuous Hopfield Attractor Snapping for Macro Delta
            torch::Tensor delta_input = theta_comp;
            if (use_hopfield_snapping) {
                auto norm_theta = torch::nn::functional::normalize(theta_comp, torch::nn::functional::NormalizeFuncOptions().dim(-1));
                auto hopfield_scores = torch::matmul(norm_theta, norm_basins.t()) * hopfield_beta;
                auto hopfield_attn = torch::softmax(hopfield_scores, -1);
                auto h_snapped = torch::matmul(hopfield_attn, norm_basins);
                delta_input = h_snapped;
                hopfield_snapped_trace.select(1, t).copy_(h_snapped);
            } else {
                hopfield_snapped_trace.select(1, t).copy_(theta_comp);
            }

            auto k_delta_t = torch::matmul(delta_input, w_k_delta.t());
            auto v_delta_t = torch::matmul(delta_input, w_v_delta.t());
            auto q_delta_t = torch::matmul(delta_input, w_q_delta.t());
            auto kv_delta_t = k_delta_t * v_delta_t;

            // Macro Delta SSD updated through Gate 2: h_delta(t) = state_delta * decay + g2 * kv_delta
            auto decay_delta_t = torch::exp(-base_a_delta.unsqueeze(0) * dt_t);
            state_delta = state_delta * decay_delta_t + g2_t * kv_delta_t;
            auto h_delta_t = torch::matmul(q_delta_t * state_delta, w_out_delta.t());
            h_delta_trace.select(1, t).copy_(h_delta_t);

            // 4. Residual Gated Modulation (Stabilized Gradient Flow):
            // Instead of compounding multiplicative scaling, we use residual gated offsets:
            auto mod_theta_a = torch::matmul(h_theta_t, w_bilinear_a_theta.t());
            auto mod_theta_b = torch::sigmoid(torch::matmul(h_theta_t, w_bilinear_b_theta.t()));
            auto pac_theta = mod_theta_a * mod_theta_b;

            auto mod_delta_a = torch::matmul(h_delta_t, w_bilinear_a_delta.t());
            auto mod_delta_b = torch::sigmoid(torch::matmul(h_delta_t, w_bilinear_b_delta.t()));
            auto pac_delta = mod_delta_a * mod_delta_b;

            auto y_t = h_gamma_t + 0.1f * (pac_theta + pac_delta);
            y_out.select(1, t).copy_(torch::matmul(y_t, w_pac_out.t()));
        }

        return std::make_tuple(
            y_out, delta_t_trace, gate1_trace, gate2_trace,
            na_trace, da_trace,
            h_gamma_trace, h_theta_trace, h_delta_trace,
            hopfield_snapped_trace
        );
    }
};
TORCH_MODULE(TriScaleHierarchicalPAC);

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
// 4. CONTINUOUS HOPFIELD MEMORY WITH CONTEXT-GATED REPULSOR DYNAMICS
// ============================================================================
class ContinuousHopfieldMemoryImpl : public torch::nn::Module {
public:
    int64_t dim;
    int64_t num_basins;
    torch::Tensor basins;

    // Context-Gated Somatic Episode Buffers (Triad: Context, Action, Valence)
    int64_t max_episodes;
    int64_t active_episodes;
    torch::Tensor context_keys;  // [max_episodes, dim]
    torch::Tensor action_keys;   // [max_episodes, dim]
    torch::Tensor valences;      // [max_episodes]

    ContinuousHopfieldMemoryImpl(int64_t dim = 256, int64_t num_basins = 32, std::string device_str = "cpu", int64_t max_episodes = 256)
        : dim(dim), num_basins(num_basins), max_episodes(max_episodes), active_episodes(0) {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        basins = register_parameter("basins", torch::randn({num_basins, dim}, torch::TensorOptions().device(device)) * (1.0f / std::sqrt(dim)));
        
        context_keys = register_buffer("context_keys", torch::zeros({max_episodes, dim}, torch::TensorOptions().device(device)));
        action_keys = register_buffer("action_keys", torch::zeros({max_episodes, dim}, torch::TensorOptions().device(device)));
        valences = register_buffer("valences", torch::zeros({max_episodes}, torch::TensorOptions().device(device)));
        
        this->to(device);
    }

    void record_somatic_episode(torch::Tensor context_key, torch::Tensor action_key, float valence) {
        auto dev = basins.device();
        auto ctx_norm = torch::nn::functional::normalize(context_key.to(dev).detach().view({1, dim}), torch::nn::functional::NormalizeFuncOptions().dim(-1));
        auto act_norm = torch::nn::functional::normalize(action_key.to(dev).detach().view({1, dim}), torch::nn::functional::NormalizeFuncOptions().dim(-1));

        int64_t slot = active_episodes % max_episodes;
        context_keys.index_put_({slot}, ctx_norm.squeeze(0));
        action_keys.index_put_({slot}, act_norm.squeeze(0));
        valences.index_put_({slot}, torch::tensor(valence, torch::TensorOptions().device(dev)));

        if (active_episodes < max_episodes) {
            active_episodes++;
        }
    }

    // Context-gated repulsion relaxation
    torch::Tensor relax_with_repulsion(torch::Tensor context_t, torch::Tensor action_t, float beta = 8.0f) {
        auto dev = action_t.device();
        auto ctx_norm = torch::nn::functional::normalize(context_t.to(dev), torch::nn::functional::NormalizeFuncOptions().dim(-1)); // [B, dim]
        auto act_norm = torch::nn::functional::normalize(action_t.to(dev), torch::nn::functional::NormalizeFuncOptions().dim(-1)); // [B, dim]

        if (active_episodes == 0) {
            return action_t;
        }

        // Active episodes view
        auto valid_ctx = context_keys.slice(0, 0, active_episodes); // [N, dim]
        auto valid_act = action_keys.slice(0, 0, active_episodes); // [N, dim]
        auto valid_val = valences.slice(0, 0, active_episodes);    // [N]

        // Context alignment: c_t^T c_i (unscaled cosine similarity between normalized vectors)
        auto ctx_sim = torch::matmul(ctx_norm, valid_ctx.t()); // [B, N]
        
        // Action alignment: a_t^T a_i (unscaled cosine similarity)
        auto act_sim = torch::matmul(act_norm, valid_act.t()); // [B, N]

        // Context gating filter: sharp gate when context similarity is high
        auto ctx_gate = torch::sigmoid((ctx_sim - 0.3f) * 12.0f); // smooth continuous gate [B, N]

        // Continuous Tripartite Valence & Forces:
        // F_attract = sum_i [ Gate_ctx,i * max(0.0, V_i) * max(0, a_t^T a_i) * a_i ]
        // F_repulse = sum_i [ Gate_ctx,i * max(0.0, -V_i) * max(0, a_t^T a_i) * a_i ]
        // When V_i == 0.0 (neutral topographic anchor), both positive and negative projections are strictly 0.0
        auto val_pos = torch::clamp(valid_val.unsqueeze(0), 0.0f, 1.0f);   // [B, N], max(0.0, V_i)
        auto val_neg = torch::clamp(-valid_val.unsqueeze(0), 0.0f, 1.0f);  // [B, N], max(0.0, -V_i)

        auto act_proj = torch::clamp(act_sim, 0.0f, 1.0f); // [B, N], max(0, a_t^T a_i)

        auto att_weights = ctx_gate * val_pos * act_proj; // [B, N]
        auto att_force = torch::matmul(att_weights, valid_act); // [B, dim]

        auto rep_weights = ctx_gate * val_neg * act_proj; // [B, N]
        auto rep_force = torch::matmul(rep_weights, valid_act); // [B, dim]

        // Trajectory displacement:
        // a_relaxed = Normalize(a_t + 0.8 * F_attract - 1.5 * F_repulse)
        auto relaxed_action = action_t + 0.8f * att_force - 1.5f * rep_force;
        return torch::nn::functional::normalize(relaxed_action, torch::nn::functional::NormalizeFuncOptions().dim(-1));
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
// 9. TSODYKS-MARKRAM SYNAPTIC DEPRESSION OP (7th Atomic Operator - Activity-Dependent Vesicular Fatigue)
// dx_i / dt = (1.0 - x_i) / tau_rec - u_depress * x_i * a_i(t)
// W_eff(t) = W * diag(x_t)
// y_t = x_t * (W * a_t)
// ============================================================================
class TsodyksMarkramSynapticDepressionOpImpl : public GraphOp {
public:
    int64_t dim;
    float tau_rec;
    float u_depress;
    std::string device_str;
    torch::Tensor w_proj;
    torch::Tensor vesicle_resource; // [B, dim] in [0.0, 1.0]
    bool state_initialized = false;

    TsodyksMarkramSynapticDepressionOpImpl(int64_t dim, std::string device_str, float tau_rec = 8.0f, float u_depress = 0.5f)
        : dim(dim), tau_rec(tau_rec), u_depress(u_depress), device_str(device_str) {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        w_proj = register_parameter("w_proj", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * (1.0f / std::sqrt((float)dim)));
        this->to(device);
    }

    void reset_state() {
        state_initialized = false;
        vesicle_resource = torch::Tensor();
    }

    torch::Tensor get_vesicle_resource() const {
        return vesicle_resource;
    }

    torch::Tensor forward(torch::Tensor x) override {
        auto B = x.size(0);
        auto device = x.device();

        if (!state_initialized || !vesicle_resource.defined() || vesicle_resource.size(0) != B || vesicle_resource.device() != device) {
            vesicle_resource = torch::ones({B, dim}, torch::TensorOptions().device(device));
            state_initialized = true;
        }

        // 1. Normalized activation intensity a_i(t) = |tanh(x_i)| in [0.0, 1.0]
        auto a_t = torch::abs(torch::tanh(x)); // [B, dim]

        // 2. Tsodyks-Markram Vesicular Dynamics integration:
        // dx_i/dt = (1.0 - x_i) / tau_rec - u_depress * x_i * a_i
        // Discrete Euler update: x_{t+1} = clamp(x_t + (1.0 - x_t) / tau_rec - u_depress * x_t * a_t, 0.01, 1.0)
        auto recovery = (1.0f - vesicle_resource) / tau_rec;
        auto depression = u_depress * vesicle_resource * a_t;
        auto next_x = torch::clamp(vesicle_resource + recovery - depression, 0.01f, 1.0f);
        
        // Update persistent state in-place without breaking gradient chain
        vesicle_resource = next_x.detach();

        // 3. Effective Synaptic Transmission:
        // W_eff = W * x_t -> projection output modulated by available transmitter pool
        auto proj = torch::matmul(x, w_proj.t()); // [B, dim]
        return next_x * proj;
    }
};

// ============================================================================
// 8h. BADDELEY MULTI-SLOT WORKING MEMORY & VECTOR SCRATCHPAD (EXP-322)
// S = 4 isolated registers in R^{S x D} with read/write/reset gating
// ============================================================================
class SlotMemoryOpImpl : public GraphOp {
public:
    int64_t dim;
    int64_t num_slots;
    std::string device_str;
    torch::Tensor w_write_key;   // [num_slots, dim]
    torch::Tensor w_write_val;   // [dim, dim]
    torch::Tensor w_read_key;    // [num_slots, dim]
    torch::Tensor w_read_out;    // [dim, dim]
    torch::Tensor w_erase_gate;  // [num_slots, dim]
    
    torch::Tensor memory_slots;  // [B, num_slots, dim]
    bool slots_initialized = false;

    SlotMemoryOpImpl(int64_t dim, std::string device_str, int64_t num_slots = 4)
        : dim(dim), num_slots(num_slots), device_str(device_str) {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        w_write_key = register_parameter("w_write_key", torch::randn({num_slots, dim}, torch::TensorOptions().device(device)) * (1.0f / std::sqrt((float)dim)));
        w_write_val = register_parameter("w_write_val", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * (1.0f / std::sqrt((float)dim)));
        w_read_key = register_parameter("w_read_key", torch::randn({num_slots, dim}, torch::TensorOptions().device(device)) * (1.0f / std::sqrt((float)dim)));
        w_read_out = register_parameter("w_read_out", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * (1.0f / std::sqrt((float)dim)));
        w_erase_gate = register_parameter("w_erase_gate", torch::randn({num_slots, dim}, torch::TensorOptions().device(device)) * (1.0f / std::sqrt((float)dim)));
        this->to(device);
    }

    void reset_state() {
        slots_initialized = false;
        memory_slots = torch::Tensor();
    }

    torch::Tensor get_memory_slots() const {
        return memory_slots;
    }

    torch::Tensor forward(torch::Tensor x) override {
        auto B = x.size(0);
        auto device = x.device();

        if (!slots_initialized || !memory_slots.defined() || memory_slots.size(0) != B || memory_slots.device() != device) {
            memory_slots = torch::zeros({B, num_slots, dim}, torch::TensorOptions().device(device));
            slots_initialized = true;
        }

        // 1. Soft Address Write Gate: alpha_write = Softmax(x * W_write_key^T / sqrt(D)) -> [B, num_slots]
        auto write_logits = torch::matmul(x, w_write_key.t()) * (1.0f / std::sqrt((float)dim)); // [B, num_slots]
        auto alpha_write = torch::softmax(write_logits, -1); // [B, num_slots]

        // 2. Candidate Value to write: v_cand = tanh(x * W_write_val^T) -> [B, dim]
        auto v_cand = torch::tanh(torch::matmul(x, w_write_val.t())); // [B, dim]

        // 3. Selective Erase Gate: e_t = Sigmoid(x * W_erase_gate^T) -> [B, num_slots]
        auto erase_gate = torch::sigmoid(torch::matmul(x, w_erase_gate.t())); // [B, num_slots]

        // 4. Update Memory Slots in R^{B x num_slots x dim}:
        // M_{t+1}[s] = (1.0 - alpha_write[s] * erase_gate[s]) * M_t[s] + alpha_write[s] * v_cand
        auto erase_factor = 1.0f - (alpha_write * erase_gate).unsqueeze(-1); // [B, num_slots, 1]
        auto write_term = alpha_write.unsqueeze(-1) * v_cand.unsqueeze(1);   // [B, num_slots, dim]
        auto next_slots = memory_slots * erase_factor + write_term;

        // Persistent update across sub-step thinking cycles
        memory_slots = next_slots.detach();

        // 5. Soft Address Read Gate: alpha_read = Softmax(x * W_read_key^T / sqrt(D)) -> [B, num_slots]
        auto read_logits = torch::matmul(x, w_read_key.t()) * (1.0f / std::sqrt((float)dim)); // [B, num_slots]
        auto alpha_read = torch::softmax(read_logits, -1); // [B, num_slots]

        // 6. Readout: read_val = Sum_s(alpha_read[s] * M_{t+1}[s]) -> [B, dim]
        auto read_val = torch::sum(alpha_read.unsqueeze(-1) * next_slots, 1); // [B, dim]

        // 7. Output Projection & Residual Bypass
        auto out = torch::matmul(read_val, w_read_out.t()); // [B, dim]
        return out;
    }
};
// ============================================================================
// 8i. NON-LINEAR TRANSFORM OPERATOR PRIMITIVE (Universal Cortical Organelle)
// Two-layer MLP with Swish/GELU activation capable of learning arbitrary
// functional non-linear transforms (cyclic shifts, reflections, modular arithmetic).
// ============================================================================
class NonLinearTransformOpImpl : public GraphOp {
public:
    int64_t dim;
    torch::Tensor w_up;
    torch::Tensor b_up;
    torch::Tensor w_down;
    torch::Tensor b_down;

    NonLinearTransformOpImpl(int64_t dim, std::string device_str) : dim(dim) {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        int64_t hidden_dim = dim * 4;
        w_up = register_parameter("w_up", torch::randn({hidden_dim, dim}, torch::TensorOptions().device(device)) * (1.0f / std::sqrt((float)dim)));
        b_up = register_parameter("b_up", torch::zeros({hidden_dim}, torch::TensorOptions().device(device)));
        w_down = register_parameter("w_down", torch::randn({dim, hidden_dim}, torch::TensorOptions().device(device)) * (1.0f / std::sqrt((float)hidden_dim)));
        b_down = register_parameter("b_down", torch::zeros({dim}, torch::TensorOptions().device(device)));
        this->to(device);
    }

    torch::Tensor forward(torch::Tensor x) override {
        // Two-layer MLP with GELU non-linearity
        auto h = torch::matmul(x, w_up.t()) + b_up;
        auto h_act = torch::gelu(h);
        auto out = torch::matmul(h_act, w_down.t()) + b_down;
        return out;
    }
};

// ============================================================================
// 9. DYNAMIC MORPHIC GRAPH & COMMUTATION ORCHESTRATOR R(h_t)
// ============================================================================
class DynamicMorphicGraphImpl : public torch::nn::Module {
public:
    int64_t dim;
    std::string device_str;
    int64_t max_nodes = 128;
    int64_t k_nodes = 0;
    int64_t total_sprouted_so_far = 0;

    std::vector<std::shared_ptr<GraphOp>> node_ops;
    std::vector<torch::Tensor> alpha_epi;
    std::vector<bool> is_core_node;
    std::vector<std::string> node_names;
    std::vector<std::string> node_types;
    std::vector<float> methylation_locks; // Susumu Ohno Gene Lock: 1.0 = Frozen, 0.0 = Plastic

    torch::Tensor w_route;
    torch::Tensor w_query; // Dynamic Cross-Node Query Projection [dim, dim]
    torch::Tensor w_key;   // Dynamic Cross-Node Key Projection [dim, dim]
    torch::Tensor organelle_signatures; // Static Functional Organelle Identity Passports [max_nodes, dim]
    torch::Tensor w_query_step; // Query projection for step chaining [dim, dim]
    torch::Tensor w_sensory_in, w_motor_out;
    torch::Tensor w_readout_ctx_proj; // Projection from context vector to readout query vector [dim, dim]
    torch::Tensor w_init_route;       // Initial Step Routing Projection from instruction vector to node logits [dim, max_nodes]

    torch::Tensor w_halt;

    DynamicMorphicGraphImpl(int64_t dim = 128, std::string device_str = "cpu", int64_t max_nodes = 128)
        : dim(dim), device_str(device_str), max_nodes(max_nodes) {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        w_route = register_parameter("w_route", torch::zeros({max_nodes, max_nodes}, torch::TensorOptions().device(device)));
        w_query = register_parameter("w_query", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * (0.1f / std::sqrt((float)dim)));
        w_key = register_parameter("w_key", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * (0.1f / std::sqrt((float)dim)));
        organelle_signatures = register_parameter("organelle_signatures", torch::randn({max_nodes, dim}, torch::TensorOptions().device(device)) * (1.0f / std::sqrt((float)dim)));
        w_query_step = register_parameter("w_query_step", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * (1.0f / std::sqrt((float)dim)));
        w_sensory_in = register_parameter("w_sensory_in", torch::eye(dim, torch::TensorOptions().device(device)));
        w_motor_out = register_parameter("w_motor_out", torch::eye(dim, torch::TensorOptions().device(device)));
        w_readout_ctx_proj = register_parameter("w_readout_ctx_proj", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * (1.0f / std::sqrt((float)dim)));
        w_init_route = register_parameter("w_init_route", torch::randn({dim, max_nodes}, torch::TensorOptions().device(device)) * (1.0f / std::sqrt((float)dim)));
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
        } else if (op_type == "TsodyksMarkram") {
            op = std::make_shared<TsodyksMarkramSynapticDepressionOpImpl>(dim, device_str, 8.0f, 0.5f);
        } else if (op_type == "SlotMemory") {
            op = std::make_shared<SlotMemoryOpImpl>(dim, device_str, 4);
        } else if (op_type == "NonLinearTransform") {
            op = std::make_shared<NonLinearTransformOpImpl>(dim, device_str);
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
        if (dst_idx < max_nodes && src_idx < max_nodes) {
            w_route[dst_idx].copy_(w_route[src_idx]);
            w_route.select(1, dst_idx).copy_(w_route.select(1, src_idx));
        }

        return dst_idx;
    }

    std::map<std::string, torch::Tensor> get_active_parameters_map() {
        std::map<std::string, torch::Tensor> params;
        params["w_route"] = w_route;
        params["w_query"] = w_query;
        params["w_key"] = w_key;
        params["organelle_signatures"] = organelle_signatures;
        params["w_query_step"] = w_query_step;
        params["w_sensory_in"] = w_sensory_in;
        params["w_motor_out"] = w_motor_out;
        params["w_readout_ctx_proj"] = w_readout_ctx_proj;
        params["w_init_route"] = w_init_route;
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

    int64_t prune_relative_darwinism(float relative_threshold_factor = 0.15f) {
        int64_t pruned_count = 0;
        torch::NoGradGuard no_grad;
        if (k_nodes <= 1) return 0;

        std::vector<float> utilities;
        float total_u = 0.0f;
        for (int64_t i = 0; i < k_nodes; ++i) {
            float gate = std::abs(std::tanh(alpha_epi[i].item<float>()));
            utilities.push_back(gate);
            total_u += gate;
        }

        float mean_u = total_u / static_cast<float>(k_nodes);
        float cutoff = relative_threshold_factor * mean_u;

        for (int64_t i = 0; i < k_nodes; ++i) {
            if (!is_core_node[i] && methylation_locks[i] < 0.5f) {
                if (utilities[i] < cutoff) {
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

    torch::Tensor get_node_gate(int64_t j) {
        if (j < (int64_t)methylation_locks.size() && methylation_locks[j] >= 1.0f) {
            return torch::tensor(1.0f, alpha_epi[j].options());
        }
        return torch::tanh(3.5f * alpha_epi[j]);
    }

    void reset_state() {
        has_persistent_states = false;
        persistent_node_states = torch::Tensor();
    }

    std::tuple<torch::Tensor, float> forward_adaptive(
        torch::Tensor x_sensory,
        torch::Tensor context_chain = torch::Tensor(),
        torch::Tensor op_first_embed = torch::Tensor(),
        int64_t max_thinking_steps = 8,
        float halt_threshold = 0.8f,
        float epsilon_halt = 1e-3f) {
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

        auto sensory_in = torch::matmul(x_sensory, w_sensory_in.t()); // [B, dim]

        // Targeted initial routing at step 0
        torch::Tensor init_routing_weights; // [B, K]
        if (op_first_embed.defined() && op_first_embed.numel() > 0) {
            auto active_w_init = w_init_route.slice(1, 0, K); // [dim, K]
            auto init_logits = torch::matmul(op_first_embed, active_w_init); // [B, K]
            init_routing_weights = torch::softmax(init_logits, -1); // [B, K]
        }

        auto active_w_route = w_route.slice(0, 0, K).slice(1, 0, K);
        auto active_signatures = organelle_signatures.slice(0, 0, K);

        std::vector<torch::Tensor> gate_factors;
        for (int64_t j = 0; j < K; ++j) {
            gate_factors.push_back(get_node_gate(j).abs());
        }
        auto gates_tensor = torch::stack(gate_factors, 0);
        auto inactive_mask = (gates_tensor < 1e-4f).unsqueeze(0).unsqueeze(0);

        int64_t actual_steps_taken = 0;
        torch::Tensor prev_aggregated;

        for (int64_t step = 0; step < max_thinking_steps; ++step) {
            actual_steps_taken++;
            auto current_node_states_b = node_states.permute({1, 0, 2}); // [B, K, dim]

            torch::Tensor active_h_sum = torch::zeros({B, dim}, torch::TensorOptions().device(device));
            for (int64_t j = 0; j < K; ++j) {
                active_h_sum = active_h_sum + get_node_gate(j) * current_node_states_b.select(1, j);
            }

            torch::Tensor curr_step_ctx;
            if (context_chain.defined() && context_chain.numel() > 0) {
                if (context_chain.dim() == 3) {
                    // [B, S_ops, dim] -> select instruction step
                    int64_t s_idx = std::min(step, context_chain.size(1) - 1);
                    curr_step_ctx = context_chain.select(1, s_idx);
                } else if (context_chain.dim() == 2) {
                    curr_step_ctx = context_chain;
                }
            }

            if (curr_step_ctx.defined() && curr_step_ctx.numel() > 0) {
                active_h_sum = active_h_sum + curr_step_ctx;
            }

            auto Q_step = torch::matmul(active_h_sum, w_query_step);
            auto sig_routing_logits = torch::matmul(Q_step, active_signatures.t()) * (1.0f / std::sqrt((float)dim));

            auto step_routing_logits = active_w_route.unsqueeze(0) + sig_routing_logits.unsqueeze(1);
            step_routing_logits = step_routing_logits.masked_fill(inactive_mask, -1e4f);
            auto step_routing_matrix = torch::softmax(step_routing_logits, 2);

            auto aggregated_inputs_b = torch::einsum("bik,bid->bkd", {step_routing_matrix, current_node_states_b});
            auto aggregated_inputs = aggregated_inputs_b.permute({1, 0, 2}); // [K, B, dim]

            if (step == 0) {
                if (init_routing_weights.defined() && init_routing_weights.numel() > 0) {
                    auto init_weights_k = init_routing_weights.t().unsqueeze(-1);
                    aggregated_inputs = aggregated_inputs + init_weights_k * sensory_in.unsqueeze(0);
                } else {
                    aggregated_inputs[0] = aggregated_inputs[0] + sensory_in;
                }
            }

            std::vector<torch::Tensor> new_states;
            for (int64_t j = 0; j < K; ++j) {
                auto raw_out = node_ops[j]->forward(aggregated_inputs[j]);
                auto graft_gate = get_node_gate(j);
                auto grafted_out = graft_gate * raw_out;
                new_states.push_back(grafted_out);
            }
            node_states = torch::stack(new_states, 0);

            // Adaptive Pondering Check
            torch::Tensor curr_aggregated = torch::zeros({B, dim}, torch::TensorOptions().device(device));
            for (int64_t j = 0; j < K; ++j) {
                curr_aggregated = curr_aggregated + get_node_gate(j) * node_states[j];
            }

            auto p_halt = torch::sigmoid(torch::matmul(curr_aggregated, w_halt.t())).mean().item<float>();

            float delta_f = 1.0f;
            if (prev_aggregated.defined()) {
                delta_f = (curr_aggregated - prev_aggregated).norm().item<float>() / (static_cast<float>(B * dim) + 1e-6f);
            }
            prev_aggregated = curr_aggregated;

            if (step >= 1 && (p_halt > halt_threshold || delta_f < epsilon_halt)) {
                break;
            }
        }

        persistent_node_states = node_states.detach();
        has_persistent_states = true;

        // Dynamic Context-Dependent Readout Attention
        auto node_states_b = node_states.permute({1, 0, 2}); // [B, K, dim]
        torch::Tensor readout_logits;
        if (context_chain.defined() && context_chain.numel() > 0) {
            torch::Tensor final_step_ctx;
            if (context_chain.dim() == 3) {
                final_step_ctx = context_chain.select(1, context_chain.size(1) - 1);
            } else {
                final_step_ctx = context_chain;
            }
            auto q_readout = torch::matmul(final_step_ctx, w_readout_ctx_proj).unsqueeze(1); // [B, 1, dim]
            readout_logits = torch::matmul(node_states_b, q_readout.transpose(-1, -2)) * (1.0f / std::sqrt((float)dim)); // [B, K, 1]
        } else {
            readout_logits = torch::zeros({B, K, 1}, torch::TensorOptions().device(device));
        }
        readout_logits = readout_logits.masked_fill((gates_tensor < 1e-4f).unsqueeze(0).unsqueeze(-1), -1e4f);
        auto readout_weights = torch::softmax(readout_logits, 1); // [B, K, 1]
        auto final_readout = torch::einsum("bk,bkd->bd", {readout_weights.squeeze(-1), node_states_b});

        auto out = torch::matmul(final_readout, w_motor_out.t());
        return std::make_tuple(out, static_cast<float>(actual_steps_taken));
    }

    torch::Tensor forward(
        torch::Tensor x_sensory,
        torch::Tensor context_chain = torch::Tensor(),
        torch::Tensor op_first_embed = torch::Tensor(),
        int64_t thinking_steps = 4) {
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

        // Targeted initial routing at step 0
        torch::Tensor init_routing_weights; // [B, K]
        if (op_first_embed.defined() && op_first_embed.numel() > 0) {
            auto active_w_init = w_init_route.slice(1, 0, K); // [dim, K]
            auto init_logits = torch::matmul(op_first_embed, active_w_init); // [B, K]
            init_routing_weights = torch::softmax(init_logits, -1); // [B, K]
        }

        auto active_w_route = w_route.slice(0, 0, K).slice(1, 0, K);
        auto active_signatures = organelle_signatures.slice(0, 0, K); // [K, dim]

        std::vector<torch::Tensor> gate_factors;
        for (int64_t j = 0; j < K; ++j) {
            gate_factors.push_back(get_node_gate(j).abs());
        }
        auto gates_tensor = torch::stack(gate_factors, 0); // [K]
        auto inactive_mask = (gates_tensor < 1e-4f).unsqueeze(0).unsqueeze(0); // [1, 1, K]

        for (int64_t step = 0; step < thinking_steps; ++step) {
            auto current_node_states_b = node_states.permute({1, 0, 2}); // [B, K, dim]

            torch::Tensor active_h_sum = torch::zeros({B, dim}, torch::TensorOptions().device(device));
            for (int64_t j = 0; j < K; ++j) {
                active_h_sum = active_h_sum + get_node_gate(j) * current_node_states_b.select(1, j);
            }

            torch::Tensor curr_step_ctx;
            if (context_chain.defined() && context_chain.numel() > 0) {
                if (context_chain.dim() == 3) {
                    // [B, S_ops, dim] -> select instruction step
                    int64_t s_idx = std::min(step, context_chain.size(1) - 1);
                    curr_step_ctx = context_chain.select(1, s_idx);
                } else if (context_chain.dim() == 2) {
                    curr_step_ctx = context_chain;
                }
            }

            if (curr_step_ctx.defined() && curr_step_ctx.numel() > 0) {
                active_h_sum = active_h_sum + curr_step_ctx;
            }

            auto Q_step = torch::matmul(active_h_sum, w_query_step); // [B, dim]
            auto sig_routing_logits = torch::matmul(Q_step, active_signatures.t()) * (1.0f / std::sqrt((float)dim)); // [B, K]

            auto step_routing_logits = active_w_route.unsqueeze(0) + sig_routing_logits.unsqueeze(1); // [B, K_src, K_tgt]
            step_routing_logits = step_routing_logits.masked_fill(inactive_mask, -1e4f);
            auto step_routing_matrix = torch::softmax(step_routing_logits, 2); // [B, K_src, K_tgt]

            auto aggregated_inputs_b = torch::einsum("bik,bid->bkd", {step_routing_matrix, current_node_states_b});
            auto aggregated_inputs = aggregated_inputs_b.permute({1, 0, 2}); // [K, B, dim]

            if (step == 0) {
                if (init_routing_weights.defined() && init_routing_weights.numel() > 0) {
                    auto init_weights_k = init_routing_weights.t().unsqueeze(-1); // [K, B, 1]
                    aggregated_inputs = aggregated_inputs + init_weights_k * sensory_in.unsqueeze(0);
                } else {
                    aggregated_inputs[0] = aggregated_inputs[0] + sensory_in;
                }
            }

            std::vector<torch::Tensor> new_states;
            for (int64_t j = 0; j < K; ++j) {
                auto raw_out = node_ops[j]->forward(aggregated_inputs[j]);
                auto graft_gate = get_node_gate(j);
                auto grafted_out = graft_gate * raw_out;
                new_states.push_back(grafted_out);
            }
            node_states = torch::stack(new_states, 0);
        }

        persistent_node_states = node_states.detach();
        has_persistent_states = true;

        // Dynamic Context-Dependent Readout Attention targeting the final executed operation
        auto node_states_b = node_states.permute({1, 0, 2}); // [B, K, dim]
        torch::Tensor readout_logits;
        if (context_chain.defined() && context_chain.numel() > 0) {
            torch::Tensor final_step_ctx;
            if (context_chain.dim() == 3) {
                final_step_ctx = context_chain.select(1, context_chain.size(1) - 1);
            } else {
                final_step_ctx = context_chain;
            }
            auto q_readout = torch::matmul(final_step_ctx, w_readout_ctx_proj).unsqueeze(1); // [B, 1, dim]
            readout_logits = torch::matmul(node_states_b, q_readout.transpose(-1, -2)) * (1.0f / std::sqrt((float)dim)); // [B, K, 1]
        } else {
            readout_logits = torch::zeros({B, K, 1}, torch::TensorOptions().device(device));
        }
        readout_logits = readout_logits.masked_fill((gates_tensor < 1e-4f).unsqueeze(0).unsqueeze(-1), -1e4f);
        auto readout_weights = torch::softmax(readout_logits, 1); // [B, K, 1]
        auto final_readout = torch::einsum("bk,bkd->bd", {readout_weights.squeeze(-1), node_states_b});

        return torch::matmul(final_readout, w_motor_out.t());
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward_with_diagnostics(
        torch::Tensor x_sensory,
        torch::Tensor context_chain = torch::Tensor(),
        torch::Tensor op_first_embed = torch::Tensor(),
        int64_t thinking_steps = 4) {
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

        torch::Tensor init_routing_weights;
        if (op_first_embed.defined() && op_first_embed.numel() > 0) {
            auto active_w_init = w_init_route.slice(1, 0, K);
            auto init_logits = torch::matmul(op_first_embed, active_w_init);
            init_routing_weights = torch::softmax(init_logits, -1);
        } else {
            init_routing_weights = torch::zeros({B, K}, torch::TensorOptions().device(device));
            init_routing_weights.select(1, 0).fill_(1.0f);
        }

        auto active_w_route = w_route.slice(0, 0, K).slice(1, 0, K);
        auto active_signatures = organelle_signatures.slice(0, 0, K);

        std::vector<torch::Tensor> gate_factors;
        for (int64_t j = 0; j < K; ++j) {
            gate_factors.push_back(get_node_gate(j).abs());
        }
        auto gates_tensor = torch::stack(gate_factors, 0);
        auto inactive_mask = (gates_tensor < 1e-4f).unsqueeze(0).unsqueeze(0);

        std::vector<torch::Tensor> step_routing_matrices;

        for (int64_t step = 0; step < thinking_steps; ++step) {
            auto current_node_states_b = node_states.permute({1, 0, 2});

            torch::Tensor active_h_sum = torch::zeros({B, dim}, torch::TensorOptions().device(device));
            for (int64_t j = 0; j < K; ++j) {
                active_h_sum = active_h_sum + get_node_gate(j) * current_node_states_b.select(1, j);
            }

            torch::Tensor curr_step_ctx;
            if (context_chain.defined() && context_chain.numel() > 0) {
                if (context_chain.dim() == 3) {
                    int64_t s_idx = std::min(step, context_chain.size(1) - 1);
                    curr_step_ctx = context_chain.select(1, s_idx);
                } else if (context_chain.dim() == 2) {
                    curr_step_ctx = context_chain;
                }
            }

            if (curr_step_ctx.defined() && curr_step_ctx.numel() > 0) {
                active_h_sum = active_h_sum + curr_step_ctx;
            }

            auto Q_step = torch::matmul(active_h_sum, w_query_step);
            auto sig_routing_logits = torch::matmul(Q_step, active_signatures.t()) * (1.0f / std::sqrt((float)dim));

            auto step_routing_logits = active_w_route.unsqueeze(0) + sig_routing_logits.unsqueeze(1);
            step_routing_logits = step_routing_logits.masked_fill(inactive_mask, -1e4f);
            auto step_routing_matrix = torch::softmax(step_routing_logits, 2);
            step_routing_matrices.push_back(step_routing_matrix);

            auto aggregated_inputs_b = torch::einsum("bik,bid->bkd", {step_routing_matrix, current_node_states_b});
            auto aggregated_inputs = aggregated_inputs_b.permute({1, 0, 2});

            if (step == 0) {
                auto init_weights_k = init_routing_weights.t().unsqueeze(-1);
                aggregated_inputs = aggregated_inputs + init_weights_k * sensory_in.unsqueeze(0);
            }

            std::vector<torch::Tensor> new_states;
            for (int64_t j = 0; j < K; ++j) {
                auto raw_out = node_ops[j]->forward(aggregated_inputs[j]);
                auto graft_gate = get_node_gate(j);
                auto grafted_out = graft_gate * raw_out;
                new_states.push_back(grafted_out);
            }
            node_states = torch::stack(new_states, 0);
        }

        persistent_node_states = node_states.detach();
        has_persistent_states = true;

        auto node_states_b = node_states.permute({1, 0, 2});
        torch::Tensor readout_logits;
        if (context_chain.defined() && context_chain.numel() > 0) {
            torch::Tensor final_step_ctx;
            if (context_chain.dim() == 3) {
                final_step_ctx = context_chain.select(1, context_chain.size(1) - 1);
            } else {
                final_step_ctx = context_chain;
            }
            auto q_readout = torch::matmul(final_step_ctx, w_readout_ctx_proj).unsqueeze(1);
            readout_logits = torch::matmul(node_states_b, q_readout.transpose(-1, -2)) * (1.0f / std::sqrt((float)dim));
        } else {
            readout_logits = torch::zeros({B, K, 1}, torch::TensorOptions().device(device));
        }
        readout_logits = readout_logits.masked_fill((gates_tensor < 1e-4f).unsqueeze(0).unsqueeze(-1), -1e4f);
        auto readout_weights = torch::softmax(readout_logits, 1);
        auto final_readout = torch::einsum("bk,bkd->bd", {readout_weights.squeeze(-1), node_states_b});

        auto out = torch::matmul(final_readout, w_motor_out.t());
        torch::Tensor all_step_routings = torch::stack(step_routing_matrices, 1); // [B, steps, K_src, K_tgt]
        return std::make_tuple(out, readout_weights.squeeze(-1), init_routing_weights, all_step_routings);
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
        .def(py::init<int64_t, std::string, float, float, float, float, int64_t, bool, float>(),
             py::arg("dim") = 128, py::arg("device") = "cpu",
             py::arg("fast_min_decay") = 0.05f, py::arg("fast_max_decay") = 0.5f,
             py::arg("slow_min_decay") = 0.0005f, py::arg("slow_max_decay") = 0.01f,
             py::arg("num_hopfield_basins") = 256,
             py::arg("use_hopfield_snapping") = true,
             py::arg("hopfield_beta") = 12.0f)
        .def("forward", [](EndogenousThetaGammaPACImpl& self, torch::Tensor x, std::optional<torch::Tensor> free_energy, std::optional<torch::Tensor> init_h_fast, std::optional<torch::Tensor> init_h_slow) {
            return self.forward(x,
                                free_energy.has_value() ? free_energy.value() : torch::Tensor(),
                                init_h_fast.has_value() ? init_h_fast.value() : torch::Tensor(),
                                init_h_slow.has_value() ? init_h_slow.value() : torch::Tensor());
        }, py::arg("x"), py::arg("free_energy") = py::none(), py::arg("init_h_fast") = py::none(), py::arg("init_h_slow") = py::none())
        .def("__call__", [](EndogenousThetaGammaPACImpl& self, torch::Tensor x, std::optional<torch::Tensor> free_energy, std::optional<torch::Tensor> init_h_fast, std::optional<torch::Tensor> init_h_slow) {
            return self.forward(x,
                                free_energy.has_value() ? free_energy.value() : torch::Tensor(),
                                init_h_fast.has_value() ? init_h_fast.value() : torch::Tensor(),
                                init_h_slow.has_value() ? init_h_slow.value() : torch::Tensor());
        }, py::arg("x"), py::arg("free_energy") = py::none(), py::arg("init_h_fast") = py::none(), py::arg("init_h_slow") = py::none());

    py::class_<TriScaleHierarchicalPACImpl, torch::nn::Module, std::shared_ptr<TriScaleHierarchicalPACImpl>>(m, "TriScaleHierarchicalPAC")
        .def(py::init<int64_t, std::string, float, float, float, float, float, float, int64_t, bool, float>(),
             py::arg("dim") = 128, py::arg("device") = "cpu",
             py::arg("gamma_min_decay") = 0.05f, py::arg("gamma_max_decay") = 0.50f,
             py::arg("theta_min_decay") = 0.005f, py::arg("theta_max_decay") = 0.05f,
             py::arg("delta_min_decay") = 0.0001f, py::arg("delta_max_decay") = 0.001f,
             py::arg("num_hopfield_basins") = 256,
             py::arg("use_hopfield_snapping") = true,
             py::arg("hopfield_beta") = 12.0f)
        .def("forward", [](TriScaleHierarchicalPACImpl& self, torch::Tensor x, std::optional<torch::Tensor> free_energy,
                           std::optional<torch::Tensor> init_h_gamma, std::optional<torch::Tensor> init_h_theta, std::optional<torch::Tensor> init_h_delta) {
            return self.forward(x,
                                free_energy.has_value() ? free_energy.value() : torch::Tensor(),
                                init_h_gamma.has_value() ? init_h_gamma.value() : torch::Tensor(),
                                init_h_theta.has_value() ? init_h_theta.value() : torch::Tensor(),
                                init_h_delta.has_value() ? init_h_delta.value() : torch::Tensor());
        }, py::arg("x"), py::arg("free_energy") = py::none(),
           py::arg("init_h_gamma") = py::none(), py::arg("init_h_theta") = py::none(), py::arg("init_h_delta") = py::none())
        .def("__call__", [](TriScaleHierarchicalPACImpl& self, torch::Tensor x, std::optional<torch::Tensor> free_energy,
                            std::optional<torch::Tensor> init_h_gamma, std::optional<torch::Tensor> init_h_theta, std::optional<torch::Tensor> init_h_delta) {
            return self.forward(x,
                                free_energy.has_value() ? free_energy.value() : torch::Tensor(),
                                init_h_gamma.has_value() ? init_h_gamma.value() : torch::Tensor(),
                                init_h_theta.has_value() ? init_h_theta.value() : torch::Tensor(),
                                init_h_delta.has_value() ? init_h_delta.value() : torch::Tensor());
        }, py::arg("x"), py::arg("free_energy") = py::none(),
           py::arg("init_h_gamma") = py::none(), py::arg("init_h_theta") = py::none(), py::arg("init_h_delta") = py::none());
    py::class_<ParallelOperatorBankImpl, torch::nn::Module, std::shared_ptr<ParallelOperatorBankImpl>>(m, "ParallelOperatorBank")
        .def(py::init<int64_t, int64_t, int64_t>(), py::arg("dim") = 256, py::arg("state_dim") = 128, py::arg("num_operators") = 8)
        .def("compute_operators", &ParallelOperatorBankImpl::compute_operators);

    py::class_<ContinuousHopfieldMemoryImpl, torch::nn::Module, std::shared_ptr<ContinuousHopfieldMemoryImpl>>(m, "ContinuousHopfieldMemory")
        .def(py::init<int64_t, int64_t, std::string, int64_t>(), py::arg("dim") = 256, py::arg("num_basins") = 32, py::arg("device_str") = "cpu", py::arg("max_episodes") = 256)
        .def("record_somatic_episode", &ContinuousHopfieldMemoryImpl::record_somatic_episode, py::arg("context_key"), py::arg("action_key"), py::arg("valence"))
        .def("relax_with_repulsion", &ContinuousHopfieldMemoryImpl::relax_with_repulsion, py::arg("context_t"), py::arg("action_t"), py::arg("beta") = 8.0f)
        .def_readonly("active_episodes", &ContinuousHopfieldMemoryImpl::active_episodes)
        .def_readonly("max_episodes", &ContinuousHopfieldMemoryImpl::max_episodes)
        .def_readonly("valences", &ContinuousHopfieldMemoryImpl::valences)
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

    py::class_<TsodyksMarkramSynapticDepressionOpImpl, torch::nn::Module, std::shared_ptr<TsodyksMarkramSynapticDepressionOpImpl>>(m, "TsodyksMarkramSynapticDepressionOp")
        .def(py::init<int64_t, std::string, float, float>(), py::arg("dim"), py::arg("device_str") = "cpu", py::arg("tau_rec") = 8.0f, py::arg("u_depress") = 0.5f)
        .def("reset_state", &TsodyksMarkramSynapticDepressionOpImpl::reset_state)
        .def("get_vesicle_resource", &TsodyksMarkramSynapticDepressionOpImpl::get_vesicle_resource)
        .def("forward", &TsodyksMarkramSynapticDepressionOpImpl::forward)
        .def("__call__", &TsodyksMarkramSynapticDepressionOpImpl::forward);

    py::class_<SlotMemoryOpImpl, torch::nn::Module, std::shared_ptr<SlotMemoryOpImpl>>(m, "SlotMemoryOp")
        .def(py::init<int64_t, std::string, int64_t>(), py::arg("dim"), py::arg("device_str") = "cpu", py::arg("num_slots") = 4)
        .def("reset_state", &SlotMemoryOpImpl::reset_state)
        .def("get_memory_slots", &SlotMemoryOpImpl::get_memory_slots)
        .def("forward", &SlotMemoryOpImpl::forward)
        .def("__call__", &SlotMemoryOpImpl::forward);

    py::class_<NonLinearTransformOpImpl, torch::nn::Module, std::shared_ptr<NonLinearTransformOpImpl>>(m, "NonLinearTransformOp")
        .def(py::init<int64_t, std::string>(), py::arg("dim"), py::arg("device_str") = "cpu")
        .def("forward", &NonLinearTransformOpImpl::forward)
        .def("__call__", &NonLinearTransformOpImpl::forward);

    py::class_<DynamicMorphicGraphImpl, torch::nn::Module, std::shared_ptr<DynamicMorphicGraphImpl>>(m, "DynamicMorphicGraph")
        .def(py::init<int64_t, std::string, int64_t>(), py::arg("dim") = 128, py::arg("device_str") = "cpu", py::arg("max_nodes") = 128)
        .def_readonly("k_nodes", &DynamicMorphicGraphImpl::k_nodes)
        .def("add_node", &DynamicMorphicGraphImpl::add_node, py::arg("name"), py::arg("op_type"), py::arg("is_core") = false, py::arg("initial_alpha") = 0.0f)
        .def("lock_node", &DynamicMorphicGraphImpl::lock_node, py::arg("idx"), py::arg("lock_value") = 1.0f)
        .def("duplicate_node", &DynamicMorphicGraphImpl::duplicate_node, py::arg("src_idx"), py::arg("new_name"), py::arg("initial_alpha") = 0.0f)
        .def("prune_inactive_nodes", &DynamicMorphicGraphImpl::prune_inactive_nodes, py::arg("threshold") = 0.02f)
        .def("prune_relative_darwinism", &DynamicMorphicGraphImpl::prune_relative_darwinism, py::arg("relative_threshold_factor") = 0.15f)
        .def("reset_state", &DynamicMorphicGraphImpl::reset_state)
        .def("forward", &DynamicMorphicGraphImpl::forward, py::arg("x_sensory"), py::arg("context_chain") = torch::Tensor(), py::arg("op_first_embed") = torch::Tensor(), py::arg("thinking_steps") = 4)
        .def("__call__", &DynamicMorphicGraphImpl::forward, py::arg("x_sensory"), py::arg("context_chain") = torch::Tensor(), py::arg("op_first_embed") = torch::Tensor(), py::arg("thinking_steps") = 4)
        .def("forward_adaptive", &DynamicMorphicGraphImpl::forward_adaptive, py::arg("x_sensory"), py::arg("context_chain") = torch::Tensor(), py::arg("op_first_embed") = torch::Tensor(), py::arg("max_thinking_steps") = 8, py::arg("halt_threshold") = 0.8f, py::arg("epsilon_halt") = 1e-3f)
        .def("forward_with_diagnostics", &DynamicMorphicGraphImpl::forward_with_diagnostics, py::arg("x_sensory"), py::arg("context_chain") = torch::Tensor(), py::arg("op_first_embed") = torch::Tensor(), py::arg("thinking_steps") = 4)
        .def("get_topology_manifest", &DynamicMorphicGraphImpl::get_topology_manifest)
        .def("get_methylation_locks", &DynamicMorphicGraphImpl::get_methylation_locks)
        .def("set_methylation_locks", &DynamicMorphicGraphImpl::set_methylation_locks, py::arg("locks"))
        .def("named_parameters_map", [](std::shared_ptr<DynamicMorphicGraphImpl> m) {
            return m->get_active_parameters_map();
        });
}
