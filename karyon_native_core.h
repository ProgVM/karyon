// karyon_native_core.h — Pure C++20 Sovereign Unified Substrate Core Engine v7.1
// Standard: KEP v20.0 Sovereign Master (Principle 2 to 27 Compliant)
//
// Advancements in v7.1 (Supersonic Analytical Architecture):
// 1. Explicit AVX2 SIMD FMA intrinsics for dense linear & bilinear GEMV (_mm256_fmadd_ps)
// 2. Analytical Adjoint Gradient for continuous motor actuation (replacing numerical finite-difference perturbations)
// 3. Reflex-Modulated Thinking: Mental rollout triggered on high Free Energy, fast reflex on stable flow
// 4. Zero-allocation memory layout across all operations

#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <random>
#include <algorithm>
#include <span>
#include <immintrin.h>

namespace karyon {

struct CoreConfig {
    size_t state_dim = 256;             // Dimension of unified continuous state x_t in R^D
    size_t action_dim = 4;              // Dimension of continuous motor control u_t in R^M
    size_t num_primitives = 2;          // Number of continuous super-primitives in stack
    size_t min_horizon_k = 1;           // Minimum mental trajectory simulation depth
    size_t max_horizon_k = 4;           // Maximum mental trajectory simulation depth
    float plasticity_rate = 0.02f;      // Baseline rate of autonomous structural plasticity
    float noradrenaline_decay = 0.03f;  // Decay rate of somatic arousal
    float da_baseline = 0.50f;          // Baseline dopaminergic gain
    size_t hamiltonian_steps = 2;       // Analytical gradient descent steps along motor field
};

class CoreScratchpad {
public:
    std::vector<float> buf_a;
    std::vector<float> buf_b;
    std::vector<float> buf_sim;
    std::vector<float> buf_grad;
    std::vector<float> buf_quad;
    std::vector<float> buf_adjoint;

    explicit CoreScratchpad(size_t state_dim = 256, size_t action_dim = 16) {
        buf_a.resize(state_dim, 0.0f);
        buf_b.resize(state_dim, 0.0f);
        buf_sim.resize(state_dim, 0.0f);
        buf_grad.resize(action_dim, 0.0f);
        buf_quad.resize(state_dim, 0.0f);
        buf_adjoint.resize(state_dim, 0.0f);
    }
};

class ContinuousSuperPrimitive {
public:
    size_t id;
    size_t dim;

    float alpha_lin;
    float alpha_bilin;
    float alpha_rec;
    float alpha_res;

    float epigenetic_lock;
    float cumulative_causal_utility;

    alignas(32) std::vector<float> W_lin;
    alignas(32) std::vector<float> W_bilin;
    alignas(32) std::vector<float> state;

    ContinuousSuperPrimitive(size_t p_id, size_t d, std::mt19937& rng);

    // AVX2 vectorized forward pass
    void forward(std::span<const float> in, std::span<float> out, std::span<float> scratch_quad);

    void update_plasticity(std::span<const float> in,
                           std::span<const float> target,
                           float plasticity_rate);

    void adapt_coordinates(float dF_dalpha_lin, float dF_dalpha_bilin, float plasticity_rate);
    void update_epigenetic_lock(float baseline_fe_reduction);
};

struct CoreTelemetry {
    std::vector<float> action_vector;
    float free_energy;
    float sensory_prediction_error;
    float noradrenaline_arousal;
    size_t mental_horizon_k;
    float lock_ratio;
};

class KaryonSovereignCore {
public:
    explicit KaryonSovereignCore(const CoreConfig& config = CoreConfig());

    // Analytical Adjoint Least-Action motor control (O(1) simulation instead of O(M * steps))
    void decide_continuous_action(std::span<const float> current_state,
                                  std::span<const float> target_goal,
                                  std::span<float> action_out);

    CoreTelemetry step_environment(std::span<const float> current_state,
                                   std::span<const float> target_goal,
                                   std::span<const float> next_state_observed,
                                   std::span<const float> action_taken);

    size_t decide_action(const std::vector<float>& current_state,
                         const std::vector<float>& target_goal);

    CoreTelemetry step_environment(const std::vector<float>& current_state,
                                   const std::vector<float>& target_goal,
                                   const std::vector<float>& next_state_observed,
                                   size_t discrete_action);

    float get_noradrenaline() const { return noradrenaline_arousal_; }
    size_t get_current_horizon() const { return current_horizon_k_; }
    float get_average_epigenetic_lock() const;

private:
    CoreConfig config_;
    std::mt19937 rng_;
    std::vector<std::unique_ptr<ContinuousSuperPrimitive>> primitives_;

    alignas(32) std::vector<float> W_motor_;

    float noradrenaline_arousal_;
    float dopamine_vigor_;
    float somatic_energy_;
    size_t current_horizon_k_;

    CoreScratchpad scratchpad_;

    void adapt_endogenous_horizon(float fe_error);
};

} // namespace karyon
