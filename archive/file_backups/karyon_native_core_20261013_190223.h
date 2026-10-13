// karyon_native_core.h — Pure C++20 Sovereign Unified Substrate Core Engine v7.4
// Standard: KEP v20.0 Sovereign Master (Principles 2 to 27 Compliant)
//
// Advancements in v7.4 (Mental Sandbox Trajectory Rollout & True Quality):
// 1. Dual Control Modality:
//    - Real-world discrete action planning via Endogenous Mental Horizon Rollout (evaluate_trajectory)
//    - High-speed continuous analog adjoint steering for high-dim video manifolds
// 2. Continuous Super-Primitives with AVX2-FMA Vectorized Plasticity and Net2Net Smooth Sprouting
// 3. Bidirectional Laminar Field Compression (256D <-> 32D)
// 4. Ground-truth Goal Attraction & Active Inference Trajectory Selection

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
    size_t sensory_dim = 256;             // Sensory observation dimension
    size_t latent_dim = 32;               // High-speed compressed manifold dimension
    size_t action_dim = 4;                // Action count (discrete actions or continuous thrusters)
    size_t initial_primitives = 2;         // Initial super-primitives in stack
    size_t max_primitives = 6;            // Ceiling for dynamic autopoiesis
    size_t min_horizon_k = 1;             // Minimum mental simulation depth
    size_t max_horizon_k = 6;             // Maximum mental simulation depth
    float plasticity_rate = 0.02f;        // Base rate of structural plasticity
    float noradrenaline_decay = 0.02f;    // Somatic arousal decay
    float da_baseline = 0.50f;            // Baseline dopaminergic vigor
    float discount_gamma = 0.90f;         // Horizon temporal discounting
    float sprouting_fe_threshold = 0.15f; // Stress threshold triggering morphogenetic sprouting
    size_t sprouting_cooldown_ticks = 1000;
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
    float birth_gain;

    alignas(32) std::vector<float> W_lin;
    alignas(32) std::vector<float> W_bilin;
    alignas(32) std::vector<float> state;

    ContinuousSuperPrimitive(size_t p_id, size_t d, std::mt19937& rng, bool sprouted_zero_shock = false);

    void forward(std::span<const float> in, std::span<float> out, std::span<float> scratch_quad);
    void update_plasticity_avx2(std::span<const float> in,
                                std::span<const float> target,
                                float plasticity_rate,
                                std::span<const float> scratch_quad);
    void adapt_coordinates(float dF_dalpha_lin, float dF_dalpha_bilin, float plasticity_rate);
    void update_epigenetic_lock(float baseline_fe_reduction);
    void mature_birth_gain(float rate = 0.005f);
};

struct CoreTelemetry {
    std::vector<float> action_vector;
    size_t action_discrete_chosen = 0;
    float free_energy = 0.0f;
    float sensory_prediction_error = 0.0f;
    float noradrenaline_arousal = 0.0f;
    size_t mental_horizon_k = 1;
    float lock_ratio = 0.0f;
    size_t active_primitives_count = 0;
    bool event_sprouted = false;
    bool event_pruned = false;
};

class KaryonSovereignCore {
public:
    explicit KaryonSovereignCore(const CoreConfig& config = CoreConfig());

    // 1. Continuous Analog Motor Control
    void decide_continuous_action(std::span<const float> current_state,
                                  std::span<const float> target_goal,
                                  std::span<float> action_out);

    // 2. Discrete Action Decision via Mental Trajectory Rollout Sandbox
    size_t decide_action(const std::vector<float>& current_state,
                         const std::vector<float>& target_goal);

    // Step with Continuous Action
    CoreTelemetry step_environment(std::span<const float> current_state,
                                   std::span<const float> target_goal,
                                   std::span<const float> next_state_observed,
                                   std::span<const float> action_taken);

    // Step with Discrete Action
    CoreTelemetry step_environment(const std::vector<float>& current_state,
                                   const std::vector<float>& target_goal,
                                   const std::vector<float>& next_state_observed,
                                   size_t discrete_action);

    float get_noradrenaline() const { return noradrenaline_arousal_; }
    size_t get_current_horizon() const { return current_horizon_k_; }
    float get_average_epigenetic_lock() const;
    size_t get_primitives_count() const { return primitives_.size(); }

    bool trigger_morphogenetic_sprout();
    bool prune_least_causal_primitive();

private:
    CoreConfig config_;
    std::mt19937 rng_;
    std::vector<std::unique_ptr<ContinuousSuperPrimitive>> primitives_;

    alignas(32) std::vector<float> W_sensory_down_;
    alignas(32) std::vector<float> W_sensory_up_;
    alignas(32) std::vector<float> W_motor_;
    std::vector<float> action_delta_table_; // [action_dim x sensory_dim] for trajectory rollout

    float noradrenaline_arousal_;
    float dopamine_vigor_;
    float somatic_energy_;
    size_t current_horizon_k_;
    size_t ticks_since_last_sprout_;
    float moving_fe_ema_;

    // Internal scratchpad
    std::vector<float> buf_sensory_in_;
    std::vector<float> buf_sensory_out_;
    std::vector<float> buf_latent_a_;
    std::vector<float> buf_latent_b_;
    std::vector<float> buf_quad_;
    std::vector<float> buf_adjoint_sensory_;

    void adapt_endogenous_horizon(float fe_error);
    void evaluate_autopoietic_morphogenesis(float fe_error);
    float evaluate_trajectory(size_t action_id,
                              const std::vector<float>& s0,
                              const std::vector<float>& target_goal,
                              size_t horizon_k);
};

} // namespace karyon
