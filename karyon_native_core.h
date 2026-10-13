// karyon_native_core.h
// Karyon Native Core v7.6 - Zero-Allocation Autonomous C++20 Cybernetic Substrate
// Standard: KEP v20.0 Sovereign Master (Principles 2, 3, 4, 5, 6, 7, 27, 28)
//
// Key Advancements in v7.6:
// 1. Direct Aversive Reflex Loop & Action-Valence Landscape:
//    - Action-specific continuous pain/threat expectation vector: Q_valence_[action_dim x sensory_dim].
//    - Real-time inhibitory synaptic depression on actions causing negative valence.
// 2. Valence-Coupled Mental Trajectory Evaluation:
//    - evaluate_trajectory incorporates predicted valence penalty:
//      Cost = Pragmatic_Dist - Epistemic_Novelty + Lambda_Pain * Expected_Aversion(a, s)
//    - Direct reflex veto prevents taking actions with high predicted pain in given sensory context!
// 3. Positive Reinforcement Hebbian Motor Facilitation:
//    - Actions with positive valence (+1.0) are amplified and epigenetically protected.

#pragma once

#include <vector>
#include <memory>
#include <random>
#include <span>
#include <immintrin.h>
#include <cstddef>
#include <cmath>
#include <algorithm>

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
    float stagnation_decay = 0.03f;       // Continuous exponential decay of stagnation field
    float aversive_weight = 5.0f;         // Weight of pain expectation in mental simulation
};

class ContinuousSuperPrimitive {
public:
    size_t id;
    size_t dim;

    float alpha_lin;
    float alpha_bilin;
    float alpha_rec;
    float tau_time;
    float epigenetic_lock;
    float birth_gain;

    alignas(32) std::vector<float> W_lin;
    alignas(32) std::vector<float> W_bilin;
    alignas(32) std::vector<float> W_rec;
    std::vector<float> state_trace;

    ContinuousSuperPrimitive(size_t pid, size_t d, std::mt19937& rng);

    void forward(std::span<const float> x_in, std::span<float> h_out, std::span<float> quad_buf);
    void update_plasticity_avx2(std::span<const float> pre, std::span<const float> post, float lr, std::span<float> quad_buf);
    void mature_birth_gain(float rate = 0.005f);
    void update_epigenetic_lock(float utility);
    void adapt_coordinates(float d_lin, float d_bilin, float d_rec);
};

struct CoreTelemetry {
    float free_energy = 0.0f;
    float sensory_prediction_error = 0.0f;
    float noradrenaline = 0.0f;
    float dopamine = 0.0f;
    float somatic_energy = 0.0f;
    float stagnation_potential = 0.0f;
    float expected_valence = 0.0f;
    size_t active_primitives_count = 0;
    size_t horizon_k = 1;
    float lock_ratio = 0.0f;
    bool event_sprouted = false;
    bool event_pruned = false;
    size_t action_discrete_chosen = 0;
};

class KaryonSovereignCore {
public:
    explicit KaryonSovereignCore(const CoreConfig& config);
    ~KaryonSovereignCore() = default;

    // 1. Continuous Analog Motor Control
    void decide_continuous_action(std::span<const float> current_state,
                                  std::span<const float> target_goal,
                                  std::span<float> action_out);

    // 2. Discrete Action Decision via Mental Trajectory Rollout Sandbox with Epistemic & Valence Fields
    size_t decide_action(const std::vector<float>& current_state,
                         const std::vector<float>& target_goal);

    // Step with Continuous Action (with external valence / mentorship scalar)
    CoreTelemetry step_environment(std::span<const float> current_state,
                                   std::span<const float> target_goal,
                                   std::span<const float> next_state_observed,
                                   std::span<const float> action_taken,
                                   float external_valence = 0.0f);

    // Step with Discrete Action (with external valence / mentorship scalar)
    CoreTelemetry step_environment(const std::vector<float>& current_state,
                                   const std::vector<float>& target_goal,
                                   const std::vector<float>& next_state_observed,
                                   size_t discrete_action,
                                   float external_valence = 0.0f);

    float get_noradrenaline() const { return noradrenaline_arousal_; }
    float get_dopamine() const { return dopamine_vigor_; }
    float get_stagnation_potential() const { return stagnation_potential_; }
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
    std::vector<float> W_action_valence_;   // [action_dim x sensory_dim] associative pain/threat landscape

    float noradrenaline_arousal_;
    float dopamine_vigor_;
    float somatic_energy_;
    float stagnation_potential_;           // Continuous non-static stagnation field trace
    size_t current_horizon_k_;
    size_t ticks_since_last_sprout_;
    float moving_fe_ema_;

    // Continuous running barycenter / velocity trace of sensory manifold
    std::vector<float> continuous_sensory_ema_;

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

// C-ABI Export Functions
extern "C" {
    void* karyon_core_create(const CoreConfig* cfg);
    void karyon_core_destroy(void* handle);
    void karyon_core_decide_action(void* handle, const float* current_state, const float* target_goal, float* action_out);
    size_t karyon_core_decide_discrete(void* handle, const float* current_state, const float* target_goal);
    void karyon_core_step(void* handle,
                          const float* current_state,
                          const float* target_goal,
                          const float* next_state_observed,
                          const float* action_taken,
                          float external_valence,
                          CoreTelemetry* telem_out);
}

} // namespace karyon
