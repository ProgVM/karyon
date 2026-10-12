// karyon_native_core.h — Pure C++20 Sovereign Unified Substrate Core Engine v6.0
// Standard: KEP v20.0 Sovereign Master (Principle 2 to 27 Compliant)
// Fully Integrated Breakthroughs (EXP-470 to EXP-479):
// 1. Continuous Tensor-Field Super-Primitives with 6 Coordinate Vector Dynamics
// 2. Continuous Causal Credit Assignment & Self-Synthesizing Plasticity
// 3. Epigenetic Smooth Sprouting & Invariant Locking (tanh(alpha_epi) -> 0 at birth)
// 4. Closed-Loop Sensorimotor Coupled Agency & Unified Efference-Afference Manifold
// 5. Hamiltonian Principle of Least Action in Latent Space (Mental Sandbox Rollout)
// 6. Dynamic Endogenous Horizon Scaling K(t) ~ Ashby Somatic Arousal (NA_t, F_t)

#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <random>
#include <algorithm>
#include <immintrin.h>

namespace karyon {

// ---------------------------------------------------------------------------
// 1. Unified Substrate Core Configuration
// ---------------------------------------------------------------------------
struct CoreConfig {
    size_t state_dim = 16;              // Continuous hidden state dimension
    size_t num_primitives = 2;          // Dynamic continuous super-primitives count
    size_t num_actions = 4;             // Discrete action efference space
    size_t min_horizon_k = 1;           // Minimum mental rollout depth (reflex)
    size_t max_horizon_k = 8;           // Maximum mental rollout depth (deep reflection)

    float base_alpha_lin = 0.80f;       // Linear transformation weight
    float base_alpha_bilin = 0.20f;     // Bilinear gating weight
    float base_alpha_rec = 0.85f;       // Temporal integration timescale
    float base_alpha_res = 0.30f;       // Residual highway weight

    float epigenetic_threshold = 0.02f; // Causal utility threshold to trigger epigenetic invariant lock
    float plasticity_rate = 0.02f;      // Structural and parameter adaptation rate
    float discount_gamma = 0.85f;       // Trajectory free energy temporal discount
};

// ---------------------------------------------------------------------------
// 2. Comprehensive Step Telemetry
// ---------------------------------------------------------------------------
struct CoreTelemetry {
    float free_energy = 0.0f;
    float rolling_free_energy = 0.0f;
    float noradrenaline_arousal = 0.0f;
    size_t mental_horizon_k = 1;
    size_t action_executed = 0;
    float goal_distance = 0.0f;
    bool goal_reached = false;
    size_t goals_achieved = 0;
    float primary_primitive_lock = 0.0f;
    float primary_causal_utility = 0.0f;
};

// ---------------------------------------------------------------------------
// 3. Continuous Tensor-Field Super-Primitive (KEP Principle 3 & 4)
// ---------------------------------------------------------------------------
struct ContinuousSuperPrimitive {
    size_t id;
    size_t dim;

    // Continuous Equation Coordinates
    float alpha_lin;
    float alpha_bilin;
    float alpha_rec;
    float alpha_res;

    // Epigenetic Protection & Causal History (Principle 5 & 6)
    float epigenetic_lock;
    float cumulative_causal_utility;

    // Parameter Tensors
    std::vector<float> W_lin;
    std::vector<float> W_bilin;
    std::vector<float> bias;
    std::vector<float> state;

    // Transient Cache
    std::vector<float> last_x;
    std::vector<float> last_out;

    ContinuousSuperPrimitive(size_t p_id, size_t d, std::mt19937& rng);

    std::vector<float> forward(const std::vector<float>& x);
    void causal_update(const std::vector<float>& error, float global_fe_reduction, float plasticity_rate);
};

// ---------------------------------------------------------------------------
// 4. Karyon-CoRE Sovereign Autonomous Agent Engine (v6.0 Sovereign)
// ---------------------------------------------------------------------------
class KaryonSovereignCore {
public:
    explicit KaryonSovereignCore(const CoreConfig& config = CoreConfig());
    ~KaryonSovereignCore() = default;

    void reset();

    // Endogenous Mental Rollout & Action Selection (EXP-478 & EXP-479)
    size_t decide_action(const std::vector<float>& current_state,
                         const std::vector<float>& target_goal);

    // Online Closed-Loop Perception-Action Learning Step
    CoreTelemetry step_environment(const std::vector<float>& current_state,
                                   const std::vector<float>& target_goal,
                                   const std::vector<float>& next_state_observed,
                                   size_t action_taken);

    // Mental Sandbox Simulation
    float evaluate_trajectory(size_t action_id,
                              const std::vector<float>& s0,
                              const std::vector<float>& target_goal,
                              size_t horizon_k);

    const CoreConfig& config() const { return config_; }
    float get_noradrenaline() const { return noradrenaline_arousal_; }
    size_t get_current_horizon() const { return current_horizon_k_; }

private:
    CoreConfig config_;
    std::mt19937 rng_{42};

    std::vector<std::unique_ptr<ContinuousSuperPrimitive>> primitives_;
    std::vector<float> action_delta_table_; // [num_actions x dim] Physical efference manifold

    // Ashby Somatic Interoception
    float rolling_free_energy_ = 0.10f;
    float noradrenaline_arousal_ = 0.05f;
    size_t current_horizon_k_ = 1;
    size_t total_goals_achieved_ = 0;

    size_t compute_endogenous_horizon();
};

} // namespace karyon
