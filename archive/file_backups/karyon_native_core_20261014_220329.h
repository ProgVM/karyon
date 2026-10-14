// karyon_native_core.h
// Karyon Native Core v8.0 - Autonomous C++20 Cybernetic Substrate with Sovereign Endogenous Hormones
// Standard: KEP v20.0 Sovereign Master (Principles 2, 3, 4, 5, 6, 7, 27, 28)
//
// Key Advancements in v8.0 (Sovereign Endogenous Hormones & Dynamic Endocrine Morphogenesis):
// 1. Eradication of Hardcoded Anthropomorphic Neurotransmitters:
//    - Replaces static 'noradrenaline' and 'dopamine' with a dynamic, sovereign vector field of hormones h in R^H.
// 2. Sovereign Endogenous Hormones (SovereignHormone):
//    - Each hormone possesses its own learned emission dynamics, decay time constant (gamma_k), basal setpoint,
//      and continuous sensitivity to sensory patterns, valence, and surprise.
// 3. Dynamic Receptor Receptive Fields (Multi-Effector Modulation):
//    - Effector properties (plasticity rate, trajectory horizon, motor vigor, threat sensitivity, stochastic exploration)
//      are endogenous projections of the sovereign hormone vector: Target = F(Omega_mod * h).
// 4. Endocrine Autopoiesis & Morphogenesis:
//    - Sprouting: Karyon autonomously synthesizes new hormone axes under persistent unresolved Free Energy.
//    - Pruning: Hormones that provide zero causal utility to Free Energy minimization decay and are pruned.
// 5. Zero-Allocation & AVX2-FMA Performance Guarantee.

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
    size_t initial_latent_dim = 16;       // Starting latent dimensionality
    size_t max_latent_dim = 64;           // Dimensional morphogenesis ceiling
    size_t action_dim = 4;                // Action count (discrete actions or continuous thrusters)
    size_t initial_primitives = 2;         // Initial super-primitives in stack
    size_t max_primitives = 6;            // Ceiling for dynamic autopoiesis
    size_t initial_hormones = 3;          // Initial count of sovereign hormones (default: 3)
    size_t max_hormones = 8;              // Ceiling for dynamic hormone autopoiesis
    size_t min_horizon_k = 1;             // Minimum mental simulation depth
    size_t max_horizon_k = 6;             // Maximum mental simulation depth
    float plasticity_rate = 0.02f;        // Base rate of structural plasticity
    float discount_gamma = 0.90f;         // Horizon temporal discounting
    float td_lambda = 0.70f;              // Eligibility trace decay for predictive threat assignment
    float sprouting_fe_threshold = 0.15f; // Stress threshold triggering morphogenetic sprouting
    size_t sprouting_cooldown_ticks = 1000;
    float stagnation_decay = 0.03f;       // Continuous exponential decay of stagnation field
    float aversive_weight = 20.0f;        // Base weight of predicted threat in mental simulation
    float peril_threshold = 0.35f;        // Deprecated fallback threshold
    float metric_curvature_scale = 10.0f; // Scale of Riemannian curvature barrier
};

// =========================================================================
// Continuous Sovereign Tensor-Field Super-Primitive (KEP Principle 3)
// Replaces discrete operator zoos with a single unified parametric continuous field:
// dh/dt = tau * [ (1 - gate) * (W_dyn * [x; h]) + gate * sigma(W_gate * x) * (W_dyn * x) - gamma * h ]
// By varying coordinates (gate, gamma, tau), this single operator continuously
// degenerates into pure linear projection, bilinear gating (SwiGLU-like),
// leaky integrator (SSD/Mamba-like), or non-linear attractor basin.
// =========================================================================
class ContinuousSuperPrimitive {
public:
    size_t id;
    size_t dim;

    // Continuous manifold coordinates governing operator state:
    float gate_bias;       // [0, 1]: Gating coordinate (0 = linear/recurrent flow, 1 = multiplicative gating)
    float decay_rate;      // [0.01, 1.0]: Leaky dissipation / attractor depth gamma
    float tau_time;        // Time-scale integration constant
    float epigenetic_lock; // Invariant protection factor [0, 1]
    float birth_gain;      // Net2Net identity scaling [0, 1]

    // Unified dynamic field weights:
    // W_field: [dim x (2 * dim)] projecting concatenated [x; h_prev]
    alignas(32) std::vector<float> W_field;
    // W_gate: [dim x dim] projecting gating field
    alignas(32) std::vector<float> W_gate;
    
    std::vector<float> state_trace; // h(t)

    ContinuousSuperPrimitive(size_t pid, size_t d, std::mt19937& rng);

    void forward(std::span<const float> x_in, std::span<float> h_out, std::span<float> quad_buf);
    void update_plasticity_avx2(std::span<const float> pre, std::span<const float> post, float lr, std::span<float> quad_buf);
    void mature_birth_gain(float rate = 0.005f);
    void update_epigenetic_lock(float utility);
    void adapt_manifold_coordinates(float d_gate, float d_decay, float lr);

    // Dynamic Dimensional Morphogenesis (Net2Net Latent Sprouting)
    void sprout_latent_dimension();
};

// =========================================================================
// Sovereign Endogenous Hormone (Autonomous Modulatory Axis)
// =========================================================================
struct SovereignHormone {
    size_t id = 0;
    float concentration = 0.5f;          // Current continuous concentration
    float basal_level = 0.5f;            // Baseline homeostatic attractor
    float decay_rate = 0.05f;            // Half-life decay constant gamma_k
    float valence_sensitivity = 0.2f;    // Response to positive/negative valence (+/-)
    float surprise_sensitivity = 0.3f;   // Response to sensory prediction error / Free Energy
    float causal_utility = 1.0f;         // Running trace of utility for pruning
    float birth_gain = 1.0f;             // Net2Net smooth birth scaling

    // Receptor projection weights to system effectors:
    float w_plasticity = 0.0f;           // Modulates learning rate
    float w_horizon = 0.0f;              // Modulates mental rollout horizon K
    float w_motor_vigor = 0.0f;          // Modulates motor drive/magnitude
    float w_threat_aversive = 0.0f;      // Modulates sensitivity to peril/danger
    float w_exploration = 0.0f;          // Modulates stochastic exploratory jitter

    // Sensory emission receptive field (weights responding to environmental features)
    alignas(32) std::vector<float> W_sensory_emission;

    SovereignHormone(size_t hid, size_t sensory_dim, std::mt19937& rng);
};
// =========================================================================
// Continuous Dynamic Episodic Memory Bank (Unbounded Non-Parametric Memory)
// Implements sparse, dynamically expanding topological memory slots:
// M = {(k_i, v_i, u_i, n_i, t_i)}
// with AVX2 vectorized associative cosine retrieval, novelty detection,
// and inhibition-of-return to permanently solve deep non-convex mazes.
// =========================================================================
struct EpisodicMemorySlot {
    std::vector<float> key;       // Latent spatial/sensory coordinate k_i
    std::vector<float> value;     // Consequence vector v_i (action outcomes, valence, threat)
    float utility;                // Epigenetic protection / causal importance u_i
    float visit_count;            // Familiarity / visit frequency n_i
    uint64_t last_accessed_step;  // Recency timestamp t_i
};

class DynamicEpisodicMemory {
public:
    size_t key_dim;
    size_t value_dim;
    float similarity_threshold;   // Radius for clustering new memory slots
    std::vector<EpisodicMemorySlot> slots;

    DynamicEpisodicMemory(size_t k_dim, size_t v_dim, float sim_thresh = 0.92f)
        : key_dim(k_dim), value_dim(v_dim), similarity_threshold(sim_thresh) {
        slots.reserve(1024); // Zero-allocation buffer pre-reserve
    }

    // Dynamic Net2Net dimensional sprouting support
    void sprout_dimension() {
        key_dim++;
        for (auto& slot : slots) {
            slot.key.push_back(0.0f);
        }
    }

    // Query associative memory with AVX2 dot product:
    // Returns {best_similarity, best_slot_index, familiarity, read_value}
    struct QueryResult {
        float max_sim = -1.0f;
        int best_idx = -1;
        float familiarity = 0.0f;
        std::vector<float> retrieved_value;
    };

    QueryResult query(const std::vector<float>& query_key, float* scratch_read = nullptr) const;

    // Record or update memory trace at current state
    void record(const std::vector<float>& key,
                const std::vector<float>& value,
                float utility,
                uint64_t current_step);

    // Compute Inhibition-of-Return penalty for a candidate next state
    float compute_inhibition_of_return(const std::vector<float>& candidate_key) const;

    size_t size() const { return slots.size(); }
    void clear() { slots.clear(); }
};

struct CoreTelemetry {
    float free_energy = 0.0f;
    float sensory_prediction_error = 0.0f;
    float primary_hormone = 0.0f;         // First hormone concentration (replaces fixed noradrenaline)
    float secondary_hormone = 0.0f;       // Second hormone concentration (replaces fixed dopamine)
    float effective_plasticity = 0.0f;    // Endogenously modulated plasticity
    float effective_motor_vigor = 0.0f;   // Endogenously modulated vigor
    float somatic_energy = 0.0f;
    float stagnation_potential = 0.0f;
    float expected_valence = 0.0f;
    float max_predicted_peril = 0.0f;
    size_t active_primitives_count = 0;
    size_t active_hormones_count = 0;
    size_t current_latent_dim = 16;
    size_t horizon_k = 1;
    float lock_ratio = 0.0f;
    bool event_sprouted = false;
    bool event_pruned = false;
    bool event_hormone_sprouted = false;
    bool event_hormone_pruned = false;
    bool event_latent_sprouted = false;
    size_t action_discrete_chosen = 0;
    size_t episodic_memory_slots = 0;     // Number of unbounded episodic memory nodes
    float episodic_familiarity = 0.0f;    // Familiarity / novelty index of current state
};

class KaryonSovereignCore {
public:
    explicit KaryonSovereignCore(const CoreConfig& config);
    ~KaryonSovereignCore() = default;

    // 1. Continuous Analog Motor Control via Variational Adjoint Field
    void decide_continuous_action(std::span<const float> current_state,
                                  std::span<const float> target_goal,
                                  std::span<float> action_out);

    // 2. Continuous Trajectory Rollout & Variational Optimization in Control Space
    // Optimizes the continuous action vector directly along the manifold gradient
    void optimize_continuous_action_trajectory(std::span<const float> current_state,
                                               std::span<const float> target_goal,
                                               std::span<float> action_inout,
                                               size_t variational_steps = 3);

    // 3. Discrete Action Decision (backed by Continuous Trajectory Rollout)
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

    // Endogenous Hormonal Telemetry
    float get_hormone(size_t index) const {
        if (index < hormones_.size()) return hormones_[index].concentration;
        return 0.0f;
    }
    size_t get_hormones_count() const { return hormones_.size(); }
    float get_effective_plasticity() const { return effective_plasticity_; }
    float get_effective_motor_vigor() const { return effective_motor_vigor_; }
    float get_stagnation_potential() const { return stagnation_potential_; }
    size_t get_current_horizon() const { return current_horizon_k_; }
    float get_average_epigenetic_lock() const;
    size_t get_primitives_count() const { return primitives_.size(); }

    // Autopoietic Genesis of Primitives & Hormones & Latent Dimensions
    bool trigger_morphogenetic_sprout();
    bool prune_least_causal_primitive();

    bool trigger_hormone_sprout();
    bool prune_least_causal_hormone();

    bool trigger_latent_dimension_sprout();
    size_t get_current_latent_dim() const { return current_latent_dim_; }

private:
    CoreConfig config_;
    std::mt19937 rng_;
    std::vector<std::unique_ptr<ContinuousSuperPrimitive>> primitives_;
    std::vector<SovereignHormone> hormones_;

    alignas(32) std::vector<float> W_sensory_down_;
    alignas(32) std::vector<float> W_sensory_up_;
    alignas(32) std::vector<float> W_motor_;
    std::vector<float> action_delta_table_; // [action_dim x sensory_dim] for trajectory rollout
    std::vector<float> W_action_valence_;   // [action_dim x sensory_dim] linear direct reflex
    alignas(32) std::vector<float> W_peril_latent_; // [action_dim x latent_dim] Non-linear Peril Field

    // Dual-Level Hierarchical Active Inference & Successor Diffusion Attractor
    alignas(32) std::vector<float> M_successor_;        // [latent_dim x latent_dim] Topological graph diffusion
    alignas(32) std::vector<float> goal_successor_map_; // [latent_dim] Diffused topological potential of goal
    alignas(32) std::vector<float> macro_subgoal_latent_; // [latent_dim] Level 2 Slow Macro-Attractor
    float macro_subgoal_timer_;                        // Macro-attractor phase counter

    // Temporal Eligibility Traces for Backward Threat Attribution
    std::vector<float> action_eligibility_trace_; // [action_dim x sensory_dim]

    // Endogenous Effector Modulations computed from Sovereign Hormones
    float effective_plasticity_;
    float effective_motor_vigor_;
    float effective_threat_weight_;
    float effective_exploration_;
    float somatic_energy_;
    float stagnation_potential_;           // Continuous non-static stagnation field trace
    float appetitive_hunger_;              // Continuous goal frustration / appetitive drive
    float last_goal_dist_sq_;              // Squared distance to goal on previous step
    size_t current_latent_dim_;            // Sovereign Dynamic Latent Dimensionality
    size_t current_horizon_k_;
    size_t ticks_since_last_sprout_;
    size_t ticks_since_last_hormone_sprout_;
    size_t ticks_since_last_latent_sprout_;
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

    void update_sovereign_hormones(std::span<const float> sensory_state, float surprise, float valence);
    void compute_effector_modulations();
    void evaluate_autopoietic_morphogenesis(float fe_error);
    
    // Non-linear Peril Predictor in Latent Space
    float compute_latent_peril(size_t action_id, const std::vector<float>& h_latent);

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
